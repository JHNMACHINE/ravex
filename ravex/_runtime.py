"""The Ravex runtime.

One instance per process. It owns the registry, the checkpoint backend and the
resume logic, and it is what the PyTorch patches call into.

Two invariants shape everything here:

*Never break the training run.* Every entry point is wrapped; on an unexpected
error the user's code carries on as if Ravex were not installed
(``fallback_on_error``). A failed checkpoint costs that checkpoint and is tried
again at the next one — it does not turn the runtime off, because the runtime
is per process and one rank switching itself off is how the other seven end up
waiting in a collective it will never enter.

*Never write to stdout.* Training output belongs to the user. Logs go to the
configured file, or to stderr at WARNING and above when no file is set —
silently swallowing a checkpoint failure would be worse than three lines of
stderr.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Any, List, Optional

from ravex._backends import get_backend
from ravex._config import RavexConfig
from ravex._dist.collectives import (
    all_ranks_agree,
    get_rank,
    get_world_size,
    is_main_process,
)
from ravex._patches import install_all_patches
from ravex._registry import ObjectRegistry
from ravex._resume import ResumeManager

logger = logging.getLogger("ravex")

_LOG_FORMAT = "%(asctime)s [ravex] %(levelname)s %(message)s"

#: How many further optimizer steps a due round may survive before the
#: runtime says out loud that nobody is closing it. The dataloader path
#: cannot reach this: the boundary is the top of the next iteration, which
#: comes before the next `optimizer.step()` even under gradient
#: accumulation. The slack is for a loop that steps the same optimizer
#: more than once per iteration, which is unusual but not wrong.
_ROUND_BOUNDARY_GRACE_STEPS = 2


def _drain_accelerator() -> None:
    """Wait for this rank's queued device work, and let the caller bill it.

    The training loop is asynchronous: Python runs ahead of the device, so when
    a checkpoint is taken there is still work in flight, and none of the state
    can be read until it lands. Something has to wait for it. The question is
    only what the waiting gets called.

    Measured on 8x RTX 5060 Ti with a 1.5B model, FSDP2 per-rank: this drain is
    2.98 s at the first checkpoint of a run and ~7.1 s at every one after, while
    the collection it precedes — `get_state_dict` plus the device-to-host copy —
    is 1.55 s and does not move. Folded together, as they were, `collect` read
    5 s and then 10 s, and the doubling looked like a defect in the collection.
    It was the loop's lead over the device growing from one queued step to two.

    Timing it separately costs nothing and stops the phase breakdown from
    charging training compute to the checkpoint. The wall time is unchanged:
    this synchronize is the one `collect_state` would have done implicitly a
    moment later.

    Silent when there is no accelerator, and deliberately not `torch.cuda`
    specific — a CPU run has nothing to drain and must not import cuda to find
    that out.
    """
    torch = sys.modules.get("torch")
    if torch is None:  # pragma: no cover - checkpointing implies torch
        return
    try:
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None and accelerator.is_available():
            accelerator.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not drain the accelerator queue: %s", exc)


class RavexRuntime:
    """Singleton runtime. Build it through :func:`get_runtime`."""

    #: Defaults at class level as well as in `__init__`, because the tests —
    #: and any future caller — build a runtime with `__new__` to exercise one
    #: method without standing up the whole thing. False is the safe reading:
    #: a job that has not established a transport has not got one.
    _byte_transport_ok = False
    _byte_group = None
    #: None means the socket ring has not been tried yet, False that it was
    #: and could not be had. Both roads move the same bytes, so a runtime
    #: built with `__new__` starting on the collective one is the right
    #: reading rather than a degraded one.
    _ring_link = None
    #: Set only from `_install_signal_handler`'s closure - the one thing that
    #: code is allowed to do. See `on_step` for why.
    _emergency_requested = False
    #: Whether this rank participates in the SIGTERM detection channel at
    #: all: sharded, multi-machine, and turned on. Decided once from facts
    #: identical on every rank (config plus launcher environment), the same
    #: way `_storage_split` is - so it needs no agreement collective of its
    #: own to be safe to branch on.
    _emergency_active: Optional[bool] = None
    _emergency_group = None
    #: The rendezvous store the alarm is announced on, once asked for. `False`
    #: is "asked and could not be had", the same distinction `_ring_link`
    #: makes above; `None` at class level so a `__new__`-built runtime asks.
    _emergency_store = None
    #: The step every rank has agreed to save at, once one has been announced.
    _emergency_round: Optional[int] = None
    #: Set once the announced round is behind this rank, whether it attended
    #: it or read the alarm too late to. Both mean the same thing to
    #: `_emergency_round_has_come`: there is nothing further to do on this
    #: road, and a second attempt would be a rendezvous nobody announced.
    _emergency_settled = False

    def __init__(self, config: Optional[RavexConfig] = None):
        self.config = config or RavexConfig.load()
        self.registry = ObjectRegistry()

        self._enabled = self.config.enabled
        self._patches = None
        self._backend = None
        self._resume_manager: Optional[ResumeManager] = None
        self._storage_announced = False
        #: Set by the probe: storage is local *and* each machine has its own.
        self._storage_split = False
        #: Whether this job can move bytes between ranks at all, and the group
        #: to do it on. Settled once at activation; see `byte_transport_group`.
        self._byte_transport_ok = False
        self._byte_group = None
        #: The socket ring, built on the first replication round that wants it
        #: and kept for the process. `False` means it was tried and could not
        #: be had, which is not the same as "not tried yet" and must not be
        #: retried every round — see `_replication_link`.
        self._ring_link: Any = None
        self._last_replicated_step: Optional[int] = None
        self._leader_optimizer_id: Optional[int] = None
        self._checkpoint_due = False
        self._resume_attempted = False
        self._restoring = False
        self._last_saved_step: Optional[int] = None
        # For the cadence warning: when the previous handoff finished,
        # and whether the observation has already been made.
        self._last_checkpoint_end: Optional[float] = None
        self._cadence_warned = False
        self._emergency_requested = False
        self._emergency_active: Optional[bool] = None
        self._emergency_group = None
        self._emergency_store: Any = None
        self._emergency_round: Optional[int] = None
        self._emergency_settled = False
        #: The outer loop and its transport (GPU-113), built on the first step
        #: that wants them. `False` means it was tried and could not be had —
        #: not the same as "not tried yet", and not retried every step, the
        #: same distinction `_ring_link` makes.
        self._outer: Any = None
        self._exchange: Any = None
        self._outer_peers: List[int] = []
        #: Who is in the run, evaluated per round. None until the outer loop is
        #: built, because it needs the rendezvous store the exchange takes its
        #: addresses from.
        self._membership: Any = None
        #: Kept so the round boundary does not go looking for the store again
        #: on a path that runs once per round.
        self.store_for_joins: Any = None
        self._round_due = False
        #: The step the round came due at, for the warning below. Only
        #: meaningful while `_round_due` is up.
        self._round_due_at: Optional[int] = None
        self._round_boundary_warned = False
        #: Whether anything is handing us the top of the iteration — the
        #: dataloader wrapper, or `ravex.batch_boundary()` called by hand.
        self._boundary_delivered = False
        self._shutdown_done = False
        self._lock = threading.RLock()
        self._framework = "unknown"
        #: The adapter for that framework, from activation onward. None before
        #: it, and on a run whose activation failed.
        self._adapter: Any = None
        #: The adapter's answer to `should_intercept_step`, asked once at
        #: activation. See `on_step` for what a False costs.
        self._count_steps = True

        self._setup_logging()

        for problem in self.config.problems:
            logger.warning("Configuration: %s", problem)

    # ─── lifecycle ──────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        logger.handlers.clear()
        logger.propagate = False  # the user's root logger stays untouched

        level = getattr(logging, self.config.log_level, logging.INFO)
        if self.config.log_file:
            try:
                handler: logging.Handler = logging.FileHandler(
                    self.config.log_file, encoding="utf-8"
                )
                handler.setLevel(level)
            except OSError:
                handler = logging.StreamHandler(sys.stderr)
                handler.setLevel(logging.WARNING)
        else:
            # No log file: stay out of the way, but do not hide failures.
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(max(level, logging.WARNING))

        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)

    def activate(self) -> None:
        """Install the PyTorch patches and get ready to checkpoint."""
        if not self._enabled:
            logger.debug("Ravex is disabled by configuration")
            return

        try:
            from ravex._frameworks import detect_framework, get_adapter

            self._framework = detect_framework()
            self._adapter = get_adapter(self._framework)
            self._resolve_step_interception()
            self._patches = install_all_patches(self.registry, self)
            self._install_signal_handler()

            logger.info(
                "Ravex active - %s framework=%s rank=%d/%d",
                self.config.describe(),
                self._framework,
                get_rank(),
                get_world_size(),
            )
        except Exception as exc:
            logger.error("Activation failed: %s", exc, exc_info=True)
            self._disable("activation failed")

    def _resolve_step_interception(self) -> None:
        """Ask the adapter, once, whether ``optimizer.step()`` is the loop's step.

        Once rather than per step. The answer is a property of the framework,
        the framework is settled here and cannot change under a running loop,
        and `on_step` is the hottest path Ravex has: a bool test belongs in it,
        a dispatch through an adapter does not.

        Asking *here* is worth something the autoloader could not have had.
        Detection reads ``sys.modules``, and the decorator runs when the
        training function is called — with transformers, or Lightning, or
        DeepSpeed already imported by the script above it. The autoloader asked
        at interpreter startup, before the script had imported anything, and
        the honest answer at that point was always "vanilla".
        """
        try:
            self._count_steps = bool(self._adapter.should_intercept_step())
        except Exception as exc:
            logger.warning(
                "Adapter %s could not say whether to count optimizer steps "
                "(%s: %s) - counting them, which is what every framework "
                "Ravex ships an adapter for asks for",
                self._framework,
                type(exc).__name__,
                exc,
            )
            self._count_steps = True
            return

        if not self._count_steps:
            logger.warning(
                "Adapter %s has taken over step counting: Ravex will not count "
                "optimizer.step(), and nothing is checkpointed until something "
                "else advances the step count",
                self._framework,
            )

    def _announce_storage_topology(self) -> None:
        """Say which of the three situations this multi-machine job is in.

        A shared filesystem and a local disk are indistinguishable from the
        configuration — both are ``storage.type: local`` pointing at a
        directory that exists — so this asks the filesystem instead of
        guessing. Guessing wrong costs either a false alarm on a cluster that
        is perfectly safe, or silence on a job whose checkpoint is being split
        across disks that cannot see each other.

        Said before the first checkpoint, which is the only moment it can
        prevent anything: after that the shards have been written to the wrong
        places and the warning has nothing left to do. Not at activation
        though — Ravex activates on import, before the training script calls
        ``init_process_group``, and the probe is a collective.

        Every rank probes; one rank per machine speaks. Gating the probe on the
        local rank would leave the others waiting inside a collective the
        speaker alone entered.
        """
        from ravex._dist.collectives import (
            byte_transport_group,
            is_local_main_process,
            local_world_size,
            spans_several_machines,
            storage_is_shared,
        )

        if self._storage_announced:
            return
        self._storage_announced = True

        # Both conditions come from config and launcher environment, identical
        # on every rank, so the ranks take this branch together or not at all.
        if self.config.storage.is_remote or not spans_several_machines():
            return

        shared = storage_is_shared(self.config.storage.path)
        # Kept: it is the difference between needing replication and not, and
        # asking again per checkpoint would be a collective per checkpoint.
        self._storage_split = not shared

        # Before the announcement, because what it is allowed to promise
        # depends on the answer — and collective, so it happens on every rank
        # rather than only on the one that logs. Reached by all of them: the
        # branches above are taken together or not at all.
        if self._storage_split:
            self._byte_transport_ok, self._byte_group = byte_transport_group()

        if not is_local_main_process():
            return

        machines = -(-get_world_size() // max(local_world_size(), 1))
        if shared:
            logger.info(
                "Checkpoint storage (%s) is shared across all %d machines - "
                "per-rank checkpoints resume no matter which machine gets "
                "which ranks.",
                self.config.storage.path,
                machines,
            )
            return

        from ravex._dist.replication import replication_ring

        ring = replication_ring(get_rank(), get_world_size(), local_world_size())
        replicating = (
            self.config.replicate_every > 0
            and ring is not None
            and self._byte_transport_ok
        )

        if replicating:
            logger.info(
                "This job spans %d machines and each writes checkpoints to its "
                "own local storage (%s) - verified, not assumed. Each rank's "
                "store is copied to a peer on another machine every %d "
                "checkpoints, so losing one machine costs at most that much "
                "progress. Keeping the checkpoints after the run ends is "
                "yours: copy them off before the machines go away, or point "
                "storage at S3.",
                machines,
                self.config.storage.path,
                self.config.replicate_every,
            )
            return

        # Off, and the two reasons are not the same. Saying which one matters:
        # a layout that cannot be replicated looks exactly like one nobody
        # asked to replicate, and only one of them is the user's doing.
        if ring is None and self.config.replicate_every > 0:
            reason = (
                "copies between machines are switched on but cannot be placed "
                "here: the ranks are not spread evenly over the machines, so "
                "no peer can be shown to be on a different one"
            )
        elif not self._byte_transport_ok and self.config.replicate_every > 0:
            reason = (
                "copies between machines are switched on but there is no way "
                "to move bytes between the ranks: this job's backend does not "
                "carry host tensors and no gloo group could be opened"
            )
        else:
            reason = "copies between machines are off (replicate_every=0)"

        logger.warning(
            "This job spans %d machines and each writes checkpoints to its own "
            "local storage (%s) - verified, not assumed. Every machine holds "
            "only its own ranks' shards, and %s, so the checkpoint resumes "
            "only if every machine is given the same ranks again - which no "
            "launcher promises - and not at all if one machine is lost. Set "
            "replicate_every, or point storage at a filesystem every node "
            "shares.",
            machines,
            self.config.storage.path,
            reason,
        )

    def _ensure_backend(self) -> bool:
        """Build the checkpoint backend on first real use.

        Not at activation time. A `torchrun` launcher, a dataloader worker, or
        any other helper process that happens to import torch inside a project
        with a ravex.yaml would otherwise each construct a checkpoint manager —
        creating directories and, with S3 or R2 configured, opening connections
        on behalf of a process that will never train a single step.
        """
        if self._backend is not None:
            return True
        if not self._enabled:
            return False
        try:
            self._backend = get_backend(self.config, per_rank=self._per_rank_active())
            self._resume_manager = ResumeManager(
                self._backend, self.registry, self.config
            )
        except Exception as exc:
            # Reported, not disabled. This runs inside `checkpoint`, between
            # the collective collect and the verdict every rank agrees on, and
            # turning *this* process off there is the asymmetry that strands
            # the others. The caller folds the False into that verdict.
            logger.error("Could not open the checkpoint backend: %s", exc)
            return False
        return True

    def _per_rank_active(self) -> bool:
        """Whether this run really is checkpointing per rank.

        Asking the config is not enough: ``per_rank`` downgrades to a gather
        for a model whose shards are not DTensors, and then rank 0 is again the
        only writer. Store layout and write path have to agree — a rank writing
        into ``rank_3/`` that never writes leaves a store the resume will find
        empty — so both ask the registry the same structural question.

        **World size one is not automatically "not per rank".** It used to be,
        and that cost an elastic job everything it had done: a cluster that
        shrinks to a single node comes back, finds ``per_rank`` switched off by
        the world size alone, looks in ``checkpoints/`` instead of in
        ``checkpoints/rank_<n>/``, and reports *No checkpoint found - starting
        from scratch* while the shards sit right there beside it. The reshard
        planner has always handled ``N -> 1`` — ``tests/test_dist_reshard.py`` calls
        it "the shrink taken to its limit" — but the resume path never asked it
        to, because this method answered before the question was reached.
        Reproduced end to end in ``integration/elastic/probe.sh``: at
        ``AGENTS=3`` the survivors resume, at ``AGENTS=2`` they silently do not.

        So a lone rank still takes the per-rank path when per-rank stores are
        already on disk — meaning this run is continuing a wider one. That
        keeps reading and writing on the same layout, which matters as much as
        the read itself: resuming from ``rank_<n>/`` and then writing to
        ``checkpoints/`` would leave the next resume choosing between two
        histories, and it would choose the stale one.

        A lone rank with no such stores is an ordinary single-rank run and
        writes an ordinary store, exactly as before. Remote storage answers
        False for the same reason :func:`~ravex._backends.visible_rank_stores`
        returns nothing there — every node sees the same bucket, so the
        question carries no information — and resharding refuses remote storage
        one layer down in any case.
        """
        if self.config.sharded_checkpoints != "per_rank":
            return False
        if self.registry.sharded_layout("per_rank") != "per_rank":
            return False
        if get_world_size() > 1:
            return True

        from ravex._backends import visible_rank_stores

        return bool(visible_rank_stores(self.config))

    def _disable(self, reason: str) -> None:
        if not self._enabled:
            return
        self._enabled = False
        logger.error("Ravex disabled (%s) - training continues unaffected", reason)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self):
        return self._backend

    @property
    def step(self) -> int:
        return self.registry.step_count

    # ─── hooks called from the patches ──────────────────────────────

    def on_step(self, optimizer: Any) -> None:
        """Called after every ``optimizer.step()``.

        The counter advances in optimizer steps, not micro-batches, so gradient
        accumulation needs no special handling: ``step()`` fires once per
        accumulation cycle by construction.

        With several optimizers (a GAN's generator and discriminator, say) only
        the first one registered drives the counter, so one training iteration
        stays one step.
        """
        if not self._enabled:
            return

        # Loading sharded optimizer state calls optimizer.step() with zero
        # gradients, to allocate the state tensors before filling them in
        # (torch.distributed.checkpoint.state_dict._init_optim_state). That
        # step changes nothing and must not be counted: left unguarded, every
        # FSDP resume silently burns one step of the budget and misnumbers
        # every checkpoint after it, compounding on each restart.
        if self._restoring:
            return

        if self._leader_optimizer_id is None:
            self._leader_optimizer_id = id(optimizer)
        elif id(optimizer) != self._leader_optimizer_id:
            return

        # Last chance to resume: reached only when the run has no DataLoader,
        # since DataLoader.__iter__ fires earlier.
        if not self._resume_attempted:
            self._try_resume()

        # The adapter's veto, settled once at activation. After the resume
        # above rather than before it: a framework that counts its own steps
        # still has to be resumed, and for a loop with no DataLoader this is
        # the last point where that can happen.
        if not self._count_steps:
            return

        self.registry.step_count += 1

        if self.config.outer_loop:
            outer = self._outer_loop()
            if outer is not None:
                self._warn_if_no_round_boundary(outer)
                outer.record_step()
                if outer.round_is_over():
                    # Flagged, not closed. Same discipline as `_checkpoint_due`
                    # right below and for a stronger version of the same
                    # reason: this is the middle of `optimizer.step()`, before
                    # the LR scheduler has run, and closing a round here would
                    # write the outer parameters into the model underneath an
                    # optimizer that is still mid-step — after having spent
                    # however many minutes of network inside it.
                    if not self._round_due:
                        # The transition, not the state: `round_is_over()`
                        # stays true from the step it first answers yes until
                        # a round actually closes, so re-stamping it here
                        # would reset the clock every step and the warning
                        # below would never come due.
                        self._round_due_at = self.registry.step_count
                    self._round_due = True

        if self._emergency_coordination_active():
            if self.registry.step_count % self.config.emergency_check_every == 0:
                self._check_emergency_signal()

        if self.registry.step_count % self.config.checkpoint_every == 0:
            # Not here: this is the middle of the iteration, before the LR
            # scheduler has stepped. Flag it and collect at the top of the next
            # one — see _BatchBoundaryIterator.
            self._checkpoint_due = True
            if not self.registry.dataloaders and not self._boundary_delivered:
                # No dataloader to give us that boundary (a loop over
                # pre-batched tensors) and nobody calling
                # `ravex.batch_boundary()` either. Mid-iteration is then the
                # best available point — at the cost this hatch has always
                # had, a saved learning rate one step stale.
                self._checkpoint_due = False
                self.checkpoint()

    def on_batch_boundary(self) -> None:
        """Called at the top of each training iteration, before the batch.

        Two callers: the dataloader wrapper, which gets this moment for free,
        and `ravex.batch_boundary()`, which is how a loop without a dataloader
        hands it over.
        """
        if not self._enabled:
            return
        # Remembered because the *checkpoint* hatch in `on_step` reads "no
        # dataloader" as "no boundary". A loop that calls
        # `ravex.batch_boundary()` has the boundary without having a
        # dataloader, and its checkpoints belong here rather than mid-step.
        self._boundary_delivered = True
        if self._round_due:
            self._round_due = False
            self._close_outer_round()
        if not self._checkpoint_due:
            return
        self._checkpoint_due = False
        self.checkpoint()

    # ─── the outer loop, GPU-113 ────────────────────────────────────────

    def _outer_loop(self):
        """The outer loop and its exchange, built once. None if unavailable.

        Built here rather than at activation because it needs a model, and at
        activation there may not be one yet — the registry fills up as the
        training loop runs. First step that has one is the earliest honest
        moment, and it is before any round could be due.

        ``False`` is remembered as "asked and could not be had", so a job
        without a rendezvous does not retry, and log, every step of the run.
        """
        if self._outer is False:
            return None
        if self._outer is not None:
            return self._outer

        if not self.registry.models:
            # Not a refusal, and this distinction is load bearing: a model
            # built inside the decorated function is registered by the patches
            # partway through, so "none yet" at step one is ordinary and
            # remembering it as a failure would leave the run training alone
            # for a reason that stopped being true immediately after.
            return None

        try:
            self._outer = self._build_outer_loop()
        except Exception as exc:
            logger.warning(
                "Could not start the outer loop (%s). Training continues "
                "locally, which for a job that expected to train with peers "
                "is not a degraded mode - it is a different run.",
                exc,
            )
            self._outer = False
            return None
        return self._outer or None

    def _build_outer_loop(self):
        """Open the round exchange and the outer loop, or explain the refusal.

        Raises rather than returning None so that every reason ends up in one
        log line at the caller, said the same way.
        """
        from ravex._dist.agreement import rendezvous_store
        from ravex._dist.collectives import get_rank, get_world_size
        from ravex._dist.exchange import SEED_ROUND, DeltaExchange, adopt_outer_state
        from ravex._dist.membership import Membership
        from ravex._dist.outer import OuterLoop

        models = self.registry.models
        if len(models) > 1:
            raise RuntimeError(
                "%d models are registered and the outer loop averages one. "
                "Name the one being trained with ravex.track(model=...)"
                % len(models)
            )

        store = rendezvous_store()
        if store is None:
            # Worth being precise about, because it reads like a
            # fault-tolerance limit and is not one. The process group is used
            # for its *store* — addresses and the job token — and never for
            # the exchange, so a rank dying breaks a group nothing here
            # touches. What it constrains is the start: `init_process_group`
            # is collective, so everyone has to be present once.
            raise RuntimeError(
                "no rendezvous store: the outer loop takes peer addresses and "
                "the job token from torch's store, which exists only after "
                "init_process_group. Launch under torchrun (across machines "
                "with --rdzv-backend=c10d)"
            )

        world = get_world_size()
        if world < 2:
            raise RuntimeError("world size is %d; a round needs peers" % world)

        rank = get_rank()
        # The set this node exchanges with is now asked for **per round**
        # rather than computed once (GPU-121). Computed once, a node that joins
        # a run in progress appears in it nowhere and no amount of announcing
        # helps; asked per round, membership is a function of the round number
        # and every member computes the same answer for the same round.
        self._membership = Membership(store, rank, world)
        self.store_for_joins = store
        self._outer_peers = self._membership.peers_at(0)
        root = self.config.outer_root or os.path.join(
            self.config.storage.path, "rounds", str(rank)
        )

        exchange = DeltaExchange(
            rank,
            store,
            root=root,
            node=str(rank),
            compression_level=self.config.compression_level,
            save_dtype=self.config.outer_save_dtype,
        )
        if not exchange.start():
            raise RuntimeError("the round exchange could not open its listener")
        self._exchange = exchange

        loop = OuterLoop(
            models[0],
            inner_steps=self.config.outer_inner_steps,
            round_seconds=self.config.outer_round_seconds or None,
            lr=self.config.outer_lr,
            momentum=self.config.outer_momentum,
            combine_mode=self.config.outer_combine,
            # **On the host, and it is not a preference** (GPU-124). A round
            # report is a moonclip snapshot and moonclip reads host memory, so
            # an outer state in VRAM is not publishable at all: with the model
            # on CUDA this raised before the seed round, Ravex said so, and the
            # run carried on training locally - which is two boxes holding two
            # models, not a degraded one. Left unpassed, the snapshot was born
            # wherever the model was, and every test and bench in this
            # repository is on CPU, so nothing ever exercised the other case.
            #
            # `snapshot` already documented the second reason: two extra copies
            # of the parameters live here, and keeping them off the accelerator
            # is the difference between a model fitting on a cheap box and not.
            # The cost is one host-device transfer per round, against a round
            # measured in seconds of network.
            device="cpu",
            node=str(rank),
        )

        # Before anything else, and it is not optional: every node has to hold
        # the same outer parameters or the difference between them is never
        # touched again. See `adopt_outer_state` for what that looks like from
        # the outside, which is a run that appears to be working.
        if not adopt_outer_state(
            loop, exchange, 0, rank, time.monotonic() + self.config.outer_deadline
        ):
            raise RuntimeError(
                "could not take the starting parameters from rank 0 within "
                "%ds. Training on this node's own would put it a fixed "
                "distance from every peer for the rest of the run"
                % self.config.outer_deadline
            )
        # Training rounds start after the seed round, which used number 0.
        loop.round_number = SEED_ROUND + 1
        logger.info(
            "Outer loop on: rank %d of %d, %d inner step(s) per round%s, "
            "reports staged in %s.",
            rank,
            world,
            self.config.outer_inner_steps,
            " or %.0fs" % self.config.outer_round_seconds
            if self.config.outer_round_seconds
            else "",
            root,
        )
        return loop

    def _warn_if_no_round_boundary(self, outer) -> None:
        """Say it when a round is due and nothing ever closes it.

        `_round_due` goes up inside `optimizer.step()` and comes down at the
        next batch boundary. A loop with no DataLoader is handed no boundary
        for free, so unless it calls `ravex.batch_boundary()` the flag stays
        up for the rest of the run: no round closes, no delta is exchanged,
        and every node trains its own model to the end while the log says
        nothing at all. The silence is the defect worth fixing here — more
        than the missing call, which the message names.

        Once per process. A run that has been told has been told.
        """
        if self._round_boundary_warned or not self._round_due:
            return
        if self._round_due_at is None:
            return
        waited = self.registry.step_count - self._round_due_at
        if waited < _ROUND_BOUNDARY_GRACE_STEPS:
            return

        self._round_boundary_warned = True
        logger.warning(
            "Outer round %d has been ready to close for %d optimizer steps "
            "and nothing has closed it. Ravex takes the batch boundary from "
            "the DataLoader iterator, and this loop has no DataLoader "
            "registered, so that moment never arrives: no round will close, "
            "no delta will be exchanged with the peers, and this node will "
            "train alone for the rest of the run. Call ravex.batch_boundary() "
            "at the top of each training iteration.",
            outer.round_number,
            waited,
        )

    def _close_outer_round(self) -> None:
        """Exchange deltas with the peers and take the outer step.

        Never raises. A round that could not be closed leaves the model exactly
        where the local training put it and the next round carries on from
        there — which is worse than a round that worked and is very much better
        than a training script that stops because a peer's disk was full.
        """
        from ravex._dist.exchange import close_round
        from ravex._dist.membership import MembershipError, serve_joins

        if not self._outer or self._exchange is None:
            return
        try:
            started = time.monotonic()
            if self._membership is not None:
                # At the boundary, before the round's own work: writing the
                # outer parameters down for a candidate and taking in an
                # announcement both have to happen at the round number every
                # member agrees on, which is this one.
                serve_joins(
                    self.store_for_joins, self._exchange, self._outer,
                    self._membership,
                )
                self._outer_peers = self._membership.peers_at(
                    self._outer.round_number
                )
            report = close_round(
                self._outer,
                self._exchange,
                self._outer_peers,
                time.monotonic() + self.config.outer_deadline,
            )
            # The split, not just the total. On loopback this line read
            # "took 0.0s" every time, which is the one number the whole
            # architecture is chosen around and it was never observed - see
            # `close_round` and GPU-117. `network` is what a slower link makes
            # bigger; the rest is the model's size, not the link's speed.
            logger.info(
                "Outer round %d took %.1fs over %d node(s): %.1fs network "
                "(%.1fs of it waiting for a peer to reach the round), "
                "%.1fs delta, %.1fs publish (%.1fs waiting on a fetch), "
                "%.1fs outer step.",
                report["round"],
                time.monotonic() - started,
                report["nodes"],
                report.get("gather_seconds", 0.0),
                report.get("gather_wait_seconds", 0.0),
                report.get("delta_seconds", 0.0),
                report.get("publish_seconds", 0.0),
                report.get("publish_wait_seconds", 0.0),
                report.get("apply_seconds", 0.0),
            )
        except MembershipError:
            # **Not swallowed like the rest.** Every other failure here leaves
            # this node's own parameters where its training put them and costs
            # the run one round; this one says the averages have already
            # differed between nodes, so there is no round to carry on to and
            # the honest thing is to stop rather than keep training a second
            # model that looks like the first.
            raise
        except Exception as exc:
            logger.warning("Outer round failed: %s", exc)
            # Advance anyway. A node that stays on a number its peers have left
            # behind is asking for a round they retired while they ask for one
            # it never reaches - both still running, both still logging closed
            # rounds, permanently invisible to each other.
            try:
                self._outer.abandon_round()
            except Exception:
                pass

    def should_stop(self) -> bool:
        """Whether the configured step budget has been spent.

        A resumed script re-runs its own loop bounds from the top — it has no
        idea 3000 of its 5000 steps already happened in a previous process. So
        when ``max_steps`` is set, the dataloaders start returning empty:
        remaining epochs fall through instantly and the script exits on its own
        terms, without Ravex raising anything into user code.
        """
        if not self._enabled or self.config.max_steps is None:
            return False
        return self.registry.step_count >= self.config.max_steps

    def on_dataloader_iter(self, _loader: Any) -> None:
        """Called on ``iter(dataloader)``.

        This is the moment to resume: the model, the optimizer and the loader
        all exist, and no batch has been drawn yet — so the restored dataset
        position and RNG state are applied before they are used.
        """
        if not self._enabled or self._resume_attempted:
            return
        if self.registry.is_empty():
            return  # nothing to restore into yet
        self._try_resume(defer_rng=True)

    def after_dataloader_iter(self) -> None:
        """Called once the loader's iterator exists.

        Constructing it draws a worker base seed from the global RNG, so the
        restored RNG state is applied here rather than in
        :meth:`on_dataloader_iter` — otherwise the first resumed step would run
        one draw ahead of the run it is meant to continue.
        """
        if self._enabled:
            self.registry.apply_pending_rng()

    def _try_resume(self, defer_rng: bool = False) -> None:
        self._resume_attempted = True
        # Before the `resume` guard: a run that never resumes still deserves to
        # be told its checkpoints are being split, and this is the first point
        # where the process group is up to find out.
        self._announce_storage_topology()
        if not self.config.resume:
            return
        if not self._ensure_backend():
            return
        # Before anything asks what is on disk: a machine that was replaced has
        # nothing of its own. With a bucket the way back is a download; without
        # one it is a peer that has been holding a copy.
        self._restore_from_remote_if_empty()
        self._recover_missing_stores()
        # Bound locally: `_ensure_backend` sets it, but only a local name makes
        # that visible to a type checker, and the package ships `py.typed`.
        resume_manager = self._resume_manager
        if resume_manager is None:  # pragma: no cover - _ensure_backend sets it
            return
        try:
            # Anything the loading machinery does to the optimizer is not
            # training; see the guard in on_step.
            self._restoring = True
            # Before the ordinary resume, not instead of it: a foreign
            # checkpoint is converted here and the normal path is skipped only
            # when that succeeded. If it did not, the run falls through and
            # `try_resume` says what it finds, which for a directory that is
            # not a Ravex store is "nothing to resume" — the right answer.
            if self._convert_foreign(defer_rng):
                return
            resume_manager.try_resume(
                defer_rng=defer_rng, per_rank=self._per_rank_active()
            )
            self._restore_extra_state(resume_manager.restored_extra)
        except Exception as exc:
            logger.warning("Resume failed (%s) - starting from scratch", exc)
        finally:
            self._restoring = False

    def _convert_foreign(self, defer_rng: bool) -> bool:
        """Restore from a checkpoint another framework wrote, if there is one
        and the config allows it. **Collective when it acts.**

        The work is in :mod:`ravex._interop.resume`, next to the readers whose
        output it consumes; what stays here is the call, so that the attempt
        reads in order alongside the ordinary resume above.
        """
        from ravex._interop.resume import convert_on_resume

        return convert_on_resume(self.config, self.registry, defer_rng)

    def _collect_extra_state(self) -> Optional[dict]:
        """The adapter's own state, or None when there is none to have.

        Runs inside the collective window of `checkpoint`, which is why it does
        not get to fail loudly: an adapter raising here would take the weights
        down with it, and on a sharded run it would take them down on one rank
        while the other seven wrote theirs — a checkpoint that exists in pieces
        is worse than one that does not exist.

        The framework's name travels with the state because a resume has to be
        able to tell "nothing extra was saved" from "something was saved by a
        framework this run is not using".
        """
        adapter = self._adapter
        if adapter is None:
            return None

        try:
            collected = adapter.collect_extra_state()
        except Exception as exc:
            logger.warning(
                "Adapter %s could not collect its framework state (%s: %s) - "
                "the checkpoint is written without it",
                self._framework,
                type(exc).__name__,
                exc,
            )
            return None

        if not collected:
            return None
        return {"framework": self._framework, "state": collected}

    def _restore_extra_state(self, extra: Optional[dict]) -> None:
        """Hand back what `_collect_extra_state` saved, if it is this run's.

        Called once the model, the optimizer, the schedulers and the dataset
        position are back: an adapter restoring loop counters is entitled to
        assume the run around them exists.

        Three things arrive here and only one of them is state to restore. A
        checkpoint written before this seam existed — or by an adapter that
        collects nothing, which is every adapter Ravex ships today — has no
        `extra` at all, and must resume exactly as it always did. A checkpoint
        written under another framework holds state this adapter has no way to
        read, and saying so is more useful than either ignoring it or dying on
        it. Only the third is ours.
        """
        adapter = self._adapter
        if adapter is None or not extra:
            return

        written_by = extra.get("framework")
        if written_by != self._framework:
            logger.warning(
                "This checkpoint carries framework state written under %s and "
                "this run is %s, so it is being skipped. Everything else - "
                "weights, optimizer, RNG, dataset position - resumed normally",
                written_by,
                self._framework,
            )
            return

        try:
            adapter.restore_extra_state(extra.get("state") or {})
        except Exception as exc:
            logger.warning(
                "Adapter %s could not restore its framework state (%s: %s) - "
                "the run continues from the framework's own defaults, with "
                "everything else resumed",
                self._framework,
                type(exc).__name__,
                exc,
            )

    # ─── checkpointing ──────────────────────────────────────────────

    def checkpoint(self, final: bool = False, emergency: bool = False) -> bool:
        """Collect and hand off a checkpoint. Returns True if one was written.

        Collection runs on the training thread on purpose: the state has to be
        consistent with the step that just finished. The backend copies it and
        writes in the background, so what the loop actually pays for is the
        copy, not the I/O.

        ``emergency`` only adds a metadata flag (see below) — it changes
        nothing about how this function collects or writes state. By the time
        it is called with ``emergency=True``, from
        ``_check_emergency_signal``, every rank that needs to be here already
        is: the flag exists for recovery analysis afterward, not to change
        this function's own safety logic, which does not know or need to know
        why a given call happened.
        """
        if not self._enabled:
            return False

        # Checked here rather than at activation because at activation there is
        # no model yet to look at. Safe to act on unilaterally despite the rule
        # about one rank switching itself off: this is a property of the
        # DeepSpeed configuration, identical on every rank, so they all reach
        # the same verdict at the same checkpoint and none is left waiting.
        if self.registry.parameters_are_partitioned_away():
            self._disable(
                "the model's parameters are partitioned away from the module - "
                "DeepSpeed ZeRO stage 3 does this, and what is left here is the "
                "right keys with no data behind them. Reaching the real values "
                "needs a gather through DeepSpeed's own machinery, which Ravex "
                "does not drive, so a checkpoint taken from here would restore "
                "nothing while looking like it had worked. Use DeepSpeed's "
                "`engine.save_checkpoint()` for this run; Ravex can read what "
                "it writes (convert_foreign)"
            )
            return False

        # With a sharded model, collecting is a collective: every rank has to
        # call it or the ones that do will block forever waiting for the ones
        # that did not. So the rank gate moves *after* collection — every rank
        # gathers, only rank 0 writes.
        sharded = self.registry.has_sharded_models()
        if not sharded and not is_main_process():
            return False

        step = self.registry.step_count
        if self._last_saved_step == step and not final:
            return False

        started = time.perf_counter()
        phases: dict = {}
        # `drain` is one `synchronize()`, and its justification for being
        # excluded from the reported cost is that it waits for work the
        # training loop had already queued — true for compute, false for a
        # rank waiting on a peer that is late *because it was checkpointing*.
        # Measured 2026-08-30 on two machines, the same five checkpoints:
        # 34.6 s flat on one rank and 104-139 s on the other, which is not a
        # difference compute explains. A barrier first splits them: what it
        # absorbs is skew, what is left in `drain` after it is queue. See
        # GPU-98. `skew` is not excluded from `cost` below — it is exactly the
        # part of the old `drain` number that was never queued compute.
        #
        # Same guard as the verdict at the end of the save: every rank of a
        # sharded, multi-rank run reaches this line, because `sharded` and
        # `get_world_size()` read the same on every rank. A replicated
        # (non-sharded) job leaves the non-main ranks at the early `return`
        # above, so a barrier here without this guard would send rank 0 alone
        # into a collective the other seven already left — a hang until NCCL
        # times out, not an error. On one rank there is no peer to be skewed
        # from and nothing to measure, so the guard also just skips the cost
        # of a no-op barrier there.
        if sharded and get_world_size() > 1:
            from ravex._dist.collectives import barrier

            barrier()
            skew_done = time.perf_counter()
            phases["skew"] = skew_done - started
            started = skew_done
        _drain_accelerator()
        collect_started = time.perf_counter()
        phases["drain"] = collect_started - started

        # From here to the verdict below there is no early `return`, and that
        # is the point. Every rank of a sharded run is about to enter a
        # collective, so every rank has to come out of it the same way — a
        # rank that leaves early is one the others wait for until NCCL times
        # out. Failures set `ok` and are settled together at the bottom.
        ok = True
        wrote = False
        # Kept because `except` unbinds its name on the way out, and the
        # `fallback_on_error=False` path below wants the original, not a
        # summary of it.
        failure: Optional[BaseException] = None
        try:
            state = self.registry.collect_state(
                track_rng=self.config.track_rng,
                sharded_layout=self.config.sharded_checkpoints,
            )
            phases["collect"] = time.perf_counter() - collect_started

            # Under its own key on purpose. `models`, `sharded` and `rng` have
            # formats this run's readers understand; a framework's own state is
            # opaque to everything except the adapter that produced it, and
            # mixing it into a tree the reshard walks would make it a shape
            # error waiting for a topology change.
            extra = self._collect_extra_state()
            if extra is not None:
                state["extra"] = extra

            # Under `per_rank` there is nothing on rank 0 to write for the
            # other ranks — each holds its own shard and writes it into its own
            # store. The layout is read back from what was collected rather
            # than from the config: the registry downgrades to a gather when
            # the model cannot be split that way, and then rank 0 is once again
            # the only one holding anything.
            if not self._collected_per_rank(state) and not is_main_process():
                pass  # nothing of its own to write; still answers below
            else:
                backend = self._backend if self._ensure_backend() else None
                if backend is None:
                    ok = False
                else:
                    metadata = {
                        "step": str(step),
                        "framework": self._framework,
                        "world_size": str(get_world_size()),
                    }
                    if self.config.run_id:
                        metadata["run_id"] = self.config.run_id
                    if final:
                        metadata["final"] = "true"
                    if emergency:
                        # For recovery analysis, not for resume: nothing reads
                        # this back to make a decision. It says this was
                        # written because a signal came in, not because the
                        # process exited on its own - useful when reading a
                        # manifest after the fact, irrelevant to whether the
                        # checkpoint loads.
                        metadata["emergency"] = "true"

                    phases.update(backend.save(step, state, metadata) or {})
                    wrote = True
        except Exception as exc:
            logger.error("Checkpoint at step %d failed: %s", step, exc, exc_info=True)
            ok = False
            failure = exc

        # One rank that cannot write is every rank's problem.
        #
        # This used to be `self._disable("checkpoint failed")`, and `_disable`
        # is per process: a full disk on one node of eight turned that rank
        # off while the other seven stayed on. At the next checkpoint the
        # seven called `collect_state` — a collective — and the eighth
        # returned at the first line and ran ahead into the next forward. The
        # seven then wait for a participant that never comes: NCCL takes half
        # an hour to say so, and all eight GPUs stay allocated and billing
        # while it does. Not a fast error, an expensive hang.
        #
        # So the verdict is taken once, by everyone, and everyone acts on it.
        if sharded and get_world_size() > 1:
            ok = all_ranks_agree(ok)

        if not ok:
            if not self.config.fallback_on_error:
                # The user asked to hear about it rather than be protected
                # from it. Raised after the agreement, so every rank raises —
                # a single rank unwinding out of here is the asymmetry above.
                # A rank whose own save was fine has nothing to re-raise and
                # says so; the rank that failed carries its own traceback.
                if failure is not None:
                    raise failure
                raise RuntimeError(
                    f"Ravex: checkpoint at step {step} failed on another rank"
                )
            # Nothing records the step, so the next checkpoint is a fresh
            # attempt rather than a permanent surrender. `No space left on
            # device` is often gone ten seconds later, and a long run that
            # quietly drops its protection for the hours that remain is the
            # worse outcome. A fault that really is permanent costs one logged
            # attempt per checkpoint — noise, but noise that says the truth.
            logger.warning(
                "Checkpoint at step %d failed on at least one rank - skipped "
                "on all of them; retrying at the next checkpoint",
                step,
            )
            return False

        self._last_saved_step = step
        replicate_started = time.perf_counter()
        self._replicate_if_due(step)
        # Named, because on a slow link this *is* the handoff. Measured on two
        # machines with a 100 Mbps link between them: 151s of a 186s handoff,
        # absent from this breakdown and therefore absent from the cadence
        # observation below, which sums what this dict names. It stayed silent
        # while the run spent 64% of its wall time stopped here.
        #
        # On one machine the copy goes over loopback and this reads 0.000s.
        # Worth printing anyway: a phase that costs nothing is the fastest way
        # to rule it out.
        phases["replicate"] = time.perf_counter() - replicate_started
        if not wrote:
            return False

        # The breakdown, not just the total: a handoff that costs seconds is
        # seconds the training loop is stopped, and every investigation of one
        # so far has started by re-running the job to find out which phase it
        # was. Two perf_counter calls per phase is a cheap way not to.
        finished = time.perf_counter()
        logger.info(
            "Checkpoint at step %d handed off in %.3fs (%s)",
            step,
            finished - started,
            ", ".join("%s %.3fs" % item for item in phases.items()),
        )
        self._warn_if_cadence_is_expensive(step, started, finished, phases)
        return True

    def _restore_from_remote_if_empty(self) -> None:
        """Fetch this rank's store back out of the bucket, if it has none.

        Purely local — each rank pulls its own prefix and nothing waits on
        anyone — so unlike the peer recovery below there is no pairing to get
        right.

        Only when the local store is empty. A store that is present is the one
        this run has been writing, and a download that overwrote it would be
        undoing work rather than recovering it; whether a *stale* local store
        should also be refreshed is a separate question, and not this one.

        Measured on a six-node bench on 2026-08-19: with the remote push-only,
        a node whose disk had been replaced started from scratch while its data
        sat in the bucket — and took every other rank with it, since a resume
        is agreed at the oldest step everyone holds.
        """
        if not self.config.storage.is_remote or self._backend is None:
            return
        try:
            if self._backend.has_checkpoint():
                return
            if not self._backend.restore_from_remote():
                return
        except Exception as exc:
            logger.warning("Could not fetch this rank's store from the remote: %s", exc)
            return

        logger.info(
            "This rank had no store of its own and fetched one back from remote "
            "storage."
        )

    def _reread_store(self) -> None:
        """Build the backend again over a store that appeared under it.

        The backend read the directory when it was empty and its answer is now
        wrong. Cheap to drop: nothing has trained yet at this point.
        """
        self._backend = None
        self._resume_manager = None
        self._ensure_backend()

    def _recover_missing_stores(self) -> None:
        """Give a rank back a store it came up without.

        Two ways, and the cheap one goes first.

        **A copy already on this disk.** The ring addresses a peer by rank, but
        a copy travels with the disk it was written to, so a reshuffle of the
        nodes leaves every rank sitting on the copy of its own store — present,
        complete, and unreachable by asking anyone. Promoting it is a local
        file copy: no pairing, no transfer, nothing on the wire.

        **A copy on the peer that has been holding it.** The ring run
        backwards, for the machine that was replaced and arrived with an empty
        disk. Without this the copies exist and nobody reads them, which
        protects the bytes and not the run.

        Every rank takes part, and both decisions come out of one gathered
        picture: who is missing a store, who holds a copy of whose, and which
        history each of those copies belongs to. A send posted without a
        receive waiting would hang here at startup, before anyone is watching —
        so the pairing is computed from the picture *after* the local
        promotions, which every rank works out identically rather than
        announcing.

        Best-effort. Failing to recover means starting from scratch, which is
        what would have happened anyway; failing loudly at startup would not.
        """
        from ravex._backends import (
            per_rank_store_path,
            replica_store_path,
            visible_rank_stores,
        )
        from ravex._dist.collectives import gather_objects, local_world_size
        from ravex._dist.identity import run_id_at
        from ravex._dist.replication import (
            exchange_stores,
            local_recoveries,
            promote_copy,
            recovery_roles,
            replica_is_complete,
            replication_ring,
            unmark_replica,
        )

        if not self._storage_split or not self._per_rank_active():
            return

        rank = get_rank()
        ring = replication_ring(rank, get_world_size(), local_world_size())
        if ring is None:
            return
        send_to, receive_from = ring

        own = per_rank_store_path(self.config, rank)
        held = replica_store_path(self.config, receive_from)
        mine_here = replica_store_path(self.config, rank)

        try:
            needs_mine = rank not in visible_rank_stores(self.config)
            holds_whole = replica_is_complete(held)
            own_copy = replica_is_complete(mine_here)
            copy_run = run_id_at(mine_here)
            store_run = None if needs_mine else run_id_at(own)
        except Exception:  # pragma: no cover - never worth the run
            needs_mine, holds_whole, own_copy = False, False, False
            copy_run, store_run = None, None

        # Unconditional, and before any of the decisions below: the pairing is
        # only exact if every rank contributes to the same picture. It carries
        # the identities too, because a copy can only be promoted once the
        # history it belongs to is known, and that is not a local fact.
        state = gather_objects(
            (bool(needs_mine), bool(holds_whole), bool(own_copy), copy_run, store_run)
        )
        needs = [bool(entry[0]) for entry in state]
        holds = [bool(entry[1]) for entry in state]
        own_copies = [bool(entry[2]) for entry in state]
        copy_runs = [entry[3] for entry in state]
        store_runs = [entry[4] for entry in state]

        if not any(needs):
            return

        promotions = local_recoveries(needs, own_copies, copy_runs, store_runs)

        if promotions[rank]:
            if promote_copy(mine_here, own):
                logger.info(
                    "This rank had no store of its own and rebuilt one from the "
                    "copy that came back on this machine's disk - nothing had to "
                    "be fetched."
                )
                self._reread_store()
            else:
                logger.warning(
                    "This rank has no store and the copy on its own disk could "
                    "not be promoted - starting from scratch."
                )

        # Recomputed, not re-gathered. A promotion is a local act with a local
        # outcome, and asking again would cost a second collective to learn
        # something every rank can already derive. A promotion that failed
        # leaves this rank without a store and nobody sending it one, which is
        # the same place it would have been anyway — and not a hang, which is
        # what an inexact pairing would cost.
        needs = [need and not taken for need, taken in zip(needs, promotions)]
        if not any(needs):
            return

        # Promoting a copy off this machine's own disk needed no transport, and
        # has already happened above. Fetching one from a peer does, and on a
        # job with nowhere to put host bytes there is nothing further to try.
        if not self._byte_transport_ok:
            return

        do_send, do_receive = recovery_roles(rank, send_to, receive_from, needs, holds)

        try:
            # Backwards: the copy travels to the rank it belongs to, so this
            # sends to the peer it normally receives from, and the other way.
            ok = exchange_stores(
                held if do_send else None,
                own if do_receive else None,
                send_to=receive_from,
                receive_from=send_to,
                group=self._byte_group,
                link=self._replication_link(send_to, receive_from),
                # The copy travels to the rank it belongs to, so this round
                # runs against the ring. On the socket road that cannot be read
                # off the rank numbers - on two ranks the successor and the
                # predecessor are the same peer - so it is said here.
                forward=False,
            )
        except Exception as exc:
            logger.warning("Recovering this rank's store from a copy failed: %s", exc)
            return

        if do_receive and ok:
            # The copy has come home, so this directory is now this rank's own
            # store and not a copy of anybody's. `StoreWriter` marked it whole
            # on the way in, which was the right thing to do while the bytes
            # were still arriving and the wrong thing to leave behind. Removing
            # it here is what `promote_copy` does at the end of the local road.
            unmark_replica(own)
            logger.info(
                "This rank had no store of its own and took one back from rank "
                "%d, where a copy had been kept.",
                send_to,
            )
            self._reread_store()
        elif do_receive:
            logger.warning(
                "This rank has no store and the copy on rank %d could not be "
                "brought back whole - starting from scratch.",
                send_to,
            )

    def _replication_link(self, send_to: int, receive_from: int):
        """The socket ring for replication, or None to use the collectives.

        Built once and kept: the connections outlive the round, which is what
        makes the second round cheap — the manifest exchange happens over a
        connection that is already up, and only the delta crosses. Built
        *lazily* rather than at activation because a job that never replicates
        should not open connections it will not use, and because the ring's
        shape comes from `replication_ring`, which the caller has just worked
        out.

        A failure to build is remembered. The reasons a link cannot be made —
        an unreachable hostname, a firewall between nodes — do not change from
        one round to the next, and retrying every ten steps would turn a
        one-line explanation into a log nobody reads.
        """
        if self.config.replication_transport == "collectives":
            return None
        if self._ring_link is False:
            return None
        if self._ring_link is not None:
            return self._ring_link

        from ravex._dist.replication import RingLink

        store = None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                # Private, and deliberately reached through one `try`: this is
                # the rendezvous every rank of a torchrun job already shares,
                # and there is no public way to ask for it. If a torch release
                # moves it, the socket road quietly stops being taken and the
                # collective one keeps working.
                store = dist.distributed_c10d._get_default_store()
        except Exception:
            store = None

        if store is None:
            if self.config.replication_transport == "sockets":
                logger.warning(
                    "replication_transport=sockets, but there is no rendezvous "
                    "store to advertise addresses on - falling back to the "
                    "collectives for this run"
                )
            self._ring_link = False
            return None

        link = RingLink.connect(get_rank(), send_to, receive_from, store)
        self._ring_link = link if link is not None else False
        return link

    def _replicate_if_due(self, step: int) -> None:
        """Trade stores around the ring, so no machine is the only copy.

        Reached by every rank or by none. Each condition below is the same on
        all of them — the layout, the probe's verdict, the config, and a step
        the ranks have already agreed on — which is what makes it safe to run
        collectives in here.

        A round is a recovery point only if **every** rank's copy landed. Three
        out of four is not a checkpoint anyone can resume from, and recording
        it as one would be worse than skipping: the next failure would find a
        set that cannot be assembled and no sign that anything was wrong.
        """
        from ravex._backends import per_rank_store_path, replica_store_path
        from ravex._dist.collectives import all_ranks_agree, local_world_size
        from ravex._dist.replication import exchange_stores, replication_ring

        if not self._replication_due(step):
            return

        ring = replication_ring(
            get_rank(), get_world_size(), local_world_size()
        )
        if ring is None:
            return
        send_to, receive_from = ring

        source = per_rank_store_path(self.config, get_rank())
        destination = replica_store_path(self.config, receive_from)

        ok = True
        try:
            # The copy has to stand on its own once it lands. See
            # `CheckpointBackend.consolidate`.
            if self._backend is not None:
                self._backend.consolidate()
            ok = exchange_stores(
                source,
                destination,
                send_to,
                receive_from,
                group=self._byte_group,
                link=self._replication_link(send_to, receive_from),
            )
        except Exception as exc:
            logger.warning("Replication at step %d failed here: %s", step, exc)
            ok = False

        # One rank short and nobody has a recovery point at this step.
        if not all_ranks_agree(ok):
            logger.warning(
                "Replication at step %d did not complete on every rank - this "
                "step is not a recovery point. Training continues; the "
                "previous replicated step still stands.",
                step,
            )
            return

        self._last_replicated_step = step
        logger.info(
            "Replicated step %d to rank %d, and took rank %d's copy.",
            step,
            send_to,
            receive_from,
        )

    def _replication_due(self, step: int) -> bool:
        """Whether this checkpoint is one of the replicated ones.

        Derived from the step and the configuration alone, so every rank gets
        the same answer without asking each other. Off unless the storage was
        actually found to be split: with a shared filesystem or a bucket, a
        copy protects nothing and costs bandwidth.
        """
        from ravex._dist.collectives import local_world_size

        if self.config.replicate_every <= 0 or not self._storage_split:
            return False
        if not self._byte_transport_ok:
            # Said once at activation rather than once per checkpoint.
            return False
        if not self._per_rank_active():
            return False
        if get_world_size() <= max(local_world_size(), 1):
            return False

        taken = step // max(self.config.checkpoint_every, 1)
        return taken % self.config.replicate_every == 0

    #: Fraction of wall time going into checkpoint handoff above which the
    #: cadence is worth mentioning. A fifth is the point where a reader would
    #: rather have been told: below it the cost reads as overhead, above it as
    #: a choice.
    _CADENCE_WARN_FRACTION = 0.2

    def _warn_if_cadence_is_expensive(
        self, step: int, started: float, finished: float, phases: dict
    ) -> None:
        """Say what fraction of the loop is going into checkpoints. Once.

        Deliberately a statement of what happened, not a prediction. Measured on
        8x RTX 5060 Ti with a 1.5B model: at ``checkpoint_every=2`` the handoff
        is a third of wall time and nothing breaks — the writer keeps up, the
        run is simply slower than the author probably meant it to be. So this
        does not claim the cadence is unsustainable, and it does not silently
        raise it. Both would be guesses about a machine this process cannot
        see; the ratio is a fact it can.

        ``drain`` is excluded on purpose. That phase is the training loop's own
        queued GPU work coming due, not a cost of checkpointing, and counting it
        would make every cadence look expensive on a fast writer.

        ``replicate`` does count, and for the mirror reason: it exists only
        because a copy is being pushed to another machine, and the loop is
        stopped for the whole of it. Over loopback it is milliseconds and says
        nothing; over a 100 Mbps link it was 81% of the handoff.
        """
        previous = self._last_checkpoint_end
        self._last_checkpoint_end = finished
        if previous is None or self._cadence_warned:
            return

        interval = finished - previous
        cost = sum(v for k, v in phases.items() if k != "drain")
        if interval <= 0 or cost / interval < self._CADENCE_WARN_FRACTION:
            return

        self._cadence_warned = True
        logger.warning(
            "checkpoint_every=%d is costing %.0f%% of wall time: %.1fs of "
            "handoff per %.1fs between checkpoints, at step %d. The run will "
            "still complete and checkpoints are still durable - it is just "
            "spending that share on them. Raising checkpoint_every reduces it "
            "proportionally; the trade is how much progress a crash costs.",
            self.config.checkpoint_every,
            100 * cost / interval,
            cost,
            interval,
            step,
        )

    @staticmethod
    def _collected_per_rank(state: dict) -> bool:
        """Whether every sharded group in this state is one rank's own shard.

        All-or-nothing by construction (see ``_sharded_layout``); read as a
        conjunction anyway, because the one thing that must never happen is a
        rank writing a checkpoint it only holds part of.
        """
        groups = state.get("sharded") or {}
        return bool(groups) and all(
            group.get("layout") == "per_rank" for group in groups.values()
        )

    def flush(self) -> None:
        if self._backend is not None:
            try:
                self._backend.flush()
            except Exception as exc:
                logger.warning("Flush failed: %s", exc)

    # ─── shutdown ───────────────────────────────────────────────────

    def _install_signal_handler(self) -> None:
        """Catch SIGTERM — the signal a preempted spot instance gets.

        Only installed when nothing else has claimed SIGTERM and only on the
        main thread; stealing the user's handler would be worse than missing
        the last few steps.

        The handler itself does almost nothing. A signal handler runs inside
        whatever the main thread was doing when the signal landed — which, on
        a sharded job, can be a training-thread collective already in flight.
        Calling `shutdown()` from here, as this used to do, means posting a
        *second* collective from inside a context that might already be
        halfway through a first one: not a hang this code introduces on
        purpose, but one it could trigger by accident. So the handler only
        raises a flag; `on_step` — ordinary code, on the training thread,
        between collectives rather than astride one — does the rest at the
        next safe point. See `_check_emergency_signal`.
        """
        if not self.config.handle_sigterm or os.name == "nt":
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
                logger.debug("SIGTERM already handled elsewhere; not installing")
                return

            def handler(signum, frame):
                # Signal-safe on purpose: a plain attribute write is atomic in
                # CPython and nothing here can block or re-enter a collective.
                # The process does not die from this - Python's own handling
                # of SIGTERM only terminates the process when nothing has
                # claimed the signal, and this function only reaches here
                # having confirmed that was true at install time. It stays
                # alive until `_check_emergency_signal` finishes with it, or
                # until the cloud's own grace period runs out and sends
                # SIGKILL, which no handler anywhere can catch or delay.
                self._emergency_requested = True

            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
            logger.debug("Could not install SIGTERM handler: %s", exc)

    def _emergency_coordination_active(self) -> bool:
        """Whether this run should probe for SIGTERM on other ranks at all.

        Decided once and cached, from facts identical on every rank by
        construction: the config (loaded the same way everywhere) and
        `has_sharded_models` / `spans_several_machines`, both local,
        structural questions with no rank-specific answer. That means no
        agreement collective is needed to make this safe to branch on -
        every rank reaches the same verdict without asking, which is not
        true of every flag in this class and is why this one says so.

        Off for a plain-replicated (DDP) job: rank 0 already holds the whole
        state locally and writes it without needing anything from anyone,
        which is already what happens on SIGTERM today. Off for a single
        machine too: every local process gets SIGTERM at effectively the same
        instant from the same source, so there is nothing to coordinate
        across a channel for. On only for the one case that needs it -
        sharded models split across more than one machine - which is the
        case `_final_checkpoint_is_safe` otherwise skips outright.
        """
        if self._emergency_active is None:
            from ravex._dist.collectives import spans_several_machines

            self._emergency_active = bool(
                self.config.handle_sigterm
                and self.config.emergency_coordination
                and self.registry.has_sharded_models()
                and spans_several_machines()
            )
        return self._emergency_active

    def _emergency_store_road(self):
        """The store to announce the emergency round on, or None.

        Decided once, and `False` is remembered as "asked and could not be
        had" — the same distinction `_ring_link` and `_outer` make, and for
        the same reason: a job with no rendezvous must not retry, and log,
        every step of the run.

        No store means the collective road, which is what this channel has
        always been and is not a degraded version of anything. It is also what
        `emergency_transport: collectives` asks for explicitly, for a job that
        would rather pay a collective per step than depend on a key-value
        store for its preemption path.
        """
        if self._emergency_store is False:
            return None
        if self._emergency_store is not None:
            return self._emergency_store

        wanted = str(self.config.emergency_transport)
        if wanted == "collectives":
            self._emergency_store = False
            return None

        from ravex._dist.agreement import rendezvous_store

        store = rendezvous_store()
        if store is None:
            if wanted == "store":
                logger.warning(
                    "emergency_transport=store, but there is no rendezvous "
                    "store to announce on. Falling back to the detection "
                    "collective, which is what this channel did before "
                    "GPU-111 and is not a degraded mode."
                )
            self._emergency_store = False
            return None
        self._emergency_store = store
        return store

    def _emergency_round_has_come(self) -> bool:
        """Whether this is the step the ranks agreed to save at.

        **The asynchronous half of GPU-111, and the design is one sentence:**
        the alarm is not "somebody wants to save", it is "everybody saves at
        step R". Different ranks learn it at different moments and still act
        at the same one, which is the property the per-step `all_reduce` gave
        for free and is the only thing that has to be earned back.

        What that buys is the common case. A run that is not being preempted —
        which is every step of almost every run — now pays one `check` on a
        key that is not there, answered by the store server with nobody
        waiting for anybody: 83 µs at 4 ranks against the collective's 376,
        and 92 µs against 5.8 ms when one rank is 5 ms late. A gather pays the
        straggler on any medium; the absence of a key does not.

        **The collective is not removed, it is moved to the one step where it
        is worth its cost.** At R it is what proves every rank arrived. A rank
        that never saw the alarm never posts it, the others run out of
        `emergency_timeout` — seconds, on the isolated group — and nobody
        enters the save. Without that proof, a rank that missed the alarm
        would leave the others inside a *sharded* collective on the default
        group, whose timeout is thirty minutes of billed and idle GPUs.

        **A rank that reads the alarm after R has missed it, and does not
        reschedule.** Announcing a second round would put the ranks back in
        disagreement, which is the whole thing this protocol exists to avoid.
        The fallback is exactly the behaviour of a job with the channel off:
        the periodic checkpoint stands and the preempted rank still flushes
        before it dies.

        **The rendezvous is a step number, so it holds where step numbers hold
        — a job with a collective every step.** That is DDP and FSDP, and it is
        also the only shape where this channel is on at all, since it needs
        `has_sharded_models()`. Nodes that train independently between rounds
        drift, and there a step number is not a meeting point: GPU-125, which
        is a larger problem than this one and is about the group's membership
        rather than its transport.
        """
        if self._emergency_settled:
            return False

        if self._emergency_round is None:
            from ravex._dist.agreement import announce_emergency, announced_emergency

            store = self._emergency_store_road()
            cadence = self.config.emergency_check_every
            if self._emergency_requested:
                self._emergency_round = announce_emergency(
                    store, self.registry.step_count, cadence
                )
            else:
                self._emergency_round = announced_emergency(store)
            if self._emergency_round is None:
                return False

            if self._emergency_round < self.registry.step_count:
                # Read too late to attend. Said as a warning rather than
                # swallowed: this rank is about to *not* take part in a save
                # the others are attempting, and a silent non-participation is
                # the failure that looks like nothing happening.
                self._emergency_settled = True
                logger.warning(
                    "An emergency checkpoint was announced for step %d and "
                    "this rank only read the announcement at step %d. Too "
                    "late to join it: no coordinated save here, and the last "
                    "periodic checkpoint stands.",
                    self._emergency_round,
                    self.registry.step_count,
                )
                return False

            logger.warning(
                "SIGTERM reported on this job: every rank saves together at "
                "step %d, and this rank is at %d.",
                self._emergency_round,
                self.registry.step_count,
            )

        if self.registry.step_count < self._emergency_round:
            return False

        # The announced step, reached once. Settled before the collective
        # rather than after it, because "we tried and it timed out" and "we
        # tried and it worked" are the same answer to the question of whether
        # to try again: no. Retrying would be a second rendezvous that nobody
        # announced and that the ranks which already saved are not attending.
        self._emergency_settled = True
        return True

    def _check_emergency_signal(self) -> None:
        """Look for a SIGTERM on any rank; save together if one landed.

        Reached by every rank at the same `emergency_check_every` cadence,
        unconditionally - the same discipline the checkpoint path's own
        barrier follows, extended from "the same rank count" to "the same
        step". A rank that acted on its own flag the moment it noticed it,
        instead of waiting for this shared cadence, would be exactly the
        single-rank-in-a-collective hang this mechanism exists to avoid. The
        flag is only ever *read* here; it is never what triggers the
        collective by itself.

        Any failure below - the group failing to open, the detection
        collective's own short timeout firing, anything else - is caught and
        treated as "no emergency this round". That is the whole of the
        fallback: training continues exactly as it would have if
        `emergency_coordination` were off, which is also exactly what happens
        today for every sharded job. This is a best-effort attempt, not a
        guarantee - see docs/configuration.md and the CHANGELOG for the
        measured numbers behind that, most importantly that the local handoff
        alone has been measured close to or past the whole SIGTERM budget on
        its own, before this channel's own cost.
        """
        if self._emergency_group is None:
            from ravex._dist.collectives import emergency_group

            usable, self._emergency_group = emergency_group(
                self.config.emergency_timeout
            )
            if not usable:
                # Asked once and remembered, the same as a failed byte
                # transport: retrying every cadence would just repeat the
                # same failure and the same log line for the rest of the run.
                self._emergency_active = False
                return

        if self._emergency_store_road() is not None:
            # The cheap half. On every step of every run that is not being
            # preempted this is one `check` on a key that is not there, and
            # the function below returns False without any rank having waited
            # for any other. The collective under it is reached once per run.
            if not self._emergency_round_has_come():
                return

        try:
            from ravex._dist.collectives import emergency_signalled

            signalled = emergency_signalled(
                self._emergency_requested, self._emergency_group
            )
        except Exception as exc:
            logger.warning(
                "Emergency-signal check failed (%s) - continuing without a "
                "coordinated preemption checkpoint this round.",
                exc,
            )
            return

        if not signalled:
            return

        logger.warning(
            "SIGTERM reported by at least one rank - attempting a "
            "coordinated emergency checkpoint across %d ranks. SIGTERM "
            "gives ~10s and this is not guaranteed to land in time.",
            get_world_size(),
        )
        wrote = self.checkpoint(final=True, emergency=True)
        logger.warning(
            "Emergency checkpoint %s at step %d.",
            "written" if wrote else "did not complete",
            self.registry.step_count,
        )

        # Only the rank(s) actually preempted terminate.
        #
        # What the survivors do next is *not* "go back to ordinary training",
        # though this said so until it was run on two real machines. The
        # moment the preempted rank's process exits, the training process
        # group is broken: the survivors' next collective fails with
        # `ncclRemoteError: remote process exited`. Continuing requires
        # rebuilding the group without the departed rank - see GPU-94, where
        # exactly that regroup is validated on real GPUs. This function's job
        # ends at getting the state safely onto disk; who can carry on
        # afterwards is that mechanism's question, not this one's.
        if self._emergency_requested:
            # Durability before death, and this is the whole point of the
            # emergency path rather than a tidy-up.
            #
            # `checkpoint()` only *hands off* to the background writer. Killing
            # the process here without waiting kills the writer mid-flight, and
            # on 2026-08-31 that is exactly what two real machines showed: the
            # preempted rank logged "Emergency checkpoint written at step 9"
            # and its store held steps 4 and 8 and nothing else. With
            # `per_rank` every rank's shard is needed, so the survivor's step 9
            # was unusable on its own and the resume fell back to step 8 - the
            # coordinated save bought nothing, silently, on the one rank the
            # feature exists for. See GPU-100.
            #
            # Bounded, because the budget is not ours: SIGTERM gives ~10s
            # before SIGKILL, which no handler can catch or delay, and the
            # local handoff alone has been measured close to that. A flush that
            # does not finish in time is reported and then abandoned - dying
            # with a partial write announced beats dying with it hidden, and
            # beats not dying at all while the cloud's own timer runs out.
            self._flush_before_dying()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    def _flush_before_dying(self) -> None:
        """Wait for the emergency checkpoint to reach the disk. **Bounded.**

        Split out from `_check_emergency_signal` so the one thing that must
        never raise from a signal-adjacent path is a single, obviously
        total try/except: every failure here ends in the process dying
        anyway, and an exception escaping would replace a partial checkpoint
        with a traceback and no checkpoint at all.
        """
        # `getattr` rather than `self._backend`, for the same reason the flags
        # at the top of this class have defaults: a runtime built with
        # `__new__` to exercise one method has no `_backend` at all, and this
        # path must not be the one that turns that into an AttributeError.
        backend = getattr(self, "_backend", None)
        if backend is None:  # pragma: no cover - no backend, nothing pending
            return

        started = time.time()
        try:
            backend.flush()
        except Exception as exc:
            logger.warning(
                "The emergency checkpoint could not be flushed to disk (%s). "
                "This rank is being preempted and its shard for this step may "
                "be incomplete; with per-rank checkpoints that means the whole "
                "step is unusable and the resume falls back to the previous "
                "one.",
                exc,
            )
            return

        logger.warning(
            "Emergency checkpoint flushed to disk in %.2fs before exiting.",
            time.time() - started,
        )

    def _final_checkpoint_is_safe(self) -> bool:
        """Whether a last checkpoint can be taken without risking a hang.

        For a sharded model this needs a collective, and shutdown is precisely
        when ranks stop being in lockstep: user code may already have called
        ``destroy_process_group``, or one rank may exit ahead of the others. A
        rank waiting on a gather that nobody else will join hangs until
        something kills it.

        Losing the last few steps is a known, bounded cost. A hang is not, so
        with sharded models the periodic cadence is what you get.
        """
        if not self.registry.has_sharded_models():
            return True
        logger.info(
            "Skipping the final checkpoint: the model is sharded and a gather "
            "at shutdown can hang. Last periodic checkpoint stands at step %s.",
            self._last_saved_step,
        )
        return False

    def shutdown(self) -> None:
        """Final checkpoint plus flush. Safe to call more than once."""
        with self._lock:
            if self._shutdown_done or not self._enabled:
                return
            self._shutdown_done = True

        try:
            if (
                self.config.checkpoint_on_exit
                and self.registry.step_count > 0
                and self.registry.step_count != self._last_saved_step
                and self._final_checkpoint_is_safe()
            ):
                self.checkpoint(final=True)
            if self._backend is not None:
                self._backend.close()
            if self._ring_link:
                self._ring_link.close()
                self._ring_link = None
            if self._exchange is not None:
                # Lingering, because leaving is not free: a node that shuts its
                # listener the moment it is done takes its last report with it,
                # and the peer still fetching it closes that round one
                # contributor short - see `DeltaExchange.close`.
                self._exchange.close(
                    linger=min(60.0, float(self.config.outer_deadline)),
                    expect=self._outer_peers,
                )
                self._exchange = None
            logger.info("Ravex shutdown complete at step %d", self.registry.step_count)
        except Exception as exc:
            logger.warning("Shutdown error: %s", exc)


# ─── process-wide instance ──────────────────────────────────────────

_runtime: Optional[RavexRuntime] = None
_runtime_lock = threading.Lock()


def get_runtime(create: bool = True) -> Optional[RavexRuntime]:
    """Return the process runtime, building it on first use."""
    global _runtime
    if _runtime is None and create:
        with _runtime_lock:
            if _runtime is None:
                _runtime = RavexRuntime()
    return _runtime


def reset_runtime() -> None:
    """Tear the runtime down. For tests; not part of the public API."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            if _runtime._patches is not None:
                _runtime._patches.uninstall()
            try:
                if _runtime._backend is not None:
                    _runtime._backend.close()
            except Exception:
                pass
            import torch

            if hasattr(torch.nn.Module, "_ravex_patched"):
                delattr(torch.nn.Module, "_ravex_patched")
        _runtime = None
