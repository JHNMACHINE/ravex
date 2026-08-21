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
from typing import Any, Optional

from ravex._backends import get_backend
from ravex._config import RavexConfig
from ravex._distributed import (
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
        self._last_replicated_step: Optional[int] = None
        self._leader_optimizer_id: Optional[int] = None
        self._checkpoint_due = False
        self._resume_attempted = False
        self._restoring = False
        self._last_saved_step: Optional[int] = None
        self._shutdown_done = False
        self._lock = threading.RLock()
        self._framework = "unknown"

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
            from ravex._frameworks import detect_framework

            self._framework = detect_framework()
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
        from ravex._distributed import (
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

        from ravex._replication import replication_ring

        ring = replication_ring(get_rank(), get_world_size(), local_world_size())
        replicating = self.config.replicate_every > 0 and ring is not None

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
        """
        if self.config.sharded_checkpoints != "per_rank" or get_world_size() <= 1:
            return False
        return self.registry.sharded_layout("per_rank") == "per_rank"

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

        self.registry.step_count += 1
        if self.registry.step_count % self.config.checkpoint_every == 0:
            # Not here: this is the middle of the iteration, before the LR
            # scheduler has stepped. Flag it and collect at the top of the next
            # one — see _BatchBoundaryIterator.
            self._checkpoint_due = True
            if not self.registry.dataloaders:
                # No dataloader to give us that boundary (a loop over
                # pre-batched tensors). Mid-iteration is then the best
                # available point.
                self._checkpoint_due = False
                self.checkpoint()

    def on_batch_boundary(self) -> None:
        """Called at the top of each training iteration, before the batch."""
        if not self._enabled or not self._checkpoint_due:
            return
        self._checkpoint_due = False
        self.checkpoint()

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
            resume_manager.try_resume(
                defer_rng=defer_rng, per_rank=self._per_rank_active()
            )
        except Exception as exc:
            logger.warning("Resume failed (%s) - starting from scratch", exc)
        finally:
            self._restoring = False

    # ─── checkpointing ──────────────────────────────────────────────

    def checkpoint(self, final: bool = False) -> bool:
        """Collect and hand off a checkpoint. Returns True if one was written.

        Collection runs on the training thread on purpose: the state has to be
        consistent with the step that just finished. The backend copies it and
        writes in the background, so what the loop actually pays for is the
        copy, not the I/O.
        """
        if not self._enabled:
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
        self._replicate_if_due(step)
        if not wrote:
            return False

        # The breakdown, not just the total: a handoff that costs seconds is
        # seconds the training loop is stopped, and every investigation of one
        # so far has started by re-running the job to find out which phase it
        # was. Two perf_counter calls per phase is a cheap way not to.
        logger.info(
            "Checkpoint at step %d handed off in %.3fs (%s)",
            step,
            time.perf_counter() - started,
            ", ".join("%s %.3fs" % item for item in phases.items()),
        )
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
        from ravex._distributed import gather_objects, local_world_size
        from ravex._identity import run_id_at
        from ravex._replication import (
            exchange_stores,
            local_recoveries,
            promote_copy,
            recovery_roles,
            replica_is_complete,
            replication_ring,
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

        do_send, do_receive = recovery_roles(rank, send_to, receive_from, needs, holds)

        try:
            # Backwards: the copy travels to the rank it belongs to, so this
            # sends to the peer it normally receives from, and the other way.
            ok = exchange_stores(
                held if do_send else None,
                own if do_receive else None,
                send_to=receive_from,
                receive_from=send_to,
            )
        except Exception as exc:
            logger.warning("Recovering this rank's store from a copy failed: %s", exc)
            return

        if do_receive and ok:
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
        from ravex._distributed import all_ranks_agree, local_world_size
        from ravex._replication import exchange_stores, replication_ring

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
            ok = exchange_stores(source, destination, send_to, receive_from)
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
        from ravex._distributed import local_world_size

        if self.config.replicate_every <= 0 or not self._storage_split:
            return False
        if not self._per_rank_active():
            return False
        if get_world_size() <= max(local_world_size(), 1):
            return False

        taken = step // max(self.config.checkpoint_every, 1)
        return taken % self.config.replicate_every == 0

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
        """Checkpoint on SIGTERM — the signal a preempted spot instance gets.

        Only installed when nothing else has claimed SIGTERM and only on the
        main thread; stealing the user's handler would be worse than missing
        the last few steps.
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
                logger.warning("SIGTERM received - saving before exit")
                self.shutdown()
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
            logger.debug("Could not install SIGTERM handler: %s", exc)

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
