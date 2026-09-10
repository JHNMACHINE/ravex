"""Resume logic.

Restoring is a best-effort operation by design. A checkpoint written by an
earlier version of the user's code may no longer line up with the objects in
memory; when that happens the mismatch is logged and training starts from
scratch, because a run that starts over is recoverable and a run that crashes
at startup on a rented GPU is money on fire.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("ravex")

#: Bumped when the on-disk state layout changes incompatibly.
STATE_VERSION = 1


def _records(seen):
    """Every owner record on every machine, once each."""
    for entry in seen or ():
        if isinstance(entry, dict):
            for record in entry.values():
                if isinstance(record, dict) and record:
                    yield record


def _torn_copy_here(config, missing) -> str:
    """The third possibility, for the machine that can see it.

    A copy caught mid-transfer is neither a store nor an absence: the directory
    is here and full of bytes, and it is disqualified because ``StoreWriter``
    removes the completeness marker before the first one lands. Saying "no
    store anywhere" then blames a machine that is powered on and holding the
    data, and sends whoever reads it looking for a hardware fault.

    Not a rare corner on a slow link. Measured on two machines with 100 Mbps
    between them: a ~670 MB store takes 102s to copy against a 244s cycle, so
    the copy is unusable 42% of the time. Losing a machine during a transfer is
    close to a coin toss there.

    Only the machine holding the copy adds this: it is the one that can tell
    the two apart, and the others say what they see.
    """
    if config is None:
        return ""
    try:
        from ravex._backends import replica_store_path, visible_replica_stores
        from ravex._dist.replication import replica_is_complete

        here = visible_replica_stores(config)
        torn = sorted(
            rank
            for rank in missing
            if rank in here
            and not replica_is_complete(replica_store_path(config, rank))
        )
    except Exception:  # pragma: no cover - diagnosis must not cost the run
        return ""

    if not torn:
        return ""
    return (
        " This machine does hold a copy of rank(s) %s, but it was interrupted "
        "part-way through and cannot be trusted - the bytes are here and are "
        "not a checkpoint." % ", ".join(str(rank) for rank in torn)
    )


def _where_written(seen, ranks) -> str:
    """" (written on node0, node1)", or nothing if the stores never said.

    Naming the machine is the difference between knowing the placement moved
    and knowing which box to put back. Empty when the stores predate owner
    records, so an older checkpoint degrades the sentence rather than breaking
    it.
    """
    wanted = set(ranks)
    hosts = []
    for entry in seen or ():
        if not isinstance(entry, dict):
            continue
        for rank, record in entry.items():
            if rank in wanted and isinstance(record, dict):
                host = record.get("host")
                if host and host not in hosts:
                    hosts.append(str(host))
    if not hosts:
        return ""
    return " (written on %s)" % ", ".join(sorted(hosts))


def _several_runs(seen) -> str:
    """A sentence about two histories sharing these disks, or nothing.

    A run that starts from scratch beside an older one writes into the same
    directory names, and from then on the disk holds two unrelated trainings.
    Reproduced on 2026-08-19, where a later restart resumed the accidental one
    while the original sat beside it. The ids are what make it sayable.
    """
    ids = {str(record["run_id"]) for record in _records(seen) if record.get("run_id")}
    if len(ids) < 2:
        return ""
    return (
        " These machines hold stores from %d different runs (%s), so some of "
        "what is here belongs to a training this one is not continuing."
        % (len(ids), ", ".join(sorted(ids)))
    )


def _per_rank_groups(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{group: {"model": tree, "optimizer": tree}}`` for the per-rank groups.

    Groups written with ``sharded_checkpoints=gather`` are skipped: their state
    is already topology-independent, so there is nothing in them to reshape and
    reshaping it would be an error.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for key, saved in (snapshot.get("sharded") or {}).items():
        if not isinstance(saved, dict) or saved.get("layout") != "per_rank":
            continue
        groups[key] = {
            "model": saved.get("model"),
            "optimizer": saved.get("optimizer"),
        }
    return groups


def _paths_of(live_groups, new_extents, rank) -> Dict[Any, Dict[Any, list]]:
    """Every sharded tensor, with the length each new rank holds of it.

    Keyed by ``(group, half)`` then by path, and the value is one length per
    new rank in rank order — which is precisely what :func:`plan_reshard` takes
    as its second argument.

    A tensor that some rank does not report is refused rather than planned
    around. It means the ranks are not running the same model, and a plan built
    from a partial list would put the boundaries in the wrong places for
    everybody, not only for the rank that was quiet.
    """
    world = len(new_extents)
    out: Dict[Any, Dict[Any, list]] = {}
    for group in (new_extents[rank] or {}):
        key, half = group
        if key not in live_groups:  # pragma: no cover - built from live_groups
            continue
        for path in new_extents[rank][group]:
            lengths = []
            for r in range(world):
                extents = (new_extents[r] or {}).get(group) or {}
                if path not in extents:
                    raise ValueError(
                        "rank %d has no shard for %s in group %s: the ranks are "
                        "not running the same model" % (r, path, key)
                    )
                lengths.append(int(extents[path]))
            out.setdefault(group, {})[path] = lengths
    return out


def _extent(extents, group, path, q) -> int:
    """One old rank's shard length, or a message naming what is missing."""
    found = (extents or {}).get(group)
    if found is None or path not in found:
        key, half = group
        raise ValueError(
            "rank %d's store has no %s shard for %s in group %s, so where the "
            "other shards start cannot be worked out" % (q, half, path, key)
        )
    return int(found[path])


def _drop_per_rank_randomness(state: Dict[str, Any], old_world: int, world: int) -> None:
    """Take out the two things that do not survive a change of world size.

    **The RNG.** There were ``old_world`` generator states and there are now
    ``world`` ranks. There is no correct mapping — restoring rank 0's onto
    every rank would have them all draw the same numbers, which is worse than
    starting from the seeding the script already did. Dropping it lets
    ``restore_state`` leave the live generators alone, and the log line says so
    rather than leaving it to be discovered.

    **The data position.** The sampler partitions the epoch by world size, so
    ``position`` counts this rank's batches and means a different amount of
    data at a different world size. It is rescaled to preserve the *total*
    consumed, which is the honest half of the promise: the resumed run
    continues the model, not the run. Every sample is still seen once per
    epoch; the order is not the order the original run would have taken, and
    it cannot be.
    """
    if state.pop("rng", None) is not None:
        logger.info(
            "Resharded resume: the saved RNG states belong to a %d-rank run "
            "and are not restored. Random draws continue from this run's own "
            "seeding.",
            old_world,
        )

    loaders = state.get("dataloaders") or {}
    for key, saved in loaders.items():
        if not isinstance(saved, dict) or "position" not in saved:
            continue
        try:
            position = int(saved["position"])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        rescaled = (position * old_world) // max(world, 1)
        saved["position"] = rescaled
        logger.info(
            "Resharded resume: dataloader %s resumes %d batch(es) in rather "
            "than %d, so the run has consumed the same amount of data at %d "
            "ranks as it had at %d. The order within the epoch differs from "
            "the original run and cannot be made to match.",
            key,
            rescaled,
            position,
            world,
            old_world,
        )


class ResumeManager:
    def __init__(self, backend, registry, config=None):
        self.backend = backend
        self.registry = registry
        # The *base* config, before `_per_rank_config` appended `rank_<n>` to
        # the path. Needed to look at the sibling stores rather than only this
        # rank's own, which is what tells a fresh run apart from a checkpoint
        # sitting on the wrong machine.
        self.config = config
        self.attempted = False
        self.restored_step: Optional[int] = None
        #: Framework state carried by the checkpoint that was applied, exactly
        #: as it was written: ``{"framework": ..., "state": ...}``, or None.
        #: Read back out by the runtime, which hands it to the adapter — this
        #: class has no idea which framework is running and should not grow
        #: one. See :meth:`RavexRuntime._restore_extra_state`.
        self.restored_extra: Optional[Dict[str, Any]] = None
        #: World size that wrote the stores, once `_reshard_wanted` has looked.
        self._old_world: Optional[int] = None
        #: Set once the ranks have agreed; see :meth:`_settle_run_identity`.
        self.run_id: Optional[str] = None

    def try_resume(self, defer_rng: bool = False, per_rank: bool = False) -> bool:
        """Restore the latest checkpoint into the live objects.

        Returns True if state was applied. ``defer_rng`` is set when resuming
        from ``DataLoader.__iter__``; see :meth:`ObjectRegistry.restore_state`.
        ``per_rank`` says the checkpoint lives in one store per rank, which
        makes "the latest" a question the ranks have to answer together.
        """
        self.attempted = True

        if per_rank:
            return self._resume_per_rank(defer_rng)

        if not self.backend.has_checkpoint():
            logger.info("No checkpoint found - starting from scratch")
            return False

        state: Optional[Dict[str, Any]] = self.backend.load_latest()
        if not state:
            logger.warning("Checkpoint present but empty - starting from scratch")
            return False

        return self._apply(state, defer_rng)

    def _resume_per_rank(self, defer_rng: bool) -> bool:
        """Resume from per-rank stores, at a step every rank actually holds.

        Each rank writes on its own, so a kill lands between two ranks' writes
        as often as not: rank 3 has step 8 on disk and rank 5 stopped at 6.
        Restoring each rank's newest would build a model out of two different
        moments in training — every shard individually valid, the whole thing
        wrong, and nothing downstream able to notice.

        Both collectives here are unconditional. A rank that returned early
        because its own store was empty would strand the others waiting on it,
        so "nothing here" is a value (``NOTHING_TO_RESUME``) that travels
        through the agreement like any other and makes the answer that for
        everybody. Since GPU-111 it is also what the agreement returns when a
        rank never answers at all, which lands in the same branch below.
        """
        from ravex._dist.collectives import (
            NOTHING_TO_RESUME,
            agree_on_step,
            all_ranks_agree,
        )

        # Before anything else, because at a different world size "this rank's
        # own store" is not this rank's own history: after a shrink `rank_5`
        # may not exist, and after a growth it belongs to a rank that no longer
        # runs here. The mismatch is detected always and acted on only when
        # asked — see `reshard_on_resume`.
        if self._reshard_wanted():
            return self._resume_resharded(defer_rng)

        local = self.backend.latest_step()
        step = agree_on_step(local if local is not None else NOTHING_TO_RESUME)

        resumed = False
        try:
            if step < 0:
                # Reached by every rank or by none: `step` is the agreed
                # minimum, so it is the same number everywhere. Which is what
                # makes it safe to run another collective in here.
                self._explain_nothing_to_resume(local)
                return False
            if local != step:
                logger.info(
                    "Resuming at step %s, the newest every rank has (this rank had %s)",
                    step,
                    local,
                )

            state = self.backend.load_step(step)
            if not all_ranks_agree(bool(state)):
                logger.warning(
                    "Step %s could not be read on every rank - starting from "
                    "scratch on all of them",
                    step,
                )
                return False

            resumed = self._apply(state, defer_rng)
            return resumed
        finally:
            # After the verdict, because whether this run continues an existing
            # history or begins a new one is exactly what the identity records.
            # Every rank arrives here, and with the same `resumed`: each return
            # above is taken by all of them together or by none.
            self._settle_run_identity(resumed)

    # ─── resuming onto a different number of ranks ──────────────────

    def _old_world_size(self) -> Optional[int]:
        """How many ranks wrote the stores on this machine, if they agree.

        Read from the owner records rather than from the number of ``rank_<n>``
        directories: the directories on *this* machine are only the ones it
        happens to hold, and after a shrink that is a fraction of the run.
        ``write_owner`` records the world size beside every store, which is the
        whole world's answer written down N times.

        ``None`` when nothing says, and equally when the records disagree —
        two runs' leftovers sharing a directory is a real case (see
        :func:`_several_runs`), and guessing which of them is being resumed is
        not a decision to make silently.
        """
        if self.config is None or self.config.storage.is_remote:
            return None

        try:
            from ravex._backends import visible_store_owners

            records = visible_store_owners(self.config)
        except Exception:  # pragma: no cover - never worth the run
            return None

        sizes = {
            int(record["world_size"])
            for record in records.values()
            if isinstance(record, dict) and record.get("world_size")
        }
        if len(sizes) != 1:
            if len(sizes) > 1:
                logger.warning(
                    "The stores here were written by runs of %s ranks. Which "
                    "one this job continues is not something to guess at, so "
                    "no resharding is attempted.",
                    " and ".join(str(size) for size in sorted(sizes)),
                )
            return None
        return sizes.pop()

    def _reshard_wanted(self) -> bool:
        """Whether this resume has to reshape shards, and may. **Collective.**

        Detection is unconditional and acting on it is not. A resume that
        reshards silently is a resume that silently succeeds when the launcher
        was misconfigured and started three ranks where the job wants four: the
        run continues, the loss curve looks plausible, and nothing anywhere
        says the world shrank. So the mismatch is always said out loud, and
        turning it into an action is ``reshard_on_resume``.

        The agreement at the end is what keeps this from stranding a
        collective. Ranks decide from what their own machine holds, and on a
        job spread over several machines they can reach different verdicts;
        ``all_ranks_agree`` turns any disagreement into "everybody takes the
        ordinary path", which is the branch that is safe to take alone.
        """
        from ravex._dist.collectives import all_ranks_agree, get_world_size

        world = get_world_size()
        old = self._old_world_size()
        enabled = bool(getattr(self.config, "reshard_on_resume", False))
        remote = self.config is not None and self.config.storage.is_remote

        wanted = False
        if old is not None and old != world:
            if not enabled:
                logger.warning(
                    "This checkpoint was written per-rank by a %d-rank run and "
                    "this one has %d. Resuming it means rebuilding every shard "
                    "out of the old ones, which Ravex will do on request: set "
                    "reshard_on_resume=true (RAVEX_RESHARD_ON_RESUME=1). It is "
                    "off by default so that a launcher that started the wrong "
                    "number of ranks fails visibly instead of training on.",
                    old,
                    world,
                )
            elif remote:  # pragma: no cover - needs a bucket
                logger.warning(
                    "reshard_on_resume is set and the world changed from %d to "
                    "%d, but this store is remote and resharding is only "
                    "implemented for local storage so far. Starting from "
                    "scratch rather than half-resuming.",
                    old,
                    world,
                )
            else:
                wanted = True

        # Unconditional, on every rank, exactly once: the branch below has
        # collectives in it and the branch beside it has different ones.
        agreed = all_ranks_agree(wanted)
        if wanted and not agreed:  # pragma: no cover - needs several machines
            logger.warning(
                "Only some ranks can see a topology change from here, so no "
                "resharding is attempted. This is what a per-rank checkpoint "
                "split across machines looks like from one of them; remote or "
                "shared storage is what makes the picture whole."
            )
        self._old_world = old if agreed else None
        return agreed

    def _resume_resharded(self, defer_rng: bool) -> bool:
        """Rebuild this rank's shards out of a differently-shaped checkpoint.

        The shape of it, in order: agree on a step every *old* store holds,
        measure both topologies, plan, read each old store once more taking
        only the slices this rank needs, and hand the result to the ordinary
        apply path as though it had been written at this world size.

        Step agreement moves to the old stores and off the live process group,
        and it has to: the live group answers "what does rank r hold", and
        after a shrink rank 5's store is not held by anybody — it is read from
        a copy, by whichever rank got there. Every rank reads the same set of
        stores here, so every rank reaches the same number without a
        collective, and the ``all_ranks_agree`` below is a check rather than
        the decision.

        **Memory.** Each rank reads the old stores one at a time and keeps only
        the slices it needs, so the peak is one old snapshot plus this rank's
        new shards — not the whole checkpoint. The global tensor is never
        materialised anywhere, which is the entire point of per-rank
        checkpointing and survives this feature intact.
        """
        from ravex._dist.collectives import all_ranks_agree

        resumed = False
        try:
            state = self._resharded_state()
            if not all_ranks_agree(state is not None):
                logger.warning(
                    "The reshard could not be completed on every rank - "
                    "starting from scratch on all of them, because a partial "
                    "resume would be worse than none."
                )
                return False
            assert state is not None  # all_ranks_agree said so
            resumed = self._apply(state, defer_rng)
            return resumed
        finally:
            self._settle_run_identity(resumed)

    def _resharded_state(self) -> Optional[Dict[str, Any]]:
        """The state this rank should restore, or None with the reason logged.

        Split out from :meth:`_resume_resharded` so that every way of failing
        is a ``return None`` in one place and the caller can turn any of them
        into the same all-or-nothing verdict.

        **Every early exit here is taken by all the ranks or by none.** The
        checks before the first collective are answered from what each machine
        holds, and on a job spread over several machines the ranks can answer
        them differently — one machine reaching every old store through its
        copies while another reaches half. A rank that returned early there
        would leave the rest inside ``local_sharded_state`` waiting for a
        participant that has gone home. So the local verdicts are pooled first,
        in one gather, and they have to *match* rather than merely all be
        favourable: two ranks resuming from two different steps is worse than
        two ranks not resuming.
        """
        from ravex._backends import reachable_rank_stores
        from ravex._dist.collectives import (
            gather_objects,
            get_rank,
            get_world_size,
            local_sharded_state,
            shard_extents,
        )
        from ravex._dist.reshard import ReshardUnsupported

        old_world = self._old_world
        if old_world is None:  # pragma: no cover - guarded by `_reshard_wanted`
            return None
        world, rank = get_world_size(), get_rank()

        # Every old shard is needed, including the ones this rank will not read
        # a single row from: the offsets of the shards it *does* read are the
        # running sum of all the lengths before them. A missing store is not a
        # slice missing from one rank's tensor, it is an unknown alignment for
        # everybody.
        wanted = set(range(old_world))
        reachable = reachable_rank_stores(self.config, wanted)
        missing = sorted(wanted - reachable)
        if missing:
            logger.warning(
                "Cannot reshard %d ranks onto %d: no store and no complete "
                "copy here for rank(s) %s. Rebuilding the others without them "
                "would produce tensors with a band of uninitialised rows - "
                "every shard valid, the model wrong, and nothing downstream "
                "able to notice - so this starts from scratch instead.%s",
                old_world,
                world,
                ", ".join(str(q) for q in missing),
                _torn_copy_here(self.config, missing),
            )

        step = None if missing else self._agree_on_old_step(sorted(wanted))

        # The pooling described above, and the only collective before the live
        # layout is read. What travels is a couple of integers per rank.
        view = (step, sorted(reachable)) if step is not None else None
        views = gather_objects(view)
        if any(other != views[0] for other in views):
            logger.warning(
                "The ranks do not see the same old stores from where they are, "
                "so no reshard is attempted: resuming half of them from one "
                "step and half from another would be worse than starting from "
                "scratch. This is what a per-rank checkpoint split across "
                "machines looks like; moving the old shards between machines "
                "is not implemented yet. Remote or shared storage makes the "
                "picture whole today."
            )
            return None
        if views[0] is None:
            # Every rank agreed there is nothing to do; the reason is already
            # in the log, once per rank, from the branch that decided it.
            return None
        assert step is not None  # `views[0]` being set says so

        # The live layout, read for its shapes only. Collective, and every rank
        # reaches it: the returns above are taken by all of them together.
        live_groups = {}
        for key, model, optimizers in self.registry.sharded_groups():
            model_live, optimizer_live = local_sharded_state(model, optimizers)
            live_groups[key] = {"model": model_live, "optimizer": optimizer_live}

        mine: Optional[Dict[Any, Any]]
        try:
            mine = {
                (key, half): shard_extents(tree)
                for key, halves in live_groups.items()
                for half, tree in halves.items()
            }
        except (ValueError, ReshardUnsupported) as exc:
            logger.warning("Cannot reshard this checkpoint: %s", exc)
            # Not a `return`: the gather below is the next collective and the
            # other ranks are already on their way into it. Refusing is said
            # *through* it, as a None among the extents.
            mine = None

        # One dict of integers per rank. This is the only collective that
        # carries anything about the tensors, and it carries their lengths.
        new_extents = gather_objects(mine)
        if any(extents is None for extents in new_extents):
            return None

        try:
            return self._stitch(
                step, old_world, rank, world, live_groups, new_extents
            )
        except (ValueError, ReshardUnsupported) as exc:
            logger.warning(
                "Reshard from %d ranks to %d failed: %s. Starting from scratch.",
                old_world,
                world,
                exc,
            )
            return None
        except Exception as exc:  # pragma: no cover - a resume never kills a run
            logger.warning(
                "Reshard from %d ranks to %d hit an unexpected error (%s). "
                "Starting from scratch.",
                old_world,
                world,
                exc,
            )
            return None

    def _agree_on_old_step(self, old_ranks) -> Optional[int]:
        """The newest step every old store holds, or None having said why.

        Opening each store to ask is the cost of the question; ``latest_step``
        reads a manifest and no tensors, so it is a cost paid in file reads
        rather than in gigabytes.
        """
        from ravex._backends import open_rank_store

        steps = []
        for q in old_ranks:
            store = open_rank_store(self.config, q)
            if store is None:  # pragma: no cover - reachability already checked
                logger.warning("Rank %d's store went away mid-resume", q)
                return None
            try:
                steps.append(store.latest_step())
            finally:
                store.close()

        if any(step is None for step in steps):
            blank = [q for q, step in zip(old_ranks, steps) if step is None]
            logger.warning(
                "Nothing to reshard: the store(s) for rank(s) %s hold no "
                "checkpoint. Starting from scratch.",
                ", ".join(str(q) for q in blank),
            )
            return None

        step = min(int(s) for s in steps)
        if len(set(steps)) > 1:
            logger.info(
                "Resharding from step %s, the newest every old store holds "
                "(they range from %s to %s)",
                step,
                min(int(s) for s in steps),
                max(int(s) for s in steps),
            )
        return step

    def _stitch(
        self, step, old_world, rank, world, live_groups, new_extents
    ) -> Dict[str, Any]:
        """Read the old stores and assemble this rank's state. Two passes.

        The first pass measures — how long is each old shard — and keeps
        nothing. The second cuts out the intervals the plan asked for. Reading
        twice buys the memory bound: knowing where a shard starts needs every
        length before it, so a single pass would mean holding every old
        snapshot at once, which is the whole checkpoint per rank.

        Only the second pass reads tensors. The first asks each store to
        *describe* itself — a few kilobytes of template carrying the structure,
        the shapes and the placements — because measuring never needed the
        bytes and loading them was the larger half of a reshard's cost: N
        complete reads of N checkpoints, every one of them discarded. A backend
        that cannot describe still gets loaded, so nothing depends on the
        optimisation being available.
        """
        from ravex._dist.collectives import (
            build_resharded_tree,
            shard_extents,
            take_shard_slices,
        )
        from ravex._dist.reshard import check_covered, plan_reshard

        old_ranks = list(range(old_world))
        base = old_ranks[0]

        # ── pass one: lengths ───────────────────────────────────────
        old_extents: Dict[int, Any] = {}
        for q in old_ranks:
            snapshot = self._describe_old(q, step)
            old_extents[q] = {
                (key, half): shard_extents(tree)
                for key, halves in _per_rank_groups(snapshot).items()
                for half, tree in halves.items()
            }
            del snapshot

        # ── the plan, one tensor at a time ──────────────────────────
        wanted: Dict[int, Dict[Any, Any]] = {q: {} for q in old_ranks}
        for group, paths in _paths_of(live_groups, new_extents, rank).items():
            if group not in old_extents[base]:
                # A group the live model shards and the checkpoint holds under
                # `gather`. Its state is already topology-independent, so there
                # is nothing to reshape — and a checkpoint mixing the two
                # layouts should not lose the half that was fine.
                continue
            for path, new_lengths in paths.items():
                old_lengths = [
                    _extent(old_extents[q], group, path, q) for q in old_ranks
                ]
                pieces = plan_reshard(old_lengths, new_lengths)[rank]
                check_covered(pieces, new_lengths[rank], "%s %s" % (group, path))
                for q, (start, stop), _ in pieces:
                    wanted[q].setdefault(group, {}).setdefault(path, []).append(
                        (start, stop)
                    )

        # ── pass two: the slices, and the base snapshot ─────────────
        slices: Dict[Any, Dict[Any, list]] = {}
        state: Optional[Dict[str, Any]] = None
        for q in old_ranks:
            snapshot = self._load_old(q, step)
            groups = _per_rank_groups(snapshot)
            for group, paths in wanted[q].items():
                key, half = group
                tree = groups.get(key, {}).get(half)
                if tree is None:
                    raise ValueError(
                        "rank %d's store has no %s state for group %s" % (q, half, key)
                    )
                for path, parts in take_shard_slices(tree, paths).items():
                    slices.setdefault(group, {}).setdefault(path, []).extend(parts)
            if q == base:
                state = snapshot
            else:
                del snapshot

        if state is None:  # pragma: no cover - `base` is always in `old_ranks`
            raise ValueError("the base store produced nothing to build on")

        # ── assembly ────────────────────────────────────────────────
        for key, saved in state.get("sharded", {}).items():
            if saved.get("layout") != "per_rank":
                continue
            live = live_groups.get(key)
            if live is None:
                # A group in the checkpoint that this run does not have. The
                # registry already reports these; leaving it alone keeps that
                # one report rather than raising a second, different one here.
                continue
            for half in ("model", "optimizer"):
                saved[half] = build_resharded_tree(
                    saved[half],
                    live[half],
                    slices.get((key, half), {}),
                )
            # Written by a run of `old_world`, and now shaped for this one.
            # Said in the checkpoint so the apply path takes the ordinary
            # branch instead of the "cannot be reshaped" one.
            saved["world_size"] = world
            saved["rank"] = rank
            saved["resharded_from"] = old_world

        _drop_per_rank_randomness(state, old_world, world)
        return state

    def _load_old(self, q: int, step: int) -> Dict[str, Any]:
        """One old store's snapshot at ``step``, opened and closed around it."""
        from ravex._backends import open_rank_store

        store = open_rank_store(self.config, q)
        if store is None:  # pragma: no cover - reachability already checked
            raise ValueError("rank %d's store is no longer readable" % q)
        try:
            snapshot = store.load_step(step)
        finally:
            store.close()
        if not snapshot:
            raise ValueError("rank %d's store holds nothing at step %s" % (q, step))
        return snapshot

    def _describe_old(self, q: int, step: int) -> Dict[str, Any]:
        """One old store's snapshot at ``step``, measured rather than read.

        The same tree :meth:`_load_old` returns with something carrying a
        ``shape`` where each tensor would be, which is all the measuring pass
        of :meth:`_stitch` looks at. A backend that cannot answer that way is
        loaded instead — the answer is the same, it costs the checkpoint.
        """
        from ravex._backends import open_rank_store

        store = open_rank_store(self.config, q)
        if store is None:  # pragma: no cover - reachability already checked
            raise ValueError("rank %d's store is no longer readable" % q)
        try:
            described = store.describe_step(step)
            if described:
                return described
            # Not an error and not worth a warning: `torch_save` stores have no
            # way to describe themselves, and this is their ordinary path.
            logger.debug(
                "Rank %d's store cannot describe step %s; reading it to measure it",
                q,
                step,
            )
            snapshot = store.load_step(step)
        finally:
            store.close()
        if not snapshot:
            raise ValueError("rank %d's store holds nothing at step %s" % (q, step))
        return snapshot

    def _settle_run_identity(self, resumed: bool) -> None:
        """Agree on which run this is, and record it beside this rank's store.

        A run that resumed **continues** the history it read, so it keeps that
        history's id. A run that started from scratch is a new one, even when
        it writes into directories an older run left behind — and that is the
        case worth separating. Reproduced on 2026-08-19: a restart with a
        different rank-to-node placement began again from zero next to a
        checkpoint at step 24, and a later restart resumed the accidental
        history instead. With distinct ids the two are tellable apart; without,
        nothing on disk says they are different runs at all.

        Inheritance is therefore offered only when this rank actually resumed.
        The agreement then prefers an inherited id over an invented one, so a
        run that lost one machine and resumed on the rest keeps its name rather
        than taking the replacement's.

        Best-effort throughout: an identity that cannot be worked out or
        written costs a future diagnosis, not this run.
        """
        from ravex._backends import per_rank_store_path
        from ravex._dist.collectives import agree_on_run_id, get_rank, get_world_size
        from ravex._dist.identity import local_run_id, write_owner

        if self.config is None:
            return

        try:
            store = (
                None
                if self.config.storage.is_remote
                else per_rank_store_path(self.config, get_rank())
            )
            provenance, candidate = local_run_id(
                self.config, store if resumed else None
            )
        except Exception:  # pragma: no cover - never worth the run
            return

        # Collective, and unconditional for the same reason as the two above.
        self.run_id = agree_on_run_id(provenance, candidate)

        if store is not None:
            write_owner(store, self.run_id, get_rank(), get_world_size())

    def _explain_nothing_to_resume(self, local) -> None:
        """Say *why* nothing is being resumed, because it is not one situation.

        Three of them share this exit, and they call for different reactions:

        * nothing was ever written, which is an ordinary first run;
        * the previous run had more ranks, so the new ones have no store;
        * every store exists, but on machines other than the ones now hosting
          those ranks. Nothing is lost and nothing is reachable.

        The third is the one worth spelling out. With local storage a per-rank
        checkpoint can only be read back where it was written, and neither
        torchrun nor SLURM promises that a node gets the same rank range
        twice. Left unexplained it reads as a fresh run — the training simply
        starts over, exit code zero, with hours of checkpoints intact one
        machine away.

        The collective below is one small object per rank. See
        :func:`gather_visible_stores` on why it is not the way to move the
        checkpoints themselves.
        """
        from ravex._backends import visible_store_owners
        from ravex._dist.collectives import gather_visible_stores, get_world_size

        try:
            mine = visible_store_owners(self.config) if self.config is not None else {}
        except Exception:  # pragma: no cover - diagnosis must not cost the run
            mine = {}

        # Unconditional: a rank that skipped it would strand the others here,
        # at the one moment they are all waiting to agree on something.
        seen = gather_visible_stores(mine)
        everywhere = set().union(*seen) if seen else set()
        world = get_world_size()

        if not everywhere:
            logger.info(
                "No checkpoint found on any node (this rank had %s) - "
                "starting from scratch.",
                local,
            )
            return

        # A store whose rank is running somewhere that cannot see it.
        stranded = sorted(
            rank
            for rank in range(min(world, len(seen)))
            if rank in everywhere and rank not in seen[rank]
        )
        # And a rank with no store anywhere, which is two situations wearing
        # one face: a machine that is gone, or a run that had fewer ranks.
        #
        # The rank numbers separate them. Stores are written by a contiguous
        # range starting at 0, so a *gap* — nothing for rank 2 while rank 3 is
        # here — cannot come from a smaller run. It can only be a store that
        # existed and is no longer reachable, which on rented hardware means
        # the machine holding it was taken away. Missing ranks above every
        # store present are genuinely ambiguous, and get said as such.
        missing = sorted(set(range(world)) - everywhere)
        highest = max(everywhere)
        gap = [rank for rank in missing if rank < highest]
        beyond = sorted(rank for rank in everywhere if rank >= world)

        if stranded:
            logger.warning(
                "A checkpoint exists but this topology cannot reach it - "
                "starting from scratch. The stores for rank(s) %s are on "
                "machines other than the ones now running them%s, which is "
                "what happens when the nodes come back in a different order. "
                "Nothing was lost: rerun with the previous rank-to-node "
                "placement, or use remote storage, which does not depend on "
                "where a rank lands.%s",
                ", ".join(str(rank) for rank in stranded),
                _where_written(seen, stranded),
                _several_runs(seen),
            )
            return

        if gap:
            logger.warning(
                "A checkpoint exists but part of it is gone - starting from "
                "scratch. No store anywhere for rank(s) %s, while rank %d is "
                "present, so this is not a smaller run that wrote fewer: the "
                "machine that held them is not in this job. Their shards "
                "cannot be rebuilt from the others, and the rest of the "
                "checkpoint is unusable without them. Remote storage, or a "
                "copy on a second machine, is what survives losing one.%s",
                ", ".join(str(rank) for rank in gap),
                highest,
                _several_runs(seen),
            )
            return

        if missing:
            logger.info(
                "No store anywhere for rank(s) %s - starting from scratch. "
                "Either the run that wrote these had fewer ranks, or the "
                "machines holding the last ones are gone; from here the two "
                "look the same.%s%s%s",
                ", ".join(str(rank) for rank in missing),
                _torn_copy_here(self.config, missing),
                (
                    " The stores that do exist go up to rank %d, past this "
                    "run's %d." % (max(beyond), world - 1)
                    if beyond
                    else ""
                ),
                _several_runs(seen),
            )
            return

        # Every rank can see its own store and still none of them could name a
        # step: the directories are there and hold nothing a backend will read.
        logger.info(
            "No checkpoint that every rank holds (this rank had %s) - "
            "starting from scratch, because a partial resume would be worse "
            "than none.%s",
            local,
            _several_runs(seen),
        )

    def _apply(self, state: Dict[str, Any], defer_rng: bool) -> bool:
        version = state.get("ravex_version", 0)
        if version > STATE_VERSION:
            logger.warning(
                "Checkpoint was written by a newer Ravex (state v%s > v%s) - "
                "starting from scratch",
                version,
                STATE_VERSION,
            )
            return False

        step = state.get("step", 0)
        self.registry.restore_state(state, defer_rng=defer_rng)
        self.restored_step = self.registry.step_count
        # Read, not applied. A reshard reaches here with a stitched state whose
        # non-sharded halves come from one old rank's snapshot, so this arrives
        # on that path too.
        self.restored_extra = state.get("extra")

        sharded = len(state.get("sharded", {}))
        logger.info(
            "Resumed at step %s (%d model(s), %d optimizer(s), %d sharded group(s))",
            step,
            len(state.get("models", {})),
            len(state.get("optimizers", {})),
            sharded,
        )
        return True
