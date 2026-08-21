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
        from ravex._replication import replica_is_complete

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
        so "nothing here" is a value (-1) that travels through the agreement
        like any other and makes the answer -1 for everybody.
        """
        from ravex._distributed import agree_on_step, all_ranks_agree

        local = self.backend.latest_step()
        step = agree_on_step(local if local is not None else -1)

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
        from ravex._distributed import agree_on_run_id, get_rank, get_world_size
        from ravex._identity import local_run_id, write_owner

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
        from ravex._distributed import gather_visible_stores, get_world_size

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

        sharded = len(state.get("sharded", {}))
        logger.info(
            "Resumed at step %s (%d model(s), %d optimizer(s), %d sharded group(s))",
            step,
            len(state.get("models", {})),
            len(state.get("optimizers", {})),
            sharded,
        )
        return True
