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


class ResumeManager:
    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry
        self.attempted = False
        self.restored_step: Optional[int] = None

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

        if step < 0:
            logger.info(
                "No checkpoint that every rank holds (this rank had %s) - "
                "starting from scratch. With per-rank checkpoints a change in "
                "world size leaves the new ranks with nothing to read, and a "
                "partial resume would be worse than none.",
                local,
            )
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

        return self._apply(state, defer_rng)

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
