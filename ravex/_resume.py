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

    def try_resume(self, defer_rng: bool = False) -> bool:
        """Restore the latest checkpoint into the live objects.

        Returns True if state was applied. ``defer_rng`` is set when resuming
        from ``DataLoader.__iter__``; see :meth:`ObjectRegistry.restore_state`.
        """
        self.attempted = True

        if not self.backend.has_checkpoint():
            logger.info("No checkpoint found - starting from scratch")
            return False

        state: Optional[Dict[str, Any]] = self.backend.load_latest()
        if not state:
            logger.warning("Checkpoint present but empty - starting from scratch")
            return False

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
        logger.info(
            "Resumed at step %s (%d model(s), %d optimizer(s))",
            step,
            len(state.get("models", {})),
            len(state.get("optimizers", {})),
        )
        return True
