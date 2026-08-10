"""Training-framework detection.

Ravex intercepts PyTorch itself, so anything built on top of PyTorch —
HuggingFace ``Trainer``, Lightning, Accelerate — is already covered by the
optimizer and dataloader hooks. Detection exists so that checkpoints record
what produced them, and so the runtime can warn when a framework's own
checkpointing would duplicate the work.

Framework-specific state (``Trainer.state``, Lightning's loop counters) is a
Sprint 2 concern; the adapters below are the seam it will plug into.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

logger = logging.getLogger("ravex")


def detect_framework() -> str:
    """Identify the training stack from what is already imported.

    Only ``sys.modules`` is inspected — importing candidates to test for them
    would be both slow and rude.
    """
    modules = sys.modules

    if "transformers" in modules:
        return "huggingface"
    if "lightning" in modules or "pytorch_lightning" in modules:
        return "lightning"
    if "accelerate" in modules:
        return "accelerate"
    if "deepspeed" in modules:
        return "deepspeed"
    return "vanilla"


class FrameworkAdapter:
    """Default adapter: intercept everything, add nothing."""

    name = "vanilla"

    def should_intercept_step(self) -> bool:
        return True

    def collect_extra_state(self) -> Dict[str, Any]:
        return {}

    def restore_extra_state(self, state: Dict[str, Any]) -> None:
        return None


class HuggingFaceAdapter(FrameworkAdapter):
    """HuggingFace ``Trainer``.

    ``Trainer`` has its own checkpointing, driven by ``save_steps``. Running
    both wastes I/O without adding safety, so the recommendation is to set
    ``save_strategy="no"`` and let Ravex handle it — its checkpoints are
    delta-compressed and survive the instance disappearing.
    """

    name = "huggingface"


class LightningAdapter(FrameworkAdapter):
    """PyTorch Lightning.

    Lightning's ``ModelCheckpoint`` callback covers the model but not the RNG
    or the dataset position, which is what makes a preemption-resume exact.
    """

    name = "lightning"


_ADAPTERS = {
    "huggingface": HuggingFaceAdapter,
    "lightning": LightningAdapter,
}


def get_adapter(framework: str) -> FrameworkAdapter:
    return _ADAPTERS.get(framework, FrameworkAdapter)()
