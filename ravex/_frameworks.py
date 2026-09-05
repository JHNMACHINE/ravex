"""Training-framework detection.

Ravex intercepts PyTorch itself, so anything built on top of PyTorch —
HuggingFace ``Trainer``, Lightning, Accelerate — is already covered by the
optimizer and dataloader hooks. Detection exists so that checkpoints record
what produced them, and so the runtime can warn when a framework's own
checkpointing would duplicate the work.

**DeepSpeed was on that list and should not have been.** Intercepting
``Optimizer.step`` is enough only while the optimizer holds the model's
parameters, and ZeRO hands the base optimizer flat partition buffers instead.
Measured on 2026-08-31: a four-parameter module under ZeRO stage 1, one
parameter owned by ``FusedAdam``, none of them the module's. Both of the
registry's tests for "is this a model" then answered no — the user's model
because it is contained in DeepSpeed's engine, the engine because nothing owns
its parameters — and Ravex wrote checkpoints holding an optimizer, a dataloader
and no weights, reporting a successful handoff each time.

It works now, and it works by :meth:`~ravex._registry.ObjectRegistry._root_models`
noticing that the ownership question has no answer for this run rather than
reading it as a no. The lesson is narrower than "DeepSpeed is special": *a
framework that replaces the optimizer's parameters is outside what hooking
``Optimizer.step`` can see*, and the argument from where the hooks are is an
argument. ``integration/test_deepspeed.py`` is the evidence.

Framework-specific state — ``Trainer.state``, Lightning's loop counters — is
what the adapters below are for, and since GPU-69 the runtime calls them:
:meth:`FrameworkAdapter.should_intercept_step` once at activation,
:meth:`~FrameworkAdapter.collect_extra_state` inside
``RavexRuntime.checkpoint``, :meth:`~FrameworkAdapter.restore_extra_state` on
the way out of a resume.

No adapter overrides any of the three yet, and connecting them first is
deliberate: the wiring is testable on its own — a checkpoint that carries extra
state, one that does not, one written under another framework — while
behaviour bolted onto absent wiring is a rewrite of both at once. Read the
contract on each method before writing the first override.
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
    """Default adapter: intercept everything, add nothing.

    Three methods, and the runtime holds each to a contract:

    ``should_intercept_step``
        Asked **once**, at activation, before any step has run. ``False`` says
        this framework's ``optimizer.step()`` is not the training loop's step
        and Ravex must not count it — at which point advancing
        ``registry.step_count`` becomes the adapter's job, and until something
        does it nothing is ever checkpointed. Say ``False`` only with the rest
        of that sentence in hand.

    ``collect_extra_state``
        Called on the training thread with the loop stopped, inside the same
        window as the sharded collective, with every rank at the same point.
        So: no I/O, nothing slow, and **no collective of its own** unless every
        rank makes the same call — a rank that enters a collective its peers do
        not is a hang, not an error. Return something the backend can write, or
        an empty mapping for nothing.

    ``restore_extra_state``
        Called after the model, the optimizer, the schedulers and the dataset
        position are back, so an adapter restoring loop counters may assume the
        run around them exists. It gets what a *previous* run of this same
        framework collected: state written under a different framework is
        skipped with a warning, and a checkpoint from before any of this
        existed carries nothing and the method is not called at all.

    Raising from any of them costs the adapter's contribution and nothing else.
    The exception is logged, the extra state is treated as absent, and the
    checkpoint — or the resume, or the run — carries on. Framework state is a
    bonus on top of weights and moments, never a reason to lose them.
    """

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
