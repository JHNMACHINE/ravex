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

The wiring came first and on its own — a checkpoint that carries extra state,
one that does not, one written under another framework — because behaviour
bolted onto absent wiring is a rewrite of both at once. Read the contract on
each method before writing an override: :class:`HuggingFaceAdapter` is the
first, and what it restores is the counter ``Trainer`` stops on.
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


def _runtime_registry() -> Any:
    """The active run's registry, or None outside a ``train_loop``."""
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    return runtime.registry if runtime is not None else None


def _live_instances(cls: type) -> list:
    """Every live instance of ``cls`` or of a subclass of it, in one heap scan.

    An instance of a Python class holds a reference to its type, so asking the
    garbage collector who refers to the class hierarchy finds the instances
    without anyone having recorded them. Measured on 2026-09-14 against a heap
    of 392 thousand tracked objects with a 124M-parameter model in it: 18 ms,
    and a ``Trainer`` subclass found alongside the base class. It is paid once
    per run — callers cache what it returns.
    """
    import gc

    classes, pending = [], [cls]
    while pending:
        current = pending.pop()
        classes.append(current)
        pending.extend(current.__subclasses__())
    return [obj for obj in gc.get_referrers(*classes) if isinstance(obj, cls)]


#: The parts of ``TrainerState`` that are the *run's progress* rather than this
#: process's configuration. ``max_steps``, ``num_train_epochs``, the batch size
#: and the logging cadences are left alone on purpose: ``Trainer`` computes them
#: from the arguments of the run doing the resuming, and a checkpoint that
#: overwrote them would make a changed ``TrainingArguments`` silently not apply.
_TRAINER_PROGRESS = (
    "global_step",
    "epoch",
    "log_history",
    "best_metric",
    "best_global_step",
    "best_model_checkpoint",
    "total_flos",
    "num_input_tokens_seen",
)


class HuggingFaceAdapter(FrameworkAdapter):
    """HuggingFace ``Trainer``: its progress, and a warning about its own saves.

    **Why this is not optional.** ``Trainer.train()`` builds a fresh
    ``TrainerState`` before the first batch, and it stops when
    ``state.global_step`` reaches ``state.max_steps``. Ravex restores the model,
    the optimizer and the dataset position underneath it, but without this the
    step counter the loop stops on starts again from zero. Measured on
    2026-09-14: a 20-step run killed at step 10 with a checkpoint at step 8
    trained **20** more steps on resume, 28 in total, and reported nothing
    wrong. With the progress restored it trains 12 and stops at 20.

    **Finding the Trainer.** Ravex hooks PyTorch, not ``Trainer``, so it holds
    no reference to one. Every live instance is found from the class instead
    (:func:`_live_instances`) — which also covers a ``Trainer`` built before the
    decorated function was called — and the one whose ``model`` Ravex is
    tracking is chosen when there are several. If ``transformers.trainer`` was
    never imported there is nothing to look for: that is a plain loop over a
    model from the hub, and it is ``transformers`` being imported that made
    detection say ``huggingface``.

    **What does not come back.** ``state.epoch`` is restored, and then
    ``Trainer`` overwrites it at the next step from a loop variable local to
    ``_inner_training_loop`` that started at zero. So after a resume it counts
    the epochs of *this process*: the run still stops at the right step, and
    logs and epoch-based evaluation see an epoch number that is too small.
    Nothing outside ``Trainer`` can reach that variable.

    **The duplicate checkpoint.** ``Trainer`` has its own checkpointing, driven
    by ``save_strategy``, and with both on the weights and the optimizer are
    written twice for no extra safety. Said once, at the first moment the
    ``Trainer`` is found.
    """

    name = "huggingface"

    def __init__(self) -> None:
        #: A strong reference, on purpose, for the same reason as
        #: `ObjectRegistry.pin_live`: the final checkpoint is written after the
        #: training function returned, and a `Trainer` held weakly is gone by
        #: then — taking the progress with it, so a resume of a finished run
        #: would train its whole budget again. The adapter lives exactly as
        #: long as the runtime, so this is released at the end of the run.
        self._found: Any = None
        self._looked = False
        self._warned_duplicate = False

    def _trainer(self) -> Any:
        if self._found is not None:
            return self._found

        module = sys.modules.get("transformers.trainer")
        trainer_class = getattr(module, "Trainer", None)
        if trainer_class is None:
            return None

        candidates = _live_instances(trainer_class)
        trainer = None
        if len(candidates) == 1:
            trainer = candidates[0]
        elif candidates:
            registry = _runtime_registry()
            tracked = {id(model) for model in registry.models} if registry else set()
            training = [t for t in candidates if id(getattr(t, "model", None)) in tracked]
            if len(training) == 1:
                trainer = training[0]
            elif not self._looked:
                logger.warning(
                    "Found %d HuggingFace Trainers and could not tell which one "
                    "is training - their progress (global_step, log_history, "
                    "best_metric) is not checkpointed, so a resumed run starts "
                    "its step budget over",
                    len(candidates),
                )
        self._looked = True

        if trainer is not None:
            self._found = trainer
            self._warn_if_duplicated(trainer)
        return trainer

    def _warn_if_duplicated(self, trainer: Any) -> None:
        if self._warned_duplicate:
            return
        self._warned_duplicate = True
        strategy = getattr(getattr(trainer, "args", None), "save_strategy", "no")
        strategy = getattr(strategy, "value", strategy)
        if str(strategy) != "no":
            logger.warning(
                "HuggingFace Trainer is checkpointing too (save_strategy=%r): "
                "the model and the optimizer are written twice, once by each, "
                "and the second copy adds no safety. Set "
                'TrainingArguments(save_strategy="no") to leave it to Ravex',
                str(strategy),
            )

    def collect_extra_state(self) -> Dict[str, Any]:
        import copy

        trainer = self._trainer()
        if trainer is None:
            return {}
        state = trainer.state
        # A copy, because `log_history` is a list the loop keeps appending to
        # and a background writer may serialize it steps from now.
        return {
            "trainer_state": copy.deepcopy(
                {key: getattr(state, key) for key in _TRAINER_PROGRESS if hasattr(state, key)}
            )
        }

    def restore_extra_state(self, state: Dict[str, Any]) -> None:
        import os

        progress = state.get("trainer_state")
        if not progress:
            return
        trainer = self._trainer()
        if trainer is None:
            logger.warning(
                "This checkpoint carries HuggingFace Trainer progress (step %s) "
                "and no Trainer was found to restore it into, so the step "
                "budget starts over",
                progress.get("global_step"),
            )
            return

        best = progress.get("best_model_checkpoint")
        if best is not None and not os.path.isdir(best):
            # What `Trainer` itself does when resuming from a state whose best
            # checkpoint has since been deleted.
            progress = dict(progress, best_model_checkpoint=None)
        for key, value in progress.items():
            if hasattr(trainer.state, key):
                setattr(trainer.state, key, value)
        logger.info(
            "Restored HuggingFace Trainer progress: global_step=%s",
            trainer.state.global_step,
        )


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
