"""Monkey patches on PyTorch's base classes.

Every patch follows the same contract: call the original first, then do
Ravex's bookkeeping inside a ``try``. If the bookkeeping raises, the user's
call still returns the original result. A training run must never fail because
the checkpointer got confused.

What gets patched, and why that specific hook:

``nn.Module.__init__``
    Records every module ever built. Which of them are *models* is decided
    later — see ``ObjectRegistry._root_models`` — because at ``__init__`` time a
    module has neither parameters nor children yet.

``nn.Module.train``
    A module the user explicitly puts in training mode is a model, even if no
    optimizer owns its parameters (EMA copies, frozen teachers).

``optim.Optimizer.__init__``
    The single choke point every optimizer subclass goes through. The step hook
    is attached here, per instance: patching ``Optimizer.step`` on the class
    would miss ``Adam``, ``SGD`` and friends entirely, since each one overrides
    ``step`` and never calls ``super().step()``.

``DataLoader.__init__`` / ``DataLoader.__iter__``
    ``__init__`` swaps in the position-tracking sampler; ``__iter__`` is where
    resume is triggered, because it is the last moment before the first batch
    is drawn at which the model, the optimizer and the dataloader all exist.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List, Tuple

logger = logging.getLogger("ravex")

_PATCH_FLAG = "_ravex_patched"


class PatchSet:
    """Records installed patches so they can be undone (used by the tests)."""

    def __init__(self) -> None:
        self._undo: List[Tuple[Any, str, Any]] = []

    def apply(self, owner: Any, name: str, replacement: Callable) -> None:
        original = getattr(owner, name)
        setattr(replacement, "_ravex_original", original)
        setattr(owner, name, replacement)
        self._undo.append((owner, name, original))

    def uninstall(self) -> None:
        for owner, name, original in reversed(self._undo):
            try:
                setattr(owner, name, original)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not restore %s.%s: %s", owner, name, exc)
        self._undo.clear()

    def __len__(self) -> int:
        return len(self._undo)


def _wrap(original: Callable, after: Callable) -> Callable:
    """Call ``original``, then ``after(self, result)`` defensively."""

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            after(self, result)
        except Exception as exc:
            logger.warning(
                "Ravex hook failed in %s: %s", getattr(original, "__qualname__", original), exc
            )
        return result

    return wrapper


def install_all_patches(registry, runtime) -> PatchSet:
    """Install every patch. Idempotent: a second call is a no-op."""
    import torch

    patches = PatchSet()

    if getattr(torch.nn.Module, _PATCH_FLAG, False):
        logger.debug("PyTorch is already patched; skipping")
        return patches

    _patch_module(patches, registry, torch)
    _patch_optimizer(patches, registry, runtime, torch)
    _patch_scheduler(patches, registry, torch)
    _patch_scaler(patches, registry, torch)
    if runtime.config.track_dataloaders:
        _patch_dataloader(patches, registry, runtime, torch)

    setattr(torch.nn.Module, _PATCH_FLAG, True)
    logger.info("Installed %d PyTorch patches", len(patches))
    return patches


# ─── nn.Module ──────────────────────────────────────────────────────


def _patch_module(patches: PatchSet, registry, torch) -> None:
    def on_init(module, _result):
        registry.note_module(module)

    patches.apply(
        torch.nn.Module, "__init__", _wrap(torch.nn.Module.__init__, on_init)
    )

    original_train = torch.nn.Module.train

    @functools.wraps(original_train)
    def patched_train(self, mode: bool = True):
        result = original_train(self, mode)
        if mode:
            try:
                registry.register_model(self)
            except Exception as exc:
                logger.warning("Could not register model: %s", exc)
        return result

    patches.apply(torch.nn.Module, "train", patched_train)


# ─── optimizers ─────────────────────────────────────────────────────


def _patch_optimizer(patches: PatchSet, registry, runtime, torch) -> None:
    def on_init(optimizer, _result):
        registry.register_optimizer(optimizer)
        _attach_step_hook(optimizer, runtime)

    patches.apply(
        torch.optim.Optimizer,
        "__init__",
        _wrap(torch.optim.Optimizer.__init__, on_init),
    )


def _attach_step_hook(optimizer, runtime) -> None:
    """Fire ``runtime.on_step`` after each ``optimizer.step()``.

    ``register_step_post_hook`` (torch >= 2.0) is the supported way in and is
    immune to how a subclass implements ``step``. Older versions get an
    instance-level wrapper, which shadows the class method for this object
    only — still safe with LR schedulers, which wrap whatever ``optimizer.step``
    is at the time they are built.
    """
    if getattr(optimizer, "_ravex_hooked", False):
        return

    register = getattr(optimizer, "register_step_post_hook", None)
    if callable(register):
        register(lambda opt, args, kwargs: runtime.on_step(opt))
        optimizer._ravex_hooked = True
        return

    original_step = optimizer.step

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        try:
            runtime.on_step(optimizer)
        except Exception as exc:
            logger.warning("Step hook failed: %s", exc)
        return result

    optimizer.step = step
    optimizer._ravex_hooked = True


def _patch_scheduler(patches: PatchSet, registry, torch) -> None:
    scheduler_module = torch.optim.lr_scheduler
    base = getattr(scheduler_module, "LRScheduler", None) or getattr(
        scheduler_module, "_LRScheduler", None
    )

    def on_init(scheduler, _result):
        registry.register_scheduler(scheduler)

    for cls in (base, getattr(scheduler_module, "ReduceLROnPlateau", None)):
        if cls is None:
            continue
        patches.apply(cls, "__init__", _wrap(cls.__init__, on_init))


def _patch_scaler(patches: PatchSet, registry, torch) -> None:
    """Patch the AMP grad scaler, wherever this torch version keeps it."""

    def on_init(scaler, _result):
        registry.register_scaler(scaler)

    scaler_cls = None
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        scaler_cls = amp.GradScaler
    else:  # torch < 2.3
        cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
        scaler_cls = getattr(cuda_amp, "GradScaler", None)

    if scaler_cls is None:
        logger.debug("No GradScaler in this torch build; skipping AMP patch")
        return

    patches.apply(scaler_cls, "__init__", _wrap(scaler_cls.__init__, on_init))


# ─── dataloaders ────────────────────────────────────────────────────


def _patch_dataloader(patches: PatchSet, registry, runtime, torch) -> None:
    dataloader_cls = torch.utils.data.DataLoader

    def on_init(loader, _result):
        attach_tracker(loader)
        registry.register_dataloader(loader)

    patches.apply(
        dataloader_cls, "__init__", _wrap(dataloader_cls.__init__, on_init)
    )

    original_iter = dataloader_cls.__iter__

    @functools.wraps(original_iter)
    def patched_iter(self):
        try:
            runtime.on_dataloader_iter(self)
            if runtime.should_stop():
                # Budget spent. An empty epoch lets the user's loop unwind by
                # itself, which is gentler than raising through their code.
                logger.info("Step budget reached - yielding an empty epoch")
                return iter(())
        except Exception as exc:
            logger.warning("Resume hook failed: %s", exc)

        iterator = original_iter(self)

        # Deliberately after the iterator exists: building it draws the worker
        # base seed from the global RNG, and the restored state has to land on
        # the far side of that draw to match the run being continued.
        try:
            runtime.after_dataloader_iter()
            return _BatchBoundaryIterator(
                iterator, runtime, getattr(self, "_ravex_sampler", None)
            )
        except Exception as exc:
            logger.warning("Could not wrap the dataloader iterator: %s", exc)
            return iterator

    patches.apply(dataloader_cls, "__iter__", patched_iter)


class _BatchBoundaryIterator:
    """Wraps a dataloader iterator to expose the top of the training loop.

    A checkpoint has to describe a state the loop can be re-entered from, and
    the only such point is *just before a batch is fetched*: the previous
    iteration is complete — optimizer stepped, LR scheduler stepped, RNG
    advanced — and the next one has not started.

    Taking the checkpoint inside the ``optimizer.step()`` hook instead lands
    mid-iteration, before ``scheduler.step()`` runs. The saved learning rate is
    then one step stale, and a resumed run trains with the wrong LR from its
    very first step. So the step hook only raises a flag, and the actual
    collection happens here.

    Attribute access falls through to the real iterator so that frameworks
    poking at ``_shutdown_workers``, ``_reset`` and friends see what they
    expect.
    """

    def __init__(self, inner, runtime, tracked):
        self._inner = inner
        self._runtime = runtime
        self._tracked = tracked

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._runtime.on_batch_boundary()

            # Enforced here rather than only at the top of an epoch, so the
            # budget is a step count and not "that many steps, rounded up to
            # the end of whatever epoch we were in". The boundary check above
            # runs first, so the last step's checkpoint is still collected.
            if self._runtime.should_stop():
                raise StopIteration
        except StopIteration:
            raise
        except Exception as exc:
            logger.warning("Checkpoint at batch boundary failed: %s", exc)

        batch = next(self._inner)  # StopIteration ends the epoch, as usual

        if self._tracked is not None:
            self._tracked.note_consumed()
        return batch

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def attach_tracker(loader) -> None:
    """Swap the loader's index source for a :class:`TrackedSampler`.

    Assignment goes through ``__dict__`` on purpose: ``DataLoader.__setattr__``
    refuses to reassign ``sampler`` or ``batch_sampler`` once the loader has
    been initialised, and this runs immediately after ``__init__``.
    """
    import torch
    from ravex._sampler import TrackedSampler

    if getattr(loader, "_ravex_sampler", None) is not None:
        return

    if isinstance(getattr(loader, "dataset", None), torch.utils.data.IterableDataset):
        # An iterable dataset has no index sampler: its position lives inside
        # the user's generator and cannot be replayed by skipping indices.
        logger.info(
            "DataLoader over an IterableDataset - dataset position will not be "
            "restored on resume (model, optimizer and RNG still are)"
        )
        return

    for attribute in ("batch_sampler", "sampler"):
        original = getattr(loader, attribute, None)
        if original is None:
            continue
        tracked = TrackedSampler(original)
        loader.__dict__[attribute] = tracked
        loader.__dict__["_ravex_sampler"] = tracked
        return
