"""Dataset position tracking.

``TrackedSampler`` wraps whatever index source a ``DataLoader`` uses (its
``batch_sampler``, or its ``sampler`` when auto-collation is off) and records
how far into the current epoch iteration has got.

Fast-forward happens at the index level: on resume we consume — and throw away
— the first *N* index batches. Those are lists of integers produced in the main
process, so no sample is ever fetched, collated, or sent to a worker. Skipping
50k batches costs milliseconds, not a data-loading pass.

For the replayed order to match the original run, the shuffling RNG must be
back where it was when the epoch started. That is why we capture the sampler's
generator state at ``__iter__`` time rather than at checkpoint time, and why an
unseeded ``RandomSampler`` gets an explicit generator attached (see
``_ensure_generator``): without one, PyTorch draws a fresh seed from the global
RNG on every epoch and the order is unreproducible after a restart.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("ravex")


def _inner_sampler(obj: Any) -> Any:
    """Reach the object that actually owns the shuffling RNG.

    A ``BatchSampler`` delegates to ``.sampler``; a plain sampler is its own
    inner object.
    """
    return getattr(obj, "sampler", obj)


def _install_epoch_offset(inner: Any, offset: int) -> None:
    """Make ``set_epoch()`` continue the run rather than restart it.

    ``DistributedSampler`` derives its shuffle purely from ``seed + epoch``, and
    the epoch comes from the user's own loop counter:

        for epoch in range(EPOCHS):
            sampler.set_epoch(epoch)

    A resumed script starts that counter at 0 again. Restoring the epoch once
    fixes only the first pass — from the second onwards the run would replay
    epochs it already did, in the wrong order relative to the run it is
    continuing.

    So the sampler's ``set_epoch`` is shifted by the epoch the checkpoint was
    taken in. The user keeps counting from zero and the data keeps moving
    forward.
    """
    if offset <= 0:
        return
    original = getattr(inner, "set_epoch", None)
    if original is None or getattr(original, "_ravex_offset", None) is not None:
        return

    def set_epoch(epoch, *args, **kwargs):
        return original(epoch + offset, *args, **kwargs)

    set_epoch._ravex_offset = offset
    try:
        inner.set_epoch = set_epoch
    except AttributeError:  # pragma: no cover - exotic sampler
        logger.warning("Could not shift set_epoch on %r", inner)


def _ensure_generator(sampler: Any) -> None:
    """Give an unseeded shuffling sampler a generator we can snapshot."""
    if not hasattr(sampler, "generator"):
        return
    if sampler.generator is not None:
        return
    try:
        import torch

        generator = torch.Generator()
        # Seed from the global RNG so the run stays reproducible end to end
        # under torch.manual_seed(), instead of from entropy.
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)
        sampler.generator = generator
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not attach a generator to %r: %s", sampler, exc)


class TrackedSampler:
    """Position-tracking proxy around a sampler or batch sampler."""

    def __init__(self, original: Any):
        self._original = original
        # Batches the *training loop* has taken, which is not the same as
        # batches the sampler has produced: with num_workers > 0 the sampler
        # runs ahead by the prefetch depth. The loop's count is the one that
        # describes where training actually is, so it is fed in from the
        # dataloader wrapper via note_consumed().
        self._consumed = 0
        self._epoch_index = 0
        self._epoch_generator_state = None
        self._pending_skip = 0
        self._pending_generator_state = None

        _ensure_generator(_inner_sampler(original))

    # ─── iteration ──────────────────────────────────────────────────

    def __iter__(self):
        original = self._original
        inner = _inner_sampler(original)

        # A pending restore must be applied before the underlying sampler draws
        # its permutation, i.e. right here.
        if self._pending_generator_state is not None:
            self._set_generator_state(inner, self._pending_generator_state)
            self._pending_generator_state = None

        self._epoch_generator_state = self._generator_state(inner)

        skip = self._pending_skip
        self._pending_skip = 0
        self._consumed = skip

        iterator = iter(original)

        if skip > 0:
            logger.info("Fast-forwarding dataloader by %d batches", skip)
            skipped = 0
            for _ in iterator:
                skipped += 1
                if skipped >= skip:
                    break
            if skipped < skip:
                # The epoch was shorter than the recorded position (dataset
                # changed, or the sampler is not replayable). Carry on from
                # here rather than failing the run.
                logger.warning(
                    "Dataloader exhausted after %d of %d skipped batches; "
                    "continuing from the start of the next epoch",
                    skipped,
                    skip,
                )
                self._consumed = skipped

        yield from iterator

        self._epoch_index += 1

    def __len__(self):
        return len(self._original)

    # ─── transparency ───────────────────────────────────────────────

    def __getattr__(self, name):
        # Only reached for attributes not found on the instance, so the
        # bookkeeping fields set in __init__ never come through here.
        if name.startswith("__") or name == "_original":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_original"), name)

    def __repr__(self):
        return f"TrackedSampler({self._original!r})"

    @property
    def original(self) -> Any:
        return self._original

    @property
    def position(self) -> int:
        """Batches the training loop has taken this epoch."""
        return self._consumed

    def note_consumed(self) -> None:
        """Record that the loop received one more batch."""
        self._consumed += 1

    # ─── state ──────────────────────────────────────────────────────

    @staticmethod
    def _generator_state(inner: Any):
        generator = getattr(inner, "generator", None)
        if generator is None:
            return None
        try:
            return generator.get_state().clone()
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _set_generator_state(inner: Any, state) -> None:
        generator = getattr(inner, "generator", None)
        if generator is None or state is None:
            return
        try:
            generator.set_state(state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not restore sampler generator state: %s", exc)

    def state(self) -> Dict[str, Any]:
        """Capture where in the epoch we are, and what to replay from.

        The pair recorded is always *(state the epoch was drawn from, how many
        batches were consumed since)* — never an attempt to predict where the
        next epoch will start. Predicting is tempting when the checkpoint lands
        on the last batch, but wrong: ``RandomSampler`` draws an extra
        permutation when its iterator is exhausted, which has not happened yet
        at that point. Replaying the epoch and discarding it costs microseconds
        and leaves the generator exactly where the original run left it.
        """
        inner = _inner_sampler(self._original)
        generator_state = self._epoch_generator_state
        if generator_state is None:
            # state() called before the first iteration: the generator is
            # already at the epoch start.
            generator_state = self._generator_state(inner)

        state: Dict[str, Any] = {
            "position": self._consumed,
            "epoch_index": self._epoch_index,
            "length": self._safe_len(),
        }
        if generator_state is not None:
            state["generator_state"] = generator_state
        if hasattr(inner, "epoch"):
            state["sampler_epoch"] = inner.epoch
        return state

    def restore(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        inner = _inner_sampler(self._original)

        self._epoch_index = int(state.get("epoch_index", 0))

        if "sampler_epoch" in state and hasattr(inner, "set_epoch"):
            epoch = int(state["sampler_epoch"])
            try:
                inner.set_epoch(epoch)  # before the shift is installed
                _install_epoch_offset(inner, epoch)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("set_epoch failed during restore: %s", exc)

        # A position at (or past) the end of the epoch means the whole epoch is
        # skipped: the next pass through the loader yields nothing and the one
        # after it carries on with the right order.
        self._pending_skip = int(state.get("position", 0))
        self._pending_generator_state = state.get("generator_state")

    def _safe_len(self) -> Optional[int]:
        try:
            return len(self._original)
        except Exception:
            return None
