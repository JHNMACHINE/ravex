"""Checkpoint backends.

A backend takes the state dict the registry collected and gets it durably
stored, without blocking the training loop for longer than the copy takes.

Two implementations ship with Ravex:

``moonclip``
    The default. Per-tensor delta tracking, zstd, S3/R2 sync, background
    writer. State is flattened into individual tensors so unchanged weights
    cost zero I/O on the next checkpoint.

``torch_save``
    The fallback for environments without Moonclip. Plain ``torch.save`` on a
    single background thread. Correct, just slower and much larger on disk.

Both copy the state to CPU memory *before* returning from ``save``. That copy
is the whole point: it lets the training loop mutate the live tensors while the
writer is still working, without the writer reading half-updated weights.

``save`` reports how long each phase of that handoff took, and the runtime puts
those numbers in its per-checkpoint log line. On 8× RTX 5060 Ti the handoff came
out at 10.6 s against 1.5 s of state collection, and three A/B runs against the
suspects — thread count, compression level, checkpoint cadence — each moved it
by under a second. Guessing has a poor record here; the breakdown is cheap
enough to always be on.
"""

from __future__ import annotations

import glob
import inspect
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

logger = logging.getLogger("ravex")


class CheckpointBackend(ABC):
    """Storage interface used by the runtime."""

    @abstractmethod
    def save(
        self, step: int, state: Dict[str, Any], metadata: Dict[str, str]
    ) -> Dict[str, float]:
        """Persist ``state``. Returns once the state has been copied, not
        once it has been written.

        The return value is how long each phase of that copy took, in seconds,
        in the order the phases ran. It exists so the runtime's log line can say
        *where* a slow handoff went instead of only how long it took — the
        difference between a number that ends an investigation and one that
        starts another.
        """

    @abstractmethod
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent checkpoint, or None if there is none."""

    def latest_step(self) -> Optional[int]:
        """Step of the newest stored checkpoint, without loading its tensors."""
        return None

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Return the checkpoint written at ``step``, or None.

        Only per-rank checkpointing needs this: the ranks have to agree on a
        step every one of them holds, and "the latest" is not that step when a
        kill landed between two ranks' writes.
        """
        return None

    @abstractmethod
    def has_checkpoint(self) -> bool:
        """Whether a resumable checkpoint exists. Cheap; no tensor loading."""

    @abstractmethod
    def flush(self) -> None:
        """Block until every pending write has completed."""

    def close(self) -> None:
        self.flush()


# ─── moonclip ───────────────────────────────────────────────────────

#: Single prefix for the whole state tree. Tensor names come out as
#: ``ravex/models/<key>/<param>``, stable across steps, which is exactly
#: what Moonclip's per-tensor delta tracking keys on.
_PREFIX = "ravex"


class MoonclipBackend(CheckpointBackend):
    """Default backend, built on the Moonclip checkpoint engine."""

    def __init__(self, config):
        import moonclip

        self._moonclip = moonclip
        storage = config.storage

        kwargs: Dict[str, Any] = {
            "storage_root": storage.path,
            "compression_level": config.compression_level,
            "max_total_snapshots": config.keep_last,
            "async_save": True,
            # Always single-rank, stated explicitly. Moonclip otherwise infers
            # world_size from RANK/WORLD_SIZE in the environment, and under
            # torchrun it then rejects the single-rank save API outright:
            # "Multi-rank save requires explicit create_snapshot/save_rank/
            # finalize flow". Ravex does not need that flow — sharded state is
            # gathered before it gets here and exactly one rank writes — but
            # the mismatch is silent apart from a log line, so every
            # distributed run would lose checkpointing altogether.
            "world_size": 1,
            "rank": 0,
        }

        if not config.delta:
            # No dedicated switch in Moonclip: a full snapshot on every step is
            # exactly "delta disabled".
            kwargs["full_every_steps"] = 1

        if storage.is_remote:
            if not (storage.access_key and storage.secret_key):
                raise ValueError(
                    "storage.type is %r but no credentials were found. Set "
                    "RAVEX_S3_ACCESS_KEY / RAVEX_S3_SECRET_KEY (or "
                    "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)." % storage.type
                )
            kwargs.update(
                s3_bucket=storage.bucket,
                s3_region=storage.region,
                s3_prefix=storage.prefix,
                s3_endpoint=storage.endpoint,
                s3_access_key=storage.access_key,
                s3_secret_key=storage.secret_key,
                s3_path_style=storage.path_style,
            )

        self._manager = moonclip.CheckpointManager(**kwargs)

        # `as_tensors` arrived after the first released Moonclip. Passing it to
        # a build that predates it is a TypeError in the middle of a training
        # run, which is exactly the failure mode this backend exists to avoid,
        # so ask once here rather than guess later.
        try:
            self._as_tensors = "as_tensors" in inspect.signature(
                moonclip.flatten_state_dict
            ).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive
            self._as_tensors = False
        if not self._as_tensors:
            logger.info(
                "Moonclip predates flatten_state_dict(as_tensors=); checkpoints "
                "will block the training loop for roughly 5x longer per save"
            )

    def save(
        self, step: int, state: Dict[str, Any], metadata: Dict[str, str]
    ) -> Dict[str, float]:
        # `as_tensors` hands Moonclip the tensors and lets it take the shadow
        # copy itself, on all cores with the GIL released. Flattening to bytes
        # here instead cost two serial copies of the whole state — one in
        # `.tobytes()`, one on the way into Rust — and the training loop was
        # blocked for both: 1393 ms against 271 ms on 3.8 GiB of weights.
        #
        # The copy is still taken before this method returns, which is the
        # contract this class documents. It just happens one line later, inside
        # `save_raw`. That matters because for a model already on CPU the dict
        # below aliases live parameter memory rather than owning a copy of it.
        started = time.perf_counter()
        if self._as_tensors:
            tensors, _ = self._moonclip.flatten_state_dict(
                state, _PREFIX, as_tensors=True
            )
        else:
            tensors, _ = self._moonclip.flatten_state_dict(state, _PREFIX)
        flattened = time.perf_counter()
        self._manager.save_raw(step=step, tensors=tensors, metadata=metadata)
        # `store` is not the write: that runs in the background. It is the
        # shadow copy, plus however long the previous checkpoint's writer still
        # needed — Moonclip allows one save in flight, so a writer that has not
        # drained is backpressure landing on this line. `MOONCLIP_PROFILE=1`
        # separates the two.
        return {
            "flatten": flattened - started,
            "store": time.perf_counter() - flattened,
        }

    def load_latest(self) -> Optional[Dict[str, Any]]:
        snapshots = self._manager.list_snapshots()
        if not snapshots:
            return None
        _, loaded = self._manager.load_latest()
        state = loaded.get(_PREFIX)
        if state is None:
            logger.warning(
                "Latest snapshot has no %r payload - it was probably written by "
                "something other than Ravex. Ignoring it.",
                _PREFIX,
            )
            return None
        return state

    def latest_step(self) -> Optional[int]:
        try:
            snapshots = self._manager.list_snapshots()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list snapshots: %s", exc)
            return None
        steps = [int(s["step"]) for s in snapshots if s.get("step") is not None]
        return max(steps) if steps else None

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        try:
            snapshots = self._manager.list_snapshots()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list snapshots: %s", exc)
            return None
        # Newest first: a step can appear more than once if a run was restarted
        # and rewrote it, and the last write is the one that counts.
        for snapshot in reversed(snapshots):
            if int(snapshot.get("step", -1)) != step:
                continue
            loaded = self._manager.load(snapshot["id"])
            return loaded.get(_PREFIX)
        return None

    def has_checkpoint(self) -> bool:
        try:
            return bool(self._manager.list_snapshots())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list snapshots: %s", exc)
            return False

    def flush(self) -> None:
        self._manager.flush()

    def close(self) -> None:
        self.flush()
        if getattr(self, "_manager", None) is not None:
            try:
                self._manager.sync_now()
            except Exception as exc:
                logger.warning("Final remote sync failed: %s", exc)


# ─── torch.save fallback ────────────────────────────────────────────

_STEP_FILE = re.compile(r"step_(\d+)\.pt$")


def _step_of(path: str) -> int:
    """The step a checkpoint filename encodes."""
    match = _STEP_FILE.search(path)
    if match is None:  # pragma: no cover - _files() only yields names that match
        raise ValueError("not a checkpoint filename: %s" % path)
    return int(match.group(1))


def _cpu_copy(value: Any) -> Any:
    """Deep-copy a state tree onto CPU memory.

    ``to("cpu", copy=True)`` rather than ``.cpu()``: for a model already on CPU
    the latter is a no-op and the writer would race the next training step.
    """
    import torch

    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return value


class TorchSaveBackend(CheckpointBackend):
    """Fallback backend: one ``.pt`` file per checkpoint."""

    def __init__(self, config):
        self.directory = config.storage.path
        self.keep_last = config.keep_last
        os.makedirs(self.directory, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ravex-writer"
        )
        self._pending: Optional[Future] = None

    def save(
        self, step: int, state: Dict[str, Any], metadata: Dict[str, str]
    ) -> Dict[str, float]:
        import torch

        started = time.perf_counter()
        snapshot = _cpu_copy(state)
        snapshot["_ravex_metadata"] = dict(metadata)
        path = os.path.join(self.directory, f"step_{step:012d}.pt")
        copied = time.perf_counter()

        # Serialize writes: two concurrent torch.save calls on one disk are
        # slower than one, and ordering matters for pruning.
        self.flush()
        try:
            self._pending = self._executor.submit(self._write, torch, snapshot, path)
        except RuntimeError:
            # "cannot schedule new futures after shutdown": concurrent.futures
            # registers its own atexit hook, and by the time ours runs the pool
            # is closed. This is the checkpoint_on_exit path - the last one a
            # normally-finishing run takes - so write it here instead of losing
            # it. Nothing is racing us: the training loop is over.
            self._write(torch, snapshot, path)

        # `queue` is the previous checkpoint's `torch.save` finishing: one
        # writer thread, and `flush()` above waits for it.
        return {
            "copy": copied - started,
            "queue": time.perf_counter() - copied,
        }

    def _write(self, torch, snapshot: Dict[str, Any], path: str) -> None:
        temporary = path + ".tmp"
        try:
            torch.save(snapshot, temporary)
            os.replace(temporary, path)  # atomic: never leave a half file
            self._prune()
        except Exception as exc:
            logger.error("Checkpoint write failed for %s: %s", path, exc)
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass

    def _files(self):
        files = glob.glob(os.path.join(self.directory, "step_*.pt"))
        return sorted(files, key=_step_of)

    def _prune(self) -> None:
        files = self._files()
        for stale in files[: max(0, len(files) - self.keep_last)]:
            try:
                os.remove(stale)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not delete old checkpoint %s: %s", stale, exc)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        import torch

        files = self._files()
        if not files:
            return None
        state = torch.load(files[-1], map_location="cpu", weights_only=False)
        state.pop("_ravex_metadata", None)
        return state

    def latest_step(self) -> Optional[int]:
        files = self._files()
        if not files:
            return None
        return _step_of(files[-1])

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        import torch

        path = os.path.join(self.directory, f"step_{step:012d}.pt")
        if not os.path.exists(path):
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        state.pop("_ravex_metadata", None)
        return state

    def has_checkpoint(self) -> bool:
        return bool(self._files())

    def flush(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)


# ─── selection ──────────────────────────────────────────────────────


def _per_rank_config(config, per_rank: bool):
    """Give this rank a store of its own, under ``rank_<n>``.

    Per-rank checkpointing has every rank writing at once. Pointed at one
    store they would be N writers against one manifest, each reading it,
    adding itself and writing it back with no lock between them — a lost
    update every time two land together. Separate stores need no coordination
    at all, and each rank keeps the background writer, the delta chain and the
    retention it would have had on its own.

    The cost is N manifests, and a resume that has to agree on a step (see
    ``agree_on_step``).
    """
    from dataclasses import replace

    from ravex._distributed import get_rank

    if not per_rank:
        return config

    suffix = "rank_%d" % get_rank()
    storage = replace(
        config.storage,
        path=os.path.join(config.storage.path, suffix),
        prefix=(
            "%s/%s" % (config.storage.prefix.rstrip("/"), suffix)
            if config.storage.prefix
            else suffix
        ),
    )
    return replace(config, storage=storage)


def get_backend(config, per_rank: bool = False) -> CheckpointBackend:
    """Build the configured backend, falling back to ``torch_save``.

    A missing or broken Moonclip must never stop a training run: the whole
    proposition is that Ravex is invisible when it works and harmless when
    it does not.

    ``per_rank`` gives this rank a store of its own. The caller decides, not
    the config: whether per-rank checkpointing is actually in force depends on
    the model as well as the setting, and the store has to match what gets
    written into it.
    """
    config = _per_rank_config(config, per_rank)
    name = config.backend

    if name == "moonclip":
        try:
            return MoonclipBackend(config)
        except ImportError:
            logger.warning("Moonclip is not installed - falling back to torch.save")
        except Exception as exc:
            logger.warning("Moonclip backend unavailable (%s) - falling back", exc)
        return TorchSaveBackend(config)

    if name in ("torch_save", "torch"):
        return TorchSaveBackend(config)

    logger.warning("Unknown backend %r - using torch.save", name)
    return TorchSaveBackend(config)
