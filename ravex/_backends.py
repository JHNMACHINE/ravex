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
those numbers in its per-checkpoint log line. Each phase is meant to have one
cause and one lever: where a single number covered two of them it has been
split, which is why the wait for the previous writer is reported apart from the
copy that precedes it rather than added to it.

On 8× RTX 5060 Ti the handoff came out at 10.6 s against 1.5 s of state
collection, and three A/B runs against the suspects — thread count, compression
level, checkpoint cadence — each moved it by under a second. Guessing has a poor
record here; the breakdown is cheap enough to always be on.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

        A phase is worth reporting separately when it has a cause of its own.
        Two costs with different causes summed into one name is the shape of a
        wrong investigation: the number moves, and the reason it moved is not
        in the log.
        """

    @abstractmethod
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent checkpoint, or None if there is none."""

    def latest_step(self) -> Optional[int]:
        """Step of the newest stored checkpoint, without loading its tensors."""
        return None

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Return the checkpoint written at ``step``, or None.

        Per-rank checkpointing needs this - the ranks have to agree on a step
        every one of them holds, and "the latest" is not that step when a kill
        landed between two ranks' writes - and so does `resume_step`, which is
        somebody naming the point to come back to.
        """
        return None

    def known_steps(self) -> List[int]:
        """Every step this store holds, oldest first.

        For telling somebody what they can actually ask for: `resume_step: 900`
        on a store whose retention kept 1000 and 1050 is a mistake worth
        answering with the list rather than with "not found". The dashboard
        draws the same list as the points a run can be forked from.
        """
        return []

    def describe_step(self, step: int) -> Optional[Dict[str, Any]]:
        """The shape of the checkpoint at ``step``, with no tensors in it.

        The same tree :meth:`load_step` returns, except that every tensor is
        replaced by something carrying only its ``shape`` and ``dtype``.
        Everything that was never a tensor — placements, layout tags, scalars —
        is there unchanged, which is what makes the result measurable.

        Resharding is why this exists. Planning how N old shards map onto M new
        ones needs every old length before the first slice can be cut, so the
        measuring pass read each old checkpoint in full and threw it away.
        Whether that is affordable is not a matter of degree: it is the entire
        old checkpoint per rank, and between machines it would be that over the
        network, which would also undo the ``ceil(N/M)+1`` memory bound the
        reshard is built to hold.

        ``None`` means this backend cannot answer without loading, and the
        caller falls back to :meth:`load_step`. Not abstract for that reason —
        a backend built on ``torch.save`` has one pickle and no way to read a
        shape out of it short of unpickling the lot.
        """
        return None

    @abstractmethod
    def has_checkpoint(self) -> bool:
        """Whether a resumable checkpoint exists. Cheap; no tensor loading."""

    @abstractmethod
    def flush(self) -> None:
        """Block until every pending write has completed."""

    def restore_from_remote(self) -> bool:
        """Pull this store back from remote storage. Did anything come?

        For a machine that came up without the store it had. Backends with no
        remote of their own have nothing to pull, which is why this is not
        abstract — ``torch_save`` writes to a path and that is all.
        """
        return False

    def consolidate(self) -> None:
        """Leave the store readable on its own, with nothing outside it.

        Called before a store is copied to another machine: a copy that still
        depends on something left behind is not a copy of anything. Backends
        that never write a delta have nothing to do, which is why this is not
        abstract.
        """
        self.flush()

    def close(self) -> None:
        self.flush()

    #: The directory this store lives in on local disk. With a remote store it
    #: is the staging directory the bucket is synced from. The audit trail is
    #: written here (GPU-93), inside the store it describes.
    store_root: Optional[str] = None

    def fingerprint(self, step: int) -> Tuple[Optional[str], str]:
        """A content fingerprint of the durable checkpoint at ``step``, and its kind.

        Asked only once the checkpoint is durable — see
        :class:`ravex._audit.AuditTrail` for how that is known. The kind travels
        with the value because the two backends answer with hashes of different
        strength, and an audit entry that did not say which would invite
        reading one as the other. ``(None, "unavailable")`` from a backend that
        cannot say, which still gets its entry.
        """
        return None, "unavailable"


# ─── moonclip ───────────────────────────────────────────────────────

#: Single prefix for the whole state tree. Tensor names come out as
#: ``ravex/models/<key>/<param>``, stable across steps, which is exactly
#: what Moonclip's per-tensor delta tracking keys on.
_PREFIX = "ravex"


#: First Moonclip that accepts `save_dtype` as a mapping, and the first that
#: knows `fp64`. Below it the argument is a plain string with a shorter
#: vocabulary.
_SAVE_DTYPE_SINCE = (0, 0, 9)


def _supports_save_dtype(moonclip) -> bool:
    """Whether the installed Moonclip understands this `save_dtype`.

    An unreadable or unexpected version is taken as *supported*. The floor is
    declared in `pyproject.toml`, so being here with something older is
    already unusual; guessing "no" on a version string this cannot parse
    would silently drop a setting that probably works, which is the failure
    this function exists to prevent rather than to reproduce.
    """
    raw = getattr(moonclip, "__version__", None)
    if not isinstance(raw, str):
        return True
    try:
        parts = tuple(int(p) for p in raw.split(".")[:3])
    except ValueError:
        return True
    return parts >= _SAVE_DTYPE_SINCE


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
            "async_save": config.async_save,
            # Ravex owns the topology; Moonclip is handed a value and never
            # asked to work one out.
            #
            # Every store here is single-rank by construction. Where ranks
            # write separately they get a *directory* each — see
            # `_per_rank_config` — rather than sharing one store with a rank
            # id, so from Moonclip's side there is exactly one writer and one
            # manifest whatever the job looks like. That is the whole reason
            # this pair is stated rather than left to a default.
            "world_size": 1,
            "rank": 0,
        }

        # Passed only when it differs from Moonclip's own default.
        if not config.keep_base_in_memory:
            kwargs["keep_base_in_memory"] = False

        # Component names became globs on the way out of the config: Moonclip
        # matches names and deliberately does not know what an optimizer is.
        save_dtype = config.resolve_save_dtype()
        if save_dtype is not None and not _supports_save_dtype(moonclip):
            # Stated as a version requirement, out loud, rather than left to
            # be discovered as an exception. Moonclip before 0.0.9 declares
            # `save_dtype` as a string, so the mapping form arrives as a
            # `TypeError` — and `get_backend` catches everything, so the run
            # would lose Moonclip checkpointing *entirely* over one setting,
            # reported as "Moonclip backend unavailable". One config option,
            # the whole run downgraded to `torch.save`, and a message that
            # blames the wrong thing.
            #
            # So the narrow thing degrades instead of the broad one: the
            # option is dropped, checkpointing continues, and the reason names
            # itself. `fp64` and the per-component form both arrived in 0.0.9,
            # which is why this covers the setting rather than one spelling of
            # it — emulating the older vocabulary here would put a copy of
            # Moonclip's dtype table in the wrong repository.
            logger.warning(
                "save_dtype needs Moonclip >= 0.0.9 (installed: %s); storing "
                "every tensor at the precision it arrives in. Upgrade "
                "Moonclip to use it — checkpointing is otherwise unaffected.",
                getattr(moonclip, "__version__", "unknown"),
            )
            save_dtype = None
            # Already told, and told something more useful. Pointing at a
            # setting they just used would read as if it had been ignored for
            # no reason.
            configured = True
        else:
            configured = save_dtype is not None

        if save_dtype is not None:
            kwargs["save_dtype"] = save_dtype
            logger.info("save_dtype=%s", save_dtype)
        elif not configured:
            # Said once per run, and not as a warning, because nothing is
            # wrong — the default is deliberately "store what arrived". It is
            # here because the setting is worth a great deal and is invisible
            # otherwise: on a 1.5B model under FSDP2 the optimizer moments are
            # ~85% of the bytes written and barely delta at all, and they are
            # the part that tolerates the least precision. Someone paying for
            # that every checkpoint should at least know the knob exists.
            logger.info(
                "save_dtype is unset: every tensor is stored at the precision "
                "it arrives in. Optimizer state is typically ~85% of a "
                "checkpoint and compresses worst; save_dtype={optimizer: "
                "bf16} halves that part and leaves the model untouched."
            )

        if not config.async_save:
            logger.warning(
                "async_save=false: the training loop will block until each "
                "checkpoint is durable, not just copied"
            )

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

        # `MoonclipManager`, not `CheckpointManager`. The latter is Moonclip's
        # convenience layer: it reads RANK/WORLD_SIZE from the environment when
        # it is not told, and under torchrun that made it refuse the
        # single-rank save API outright — a refusal this backend caught and
        # answered by falling back to `torch.save` with one log line, so a
        # distributed run lost Moonclip checkpointing and said almost nothing.
        # `MoonclipManager` is the explicit layer underneath and infers
        # nothing; every keyword above is one it already takes.
        self._manager = moonclip.MoonclipManager(**kwargs)
        self.store_root = storage.path

        #: Fingerprints taken while their snapshot still existed, by step. The
        #: audit entry for a step is written after the *next* save returns,
        #: and that next save is the one whose retention can remove it - with
        #: `keep_last: 1` it always does. So each step is fingerprinted at the
        #: start of the save after it, before Moonclip is handed anything
        #: (GPU-136). Armed from the config, before the first save, for the
        #: reason `TorchSaveBackend.fingerprints` is.
        self.fingerprints: Optional[Dict[int, str]] = (
            {} if getattr(config, "audit_log", False) else None
        )
        self._last_saved_step: Optional[int] = None

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
        tensors, _ = self._moonclip.flatten_state_dict(state, _PREFIX, as_tensors=True)
        flattened = time.perf_counter()
        waited_here, fingerprinted = self._fingerprint_previous()
        handed = time.perf_counter()
        self._manager.save_tensors(step=step, tensors=tensors, metadata=metadata)
        inside_save = time.perf_counter() - handed
        self._last_saved_step = step

        # Neither of these is the write — that runs in the background. They are
        # the two things the calling thread pays for before it gets back:
        #
        # `store`, the shadow copy, without which the writer would be reading
        # memory the next step is about to overwrite. Memory bandwidth; it
        # scales with the model and comes down by making the state smaller.
        #
        # `backpressure`, the wait for the previous checkpoint's writer.
        # Moonclip allows one save in flight, so a writer that has not drained
        # stops this line before any work starts. It scales with the cadence
        # and with the storage, and comes down by checkpointing less often or
        # writing somewhere faster.
        #
        # They were one number until GPU-61, and the sum reads like the writer:
        # watching `store` grow is what opened GPU-55 against a writer in
        # deficit, when the phase was a 2 GiB memcpy the whole time. Measured
        # on 2026-08-18 the wait was 4-12% of it.
        #
        # `max` because the two are read off different clocks — Moonclip's
        # `Instant` against `perf_counter` here — and a phase reported as
        # slightly negative would be a worse lie than a rounding error.
        #
        # With the audit log on, most of that wait moves out of `save_tensors`
        # and into `_fingerprint_previous`, which has to see the previous
        # snapshot durable before it can describe it. It is the same wait, so
        # it is added back here rather than left to vanish from the phases.
        waited = self._manager.last_queue_wait()
        phases = {
            "flatten": flattened - started,
            "store": max(inside_save - waited, 0.0),
            "backpressure": waited + waited_here,
        }
        if self.fingerprints is not None:
            phases["fingerprint"] = fingerprinted
        return phases

    def _fingerprint_previous(self) -> Tuple[float, float]:
        """Fingerprint the last step saved while it is still in the store.

        Returns the time spent waiting for its writer and the time spent
        describing it, separately: the first is backpressure the next save
        would have paid anyway, the second is what the audit log costs -
        measured at 0.7 ms for 300 tensors and 4 ms for 2000, on 1.1 GiB,
        against 65-115 ms for the save itself.
        """
        step = self._last_saved_step
        if self.fingerprints is None or step is None or step in self.fingerprints:
            return 0.0, 0.0
        started = time.perf_counter()
        snapshots = self._manager.list_snapshots()  # waits for the writer
        listed = time.perf_counter()
        try:
            found, _ = self._describe_fingerprint(step, snapshots)
        except Exception as exc:
            # The audit trail must never cost a checkpoint; the entry falls
            # back to asking again later, and says `unavailable` if it can't.
            logger.warning("Audit: could not fingerprint step %d: %s", step, exc)
            found = None
        if found is not None:
            self.fingerprints[step] = found
        return listed - started, time.perf_counter() - listed

    def _rebuild(self, raw) -> Optional[Dict[str, Any]]:
        """Ravex's own payload out of a snapshot's flat tensor map.

        `MoonclipManager` hands back `{name: bytes}` and applies nothing,
        which is what this backend wants: there is no live model here to load
        into, only a state tree to give the registry. `unflatten_state_dict`
        is the supported way back, and reimplementing it here would be a copy
        of Moonclip's own format living in the wrong repository.
        """
        return self._moonclip.unflatten_state_dict(raw).get(_PREFIX)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        snapshots = self._manager.list_snapshots()
        if not snapshots:
            return None
        _, loaded = self._manager.load_latest()
        state = self._rebuild(loaded)
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

    def known_steps(self) -> List[int]:
        try:
            snapshots = self._manager.list_snapshots()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list snapshots: %s", exc)
            return []
        steps = {int(s["step"]) for s in snapshots if s.get("step") is not None}
        return sorted(steps)

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
            return self._rebuild(loaded)
        return None

    def describe_step(self, step: int) -> Optional[Dict[str, Any]]:
        """The tree at ``step`` with stubs where its tensors are.

        Two small reads and no tensors: the ``<prefix>._metadata`` entry is a
        few kilobytes holding the whole template — the structure, every shape
        and dtype, and everything that was never a tensor — and Moonclip reads
        only the byte range it occupies rather than the rank's pack.

        Needs Moonclip >= 0.1.0 for ``load_tensors``/``describe_state_dict``.
        On anything older this returns None and the caller loads, which is what
        it did before this existed: slower, never wrong.
        """
        if not hasattr(self._manager, "load_tensors") or not hasattr(
            self._moonclip, "describe_state_dict"
        ):
            return None
        try:
            snapshots = self._manager.list_snapshots()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list snapshots: %s", exc)
            return None
        # Newest first, exactly as `load_step` chooses: a step can appear more
        # than once if a run was restarted and rewrote it.
        for snapshot in reversed(snapshots):
            if int(snapshot.get("step", -1)) != step:
                continue
            try:
                raw = self._manager.load_tensors(
                    snapshot["id"], ["%s._metadata" % _PREFIX]
                )
            except Exception as exc:
                # A snapshot written by something other than Ravex has no
                # payload under this prefix. Falling back to a full load will
                # not conjure one either, but it is the caller's existing path
                # and it reports the absence in its own words.
                logger.debug("No %s template at step %s: %s", _PREFIX, step, exc)
                return None
            return self._moonclip.describe_state_dict(raw).get(_PREFIX)
        return None

    def has_checkpoint(self) -> bool:
        try:
            return bool(self._manager.list_snapshots())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list snapshots: %s", exc)
            return False

    def flush(self) -> None:
        self._manager.flush()

    def restore_from_remote(self) -> bool:
        return bool(self._manager.restore_from_remote())

    def consolidate(self) -> None:
        """Fold the delta chain into one full snapshot.

        A delta names a base, and a base that stayed on the machine the copy
        left is a base the copy cannot reach. ``merge_now`` collapses the chain
        so what gets sent stands on its own.

        It waits for the writer to drain and then forces a merge, which is the
        path bounded to 300 s in moonclip's ``FORCED_MERGE_WAIT``. Before that
        bound existed this could park here for twenty minutes on a stuck
        reader.
        """
        self._manager.flush()
        self._manager.merge_now()

    def close(self) -> None:
        self.flush()
        if getattr(self, "_manager", None) is not None:
            try:
                self._manager.sync_now()
            except Exception as exc:
                logger.warning("Final remote sync failed: %s", exc)

    def fingerprint(self, step: int) -> Tuple[Optional[str], str]:
        """SHA-256 over the per-tensor xxHash3 values Moonclip reports for ``step``.

        Nothing is rehashed: Moonclip already hashed every tensor's raw bytes
        to decide what to skip. So this is as strong as xxHash3-128 — certain
        against corruption, not against a collision crafted on purpose — and
        the kind says so.

        **Asked through** ``describe()``, **never by reading** ``manifest.json``.
        ``describe()`` waits for Moonclip's own writer. Reading the file does
        not: on Windows a reader holding it open made Moonclip's rename of the
        next manifest over it fail, and the save being written was lost —
        measured on 2026-09-14, five runs in five.

        Usually answered from ``fingerprints``, filled at the start of the next
        save while the snapshot still existed; asked of Moonclip directly only
        for the last step of a run, which nothing comes after.

        ``None`` when the snapshot is gone and was never fingerprinted, or when
        this Moonclip does not report tensor hashes — ``hash_raw`` arrives in
        0.1.1 — which is said once rather than per checkpoint.
        """
        known = (getattr(self, "fingerprints", None) or {}).get(step)
        if known is not None:
            return known, "sha256:tensor-xxh3"
        return self._describe_fingerprint(step, self._manager.list_snapshots())

    def _describe_fingerprint(
        self, step: int, snapshots: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], str]:
        from ravex._audit import tensor_fingerprint

        snapshot = None
        for candidate in snapshots:
            if candidate.get("step") == step:
                snapshot = candidate
        if snapshot is None:
            return None, "unavailable"

        tensors = self._manager.describe(snapshot["id"]).get("tensors", [])
        found = tensor_fingerprint(tensors)
        if found is None:
            if tensors and not getattr(self, "_said_no_hashes", False):
                self._said_no_hashes = True
                logger.warning(
                    "audit_log: Moonclip %s does not report tensor hashes, so "
                    "checkpoint fingerprints are recorded as null. Moonclip "
                    "0.1.1 and later report them",
                    getattr(self._moonclip, "__version__", "?"),
                )
            return None, "unavailable"
        return found, "sha256:tensor-xxh3"


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
        self.store_root = self.directory
        #: Step -> SHA-256 of its file, filled by the writer while the file
        #: is certainly still there (``None`` means nobody will ask). See
        #: `_write`.
        #:
        #: Armed here, from the config, and not at the first audit entry -
        #: which is what it used to be, and it left a hole. `_record_audit`
        #: builds the audit trail lazily and turns this dict on there, but it
        #: runs *after* `save()` has returned, so the **first** checkpoint of
        #: every audited run was written with this still `None` and never had
        #: its hash taken. Its entry then fell back to hashing the file off
        #: disk, and with `keep_last: 1` that file is deleted by the very next
        #: write: a race the audit thread wins on an idle machine and loses on
        #: a loaded one, leaving `fingerprint: null` in a record whose whole
        #: purpose is to say what was written (GPU-93). The config knows from
        #: the start whether anyone will ask, so it is asked from the start.
        self.fingerprints: Optional[Dict[int, str]] = (
            {} if getattr(config, "audit_log", False) else None
        )

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
            if self.fingerprints is not None:
                # Here, between the rename and the prune, because the prune
                # may delete the *previous* file — and with `keep_last: 1` the
                # audit trail would otherwise go looking for a checkpoint that
                # no longer exists. Hashing on this thread keeps the cost off
                # the training loop.
                from ravex._audit import file_fingerprint

                self.fingerprints[_step_of(path)] = file_fingerprint(path)
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

    def known_steps(self) -> List[int]:
        return [_step_of(path) for path in self._files()]

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

    def fingerprint(self, step: int) -> Tuple[Optional[str], str]:
        """SHA-256 of the checkpoint's ``.pt`` file: a commitment to every byte."""
        known = (self.fingerprints or {}).get(step)
        if known is not None:
            return known, "sha256:file"
        path = os.path.join(self.directory, f"step_{step:012d}.pt")
        if os.path.exists(path):
            from ravex._audit import file_fingerprint

            return file_fingerprint(path), "sha256:file"
        return None, "unavailable"


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
    from ravex._dist.collectives import get_rank

    if not per_rank:
        return config

    return store_config_at(config, rank_suffix(get_rank()))


def store_config_at(config, *parts: str):
    """The same config, re-rooted at ``<path>/<parts...>``.

    Both the local path and the remote prefix move together, because a store
    is one place addressed two ways and letting them drift would put a rank's
    bytes on disk under one name and in the bucket under another.

    Public because :func:`open_rank_store` needs to point a backend at a
    directory this process did not write, which is the whole of what reading a
    foreign store amounts to: the store is single-rank like every other, and
    only the path is different.
    """
    from dataclasses import replace

    suffix = "/".join(parts)
    storage = replace(
        config.storage,
        path=os.path.join(config.storage.path, *parts),
        prefix=(
            "%s/%s" % (config.storage.prefix.rstrip("/"), suffix)
            if config.storage.prefix
            else suffix
        ),
    )
    return replace(config, storage=storage)


def rank_suffix(rank: int) -> str:
    """The directory one rank's store lives in, under the configured path."""
    return "rank_%d" % rank


def per_rank_store_path(config, rank: int) -> str:
    """Where one rank's store is on this machine.

    Only meaningful for local storage; the caller is expected to have checked.
    Here rather than rebuilt by callers so the layout has one definition — the
    resume diagnosis and the owner record both need it, and a second copy of
    ``rank_%d`` is a second thing to keep in step with ``_per_rank_config``.
    """
    return os.path.join(config.storage.path, rank_suffix(rank))


def visible_rank_stores(config) -> "set[int]":
    """Which ranks' per-rank stores this machine can actually reach.

    Reads the directory names only — ``rank_<n>`` as written by
    ``_per_rank_config`` — and never opens a store. What is in them is each
    rank's own business, and asking would mean constructing a backend per
    foreign store just to answer a question about topology.

    Empty for remote storage, and deliberately so: every node sees the same
    bucket there, so "what I can reach" carries no information. It is only the
    local case where two machines hold different halves of one checkpoint.
    """
    if config.storage.is_remote:
        return set()

    base = config.storage.path
    try:
        entries = os.listdir(base)
    except OSError:
        # No directory yet is the ordinary first-run case, not a failure.
        return set()

    found = set()
    for name in entries:
        if not name.startswith("rank_"):
            continue
        try:
            rank = int(name[len("rank_") :])
        except ValueError:
            continue
        path = os.path.join(base, name)
        try:
            if os.path.isdir(path) and os.listdir(path):
                found.add(rank)
        except OSError:
            continue
    return found


def replica_store_path(config, rank: int) -> str:
    """Where a copy of another rank's store lands on this machine.

    Under ``replica/`` rather than beside the real stores: everything that
    scans for ``rank_<n>`` is asking "what does this machine own", and a copy
    answering that question would be read as an original.
    """
    from ravex._dist.replication import REPLICA_DIR

    return os.path.join(config.storage.path, REPLICA_DIR, rank_suffix(rank))


def visible_replica_stores(config) -> "set[int]":
    """Which ranks this machine holds a *copy* of.

    Separate from :func:`visible_rank_stores` because the two answer different
    questions. "Rank 2's store is not here" and "rank 2's store is not here but
    its copy is" lead to opposite conclusions — the first is a checkpoint that
    cannot be reached, the second is one that can.
    """
    from ravex._dist.replication import REPLICA_DIR

    if config.storage.is_remote:
        return set()

    base = os.path.join(config.storage.path, REPLICA_DIR)
    try:
        entries = os.listdir(base)
    except OSError:
        return set()

    found = set()
    for name in entries:
        if not name.startswith("rank_"):
            continue
        try:
            rank = int(name[len("rank_") :])
        except ValueError:
            continue
        path = os.path.join(base, name)
        try:
            if os.path.isdir(path) and os.listdir(path):
                found.add(rank)
        except OSError:
            continue
    return found


def visible_store_owners(config) -> "dict[int, dict]":
    """:func:`visible_rank_stores`, plus who wrote each one.

    The record is what turns "rank 2's store is not here" into "rank 2's store
    was written on node0, which is not running rank 2 now" — the difference
    between knowing something is wrong and knowing where to look. Missing or
    unreadable records map to an empty dict: the store is still present, and
    saying so with less detail beats not mentioning it.
    """
    from ravex._dist.identity import read_owner

    return {
        rank: (read_owner(per_rank_store_path(config, rank)) or {})
        for rank in visible_rank_stores(config)
    }


def open_rank_store(config, rank: int) -> Optional[CheckpointBackend]:
    """A handle on some *other* rank's per-rank store, or None if unreachable.

    The seam :func:`visible_rank_stores` declined to open. It answers "does
    ``rank_<q>`` exist on this machine" by reading directory names, on the
    grounds that opening a store to answer a question about topology is a lot
    of machinery for a small fact. Resharding needs the bytes, so this is where
    the store gets opened — and it needs nothing new underneath: every per-rank
    store is an ordinary single-rank store (``world_size=1, rank=0``, see
    :class:`MoonclipBackend`) and the only thing that differs is the path.

    Falls back to a *complete* copy under ``replica/`` when the original is not
    here, which is what makes a shrink survivable at all: after losing a
    machine, the shard it wrote exists only as the copy a neighbour holds. A
    copy caught mid-transfer is not offered — it is bytes, and it is not a
    checkpoint.

    Read-only by intent, not by enforcement: the backend it returns can write,
    and nothing here should. The caller closes it.
    """
    if config.storage.is_remote:
        # Every node already sees every store, so there is no foreign store to
        # open — `get_backend` on the ordinary per-rank path reaches it.
        return None

    from ravex._dist.replication import REPLICA_DIR, replica_is_complete

    direct = per_rank_store_path(config, rank)
    if os.path.isdir(direct) and os.listdir(direct):
        return get_backend(store_config_at(config, rank_suffix(rank)))

    copy = replica_store_path(config, rank)
    if os.path.isdir(copy) and os.listdir(copy) and replica_is_complete(copy):
        return get_backend(store_config_at(config, REPLICA_DIR, rank_suffix(rank)))

    return None


def reachable_rank_stores(config, ranks: Iterable[int]) -> "set[int]":
    """Which of ``ranks`` this machine could open, original or complete copy.

    Cheaper than opening them and the same verdict, so the reshard can decide
    whether it is possible before it starts reading tensors. Deciding late is
    the expensive kind of failure here: half the shards are in memory by then.
    """
    from ravex._dist.replication import replica_is_complete

    if config.storage.is_remote:
        return set(ranks)

    found = set()
    for rank in ranks:
        direct = per_rank_store_path(config, rank)
        try:
            if os.path.isdir(direct) and os.listdir(direct):
                found.add(rank)
                continue
        except OSError:  # pragma: no cover - a directory that vanished
            pass
        copy = replica_store_path(config, rank)
        try:
            if os.path.isdir(copy) and os.listdir(copy) and replica_is_complete(copy):
                found.add(rank)
        except OSError:  # pragma: no cover - a directory that vanished
            continue
    return found


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
