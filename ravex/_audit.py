"""An audit trail of checkpoints: what was written, when, and what it held.

The question this answers is the one GPU-93 was opened on — *which checkpoint
produced the model deployed six months ago, under which configuration* — and
its shape is a file, not a service: one JSON object per line, appended once per
checkpoint, living inside the store it describes.

Three things are recorded about each checkpoint, and each one is what a reader
of this file would otherwise have no way to reconstruct:

``fingerprint``
    A SHA-256 over the checkpoint's content. How it is computed depends on the
    backend, and the difference matters enough to be recorded next to it as
    ``fingerprint_kind``:

    * ``torch_save`` — ``sha256:file``, SHA-256 of the ``.pt`` file itself. A
      cryptographic commitment to every byte.
    * ``moonclip`` — ``sha256:tensor-xxh3``, SHA-256 over the per-tensor hashes
      Moonclip already keeps, with each tensor's name, dtype and shape. That
      costs a ``describe()`` instead of rehashing gigabytes, and it is **exactly
      as strong as those per-tensor hashes, which are xxHash3-128**: it detects
      corruption and an accidental swap with certainty for practical purposes,
      and it does *not* prove that nobody crafted a colliding tensor on
      purpose. Saying so here is the point — a compliance claim that rests on
      a hash nobody checked the strength of is worse than none.

``config_sha256``
    The resolved configuration, credentials removed, so a checkpoint can be
    tied to the settings that wrote it.

``previous_sha256`` / ``entry_sha256``
    Every entry carries the hash of the one before it. Editing, deleting or
    reordering any entry breaks the chain from that line onward, and
    :func:`verify` says where. What the chain cannot do on its own is detect a
    file rewritten *from scratch* with a consistent chain; for that, the last
    ``entry_sha256`` has to be kept somewhere the writer of the file cannot
    reach — a ticket, a release note, a signed commit. Signing is left to the
    operator on purpose: Ravex would otherwise need a key, and a key on the
    training machine protects nothing from whoever controls that machine.

**When an entry is written.** Not at the checkpoint's call, because a
checkpoint is handed off before it is durable and a fingerprint of a file still
being written is a fingerprint of nothing. Both backends wait for the previous
write at the start of the next ``save``, so :class:`AuditTrail` records a step
as pending and writes its entry once the *next* save has returned, or at
shutdown after the backend is closed. The hashing itself runs on a thread of
its own: SHA-256 of a large ``.pt`` file is seconds, and the training loop is
not the place to spend them.

**Nothing here opens a Moonclip file.** The first version read
``manifest.json`` to get the per-tensor hashes, and on Windows that cost
checkpoints: Moonclip persists the manifest by renaming a new version over it,
a rename over a file someone has open fails there — even opened sharing
delete, which was tried — and Moonclip lost the save it was writing, five runs
in five. The hashes are asked of Moonclip through ``describe()``, which waits
for its own writer.

**What is not here.** The loss: Ravex never sees it. And the data: a hash of
the dataset is the training script's to compute, because only it knows what
the dataset is.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("ravex")

#: The file name inside a store. One per store, so one per rank under
#: ``sharded_checkpoints: per_rank`` — each rank's store is its own history.
AUDIT_FILE = "audit.jsonl"

#: Written into the chain's first entry as its predecessor.
GENESIS = "0" * 64

#: Configuration fields that never enter the digest. Credentials, because the
#: digest is meant to be shared; the rest because they say where the config
#: came from rather than what it is.
_EXCLUDED_CONFIG = {"access_key", "secret_key", "metrics_token", "source", "problems"}

_CHUNK = 8 * 1024 * 1024


def _canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _entry_hash(entry: Dict[str, Any]) -> str:
    body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def file_fingerprint(path: str) -> str:
    """SHA-256 of a file, streamed so a large checkpoint is never in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(tensors: Iterable[Dict[str, Any]]) -> Optional[str]:
    """SHA-256 over a snapshot's per-tensor hashes, as Moonclip describes them.

    ``tensors`` is the ``tensors`` list of a Moonclip ``describe()``. Each one
    contributes its name, its dtype — the original one, which is what
    ``describe()`` calls ``dtype`` — its shape and its ``hash_raw``. Sorted
    first, so the order Moonclip happened to write them in does not matter;
    and ``hash_raw`` is the hash of the tensor's *raw* bytes however it is
    stored, so a delta or a skipped tensor fingerprints the same as a full one
    holding the same values.

    ``None`` when any tensor carries no hash — a Moonclip that predates
    reporting it — or there are no tensors: a fingerprint over part of a
    checkpoint would identify something that is not the checkpoint.
    """
    lines: List[str] = []
    for tensor in tensors:
        hashed = tensor.get("hash_raw")
        if not hashed:
            return None
        shape = ",".join(str(dim) for dim in tensor.get("shape", ()))
        lines.append("%s|%s|%s|%s" % (tensor.get("name"), tensor.get("dtype", ""), shape, hashed))
    if not lines:
        return None

    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def config_digest(config: Any) -> str:
    """SHA-256 of the resolved configuration, with credentials removed."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(entry)
                for key, entry in value.items()
                if key not in _EXCLUDED_CONFIG
            }
        if isinstance(value, (list, tuple)):
            return [scrub(entry) for entry in value]
        return value

    payload = scrub(dataclasses.asdict(config))
    return hashlib.sha256(_canonical(payload)).hexdigest()


def read_entries(path: str) -> List[Dict[str, Any]]:
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def verify(path: str) -> List[str]:
    """Every break in the chain, as sentences. Empty means intact.

    Checked per line rather than stopping at the first problem, because the
    useful answer to "was this edited" is *which* entries, and one edited line
    makes every later ``previous_sha256`` disagree — so a broken link is
    reported once, where it breaks, and a line whose own hash is wrong is
    reported as edited.
    """
    problems = []
    try:
        entries = read_entries(path)
    except (OSError, ValueError) as exc:
        return ["cannot read %s: %s" % (path, exc)]

    previous = GENESIS
    for number, entry in enumerate(entries, start=1):
        if entry.get("entry_sha256") != _entry_hash(entry):
            problems.append(
                "line %d (step %s) was modified after it was written"
                % (number, entry.get("step"))
            )
        if entry.get("previous_sha256") != previous:
            problems.append(
                "line %d (step %s) does not follow the line before it - an "
                "entry was removed, inserted or reordered here"
                % (number, entry.get("step"))
            )
        previous = entry.get("entry_sha256", "")
    return problems


def find(path: str, fingerprint: str) -> List[Dict[str, Any]]:
    """Entries whose fingerprint starts with ``fingerprint``.

    A prefix, because a person reading one off a ticket types the first twelve
    characters and not sixty-four.
    """
    fingerprint = fingerprint.lower()
    return [
        entry
        for entry in read_entries(path)
        if str(entry.get("fingerprint") or "").startswith(fingerprint)
    ]


class AuditLog:
    """The file: append an entry, keeping the chain."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._previous = GENESIS
        if os.path.exists(path):
            entries = read_entries(path)
            if entries:
                self._previous = entries[-1].get("entry_sha256", GENESIS)

    def append(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(fields)
        entry["previous_sha256"] = self._previous
        entry["entry_sha256"] = _entry_hash(entry)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous = entry["entry_sha256"]
        return entry


class AuditTrail:
    """What the runtime drives: a step is written, later it is durable.

    ``fingerprint`` is the backend's half — a callable from a step to
    ``(fingerprint, kind)`` — so this class knows nothing about files or
    manifests, and a backend with no way to fingerprint passes one that
    returns ``(None, kind)`` and still gets its entries.
    """

    def __init__(self, store_root: str, config: Any, fingerprint) -> None:
        self.log = AuditLog(os.path.join(store_root, AUDIT_FILE))
        self._fingerprint = fingerprint
        self._config_sha256 = config_digest(config)
        self._pending: List[Tuple[int, Dict[str, str], str]] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ravex-audit")
        self._last: Optional[Future] = None

    def saved(self, step: int, metadata: Dict[str, str]) -> None:
        """A save for ``step`` has returned: every earlier one is durable."""
        durable, self._pending = self._pending, []
        self._submit(durable)
        written_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._pending.append((step, dict(metadata), written_at))

    def close(self) -> None:
        """The backend is closed, so everything pending is durable. Waits."""
        durable, self._pending = self._pending, []
        self._submit(durable)
        self._executor.shutdown(wait=True)

    def _submit(self, durable: Iterable[Tuple[int, Dict[str, str], str]]) -> None:
        durable = list(durable)
        if not durable:
            return
        self._last = self._executor.submit(self._write, durable)

    def _write(self, durable: List[Tuple[int, Dict[str, str], str]]) -> None:
        for step, metadata, written_at in durable:
            try:
                fingerprint, kind = self._fingerprint(step)
            except Exception as exc:
                logger.warning("Audit: could not fingerprint step %d: %s", step, exc)
                fingerprint, kind = None, "unavailable"
            fields = {
                "step": step,
                "written_at": written_at,
                "fingerprint": fingerprint,
                "fingerprint_kind": kind,
                "config_sha256": self._config_sha256,
                "metadata": metadata,
            }
            try:
                self.log.append(fields)
            except Exception as exc:
                logger.warning("Audit: could not append step %d: %s", step, exc)
