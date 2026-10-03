"""A directory of compiled things, shared between machines that would compile it the same way (GPU-181).

A fresh machine spends its first minutes - for flash-attn, hours - compiling
what another machine like it already compiled: wheels without a binary on
PyPI, and the kernels Triton, Inductor and TileLang build on first use and
keep in a cache directory that dies with the machine. ``ravex cache pull``
fills such a directory from a store before the work starts, and ``ravex cache
push`` sends back what the work added.

**The key is everything that changes the binary.** Not "the architecture":
the CPU, every GPU's compute capability, CUDA, torch, Python, and the version
of each library named with ``--with``. A key too wide hands a machine a kernel
built for another combination, which crashes at best and computes the wrong
numbers in silence at worst. So the whole key is stored beside the files, and
a pull **compares it**, field by field, with the one this machine computes:
the short id in the store's path only finds the candidate, it does not vouch
for it.

**Who may write a store is the store's business, not this file's.** A shared
store that the machines it serves can write is a way to run code on each
other's machines, so a common cache is written by one trusted builder and only
read everywhere else; a person's own cache, in their own bucket, is theirs.
Nothing here can tell the two apart, which is why nothing here decides it.

**A directory is a set of files, never rewritten.** Wheels and the kernel
caches are named by their content, so a name already in the store already
holds those bytes: a push sends only the names the store lacks, a pull only
the names the directory lacks, and two machines pushing the same key at once
write the same bytes twice at worst.

The layout under the store's root::

    <name>/<id>/key.json      the whole key, written by the first push
    <name>/<id>/files/<path>  one object per file of the directory

Only the standard library, like the rest of Ravex: the store is a small
interface, :class:`Store`, so the transport is whatever implements it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

#: Characters of the key's hash in the store's path. Only for finding the key:
#: what decides is the comparison of the whole of it.
ID_CHARS = 16

#: Beside the files, the key they were built under.
KEY_FILE = "key.json"

#: Bumped when the layout or the key's fields change, so a store written by
#: one version is never read as another's.
LAYOUT = 1


class Store(Protocol):
    """Where the cache lives: objects addressed by a relative path."""

    def get(self, path: str) -> Optional[bytes]:
        """The object's bytes, or None when there is none."""

    def put(self, path: str, data: bytes) -> None:
        """Write the whole object."""

    def list(self, prefix: str) -> List[str]:
        """Every path under ``prefix``, as given to :meth:`put`."""


class LocalStore:
    """A store on a disk: for tests, and for machines sharing a filesystem."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def _path(self, path: str) -> str:
        return os.path.join(self.root, *path.split("/"))

    def get(self, path: str) -> Optional[bytes]:
        try:
            with open(self._path(path), "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    def put(self, path: str, data: bytes) -> None:
        _write_atomic(self._path(path), data)

    def list(self, prefix: str) -> List[str]:
        base = self._path(prefix)
        found = []
        for folder, _dirs, files in os.walk(base):
            for name in files:
                full = os.path.join(folder, name)
                found.append(prefix.rstrip("/") + "/" + os.path.relpath(full, base).replace(os.sep, "/"))
        return sorted(found)


@dataclass(frozen=True)
class Key:
    """What a directory's files were built under, and what a pull must match."""

    fields: Dict[str, object]

    @property
    def id(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:ID_CHARS]

    def canonical(self) -> str:
        return json.dumps(self.fields, sort_keys=True, separators=(",", ":"))

    def differences(self, other: Dict[str, object]) -> List[str]:
        """The fields that differ from ``other``, by name; empty when they match."""
        names = sorted(set(self.fields) | set(other))
        return [name for name in names if self.fields.get(name) != other.get(name)]


def gpu_capabilities() -> List[str]:
    """Every GPU's compute capability, as ``sm_89``, from the driver.

    Asked of ``nvidia-smi`` rather than of torch: importing torch to read one
    number costs seconds and starts CUDA, before a job that may want to set up
    CUDA its own way. An empty list is a machine without an NVIDIA GPU, which
    is a key of its own, not an error.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({"sm_" + line.strip().replace(".", "") for line in out.splitlines() if line.strip()})


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def compute_key(name: str, libraries: Sequence[str] = (), gpus: Optional[Sequence[str]] = None) -> Key:
    """This machine's key for the directory called ``name``.

    torch's version is read with its local part, ``2.11.0+cu128``, which is
    where a wheel says which CUDA it was built for: the CUDA a kernel links
    against is torch's, not whichever toolkit the machine has installed.
    """
    fields: Dict[str, object] = {
        "layout": LAYOUT,
        "name": name,
        "cpu": platform.machine().lower(),
        "os": sys.platform,
        "python": "cp%d%d" % sys.version_info[:2],
        "gpus": list(gpu_capabilities() if gpus is None else sorted(gpus)),
        "torch": _version("torch"),
    }
    for library in sorted(set(libraries)):
        fields["lib:" + library] = _version(library)
    return Key(fields)


def _write_atomic(path: str, data: bytes) -> None:
    """Write beside the target and rename, so a reader never sees half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".ravex-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def _local_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    found = []
    for folder, _dirs, files in os.walk(directory):
        for name in files:
            if name.startswith(".ravex-"):
                continue
            found.append(os.path.relpath(os.path.join(folder, name), directory).replace(os.sep, "/"))
    return sorted(found)


def _inside(directory: str, relative: str) -> Optional[str]:
    """The file's path in ``directory``, or None for a name that would leave it.

    The names come from the store, which this machine did not write: a name
    with ``..`` or an absolute one would put a file anywhere a pull can reach.
    """
    if not relative or relative.startswith("/") or "\\" in relative:
        return None
    if any(part in ("", ".", "..") for part in relative.split("/")):
        return None
    root = os.path.realpath(directory)
    target = os.path.realpath(os.path.join(root, *relative.split("/")))
    return target if os.path.commonpath([root, target]) == root else None


@dataclass
class Outcome:
    """What a pull or a push did, for the line the caller prints."""

    hit: bool
    files: int
    bytes: int
    reason: str = ""


def pull(store: Store, key: Key, directory: str) -> Outcome:
    """Fill ``directory`` with the files the store holds under ``key``.

    Loads nothing unless the stored key equals this one: a missing key, an
    unreadable one or one that differs in any field is a miss, and the reason
    says which.
    """
    base = "%s/%s" % (key.fields["name"], key.id)
    raw = store.get(base + "/" + KEY_FILE)
    if raw is None:
        return Outcome(False, 0, 0, "nothing stored for this key")
    try:
        stored = json.loads(raw)
    except ValueError:
        return Outcome(False, 0, 0, "the stored key is not JSON")
    if not isinstance(stored, dict):
        return Outcome(False, 0, 0, "the stored key is not an object")
    differ = key.differences(stored)
    if differ:
        return Outcome(False, 0, 0, "the stored key differs in " + ", ".join(differ))
    have = set(_local_files(directory))
    prefix = base + "/files/"
    count = size = 0
    for path in store.list(prefix):
        relative = path[len(prefix):]
        target = _inside(directory, relative)
        if target is None:
            raise ValueError("the store names a file outside the directory: %r" % relative)
        if relative in have:
            continue
        data = store.get(path)
        if data is None:
            continue
        _write_atomic(target, data)
        count += 1
        size += len(data)
    return Outcome(True, count, size)


def push(store: Store, key: Key, directory: str) -> Outcome:
    """Send the store the files of ``directory`` it does not hold under ``key``.

    The key goes first, so that the files are never there without it; a key
    already stored must be this one, or the push refuses rather than mix two
    builds under one id.
    """
    base = "%s/%s" % (key.fields["name"], key.id)
    raw = store.get(base + "/" + KEY_FILE)
    if raw is None:
        store.put(base + "/" + KEY_FILE, key.canonical().encode())
    else:
        try:
            stored = json.loads(raw)
        except ValueError:
            stored = None
        if not isinstance(stored, dict) or key.differences(stored):
            raise ValueError("%s holds another key; refusing to write over it" % base)
    prefix = base + "/files/"
    there = {path[len(prefix):] for path in store.list(prefix)}
    count = size = 0
    for relative in _local_files(directory):
        if relative in there:
            continue
        with open(os.path.join(directory, *relative.split("/")), "rb") as handle:
            data = handle.read()
        store.put(prefix + relative, data)
        count += 1
        size += len(data)
    return Outcome(raw is not None, count, size)


def open_store(address: str) -> Store:
    """The store at ``address``: ``file://<path>`` or a plain path.

    A bucket is not here yet: which S3 client Ravex uses for it is still to be
    decided (GPU-181), and until then a pull or push to ``s3://`` says so
    rather than guess.
    """
    if address.startswith("s3://"):
        raise NotImplementedError("ravex cache does not reach a bucket yet; use a directory (GPU-181)")
    if address.startswith("file://"):
        address = address[len("file://"):]
    return LocalStore(address)


def describe(outcome: Outcome, action: str, key: Key) -> str:
    if action == "pull" and not outcome.hit:
        return "cache %s %s: miss (%s)" % (key.fields["name"], key.id, outcome.reason)
    verb = "got" if action == "pull" else "sent"
    return "cache %s %s: %s, %s %d file(s), %d bytes" % (
        key.fields["name"], key.id, "hit" if outcome.hit else "new", verb, outcome.files, outcome.bytes
    )


def libraries(values: Iterable[str]) -> List[str]:
    """``--with`` values, each one or several comma-separated names."""
    return sorted({name.strip() for value in values for name in value.split(",") if name.strip()})
