"""A run as a document in its own store (GPU-148).

Two files, and the split between them is the point.

``run.json`` is **what this run is**, written once when the store is born and
never rewritten: its id, the name a person gave it, when it started, the
configuration it started with, and - once forks exist (GPU-149) - which run and
which step it came from. A resume does not touch it, because a resume is the
same run coming back; if it rewrote the file, the id would change under
everything that had quoted it.

``status.json`` is **where it has got to**: running, finished or failed, the
last step, the time of that answer. It is overwritten as the run goes, which is
also why it cannot live in the first file.

**An id is not a name.** The name is what somebody reads and can repeat between
experiments - three runs called ``baseline`` is normal. The id is what gets
quoted in a command, a message or a fork's lineage, and never repeats. The
dashboard shows both, so both have to exist.

Nothing here imports the Rust core or torch: the reader in :mod:`ravex.runs`
has to load where neither can, such as Cloudflare's Python Workers.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("ravex")

#: Written once, at the store's birth.
RUN_FILE = "run.json"

#: Overwritten as the run goes.
STATUS_FILE = "status.json"

FORMAT_VERSION = 1

#: Configuration fields that never reach disk here. The same list the audit
#: trail scrubs, for the same reason: this file is meant to be read by a
#: dashboard, which means it is meant to be shared.
_EXCLUDED = {"access_key", "secret_key", "source", "problems"}


def new_run_id() -> str:
    """Short, sortable by birth, and not repeated.

    The time prefix is what makes a list of ids readable in the order the runs
    happened; the random tail is what keeps two runs started in the same second
    on two machines apart.
    """
    return "r%s-%s" % (time.strftime("%y%m%d%H%M", time.gmtime()), uuid.uuid4().hex[:6])


def default_name(store_path: str) -> str:
    """The store's own directory name, which is what a person already called it."""
    cleaned = os.path.basename(os.path.abspath(store_path))
    return cleaned or "run"


def scrub_config(config: Any) -> Dict[str, Any]:
    """The configuration as JSON, credentials removed."""
    import dataclasses

    def plain(value: Any) -> Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: plain(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if field.name not in _EXCLUDED
            }
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items() if key not in _EXCLUDED}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    scrubbed = plain(config)
    return scrubbed if isinstance(scrubbed, dict) else {}


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: str, payload: Dict[str, Any]) -> bool:
    """Write the file atomically. Returns whether it landed.

    Atomically because a dashboard reads these while a run writes them: a
    half-written ``status.json`` is a run that appears to have no status, which
    reads as a crashed run.
    """
    partial = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        os.replace(partial, path)
        return True
    except OSError as exc:
        logger.warning("Cannot write %s: %s", path, exc)
        return False


def ensure_run(
    directory: str,
    *,
    run_id: Optional[str] = None,
    name: Optional[str] = None,
    config: Any = None,
    parent: Optional[Dict[str, Any]] = None,
    version: str = "",
) -> Dict[str, Any]:
    """Return this store's ``run.json``, writing it if the store is new.

    An existing file wins over every argument, including a ``run_id`` from the
    configuration: the store already has an identity, and a resume that renamed
    it would break every reference to it. A configuration that disagrees is
    worth one line in the log, not an overwrite.
    """
    path = os.path.join(directory, RUN_FILE)
    existing = read_json(path)
    if existing is not None:
        if run_id and existing.get("run_id") not in (None, run_id):
            logger.info(
                "This store belongs to run %s; run_id=%s from the configuration is ignored",
                existing.get("run_id"),
                run_id,
            )
        return existing

    record = {
        "format": FORMAT_VERSION,
        "run_id": run_id or new_run_id(),
        "name": name or default_name(directory),
        "created_at": time.time(),
        "created_on": socket.gethostname(),
        "ravex": version,
        "parent": parent,
        "config": scrub_config(config) if config is not None else {},
    }
    write_json(path, record)
    return record


def write_status(
    directory: str,
    *,
    state: str,
    step: int,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Overwrite ``status.json``. ``state`` is running, finished or failed."""
    payload: Dict[str, Any] = {
        "format": FORMAT_VERSION,
        "run_id": run_id,
        "state": state,
        "step": step,
        "updated_at": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    write_json(os.path.join(directory, STATUS_FILE), payload)
    return payload
