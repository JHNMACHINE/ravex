"""Read what a run is and where it has got to (GPU-148).

::

    import ravex.runs

    run = ravex.runs.describe("./checkpoints")
    run["run_id"], run["name"], run["parent"]
    run["status"]["state"], run["status"]["step"]

The path is the run's ``storage.path``, the same one its checkpoints and its
metrics are in. ``None`` means no Ravex run has written there.

**Name and id are different things**, and both are here. The name is what a
person called it and may repeat between experiments; the id is what a command
or a fork's lineage quotes, and does not. A store that predates this file has
neither, and :func:`describe` says so by returning ``None`` rather than
inventing them.

**Whether a run is alive** is ``status.state``, plus how old ``status.updated_at``
is. A process killed outright leaves ``running`` behind for ever - nothing gets
to write "I was killed" - so a reader treats a stale timestamp as stopped. The
staleness that counts is the run's own checkpoint cadence, which the
configuration in ``run.json`` carries, so the judgement belongs to whoever is
displaying it: :func:`looks_alive` is the default answer, not the only one.

Like :mod:`ravex.metrics`, this imports nothing compiled.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ravex._runs import RUN_FILE, STATUS_FILE, read_json

__all__ = ["describe", "discover", "looks_alive"]

#: How long a ``running`` status can go unrefreshed before a reader stops
#: believing it. The runtime writes one at every checkpoint and at exit.
STALE_SECONDS = 120.0


def describe(path: str) -> Optional[Dict[str, Any]]:
    """What the store at ``path`` says about its run, or ``None``."""
    record = read_json(os.path.join(path, RUN_FILE))
    if record is None:
        return None
    described = dict(record)
    described["status"] = read_json(os.path.join(path, STATUS_FILE)) or {
        "state": "unknown",
        "step": None,
        "updated_at": None,
    }
    return described


def looks_alive(described: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Whether this run is probably still writing.

    ``running`` and refreshed recently. A killed process never gets to say it
    stopped, so an old timestamp counts as stopped however the state reads.
    """
    import time

    status = described.get("status") or {}
    if status.get("state") != "running":
        return False
    updated = status.get("updated_at")
    if not isinstance(updated, (int, float)):
        return False
    return (now or time.time()) - updated < STALE_SECONDS


def discover(root: str) -> List[Dict[str, Any]]:
    """Every run in the directories directly under ``root``, newest first.

    One directory, one store, one run - which is how a dashboard is pointed at
    a machine's runs without being told each one.
    """
    found: List[Dict[str, Any]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return found
    for name in sorted(names):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        described = describe(path)
        if described is not None:
            described["path"] = path
            found.append(described)
    found.sort(key=lambda run: run.get("created_at") or 0, reverse=True)
    return found
