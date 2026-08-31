"""Who wrote a per-rank store, and as part of which run.

With ``per_rank`` and local storage each machine holds a different part of one
checkpoint, and nothing on disk used to say which machine that was or which
training run it belonged to. Both gaps cost something concrete:

* A resume that finds nothing can tell "this checkpoint was never written" from
  "it was written on a machine that is not in this job" only if the stores say
  where they came from.
* Two runs can leave their stores side by side on one disk. Reproduced on
  2026-08-19: a run restarted with a different rank-to-node placement started
  from scratch, wrote a second history next to the first, and a later restart
  resumed the accidental one while the original sat one directory away. Without
  a run identity the two are indistinguishable, and any scheme that copies
  shards between machines could copy from the wrong one.

The record lives in a small JSON file beside the store rather than inside the
checkpoint, because the code that needs it is deciding *whether* to open a
checkpoint. Reading it must not require constructing a backend.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from typing import Any, Dict, Optional, Tuple

#: Written inside each ``rank_<n>`` store.
OWNER_FILE = ".ravex-owner"

#: How a run id was arrived at, best first. A run where one rank inherited an
#: id from its own store and another had to invent one should keep the
#: inherited one — the invented id names nothing, while the inherited id names
#: the history being continued.
FROM_CONFIG = 0
FROM_SCHEDULER = 1
FROM_STORE = 2
GENERATED = 3

PROVENANCE_NAMES = {
    FROM_CONFIG: "config",
    FROM_SCHEDULER: "scheduler",
    FROM_STORE: "inherited",
    GENERATED: "generated",
}


def scheduler_run_id() -> Optional[str]:
    """A job-wide id from whatever launched this, or None.

    ``TORCHELASTIC_RUN_ID`` is only meaningful when ``torchrun`` was given
    ``--rdzv-id``. With the static rendezvous — which is what a plain
    ``--nnodes/--node_rank`` invocation uses — it is set to the literal string
    ``"none"``, verified on 2026-08-19. Taking it at face value would give
    every unrelated run on the machine the same identity, which is worse than
    having none.
    """
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        return "slurm-%s" % slurm

    elastic = os.environ.get("TORCHELASTIC_RUN_ID")
    if elastic and elastic.lower() not in ("none", "null", ""):
        return "torchelastic-%s" % elastic

    return None


def local_run_id(config, store_path: Optional[str] = None) -> Tuple[int, str]:
    """This rank's best guess at the run identity, with where it came from.

    The provenance matters more than the id: it is what lets the ranks agree
    on the best answer among them rather than on an arbitrary one. See
    :func:`ravex._dist.collectives.agree_on_run_id`.
    """
    if getattr(config, "run_id", None):
        return FROM_CONFIG, str(config.run_id)

    scheduled = scheduler_run_id()
    if scheduled:
        return FROM_SCHEDULER, scheduled

    if store_path:
        existing = read_owner(store_path)
        if existing and existing.get("run_id"):
            return FROM_STORE, str(existing["run_id"])

    return GENERATED, "run-%s" % uuid.uuid4().hex[:16]


def describe_machine() -> Dict[str, Any]:
    """Enough to name the machine in a message a person has to act on.

    The hostname is what appears in a scheduler's node list and in the shell
    prompt of the box being debugged. ``GROUP_RANK`` is torchrun's index for
    the node, which survives a hostname that is a container id.
    """
    node_rank = os.environ.get("GROUP_RANK")
    try:
        node = int(node_rank) if node_rank is not None else None
    except ValueError:
        node = None

    return {"host": socket.gethostname(), "node_rank": node}


def owner_path(store_path: str) -> str:
    return os.path.join(store_path, OWNER_FILE)


def read_owner(store_path: str) -> Optional[Dict[str, Any]]:
    """The record beside a store, or None if there is not a readable one.

    Never raises. Every caller is deciding what to *say* about a checkpoint,
    and a broken sidecar must degrade the explanation rather than the run.
    """
    try:
        with open(owner_path(store_path), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def run_id_at(store_path: str) -> Optional[str]:
    """Which run a store — or a copy of one — belongs to, or None.

    The narrow question, separated from :func:`read_owner` because one caller
    asks nothing else and asks it about directories that may not exist. A copy
    with no readable record cannot be attributed to any history, and a copy
    that cannot be attributed is one nothing should be resumed from.
    """
    record = read_owner(store_path) or {}
    value = record.get("run_id")
    return str(value) if value else None


def write_owner(store_path: str, run_id: str, rank: int, world_size: int) -> None:
    """Record who this store belongs to. Never raises.

    Best-effort on purpose: this is metadata for a future diagnosis, and a
    read-only or full filesystem must not take down a run that is otherwise
    checkpointing fine. The checkpoint itself is written by the backend, which
    reports its own failures.
    """
    record = {"run_id": run_id, "rank": rank, "world_size": world_size}
    record.update(describe_machine())

    try:
        os.makedirs(store_path, exist_ok=True)
        # Written whole and replaced, so a reader never sees half a record.
        temporary = owner_path(store_path) + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
        os.replace(temporary, owner_path(store_path))
    except OSError:
        return
