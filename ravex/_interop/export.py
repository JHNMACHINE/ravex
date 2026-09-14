"""Writing a Ravex checkpoint out as a torch distributed checkpoint — GPU-90.

The reading half is the rest of this package: a DeepSpeed ZeRO store or a DCP
directory becomes something Ravex's resume consumes. This is the other
direction, and the one the issue's own example needed — *start in FSDP, resume
in Megatron* — because Megatron-core, a plain FSDP job and anything built on
``torch.distributed.checkpoint.load`` read DCP. Ravex's store is a format only
Ravex reads; this is how a checkpoint leaves it.

**What is written, and how exact each half is.** It depends on what the
checkpoint holds, and the difference is reported as notes rather than smoothed
over:

* A **sharded group in** ``gather`` **layout** exports exactly. Its model state
  is keyed by fully qualified name and so is its optimizer state, because it
  was collected through ``get_state_dict`` — which is the form a DCP loader on
  the other side expects.
* A **plain model** exports its weights exactly. Its optimizer does not come
  out keyed by name, because it was collected with ``optimizer.state_dict()``
  and that keys the moments by the parameter's *position* in an ordering only
  the writing run knew. Written as it is and said so, which is exactly what
  :func:`ravex._interop.convert._unify_dcp` recognises and reports on the way
  back in. Inventing names by assuming the optimizer was built from
  ``model.parameters()`` in order would be right most of the time and silently
  wrong the rest — moments on the wrong parameters, with every shape plausible.
* A ``per_rank`` checkpoint is **refused**. Each rank's store holds a shard
  that only means something at the topology that wrote it; exporting one would
  write a fraction of a model under a format that does not record it is one.

Written with ``dcp.save(..., no_dist=True)``, the public entry point, present
since well before torch 2.8 — a single process writing a directory, no process
group, for the same reason the reader needs none.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ravex._interop.convert import CannotConvert


def _choose(state: Dict[str, Any], key: Optional[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[str]]:
    """The model and optimizer state to write, and what is worth saying about them."""
    sharded = state.get("sharded") or {}
    models = state.get("models") or {}
    optimizers = state.get("optimizers") or {}

    candidates = list(sharded) + [name for name in models if name not in sharded]
    if key is None:
        if len(candidates) != 1:
            raise CannotConvert(
                "this checkpoint holds %d models (%s); name the one to export"
                % (len(candidates), ", ".join(candidates) or "none")
            )
        key = candidates[0]
    elif key not in candidates:
        raise CannotConvert(
            "no model %r in this checkpoint; it holds %s"
            % (key, ", ".join(candidates) or "none")
        )

    if key in sharded:
        group = sharded[key]
        if group.get("layout") == "per_rank":
            raise CannotConvert(
                "%r was written per rank (rank %s of %s): each store holds one "
                "shard, meaningful only at the topology that wrote it. Resume "
                "it at that size, or reshard it, and export the gathered result"
                % (key, group.get("rank"), group.get("world_size"))
            )
        notes = ["model and optimizer taken from sharded group %r, keyed by name" % key]
        optimizer = group.get("optimizer") or None
        if not optimizer:
            notes.append("the group carries no optimizer state")
        return group["model"], optimizer, notes

    notes = ["model taken from %r" % key]
    optimizer = None
    if len(optimizers) == 1:
        optimizer = next(iter(optimizers.values()))
        notes.append(
            "the optimizer state is keyed by position, as optimizer.state_dict() "
            "writes it: it restores onto a model whose optimizer lists the same "
            "parameters in the same order, and cannot be matched by name"
        )
    elif optimizers:
        notes.append(
            "%d optimizers in this checkpoint and no way to tell which one "
            "belongs to %r; none exported" % (len(optimizers), key)
        )
    else:
        notes.append("no optimizer state in this checkpoint")
    return models[key], optimizer, notes


def to_dcp(state: Dict[str, Any], out_dir: str, key: Optional[str] = None) -> List[str]:
    """Write one model of a Ravex checkpoint, and its optimizer, as DCP.

    ``state`` is a checkpoint as a backend's ``load_step`` or ``load_latest``
    returns it. The directory gets ``{"model": ..., "optim": ...}`` — the two
    top-level keys a DCP reader looks for first — plus ``ravex``, which records
    the step and where the state came from so the export can be traced back.

    Returns notes on what was and was not exported exactly.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter

    model, optimizer, notes = _choose(state, key)
    payload: Dict[str, Any] = {
        "model": dict(model),
        "ravex": {"step": state.get("step"), "exported_from": "ravex"},
    }
    if optimizer:
        payload["optim"] = optimizer

    dcp.save(payload, storage_writer=FileSystemWriter(out_dir), no_dist=True)
    return notes


def load_store(path: str, backend: str = "moonclip", step: Optional[int] = None) -> Dict[str, Any]:
    """Open a Ravex store read-only and return one checkpoint from it.

    The latest when ``step`` is not given. Raises :class:`CannotConvert` when
    there is nothing there, rather than handing an empty state to the writer.
    """
    from ravex._backends import get_backend
    from ravex._config import RavexConfig

    config = RavexConfig()
    config.storage.path = path
    config.backend = backend
    config._normalize()
    store = get_backend(config)
    try:
        state = store.load_latest() if step is None else store.load_step(step)
    finally:
        store.close()
    if not state:
        raise CannotConvert(
            "no checkpoint %sin the %s store at %s"
            % ("" if step is None else "at step %d " % step, backend, path)
        )
    return state
