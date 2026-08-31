"""Reading a torch distributed checkpoint — GPU-90.

This is the format Megatron-core writes. GPU-90 was raised when Megatron had a
manifest of its own and the work looked like three adapters; it has since moved
onto ``torch.distributed.checkpoint``, which a plain FSDP job also writes, so
one reader serves both and there is no Megatron-specific code here at all.

**Almost none of this is a reimplementation, and that is the point.** Unlike
the DeepSpeed side — where the layout had to be learned and the reassembly
written and then checked against DeepSpeed's own reader — torch ships the
reader for its own format. What was worth establishing was only *which* entry
point survives a version change, because the obvious one does not: torch 2.12
exposes ``format_utils._load_state_dict_from_keys`` and torch 2.8 does not, so
a reader built on it works where it was written and fails in the container.
``_EmptyStateDictLoadPlanner`` is in both, and is what torch's own
``dcp_to_torch_save`` uses.

Both names are private, which is a real risk and is handled by falling back
rather than by hoping: if they go, :func:`read` goes through
``dcp_to_torch_save`` and a temporary file instead. Slower, and the point is
that it still answers.

**What this does not do is reshape anything.** A DCP checkpoint written by a
job with tensor or pipeline parallelism records how the pieces of each tensor
were laid out, and the metadata says so; turning that into a different
parallelism is the same class of problem as :mod:`ravex._reshard` and is not
attempted here. What comes back is what the checkpoint holds, in the structure
it was saved in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DcpUnavailable(Exception):
    """This torch cannot read a distributed checkpoint offline.

    Distinct from a corrupt or unreadable checkpoint: the file may be perfect
    and this interpreter still unable to open it without a model to load into.
    Saying which of the two it is saves the next person from checking the
    checkpoint.
    """


def describe(path: str) -> Dict[str, Dict[str, Any]]:
    """Every entry's shape and dtype, without reading a tensor.

    ``{key: {"shape": (...), "dtype": "torch.float32"}}``, flattened the way
    DCP flattens it — ``model.0.weight``, ``optim.state.0.exp_avg`` — because
    that is the shape the checkpoint's own metadata is in and rearranging it
    here would be inventing a second convention beside the format's.

    This is the operation a conversion has to be able to do *before* deciding
    to convert: what is in here, how big is it, does it match the model this
    run is holding. Doing it by loading the tensors and measuring them is what
    makes a plan cost as much as the thing it is planning — the same point
    GPU-101 makes about Ravex's own stores, which is worth noting has an answer
    here and not there.

    Entries that are not tensors come back with a ``kind`` instead of a shape.
    A checkpoint carries more than tensors — step counters, parameter group
    hyperparameters — and dropping them silently would make this a description
    of the tensors rather than of the checkpoint.
    """
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(path).read_metadata()

    out: Dict[str, Dict[str, Any]] = {}
    for key, entry in metadata.state_dict_metadata.items():
        size = getattr(entry, "size", None)
        if size is None:
            out[str(key)] = {"kind": type(entry).__name__}
            continue
        properties = getattr(entry, "properties", None)
        out[str(key)] = {
            "shape": tuple(int(extent) for extent in size),
            "dtype": str(getattr(properties, "dtype", "")) or None,
        }
    return out


def read(path: str) -> Dict[str, Any]:
    """The whole checkpoint, in the structure it was saved in.

    Nested rather than flattened: ``dcp.save({"model": ..., "optim": ...})``
    comes back as those two keys. The flattening in :func:`describe` is the
    metadata's own and is a different view of the same thing, not a different
    answer — descriptions want one entry per tensor, restores want the shape
    the caller saved.

    ``no_dist=True`` throughout: this reads a checkpoint, it does not
    participate in one. A converter is a single process looking at a directory,
    and requiring a process group to look at a directory would mean a
    checkpoint could only be inspected by a job the size of the one that wrote
    it.
    """
    state: Dict[str, Any] = {}

    try:
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.format_utils import (
            _EmptyStateDictLoadPlanner,
            _load_state_dict,
        )
    except ImportError:
        return _read_the_long_way(path)

    _load_state_dict(
        state,
        storage_reader=FileSystemReader(path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state


def _read_the_long_way(path: str) -> Dict[str, Any]:
    """Through ``dcp_to_torch_save`` and a temporary file.

    Only reached when the private names :func:`read` prefers have moved. It
    writes the whole checkpoint out and reads it back, so it costs a full copy
    on disk and is not what anyone would choose — but a slow answer is a
    different thing from no answer, and a converter that stops working on a
    torch upgrade is worse than one that gets slower.
    """
    import os
    import tempfile

    import torch

    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    except ImportError as exc:
        raise DcpUnavailable(
            "this torch offers no way to read a distributed checkpoint without "
            "a model to load into (%s)" % exc
        ) from exc

    handle, target = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    try:
        dcp_to_torch_save(path, target)
        return torch.load(target, map_location="cpu", weights_only=False)
    finally:
        try:
            os.unlink(target)
        except OSError:  # pragma: no cover - a temp file that vanished
            pass


def tensors(state: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested state dict to ``{dotted.key: tensor}``.

    The bridge between what :func:`read` returns and what :func:`describe`
    lists, so the two can be compared key for key — which is how a conversion
    checks it read everything the metadata said was there rather than trusting
    that it did.

    Lists are indexed by position, because that is what DCP's own flattening
    does: ``optim.param_groups.0.lr``. Following the format's convention
    matters more here than picking a nicer one.
    """
    import torch

    out: Dict[str, Any] = {}
    if torch.is_tensor(state):
        out[prefix] = state
        return out
    if isinstance(state, dict):
        for key, value in state.items():
            out.update(tensors(value, "%s.%s" % (prefix, key) if prefix else str(key)))
        return out
    if isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            out.update(tensors(value, "%s.%d" % (prefix, index) if prefix else str(index)))
        return out
    return out


def missing_from(read_state: Any, described: Dict[str, Dict[str, Any]]) -> List[str]:
    """Keys the metadata promised that the read did not produce, sorted.

    Empty is the expected answer, and the reason to ask anyway is that the
    failure it catches is silent: a checkpoint read through a planner that
    skipped something comes back looking complete, and the only witness is the
    metadata it was written with.
    """
    have = set(tensors(read_state))
    want = {key for key, entry in described.items() if "shape" in entry}
    return sorted(want - have)
