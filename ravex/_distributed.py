"""Distributed helpers.

With a replicated model only rank 0 writes checkpoints. Every rank must still
*load* on resume, so the read path is not gated on rank.

Sharded models have two ways through here, and they trade against each other:

``gather_sharded_state``
    Collect the whole unsharded state on rank 0. The checkpoint comes out
    independent of the topology that wrote it — eight GPUs in, one out — and it
    does not scale: measured on 8 GPUs with a 1.48B model, 14.2 s per gather
    (FSDP2) and 18.1 GiB of resident memory on rank 0, which is the entire
    state.

``local_sharded_state``
    Each rank keeps its own shard and writes it itself. Nothing is gathered, so
    nothing is bounded by one rank's memory — and the checkpoint is now tied to
    the topology: it can only be resumed at the same world size, with the same
    sharding.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("ravex")


def _dist():
    try:
        import torch.distributed as dist
    except Exception:
        return None
    return dist


_torch_numpy: Optional[bool] = None


def _torch_can_reach_numpy() -> bool:
    """Whether ``tensor.numpy()`` works in this interpreter.

    Not ``import numpy``: torch initialises NumPy itself, and a version it was
    not built against imports perfectly well and then fails the conversion. The
    probe asks the question the object collectives actually ask, once.

    ``RAVEX_ASSUME_NO_NUMPY=1`` forces the answer to no. That exists because
    the NumPy-free path in :func:`_all_gather_object` is otherwise unreachable
    on any ordinary machine — every image that matters ships NumPy — so the
    code that replaces torch's object collectives would go to a release having
    never run on a network. A diagnostic, not a feature: it makes an untested
    path testable, and there is no reason to set it in a real run.
    """
    global _torch_numpy
    if _torch_numpy is None:
        if os.environ.get("RAVEX_ASSUME_NO_NUMPY", "") not in ("", "0", "false"):
            _torch_numpy = False
            return _torch_numpy
        try:
            import torch

            torch.zeros(1, dtype=torch.uint8).numpy()
            _torch_numpy = True
        except Exception:
            _torch_numpy = False
    return _torch_numpy


def _object_device(dist):
    """Where the byte tensors behind an object collective have to live.

    NCCL only moves GPU tensors, which is why torch's own object collectives
    reach for the current CUDA device; gloo stays on the host.
    """
    import torch

    try:
        backend = str(dist.get_backend())
    except Exception:
        backend = ""
    if "nccl" in backend and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _all_gather_object(dist, value) -> List[Any]:
    """One picklable value from every rank, in rank order. **Collective.**

    ``dist.all_gather_object`` decodes what it gathered with
    ``tensor.numpy().tobytes()``, so on an install where that conversion is
    unavailable every object collective in this module raises "Numpy is not
    available" — after the wire work is done, which makes it a crash rather
    than something a caller could fall back from. Ravex depends on PyYAML and
    nothing else, and torch itself does not require NumPy, so that install is a
    configuration we have to keep working in.

    The replacement posts the same two ``all_gather`` calls, in the same order
    and with the same shapes and dtypes as torch's own, so a rank that took
    this path and a rank that did not still meet on the wire.
    """
    world = dist.get_world_size()
    if _torch_can_reach_numpy():
        # Pre-sized because `all_gather_object` fills the list in place.
        gathered: List[Any] = [None] * world
        dist.all_gather_object(gathered, value)
        return gathered

    import pickle

    import torch

    device = _object_device(dist)
    payload = torch.frombuffer(bytearray(pickle.dumps(value)), dtype=torch.uint8)
    payload = payload.to(device)

    sizes = torch.zeros(world, dtype=torch.long, device=device)
    length = torch.tensor([payload.numel()], dtype=torch.long, device=device)
    dist.all_gather([sizes[i].unsqueeze(0) for i in range(world)], length)

    # Every rank sends the same number of bytes, so the short ones are padded
    # and the length gathered above says where each one really ends.
    widest = int(sizes.max().item())
    padded = torch.zeros(widest, dtype=torch.uint8, device=device)
    padded[: payload.numel()] = payload
    chunks = [torch.empty(widest, dtype=torch.uint8, device=device) for _ in range(world)]
    dist.all_gather(chunks, padded)

    # `bytes(tensor.tolist())` is the NumPy-free spelling of `.numpy().tobytes()`.
    return [
        pickle.loads(bytes(chunk[: int(size)].cpu().tolist()))
        for chunk, size in zip(chunks, sizes.tolist())
    ]


def get_rank() -> int:
    """Global rank of this process, 0 when not distributed."""
    dist = _dist()
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def get_world_size() -> int:
    dist = _dist()
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def is_main_process() -> bool:
    return get_rank() == 0


def local_world_size() -> int:
    """Ranks sharing this machine, as the launcher reports it.

    ``torchrun`` sets ``LOCAL_WORLD_SIZE``; SLURM's equivalent is
    ``SLURM_NTASKS_PER_NODE``. Read from the environment rather than derived,
    because the process group knows how many ranks there are and nothing about
    which of them share a filesystem.

    Falls back to the whole world, which makes a single-node job the assumption
    when nothing says otherwise — the reading that raises no false alarm.
    """
    for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
        value = os.environ.get(var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return get_world_size()


def spans_several_machines() -> bool:
    """Whether this job runs on more than one machine."""
    return get_world_size() > local_world_size()


def is_local_main_process() -> bool:
    """First rank on this machine. ``True`` when nothing says otherwise.

    For things that are true once per machine rather than once per rank —
    anything about the local filesystem — where logging per rank would repeat
    itself once per GPU.
    """
    value = os.environ.get("LOCAL_RANK")
    if value is None:
        return is_main_process()
    try:
        return int(value) == 0
    except ValueError:
        return is_main_process()


def barrier() -> None:
    """Synchronize all ranks, if a process group is up."""
    dist = _dist()
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()


def byte_transport_group():
    """A group that can carry **CPU** tensors: ``(usable, group)``. Collective.

    Replication moves a store as bytes, and bytes live on the host. NCCL is a
    GPU collective library and refuses them outright — measured on 8x RTX 5060
    Ti on 2026-08-21, where every replication round failed with
    ``No backend type associated with device type cpu`` and not one copy was
    ever made. The job carried on believing it was protected, because the
    announcement at activation had promised it was.

    Three answers, and the caller has to tell them apart:

    * ``(True, None)`` — the default group already carries CPU tensors, which
      is the case on a gloo job. Building a second group there would cost a
      collective and buy nothing.
    * ``(True, group)`` — a gloo subgroup, built once here and reused. Every
      rank is a member; the ring only ever pairs ranks inside it.
    * ``(False, None)`` — no transport for bytes on this job. The caller must
      then say replication is **off**, at activation, rather than let every
      round fail one warning at a time.

    ``new_group`` is itself collective: every rank calls it, or the ones that
    skipped hang the ones that did. Hence "collective" above, and hence being
    called from a branch every rank takes together.
    """
    dist = _dist()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return False, None

    try:
        backend = str(dist.get_backend()).lower()
    except Exception:  # pragma: no cover - a group in an odd state
        return False, None

    if "gloo" in backend:
        return True, None

    if not dist.is_gloo_available():
        # Nothing to fall back to. Saying so beats failing per round.
        return False, None

    try:
        group = dist.new_group(backend="gloo")
    except Exception as exc:
        logger.warning(
            "Could not open a gloo group for moving checkpoint bytes between "
            "machines: %s. Copies between machines are off for this run.",
            exc,
        )
        return False, None

    return True, group


def is_sharded(model) -> bool:
    """Whether this model's parameters are split across ranks.

    Covers both generations: FSDP1 wraps modules in
    ``FullyShardedDataParallel``, while FSDP2's ``fully_shard`` leaves the
    module in place and turns its parameters into ``DTensor``s. Either way an
    ordinary ``state_dict()`` yields this rank's shard, which is useless on its
    own.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - defensive
        return False

    # FSDP2 / any DTensor-parameterised module.
    try:
        from torch.distributed.tensor import DTensor

        for param in model.parameters():
            if isinstance(param, DTensor):
                return True
            break  # one parameter is enough to tell
    except Exception:
        pass

    # FSDP1, whether the root is wrapped or only some submodules are.
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(model, FullyShardedDataParallel):
            return True
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                return True
    except Exception:
        pass

    return False


def gather_sharded_state(model, optimizers):
    """Gather a sharded model and its optimizers into full CPU tensors.

    **Collective.** Every rank must call this, at the same point, or the ones
    that do will wait forever for the ones that did not.

    The result is a full, unsharded state dict, which is what makes the
    checkpoint independent of the topology that wrote it: a run sharded over
    eight GPUs can be resumed on one. The cost is a gather and the peak memory
    that comes with it, which is why ``cpu_offload`` is on.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_state_dict(model, list(optimizers), options=options)


#: Marks a value in a saved state tree as one rank's shard of a DTensor.
_SHARD_TAG = "__ravex_shard__"


def _dtensor_class() -> Any:
    """``DTensor``, or None on a torch too old to have it."""
    try:
        from torch.distributed.tensor import DTensor

        return DTensor
    except Exception:
        pass
    try:  # torch < 2.5 kept it private, and private is the whole point here
        from torch.distributed._tensor import (
            DTensor,  # pyright: ignore[reportPrivateImportUsage]
        )

        return DTensor
    except Exception:
        return None


def _is_sharded_tensor(value: Any) -> bool:
    """The pre-DTensor sharded type, which this path cannot rebuild."""
    try:
        # Private by nature: there is no public name for the type this exists
        # to recognise and refuse.
        from torch.distributed._shard.sharded_tensor import (
            ShardedTensor,  # pyright: ignore[reportPrivateImportUsage]
        )
    except Exception:
        return False
    return isinstance(value, ShardedTensor)


def _encode_shards(value: Any) -> Any:
    """Replace every DTensor in a state tree with this rank's own shard.

    What comes back is plain tensors and plain data, which is all the storage
    backends can take. The global shape and the placements ride along so a
    resume onto a different layout can *say so* instead of failing with a shape
    error from inside ``set_state_dict`` — and, since :mod:`ravex._reshard`,
    so it can do something about it.

    The placements are stored as data rather than as ``str(placement)``. The
    old form is still read (see :func:`ravex._reshard.decode_placements`); it
    is not still written, because the reshard planner has to ask which
    dimension a tensor was split along and parsing torch's ``repr`` for the
    answer is a dependency on a string nobody promised to keep.
    """
    DTensor = _dtensor_class()
    if DTensor is not None and isinstance(value, DTensor):
        from ravex._reshard import encode_placement

        return {
            _SHARD_TAG: 1,
            "local": value.to_local().detach(),
            "global_shape": list(value.shape),
            "placements": [encode_placement(p) for p in value.placements],
            # The mesh the placements index into. Recorded so a reshard can
            # refuse a 2-D mesh by looking at the checkpoint instead of
            # inferring it from how many placements happen to be shards.
            "mesh_shape": _mesh_shape(value),
        }
    if _is_sharded_tensor(value):
        # FSDP1, whatever `use_orig_params` is set to. Measured on torch
        # 2.12: with `use_orig_params=True` the parameters are plain
        # `Parameter`s and the sharded state dict still yields
        # `ShardedTensor`, which carries no mesh and no placements and so
        # cannot be rebuilt the way `_rebuild_dtensor` rebuilds a shard.
        raise TypeError(
            "per-rank checkpointing needs DTensor-backed shards; this model "
            "yields ShardedTensor, which is what FSDP1 gives on this torch"
        )
    if isinstance(value, dict):
        return {k: _encode_shards(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_shards(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_encode_shards(v) for v in value)
    return value


def _mesh_shape(value: Any) -> Optional[List[int]]:
    """The device mesh's shape as plain integers, or None if it will not say.

    Best-effort by design: the mesh is recorded to make a refusal specific, and
    a checkpoint that cannot describe its mesh should still be written. The
    reshard planner already refuses on the placements alone.
    """
    mesh = getattr(value, "device_mesh", None)
    if mesh is None:
        return None
    shape = getattr(mesh, "shape", None)
    try:
        return [int(n) for n in shape] if shape is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _rebuild_dtensor(saved: Dict[str, Any], live: Any) -> Any:
    """Wrap a saved local shard in the layout the live model is using now.

    The mesh and the placements are taken from ``live`` rather than from the
    checkpoint: a device mesh holds process groups and does not survive being
    written to storage, and the layout that matters is the one this process is
    running, not the one that wrote the bytes.
    """
    DTensor = _dtensor_class()
    if DTensor is None:  # pragma: no cover - `live` being a DTensor rules it out
        raise ValueError("this torch has no DTensor to rebuild a shard into")
    reference = live.to_local()
    local = saved["local"]

    if tuple(local.shape) != tuple(reference.shape):
        raise ValueError(
            "saved shard is %s, this rank holds %s — the sharding differs from "
            "the run that wrote it" % (tuple(local.shape), tuple(reference.shape))
        )

    local = local.to(device=reference.device, dtype=reference.dtype)

    # `shape`/`stride` are passed so uneven shards survive: without them
    # `from_local` infers the global shape as local × mesh size, which is wrong
    # for the last rank of a dimension that does not divide evenly. Torch is
    # not a declared dependency — Ravex attaches to whatever build is already
    # installed — so a build without them is reachable, and the `except` below
    # is what happens then.
    try:
        return DTensor.from_local(
            local,
            live.device_mesh,
            live.placements,
            run_check=False,
            shape=live.shape,
            stride=live.stride(),
        )
    except TypeError:
        # An older signature, with no way to state the global shape. What it
        # infers instead is local x mesh size, which is right whenever the
        # tensor divides evenly across the mesh and wrong otherwise — so the
        # fallback is usable, but only after checking that this is one of the
        # cases where it agrees.
        #
        # Checked rather than assumed because the disagreement is silent: the
        # rebuilt DTensor would carry a global shape nobody asked for, every
        # shard individually valid, and the first symptom somewhere far away.
        # Every rank reaches the same verdict — with an uneven split the ranks
        # holding a full chunk infer too much and the short one too little, so
        # none of them agrees with `live.shape` — which is what keeps this from
        # stranding some ranks inside a collective while others raise.
        inferred = _inferred_global_shape(local.shape, live.placements, live.device_mesh)
        if tuple(inferred) != tuple(live.shape):
            raise ValueError(
                "this torch's DTensor.from_local does not take shape=/stride=, "
                "so it would rebuild this shard as %s instead of %s. The tensor "
                "does not divide evenly across the mesh, and the difference is "
                "exactly what those arguments exist to carry. Upgrade to a "
                "torch whose from_local accepts them, or run at a world size "
                "that divides this tensor evenly."
                % (tuple(inferred), tuple(live.shape))
            )
        return DTensor.from_local(local, live.device_mesh, live.placements)


def _inferred_global_shape(local_shape, placements, mesh) -> tuple:
    """The global shape `from_local` works out when it is not given one.

    It multiplies each sharded dimension by the size of the mesh dimension it
    is sharded over, which is the even-split assumption. Kept as a function of
    plain values so the arithmetic can be tested without a process group —
    the path that needs it only runs on a torch this dev box does not have.

    A placement is a shard exactly when it carries a `dim`; `Replicate` and
    `Partial` do not, and neither multiplies anything.
    """
    inferred = list(local_shape)
    for mesh_dim, placement in enumerate(placements):
        dim = getattr(placement, "dim", None)
        if dim is not None:
            inferred[dim] *= mesh.size(mesh_dim)
    return tuple(inferred)


def _decode_shards(saved: Any, live: Any) -> Any:
    """Undo :func:`_encode_shards`, walking the live state tree alongside."""
    if isinstance(saved, dict) and saved.get(_SHARD_TAG):
        DTensor = _dtensor_class()
        if DTensor is None or not isinstance(live, DTensor):
            raise ValueError(
                "checkpoint holds a shard for a tensor this run does not shard"
            )
        return _rebuild_dtensor(saved, live)
    if isinstance(saved, dict):
        return {
            key: _decode_shards(
                value, live.get(key) if isinstance(live, dict) else None
            )
            for key, value in saved.items()
        }
    if isinstance(saved, (list, tuple)):
        live_items = live if isinstance(live, (list, tuple)) else ()
        decoded = [
            _decode_shards(item, live_items[i] if i < len(live_items) else None)
            for i, item in enumerate(saved)
        ]
        return type(saved)(decoded) if isinstance(saved, tuple) else decoded
    return saved


#: Where one shard sits in an encoded state tree: dict keys and list indices,
#: from the root down. Used as a dictionary key and sent through
#: ``all_gather_object``, so it is a tuple of plain strings and integers.
ShardPath = Tuple[Any, ...]


def walk_shards(tree: Any, path: ShardPath = ()):
    """Yield ``(path, node)`` for every shard in an encoded state tree.

    A shard node is a leaf as far as this is concerned: it is a dict, but the
    walk stops there rather than descending into ``local`` and ``placements``.
    """
    if isinstance(tree, dict):
        if tree.get(_SHARD_TAG):
            yield path, tree
            return
        for key, value in tree.items():
            yield from walk_shards(value, path + (key,))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from walk_shards(value, path + (index,))


def node_at(tree: Any, path: ShardPath) -> Any:
    """The node ``path`` names, or None if this tree does not have one there."""
    node = tree
    for step in path:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return None
    return node


def path_name(path: ShardPath) -> str:
    """A path as something a log line can print."""
    return ".".join(str(step) for step in path) or "<root>"


def shard_extents(tree: Any) -> Dict[ShardPath, int]:
    """How long each shard in this tree is, along the dimension it is split on.

    Replicated tensors are left out: they have no extent to add up, and every
    rank holds the same bytes, so resharding one is a copy rather than a
    stitch. Excluding them here is what keeps them out of the plan entirely.

    Works on a saved tree and on a live one, which is the point — the reshard
    compares an old measurement against a new one, and neither side is allowed
    to be a guess about how torch chunks tensors.
    """
    from ravex._reshard import decode_placements, shard_dim

    extents: Dict[ShardPath, int] = {}
    for path, node in walk_shards(tree):
        placements = decode_placements(node.get("placements"))
        dim = shard_dim(placements, path_name(path))
        if dim is None:
            continue
        local = node.get("local")
        shape = getattr(local, "shape", None)
        if shape is None or dim >= len(shape):
            raise ValueError(
                "%s: placements say dimension %d, the stored shard has %d"
                % (path_name(path), dim, len(shape) if shape is not None else 0)
            )
        extents[path] = int(shape[dim])
    return extents


def take_shard_slices(
    tree: Any, wanted: Dict[ShardPath, List[Tuple[int, int]]]
) -> Dict[ShardPath, List[Any]]:
    """Cut the requested intervals out of one old rank's tree and keep only those.

    Called once per old store, with the store's whole snapshot in hand and the
    intention of not keeping it. ``narrow`` is a view, so the pieces are cloned
    — a view would pin the entire tensor it was taken from, and pinning the
    whole old checkpoint is the thing this function exists to avoid.
    """
    from ravex._reshard import decode_placements, shard_dim

    taken: Dict[ShardPath, List[Any]] = {}
    for path, intervals in wanted.items():
        node = node_at(tree, path)
        if not isinstance(node, dict) or not node.get(_SHARD_TAG):
            raise ValueError(
                "%s: this store has no shard where the others do" % path_name(path)
            )
        dim = shard_dim(decode_placements(node.get("placements")), path_name(path))
        if dim is None:  # pragma: no cover - `wanted` only holds sharded paths
            continue
        local = node["local"]
        taken[path] = [
            local.narrow(dim, start, stop - start).clone()
            for start, stop in intervals
        ]
    return taken


def build_resharded_tree(
    base: Any,
    live: Any,
    pieces: Dict[ShardPath, List[Any]],
) -> Any:
    """One old tree's structure, with every shard replaced by this rank's own.

    ``base`` supplies the shape of the result and every non-shard leaf —
    optimizer hyperparameters, step counters, the keys themselves — which are
    the same on every rank and are taken from one of them rather than merged.
    ``live`` supplies the layout each new shard has to match. ``pieces`` holds
    the slices already cut from the old stores, in the order they concatenate.

    Non-shard leaves are passed through by reference, not copied: they are
    about to be handed straight to ``set_state_dict`` and nothing here mutates
    them.
    """
    import torch

    from ravex._reshard import decode_placements, shard_dim

    def rebuild(node: Any, path: ShardPath) -> Any:
        if isinstance(node, dict) and node.get(_SHARD_TAG):
            return rebuild_shard(node, path)
        if isinstance(node, dict):
            return {key: rebuild(value, path + (key,)) for key, value in node.items()}
        if isinstance(node, list):
            return [rebuild(value, path + (i,)) for i, value in enumerate(node)]
        if isinstance(node, tuple):
            return tuple(rebuild(value, path + (i,)) for i, value in enumerate(node))
        return node

    def rebuild_shard(node: Dict[str, Any], path: ShardPath) -> Dict[str, Any]:
        live_node = node_at(live, path)
        if not isinstance(live_node, dict) or not live_node.get(_SHARD_TAG):
            raise ValueError(
                "%s is a shard in the checkpoint and not in this run's model"
                % path_name(path)
            )

        placements = live_node.get("placements")
        dim = shard_dim(decode_placements(placements), path_name(path))

        if dim is None:
            # Replicated: every old rank wrote the same bytes, so this rank's
            # new shard is any one of them. `base` is one of them.
            local = node["local"]
        else:
            parts = pieces.get(path)
            if not parts:
                raise ValueError(
                    "%s: nothing was read for this tensor" % path_name(path)
                )
            local = parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)

        return {
            _SHARD_TAG: 1,
            "local": local,
            # From the live model, not from the checkpoint: this shard is being
            # rebuilt to fit the topology running now.
            "global_shape": list(live_node.get("global_shape", [])),
            "placements": placements,
            "mesh_shape": live_node.get("mesh_shape"),
        }

    return rebuild(base, ())


def local_sharded_state(
    model, optimizers, cpu_offload: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """This rank's own shard of a sharded model and its optimizers.

    **Collective.** Every rank must call this at the same point. Nothing is
    gathered across ranks here, but the state-dict machinery still exchanges
    metadata, and a rank that skips the call strands the ones that did not.

    Nothing is bounded by a single rank's memory, which is the whole point: the
    gather that this replaces moved the entire state — 16.5 GiB for a 1.48B
    model with Adam — through rank 0 before a byte reached storage.

    The price is that the result only means something at the same world size,
    with the same sharding. :func:`apply_local_sharded_state` checks that.

    ``cpu_offload`` decides *who* copies the shards off the device. Left on,
    torch does it, into pageable host memory, before Moonclip is handed
    anything — the same thing that makes the gather path immune to Moonclip's
    pinned staging (measured at 9.4x on the transfer, 7.3x on the stall).
    Turned off, the shards stay on the device and the staging does the copy.
    It defaults to on because that is the conservative shape, and because the
    claim in the sentence above is about the *gather*: whether it holds here is
    a question for a box with GPUs on it, not for this docstring.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    options = StateDictOptions(full_state_dict=False, cpu_offload=cpu_offload)
    model_state, optimizer_state = get_state_dict(model, list(optimizers), options=options)
    return _encode_shards(model_state), _encode_shards(optimizer_state)


def apply_local_sharded_state(model, optimizers, model_state, optimizer_state) -> None:
    """Put this rank's saved shard back. **Collective.**

    The live sharded state is read first, purely for its layout: each saved
    shard is wrapped in the mesh and placements the running model is using, so
    a checkpoint written by a differently-shaped run is rejected here rather
    than half-applied.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    live_model, live_optimizer = get_state_dict(model, list(optimizers), options=options)

    set_state_dict(
        model,
        list(optimizers),
        model_state_dict=_decode_shards(model_state, live_model),
        optim_state_dict=_decode_shards(optimizer_state, live_optimizer),
        options=options,
    )


def agree_on_step(local_step: int) -> int:
    """The newest step *every* rank has on disk. **Collective.**

    Per-rank checkpoints land independently — separate writers, separate
    manifests — so a kill can leave rank 3 holding step 8 and rank 5 only step
    6. Restoring those side by side would assemble a model out of two different
    moments in training, which is not a slightly stale model but a wrong one.
    The oldest of the newest is the last step every rank can actually produce.
    """
    dist = _dist()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return local_step

    steps = _all_gather_object(dist, int(local_step))
    return min(int(step) for step in steps)


def gather_visible_stores(visible):
    """What every rank can reach on its own machine. **Collective.**

    Takes and returns a mapping of rank to that store's owner record, or any
    container of rank numbers. One entry per rank, in rank order. Ranks on the
    same machine report the same thing — that is not waste, it is how the
    picture stays readable without anyone having to know which ranks are
    co-located.

    Payload is a handful of integers per rank, so this costs what
    :func:`agree_on_step` costs. It is deliberately *not* a way to move
    checkpoint data: `all_gather_object` would leave every rank holding
    `world_size` copies, which for real shards is tens of gigabytes per
    process. Moving bytes is point-to-point work, and it is not this.
    """
    dist = _dist()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return [visible]

    seen = _all_gather_object(dist, visible)
    return [entry if entry else {} for entry in seen]


def agree_on_run_id(provenance: int, candidate: str) -> str:
    """The run identity every rank will use. **Collective.**

    Not simply rank 0's answer. A rank that inherited an id from its own store
    is naming the history this job is continuing; a rank that had to invent one
    is naming nothing. So the best-sourced candidate wins, and ties go to the
    lowest rank for determinism — see the provenance constants in
    :mod:`ravex._identity`.

    The case that makes this worth the care: rank 0's machine was replaced, so
    it generates, while ranks 1..n read the id of the run they are resuming. If
    rank 0 won, a resumed run would rename itself on every restart that lost
    the first node, and the lineage would be unreadable exactly when it matters.
    """
    dist = _dist()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return candidate

    votes = _all_gather_object(dist, (int(provenance), str(candidate)))

    best = None
    for entry in votes:
        if not entry:
            continue
        try:
            rank_provenance, value = int(entry[0]), str(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if best is None or rank_provenance < best[0]:
            best = (rank_provenance, value)
    return best[1] if best else candidate


def storage_is_shared(path: str) -> bool:
    """Whether every machine in this job sees the same directory. **Collective.**

    Asked rather than deduced. A path on a local disk and a path on an NFS or
    Lustre mount are indistinguishable from the configuration — both are
    ``storage.type: local`` pointing at a directory that exists — and guessing
    wrong is expensive in both directions: a false alarm on a cluster that is
    fine, or silence on a job whose checkpoint is being split across disks.

    Every rank drops a uniquely named marker and then looks for everyone
    else's. The gather is the synchronisation: a rank's name only reaches the
    others after it has written the file, so by the time the list comes back
    every marker exists on the filesystem that will hold it.

    A directory that cannot be written to answers False. That is the safe
    reading — it makes the caller assume the checkpoint is split — and an
    unwritable checkpoint directory is a larger problem that the backend will
    report on its own.
    """
    import uuid

    dist = _dist()
    marker = ".ravex-shared-%s" % uuid.uuid4().hex[:12]

    try:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, marker), "w", encoding="utf-8") as handle:
            handle.write("probe")
    except OSError:
        return False

    try:
        if dist is None or not dist.is_available() or not dist.is_initialized():
            # One process sees whatever it wrote, which is the honest answer
            # for a job that is not distributed at all.
            return True

        names = _all_gather_object(dist, marker)
        shared = all(
            isinstance(name, str) and os.path.exists(os.path.join(path, name))
            for name in names
        )

        # Nobody removes a marker another rank is still looking for.
        barrier()
        return shared
    finally:
        try:
            os.remove(os.path.join(path, marker))
        except OSError:
            pass


def gather_objects(value):
    """One small picklable value from every rank, in rank order. **Collective.**

    For facts the ranks must agree on before pairing up point-to-point sends:
    who lost a store, who holds a whole copy of whose. Both ends deciding from
    the same list is what keeps a send from being posted with no receive
    waiting for it.
    """
    dist = _dist()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return [value]

    return _all_gather_object(dist, value)


def all_ranks_agree(ok: bool) -> bool:
    """Whether *every* rank reports success. **Collective.**

    Used to keep a resume all-or-nothing. One rank quietly starting from
    scratch while the others restore is not a degraded resume: the ones that
    restored go on to a collective the odd one out will never join, and the job
    stops making progress without failing.
    """
    dist = _dist()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return bool(ok)

    flags = _all_gather_object(dist, bool(ok))
    return all(bool(flag) for flag in flags)


def apply_sharded_state(model, optimizers, model_state, optimizer_state) -> None:
    """Scatter a full state dict back onto a sharded model. **Collective.**

    Every rank passes the full state it read from storage, so no broadcast is
    needed — the ranks are reading identical bytes from the same checkpoint.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_state_dict(
        model,
        list(optimizers),
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=options,
    )


def unwrap_model(model):
    """Strip DDP / FSDP / compile wrappers to reach the user's module.

    Checkpointing the wrapper would bake ``module.`` prefixes into every key,
    which then fail to load in a single-GPU rerun.
    """
    seen = 0
    while seen < 8:
        inner = getattr(model, "module", None)
        if inner is None or inner is model:
            break
        # torch.compile stores the original under _orig_mod
        model = inner
        seen += 1

    orig = getattr(model, "_orig_mod", None)
    if orig is not None and orig is not model:
        model = orig
    return model
