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

import os
from typing import Any, Dict, List, Tuple


def _dist():
    try:
        import torch.distributed as dist
    except Exception:
        return None
    return dist


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
    error from inside ``set_state_dict``.
    """
    DTensor = _dtensor_class()
    if DTensor is not None and isinstance(value, DTensor):
        return {
            _SHARD_TAG: 1,
            "local": value.to_local().detach(),
            "global_shape": list(value.shape),
            "placements": [str(p) for p in value.placements],
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
    # for the last rank of a dimension that does not divide evenly.
    try:
        return DTensor.from_local(
            local,
            live.device_mesh,
            live.placements,
            run_check=False,
            shape=live.shape,
            stride=live.stride(),
        )
    except TypeError:  # older signature
        return DTensor.from_local(local, live.device_mesh, live.placements)


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

    # Pre-sized because `all_gather_object` fills the list in place.
    steps: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(steps, int(local_step))
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

    seen: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(seen, visible)
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

    votes: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(votes, (int(provenance), str(candidate)))

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
    else's. ``all_gather_object`` is the synchronisation: a rank's name only
    reaches the others after it has written the file, so by the time the list
    comes back every marker exists on the filesystem that will hold it.

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

        names: List[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(names, marker)
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

    flags: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(flags, bool(ok))
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
