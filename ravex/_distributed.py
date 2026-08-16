"""Distributed helpers.

Only rank 0 writes checkpoints. Every rank must still *load* on resume, so the
read path is not gated on rank.
"""

from __future__ import annotations

import os


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
