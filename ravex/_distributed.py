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


def is_distributed() -> bool:
    dist = _dist()
    if dist is not None and dist.is_available() and dist.is_initialized():
        return True
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


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
