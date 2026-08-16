"""FSDP on real GPUs, at a size where the sharding is not decorative.

    torchrun --nproc_per_node=8 train_fsdp_cuda.py --params 1.5e9 --api fsdp2

The existing `train_fsdp.py` runs FSDP2 on CPU with gloo over a model of about
ninety parameters. That covers the code path — Ravex sees DTensors and gathers
them — and nothing else. It cannot cover:

* **FSDP1 at all.** `FullyShardedDataParallel` refuses to initialise without an
  accelerator ("FSDP needs a non-CPU accelerator device"), so the wrapper path
  has never once been executed, on any machine.
* **NCCL.** The CPU test uses gloo; a real run gathers over NCCL.
* **Anything about size.** At ninety parameters nothing is learned about what
  the gather costs, or about the known limit that rank 0 must hold the entire
  unsharded state in host RAM.

This script exists for those three. It is deliberately shaped so that a single
card cannot hold the run: 1.5B parameters in fp32 is 6 GiB of weights, and Adam
brings the total past 24 GiB against a 16 GiB card. If it runs, FSDP is doing
the work.

What it reports, per rank and for the gather:

* device memory after sharding — evidence the shards are actually split
* the wall time of `gather_sharded_state`, which is the FSDP-specific cost and
  is *not* the one the pinned staging speeds up: that gather is done inside
  torch, with cpu_offload, before Moonclip is handed anything
* peak host RSS on rank 0 while the full state exists there
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def human_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def build_model(target_params, hidden):
    """A stack of square Linears sized to a parameter count.

    Boring on purpose: what is under test is the sharding and the gather, not
    the architecture.
    """
    per_layer = hidden * hidden + hidden
    layers = max(2, round(target_params / per_layer))
    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(hidden, hidden), nn.GELU()]
    return nn.Sequential(*blocks), layers


def shard_fsdp2(model, device, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cuda", (world_size,))
    # Leaves first, then the root, as FSDP2 expects.
    for layer in model:
        if any(True for _ in layer.parameters()):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model.to(device)


def shard_fsdp1(model, device, world_size):
    """The wrapper API. Never executed before this script existed.

    `auto_wrap_policy` is not optional at this size. Without it FSDP1 makes the
    whole model one flat unit, so every rank materialises all of it during the
    forward and the sharding saves nothing where it matters — a 1.5B model then
    OOMs on a 16 GiB card even across eight of them. Wrapping each Linear gives
    FSDP1 the same granularity `fully_shard` gets per layer in FSDP2, which is
    what makes the two comparable.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    return FSDP(
        model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
    )


def peak_rss_bytes():
    try:
        import resource

        # ru_maxrss is KiB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=1.5e9)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--api", choices=["fsdp1", "fsdp2"], default="fsdp2")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(0)
    model, layers = build_model(args.params, args.hidden)
    dense_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"model: {dense_params / 1e9:.2f}B parameters, {layers} layers, "
              f"hidden {args.hidden}")
        print(f"       {human_bytes(dense_params * 4)} of fp32 weights — "
              f"{human_bytes(dense_params * 16)} with Adam and gradients")
        print(f"api:   {args.api}, world_size {world_size}\n")

    shard = shard_fsdp2 if args.api == "fsdp2" else shard_fsdp1
    model = shard(model, device, world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(args.steps):
        x = torch.randn(args.batch, args.hidden, device=device)
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()

    def local_bytes(p):
        """Bytes this rank actually holds.

        `p.numel()` on a DTensor reports the *global* shape, so summing it
        gives the size of the whole model on every rank and makes a sharded
        run look unsharded. `to_local()` is the shard.
        """
        local = p.to_local() if hasattr(p, "to_local") else p
        return local.numel() * local.element_size()

    local_shard = sum(local_bytes(p) for p in model.parameters())
    # The dense model, not the sum of this rank's views: FSDP1 with
    # use_orig_params hands back local shapes, so summing those and comparing
    # them to themselves would report 100% on every rank however well sharded.
    global_size = dense_params * 4
    reserved = torch.cuda.memory_reserved(device)
    print(f"rank {rank}: holds {human_bytes(local_shard)} of "
          f"{human_bytes(global_size)} ({local_shard / global_size:.0%}), "
          f"{human_bytes(reserved)} reserved on device")

    # ── The FSDP-specific cost ──────────────────────────────────────
    import sys
    sys.path.insert(0, "/root/ravex")
    from ravex._distributed import gather_sharded_state

    dist.barrier()
    started = time.perf_counter()
    model_state, optimizer_state = gather_sharded_state(model, [optimizer])
    gather_seconds = time.perf_counter() - started

    if rank == 0:
        def tree_bytes(tree):
            if torch.is_tensor(tree):
                return tree.numel() * tree.element_size()
            if isinstance(tree, dict):
                return sum(tree_bytes(v) for v in tree.values())
            if isinstance(tree, (list, tuple)):
                return sum(tree_bytes(v) for v in tree)
            return 0

        model_bytes = tree_bytes(model_state)
        optim_bytes = tree_bytes(optimizer_state)
        gathered = model_bytes + optim_bytes
        on_cuda = sum(
            1 for v in model_state.values()
            if torch.is_tensor(v) and v.is_cuda
        )
        print(f"\ngather: {gather_seconds * 1000:.0f} ms for "
              f"{human_bytes(gathered)} "
              f"({human_bytes(model_bytes)} model + "
              f"{human_bytes(optim_bytes)} optimizer) "
              f"— {gathered / gather_seconds / 1e9:.2f} GB/s")
        print(f"        {len(model_state)} tensors, {on_cuda} still on the device")
        print(f"        peak host RSS on rank 0: {human_bytes(peak_rss_bytes())}")
        print("\n        cpu_offload=True means torch did the device-to-host "
              "copy itself,\n        which is why the pinned staging in Moonclip "
              "cannot help this path.")

        if args.trace:
            with open(args.trace, "w") as fh:
                json.dump(
                    {
                        "api": args.api,
                        "params": dense_params,
                        "world_size": world_size,
                        "gather_seconds": gather_seconds,
                        "gathered_bytes": gathered,
                        "tensors_on_cuda_after_gather": on_cuda,
                        "peak_rss_bytes": peak_rss_bytes(),
                    },
                    fh,
                )

    dist.barrier()
    dist.destroy_process_group()
    print(f"rank {rank} ok")


if __name__ == "__main__":
    main()
