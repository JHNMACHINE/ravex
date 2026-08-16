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

What it reports, per rank:

* device memory after sharding — evidence the shards are actually split
* the wall time of collecting the state, and the peak host RSS it costs
* how much of what was collected is still on the device, which is the
  condition for Moonclip's pinned staging to be worth anything

`--measure both` runs the comparison the per-rank work exists for:

    torchrun --nproc_per_node=8 train_fsdp_cuda.py --measure both \\
        --trace measure.json --ravex-path /root/ravex

    gather                  whole state through rank 0, cpu_offload
    per_rank offload=on     this rank's shard, torch copies it to host
    per_rank offload=off    this rank's shard, left on the device

The third line is the one with an open question attached. With cpu_offload the
device-to-host copy happens inside torch, into pageable memory, before Moonclip
sees anything — which is why the gather path cannot benefit from the pinned
staging (9.4x on the transfer, 7.3x on the single-GPU stall). Without it the
shards arrive on the device and the staging does the copy. Whether that holds
in practice is what the box is for.
"""

import argparse
import json
import os
import signal
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
    parser.add_argument("--die-at", type=int, default=0)
    parser.add_argument("--measure-gather", action="store_true")
    parser.add_argument(
        "--measure",
        choices=["none", "gather", "per_rank", "both"],
        default=None,
        help="which collection path to time; 'both' is the comparison",
    )
    parser.add_argument("--ravex-path", default="/root/ravex")
    args = parser.parse_args()

    if args.measure is None:
        args.measure = "gather" if args.measure_gather else "none"

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

    # Ravex resumes transparently before the first step, so anything it
    # restored is already in place here. `started` therefore measures the
    # training, not the restore; the restore shows up in ravex.log.
    trace = None
    if args.trace and rank == 0:
        trace = open(args.trace, "a", buffering=1)

    started_training = time.perf_counter()
    for step in range(args.steps):
        # The input is a function of the step, not of the RNG stream, so a
        # resumed run sees exactly the batches the original would have. RNG
        # restoration is covered by integration/test_cuda.py; conflating the
        # two here would make a failure ambiguous.
        generator = torch.Generator(device="cuda").manual_seed(1000 + step)
        x = torch.randn(
            args.batch, args.hidden, device=device, generator=generator
        )
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if trace is not None:
            trace.write(json.dumps({"step": step, "loss": repr(loss.item())}) + "\n")

        if args.die_at and step + 1 >= args.die_at:
            if trace is not None:
                trace.flush()
                os.fsync(trace.fileno())
            dist.barrier()
            os.kill(os.getpid(), signal.SIGKILL)

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"training: {args.steps} steps in "
              f"{(time.perf_counter() - started_training):.1f} s")
    if trace is not None:
        trace.close()

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
    if args.measure == "none":
        dist.barrier()
        dist.destroy_process_group()
        print(f"rank {rank} ok")
        return

    import sys
    sys.path.insert(0, args.ravex_path)
    from ravex._distributed import gather_sharded_state, local_sharded_state

    def tree_bytes(tree):
        """Bytes a collected state tree holds, shard records included."""
        if torch.is_tensor(tree):
            return tree.numel() * tree.element_size()
        if isinstance(tree, dict):
            return sum(tree_bytes(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(tree_bytes(v) for v in tree)
        return 0

    def tensors_on_device(tree):
        """How much of what was collected is still on the GPU.

        The number the pinned staging turns on. Anything already copied to host
        by torch is a copy Moonclip did not get to make, at 1.5 GB/s through
        pageable memory instead of 14.1 GB/s through pinned buffers.
        """
        if torch.is_tensor(tree):
            return (1, 1 if tree.is_cuda else 0)
        children = ()
        if isinstance(tree, dict):
            children = tree.values()
        elif isinstance(tree, (list, tuple)):
            children = tree
        total = cuda = 0
        for child in children:
            child_total, child_cuda = tensors_on_device(child)
            total += child_total
            cuda += child_cuda
        return (total, cuda)

    def timed(label, collect):
        """Run one collection path on every rank and report it from rank 0.

        Timed after a barrier and reported per rank: under `per_rank` there is
        no single rank doing the work, so a number taken only on rank 0 would
        be describing whichever rank happened to be fastest.
        """
        dist.barrier()
        torch.cuda.synchronize()
        before_rss = peak_rss_bytes()
        started = time.perf_counter()
        model_state, optimizer_state = collect()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started

        held = tree_bytes(model_state) + tree_bytes(optimizer_state)
        total, on_cuda = tensors_on_device(model_state)
        rss = peak_rss_bytes()

        # Every rank prints: the whole question is whether one rank is carrying
        # the run, and only rank 0's line cannot answer it.
        print(f"[{label}] rank {rank}: {seconds * 1000:.0f} ms, "
              f"holds {human_bytes(held)}, "
              f"{on_cuda}/{total} model tensors still on device, "
              f"peak RSS {human_bytes(rss)} (was {human_bytes(before_rss)})")

        del model_state, optimizer_state
        dist.barrier()
        return {
            "seconds": seconds,
            "held_bytes": held,
            "model_tensors": total,
            "model_tensors_on_cuda": on_cuda,
            "peak_rss_bytes": rss,
        }

    results = {}
    if args.measure in ("gather", "both"):
        results["gather"] = timed(
            "gather", lambda: gather_sharded_state(model, [optimizer])
        )
    if args.measure in ("per_rank", "both"):
        # Both offload settings, because the difference between them is the
        # open question: with cpu_offload torch copies the shards to pageable
        # host memory itself, and Moonclip's pinned staging never sees a device
        # tensor. Without it, the staging does the copy.
        results["per_rank_cpu_offload"] = timed(
            "per_rank offload=on",
            lambda: local_sharded_state(model, [optimizer], cpu_offload=True),
        )
        results["per_rank_on_device"] = timed(
            "per_rank offload=off",
            lambda: local_sharded_state(model, [optimizer], cpu_offload=False),
        )

    if rank == 0:
        print()
        for label, result in results.items():
            print(f"{label:24s} {result['seconds'] * 1000:8.0f} ms  "
                  f"{human_bytes(result['held_bytes']):>10s}  "
                  f"peak RSS {human_bytes(result['peak_rss_bytes']):>10s}  "
                  f"{result['model_tensors_on_cuda']}/{result['model_tensors']} "
                  f"on device")

        if args.trace:
            with open(args.trace, "w") as fh:
                json.dump(
                    {
                        "api": args.api,
                        "params": dense_params,
                        "world_size": world_size,
                        "results": results,
                    },
                    fh,
                    indent=2,
                )

    dist.barrier()
    dist.destroy_process_group()
    print(f"rank {rank} ok")


if __name__ == "__main__":
    main()
