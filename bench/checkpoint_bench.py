"""What Ravex costs, and what Moonclip buys.

Three numbers, because they answer different questions:

**Hook overhead** - the price of being watched at all, with no checkpoint ever
taken. Patched `nn.Module.__init__`, a post-step hook per optimizer, and a
wrapper around the dataloader iterator. If this is not ~free the whole approach
is wrong.

**Checkpoint stall** - how long a single training step takes when it is the one
that collects. Collection runs on the training thread on purpose: the state has
to be consistent with the step that just finished, and the writer must not read
tensors while the next step mutates them. So the loop pays for a copy of the
model, and this is that copy, measured.

**Backend throughput** - Moonclip against `torch.save` on the same state: time
to hand off, time to durability, and bytes on disk.

    python bench/checkpoint_bench.py --params 1e9

Run it on the machine you care about; a checkpoint benchmark on a laptop tells
you about the laptop's disk.
"""

import argparse
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn


def build_model(target_params: float, device):
    """A stack of square Linears sized to hit a parameter count.

    Deliberately boring: what a checkpoint benchmark measures is the size and
    shape of the state, not how cleverly it was produced.
    """
    hidden = 4096
    per_layer = hidden * hidden + hidden
    layers = max(2, round(target_params / per_layer))

    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(hidden, hidden), nn.GELU()]
    model = nn.Sequential(*blocks).to(device)

    actual = sum(p.numel() for p in model.parameters())
    return model, actual, hidden


def human(count):
    for unit in ("", "K", "M", "G", "T"):
        if abs(count) < 1000:
            return f"{count:.1f}{unit}"
        count /= 1000
    return f"{count:.1f}P"


def human_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def directory_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def train_steps(model, optimizer, hidden, device, steps, batch=4):
    """Run `steps` optimizer steps, returning per-step wall times."""
    timings = []
    for _ in range(steps):
        x = torch.randn(batch, hidden, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    return timings


def measure_backends(model, optimizer, hidden, device, root, repeats=4, between=5):
    """Compare the ways of writing the same checkpoint.

    Real training steps run between snapshots rather than random perturbation.
    That matters for the delta engine: gradient descent leaves most of a
    float's bits alone, while noise scrambles the mantissa and makes every
    tensor incompressible - which would measure nothing except how well zstd
    handles entropy.

    Three writers:

    - ``torch.save``      what the fallback backend does
    - ``moonclip/bytes``  flatten_state_dict to bytes, which is what the
                          Moonclip backend did before it passed tensors
    - ``moonclip/native`` CheckpointManager.save(model=...), which passes
                          tensors straight through to Rust — the same handoff
                          the backend takes today

    Each writer is timed only once every other writer is idle. Moonclip's
    background save is a full core-saturating zstd pass, and its `submit`
    blocks on the previous one, so timing a writer while another is still
    draining measures the drain. That is not a small effect: it is what made
    an earlier run of this benchmark report the tensor path as the slowest of
    the three when, measured alone, it is the fastest by 5x.
    """
    results = {}
    paths = {name: Path(root) / name for name in ("torch_save", "moon_bytes", "moon_native")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    try:
        import moonclip
    except ImportError:
        moonclip = None

    bytes_manager = native_manager = None
    if moonclip is not None:
        bytes_manager = moonclip.CheckpointManager(
            storage_root=str(paths["moon_bytes"]), world_size=1, rank=0, async_save=True
        )
        native_manager = moonclip.CheckpointManager(
            storage_root=str(paths["moon_native"]), world_size=1, rank=0, async_save=True
        )

    timings = {"torch.save": [], "moonclip/bytes": [], "moonclip/native": []}

    for index in range(repeats):
        train_steps(model, optimizer, hidden, device, between)
        state = model.state_dict()

        if moonclip is not None:
            bytes_manager.flush()
            native_manager.flush()
        started = time.perf_counter()
        snapshot = {k: v.detach().to("cpu", copy=True) for k, v in state.items()}
        torch.save(snapshot, paths["torch_save"] / f"step_{index}.pt")
        timings["torch.save"].append(time.perf_counter() - started)
        del snapshot

        if moonclip is None:
            continue

        bytes_manager.flush()
        native_manager.flush()
        started = time.perf_counter()
        tensors, _ = moonclip.flatten_state_dict(state, "model")
        bytes_manager.save_raw(step=index, tensors=tensors)
        timings["moonclip/bytes"].append(time.perf_counter() - started)
        del tensors

        bytes_manager.flush()
        native_manager.flush()
        started = time.perf_counter()
        native_manager.save(step=index, model=model)
        timings["moonclip/native"].append(time.perf_counter() - started)

    if moonclip is not None:
        bytes_manager.flush()
        native_manager.flush()

    for name, path in (
        ("torch.save", paths["torch_save"]),
        ("moonclip/bytes", paths["moon_bytes"]),
        ("moonclip/native", paths["moon_native"]),
    ):
        if not timings[name]:
            continue
        results[name] = {
            "median_s": statistics.median(timings[name]),
            "bytes": directory_size(path),
            "snapshots": repeats,
        }
    return results


def breakdown(model, optimizer, device):
    """Split the checkpoint stall into its parts.

    The stall is one number until you know which part of it is which. The
    designed fix - an asynchronous GPU-to-pinned-CPU copy on a side stream,
    which the plan calls a shadow copy - only helps if the device transfer
    dominates. If the CPU-side serialisation dominates instead, that fix buys
    nothing and the answer is somewhere else entirely.
    """
    import moonclip

    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}

    def total_bytes(tree):
        if torch.is_tensor(tree):
            return tree.numel() * tree.element_size()
        if isinstance(tree, dict):
            return sum(total_bytes(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(total_bytes(v) for v in tree)
        return 0

    size = total_bytes(state)
    print(f"\nstate to move: {human_bytes(size)} (model + optimizer)")

    # 1. device -> host, nothing else
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    on_cpu = {
        section: {k: v.detach().to("cpu", copy=True) for k, v in part.items()}
        for section, part in (("model", state["model"]),)
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
    transfer = time.perf_counter() - started

    # 2. flattening the state tree, which no longer converts anything: with
    #    as_tensors the tensors are handed over as they are, so this is dict
    #    building and should be microseconds. It is timed anyway, because the
    #    version of this that called .tobytes() cost 484 ms and looked just as
    #    much like bookkeeping from the outside.
    started = time.perf_counter()
    tensors, _ = moonclip.flatten_state_dict(on_cpu["model"], "model", as_tensors=True)
    serialise = time.perf_counter() - started

    # 3. handing those tensors to Rust — this is where the copy happens now
    root = tempfile.mkdtemp(prefix="ravex_breakdown_")
    try:
        manager = moonclip.CheckpointManager(
            storage_root=root, world_size=1, rank=0, async_save=True
        )
        started = time.perf_counter()
        manager.save_raw(step=0, tensors=tensors)
        handoff = time.perf_counter() - started
        started = time.perf_counter()
        manager.flush()
        write = time.perf_counter() - started
    finally:
        shutil.rmtree(root, ignore_errors=True)

    model_bytes = total_bytes(state["model"])
    print(f"  (measured on the model alone: {human_bytes(model_bytes)})")
    for label, seconds in (
        ("GPU -> CPU copy", transfer),
        ("flatten state tree", serialise),
        ("copy into Rust", handoff),
        ("background write", write),
    ):
        rate = model_bytes / seconds / 1e9 if seconds > 0 else float("inf")
        print(f"  {label:<24}{seconds * 1000:8.1f} ms   {rate:5.1f} GB/s")

    blocking = transfer + serialise + handoff
    print(f"  {'blocks training':<24}{blocking * 1000:8.1f} ms")
    print(
        f"\n  an async device copy could hide {transfer / blocking:.0%} of that; "
        f"the remaining {handoff / blocking:.0%} is the shadow copy, which is one "
        f"pass over memory and has nowhere much left to go"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=3e8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="split the checkpoint stall into transfer, serialisation and write",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("device: CPU (numbers will not mean much)")

    model, params, hidden = build_model(args.params, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"model:  {human(params)} parameters, {human_bytes(params * 4)} of fp32 weights")

    import ravex

    active = ravex.is_active()
    print(f"ravex:  {'active' if active else 'not active'}")

    # Warm up: allocator, cuDNN autotune, and Adam's state tensors.
    train_steps(model, optimizer, hidden, device, args.warmup, args.batch)

    timings = train_steps(model, optimizer, hidden, device, args.steps, args.batch)
    median = statistics.median(timings)
    slowest = max(timings)
    print(
        f"\nstep time: median {median * 1000:.1f} ms, "
        f"slowest {slowest * 1000:.1f} ms "
        f"({slowest / median:.1f}x median)"
    )
    if active:
        print("  the slowest step is the one that collected a checkpoint")

    if args.breakdown:
        breakdown(model, optimizer, device)
        return

    root = tempfile.mkdtemp(prefix="ravex_bench_")
    try:
        results = measure_backends(model, optimizer, hidden, device, root)
        print(f"\n{'writer':<18}{'handoff':>12}{'on disk':>14}{'vs torch.save':>26}")
        baseline = results["torch.save"]
        for name, data in results.items():
            comparison = ""
            if name != "torch.save":
                comparison = (
                    f"{baseline['median_s'] / data['median_s']:.1f}x faster, "
                    f"{baseline['bytes'] / max(data['bytes'], 1):.1f}x smaller"
                )
            print(
                f"{name:<18}{data['median_s'] * 1000:>9.1f} ms"
                f"{human_bytes(data['bytes']):>14}{comparison:>26}"
            )
        print(f"\n({baseline['snapshots']} snapshots, real training steps in between)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
