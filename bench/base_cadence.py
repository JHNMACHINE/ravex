"""What does a full snapshot more often cost? (GPU-241)

With a delta at every step (GPU-209) every delta in the window is written
against one full, the base. A base that will not read takes the whole window
with it, and the next good point is whatever full retention still holds. This
measures the price of the obvious remedy - a full more often, so that a broken
base costs fewer steps - in the two currencies it is paid in: bytes per step
(disk, and the bucket's upload) and time per step on the training loop.

A real training loop through ``ravex._activate``, not synthetic saves: a delta
is only as small as the change one optimizer step makes, and Adam moves every
weight a little at every step. The window is the whole run
(``checkpoint_every = steps``), so retention deletes nothing and every snapshot
is still listed at the end with its size.

    python bench/base_cadence.py --width 2048 --layers 5 --steps 120

CPU only, so the time per step is the host side of a save - the copy and the
compression - which is the part a full makes more expensive; a GPU adds the
device-to-host copy, the same for a full and a delta.
"""

import argparse
import os
import shutil
import statistics
import tempfile
import time

import moonclip
import torch
import torch.nn as nn

import ravex


def model_of(width: int, layers: int) -> nn.Module:
    torch.manual_seed(0)
    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(width, width), nn.GELU()]
    return nn.Sequential(*blocks)


def run(full_every: int, width: int, layers: int, steps: int, root: str):
    path = os.path.join(root, "full_every_%d" % full_every)
    os.environ["RAVEX_STORAGE_PATH"] = path
    ravex._activate(
        backend="moonclip",
        checkpoint_every=steps,
        full_every=full_every,
        max_steps=steps,
        resume=False,
    )
    model = model_of(width, layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = torch.randn(16, width)

    times = []
    for _ in range(steps):
        started = time.perf_counter()
        loss = model(batch).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        times.append(time.perf_counter() - started)
    ravex.deactivate()

    # Asked once the run has closed the store, through the API.
    snapshots = moonclip.MoonclipManager(storage_root=path).list_snapshots()

    fulls = [s["total_compressed"] for s in snapshots if not s["is_delta"]]
    deltas = [s["total_compressed"] for s in snapshots if s["is_delta"]]
    return {
        "full_every": full_every,
        "fulls": len(fulls),
        "full_mib": statistics.mean(fulls) / 2**20 if fulls else 0.0,
        "delta_mib": statistics.mean(deltas) / 2**20 if deltas else 0.0,
        "bytes_per_step_mib": (sum(fulls) + sum(deltas)) / max(len(snapshots), 1) / 2**20,
        # The first steps hold the first full and the warm-up of everything.
        "step_ms": statistics.median(times[5:]) * 1e3,
        "mean_step_ms": statistics.mean(times[5:]) * 1e3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--full-every", type=int, nargs="+", default=[1000, 100, 40, 20])
    args = parser.parse_args()

    params = sum(p.numel() for p in model_of(args.width, args.layers).parameters())
    print("model: %.1fM parameters, Adam, fp32; %d steps" % (params / 1e6, args.steps))
    root = tempfile.mkdtemp(prefix="base-cadence-")
    try:
        print("full_every  fulls  full MiB  delta MiB  MiB/step  median ms/step  mean ms/step")
        for full_every in args.full_every:
            r = run(full_every, args.width, args.layers, args.steps, root)
            print(
                "%10d  %5d  %8.1f  %9.2f  %8.2f  %14.1f  %12.1f"
                % (r["full_every"], r["fulls"], r["full_mib"], r["delta_mib"],
                   r["bytes_per_step_mib"], r["step_ms"], r["mean_step_ms"])
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
