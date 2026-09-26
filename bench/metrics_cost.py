"""What ``log_metrics`` costs a training step on a GPU (GPU-163).

``log_metrics`` is built not to synchronise the CUDA stream from the training
thread: a tensor is detached and reduced on the device (``histc`` for a
histogram), and a writer thread of its own brings the result home. This is the
measurement of whether that holds, on a real step:

``off``
    ``metrics: false`` - no metrics at all, the floor.
``scalars``
    the loss, the gradient norm and the learning rate every step, as device
    tensors, plus Ravex's automatic step and system metrics.
``+hist``
    the same, plus a 64-bin histogram of every weight matrix every ten steps.
``+ship``
    the same again, with the run also sent to a backend while it trains
    (``metrics_endpoint``). Only when there is one to send to - under an agent
    that hands out ``RAVEX_METRICS_ENDPOINT``, for instance.

Every arm is the same model, data and optimizer from the same seed; they run
interleaved, ``--repeats`` times each, so drift in the machine's clock or
temperature lands on all of them alike. Checkpoints are off in every arm: they
have their own bench, and here they would only add noise. Time is taken with
``torch.cuda.synchronize()`` at both ends of the measured steps, never inside
them.

    python bench/metrics_cost.py
    python bench/metrics_cost.py --width 4096 --steps 1000 --repeats 5

**Measured**, 2026-09-26, on an NVIDIA L4 (RunPod, secure cloud) - see the
table printed by the run and the numbers in the README.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time

import torch

import ravex

#: What an agent hands a job that would make every arm write to its store, its
#: bucket and its run. Each arm gets a store of its own on this disk instead.
JOB_ENV = (
    "RAVEX_RUN_ID", "RAVEX_NAME", "RAVEX_STORAGE_PATH",
    "RAVEX_STORAGE_TYPE", "RAVEX_STORAGE_BUCKET", "RAVEX_STORAGE_PREFIX",
    "RAVEX_STORAGE_ENDPOINT", "RAVEX_STORAGE_REGION", "RAVEX_STORAGE_PATH_STYLE",
)
SHIP_ENV = ("RAVEX_METRICS_ENDPOINT", "RAVEX_METRICS_TOKEN")


def model_for(width: int, depth: int, device) -> torch.nn.Module:
    layers = []
    for _ in range(depth):
        layers += [torch.nn.Linear(width, width), torch.nn.GELU()]
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers).to(device)


def one_arm(arm: str, args, device, root: str, rep: int, shipping: dict) -> float:
    """Milliseconds per step for one arm, over ``args.steps`` measured steps."""
    for name in JOB_ENV + SHIP_ENV:
        os.environ.pop(name, None)
    os.environ["RAVEX_STORAGE_PATH"] = os.path.join(root, "%s-%d" % (arm, rep))
    if arm == "+ship":
        os.environ.update(shipping)
        os.environ["RAVEX_NAME"] = "metrics-cost-ship-%d" % rep

    histograms = arm in ("+hist", "+ship")
    result = {}

    @ravex.train_loop(
        checkpoint_every=10**9,
        checkpoint_on_exit=False,
        metrics=(arm != "off"),
        backend="torch_save",
    )
    def train():
        torch.manual_seed(0)
        model = model_for(args.width, args.depth, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(args.batch, args.width, device=device)
        y = torch.randn(args.batch, 1, device=device)
        matrices = [(n, p) for n, p in model.named_parameters() if p.ndim == 2]

        started = None
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
            ravex.batch_boundary()
            loss = torch.nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if arm != "off":
                values = {"train/loss": loss, "train/grad_norm": norm,
                          "train/lr": optimizer.param_groups[0]["lr"]}
                if histograms and step % 10 == 0:
                    for name, weight in matrices:
                        values["weights/" + name] = weight
                ravex.log_metrics(values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        result["ms"] = (time.perf_counter() - started) * 1000.0 / args.steps

    train()
    return result["ms"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    card = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    shipping = {k: os.environ[k] for k in SHIP_ENV if os.environ.get(k)}
    arms = ["off", "scalars", "+hist"] + (["+ship"] if "RAVEX_METRICS_ENDPOINT" in shipping else [])
    params = sum(p.numel() for p in model_for(args.width, args.depth, "cpu").parameters())
    print("%s, %.1fM parameters, batch %d, %d measured steps per arm, %d repeats; arms %s"
          % (card, params / 1e6, args.batch, args.steps, args.repeats, ", ".join(arms)), flush=True)

    root = tempfile.mkdtemp(prefix="metrics-cost-")
    times = {arm: [] for arm in arms}
    for rep in range(args.repeats):
        for arm in arms:
            ms = one_arm(arm, args, device, root, rep, shipping)
            times[arm].append(ms)
            print("repeat %d  %-8s %.3f ms/step" % (rep, arm, ms), flush=True)

    floor = statistics.median(times["off"])
    print()
    print("%-8s %10s %10s %9s" % ("arm", "ms/step", "spread", "vs off"))
    rows = []
    for arm in arms:
        median = statistics.median(times[arm])
        spread = max(times[arm]) - min(times[arm])
        overhead = (median - floor) / floor * 100.0
        rows.append({"arm": arm, "ms_per_step": median, "spread_ms": spread,
                     "overhead_percent": overhead, "all": times[arm]})
        print("%-8s %10.3f %10.3f %+8.1f%%" % (arm, median, spread, overhead))
    print(json.dumps({"card": card, "parameters": params, "batch": args.batch, "rows": rows}))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
