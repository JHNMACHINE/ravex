"""Produce a genuine DeepSpeed ZeRO checkpoint, on CPU — groundwork for GPU-90.

GPU-90 wants to convert between framework checkpoint formats. Writing that
against a format read about rather than held is how you get an adapter that
parses a plausible file nobody ever wrote, so the first question is where real
artefacts come from. This answers it for DeepSpeed: they can be produced here,
on this machine, with no GPU — ``deepspeed`` selects its CPU accelerator on its
own when it finds no device, and ZeRO's partitioning is Python.

Run under torchrun with at least two ranks, or ZeRO has nothing to partition
across and the shards it writes are not the shape a real job produces::

    torchrun --nproc-per-node=2 make_zero.py --stage 1 --out /tmp/zero1

What comes out is the input to an adapter and, just as usefully, its oracle:
``deepspeed.utils.zero_to_fp32`` reconstructs the unsharded state from those
same files, so any conversion Ravex learns to do can be checked against
DeepSpeed's own answer rather than against an assumption.
"""

import argparse
import json
import os

import torch
import torch.nn as nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--out", required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--pair",
        action="store_true",
        help="also save one step later, for the Adam-consistency check",
    )
    args = parser.parse_args()

    import deepspeed

    deepspeed.init_distributed(dist_backend="gloo")

    torch.manual_seed(0)
    model = nn.Sequential(
        *[
            layer
            for _ in range(args.layers)
            for layer in (nn.Linear(args.hidden, args.hidden), nn.ReLU())
        ]
    )

    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
        "zero_optimization": {"stage": args.stage},
        # fp32 throughout: the point here is the *layout* of a ZeRO
        # checkpoint, and a CPU box has no bf16 story worth trusting.
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "wall_clock_breakdown": False,
    }

    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=config
    )

    for _ in range(args.steps):
        loss = engine(torch.randn(4, args.hidden)).sum()
        engine.backward(loss)
        engine.step()

    engine.save_checkpoint(args.out, tag="step3")

    # A second checkpoint, one step later. Not a spare: `zero_to_fp32`
    # reconstructs parameters and nothing else, so there is no oracle for the
    # optimizer moments — and two consecutive checkpoints *are* one, because
    # Adam's step is a deterministic function of the parameters and moments it
    # is given. `check_zero_moments.py` predicts the later parameters from the
    # earlier ones and checks. A moment reassembled onto the wrong parameter
    # has the right shape, plausible values, and fails that.
    if args.pair:
        loss = engine(torch.randn(4, args.hidden)).sum()
        engine.backward(loss)
        engine.step()
        engine.save_checkpoint(args.out, tag="step4")

    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(describe(args.out), indent=2, sort_keys=True))


def describe(root: str) -> dict:
    """Every file the checkpoint is made of, with its size. The format, as data."""
    found = {}
    for base, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(base, name)
            found[os.path.relpath(path, root).replace(os.sep, "/")] = os.path.getsize(path)
    return found


if __name__ == "__main__":
    main()
