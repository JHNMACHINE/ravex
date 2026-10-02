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
    # What a real job's checkpoint has and a small fp32 one does not (GPU-145).
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="mixed precision: bf16 weights in the module, fp32 masters in the optimizer",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        choices=[1, 2],
        help="2: weights with decay and biases without, as Hugging Face's Trainer splits them",
    )
    parser.add_argument(
        "--sub-group-size",
        type=int,
        default=None,
        help="stage 3: cut each rank's flat partition into sub-groups of this many elements, "
        "as DeepSpeed does past its default of 1e9 on a large model",
    )
    args = parser.parse_args()

    import deepspeed

    # NCCL where there is a GPU: the backend a real job's partitions were
    # made under, and the only one DeepSpeed's CUDA path takes.
    deepspeed.init_distributed(dist_backend="nccl" if torch.cuda.is_available() else "gloo")

    torch.manual_seed(0)
    model = nn.Sequential(
        *[
            layer
            for _ in range(args.layers)
            for layer in (nn.Linear(args.hidden, args.hidden), nn.ReLU())
        ]
    )

    zero = {"stage": args.stage}
    if args.sub_group_size:
        zero["sub_group_size"] = args.sub_group_size
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
        "zero_optimization": zero,
        # fp32 unless asked: the first point here was the *layout* of a ZeRO
        # checkpoint, and a CPU box has no bf16 story worth trusting.
        "fp16": {"enabled": False},
        "bf16": {"enabled": args.bf16},
        "wall_clock_breakdown": False,
    }

    if args.groups == 2:
        parameters = [
            {"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": 0.01},
            {"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0.0},
        ]
    else:
        parameters = model.parameters()
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=parameters, config=config)

    def batch():
        held = next(engine.module.parameters())
        return torch.randn(4, args.hidden).to(engine.device, dtype=torch.bfloat16 if args.bf16 else held.dtype)

    for _ in range(args.steps):
        loss = engine(batch()).float().sum()
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
        loss = engine(batch()).float().sum()
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
