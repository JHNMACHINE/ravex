"""A DeepSpeed training script, Ravex-attached by one decorator and no more.

The counterpart to ``train_plain.py``, and it exists to answer a question
``ravex/_frameworks.py`` asserts an answer to without ever having run it:
Ravex intercepts PyTorch itself, so anything built on top of PyTorch is
"already covered". DeepSpeed is the interesting test of that claim, because its
engine wraps the optimizer in something of its own and drives the step from
there.

Prints the step it reached and the first weight's checksum, so a second run can
say whether it continued or started over.
"""

import argparse

import ravex
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@ravex.train_loop()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    import deepspeed

    deepspeed.init_distributed(dist_backend="gloo")

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(args.hidden, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
    )

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.01}},
            "zero_optimization": {"stage": args.stage},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
        },
    )

    loader = DataLoader(
        TensorDataset(torch.randn(args.steps * 2, args.hidden)), batch_size=2
    )

    seen = 0
    for (batch,) in loader:
        loss = engine(batch).sum()
        engine.backward(loss)
        engine.step()
        seen += 1

    first = next(engine.module.parameters())
    print("STEPS_RUN %d" % seen)
    print("CHECKSUM %.6f" % float(first.detach().float().sum()))


if __name__ == "__main__":
    main()
