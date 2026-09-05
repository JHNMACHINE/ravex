"""A DeepSpeed training script. It has never heard of Ravex.

The same contract as every other script here: one decorator on ``main``, a
``ravex.yaml`` in the working directory does the rest. DeepSpeed is the case
that tests ``ravex/_frameworks.py``'s claim rather than illustrating it,
because its engine wraps the user's module in an ``nn.Module`` of its own and
hands the base optimizer flat partition buffers instead of the model's
parameters — so both of the registry's tests for "is this a model" come back
negative for reasons that have nothing to do with the model.

Run under torchrun: DeepSpeed's ``init_distributed`` falls back to MPI without
the launcher's environment variables, and mpi4py is not installed.
"""

import argparse
import json

import ravex
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@ravex.train_loop()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--trace", default="trace.jsonl")
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

    torch.manual_seed(7)
    data = TensorDataset(torch.randn(args.steps * 2, args.hidden))
    loader = DataLoader(data, batch_size=2)

    with open(args.trace, "a", encoding="utf-8") as trace:
        for step, (batch,) in enumerate(loader):
            loss = engine(batch).sum()
            engine.backward(loss)
            engine.step()
            trace.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")


if __name__ == "__main__":
    main()
