"""A training script with no idea Ravex exists.

Deliberately contains zero references to ravex: no import, no activate, no
callback. Whatever happens to it happens because the autoloader is installed
and a ravex.yaml sits in the working directory.

Every step appends one JSON line to --trace, so a test can compare the loss
sequence of an interrupted-and-resumed run against an uninterrupted one.

--die-at N sends this process SIGKILL after step N: no atexit, no SIGTERM
handler, no final checkpoint. The worst case a preempted instance can produce.
"""

import argparse
import json
import os
import signal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLES = 64
BATCH = 8
SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--die-at", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(SEED)

    model = nn.Sequential(
        nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=BATCH,
        shuffle=True,
        num_workers=args.workers,
    )

    model.train()
    # Line buffered: a SIGKILL must not take the last few steps with it.
    trace = open(args.trace, "a", buffering=1)

    local_step = 0
    for _ in range(args.epochs):
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            local_step += 1
            trace.write(
                json.dumps(
                    {
                        "loss": repr(loss.item()),
                        "lr": repr(optimizer.param_groups[0]["lr"]),
                    }
                )
                + "\n"
            )

            if args.die_at and local_step >= args.die_at:
                trace.flush()
                os.fsync(trace.fileno())
                os.kill(os.getpid(), signal.SIGKILL)

    trace.close()
    print(f"done after {local_step} steps")


if __name__ == "__main__":
    main()
