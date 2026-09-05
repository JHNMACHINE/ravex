"""A DDP training script that knows nothing about Ravex.

Launched with:

    torchrun --nproc_per_node=2 train_ddp.py --trace trace.jsonl

Uses the gloo backend so it runs on CPU. Rank 0 writes the trace; both ranks
train, and both have to resume — only rank 0 writes checkpoints, but a rank
that came back with fresh weights would corrupt the all-reduce on its first
step.
"""

import argparse
import json
import os
import signal

import ravex
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

SAMPLES = 64
BATCH = 8
SEED = 0


@ravex.train_loop()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--die-at", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()

    torch.manual_seed(SEED)

    model = nn.Sequential(
        nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1)
    )
    ddp_model = DDP(model)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    dataset = TensorDataset(features, targets)
    sampler = DistributedSampler(dataset, shuffle=True, seed=SEED)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)

    ddp_model.train()
    trace = open(args.trace, "a", buffering=1) if rank == 0 else None

    local_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            loss = ((ddp_model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            local_step += 1
            if trace is not None:
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
                if trace is not None:
                    trace.flush()
                    os.fsync(trace.fileno())
                # Kill every rank, the way a dying node would.
                os.kill(os.getpid(), signal.SIGKILL)

    if trace is not None:
        trace.close()
    dist.destroy_process_group()
    print(f"rank {rank} done after {local_step} steps")


if __name__ == "__main__":
    main()
