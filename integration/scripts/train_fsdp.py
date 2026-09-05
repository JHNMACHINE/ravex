"""An FSDP training script that knows nothing about Ravex.

    torchrun --nproc_per_node=2 train_fsdp.py --trace trace.jsonl

Uses FSDP2 (``fully_shard``) rather than the ``FullyShardedDataParallel``
wrapper, because FSDP1 refuses to initialise without an accelerator — "FSDP
needs a non-CPU accelerator device" — and there is no GPU in this environment.

That is a limitation of the test, not of the coverage: Ravex sees FSDP2's
parameters as DTensors and FSDP1's through the wrapper, but from there both go
down the same path — ``torch.distributed.checkpoint.state_dict`` gathers either
one into a full state dict.

What matters here is the sharding: each rank holds a slice of every parameter
and of the optimizer moments that go with it, so an ordinary ``state_dict()``
would save a fragment.
"""

import argparse
import json
import os
import signal

import ravex
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
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
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))

    torch.manual_seed(SEED)

    model = nn.Sequential(
        nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1)
    )
    # Shard the leaves first, then the root, as FSDP2 expects.
    for layer in model:
        if any(True for _ in layer.parameters()):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    dataset = TensorDataset(features, targets)
    sampler = DistributedSampler(dataset, shuffle=True, seed=SEED)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)

    model.train()
    trace = open(args.trace, "a", buffering=1) if rank == 0 else None

    local_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
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
                os.kill(os.getpid(), signal.SIGKILL)

    if trace is not None:
        trace.close()
    dist.destroy_process_group()
    print(f"rank {rank} done after {local_step} steps")


if __name__ == "__main__":
    main()
