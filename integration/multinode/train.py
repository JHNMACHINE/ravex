"""A sharded training script that knows nothing about Ravex.

One rank per container, so the process group is formed from the environment
rather than by ``torchrun``:

    RANK=3 WORLD_SIZE=6 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR=node0 MASTER_PORT=29500 python train.py --trace /out/rank3.jsonl

``LOCAL_WORLD_SIZE=1`` with ``WORLD_SIZE=6`` is what makes Ravex count six
machines. It is read from the environment and not derived, because the process
group knows how many ranks there are and nothing about which of them share a
filesystem — which is the whole subject of this bench.

FSDP2 (``fully_shard``) rather than the FSDP1 wrapper: FSDP1 refuses to
initialise without an accelerator, and there is no GPU here. What matters is
that the state is genuinely sharded, so ``per_rank`` has something to write.
"""

import argparse
import json
import os
import signal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

SAMPLES = 96
BATCH = 8
SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--die-at", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))

    torch.manual_seed(SEED)

    model = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 1))
    for layer in model:
        if any(True for _ in layer.parameters()):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    dataset = TensorDataset(features, targets)
    sampler = DistributedSampler(dataset, shuffle=True, seed=SEED)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)

    model.train()

    # Every rank writes its own trace. Rank 0 alone would hide exactly what
    # this bench is about: the ranks disagreeing about what they hold.
    trace = open(args.trace, "a", buffering=1)

    local_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            local_step += 1
            trace.write(
                json.dumps({"rank": rank, "step": local_step, "loss": repr(loss.item())})
                + "\n"
            )

            if args.die_at and local_step >= args.die_at:
                trace.flush()
                os.fsync(trace.fileno())
                # SIGKILL, not an exception: a preempted box does not unwind.
                os.kill(os.getpid(), signal.SIGKILL)

    trace.close()
    dist.destroy_process_group()
    print("rank %d done after %d steps" % (rank, local_step))


if __name__ == "__main__":
    main()
