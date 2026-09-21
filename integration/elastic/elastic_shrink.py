"""One FSDP2 rank of an elastic job, for probing GPU-94.

Deliberately does nothing about elasticity itself. The whole point of the
experiment is whether ``torchrun``'s own elastic agent plus Ravex's ordinary
resume already deliver what GPU-94 asks for, so this script must not help:
it builds a model, steps an optimizer, and writes down what world size it
found itself in. Everything else is the launcher's and Ravex's job.
"""

import datetime
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

import ravex

STEPS = int(os.environ.get("STEPS", "40"))
TRACE = os.environ.get("TRACE", "trace.jsonl")

#: How long a collective waits for a rank that is not coming back. The default
#: is **1800 seconds**, and that default is the whole reason this knob is here:
#: when a peer disappears, the surviving rank does not fail, it blocks - and
#: `torchrun`'s elastic agent restarts a worker group on *failure*, so until
#: this expires there is nothing for the agent to react to. A cluster that
#: loses a node looks perfectly healthy for half an hour.
TIMEOUT = int(os.environ.get("GLOO_TIMEOUT", "1800"))


# The explicit entry since ravex 0.1.0: `ravex.activate()` went with the
# autoloader, and this line was still calling it (caught by
# tests/test_integration_kit_names.py, GPU-130).
@ravex.train_loop()
def main() -> None:
    dist.init_process_group(
        "gloo", timeout=datetime.timedelta(seconds=TIMEOUT)
    )
    rank, world = dist.get_rank(), dist.get_world_size()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
    fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for step in range(STEPS):
        model(torch.randn(8, 64)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            with open(TRACE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "step": step, "world": world, "pid": os.getpid(),
                    "at": time.time(),
                }) + "\n")
        time.sleep(0.15)

    ravex.flush()
    dist.destroy_process_group()
    print("rank %d of %d finished" % (rank, world), file=sys.stderr)


if __name__ == "__main__":
    main()
