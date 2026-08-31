"""GPU-92 on two real machines: does a SIGTERM landing on ONE rank pull every
rank into a coordinated emergency checkpoint?

The unit tests for this run two gloo processes on one box, which cannot
reproduce the situation the mechanism exists for. On one machine every local
process gets SIGTERM at effectively the same instant from the same source -
which is exactly why ``_emergency_coordination_active`` turns the whole path
*off* there. The channel only ever runs when the job is sharded **and** spans
more than one machine, so a single box can only ever test the branch that
does nothing.

The contract being checked, straight from ``_check_emergency_signal``:

1. SIGTERM lands on one rank. Its handler only raises a flag - the process
   does not die from it.
2. At the next ``on_step``, every rank posts the detection collective on the
   isolated gloo subgroup. The OR-reduce means one rank's flag is enough.
3. Every rank - not just the signalled one - writes a checkpoint tagged
   ``emergency=true`` in its metadata.
4. Only the signalled rank then re-raises SIGTERM and dies. Everyone else
   goes back to ordinary training, which is the entire point of coordinating
   rather than each rank acting alone.

Point 4 is the one a single box cannot even express, and point 3 is the one
that matters for the checkpoint actually being resumable: a sharded save is
collective, so a rank saving alone would hang rather than write.

Run under torchrun with one rank per machine (WORLD_SIZE=2,
LOCAL_WORLD_SIZE=1), which is what makes ``spans_several_machines()`` true.
"""

import json
import os
import signal
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard


def log(msg):
    print("[rank %s] %s" % (os.environ.get("RANK", "?"), msg), flush=True)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    steps = int(os.environ.get("STEPS", "40"))
    die_at = int(os.environ.get("DIE_AT", "12"))
    hidden = int(os.environ.get("HIDDEN", "1024"))
    layers = int(os.environ.get("LAYERS", "4"))
    # The rank that gets preempted. Deliberately not rank 0: rank 0 is the
    # one that would have coped on its own under the old behaviour, so a test
    # that signalled rank 0 could pass without the channel doing anything.
    victim = int(os.environ.get("VICTIM", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    import ravex
    from ravex._distributed import spans_several_machines

    torch.manual_seed(0)
    blocks = []
    for _ in range(layers):
        blocks.append(nn.Linear(hidden, hidden))
        blocks.append(nn.ReLU())
    blocks.append(nn.Linear(hidden, 8))
    model = nn.Sequential(*blocks).to(device)
    fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    runtime = getattr(ravex, "_runtime_instance", None)
    log("ravex active=%s spans_several_machines=%s" % (
        runtime is not None, spans_several_machines()))

    reached = 0
    for step in range(1, steps + 1):
        model(torch.randn(8, hidden, device=device)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        reached = step

        if step == die_at and rank == victim:
            log("raising SIGTERM on myself at step %d (the preempted box)" % step)
            os.kill(os.getpid(), signal.SIGTERM)

        # Past the victim's death the survivors must keep going. Slowed a
        # little so "it kept training" is a real observation rather than a
        # race the loop won by finishing first.
        if step >= die_at:
            time.sleep(0.2)

    log("finished all %d steps" % reached)
    print("SURVIVED " + json.dumps({"rank": rank, "steps_reached": reached}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print("SURVIVED " + json.dumps({
            "rank": os.environ.get("RANK"), "error": str(exc)}), flush=True)
        sys.exit(1)
