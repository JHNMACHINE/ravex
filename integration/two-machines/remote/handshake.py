"""Does this pair of boxes form a process group at all — on both transports.

NCCL over the overlay is what training needs. The gloo subgroup is what the
replication needs: since GPU-82 the store bytes travel as host tensors on a
separate gloo group, and NAT between the machines is exactly the thing that
group has never met.
"""

import os
import socket

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)

# Hostnames, gathered: on one box they coincide, and telling two machines apart
# is half of what the .ravex-owner records are for (GPU-59).
names = [None] * world
dist.all_gather_object(names, socket.gethostname())

t = torch.full((1024, 1024), float(rank + 1), device="cuda")
dist.all_reduce(t)
nccl_ok = int(t[0, 0].item()) == sum(range(1, world + 1))

byte_group = dist.new_group(ranks=list(range(world)), backend="gloo")
h = torch.full((4 * 1024 * 1024,), rank + 1, dtype=torch.uint8)
dist.all_reduce(h, group=byte_group)
gloo_ok = int(h[0].item()) == sum(range(1, world + 1)) % 256

if rank == 0:
    print(f"  world {world}, machines {len(set(names))}: {sorted(set(names))}")
    print(f"  nccl all_reduce on cuda: {'ok' if nccl_ok else 'WRONG RESULT'}")
    print(f"  gloo all_reduce on host: {'ok' if gloo_ok else 'WRONG RESULT'}")
    print("  ready" if nccl_ok and gloo_ok else "  NOT ready")

dist.destroy_process_group()
