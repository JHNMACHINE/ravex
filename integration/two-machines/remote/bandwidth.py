"""What the two transports carry between these boxes, measured as they are used.

Gloo first, chunked at 4 MiB and sending in both directions at once, because
that is exactly what `exchange_stores` does around the ring: every rank sends
one store and receives another at the same time, so a one-way number would
overstate what the replication gets.

Then NCCL, which is what training runs on. It is not the replication's
transport since GPU-82 — the store bytes are host tensors on a gloo subgroup —
but if NCCL cannot cross this overlay, nothing above it matters.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

CHUNK = 4 * 1024 * 1024  # ravex._replication.CHUNK

parser = argparse.ArgumentParser()
parser.add_argument("--size-mb", type=int, default=2048)
args = parser.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
peer = (rank + 1) % world

total = args.size_mb * 1024 * 1024
chunks = total // CHUNK

byte_group = dist.new_group(ranks=list(range(world)), backend="gloo")

out = torch.ones(CHUNK, dtype=torch.uint8)
inp = torch.empty(CHUNK, dtype=torch.uint8)

dist.barrier()
started = time.perf_counter()
for _ in range(chunks):
    handle = dist.isend(out, dst=peer, group=byte_group)
    dist.recv(inp, src=peer, group=byte_group)
    handle.wait()
gloo_seconds = time.perf_counter() - started
dist.barrier()

buf = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
dist.barrier()
started = time.perf_counter()
for _ in range(4):
    dist.all_reduce(buf)
torch.cuda.synchronize()
nccl_seconds = time.perf_counter() - started

if rank == 0:
    moved = chunks * CHUNK
    print(f"  gloo  {moved / gloo_seconds / 1e6:7.0f} MB/s each way "
          f"({moved / 1e6:.0f} MB in {gloo_seconds:.1f} s, 4 MiB chunks, "
          "both directions at once)")
    # ring all-reduce moves 2*(n-1)/n of the buffer per rank
    bus = 4 * buf.numel() * 4 * 2 * (world - 1) / world
    print(f"  nccl  {bus / nccl_seconds / 1e6:7.0f} MB/s bus "
          f"({nccl_seconds:.1f} s for 4 all_reduce of 1 GiB)")
    rate = moved / gloo_seconds  # bytes per second, one direction
    print(f"  a 3 GiB store therefore takes about "
          f"{3 * 1024 ** 3 / rate:.0f} s to replicate — compare that with the "
          "interval 30-measure.sh reports")

dist.destroy_process_group()
