"""Where a replica transfer's time actually goes, with the network taken out.

Two processes on one machine over gloo on loopback, so the link is not a
variable and whatever the number is, it is ours. Three arms, and the point is
the gap between them:

**wire** - gloo moving the same chunks with no files anywhere. The ceiling the
other two are measured against.

**encode** - `encode_store` read off disk and thrown away, sender side only.
Framing plus disk, no network.

**exchange** - `exchange_stores`, the path a replication round actually takes.

GPU-84 ran exactly this on 2026-08-21 (128 cores, 4 GiB in 8 files) and found
1462 MB/s on the wire, 3411 MB/s encoding, and **529 MB/s** through the real
path - a third of the wire, getting *worse* as the chunk grew: 244 MB/s at
16 MiB and 113 MB/s at 64 MiB. The script that produced it was a throwaway in a
session scratchpad and is gone; this is it, in the repo, because GPU-109 turns
on whether that number still holds.

Two of the four copies GPU-84 blamed have since been removed - see the "No copy
on the way out" comment in `exchange_stores` and the fast path in
`StoreWriter.feed`. What has *not* been addressed is the lockstep: send,
receive and disk write take turns, while each of them alone runs several times
faster. So the shape of the result matters as much as the rate. If exchange now
tracks the wire, the transport is not the cost and GPU-109 closes as a note.

    python bench/transport_ceiling.py --size 1 --files 8

Run it where the answer matters. On a laptop this measures the laptop, and on
Windows it measures a gloo that is not the one production uses. The store is
read repeatedly and will be in page cache: that is deliberate, since the
question is what the transport costs, not what a cold disk costs.
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ravex._dist.replication import (
    encode_store,
    encoded_size,
    exchange_stores,
    fixed_chunks,
)

MIB = 1 << 20


def build_store(path: str, total: int, files: int) -> int:
    """`files` files of equal size summing to `total`, filled with junk.

    Random-ish rather than zeros, in case anything downstream is ever tempted
    to compress: this benchmark should never be able to report a rate the real
    thing cannot reach.
    """
    os.makedirs(path, exist_ok=True)
    each = total // files
    block = os.urandom(min(each, 8 * MIB))
    written = 0
    for index in range(files):
        target = each if index < files - 1 else total - written
        with open(os.path.join(path, "shard_%02d.bin" % index), "wb") as handle:
            left = target
            while left > 0:
                take = min(left, len(block))
                handle.write(block[:take])
                left -= take
        written += target
    return written


def rate(byte_count: int, seconds: float) -> float:
    return byte_count / seconds / MIB if seconds > 0 else float("inf")


def arm_wire(rank: int, total: int, chunk: int) -> float:
    """gloo alone, same chunking, no files. Rank 0 sends, rank 1 receives."""
    block = bytes(chunk)
    count = -(-total // chunk)

    dist.barrier()
    started = time.perf_counter()
    for index in range(count):
        size = min(chunk, total - index * chunk)
        if rank == 0:
            tensor = torch.frombuffer(block, dtype=torch.uint8)[:size]
            dist.send(tensor, dst=1)
        else:
            buffer = torch.empty(size, dtype=torch.uint8)
            dist.recv(buffer, src=0)
    elapsed = time.perf_counter() - started
    dist.barrier()
    return elapsed


def arm_encode(rank: int, source: str, chunk: int) -> float:
    """Disk and framing, sender side only. Rank 1 waits."""
    dist.barrier()
    started = time.perf_counter()
    if rank == 0:
        for _ in fixed_chunks(encode_store(source, chunk=chunk), chunk):
            pass
    elapsed = time.perf_counter() - started
    dist.barrier()
    return elapsed


def arm_exchange(rank: int, source: str, destination: str, chunk: int) -> float:
    """The real path, one way: rank 0's store lands in rank 1's directory."""
    if rank == 1:
        shutil.rmtree(destination, ignore_errors=True)
        os.makedirs(destination, exist_ok=True)

    dist.barrier()
    started = time.perf_counter()
    if rank == 0:
        ok = exchange_stores(source, None, send_to=1, receive_from=-1, chunk=chunk)
    else:
        ok = exchange_stores(None, destination, send_to=-1, receive_from=0, chunk=chunk)
    elapsed = time.perf_counter() - started
    dist.barrier()

    if not ok:
        raise RuntimeError("the exchange reported an incomplete store")
    return elapsed


def worker(rank: int, args, root: str, results) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    dist.init_process_group("gloo", rank=rank, world_size=2)

    source = os.path.join(root, "source")
    destination = os.path.join(root, "destination")
    total = int(args.size * (1 << 30))

    if rank == 0:
        total = build_store(source, total, args.files)
    payload = torch.tensor([total], dtype=torch.int64)
    dist.broadcast(payload, src=0)
    total = int(payload[0].item())

    # What actually crosses is the framed store, not the sum of the files: the
    # manifest and the per-file headers ride along. Small at this scale, and
    # reported rather than assumed away.
    framed = encoded_size(source) if rank == 0 else total

    for chunk_mib in args.chunks:
        chunk = chunk_mib * MIB
        seconds = arm_wire(rank, total, chunk)
        if rank == 1:
            results.append(("wire", chunk_mib, seconds, total))

    seconds = arm_encode(rank, source, args.chunks[0] * MIB)
    if rank == 0:
        results.append(("encode", args.chunks[0], seconds, framed))

    for chunk_mib in args.chunks:
        chunk = chunk_mib * MIB
        seconds = arm_exchange(rank, source, destination, chunk)
        if rank == 1:
            results.append(("exchange", chunk_mib, seconds, framed))

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float, default=1.0, help="store size, GiB")
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[4, 16, 64],
        metavar="MIB",
        help="chunk sizes to sweep; GPU-84 found the real path degrading here",
    )
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--keep", action="store_true", help="leave the store on disk")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="ravex-transport-")
    manager = mp.Manager()
    results = manager.list()
    try:
        mp.spawn(worker, args=(args, root, results), nprocs=2, join=True)
        rows = list(results)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    order = {"wire": 0, "encode": 1, "exchange": 2}
    rows.sort(key=lambda row: (order[row[0]], row[1]))

    print()
    print("%-10s %8s %10s %12s" % ("arm", "chunk", "seconds", "MB/s"))
    for name, chunk_mib, seconds, byte_count in rows:
        print(
            "%-10s %6d M %10.2f %12.0f"
            % (name, chunk_mib, seconds, rate(byte_count, seconds))
        )
    print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "arm": name,
                        "chunk_mib": chunk_mib,
                        "seconds": seconds,
                        "bytes": byte_count,
                        "mb_per_second": rate(byte_count, seconds),
                    }
                    for name, chunk_mib, seconds, byte_count in rows
                ],
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
