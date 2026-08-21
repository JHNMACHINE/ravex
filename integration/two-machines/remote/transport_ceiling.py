"""Where the 500 MB/s in the replication path actually comes from.

Measured on loopback, so the network is not a variable at all: whatever this
finds is a property of the code. Four arms, each isolating one suspect.

  raw          plain gloo isend/recv of the same volume, no filesystem
  encode       encode_store read from disk, thrown away, never sent
  exchange     the real thing: encode + send + StoreWriter on the far end
  exchange@N   the same, with a bigger chunk

If `raw` is fast and `exchange` is not, the transport is not the problem and a
faster link buys nothing. If both sit at the same number, the chunked
isend/recv is the ceiling — and that is worth knowing before anyone rents two
machines to measure a link they cannot use.
"""

import argparse
import os
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

from ravex._replication import StoreWriter, encode_store, encoded_size, fixed_chunks


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def make_store(path, total, files=8):
    """A store-shaped directory: a handful of large files, like Moonclip's."""
    os.makedirs(path, exist_ok=True)
    block = os.urandom(8 * 1024 * 1024)
    per_file = total // files
    for index in range(files):
        with open(os.path.join(path, f"chunk_{index}.bin"), "wb") as fh:
            written = 0
            while written < per_file:
                fh.write(block[: min(len(block), per_file - written)])
                written += len(block)
    os.sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=4.0)
    parser.add_argument("--chunks-mib", default="4,16,64")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    peer = 1 - rank
    total = int(args.gib * 1024 ** 3)

    root = tempfile.mkdtemp(prefix=f"ceiling{rank}-", dir="/root")
    source = os.path.join(root, "store")
    make_store(source, total)
    size = encoded_size(source)
    if rank == 0:
        print(f"  store: {human(size)} in 8 files, on {root}\n")

    def report(label, seconds):
        if rank == 0:
            print(f"  {label:<18s} {seconds:6.1f} s   "
                  f"{size / seconds / 1e6:6.0f} MB/s")

    # 1. The wire alone. Same volume, same chunking, nothing read or written.
    for chunk_mib in [int(c) for c in args.chunks_mib.split(",")]:
        chunk = chunk_mib * 1024 * 1024
        out = torch.ones(chunk, dtype=torch.uint8)
        inp = torch.empty(chunk, dtype=torch.uint8)
        count = size // chunk
        dist.barrier()
        started = time.perf_counter()
        for _ in range(count):
            handle = dist.isend(out, dst=peer)
            dist.recv(inp, src=peer)
            handle.wait()
        report(f"raw gloo @{chunk_mib}M", time.perf_counter() - started)

    # 2. Reading and framing the store, with nothing sent. The disk's share.
    dist.barrier()
    started = time.perf_counter()
    consumed = 0
    for block in fixed_chunks(encode_store(source, chunk=4 * 1024 * 1024),
                              4 * 1024 * 1024):
        consumed += len(block)
    report("encode only @4M", time.perf_counter() - started)

    # 3. The real path, at each chunk size.
    from ravex import _replication

    for chunk_mib in [int(c) for c in args.chunks_mib.split(",")]:
        destination = os.path.join(root, f"copy{chunk_mib}")
        shutil.rmtree(destination, ignore_errors=True)
        dist.barrier()
        started = time.perf_counter()
        ok = _replication.exchange_stores(
            source, destination, peer, peer, chunk=chunk_mib * 1024 * 1024
        )
        seconds = time.perf_counter() - started
        report(f"exchange @{chunk_mib}M", seconds)
        if rank == 0 and not ok:
            print("    ** the copy did not arrive whole **")
        shutil.rmtree(destination, ignore_errors=True)

    shutil.rmtree(root, ignore_errors=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
