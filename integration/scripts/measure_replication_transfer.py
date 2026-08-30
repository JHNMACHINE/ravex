"""What a replica costs to move, and where the time goes when it costs too much.

    python integration/scripts/measure_replication_transfer.py --gib 1

Two gloo processes on loopback. That is not a compromise: the same path was
measured at 529 MB/s over loopback and 562 MB/s over real TCP between two
containers on 2026-08-21, so whatever binds it is not the transport, and it can
be studied on a laptop instead of on two rented machines.

Four things are timed, and the first two are the bounds the third has to be
read against:

    wire      isend/recv of the same volume, no file touched
    read      `encode_store` off the disk and thrown away, nothing sent
    exchange  `exchange_stores`, the path replication actually uses
    phases    the same loop with a stopwatch on each step

`exchange` sitting far below `wire` and `read` is the finding this script
exists to catch. It did, in GPU-84: 529 MB/s against a wire carrying 1462 and a
disk delivering 3411, because every chunk was copied four times. Removing two of
those copies took the transfer from 355 to 527 MB/s on the machine this was
written on — the wire's own speed — and this script is what would notice it
sliding back.

Nothing here needs a GPU, and it runs on Windows as well as Linux.

**`phases` mirrors the body of `exchange_stores` on purpose**, because a timer
wrapped around the function cannot see inside it. It has to be updated when
that loop changes; if the two disagree, the loop in `ravex/_replication.py` is
the truth and this is stale.

Its own rate is always lower than `exchange`'s — a stopwatch on every chunk is
not free — so read `phases` for the *proportions* and `exchange` for the
number. Comparing the two rates would only measure this script.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import torch.distributed as dist

from ravex import _replication
from ravex._replication import (
    StoreWriter,
    encode_store,
    encoded_size,
    fixed_chunks,
)


def human(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def build_store(path, total, files=8):
    """A directory shaped like a store: a few large files, as Moonclip writes.

    The shape matters. Many small files would spend the run in the framing
    path and say nothing about the bulk transfer, which is what a checkpoint
    actually is.
    """
    os.makedirs(path, exist_ok=True)
    block = os.urandom(8 * 1024 * 1024)
    per_file = total // files
    for index in range(files):
        with open(os.path.join(path, f"chunk_{index}.bin"), "wb") as handle:
            written = 0
            while written < per_file:
                handle.write(block[: min(len(block), per_file - written)])
                written += len(block)


def time_wire(peer, size, chunk):
    out = torch.ones(chunk, dtype=torch.uint8)
    inp = torch.empty(chunk, dtype=torch.uint8)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(size // chunk):
        handle = dist.isend(out, dst=peer)
        dist.recv(inp, src=peer)
        handle.wait()
    return time.perf_counter() - started


def time_read(source, chunk):
    dist.barrier()
    started = time.perf_counter()
    for _ in fixed_chunks(encode_store(source, chunk=chunk), chunk):
        pass
    return time.perf_counter() - started


def time_exchange(source, destination, peer, chunk):
    shutil.rmtree(destination, ignore_errors=True)
    dist.barrier()
    started = time.perf_counter()
    ok = _replication.exchange_stores(source, destination, peer, peer, chunk=chunk)
    seconds = time.perf_counter() - started
    if not ok:
        print("    ** the copy did not arrive whole **")
    shutil.rmtree(destination, ignore_errors=True)
    return seconds


def time_phases(source, destination, peer, chunk, report):
    """`exchange_stores`, step by step. See the note at the top of the file.

    Destination is emptied before every round (see `time_exchange`), so the
    manifest round trip below never finds anything to skip — this arm is
    still the right place to see what that round trip costs on its own, as
    the fixed floor GPU-97's fix adds to every replication, not to see the
    incremental win, which needs a destination that already holds a copy.
    """
    shutil.rmtree(destination, ignore_errors=True)
    existing_blob = _replication._encode_manifest(_replication.store_files(destination))

    dist.barrier()
    started = time.perf_counter()

    spent = {
        "manifest": 0.0, "read": 0.0, "isend": 0.0, "recv": 0.0,
        "write": 0.0, "wait": 0.0,
    }

    mark = time.perf_counter()
    blob_len = torch.tensor([len(existing_blob)], dtype=torch.int64)
    handle = dist.isend(blob_len, dst=peer)
    blob_len_in = torch.zeros(1, dtype=torch.int64)
    dist.recv(blob_len_in, src=peer)
    incoming_blob_len = int(blob_len_in[0].item())
    handle.wait()

    handle = None
    if existing_blob:
        blob_tensor = _replication._wire_tensor(existing_blob)
        handle = dist.isend(blob_tensor, dst=peer)
    peer_existing = {}
    if incoming_blob_len:
        blob_buffer = torch.empty(incoming_blob_len, dtype=torch.uint8)
        dist.recv(blob_buffer, src=peer)
        peer_existing = dict(_replication._parse_manifest(blob_buffer.numpy().tobytes()))
    if handle is not None:
        handle.wait()
    spent["manifest"] = time.perf_counter() - mark

    skip = {
        relative
        for relative, size in _replication.store_files(source)
        if peer_existing.get(relative) == size
    }
    outgoing = encoded_size(source, skip=skip)

    size = torch.tensor([outgoing], dtype=torch.int64)
    handle = dist.isend(size, dst=peer)
    size_in = torch.zeros(1, dtype=torch.int64)
    dist.recv(size_in, src=peer)
    incoming = int(size_in[0].item())
    handle.wait()

    mine = fixed_chunks(encode_store(source, chunk=chunk, skip=skip), chunk)
    writer = StoreWriter(destination)
    my_chunks = -(-outgoing // chunk)
    their_chunks = -(-incoming // chunk)

    for index in range(max(my_chunks, their_chunks)):
        handle = None
        if index < my_chunks:
            mark = time.perf_counter()
            block = next(mine)
            spent["read"] += time.perf_counter() - mark

            mark = time.perf_counter()
            outgoing_chunk = _replication._wire_tensor(block)
            handle = dist.isend(outgoing_chunk, dst=peer)
            spent["isend"] += time.perf_counter() - mark

        if index < their_chunks:
            expected = min(chunk, incoming - index * chunk)
            mark = time.perf_counter()
            buffer = torch.empty(expected, dtype=torch.uint8)
            dist.recv(buffer, src=peer)
            spent["recv"] += time.perf_counter() - mark

            mark = time.perf_counter()
            writer.feed(buffer.numpy())
            spent["write"] += time.perf_counter() - mark

        if handle is not None:
            mark = time.perf_counter()
            handle.wait()
            spent["wait"] += time.perf_counter() - mark

    writer.close()
    writer.commit()
    seconds = time.perf_counter() - started
    shutil.rmtree(destination, ignore_errors=True)

    if report:
        print()
        for name, value in sorted(spent.items(), key=lambda item: -item[1]):
            print(f"    {name:<8}{value:>8.2f} s{value / seconds:>7.0%}")
        print()
    return seconds


def child():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=float, default=1.0)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--arms", default="wire,read,exchange,phases")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="rounds; arms alternate and the median is reported, because a "
        "single run of each is not enough to tell a change from the weather",
    )
    parser.add_argument("--dir", default=None, help="where the scratch store goes")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    peer = 1 - rank
    report = rank == 0
    chunk = args.chunk_mib * 1024 * 1024

    root = tempfile.mkdtemp(prefix=f"ravex-transfer-{rank}-", dir=args.dir)
    source = os.path.join(root, "store")
    build_store(source, int(args.gib * 1024 ** 3))
    size = encoded_size(source)
    if report:
        print(f"\n  store {human(size)} in 8 files, chunk {args.chunk_mib} MiB, "
              f"{args.repeat} rounds\n")

    runs = {arm: [] for arm in args.arms.split(",")}
    try:
        for round_index in range(args.repeat):
            for arm in runs:
                if arm == "wire":
                    seconds = time_wire(peer, size, chunk)
                elif arm == "read":
                    seconds = time_read(source, chunk)
                elif arm == "exchange":
                    seconds = time_exchange(source, os.path.join(root, "copy"), peer, chunk)
                elif arm == "phases":
                    seconds = time_phases(
                        source, os.path.join(root, "copy"), peer, chunk,
                        report and round_index == 0,
                    )
                else:
                    raise SystemExit(f"unknown arm: {arm}")
                runs[arm].append(seconds)

        if report:
            for arm, seconds in runs.items():
                median = statistics.median(seconds)
                rates = sorted(round(size / value / 1e6) for value in seconds)
                print(f"  {arm:<10}{median:>7.2f} s{size / median / 1e6:>8.0f} MB/s"
                      f"   runs {rates}")
    finally:
        dist.destroy_process_group()
        shutil.rmtree(root, ignore_errors=True)


def main():
    """One process per rank: a process group wants two participants.

    Spawned here rather than left to `torchrun` so the script runs the same way
    on a laptop as on a box, and so `--repeat` means rounds of the same pair
    rather than two unrelated processes racing.
    """
    env = dict(os.environ)
    env.update(
        MASTER_ADDR=env.get("MASTER_ADDR", "127.0.0.1"),
        MASTER_PORT=env.get("MASTER_PORT", "29711"),
        WORLD_SIZE="2",
        RAVEX_ENABLED="0",
        RAVEX_TRANSFER_CHILD="1",
    )
    processes = []
    for rank in (0, 1):
        env["RANK"] = str(rank)
        processes.append(subprocess.Popen([sys.executable, __file__] + sys.argv[1:], env=env))
    return max(process.wait() for process in processes)


if __name__ == "__main__":
    sys.exit(child() if os.environ.get("RAVEX_TRANSFER_CHILD") else main())
