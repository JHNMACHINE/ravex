"""A store must arrive byte for byte, over a real process group.

    python integration/scripts/verify_replication_transfer.py

`tests/test_replication.py` drives `StoreWriter` in one process, which covers
the framing. This covers the thing that unit test cannot: `exchange_stores`
between two processes, with sizes agreed on the wire, chunks arriving in the
order the sender posted them, and the completeness marker written at the end.

The store is built to be awkward on purpose. Since 2026-08-21 the receiving
side writes straight from the wire's memory whenever nothing is half-parsed,
and falls back to buffering for headers and file boundaries — so the cases
worth suspecting are the ones where the buffer used to do the work:

    empty.bin       zero length: created by being reached, not by being written
    tiny.bin        one byte
    exact.bin       exactly one chunk, so a file ends where a chunk ends
    exact_plus.bin  one byte past a chunk boundary
    big.bin         several chunks with a ragged remainder
    nested/...      subdirectories, whose names travel in the header

A small chunk is used so those boundaries come up often. Exits non-zero if any
byte differs, so it can be wired into a check.

No GPU, and it runs on Windows as well as Linux.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import torch.distributed as dist

from ravex._replication import COMPLETE_MARKER, encoded_size, exchange_stores

CHUNK = 64 * 1024


def build_awkward_store(path, chunk):
    plan = [
        ("empty.bin", 0),
        ("tiny.bin", 1),
        ("small.bin", 1024),
        ("exact.bin", chunk),
        ("exact_plus.bin", chunk + 1),
        ("big.bin", chunk * 3 + 12345),
        (os.path.join("nested", "a.bin"), chunk // 2),
        (os.path.join("nested", "deeper", "b.bin"), chunk * 2 - 7),
    ]
    for name, size in plan:
        target = os.path.join(path, name)
        os.makedirs(os.path.dirname(target) or path, exist_ok=True)
        with open(target, "wb") as handle:
            written = 0
            seed = 0
            while written < size:
                # Content that differs per file and per position, so a chunk
                # delivered to the wrong offset cannot go unnoticed.
                block = hashlib.sha256(f"{name}-{seed}".encode()).digest() * 1024
                handle.write(block[: min(len(block), size - written)])
                written += min(len(block), size - written)
                seed += 1
    return plan


def tree_digest(root):
    """Every name and every byte that has to survive the trip."""
    entries = {}
    for base, _, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(base, name)
            relative = os.path.relpath(full, root).replace(os.sep, "/")
            if relative == COMPLETE_MARKER:
                continue
            with open(full, "rb") as handle:
                entries[relative] = hashlib.sha256(handle.read()).hexdigest()
    return entries


def child():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    peer = 1 - rank
    report = rank == 0

    root = tempfile.mkdtemp(prefix=f"ravex-verify-{rank}-")
    source = os.path.join(root, "store")
    build_awkward_store(source, CHUNK)
    expected = tree_digest(source)
    destination = os.path.join(root, "copy")

    if report:
        print(f"\n  {len(expected)} files, {encoded_size(source)} encoded bytes, "
              f"{CHUNK // 1024} KiB chunks")

    dist.barrier()
    ok = exchange_stores(source, destination, peer, peer, chunk=CHUNK)
    got = tree_digest(destination)
    marker = os.path.exists(os.path.join(destination, COMPLETE_MARKER))

    problems = []
    if not ok:
        problems.append("exchange_stores reported an incomplete arrival")
    if not marker:
        problems.append(f"{COMPLETE_MARKER} is missing")
    for name in sorted(set(expected) | set(got)):
        if expected.get(name) != got.get(name):
            problems.append(
                f"{name}: expected {expected.get(name, '<absent>')[:12]}, "
                f"got {got.get(name, '<absent>')[:12]}"
            )

    if report:
        if problems:
            for problem in problems:
                print(f"  FAIL  {problem}")
        else:
            print(f"  ok    {len(got)} files, byte for byte, marker written")

    dist.barrier()
    dist.destroy_process_group()
    shutil.rmtree(root, ignore_errors=True)
    return 1 if problems else 0


def main():
    env = dict(os.environ)
    env.update(
        MASTER_ADDR=env.get("MASTER_ADDR", "127.0.0.1"),
        MASTER_PORT=env.get("MASTER_PORT", "29712"),
        WORLD_SIZE="2",
        RAVEX_ENABLED="0",
        RAVEX_VERIFY_CHILD="1",
    )
    processes = []
    for rank in (0, 1):
        env["RANK"] = str(rank)
        processes.append(subprocess.Popen([sys.executable, __file__] + sys.argv[1:], env=env))
    return max(process.wait() for process in processes)


if __name__ == "__main__":
    sys.exit(child() if os.environ.get("RAVEX_VERIFY_CHILD") else main())
