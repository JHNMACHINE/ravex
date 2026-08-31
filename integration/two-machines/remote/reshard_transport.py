"""What it costs to move an old shard to the machine that needs it — GPU-96.

The issue asks for this measurement *before* the transport is written, and the
order is the point: if moving the bytes between machines is slower than
consolidating them on remote storage, then the honest answer to a
cross-machine reshard is "put the checkpoint somewhere both machines can see"
and there is no transport to write. That answer is only worth having if it is
measured on two real machines — on one box every number here reads zero, which
is the lesson GPU-83 paid for.

Three legs, and the gap between them is the finding:

  p2p     ``ravex._elastic.prestage_send`` / ``prestage_receive``, which is the
          socket a slice mover would inherit. A whole store rather than a
          slice, deliberately: the transport does not exist yet, and a whole
          store is the honest ceiling on what one would achieve.

  resync  the same call again with the destination already holding the bytes.
          The manifest/skip path, which is what a pre-staged reshard would pay
          on its later rounds rather than the full price every time.

  remote  the same store up to object storage and back down. This is the
          alternative the issue names, and it moves *more* bytes than the p2p
          leg does — up and then down — so if it still wins, it wins clearly.

What the numbers are divided into is not this store's size. It is the volume
:func:`ravex._reshard.crossing_bytes` computes for a given reshard, which is
usually a small fraction of a checkpoint and is sometimes exactly zero. Sizing
a transport off "a reshard moves everything" would be sizing it off a case
that mostly does not happen — see ``tests/test_reshard_locality.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch.distributed as dist

from ravex._elastic import prestage_receive, prestage_send
from ravex._replication import store_files


def directory_bytes(path: str) -> int:
    """What ``store_files`` says is there, which is what the transport moves."""
    return sum(size for _, size in store_files(path))


def human(rate: float) -> str:
    return "%.1f MB/s" % (rate / 1e6)


def leg_p2p(args, rank: int, results: dict) -> None:
    """Node 1 sends its store to node 0 over a plain socket, twice.

    The second round is not a repeat for confidence — it is a different
    measurement. Everything is already on the far side by then, so what it
    times is the manifest exchange and the skip decision with no payload
    behind them, which is the cost a pre-staging scheme pays on every round
    after its first.
    """
    port = args.port
    if rank == 0:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.self_addr, port))
        listener.listen(1)
        dist.barrier()
        conn, _ = listener.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            for round_name in ("p2p", "resync"):
                dist.barrier()
                started = time.perf_counter()
                complete = prestage_receive(conn, args.into)
                results[round_name + "_seconds"] = time.perf_counter() - started
                results[round_name + "_complete"] = bool(complete)
                results[round_name + "_landed_bytes"] = directory_bytes(args.into)
        finally:
            conn.close()
            listener.close()
        return

    dist.barrier()
    # A short retry: the listener is up before the barrier, but a container
    # bridge can still refuse the first SYN, and one failed connect should not
    # cost the phase.
    sock = None
    for attempt in range(20):
        try:
            sock = socket.create_connection((args.peer_addr, port), timeout=30)
            break
        except OSError:
            if attempt == 19:
                raise
            time.sleep(0.5)
    assert sock is not None
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        for round_name in ("p2p", "resync"):
            dist.barrier()
            started = time.perf_counter()
            prestage_send(sock, args.store)
            results[round_name + "_seconds"] = time.perf_counter() - started
    finally:
        sock.close()


def leg_remote(args, rank: int, results: dict) -> None:
    """The same store up to object storage and back down, from node 1 only.

    Node 1 alone, because the comparison is between two ways of getting one
    machine's shard to another, not between two ways of saturating a cluster.
    A parallel upload from every rank would measure the provider's aggregate,
    which flatters this leg for a reason that has nothing to do with the
    choice being made.

    Skipped rather than faked when there are no credentials: a leg that
    silently reports the local filesystem's speed as object storage's is worse
    than a leg that does not run.
    """
    if rank != 0:
        return

    key_id = os.environ.get("RAVEX_S3_ACCESS_KEY")
    secret = os.environ.get("RAVEX_S3_SECRET_KEY")
    bucket = os.environ.get("RAVEX_S3_BUCKET")
    endpoint = os.environ.get("RAVEX_S3_ENDPOINT")
    if not (key_id and secret and bucket):
        results["remote_skipped"] = (
            "no RAVEX_S3_ACCESS_KEY / RAVEX_S3_SECRET_KEY / RAVEX_S3_BUCKET"
        )
        return

    import boto3
    from concurrent.futures import ThreadPoolExecutor

    client = boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        endpoint_url=endpoint or None,
        region_name=os.environ.get("RAVEX_S3_REGION", "auto"),
    )
    prefix = "gpu96/%s" % os.environ.get("RUNPOD_POD_ID", socket.gethostname())
    files = store_files(args.store)
    total = sum(size for _, size in files)

    # Concurrent, because a serial upload of many small files measures
    # round-trip latency rather than bandwidth, and nobody consolidating a
    # checkpoint on purpose would do it serially. This is the alternative at
    # its best, which is the only version worth comparing against.
    def put(item):
        relative, _ = item
        client.upload_file(
            os.path.join(args.store, relative), bucket, "%s/%s" % (prefix, relative)
        )

    def get(item):
        relative, _ = item
        target = os.path.join(args.into + "-remote", relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        client.download_file(bucket, "%s/%s" % (prefix, relative), target)

    with ThreadPoolExecutor(max_workers=args.remote_workers) as pool:
        started = time.perf_counter()
        list(pool.map(put, files))
        results["upload_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        list(pool.map(get, files))
        results["download_seconds"] = time.perf_counter() - started

    results["remote_files"] = len(files)
    results["remote_bytes"] = total
    results["remote_workers"] = args.remote_workers

    for relative, _ in files:  # leave the bucket as it was found
        try:
            client.delete_object(Bucket=bucket, Key="%s/%s" % (prefix, relative))
        except Exception:  # pragma: no cover - a leftover object is not a failure
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, help="the store to move (node 1)")
    parser.add_argument("--into", required=True, help="where it lands (node 0)")
    parser.add_argument("--self-addr", required=True)
    parser.add_argument("--peer-addr", required=True)
    parser.add_argument("--port", type=int, default=29777)
    parser.add_argument("--remote-workers", type=int, default=16)
    parser.add_argument("--out", default="")
    parser.add_argument("--legs", default="p2p,remote")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    results: dict = {"rank": rank, "host": socket.gethostname()}

    if rank == 1:
        results["store_bytes"] = directory_bytes(args.store)
        results["store_files"] = len(store_files(args.store))

    legs = args.legs.split(",")
    if "p2p" in legs:
        leg_p2p(args, rank, results)
    dist.barrier()
    if "remote" in legs:
        # Caught, not allowed to propagate. The first run of this phase lost a
        # perfectly good p2p measurement - minutes of a rented link - because
        # an S3 token without write permission raised here, and the report is
        # written after both legs. A leg that cannot run is a fact to record
        # beside the other leg's numbers, not a reason to discard them.
        try:
            leg_remote(args, rank, results)
        except Exception as exc:
            results["remote_error"] = "%s: %s" % (type(exc).__name__, exc)
    dist.barrier()

    # Both halves of every rate live on different ranks — node 1 knows how big
    # the store was, node 0 knows how long the read took — so the numerator and
    # the denominator are put together in one place rather than each rank
    # printing half a fraction.
    everyone: list = [None, None]
    dist.all_gather_object(everyone, results)

    if rank == 0:
        merged = {}
        for one in everyone:
            merged.update({k: v for k, v in one.items() if k not in ("rank", "host")})
        merged["hosts"] = [one["host"] for one in everyone]
        report(merged)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(merged, handle, indent=2, sort_keys=True)

    dist.destroy_process_group()


def report(m: dict) -> None:
    size = m.get("store_bytes", 0)
    print("\n  store            %.0f MB in %d files (%s -> %s)"
          % (size / 1e6, m.get("store_files", 0), m["hosts"][1], m["hosts"][0]))

    if "p2p_seconds" in m:
        seconds = m["p2p_seconds"]
        print("  p2p              %s (%.1f s, socket, ravex._elastic.prestage_*)"
              % (human(size / seconds if seconds else 0), seconds))
        # `complete` is StoreWriter's own verdict, and it is the one that
        # counts. Comparing the two directories' byte totals looked like a
        # stronger check and is a worse one: the destination's manifest is
        # rewritten as it is received and legitimately differs from the
        # source's by a few bytes, so equality reports a failure on a
        # transfer that worked. The size is still printed, because a *large*
        # gap would mean something else.
        if not m.get("p2p_complete"):
            print("  ** the store did not arrive complete **")
        landed = m.get("p2p_landed_bytes", 0)
        if abs(landed - size) > 4096:
            print("  ** landed %d bytes against %d sent **" % (landed, size))

    if "resync_seconds" in m:
        seconds = m["resync_seconds"]
        print("  resync           %.2f s for nothing to move (manifest + skip only)"
              % seconds)

    if m.get("remote_error"):
        print("  remote           failed: %s" % m["remote_error"])
        print("                   AccessDenied on PutObject means the token can "
              "read the bucket and\n                   not write it - an R2 token "
              "needs Object Read *and* Write.")
    elif m.get("remote_skipped"):
        print("  remote           skipped: %s" % m["remote_skipped"])
        print("                   without it there is no comparison to make - this "
              "phase measured\n                   one of the two options and not "
              "the choice between them.")
    elif "upload_seconds" in m:
        up, down = m["upload_seconds"], m["download_seconds"]
        rb = m.get("remote_bytes", size)
        print("  remote up        %s (%.1f s, %d workers)"
              % (human(rb / up if up else 0), up, m.get("remote_workers", 0)))
        print("  remote down      %s (%.1f s)"
              % (human(rb / down if down else 0), down))
        print("  remote round     %s effective (%.1f s for up *and* down)"
              % (human(rb / (up + down) if up + down else 0), up + down))

        p2p = m.get("p2p_seconds")
        if p2p:
            factor = p2p / (up + down)
            if factor > 1:
                print("\n  -> consolidating on object storage is %.1fx faster than "
                      "moving the\n     same bytes machine to machine, and that is "
                      "with it carrying them twice." % factor)
            else:
                print("\n  -> the direct link wins by %.1fx even against a single "
                      "upload plus\n     download, so a point-to-point transport is "
                      "worth writing." % (1 / factor))


if __name__ == "__main__":
    main()
