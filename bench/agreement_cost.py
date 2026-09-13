"""What the per-step agreement costs, before anyone rewrites it.

Ravex asks its ranks to agree about small things, and one of those questions is
asked **every step**: `emergency_signalled`, an `all_reduce(MAX)` of one int32
on a short-timeout gloo group, which is how a rank that caught SIGTERM pulls
everyone else into a checkpoint with it. GPU-111 proposes replacing that family
with a key-value round on the rendezvous store, and the first thing that issue
asks for is this number, because the same rule applied to the transport
(GPU-109) changed the premise twice.

Four arms, and the fourth is the one that decides:

**signal** - `emergency_signalled(False, group)`, the real per-step call.

**gather** - `all_ranks_agree(True)`, the other shape of the same family. Since
GPU-111 it takes the store road where there is a store, so this arm measures
whatever the shipped code actually does.

**gather-collectives** - the implementation it replaced, called directly. A
private function, reached on purpose: without it this bench stops being able to
answer "and what did it cost before", which is the question the next change will
ask too.

**store-check** - `store.check([missing])` on the rendezvous store. Since
GPU-111 this is not a hypothetical arm but **what a step actually costs**: the
alarm is announced on a key, and a run nobody is preempting spends its per-step
budget discovering that the key is not there. Compare it against **signal**,
which is what that step cost before.

**+skew** - the same calls, with one rank arriving late on purpose, and the
arms that keep the story honest in both directions. A key that is simply *not
there* is answered without waiting for anyone, so `store-check` does not move.
A **gather** cannot do that on any medium: it needs everyone's answer, so both
roads pay the straggler and the skew arms say so. What the store changes for a
gather is the cost of the mechanism, not the cost of waiting — and conflating
the two is how a 1.7x becomes a claim of 20x.

    python bench/agreement_cost.py --ranks 4 --iterations 2000 --skew-ms 5

Gloo on loopback, so this is the floor: on a real fabric the collective's
number grows with the fabric and the store's grows with the store server's
latency, and the two do not grow together.

Run on 2026-09-13, Windows, gloo on loopback, mean microseconds per call:

    ranks   signal   gather   gather-collectives   store-check
        2      458      332                  634            48
        4      376      982                 1416            83
        8      773     3305                 3097           162

Two readings, and the second is the one that moved GPU-111. **`signal` is
already the cheapest arm of its family** - 773 µs at 8 ranks against the 3305
of the store gather that replaced `all_ranks_agree` - so porting it to a store
*gather* would have been a 4x regression, growing with the rank count. What
justifies the change is not the mean but the straggler: with 5 ms of skew on
one rank, `signal+skew` reaches 5.8 ms on every other rank while
`store-check+skew` stays at ~92 µs, because the absence of a key is answered
without waiting for anybody and a gather never is.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ravex._dist.collectives import (
    _all_gather_object,
    all_ranks_agree,
    emergency_group,
    emergency_signalled,
)

MISSING_KEY = "ravex/bench/agreement/never-written"


def timed(call, iterations, warmup=50):
    """Per-call durations in microseconds, warmup discarded."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1e6)
    return samples


def timed_with_skew(call, iterations, rank, skew_seconds, warmup=10):
    """The same, with rank 0 arriving late on purpose.

    The barrier before each round is what makes the delay a *skew* rather than
    an offset that averages out: every rank starts the round together, then one
    of them stalls. What the others pay for that is the number this arm exists
    to produce.
    """
    samples = []
    for index in range(warmup + iterations):
        dist.barrier()
        if rank == 0:
            time.sleep(skew_seconds)
        started = time.perf_counter()
        call()
        elapsed = (time.perf_counter() - started) * 1e6
        if index >= warmup:
            samples.append(elapsed)
    return samples


def summarise(samples):
    ordered = sorted(samples)
    return {
        "calls": len(ordered),
        "mean_us": statistics.fmean(ordered),
        "median_us": statistics.median(ordered),
        "p99_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "max_us": ordered[-1],
    }


def worker(rank, args, results) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    dist.init_process_group("gloo", rank=rank, world_size=args.ranks)

    usable, group = emergency_group(args.timeout)
    if not usable:
        results.append((rank, "signal", None))
        dist.destroy_process_group()
        return

    store = dist.distributed_c10d._get_default_store()

    dist.barrier()
    results.append(
        (rank, "signal", summarise(timed(lambda: emergency_signalled(False, group), args.iterations)))
    )

    dist.barrier()
    results.append(
        (rank, "gather", summarise(timed(lambda: all_ranks_agree(True), args.iterations)))
    )

    dist.barrier()
    results.append(
        (
            rank,
            "gather-collectives",
            summarise(timed(lambda: _all_gather_object(dist, True), args.iterations)),
        )
    )

    dist.barrier()
    results.append(
        (rank, "store-check", summarise(timed(lambda: store.check([MISSING_KEY]), args.iterations)))
    )

    skew = args.skew_ms / 1000.0
    for label, call in (
        ("gather+skew", lambda: all_ranks_agree(True)),
        ("gather-collectives+skew", lambda: _all_gather_object(dist, True)),
    ):
        dist.barrier()
        results.append(
            (
                rank,
                label,
                summarise(
                    timed_with_skew(call, args.skew_iterations, rank, skew)
                ),
            )
        )

    dist.barrier()
    results.append(
        (
            rank,
            "signal+skew",
            summarise(
                timed_with_skew(
                    lambda: emergency_signalled(False, group), args.skew_iterations, rank, skew
                )
            ),
        )
    )

    dist.barrier()
    results.append(
        (
            rank,
            "store-check+skew",
            summarise(
                timed_with_skew(
                    lambda: store.check([MISSING_KEY]), args.skew_iterations, rank, skew
                )
            ),
        )
    )

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--skew-iterations", type=int, default=200)
    parser.add_argument(
        "--skew-ms",
        type=float,
        default=5.0,
        help="how late rank 0 arrives; what the others pay for it is the point",
    )
    parser.add_argument("--step-ms", type=float, default=100.0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--port", type=int, default=29873)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    manager = mp.Manager()
    results = manager.list()
    mp.spawn(worker, args=(args, results), nprocs=args.ranks, join=True)
    rows = list(results)

    if any(row[2] is None for row in rows):
        print("no emergency group could be built here; nothing to measure")
        return

    order = [
        "signal",
        "gather",
        "gather-collectives",
        "store-check",
        "signal+skew",
        "gather+skew",
        "gather-collectives+skew",
        "store-check+skew",
    ]
    print()
    print(
        "%-18s %6s %10s %10s %10s %10s"
        % ("arm", "rank", "mean us", "median", "p99", "max")
    )
    for name in order:
        for rank, arm, stats in sorted(rows, key=lambda row: row[0]):
            if arm != name:
                continue
            print(
                "%-18s %6d %10.1f %10.1f %10.1f %10.1f"
                % (name, rank, stats["mean_us"], stats["median_us"], stats["p99_us"], stats["max_us"])
            )
    print()

    # The one derived number, with its assumption named rather than buried: a
    # step is not measured here, it is supplied.
    for name in ("signal", "store-check"):
        costs = [stats["mean_us"] for _rank, arm, stats in rows if arm == name]
        if costs:
            share = statistics.fmean(costs) / (args.step_ms * 1000.0) * 100.0
            print(
                "%-12s %.1f us/step = %.4f%% of a %.0f ms step"
                % (name, statistics.fmean(costs), share, args.step_ms)
            )
    print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                [{"rank": rank, "arm": arm, **stats} for rank, arm, stats in rows],
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
