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

**gather** - `all_ranks_agree(True)`, the other shape of the same family: an
all-gather of a Python object rather than a reduce of a tensor. Not per-step —
it gates a resume — but it is what a store round would replace first.

**store-check** - `store.check([missing])` on the rendezvous store: the whole
cost of the announced-flag design in the common case, where nobody is asking
for anything and the answer is the absence of one key.

**skew** - the same two, with one rank arriving late on purpose. This is the
arm that matters. A collective makes every rank wait for the slowest, so a
straggler's delay is paid by everybody; a key that is simply not there is
answered without anyone waiting for anyone. On an idle box the first three arms
understate the difference to nothing, because there is no skew to absorb — and
skew is what a training job has.

    python bench/agreement_cost.py --ranks 4 --iterations 2000 --skew-ms 5

Gloo on loopback, so this is the floor: on a real fabric the collective's
number grows with the fabric and the store's grows with the store server's
latency, and the two do not grow together.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ravex._dist.collectives import all_ranks_agree, emergency_group, emergency_signalled

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
        (rank, "store-check", summarise(timed(lambda: store.check([MISSING_KEY]), args.iterations)))
    )

    skew = args.skew_ms / 1000.0
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

    order = ["signal", "gather", "store-check", "signal+skew", "store-check+skew"]
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
