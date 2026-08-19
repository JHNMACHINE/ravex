"""Does retaining the delta base make the *next* collection slower?

GPU-54: on the 8-GPU box, `collect` costs 5.33 s at step 2 and ~10.4 s from
step 4 on. Contention for Moonclip's compression pool was the obvious
explanation and has already been falsified — the doubling is identical with
`MOONCLIP_THREADS=1`. What is left is host memory: from the first checkpoint
onwards Moonclip holds a shadow copy *and*, with `keep_base_in_memory` on, a
full extra copy of the saved state. The next collection has to obtain its
buffers from an allocator that can no longer hand back the pages it just used.

This models that half of the problem and nothing else. There is no GPU here,
no FSDP and no device-to-host copy: `collect` below is a plain host-to-host
clone of a resident state dict, which is the allocation the real one ends in.
So a doubling here is evidence that the mechanism is host-side and reproducible
without renting anything. **The converse does not hold** — if the doubling does
not appear, the real cost may still be in the CUDA copy, which this cannot see.

    python bench/gpu54_base_retention.py --mib 512 --rounds 6

Run it on Linux if the answer is meant to inform the 8-GPU box. Page reuse is
allocator behaviour, and glibc's is not Windows'.
"""

import argparse
import gc
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple

import torch

from ravex._backends import MoonclipBackend
from ravex._config import RavexConfig


def make_state(mib: int, tensors: int = 64) -> Dict[str, torch.Tensor]:
    """A state dict of roughly ``mib`` MiB, in fp32, split across ``tensors``."""
    per = (mib * 1024 * 1024) // (tensors * 4)
    return {"p%02d" % i: torch.randn(per, dtype=torch.float32) for i in range(tensors)}


def mutate(state: Dict[str, torch.Tensor]) -> None:
    """A training step's worth of change: small, dense, everywhere.

    Deltas have to stay realistic — a state that never changes takes Moonclip's
    early-out path and retains nothing worth measuring.
    """
    for t in state.values():
        t.mul_(1.0 + 1e-3)


def collect(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fresh host buffers holding the current state.

    Stands in for `get_state_dict(cpu_offload=True)`, whose last act is exactly
    this allocation.
    """
    return {name: tensor.clone() for name, tensor in state.items()}


def run_control(mib: int, rounds: int) -> List[Tuple[float, float]]:
    """The same clone loop with no backend at all.

    The control that decides whether any of this is about checkpointing. If
    allocating and releasing the same buffers in a loop already costs twice as
    much from the second round on, then Moonclip, the delta and the base are
    all bystanders and the answer is in the allocator.
    """
    live = make_state(mib)
    timings: List[Tuple[float, float]] = []
    for _ in range(rounds):
        mutate(live)
        started = time.perf_counter()
        collected = collect(live)
        elapsed = time.perf_counter() - started
        timings.append((elapsed, elapsed))
        del collected

    del live
    gc.collect()
    return timings


def run(
    mib: int, rounds: int, keep_base: bool, async_save: bool = True
) -> List[Tuple[float, float]]:
    config = RavexConfig()
    config.storage.path = tempfile.mkdtemp(prefix="gpu54-")
    config.keep_base_in_memory = keep_base
    config.async_save = async_save
    # Retention is off on purpose: the question is what holding every base
    # costs, so nothing may be pruned mid-run. It is also why the `finally`
    # below is not optional — with `keep_last` this high the store only grows,
    # and `--ranks 8` runs this in eight processes at once.
    config.keep_last = rounds + 1
    backend = MoonclipBackend(config)

    try:
        live = make_state(mib)
        timings: List[Tuple[float, float]] = []
        for step in range(1, rounds + 1):
            mutate(live)
            started = time.perf_counter()
            collected = collect(live)
            collected_at = time.perf_counter()
            backend.save(step * 2, collected, {"step": str(step * 2)})
            # Both numbers, because they answer different questions. `collect`
            # is what GPU-54 measured. The total is what the training loop
            # actually waits for, and a writer that makes the first cheaper by
            # making the second dearer has not helped anyone.
            timings.append((collected_at - started, time.perf_counter() - started))
            # Ravex drops its reference here too; the point is that Moonclip's
            # copies outlive it.
            del collected

        del live, backend
        gc.collect()
        return timings
    finally:
        # After the backend is gone, so the background writer is not still
        # draining into a directory being removed. `ignore_errors` because a
        # bench that cannot tidy up should still report its numbers — the
        # measurement is the point, and the caller sees the disk either way.
        shutil.rmtree(config.storage.path, ignore_errors=True)


def _rank_worker(mib, rounds, keep_base, async_save, out):
    out.put(run(mib, rounds, keep_base, async_save))


def run_ranks(
    ranks: int, mib: int, rounds: int, keep_base: bool, async_save: bool = True
) -> List[List[Tuple[float, float]]]:
    """The same loop in ``ranks`` processes at once, each with its own manager.

    This is the shape the 8-GPU box actually has, and the shape a single
    process cannot show. Ravex builds Moonclip with ``world_size=1, rank=0`` on
    every rank, so eight co-located ranks are eight independent managers, each
    retaining its own full base. Moonclip sizes `keep_base_in_memory` at "+1.00x
    the saved state" — a figure written for one process, and multiplied by the
    rank count here without anyone choosing it.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out: Any = ctx.Queue()
    procs = [
        ctx.Process(
            target=_rank_worker, args=(mib, rounds, keep_base, async_save, out)
        )
        for _ in range(ranks)
    ]
    for proc in procs:
        proc.start()
    collected = [out.get() for _ in procs]
    for proc in procs:
        proc.join()
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=512, help="state size, MiB")
    parser.add_argument("--rounds", type=int, default=6, help="checkpoints per run")
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="concurrent processes, each with its own manager (the box has 8)",
    )
    args = parser.parse_args()

    if args.ranks > 1:
        # The shape the box actually has. Ravex builds Moonclip with
        # `world_size=1, rank=0` on every rank, so co-located ranks are that
        # many independent managers, each retaining its own full base.
        print(
            "state %d MiB x %d ranks, %d rounds  (peak ~%.1f GiB with base)"
            % (args.mib, args.ranks, args.rounds, args.mib * args.ranks * 3 / 1024)
        )
        for keep in (True, False):
            per_rank = run_ranks(args.ranks, args.mib, args.rounds, keep)
            print()
            print("keep_base=%s" % keep)
            for index, timings in enumerate(per_rank):
                first = timings[0][0]
                rest = [t[0] for t in timings[1:]]
                mean = sum(rest) / len(rest)
                total = sum(t[1] for t in timings[1:]) / len(rest)
                print(
                    "  rank %d  collect %.3fs -> %.3fs (%.2fx)   total stall %.3fs"
                    % (index, first, mean, mean / first if first else 0, total)
                )
        return

    print("state %d MiB, %d rounds\n" % (args.mib, args.rounds))
    # The fourth row is the one that decides it. Everything else is there to
    # be compared against: the control says what the clone costs with nothing
    # else happening, and the two async rows say that base retention is not the
    # variable.
    results = {
        "no backend": run_control(args.mib, args.rounds),
        "async, keep_base": run(args.mib, args.rounds, True, async_save=True),
        "async, no base": run(args.mib, args.rounds, False, async_save=True),
        "sync, keep_base": run(args.mib, args.rounds, True, async_save=False),
    }

    header = "  ".join("step %-2d" % (i * 2) for i in range(1, args.rounds + 1))
    for which, index in (("collect", 0), ("total stall", 1)):
        print()
        print("%-18s  %s" % (which, header))
        for label, timings in results.items():
            cells = "  ".join("%6.3fs" % t[index] for t in timings)
            print("%-18s  %s" % (label, cells))

    print()
    for label, timings in results.items():
        first, rest = timings[0], timings[1:]
        if not rest:
            continue
        c_mean = sum(t[0] for t in rest) / len(rest)
        t_mean = sum(t[1] for t in rest) / len(rest)
        print(
            "%-18s  collect %.3fs -> %.3fs (%.2fx)   total stall, mean %.3fs"
            % (label, first[0], c_mean, c_mean / first[0] if first[0] else 0, t_mean)
        )


if __name__ == "__main__":
    main()
