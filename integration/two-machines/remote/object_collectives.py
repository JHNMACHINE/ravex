"""The six agreements, across two machines, on whichever object path is live.

Every place two ranks have to reach the same answer before a resume goes
through now runs on ``_all_gather_object``: the step to resume from, which
stores each machine can see, the run id, whether the storage is shared,
whether every rank succeeded. That function has two implementations and picks
the NumPy-free one only when torch cannot reach NumPy — which no ordinary
image reproduces, so ``RAVEX_ASSUME_NO_NUMPY=1`` is what makes it reachable.

Run under torchrun with two nodes. What it asserts is what one box cannot
answer honestly:

* the ranks converge on the *same* value, having started from different ones;
* ``storage_is_shared`` says **no** — two machines with local disks do not
  share a filesystem, and on one box the same call says yes, which is exactly
  why a green single-box bench proves nothing here;
* whichever object path each rank took, they still meet.
"""

import os
import socket
import sys

import torch.distributed as dist


def main() -> int:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    from ravex import _distributed as D
    from ravex._identity import FROM_STORE

    D._torch_numpy = None
    numpy_free = not D._torch_can_reach_numpy()
    host = socket.gethostname()

    def say(*parts):
        print(f"[rank{rank}]", *parts, flush=True)

    say(f"host={host} world={world} numpy_free_path={numpy_free}")

    failures = []

    def check(name, got, want):
        ok = got == want
        say(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" != {want!r}"))
        if not ok:
            failures.append(name)

    # Every rank arrives with a different answer; they must leave with one.
    check("agree_on_step", D.agree_on_step(100 + rank * 7), 100)
    check("all_ranks_agree(True)", D.all_ranks_agree(True), True)
    check("all_ranks_agree(mixed)", D.all_ranks_agree(rank == 0), False)
    check("agree_on_run_id", D.agree_on_run_id(FROM_STORE, "run-abc"), "run-abc")

    # Payloads of very different sizes: with equal ones the padding in the
    # NumPy-free path never has to do anything.
    payload = {"rank": rank, "host": host, "blob": "x" * (1 + rank * 4000)}
    gathered = D.gather_objects(payload)
    check("gather_objects/len", len(gathered), world)
    check("gather_objects/order", [g["rank"] for g in gathered], list(range(world)))
    hosts = sorted({g["host"] for g in gathered})
    check("gather_objects/two hosts", len(hosts), world)
    check(
        "gather_objects/sizes",
        [len(g["blob"]) for g in gathered],
        [1 + r * 4000 for r in range(world)],
    )

    # The assertion that only two machines can make. Each box has its own
    # local disk; on a single box this same call answers True.
    shared = D.storage_is_shared(os.environ.get("KIT_ROOT", "/root") + "/run")
    check("storage_is_shared (must be False on two boxes)", shared, False)

    every = D.gather_objects(numpy_free)
    say(f"paths across the job: {every}")

    verdict = D.all_ranks_agree(not failures)
    if rank == 0:
        print(
            f"\n=== object collectives: {'PASS' if verdict else 'FAIL'} "
            f"(paths={every}) ===",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
