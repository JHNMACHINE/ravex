"""Measured rates, turned into an answer about specific reshards — GPU-96.

A bandwidth number on its own does not decide anything here. What decides it is
that number divided into *the bytes a particular reshard actually moves*, and
that quantity is far smaller than a checkpoint and is often exactly zero. The
arithmetic comes from :mod:`ravex._reshard`, tested exhaustively and offline in
``tests/test_reshard_locality.py``; this file only pairs it with what the link
between two real machines was measured to carry.

One simplification, stated rather than hidden: every tensor is treated as
having the same row width, so a scenario's cost is a *fraction* of the
checkpoint. That is close to exact for what is in scope — a 1-D FSDP mesh
sharded on dim 0, where the fraction depends on where the machine boundaries
fall and not on how wide a row is — and it would stop being true for a 2-D
mesh, which :func:`ravex._reshard.shard_dim` refuses by name anyway.
"""

from __future__ import annotations

import argparse
import json

from ravex._reshard import contiguous_homes, crossing_rows, plan_reshard


#: The reshards worth pricing, as ``(label, old_world, new_world, old_home,
#: new_home)``. Homes are functions of the worlds so the table stays readable.
#: The three families are deliberately different in kind: a shrink whose
#: boundaries line up, one whose boundaries do not, and a machine that is gone.
def scenarios():
    yield (
        "8 -> 4, two machines, both halves kept",
        8, 4,
        lambda: contiguous_homes(8, 2),
        lambda: contiguous_homes(4, 2),
    )
    yield (
        "8 -> 6, two machines (uneven: 3 and 3 onto 4 and 2)",
        8, 6,
        lambda: contiguous_homes(8, 2),
        lambda: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1},
    )
    yield (
        "8 -> 5, two machines (odd new world, boundary off-centre)",
        8, 5,
        lambda: contiguous_homes(8, 2),
        lambda: contiguous_homes(5, 2),
    )
    yield (
        "6 -> 4, one of three machines lost, its shards read from a survivor",
        6, 4,
        lambda: {0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 2},
        lambda: {0: 0, 1: 0, 2: 2, 3: 2},
    )
    yield (
        "8 -> 4 onto one surviving machine holding complete copies",
        8, 4,
        lambda: {q: 0 for q in range(8)},
        lambda: {r: 0 for r in range(4)},
    )


def fraction_crossing(old_world: int, new_world: int, old_home, new_home) -> float:
    """Of the whole checkpoint, how much has to cross a machine boundary."""
    extent = old_world * new_world
    plan = plan_reshard([new_world] * old_world, [old_world] * new_world)
    crossing = crossing_rows(plan, old_home(), new_home())
    return sum(crossing.values()) / float(extent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", required=True, help="gpu96.transport.json")
    parser.add_argument(
        "--checkpoint-gb",
        type=float,
        default=0.0,
        help="the whole checkpoint across all ranks; defaults to the measured "
             "store times the old world of each scenario",
    )
    args = parser.parse_args()

    with open(args.measured, encoding="utf-8") as handle:
        m = json.load(handle)

    store = m.get("store_bytes", 0)
    p2p_rate = store / m["p2p_seconds"] if m.get("p2p_seconds") else 0.0
    remote_bytes = m.get("remote_bytes", store)
    # Kept apart. Object storage is rarely symmetric, and a single round-trip
    # average would hide which direction is the expensive one - which matters,
    # because the two ways of using storage below lean on different directions.
    up_rate = remote_bytes / m["upload_seconds"] if m.get("upload_seconds") else 0.0
    down_rate = remote_bytes / m["download_seconds"] if m.get("download_seconds") else 0.0

    print("\n  measured on this pair of machines")
    print("    point to point   %s" % _rate(p2p_rate))
    if up_rate and down_rate:
        print("    storage up       %s" % _rate(up_rate))
        print("    storage down     %s" % _rate(down_rate))
    else:
        print("    object storage   not measured - the comparison is incomplete")

    print("\n  what that costs, per reshard")
    for label, old_world, new_world, old_home, new_home in scenarios():
        share = fraction_crossing(old_world, new_world, old_home, new_home)
        whole = args.checkpoint_gb * 1e9 if args.checkpoint_gb else store * old_world
        moving = share * whole

        if moving == 0:
            print("    %-68s nothing crosses" % label)
            continue

        # How many machines can work at once on each side. Uploading is done
        # by whoever holds an old store, downloading by whoever runs a new
        # rank, and both happen concurrently - a serial figure would flatter
        # the point-to-point option for a reason that has nothing to do with
        # the choice being made.
        senders = len(set(old_home().values()))
        receivers = len(set(new_home().values()))

        line = "    %-68s %5.1f%% = %.1f GB" % (label, 100 * share, moving / 1e9)
        if p2p_rate:
            line += "\n      %-66s %s" % ("point to point", _seconds(moving / p2p_rate))
        if up_rate and down_rate:
            # Two honest versions of the alternative, and they are far apart.
            #
            # Consolidating *at reshard time* is what the issue names: every
            # machine pushes its stores to the bucket, then every machine
            # pulls what it is missing. The whole checkpoint goes up, not just
            # the part that crosses, which is why this can lose badly even on
            # a fast link to storage.
            consolidate = (whole / senders) / up_rate + (moving / receivers) / down_rate
            # Writing there *all along* is the other one, and it is not a
            # reshard cost at all: the uploads already happened as ordinary
            # checkpoint writes during training, `storage.is_remote` makes
            # every rank see every store, and the refusal this issue is about
            # never fires. What remains is reading what this rank needs.
            already_there = (moving / receivers) / down_rate
            line += "\n      %-66s %s" % (
                "consolidate on storage first, then read", _seconds(consolidate))
            line += "\n      %-66s %s" % (
                "storage from the start (the reshard reads only)",
                _seconds(already_there))
        print(line)

    print("\n  the last two lines are the ones GPU-96 is about: a lost machine "
          "whose\n  shards are readable somewhere else. The first three are the "
          "cases a\n  transport would be written for and mostly never used on.")


def _rate(rate: float) -> str:
    return "%.1f MB/s" % (rate / 1e6) if rate else "n/a"


def _seconds(seconds: float) -> str:
    if seconds < 90:
        return "%.0f s" % seconds
    return "%.0f min" % (seconds / 60)


if __name__ == "__main__":
    main()
