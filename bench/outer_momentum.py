"""What the outer momentum costs in the first rounds, and what to default to.

GPU-192 (GPU-190 is its duplicate). Adding a "every node alone" control to
GPU-131 showed the outer loop with DiLoCo's defaults - Nesterov, momentum 0.9,
outer lr 0.7 - *losing* to one node training by itself for the first ~30
rounds, and only overtaking it later. That was one run, on two machines, with
the numbers carried over from a session nobody can open any more. This bench
asks the same question on one CPU, on several seeds, and against the
alternatives the ticket names.

**Why the first rounds.** At round 0 the momentum buffer *is* the
pseudo-gradient, so the Nesterov update is ``grad + 0.9 * grad`` and the outer
step travels 1.9 x 0.7 = 1.33 times the average delta: past where the nodes
went, not to it. The buffer then grows towards ``1 / (1 - 0.9)`` = 10 deltas.
DiLoCo runs thousands of rounds and amortises that start; a run here does tens,
and the start is most of it. The delta of the first rounds is also the
noisiest, because it is measured from an initialisation the inner optimizer is
still leaving fast.

**The task is GPU-131's teacher**, scaled to a CPU: a fixed linear map with
unit-variance targets, a stack of square Linears with no GELU at the end (the
reason is in ``integration/two-machines/remote/outer_run.py``), Adam inside,
each node its own input stream, and a held-out set nobody trains on. The
nodes run in one process and exchange through ``OuterLoop.apply`` - no
transport, because the link changes when a round's numbers arrive and not what
they are.

**The arms.** ``alone`` is one node taking the same local steps, which is the
same wall clock and the bar the group has to clear to be worth renting. A
schedule is ``M`` for a constant momentum, ``0`` for plain averaging at the
same outer lr, or ``warmM:R`` for a momentum that rises linearly from 0 to
``M`` over the first ``R`` rounds. Every arm on a seed shares the
initialisation and the data, so the comparison within a seed is paired.

**Measured**, 2026-10-15, five seeds, H=50, held-out MSE at 2000 steps minus
one node alone's on the same seed (negative is the group winning):

=========================== ========= ========= ========= =============
inner optimizer             0.9       0.7       0.5       warm0.9:10
=========================== ========= ========= ========= =============
Adam 1e-4, batch 8          **-0.25** -0.20     -0.10     -0.24
Adam 1e-3, batch 64         -0.02     -0.02     -0.02     **-0.04**
Adam 1e-3, batch 8          +0.19     -0.07     **-0.08** -0.04
=========================== ========= ========= ========= =============

**0.9 is the best arm where the inner optimizer moves slowly and the worst
where it does not**: with batch 8 at 1e-3 it loses to one node on every seed
and at every step measured, and the warmup does not rescue it at H=25, where
it loses too. 0.7 and 0.5 beat one node on every seed of every arm, at H=25
as well. GPU-131's own configuration (``--hidden 2048 --layers 5 --batch 64
--lr 1e-4``, one seed, 18 minutes) comes back digit for digit - alone 0.749,
0.9 0.810, 0.5 0.680 at 500 steps - with 0.7 the best arm at 0.664 there and
0.478 against 0.607 at 1000. Hence the default of 0.7.

    python bench/outer_momentum.py
    python bench/outer_momentum.py --h 25 50 --seeds 0 1 2 3 4 --at 500 1000 2000
"""

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn as nn

from ravex._dist.outer import OuterLoop


def make_teacher(hidden, held_out=512):
    """The same task as ``outer_run.py``: identical on every node, by seed."""
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn(hidden, hidden, generator=generator) / hidden ** 0.5

    def teacher(x):
        return x @ weight.T

    held_x = torch.randn(held_out, hidden, generator=generator)
    return teacher, held_x, teacher(held_x)


def make_model(hidden, layers, seed):
    torch.manual_seed(seed)
    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(hidden, hidden), nn.GELU()]
    return nn.Sequential(*blocks[:-1])


def heldout(model, held_x, held_y):
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(held_x), held_y))


def outer_options(schedule):
    """``OuterLoop`` keywords for a schedule: ``M``, ``0`` or ``warmM:R``."""
    if schedule.startswith("warm"):
        target, rounds = schedule[4:].split(":")
        return {"momentum": float(target), "momentum_warmup": int(rounds)}
    if float(schedule) == 0:
        # Plain averaging: Nesterov has nothing to look ahead with.
        return {"momentum": 0.0, "nesterov": False}
    return {"momentum": float(schedule)}


def run(args, seed, schedule, h, nodes):
    """Held-out loss at every step in ``args.at``, for one arm on one seed.

    ``schedule`` None means one node, alone, with no outer loop at all.
    """
    teacher, held_x, held_y = make_teacher(args.hidden)
    models = [make_model(args.hidden, args.layers, 3 + 1000 * seed) for _ in range(nodes)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in models]
    # Node i's stream is the same in every arm, so `alone` is node 0 of the
    # group running without the others.
    streams = [torch.Generator().manual_seed(11 + i + 1000 * seed) for i in range(nodes)]

    loops = None
    if schedule is not None:
        loops = [
            OuterLoop(m, inner_steps=h, lr=args.outer_lr, **outer_options(schedule))
            for m in models
        ]

    wanted = set(args.at)
    seen = {}
    for step in range(1, max(args.at) + 1):
        for model, optimizer, stream in zip(models, optimizers, streams):
            x = torch.randn(args.batch, args.hidden, generator=stream)
            loss = nn.functional.mse_loss(model(x), teacher(x))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if loops is not None:
            for loop in loops:
                loop.record_step()
            if loops[0].round_is_over():
                everyone = [loop.contribution() for loop in loops]
                for loop in loops:
                    loop.apply(everyone)
        if step in wanted:
            seen[step] = heldout(models[0], held_x, held_y)
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--h", type=int, nargs="+", default=[50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--at", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--schedules", type=str, nargs="+",
                        default=["0.9", "0.5", "warm0.9:10", "0"])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    for h in args.h:
        bad = [a for a in args.at if a % h]
        if bad:
            sys.exit("--at %s are not whole rounds of H=%d: the nodes would "
                     "hold different models there" % (bad, h))

    results = []
    for seed in args.seeds:
        started = time.time()
        alone = run(args, seed, None, None, 1)
        results.append({"seed": seed, "arm": "alone", "h": None, "loss": alone})
        for h in args.h:
            for schedule in args.schedules:
                loss = run(args, seed, schedule, h, args.nodes)
                results.append({"seed": seed, "arm": schedule, "h": h, "loss": loss})
        print("seed %d done in %.0fs" % (seed, time.time() - started), file=sys.stderr)

    # Mean over seeds, and the paired difference against `alone`: negative is
    # the group beating one node, and a row whose worst seed is negative beats
    # it on every seed.
    arms = [("alone", None)] + [(s, h) for h in args.h for s in args.schedules]
    print("held-out MSE, mean over %d seed(s); vs alone = paired mean (worst seed)"
          % len(args.seeds))
    header = "%-14s %4s" % ("arm", "H") + "".join(
        "  %8s %17s" % ("@%d" % a, "vs alone") for a in args.at)
    print(header)
    for arm, h in arms:
        rows = [r for r in results if r["arm"] == arm and r["h"] == h]
        line = "%-14s %4s" % (arm, h or "-")
        for a in args.at:
            values = [r["loss"][a] for r in rows]
            diffs = [
                r["loss"][a] - next(x["loss"][a] for x in results
                                    if x["arm"] == "alone" and x["seed"] == r["seed"])
                for r in rows
            ]
            line += "  %8.4f %+8.4f (%+.4f)" % (
                statistics.mean(values), statistics.mean(diffs), max(diffs))
        print(line)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"args": vars(args), "results": results}, handle, indent=1)


if __name__ == "__main__":
    main()
