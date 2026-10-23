"""What letting the link set H is worth, against fixing it, at equal wall clock.

GPU-231. ``outer_inner_steps`` was 500 because DiLoCo's paper says 500, and
``bench/outer_convergence.py`` half contradicts it: at equal tokens H=64 beat
H=512. That bench cannot see the network, though - its nodes share a process -
and on a link between continents the network is what decides how many rounds
and how many steps a run gets for its hours. ``outer_auto_steps`` moves H after
every round toward the value at which the training waits a fixed share of the
round (``ravex._dist.outer.next_inner_steps``). This asks what that does to the
model.

**The clock is simulated, the training is not.** The same byte-level
transformer and contiguous shards as ``bench/outer_convergence.py``, two nodes
each training for real on their own shard; what is simulated is the time a
step and a round's network would take on the link, ``--step-seconds`` and
``--network-seconds`` (varied by ``--jitter`` from round to round, so the rule
sees noise as a real link gives it). Every arm trains until the simulated
budget, ``--budget-seconds``, is spent, and is scored on held-out cross
entropy, nats per byte. So a small H pays for its many rounds in network time
and gets fewer steps; a large one gets more steps and fewer, older exchanges -
the trade the fixed-H benches could only see one side of.

Blocking, a round costs ``H * step + network``. With ``--overlap`` it costs
``H * step`` plus whatever of the previous round's network outlasts it, and
the previous round lands into this one ``network / step`` steps in - the
runtime's semantics (GPU-230), including the new H taking effect one round
later than blocking (``OuterLoop.set_inner_steps``).

Arms: each H in ``--h``, ``auto`` starting from ``--auto-start``, and one node
alone for the same budget, which is what a group has to beat to be worth its
network.

    python bench/round_steps_auto.py
    python bench/round_steps_auto.py --overlap
    python bench/round_steps_auto.py --network-seconds 300 --budget-seconds 20000

**Measured** - see the GPU-231 comment on Linear for the table and what it
says; the numbers depend on the simulated link, and the defaults are one
inter-continental link for a small model, not a law.
"""

import argparse
import copy
import json
import math
import random
import time

import torch

from outer_convergence import corpus_bytes, evaluate, model_for, shards_for, train

from ravex._dist.outer import (
    DEFAULT_MAX_INNER_STEPS,
    DEFAULT_MIN_INNER_STEPS,
    DEFAULT_NETWORK_SHARE,
    OuterLoop,
    next_inner_steps,
)


def network(args, rng):
    """One round's network time, with the link's jitter."""
    return args.network_seconds * (1.0 + args.jitter * rng.uniform(-1.0, 1.0))


def run(shards, args, arm, overlap):
    """Two nodes for ``--budget-seconds`` of simulated clock; the model and
    the round's record."""
    auto = arm == "auto"
    start = args.auto_start if auto else int(arm)
    base = model_for(args)
    models = [copy.deepcopy(base) for _ in shards]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=args.lr) for m in models]
    loops = [
        OuterLoop(m, inner_steps=start, lr=args.outer_lr, momentum=args.outer_momentum, node=str(i))
        for i, m in enumerate(models)
    ]
    generators = [
        torch.Generator().manual_seed(100 + index + 1000 * args.seed) for index in range(len(shards))
    ]
    rng = random.Random(args.seed)
    step = args.step_seconds

    def decide(trained_steps, network_seconds):
        if not auto:
            return
        # Every node measured the same simulated round, so the store's answer
        # is the one proposal; `decide_inner_steps` is tested on its own.
        current = loops[0].inner_steps
        decided = next_inner_steps(
            current,
            trained_seconds=trained_steps * step,
            trained_steps=trained_steps,
            network_seconds=network_seconds,
            overlap=overlap,
            share=args.share,
            low=args.low,
            high=args.high,
        )
        for loop in loops:
            loop.set_inner_steps(decided, now=not overlap)

    clock = waited = 0.0
    history = []
    flights = None
    previous = None  # (steps trained, network seconds) of the round in flight
    while True:
        h = loops[0].inner_steps
        if clock + h * step > args.budget_seconds:
            break
        if not overlap:
            for index, (shard, loop) in enumerate(zip(shards, loops)):
                train(models[index], optimizers[index], shard, h, args, generators[index])
                for _ in range(h):
                    loop.record_step()
            mine = [loop.contribution() for loop in loops]
            for loop in loops:
                loop.apply(mine)
            seconds = network(args, rng)
            clock += h * step + seconds
            waited += seconds
            history.append(h)
            decide(h, seconds)
            continue

        # Overlapped: the round in flight lands `network / step` steps into
        # this one, or at its end if the network outlasts it.
        landing = h if previous is None else min(h, int(math.ceil(previous[1] / step)))
        for index, (shard, loop) in enumerate(zip(shards, loops)):
            train(models[index], optimizers[index], shard, landing, args, generators[index])
            if flights is not None:
                loop.land(flights[index])
        if previous is not None:
            decide(*previous)
        for index, (shard, loop) in enumerate(zip(shards, loops)):
            train(models[index], optimizers[index], shard, h - landing, args, generators[index])
            for _ in range(h):
                loop.record_step()
        stall = max(0.0, previous[1] - h * step) if previous is not None else 0.0
        clock += h * step + stall
        waited += stall
        history.append(h)
        flights = [loop.launch() for loop in loops]
        mine = [flight.contribution() for flight in flights]
        for flight in flights:
            flight.apply(mine)
        previous = (h, network(args, rng))
    if flights is not None:
        # The last round lands with nothing trained after it, as at the
        # runtime's shutdown; its network is past the budget, and not counted.
        for loop, flight in zip(loops, flights):
            loop.land(flight)
    return models, {
        "rounds": len(history),
        "steps": sum(history),
        "waited": waited / clock if clock else 0.0,
        "h": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=str, nargs="+", default=["64", "500", "auto"],
                        help="fixed H values, and 'auto'")
    parser.add_argument("--auto-start", type=int, default=500)
    parser.add_argument("--share", type=float, default=DEFAULT_NETWORK_SHARE)
    parser.add_argument("--low", type=int, default=DEFAULT_MIN_INNER_STEPS)
    parser.add_argument("--high", type=int, default=DEFAULT_MAX_INNER_STEPS)
    parser.add_argument("--step-seconds", type=float, default=0.5,
                        help="simulated seconds of one local step")
    parser.add_argument("--network-seconds", type=float, default=60.0,
                        help="simulated seconds of one round's exchange")
    parser.add_argument("--jitter", type=float, default=0.2,
                        help="the network time varies by up to this fraction")
    parser.add_argument("--budget-seconds", type=float, default=2400.0)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--text", type=str, nargs="+", default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    data = corpus_bytes(args.text)
    shards, held = shards_for(data, args.nodes, args.iid)
    print(
        "%.1f KB of corpus, %d nodes, %s; a step %.2fs, a round's network %.0fs +-%.0f%%, "
        "a budget of %.0fs"
        % (len(data) / 1024, args.nodes, "overlapped" if args.overlap else "blocking",
           args.step_seconds, args.network_seconds, 100 * args.jitter, args.budget_seconds)
    )
    print("%-12s %5s %9s %7s %8s %9s  %s" % ("arm", "seed", "loss", "rounds", "steps", "waiting", "H"))
    rows = []
    for seed in args.seeds:
        args.seed = seed
        started = time.perf_counter()
        alone_steps = int(args.budget_seconds // args.step_seconds)
        model = model_for(args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(model, optimizer, torch.cat(shards), alone_steps, args,
              torch.Generator().manual_seed(100 + 1000 * seed))
        loss = evaluate(model, held, args)
        rows.append({"arm": "one node", "seed": seed, "loss": loss, "steps": alone_steps})
        print("%-12s %5d %9.4f %7s %8d %9s  (%.0fs)"
              % ("one node", seed, loss, "-", alone_steps, "-", time.perf_counter() - started),
              flush=True)
        for arm in args.h:
            started = time.perf_counter()
            models, record = run(shards, args, arm, args.overlap)
            loss = evaluate(models[0], held, args)
            path = record["h"]
            shown = path if len(path) <= 8 else path[:4] + ["..."] + path[-3:]
            rows.append({"arm": arm, "seed": seed, "loss": loss, **record})
            print("%-12s %5d %9.4f %7d %8d %8.0f%%  %s (%.0fs)"
                  % ("H=" + arm, seed, loss, record["rounds"], record["steps"],
                     100 * record["waited"], " ".join(str(h) for h in shown),
                     time.perf_counter() - started),
                  flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
