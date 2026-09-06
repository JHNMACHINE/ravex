"""Does the outer loop converge, and which way of averaging is right.

GPU-114. The unit tests pin the arithmetic of ``ravex._dist.outer``; this
answers the question the arithmetic cannot, which is whether a model trained by
several nodes exchanging parameter deltas every H steps actually lands
somewhere useful, and what it costs when the nodes run at different speeds.

Four things it measures, and the last two are the ones worth arguing about:

**convergence** — the loop against a single node with the same wall clock.
Not the same *compute*: N nodes take N times as many optimizer steps in the
same time, and that is the point of having them.

**the outer momentum** — with and without. It is the one thing separating this
from parameter averaging, so if it does not pay it should go.

**the combine modes under skew** — one node five times slower than the other,
which is what a cheaper box in another region looks like. ``mean``,
``step_weighted`` and ``normalized`` (see :func:`ravex._dist.outer.combine`).

**iid shards against skewed ones** — and this arm exists because the first
version of this bench could not see what it was measuring. If every node draws
from the same distribution, over-weighting the fast node costs nothing at all,
because its data says the same thing as everyone else's. The whole argument
against weighting by step count is about *bias toward the fast node's data*, so
a bench with identically distributed shards cannot observe it, whatever number
it prints. ``--skew-data`` gives each node its own target to make the bias
measurable.

    python bench/outer_loop.py --rounds 10 --inner 20
    python bench/outer_loop.py --skew-data

A tiny model on a tiny problem, so these numbers are a direction and not a
result. What they are good for is telling apart "the mechanism works" from
"the mechanism runs", which is the difference this file exists for.
"""

import argparse
import copy

import torch

from ravex._dist.outer import COMBINE_MODES, OuterLoop

BATCH = 32


def model_for(seed=3, features=4, hidden=8):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(features, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )


def problem(n=512, features=4, seed=7, tilt=0.0):
    """A linear target with noise. ``tilt`` moves this shard's target away.

    ``tilt`` is what makes a shard *different* rather than merely separate: at
    0 every node is learning the same function from different samples, which is
    the easy case and the one that hides a weighting mistake.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, features, generator=generator)
    truth = torch.randn(features, 1, generator=generator)
    if tilt:
        truth = truth + tilt * torch.randn(features, 1, generator=generator)
    y = x @ truth + 0.05 * torch.randn(n, 1, generator=generator)
    return x, y


def shards_for(nodes, skew_data, n=512):
    """One shard per node, and the whole set to score against."""
    whole_x, whole_y = problem(n=n)
    if not skew_data:
        size = n // nodes
        return (
            [(whole_x[i * size : (i + 1) * size], whole_y[i * size : (i + 1) * size])
             for i in range(nodes)],
            (whole_x, whole_y),
        )

    shards = [problem(n=n // nodes, seed=7 + i, tilt=0.6) for i in range(nodes)]
    every_x = torch.cat([s[0] for s in shards])
    every_y = torch.cat([s[1] for s in shards])
    return shards, (every_x, every_y)


def train_locally(model, x, y, steps, lr, start=0, optimizer=None):
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for step in range(steps):
        begin = ((start + step) * BATCH) % len(x)
        xb, yb = x[begin : begin + BATCH], y[begin : begin + BATCH]
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()
    return optimizer


def loss_of(model, x, y):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x), y).item()


def run(shards, rounds, inner, lr, mode="mean", momentum=0.9, speeds=None):
    """N nodes on one process. ``speeds`` scales each node's inner steps."""
    base = model_for()
    models = [copy.deepcopy(base) for _ in shards]
    loops = [
        OuterLoop(
            models[i],
            inner_steps=inner,
            combine_mode=mode,
            momentum=momentum,
            nesterov=bool(momentum),
            node=str(i),
        )
        for i in range(len(shards))
    ]
    optimizers = [None] * len(shards)
    speeds = speeds or [1.0] * len(shards)

    for round_number in range(rounds):
        contributions = []
        for index, (x, y) in enumerate(shards):
            steps = max(1, int(inner * speeds[index]))
            optimizers[index] = train_locally(
                models[index], x, y, steps, lr,
                start=round_number * steps, optimizer=optimizers[index],
            )
            for _ in range(steps):
                loops[index].record_step()
            contributions.append(loops[index].contribution())
        for loop in loops:
            loop.apply(contributions)
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--slow", type=float, default=0.2,
                        help="the last node's speed, as a fraction of the rest")
    parser.add_argument("--skew-data", action="store_true",
                        help="give each node its own target, so weighting bias "
                             "becomes visible at all")
    args = parser.parse_args()

    shards, (x, y) = shards_for(args.nodes, args.skew_data)
    even = [1.0] * args.nodes
    skewed = [1.0] * (args.nodes - 1) + [args.slow]

    print("shards: %s, %d node(s), %d rounds x %d inner steps"
          % ("skewed targets" if args.skew_data else "iid",
             args.nodes, args.rounds, args.inner))
    print()
    print("%-46s%10s" % ("", "loss"))
    print("%-46s%10.4f" % ("untrained", loss_of(model_for(), x, y)))

    alone = model_for()
    train_locally(alone, x, y, args.rounds * args.inner, args.lr)
    print("%-46s%10.4f" % ("one node, same wall clock", loss_of(alone, x, y)))
    print()

    for momentum, label in ((0.0, "no outer momentum"), (0.9, "outer momentum 0.9")):
        models = run(shards, args.rounds, args.inner, args.lr, momentum=momentum)
        print("%-46s%10.4f" % (label, loss_of(models[0], x, y)))
    print()

    for mode in COMBINE_MODES:
        for speeds, label in ((even, "every node the same speed"),
                              (skewed, "one node %gx slower" % (1 / args.slow))):
            models = run(shards, args.rounds, args.inner, args.lr,
                         mode=mode, speeds=speeds)
            print("%-16s%-30s%10.4f" % (mode, label, loss_of(models[0], x, y)))


if __name__ == "__main__":
    main()
