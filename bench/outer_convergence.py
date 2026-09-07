"""What H and ``save_dtype`` cost the loss.

GPU-118, under GPU-113. Two levers make a round affordable on a slow link and
both were built without ever being paid for: **H**, the local steps per round,
and **``save_dtype``**, the cast that shrinks the delta on the wire. This is the
bill.

**Why not ``bench/outer_loop.py``.** That one answers "does the mechanism work"
on a four-feature regression, and a regression converges whatever you do to it:
it cannot tell H=20 from H=500 because there is nothing in it that drifting
apart for 500 steps would damage. Everything here is the same outer loop on a
model with a curve — a byte-level transformer over this repository's own prose,
which is a real language-modelling task, offline, and the same corpus for anyone
who has the repo checked out.

**The shards are contiguous, and that is the point.** Node *i* gets the *i*-th
block of the corpus, so the nodes see different code, different docstrings,
different vocabulary — which is what "nodes on different continents with
different data" means, and what makes drifting apart cost something. ``--iid``
interleaves them instead, and exists because a bench that cannot see the
difference between the two is a bench that cannot see the thing it measures
(the lesson ``ravex._dist.outer.combine`` was written after).

**At equal tokens seen.** Every H arm takes the same number of local steps per
node on the same stream of batches; only how often they exchange changes. Two
single-node baselines stand next to them: one that took the same steps — the
same wall clock, one machine — and one that took N times as many, which is the
same *tokens* the group saw. The group has to beat the first to be worth
anything and cannot beat the second by much.

**The quantization goes through moonclip, not through ``.to(torch.bfloat16)``.**
Each node's delta is written to a store with the ``save_dtype`` under test and
read back, which is exactly what crosses the wire, per-tensor float8 scales
included. Two extra arms:

``+ef``
    error feedback: the residual a cast throws away is carried into the next
    round instead of being lost. Standard, and it matters here because
    quantization error is **not** independent between rounds — a delta truncated
    the same way every round accumulates its error rather than averaging it out.
    Costs one extra copy of the parameters per node.

``own delta exact``
    what the code did before this issue: a node averaged its *own* delta uncast
    while its peers averaged the cast one, so no two nodes took the same outer
    step. The arm reproduces it to show what it was worth, and the spread column
    is where it shows — see ``DeltaExchange.as_published``.

**The cast arms are the slow ones.** Each writes both nodes' deltas to a
moonclip store and reads them back, every round — which is the point, since a
cast simulated in torch is not the cast that crosses the wire, but it means an
arm at H=8 pays a store round trip 256 times per node. Raise ``--dtype-min-h``
to skip the small-H cast cells if a run has to fit in a coffee break.

    python bench/outer_convergence.py
    python bench/outer_convergence.py --steps 4000 --h 1 32 256 1024

A small model on a small corpus: these are a direction, not a scaling law. What
they are good for is the shape — where the curve turns over, and whether a cast
costs a little or costs the run.

**Measured**, 2026-09-07: 696 KB of this repository, contiguous shards, 2 nodes,
0.48M parameters, 2048 local steps per node in every arm. Held-out cross
entropy, nats per byte:

======== ======== ======== ======== ==========
H        none     bf16     fp8      fp8+ef
======== ======== ======== ======== ==========
1        2.6906   —        —        —
8        1.5149   1.5100   1.5164   1.4948
64       **1.4455**  1.4452   **1.4434**  1.4442
512      1.7836   1.7836   1.7830   1.7841
======== ======== ======== ======== ==========

against ``untrained`` 5.6950, ``one node, same steps`` 1.5527, and ``one node,
2x the steps`` — the same tokens the pair saw — 1.3805.

**The cast costs nothing measurable, at any H.** Every dtype column sits inside
±0.005 of ``none``, which is smaller than the gap between neighbouring arms of
the same column; fp8 comes out marginally *ahead* at H=64, which is how you
know you are reading noise. So the 2.56x of bf16 and the 4.84x of fp8 measured
in :mod:`ravex._dist.report` are available, and the worry they were held back
for is not there — including the specific one, that a delta truncated the same
way every round accumulates its error rather than averaging it out. At H=8
there are 256 rounds to accumulate over and the fp8 column is still flat, so
**error feedback is not needed at these sizes**: ``fp8+ef`` neither helps nor
hurts beyond noise.

**H has an optimum, and both sides of it are real.** H=64 beats a single node
given the same wall clock by 7% and lands most of the way to a single node
given the same *tokens*, which is what the whole arrangement is for. H=512 is
worse than doing nothing distributed at all.

**But the H axis and the round-count axis are the same axis here, and that is a
property of the question rather than of the bench.** At equal tokens seen —
which is what makes the comparison fair — a larger H is fewer rounds: H=512 is
four exchanges in the whole run. So what H=512 shows is not "H is too large",
it is "four exchanges is too few", and a real run at H=512 for 200k steps gets
390 of them. The ceiling this can honestly report is on **rounds**, not on H.

**H=1 is the surprise, and it is mostly the outer learning rate.** 2.69 against
1.45 at H=64. An outer step at every inner step compounds a Nesterov buffer at
``outer_lr=0.7`` 2048 times and overwrites the inner AdamW's progress each time,
and those defaults are DiLoCo's, calibrated for H in the hundreds. Re-run at
``--outer-lr 0.1`` it comes back to 1.79 — most of the gap — **and still loses
to the single node in the same run** (1.60). So the conclusion survives the
retune, and it is worth knowing because "set H low and it degrades gracefully
toward synchronous training" is the natural assumption and it is false: there
is no small-H limit where this becomes ordinary data parallelism.

**What the ``own delta exact`` arm cost, and where it did not show.** At H=64
with fp8: loss 1.4100 against 1.4104 with the fix — indistinguishable, and both
perfectly healthy models. The ``spread`` column is the whole story:
**3.42e-02** against 0.00e+00. Nothing about the loss curve, the round reports,
or the models themselves would tell you that the two nodes had stopped training
one model; only the invariant does. Which is why the test that guards it uses
``torch.equal`` and not ``allclose``.

**One caveat that is this bench's own fault.** The corpus is the repository, so
editing the repository changes the corpus — three runs in one afternoon read
696, 705 and 710 KB, and their baselines moved by 0.05 with it. Arms *within* a
run are comparable because the corpus is loaded once; numbers from different
runs are not, and every comparison above is within a run. ``--text`` pins a
corpus if that matters.
"""

import argparse
import copy
import glob
import json
import os
import shutil
import tempfile
import time

import torch

from ravex._dist import report as _report
from ravex._dist.outer import Contribution, OuterLoop

#: The corpus, relative to the repository root. Prose-heavy Python and the
#: docs: this project writes long docstrings, which is what makes its own
#: source a language corpus rather than a token soup.
CORPUS = ("ravex/**/*.py", "docs/*.md", "README.md", "CHANGELOG.md")

VOCAB = 256  # bytes, so there is no tokenizer to agree about


def corpus_bytes(paths=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if paths:
        blobs = [open(path, "rb").read() for path in paths]
    else:
        files = []
        for pattern in CORPUS:
            files += sorted(glob.glob(os.path.join(root, pattern), recursive=True))
        blobs = [
            open(path, "rb").read()
            for path in files
            if "__pycache__" not in path and ".venv" not in path
        ]
    data = b"\n".join(blobs)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def shards_for(data, nodes, iid, holdout=0.1):
    """One shard per node, and a held-out slice that spans every node's data.

    Contiguous by default: node *i* gets block *i*, so the shards differ in
    what they are about and not only in which samples they drew. The held-out
    part is taken from the tail of each shard, so the score is over everybody's
    distribution rather than over the luckiest node's.
    """
    if iid:
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(11))
        data = data[order]

    size = len(data) // nodes
    train, held = [], []
    for index in range(nodes):
        block = data[index * size : (index + 1) * size]
        cut = int(len(block) * (1 - holdout))
        train.append(block[:cut])
        held.append(block[cut:])
    return train, torch.cat(held)


class Block(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = torch.nn.MultiheadAttention(
            dim, heads, batch_first=True, bias=True
        )
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * dim, dim),
        )

    def forward(self, x, mask):
        h = self.norm1(x)
        attended, _ = self.attention(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class Model(torch.nn.Module):
    """A decoder-only transformer over bytes. Small, and a real one."""

    def __init__(self, dim=128, depth=2, heads=4, block=128):
        super().__init__()
        self.block = block
        self.embed = torch.nn.Embedding(VOCAB, dim)
        self.position = torch.nn.Embedding(block, dim)
        self.blocks = torch.nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, VOCAB)
        # Registered rather than rebuilt per forward, and non-persistent so it
        # is not a buffer the outer loop has to have an opinion about.
        self.register_buffer(
            "mask",
            torch.triu(torch.full((block, block), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.position(positions)
        mask = self.mask[:length, :length]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


def model_for(args, seed=5):
    torch.manual_seed(seed)
    return Model(dim=args.dim, depth=args.depth, heads=args.heads, block=args.block)


def batches(data, count, block, generator):
    starts = torch.randint(
        0, len(data) - block - 1, (count,), generator=generator
    )
    x = torch.stack([data[start : start + block] for start in starts])
    y = torch.stack([data[start + 1 : start + block + 1] for start in starts])
    return x, y


def train(model, optimizer, data, steps, args, generator):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        x, y = batches(data, args.batch, args.block, generator)
        optimizer.zero_grad()
        logits = model(x)
        loss_fn(logits.reshape(-1, VOCAB), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate(model, held, args):
    """Cross entropy on the held-out slice, in nats per byte."""
    generator = torch.Generator().manual_seed(1234)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = batches(held, args.batch, args.block, generator)
            total += loss_fn(
                model(x).reshape(-1, VOCAB), y.reshape(-1)
            ).item()
    return total / args.eval_batches


class Wire:
    """A node's delta as its peers receive it: through the real store, or not.

    ``dtype`` None is the identity and does no I/O — the arm where nothing is
    cast should not be paying for a disk round trip that the product would not
    make either.
    """

    def __init__(self, root, nodes, dtype, feedback):
        self.dtype = dtype
        self.feedback = feedback
        self.stores = (
            [
                _report.open_store(os.path.join(root, "n%d" % index),
                                   save_dtype=dtype)
                for index in range(nodes)
            ]
            if dtype
            else None
        )
        self.residuals = [{} for _ in range(nodes)] if feedback else None

    def send(self, delta, node, round_number):
        if not self.dtype:
            return delta

        outgoing = dict(delta)
        if self.residuals is not None:
            carried = self.residuals[node]
            outgoing = {
                name: value + carried[name] if name in carried else value
                for name, value in outgoing.items()
            }

        store = self.stores[node]
        _report.write(store, outgoing, round_number, 0, str(node))
        received = _report.read(
            store, round_number, _report.expectation(outgoing)
        ).delta

        if self.residuals is not None:
            # What the cast threw away, to be added to the next round's delta.
            # Without this the same corner of every tensor is truncated the
            # same way every round, and the error accumulates instead of
            # averaging out.
            self.residuals[node] = {
                name: outgoing[name] - received[name] for name in outgoing
            }
        return received


def spread(models):
    """The largest gap between any two nodes' parameters.

    Zero is the only correct answer: every node applies the same combined
    pseudo-gradient to the same outer parameters. Anything else is nodes
    training models that are quietly not the same one.
    """
    if len(models) < 2:
        return 0.0
    first = dict(models[0].named_parameters())
    worst = 0.0
    for other in models[1:]:
        for name, value in other.named_parameters():
            worst = max(worst, (first[name] - value).abs().max().item())
    return worst


def run(shards, args, inner, dtype=None, feedback=False, own="published"):
    """N nodes on one process, exchanging every ``inner`` steps."""
    base = model_for(args)
    models = [copy.deepcopy(base) for _ in shards]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=args.lr) for model in models
    ]
    loops = [
        OuterLoop(
            models[index],
            inner_steps=inner,
            lr=args.outer_lr,
            momentum=args.outer_momentum,
            node=str(index),
        )
        for index in range(len(shards))
    ]
    generators = [
        torch.Generator().manual_seed(100 + index) for index in range(len(shards))
    ]

    root = tempfile.mkdtemp(prefix="ravex-convergence-")
    wire = Wire(root, len(shards), dtype, feedback)
    try:
        for round_number in range(max(1, args.steps // inner)):
            mine = []
            for index, shard in enumerate(shards):
                train(models[index], optimizers[index], shard, inner, args,
                      generators[index])
                for _ in range(inner):
                    loops[index].record_step()
                mine.append(loops[index].contribution())

            on_the_wire = [
                wire.send(contribution.delta, index, round_number)
                for index, contribution in enumerate(mine)
            ]
            for index, loop in enumerate(loops):
                loop.apply(
                    [
                        Contribution(
                            # `own="exact"` is the defect: this node averages
                            # the delta it computed, everyone else averages the
                            # cast one, and no two nodes take the same step.
                            delta=(
                                mine[other].delta
                                if own == "exact" and other == index
                                else on_the_wire[other]
                            ),
                            steps=mine[other].steps,
                            node=mine[other].node,
                        )
                        for other in range(len(mine))
                    ]
                )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return models


def alone(shard, args, steps):
    """One node, one model, no rounds."""
    model = model_for(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, optimizer, shard, steps, args, torch.Generator().manual_seed(100))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200,
                        help="local steps per node, the same for every arm")
    parser.add_argument("--h", type=int, nargs="+", default=[1, 8, 64, 512],
                        metavar="H", help="inner steps per round to walk")
    parser.add_argument("--dtypes", type=str, nargs="+",
                        default=["none", "bf16", "fp8", "fp8+ef"],
                        help="save_dtype arms; +ef adds error feedback")
    parser.add_argument("--dtype-min-h", type=int, default=8,
                        help="skip cast arms below this H: they are a disk "
                             "round trip per round, not a finding")
    parser.add_argument("--iid", action="store_true",
                        help="interleave the shards instead of splitting the "
                             "corpus into blocks")
    parser.add_argument("--own-delta", type=str, default="published",
                        choices=("published", "exact", "both"),
                        help="whether a node averages its own delta as its "
                             "peers see it; `both` measures the difference")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--text", type=str, nargs="+", default=None,
                        help="files to use as the corpus instead of the repo")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    data = corpus_bytes(args.text)
    shards, held = shards_for(data, args.nodes, args.iid)
    parameters = sum(p.numel() for p in model_for(args).parameters())

    print("%.1f KB of corpus, %s shards, %d node(s), %.2fM parameters"
          % (len(data) / 1024, "iid" if args.iid else "contiguous",
             args.nodes, parameters / 1e6))
    print("%d local steps per node in every arm, %d x %d tokens per step"
          % (args.steps, args.batch, args.block))
    print(flush=True)

    rows = []

    def report(label, model, models, seconds):
        loss = evaluate(model, held, args)
        gap = spread(models) if models else 0.0
        rows.append({"arm": label, "loss": loss, "spread": gap,
                     "seconds": seconds})
        # Flushed per arm: the cast arms take minutes each, and a run whose
        # output only appears at the end is a run you cannot tell from a hang.
        print("%-34s %9.4f %11.2e %8.0fs" % (label, loss, gap, seconds),
              flush=True)

    print("%-34s %9s %11s %9s" % ("", "loss", "spread", "took"))

    started = time.perf_counter()
    report("untrained", model_for(args), None, 0.0)

    started = time.perf_counter()
    model = alone(torch.cat(shards), args, args.steps)
    report("one node, same steps", model, None, time.perf_counter() - started)

    started = time.perf_counter()
    model = alone(torch.cat(shards), args, args.steps * args.nodes)
    report("one node, %dx the steps" % args.nodes, model, None,
           time.perf_counter() - started)
    print()

    owns = ("published", "exact") if args.own_delta == "both" else (args.own_delta,)
    for inner in args.h:
        for dtype in args.dtypes:
            cast = None if dtype == "none" else dtype.split("+")[0]
            feedback = dtype.endswith("+ef")
            if cast and inner < args.dtype_min_h:
                continue
            for own in owns if cast else ("published",):
                label = "H=%-5d %s" % (inner, dtype)
                if own == "exact":
                    label += " (own delta exact)"
                started = time.perf_counter()
                models = run(shards, args, inner, dtype=cast,
                             feedback=feedback, own=own)
                report(label, models[0], models, time.perf_counter() - started)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
