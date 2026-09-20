"""What resuming from a cast checkpoint costs the loss.

GPU-58 gave `save_dtype` a target per component — model in fp32, optimizer in
bf16, and so on — and `tests/test_backends.py` proves the mechanism: the cast
reaches both the plain and the sharded copy, the components left alone do not
move a bit, and what moves moves by no more than bf16 rounding. What none of
that says is **what it costs**, because the tensors are compared where they
land, not where they are used.

The interesting one is the optimizer, and it is interesting for a specific
reason. Adam divides the step by `sqrt(exp_avg_sq)`, so a relative error in the
second moment is a relative error in every step size that follows — and bf16
keeps eight bits of mantissa, about two decimal digits. Whether that is
harmless or expensive is not derivable from the rounding error of one tensor;
it is a question about a training run, and the only honest answer is a run.

It is also not an academic setting. On a link at 7 MB/s the first lever anyone
reaches for to shrink a checkpoint is the optimizer, because it is the biggest
part of one — Adam carries two moments per parameter, so it is twice the model.

**The comparison.** Every arm trains `--before` steps, is checkpointed through
the real Moonclip backend with the `save_dtype` under test, is reloaded into a
*fresh* model and optimizer, and trains `--after` more. The control never
checkpoints at all. The two phases draw their batches from generators seeded
the same way in every arm, so the batch stream is identical everywhere and the
only difference left is what the round trip through the store threw away.

`reload fp32` is the arm that matters most and looks like it says nothing: it
is the same round trip with no cast at all. If it does not land on the control,
the harness is measuring itself and no other row can be believed.

    python bench/resume_precision.py
    python bench/resume_precision.py --before 800 --after 800 \
        --dtypes none optimizer:bf16 optimizer:fp8 model:bf16

**Measured**, 2026-09-20: 0.48M parameters, 600 steps then 600 more, AdamW at
1e-3, one seed. `weights` and `exp_avg_sq` are the worst relative error a
tensor came back with; `flushed` is the share of second-moment entries that
were nonzero and returned as zero.

================================ ======== ========= ========== =========
arm                               loss     weights  exp_avg_sq  flushed
================================ ======== ========= ========== =========
never checkpointed                1.8964     n/a       n/a        n/a
reload fp32                       1.8964   0.00e+00  0.00e+00    0.00%
reload optimizer:bf16             1.8964   0.00e+00  3.89e-03    0.00%
reload optimizer:fp8              1.8965   0.00e+00  1.00e+00    0.71%
reload model:bf16                 1.8964   3.89e-03  0.00e+00    0.00%
reload optimizer:bf16,model:bf16  1.8964   3.89e-03  3.89e-03    0.00%
================================ ======== ========= ========== =========

**Every cast lands where the config says and nowhere else**, which is GPU-58's
claim checked at the consequence rather than at the tensor. **And none of them
costs the loss**, fp8 on the optimizer included. bf16 comes back at 3.89e-03
everywhere, which is 2 ** -8 — eight bits of mantissa, the constant the theory
predicts, arrived at without being told.

The honest limit is the second phase: 600 steps of training after the resume
absorb the damage, and Adam rebuilds the second moment it was handed. A run
that resumes and is evaluated at once, or that ends soon after, is a case this
does not cover.

A small model on a small corpus: a direction, not a scaling law. What it is
good for is the shape — whether a cast costs nothing, a little, or the run.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The corpus, the model and the evaluation come from the convergence bench on
# purpose: same byte-level transformer over this repository's own prose, same
# held-out slice, so a number here can be read next to a number there instead
# of being a second toy with its own scale.
from outer_convergence import (  # noqa: E402
    corpus_bytes,
    evaluate,
    model_for,
    shards_for,
    train,
)

from ravex._config import RavexConfig  # noqa: E402


def optimizer_for(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr)


def phase(model, optimizer, data, steps, args, seed):
    """Train `steps` on a batch stream that is the same in every arm."""
    train(model, optimizer, data, steps, args, torch.Generator().manual_seed(seed))


def as_state(model, optimizer, step):
    """The layout Ravex's backends write, with nothing else in it."""
    return {
        "ravex_version": 1,
        "step": step,
        "models": {"model": model.state_dict()},
        "optimizers": {"optimizer": optimizer.state_dict()},
        "schedulers": {},
        "scalers": {},
        "dataloaders": {},
        "sharded": {},
    }


def parse_dtype(text):
    """`none`, `bf16`, or `optimizer:bf16,model:none` as the config takes it."""
    if text == "none":
        return None
    if ":" not in text:
        return text
    rules = {}
    for clause in text.split(","):
        key, _, dtype = clause.partition(":")
        rules[key.strip()] = dtype.strip()
    return rules


def round_trip(model, optimizer, args, save_dtype, step):
    """Save through Moonclip with `save_dtype`, load into fresh objects.

    Fresh on purpose. Loading back into the same model and optimizer would
    leave every tensor the cast damaged still sitting in memory at full
    precision, and the arm would measure nothing at all — which is the shape of
    mistake this whole bench exists to catch elsewhere.
    """
    from ravex._backends import MoonclipBackend

    root = tempfile.mkdtemp(prefix="ravex-resume-precision-")
    try:
        config = RavexConfig()
        config.storage.path = os.path.join(root, "checkpoints")
        config.backend = "moonclip"
        config.async_save = False
        if save_dtype is not None:
            config.save_dtype = save_dtype
        config._normalize()
        if config.problems:
            raise SystemExit("bad config: %s" % (config.problems,))

        backend = MoonclipBackend(config)
        backend.save(step, as_state(model, optimizer, step), {})
        backend.flush()
        backend.close()

        got = MoonclipBackend(config).load_latest()
        if got is None:
            raise SystemExit("nothing came back out of the store")

        restored = model_for(args)
        restored.load_state_dict(got["models"]["model"])
        restored_optimizer = optimizer_for(restored, args)
        restored_optimizer.load_state_dict(got["optimizers"]["optimizer"])
        return restored, restored_optimizer
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def second_moment_error(before, after):
    """How far Adam's `exp_avg_sq` moved: the worst entry, and how many died.

    Relative and not absolute because that is what reaches the step: the update
    divides by its square root, so what matters is the fraction by which an
    entry is wrong, not its size.

    **Two numbers, because one of them lies.** The worst relative error reads
    1.00 for fp8, which looks like the second moment being destroyed and is
    not: Moonclip scales float8 per tensor (GPU-91), so the range is fine and
    what suffers is the dynamic range *inside* a tensor. Measured on this
    model, fp8 leaves the largest entry untouched to four figures and flushes
    about 1.8% of the smallest ones to zero — and an entry that was small and
    becomes zero scores a relative error of exactly 1, one entry or all of
    them. `flushed` is the number that separates those two worlds.
    """
    worst = 0.0
    died = 0
    total = 0
    for group_before, group_after in zip(
        before.state_dict()["state"].values(), after.state_dict()["state"].values()
    ):
        a = group_before.get("exp_avg_sq")
        b = group_after.get("exp_avg_sq")
        if a is None or b is None:
            continue
        scale = a.abs().clamp_min(1e-12)
        worst = max(worst, ((a - b).abs() / scale).max().item())
        died += int(((a != 0) & (b == 0)).sum())
        total += a.numel()
    return worst, (died / total if total else 0.0)


def model_error(before, after):
    """How far the weights moved, relative to themselves.

    Without this column an arm that casts the model reads exactly like an arm
    that casts nothing: `exp_avg_sq` is untouched either way, and the loss
    after `--after` steps of training is the same to four decimals. Which is a
    finding *or* a bug in this file, and there is no way to tell them apart
    from the two columns that were here first.
    """
    worst = 0.0
    a = before.state_dict()
    b = after.state_dict()
    for name, value in a.items():
        other = b.get(name)
        if other is None or not torch.is_floating_point(value):
            continue
        scale = value.abs().clamp_min(1e-12)
        worst = max(worst, ((value - other).abs() / scale).max().item())
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=int, default=600,
                        help="steps before the checkpoint")
    parser.add_argument("--after", type=int, default=600,
                        help="steps after resuming from it")
    parser.add_argument("--dtypes", type=str, nargs="+",
                        default=["none", "optimizer:bf16", "optimizer:fp8",
                                 "model:bf16", "optimizer:bf16,model:bf16"],
                        help="save_dtype arms, in the config's own spelling")
    parser.add_argument("--lr", type=float, default=1e-3)
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
    (shard,), held = shards_for(data, 1, iid=False)
    parameters = sum(p.numel() for p in model_for(args).parameters())

    print("%.1f KB of corpus, %.2fM parameters, %d steps then %d more"
          % (len(data) / 1024, parameters / 1e6, args.before, args.after))
    print("%d x %d tokens per step, AdamW at lr=%g"
          % (args.batch, args.block, args.lr))
    print(flush=True)
    print("%-32s %9s %11s %11s %9s %8s"
          % ("", "loss", "weights", "exp_avg_sq", "flushed", "took"))

    rows = []

    def report(label, loss, weights, moved, flushed, seconds):
        rows.append({"arm": label, "loss": loss, "weight_error": weights,
                     "exp_avg_sq_error": moved, "exp_avg_sq_flushed": flushed,
                     "seconds": seconds})
        # Plain ASCII: this bench is meant to be run over ssh on a rented box,
        # and a console that cannot encode an em dash is a traceback in the
        # middle of a measurement.
        show = lambda v: "n/a" if v is None else "%.2e" % v
        print("%-32s %9.4f %11s %11s %8s %7.0fs"
              % (label, loss, show(weights), show(moved),
                 "n/a" if flushed is None else "%.2f%%" % (100 * flushed),
                 seconds), flush=True)

    # The control: one run, no store in the middle, same two batch streams.
    started = time.perf_counter()
    model = model_for(args)
    optimizer = optimizer_for(model, args)
    phase(model, optimizer, shard, args.before, args, seed=11)
    midpoint = copy.deepcopy(model), copy.deepcopy(optimizer)
    phase(model, optimizer, shard, args.after, args, seed=22)
    report("never checkpointed", evaluate(model, held, args), None, None, None,
           time.perf_counter() - started)
    print()

    for text in args.dtypes:
        started = time.perf_counter()
        # Every arm restarts from the *same* midpoint, so the phase before the
        # checkpoint is trained once rather than once per arm - and, more to
        # the point, no arm can differ from another because of it.
        model, optimizer = copy.deepcopy(midpoint[0]), copy.deepcopy(midpoint[1])
        restored, restored_optimizer = round_trip(
            model, optimizer, args, parse_dtype(text), args.before
        )
        moved, flushed = second_moment_error(optimizer, restored_optimizer)
        # Measured before the second phase, which would train the difference
        # away: what is wanted is what came back out of the store, not what
        # six hundred more steps made of it.
        weights = model_error(model, restored)
        phase(restored, restored_optimizer, shard, args.after, args, seed=22)
        label = "reload fp32" if text == "none" else "reload %s" % text
        report(label, evaluate(restored, held, args), weights, moved, flushed,
               time.perf_counter() - started)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
