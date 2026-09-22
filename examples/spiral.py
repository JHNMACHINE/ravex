"""One run on this machine's CPU: a small MLP learning two interleaved spirals.

    RAVEX_STORAGE_PATH=runs/mine python examples/spiral.py --lr 5e-3 --steps 3000

The same model as ``dashboard_demo.py``, but one run whose store, name and
resume point come from the ``RAVEX_*`` environment instead of being written
here. That is what makes it launchable by somebody else - the platform's agent
sets ``RAVEX_STORAGE_PATH``, ``RAVEX_NAME``, ``RAVEX_RESUME_STEP`` and
``RAVEX_METRICS_ENDPOINT``, and passes the run's parameters as the flags below.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import ravex

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard_demo import model_for, spiral  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--pace",
        type=float,
        default=0.02,
        help="seconds to sleep per step, so a run lasts long enough to watch",
    )
    args = parser.parse_args()

    points, labels = spiral(2048, args.seed)
    holdout, holdout_labels = spiral(512, args.seed + 1000)

    @ravex.train_loop(
        backend="torch_save",
        checkpoint_every=100,
        keep_last=10,
        metrics_chunk_every=2,
        system_metrics_every=5,
        metrics_every=5,
    )
    def run():
        torch.manual_seed(args.seed)
        model = model_for(args.width)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
        loss_fn = torch.nn.CrossEntropyLoss()
        ravex.track(model=model, optimizer=optimizer, scheduler=schedule)

        while ravex.step() < args.steps:
            ravex.batch_boundary()
            start = (ravex.step() * args.batch) % (len(points) - args.batch)
            batch = points[start : start + args.batch]
            target = labels[start : start + args.batch]
            logits = model(batch)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            schedule.step()

            step = ravex.step()
            ravex.log_metrics(
                {
                    "train/loss": loss,
                    "train/accuracy": (logits.argmax(1) == target).float().mean(),
                    "train/grad_norm": grad_norm,
                }
            )
            if step % 25 == 0:
                with torch.no_grad():
                    predicted = model(holdout)
                    ravex.log_metrics(
                        {
                            "eval/loss": loss_fn(predicted, holdout_labels),
                            "eval/accuracy": (predicted.argmax(1) == holdout_labels).float().mean(),
                            "weights/layer0": model[0].weight,
                            "weights/output": model[4].weight,
                        }
                    )
            if step % 100 == 0:
                print("step %d  loss %.4f" % (step, loss.item()), flush=True)
            if args.pace:
                time.sleep(args.pace)

    run()


if __name__ == "__main__":
    main()
