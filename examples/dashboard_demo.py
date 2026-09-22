"""Runs to look at, trained on this machine's CPU in about a minute.

    python examples/dashboard_demo.py --root runs         # three runs, one resumed
    python examples/dashboard_demo.py --root runs --live watch-me   # keeps one going

Nothing here is platform-specific: it is a training loop with
``@ravex.train_loop`` and a few ``ravex.log_metrics`` calls. It exists because
a dashboard with nothing in it teaches nobody anything, and because the runs it
makes are the ones worth looking at - a good one, one whose learning rate is far
too high, and one killed between checkpoints and resumed.

One run is deliberately killed between checkpoints and restarted, because that
is the case the timeline rule exists for: what it logged after its last
checkpoint describes a model that no longer exists, and the dashboard must not
draw it.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

import ravex


def spiral(samples: int, seed: int = 0):
    """Two interleaved spirals: small, and not separable by a straight line."""
    generator = torch.Generator().manual_seed(seed)
    angle = torch.rand(samples, generator=generator) * 4 * math.pi
    radius = angle / (4 * math.pi)
    label = (torch.rand(samples, generator=generator) < 0.5).long()
    turn = angle + label * math.pi
    points = torch.stack([radius * torch.cos(turn), radius * torch.sin(turn)], dim=1)
    return points + torch.randn(samples, 2, generator=generator) * 0.03, label


def model_for(width: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(2, width),
        torch.nn.Tanh(),
        torch.nn.Linear(width, width),
        torch.nn.Tanh(),
        torch.nn.Linear(width, 2),
    )


def train(store: str, *, lr: float, width: int, steps: int, seed: int, pace: float = 0.0):
    points, labels = spiral(2048, seed)
    holdout, holdout_labels = spiral(512, seed + 1000)

    @ravex.train_loop(
        storage={"path": store},
        backend="torch_save",
        checkpoint_every=50,
        metrics_chunk_every=2,
        system_metrics_every=2,
        metrics_every=5,
    )
    def run():
        torch.manual_seed(seed)
        model = model_for(width)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        loss_fn = torch.nn.CrossEntropyLoss()
        ravex.track(model=model, optimizer=optimizer, scheduler=schedule)

        while ravex.step() < steps:
            ravex.batch_boundary()
            start = (ravex.step() * 128) % (len(points) - 128)
            batch, target = points[start : start + 128], labels[start : start + 128]
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
                            # A tensor, so it is stored as a histogram.
                            "weights/layer0": model[0].weight,
                            "weights/output": model[4].weight,
                        }
                    )
            if pace:
                time.sleep(pace)

    run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs", help="where the runs' stores go")
    parser.add_argument("--live", help="name of a run to keep training slowly, for watching")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)

    def store(name: str) -> str:
        return os.path.join(args.root, name)

    if args.live:
        print(f"training {args.live} slowly; open the dashboard and watch. Ctrl-C to stop.")
        train(store(args.live), lr=5e-3, width=96, steps=10**6, seed=7, pace=0.25)
        return

    print("lr-5e-3: a plain run")
    train(store("lr-5e-3"), lr=5e-3, width=96, steps=args.steps, seed=1)

    print("lr-5e-1: same model, a learning rate that is far too high")
    train(store("lr-5e-1"), lr=5e-1, width=96, steps=args.steps, seed=1)

    print("interrupted: killed between checkpoints, then resumed")
    try:
        train(store("interrupted"), lr=5e-3, width=96, steps=args.steps // 2 + 37, seed=2)
    except KeyboardInterrupt:
        pass
    # Kill it the way a preemption does: after the last checkpoint, with no
    # final one. `checkpoint_on_exit=False` is what makes those steps orphans.
    @ravex.train_loop(
        storage={"path": store("interrupted")},
        backend="torch_save",
        checkpoint_every=50,
        checkpoint_on_exit=False,
        metrics_chunk_every=2,
    )
    def cut_short():
        torch.manual_seed(2)
        model = model_for(96)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        ravex.track(model=model, optimizer=optimizer)
        points, labels = spiral(2048, 2)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(60):
            ravex.batch_boundary()
            logits = model(points[:128])
            loss = loss_fn(logits, labels[:128])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Diverging on purpose: these are the points that must not be drawn.
            ravex.log_metrics({"train/loss": loss * 3 + 1.0})
        raise KeyboardInterrupt

    try:
        cut_short()
    except KeyboardInterrupt:
        pass
    print("interrupted: resuming")
    train(store("interrupted"), lr=5e-3, width=96, steps=args.steps, seed=2)

    print(f"\ndone. Serve them with:  python -m gpuzero --root {args.root}")


if __name__ == "__main__":
    main()
