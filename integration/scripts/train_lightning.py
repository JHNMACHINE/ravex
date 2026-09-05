"""Training driven by Lightning. No mention of Ravex anywhere.

Lightning owns the loop, builds the optimizer through `configure_optimizers`,
and calls `optimizer.step()` through a `LightningOptimizer` wrapper rather than
directly. Its own checkpointing is switched off so Ravex is the only one
saving anything.
"""

import argparse
import json
import os
import signal

import lightning as L
import ravex
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLES = 64
BATCH = 8
SEED = 0


class Regressor(L.LightningModule):
    def __init__(self, trace_path, die_at, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 12), nn.Tanh(), nn.Dropout(dropout), nn.Linear(12, 1)
        )
        self.trace = open(trace_path, "a", buffering=1)
        self.die_at = die_at
        self.seen = 0

    def training_step(self, batch, batch_index):
        x, y = batch
        loss = ((self.net(x) - y) ** 2).mean()
        self.trace.write(json.dumps({"loss": repr(loss.item())}) + "\n")
        return loss

    def on_train_batch_end(self, outputs, batch, batch_index):
        self.seen += 1
        if self.die_at and self.seen >= self.die_at:
            self.trace.flush()
            os.fsync(self.trace.fileno())
            os.kill(os.getpid(), signal.SIGKILL)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-3)


@ravex.train_loop()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--die-at", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    # Strips the two sources of per-step randomness, to separate "did the state
    # come back" from "did the replayed data and RNG line up".
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-dropout", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    generator = torch.Generator().manual_seed(SEED)
    features = torch.randn(SAMPLES, 6, generator=generator)
    targets = features.sum(dim=1, keepdim=True)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=BATCH,
        shuffle=not args.no_shuffle,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,  # Ravex is the only checkpointer here
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model = Regressor(args.trace, args.die_at, dropout=0.0 if args.no_dropout else 0.2)
    trainer.fit(model, loader)
    print("done")


if __name__ == "__main__":
    main()
