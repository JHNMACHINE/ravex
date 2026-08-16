"""Training driven by HuggingFace `Trainer`. No mention of Ravex anywhere.

The model is a handful of Linear layers rather than anything from the hub:
`Trainer` accepts any `nn.Module` whose forward returns a loss, and pulling a
pretrained checkpoint would make this test depend on the network and on
whatever the hub serves that day.

What matters is that `Trainer` owns everything Ravex hooks into. It builds the
optimizer, it builds the dataloader (through Accelerate, which wraps it in a
subclass of its own), and it runs the loop. Ravex's own checkpointing is the
only one active: `save_strategy="no"` keeps Trainer from writing its own.
"""

import argparse
import json
import os
import signal

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

SAMPLES = 64
BATCH = 8
SEED = 0


class Rows(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(SEED)
        self.x = torch.randn(SAMPLES, 6, generator=generator)
        self.y = self.x.sum(dim=1, keepdim=True)

    def __len__(self):
        return SAMPLES

    def __getitem__(self, index):
        return {"x": self.x[index], "labels": self.y[index]}


class Model(nn.Module):
    def __init__(self, trace_path, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 12), nn.Tanh(), nn.Dropout(dropout), nn.Linear(12, 1)
        )
        self.trace = open(trace_path, "a", buffering=1)

    def forward(self, x, labels=None):
        prediction = self.net(x)
        loss = ((prediction - labels) ** 2).mean()
        if self.training:
            self.trace.write(json.dumps({"loss": repr(loss.item())}) + "\n")
        return {"loss": loss, "logits": prediction}


class SequentialTrainer(Trainer):
    """Trainer with the shuffling taken out, for the deterministic variant."""

    def _get_train_sampler(self, *args, **kwargs):
        from torch.utils.data import SequentialSampler

        return SequentialSampler(self.train_dataset)


class DieAt(TrainerCallback):
    """Stands in for the machine going away mid-run."""

    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step and state.global_step >= self.step:
            os.kill(os.getpid(), signal.SIGKILL)


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
    model = Model(args.trace, dropout=0.0 if args.no_dropout else 0.2)

    arguments = TrainingArguments(
        output_dir="./hf-output",
        per_device_train_batch_size=BATCH,
        num_train_epochs=args.epochs,
        learning_rate=5e-3,
        save_strategy="no",  # Ravex is the only checkpointer here
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=SEED,
        dataloader_num_workers=0,
        use_cpu=True,
    )

    trainer_class = SequentialTrainer if args.no_shuffle else Trainer
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=Rows(),
        callbacks=[DieAt(args.die_at)] if args.die_at else [],
    )
    trainer.train()
    print("done")


if __name__ == "__main__":
    main()
