"""Load a checkpoint written by the DDP run into a plain, unwrapped model.

The point is the key names. `DistributedDataParallel` puts every parameter
behind a `module.` prefix; `torch.compile` adds `_orig_mod.`. A checkpoint that
kept those prefixes would load fine on the cluster that wrote it and fail on a
single GPU afterwards — which is exactly when you reach for it.

Prints OK on success so a test can assert on it.
"""

import argparse
import glob
import os

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.checkpoints, "step_*.pt")))
    if not files:
        raise SystemExit("no checkpoint found")

    state = torch.load(files[-1], map_location="cpu", weights_only=False)
    models = state["models"]
    if len(models) != 1:
        raise SystemExit(f"expected exactly one model, got {list(models)}")

    state_dict = next(iter(models.values()))
    for key in state_dict:
        if key.startswith("module.") or key.startswith("_orig_mod."):
            raise SystemExit(f"wrapper prefix leaked into the checkpoint: {key}")

    # No DDP here, no process group, no torchrun.
    model = nn.Sequential(
        nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1)
    )
    model.load_state_dict(state_dict)

    print(f"OK step={state['step']} tensors={len(state_dict)}")


if __name__ == "__main__":
    main()
