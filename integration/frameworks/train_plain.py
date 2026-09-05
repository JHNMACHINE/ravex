"""A plain PyTorch training script. It has never heard of DeepSpeed.

Of Ravex it has heard exactly once — the decorator on
``main`` — and nothing else here changes: no callback, no checkpoint call, no
resume logic, and the settings still come from the ``ravex.yaml`` in the working
directory. The point survives the change, and it is worth restating in the
narrower form it now has: **Ravex must not alter how this script trains.** The architecture matches what
``make_zero.py`` trains, because the checkpoint being converted is of that.

Writes its state dict out after the first iteration, which is after Ravex's
resume has fired — the resume happens at ``DataLoader.__iter__``, before any
batch is drawn.
"""

import argparse

import ravex
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@ravex.train_loop()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=97)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(1234)  # deliberately not the seed make_zero.py used
    model = nn.Sequential(
        *[
            layer
            for _ in range(args.layers)
            for layer in (nn.Linear(args.hidden, args.hidden), nn.ReLU())
        ]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    loader = DataLoader(
        TensorDataset(torch.randn(8, args.hidden)), batch_size=4
    )

    for (batch,) in loader:
        # The state is written before the step, so what lands in the file is
        # what the resume put there and not what one iteration of this script
        # did to it.
        torch.save(model.state_dict(), args.out)
        model(batch).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        break

    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
