"""A plain PyTorch training loop. Nothing in it knows Ravex exists.

Run it, kill it with Ctrl-C part way through, then run it again:

    python examples/train_mlp.py
    # ^C somewhere around step 300
    python examples/train_mlp.py

The second run continues from the last checkpoint instead of starting over.
Ravex activates because examples/ravex.yaml sits next to this file.

Without `ravex enable`, add the explicit form:

    RAVEX_ENABLED=1 python -c "import ravex; ravex.activate()" ...

or just put `import ravex; ravex.activate()` at the top of your own script.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLES = 2048
BATCH = 32
EPOCHS = 20


def main():
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    features = torch.randn(SAMPLES, 16)
    targets = (features * torch.linspace(1, 2, 16)).sum(dim=1, keepdim=True)
    loader = DataLoader(
        TensorDataset(features, targets), batch_size=BATCH, shuffle=True
    )

    model.train()
    step = 0
    for epoch in range(EPOCHS):
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch:2d}  step {step:5d}  loss {loss.item():.5f}")

    print("done")


if __name__ == "__main__":
    main()
