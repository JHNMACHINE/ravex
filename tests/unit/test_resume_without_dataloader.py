"""A loop with no dataloader, resumed (GPU-275).

Until then the resume of such a loop happened in the hook after its first
``optimizer.step()``: the forward and backward of that step ran on the fresh
weights, the restored state overwrote the update, and the step was counted
anyway - so the resumed run was one batch short, and the batch it skipped was
the one its script picked with the step number, still 0 when it asked. Nothing
failed; the loss at the first resumed step was the loss at step 1.

The assertion that pins it is the strongest one there is: a run split in two
ends on exactly the weights of the same run done in one go.
"""

import torch

import ravex


def run(steps, seen=None):
    """Train to `steps`, picking each batch from the step it is for."""

    @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
    def train():
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if seen is not None:
            seen.append(ravex.step())
        while ravex.step() < steps:
            ravex.batch_boundary()
            x = torch.randn(8, 4, generator=torch.Generator().manual_seed(ravex.step()))
            optimizer.zero_grad()
            model(x).pow(2).mean().backward()
            optimizer.step()
        return [p.detach().clone() for p in model.parameters()]

    return train()


def test_a_split_run_ends_where_the_unbroken_one_does(tmp_path, monkeypatch):
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "whole"))
    whole = run(8)

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "split"))
    run(5)
    split = run(8)

    for left, right in zip(whole, split):
        assert torch.equal(left, right)


def test_the_step_says_where_the_run_resumed_before_the_first_step(storage):
    run(5)
    seen = []
    run(8, seen)
    assert seen == [5]
