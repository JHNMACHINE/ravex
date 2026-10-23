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

import pytest
import torch

import ravex
from ravex._runtime import get_runtime


def run(steps, seen=None, hand_over=True):
    """Train to `steps`, picking each batch from the step it is for.

    `hand_over=False` leaves the top of the iteration to the next forward of
    the model, which Ravex takes on its own (GPU-276).
    """

    @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
    def train():
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if seen is not None:
            seen.append(ravex.step())
        while ravex.step() < steps:
            if hand_over:
                ravex.batch_boundary()
            x = torch.randn(8, 4, generator=torch.Generator().manual_seed(ravex.step()))
            optimizer.zero_grad()
            model(x).pow(2).mean().backward()
            optimizer.step()
        return [p.detach().clone() for p in model.parameters()]

    return train()


@pytest.mark.parametrize("hand_over", [True, False], ids=["by hand", "from the forward"])
def test_a_split_run_ends_where_the_unbroken_one_does(tmp_path, monkeypatch, hand_over):
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "whole"))
    whole = run(8, hand_over=hand_over)

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "split"))
    run(5, hand_over=hand_over)
    split = run(8, hand_over=hand_over)

    for left, right in zip(whole, split):
        assert torch.equal(left, right)


def test_the_step_says_where_the_run_resumed_before_the_first_step(storage):
    run(5)
    seen = []
    run(8, seen)
    assert seen == [5]


def test_one_boundary_per_step_however_many_micro_batches(storage, monkeypatch):
    """Gradient accumulation: four forwards a step, one top of the iteration -
    the first forward after the step, and the very first of the run (GPU-276)."""
    boundaries = []

    @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
    def train():
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        runtime = get_runtime()
        original = runtime.on_batch_boundary
        monkeypatch.setattr(runtime, "on_batch_boundary", lambda: (boundaries.append(ravex.step()), original())[1])
        for _ in range(3):
            for _ in range(4):
                model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

    train()
    # Before step 1, 2 and 3; nothing follows the last step to be its top.
    assert boundaries == [0, 1, 2]


def test_the_hooks_come_off_when_the_run_ends(storage):
    """A model evaluated after the run must not call into a runtime that is gone."""
    kept = {}

    @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
    def train():
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        assert len(model._forward_pre_hooks) == 1
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        kept["model"] = model

    train()
    assert len(kept["model"]._forward_pre_hooks) == 0
    kept["model"](torch.randn(2, 4))
