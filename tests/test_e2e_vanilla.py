"""The test the whole project exists to pass.

Train a model, kill the process mid-run, start the same script again, and check
that what comes out is bit-for-bit what an uninterrupted run would have
produced — same losses, step by step, and the same final weights.

The model deliberately contains dropout and the loader shuffles, so passing
requires the global RNG, the sampler's generator and the dataset position all
to come back exactly where they were. Restoring only the weights would give
plausible-looking losses that quietly differ.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ravex

SAMPLES = 64
BATCH = 8
STEPS_PER_EPOCH = SAMPLES // BATCH
TOTAL_STEPS = 40
CRASH_AT = 20
CHECKPOINT_EVERY = 4
SEED = 0

#: A hard kill loses whatever happened after the last checkpoint. Step 16 is
#: the last one whose checkpoint was actually collected — the one due at step 20
#: would have been taken at the top of iteration 21, which never ran. So the
#: resumed run replays 17-20 (identically) before reaching new ground.
RESUME_FROM = 17


class Killed(RuntimeError):
    """Stands in for the instance disappearing under the training loop."""


def build():
    """Everything the user's script would build. Deterministic given SEED."""
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(features, targets), batch_size=BATCH, shuffle=True)
    return model, optimizer, scheduler, loader


def train(epochs=20, die_at=None):
    """A plain PyTorch loop. Not one line of it knows Ravex exists."""
    model, optimizer, scheduler, loader = build()
    losses = {}
    try:
        for _ in range(epochs):
            for x, y in loader:
                loss = ((model(x) - y) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses[ravex.step()] = loss.item()
                if die_at is not None and ravex.step() >= die_at:
                    raise Killed
    except Killed:
        pass
    return model, losses


def weights(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def run_scenario(tmp_path, monkeypatch, backend):
    options = dict(
        backend=backend, checkpoint_every=CHECKPOINT_EVERY, max_steps=TOTAL_STEPS
    )

    # ── the run we are trying to reproduce ──────────────────────────
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "reference"))
    ravex.activate(**options)
    reference_model, reference = train()
    reference_weights = weights(reference_model)
    ravex.deactivate()

    assert sorted(reference) == list(range(1, TOTAL_STEPS + 1))

    # ── same script, killed at step 20 ──────────────────────────────
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "interrupted"))
    ravex.activate(checkpoint_on_exit=False, **options)
    _, first_half = train(die_at=CRASH_AT)
    ravex.deactivate()

    assert max(first_half) == CRASH_AT
    for step, loss in first_half.items():
        assert loss == reference[step], f"step {step} diverged before the crash"

    # ── the process comes back, same storage, no code change ────────
    ravex.activate(**options)
    resumed_model, second_half = train()
    ravex.deactivate()

    return reference, reference_weights, second_half, weights(resumed_model)


def test_resume_reproduces_the_uninterrupted_run(tmp_path, monkeypatch):
    reference, reference_weights, second_half, resumed_weights = run_scenario(
        tmp_path, monkeypatch, "torch_save"
    )

    # It picked up where the last checkpoint left off and stopped at the
    # configured budget, rather than running a full extra script's worth.
    assert sorted(second_half) == list(range(RESUME_FROM, TOTAL_STEPS + 1))

    for step, loss in second_half.items():
        assert loss == reference[step], (
            f"loss at step {step} differs after resume: "
            f"{loss!r} vs {reference[step]!r}"
        )

    for key, value in resumed_weights.items():
        assert torch.equal(value, reference_weights[key]), f"{key} differs"


@pytest.mark.skipif(
    pytest.importorskip("moonclip", reason="moonclip not installed") is None,
    reason="moonclip not installed",
)
def test_resume_reproduces_the_uninterrupted_run_on_moonclip(tmp_path, monkeypatch):
    reference, reference_weights, second_half, resumed_weights = run_scenario(
        tmp_path, monkeypatch, "moonclip"
    )

    assert sorted(second_half) == list(range(RESUME_FROM, TOTAL_STEPS + 1))

    for step, loss in second_half.items():
        assert loss == reference[step], f"loss at step {step} differs after resume"

    for key, value in resumed_weights.items():
        assert torch.equal(value, reference_weights[key]), f"{key} differs"


def test_a_fresh_run_starts_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "empty"))
    ravex.activate(backend="torch_save", checkpoint_every=4, max_steps=8)
    _, losses = train()
    ravex.deactivate()

    assert sorted(losses) == list(range(1, 9))
