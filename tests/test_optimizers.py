"""Optimizers other than Adam, and more than one of them.

The end-to-end test in ``test_e2e_vanilla`` holds a resume to bit-for-bit with
Adam. Nothing in Ravex is written for Adam, but "nothing is written for it" is
an argument, and this is the measurement: the same kill-and-resume on the
optimizers people actually reach for next.

**Muon is the interesting one**, and not because of its update. It only takes
matrices, so it is always used in a pair - Muon on the 2-D weights, AdamW on
biases and norms - and a pair is two ``optimizer.step()`` calls per iteration.
Ravex counts steps with a hook on ``step``; if it counted both, every
checkpoint cadence and every resume point would be off by a factor of two.
"""

import importlib.util

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ravex

HAVE_MOONCLIP = importlib.util.find_spec("moonclip") is not None
HAVE_MUON = hasattr(torch.optim, "Muon")

SAMPLES = 64
BATCH = 8
TOTAL_STEPS = 40
CRASH_AT = 20
CHECKPOINT_EVERY = 4
RESUME_FROM = 17  # see test_e2e_vanilla: the checkpoint due at 20 never ran


class Killed(RuntimeError):
    pass


def muon_pair(model):
    matrices = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    return [
        torch.optim.Muon(matrices, lr=0.02, momentum=0.95),
        torch.optim.AdamW(rest, lr=5e-3, weight_decay=0.01),
    ]


OPTIMIZERS = {
    "muon+adamw": muon_pair,
    "sgd-nesterov": lambda model: [
        torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    ],
    "adamw": lambda model: [torch.optim.AdamW(model.parameters(), lr=5e-3)],
    "rmsprop": lambda model: [torch.optim.RMSprop(model.parameters(), lr=1e-3)],
}


def build(make):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1))
    optimizers = make(model)
    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(features, targets), batch_size=BATCH, shuffle=True)
    return model, optimizers, loader


def train(make, die_at=None):
    model, optimizers, loader = build(make)
    losses = {}
    try:
        for _ in range(20):
            for x, y in loader:
                loss = ((model(x) - y) ** 2).mean()
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                losses[ravex.step()] = loss.item()
                if die_at is not None and ravex.step() >= die_at:
                    raise Killed
    except Killed:
        pass
    return model, optimizers, losses


def states(model, optimizers):
    return (
        {k: v.clone() for k, v in model.state_dict().items()},
        [o.state_dict()["state"] for o in optimizers],
    )


@pytest.mark.parametrize(
    "backend",
    ["torch_save", pytest.param("moonclip", marks=pytest.mark.skipif(
        not HAVE_MOONCLIP, reason="moonclip not installed"))],
)
@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_resume_reproduces_the_uninterrupted_run(tmp_path, monkeypatch, backend, name):
    if name.startswith("muon") and not HAVE_MUON:
        pytest.skip("torch.optim.Muon needs torch 2.9 or later")
    make = OPTIMIZERS[name]
    options = dict(backend=backend, checkpoint_every=CHECKPOINT_EVERY, max_steps=TOTAL_STEPS)

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "reference"))
    ravex._activate(**options)
    model, optimizers, reference = train(make)
    reference_weights, reference_state = states(model, optimizers)
    ravex.deactivate()
    # One step per iteration, however many optimizers took one.
    assert sorted(reference) == list(range(1, TOTAL_STEPS + 1))

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "interrupted"))
    ravex._activate(checkpoint_on_exit=False, **options)
    train(make, die_at=CRASH_AT)
    ravex.deactivate()

    ravex._activate(**options)
    model, optimizers, second_half = train(make)
    resumed_weights, resumed_state = states(model, optimizers)
    ravex.deactivate()

    assert sorted(second_half) == list(range(RESUME_FROM, TOTAL_STEPS + 1))
    for step, loss in second_half.items():
        assert loss == reference[step], "step %d differs after resume" % step
    for key, value in resumed_weights.items():
        assert torch.equal(value, reference_weights[key]), "%s differs" % key
    for mine, theirs in zip(resumed_state, reference_state):
        assert mine.keys() == theirs.keys()
        for index in mine:
            for field, value in mine[index].items():
                other = theirs[index][field]
                if torch.is_tensor(value):
                    assert torch.equal(value, other), "optimizer %s.%s" % (index, field)
                else:
                    assert value == other, "optimizer %s.%s" % (index, field)
