"""The final checkpoint of a training function that returns.

The registry holds everything weakly, and the decorator writes the final
checkpoint in its ``finally`` — by which point a function that *returned* has
already released its locals. Until the fix this wrote a checkpoint with no
weights and no optimizer, and it was the newest one: the next run resumed onto
initial weights at the final step. Every test below is a run that returns
normally, which is the case the rest of the suite never had, because every
other test either keeps the model alive or is killed.
"""

import gc
import weakref

import pytest
import torch

import ravex
from ravex._backends import TorchSaveBackend
from ravex._config import RavexConfig


def backend_at(storage):
    config = RavexConfig()
    config.storage.path = str(storage)
    config.backend = "torch_save"
    config._normalize()
    return TorchSaveBackend(config)


def returning_run(steps, checkpoint_every, seen):
    """Train and return, keeping nothing but copies of the final weights."""

    @ravex.train_loop(backend="torch_save", checkpoint_every=checkpoint_every)
    def train():
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        seen["started_at"] = None
        for _ in range(steps):
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            if seen["started_at"] is None:
                seen["started_at"] = ravex.step() - 1
        seen["weight"] = model.weight.detach().clone()
        seen["model"] = weakref.ref(model)

    train()


def test_the_final_checkpoint_carries_the_weights_and_the_optimizer(storage):
    seen = {}
    returning_run(steps=10, checkpoint_every=4, seen=seen)

    state = backend_at(storage).load_step(10)
    assert state is not None, "no final checkpoint was written"
    assert state["models"], "the final checkpoint has no weights"
    assert state["optimizers"], "the final checkpoint has no optimizer"

    (weights,) = state["models"].values()
    assert torch.equal(weights["weight"], seen["weight"]), (
        "the final checkpoint holds weights from before the last steps"
    )


def test_a_run_shorter_than_the_cadence_still_saves_on_return(storage):
    """No periodic checkpoint to pin at, so the pin comes from the first step."""
    seen = {}
    returning_run(steps=3, checkpoint_every=100, seen=seen)

    state = backend_at(storage).load_step(3)
    assert state is not None and state["models"]


def test_the_next_run_resumes_onto_the_trained_weights(storage):
    first = {}
    returning_run(steps=10, checkpoint_every=4, seen=first)

    second = {}
    returning_run(steps=2, checkpoint_every=4, seen=second)

    assert second["started_at"] == 10, "the second run did not resume"
    state = backend_at(storage).load_step(10)
    (weights,) = state["models"].values()
    assert torch.equal(weights["weight"], first["weight"]), (
        "the checkpoint the second run resumed from is not where the first ended"
    )
    later = backend_at(storage).load_step(12)
    assert later is not None and later["models"]


def test_nothing_is_kept_alive_after_the_run(storage):
    """The pin is a bounded exception to the weak registry, not a leak."""
    seen = {}
    returning_run(steps=10, checkpoint_every=4, seen=seen)
    gc.collect()
    assert seen["model"]() is None, "Ravex kept the model alive past shutdown"
