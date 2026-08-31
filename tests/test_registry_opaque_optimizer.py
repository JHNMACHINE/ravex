"""When the optimizer's parameters are not the model's — GPU-90 follow-up.

The registry decides what to checkpoint by asking, of each outermost module
with parameters, whether an optimizer owns one of them. That question has an
answer for every ordinary run and none at all for DeepSpeed: ZeRO hands the
base optimizer flat partition buffers, so nothing owns anything and the *test*
comes back negative for reasons that have nothing to do with the module.

Measured on 2026-08-31 with ZeRO stage 1 in a container: a four-parameter
module, one parameter owned by ``FusedAdam``, none of them the module's — and
both of the registry's filters missed. The user's model was dropped as
contained in the DeepSpeed engine, the engine was dropped as unowned, and Ravex
wrote checkpoints holding an optimizer, a dataloader and **no weights**,
logging a successful handoff each time. At stage 3 it was worse: the base
optimizer's ``state_dict()`` raised a bare ``KeyError`` on a parameter id and
took the whole checkpoint down with it, every interval.

None of that needs DeepSpeed to reproduce. The shape is what matters — a
wrapper module holding the real model, and an optimizer over buffers that
belong to neither — and it is built here out of plain torch.
"""

import pytest

import torch
import torch.nn as nn

from ravex._registry import ObjectRegistry


class Engine(nn.Module):
    """A wrapper that owns the model, the way DeepSpeed's engine does."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):  # pragma: no cover - never called
        return self.module(x)


def flat_optimizer(numel=16):
    """An optimizer over a buffer that is nobody's parameter. ZeRO's shape."""
    partition = nn.Parameter(torch.zeros(numel))
    return torch.optim.SGD([partition], lr=0.1)


@pytest.fixture
def registry():
    return ObjectRegistry()


@pytest.fixture
def alive():
    """Somewhere to keep strong references for the length of a test.

    The registry holds everything weakly, on purpose — it says so: a registry
    that kept a model alive would turn "we checkpoint your training" into "we
    leak your training". So an optimizer built inline as an argument is
    collected before the assertion runs, and the test fails for a reason that
    has nothing to do with what it is testing. Ask this fixture to hold it.
    """
    kept = []

    def keep(obj):
        kept.append(obj)
        return obj

    return keep


# ── the bug ─────────────────────────────────────────────────────────────────


def test_a_wrapped_model_is_still_checkpointed_when_nothing_owns_it(registry, alive):
    """The whole failure, in three lines of setup.

    Without the fallback this returns no models at all, and the checkpoint that
    follows has no weights in it.
    """
    model = nn.Sequential(nn.Linear(4, 4))
    engine = Engine(model)
    registry.note_module(model)
    registry.note_module(engine)
    registry.register_optimizer(alive(flat_optimizer()))

    roots = registry.models
    assert roots, "no model to checkpoint: this is the bug"
    assert len(roots) == 1
    # The engine is the root; `unwrap_model` strips it when the state is taken,
    # which is how DDP has always been handled.
    assert roots[0] is engine


def test_the_state_that_comes_out_has_the_models_own_names(registry, alive):
    """`unwrap_model` follows `.module`, so the keys are the user's, not the wrapper's."""
    model = nn.Sequential(nn.Linear(4, 4))
    engine = Engine(model)
    registry.note_module(model)
    registry.note_module(engine)
    registry.register_optimizer(alive(flat_optimizer()))

    state = registry.collect_state(track_rng=False)
    saved = next(iter(state["models"].values()))
    assert sorted(saved) == ["0.bias", "0.weight"]


# ── and no regression where the test *does* answer ──────────────────────────


def test_an_incidental_module_is_still_dropped(registry, alive):
    """The fallback must not turn every module with parameters into a model.

    A loss wrapper holding buffers is the case the ownership test exists for,
    and it is only safe to ignore that test when the test could not have
    succeeded for anybody.
    """
    model = nn.Sequential(nn.Linear(4, 4))
    incidental = nn.Linear(3, 3)  # nobody optimises this
    registry.note_module(model)
    registry.note_module(incidental)
    registry.register_optimizer(alive(torch.optim.SGD(model.parameters(), lr=0.1)))

    roots = registry.models
    assert model in roots
    assert incidental not in roots


def test_no_optimizer_at_all_is_not_the_same_as_an_opaque_one(registry):
    """A model built and never trained is not a run whose optimizer is opaque.

    Keying the fallback on "there are optimizers" rather than on "they own
    something" is what separates the two — and it is also what makes it fire at
    ZeRO stage 3, where the base optimizer's parameter groups come back empty
    altogether. An earlier version keyed it on the parameters being non-empty
    and missed exactly that stage.
    """
    incidental = nn.Linear(3, 3)
    registry.note_module(incidental)

    assert registry.models == []


# ── parameters that are not there ───────────────────────────────────────────


def test_partitioned_away_is_recognised(registry, alive):
    """ZeRO stage 3 leaves the keys and takes the data.

    Measured: a four-parameter module straight after `deepspeed.initialize` at
    stage 3 reports four parameters, every one `numel() == 0`, where stages 1
    and 2 report them at full size.
    """
    hollow = nn.Linear(4, 4)
    with torch.no_grad():
        for param in hollow.parameters():
            param.data = torch.empty(0)
    registry.note_module(hollow)
    registry.register_optimizer(alive(flat_optimizer()))

    assert registry.models == [hollow]
    assert registry.parameters_are_partitioned_away() is True


def test_a_model_with_real_parameters_is_not_partitioned_away(registry, alive):
    model = nn.Linear(4, 4)
    registry.note_module(model)
    registry.register_optimizer(alive(torch.optim.SGD(model.parameters(), lr=0.1)))
    assert registry.parameters_are_partitioned_away() is False


def test_no_models_is_not_partitioned_away(registry):
    """Nothing to be wrong about is a different answer from everything empty."""
    assert registry.parameters_are_partitioned_away() is False


# ── an optimizer that cannot describe itself ────────────────────────────────


class Hostile(torch.optim.SGD):
    """An optimizer whose `state_dict()` raises, the way ZeRO-3's base does."""

    def state_dict(self):
        raise KeyError(140234567890)


def test_one_optimizer_failing_does_not_cost_the_checkpoint_its_weights(registry, alive):
    """The weights are the part a resume cannot do without.

    Before this, a `KeyError` here took the whole collection down — every
    interval — and reported it as `Checkpoint at step 4 failed:
    133268936989168`: a number, with nothing to say what it belonged to.
    """
    model = nn.Sequential(nn.Linear(4, 4))
    registry.note_module(model)
    registry.register_optimizer(alive(Hostile(model.parameters(), lr=0.1)))

    state = registry.collect_state(track_rng=False)

    assert state["optimizers"] == {}
    saved = next(iter(state["models"].values()))
    assert sorted(saved) == ["0.bias", "0.weight"]


def test_a_healthy_optimizer_beside_a_broken_one_still_comes_through(registry, alive):
    model = nn.Sequential(nn.Linear(4, 4))
    other = nn.Sequential(nn.Linear(2, 2))
    registry.note_module(model)
    registry.note_module(other)
    registry.register_optimizer(alive(Hostile(model.parameters(), lr=0.1)))
    registry.register_optimizer(alive(torch.optim.SGD(other.parameters(), lr=0.1)))

    state = registry.collect_state(track_rng=False)
    assert len(state["optimizers"]) == 1
