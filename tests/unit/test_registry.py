import torch
import torch.nn as nn

from ravex._registry import ObjectRegistry


class Net(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.body(x)


def note_tree(registry, module):
    """Stand in for the nn.Module.__init__ patch, which sees every module."""
    for sub in module.modules():
        registry.note_module(sub)


def test_only_the_outermost_module_counts_as_a_model():
    registry = ObjectRegistry()
    net = Net()
    note_tree(registry, net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    registry.register_optimizer(optimizer)

    # Net has 5 modules with parameters underneath it; only Net itself is the model.
    assert registry.models == [net]


def test_modules_no_optimizer_owns_are_ignored():
    registry = ObjectRegistry()
    trained = Net()
    incidental = Net()  # e.g. a metric or loss helper a library built
    note_tree(registry, trained)
    note_tree(registry, incidental)
    optimizer = torch.optim.SGD(trained.parameters(), lr=0.1)
    registry.register_optimizer(optimizer)

    assert registry.models == [trained]


def test_a_module_put_in_train_mode_counts_even_without_an_optimizer():
    registry = ObjectRegistry()
    ema = Net()
    note_tree(registry, ema)
    registry.register_model(ema)  # what Module.train() triggers

    assert registry.models == [ema]


def test_fingerprint_is_stable_across_instances_and_processes():
    # Same architecture, different random init: the checkpoint from one run has
    # to apply to the other, so the key must match.
    first, second = Net(), Net()
    assert ObjectRegistry.fingerprint_model(first) == ObjectRegistry.fingerprint_model(
        second
    )
    # And it must not depend on PYTHONHASHSEED-style randomness.
    assert ObjectRegistry.fingerprint_model(first).startswith("model_")


def test_fingerprint_changes_with_the_architecture():
    assert ObjectRegistry.fingerprint_model(Net(hidden=8)) != (
        ObjectRegistry.fingerprint_model(Net(hidden=16))
    )


def test_identical_twins_get_distinct_keys():
    registry = ObjectRegistry()
    generator, discriminator = Net(), Net()
    note_tree(registry, generator)
    note_tree(registry, discriminator)
    registry.register_model(generator)
    registry.register_model(discriminator)

    keys = [key for key, _ in registry.keyed_models()]
    assert len(keys) == 2
    assert len(set(keys)) == 2, "twin models must not collide on one key"
    assert keys[1].endswith("#1")


def test_dead_models_drop_out_of_the_registry():
    registry = ObjectRegistry()
    net = Net()
    note_tree(registry, net)
    registry.register_model(net)
    assert len(registry.models) == 1

    del net
    import gc

    gc.collect()
    assert registry.models == []


def test_state_roundtrip_restores_weights_and_step():
    registry = ObjectRegistry()
    net = Net()
    note_tree(registry, net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # noqa: F841 - held alive
    registry.register_optimizer(optimizer)
    registry.step_count = 17

    state = registry.collect_state()

    # Move the weights somewhere else entirely, then restore.
    with torch.no_grad():
        for param in net.parameters():
            param.add_(1.0)
    registry.step_count = 0

    registry.restore_state(state)

    assert registry.step_count == 17
    for key, saved in state["models"][next(iter(state["models"]))].items():
        assert torch.equal(net.state_dict()[key], saved)


def test_restore_survives_an_architecture_change(caplog):
    registry = ObjectRegistry()
    small = Net(hidden=8)
    note_tree(registry, small)
    registry.register_model(small)
    state = registry.collect_state()

    # Fresh process, user changed the model: the key no longer matches.
    other = ObjectRegistry()
    big = Net(hidden=16)
    note_tree(other, big)
    other.register_model(big)

    before = {k: v.clone() for k, v in big.state_dict().items()}
    other.restore_state(state)  # must warn, not raise

    for key, value in big.state_dict().items():
        assert torch.equal(value, before[key]), "weights must be left untouched"


def test_rng_state_roundtrip():
    registry = ObjectRegistry()
    net = Net()
    note_tree(registry, net)
    registry.register_model(net)

    torch.manual_seed(1234)
    state = registry.collect_state(track_rng=True)
    expected = torch.randn(5)

    torch.manual_seed(999)  # scramble
    registry.restore_state(state)

    assert torch.equal(torch.randn(5), expected)
