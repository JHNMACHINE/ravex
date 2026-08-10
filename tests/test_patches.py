import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ravex
from ravex._runtime import get_runtime


def make_loop():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.randn(32, 4), torch.randn(32, 1))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return model, optimizer, loader


def run_steps(model, optimizer, loader, steps):
    done = 0
    while done < steps:
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            done += 1
            if done >= steps:
                break
    return done


def test_objects_are_registered_without_touching_user_code(storage):
    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    model.train()

    registry = get_runtime().registry
    assert model in registry.models
    assert optimizer in registry.optimizers
    assert loader in registry.dataloaders


def test_optimizer_steps_are_counted(storage):
    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()

    run_steps(model, optimizer, loader, 5)
    assert ravex.step() == 5


def test_gradient_accumulation_counts_optimizer_steps_not_microbatches(storage):
    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()

    accumulation = 4
    microbatches = 0
    for x, y in loader:
        loss = ((model(x) - y) ** 2).mean() / accumulation
        loss.backward()
        microbatches += 1
        if microbatches % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

    assert microbatches == 4
    assert ravex.step() == 1, "one accumulation cycle is one step"


def test_a_second_optimizer_does_not_double_count(storage):
    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    generator = nn.Linear(4, 4)
    discriminator = nn.Linear(4, 1)
    optimizer_g = torch.optim.SGD(generator.parameters(), lr=0.1)
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.1)

    for _ in range(3):
        x = torch.randn(8, 4)
        discriminator(generator(x)).mean().backward()
        optimizer_g.step()
        optimizer_d.step()

    assert ravex.step() == 3, "one training iteration is one step"


def test_a_checkpoint_lands_on_the_configured_cadence(storage):
    ravex.activate(backend="torch_save", checkpoint_every=3)
    model, optimizer, loader = make_loop()

    # Seven steps, not six: a checkpoint due at step N is collected at the top
    # of iteration N+1, where the state is consistent again.
    run_steps(model, optimizer, loader, 7)
    ravex.flush()

    written = sorted(p.name for p in storage.glob("step_*.pt"))
    assert written == ["step_000000000003.pt", "step_000000000006.pt"]


def test_a_checkpoint_pending_at_exit_still_gets_written(storage):
    ravex.activate(backend="torch_save", checkpoint_every=3)
    model, optimizer, loader = make_loop()

    run_steps(model, optimizer, loader, 6)  # step 6 is due but not yet collected
    ravex.deactivate()

    written = sorted(p.name for p in storage.glob("step_*.pt"))
    assert written == ["step_000000000003.pt", "step_000000000006.pt"]


def test_old_checkpoints_are_pruned(storage):
    ravex.activate(backend="torch_save", checkpoint_every=1, keep_last=2)
    model, optimizer, loader = make_loop()

    run_steps(model, optimizer, loader, 5)
    ravex.flush()

    assert len(list(storage.glob("step_*.pt"))) == 2


def test_deactivate_restores_pytorch(storage):
    original_init = torch.nn.Module.__init__
    original_step_owner = torch.optim.Optimizer.__init__

    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    assert torch.nn.Module.__init__ is not original_init

    ravex.deactivate()

    assert torch.nn.Module.__init__ is original_init
    assert torch.optim.Optimizer.__init__ is original_step_owner
    assert not hasattr(torch.nn.Module, "_ravex_patched")


def test_a_broken_backend_disables_ravex_instead_of_the_run(storage, monkeypatch):
    ravex.activate(backend="torch_save", checkpoint_every=1)
    runtime = get_runtime()

    def explode(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(runtime.backend, "save", explode)

    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 3)  # must not raise

    assert not ravex.is_active()
