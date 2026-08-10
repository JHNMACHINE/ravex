import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ravex
from ravex._backends import TorchSaveBackend
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


def run_amp_steps(model, optimizer, loader, scaler, steps):
    done = 0
    while done < steps:
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # calls optimizer.step() internally
            scaler.update()
            done += 1
            if done >= steps:
                return


def test_the_amp_scaler_is_tracked_and_its_scale_survives_a_resume(storage):
    """The loss scale is tuned state, not a constant.

    A resume that reset it to init_scale would spend the next few hundred steps
    re-discovering the scale, overflowing gradients on the way. Exercised on
    CPU here; the mechanism is device-independent since the hook is on
    GradScaler.__init__.
    """
    ravex.activate(backend="torch_save", checkpoint_every=2)
    model, optimizer, loader = make_loop()
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)

    assert scaler in get_runtime().registry.scalers

    run_amp_steps(model, optimizer, loader, scaler, 2)
    scaler.update(new_scale=512.0)  # as an overflow would
    run_amp_steps(model, optimizer, loader, scaler, 3)
    ravex.deactivate()

    # Fresh process: a new scaler back at its initial scale.
    ravex.activate(backend="torch_save", checkpoint_every=2)
    model, optimizer, loader = make_loop()
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    run_amp_steps(model, optimizer, loader, scaler, 1)

    assert scaler.get_scale() == 512.0
    assert ravex.step() > 1, "the step counter should have resumed, not restarted"


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

    def explode(*args, **kwargs):
        raise RuntimeError("disk on fire")

    # Patched on the class: the backend instance does not exist yet, since it
    # is only built when something actually needs to read or write.
    monkeypatch.setattr(TorchSaveBackend, "save", explode)

    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 3)  # must not raise

    assert not ravex.is_active()
