import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ravex
from ravex import _runtime as runtime_module
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


def test_a_broken_backend_costs_the_checkpoint_not_the_run(storage, monkeypatch):
    """A failed write must not take the run down — and must not give up either.

    It used to `_disable`, which is per process. On one rank of eight that is
    the worst of both: the run keeps going without protection, and the next
    checkpoint hangs the other seven in a collective this rank no longer
    enters. So the failure now costs one checkpoint, and the next one is tried
    on its own merits — a full disk is often not full a minute later.
    """
    ravex.activate(backend="torch_save", checkpoint_every=1)

    attempts = []

    def explode(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("disk on fire")

    # Patched on the class: the backend instance does not exist yet, since it
    # is only built when something actually needs to read or write.
    monkeypatch.setattr(TorchSaveBackend, "save", explode)

    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 3)  # must not raise

    assert ravex.is_active(), (
        "the runtime turned itself off; on a sharded run that is the rank "
        "the others wait for"
    )
    assert len(attempts) >= 2, (
        f"it gave up after the first failure ({len(attempts)} attempt(s) in "
        "3 steps at checkpoint_every=1)"
    )


def _as_one_rank_of_eight(monkeypatch, runtime, main: bool):
    """Make this process look like a rank of a sharded eight-rank run."""
    monkeypatch.setattr(runtime.registry, "has_sharded_models", lambda: True)
    monkeypatch.setattr(runtime_module, "get_world_size", lambda: 8)
    monkeypatch.setattr(runtime_module, "is_main_process", lambda: main)


def test_a_rank_with_nothing_to_write_still_answers_the_verdict(storage, monkeypatch):
    """The gather layout leaves the non-main ranks holding nothing.

    They used to `return` as soon as they saw that, which is fine only because
    nothing collective came after. Now the verdict does, and a collective that
    seven ranks enter and one skips is exactly the hang this is all about.
    """
    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 1)

    runtime = get_runtime()
    _as_one_rank_of_eight(monkeypatch, runtime, main=False)

    asked = []
    monkeypatch.setattr(
        runtime_module, "all_ranks_agree", lambda ok: (asked.append(ok), ok)[1]
    )

    runtime.checkpoint()
    assert asked == [True], "a rank with nothing of its own to write skipped the agreement"


def test_a_failure_on_another_rank_stops_this_one_too(storage, monkeypatch):
    """This rank's own save was fine. Somebody else's was not.

    It has to skip anyway, and — because the step goes unrecorded on every
    rank — come back to it at the next checkpoint rather than treating it as
    done.
    """
    ravex.activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 1)

    runtime = get_runtime()
    _as_one_rank_of_eight(monkeypatch, runtime, main=True)
    monkeypatch.setattr(runtime_module, "all_ranks_agree", lambda ok: False)

    step = runtime.registry.step_count
    assert runtime.checkpoint() is False
    assert runtime._last_saved_step != step, (
        "the step was recorded as saved while another rank had failed, so no "
        "rank will come back to it"
    )
    assert ravex.is_active(), "one rank's failure turned the others off for good"
