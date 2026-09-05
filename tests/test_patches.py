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
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    model.train()

    registry = get_runtime().registry
    assert model in registry.models
    assert optimizer in registry.optimizers
    assert loader in registry.dataloaders


def test_optimizer_steps_are_counted(storage):
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()

    run_steps(model, optimizer, loader, 5)
    assert ravex.step() == 5


def test_gradient_accumulation_counts_optimizer_steps_not_microbatches(storage):
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
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
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
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
    ravex._activate(backend="torch_save", checkpoint_every=3)
    model, optimizer, loader = make_loop()

    # Seven steps, not six: a checkpoint due at step N is collected at the top
    # of iteration N+1, where the state is consistent again.
    run_steps(model, optimizer, loader, 7)
    ravex.flush()

    written = sorted(p.name for p in storage.glob("step_*.pt"))
    assert written == ["step_000000000003.pt", "step_000000000006.pt"]


def test_a_checkpoint_pending_at_exit_still_gets_written(storage):
    ravex._activate(backend="torch_save", checkpoint_every=3)
    model, optimizer, loader = make_loop()

    run_steps(model, optimizer, loader, 6)  # step 6 is due but not yet collected
    ravex.deactivate()

    written = sorted(p.name for p in storage.glob("step_*.pt"))
    assert written == ["step_000000000003.pt", "step_000000000006.pt"]


def test_old_checkpoints_are_pruned(storage):
    ravex._activate(backend="torch_save", checkpoint_every=1, keep_last=2)
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
    ravex._activate(backend="torch_save", checkpoint_every=2)
    model, optimizer, loader = make_loop()
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)

    assert scaler in get_runtime().registry.scalers

    run_amp_steps(model, optimizer, loader, scaler, 2)
    scaler.update(new_scale=512.0)  # as an overflow would
    run_amp_steps(model, optimizer, loader, scaler, 3)
    ravex.deactivate()

    # Fresh process: a new scaler back at its initial scale.
    ravex._activate(backend="torch_save", checkpoint_every=2)
    model, optimizer, loader = make_loop()
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    run_amp_steps(model, optimizer, loader, scaler, 1)

    assert scaler.get_scale() == 512.0
    assert ravex.step() > 1, "the step counter should have resumed, not restarted"


def test_deactivate_restores_pytorch(storage):
    original_init = torch.nn.Module.__init__
    original_step_owner = torch.optim.Optimizer.__init__

    ravex._activate(backend="torch_save", checkpoint_every=10_000)
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
    ravex._activate(backend="torch_save", checkpoint_every=1)

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
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 1)

    runtime = get_runtime()
    _as_one_rank_of_eight(monkeypatch, runtime, main=False)

    asked = []
    monkeypatch.setattr(
        runtime_module, "all_ranks_agree", lambda ok: (asked.append(ok), ok)[1]
    )

    runtime.checkpoint()
    # One: the verdict, which is what the test was written for. Asserted as a
    # sequence rather than as `True in asked` so a rank that skipped it
    # entirely — zero calls rather than one — would also fail this.
    assert asked == [True], (
        "a rank with nothing of its own to write skipped the verdict collective"
    )


def test_a_sharded_multirank_checkpoint_splits_skew_from_drain_by_default(
    storage, monkeypatch
):
    """GPU-98: `skew` used to require `RAVEX_SPLIT_DRAIN`. It is unconditional
    now for a sharded run with more than one rank — the diagnostic itself
    established, on two real machines, that what a single `drain` number was
    hiding was mostly a rank waiting on a checkpointing peer, not queued
    device work, so the split no longer needs a flag to opt into.
    """
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 1)

    runtime = get_runtime()
    _as_one_rank_of_eight(monkeypatch, runtime, main=True)

    from ravex._dist import collectives as distributed_module

    barriers = []
    monkeypatch.setattr(distributed_module, "barrier", lambda: barriers.append(1))
    monkeypatch.setattr(runtime_module, "all_ranks_agree", lambda ok: ok)

    seen_phases = []
    monkeypatch.setattr(
        runtime,
        "_warn_if_cadence_is_expensive",
        lambda step, started, finished, phases: seen_phases.append(phases),
    )

    assert runtime.checkpoint() is True
    assert barriers == [1], (
        "no barrier ran before the drain on a sharded, multi-rank checkpoint"
    )
    assert seen_phases and "skew" in seen_phases[0], (
        "the handoff phases did not include `skew`"
    )
    assert "drain" in seen_phases[0]


def test_a_replicated_job_posts_no_collective_from_rank_zero_alone(
    storage, monkeypatch
):
    """The gather layout sends only rank 0 past the guard above.

    Every rank of a *sharded* run reaches the collectives in the save, which
    is what makes them safe. A replicated run is the opposite: the non-main
    ranks leave at the early `return`, and rank 0 is alone from there on. A
    collective posted there waits for seven ranks that already went home, and
    the job stops at the timeout rather than at an error.

    That is not hypothetical — it is how the `RAVEX_SPLIT_DRAIN` diagnostic's
    agreement was written the first time, and it turned the CI integration job
    into a ten-minute wall. The skew barrier that diagnostic grew into (see
    GPU-98) is unconditional now, gated only by `sharded and get_world_size()
    > 1` — the same guard the verdict below it carries, and for the same
    reason. This pins both: a replicated job must enter neither the barrier
    nor the verdict collective from rank 0 alone.
    """
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
    model, optimizer, loader = make_loop()
    run_steps(model, optimizer, loader, 1)

    runtime = get_runtime()
    # Replicated, not sharded: eight ranks, and this is the only one that gets
    # past the guard.
    monkeypatch.setattr(runtime.registry, "has_sharded_models", lambda: False)
    monkeypatch.setattr(runtime_module, "get_world_size", lambda: 8)
    monkeypatch.setattr(runtime_module, "is_main_process", lambda: True)

    asked = []
    monkeypatch.setattr(
        runtime_module, "all_ranks_agree", lambda ok: (asked.append(ok), ok)[1]
    )
    from ravex._dist import collectives as distributed_module

    barriers = []
    monkeypatch.setattr(distributed_module, "barrier", lambda: barriers.append(1))

    runtime.checkpoint()

    assert asked == [], (
        "rank 0 posted a collective the other seven ranks had already left before"
    )
    assert barriers == [], (
        "rank 0 entered the skew barrier alone on a replicated job"
    )


def test_a_failure_on_another_rank_stops_this_one_too(storage, monkeypatch):
    """This rank's own save was fine. Somebody else's was not.

    It has to skip anyway, and — because the step goes unrecorded on every
    rank — come back to it at the next checkpoint rather than treating it as
    done.
    """
    ravex._activate(backend="torch_save", checkpoint_every=10_000)
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
