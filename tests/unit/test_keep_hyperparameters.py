"""A run restarted with changed parameters keeps them (the platform's "save").

A resume is the same run coming back, so by default the checkpoint's learning
rate wins. With `keep_hyperparameters` the script's does - otherwise a new
learning rate on the command line would be read, and then silently replaced by
the old one from the checkpoint. The schedule's shape is the script's too; only
how far along it the run is comes from the checkpoint.
"""

import math

import pytest
import torch

import ravex


def script(lr, t_max, total, seen):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    ravex.track(model=model, optimizer=optimizer, scheduler=schedule)
    while ravex.step() < total:
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        if "lr" not in seen:
            # The rate the first step after the restart was taken with.
            seen["lr"] = optimizer.param_groups[0]["lr"]
            seen["start"] = ravex.step() - 1
        schedule.step()
        optimizer.zero_grad()


def cosine(lr, step, t_max):
    return lr * (1 + math.cos(math.pi * step / t_max)) / 2


def first_run(storage):
    ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(0.1, 10, 4, {}))()


class TestKeepHyperparameters:
    def test_a_plain_resume_keeps_the_checkpoints_rate(self, storage):
        first_run(storage)
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(0.5, 10, 6, seen))()
        assert seen["start"] == 4
        assert seen["lr"] == pytest.approx(cosine(0.1, 4, 10))

    def test_with_it_the_scripts_rate_and_schedule_win(self, storage):
        first_run(storage)
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2, keep_hyperparameters=True)(
            lambda: script(0.5, 20, 6, seen)
        )()
        assert seen["start"] == 4
        # The new rate, on the new schedule's shape, at the step the run was at.
        assert seen["lr"] == pytest.approx(cosine(0.5, 4, 20))

    def test_the_environment_asks_for_it_too(self, storage, monkeypatch):
        first_run(storage)
        monkeypatch.setenv("RAVEX_KEEP_HYPERPARAMETERS", "1")
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(0.5, 10, 6, seen))()
        assert seen["lr"] == pytest.approx(cosine(0.5, 4, 10))

    def test_a_restart_into_another_model_stops(self, storage):
        # "save" with a parameter that changes the model: the checkpoint does
        # not fit, and carrying on would corrupt the run.
        first_run(storage)

        def wider():
            model = torch.nn.Linear(4, 3)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            ravex.track(model=model, optimizer=optimizer)
            while ravex.step() < 6:
                ravex.batch_boundary()
                model(torch.randn(2, 4)).sum().backward()
                optimizer.step()
                optimizer.zero_grad()

        with pytest.raises(RuntimeError, match="does not fit"):
            ravex.train_loop(backend="torch_save", checkpoint_every=2, keep_hyperparameters=True)(wider)()


class TestPlatformOverride:
    """RAVEX_OVERRIDE: settings chosen for this run on the platform's run page.
    They win over the decorator's arguments, or a changed checkpoint_every
    would be read and ignored."""

    def test_it_wins_over_the_decorator(self, storage, monkeypatch):
        import os

        monkeypatch.setenv("RAVEX_OVERRIDE", '{"checkpoint_every": 3}')
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(0.1, 10, 7, {}))()
        written = sorted(int(n[5:17]) for n in os.listdir(str(storage)) if n.startswith("step_"))
        assert 3 in written and 6 in written and 2 not in written and 4 not in written

    def test_an_unknown_setting_is_an_error(self, storage, monkeypatch):
        monkeypatch.setenv("RAVEX_OVERRIDE", '{"checkpoint_evry": 3}')
        with pytest.raises(TypeError, match="checkpoint_evry"):
            ravex.train_loop(backend="torch_save")(lambda: script(0.1, 10, 2, {}))()
