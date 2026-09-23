"""A fork: a new run that starts from step N of another (GPU-149).

What these tests hold Ravex to is the issue's "done when": two forks from the
same step of a parent, with different learning rates, run and write separate
stores; each starts from the parent's state at that step; and the parent is
left exactly as it was. Plus the two ways a fork can be asked for wrongly.
"""

import os

import pytest
import torch

import ravex
import ravex.runs
from ravex._resume import ResumeStepMissing


def train_to(total, lr=0.1, seen=None):
    """A loop that trains until `ravex.step()` reaches `total`, the way a
    resumable script is written - so a fork continues where it starts.

    `seen["start"]` is the step the run started from. Read after the first
    optimizer step, because that is where Ravex restores: before it the count
    is still 0 whether or not there is anything to resume.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ravex.track(model=model, optimizer=optimizer)
    while ravex.step() < total:
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        if seen is not None and "start" not in seen:
            seen["start"] = ravex.step() - 1
    return model


def parent_run(path, monkeypatch, steps=6):
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(path))

    @ravex.train_loop(backend="torch_save", checkpoint_every=2, name="base")
    def train():
        train_to(steps)

    train()


def weight_at(path, step):
    state = torch.load(os.path.join(str(path), "step_%012d.pt" % step), weights_only=False)
    (model,) = state["models"].values()
    return model["weight"]


class TestFork:
    def test_two_forks_from_one_step_write_their_own_stores(self, tmp_path, monkeypatch):
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        before = sorted(os.listdir(parent))
        parent_id = ravex.runs.describe(str(parent))["run_id"]

        seen = {}
        for lr in (0.3, 0.01):
            child = tmp_path / ("lr-%s" % lr)
            monkeypatch.setenv("RAVEX_STORAGE_PATH", str(child))
            record = {}

            @ravex.train_loop(
                backend="torch_save",
                checkpoint_every=2,
                name="base",
                fork_from=str(parent),
                fork_step=4,
            )
            def train():
                train_to(8, lr=lr, seen=record)

            train()
            seen[lr] = record
            described = ravex.runs.describe(str(child))
            # A run of its own, whose document says where it came from.
            assert described["run_id"] != parent_id
            assert described["parent"] == {"run": parent_id, "step": 4}
            assert described["status"]["state"] == "finished"
            assert described["status"]["step"] == 8
            # Its checkpoints are in its own store, from the fork on.
            written = sorted(n for n in os.listdir(child) if n.startswith("step_"))
            assert written and all(int(n[5:17]) > 4 for n in written)

        # Both started from the parent's step 4.
        for record in seen.values():
            assert record["start"] == 4
        # And the parent has not noticed.
        assert sorted(os.listdir(parent)) == before
        assert ravex.runs.describe(str(parent))["parent"] is None

    def test_a_fork_starts_from_the_parents_weights(self, tmp_path, monkeypatch):
        # With lr=0 nothing moves, so what the fork ends with is exactly what it
        # started from - which has to be the parent's step 4, not a fresh model
        # and not the parent's newest.
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))
        models = []

        @ravex.train_loop(backend="torch_save", checkpoint_every=2, fork_from=str(parent), fork_step=4)
        def train():
            models.append(train_to(6, lr=0.0))

        train()
        assert torch.equal(models[0].weight.detach(), weight_at(parent, 4))
        assert not torch.equal(weight_at(parent, 4), weight_at(parent, 6))

    def test_the_fork_keeps_its_own_schedule(self, tmp_path, monkeypatch):
        # The same script forked with another learning rate. The parent's
        # optimizer and scheduler state come over - the schedule is at step 4
        # - and the rate is the child's cosine evaluated there: not the
        # parent's, and not the child's starting one.
        import math

        def script(lr, total, seen=None):
            torch.manual_seed(0)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            ravex.track(model=model, optimizer=optimizer, scheduler=schedule)
            while ravex.step() < total:
                ravex.batch_boundary()
                model(torch.randn(2, 4)).sum().backward()
                optimizer.step()
                if seen is not None and "lr" not in seen:
                    # The rate the first step after the fork was taken with.
                    seen["lr"] = optimizer.param_groups[0]["lr"]
                schedule.step()
                optimizer.zero_grad()

        parent = tmp_path / "base"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(parent))
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(0.1, 6))()

        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2, fork_from=str(parent), fork_step=4)(
            lambda: script(0.5, 6, seen)
        )()
        assert seen["lr"] == pytest.approx(0.5 * (1 + math.cos(math.pi * 4 / 10)) / 2)

    def test_a_fork_that_starts_again_resumes_from_its_own_checkpoint(self, tmp_path, monkeypatch):
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        child = tmp_path / "child"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(child))
        seen = {}

        for total in (6, 8):
            seen.clear()  # each start records where it began

            @ravex.train_loop(backend="torch_save", checkpoint_every=2, fork_from=str(parent), fork_step=2)
            def train():
                train_to(total, seen=seen)

            train()
        # The second start found the child's own step 6, not the parent's 2.
        assert seen["start"] == 6
        assert ravex.runs.describe(str(child))["parent"]["step"] == 2

    def test_without_a_step_the_fork_takes_the_parents_newest(self, tmp_path, monkeypatch):
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        child = tmp_path / "child"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(child))

        @ravex.train_loop(backend="torch_save", checkpoint_every=2, fork_from=str(parent))
        def train():
            train_to(7)

        train()
        assert ravex.runs.describe(str(child))["parent"]["step"] == 6

    def test_a_step_the_parent_does_not_hold_stops_the_run(self, tmp_path, monkeypatch):
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))

        @ravex.train_loop(backend="torch_save", checkpoint_every=2, fork_from=str(parent), fork_step=3)
        def train():
            train_to(8)

        with pytest.raises(ResumeStepMissing) as raised:
            train()
        # The steps it could have asked for, which is the useful half of the answer.
        assert "fork_step=3" in str(raised.value)
        assert raised.value.available == [2, 4, 6]

    def test_the_environment_asks_for_a_fork_too(self, tmp_path, monkeypatch):
        # How the platform's agent asks: RAVEX_FORK_FROM and RAVEX_FORK_STEP.
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        child = tmp_path / "child"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(child))
        monkeypatch.setenv("RAVEX_FORK_FROM", str(parent))
        monkeypatch.setenv("RAVEX_FORK_STEP", "4")

        @ravex.train_loop(backend="torch_save", checkpoint_every=2)
        def train():
            train_to(6)

        train()
        assert ravex.runs.describe(str(child))["parent"]["step"] == 4

    def test_with_moonclip_the_parents_step_is_pinned(self, tmp_path, monkeypatch):
        # Retention on a parent that keeps training must not take the step a
        # fork started from. Moonclip can hold it; torch_save cannot, and says so.
        moonclip = pytest.importorskip("moonclip")
        parent = tmp_path / "base"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(parent))
        ravex.train_loop(backend="moonclip", checkpoint_every=2, async_save=False)(lambda: train_to(6))()

        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))
        ravex.train_loop(
            backend="moonclip", checkpoint_every=2, async_save=False, fork_from=str(parent), fork_step=4
        )(lambda: train_to(8))()

        manager = moonclip.MoonclipManager(str(parent))
        if not hasattr(manager, "pinned_steps"):
            pytest.skip("this Moonclip predates pin()")
        assert 4 in manager.pinned_steps()
        assert ravex.runs.describe(str(tmp_path / "child"))["parent"]["step"] == 4

    def test_a_fork_into_another_model_stops_instead_of_starting_over(self, tmp_path, monkeypatch):
        # A parameter that changes the model was changed: the parent's weights
        # do not fit. Training from scratch under the fork's lineage would be a
        # run whose history is a lie; it stops and says why.
        parent = tmp_path / "base"
        parent_run(parent, monkeypatch)
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))

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
            ravex.train_loop(backend="torch_save", checkpoint_every=2, fork_from=str(parent), fork_step=4)(wider)()


def logged_to(total, lr=0.1):
    """train_to, logging the loss at every step."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ravex.track(model=model, optimizer=optimizer)
    while ravex.step() < total:
        ravex.batch_boundary()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ravex.log_metrics({"train/loss": loss})


class TestReadingALineage:
    """ravex.metrics.read follows a fork back to its parent, without the platform."""

    def test_a_forks_series_begins_with_its_parents(self, tmp_path, monkeypatch):
        import ravex.metrics

        parent = tmp_path / "base"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(parent))
        ravex.train_loop(backend="torch_save", checkpoint_every=2, metrics_chunk_every=0)(lambda: logged_to(6))()
        child = tmp_path / "child"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(child))
        ravex.train_loop(
            backend="torch_save", checkpoint_every=2, metrics_chunk_every=0, fork_from=str(parent), fork_step=4
        )(lambda: logged_to(8, lr=0.01))()

        whole = ravex.metrics.read(str(child))
        assert whole["scalars"]["train/loss"]["step"] == [1, 2, 3, 4, 5, 6, 7, 8]
        # Where the line changes hands: the chart's branch point.
        assert whole["parent"] == {"store": str(parent), "step": 4}
        # The parent's first four are the parent's own values.
        assert whole["scalars"]["train/loss"]["value"][:4] == ravex.metrics.read(str(parent))["scalars"]["train/loss"]["value"][:4]

        own = ravex.metrics.read(str(child), inherited=False)
        assert own["scalars"]["train/loss"]["step"] == [5, 6, 7, 8]
        assert "parent" not in own

    def test_a_parent_that_moved_leaves_the_forks_own_points(self, tmp_path, monkeypatch):
        import shutil

        import ravex.metrics

        parent = tmp_path / "base"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(parent))
        ravex.train_loop(backend="torch_save", checkpoint_every=2, metrics_chunk_every=0)(lambda: logged_to(4))()
        child = tmp_path / "child"
        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(child))
        ravex.train_loop(
            backend="torch_save", checkpoint_every=2, metrics_chunk_every=0, fork_from=str(parent), fork_step=2
        )(lambda: logged_to(4))()
        shutil.rmtree(parent)
        assert ravex.metrics.read(str(child))["scalars"]["train/loss"]["step"] == [3, 4]


class TestAParentStillTraining:
    """The issue's last condition: the parent carries on without noticing."""

    def test_a_fork_starts_while_the_parent_keeps_training(self, tmp_path, monkeypatch):
        # Two processes on one Moonclip store: the parent writing its manifest
        # at every checkpoint, the child opening it to read step 4 and pin it.
        # The pin must survive the parent's later writes.
        import subprocess
        import sys
        import time

        moonclip = pytest.importorskip("moonclip")
        parent = tmp_path / "base"
        code = f"""
import os, time, torch, ravex
os.environ["RAVEX_STORAGE_PATH"] = {str(parent)!r}

@ravex.train_loop(backend="moonclip", checkpoint_every=2, async_save=False)
def train():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    while ravex.step() < 40:
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        time.sleep(0.1)

train()
"""
        process = subprocess.Popen([sys.executable, "-c", code])
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                described = ravex.runs.describe(str(parent))
                if described and (described["status"].get("step") or 0) >= 6:
                    break
                time.sleep(0.2)
            else:
                raise AssertionError("the parent never reached step 6")

            monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))
            seen = {}
            ravex.train_loop(
                backend="moonclip", checkpoint_every=2, async_save=False, fork_from=str(parent), fork_step=4
            )(lambda: train_to(8, seen=seen))()
            assert seen["start"] == 4
            # Forked while the parent was still going.
            assert process.poll() is None
            assert process.wait(timeout=120) == 0
        finally:
            if process.poll() is None:
                process.kill()

        described = ravex.runs.describe(str(parent))
        assert described["status"]["state"] == "finished" and described["status"]["step"] == 40
        manager = moonclip.MoonclipManager(str(parent))
        if hasattr(manager, "pinned_steps"):
            assert 4 in manager.pinned_steps(), "the parent's own writes dropped the fork's pin"
