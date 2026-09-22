"""`run.json` and `status.json`: what a run is, and where it got to (GPU-148).

The split is what these tests are about. A resume is the same run coming back,
so the document that names it must survive untouched - otherwise every id ever
quoted for that run points at nothing. The status is the opposite: it is only
ever the latest answer, and it is overwritten.
"""

import json
import os
import time

import pytest
import torch

import ravex
import ravex.metrics
import ravex.runs
from ravex._runs import RUN_FILE, STATUS_FILE


def tiny(steps=3):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    for _ in range(steps):
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()


class TestTheDocument:
    def test_a_run_describes_itself(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=2)
        def train():
            tiny(3)

        train()
        described = ravex.runs.describe(str(storage))
        assert described is not None
        assert described["run_id"]
        assert described["name"] == os.path.basename(str(storage))
        assert described["parent"] is None
        assert described["status"]["state"] == "finished"
        assert described["status"]["step"] == 3

    def test_the_name_is_yours_and_the_id_is_not_the_name(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, name="baseline")
        def train():
            tiny(1)

        train()
        described = ravex.runs.describe(str(storage))
        assert described["name"] == "baseline"
        assert described["run_id"] != "baseline"

    def test_two_runs_can_share_a_name_and_never_an_id(self, tmp_path, monkeypatch):
        ids = []
        for number in (1, 2):
            path = tmp_path / f"store{number}"
            monkeypatch.setenv("RAVEX_STORAGE_PATH", str(path))

            @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, name="baseline")
            def train():
                tiny(1)

            train()
            ids.append(ravex.runs.describe(str(path))["run_id"])
        assert ids[0] != ids[1]

    def test_a_resume_keeps_the_identity_it_was_born_with(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=2)
        def first():
            tiny(2)

        first()
        born = ravex.runs.describe(str(storage))

        @ravex.train_loop(backend="torch_save", checkpoint_every=2)
        def second():
            tiny(2)

        second()
        again = ravex.runs.describe(str(storage))
        assert again["run_id"] == born["run_id"]
        assert again["created_at"] == born["created_at"]
        assert again["status"]["step"] == 4, "the status moves even though the document does not"

    def test_a_run_id_from_the_config_does_not_rename_an_existing_store(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def first():
            tiny(1)

        first()
        born = ravex.runs.describe(str(storage))["run_id"]

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, run_id="something-else")
        def second():
            tiny(1)

        second()
        assert ravex.runs.describe(str(storage))["run_id"] == born

    def test_credentials_never_reach_the_document(self, storage, monkeypatch):
        monkeypatch.setenv("RAVEX_S3_ACCESS_KEY", "AKIAsecret")
        monkeypatch.setenv("RAVEX_S3_SECRET_KEY", "shhh")

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            tiny(1)

        train()
        text = (storage / RUN_FILE).read_text(encoding="utf-8")
        assert "AKIAsecret" not in text and "shhh" not in text
        assert json.loads(text)["config"]["checkpoint_every"] == 10_000


class TestWhetherItIsAlive:
    def test_a_finished_run_is_not_alive(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            tiny(1)

        train()
        assert ravex.runs.looks_alive(ravex.runs.describe(str(storage))) is False

    def test_a_running_status_nobody_refreshed_stops_counting(self, storage):
        """A killed process never writes "I was killed": staleness is the evidence."""

        @ravex.train_loop(backend="torch_save", checkpoint_every=1)
        def train():
            tiny(1)
            described = ravex.runs.describe(str(storage))
            assert described["status"]["state"] == "running"
            assert ravex.runs.looks_alive(described) is True
            stale = described["status"]["updated_at"] + ravex.runs.STALE_SECONDS + 1
            assert ravex.runs.looks_alive(described, now=stale) is False

        train()

    def test_the_status_survives_a_reader_arriving_mid_write(self, storage):
        """Written to a temporary name and renamed, so a reader sees one or the other."""

        @ravex.train_loop(backend="torch_save", checkpoint_every=1)
        def train():
            tiny(2)

        train()
        assert not (storage / (STATUS_FILE + ".tmp")).exists()
        assert json.loads((storage / STATUS_FILE).read_text(encoding="utf-8"))["state"] == "finished"


class TestReadingThem:
    def test_discover_lists_the_runs_under_a_directory_newest_first(self, tmp_path, monkeypatch):
        for name in ("older", "newer"):
            monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / name))

            @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, name=name)
            def train():
                tiny(1)

            train()
            time.sleep(0.01)
        found = ravex.runs.discover(str(tmp_path))
        assert [run["name"] for run in found] == ["newer", "older"]
        assert all(run["path"] for run in found)

    def test_a_directory_ravex_never_wrote_to_is_not_a_run(self, tmp_path):
        (tmp_path / "notes").mkdir()
        assert ravex.runs.describe(str(tmp_path / "notes")) is None
        assert ravex.runs.discover(str(tmp_path)) == []

    def test_the_reader_imports_without_the_rust_core(self):
        """The dashboard runs on Python Workers, which cannot load a compiled module."""
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['ravex._core'] = None; sys.modules['torch'] = None\n"
            "import ravex.runs\n"
            "print(ravex.runs.describe('nowhere') is None)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


def test_metrics_and_the_document_agree_on_the_step(storage):
    """Both are written from the same runtime; a reader joins them by run."""

    @ravex.train_loop(backend="torch_save", checkpoint_every=2)
    def train():
        tiny(4)
        ravex.log_metrics({"loss": 1.0})

    train()
    history = ravex.metrics.read(str(storage))
    described = ravex.runs.describe(str(storage))
    assert history["scalars"]["loss"]["step"] == [4]
    assert described["status"]["step"] == 4


@pytest.mark.parametrize("backend", ["torch_save", "moonclip"])
def test_the_document_lands_whatever_the_backend(storage, backend):
    @ravex.train_loop(backend=backend, checkpoint_every=2)
    def train():
        tiny(2)

    train()
    assert ravex.runs.describe(str(storage))["status"]["step"] == 2
