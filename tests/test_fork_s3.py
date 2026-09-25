"""A fork from a parent that lives in a bucket (GPU-159).

On a platform of rented machines the parent was written somewhere else: its
store is a prefix in the bucket, and a fork on a new machine names it by the
``store_uri`` the parent reported - ``s3://bucket/prefix``. The copy comes down,
the step is read from it, and the copy goes away.

The address parsing is tested everywhere; the fork itself needs an S3 endpoint
and skips without one, with the same setup as ``tests/test_metrics_s3.py``::

    RAVEX_TEST_S3_ENDPOINT=http://127.0.0.1:9000 pytest tests/test_fork_s3.py
"""

import os
import uuid

import pytest
import torch

import ravex
import ravex.runs
from ravex._resume import ResumeStepMissing
from ravex._runtime import _bucket_uri

ENDPOINT = os.environ.get("RAVEX_TEST_S3_ENDPOINT")
BUCKET = os.environ.get("RAVEX_TEST_S3_BUCKET", "ravex-test")
ACCESS = os.environ.get("RAVEX_TEST_S3_ACCESS_KEY", "minioadmin")
SECRET = os.environ.get("RAVEX_TEST_S3_SECRET_KEY", "minioadmin")

needs_s3 = pytest.mark.skipif(
    not ENDPOINT, reason="RAVEX_TEST_S3_ENDPOINT is not set; no S3 endpoint to test against"
)


class TestTheAddress:
    def test_a_path_is_not_a_bucket(self, tmp_path):
        assert _bucket_uri(str(tmp_path)) is None
        assert _bucket_uri("runs/harold-001") is None

    def test_bucket_and_prefix(self):
        assert _bucket_uri("s3://gpuzero/runs/harold-001") == ("gpuzero", "runs/harold-001")
        assert _bucket_uri("s3://gpuzero/runs/harold-001/") == ("gpuzero", "runs/harold-001")
        assert _bucket_uri("s3://gpuzero") == ("gpuzero", "")

    def test_no_bucket_is_refused(self):
        with pytest.raises(ValueError):
            _bucket_uri("s3:///runs/harold-001")


def train_to(total, lr=0.1, seen=None):
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


def in_bucket(monkeypatch, path, prefix):
    monkeypatch.setenv("RAVEX_STORAGE_TYPE", "s3")
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(path))
    monkeypatch.setenv("RAVEX_STORAGE_BUCKET", BUCKET)
    monkeypatch.setenv("RAVEX_STORAGE_PREFIX", prefix)
    monkeypatch.setenv("RAVEX_STORAGE_ENDPOINT", ENDPOINT or "")
    monkeypatch.setenv("RAVEX_STORAGE_PATH_STYLE", "true")
    monkeypatch.setenv("RAVEX_S3_ACCESS_KEY", ACCESS)
    monkeypatch.setenv("RAVEX_S3_SECRET_KEY", SECRET)


@needs_s3
class TestForkFromABucket:
    def test_a_fork_on_another_machine_starts_from_the_parents_step(self, tmp_path, monkeypatch):
        pytest.importorskip("moonclip")
        base = "fork-test-" + uuid.uuid4().hex[:8]
        # The parent, on "one machine": a staging directory that is deleted
        # afterwards, so the only copy left is the bucket's.
        parent_dir = tmp_path / "machine-a" / "base"
        in_bucket(monkeypatch, parent_dir, base + "/base")
        ravex.train_loop(backend="moonclip", checkpoint_every=2, async_save=False, name="base")(
            lambda: train_to(6)
        )()
        parent_id = ravex.runs.describe(str(parent_dir))["run_id"]
        import shutil

        shutil.rmtree(tmp_path / "machine-a")

        # The fork, on "another": nothing of the parent on its disk.
        child_dir = tmp_path / "machine-b" / "child"
        in_bucket(monkeypatch, child_dir, base + "/child")
        seen = {}
        ravex.train_loop(
            backend="moonclip",
            checkpoint_every=2,
            async_save=False,
            fork_from="s3://%s/%s/base" % (BUCKET, base),
            fork_step=4,
        )(lambda: train_to(8, seen=seen))()

        assert seen["start"] == 4
        described = ravex.runs.describe(str(child_dir))
        assert described["parent"] == {"run": parent_id, "step": 4}
        assert described["status"]["state"] == "finished"
        # The copy of the parent went away with the fork's start.
        assert [n for n in os.listdir(tmp_path / "machine-b") if n.startswith(".fork-parent-")] == []

    def test_a_prefix_with_nothing_in_it_stops_the_run(self, tmp_path, monkeypatch):
        pytest.importorskip("moonclip")
        base = "fork-test-" + uuid.uuid4().hex[:8]
        in_bucket(monkeypatch, tmp_path / "child", base + "/child")
        with pytest.raises(ResumeStepMissing):
            ravex.train_loop(
                backend="moonclip",
                checkpoint_every=2,
                async_save=False,
                fork_from="s3://%s/%s/nobody" % (BUCKET, base),
            )(lambda: train_to(4))()
        assert [n for n in os.listdir(tmp_path) if n.startswith(".fork-parent-")] == []
