"""Metrics reach the bucket while the run is still going (GPU-152).

Needs an S3 endpoint, so it skips itself without one - the same arrangement as
Moonclip's ``tests/s3_minio.rs``. Locally::

    mkdir -p "$DATA/ravex-test"
    docker run -d --name vgw -p 9000:9000 -v "$DATA:/data" \\
        -e ROOT_ACCESS_KEY=minioadmin -e ROOT_SECRET_KEY=minioadmin \\
        --entrypoint sh versity/versitygw -c "versitygw --port :9000 posix /data"
    RAVEX_TEST_S3_ENDPOINT=http://127.0.0.1:9000 RAVEX_TEST_S3_DATA_DIR="$DATA" \\
        pytest tests/test_metrics_s3.py

The image listens on 7070 unless told otherwise, hence ``--port``.

Two ways of reading the bucket back, for two different questions:

* **During the run**, the object keys themselves, handed to
  :func:`ravex.metrics.resolve` the way the dashboard's Worker does. That needs
  the bucket's files: versitygw's ``posix`` backend keeps one file per object,
  so with its data directory mounted from the host, set
  ``RAVEX_TEST_S3_DATA_DIR`` to it. Moonclip cannot answer this one - its
  ``restore_from_remote`` pulls nothing until the bucket holds a manifest, and
  mid-run it may not yet.
* **After the run**, through Moonclip: a second manager on an empty directory,
  ``restore_from_remote()``. That is how a machine that lost its disk gets its
  store back, so it is what a replacement node would see.
"""

import os
import time
import uuid

import pytest
import torch

import ravex
import ravex.metrics

ENDPOINT = os.environ.get("RAVEX_TEST_S3_ENDPOINT")
BUCKET = os.environ.get("RAVEX_TEST_S3_BUCKET", "ravex-test")
DATA_DIR = os.environ.get("RAVEX_TEST_S3_DATA_DIR")
ACCESS = os.environ.get("RAVEX_TEST_S3_ACCESS_KEY", "minioadmin")
SECRET = os.environ.get("RAVEX_TEST_S3_SECRET_KEY", "minioadmin")

pytestmark = pytest.mark.skipif(
    not ENDPOINT, reason="RAVEX_TEST_S3_ENDPOINT is not set; no S3 endpoint to test against"
)


@pytest.fixture
def remote(monkeypatch, tmp_path):
    prefix = "metrics-test-" + uuid.uuid4().hex[:8]
    monkeypatch.setenv("RAVEX_STORAGE_TYPE", "s3")
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "staging"))
    monkeypatch.setenv("RAVEX_STORAGE_BUCKET", BUCKET)
    monkeypatch.setenv("RAVEX_STORAGE_PREFIX", prefix)
    monkeypatch.setenv("RAVEX_STORAGE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("RAVEX_STORAGE_PATH_STYLE", "true")
    monkeypatch.setenv("RAVEX_S3_ACCESS_KEY", ACCESS)
    monkeypatch.setenv("RAVEX_S3_SECRET_KEY", SECRET)
    return prefix


def pull(prefix, into):
    """Everything under ``prefix`` in the bucket, copied into ``into``."""
    import moonclip

    manager = moonclip.MoonclipManager(
        storage_root=str(into),
        s3_bucket=BUCKET,
        s3_prefix=prefix,
        s3_endpoint=ENDPOINT,
        s3_access_key=ACCESS,
        s3_secret_key=SECRET,
        s3_path_style=True,
    )
    manager.restore_from_remote()
    return into


def bucket_chunks(prefix):
    """The metric objects under ``prefix``, keyed as a Worker would list them."""
    root = os.path.join(DATA_DIR, BUCKET, prefix)
    chunks = {}
    for directory, _dirs, files in os.walk(os.path.join(root, "metrics")):
        for name in files:
            path = os.path.join(directory, name)
            key = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, encoding="utf-8") as handle:
                chunks[key] = handle.read()
    return chunks


@pytest.mark.skipif(not DATA_DIR, reason="RAVEX_TEST_S3_DATA_DIR is not set")
def test_metrics_are_in_the_bucket_before_the_run_ends(remote):
    seen = []

    @ravex.train_loop(checkpoint_every=2, metrics_chunk_every=0.2)
    def train():
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(4):
            ravex.batch_boundary()
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            ravex.log_metrics({"train/loss": float(ravex.step())})
        # Long enough for a chunk to close and for the sync thread to take it.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            loss = ravex.metrics.resolve(bucket_chunks(remote))["scalars"].get("train/loss")
            seen.append(loss)
            if loss and loss["step"] == [1, 2, 3, 4]:
                return
            time.sleep(0.25)

    train()
    assert seen and seen[-1] and seen[-1]["step"] == [1, 2, 3, 4], (
        "the bucket never held the run's metrics while it was running: %r" % seen[-3:]
    )


def test_the_bucket_resolves_to_the_same_timeline_as_the_disk(remote, tmp_path):
    @ravex.train_loop(checkpoint_every=5, checkpoint_on_exit=False)
    def first():
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(8):
            ravex.batch_boundary()
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            ravex.log_metrics({"loss": float(ravex.step())})
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        first()

    @ravex.train_loop(checkpoint_every=5)
    def second():
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            ravex.batch_boundary()
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            ravex.log_metrics({"loss": ravex.step() + 100.0})

    second()
    staging = os.environ["RAVEX_STORAGE_PATH"]
    mirror = pull(remote, tmp_path / "after")
    from_bucket = ravex.metrics.read(str(mirror))["scalars"]["loss"]
    assert from_bucket == ravex.metrics.read(staging)["scalars"]["loss"]
    assert from_bucket["value"] == [1.0, 2.0, 3.0, 4.0, 5.0, 106.0, 107.0, 108.0]
