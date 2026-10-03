"""The cache of compiled dependencies in a real bucket (GPU-181).

``tests/test_cache.py`` holds the cache to its rules against a directory;
this is the same round trip through ``_core.S3Store`` and an S3 server, which
is the only proof that the signing in ``src/s3.rs`` is one a service accepts.
It skips without an endpoint, with the same setup as ``tests/test_fork_s3.py``::

    RAVEX_TEST_S3_ENDPOINT=http://127.0.0.1:9000 pytest tests/test_cache_s3.py
"""

import os
import uuid

import pytest

from ravex import _cache
from ravex._cli import main

ENDPOINT = os.environ.get("RAVEX_TEST_S3_ENDPOINT")
BUCKET = os.environ.get("RAVEX_TEST_S3_BUCKET", "ravex-test")
ACCESS = os.environ.get("RAVEX_TEST_S3_ACCESS_KEY", "minioadmin")
SECRET = os.environ.get("RAVEX_TEST_S3_SECRET_KEY", "minioadmin")

pytestmark = pytest.mark.skipif(
    not ENDPOINT, reason="RAVEX_TEST_S3_ENDPOINT is not set; no S3 endpoint to test against"
)


@pytest.fixture
def bucket(monkeypatch):
    """A prefix of its own in the test bucket, and the keys in the environment."""
    monkeypatch.setenv("RAVEX_CACHE_ACCESS_KEY", ACCESS)
    monkeypatch.setenv("RAVEX_CACHE_SECRET_KEY", SECRET)
    monkeypatch.setenv("RAVEX_CACHE_ENDPOINT", ENDPOINT or "")
    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_89"])
    return "s3://%s/cache-%s" % (BUCKET, uuid.uuid4().hex[:8])


def test_what_one_machine_pushed_another_pulls(tmp_path, bucket, capsys):
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "ab").mkdir(parents=True)
    # A name with characters S3 escapes, and one past a single read buffer.
    (first / "ab" / "kernel a&b+1.cubin").write_bytes(b"\x7fELF" * 300_000)
    (first / "flash_attn-2.7.4+cu128.whl").write_bytes(b"wheel")

    assert main(["cache", "push", "--name", "triton", "--dir", str(first), "--store", bucket]) == 0
    assert main(["cache", "pull", "--name", "triton", "--dir", str(second), "--store", bucket]) == 0

    assert "hit" in capsys.readouterr().out.splitlines()[-1]
    assert (second / "ab" / "kernel a&b+1.cubin").read_bytes() == b"\x7fELF" * 300_000
    assert (second / "flash_attn-2.7.4+cu128.whl").read_bytes() == b"wheel"


def test_another_gpu_loads_nothing_from_the_bucket(tmp_path, bucket, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cubin").write_bytes(b"built for sm_89")
    store = _cache.open_store(bucket)
    _cache.push(store, _cache.compute_key("triton"), str(source))

    monkeypatch.setattr(_cache, "gpu_capabilities", lambda: ["sm_90"])
    got = _cache.pull(store, _cache.compute_key("triton"), str(tmp_path / "other"))

    assert not got.hit
    assert not (tmp_path / "other").exists()


def test_a_missing_object_is_none_not_an_error(bucket):
    store = _cache.open_store(bucket)

    assert store.get("nothing/here") is None
    assert store.list("nothing/") == []


def test_wrong_keys_are_an_error_not_a_miss(tmp_path, bucket, monkeypatch):
    """A cache that turned a refused signature into "nothing stored" would
    have every node compile from scratch, forever, and say nothing."""
    monkeypatch.setenv("RAVEX_CACHE_SECRET_KEY", "not-the-secret")
    store = _cache.open_store(bucket)

    with pytest.raises(OSError):
        _cache.pull(store, _cache.compute_key("triton"), str(tmp_path / "x"))
