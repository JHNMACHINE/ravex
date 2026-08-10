import os

import pytest

import ravex


@pytest.fixture(autouse=True)
def clean_runtime():
    """Every test starts with no runtime and leaves torch unpatched."""
    ravex.deactivate()
    yield
    ravex.deactivate()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Keep tests away from any ravex.yaml or RAVEX_* on the machine."""
    for name in list(os.environ):
        if name.startswith("RAVEX_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAVEX_CONFIG", str(tmp_path / "does-not-exist.yaml"))


@pytest.fixture
def storage(monkeypatch, tmp_path):
    """Point checkpoints at a per-test directory."""
    path = tmp_path / "checkpoints"
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(path))
    return path
