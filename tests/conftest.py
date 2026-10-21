import os
import sys

import pytest

import ravex

# The suite is in two directories (GPU-256), and a test of one may import the
# helpers of a test module in the other - the distributed half of a subject
# takes its fakes from the unit half. Both are put on the path here, once, so
# that does not depend on which directory pytest happens to collect first.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _directory in ("unit", "distributed"):
    _path = os.path.join(_HERE, _directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)


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
