"""One version number, written once.

``ravex/__init__.py`` holds the literal and the build reads it from there
(``[tool.hatch.version]``). From GPU-105 to GPU-267 maturin forced a second
copy into ``Cargo.toml``, and this file guarded the two against drifting: a
bump made in one alone ships a wheel whose CLI reports another release. With
the compiled core gone there is one copy again, and what is left to guard is
that nobody adds a second.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_pyproject_takes_the_version_from_the_package():
    """`dynamic = ["version"]`, read from `ravex/__init__.py`, not a literal."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "ravex/__init__.py"' in pyproject
    assert not re.search(r'^version\s*=\s*"', pyproject, re.MULTILINE)


def test_there_is_no_cargo_manifest_to_drift_from():
    assert not (REPO / "Cargo.toml").exists()
