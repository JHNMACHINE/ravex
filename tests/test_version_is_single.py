"""One version number, written in two files that maturin forces apart.

Until GPU-105 there was genuinely one place: ``ravex/__init__.py`` held the
literal and ``[tool.setuptools.dynamic]`` read it. maturin has no equivalent —
it takes the wheel's version from ``Cargo.toml`` and will not read a Python
attribute — so the literal has to be repeated, and the repetition has to be
guarded.

What the drift looks like if it is not: ``ravex status`` and ``ravex --version``
print ``ravex.__version__``, so a bump made in ``Cargo.toml`` alone ships a
wheel whose CLI reports the *previous* release. That is worse than an obviously
wrong number, because the installed version and the reported one are both
plausible and only one of them is true.

The tests read the file rather than importing anything that parses TOML:
``tomllib`` is 3.11+ and would be fine, but ``Cargo.toml`` is not a Python
manifest and a regex over one line is the whole of what is being asserted.
"""

import re
from pathlib import Path

import pytest

import ravex

REPO = Path(__file__).resolve().parent.parent


def first_version_in(path: Path) -> str:
    """The first top-level ``version = "..."`` line, which is the package's.

    First rather than only: ``Cargo.toml`` has a ``version`` under every
    dependency, and they are all below ``[package]``.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if match is not None:
            return match.group(1)
    pytest.fail("no top-level version line in %s" % path.name)


def test_cargo_and_the_package_agree():
    assert first_version_in(REPO / "Cargo.toml") == ravex.__version__


def test_the_compiled_core_reports_the_same_version():
    """The extension carries ``CARGO_PKG_VERSION``, baked in when it was built.

    This is the one of the three that cannot be satisfied by editing a file: it
    fails when the ``.pyd``/``.so`` in the tree was compiled from a different
    version of the source than the checkout being tested, which is what a stale
    editable install looks like from the inside.
    """
    from ravex import _core

    assert _core.__version__ == ravex.__version__


def test_pyproject_still_takes_the_version_from_cargo():
    """`dynamic = ["version"]`, not a third literal.

    If someone answers a maturin warning by pasting the number into
    ``[project]`` as well, this is the test that says no: two guarded copies are
    the cost of the build backend, three is a choice.
    """
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert not re.search(r'^version\s*=\s*"', pyproject, re.MULTILINE)


def test_the_moonclip_floor_covers_what_the_round_exchange_calls():
    """The floor is a claim about methods, and one of them has no fallback.

    ``_dist/report.py`` calls ``describe()`` on a peer's store to check the
    payload before allocating it, and Moonclip added ``describe()`` in 0.1.0.
    Against an older engine that is an ``AttributeError`` inside the
    ``except Exception`` that treats a peer as unreadable — so every round
    closes with zero contributors, each node trains its own model, and every
    log line says the round closed. A version pin is the only place that can
    be caught, which is the same argument the ``>=0.0.8`` note makes about
    ``restore_from_remote``.

    Two halves, and both are needed: the floor has to *say* 0.1.0, and the
    engine actually installed has to *have* the method. The first would pass on
    a machine where the second is false, which is the machine the tests are
    running on.
    """
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"moonclip>=(\d+)\.(\d+)\.(\d+)"', pyproject)
    assert match is not None, "no moonclip floor in pyproject.toml"
    assert tuple(int(part) for part in match.groups()) >= (0, 1, 0), (
        "the floor is below the release that added describe(), which "
        "_dist/report.py calls without a fallback"
    )

    moonclip = pytest.importorskip("moonclip")
    assert hasattr(moonclip.MoonclipManager, "describe")
