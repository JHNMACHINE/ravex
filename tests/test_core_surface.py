"""What the compiled core exports, what the stub declares, what the shim re-exports.

Three lists that have to be the same list, kept in three files that no compiler
checks against each other. GPU-105 moved the reshard planner into Rust and left
``ravex/_dist/reshard.py`` as the name it is imported by; the risk that came
with that is quiet and one-directional.

* A function added to ``src/python.rs`` and not to the shim is simply not
  reachable — it exists in the wheel and no caller can see it.
* A name dropped from the Rust side and left in ``ravex/_core.pyi`` is worse: a
  stub that has drifted type-checks code against a function that is not there,
  so the editor is green and the import fails at runtime.

Neither shows up in the reshard suites, because those import the names that do
exist. This is the test that notices the ones that stopped existing.
"""

import ast
from pathlib import Path

from ravex import _core
from ravex._dist import reshard

STUB = Path(__file__).resolve().parent.parent / "ravex" / "_core.pyi"

#: Not part of the exported surface: dunders, and the version the extension
#: carries for `test_version_is_single.py` to check.
PRIVATE = {"__version__"}


def public(names) -> set:
    return {name for name in names if not name.startswith("_")} - PRIVATE


def declared_in_the_stub() -> set:
    """Top-level `def` and `class` names in ``_core.pyi``.

    Parsed rather than imported: a ``.pyi`` is not importable, and reading it
    with ``ast`` is what a type checker does with it anyway.
    """
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def test_the_stub_declares_exactly_what_the_extension_exports():
    assert declared_in_the_stub() == public(dir(_core))


def test_the_shim_re_exports_everything_the_core_has():
    """Nothing is stranded inside the extension.

    ``ravex._dist.reshard`` is the public name of this module and always has
    been; a planner function only reachable as ``ravex._core.something`` is one
    nobody will find.
    """
    assert public(dir(_core)) <= set(reshard.__all__)


def test_every_name_in_all_actually_resolves():
    """`__all__` is hand-written, so it can name something that is not there."""
    for name in reshard.__all__:
        assert hasattr(reshard, name), name


def test_the_type_aliases_stay_python_side():
    """``Placement``, ``Piece`` and ``Homes`` are annotations, not exports.

    They were type aliases in the Python module and they still are — the
    extension has no reason to carry them — but they are part of what
    ``ravex._dist.reshard`` offers, so ``__all__`` names them and this says so.
    """
    for alias in ("Placement", "Piece", "Homes"):
        assert alias in reshard.__all__
        assert not hasattr(_core, alias)
