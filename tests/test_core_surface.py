"""What the compiled core exports, what the stub declares, what the shims re-export.

Three lists that have to be the same list, kept in files that no compiler checks
against each other. GPU-105 moved the reshard planner into Rust and left
``ravex/_dist/reshard.py`` as the name it is imported by; GPU-109 did the same
for the replica transport, behind ``ravex/_dist/replication.py``. The risk that
came with both is quiet and one-directional.

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

PACKAGE = Path(__file__).resolve().parent.parent / "ravex"
STUB = PACKAGE / "_core.pyi"
#: The modules that stand in front of the transport half of the core. Two,
#: because the framing is `replication`'s and the socket that carries it
#: between machines is `elastic`'s.
SHIMS = (PACKAGE / "_dist" / "replication.py", PACKAGE / "_dist" / "elastic.py")

#: Not part of the exported surface: dunders, and the version the extension
#: carries for `test_version_is_single.py` to check.
PRIVATE = {"__version__"}


def public(names) -> set:
    return {name for name in names if not name.startswith("_")} - PRIVATE


def declared_in_the_stub() -> set:
    """Top-level names in ``_core.pyi``: functions, classes and constants.

    Parsed rather than imported: a ``.pyi`` is not importable, and reading it
    with ``ast`` is what a type checker does with it anyway.

    Annotated constants count: ``CHUNK`` and ``COMPLETE_MARKER`` are module
    attributes of the extension rather than callables, and a stub that omitted
    them would leave two of the names this file tracks outside the comparison.

    A *plain* assignment does not, and the difference is the whole reason this
    reads the syntax rather than importing anything: `CHUNK: int` declares
    something the extension exports, while `Placement = Dict[str, Any]` is a
    type alias that lives on this side of the boundary and must not — see
    `test_the_type_aliases_stay_python_side`.
    """
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    declared = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            declared.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
    # The same filter both sides get, so `__version__` — declared here and
    # exported there — is out of the comparison on both.
    return public(declared)


def test_the_stub_declares_exactly_what_the_extension_exports():
    assert declared_in_the_stub() == public(dir(_core))


def test_no_name_is_stranded_inside_the_extension():
    """Every compiled name is reachable from the module it belongs to.

    Three shims now, and they do not answer the question the same way. The
    planner's names are re-exported one for one, so ``reshard.__all__`` is the
    list. The transport's are not: ``ravex._dist.replication`` has always
    spelled two of them with a leading underscore and has always hidden the
    encoder behind ``encode_store``. So what is checked for those is that a
    module *reaches* the name — a `_core.something` nothing mentions is one
    nobody will find, which is the failure this test is for either way.
    """
    reached = "".join(shim.read_text(encoding="utf-8") for shim in SHIMS)
    for name in public(dir(_core)):
        assert name in set(reshard.__all__) or ("_core.%s" % name) in reached, name


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
