"""Every ravex name the integration kit reaches for still exists.

The kit under ``integration/`` is not installed with the package and most of it
only runs on rented GPUs, so a rename inside ravex does not break anything here:
it breaks on a box that is already being paid for. That happened twice.
``ravex_timing.py`` imported ``ravex._replication`` for a whole release after
GPU-105 moved it under ``_dist`` (GPU-130), and ``gpu92_sigterm.py`` asked for
``ravex._runtime_instance``, which never existed, and reported ``active=False``
instead of failing.

So this reads the kit instead of running it. Every ``import`` of a ravex module
and every attribute chain hanging off a name bound to one, such as
``_backends.MoonclipBackend.consolidate``, is resolved against the installed
package. Nothing in the kit is executed, which is the point: the scripts want
CUDA, NCCL and a second machine, while the names they touch can be checked here.

What it cannot see: names built at runtime (``getattr(ravex, name)``), and a
local variable that shadows an imported ravex name. The second would show up as
a false failure, not a silent pass.
"""

import ast
import importlib
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent.parent / "integration"
SOURCES = sorted(p for p in KIT.rglob("*.py") if "__pycache__" not in p.parts)


def is_ravex(module):
    return module == "ravex" or module.startswith("ravex.")


def resolve(path):
    """Walk a dotted ravex path, importing submodules the way Python would."""
    obj = importlib.import_module(path[0])
    walked = path[0]
    for part in path[1:]:
        walked = walked + "." + part
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        if not isinstance(obj, type(importlib)):
            raise AttributeError(walked)
        try:
            obj = importlib.import_module(walked)
        except ModuleNotFoundError as exc:
            if exc.name != walked:
                raise
            raise AttributeError(walked) from None
    return obj


def chain(node):
    """``a.b.c`` as ``["a", "b", "c"]``, or None when the root is not a name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def ravex_references(tree):
    """``(lineno, dotted path)`` for every ravex name the source relies on."""
    bound = {}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_ravex(alias.name):
                    continue
                found.append((node.lineno, alias.name.split(".")))
                if alias.asname:
                    bound[alias.asname] = alias.name.split(".")
                else:
                    bound["ravex"] = ["ravex"]
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or not is_ravex(node.module):
                continue
            base = node.module.split(".")
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.append((node.lineno, base + [alias.name]))
                bound[alias.asname or alias.name] = base + [alias.name]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # `_backends.MoonclipBackend.consolidate = wrapper` creates the last
        # name; what has to exist beforehand is everything to its left.
        target = node.value if isinstance(node.ctx, ast.Store) else node
        parts = chain(target)
        if parts is None or parts[0] not in bound:
            continue
        found.append((node.lineno, bound[parts[0]] + parts[1:]))
    return found


def test_the_kit_is_where_this_test_looks():
    assert any(p.name == "ravex_timing.py" for p in SOURCES)


@pytest.mark.parametrize(
    "source", SOURCES, ids=[p.relative_to(KIT).as_posix() for p in SOURCES]
)
def test_every_ravex_name_the_kit_uses_exists(source):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    missing = []
    for lineno, path in ravex_references(tree):
        try:
            resolve(path)
        except AttributeError:
            missing.append("line %d: %s" % (lineno, ".".join(path)))
    assert not missing, "%s uses ravex names that no longer exist:\n%s" % (
        source.relative_to(KIT).as_posix(),
        "\n".join(sorted(set(missing))),
    )
