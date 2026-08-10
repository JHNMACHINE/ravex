"""``ravex`` command line.

Its only real job is managing the autoloader: a one-line ``.pth`` file in
site-packages that Python executes at interpreter startup. That file is what
makes checkpointing work without touching the training script.

Keeping enable/disable explicit — rather than writing the ``.pth`` from a
post-install hook — means installing the package never changes the behaviour of
unrelated Python processes on the machine.
"""

from __future__ import annotations

import argparse
import os
import sys
import sysconfig
from pathlib import Path

PTH_NAME = "ravex_autoload.pth"
PTH_CONTENT = "import ravex._bootstrap\n"


def pth_path() -> Path:
    """Where the autoloader lives for the *current* interpreter."""
    return Path(sysconfig.get_paths()["purelib"]) / PTH_NAME


def _enable(_args) -> int:
    path = pth_path()
    try:
        path.write_text(PTH_CONTENT, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write {path}: {exc}", file=sys.stderr)
        print("Try again inside a virtualenv, or with elevated permissions.", file=sys.stderr)
        return 1

    print(f"Autoloader installed: {path}")
    print()
    print("Ravex now starts with every Python process in this environment,")
    print("but only activates for projects that have a ravex.yaml (or when")
    print("RAVEX_ENABLED=1 is set). Nothing else changes.")
    return 0


def _disable(_args) -> int:
    path = pth_path()
    if not path.exists():
        print(f"Autoloader is not installed ({path})")
        return 0
    try:
        path.unlink()
    except OSError as exc:
        print(f"Could not remove {path}: {exc}", file=sys.stderr)
        return 1
    print(f"Autoloader removed: {path}")
    return 0


def _status(_args) -> int:
    from ravex import __version__
    from ravex._bootstrap import should_activate
    from ravex._config import RavexConfig, find_config_file

    print(f"ravex {__version__}")
    print(f"  python       {sys.version.split()[0]} ({sys.executable})")

    path = pth_path()
    print(f"  autoloader   {'installed' if path.exists() else 'not installed'} ({path})")

    config_file = find_config_file()
    print(f"  config file  {config_file or 'none found'}")

    flag = os.environ.get("RAVEX_ENABLED")
    print(f"  RAVEX_ENABLED  {flag if flag is not None else '<unset>'}")
    print(f"  would activate here: {'yes' if should_activate() else 'no'}")

    config = RavexConfig.load()
    print(f"  resolved     {config.describe()}")

    for name in ("torch", "moonclip", "yaml"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "?")
            print(f"  {name:<12} {version}")
        except ImportError:
            print(f"  {name:<12} not installed")

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ravex",
        description="Transparent checkpointing for PyTorch training.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("enable", help="install the startup autoloader").set_defaults(
        handler=_enable
    )
    subparsers.add_parser("disable", help="remove the startup autoloader").set_defaults(
        handler=_disable
    )
    subparsers.add_parser("status", help="show what is installed and configured").set_defaults(
        handler=_status
    )

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
