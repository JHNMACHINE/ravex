"""``ravex`` command line.

Its subject is the autoloader: a one-line ``.pth`` file in site-packages that
Python executes at interpreter startup. That file is what makes checkpointing
work without touching the training script, and since 0.0.4 **the wheel ships
it** — ``pip install ravex`` puts it there.

So ``enable`` is no longer how the autoloader arrives. What it is for is
putting it back after ``disable``, which is the command that still earns its
keep: somebody who does not want a line of ours running in every interpreter
of their environment should be able to say so, and to change their mind.

The property this used to protect — that installing the package changes
nothing for unrelated processes — is now defended in the file itself rather
than by withholding it. ``ravex._bootstrap`` imports only ``os`` and ``sys``,
looks for a ``ravex.yaml`` above the working directory, and installs nothing
at all when there is not one. Measured at about a millisecond, against roughly
twelve when it still pulled in ``typing``.
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
    print()
    print("Since 0.0.4 the wheel ships this file, so a normal install already")
    print("has it. This command is here to put it back after `ravex disable`.")
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
    print()
    print("Note that `pip install --upgrade ravex` will put it back: the file")
    print("ships in the wheel. Run `ravex disable` again after an upgrade, or")
    print("set RAVEX_ENABLED=0 to turn Ravex off without removing anything.")
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

    subparsers.add_parser(
        "enable", help="reinstall the startup autoloader (the wheel ships it)"
    ).set_defaults(
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
