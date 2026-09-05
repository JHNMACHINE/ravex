"""``ravex`` command line.

One subcommand, and it answers one question: *what would Ravex do if I ran my
training script from here?* Which config file it would find, what the resolved
settings are, and whether the pieces it depends on are installed.

Until 0.0.5 this had two more — ``enable`` and ``disable`` — whose subject was a
``.pth`` file in site-packages that ran in every interpreter on the machine.
That file is gone (see the changelog), and with it the whole category of
question the CLI used to exist for: whether Ravex was armed, where, and how to
disarm it. There is nothing to disarm now. A training run is checkpointed if its
entry point is decorated with ``@ravex.train_loop`` and not otherwise, which is
a fact you can read in the source rather than one you have to interrogate the
environment about.

``status`` survives because config resolution is still worth being able to ask
about: ``ravex.yaml`` is searched for *above the working directory*, and
``RAVEX_*`` variables win over it, so "which settings am I actually going to
get" is not always obvious from where you are standing.
"""

from __future__ import annotations

import argparse
import os
import sys


def _status(_args) -> int:
    from ravex import __version__
    from ravex._config import RavexConfig, find_config_file

    print(f"ravex {__version__}")
    print(f"  python       {sys.version.split()[0]} ({sys.executable})")

    config_file = find_config_file()
    print(f"  config file  {config_file or 'none found'}")

    flag = os.environ.get("RAVEX_ENABLED")
    print(f"  RAVEX_ENABLED  {flag if flag is not None else '<unset>'}")

    config = RavexConfig.load()
    print(f"  resolved     {config.describe()}")

    # The compiled core is not optional since GPU-105, so its absence is a
    # broken install rather than a missing extra — worth saying plainly here,
    # because the alternative is an ImportError deep inside a resume.
    try:
        from ravex import _core

        print(f"  core         {_core.__version__} (compiled)")
    except ImportError as exc:
        print(f"  core         MISSING - {exc}")

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
        description="Checkpoint and resume for PyTorch training.",
        epilog=(
            "Ravex attaches to a training run through the @ravex.train_loop "
            "decorator on its entry point. Nothing on this command line turns "
            "it on or off."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "status", help="show what is configured and what is installed"
    ).set_defaults(handler=_status)

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
