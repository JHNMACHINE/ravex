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

``export`` is the other kind of question a command line is for: an operation on
a store after the run that wrote it is gone. It writes a checkpoint out as a
``torch.distributed.checkpoint`` directory (GPU-90) — the format other
frameworks read, where Ravex's own store is a format only Ravex reads.

``audit`` is the same kind: reading, listing and verifying the audit log a run
with ``audit_log: true`` left in its store (GPU-93). The person who needs it is
usually not the one who ran the training, and is asking months later.
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


def _export(args) -> int:
    from ravex._interop.convert import CannotConvert
    from ravex._interop.export import load_store, to_dcp

    # Refused rather than written into. DCP names its files by rank and
    # overwrites its metadata, so an export on top of an earlier one leaves
    # the earlier one's extra files beside a manifest that no longer lists
    # them — a directory that loads, and holds more than it says.
    if os.path.isdir(args.out) and os.listdir(args.out):
        print(
            f"ravex export: {args.out} exists and is not empty; export into a "
            "new directory",
            file=sys.stderr,
        )
        return 2

    try:
        state = load_store(args.storage, backend=args.backend, step=args.step)
        notes = to_dcp(state, args.out, key=args.model)
    except CannotConvert as exc:
        print(f"ravex export: {exc}", file=sys.stderr)
        return 2

    print(f"exported step {state.get('step')} of {args.storage} to {args.out}")
    for note in notes:
        print(f"  {note}")
    return 0


def _audit_log(args) -> int:
    import json

    from ravex import _audit as audit

    path = args.storage
    if os.path.isdir(path):
        path = os.path.join(path, audit.AUDIT_FILE)
    if not os.path.exists(path):
        print(
            f"ravex audit: no audit log at {path} - it is only written by a run "
            "started with audit_log: true (RAVEX_AUDIT_LOG=1). Under "
            "sharded_checkpoints: per_rank each rank has its own, in rank_<n>/",
            file=sys.stderr,
        )
        return 2

    if args.action == "verify":
        problems = audit.verify(path)
        if problems:
            for problem in problems:
                print(f"BROKEN  {problem}")
            return 1
        entries = audit.read_entries(path)
        if not entries:
            print("intact: 0 entries")
            return 0
        print(
            f"intact: {len(entries)} entries, steps {entries[0]['step']} to "
            f"{entries[-1]['step']}"
        )
        # Printed because it is the one thing the chain cannot check by itself:
        # a file rewritten from scratch verifies too. Kept somewhere the writer
        # of this file cannot reach, it can.
        print(f"last entry_sha256 {entries[-1]['entry_sha256']}")
        print("  keep this value outside the store; it is what proves the log was not rewritten")
        return 0

    try:
        entries = audit.read_entries(path)
    except ValueError as exc:
        print(f"ravex audit: {path} is not a readable audit log ({exc}); run verify", file=sys.stderr)
        return 2

    if args.action == "list":
        for entry in entries:
            fingerprint = entry.get("fingerprint") or "unavailable"
            print(
                f"step {entry.get('step'):>10}  {entry.get('written_at')}  "
                f"{fingerprint[:16]}  {entry.get('fingerprint_kind')}"
            )
        return 0

    if not args.fingerprint:
        print("ravex audit find: give a fingerprint, or its first characters", file=sys.stderr)
        return 2
    matches = audit.find(path, args.fingerprint)
    if not matches:
        print(f"ravex audit: no checkpoint with fingerprint {args.fingerprint} in {path}", file=sys.stderr)
        return 1
    for entry in matches:
        print(json.dumps(entry, indent=2, sort_keys=True))
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

    export = subparsers.add_parser(
        "export",
        help="write a checkpoint out as a torch distributed checkpoint (DCP)",
    )
    export.add_argument("--storage", required=True, help="the Ravex store to read")
    export.add_argument("--out", required=True, help="a new directory to write the DCP into")
    export.add_argument(
        "--backend",
        default="moonclip",
        choices=("moonclip", "torch_save"),
        help="the backend that wrote the store (default: moonclip)",
    )
    export.add_argument("--step", type=int, default=None, help="default: the latest")
    export.add_argument(
        "--model", default=None, help="which model, when the checkpoint holds several"
    )
    export.set_defaults(handler=_export)

    audit = subparsers.add_parser(
        "audit", help="verify, list or search a store's audit log (audit_log: true)"
    )
    audit.add_argument("action", choices=("verify", "list", "find"))
    audit.add_argument(
        "fingerprint", nargs="?", default=None, help="for find: a fingerprint, or its first characters"
    )
    audit.add_argument(
        "--storage", required=True, help="the store directory, or the audit.jsonl inside it"
    )
    audit.set_defaults(handler=_audit_log)

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
