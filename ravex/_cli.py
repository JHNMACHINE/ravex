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

``ship`` too acts on a store whose run is gone: it sends a backend what that
run could not (GPU-156), for a run that ended while its backend was down.
"""

from __future__ import annotations

import argparse
import json
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


def _rendezvous(args) -> int:
    """Hold the store outer-loop nodes meet at, until interrupted (GPU-129)."""
    import time

    from ravex._dist import rendezvous

    try:
        server = rendezvous.serve(args.host, args.port)
    except Exception as exc:
        print(
            f"ravex rendezvous: could not listen on {args.host}:{args.port}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        f"Rendezvous listening on {args.host}:{args.port}. Nodes meet here with "
        f"outer_rendezvous (or {rendezvous.ENV}) set to this machine's address "
        f"and port {args.port}.",
        flush=True,
    )
    if rendezvous.job_token() is not None:
        print(
            f"Gated by {rendezvous.TOKEN_ENV}: only nodes started with the same "
            "one get in. Jobs on this server share it, so run one server per "
            "run unless its jobs trust each other.",
            flush=True,
        )
    else:
        print(
            f"No authentication: {rendezvous.TOKEN_ENV} is not set, so whoever "
            "reaches this port can take a number, write any key and read the "
            "job token. Keep it on a private network, or set the token here "
            "and on every node.",
            flush=True,
        )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        del server


def _held_steps(storage):
    """The steps a local store holds, read the way the run that wrote it would.

    ``None`` when that cannot be said from here: a store in a bucket needs its
    credentials, and a run.json without a config does not say which backend
    wrote it. The metrics still go; only the checkpoint list is left alone.
    """
    from ravex._backends import get_backend
    from ravex._config import RavexConfig

    try:
        with open(os.path.join(storage, "run.json"), encoding="utf-8") as handle:
            written = json.load(handle).get("config") or {}
    except (OSError, ValueError):
        return None
    if not written.get("backend") or (written.get("storage") or {}).get("type", "local") != "local":
        return None
    config = RavexConfig()
    config.storage.path = storage
    config.backend = written["backend"]
    config._normalize()
    try:
        store = get_backend(config)
        try:
            return store.known_steps()
        finally:
            store.close()
    except Exception as exc:
        print(f"ravex ship: cannot list the checkpoints in {storage}: {exc}", file=sys.stderr)
        return None


def _ship(args) -> int:
    """Send a backend what a store's executions never got to send (GPU-156).

    A run posts itself while it trains, but closing does not wait out a backend
    that is down: a run that has finished training should not sit there because
    the dashboard is. What was left stays on disk for "the next execution on
    this store" - which a finished run never gets. This is that execution
    without the training: whoever outlives the run (the platform's agent, or a
    person) calls it until it says everything arrived.
    """
    from ravex._ship import CLOSE_SECONDS, Shipper

    if not os.path.isfile(os.path.join(args.storage, "run.json")):
        print(f"ravex ship: {args.storage} holds no run.json, so it is not a run's store", file=sys.stderr)
        return 2
    token = args.token or os.environ.get("RAVEX_METRICS_TOKEN") or None
    # recover=True is the whole command: it queues every chunk the ledgers do
    # not list as confirmed, and marks the run document for sending, so the
    # final status goes too.
    shipper = Shipper(args.endpoint, args.storage, token=token, recover=True)
    steps = _held_steps(args.storage)
    if steps is not None:
        # The list a resume is picked from. The backend's copy is whatever the
        # run last managed to send, which after retention can name steps the
        # store no longer holds - a resume the page offers and Ravex refuses.
        shipper.checkpoints(steps)
    shipper.close(timeout=args.wait if args.wait is not None else CLOSE_SECONDS)
    left = shipper.pending
    if left:
        # Nothing is lost: the chunks stay, unconfirmed, for the next attempt.
        print(f"ravex ship: {left} chunk(s) of {args.storage} did not reach {args.endpoint}", file=sys.stderr)
        return 1
    print(f"{args.storage}: everything it holds is at {args.endpoint}")
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

    rendezvous = subparsers.add_parser(
        "rendezvous",
        help="hold the store outer-loop nodes meet at, without torchrun (outer_rendezvous)",
    )
    rendezvous.add_argument(
        "--host", default="0.0.0.0", help="the interface to listen on (default: all)"
    )
    rendezvous.add_argument("--port", type=int, default=29400, help="default: 29400")
    rendezvous.set_defaults(handler=_rendezvous)

    ship = subparsers.add_parser(
        "ship",
        help="send a backend what a store's runs never got to send (metrics_endpoint)",
    )
    ship.add_argument("--storage", required=True, help="the run's store")
    ship.add_argument("--endpoint", required=True, help="the backend, as metrics_endpoint")
    ship.add_argument(
        "--token", default=None, help="default: RAVEX_METRICS_TOKEN, as metrics_token"
    )
    ship.add_argument(
        "--wait",
        type=float,
        default=None,
        help="seconds to spend on a backlog that is going out (default: as at the end of a run)",
    )
    ship.set_defaults(handler=_ship)

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
