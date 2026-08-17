"""What the autoloader decides before any of Ravex exists.

The .pth hook runs in every Python process on the machine, so the questions here
are all about *not* acting: not for a process that never opted in, and not for
one that starts a training run without ever doing any training itself.
"""

import sys

from ravex import _bootstrap

PYTHON = "/usr/bin/python"


def launched_as(monkeypatch, *arguments, argv0="train.py"):
    """Make this process look like it was started with `python <arguments>`.

    ``orig_argv`` is the whole command line as the interpreter received it;
    ``argv`` is what the script sees. They differ, and the difference is the
    point — see ``_is_launcher_process``.
    """
    monkeypatch.setattr(sys, "orig_argv", [PYTHON, *arguments], raising=False)
    monkeypatch.setattr(sys, "argv", [argv0])


def test_the_torchrun_launcher_is_not_a_rank(monkeypatch):
    """The launcher imports torch to parse its own arguments, so it reaches the
    autoloader seconds before the ranks it is about to spawn.

    It has never done damage - it holds no model, so it never checkpoints and
    never opens the backend - but that is by construction, not by design. One
    future path that opens storage before objects are registered and the
    launcher is creating directories, or S3 connections, for a process that will
    never train a step.
    """
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    # `python -m torch.distributed.run script.py`. At the moment this runs,
    # argv[0] is still "-m" and __main__ has no spec: runpy pulled torch in
    # while resolving the module, before it rewrote either.
    launched_as(monkeypatch, "-m", "torch.distributed.run", "train.py", argv0="-m")
    assert _bootstrap._is_launcher_process()

    # The deprecated launcher, and an interpreter option before the -m.
    launched_as(monkeypatch, "-m", "torch.distributed.launch", "train.py", argv0="-m")
    assert _bootstrap._is_launcher_process()
    launched_as(monkeypatch, "-u", "-m", "torch.distributed.run", argv0="-m")
    assert _bootstrap._is_launcher_process()

    # The console script: a shebang wrapper, so the module name never appears
    # and argv[0] is what gives it away.
    launched_as(monkeypatch, "/usr/local/bin/torchrun", "-c", argv0="/usr/local/bin/torchrun")
    assert _bootstrap._is_launcher_process()
    launched_as(monkeypatch, r"C:\venv\Scripts\torchrun.exe", argv0=r"C:\venv\Scripts\torchrun.exe")
    assert _bootstrap._is_launcher_process()


def test_the_ranks_it_spawns_are(monkeypatch):
    """The expensive mistake would be the mirror image of the bug: a worker
    mistaken for a launcher trains a whole run with no checkpoints at all."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    launched_as(monkeypatch, "-u", "train.py", "--epochs", "3")
    assert not _bootstrap._is_launcher_process()

    # A worker started by the old launcher gets --local_rank as an argument
    # rather than an env var, so the structural test has to stand on its own.
    monkeypatch.delenv("LOCAL_RANK")
    launched_as(monkeypatch, "train.py", "--local_rank=3")
    assert not _bootstrap._is_launcher_process()


def test_a_plain_script_is_not_a_launcher(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    launched_as(monkeypatch, "train.py")
    assert not _bootstrap._is_launcher_process()

    # A training script run as a module, and one whose own arguments happen to
    # mention the launcher.
    launched_as(monkeypatch, "-m", "mypackage.train", argv0="-m")
    assert not _bootstrap._is_launcher_process()
    launched_as(monkeypatch, "train.py", "-m", "torch.distributed.run")
    assert not _bootstrap._is_launcher_process()

    # An interpreter with no script at all, and one whose orig_argv is missing
    # (Python 3.9): neither may be read as a launcher.
    monkeypatch.setattr(sys, "orig_argv", [PYTHON], raising=False)
    monkeypatch.setattr(sys, "argv", [""])
    assert not _bootstrap._is_launcher_process()
    monkeypatch.delattr(sys, "orig_argv", raising=False)
    monkeypatch.setattr(sys, "argv", ["train.py"])
    assert not _bootstrap._is_launcher_process()
