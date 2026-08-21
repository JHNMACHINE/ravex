"""What the autoloader decides before any of Ravex exists.

The .pth hook runs in every Python process on the machine, so the questions here
are all about *not* acting: not for a process that never opted in, and not for
one that starts a training run without ever doing any training itself.
"""

import os
import subprocess
import sys

import pytest

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


@pytest.mark.skipif(os.name != "nt", reason="a backslash separates paths only on Windows")
def test_the_windows_console_script_is_recognised_too(monkeypatch):
    """Split out of the test above because it can only pass where it is true.

    ``_is_launcher_process`` splits ``argv[0]`` with ``os.path``, which is
    ``ntpath`` on Windows and ``posixpath`` everywhere else. On POSIX a
    backslash is an ordinary character in a filename rather than a separator,
    so a Windows path stays whole and never matches ``torchrun``.

    That is the right answer, not a gap to paper over in the library: such a
    path cannot be ``argv[0]`` on a POSIX interpreter, and splitting on
    backslashes there would mis-read a legitimate filename. Asserting it
    unguarded is what turned the CI red on Linux while it passed on the
    developer's Windows box.
    """
    monkeypatch.delenv("LOCAL_RANK", raising=False)
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


# --- what the .pth costs everybody -----------------------------------
#
# From 0.0.4 the wheel ships `ravex_autoload.pth`, so this module runs in every
# Python process in the environment rather than only for people who ran
# `ravex enable`. What used to be a promise about a file not existing is now a
# promise about what the file does, and these are that promise.




# `install()` is deliberately **not** called in-process anywhere below.
#
# It looks harmless and is not. When `torch` is already in `sys.modules` —
# which it is as soon as any other test module has imported it — `install()`
# skips the watcher entirely and calls `_activate()` on the spot, which builds
# the runtime and installs the PyTorch patches for the whole session. Running
# this file alone hid that: nothing here imports torch, so the other branch
# ran and the tests passed. In the full suite it broke ten tests in
# `test_patches.py`, which reasonably expects to control when patching happens.
#
# So: the pure half (`should_activate`) is tested here, and anything that arms
# is tested in a fresh interpreter, where torch really is absent and the
# branch under test is the one that runs in production.


def test_no_config_means_no_activation(tmp_path, monkeypatch):
    """The whole claim of shipping the .pth: installing the package starts
    nothing. Not "it writes no file" — it decides not to arm."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_ENABLED", raising=False)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)

    assert _bootstrap.should_activate() is False


def test_a_config_above_the_working_directory_is_enough(tmp_path, monkeypatch):
    """Found by walking up, so a script run from a subdirectory of the project
    still gets checkpointed."""
    (tmp_path / "ravex.yaml").write_text("checkpoint_every: 10\n", encoding="utf-8")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    monkeypatch.delenv("RAVEX_ENABLED", raising=False)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)

    assert _bootstrap.should_activate() is True


def test_a_malformed_config_is_not_read_at_startup(tmp_path, monkeypatch):
    """Startup asks whether a config *exists*, never what is in it.

    Worth pinning: the parsing happens later, inside `_activate`'s except, and
    a future `should_activate` that decided to peek would put YAML parsing on
    the critical path of every interpreter in the environment.
    """
    (tmp_path / "ravex.yaml").write_text("{[not: yaml: at all", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_ENABLED", raising=False)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)

    assert _bootstrap.should_activate() is True


def test_an_explicit_config_path_wins_over_the_walk(tmp_path, monkeypatch):
    """`RAVEX_CONFIG` is answered directly and the walk never happens, which
    is what lets a platform — or this suite's own conftest — point Ravex at a
    file that is not there and be sure no stray `ravex.yaml` is picked up.
    """
    (tmp_path / "ravex.yaml").write_text("checkpoint_every: 10\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_ENABLED", raising=False)
    monkeypatch.setenv("RAVEX_CONFIG", str(tmp_path / "not-here.yaml"))

    assert _bootstrap.should_activate() is False


def test_a_working_directory_that_cannot_be_read_does_not_raise(monkeypatch):
    """`os.getcwd()` fails on a directory deleted under a running process,
    which happens to build agents more often than it sounds."""

    def gone():
        raise OSError("the working directory is not there any more")

    monkeypatch.delenv("RAVEX_ENABLED", raising=False)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.setattr(_bootstrap.os, "getcwd", gone)

    assert _bootstrap.should_activate() is False


def _in_a_fresh_interpreter(code, env_extra=None, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env.pop("RAVEX_ENABLED", None)
    env.pop("RAVEX_CONFIG", None)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_startup_path_does_not_import_typing(tmp_path):
    """`typing` was 85% of what the .pth cost — 9.8 ms of 11.6 — for three
    names used only in annotations that `from __future__ import annotations`
    already turns into strings.

    A regression here is invisible: putting `from typing import ...` back into
    `ravex/__init__.py` breaks no other test and slows every Python process in
    the environment. Hence this one.
    """
    answer = _in_a_fresh_interpreter(
        "import ravex._bootstrap, sys; print('typing' in sys.modules)",
        cwd=str(tmp_path),
    )
    assert answer == "False", "typing is back on the startup path"


def test_an_uninterested_process_installs_nothing(tmp_path):
    """No `ravex.yaml`, no flag: nothing goes into `sys.meta_path` at all.

    In a fresh interpreter because that is the situation being described — the
    .pth running in some unrelated process on the machine.
    """
    answer = _in_a_fresh_interpreter(
        "import ravex._bootstrap as b, sys;"
        "print(sum(isinstance(f, b._TorchImportWatcher) for f in sys.meta_path))",
        cwd=str(tmp_path),
    )
    assert answer == "0"


def test_an_interested_process_arms_without_loading_torch(tmp_path):
    """Armed and still not loading torch is the point of the whole design: a
    process that imports torch seconds later pays then, and one that never
    does pays never."""
    answer = _in_a_fresh_interpreter(
        "import ravex._bootstrap as b, sys;"
        "w = sum(isinstance(f, b._TorchImportWatcher) for f in sys.meta_path);"
        "print(w, 'torch' in sys.modules)",
        env_extra={"RAVEX_ENABLED": "1"},
        cwd=str(tmp_path),
    )
    assert answer == "1 False"


def test_arming_twice_does_not_stack_watchers(tmp_path):
    """`install()` is reachable by hand as well as from the .pth, and two
    finders would mean the second never fires and never leaves `meta_path`."""
    answer = _in_a_fresh_interpreter(
        "import ravex._bootstrap as b, sys;"
        "b.install();"
        "print(sum(isinstance(f, b._TorchImportWatcher) for f in sys.meta_path))",
        env_extra={"RAVEX_ENABLED": "1"},
        cwd=str(tmp_path),
    )
    assert answer == "1"
