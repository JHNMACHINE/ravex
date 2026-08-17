"""Startup hook, executed by ``ravex_autoload.pth``.

This module runs in **every** Python process on the machine once the ``.pth``
file is installed, before any user code. Two consequences drive its design:

*It must be nearly free.* Nothing heavy is imported here — no torch, no yaml,
not even the rest of Ravex. Importing torch at interpreter startup would
add seconds to every ``python -c`` on the box and would drag CUDA
initialisation into processes that never asked for it.

*It must not activate uninvited.* Installing a package should not change how
unrelated scripts behave. Without an explicit ``RAVEX_ENABLED``, the
runtime only wakes up for projects that opted in by having a ``ravex.yaml``
somewhere above the working directory.

The actual work is deferred: a meta-path finder waits for ``import torch`` and
only then loads the runtime and installs the patches. A process that never
imports torch never pays anything beyond this module's few hundred bytes.
"""

import os
import sys

_FALSE = ("0", "false", "no", "off")
_CONFIG_NAMES = ("ravex.yaml", "ravex.yml")

#: Modules that are a launcher when run as ``__main__``.
_LAUNCHER_MODULES = ("torch.distributed.run", "torch.distributed.launch")

#: Console scripts that are a launcher. Matched on the stem of ``argv[0]``, so
#: ``torchrun.exe`` counts.
_LAUNCHER_SCRIPTS = ("torchrun",)


def _is_launcher_process():
    """Whether this process is a distributed launcher rather than a rank.

    ``torchrun`` imports torch to parse its own arguments, so the launcher goes
    through the autoloader like everything else: every distributed run had one
    more Ravex instance than it had ranks, announcing itself a few seconds ahead
    of the real ones as ``rank=0/1`` — before ``WORLD_SIZE`` exists.

    Nothing has gone wrong because of it: the launcher holds no model and no
    optimizer, so it never checkpoints and never reaches ``_ensure_backend``.
    But that is harmless *by construction*, not by design, and it is precisely
    what ``_ensure_backend`` exists to prevent. One future path that opens the
    backend before any object is registered, and the launcher starts creating
    directories — or opening S3 connections — for a process that will never
    train a step.

    Deliberately structural rather than a guess at intent: a worker's argv is
    the user's script and its ``LOCAL_RANK`` is set, so nothing a rank does can
    look like this.
    """
    if os.environ.get("LOCAL_RANK") is not None:
        return False  # a worker torchrun started, and workers do train

    # `python -m torch.distributed.run` has to be read off the *original*
    # command line, because neither `sys.argv` nor ``__main__.__spec__`` knows
    # yet: runpy imports `torch.distributed` while it is still resolving which
    # module to run, so torch — and with it this hook — fires before runpy
    # rewrites either. Measured at that instant: ``argv == ['-m', ...]`` and
    # ``__main__.__spec__ is None``.
    #
    # `sys.orig_argv` is 3.10+; on 3.9 the `-m` form goes undetected, which
    # leaves the status quo rather than a wrong answer.
    arguments = list(getattr(sys, "orig_argv", ()))[1:]  # [0] is the interpreter
    for flag, value in zip(arguments, arguments[1:]):
        if flag == "-m":
            return value in _LAUNCHER_MODULES
        if not flag.startswith("-"):
            break  # past the interpreter's own options; the rest is the script's

    # The console script, where argv[0] is the `torchrun` wrapper itself.
    argv0 = sys.argv[0] if sys.argv else ""
    return os.path.splitext(os.path.basename(argv0))[0] in _LAUNCHER_SCRIPTS


def _has_config_file():
    explicit = os.environ.get("RAVEX_CONFIG")
    if explicit:
        return os.path.isfile(os.path.expanduser(explicit))
    try:
        directory = os.getcwd()
    except OSError:
        return False
    while True:
        for name in _CONFIG_NAMES:
            if os.path.isfile(os.path.join(directory, name)):
                return True
        parent = os.path.dirname(directory)
        if parent == directory:
            return False
        directory = parent


def should_activate():
    """Whether this process should run Ravex."""
    flag = os.environ.get("RAVEX_ENABLED")
    if flag is not None:
        return flag.strip().lower() not in _FALSE
    return _has_config_file()


def _activate():
    """Build the runtime and patch torch. Called once torch is loaded."""
    if _is_launcher_process():
        if os.environ.get("RAVEX_DEBUG"):
            sys.stderr.write("[ravex] not activating: this is a launcher process\n")
        return

    try:
        from ravex._runtime import get_runtime

        runtime = get_runtime()
        if runtime is None:  # pragma: no cover - only when Ravex is disabled
            return
        runtime.activate()

        import atexit

        atexit.register(runtime.shutdown)
    except Exception as exc:
        # Never let the autoloader take a training run down with it - but never
        # let it disappear quietly either. Failing here means no checkpoints at
        # all, on a run that asked for them; one line of stderr is a far
        # smaller cost than finding out when the machine dies.
        sys.stderr.write(
            f"[ravex] did not start: {type(exc).__name__}: {exc}\n"
            f"[ravex] training continues without checkpointing. "
            f"Set RAVEX_DEBUG=1 for the traceback.\n"
        )
        if os.environ.get("RAVEX_DEBUG"):
            import traceback

            traceback.print_exc()


class _LoaderProxy:
    """Wraps a module loader to run a callback once execution finishes."""

    def __init__(self, loader, callback):
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._callback()

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _TorchImportWatcher:
    """Meta-path finder that fires once ``torch`` has finished importing.

    It never resolves anything itself: it asks the rest of ``sys.meta_path``
    for the real spec and hands back that spec with a wrapped loader. If
    anything looks unfamiliar it returns None and the import proceeds
    untouched.
    """

    def __init__(self, callback):
        self._callback = callback
        self._fired = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch" or self._fired:
            return None

        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break

        if spec is None or spec.loader is None:
            return None

        self._fired = True
        self._remove()
        spec.loader = _LoaderProxy(spec.loader, self._callback)
        return spec

    def _remove(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def install():
    """Arm the deferred activation. Returns True if armed or already active."""
    if not should_activate():
        return False

    if "torch" in sys.modules:
        # Unusual at .pth time, normal when this is called by hand.
        _activate()
        return True

    for finder in sys.meta_path:
        if isinstance(finder, _TorchImportWatcher):
            return True

    sys.meta_path.insert(0, _TorchImportWatcher(_activate))
    return True


install()
