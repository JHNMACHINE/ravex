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
