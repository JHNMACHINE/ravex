"""Ravex — transparent checkpointing for PyTorch training.

Two ways in.

**Zero code changes** (what the project is for). Install the autoloader once,
drop a ``ravex.yaml`` next to your code, and run your script unchanged::

    ravex enable
    python train.py

**One line**, when you would rather be explicit::

    import ravex
    ravex.activate()

Either way, from that point on every ``optimizer.step()`` is counted, a
checkpoint is written every ``checkpoint_every`` steps, and rerunning the same
script picks up where the last one stopped — model, optimizer, LR scheduler,
AMP scaler, RNG state and dataset position included.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__version__ = "0.1.0"

__all__ = [
    "activate",
    "checkpoint",
    "deactivate",
    "flush",
    "is_active",
    "status",
    "step",
    "__version__",
]


def activate(**overrides: Any):
    """Install the PyTorch patches and start checkpointing.

    Keyword arguments override the resolved configuration, e.g.
    ``ravex.activate(checkpoint_every=100, backend="torch_save")``.
    Calling this twice is harmless.
    """
    from ravex._config import RavexConfig
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    if runtime is None:
        config = RavexConfig.load()
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"unknown configuration option {key!r}")
            setattr(config, key, value)
        config._normalize()

        from ravex._runtime import RavexRuntime
        import ravex._runtime as runtime_module

        runtime = RavexRuntime(config)
        runtime_module._runtime = runtime
        runtime.activate()

        import atexit

        atexit.register(runtime.shutdown)
    elif overrides:
        raise RuntimeError(
            "Ravex is already active; configuration cannot be changed after "
            "activation. Set the options in ravex.yaml or RAVEX_* env vars."
        )
    return runtime


def deactivate() -> None:
    """Stop checkpointing and remove the patches. Mostly for tests."""
    from ravex._runtime import get_runtime, reset_runtime

    runtime = get_runtime(create=False)
    if runtime is not None:
        runtime.shutdown()
    reset_runtime()


def is_active() -> bool:
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    return runtime is not None and runtime.enabled


def checkpoint() -> bool:
    """Force a checkpoint now, outside the normal cadence."""
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    if runtime is None:
        return False
    return runtime.checkpoint()


def flush() -> None:
    """Block until every pending checkpoint write has landed."""
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    if runtime is not None:
        runtime.flush()


def step() -> int:
    """Current optimizer-step count as Ravex sees it."""
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    return runtime.step if runtime is not None else 0


def status() -> Dict[str, Optional[Any]]:
    """Snapshot of what the runtime is doing, for debugging."""
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    if runtime is None:
        return {"active": False}

    return {
        "active": runtime.enabled,
        "step": runtime.step,
        "models": len(runtime.registry.models),
        "optimizers": len(runtime.registry.optimizers),
        "dataloaders": len(runtime.registry.dataloaders),
        "resumed": runtime.registry.resumed,
        "backend": type(runtime.backend).__name__ if runtime.backend else None,
        "config": runtime.config.describe(),
    }
