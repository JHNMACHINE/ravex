"""Ravex — checkpoint and resume for PyTorch training.

One way in, and it is a decorator::

    import ravex

    @ravex.train_loop(preemption_handler=True)
    def train():
        model = build_model()
        optimizer = torch.optim.Adam(model.parameters())
        for batch in loader:
            ...

    train()

Inside that function nothing changes — not one line, not one import. Ravex
notices the model, the optimizer, the scheduler, the AMP scaler and the
dataloader as they are constructed, counts every ``optimizer.step()``, writes a
checkpoint every ``checkpoint_every`` steps, and on a rerun puts the whole thing
back: weights, optimizer moments, LR schedule, RNG state and dataset position.

**What the decorator adds over the old autoloader** is not convenience, it is a
*boundary*. Ravex now knows where the loop begins and where it ends, and — the
part that mattered enough to break the API for — it holds the function
**before the model exists**. That is what an elastic regroup needs and could
never have: see ``ravex/_dist/elastic.py``, which stops at the rendezvous layer
for exactly this reason.

Through 0.0.5 there was a second way in: a ``.pth`` file in site-packages that
ran in every interpreter on the machine and attached Ravex to any process that
had a ``ravex.yaml`` above its working directory. It is gone, deliberately —
see the changelog. Ravex is a library you adopt, not a capability switched on
underneath code that does not know about it.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar

#: The version, written here and in ``Cargo.toml``, which is what maturin builds
#: the wheel from. Two places, because maturin has no equivalent of setuptools'
#: ``dynamic = {attr = ...}``; ``tests/test_version_is_single.py`` fails if they
#: ever disagree.
__version__ = "0.2.0"

__all__ = [
    "batch_boundary",
    "checkpoint",
    "deactivate",
    "flush",
    "is_active",
    "status",
    "step",
    "track",
    "train_loop",
    "__version__",
]

F = TypeVar("F", bound=Callable[..., Any])


def train_loop(
    _function: Optional[F] = None,
    *,
    preemption_handler: Optional[bool] = None,
    elastic: bool = False,
    **overrides: object,
) -> Any:
    """Wrap a training function so Ravex checkpoints it.

    Usable bare or called::

        @ravex.train_loop
        def train(): ...

        @ravex.train_loop(checkpoint_every=100, backend="torch_save")
        def train(): ...

    Keyword arguments are configuration overrides and win over ``ravex.yaml``
    and the ``RAVEX_*`` environment variables — they are the most specific thing
    that could have said so. An unknown name raises ``TypeError`` rather than
    being ignored, because a silently-ignored ``checkpoint_evry=50`` is a run
    that checkpoints on the default cadence and never says why.

    Two of them are named rather than passed through, because they are the two
    the caller thinks in rather than the two the config file thinks in:

    ``preemption_handler``
        Catch SIGTERM and try to get a final checkpoint down before the process
        is killed — a spot instance gets roughly ten seconds' notice. Sets
        ``handle_sigterm``. Read what that promises in the changelog for GPU-92
        before relying on it: on a sharded model it is a best-effort attempt
        that will often not complete, not a guarantee.

    ``elastic``
        Resume onto a *different* number of ranks without the process exiting.
        **Not implemented**, and this raises rather than accepting it quietly.
        See the message it raises.

    The teardown is a ``finally``, so it runs whether the function returns or
    raises: final checkpoint, flush, patches removed, runtime dropped. A second
    call to the decorated function starts cleanly rather than finding the
    previous run's state — which is why the patches are uninstalled here instead
    of at interpreter exit.
    """
    if elastic:
        raise NotImplementedError(
            "elastic=True is not wired up yet, and what is missing is no "
            "longer the rebuild. Since GPU-110 ravex._dist.elastic.regroup() "
            "does the whole thing — capture the full model and optimizer "
            "state, tear the group down, bring it up at the new world size, "
            "and load the state into a freshly built model under the new "
            "mesh, with a joining rank getting the state from the broadcast "
            "rather than out of band. It is covered by "
            "tests/test_dist_elastic_remesh.py across a real 2 -> 3 change. "
            "What is missing is re-entry: your loop holds `model` and "
            "`optimizer` in its own locals, and a regroup returns new objects "
            "that ravex cannot assign into your frame. Doing this from the "
            "decorator means calling your function again and restoring into "
            "the objects it builds the second time - which is the resume path "
            "ravex already has, pointed at a regroup instead of a checkpoint. "
            "Until then: resuming onto a different world size *with* a process "
            "restart does work — set reshard_on_resume=true and relaunch."
        )

    if preemption_handler is not None:
        overrides["handle_sigterm"] = bool(preemption_handler)

    def decorate(function: F) -> F:
        @functools.wraps(function)
        def wrapper(*args: object, **kwargs: object):
            _activate(**overrides)
            try:
                return function(*args, **kwargs)
            finally:
                deactivate()

        return wrapper  # type: ignore[return-value]

    if _function is not None:
        # Used bare, as `@ravex.train_loop`. Nothing was configured, which is
        # fine — `ravex.yaml` and the environment still apply.
        return decorate(_function)
    return decorate


def track(
    model: object = None,
    optimizer: object = None,
    dataloader: object = None,
    *,
    scheduler: object = None,
    scaler: object = None,
) -> None:
    """Name an object explicitly, when leaving it to be noticed is not enough.

    Inside a ``train_loop`` Ravex finds these on its own and this is not needed.
    It is here for the cases where "on its own" is a guess: two models where only
    one is being trained, an optimizer built before the decorated function was
    entered, a dataloader that is never iterated through the patched path.

    Registering something twice is harmless. Registering an optimizer through
    this attaches the same step hook the automatic path attaches, so the step
    count keeps working — an optimizer tracked here and stepped is counted.

    Raises if there is no active ``train_loop``: the alternative is recording
    objects into a registry that is about to be replaced, which looks like it
    worked.
    """
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    if runtime is None:
        raise RuntimeError(
            "ravex.track() outside a @ravex.train_loop function. There is no "
            "runtime to register with, so this call would be silently lost."
        )

    registry = runtime.registry
    if model is not None:
        registry.register_model(model)
    if optimizer is not None:
        from ravex._patches import _attach_step_hook

        registry.register_optimizer(optimizer)
        _attach_step_hook(optimizer, runtime)
    if dataloader is not None:
        registry.register_dataloader(dataloader)
    if scheduler is not None:
        registry.register_scheduler(scheduler)
    if scaler is not None:
        registry.register_scaler(scaler)


def _activate(**overrides: object) -> None:
    """Build the runtime and install the patches. The decorator's entry half.

    Underscored because it is not the way in — :func:`train_loop` is, and it is
    the only form that also guarantees the exit. This exists as its own function
    because the test suite needs to open the runtime without wrapping every
    scenario in a decorated function, and because a seam that the decorator and
    the tests share is one seam rather than two.
    """
    from ravex._config import RavexConfig
    from ravex._runtime import RavexRuntime, get_runtime
    import ravex._runtime as runtime_module

    if get_runtime(create=False) is not None:
        raise RuntimeError(
            "a @ravex.train_loop function is already running in this process. "
            "Nesting them would mean two runtimes counting the same steps; "
            "if the inner one was meant to configure the outer, pass the "
            "options to the outer decorator instead."
        )

    config = RavexConfig.load()
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise TypeError(f"unknown configuration option {key!r}")
        if key == "storage":
            # Not `setattr`. `storage` is the one option whose value is a
            # section rather than a scalar, and assigning a dict over the
            # dataclass used to surface several frames later as
            # `AttributeError: 'dict' object has no attribute 'path'` from
            # inside `_normalize` — naming neither this option nor this
            # decorator. `strict` because a keyword typed at the call site is
            # the most specific thing that could have said so, exactly like
            # the unknown-option check above.
            config.apply_storage(value, strict=True)
            continue
        setattr(config, key, value)
    config._normalize()

    runtime = RavexRuntime(config)
    runtime_module._runtime = runtime
    runtime.activate()


def deactivate() -> None:
    """Stop checkpointing, write the final checkpoint, and unpatch PyTorch.

    The decorator's exit half, exposed because two callers still want it by
    hand: the test suite, and a notebook that wants to stop early without
    leaving the patches installed on a live interpreter.
    """
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


def batch_boundary() -> None:
    """Hand Ravex the top of a training iteration.

    A loop that iterates a ``DataLoader`` never needs this: Ravex wraps the
    iterator and takes the boundary from it. A loop over tensors that are
    already batched has no such moment to borrow — and there is no other point
    in the iteration that would do, because the work Ravex defers to the
    boundary is exactly the work that must not happen inside
    ``optimizer.step()``:

    * a **checkpoint** taken mid-step records a learning rate one step stale,
      so the resumed run trains with the wrong LR from its very first step;
    * an **outer round** (``outer_loop: true``) writes the averaged parameters
      into the model, underneath an optimizer that has not finished its step,
      after spending however many minutes of network to fetch them.

    So call it at the top of each iteration, before the batch::

        for begin in range(0, len(data), batch_size):
            ravex.batch_boundary()
            optimizer.zero_grad()
            loss_fn(model(x[begin : begin + batch_size]), y[...]).backward()
            optimizer.step()

    Cheap and idempotent: with nothing due it returns having done nothing, and
    outside an active run it does nothing at all. Calling it in a loop that
    *does* have a DataLoader is harmless for the same reason.
    """
    from ravex._runtime import get_runtime

    runtime = get_runtime(create=False)
    if runtime is not None:
        runtime.on_batch_boundary()


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


def status() -> dict[str, object | None]:
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
