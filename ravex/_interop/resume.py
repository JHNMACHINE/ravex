"""Converting a foreign checkpoint at resume time — GPU-90.

The rest of this package answers *what is this* and *what does it say*.
This is the caller: the decision to act on the answer, and the one place that
turns a converted state into something the registry will restore.

It lives here rather than on :class:`~ravex._runtime.RavexRuntime` because it
needs almost nothing from the runtime — the storage path and the flag out of
the config, and the registry to find the live model and put the state back.
Two arguments, against a class of thirty-five methods it was previously three
of. What the runtime keeps is a one-line entry point, so that the *decision to
try* still reads in order alongside the ordinary resume.

**Everything here is collective when it acts.** ``all_ranks_agree`` gates the
load precisely so that a rank which cannot see a convertible checkpoint does
not leave its peers waiting inside ``restore_state``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ravex")


def convert_on_resume(config, registry, defer_rng: bool) -> bool:
    """Restore from a checkpoint another framework wrote. **Collective when it acts.**

    Returns True when the live objects were loaded from it, so the caller
    skips the ordinary resume.

    **Detection is unconditional, acting on it is not** — the same shape as
    ``reshard_on_resume``, for a stronger version of the same reason. A
    DeepSpeed directory sitting where Ravex's own store belongs usually
    means a path points somewhere unintended, and converting it silently
    would turn that into a run training on someone else's weights with
    nothing in the log to say so. So this always says what it found, and
    only ``convert_foreign`` turns that into a load.

    **The active framework is a guard, not a hint.** A DeepSpeed run
    resuming a DeepSpeed checkpoint needs no help: its engine loads that
    checkpoint itself, and Ravex converting it as well would restore the
    same weights twice by two different routes and disagree with the
    engine about the optimizer. So that pair is declined by name.
    """
    from ravex._dist.collectives import all_ranks_agree
    from ravex._interop.foreign import identify, summary

    if config.storage.is_remote:
        # `identify` reads a local directory. A bucket is not one, and
        # answering "unknown" for every remote store would be a message
        # about this function rather than about the checkpoint.
        return False

    found = identify(config.storage.path)
    if found.format not in ("deepspeed", "dcp"):
        return False

    from ravex._frameworks import detect_framework

    active = detect_framework()
    if active == "deepspeed" and found.format == "deepspeed":
        logger.info(
            "There is a DeepSpeed checkpoint at %s and this run is "
            "DeepSpeed - leaving it to the engine, which loads its own "
            "checkpoints.",
            config.storage.path,
        )
        return False

    logger.warning(
        "%s holds a checkpoint written by something else (%s), not a Ravex "
        "store.%s",
        config.storage.path,
        summary(found),
        ""
        if config.convert_foreign
        else " Ravex can convert it on load - set convert_foreign=true "
        "(RAVEX_CONVERT_FOREIGN=1). It is off by default so that a "
        "storage path pointing somewhere unintended fails visibly "
        "instead of training on someone else's weights.",
    )
    if not config.convert_foreign:
        return False

    # Every rank reaches this, because the flag is configuration and is the
    # same everywhere; what can differ is what each machine's disk holds.
    # Ranks that disagree all take the ordinary path together rather than
    # some of them entering the collective inside `restore_state`.
    target = _target(registry)
    wanted = target is not None
    if not all_ranks_agree(wanted):
        logger.warning(
            "Not every rank can see a convertible checkpoint from where it "
            "is, so none of them converts - half a run restored from a "
            "foreign checkpoint and half from nothing is worse than "
            "neither."
        )
        return False

    assert target is not None  # `all_ranks_agree` said so on every rank
    kind, key, model, optimizers = target
    try:
        return _apply(registry, found, kind, key, model, optimizers, defer_rng)
    except Exception as exc:
        logger.warning(
            "Could not convert the checkpoint at %s (%s) - starting from "
            "scratch.",
            config.storage.path,
            exc,
        )
        return False

def _target(registry):
    """The one live group a converted checkpoint would go into, or None.

    One. Several sharded groups is not an obstacle to converting, it is an
    obstacle to knowing *which* of them the foreign checkpoint is of — and
    a converter that picks the first would be right about half the time on
    the runs where it matters.
    """
    groups = registry.sharded_groups()
    if len(groups) == 1:
        key, model, optimizers = groups[0]
        return ("sharded", key, model, optimizers)

    plain = registry.keyed_models()
    if not groups and len(plain) == 1:
        key, model = plain[0]
        # The optimizer comes along when there is exactly one, so that a
        # converted resume restores Adam's moments rather than quietly
        # dropping them. Several optimizers is the same ambiguity as
        # several models and gets the same answer: the weights are
        # restored and the moments are not, said out loud below.
        owners = registry.optimizers
        return ("plain", key, model, list(owners) if len(owners) == 1 else [])

    logger.warning(
        "Converting a foreign checkpoint needs one model to convert it "
        "into, and this run has %d sharded group(s) and %d plain model(s). "
        "Which one the checkpoint is of is not something to guess at.",
        len(groups),
        len(plain),
    )
    return None

def _apply(registry, found, kind, key, model, optimizers, defer_rng: bool) -> bool:
    from ravex._interop.convert import align, fit_param_groups, into_snapshot, rename, unify
    from ravex._dist.collectives import unwrap_model

    import torch

    loader = lambda path: torch.load(  # noqa: E731 - one expression, one use
        path, map_location="cpu", weights_only=False
    )

    unified = unify(found, loader)
    live = list(unwrap_model(model).state_dict())
    alignment = align(list(unified["model"]), live)

    logger.info("Converting from %s: %s", unified["source"], alignment.report())
    for note in unified.get("notes", []):
        logger.info("Conversion note: %s", note)

    if not alignment.mapping:
        logger.warning(
            "Not one parameter name in that checkpoint matches this model, "
            "so there is nothing to convert into it."
        )
        return False

    moved = rename(unified, alignment)
    if optimizers:
        moved = fit_param_groups(moved, optimizers[0].state_dict()["param_groups"])
        for note in moved.get("notes", [])[len(unified.get("notes", [])) :]:
            logger.info("Conversion note: %s", note)

    if kind == "sharded":
        if moved.get("optimizer_keyed_by") == "position":
            # `set_state_dict(full_state_dict=True)` identifies a group's
            # members by name. A positional state cannot be handed to it,
            # and matching those positions against a sharded model's
            # parameter order is a guess this will not make: the weights
            # come across, the moments do not, and that is said.
            logger.warning(
                "That checkpoint's optimizer state is keyed by position "
                "rather than by parameter name, and a sharded restore "
                "matches by name. The weights are converted; the moments "
                "are not."
            )
            moved = dict(moved)
            moved["optimizer"] = {"state": {}, "param_groups": []}
        snapshot = into_snapshot(moved, key)
    else:
        # A plain model takes its state dict directly; `restore_state`
        # loads it with `load_state_dict` and never reaches the sharded
        # branch, so wrapping it as a sharded group would send it through
        # a collective this run has no need of.
        from ravex._interop.convert import to_positional

        optimizers_state = {}
        if optimizers:
            names = [n for n, _ in unwrap_model(model).named_parameters()]
            positional = to_positional(moved, names)
            if positional is None:
                logger.warning(
                    "The converted optimizer state does not cover every "
                    "parameter of this model, so the moments are not "
                    "restored - only the weights. Training continues; the "
                    "first steps after this will behave as if the "
                    "optimizer had just been created."
                )
            else:
                keys = [k for k, _ in registry.keyed_optimizers()]
                if keys:
                    optimizers_state[keys[0]] = positional

        snapshot = {
            "ravex_version": 1,
            "step": int(moved.get("step") or 0),
            "models": {key: moved["model"]},
            "optimizers": optimizers_state,
            "schedulers": {},
            "scalers": {},
            "sharded": {},
        }

    registry.restore_state(snapshot, defer_rng=defer_rng)
    logger.info(
        "Resumed from a converted %s checkpoint at step %s. The data order "
        "and the RNG are not restored - a foreign checkpoint carries "
        "neither.",
        found.format,
        snapshot["step"],
    )
    return True
