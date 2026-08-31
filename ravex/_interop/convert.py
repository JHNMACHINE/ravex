"""Turning a foreign checkpoint into something Ravex can restore — GPU-90.

:mod:`ravex._interop.foreign` says what a directory is, :mod:`ravex._interop.zero` and
:mod:`ravex._interop.dcp` read it. This is the last step: putting what they read into
the shape Ravex's own resume path already consumes.

**There is less here than the issue implies, and that is the finding.** The
plan called for an adapter layer that converts between framework formats. But
Ravex already restores a *gathered* checkpoint — whole, unsharded tensors —
onto a live sharded model, through ``apply_sharded_state``, which is
``torch.distributed.checkpoint.state_dict.set_state_dict`` with
``full_state_dict=True``. That call wants exactly two things:

    model state      ``{fqn: whole tensor}``
    optimizer state  ``{"state": {fqn: {...}}, "param_groups": [...]}``

and a foreign checkpoint read whole is already both. So conversion is not a
new restore path beside the existing one — it is producing the input the
existing one takes. Nothing here scatters a tensor, chooses a shard, or knows
what a mesh is; that machinery exists and is tested, and adding a second copy
of it under a different name is how two subtly different answers get shipped.

**What is left is the names, and that is the part that actually fails.**
Formats differ in what they call a parameter far more reliably than in how they
store it: a DeepSpeed engine may write ``module.layer.weight`` where the live
model calls it ``layer.weight``, a wrapper adds a prefix, a checkpoint holds
buffers the model does not. Handing a mismatched dict to ``set_state_dict``
produces either an exception naming one key or, worse, a silent partial load.
So :func:`align` reconciles the names first and reports what it did, and
:func:`unify` refuses to guess.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class CannotConvert(Exception):
    """A checkpoint this cannot be turned into a restorable state.

    Same distinction the rest of the package draws: a refusal with a reason,
    told apart from a failure, because only the first is something the person
    holding the checkpoint can do anything about.
    """


#: Prefixes that wrappers splice in front of every parameter name. Stripped
#: only when doing so makes the whole set match — never one key at a time,
#: which would produce a state dict that is half one naming and half another.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")


def unify(found, loader) -> Dict[str, Any]:
    """One shape out of whichever format this is.

    ``{"model": {fqn: tensor}, "optimizer": {"state": ..., "param_groups":
    ...}, "step": int | None, "source": str}``.

    The DeepSpeed reader already produces this; the distributed-checkpoint one
    produces whatever structure the writer chose to save, so its top level has
    to be interpreted. That interpretation is by convention and is *reported*,
    not assumed silently: a checkpoint whose model lives under a key this does
    not recognise comes back refused with the keys it did find, which is a
    thing a caller can fix, rather than empty, which is a thing a caller
    discovers three steps later.
    """
    if found.format == "deepspeed":
        from ravex._interop.zero import unshard

        got = unshard(found, loader)
        return {
            "model": got["model"],
            "optimizer": _optimizer_with_named_groups(got),
            "step": got.get("step"),
            "source": "deepspeed (ZeRO stage %s, %s rank(s))"
            % (got.get("stage"), got.get("world_size")),
            "notes": list(got.get("notes", [])),
        }

    if found.format == "dcp":
        from ravex._interop.dcp import read

        return _unify_dcp(read(found.root))

    raise CannotConvert(
        "nothing here reads a %r checkpoint; it was identified but not opened"
        % found.format
    )


def _optimizer_with_named_groups(got: Dict[str, Any]) -> Dict[str, Any]:
    """ZeRO's parameter groups, with ``params`` as names instead of indices.

    ``set_state_dict`` identifies a group's members by fully qualified name.
    ZeRO records them positionally — ``params: [0]``, an index into the flat
    buffer — and the only record of which parameters were in which group is
    ``param_shapes``, one ordered mapping per group. So the grouping is
    rebuilt from there rather than from the optimizer's own list, which does
    not carry it.
    """
    shapes = got.get("param_shapes") or []
    groups = got.get("optimizer", {}).get("param_groups") or []

    rebuilt: List[Dict[str, Any]] = []
    for index, names in enumerate(shapes):
        source = dict(groups[index]) if index < len(groups) else {}
        source["params"] = list(names)
        rebuilt.append(source)

    if not rebuilt and groups:
        # Groups but no shapes: keep what is there rather than inventing an
        # empty membership, and let `align` report that nothing lines up.
        rebuilt = [dict(group) for group in groups]

    return {"state": got.get("optimizer", {}).get("state", {}), "param_groups": rebuilt}


#: Where a distributed checkpoint's writer usually puts each half. Tried in
#: order; the first key present wins.
_DCP_MODEL_KEYS = ("model", "model_state_dict", "module", "state_dict", "app")
_DCP_OPTIM_KEYS = ("optim", "optimizer", "optim_state_dict", "optimizer_state_dict")


def _unify_dcp(state: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    model: Optional[Dict[str, Any]] = None
    chosen = None
    for key in _DCP_MODEL_KEYS:
        candidate = state.get(key)
        if isinstance(candidate, dict) and any(
            torch.is_tensor(v) for v in candidate.values()
        ):
            model, chosen = candidate, key
            break

    if model is None:
        # The whole thing may *be* the model — `dcp.save(model.state_dict())`
        # is a perfectly ordinary thing to have written.
        if any(torch.is_tensor(v) for v in state.values()):
            model, chosen = state, "(the top level)"
        else:
            raise CannotConvert(
                "no model state found in this distributed checkpoint; its top "
                "level holds %s and none of those is a mapping of tensors"
                % ", ".join(sorted(map(str, state))[:8])
            )

    optimizer: Dict[str, Any] = {"state": {}, "param_groups": []}
    for key in _DCP_OPTIM_KEYS:
        candidate = state.get(key)
        if isinstance(candidate, dict) and "state" in candidate:
            optimizer = candidate
            break

    notes = ["model taken from %r" % chosen]
    if not optimizer["state"]:
        notes.append("no optimizer state found; the moments will not be restored")

    # **Two kinds of optimizer state reach here, and they are not the same
    # thing.** `torch.distributed.checkpoint.state_dict.get_state_dict` — what
    # an FSDP or Megatron job saves — keys the state by fully qualified name.
    # A plain `optimizer.state_dict()` keys it by the parameter's *position*
    # in the optimizer's own ordering, and that ordering belongs to the run
    # that wrote it. Telling them apart matters because everything downstream
    # matches on names: fed a positional state, `rename` finds nothing called
    # "0" and drops every moment, silently.
    keyed_by = "name"
    keys = list(optimizer.get("state") or {})
    if keys and all(str(k).isdigit() for k in keys):
        keyed_by = "position"
        notes.append(
            "the optimizer state is keyed by position, not by parameter name - "
            "which is what a plain `optimizer.state_dict()` writes. It can be "
            "restored onto a model with the same parameter order and cannot be "
            "matched by name"
        )

    return {
        "model": {str(k): v for k, v in model.items() if torch.is_tensor(v)},
        "optimizer": optimizer,
        "optimizer_keyed_by": keyed_by,
        "step": None,
        "source": "torch distributed checkpoint",
        "notes": notes,
    }


# ── names, which is where this actually goes wrong ──────────────────────────


class Alignment:
    """What matching a checkpoint's names against a live model came to."""

    def __init__(self, mapping, missing, extra, prefix=None) -> None:
        #: ``{checkpoint name: live name}`` for everything that matched.
        self.mapping: Dict[str, str] = mapping
        #: Live parameters the checkpoint has nothing for, sorted.
        self.missing: List[str] = missing
        #: Checkpoint entries the model has no home for, sorted.
        self.extra: List[str] = extra
        #: The wrapper prefix stripped to make it line up, if any.
        self.prefix: Optional[str] = prefix

    @property
    def complete(self) -> bool:
        return not self.missing

    def __repr__(self) -> str:  # pragma: no cover - a convenience
        return "Alignment(%d matched, %d missing, %d extra%s)" % (
            len(self.mapping),
            len(self.missing),
            len(self.extra),
            ", prefix %r" % self.prefix if self.prefix else "",
        )

    def report(self) -> str:
        """One paragraph a person can act on, or a short line when it all fits."""
        parts = ["%d parameter(s) matched" % len(self.mapping)]
        if self.prefix:
            parts.append("after stripping the %r prefix" % self.prefix)
        if self.missing:
            parts.append(
                "%d live parameter(s) have nothing in the checkpoint: %s"
                % (len(self.missing), ", ".join(self.missing[:6]))
            )
        if self.extra:
            parts.append(
                "%d checkpoint entr(ies) have no home in the model: %s"
                % (len(self.extra), ", ".join(self.extra[:6]))
            )
        return "; ".join(parts)


def align(checkpoint_names, live_names) -> Alignment:
    """Match a checkpoint's parameter names against a live model's.

    Tries the names as they are first. If that leaves the model unserved and
    every checkpoint name begins with the same wrapper prefix, tries again
    without it — **all of them or none**. Stripping per key is what turns a
    naming mismatch into a half-loaded model: some parameters restored, some
    left at their initialisation, and a loss curve that looks like a bad
    hyperparameter rather than like a bug.

    Nothing is renamed by similarity. Two parameters with nearly the same name
    are a normal thing in a transformer, and a fuzzy match that gets one of
    them wrong produces a model that runs.
    """
    live = list(live_names)
    checkpoint = list(checkpoint_names)
    live_set = set(live)

    direct = {name: name for name in checkpoint if name in live_set}
    if len(direct) == len(live):
        return Alignment(direct, [], sorted(set(checkpoint) - live_set))

    for prefix in _WRAPPER_PREFIXES:
        if not checkpoint or not all(name.startswith(prefix) for name in checkpoint):
            continue
        stripped = {name: name[len(prefix):] for name in checkpoint}
        matched = {
            name: short for name, short in stripped.items() if short in live_set
        }
        if len(matched) > len(direct):
            served = set(matched.values())
            return Alignment(
                matched,
                sorted(live_set - served),
                sorted(name for name in checkpoint if name not in matched),
                prefix,
            )

    served = set(direct.values())
    return Alignment(
        direct,
        sorted(live_set - served),
        sorted(set(checkpoint) - served),
    )


def rename(unified: Dict[str, Any], alignment: Alignment) -> Dict[str, Any]:
    """The same state under the live model's names.

    Applied to the optimizer as well as to the weights, and to the ``params``
    membership inside each parameter group — three places that have to agree,
    and the reason this is one function rather than three call sites.
    """
    mapping = alignment.mapping

    model = {mapping[name]: tensor
             for name, tensor in unified["model"].items() if name in mapping}

    optimizer = unified.get("optimizer") or {}
    if unified.get("optimizer_keyed_by") == "position":
        # Nothing to rename: these keys are positions, and mapping them
        # through a name table would drop all of them. Carried through as they
        # are, for `to_positional` to pass along or the sharded path to refuse.
        out = dict(unified)
        out["model"] = model
        return out
    state = {
        mapping[name]: entry
        for name, entry in (optimizer.get("state") or {}).items()
        if name in mapping
    }
    groups = []
    for group in optimizer.get("param_groups") or []:
        moved = dict(group)
        members = group.get("params")
        if isinstance(members, (list, tuple)):
            moved["params"] = [mapping[n] for n in members if n in mapping]
        groups.append(moved)

    out = dict(unified)
    out["model"] = model
    out["optimizer"] = {"state": state, "param_groups": groups}
    return out


def fit_param_groups(unified: Dict[str, Any], allowed) -> Dict[str, Any]:
    """Drop hyperparameters the receiving optimizer does not have.

    A parameter group carries the settings of the optimizer that wrote it, and
    those are not portable: DeepSpeed's Adam records ``bias_correction``, which
    torch's does not, and handing it over is not a near miss —
    ``set_state_dict`` raises ``KeyError: param_groups.0.bias_correction`` from
    inside its own unflattening, several frames from anything the caller wrote.

    Dropping rather than translating. A hyperparameter with no counterpart has
    no correct value in the target, and inventing one changes how the resumed
    run trains — quietly, and in a way no test of the checkpoint would catch.
    Saying which ones were dropped lets whoever cares set them by hand.

    ``allowed`` is the set of keys the live optimizer's own groups use, or the
    groups themselves. ``params`` always survives: it is membership, not a
    setting.
    """
    if allowed and isinstance(next(iter(allowed)), dict):
        keys = {key for group in allowed for key in group}
    else:
        keys = set(allowed)
    keys.add("params")

    optimizer = unified.get("optimizer") or {}
    kept, dropped = [], set()
    for group in optimizer.get("param_groups") or []:
        kept.append({k: v for k, v in group.items() if k in keys})
        dropped |= {k for k in group if k not in keys}

    out = dict(unified)
    out["optimizer"] = {"state": optimizer.get("state", {}), "param_groups": kept}
    if dropped:
        out["notes"] = list(out.get("notes", [])) + [
            "dropped hyperparameter(s) the receiving optimizer does not have: %s"
            % ", ".join(sorted(dropped))
        ]
    return out


def to_positional(unified: Dict[str, Any], names) -> Optional[Dict[str, Any]]:
    """The optimizer state keyed by position, for ``Optimizer.load_state_dict``.

    Everything else here works in names, because a name is the only identifier
    that survives a change of framework. But torch's *plain*
    ``Optimizer.load_state_dict`` — the one a run without FSDP uses — wants the
    state keyed by the parameter's index in the optimizer's own ordering, and
    ``param_groups[i]["params"]`` as indices too.

    ``names`` is the live parameter names in that order, which the caller gets
    from ``named_parameters()``. Position is reintroduced at the last possible
    moment and against the *live* model, never carried through from the
    checkpoint's own ordering — that ordering belongs to a different run and
    the whole reason this module works in names is that it cannot be trusted.

    ``None`` when the state does not cover the model, rather than a partial
    state dict: torch raises on a mismatched length anyway, and a partial one
    that happens to be the right length would load the wrong moments onto the
    right parameters, which nothing downstream would notice.
    """
    order = list(names)
    optimizer = unified.get("optimizer") or {}
    state = optimizer.get("state") or {}
    if not state:
        return None

    if unified.get("optimizer_keyed_by") == "position":
        # Already positional, and this is the one place that shape is usable:
        # `Optimizer.load_state_dict` uses the same convention, so passing it
        # through is what torch itself would do. Only the count is checked -
        # matching a position against a different model's ordering is not
        # something this can verify, and the note from `unify` says so.
        numbered = {int(k): v for k, v in state.items()}
        if len(numbered) != len(order):
            return None
        return {
            "state": numbered,
            "param_groups": optimizer.get("param_groups")
            or [{"params": list(range(len(order)))}],
        }

    index_of = {name: position for position, name in enumerate(order)}
    positional = {
        index_of[name]: entry for name, entry in state.items() if name in index_of
    }
    if len(positional) != len(order):
        return None

    groups = []
    for group in (unified.get("optimizer") or {}).get("param_groups") or []:
        moved = dict(group)
        members = group.get("params")
        if isinstance(members, (list, tuple)):
            moved["params"] = [index_of[n] for n in members if n in index_of]
        groups.append(moved)
    if not groups:
        groups = [{"params": list(range(len(order)))}]

    return {"state": positional, "param_groups": groups}


def into_snapshot(unified: Dict[str, Any], key: str, step: int = 0) -> Dict[str, Any]:
    """A Ravex snapshot holding one sharded group, in ``gather`` layout.

    ``gather`` is not a stand-in for something better: it is what a foreign
    checkpoint read whole genuinely *is* — topology-independent, every tensor
    entire — and it is the layout Ravex's restore path already scatters onto a
    live sharded model. ``key`` is the registry's fingerprint for the group
    this state belongs to, which only the registry can compute, so it is
    passed in rather than guessed.
    """
    return {
        "ravex_version": 1,
        "step": int(step or unified.get("step") or 0),
        "models": {},
        "optimizers": {},
        "schedulers": {},
        "scalers": {},
        "sharded": {
            key: {
                "layout": "gather",
                "model": unified["model"],
                "optimizer": unified.get("optimizer") or {},
                "parameters": sorted(unified["model"]),
                "converted_from": unified.get("source"),
            }
        },
    }


def convert(found, loader, live_names, key: str) -> Tuple[Dict[str, Any], Alignment]:
    """Read, align, rename, wrap. The whole path in one call.

    Returns the snapshot and the alignment that produced it. The alignment is
    returned rather than logged and dropped because "it converted" and "it
    converted everything" are different answers, and only the caller knows
    whether a partial one is acceptable — a fine-tune that adds a head has
    extra live parameters on purpose.
    """
    unified = unify(found, loader)
    alignment = align(list(unified["model"]), live_names)
    if not alignment.mapping:
        raise CannotConvert(
            "not one parameter name in this checkpoint matches the live model. "
            + alignment.report()
        )
    return into_snapshot(rename(unified, alignment), key), alignment
