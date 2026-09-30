"""Precision declared on the module: ``@ravex.save_dtype`` (GPU-137).

``save_dtype`` in the configuration names components or tensor-name globs, and
a glob is a string about an attribute name: rename ``experts`` to
``moe_experts`` and the rule that cast them to fp8 matches nothing, the model
goes back to bf16, and the first sign is a slightly worse curve two days later.
An annotation lives on the module, so a rename cannot detach it.

It is resolved into globs when the checkpoint backend is built — Moonclip still
matches names and still knows nothing about modules — and the resolution is
where the three rules live:

**The deepest annotation wins.** Rules are emitted deepest first, and Moonclip
applies the first one that matches, so an annotated module inside another
annotated module keeps its own dtype. That is the silent shadowing a glob
written in the wrong order has today, removed by construction.

**An instance beats its class.** ``ravex.save_dtype(model.head, "none")``
carves one module out of a class that was annotated for all of them.

**A specific config glob beats an annotation, which beats a default.** The
author knows which pieces tolerate fp8; a scalar ``save_dtype`` or a component
name in the file is a property of the run and should not undo that in silence.
A glob that names tensors is the operator being specific, and wins — which is
also how "everything fp32, for a debug" is said:
``RAVEX_SAVE_DTYPE='ravex/*:fp32'``. Not the bare ``*``: that is the catch-all
default, and ``{model: none, "*": bf16}`` has to keep meaning "everything but
the weights", which it could not if ``*`` jumped ahead of ``model``.
See :meth:`ravex._config.RavexConfig.resolve_save_dtype`.

**A declaration that does nothing says so.** Moonclip reports a pattern that
matches no tensor; an annotation can be dead in more ways than that — a class
never instantiated in a registered model, a module whose every tensor sits
under a deeper annotation, a module a config glob overrides — and each is
logged once. The point of the feature is that a declaration cannot die
silently; if it could, the problem would have moved rather than gone.

Annotations cover the module's own tensors: its parameters and buffers, under
``ravex/models/<key>/`` or ``ravex/sharded/<key>/model/``. Optimizer state stays
with the configuration's ``optimizer`` component — how much precision the
moments tolerate is a property of the run's optimizer, not of the layer.
"""

from __future__ import annotations

import logging
import re
import weakref
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("ravex")

#: Module instance → dtype, and class → dtype. Weak on both sides: a
#: declaration must not keep a model alive, and a class defined in a test or a
#: notebook cell should be free to go.
_instances: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()
_classes: "weakref.WeakKeyDictionary[type, str]" = weakref.WeakKeyDictionary()

#: Messages already logged, so a backend rebuilt over the same store does not
#: repeat itself.
_said: Set[str] = set()

#: Segments wrappers insert into module names that the state dict does not
#: carry. ``get_state_dict`` strips them, so a rule built from
#: ``named_modules()`` has to strip them too or it names tensors that do not
#: exist.
_WRAPPER_SEGMENTS = frozenset(
    {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}
)


def normalize_dtype(dtype: Any) -> str:
    """The dtype as the configuration spells it, or ``ValueError``.

    Raises rather than recording a problem, unlike a bad value in
    ``ravex.yaml``: an annotation is code, and it runs at import, where a typo
    should stop the program on the line that has it.
    """
    from ravex._config import _SAVE_DTYPES

    name = str(dtype).strip().lower()
    if name not in _SAVE_DTYPES:
        raise ValueError(
            "save_dtype %r is not a dtype Moonclip can store; use one of: %s"
            % (dtype, ", ".join(sorted(_SAVE_DTYPES)))
        )
    return name


def declare(target: Any, dtype: str) -> None:
    """Record ``dtype`` for a module class or a module instance."""
    import torch.nn as nn

    name = normalize_dtype(dtype)
    if isinstance(target, type) and issubclass(target, nn.Module):
        _classes[target] = name
    elif isinstance(target, nn.Module):
        _instances[target] = name
    else:
        raise TypeError(
            "ravex.save_dtype() takes an nn.Module subclass or instance, not %r"
            % (target,)
        )


def declared_dtype(module: Any) -> Optional[str]:
    """What ``module`` was declared as: its own annotation, else its class's."""
    own = _instances.get(module)
    if own is not None:
        return own
    for klass in type(module).__mro__:
        found = _classes.get(klass)
        if found is not None:
            return found
    return None


def _clean(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def _glob(pattern: str) -> "re.Pattern[str]":
    """Moonclip's matching: ``*`` is any run of characters, the rest literal."""
    return re.compile(".*".join(re.escape(part) for part in pattern.split("*")))


def _say(message: str) -> None:
    if message not in _said:
        _said.add(message)
        logger.warning(message)


def _tensor_names(root: Any) -> Tuple[List[str], bool]:
    """Names of every parameter and buffer, and whether they are trustworthy.

    Not ``state_dict()``: on an FSDP1 root that is a collective. The second
    value is False when FSDP1 has flattened the parameters, because then the
    names seen here are not the ones the checkpoint will hold, and a report
    built on them would be wrong.
    """
    names = [n for n, _ in root.named_parameters(remove_duplicate=False)]
    names += [n for n, _ in root.named_buffers(remove_duplicate=False)]
    trustworthy = not any("_flat_param" in n for n in names)
    return [_clean(n) for n in names], trustworthy


def _annotated(root: Any) -> List[Tuple[str, str, str]]:
    """``(fqn, dtype, label)`` for every annotated module under ``root``."""
    found = []
    for name, module in root.named_modules():
        dtype = declared_dtype(module)
        if dtype is not None:
            found.append((_clean(name), dtype, type(module).__name__))
    return found


def declared_rules(registry: Any, config_globs: Iterable[str] = ()) -> Dict[str, str]:
    """The annotations on the registered models, as ordered Moonclip globs.

    ``config_globs`` are the configuration's specific patterns, which win over
    an annotation (see the module docstring); they are passed only so that an
    annotation they completely override can be reported.
    """
    from ravex._dist.collectives import unwrap_model

    roots: List[Tuple[str, Any]] = [
        ("ravex/models/%s/" % key, unwrap_model(model))
        for key, model in registry.keyed_models()
    ]
    roots += [
        ("ravex/sharded/%s/model/" % key, unwrap_model(model))
        for key, model, _ in registry.sharded_groups()
    ]
    overriding = [(p, _glob(p)) for p in config_globs]

    rules: List[Tuple[str, str]] = []
    seen_classes: Set[type] = set()
    seen_instances: Set[int] = set()

    for prefix, root in roots:
        for _, module in root.named_modules():
            seen_instances.add(id(module))
            seen_classes.update(type(module).__mro__)

        annotated = _annotated(root)
        if not annotated:
            continue
        names, trustworthy = _tensor_names(root)

        # Deepest first: Moonclip takes the first match, so this is what makes
        # the most specific annotation the one that applies.
        annotated.sort(key=lambda entry: -(entry[0].count(".") + 1 if entry[0] else 0))
        claimed: Set[str] = set()
        for fqn, dtype, label in annotated:
            pattern = prefix + (fqn + ".*" if fqn else "*")
            rules.append((pattern, dtype))
            if not trustworthy:
                continue

            where = "%s (%s)" % (fqn or "<root>", label)
            owned = [
                n for n in names
                if n not in claimed and (not fqn or n.startswith(fqn + "."))
            ]
            claimed.update(owned)
            if not owned:
                _say(
                    "save_dtype=%s on %s covers no tensor: it holds no parameter "
                    "or buffer that a deeper annotation has not already claimed"
                    % (dtype, where)
                )
                continue
            for glob, matcher in overriding:
                if all(matcher.fullmatch(prefix + n) for n in owned):
                    _say(
                        "save_dtype=%s on %s is overridden by the configured "
                        "rule %r, which names the same tensors; the configured "
                        "rule wins" % (dtype, where, glob)
                    )
                    break

    for module in list(_instances.keys()):
        if id(module) not in seen_instances:
            _say(
                "save_dtype=%s was declared on a %s that is not part of any "
                "model this run checkpoints; it has no effect"
                % (_instances.get(module), type(module).__name__)
            )
    for klass in list(_classes.keys()):
        if klass not in seen_classes:
            _say(
                "save_dtype=%s was declared on class %s, and no model this run "
                "checkpoints contains one; it has no effect"
                % (_classes.get(klass), klass.__name__)
            )

    # A module reachable from two roots yields the same pattern twice; the
    # first one stands, as it would in Moonclip.
    ordered: Dict[str, str] = {}
    for pattern, dtype in rules:
        ordered.setdefault(pattern, dtype)
    return ordered
