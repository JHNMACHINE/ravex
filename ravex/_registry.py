"""Registry of the training objects Ravex has seen.

Everything is held through :mod:`weakref`: a registry that keeps a model alive
would turn "we checkpoint your training" into "we leak your training".

Objects are keyed by a *structural fingerprint* rather than by ``id()``, which
is meaningless across processes. The fingerprint is a digest of the shapes and
dtypes of an object's tensors, so the same architecture rebuilt in a fresh
process lands on the same key and the checkpoint applies. If the user changes
the architecture between runs, the key changes, the state simply does not match
and the resume is skipped with a warning instead of exploding.

``hashlib`` is used rather than ``hash()`` on purpose: Python randomizes string
hashing per process (``PYTHONHASHSEED``), so ``hash()``-based keys would differ
between the run that saved and the run that resumes.
"""

from __future__ import annotations

import hashlib
import logging
import random
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

from ravex._distributed import (
    apply_local_sharded_state,
    apply_sharded_state,
    gather_sharded_state,
    get_rank,
    get_world_size,
    is_sharded,
    local_sharded_state,
    unwrap_model,
)

logger = logging.getLogger("ravex")


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()


#: Segments the various wrappers splice into parameter names. A gathered full
#: state dict has them stripped; a live named_parameters() may not.
_WRAPPER_SEGMENTS = (
    "_fsdp_wrapped_module",
    "_checkpoint_wrapped_module",
    "_orig_mod",
    "module",
)


def _clean_fqn(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def _drop_on_death(entries: "OrderedDict[int, weakref.ref]", key: int):
    """Callback that removes a dead entry, unless its id was already reused."""

    def callback(ref: weakref.ref) -> None:
        if entries.get(key) is ref:
            entries.pop(key, None)

    return callback


class _WeakSet:
    """Insertion-ordered weak collection, keyed by object identity.

    Identity keying matters for throughput: ``nn.Module.__init__`` is patched,
    so a large model registers thousands of modules, and a linear membership
    scan would make construction quadratic.
    """

    def __init__(self) -> None:
        self._entries: "OrderedDict[int, weakref.ref]" = OrderedDict()

    def add(self, obj: Any) -> bool:
        """Record ``obj``. Returns True if it was not already present."""
        key = id(obj)
        existing = self._entries.get(key)
        if existing is not None and existing() is obj:
            return False
        try:
            self._entries[key] = weakref.ref(obj, _drop_on_death(self._entries, key))
        except TypeError:  # object does not support weak references
            return False
        return True

    def alive(self) -> List[Any]:
        live: List[Any] = []
        for ref in list(self._entries.values()):
            obj = ref()
            if obj is not None:
                live.append(obj)
        return live

    def __contains__(self, obj: Any) -> bool:
        existing = self._entries.get(id(obj))
        return existing is not None and existing() is obj

    def __len__(self) -> int:
        return len(self.alive())


class ObjectRegistry:
    """Tracks the PyTorch objects the user's code created."""

    def __init__(self) -> None:
        # Every module ever constructed, plus the subset the user explicitly
        # put in training mode. See _root_models() for how a model is picked
        # out of the first set.
        self._seen_modules = _WeakSet()
        self._trained_modules = _WeakSet()

        self._optimizers = _WeakSet()
        self._dataloaders = _WeakSet()
        self._schedulers = _WeakSet()
        self._scalers = _WeakSet()

        self.step_count: int = 0
        self.resumed: bool = False
        self._pending_rng: Optional[Dict[str, Any]] = None

        # Invalidates the cached root-module computation.
        self._generation: int = 0
        self._roots_cache: Optional[List[weakref.ref]] = None
        self._roots_generation: int = -1

    # ─── registration ───────────────────────────────────────────────

    def note_module(self, module: Any) -> None:
        """Record a freshly constructed module. Called for every module."""
        if self._seen_modules.add(module):
            self._generation += 1

    def register_model(self, model: Any) -> None:
        # Module.train() recurses into children, so this fires for every
        # submodule too. Which of them is the actual model is decided in
        # _root_models(); guessing here would be wrong.
        self.note_module(model)
        if self._trained_modules.add(model):
            self._generation += 1

    def register_optimizer(self, optimizer: Any) -> None:
        if self._optimizers.add(optimizer):
            self._generation += 1

    def register_dataloader(self, dataloader: Any) -> None:
        self._dataloaders.add(dataloader)

    def register_scheduler(self, scheduler: Any) -> None:
        self._schedulers.add(scheduler)

    def register_scaler(self, scaler: Any) -> None:
        self._scalers.add(scaler)

    # ─── views ──────────────────────────────────────────────────────

    @property
    def optimizers(self) -> List[Any]:
        return self._optimizers.alive()

    @property
    def dataloaders(self) -> List[Any]:
        return self._dataloaders.alive()

    @property
    def schedulers(self) -> List[Any]:
        return self._schedulers.alive()

    @property
    def scalers(self) -> List[Any]:
        return self._scalers.alive()

    @property
    def models(self) -> List[Any]:
        """Top-level models only."""
        return self._root_models()

    def _root_models(self) -> List[Any]:
        """Work out which of the observed modules are actual models.

        Two filters, applied in order:

        1. **Outermost only.** Every module is observed, so a 300-layer network
           shows up as 300 modules. A module contained in another observed
           module is already covered by its parent's ``state_dict()``.
        2. **Actually part of the training.** A root qualifies if an optimizer
           owns at least one of its parameters, or if the user explicitly put
           it in training mode. That drops the incidental modules libraries
           build behind the scenes (loss wrappers holding buffers, metric
           helpers, tokenizer heads) without needing a blocklist, while still
           catching EMA copies and frozen teachers, which no optimizer owns but
           the user does call ``.train()`` on.
        """
        if self._roots_cache is not None and self._roots_generation == self._generation:
            cached = [ref() for ref in self._roots_cache]
            # The cache holds weak references: a strong one would keep a model
            # alive for the lifetime of the process. A dead entry means the
            # user dropped a model, so recompute.
            if all(module is not None for module in cached):
                return cached

        live = self._seen_modules.alive()
        for module in self._trained_modules.alive():
            if module not in self._seen_modules:
                live.append(module)

        by_id = {id(m): m for m in live}
        contained: Set[int] = set()
        for module in live:
            try:
                for sub in module.modules():
                    if sub is not module and id(sub) in by_id:
                        contained.add(id(sub))
            except Exception:  # pragma: no cover - exotic Module override
                continue

        owned = self._optimized_parameter_ids()
        candidates = [
            module
            for module in live
            if id(module) not in contained and self._has_parameters(module)
        ]

        # **The ownership test is evidence only when it could have succeeded
        # for somebody.** It asks "does an optimizer own one of this module's
        # parameters", and answers no in two very different situations: this
        # module is not part of the training, or *no* module is — because the
        # optimizer's parameters are not any module's.
        #
        # The second is what DeepSpeed does. ZeRO hands the base optimizer its
        # own flat partition buffers, so a run with a live model, a live
        # optimizer and every step working reports zero owned parameters
        # against every candidate. Measured on 2026-08-31 with ZeRO stage 1: a
        # 4-parameter module, one parameter owned by `FusedAdam`, none of them
        # the module's. Both filters then miss — the user's model is dropped as
        # contained in the engine, and the engine is dropped as unowned — and
        # Ravex wrote checkpoints containing an optimizer, a dataloader and no
        # weights at all, logging a successful handoff each time.
        #
        # So when nothing owns anything, the question was not answered and must
        # not be read as a no. Having parameters and being outermost is what is
        # left to go on. This cannot fire on an ordinary run: there, the
        # optimizer was built from a model's parameters and that model owns
        # them.
        # Keyed on *there being optimizers*, not on their holding parameters.
        # At ZeRO stage 3 the base optimizer's param_groups come back empty
        # altogether, so a condition written as "owned and not informative"
        # misses exactly the stage that needs it most - measured, after the
        # first version of this fix left stage 3 with no roots and the old
        # symptom intact.
        informative = any(self._owns_any(module, owned) for module in candidates)
        uninformative = bool(self.optimizers) and not informative
        if uninformative:
            self._warn_once(
                "opaque-optimizer",
                "No module owns any parameter this run's optimizer(s) hold, "
                "which is what DeepSpeed's ZeRO looks like from here - the "
                "optimizer is given flat partition buffers rather than the "
                "model's parameters, and at stage 3 it holds none at all. "
                "Falling back to checkpointing the "
                "outermost module that has parameters, because the ownership "
                "test cannot answer here and reading it as a no would leave "
                "the model out of the checkpoint entirely.",
            )

        roots = []
        for module in candidates:
            if (
                module in self._trained_modules
                or self._owns_any(module, owned)
                or uninformative
            ):
                roots.append(module)

        self._roots_cache = [weakref.ref(module) for module in roots]
        self._roots_generation = self._generation
        return roots

    def _optimized_parameter_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for optimizer in self.optimizers:
            try:
                for group in optimizer.param_groups:
                    for param in group.get("params", []):
                        ids.add(id(param))
            except Exception:  # pragma: no cover - defensive
                continue
        return ids

    @staticmethod
    def _owns_any(module: Any, parameter_ids: Set[int]) -> bool:
        if not parameter_ids:
            return False
        try:
            for param in module.parameters():
                if id(param) in parameter_ids:
                    return True
        except Exception:  # pragma: no cover - defensive
            return False
        return False

    def parameters_are_partitioned_away(self) -> bool:
        """Whether the models are here but their weights are not.

        DeepSpeed's ZeRO stage 3 partitions the *parameters* as well as the
        optimizer state, and what it leaves on the module is an empty shell:
        measured on 2026-08-31, a four-parameter module under stage 3 reports
        four parameters, every one of them ``numel() == 0``, while stages 1 and
        2 report the same four at full size.

        That distinction decides whether Ravex can checkpoint the run at all.
        A state dict taken here would be the right keys, the right dtypes, and
        no data — a checkpoint that restores nothing and says nothing, which is
        the one outcome this project is built to avoid. Reaching the real
        values needs a gather through DeepSpeed's own machinery, which Ravex
        does not drive.

        True only when there is something to be wrong about: models exist, they
        have parameters, and every one of those is empty. A model that has no
        parameters at all is a different situation and is not this one.
        """
        found = False
        for module in self._root_models():
            try:
                for param in module.parameters():
                    found = True
                    if param.numel() != 0:
                        return False
            except Exception:  # pragma: no cover - defensive
                return False
        return found

    @staticmethod
    def _has_parameters(module: Any) -> bool:
        try:
            for _ in module.parameters():
                return True
        except Exception:  # pragma: no cover - defensive
            return False
        return False

    def is_empty(self) -> bool:
        return not self._root_models() and not self.optimizers

    # ─── fingerprints ───────────────────────────────────────────────

    @staticmethod
    def fingerprint_model(model: Any) -> str:
        model = unwrap_model(model)
        parts = [type(model).__name__]
        try:
            for name, param in model.named_parameters():
                parts.append(f"{name}:{tuple(param.shape)}:{param.dtype}")
            for name, buf in model.named_buffers():
                parts.append(f"b:{name}:{tuple(buf.shape)}")
        except Exception:  # pragma: no cover - defensive
            pass
        return f"model_{_digest('|'.join(parts))}"

    @staticmethod
    def fingerprint_optimizer(optimizer: Any) -> str:
        parts = [type(optimizer).__name__]
        try:
            for index, group in enumerate(optimizer.param_groups):
                params = group.get("params", [])
                shapes = ",".join(str(tuple(p.shape)) for p in params)
                parts.append(f"g{index}:{len(params)}:{shapes}")
        except Exception:  # pragma: no cover - defensive
            pass
        return f"optim_{_digest('|'.join(parts))}"

    @staticmethod
    def fingerprint_dataloader(dataloader: Any) -> str:
        parts = [type(getattr(dataloader, "dataset", None)).__name__]
        try:
            parts.append(str(len(dataloader.dataset)))  # type: ignore[arg-type]
        except Exception:
            parts.append("?")
        parts.append(str(getattr(dataloader, "batch_size", None)))
        return f"dl_{_digest('|'.join(parts))}"

    @staticmethod
    def _keyed(objects: List[Any], fingerprint) -> List[tuple]:
        """Pair each object with a unique key.

        Identical twins (a GAN's two discriminators, say) share a fingerprint,
        so a registration-order suffix is appended to keep keys unique and
        still stable across processes.
        """
        counts: Dict[str, int] = {}
        result = []
        for obj in objects:
            base = fingerprint(obj)
            index = counts.get(base, 0)
            counts[base] = index + 1
            result.append((f"{base}#{index}" if index else base, obj))
        return result

    def keyed_models(self) -> List[tuple]:
        return self._keyed(self._plain_models(), self.fingerprint_model)

    def keyed_optimizers(self) -> List[tuple]:
        return self._keyed(self._plain_optimizers(), self.fingerprint_optimizer)

    # ─── sharded models ─────────────────────────────────────────────

    def sharded_groups(self) -> List[tuple]:
        """Sharded models paired with the optimizers that own their parameters.

        FSDP state cannot be collected model-first then optimizer-second: the
        optimizer's moments are sharded against the model's flattened
        parameters, and gathering them needs both objects at once.

        Keys are positional (``sharded_0``, ``sharded_1``) rather than
        structural. A fingerprint built from parameter shapes would encode the
        world size — each rank sees only its shard — and a run sharded over
        eight GPUs would then fail to match itself when resumed on four.
        """
        groups = []
        for model in self._root_models():
            if not is_sharded(model):
                continue
            owners = [o for o in self.optimizers if self._optimizer_owns(o, model)]
            groups.append((f"sharded_{len(groups)}", model, owners))
        return groups

    def has_sharded_models(self) -> bool:
        return any(is_sharded(model) for model in self._root_models())

    def _plain_models(self) -> List[Any]:
        return [model for model in self._root_models() if not is_sharded(model)]

    def _plain_optimizers(self) -> List[Any]:
        """Optimizers not already covered by a sharded group."""
        claimed = set()
        for _, _, owners in self.sharded_groups():
            claimed.update(id(o) for o in owners)
        return [o for o in self.optimizers if id(o) not in claimed]

    @staticmethod
    def _optimizer_owns(optimizer, model) -> bool:
        model_params = {id(p) for p in model.parameters()}
        try:
            for group in optimizer.param_groups:
                for param in group.get("params", []):
                    if id(param) in model_params:
                        return True
        except Exception:  # pragma: no cover - defensive
            return False
        return False

    def keyed_dataloaders(self) -> List[tuple]:
        return self._keyed(self.dataloaders, self.fingerprint_dataloader)

    # ─── state ──────────────────────────────────────────────────────

    def collect_state(
        self, track_rng: bool = True, sharded_layout: str = "gather"
    ) -> Dict[str, Any]:
        """Snapshot everything needed to restart this run.

        Runs on the training thread, synchronously: the returned structure must
        be consistent with the step that just finished, and the background
        writer must not read tensors while the next step mutates them. The
        backend takes its own copy before returning.

        ``sharded_layout`` picks how FSDP state is taken — ``gather`` collects
        it all on rank 0, ``per_rank`` leaves each rank holding its own shard.
        See :mod:`ravex._distributed`.
        """
        import torch

        state: Dict[str, Any] = {
            "ravex_version": 1,
            "step": self.step_count,
            "models": {},
            "optimizers": {},
            "schedulers": {},
            "scalers": {},
            "dataloaders": {},
            "sharded": {},
        }

        for key, model in self.keyed_models():
            state["models"][key] = unwrap_model(model).state_dict()

        for key, optimizer in self.keyed_optimizers():
            # One optimizer refusing to describe itself must not cost the
            # checkpoint the model. Measured on 2026-08-31: under DeepSpeed
            # ZeRO stage 3 the base `FusedAdam`'s `state_dict()` raises a bare
            # `KeyError` on a parameter id, because ZeRO has taken over the
            # mapping it reads. That killed the whole collection every
            # interval, so a run that could have had its weights saved had
            # nothing saved at all, reported as `Checkpoint at step 4 failed:
            # 133268936989168` — a number, and no clue what it belonged to.
            #
            # Skipped and named, once. The weights are the part a resume
            # cannot do without; moments it can, at the cost of a few steps of
            # warm-up.
            try:
                state["optimizers"][key] = optimizer.state_dict()
            except Exception as exc:
                self._warn_once(
                    "optimizer-state-%s" % key,
                    "Optimizer %s (%s) could not produce a state dict (%s: %s), "
                    "so its moments are left out of the checkpoint. Everything "
                    "else is still saved. This is what DeepSpeed's ZeRO looks "
                    "like from here - its engine owns the optimizer's "
                    "bookkeeping and the base optimizer alone cannot describe "
                    "it."
                    % (key, type(optimizer).__name__, type(exc).__name__, exc),
                )

        # Sharded models go through a collective either way, so this must run
        # on every rank — under `gather` even though only rank 0 will write the
        # result, under `per_rank` because every rank writes its own.
        groups = self.sharded_groups()
        layout = self.sharded_layout(sharded_layout)
        for key, model, optimizers in groups:
            if layout == "per_rank":
                model_state, optimizer_state = local_sharded_state(model, optimizers)
            else:
                model_state, optimizer_state = gather_sharded_state(model, optimizers)
            state["sharded"][key] = {
                "model": model_state,
                "optimizer": optimizer_state,
                # Recorded so a resume can say *what* changed rather than
                # failing with a shape error deep inside set_state_dict.
                "parameters": sorted(model_state.keys()),
                "layout": layout,
                # Per-rank shards only mean anything at the topology that wrote
                # them, so the topology is part of the checkpoint.
                "world_size": get_world_size(),
                "rank": get_rank(),
            }

        for index, scheduler in enumerate(self.schedulers):
            try:
                state["schedulers"][f"sched_{index}"] = scheduler.state_dict()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not collect scheduler state: %s", exc)

        for index, scaler in enumerate(self.scalers):
            try:
                state["scalers"][f"scaler_{index}"] = scaler.state_dict()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not collect scaler state: %s", exc)

        for key, dataloader in self.keyed_dataloaders():
            tracked = getattr(dataloader, "_ravex_sampler", None)
            if tracked is not None:
                state["dataloaders"][key] = tracked.state()

        if track_rng:
            state["rng"] = self._collect_rng(torch)

        return state

    def sharded_layout(self, requested: str) -> str:
        """Settle on one layout for every sharded group in this checkpoint.

        All-or-nothing, and decided before anything is collected. Per group,
        one gathering and one keeping its shard would give a checkpoint that is
        partly topology-independent and partly not, and rank 0 would be storing
        its own shard as if it were the whole tensor — wrong in a way nothing
        downstream could notice.

        The test is local and structural (are the parameters DTensors?), so
        every rank reaches the same answer without communicating and the
        collectives that follow still line up. It also decides where the
        checkpoint is written, which is why the runtime asks the same question
        before it opens a backend.
        """
        groups = self.sharded_groups()
        if requested != "per_rank" or not groups:
            return "gather"

        from ravex._distributed import _dtensor_class

        DTensor = _dtensor_class()
        if DTensor is None:
            self._warn_once(
                "no-dtensor",
                "sharded_checkpoints=per_rank needs DTensor support in torch - "
                "gathering on rank 0 instead",
            )
            return "gather"

        for key, model, _ in groups:
            # FSDP2 leaves DTensors in place of the parameters. FSDP1 does not,
            # at any `use_orig_params` setting — measured on torch 2.12: the
            # parameters stay plain and the sharded state dict yields
            # `ShardedTensor`, which has no mesh to rebuild a shard against.
            # So this test is also the FSDP1 test.
            if not any(isinstance(p, DTensor) for p in model.parameters()):
                self._warn_once(
                    "not-dtensor-backed:%s" % key,
                    "Sharded group %s is not DTensor-backed - per-rank "
                    "checkpointing needs FSDP2; gathering on rank 0 instead"
                    % key,
                )
                return "gather"
        return "per_rank"

    def _warn_once(self, key: str, message: str) -> None:
        """Log once per reason. This runs on every checkpoint and every resume,
        and the same downgrade repeated 500 times buries whatever comes next."""
        seen = self.__dict__.setdefault("_warned", set())
        if key in seen:
            return
        seen.add(key)
        logger.warning("%s", message)

    @staticmethod
    def _collect_rng(torch) -> Dict[str, Any]:
        rng: Dict[str, Any] = {
            "torch": torch.random.get_rng_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            try:
                rng["cuda"] = torch.cuda.get_rng_state_all()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not collect CUDA RNG state: %s", exc)
        try:
            import numpy as np

            rng["numpy"] = np.random.get_state()
        except ImportError:
            pass
        return rng

    def restore_state(self, state: Dict[str, Any], defer_rng: bool = False) -> None:
        """Apply a checkpoint to the live objects.

        Anything that does not match is reported and skipped: a partially
        restored run that keeps going beats a crash on startup.

        ``defer_rng`` holds the global RNG back for :meth:`apply_pending_rng`.
        Building a ``DataLoader`` iterator draws its worker base seed from the
        global RNG, so restoring before that draw leaves the generator one step
        ahead of where the original run was — enough to change every dropout
        mask from the first resumed step onwards.
        """
        import torch

        self.step_count = int(state.get("step", 0))

        saved_models = state.get("models", {})
        unmatched_models = []
        for key, model in self.keyed_models():
            if key not in saved_models:
                unmatched_models.append((key, model))
                continue
            unwrap_model(model).load_state_dict(saved_models[key])

        saved_optimizers = state.get("optimizers", {})
        for key, optimizer in self.keyed_optimizers():
            if key not in saved_optimizers:
                logger.warning("No saved state for optimizer %s", key)
                continue
            optimizer.load_state_dict(saved_optimizers[key])

        saved_sharded = state.get("sharded", {})
        unmatched_sharded = []
        claimed_sharded = set()
        for key, model, optimizers in self.sharded_groups():
            saved = saved_sharded.get(key)
            if saved is None:
                unmatched_sharded.append((key, model, optimizers))
                continue
            claimed_sharded.add(key)
            expected = saved.get("parameters")
            if expected is not None:
                self._warn_on_parameter_mismatch(key, model, expected)
            # Collective: every rank has to reach this, which it does because
            # the resume fires from DataLoader.__iter__ on all of them.
            if saved.get("layout") == "per_rank":
                written_by = int(saved.get("world_size", 0))
                if written_by != get_world_size():
                    # Every rank reads the same world_size out of its own
                    # checkpoint and compares it against the same live one, so
                    # they all skip together and no collective is stranded.
                    logger.warning(
                        "Sharded group %s was written per-rank by a %d-rank run "
                        "and this run has %d. Per-rank shards cannot be "
                        "reshaped; skipping it. Write with "
                        "sharded_checkpoints=gather to resume at a different "
                        "world size.",
                        key,
                        written_by,
                        get_world_size(),
                    )
                    continue
                apply_local_sharded_state(
                    model, optimizers, saved["model"], saved["optimizer"]
                )
            else:
                apply_sharded_state(
                    model, optimizers, saved["model"], saved["optimizer"]
                )

        self._bridge_topologies(
            unmatched_models,
            unmatched_sharded,
            saved_models,
            saved_sharded,
            claimed_sharded,
        )

        saved_schedulers = state.get("schedulers", {})
        for index, scheduler in enumerate(self.schedulers):
            key = f"sched_{index}"
            if key in saved_schedulers:
                try:
                    scheduler.load_state_dict(saved_schedulers[key])
                except Exception as exc:
                    logger.warning("Could not restore scheduler %s: %s", key, exc)

        saved_scalers = state.get("scalers", {})
        for index, scaler in enumerate(self.scalers):
            key = f"scaler_{index}"
            if key in saved_scalers:
                try:
                    scaler.load_state_dict(saved_scalers[key])
                except Exception as exc:
                    logger.warning("Could not restore scaler %s: %s", key, exc)

        saved_dataloaders = state.get("dataloaders", {})
        for key, dataloader in self.keyed_dataloaders():
            tracked = getattr(dataloader, "_ravex_sampler", None)
            if tracked is not None and key in saved_dataloaders:
                tracked.restore(saved_dataloaders[key])

        rng = state.get("rng")
        if isinstance(rng, dict):
            if defer_rng:
                self._pending_rng = rng
            else:
                self._restore_rng(torch, rng)

        self.resumed = True

    def _bridge_topologies(
        self,
        unmatched_models: List[tuple],
        unmatched_sharded: List[tuple],
        saved_models: Dict[str, Any],
        saved_sharded: Dict[str, Any],
        claimed_sharded: set,
    ) -> None:
        """Match a checkpoint to a differently-shaped run.

        The gathered state is topology-independent by construction, which is
        the point of gathering it: eight GPUs in, one out. But the *keys* are
        not. A sharded run files its model under ``sharded``, an ordinary run
        under ``models``, and neither looks in the other's drawer. So resuming
        an FSDP job in a plain single-process script found nothing, kept its
        random initialisation, and carried on - with the step counter restored
        and checkpoints still being written, so the only sign was one warning
        and a loss back at its starting value.

        Whatever is left unmatched on both sides is paired up here, in order.
        """
        leftover_sharded = [key for key in saved_sharded if key not in claimed_sharded]
        leftover_plain = [
            key
            for key in saved_models
            if key not in {k for k, _ in self.keyed_models()}
        ]

        for (key, model), saved_key in zip(unmatched_models, leftover_sharded):
            group = saved_sharded[saved_key]
            logger.info(
                "Model %s takes the state saved by sharded group %s "
                "(resuming a distributed run in a plain one)",
                key,
                saved_key,
            )
            unwrap_model(model).load_state_dict(group["model"])
            self._bridge_optimizer(model, group)

        for (key, model, optimizers), saved_key in zip(
            unmatched_sharded, leftover_plain
        ):
            logger.info(
                "Sharded group %s takes the state saved by plain model %s "
                "(resuming a single-process run in a distributed one)",
                key,
                saved_key,
            )
            try:
                apply_sharded_state(model, [], saved_models[saved_key], {})
            except Exception as exc:
                logger.warning("Could not scatter %s onto the shards: %s", saved_key, exc)

        for key, model in unmatched_models[len(leftover_sharded) :]:
            logger.warning(
                "No saved state for model %s (%s) - it keeps its initial weights. "
                "Did the architecture change since the checkpoint?",
                key,
                type(unwrap_model(model)).__name__,
            )
        for key, _, _ in unmatched_sharded[len(leftover_plain) :]:
            logger.warning(
                "No saved state for sharded model %s - it keeps its initial weights",
                key,
            )

    def _bridge_optimizer(self, model: Any, group: Dict[str, Any]) -> None:
        """Carry the optimizer across too, if one owns this model.

        The saved optimizer state is keyed by parameter name rather than by
        position, which is what makes it portable; translating it back onto a
        plain optimizer is the loader's job.
        """
        optimizers = [o for o in self.optimizers if self._optimizer_owns(o, model)]
        if not optimizers or not group.get("optimizer"):
            return
        try:
            apply_sharded_state(model, optimizers, group["model"], group["optimizer"])
        except Exception as exc:
            logger.warning(
                "Model weights were restored but the optimizer state was not (%s). "
                "Training continues from the right weights with fresh moments.",
                exc,
            )

    @staticmethod
    def _warn_on_parameter_mismatch(key: str, model: Any, expected: List[str]) -> None:
        """Report an architecture change before set_state_dict trips over it.

        Sharded keys are positional, so a changed model still matches by key and
        the failure would otherwise surface as an opaque shape error from deep
        inside the loading machinery.
        """
        try:
            current = {_clean_fqn(name) for name, _ in model.named_parameters()}
        except Exception:  # pragma: no cover - defensive
            return

        wanted = {_clean_fqn(name) for name in expected}
        if current == wanted:
            return

        # Only shout when the difference is real. Wrapper infixes vary between
        # FSDP1, FSDP2 and activation checkpointing, and _clean_fqn cannot know
        # every one of them; a warning on every resume would train the user to
        # ignore it.
        if len(current) != len(wanted) or not (current & wanted):
            logger.warning(
                "Sharded model %s does not look like the checkpoint "
                "(%d parameters now, %d saved). If the architecture changed, "
                "loading is about to fail.",
                key,
                len(current),
                len(wanted),
            )

    def apply_pending_rng(self) -> bool:
        """Apply an RNG state held back by ``restore_state(defer_rng=True)``."""
        rng, self._pending_rng = self._pending_rng, None
        if rng is None:
            return False
        import torch

        self._restore_rng(torch, rng)
        return True

    @staticmethod
    def _restore_rng(torch, rng: Dict[str, Any]) -> None:
        try:
            if "torch" in rng:
                torch.random.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if "python" in rng:
                # Pickle round-trips tuples as lists; random.setstate is strict.
                pystate = rng["python"]
                if isinstance(pystate, list):
                    pystate = (pystate[0], tuple(pystate[1]), pystate[2])
                random.setstate(pystate)
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [s.cpu().to(torch.uint8) for s in rng["cuda"]]
                )
            if "numpy" in rng:
                import numpy as np

                np.random.set_state(rng["numpy"])
        except Exception as exc:
            logger.warning("Could not fully restore RNG state: %s", exc)
