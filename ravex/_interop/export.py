"""Writing a Ravex checkpoint out as a torch distributed checkpoint — GPU-90.

The reading half is the rest of this package: a DeepSpeed ZeRO store or a DCP
directory becomes something Ravex's resume consumes. This is the other
direction, and the one the issue's own example needed — *start in FSDP, resume
in Megatron* — because Megatron-core, a plain FSDP job and anything built on
``torch.distributed.checkpoint.load`` read DCP. Ravex's store is a format only
Ravex reads; this is how a checkpoint leaves it.

**What is written, and how exact each half is.** It depends on what the
checkpoint holds, and the difference is reported as notes rather than smoothed
over:

* A **sharded group in** ``gather`` **layout** exports exactly. Its model state
  is keyed by fully qualified name and so is its optimizer state, because it
  was collected through ``get_state_dict`` — which is the form a DCP loader on
  the other side expects.
* A **plain model** exports its weights exactly. Its optimizer does not come
  out keyed by name, because it was collected with ``optimizer.state_dict()``
  and that keys the moments by the parameter's *position* in an ordering only
  the writing run knew. Written as it is and said so, which is exactly what
  :func:`ravex._interop.convert._unify_dcp` recognises and reports on the way
  back in. Inventing names by assuming the optimizer was built from
  ``model.parameters()`` in order would be right most of the time and silently
  wrong the rest — moments on the wrong parameters, with every shape plausible.
* A ``per_rank`` checkpoint is **refused**. Each rank's store holds a shard
  that only means something at the topology that wrote it; exporting one would
  write a fraction of a model under a format that does not record it is one.

Written with ``dcp.save(..., no_dist=True)``, the public entry point, present
since well before torch 2.8 — a single process writing a directory, no process
group, for the same reason the reader needs none.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ravex._interop.convert import CannotConvert


def _choose(state: Dict[str, Any], key: Optional[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[str]]:
    """The model and optimizer state to write, and what is worth saying about them."""
    sharded = state.get("sharded") or {}
    models = state.get("models") or {}
    optimizers = state.get("optimizers") or {}

    candidates = list(sharded) + [name for name in models if name not in sharded]
    if key is None:
        if len(candidates) != 1:
            raise CannotConvert(
                "this checkpoint holds %d models (%s); name the one to export"
                % (len(candidates), ", ".join(candidates) or "none")
            )
        key = candidates[0]
    elif key not in candidates:
        raise CannotConvert(
            "no model %r in this checkpoint; it holds %s"
            % (key, ", ".join(candidates) or "none")
        )

    if key in sharded:
        group = sharded[key]
        if group.get("layout") == "per_rank":
            raise CannotConvert(
                "%r was written per rank (rank %s of %s): each store holds one "
                "shard, meaningful only at the topology that wrote it. Resume "
                "it at that size, or reshard it, and export the gathered result"
                % (key, group.get("rank"), group.get("world_size"))
            )
        notes = ["model and optimizer taken from sharded group %r, keyed by name" % key]
        optimizer = group.get("optimizer") or None
        if not optimizer:
            notes.append("the group carries no optimizer state")
        return group["model"], optimizer, notes

    notes = ["model taken from %r" % key]
    optimizer = None
    if len(optimizers) == 1:
        optimizer = next(iter(optimizers.values()))
        notes.append(
            "the optimizer state is keyed by position, as optimizer.state_dict() "
            "writes it: it restores onto a model whose optimizer lists the same "
            "parameters in the same order, and cannot be matched by name"
        )
    elif optimizers:
        notes.append(
            "%d optimizers in this checkpoint and no way to tell which one "
            "belongs to %r; none exported" % (len(optimizers), key)
        )
    else:
        notes.append("no optimizer state in this checkpoint")
    return models[key], optimizer, notes


def to_dcp(state: Dict[str, Any], out_dir: str, key: Optional[str] = None) -> List[str]:
    """Write one model of a Ravex checkpoint, and its optimizer, as DCP.

    ``state`` is a checkpoint as a backend's ``load_step`` or ``load_latest``
    returns it. The directory gets ``{"model": ..., "optim": ...}`` — the two
    top-level keys a DCP reader looks for first — plus ``ravex``, which records
    the step and where the state came from so the export can be traced back.

    Returns notes on what was and was not exported exactly.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter

    model, optimizer, notes = _choose(state, key)
    payload: Dict[str, Any] = {
        "model": dict(model),
        "ravex": {"step": state.get("step"), "exported_from": "ravex"},
    }
    if optimizer:
        payload["optim"] = optimizer

    dcp.save(payload, storage_writer=FileSystemWriter(out_dir), no_dist=True)
    return notes


# ─── DeepSpeed: the universal checkpoint (GPU-145) ──────────────────

#: The directory DeepSpeed's universal loader looks for, by the name it reads
#: out of ``latest_universal``. Its own converter, ``ds_to_universal``, names
#: its output the same way: the source tag with ``_universal`` after it.
UNIVERSAL_SUFFIX = "_universal"

#: The DeepSpeed version whose universal layout this writes, and the one
#: ``integration/frameworks/check_deepspeed_export.py`` resumed it with.
#: DeepSpeed parses ``ds_version`` as a version and compares against it when
#: it loads, so it has to be a real one; who wrote the files is said in
#: ``written_by`` instead.
DEEPSPEED_FORMAT_VERSION = "0.19.7"


def _named_moments(optimizer: Optional[Dict[str, Any]], model: Dict[str, Any]) -> Tuple[Optional[Dict[str, Dict[str, Any]]], List[Dict[str, Any]], List[str]]:
    """The optimizer's per-parameter state keyed by parameter name, its groups,
    and what is worth saying about how the names were found.

    Two shapes carry names, and only those two are used:

    * keyed by name already - a sharded group collected through
      ``get_state_dict`` - where ``state`` is ``{fqn: {...}}`` and each group's
      ``params`` lists fully qualified names;
    * keyed by position with ``param_names`` in each group, which torch records
      when the optimizer was built from ``model.named_parameters()``.

    Keyed by position with no names, the moments are **not** exported: which
    parameter "index 4" was is something only the run that wrote it knew.
    Assuming ``model.parameters()`` order would be right most of the time and,
    the rest of the time, put Adam's moments on the wrong tensors with every
    shape plausible - the failure the DCP export refuses for the same reason.
    """
    if not optimizer:
        return None, [], ["no optimizer state in this checkpoint: the weights only"]
    state = optimizer.get("state") or {}
    groups = list(optimizer.get("param_groups") or [])

    if state and all(isinstance(key, str) for key in state):
        named = {str(key): value for key, value in state.items()}
        return named, groups, ["optimizer state keyed by parameter name"]

    names: Dict[Any, str] = {}
    for group in groups:
        params, labels = group.get("params") or [], group.get("param_names") or []
        if len(labels) == len(params):
            names.update(zip(params, labels))
    if state and all(key in names for key in state):
        named = {names[key]: value for key, value in state.items()}
        for name in named:
            if name not in model:
                raise CannotConvert(
                    "the optimizer names a parameter %r the model does not have; "
                    "they are not from one training run" % name
                )
        return named, groups, ["optimizer state named through its param_names"]

    return None, groups, [
        "the optimizer state is keyed by position with no parameter names, so "
        "its moments are not exported: DeepSpeed resumes these weights with a "
        "fresh optimizer. Build the optimizer from model.named_parameters() for "
        "the moments to come along"
    ]


def _plain(value: Any) -> Any:
    """A hyperparameter as DeepSpeed's own files hold it: numbers and tuples,
    not one-element tensors."""
    import torch

    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value


def to_deepspeed(state: Dict[str, Any], out_dir: str, key: Optional[str] = None) -> List[str]:
    """Write one model of a Ravex checkpoint as a DeepSpeed *universal* checkpoint.

    The other direction of GPU-90's ZeRO reader: "start in FSDP, resume in
    DeepSpeed". Universal and not ZeRO's own per-rank files, because a ZeRO
    checkpoint is a shard per data-parallel rank at one stage, and writing it
    would mean choosing the world size and stage of a run that has not started
    yet. The universal format holds one file per parameter and per optimizer
    buffer, whole, and DeepSpeed partitions it itself at load time for whatever
    topology it is resumed on - its loader exists for exactly that, and its own
    converter ``ds_to_universal`` produces the layout written here::

        out_dir/latest_universal                    the tag directory's name
        out_dir/<tag>_universal/
            mp_rank_00_model_states.pt              module, param_shapes, steps
            zero_pp_rank_0_mp_rank_00_model_states.pt   the same, as stage 3 names it
            zero/optimizer_state.pt                 param_groups and ZeRO's flags
            zero/<parameter>/fp32.pt                {"param": tensor, "cat_dim": 0}
            zero/<parameter>/exp_avg.pt             the same, per Adam buffer
            zero/<parameter>/step.pt                the step, an int

    Resumed with ``"checkpoint": {"load_universal": true}`` in the DeepSpeed
    config and ``engine.load_checkpoint(out_dir)``, at ZeRO stage 1, 2 or 3 and
    any data-parallel size. ``integration/frameworks/check_deepspeed_export.py``
    checks that against DeepSpeed itself: weights and moments come back equal
    by name, and the next step equals the one from DeepSpeed's own converter.
    It needs DeepSpeed's Adam (its default), not ``torch_adam``: the universal
    loader restores the step as an int, as DeepSpeed's converter writes it,
    and torch's Adam wants a tensor there.

    The weights are written in fp32: they are what the optimizer's master copy
    starts from, and the model DeepSpeed builds casts them to its own dtype.

    Returns notes on what was and was not exported exactly.
    """
    import os
    import shutil
    from collections import OrderedDict

    import torch

    model, optimizer, notes = _choose(state, key)
    moments, groups, said = _named_moments(optimizer, model)
    notes.extend(said)

    step = int(state.get("step") or 0)
    tag = "global_step%d%s" % (step, UNIVERSAL_SUFFIX)
    root = os.path.join(out_dir, tag)
    zero = os.path.join(root, "zero")
    os.makedirs(zero, exist_ok=True)

    # Parameters are what the optimizer stepped, when it says; otherwise every
    # floating tensor of the model, and anything else is a buffer. A model
    # without an optimizer to ask cannot tell a batch-norm statistic from a
    # weight, and treats it as a weight: DeepSpeed then gives it an fp32
    # master copy it never steps, which costs memory and nothing else.
    if moments is not None:
        params = [name for name in model if name in moments]
    else:
        params = [name for name, t in model.items() if torch.is_tensor(t) and t.is_floating_point()]
    buffers = [name for name in model if name not in params]

    for name in params:
        folder = os.path.join(zero, name)
        os.makedirs(folder, exist_ok=True)
        tensor = model[name].detach().to(dtype=torch.float32, device="cpu").contiguous()
        torch.save({"param": tensor, "cat_dim": 0}, os.path.join(folder, "fp32.pt"))
        entry = (moments or {}).get(name) or {}
        for buffer, value in entry.items():
            if buffer == "step":
                torch.save(int(_plain(value)), os.path.join(folder, "step.pt"))
            elif torch.is_tensor(value) and value.numel() == tensor.numel():
                torch.save(
                    {"param": value.detach().to(dtype=torch.float32, device="cpu").reshape(tensor.shape).contiguous(), "cat_dim": 0},
                    os.path.join(folder, "%s.pt" % buffer),
                )
        if "step" not in entry:
            torch.save(step, os.path.join(folder, "step.pt"))

    # The optimizer's step, which stage 3 reads once for every parameter
    # rather than per parameter as stages 1 and 2 do.
    steps = [int(_plain(entry["step"])) for entry in (moments or {}).values() if "step" in entry]
    optimizer_step = steps[0] if steps else step

    param_groups = []
    for group in groups or [{}]:
        clean = {k: _plain(v) for k, v in group.items() if k not in ("params", "param_names")}
        clean["params"] = list(range(len(group.get("params") or [])))
        param_groups.append(clean)
    torch.save(
        {
            "param_groups": param_groups,
            # What ZeRO records about itself, at the trivial values: this is
            # not a partition of anything, and the loader partitions it anew.
            #
            # No loss scaler, overflow flag or clip_grad. DeepSpeed reads each
            # with ``sd.get(key, its own)``, so leaving them out keeps the
            # resuming run's configuration - where writing them would replace
            # it: a clip_grad of 0 here switched its gradient clipping off.
            "zero_stage": 2,
            "partition_count": [1] * len(param_groups),
            "group_paddings": [0] * len(param_groups),
            # Stage 3 takes the step and the hyperparameters from here.
            "optimizer_state_dict": {"state": {0: {"step": optimizer_step}}, "param_groups": param_groups},
            "ds_version": DEEPSPEED_FORMAT_VERSION,
            "written_by": "ravex",
        },
        os.path.join(zero, "optimizer_state.pt"),
    )

    torch.save(
        {
            "module": OrderedDict((name, model[name]) for name in model),
            "param_shapes": [OrderedDict((name, model[name].shape) for name in params)],
            "buffer_names": buffers,
            "frozen_param_shapes": OrderedDict(),
            "frozen_param_fragments": OrderedDict(),
            "shared_params": {},
            "global_steps": step,
            "global_samples": 0,
            "skipped_steps": 0,
            "dp_world_size": 1,
            "mp_world_size": 1,
            "lr_scheduler": None,
            "optimizer": None,
            "sparse_tensor_module_names": set(),
            "checkpoint_parallel_dimensions": {"pp_degree": 1, "tp_degree": 1},
            "ds_version": DEEPSPEED_FORMAT_VERSION,
            "written_by": "ravex",
        },
        os.path.join(root, "mp_rank_00_model_states.pt"),
    )
    # Stage 3 looks for its model states under the per-rank name and refuses
    # a directory without one. A single copy under rank 0's name is enough at
    # any world size - measured at 1, 2 and 3 - since what stage 3 restores
    # comes from the per-parameter files, and the model states only have to
    # be there; DeepSpeed's own converter writes one per rank of the run it
    # converted, which is a topology this checkpoint does not have.
    shutil.copyfile(
        os.path.join(root, "mp_rank_00_model_states.pt"),
        os.path.join(root, "zero_pp_rank_0_mp_rank_00_model_states.pt"),
    )
    with open(os.path.join(out_dir, "latest_universal"), "w", encoding="utf-8") as handle:
        handle.write(tag)
    notes.append(
        "written as a DeepSpeed universal checkpoint: resume with "
        '"checkpoint": {"load_universal": true} in the DeepSpeed config'
    )
    return notes


def load_store(path: str, backend: str = "moonclip", step: Optional[int] = None) -> Dict[str, Any]:
    """Open a Ravex store read-only and return one checkpoint from it.

    The latest when ``step`` is not given. Raises :class:`CannotConvert` when
    there is nothing there, rather than handing an empty state to the writer.
    """
    from ravex._backends import get_backend
    from ravex._config import RavexConfig

    config = RavexConfig()
    config.storage.path = path
    config.backend = backend
    config._normalize()
    store = get_backend(config)
    try:
        state = store.load_latest() if step is None else store.load_step(step)
    finally:
        store.close()
    if not state:
        raise CannotConvert(
            "no checkpoint %sin the %s store at %s"
            % ("" if step is None else "at step %d " % step, backend, path)
        )
    return state
