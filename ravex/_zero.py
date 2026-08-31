"""Reading a DeepSpeed ZeRO checkpoint back into whole tensors — GPU-90.

:mod:`ravex._foreign` decides *what* a directory is without opening anything.
This opens it.

**Stages 1 and 2 share an assembly; stage 3 does not.** That distinction was
not obvious from the file layouts — stage 3 looks like the others, a flat fp32
buffer per rank per parameter group — and this module was first written
assuming one algorithm covered all three. It reconstructed stages 1 and 2
bit-exactly and stage 3 to within 0.13 per element: every shape right, every
value wrong, which is exactly the failure that passes a summary statistic. The
oracle caught it. That is the entire argument for having one.

The difference is *what a partition is a partition of*:

- **Stages 1 and 2** cut the whole parameter group into one contiguous slice
  per rank. Concatenating the slices in rank order rebuilds the flat group, and
  walking the recorded shapes in order cuts it back into tensors.
- **Stage 3** partitions *each parameter individually*: every rank holds
  ``ceil(numel / world_size)`` elements of parameter one, then of parameter
  two, and so on. So the pieces of a single tensor are spread across the ranks
  at the same offset, and rebuilding it means gathering across ranks
  per parameter rather than concatenating whole partitions.

Assembling stage 3 the stage-1 way interleaves the second half of the model
into the first. Nothing about the shapes objects, because the element counts
work out either way — which is why this reader now decides its assembly from
the recorded stage and refuses a stage it does not know.

Where each stage files the partition:

===========  ===================================  =========================
stage        flat fp32 partition                  model file
===========  ===================================  =========================
1 and 2      ``single_partition_of_fp32_groups``   shared, holds real tensors
3            ``fp32_flat_groups``                  one per rank, tensors empty
===========  ===================================  =========================

**The parameters are read from the optimizer, never from ``module``.** At
stage 3 there is no choice — the model file's tensors are ``tensor(0,)``
placeholders, the data is not there. At stages 1 and 2 ``module`` does hold
real tensors, and taking them would still be wrong under mixed precision: it
is the *working* copy, in fp16 or bf16, while the fp32 master weights the
optimizer steps live in the partitions. Reading ``module`` would silently
return a lower-precision answer that looks entirely correct on an fp32 run,
which is the kind of bug that surfaces months later as a resumed model that
trains slightly worse.

**Padding is dropped by never asking for it.** Slicing exactly ``sum(numel)``
elements from the front of the reassembled group leaves any surplus behind
without anything having to know how much there was. That is not a shortcut, it
is the only thing that survives contact with the real layouts: the partitions
are **not** equal — 28518 parameters over two ranks came back as 14260 and
14258, aligned rather than divided — and ``group_paddings`` was ``[0]`` in
every configuration that could be produced, including that one. So the field
does not mean "the slack at the end", and a reader that subtracted it would be
subtracting a number whose meaning it had guessed. It is reported when it
disagrees with the surplus and otherwise ignored.

The parameter partitions and the optimizer moments are also **not the same
length**: at stage 1 the last rank's parameters are trimmed to the real
remainder while its moments keep the full aligned length. Pairing the two by
length drops that rank and loses half the model, which is what a first version
of this module did.

**On trusting this.** DeepSpeed ships ``zero_to_fp32.py`` *inside* every
checkpoint, which reconstructs the same tensors its own way. That makes this
module checkable against the thing that wrote the file rather than against its
author's reading of the format, and
``integration/frameworks/check_zero_oracle.py`` does exactly that. A
reimplementation of a format is worth having only when something independent
can say it is right.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class ZeroUnsupported(Exception):
    """A ZeRO checkpoint this reader will not attempt.

    Separate from ``ValueError`` for the same reason
    :class:`ravex._reshard.ReshardUnsupported` is: "I refuse, and here is why"
    is a different thing from "something went wrong", and only the first is
    something a user can act on.
    """


def unshard(found, loader) -> Dict[str, Any]:
    """Every fp32 parameter of a ZeRO checkpoint, whole, keyed by its name.

    ``found`` is a :class:`ravex._foreign.Foreign` describing a ``deepspeed``
    checkpoint; ``loader`` opens one ``.pt`` path and returns what is in it, so
    this module names no torch API for I/O and the caller keeps control of how
    files are read.

    Returns ``{"model": {name: tensor}, "optimizer": {...}, "step": int | None,
    "world_size": int, "stage": int}``.

    ``optimizer`` is ``{"state": {name: {"exp_avg": t, ...}}, "param_groups":
    [...]}`` — torch's own optimizer state dict shape, except keyed by
    parameter *name* instead of by position. Position is what makes an
    optimizer state dict useless across frameworks: index 4 means whatever the
    fourth parameter happened to be when it was written. The names come from
    the same ``param_shapes`` the weights are cut with, so the two halves
    cannot disagree about which parameter is which.
    """
    import os

    import torch

    ranks = _optim_files(found)
    if not ranks:
        raise ZeroUnsupported("no ZeRO optimizer shard was found in %s" % found.root)

    shapes, step = _shapes_and_step(found, loader, os)
    if shapes is None:
        raise ZeroUnsupported(
            "no model state file in %s, so the parameter shapes are unknown - "
            "an optimizer-only checkpoint cannot be turned back into named "
            "tensors" % found.root
        )

    stage: Optional[int] = None
    groups: List[List[Any]] = []
    paddings: Optional[Sequence[int]] = None
    #: ``moments[key][group]`` is the list of per-rank partitions of one
    #: optimizer buffer (``exp_avg`` and friends), in rank order — the same
    #: shape of thing as ``groups``, and reassembled by the same function.
    moments: Dict[str, List[List[Any]]] = {}
    scalars: Dict[str, Dict[int, Any]] = {}
    param_groups: Optional[Any] = None

    for rank, name in ranks:
        blob = loader(os.path.join(found.root, name))
        osd = blob.get("optimizer_state_dict", blob)
        here = int(osd.get("zero_stage", blob.get("zero_stage", 0)) or 0)
        if stage is None:
            stage = here
        elif here != stage:
            raise ZeroUnsupported(
                "rank %d says ZeRO stage %d and an earlier rank said %d: these "
                "shards are not from one checkpoint" % (rank, here, stage)
            )

        flat = _flat_partitions(osd, blob)
        if flat is None:
            raise ZeroUnsupported(
                "rank %d's shard holds no fp32 partition under any name this "
                "reader knows (looked for single_partition_of_fp32_groups and "
                "fp32_flat_groups)" % rank
            )
        if not groups:
            groups = [[] for _ in flat]
        elif len(flat) != len(groups):
            raise ZeroUnsupported(
                "rank %d has %d parameter group(s) and an earlier rank had %d"
                % (rank, len(flat), len(groups))
            )
        for index, part in enumerate(flat):
            groups[index].append(part)

        if paddings is None:
            recorded = osd.get("group_paddings")
            if isinstance(recorded, (list, tuple)):
                paddings = [int(p) for p in recorded]

        base = _base_state(osd, blob)
        if base is not None:
            if param_groups is None:
                param_groups = base.get("param_groups")
            _collect_moments(base, flat, rank, moments, scalars, len(groups))

    if len(shapes) != len(groups):
        raise ZeroUnsupported(
            "the model file describes %d parameter group(s) and the shards hold "
            "%d" % (len(shapes), len(groups))
        )

    if stage not in (1, 2, 3):
        raise ZeroUnsupported(
            "this checkpoint records ZeRO stage %r, and the assembly differs "
            "per stage - refusing rather than reading it as a stage this "
            "reader does know" % stage
        )

    notes: List[str] = []

    def assemble(parts, index, check_padding=True):
        flat = [part.flatten() for part in parts]
        if stage == 3:
            return _cut_per_parameter(flat, shapes[index], index, torch)
        return _cut(
            torch.cat(flat), shapes[index], index,
            paddings if check_padding else None, torch, notes,
        )

    model: Dict[str, Any] = {}
    for index, parts in enumerate(groups):
        model.update(assemble(parts, index))

    # The moments go through the identical assembly: they are partitioned the
    # same way as the weights, and that assembly is already proven bit-exact
    # on the weights against DeepSpeed's own reader.
    #
    # The recorded padding is deliberately not re-checked here. It describes
    # the *parameter* partition, and at stage 1 the moments do not share it —
    # the last rank's parameters are trimmed to the real remainder while its
    # moments keep the padded length. Re-using that number would reject every
    # correctly assembled moment. What still holds, and is still checked
    # inside the assembly, is that the pieces add up to what the shapes ask
    # for; the surplus falls off the end.
    state: Dict[str, Dict[str, Any]] = {}
    for key, per_group in sorted(moments.items()):
        for index, parts in enumerate(per_group):
            if not parts:
                continue
            for name, tensor in assemble(parts, index, check_padding=False).items():
                state.setdefault(name, {})[key] = tensor

    # Scalars are per group, not per element - `step` is one number for every
    # parameter in the group and does not get partitioned. Copied onto each
    # name rather than kept beside the groups so that one entry per parameter
    # is the whole story and a caller never has to look in two places.
    for key, per_group in sorted(scalars.items()):
        for index, value in per_group.items():
            if index >= len(shapes):
                continue
            for name in shapes[index]:
                state.setdefault(name, {})[key] = value

    return {
        "model": model,
        "optimizer": {"state": state, "param_groups": param_groups},
        "step": step,
        "world_size": len(ranks),
        "stage": stage,
        "notes": notes,
    }


def _base_state(osd, blob):
    """The plain torch optimizer state dict buried inside a ZeRO shard.

    Filed under a different key per stage, and neither key is where the other
    puts it: stages 1 and 2 use ``base_optimizer_state``, stage 3 nests a
    second ``optimizer_state_dict`` inside the first. Recognised by *shape* —
    a mapping carrying ``state`` — rather than by looking up the key the
    recorded stage implies, so a shard whose stage and layout disagree still
    finds the right object instead of confidently finding nothing.
    """
    seen = []
    for holder in (osd, blob):
        if not hasattr(holder, "get"):
            continue
        for key in ("base_optimizer_state", "optimizer_state_dict"):
            candidate = holder.get(key)
            if hasattr(candidate, "get") and "state" in candidate:
                return candidate
            if candidate is not None:
                seen.append(candidate)
        if "state" in holder and "param_groups" in holder:
            return holder
    for candidate in seen:
        if hasattr(candidate, "get") and "state" in candidate:
            return candidate
    return None


def _collect_moments(base, flat, rank, moments, scalars, group_count) -> None:
    """One rank's optimizer buffers, filed per group in rank order.

    **Collected without filtering, checked once at assembly.** The obvious
    guard — take a buffer as partitioned only when it is exactly as long as
    that group's flat parameter partition — is wrong, and measurably so. At
    stage 1 DeepSpeed trims the *last* rank's parameter partition down to the
    real remainder while leaving its moments at the full padded length: with
    three 97x97 layers over two ranks, rank 1 holds 14258 parameters and 14260
    elements of ``exp_avg``. A length test rejects exactly that rank, and the
    reconstruction then comes up half the model short.

    So the check that matters is the one the assembly already does: the pieces
    must add up to what the recorded shapes ask for. That is a statement about
    the whole group rather than about one rank, which is the level the padding
    lives at.

    A tensor of one element is read as a scalar rather than as a partition —
    ``step`` is stored that way by some torch versions. The two are only
    ambiguous for a parameter group holding a single element in total, which
    would be a model with one number in it.
    """
    import torch

    state = base.get("state")
    if not hasattr(state, "items"):
        return

    for raw_index, entry in state.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index >= max(group_count, len(flat)) or index >= len(flat):
            continue

        for key, value in entry.items():
            if torch.is_tensor(value) and value.numel() > 1:
                per_group = moments.setdefault(key, [[] for _ in flat])
                while len(per_group) <= index:
                    per_group.append([])
                per_group[index].append(value)
            elif torch.is_tensor(value) and value.numel() == 1:
                scalars.setdefault(key, {}).setdefault(index, value.item())
            elif isinstance(value, (int, float)):
                scalars.setdefault(key, {}).setdefault(index, value)


def _cut(flat, shapes, group: int, paddings, torch, notes) -> Dict[str, Any]:
    """Slice one flat group back into named tensors, in the recorded order.

    The order is the format's own: ``param_shapes`` is an ordered mapping
    written in the order the parameters were flattened, so walking it and
    taking ``numel`` at a time is not a convention this reader invents, it is
    the one the writer used.
    """
    wanted = sum(_numel(shape) for shape in shapes.values())
    if flat.numel() < wanted:
        raise ZeroUnsupported(
            "parameter group %d needs %d elements and the shards hold %d "
            "between them: a shard is missing or truncated"
            % (group, wanted, flat.numel())
        )

    slack = flat.numel() - wanted
    if paddings is not None and group < len(paddings) and paddings[group] != slack:
        # A note, not a refusal, and the demotion is deliberate.
        #
        # Refusing here was the first version, on the reasoning that a
        # disagreement means this reader has drifted from the checkpoint's own
        # bookkeeping. Then `group_paddings` was measured across stages 1 and
        # 2 at two world sizes and two shapes, and it was `[0]` every time —
        # including where the partitions are visibly uneven ([14260, 14258]
        # for 28518 parameters over two ranks). So the field's meaning is not
        # "how much slack there is at the end", it is something this reader
        # does not understand, and every case that exercised it was one where
        # both numbers happened to be zero.
        #
        # Gating on a field never observed to be non-zero would refuse a
        # checkpoint that reconstructs correctly — and the reconstruction is
        # proven correct by `zero_to_fp32` for every case that could be
        # produced. The direction that is actually dangerous, too few elements
        # rather than too many, is refused above.
        notes.append(
            "parameter group %d has %d element(s) left over and the checkpoint "
            "records %d of padding; the reconstruction takes what the shapes "
            "ask for and ignores both" % (group, slack, paddings[group])
        )

    out: Dict[str, Any] = {}
    at = 0
    for name, shape in shapes.items():
        size = _numel(shape)
        out[name] = flat[at : at + size].view(*shape) if shape else flat[at]
        at += size
    return out


def _cut_per_parameter(parts, shapes, group: int, torch) -> Dict[str, Any]:
    """Stage 3: gather each parameter across the ranks, one parameter at a time.

    Every rank holds ``ceil(numel / ranks)`` elements of each parameter, laid
    out in the recorded order, so the fragments of one tensor sit at the same
    offset in every rank's buffer. Walking the offset forward by that padded
    stride and concatenating across ranks rebuilds the parameter; trimming to
    ``numel`` drops the padding the last rank carried.

    The stride is the *padded* one, not ``numel``: a parameter whose element
    count does not divide by the rank count still occupies a whole slot in
    every rank's buffer, and advancing by the unpadded count would slide every
    subsequent parameter backwards by a little more each time. That is the
    failure mode that produced the right shapes and the wrong values.
    """
    ranks = len(parts)
    out: Dict[str, Any] = {}
    at = 0
    for name, shape in shapes.items():
        size = _numel(shape)
        stride = -(-size // ranks)  # ceil, without importing math for one line
        pieces = []
        for part in parts:
            if at + stride > part.numel():
                raise ZeroUnsupported(
                    "parameter %r in group %d wants elements %d..%d of a shard "
                    "holding %d: the shards are not all of one checkpoint"
                    % (name, group, at, at + stride, part.numel())
                )
            pieces.append(part[at : at + stride])
        whole = torch.cat(pieces)[:size]
        out[name] = whole.view(*shape) if shape else whole[0]
        at += stride
    return out


def _numel(shape) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _optim_files(found) -> List[tuple]:
    """``(rank, filename)`` for every data-parallel shard, in rank order.

    Rank order is what the concatenation depends on, so it is established here
    from the names rather than inherited from whatever order the directory
    listing came back in. A filesystem is under no obligation to sort.
    """
    from ravex._foreign import _ZERO_OPTIM

    found_ranks = []
    for name in found.files:
        match = _ZERO_OPTIM.match(name)
        if match:
            found_ranks.append((int(match.group(1)), name))
    found_ranks.sort()

    ranks = [rank for rank, _ in found_ranks]
    if ranks and ranks != list(range(len(ranks))):
        raise ZeroUnsupported(
            "the shards present are for rank(s) %s, which is not a complete "
            "0..N-1: rebuilding from these would join the wrong pieces together"
            % ranks
        )
    return found_ranks


def _shapes_and_step(found, loader, os):
    """``param_shapes`` and the step, from whichever model file this stage wrote.

    Any one of them will do at stage 3 — every rank writes the same shapes,
    only the tensors differ, and at stage 3 the tensors are empty anyway.
    """
    from ravex._foreign import _ZERO_MODEL_PER_RANK, _ZERO_MODEL_SHARED

    for name in found.files:
        if not (_ZERO_MODEL_SHARED.match(name) or _ZERO_MODEL_PER_RANK.match(name)):
            continue
        blob = loader(os.path.join(found.root, name))
        shapes = blob.get("param_shapes")
        if shapes is None:
            continue
        step = blob.get("global_steps")
        return [dict(group) for group in shapes], (
            int(step) if step is not None else None
        )
    return None, None


def _flat_partitions(osd, blob):
    """This rank's flat fp32 partition per parameter group, whichever stage wrote it.

    Stages 1 and 2 file it under ``single_partition_of_fp32_groups`` and stage
    3 under ``fp32_flat_groups``; both are a list with one entry per parameter
    group. Looked for by name rather than selected by the stage number, so a
    checkpoint whose recorded stage and actual layout disagree still reads
    correctly instead of reading the wrong key confidently.
    """
    for key in ("single_partition_of_fp32_groups", "fp32_flat_groups"):
        for holder in (osd, blob):
            value = holder.get(key) if hasattr(holder, "get") else None
            if isinstance(value, (list, tuple)) and value:
                return list(value)
    return None
