"""Recognising a checkpoint that something else wrote — GPU-90.

Ravex reads its own stores through :mod:`ravex._backends`, which knows where
everything is because it put it there. This module is for the other case: a
directory handed over by a team, or left behind by a run in a different
framework, where the first question is not *what step is this* but *what is
this*.

**Layout first, fields second, and in that order for a reason.** Every format
here records what it is somewhere inside itself — DeepSpeed writes
``zero_stage`` into every optimizer file, torch's distributed checkpoint writes
a ``.metadata`` — but reading a field means choosing a file to open, and
choosing a file is the question. So the shape of the directory decides which
format this is, and the recorded fields are then read to *confirm* it and to
fill in what the layout cannot say. When the two disagree the recorded value
wins and the disagreement is reported: the layout is an inference, the field is
a statement.

Nothing here imports torch or opens a tensor. Identifying a checkpoint should
cost a directory listing, because it happens before anyone has decided to load
one — and because a function that only lists names can be tested against a tree
of empty files, which is how the awkward cases get covered.

**On Megatron.** GPU-90 was written when Megatron had a manifest of its own.
It no longer does: ``megatron.core.dist_checkpointing`` is built on
``torch.distributed.checkpoint`` and writes the ordinary ``.metadata`` /
``.distcp`` pair. So there is no Megatron branch below, and its absence is the
finding rather than an omission — a Megatron checkpoint identifies as ``dcp``,
which is also what a plain torch job writes, and one reader serves both.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


#: ``zero_pp_rank_<dp>_mp_rank_<mp>_optim_states.pt`` — one per data-parallel
#: rank. The ``mp_rank`` half is the tensor/pipeline coordinate and is ``00``
#: for a job that is only data-parallel.
_ZERO_OPTIM = re.compile(
    r"^zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states\.pt$"
)
#: ``mp_rank_<mp>_model_states.pt`` at stages 1 and 2, where the parameters are
#: replicated and one copy is written; ``zero_pp_rank_<dp>_mp_rank_<mp>_model_
#: states.pt`` at stage 3, where they are partitioned like everything else.
_ZERO_MODEL_SHARED = re.compile(r"^mp_rank_(\d+)_model_states\.pt$")
_ZERO_MODEL_PER_RANK = re.compile(
    r"^zero_pp_rank_(\d+)_mp_rank_(\d+)_model_states\.pt$"
)


@dataclass
class Foreign:
    """What a directory turned out to be.

    ``format`` is the only field always meaningful. The rest is what that
    particular format makes knowable from its layout alone, and is left unset
    rather than guessed at — an ``Optional`` here means "the directory does not
    say", which a caller can act on, while a plausible default would be
    indistinguishable from a fact.
    """

    #: ``"deepspeed"``, ``"dcp"``, ``"ravex"`` or ``"unknown"``.
    format: str
    #: Where the state actually is. For DeepSpeed this is the tag directory
    #: (``latest`` names it), not the root the caller passed.
    root: str
    #: Data-parallel ranks that wrote it, when the layout says.
    world_size: Optional[int] = None
    #: ZeRO stage, inferred from the layout here and confirmed from the files
    #: by :func:`confirm_stage`.
    stage: Optional[int] = None
    #: Files that make it up, relative to :attr:`root`, sorted.
    files: List[str] = field(default_factory=list)
    #: Anything worth saying out loud about what was found.
    notes: List[str] = field(default_factory=list)

    @property
    def understood(self) -> bool:
        return self.format != "unknown"


def identify(path: str) -> Foreign:
    """What kind of checkpoint is in ``path``. A directory listing, nothing more.

    The order the branches are tried in is not arbitrary. Ravex's own layout is
    checked first because it is the one this process is most likely to be
    pointed at by accident, and because reporting a Ravex store as "unknown
    foreign format" would send a caller down a conversion path for a
    checkpoint it can already read. DeepSpeed comes before the distributed
    checkpoint because a DeepSpeed tag directory can contain files that a
    loose ``.distcp`` test would match, while the reverse is not true.
    """
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        return Foreign("unknown", path, notes=["cannot be listed: %s" % exc])

    names = set(entries)

    if "manifest.json" in names and os.path.isdir(os.path.join(path, "snapshots")):
        return Foreign("ravex", path, notes=["a Ravex store, readable as it is"])
    if any(name.startswith("rank_") for name in names):
        ranks = [n for n in entries if re.match(r"^rank_\d+$", n)]
        return Foreign(
            "ravex",
            path,
            world_size=len(ranks) or None,
            files=ranks,
            notes=["a Ravex per-rank store, readable as it is"],
        )

    zero = _identify_deepspeed(path, entries)
    if zero is not None:
        return zero

    if ".metadata" in names:
        shards = [n for n in entries if n.endswith(".distcp")]
        return Foreign(
            "dcp",
            path,
            world_size=len(shards) or None,
            files=sorted(shards),
            notes=[
                "torch distributed checkpoint - also what Megatron-core writes, "
                "so this is not by itself evidence of which framework produced it"
            ],
        )

    return Foreign("unknown", path, files=entries)


def _identify_deepspeed(path: str, entries: List[str]) -> Optional[Foreign]:
    """A DeepSpeed ZeRO checkpoint, at its root or at one of its tags.

    Both are accepted because both are handed over in practice: the root is
    what ``save_checkpoint`` was given and holds ``latest``, while the tag is
    what someone copies out of it when they want one step. Following ``latest``
    rather than picking the newest directory keeps this from disagreeing with
    DeepSpeed about which step is current, which is a disagreement that would
    only show up as the wrong weights.
    """
    if "latest" in entries and not _zero_files(entries):
        tag = _read_latest(path)
        if tag is None:
            return Foreign(
                "unknown",
                path,
                notes=["a `latest` file that could not be read"],
            )
        inner = os.path.join(path, tag)
        try:
            tag_entries = sorted(os.listdir(inner))
        except OSError:
            return Foreign(
                "unknown",
                path,
                notes=["`latest` names %r, which is not a directory here" % tag],
            )
        found = _describe_zero(inner, tag_entries)
        if found is not None:
            found.notes.insert(0, "tag %r, named by `latest`" % tag)
        return found

    return _describe_zero(path, entries)


def _describe_zero(root: str, entries: List[str]) -> Optional[Foreign]:
    optim = {}
    model_shared = []
    model_per_rank = {}
    for name in entries:
        match = _ZERO_OPTIM.match(name)
        if match:
            optim[int(match.group(1))] = name
            continue
        if _ZERO_MODEL_SHARED.match(name):
            model_shared.append(name)
            continue
        match = _ZERO_MODEL_PER_RANK.match(name)
        if match:
            model_per_rank[int(match.group(1))] = name

    if not optim:
        return None

    # Stage from the layout: at stages 1 and 2 the parameters are replicated
    # and one model file is written for everyone; at stage 3 they are
    # partitioned and there is one per rank. That separates 3 from {1, 2} and
    # cannot separate 1 from 2 — the two differ in what is partitioned at run
    # time (gradients), not in what reaches disk. `confirm_stage` reads the
    # number DeepSpeed wrote down.
    stage = 3 if model_per_rank else None
    notes = []
    if model_per_rank and model_shared:
        notes.append(
            "both a shared and a per-rank model file are present, which no "
            "single stage writes - treating it as stage 3 and confirming from "
            "the files"
        )
    if not model_per_rank and not model_shared:
        notes.append("no model state file: this checkpoint holds optimizer state only")

    files = sorted(set(list(optim.values()) + model_shared + list(model_per_rank.values())))
    ranks = sorted(optim)
    if ranks != list(range(len(ranks))):
        notes.append(
            "the data-parallel ranks present are %s, which is not a complete "
            "0..N-1 - some shard is missing" % ranks
        )
    return Foreign(
        "deepspeed",
        root,
        world_size=len(optim),
        stage=stage,
        files=files,
        notes=notes,
    )


def _zero_files(entries: List[str]) -> bool:
    return any(_ZERO_OPTIM.match(name) for name in entries)


def _read_latest(path: str) -> Optional[str]:
    try:
        with open(os.path.join(path, "latest"), encoding="utf-8") as handle:
            tag = handle.read().strip()
    except OSError:
        return None
    # A tag is a directory name and is joined onto a path, so it does not get
    # to contain a separator or climb out of the checkpoint. The file is
    # ordinarily written by DeepSpeed, but "ordinarily" is not a property of a
    # directory someone handed over.
    if not tag or os.path.isabs(tag) or os.sep in tag or "/" in tag or tag in (".", ".."):
        return None
    return tag


def confirm_stage(found: Foreign, loader) -> Foreign:
    """Read the stage DeepSpeed recorded, and say so if the layout disagreed.

    ``loader`` takes a path and returns the unpickled object, so this module
    stays free of torch and the caller decides how a ``.pt`` gets opened —
    which also lets the tests exercise the disagreement without producing a
    real checkpoint that has one.

    The recorded value wins. It is a statement; the layout was an inference,
    and the whole point of preferring measurements to derivations is that the
    thing which wrote the file knew more than the thing reading it.
    """
    if found.format != "deepspeed" or not found.files:
        return found

    optim = [name for name in found.files if _ZERO_OPTIM.match(name)]
    if not optim:
        return found

    try:
        blob = loader(os.path.join(found.root, optim[0]))
        recorded = int(blob["optimizer_state_dict"]["zero_stage"])
    except Exception as exc:
        found.notes.append(
            "the stage could not be read from %s (%s); going with the layout"
            % (optim[0], exc)
        )
        return found

    if found.stage is not None and found.stage != recorded:
        found.notes.append(
            "the layout looks like stage %d and the checkpoint says stage %d; "
            "going with what it says" % (found.stage, recorded)
        )
    found.stage = recorded
    return found


def summary(found: Foreign) -> str:
    """One line, for a log that has to say what it is about to refuse or read."""
    parts = [found.format]
    if found.stage is not None:
        parts.append("ZeRO stage %d" % found.stage)
    if found.world_size is not None:
        parts.append("%d rank(s)" % found.world_size)
    line = ", ".join(parts)
    return line if not found.notes else "%s (%s)" % (line, "; ".join(found.notes))
