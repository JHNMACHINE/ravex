"""Resuming a per-rank checkpoint onto a different number of ranks.

Per-rank checkpointing writes each rank's own slice of every tensor and
nothing else, which is what keeps a checkpoint off any single rank's memory
budget. The price, until now, was that the slices only line up with the
topology that wrote them: resume eight shards onto four ranks and the first
tensor raises *saved shard is (384, 4096), this rank holds (512, 4096)*.

This module is the arithmetic that makes them line up. It is deliberately
**pure**: no torch, no process group, no I/O. Everything here is integers and
plain dicts, so the part with no excuse for being under-tested can be tested
exhaustively over every ``N -> M`` pair in a range rather than on the one pair
a GPU box happens to have.

Two ideas carry the whole file.

**Offsets are measured, not derived.** The obvious approach is to reproduce
torch's chunking rule and work out where each shard begins. Don't: the rule
has an uneven-tail case that :func:`ravex._distributed._rebuild_dtensor`
already carries a comment about, and a second implementation of it would drift
from the first in silence. Both sides can be *observed* instead — the old
lengths are the shapes of the tensors that were saved, the new lengths are the
shapes the live model is holding — and the plan is then a matter of adding up
what torch already decided.

**A hole is worse than a failure.** If some old shard cannot be read, the
tempting move is to carry on with the ones that can. That produces a tensor
with a band of uninitialised rows: every individual shard valid, the whole
thing wrong, and nothing downstream able to notice. It is the same failure
GPU-59 and GPU-79 were about, and :func:`plan_reshard` raises rather than
reintroduce it through this door.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


#: What a placement dict looks like once decoded: ``kind`` is one of
#: ``shard``, ``replicate``, ``partial`` or ``unknown``, and ``dim`` is present
#: exactly when ``kind`` is ``shard``.
Placement = Dict[str, Any]

# How torch writes a placement, for checkpoints from before placements were
# stored as data. Two forms each, and *both* are needed: what the old
# `_encode_shards` wrote was `str(placement)`, which on torch 2.12 is the short
# form — `S(0)`, `R`, `P(sum)` — while the long form is what `repr` gives and
# what anyone reading this file would expect to have been written. Accepting
# only the one that looks canonical would have read every existing per-rank
# checkpoint as "unknown placement" and refused to reshard it.
_LEGACY_SHARD = re.compile(r"^\s*(?:_?Shard\s*\(\s*(?:dim\s*=\s*)?|S\s*\(\s*)(\d+)\s*\)\s*$")
_LEGACY_REPLICATE = re.compile(r"^\s*(?:_?Replicate\s*\(\s*\)|R)\s*$")
_LEGACY_PARTIAL = re.compile(
    r"^\s*(?:_?Partial|P)\s*\(\s*(?:reduce_op\s*=\s*)?['\"]?(\w*)"
)


def encode_placement(placement: Any) -> Placement:
    """One DTensor placement as plain data.

    What used to be written here was ``str(placement)`` — ``"Shard(dim=0)"`` —
    which reads well in a dump and badly everywhere else. The only way back
    from it is a parser, and a parser for torch's ``repr`` is a dependency on a
    string nobody promised to keep stable. Resharding has to *ask* which
    dimension a tensor was split along, so from now on the answer is written
    down as an answer.

    Duck-typed on purpose: this takes a torch object but imports no torch, so
    the encoding lives beside the planner that consumes it rather than inside
    the module that happens to hold a DTensor.

    ``Partial`` is named rather than folded in with ``Replicate``. A partial
    value is a term waiting to be summed, not a copy of the whole — treating
    one as the other would produce a tensor that is quietly a fraction of what
    it should be, which is exactly the class of error this file exists to
    avoid.
    """
    dim = getattr(placement, "dim", None)
    if dim is not None:
        return {"kind": "shard", "dim": int(dim)}

    if _asks_yes(placement, "is_replicate"):
        return {"kind": "replicate"}

    if _asks_yes(placement, "is_partial"):
        op = getattr(placement, "reduce_op", None)
        return {"kind": "partial", "op": str(op) if op is not None else "sum"}

    # Neither a shard nor anything this torch will admit to. Carrying the repr
    # keeps the checkpoint self-describing: a reshard refuses it by name
    # instead of refusing it as "something".
    return {"kind": "unknown", "repr": str(placement)}


def _asks_yes(placement: Any, question: str) -> bool:
    """``placement.is_replicate()`` and friends, tolerating their absence."""
    method = getattr(placement, question, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:  # pragma: no cover - a placement type with a hostile API
        return False


def decode_placements(saved: Any) -> List[Placement]:
    """The placement list of a saved shard, whichever way it was written.

    Two shapes reach this. Checkpoints written from here on hold the dicts
    :func:`encode_placement` produces. Checkpoints written before that hold
    torch's ``repr`` strings, and they are still resumable — a stored
    checkpoint is not a thing you get to reformat after the fact, and the
    strings are parseable well enough to recover the one field that matters.

    Anything unrecognisable comes back as ``unknown`` rather than raising. The
    caller that cares — :func:`shard_dim` — refuses on it with a message that
    can name the tensor; raising here would only be able to name the string.
    """
    if not isinstance(saved, (list, tuple)):
        return []

    decoded: List[Placement] = []
    for entry in saved:
        if isinstance(entry, dict):
            kind = entry.get("kind")
            if kind == "shard":
                try:
                    decoded.append({"kind": "shard", "dim": int(entry["dim"])})
                    continue
                except (KeyError, TypeError, ValueError):
                    decoded.append({"kind": "unknown", "repr": repr(entry)})
                    continue
            if kind in ("replicate", "partial", "unknown"):
                decoded.append(dict(entry))
                continue
            decoded.append({"kind": "unknown", "repr": repr(entry)})
            continue

        decoded.append(_decode_legacy_placement(str(entry)))
    return decoded


def _decode_legacy_placement(text: str) -> Placement:
    """Parse one ``repr`` from a checkpoint written before this was data."""
    match = _LEGACY_SHARD.match(text)
    if match is not None:
        return {"kind": "shard", "dim": int(match.group(1))}
    if _LEGACY_REPLICATE.match(text) is not None:
        return {"kind": "replicate"}
    match = _LEGACY_PARTIAL.match(text)
    if match is not None:
        return {"kind": "partial", "op": match.group(1) or "sum"}
    return {"kind": "unknown", "repr": text}


def shard_dim(placements: Sequence[Placement], where: str = "a tensor") -> Optional[int]:
    """The single dimension this tensor is split along, or None if it is not.

    ``None`` means every rank holds the same bytes — a ``Replicate`` on a 1-D
    mesh — and resharding one is copying rank 0's, which is why it is a normal
    answer rather than an error.

    Everything else in scope is a 1-D mesh with exactly one ``Shard``. A 2-D
    mesh — FSDP crossed with tensor parallel — makes the plan a cartesian
    problem rather than an interval one, and refusing it by name costs one
    check and saves a category of wrong answers. Widening later is additive:
    nothing written here has to change for a 2-D planner to be added beside it.
    """
    shards = [p for p in placements if p.get("kind") == "shard"]

    for placement in placements:
        if placement.get("kind") == "partial":
            raise ReshardUnsupported(
                "%s is stored as a Partial value, which is a term waiting to be "
                "summed rather than a piece of the tensor. Resharding one would "
                "have to know what reduction is pending; it does not." % where
            )
        if placement.get("kind") == "unknown":
            raise ReshardUnsupported(
                "%s carries a placement this version does not understand (%s)"
                % (where, placement.get("repr", "?"))
            )

    if not shards:
        return None
    if len(shards) > 1:
        raise ReshardUnsupported(
            "%s is sharded over %d mesh dimensions. Resharding covers a 1-D "
            "mesh — FSDP — where a shard is an interval; a 2-D mesh is a "
            "different problem and is not attempted." % (where, len(shards))
        )
    return int(shards[0]["dim"])


class ReshardUnsupported(Exception):
    """A checkpoint this planner will not attempt to reshape.

    Separate from ``ValueError`` so the resume path can tell "I refuse, and
    here is why" apart from "something went wrong". The first is a reason to
    fall back to starting from scratch with a message a user can act on; the
    second is a bug.
    """


def offsets_from_lengths(lengths: Sequence[int]) -> List[int]:
    """Running sum, with the leading zero: ``[3, 3, 2] -> [0, 3, 6, 8]``.

    Shard *q* covers global rows ``[offsets[q], offsets[q + 1])``. Keeping the
    fence-post form means both ends of every interval come out of the same list
    and there is no ``+ 1`` to get wrong at a call site.
    """
    running = [0]
    for length in lengths:
        running.append(running[-1] + int(length))
    return running


#: One piece of a new shard: which old rank holds it, the half-open interval
#: to take out of that rank's shard, and where it lands in the new one. The
#: destination is carried rather than implied so the caller can assert on it
#: instead of trusting the order it iterates in.
Piece = Tuple[int, Tuple[int, int], Tuple[int, int]]


def plan_reshard(
    old_lengths: Sequence[int], new_lengths: Sequence[int]
) -> List[List[Piece]]:
    """How to build each new shard out of the old ones.

    Returns one list of :data:`Piece` per new rank, in order, covering that
    rank's shard from its first row to its last with no gap and no overlap.

    Both arguments are *measurements*: ``old_lengths[q]`` is the extent of the
    shard rank *q* actually saved, ``new_lengths[r]`` the extent rank *r* is
    holding now. The two must sum to the same global extent — if they do not,
    the checkpoint is not of this tensor, and continuing would silently
    truncate or pad it.

    The general case is not a rearrangement of whole shards. Going from four
    ranks to three, no new shard equals any old one: each is stitched from two,
    and one old shard feeds two new ones. Boundaries coincide only on an exact
    halving or doubling — which is why an 8 -> 4 test passes while the general
    mechanism stays unwritten, and why the test for this is exhaustive.
    """
    old_total = sum(int(n) for n in old_lengths)
    new_total = sum(int(n) for n in new_lengths)
    if old_total != new_total:
        raise ValueError(
            "the saved shards add up to %d along the sharded dimension and the "
            "live ones to %d: this checkpoint is not of this tensor"
            % (old_total, new_total)
        )

    old_at = offsets_from_lengths(old_lengths)
    new_at = offsets_from_lengths(new_lengths)

    plan: List[List[Piece]] = []
    for r in range(len(new_lengths)):
        lo, hi = new_at[r], new_at[r + 1]
        pieces: List[Piece] = []
        for q in range(len(old_lengths)):
            a, b = old_at[q], old_at[q + 1]
            if b <= lo or a >= hi:
                continue
            start, stop = max(lo, a), min(hi, b)
            pieces.append((q, (start - a, stop - a), (start - lo, stop - lo)))
        plan.append(pieces)
    return plan


def sources_needed(plan: Iterable[Sequence[Piece]]) -> "set[int]":
    """Which old ranks a plan reads from at all.

    The reachability question is asked against this rather than against
    ``range(old_world)``: on a shrink each new rank needs at most
    ``ceil(N / M) + 1`` old shards, and demanding every store be readable when
    only some are wanted would refuse resumes that are perfectly possible.
    """
    return {piece[0] for pieces in plan for piece in pieces}


def check_covered(pieces: Sequence[Piece], length: int, where: str) -> None:
    """Assert that the pieces tile ``[0, length)`` exactly. Cheap, and load-bearing.

    :func:`plan_reshard` cannot produce a gap — it walks every old interval
    that overlaps — so this never fires on its output. It exists because the
    thing it is guarding against is unobservable downstream: a shard assembled
    with a hole in it is the right shape, the right dtype, and wrong, and the
    run would train on it for hours before anything looked odd.
    """
    covered = 0
    for _, _, (start, stop) in pieces:
        if start != covered:
            raise ValueError(
                "%s: the pieces of this shard do not join up — expected the next "
                "one to start at %d, it starts at %d" % (where, covered, start)
            )
        covered = stop
    if covered != length:
        raise ValueError(
            "%s: the pieces cover %d of %d rows" % (where, covered, length)
        )
