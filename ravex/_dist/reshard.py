"""Resuming a per-rank checkpoint onto a different number of ranks.

The arithmetic that used to be written out here now lives in ``src/reshard.rs``
and reaches Python through ``ravex._core``. This module is the name it is
imported by — unchanged, because a stored checkpoint and a resume path that has
worked for months are both worse off for a rename that buys nothing.

**Why it moved.** Everything this module does is integers: offsets, intervals,
row counts. No torch, no process group, no I/O — which is exactly the shape of
work that is cheap in Rust and, more to the point, the shape that can be moved
without dragging the interpreter across the boundary on every call. The parts
of Ravex that live on introspecting live Python objects — ``_patches``,
``_registry``, ``_runtime`` — stay where they are for the opposite reason.

**What did not change.** Every name below means what it meant, takes what it
took and raises what it raised, down to the wording of the messages. That was
the condition of the port rather than a courtesy: ``tests/test_dist_reshard.py``
and ``tests/test_dist_reshard_locality.py`` were written against the Python
implementation and were not touched for it, so they are the oracle the Rust one
is held to. A port whose tests had to be edited to pass would have proved
nothing.

The two ideas the implementation is built on are documented where it is, in the
module docstring of ``src/reshard.rs``: offsets are measured rather than
derived, and a hole is worse than a failure.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Imported eagerly, and this is the one place in the package where that is not a
# startup-cost mistake. `ravex/__init__.py` is run in every interpreter on the
# machine once the autoloader is installed, so nothing it touches may be
# expensive — but `_dist` is reached only from the resume path, and by then a
# checkpoint is already being opened.
from ravex._core import (
    ReshardUnsupported,
    check_covered,
    contiguous_homes,
    crossing_bytes,
    crossing_pieces,
    crossing_rows,
    decode_placements,
    encode_placement,
    most_sources_held,
    offsets_from_lengths,
    plan_reshard,
    shard_dim,
    sources_needed,
)

#: What a placement dict looks like once decoded: ``kind`` is one of ``shard``,
#: ``replicate``, ``partial`` or ``unknown``, and ``dim`` is present exactly
#: when ``kind`` is ``shard``.
Placement = Dict[str, Any]

#: One piece of a new shard: which old rank holds it, the half-open interval to
#: take out of that rank's shard, and where it lands in the new one. The
#: destination is carried rather than implied so the caller can assert on it
#: instead of trusting the order it iterates in.
Piece = Tuple[int, Tuple[int, int], Tuple[int, int]]

#: Which machine holds a readable copy of each old rank's store, and which
#: machine each new rank runs on. The identities are opaque and only ever
#: compared for equality — a hostname, an owner record, a node rank. Nothing
#: assumes they are integers or that they are ordered.
Homes = Dict[int, Any]

__all__ = [
    "Homes",
    "Piece",
    "Placement",
    "ReshardUnsupported",
    "check_covered",
    "contiguous_homes",
    "crossing_bytes",
    "crossing_pieces",
    "crossing_rows",
    "decode_placements",
    "encode_placement",
    "most_sources_held",
    "offsets_from_lengths",
    "plan_reshard",
    "shard_dim",
    "sources_needed",
]
