"""Types for the compiled core.

The package ships ``py.typed``, so without this file every symbol
``ravex._dist.reshard`` re-exports reads as unknown — and the first thing that
shows up is not a squiggle on the import, it is the *callers* losing the types
of what they were handed.

Kept by hand and checked by ``tests/test_core_surface.py``, which asserts that
what the extension exports and what is declared here are the same set of names.
A stub that has drifted is worse than none: it type-checks code against a
function that is no longer there.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__version__: str

Placement = Dict[str, Any]
Piece = Tuple[int, Tuple[int, int], Tuple[int, int]]
Homes = Dict[int, Any]

class ReshardUnsupported(Exception): ...

def encode_placement(placement: Any) -> Placement: ...
def decode_placements(saved: Any) -> List[Placement]: ...
def shard_dim(
    placements: Sequence[Placement], where: str = ...
) -> Optional[int]: ...
def offsets_from_lengths(lengths: Sequence[int]) -> List[int]: ...
def plan_reshard(
    old_lengths: Sequence[int], new_lengths: Sequence[int]
) -> List[List[Piece]]: ...
def sources_needed(plan: Iterable[Sequence[Piece]]) -> Set[int]: ...
def check_covered(pieces: Sequence[Piece], length: int, where: str) -> None: ...
def crossing_pieces(
    plan: Sequence[Sequence[Piece]], old_home: Homes, new_home: Homes
) -> List[List[Piece]]: ...
def crossing_rows(
    plan: Sequence[Sequence[Piece]], old_home: Homes, new_home: Homes
) -> Dict[Tuple[Any, Any], int]: ...
def crossing_bytes(
    tensors: Iterable[Tuple[Sequence[Sequence[Piece]], int]],
    old_home: Homes,
    new_home: Homes,
) -> Dict[Tuple[Any, Any], int]: ...
def contiguous_homes(world: int, machines: int) -> Dict[int, int]: ...
def most_sources_held(plan: Sequence[Sequence[Piece]]) -> int: ...
