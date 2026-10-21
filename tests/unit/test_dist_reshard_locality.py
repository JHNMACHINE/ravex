"""Which of a reshard's bytes actually have to cross a network — GPU-96.

The planner in :mod:`ravex._dist.reshard` says how a new shard is stitched out of
old ones. These tests are about the fact it does not carry on its own: *where*
each old shard can be read from. Same discipline as the tests above them —
pure arithmetic, so exhaustive rather than illustrative — because the number
these functions produce is the one that decides whether a cross-machine
transport is worth writing at all, and a number that decides something should
not rest on the one topology a rented box happened to have.

The headline the suite exists to pin down is counter-intuitive, and getting it
backwards would mean sizing a transport for a case that mostly does not occur:
**a shrink whose machine boundaries line up moves nothing.** Traffic is what
misalignment costs, not what resharding costs.
"""

import math

import pytest

from ravex._dist.reshard import (
    contiguous_homes,
    crossing_bytes,
    crossing_pieces,
    crossing_rows,
    most_sources_held,
    plan_reshard,
)


# ── how ranks are dealt to machines ─────────────────────────────────────────


def test_contiguous_homes_deals_even_blocks():
    assert contiguous_homes(8, 2) == {r: r // 4 for r in range(8)}
    assert contiguous_homes(6, 3) == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}


def test_contiguous_homes_spreads_a_remainder_over_the_first_machines():
    # Five ranks over two machines is three then two, which is what torchrun
    # does and is the only shape where the boundary can land off-centre.
    assert contiguous_homes(5, 2) == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}


def test_contiguous_homes_refuses_a_machine_with_no_rank():
    with pytest.raises(ValueError, match="cannot be dealt"):
        contiguous_homes(2, 4)
    with pytest.raises(ValueError, match="at least one machine"):
        contiguous_homes(4, 0)


# ── the headline: an aligned reshard moves nothing ──────────────────────────


@pytest.mark.parametrize("machines", [2, 3, 4])
@pytest.mark.parametrize("old_world", [2, 4, 6, 8, 12])
@pytest.mark.parametrize("new_world", [2, 4, 6, 8, 12])
def test_nothing_crosses_when_the_machine_boundaries_line_up(
    machines, old_world, new_world
):
    """Both topologies dealt contiguously and divisible by the machine count.

    Every machine's old stores cover exactly the interval its new ranks want,
    so the answer is an empty dict — not a small number, an empty one. This is
    the case a transport would never be invoked for, and it is a large share
    of the real ones: the same boxes, the same number of them, a different
    number of GPUs in use on each.
    """
    if old_world % machines or new_world % machines:
        pytest.skip("not an aligned split")

    extent = old_world * new_world
    plan = plan_reshard([new_world] * old_world, [old_world] * new_world)

    crossing = crossing_rows(
        plan,
        contiguous_homes(old_world, machines),
        contiguous_homes(new_world, machines),
    )
    assert crossing == {}, (
        "%d -> %d over %d machines should move nothing and moved %s of %d rows"
        % (old_world, new_world, machines, crossing, extent)
    )


def test_a_misaligned_boundary_is_exactly_what_crosses():
    """Two old shards, three new ones, over two machines.

    New rank 1 straddles the old machine boundary: it wants rows [2, 4) and
    machine 0 only holds [0, 3). The single row [3, 4) is the whole cost, and
    it moves from machine 1 to machine 0 — worked out by hand rather than by
    re-running the implementation, so the test can disagree with the code.
    """
    plan = plan_reshard([3, 3], [2, 2, 2])
    crossing = crossing_rows(plan, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 1})
    assert crossing == {(1, 0): 1}


def test_direction_is_kept_apart():
    """A link is a pair, not a set: who pulls and who pushes are different facts."""
    # Four old shards over two machines, three new ones dealt 2/1. New rank 1
    # reaches across from machine 0 into machine 1's half; new rank 2 reaches
    # back the other way.
    plan = plan_reshard([3, 3, 3, 3], [4, 4, 4])
    crossing = crossing_rows(plan, contiguous_homes(4, 2), {0: 0, 1: 0, 2: 1})
    assert set(crossing) == {(1, 0)}
    # New rank 1 holds [4, 8); machine 0's stores cover [0, 6); so [6, 8).
    assert crossing == {(1, 0): 2}


# ── the case the transport is actually for ──────────────────────────────────


def test_a_dead_machine_costs_only_what_its_neighbour_cannot_serve_locally():
    """Six ranks over three machines, machine 1 gone, its shards on machine 2.

    ``old_home`` says where a *readable copy* is, so a promoted replica is
    fed in as an ordinary home and needs no second code path. Machine 1's
    ranks 2 and 3 are read from machine 2, and the surviving pair of machines
    resume as four ranks.

    Two of twelve rows cross. That is the number worth staring at before
    writing a transport: losing a third of the cluster does not mean moving a
    third of the checkpoint.
    """
    plan = plan_reshard([2] * 6, [3] * 4)
    old_home = {0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 2}
    new_home = {0: 0, 1: 0, 2: 2, 3: 2}

    assert crossing_rows(plan, old_home, new_home) == {(2, 0): 2}


def test_a_replica_in_the_right_place_costs_nothing():
    """The case ``open_rank_store`` already covers, stated as arithmetic.

    Machine 1 is gone and machine 0 holds complete copies of everything it
    wrote. Every home is machine 0, every new rank runs there, and no byte
    moves — which is why the local fallback was worth having before any
    transport existed.
    """
    plan = plan_reshard([3] * 4, [6] * 2)
    everything_here = {q: 0 for q in range(4)}
    assert crossing_rows(plan, everything_here, {0: 0, 1: 0}) == {}


# ── the partition, and what the transport is handed ─────────────────────────


@pytest.mark.parametrize("old_world", range(1, 9))
@pytest.mark.parametrize("new_world", range(1, 9))
def test_crossing_and_local_partition_the_plan_exactly(old_world, new_world):
    """Every piece is either fetched or read locally — never both, never neither.

    A piece counted twice is a byte moved for nothing; a piece counted in
    neither is a band of uninitialised rows, which is the failure the whole
    module is written against. The partition is what rules both out.
    """
    extent = old_world * new_world
    plan = plan_reshard([new_world] * old_world, [old_world] * new_world)
    old_home = contiguous_homes(old_world, min(2, old_world))
    new_home = contiguous_homes(new_world, min(2, new_world))

    crossing = crossing_pieces(plan, old_home, new_home)
    assert len(crossing) == len(plan)

    for r, (all_pieces, remote) in enumerate(zip(plan, crossing)):
        remote_set = [p for p in all_pieces if old_home[p[0]] != new_home[r]]
        local_set = [p for p in all_pieces if old_home[p[0]] == new_home[r]]
        assert remote == remote_set
        assert len(remote) + len(local_set) == len(all_pieces)

    moved = sum(rows for rows in crossing_rows(plan, old_home, new_home).values())
    assert 0 <= moved <= extent


def test_an_unknown_home_raises_rather_than_guessing():
    """Neither default is safe, so there is no default.

    Assuming a missing store is remote invents traffic; assuming it is local
    invents a store. The second is the dangerous one — it is how a hole gets
    into a tensor — so both are refused by name.
    """
    plan = plan_reshard([2, 2], [4])
    with pytest.raises(KeyError, match="old rank 1"):
        crossing_pieces(plan, {0: 0}, {0: 0})
    with pytest.raises(KeyError, match="new rank 0"):
        crossing_pieces(plan, {0: 0, 1: 0}, {})


def test_machine_identities_are_opaque():
    """Hostnames, not indices. The homes come from ``.ravex-owner`` records."""
    plan = plan_reshard([3, 3], [2, 2, 2])
    crossing = crossing_rows(
        plan,
        {0: "d01a456116cf", 1: "c93f520e40b5"},
        {0: "d01a456116cf", 1: "d01a456116cf", 2: "c93f520e40b5"},
    )
    assert crossing == {("c93f520e40b5", "d01a456116cf"): 1}


# ── bytes, which is what a rate is divided into ─────────────────────────────


def test_crossing_bytes_uses_each_tensor_own_row_width():
    """Two tensors, two widths, one link.

    The pairing is per tensor on purpose: an embedding's row and a bias's row
    are not the same number of bytes, and an average over the model would be a
    figure that looks measured and is not.
    """
    plan = plan_reshard([3, 3], [2, 2, 2])  # one row crosses, machine 1 -> 0
    old_home, new_home = {0: 0, 1: 1}, {0: 0, 1: 0, 2: 1}

    wide = 4096 * 2  # bf16, hidden 4096
    narrow = 2  # bf16 scalar per row

    assert crossing_bytes([(plan, wide)], old_home, new_home) == {(1, 0): wide}
    assert crossing_bytes(
        [(plan, wide), (plan, narrow)], old_home, new_home
    ) == {(1, 0): wide + narrow}


def test_crossing_bytes_is_empty_when_nothing_crosses():
    plan = plan_reshard([4] * 4, [8] * 2)
    assert crossing_bytes([(plan, 8192)], contiguous_homes(4, 2), contiguous_homes(2, 2)) == {}


# ── the memory ceiling, which is a count ────────────────────────────────────


@pytest.mark.parametrize("old_world", range(1, 17))
@pytest.mark.parametrize("new_world", range(1, 17))
def test_a_rank_never_holds_more_than_the_documented_bound(old_world, new_world):
    """``ceil(N / M) + 1`` old shards, which is what keeps the global tensor unmaterialised.

    Exhaustive because the bound is the reason the memory story holds, and the
    only pairs where it is obviously safe are the ones a hand-picked test would
    have chosen.
    """
    plan = plan_reshard([new_world] * old_world, [old_world] * new_world)
    bound = math.ceil(old_world / new_world) + 1
    assert most_sources_held(plan) <= bound


def test_most_sources_held_of_nothing_is_nothing():
    assert most_sources_held([]) == 0
