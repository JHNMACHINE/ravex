"""GPU-94 step 1: can a rank that never existed join a live gloo group, - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_elastic_rendezvous.py.
"""


from test_dist_elastic_rendezvous import _run_three_rank_rejoin  # noqa: E402


class TestARankThatNeverExistedCanJoin:
    """The core question step 1 was meant to answer, isolated from FSDP2, from
    checkpoints, from ravex entirely: can torch's public distributed API even
    do the rendezvous shape a live, no-restart topology change needs.
    """

    def test_generation_prefixed_store_lets_a_new_rank_join_cleanly(self):
        messages, killed = _run_three_rank_rejoin()

        assert killed == [], (
            "a process had to be killed rather than exiting on its own - "
            "that is a hang, not a clean failure, and it is exactly the "
            "failure mode the naive version in the module docstring produces"
        )
        assert messages[0] == [("phase1", "ok"), ("phase2", 3.0)]
        assert messages[1] == [("phase1", "ok"), ("phase2", 3.0)]
        # Rank 2 was never part of phase 1 - it has no phase1 message at all,
        # and its only result is the phase-2 collective it joined fresh.
        assert messages[2] == [("phase2", 3.0)]
