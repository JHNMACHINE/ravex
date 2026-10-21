"""Per-rank stores across machines that share only a network - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_multinode.py.
"""


import pytest

from test_dist_multinode import NEEDS_NUMPY, _gather_across_two_ranks, _join_gloo, _store_server  # noqa: E402


# ─── the old-torch from_local fallback ──────────────────────────────


class TestTheObjectGatherOnTwoRanks:
    """What one rank cannot ask.

    With ``world_size=1`` the replacement in `_all_gather_object` barely runs:
    `widest` is always this rank's own length, so the padding — the part most
    likely to be wrong — is never exercised, and there is no second rank to
    disagree with about the wire. These use two real processes and payloads of
    deliberately different sizes.

    Two processes on one box, not two machines: the question here is the shape
    of what goes on the wire, which loopback answers honestly. What loopback
    cannot answer — bandwidth, a peer that disappears — is what the rented kit
    in `integration/two-machines` is for.
    """

    # Sizes chosen to differ by a lot, so a rank that padded to the wrong
    # length produces garbage rather than something that happens to fit.
    PAYLOADS = [{"rank": 0, "blob": "a"}, {"rank": 1, "blob": "b" * 5000}]

    def test_both_ranks_without_numpy_agree(self):
        results = _gather_across_two_ranks([True, True], self.PAYLOADS)
        assert set(results) == {0, 1}, results
        for rank, (took_new, got) in results.items():
            assert took_new is True, f"rank {rank} did not take the new path"
            assert got == self.PAYLOADS, f"rank {rank} gathered {got!r}"

    @NEEDS_NUMPY
    def test_a_rank_with_numpy_and_a_rank_without_still_meet(self):
        """The claim `_all_gather_object` makes in its own docstring, and the
        one nothing else checks: the replacement posts the same collectives, in
        the same order and with the same shapes, as torch's own — so a mixed
        pair still agrees.

        It is not hypothetical. Two machines rented from one provider can come
        up from different images, and the unit job in CI installs no NumPy
        while every other job does.
        """
        results = _gather_across_two_ranks([False, True], self.PAYLOADS)
        assert set(results) == {0, 1}, results
        assert results[0][0] is False, "rank 0 was supposed to keep torch's path"
        assert results[1][0] is True, "rank 1 was supposed to take the new one"
        for rank, (_, got) in results.items():
            assert got == self.PAYLOADS, f"rank {rank} gathered {got!r}"


def _object_broadcast_worker(rank, world_size, port, no_numpy, payload, src, out):
    """One rank of a real gloo group, receiving `payload` from `src`."""
    import os

    if no_numpy:
        os.environ["RAVEX_ASSUME_NO_NUMPY"] = "1"
    else:
        os.environ.pop("RAVEX_ASSUME_NO_NUMPY", None)

    try:
        import torch.distributed as dist

        from ravex._dist import collectives as distributed

        distributed._torch_numpy = None
        took_the_new_path = not distributed._torch_can_reach_numpy()

        _join_gloo(rank, world_size, port)
        try:
            got = distributed._broadcast_object(
                dist, payload if rank == src else None, src
            )
        finally:
            dist.destroy_process_group()
        out.put((rank, took_the_new_path, got))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _broadcast_across_two_ranks(no_numpy_by_rank, payload, src=0):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    server = _store_server()
    port = server.port
    procs = [
        ctx.Process(
            target=_object_broadcast_worker,
            args=(rank, 2, port, no_numpy_by_rank[rank], payload, src, out),
        )
        for rank in (0, 1)
    ]
    for proc in procs:
        proc.start()
    results = {}
    try:
        for _ in procs:
            rank, took_new, got = out.get(timeout=120)
            if took_new is None:
                pytest.fail("rank %d failed: %s" % (rank, got))
            results[rank] = (took_new, got)
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - a hung collective
                proc.terminate()
                proc.join(timeout=10)
    return results


class TestTheObjectBroadcastOnTwoRanks:
    """The other object collective, and the one that had been left behind.

    `_all_gather_object` replaced torch's gather because it decodes with
    `tensor.numpy().tobytes()`; `elastic.share_captured` went on calling
    `dist.broadcast_object_list`, which decodes exactly the same way. The CI
    job that installs no NumPy found it on 2026-09-13, at the worst possible
    line: the broadcast that hands every rank the captured state after an
    elastic regroup. The wire work was done and then it raised, so the run had
    already paid for the transfer and got a crash instead of a model.
    """

    # Deliberately not a small scalar: the payload a regroup broadcasts is a
    # state dict, and a length handled wrongly shows up as garbage rather than
    # as something that happens to fit.
    PAYLOAD = ({"layer.weight": [0.5] * 2000, "layer.bias": [0.25]}, {"step": 41})

    def test_both_ranks_without_numpy_agree(self):
        results = _broadcast_across_two_ranks([True, True], self.PAYLOAD)
        assert set(results) == {0, 1}, results
        for rank, (took_new, got) in results.items():
            assert took_new is True, "rank %d did not take the new path" % rank
            assert got == self.PAYLOAD, "rank %d received %r" % (rank, got)

    @NEEDS_NUMPY
    def test_a_rank_with_numpy_and_a_rank_without_still_meet(self):
        """The same claim `_all_gather_object` makes, on this primitive.

        NumPy is a property of the process, not of the job, so the receiving
        rank can be on one path while the source is on the other. They agree
        only if the replacement posts the same two broadcasts, in the same
        order and with the same dtypes, as torch's own — a `long` holding the
        length and then the payload as `uint8`.

        The source here is the rank *with* NumPy, which is the interesting
        direction: it encodes through torch and the other decodes through the
        replacement.
        """
        results = _broadcast_across_two_ranks([False, True], self.PAYLOAD, src=0)
        assert results[0][0] is False, "rank 0 was supposed to keep torch's path"
        assert results[1][0] is True, "rank 1 was supposed to take the new one"
        for rank, (_took, got) in results.items():
            assert got == self.PAYLOAD, "rank %d received %r" % (rank, got)

    @NEEDS_NUMPY
    def test_and_the_other_direction_too(self):
        """Source without NumPy, receiver with. Encoding and decoding are
        different halves of the compatibility claim and only this arm exercises
        the replacement's encoder against torch's decoder."""
        results = _broadcast_across_two_ranks([True, False], self.PAYLOAD, src=0)
        assert results[0][0] is True
        assert results[1][0] is False
        for rank, (_took, got) in results.items():
            assert got == self.PAYLOAD, "rank %d received %r" % (rank, got)


def _drain_split_worker(rank, world_size, port, sharded, out):
    """One rank deciding whether to split `drain`, the way the runtime does."""
    try:
        import torch.distributed as dist

        from ravex._dist.collectives import all_ranks_agree, barrier, get_world_size

        _join_gloo(rank, world_size, port)
        try:
            # The runtime's own guard, not a copy of it: `sharded and
            # get_world_size() > 1`, unconditional since GPU-98 — no
            # environment variable left to read asymmetrically.
            split = sharded and get_world_size() > 1
            if split:
                barrier()
            # A second collective every rank posts. If the ranks had disagreed
            # about the barrier above, this is where the mismatch would show.
            settled = all_ranks_agree(True)
        finally:
            dist.destroy_process_group()
        out.put((rank, split, settled))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _decide_drain_split_across_two_ranks(sharded):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    server = _store_server()
    port = server.port
    procs = [
        ctx.Process(target=_drain_split_worker, args=(rank, 2, port, sharded, out))
        for rank in (0, 1)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, split, settled = out.get(timeout=120)
            results[rank] = (split, settled)
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():  # pragma: no cover - a hung collective
                p.terminate()
                p.join(timeout=10)
    return results


class TestTheDrainSplitGuardMatchesTheVerdict:
    """The skew barrier (GPU-98) puts a collective in the checkpoint path,
    gated by `sharded and get_world_size() > 1` — the same guard the
    checkpoint's own verdict collective carries, and for the same reason: a
    barrier some ranks enter and others do not is two different collectives
    posted on one process group, a hang until NCCL gives up rather than an
    error.

    This used to be `RAVEX_SPLIT_DRAIN`, an opt-in diagnostic agreed on by
    every rank rather than read locally, because the variable could be set
    asymmetrically. There is no variable left to set wrong now — the guard
    comes from `sharded` and `get_world_size()`, which read the same on every
    rank by construction — but that symmetry is exactly the kind of thing that
    is easy to believe and wrong to assume: this pins it on two real
    processes, where `world_size=1` would show nothing and a mock could not
    be trusted either way.
    """

    def test_both_ranks_split_on_a_sharded_run(self):
        results = _decide_drain_split_across_two_ranks(sharded=True)
        assert set(results) == {0, 1}, results
        for rank, (split, settled) in results.items():
            assert split is True, f"rank {rank}: {split!r}"
            assert settled is True, f"rank {rank} did not reach the end: {settled!r}"

    def test_neither_rank_splits_on_a_replicated_run(self):
        """Not sharded: the guard must keep both ranks out of the barrier,
        the same way it keeps the non-main ranks of a real replicated job out
        of the verdict collective (see
        `test_a_replicated_job_posts_no_collective_from_rank_zero_alone` in
        `test_patches.py`) — a barrier only rank 0 entered is a hang, not an
        error, and this is the multi-process pin for that guard.
        """
        results = _decide_drain_split_across_two_ranks(sharded=False)
        assert set(results) == {0, 1}, results
        for rank, (split, settled) in results.items():
            assert split is False, f"rank {rank} barriered alone: {split!r}"
            assert settled is True, f"rank {rank} did not reach the end: {settled!r}"
