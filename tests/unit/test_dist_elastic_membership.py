"""GPU-94 step 3: discovery and decision, the two pieces above the
rendezvous mechanics proven in test_dist_elastic_rendezvous.py.

Discovery needs no process group at all - a candidate rank announces itself
on the raw persistent store before it has ever called
``init_process_group``, and whoever polls (rank 0) does so without blocking
its own step loop. Decision generalises GPU-92's emergency channel from a
single bit to a small object, over the same isolated gloo subgroup.

Regroup itself (destroy + reinit at a new world_size via a fresh
``PrefixStore`` generation) is exercised in test_dist_elastic_rendezvous.py and
test_dist_elastic_remesh.py - not repeated here.
"""

import multiprocessing as mp
import traceback



def _store_server():
    """A rendezvous store for one spawned group, held by the test's own process.

    On port 0, with the port read back from the store. The probe it replaces
    found a free port, closed it and handed the number to a rank to listen on;
    under `pytest -n` another worker can take it in between, and that rank dies
    with EADDRINUSE (GPU-191, as in test_emergency_checkpoint and
    test_dist_multinode). A server that is already listening has no such
    window. The caller keeps it alive until its ranks are done.
    """
    from ravex._dist.rendezvous import _tcp_store

    return _tcp_store("127.0.0.1", 0, True, 120.0)


def _store_client(port):
    """The store `_store_server` opened, from inside a rank."""
    from ravex._dist.rendezvous import _tcp_store

    return _tcp_store("127.0.0.1", port, False, 120.0)


# ─── discovery: no process group involved at all ───────────────────────


# ─── decision: real 2-rank gloo, isolated subgroup, richer payload ──────


def _decision_worker(rank, world_size, port, local_view, timeout_seconds, out):
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"

    try:
        import torch.distributed as dist

        from ravex._dist.elastic import topology_decision

        dist.init_process_group("gloo", store=_store_client(port), rank=rank, world_size=world_size)
        try:
            decision = topology_decision(local_view, timeout_seconds)
        finally:
            dist.destroy_process_group()
        out.put((rank, decision))
    except Exception:
        out.put((rank, f"EXC: {traceback.format_exc()}"))


def _run_decision_across_two_ranks(views_by_rank, timeout_seconds=20):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    server = _store_server()
    port = server.port
    procs = [
        ctx.Process(
            target=_decision_worker,
            args=(rank, 2, port, views_by_rank[rank], timeout_seconds, out),
        )
        for rank in (0, 1)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, decision = out.get(timeout=60)
            results[rank] = decision
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():  # pragma: no cover - a hung collective
                p.terminate()
                p.join(timeout=10)
    return results


class TestTopologyDecisionCarriesMoreThanABit:
    """Same channel GPU-92 built, carrying more than GPU-92 ever needed it to."""

    def test_a_group_that_cannot_be_built_falls_back_to_the_local_view_alone(
        self, monkeypatch
    ):
        """Same degrade contract `emergency_group`'s own caller already
        relies on (see test_emergency_checkpoint.py's
        TestTheDetectionRoundDegradesCleanly) - here exercised through
        topology_decision instead of emergency_signalled, single process,
        no real distributed needed since the whole point is that no group
        gets built.
        """
        import ravex._dist.elastic as elastic

        monkeypatch.setattr(
            "ravex._dist.collectives.emergency_group", lambda timeout: (False, None)
        )
        result = elastic.topology_decision({"wants_join": True}, timeout_seconds=5)
        assert result == [{"wants_join": True}]
