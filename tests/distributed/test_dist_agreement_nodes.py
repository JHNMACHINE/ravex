"""GPU-111: agreeing about one bit without a collective - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_agreement.py.
"""

import datetime
import multiprocessing as mp
import traceback

import pytest



# ─── agree_on_step, the other gather of the family ─────────────────


# ─── both roads, two real processes, a real store ───────────────────────────


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


def _roads_worker(rank, port, out):
    try:
        import torch.distributed as dist

        from ravex._dist.collectives import all_ranks_agree
        from ravex._dist.agreement import all_gather_scalar, rendezvous_store

        dist.init_process_group(
            "gloo",
            store=_store_client(port),
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=60),
        )

        store = rendezvous_store()
        if store is None:
            out.put((rank, "no-store", None))
            return

        answers = {}
        for label, votes in (("all-yes", (True, True)), ("one-no", (True, False))):
            over_store = all_gather_scalar(
                label, votes[rank], rank, 2, store, patience=30
            )
            # The public function, which prefers the store road when there is
            # one - so this is the road under test, not a second opinion.
            answers[label] = {
                "store": all(over_store) if over_store is not None else None,
                "public": all_ranks_agree(votes[rank]),
            }
            dist.barrier()

        dist.destroy_process_group()
        out.put((rank, "done", answers))
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def test_both_roads_give_the_same_answer():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    server = _store_server()
    port = server.port
    procs = [
        ctx.Process(target=_roads_worker, args=(rank, port, out)) for rank in (0, 1)
    ]
    for process in procs:
        process.start()

    results = {}
    try:
        for _ in range(2):
            try:
                rank, status, payload = out.get(timeout=120)
            except Exception:
                break
            results[rank] = (status, payload)
    finally:
        for process in procs:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                pytest.fail("a rank hung: %r" % (results,))

    for rank, (status, payload) in results.items():
        if status == "EXC":
            pytest.fail("rank %d: %s" % (rank, payload))
        if status == "no-store":
            pytest.skip("no rendezvous store to ask on")
    assert set(results) == {0, 1}, results

    for rank in (0, 1):
        answers = results[rank][1]
        assert answers["all-yes"] == {"store": True, "public": True}
        # One rank saying no is the whole job saying no, on either road.
        assert answers["one-no"] == {"store": False, "public": False}
