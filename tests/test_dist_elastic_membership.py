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

import pytest


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


#: How long a worker waits for the *other* worker to get somewhere, which is
#: mostly how long a freshly spawned interpreter takes to import torch.
#:
#: It was 15 s, and 15 s is a claim about how fast the machine is rather than
#: anything about the code under test. On a CI runner sharing four cores with
#: three other pytest workers it stopped being true: the master's wait expired,
#: it polled a store nobody had written to yet, and the red read
#: `assert None == '10.0.0.9:29500'` - a sentence about `pending_join` for a
#: candidate that was still importing torch. The rest of this suite gives the
#: same kind of step 60 s and more.
#:
#: And a wait that runs out now says so, instead of letting the poll happen
#: anyway and leaving the assertion to describe the wrong thing.
SPAWN_GRACE = 120.0


def _discovery_worker(is_master, port, ready_evt, announced_evt, out):
    try:
        import torch.distributed as dist

        from ravex._dist.elastic import announce_join, pending_join

        if is_master:
            # The test holds the server (`_store_server`); this side is the
            # one that polls, and says when it is connected.
            store = _store_client(port)
            ready_evt.set()

            # A poll before the candidate announces itself must not block -
            # `pending_join` is meant for a step loop, not a wait.
            first_look = pending_join(store, candidate_rank=2)
            out.put(("rank0_before", first_look))

            if not announced_evt.wait(timeout=SPAWN_GRACE):
                out.put(("rank0_after", "the candidate never announced itself"))
            else:
                second_look = pending_join(store, candidate_rank=2)
                out.put(("rank0_after", second_look))
        else:
            if not ready_evt.wait(timeout=SPAWN_GRACE):
                out.put(("EXC", "rank 0 never connected to the store"))
                return
            store = _store_client(port)
            announce_join(store, candidate_rank=2, address="10.0.0.9:29500")
            announced_evt.set()
            out.put(("candidate", "announced"))
    except Exception:
        out.put(("EXC", traceback.format_exc()))


def test_a_candidate_with_no_process_group_can_still_announce_itself():
    """The core claim: announcing and polling need nothing from torch.distributed
    beyond the raw store - no init_process_group anywhere in this test.
    """
    ctx = mp.get_context("spawn")
    server = _store_server()
    port = server.port
    ready_evt = ctx.Event()
    announced_evt = ctx.Event()
    out = ctx.Queue()
    procs = [
        ctx.Process(
            target=_discovery_worker,
            args=(is_master, port, ready_evt, announced_evt, out),
        )
        for is_master in (True, False)
    ]
    for p in procs:
        p.start()

    messages = {}
    try:
        for _ in range(3):
            try:
                key, value = out.get(timeout=SPAWN_GRACE + 30)
            except Exception:
                break
            messages[key] = value
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    assert messages.get("rank0_before") is None, (
        "polling before the candidate announced itself must return None, "
        "not block or raise"
    )
    assert messages.get("candidate") == "announced"
    assert messages.get("rank0_after") == "10.0.0.9:29500"


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

    def test_both_ranks_see_the_same_gathered_view(self):
        views = {
            0: {"wants_join": False, "step": 40},
            1: {"wants_join": True, "candidate": 2, "step": 40},
        }
        results = _run_decision_across_two_ranks(views)

        assert set(results) == {0, 1}, results
        for rank, decision in results.items():
            assert decision == [views[0], views[1]], f"rank {rank}: {decision}"

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
