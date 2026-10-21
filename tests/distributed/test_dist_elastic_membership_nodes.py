"""GPU-94 step 3: discovery and decision, the two pieces above the - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_elastic_membership.py.
"""

import multiprocessing as mp
import traceback

from test_dist_elastic_membership import _run_decision_across_two_ranks, _store_client, _store_server  # noqa: E402


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
