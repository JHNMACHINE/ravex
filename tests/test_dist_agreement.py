"""GPU-111: agreeing about one bit without a collective.

``all_ranks_agree`` keeps a resume all-or-nothing, and until now it did that by
pickling a Python bool through two collectives — the most expensive call in this
family, measured at 1025 µs against 41–56 µs for a store lookup, and 5.7 ms per
rank once one rank is 5 ms late (``bench/agreement_cost.py``).

The point of these tests is not the speed. It is that the answer is the *same
answer*, and that the one place the two roads differ is a difference in the
right direction: a rank that goes silent produces ``False`` here, where the
collective produced a group-wide timeout. "Not every rank succeeded" is exactly
what the caller asked, and it is better delivered as an answer than as an
outage.

Built against a store made of a dictionary, because what is under test is the
protocol — the rounds, the deadline, the reduction — and not torch's key-value
server. ``test_both_roads_give_the_same_answer`` is the one that runs two real
processes with a real ``TCPStore`` behind it.
"""

import datetime
import multiprocessing as mp
import os
import socket
import threading
import time
import traceback

import pytest

from ravex._dist import agreement
from ravex._dist.agreement import all_gather_scalar


@pytest.fixture(autouse=True)
def fresh_rounds():
    """Round numbers are per process, and a test is not a process."""
    agreement.reset_rounds()
    yield
    agreement.reset_rounds()


class FakeStore:
    """The four methods this module asks of a rendezvous store."""

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()
        self.deleted = []

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def check(self, keys):
        with self._lock:
            return all(key in self._data for key in keys)

    def get(self, key):
        with self._lock:
            return self._data[key]

    def delete_key(self, key):
        with self._lock:
            self.deleted.append(key)
            self._data.pop(key, None)


def gather_from(store, votes, world_size=None, patience=None, name="q"):
    """Every rank votes at once, from its own thread. Returns the answers."""
    world_size = world_size if world_size is not None else len(votes)
    answers = {}

    def vote(rank):
        answers[rank] = all_gather_scalar(
            name, votes[rank], rank, world_size, store, patience=patience
        )

    threads = [threading.Thread(target=vote, args=(rank,)) for rank in votes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return answers


def test_every_rank_sees_every_answer():
    store = FakeStore()
    answers = gather_from(store, {0: True, 1: False, 2: True})

    assert answers[0] == [True, False, True]
    # Same list on every rank, in rank order, which is what makes a local
    # `all()` a decision the whole job agrees on.
    assert answers[0] == answers[1] == answers[2]


def test_a_silent_rank_is_an_answer_and_not_a_wait():
    """The one place the two roads differ, and the reason to prefer this one.

    A collective waits for the group's timeout and then fails everybody. Here
    rank 2 never votes, the question has a deadline, and running out of it says
    "no" — which is what "did every rank succeed" was asking.
    """
    store = FakeStore()
    started = time.monotonic()
    answers = gather_from(store, {0: True, 1: True}, world_size=3, patience=0.5)
    elapsed = time.monotonic() - started

    assert answers[0] is None and answers[1] is None
    assert elapsed < 10, "it waited far past its own deadline"


def test_the_missing_rank_is_named(caplog):
    """A collective's timeout says the group failed. This says who."""
    import logging

    store = FakeStore()
    with caplog.at_level(logging.WARNING, logger="ravex"):
        gather_from(store, {0: True}, world_size=2, patience=0.2)

    assert "rank(s) 1" in caplog.text


def test_two_questions_in_a_row_do_not_read_each_other():
    """Rounds, which is the whole reason the key carries a number.

    Ranks reach the same call sites in the same order — the invariant a
    collective already demands — so a per-process counter is the same counter
    on every rank. Without it the second question would read the first's votes
    and answer the wrong thing, silently.
    """
    store = FakeStore()
    first = gather_from(store, {0: True, 1: True})
    second = gather_from(store, {0: True, 1: False})

    assert first[0] == [True, True]
    assert second[0] == [True, False]


def test_old_rounds_are_dropped():
    """Small keys, but one per rank per round, on a store the job shares."""
    store = FakeStore()
    for _ in range(4):
        gather_from(store, {0: True, 1: True})

    assert store.deleted, "nothing was ever cleaned up"
    # Only from rank 0, and only rounds old enough that nobody is still reading
    # them: the two most recent survive.
    assert all("/agree/q/" in key for key in store.deleted)
    assert not any(key.endswith("/3/0") or key.endswith("/3/1") for key in store.deleted)


def test_only_scalars_travel():
    """A structure needs an encoding decided on purpose, not defaulted into."""
    store = FakeStore()
    with pytest.raises(TypeError, match="scalars only"):
        all_gather_scalar("q", {"step": 3}, 0, 2, store)


def test_one_rank_asks_nobody():
    store = FakeStore()
    assert all_gather_scalar("q", True, 0, 1, store) == [True]


# ─── both roads, two real processes, a real store ───────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _roads_worker(rank, port, out):
    try:
        import torch.distributed as dist

        from ravex._dist.collectives import all_ranks_agree
        from ravex._dist.agreement import all_gather_scalar, rendezvous_store

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60)
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
    port = _free_port()
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
