"""GPU-115: collecting the round with a deadline instead of a collective.

Real sockets on loopback, a store made of a dictionary. The store is fake
because what is under test is the protocol — the rendezvous on a round number,
the deadline, the token — and not torch's key-value server, which is the same
split ``tests/unit/test_dist_agreement.py`` makes for the same reason.

The sockets are **not** fake, and that is deliberate. Every claim this module
makes is about what happens when the other end does not behave: a node that
never starts, one that answers late, one that answers with another model's
tensors, one that reached the port without belonging to the job. None of those
can be produced by a stub that stands in for the network, because in each of
them the network is the thing doing the work.
"""

import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("moonclip")

from ravex._dist import report as _report  # noqa: E402
from ravex._dist.exchange import DeltaExchange  # noqa: E402


class FakeStore:
    """The methods this module asks of a rendezvous store."""

    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.values[key] = value

    def get(self, key):
        with self.lock:
            return self.values[key]

    def check(self, keys):
        with self.lock:
            return all(key in self.values for key in keys)

    def delete_key(self, key):
        with self.lock:
            return self.values.pop(key, None) is not None

    def compare_set(self, key, expected, desired):
        """``TCPStore``'s semantics, checked against one on 2026-09-21: with
        ``expected`` empty the first writer wins and every later caller gets
        the stored value back; a missing key with ``expected`` set is left
        missing and ``expected`` comes back."""
        with self.lock:
            current = self.values.get(key)
            if current is None:
                if expected == "":
                    self.values[key] = desired.encode("utf-8")
                    return self.values[key]
                return expected.encode("utf-8")
            if current == expected.encode("utf-8"):
                self.values[key] = desired.encode("utf-8")
            return self.values[key]


def a_delta(scale=1.0):
    torch.manual_seed(0)
    return {"w": torch.randn(3, 4) * scale, "b": torch.randn(3) * scale}


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def nodes(store, tmp_path):
    """Two started exchanges, closed however the test ends."""
    made = []

    def make(rank, save_dtype=None):
        exchange = DeltaExchange(
            rank,
            store,
            root=os.path.join(str(tmp_path), "rank%d" % rank),
            node="n%d" % rank,
            patience=5.0,
            save_dtype=save_dtype,
        )
        assert exchange.start(), "rank %d could not open its listener" % rank
        made.append(exchange)
        return exchange

    yield make
    for exchange in made:
        exchange.close()


def test_a_round_is_not_servable_until_its_marker_lands(nodes, tmp_path):
    """A directory without the marker is a publish in progress, not a report."""
    root = str(tmp_path / "half-written")
    os.makedirs(_report.round_path(root, 7), exist_ok=True)

    assert not _report.round_is_complete(root, 7)

    _report.publish_round(root, 7, a_delta(), 3, "n0")
    assert _report.round_is_complete(root, 7)


# -- the round's decision (GPU-140) ------------------------------------------


def test_the_first_proposal_is_the_round_every_node_applies():
    from ravex._dist.exchange import decide_round

    store = FakeStore()
    deadline = time.monotonic() + 5
    assert decide_round(store, b"t", 3, 1, {1, 0}, deadline) == ([0, 1], 1)
    # A later node holding more, or less, still gets the first answer.
    assert decide_round(store, b"t", 3, 0, {0}, deadline) == ([0, 1], 1)
    assert decide_round(store, b"t", 3, 2, {0, 1, 2}, deadline) == ([0, 1], 1)
    # Another round, and another incarnation of the job, are other keys.
    assert decide_round(store, b"t", 4, 0, {0}, deadline) == ([0], 0)
    assert decide_round(store, b"u", 3, 2, {2}, deadline) == ([2], 2)


def test_a_store_that_does_not_answer_stops_the_node_instead_of_guessing():
    from ravex._dist.exchange import RoundSplitError, decide_round

    class Down(FakeStore):
        def compare_set(self, key, expected, desired):
            raise ConnectionError("rendezvous unreachable")

    started = time.monotonic()
    with pytest.raises(RoundSplitError, match="could not read round 3"):
        decide_round(Down(), b"t", 3, 0, {0}, started + 0.5)
    assert time.monotonic() - started < 5


def test_an_unreadable_decision_is_not_applied():
    from ravex._dist.exchange import ROUND_SET_KEY, RoundSplitError, _job_digest, decide_round

    store = FakeStore()
    store.set(ROUND_SET_KEY % (_job_digest(b"t"), 3), b"not json")
    with pytest.raises(RoundSplitError, match="unreadable"):
        decide_round(store, b"t", 3, 0, {0}, time.monotonic() + 1)


def test_the_decision_holds_on_a_real_tcpstore_with_racing_proposers():
    """``FakeStore`` imitates ``compare_set``; this is the real one, raced."""
    import datetime

    dist = pytest.importorskip("torch.distributed")
    from ravex._dist.exchange import decide_round

    server = dist.TCPStore(
        "127.0.0.1", 0, 1, True, timeout=datetime.timedelta(seconds=10),
        use_libuv=False,
    )
    answers = {}

    def propose(rank):
        client = dist.TCPStore(
            "127.0.0.1", server.port, 1, False,
            timeout=datetime.timedelta(seconds=10), use_libuv=False,
        )
        answers[rank] = decide_round(
            client, b"t", 7, rank, {rank}, time.monotonic() + 10
        )

    threads = [threading.Thread(target=propose, args=(r,)) for r in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert len(answers) == 8
    decided = set(map(repr, answers.values()))
    assert len(decided) == 1, "the proposers disagree: %s" % decided
    members, by = next(iter(answers.values()))
    assert members == [by]


def test_old_decisions_leave_the_store_and_recent_ones_stay():
    """GPU-142: one key per round for the life of a run is not a leak worth
    having on a server that outlives the run."""
    from ravex._dist.exchange import (
        APPLIED_KEY, KEEP_DECISIONS, ROUND_SET_KEY, _job_digest, decide_round,
    )

    store = FakeStore()
    exchange = DeltaExchange(0, store, root=".", node="n0")
    exchange.scope = b"t"
    digest = _job_digest(b"t")
    for round_number in range(1, KEEP_DECISIONS + 3):
        decide_round(store, b"t", round_number, 0, {0}, time.monotonic() + 1)
        exchange.applied(round_number)

    assert not store.check([ROUND_SET_KEY % (digest, 1)])
    assert not store.check([APPLIED_KEY % (digest, 2, 0)])
    assert store.check([ROUND_SET_KEY % (digest, 3)])
    assert store.check([APPLIED_KEY % (digest, KEEP_DECISIONS + 2, 0)])


def test_a_member_that_left_is_not_waited_for_and_not_asked():
    from ravex._dist.membership import Membership, acknowledge, acknowledged, leave

    store = FakeStore()
    acknowledge(store, 0, 5, 9)
    assert acknowledged(store, 5, [0, 1, 2]) == [1, 2]
    leave(store, 2, 7)
    assert acknowledged(store, 5, [0, 1, 2]) == [1], (
        "a joiner must not wait for a number that declared itself gone"
    )

    membership = Membership(store, 0, 3)
    assert membership.peers_at(8) == [1]
