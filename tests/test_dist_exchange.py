"""GPU-115: collecting the round with a deadline instead of a collective.

Real sockets on loopback, a store made of a dictionary. The store is fake
because what is under test is the protocol — the rendezvous on a round number,
the deadline, the token — and not torch's key-value server, which is the same
split ``tests/test_dist_agreement.py`` makes for the same reason.

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
from ravex._dist.exchange import ADDRESS_KEY, DeltaExchange  # noqa: E402


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


def publish(exchange, delta, round_number, steps):
    exchange.publish(delta, round_number, steps)


def test_two_nodes_trade_a_round(nodes):
    zero, one = nodes(0), nodes(1)
    mine, theirs = a_delta(), a_delta(scale=2.0)
    publish(zero, mine, 0, 10)
    publish(one, theirs, 0, 25)

    got = zero.gather([1], 0, _report.expectation(mine), time.monotonic() + 10)

    assert len(got) == 1
    assert got[0].node == "n1"
    assert got[0].steps == 25
    assert torch.equal(got[0].delta["w"], theirs["w"])


def test_the_step_counts_come_back_differing(nodes):
    """A heterogeneous round is the normal case, not an anomaly to reconcile."""
    zero, one, two = nodes(0), nodes(1), nodes(2)
    delta = a_delta()
    publish(zero, delta, 0, 100)
    publish(one, delta, 0, 20)
    publish(two, delta, 0, 63)

    got = zero.gather([1, 2], 0, _report.expectation(delta), time.monotonic() + 10)
    assert sorted(report.steps for report in got) == [20, 63]


def test_a_node_that_never_answers_is_an_absence_and_not_a_failure(nodes, caplog):
    """The whole fault tolerance of GPU-113, on a real socket.

    Rank 1 has an address on the store and nothing behind it, which is what a
    box that was killed between advertising and publishing looks like.
    """
    zero = nodes(0)
    delta = a_delta()
    publish(zero, delta, 0, 10)
    zero.store.set(ADDRESS_KEY % 1, b"127.0.0.1:1")

    with caplog.at_level("WARNING", logger="ravex"):
        started = time.monotonic()
        got = zero.gather([1], 0, _report.expectation(delta), started + 3)

    assert got == []
    assert (time.monotonic() - started) < 10
    assert "did not contribute" in caplog.text


def test_one_silent_node_does_not_cost_the_others_their_round(nodes):
    """A collective would have failed the group; here the others are untouched."""
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 0, 10)
    publish(one, delta, 0, 11)
    zero.store.set(ADDRESS_KEY % 2, b"127.0.0.1:1")

    got = zero.gather([1, 2], 0, _report.expectation(delta), time.monotonic() + 3)

    assert [report.node for report in got] == ["n1"]


def test_a_node_that_never_advertised_is_simply_not_there(nodes):
    zero = nodes(0)
    delta = a_delta()
    publish(zero, delta, 0, 10)
    got = zero.gather([7], 0, _report.expectation(delta), time.monotonic() + 1)
    assert got == []


def test_a_peer_still_finishing_its_round_is_waited_for(nodes):
    """The rendezvous. Under a wall clock the peer is about to publish.

    A refusal here would mean every node needing to know in advance when the
    others will be ready, and there is no channel that carries that.
    """
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 0, 10)

    def late():
        time.sleep(0.4)
        publish(one, delta, 0, 99)

    threading.Thread(target=late, daemon=True).start()
    got = zero.gather([1], 0, _report.expectation(delta), time.monotonic() + 10)

    assert len(got) == 1 and got[0].steps == 99


def test_a_peer_behind_by_longer_than_one_read_slice_is_still_waited_for(
    nodes, monkeypatch
):
    """The wait is bounded by the round's deadline, not by one ``recv``.

    Found on two real machines, EU and US, on 2026-09-15: the American node
    spent 56 s in its first round, the European one waited for its report for
    exactly the 30 s of one read slice, took the timeout for an absence and
    closed round 1 without it — while the American node then found the
    European report and averaged it in. One round in, two models.

    The slice is shrunk so the test does not have to wait 30 s to show it: a
    peer five slices late has to come back as a report, because the deadline
    is ten seconds away.
    """
    from ravex._dist import exchange as _exchange

    monkeypatch.setattr(_exchange, "RECV_SLICE", 0.2)
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 0, 10)

    def late():
        time.sleep(1.0)
        publish(one, delta, 0, 42)

    threading.Thread(target=late, daemon=True).start()
    got = zero.gather([1], 0, _report.expectation(delta), time.monotonic() + 10)

    assert len(got) == 1 and got[0].steps == 42, (
        "a peer one second behind was treated as absent: the read slice "
        "decided the round instead of the deadline"
    )


def test_a_peer_already_past_the_round_says_so_instead_of_hanging(nodes):
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 0, 10)
    publish(one, delta, 5, 10)  # rank 1 is five rounds ahead

    started = time.monotonic()
    got = zero.gather([1], 0, _report.expectation(delta), started + 10)

    assert got == []
    assert (time.monotonic() - started) < 5, "GONE should be immediate"


def test_a_peer_training_another_model_is_refused_and_named(nodes, caplog):
    zero, one = nodes(0), nodes(1)
    mine = a_delta()
    publish(zero, mine, 0, 10)
    publish(one, {"w": torch.randn(9, 9), "b": torch.randn(9)}, 0, 10)

    with caplog.at_level("WARNING", logger="ravex"):
        got = zero.gather([1], 0, _report.expectation(mine), time.monotonic() + 5)

    assert got == []
    assert "Unusable report" in caplog.text


def test_a_connection_without_the_job_token_gets_nothing(nodes, caplog):
    """Anything that can reach the port can knock; the token is what answers."""
    import socket as _socket

    zero = nodes(0)
    publish(zero, a_delta(), 0, 10)
    host, _, port = zero.address.rpartition(":")

    from ravex._dist.exchange import (
        KIND_ROUND,
        REQUEST,
        REQUEST_MAGIC,
        RESPONSE,
    )

    with caplog.at_level("WARNING", logger="ravex"):
        connection = _socket.create_connection((host, int(port)), timeout=5)
        try:
            connection.sendall(
                REQUEST.pack(REQUEST_MAGIC, b"0" * 32, 1, 0, KIND_ROUND)
            )
            connection.settimeout(2)
            assert connection.recv(RESPONSE.size) == b"", "a stranger was answered"
        finally:
            connection.close()

    assert "without the job token" in caplog.text


def test_a_node_cannot_be_talked_into_answering_for_a_peer_it_did_not_reach(nodes):
    """Who answered is keyed by the peer dialled, never by the payload's name.

    Rank 1 reports itself as ``n0``. If the caller believed that field, one
    machine would be counted twice in the average.
    """
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 0, 10)
    one.node = "n0"  # rank 1 reports itself under rank 0's name
    publish(one, delta, 0, 77)

    got = zero.gather([1], 0, _report.expectation(delta), time.monotonic() + 5)
    assert len(got) == 1 and got[0].steps == 77


def test_the_deadline_is_the_deadline(nodes):
    """A peer that trickles must not hold the round open past its own clock."""
    zero = nodes(0)
    delta = a_delta()
    publish(zero, delta, 0, 10)

    import socket as _socket

    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    held = []

    def accept_and_say_nothing():
        try:
            connection, _ = listener.accept()
            held.append(connection)
            time.sleep(30)
        except OSError:
            pass

    threading.Thread(target=accept_and_say_nothing, daemon=True).start()
    zero.store.set(
        ADDRESS_KEY % 1, ("127.0.0.1:%d" % listener.getsockname()[1]).encode()
    )

    try:
        started = time.monotonic()
        got = zero.gather([1], 0, _report.expectation(delta), started + 2)
        elapsed = time.monotonic() - started
    finally:
        listener.close()
        for connection in held:
            connection.close()

    assert got == []
    assert elapsed < 8, "gather ran %.1fs past a 2s deadline" % (elapsed - 2)


def test_a_round_with_no_peers_asks_nothing(nodes):
    zero = nodes(0)
    assert zero.gather([], 0, _report.expectation(a_delta()), time.monotonic() + 5) == []


def test_the_exchange_survives_several_rounds_on_one_listener(nodes):
    """The listener and the token are built once; only the payload changes."""
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    for round_number in range(4):
        publish(zero, delta, round_number, 10)
        publish(one, delta, round_number, 20 + round_number)
        got = zero.gather([1], round_number, _report.expectation(delta),
                          time.monotonic() + 10)
        assert len(got) == 1
        assert got[0].round_number == round_number
        assert got[0].steps == 20 + round_number


def test_without_a_cast_a_node_averages_exactly_what_it_wrote(nodes):
    """No ``save_dtype``, no read back: the delta is its own wire form."""
    zero = nodes(0)
    delta = a_delta()
    publish(zero, delta, 0, 10)

    assert zero.as_published(delta, 0, _report.expectation(delta)) is delta


def test_a_cast_delta_is_averaged_as_the_peers_will_read_it(nodes):
    """The one thing that keeps every node taking the *same* outer step.

    With ``save_dtype`` the report on the wire is a cast of the delta, so a
    node that averaged the one it computed would be combining a different set
    from everybody else — and the nodes would part by a quantization error per
    round while every log line said the round closed over both of them.
    """
    zero, one = nodes(0, save_dtype="bf16"), nodes(1)
    delta = a_delta()
    expected = _report.expectation(delta)
    publish(zero, delta, 0, 10)

    mine = zero.as_published(delta, 0, expected)
    theirs = one.gather([0], 0, expected, time.monotonic() + 10)

    assert len(theirs) == 1
    for name in delta:
        # The cast really happened - otherwise this test would pass on a
        # function that returned its argument.
        assert not torch.equal(mine[name], delta[name]), name
        assert torch.equal(mine[name], theirs[0].delta[name]), name


def test_closing_twice_is_not_an_error(store, tmp_path):
    exchange = DeltaExchange(0, store, root=str(tmp_path), patience=2.0)
    assert exchange.start()
    exchange.close()
    exchange.close()


def test_leaving_waits_for_the_last_report_to_be_taken(nodes):
    """The defect the end-to-end test found, kept where it cannot come back.

    A node that shuts its listener the moment it has applied its own outer step
    takes its last report with it. The peer still fetching it gets nothing,
    closes that round with one contributor fewer, and the two models end the
    run different - which is a wrong answer arriving quietly, not a crash.
    """
    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 4, 10)
    publish(one, delta, 4, 10)

    late = []

    def fetch_after_a_moment():
        time.sleep(0.5)
        late.extend(
            one.gather([0], 4, _report.expectation(delta), time.monotonic() + 10)
        )

    puller = threading.Thread(target=fetch_after_a_moment, daemon=True)
    puller.start()

    zero.close(linger=10, expect=[1])
    puller.join(10)

    assert len(late) == 1, "rank 0 left before rank 1 had taken round 4"


def test_leaving_does_not_outlive_a_peer_that_will_never_fetch(nodes):
    """Best effort, and bounded: a dead peer is not worth waiting out."""
    zero = nodes(0)
    publish(zero, a_delta(), 0, 10)

    started = time.monotonic()
    zero.close(linger=1.0, expect=[9])
    elapsed = time.monotonic() - started

    assert 0.8 < elapsed < 5, "linger ran %.1fs for a peer that never comes" % elapsed


def test_a_publish_does_not_wait_for_a_fetch_in_flight(nodes, monkeypatch):
    """GPU-119, and it is the whole issue in one assertion.

    Until each round got its own directory a lock stood between writing this
    node's report and serving it, because one store held every round and
    moonclip renames its manifest into place. A peer draining a slow link
    therefore held a publish for the length of a transfer — 2.2-2.3 s against a
    round whose own transfer was 0.30 s, and worse as nodes were added, because
    the same lock serialised two peers fetching from one node.

    The send is stubbed to be slow rather than throttled for real: what is
    under test is whether anything blocks, not how fast a socket is.
    """
    import ravex._dist.exchange as exchange_module

    zero, one = nodes(0), nodes(1)
    delta = a_delta()
    publish(zero, delta, 0, 10)

    serving = threading.Event()

    class SlowCore:
        @staticmethod
        def prestage_send(fileno, source, chunk):
            serving.set()
            time.sleep(2.0)
            return 0

        @staticmethod
        def prestage_receive(fileno, destination):
            return False

    monkeypatch.setattr(exchange_module, "_rust_core", lambda: SlowCore)

    fetching = threading.Thread(
        target=one.gather,
        args=([0], 0, _report.expectation(delta), time.monotonic() + 10),
        daemon=True,
    )
    fetching.start()
    assert serving.wait(5), "the serve never started, so nothing is in flight"

    started = time.monotonic()
    publish(zero, delta, 1, 11)
    took = time.monotonic() - started

    assert took < 0.5, (
        "publishing round 1 waited %.2fs for a fetch of round 0 that is still "
        "in flight" % took
    )
    assert zero.publish_wait < 0.5
    fetching.join(10)


def test_retention_does_not_delete_a_round_that_is_on_a_socket(nodes):
    """The one race a lock used to cover, and now the only one left.

    Nothing writes a published round any more, so the sole way a directory can
    vanish under a reader is retention. It asks — with a set lookup, not a lock
    held across a transfer.
    """
    zero = nodes(0)
    delta = a_delta()

    for round_number in range(6):
        publish(zero, delta, round_number, 10)

    kept = sorted(_report.rounds_present(zero.mine_path))
    assert kept == [3, 4, 5], kept

    # Round 3 goes on a socket, then four more rounds are published.
    zero._hold(3)
    for round_number in range(6, 10):
        publish(zero, delta, round_number, 10)

    assert _report.round_is_complete(zero.mine_path, 3), (
        "retention deleted a round that was being served"
    )

    zero._release(3)
    publish(zero, delta, 10, 10)
    assert not _report.round_is_complete(zero.mine_path, 3)


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
