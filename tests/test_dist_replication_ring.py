"""GPU-112: who is on the other end of a replication socket.

``RingLink`` opens a listening port on every rank, which is a surface the
process group never had — gloo and NCCL only ever speak to a member, and
membership is the launcher's to decide. Here anything that can reach the port
can knock, and a knock is not harmless: ``StoreWriter`` removes the completion
marker before the first byte lands, so a connection that arrives and then says
nothing is enough to stop a good replica from being read as one.

So both ends prove they hold a token rank 0 leaves on the rendezvous store.
These tests are about that handshake and nothing else, which is why they build
the ring in one process against a store made of a dictionary: the multi-process
version lives in ``test_dist_replication_sockets.py`` and answers a different
question (do the two roads leave the same replica). Here the questions are:
does a stranger get in, does a stranger's failure cost the real peer its ring,
and does a rank push its store into a port that cannot answer for itself.

**What the token is not.** A shared secret in the clear on a connection nobody
has encrypted. It identifies; it does not protect. Someone who can read the
traffic between two ranks, or take over an established connection, is not
stopped by it — and on a network where that is the threat,
``replication_transport: collectives`` is the answer and stays supported.
"""

import os
import socket
import threading

import pytest

from ravex._dist.replication import RingLink, _recv_exactly, store_files


@pytest.fixture(autouse=True)
def rendezvous_on_loopback(monkeypatch):
    """Pin what `RingLink._advertise` probes towards.

    It asks the routing table which local address reaches the rendezvous, and
    with no MASTER_ADDR it falls back to this host's name — which on a machine
    with a Hyper-V, docker or VPN interface resolves to several addresses and
    costs ten seconds per dial before the right one is tried. That fallback is
    deliberate and belongs in the code; a test that pays for it is measuring
    the local network stack rather than the ring.
    """
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")


class FakeStore:
    """The three methods `RingLink` asks of a rendezvous store.

    A dictionary rather than a real ``TCPStore`` because what is under test is
    the handshake, not torch's key-value server. `check` is non-blocking and
    `get` is only ever called after it, which is exactly how `RingLink` uses
    the real one.
    """

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def check(self, keys):
        with self._lock:
            return all(key in self._data for key in keys)

    def get(self, key):
        with self._lock:
            return self._data[key]


def build_ring(store, timeout_seconds=15):
    """Both ends of a two-rank ring, connected from one process."""
    links = {}

    def build(rank):
        links[rank] = RingLink.connect(
            rank, 1 - rank, 1 - rank, store, timeout_seconds=timeout_seconds
        )

    threads = [threading.Thread(target=build, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout_seconds + 10)
    return links.get(0), links.get(1)


def write(root, relative, content: bytes):
    path = os.path.join(root, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


def address_of(store, rank, timeout=10.0):
    """Wait for a rank to advertise itself, the way its peer would."""
    key = RingLink.ADDRESS_KEY % rank
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            if store.check([key]):
                text = store.get(key).decode("utf-8")
                host, _, port = text.rpartition(":")
                return host, int(port)
        raise AssertionError("rank %d never advertised" % rank)
    finally:
        timer.cancel()


def test_the_ring_forms_and_carries_a_store(tmp_path):
    store = FakeStore()
    zero, one = build_ring(store)
    assert zero is not None and one is not None

    source = str(tmp_path / "store")
    destination = str(tmp_path / "replica")
    write(source, "manifest.json", b'{"step": 3}')
    write(source, "step_3/shard.bin", b"D" * 9000)
    os.makedirs(destination, exist_ok=True)

    outcome = {}

    def receive():
        outcome["ok"] = one.round(None, destination)

    thread = threading.Thread(target=receive)
    thread.start()
    zero.round(source, None)
    thread.join(timeout=30)

    zero.close()
    one.close()

    assert outcome.get("ok") is True
    names = {name for name, _size in store_files(destination)}
    assert "step_3/shard.bin" in names
    assert ".ravex-replica-ok" in names


def test_a_stranger_is_refused_and_the_real_peer_still_gets_in(tmp_path, caplog):
    """The rejection must cost the stranger's connection and nothing else.

    Accepting once and trusting whatever arrived would let anything that
    reaches the port first take the predecessor's place — and the predecessor
    would then wait for a ring that never forms, on a rank that is not the one
    doing anything wrong.
    """
    store = FakeStore()
    zero = {}

    def build_zero():
        zero["link"] = RingLink.connect(0, 1, 1, store, timeout_seconds=15)

    thread = threading.Thread(target=build_zero)
    thread.start()

    # In before rank 1, saying nothing it is entitled to say.
    intruder = socket.create_connection(address_of(store, 0), timeout=10)
    intruder.sendall((0).to_bytes(4, "big") + b"z" * RingLink.TOKEN_CHARS)
    assert intruder.recv(16) == b"", "the port answered a connection with no token"
    intruder.close()

    one = RingLink.connect(1, 0, 0, store, timeout_seconds=15)
    thread.join(timeout=30)

    try:
        assert one is not None, "the stranger cost the real peer its ring"
        assert zero.get("link") is not None
    finally:
        for link in (zero.get("link"), one):
            if link is not None:
                link.close()


def test_a_port_that_cannot_answer_is_not_pushed_to(tmp_path):
    """The other half of the proof, and the one that protects the *sender*.

    Reading a peer's address from the store and pushing a checkpoint into
    whatever answers is how a store ends up somewhere nobody meant it to go —
    a port that was reused, or one that was never a rank at all.
    """
    store = FakeStore()

    impostor = socket.socket()
    impostor.bind(("127.0.0.1", 0))
    impostor.listen(1)
    impostor.settimeout(15)
    store.set(
        RingLink.ADDRESS_KEY % 1,
        ("127.0.0.1:%d" % impostor.getsockname()[1]).encode("utf-8"),
    )

    def answer_badly():
        try:
            connection, _ = impostor.accept()
            _recv_exactly(connection, 4 + RingLink.TOKEN_CHARS)
            connection.sendall(b"n" * 64)
            connection.close()
        except OSError:
            pass

    thread = threading.Thread(target=answer_badly, daemon=True)
    thread.start()

    link = RingLink.connect(0, 1, 1, store, timeout_seconds=5)
    thread.join(timeout=20)
    impostor.close()

    assert link is None, "a store would have been pushed into an unknown port"


def test_the_token_is_read_rather_than_invented_by_everyone():
    """Rank 0 makes it; the others wait for it.

    A token each rank generated for itself would be two tokens and no ring, and
    the failure would look like a network problem rather than a logic one.
    """
    store = FakeStore()
    mine = RingLink._shared_secret(store, 0, deadline=float("inf"))
    assert len(mine) == RingLink.TOKEN_CHARS
    assert RingLink._shared_secret(store, 3, deadline=float("inf")) == mine


def test_a_rank_that_finds_no_token_does_not_open_a_port():
    """No store to read it from is a fallback, not an error.

    `connect` returning None puts the round back on the collectives, which need
    no rank to be reachable by anyone — the road that still works when this one
    cannot be built.
    """
    empty = FakeStore()
    assert RingLink.connect(1, 0, 0, empty, timeout_seconds=1) is None


@pytest.mark.parametrize("peer", [0, 1, 7])
def test_the_answer_is_bound_to_the_rank_that_asked(peer):
    """So one rank's answer is not a reusable answer to another's greeting."""
    secret = b"a" * RingLink.TOKEN_CHARS
    answers = {RingLink._acknowledgement(secret, rank) for rank in (0, 1, 7)}
    assert len(answers) == 3
    assert RingLink._acknowledgement(secret, peer) != RingLink._acknowledgement(
        b"b" * RingLink.TOKEN_CHARS, peer
    )
