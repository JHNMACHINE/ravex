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
    intruder.sendall(b"z" * (4 + RingLink.NONCE_BYTES + RingLink.PROOF_CHARS))
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
            _recv_exactly(connection, 4 + RingLink.NONCE_BYTES + RingLink.PROOF_CHARS)
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


def test_a_proof_is_bound_to_everything_it_is_about():
    """Change any one input and it is a different proof (GPU-134).

    The label, so a greeting is never an answer; the nonce, so a proof seen
    once is no use later; each number, so a greeting to one rank is not a
    greeting to another; and the token, which is the point.
    """
    secret, nonce = b"a" * 32, b"n" * RingLink.NONCE_BYTES
    base = RingLink._proof(secret, b"hello", nonce, 0, 1)
    assert len(base) == RingLink.PROOF_CHARS
    variants = {
        RingLink._proof(secret, b"ack", nonce, 0, 1),
        RingLink._proof(secret, b"hello", b"m" * RingLink.NONCE_BYTES, 0, 1),
        RingLink._proof(secret, b"hello", nonce, 1, 0),
        RingLink._proof(secret, b"hello", nonce, 0, 2),
        RingLink._proof(b"b" * 32, b"hello", nonce, 0, 1),
    }
    assert base not in variants and len(variants) == 5


def test_a_token_handed_in_never_reaches_the_store(monkeypatch):
    """With RAVEX_JOB_TOKEN, the store holds nothing that proves membership.

    On a rendezvous strangers can reach, a token on the store is a token
    anybody has - GPU-134.
    """
    monkeypatch.setenv(RingLink.TOKEN_ENV, "handed-in")
    store = FakeStore()
    assert RingLink._shared_secret(store, 0, deadline=float("inf")) == b"handed-in"
    assert RingLink._shared_secret(store, 3, deadline=0.0) == b"handed-in"
    assert not store.check([RingLink.SECRET_KEY])


def test_each_life_of_the_job_is_its_own(monkeypatch):
    """The incarnation is rank 0's to make, new each time, and read by the rest."""
    store = FakeStore()
    first = RingLink._incarnation(store, 0, deadline=float("inf"))
    assert RingLink._incarnation(store, 2, deadline=float("inf")) == first
    assert RingLink._incarnation(store, 0, deadline=float("inf")) != first


def test_dialling_an_impostor_does_not_hand_it_the_token(monkeypatch):
    """The hole GPU-134 found: the greeting used to *be* the token.

    A stranger that can write the store advertises a listener under a rank's
    key and waits. Whatever the real rank sends it has to be useless to it.
    """
    monkeypatch.setenv(RingLink.TOKEN_ENV, "the-real-token")
    store = FakeStore()
    impostor = socket.socket()
    impostor.bind(("127.0.0.1", 0))
    impostor.listen(1)
    impostor.settimeout(15)
    store.set(
        RingLink.ADDRESS_KEY % 1,
        ("127.0.0.1:%d" % impostor.getsockname()[1]).encode("utf-8"),
    )
    heard = {}

    def listen():
        try:
            connection, _ = impostor.accept()
            heard["greeting"] = _recv_exactly(connection, 4 + RingLink.NONCE_BYTES + RingLink.PROOF_CHARS)
            connection.close()
        except OSError:
            pass

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    assert RingLink.connect(0, 1, 1, store, timeout_seconds=5) is None
    thread.join(timeout=20)
    impostor.close()

    assert b"the-real-token" not in heard["greeting"]
