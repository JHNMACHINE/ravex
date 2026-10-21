"""GPU-112: who is on the other end of a replication socket - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_replication_ring.py.
"""

import os
import socket
import threading


from ravex._dist.replication import RingLink, _recv_exactly, store_files
from test_dist_replication_ring import FakeStore  # noqa: E402


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
