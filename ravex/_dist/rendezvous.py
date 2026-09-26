"""A rendezvous of Ravex's own: nodes that start on their own, with no torchrun.

GPU-129, under GPU-113. Everything above this module — the round exchange, the
membership of GPU-121, the job token — talks to a key-value store and to
nothing else. Until now that store was torch's: the one ``init_process_group``
builds, which exists only once every rank has arrived, lives inside the
torchrun agent of whichever machine is the rendezvous endpoint, and dies with
that machine. For a run whose nodes are rented on different continents and
come and go, two of those are disqualifying:

* **Everybody has to be there at the start.** ``init_process_group`` is
  collective. A node that arrives later has no rank and no world size to be
  given, which is why GPU-121's join was reachable from a test and from nothing
  a user could launch.
* **The store dies with one node.** The exchange reads peer addresses from it
  on every fetch, so losing the endpoint's machine stops every round, not only
  that node's contribution.

So the store moves out of the training processes into one of its own:
``ravex rendezvous``. It is torch's ``TCPStore`` — the same object, with the
``wait`` and ``multi_get`` the agreement code already relies on — hosted by a
small process that trains nothing and can be kept alive somewhere that is not a
spot GPU. The nodes are plain ``python train.py``.

**Identity is a counter, and a number is never handed out twice.** A node takes
the next one with ``add``, which the server applies atomically. The first
``min_nodes`` are the base members: they wait for each other and start the run
together, as ranks did under torchrun. Every later number is a joiner and goes
through :func:`ravex._dist.membership.join`. A node that crashes and comes back
is a new node with a new number, which is the only reading under which
"membership is a function of the round number" survives a restart.

**What it does not do, said before anyone relies on it.**

* ``TCPStore`` has **no authentication** of its own. With ``RAVEX_JOB_TOKEN``
  set (GPU-134), on the server and on every node, it is not what listens on
  the public port: the server keeps it on loopback and puts a gate in front,
  and a node reaches it through a forwarder of its own. Both ends prove they
  hold the token before a byte of the store's protocol goes through, so a
  stranger never gets a number, a key or a place in the membership - and the
  exchange and the ring prove the same token again on their own connections.
  One token per server: jobs sharing a server share it, so they have to trust
  each other. Without the token the store is on the port for anyone, and so is
  the token rank 0 leaves on it; the port belongs on a private network.
* The server is **still a single point** — moved from a rented GPU to a process
  that is cheap to keep alive. If it restarts, the job's state on it is gone.
* A base member that dies **before** the run starts cannot be replaced by
  relaunching it: the relaunch takes the next number and arrives as a joiner,
  into a run that never started.
"""

from __future__ import annotations

import datetime
import hmac
import logging
import os
import re
import secrets
import socket
import threading
import time
from typing import List, Optional, Tuple

logger = logging.getLogger("ravex")

#: Where a node finds the server when the configuration does not say.
ENV = "RAVEX_RENDEZVOUS"

#: The port ``ravex rendezvous`` listens on unless told otherwise. Not torch's
#: 29500, so a rendezvous and a torchrun job can share a box.
DEFAULT_PORT = 29400

#: Every key a job writes lives under this, so one server holds many jobs —
#: which is what a platform hosting it needs, and costs the process nothing.
JOB_PREFIX = "ravex/job/%s/"

#: A job name goes into every key, so it is kept to characters that cannot
#: change what a key means.
_JOB_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

#: The counter node numbers come from.
NEXT_KEY = "ravex/gpu129/next"

#: How often a waiting base member looks again.
POLL = 0.05

#: The token that closes the server, and the one the exchange and the ring
#: prove (GPU-134): one secret per run, handed to the server and to every node.
TOKEN_ENV = "RAVEX_JOB_TOKEN"

#: What a gated server says first, so a node knows it reached one.
GATE_MAGIC = b"RVXGATE1"
NONCE_BYTES = 16
PROOF_CHARS = 64

#: How long either side of the gate waits for the other's half of the
#: handshake. A connection that says nothing costs this and one thread.
GATE_PATIENCE = 10.0

#: How often a node dials again a server that is not up yet.
DIAL_RETRY = 0.5

#: The forwarders of this process's gated clients. A ``PrefixStore`` takes no
#: attributes, so nothing else could keep them alive as long as their store.
_FORWARDERS: List["_Forwarder"] = []


def parse_address(value: str) -> Tuple[str, int]:
    """``host:port``, ``[v6]:port``, or a host alone for :data:`DEFAULT_PORT`."""
    text = str(value).strip()
    host, sep, port = text.rpartition(":")
    if not sep or (text.count(":") > 1 and not text.startswith("[")):
        # No colon, or a bare IPv6 address with no port.
        host, port = text, ""
    host = host.strip("[]")
    if not host:
        raise ValueError("rendezvous address %r has no host" % (value,))
    if not port:
        return host, DEFAULT_PORT
    try:
        number = int(port)
    except ValueError:
        raise ValueError(
            "rendezvous address %r has port %r, which is not a number" % (value, port)
        ) from None
    if not 0 < number < 65536:
        raise ValueError("rendezvous address %r has port %d, out of range" % (value, number))
    return host, number


def valid_job(name: object) -> bool:
    """Whether ``name`` can go into a key as it is."""
    return isinstance(name, str) and bool(_JOB_NAME.match(name))


def _tcp_store(host: str, port: int, is_master: bool, timeout: float):
    """torch's ``TCPStore``, on whichever implementation this torch has.

    libuv first, because it is torch's default and the one built for many
    clients. Windows CPU wheels are built without it and refuse the default
    with a message naming libuv — and ``USE_LIBUV=0``, which that message
    suggests, is not honoured when the argument is passed. The older
    implementation speaks the same protocol, so that is what they get.
    """
    import torch.distributed as dist

    kwargs = dict(
        world_size=None,
        is_master=is_master,
        timeout=datetime.timedelta(seconds=timeout),
        wait_for_workers=False,
    )
    try:
        return dist.TCPStore(host, port, **kwargs)
    except Exception as exc:
        if "libuv" not in str(exc):
            raise
    return dist.TCPStore(host, port, use_libuv=False, **kwargs)


def job_token(token: Optional[str] = None) -> Optional[bytes]:
    """``token``, or ``RAVEX_JOB_TOKEN``; None when neither says anything."""
    value = os.environ.get(TOKEN_ENV, "") if token is None else token
    return value.encode("utf-8") if value else None


def _hello(token: bytes, nonces: bytes) -> bytes:
    from ravex._dist.replication import RingLink

    return RingLink._proof(token, b"ravex-rendezvous-hello", nonces)


def _welcome(token: bytes, nonces: bytes) -> bytes:
    from ravex._dist.replication import RingLink

    return RingLink._proof(token, b"ravex-rendezvous-welcome", nonces)


def _recv(sock, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise OSError("closed after %d of %d bytes" % (len(data), size))
        data += chunk
    return data


def _close(*socks) -> None:
    for sock in socks:
        try:
            sock.close()
        except OSError:
            pass


def _pipe(one, other) -> None:
    """Bytes both ways between two sockets, until either end is done."""

    def pump(source, sink):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (source, sink):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    back = threading.Thread(target=pump, args=(other, one), daemon=True)
    back.start()
    pump(one, other)
    back.join()
    _close(one, other)


class _Listener:
    """A port, and a thread per connection that arrives at it."""

    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(128)
        self.port = self.sock.getsockname()[1]
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                connection, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(connection,), daemon=True).start()

    def handle(self, connection) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True
        _close(self.sock)


class _Gate(_Listener):
    """The public port of a server with a token (GPU-134).

    Speaks first, with a nonce, and lets through only a connection that
    answers it with a proof of the token; then relays bytes to the store on
    loopback, which never sees anything else. Refusing costs the stranger its
    connection and nothing more: no number, no key read, no key written.
    """

    def __init__(self, host: str, port: int, token: bytes, store_port: int):
        self.token = token
        self.store_port = store_port
        super().__init__(host, port)

    def handle(self, connection) -> None:
        inner = None
        try:
            connection.settimeout(GATE_PATIENCE)
            mine = secrets.token_bytes(NONCE_BYTES)
            connection.sendall(GATE_MAGIC + mine)
            answer = _recv(connection, NONCE_BYTES + PROOF_CHARS)
            nonces = mine + answer[:NONCE_BYTES]
            if not hmac.compare_digest(answer[NONCE_BYTES:], _hello(self.token, nonces)):
                logger.warning("Rendezvous: refused a connection without the job token.")
                _close(connection)
                return
            connection.sendall(_welcome(self.token, nonces))
            connection.settimeout(None)
            inner = socket.create_connection(("127.0.0.1", self.store_port))
        except OSError:
            _close(connection)
            if inner is not None:
                _close(inner)
            return
        _pipe(connection, inner)


class Gated:
    """A server behind a gate: what :func:`serve` returns when there is a token.

    Lives as long as this object, like the bare store does without a token.
    """

    def __init__(self, store, gate: _Gate):
        self.store = store
        self.gate = gate
        self.port = gate.port

    def close(self) -> None:
        self.gate.close()

    def __del__(self):
        self.close()


class _Forwarder(_Listener):
    """A node's side of the gate: a port on loopback its store client dials.

    Each connection the client opens is carried to the server, proved, and
    relayed. A server not up yet is dialled again until ``timeout``, because a
    node may well be started before it - the client already came through, and
    it waits on its first read rather than on a refused connect.
    """

    def __init__(self, host: str, port: int, token: bytes, timeout: float):
        self.remote = (host, port)
        self.token = token
        self.timeout = timeout
        super().__init__("127.0.0.1", 0)

    def _dial(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                return socket.create_connection(
                    self.remote, timeout=max(1.0, min(GATE_PATIENCE, deadline - time.monotonic()))
                )
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(DIAL_RETRY)

    def handle(self, connection) -> None:
        outer = None
        try:
            outer = self._dial()
            outer.settimeout(GATE_PATIENCE)
            greeting = _recv(outer, len(GATE_MAGIC) + NONCE_BYTES)
            if greeting[:len(GATE_MAGIC)] != GATE_MAGIC:
                raise OSError("it did not open with a ravex gate")
            mine = secrets.token_bytes(NONCE_BYTES)
            nonces = greeting[len(GATE_MAGIC):] + mine
            outer.sendall(mine + _hello(self.token, nonces))
            if not hmac.compare_digest(_recv(outer, PROOF_CHARS), _welcome(self.token, nonces)):
                raise OSError("it does not hold this job's token")
            outer.settimeout(None)
        except OSError as exc:
            logger.warning(
                "Rendezvous %s:%d refused or did not answer the handshake (%s). "
                "It needs the same %s as this node; one started without it "
                "never speaks first.",
                self.remote[0], self.remote[1], exc, TOKEN_ENV,
            )
            _close(connection)
            if outer is not None:
                _close(outer)
            return
        _pipe(connection, outer)


def serve(host: str = "0.0.0.0", port: int = DEFAULT_PORT, token: Optional[str] = None):
    """Open the server. It lives exactly as long as the returned object.

    With a token - ``token``, or ``RAVEX_JOB_TOKEN`` - the store listens on
    loopback only and ``host:port`` is a :class:`Gated` front: see
    :class:`_Gate`. Without one, the store itself is on ``host:port``.
    """
    secret = job_token(token)
    if secret is None:
        return _tcp_store(host, port, True, 300.0)
    store = _tcp_store("127.0.0.1", 0, True, 300.0)
    return Gated(store, _Gate(host, port, secret, store.port))


def connect(address: str, job: str, timeout: float = 300.0, token: Optional[str] = None):
    """A client of the server at ``address``, confined to ``job``'s keys.

    With a token - ``token``, or ``RAVEX_JOB_TOKEN`` - the client goes through
    a :class:`_Forwarder`, which proves it to the server's gate; the server has
    to have been started with the same one.

    ``timeout`` bounds reaching the server — a node may well be started before
    it — and any blocking read after that. Nothing above this module blocks on
    a read on purpose (it asks ``check`` before ``get``), so in practice it is
    the first.
    """
    import torch.distributed as dist

    if not valid_job(job):
        raise ValueError(
            "job name %r: use letters, digits, '.', '_' and '-', at most 128" % (job,)
        )
    host, port = parse_address(address)
    secret = job_token(token)
    if secret is not None:
        forwarder = _Forwarder(host, port, secret, timeout)
        _FORWARDERS.append(forwarder)
        host, port = "127.0.0.1", forwarder.port
    return dist.PrefixStore(JOB_PREFIX % job, _tcp_store(host, port, False, timeout))


def register(store) -> int:
    """This node's number: the next one, never handed out before."""
    return int(store.add(NEXT_KEY, 1)) - 1


def registered(store) -> int:
    """How many numbers have been handed out, including to nodes long gone."""
    return int(store.add(NEXT_KEY, 0))


def wait_for_base(store, min_nodes: int, deadline: float, poll: float = POLL) -> bool:
    """Wait until the first ``min_nodes`` nodes have registered. False at ``deadline``.

    Only base members wait. They are the ones that start the run, and its seed
    round is theirs: a base member that started training before the others had
    arrived would have nobody to take the starting parameters from, or to give
    them to.
    """
    said = False
    while True:
        count = registered(store)
        if count >= min_nodes:
            return True
        if time.monotonic() >= deadline:
            return False
        if not said:
            logger.info(
                "Waiting at the rendezvous for %d more node(s) to start the run.",
                min_nodes - count,
            )
            said = True
        time.sleep(poll)
