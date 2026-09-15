"""Collecting everyone's round report, with a deadline instead of a collective.

GPU-115, under GPU-113. :mod:`ravex._dist.outer` decides what to do with a list
of contributions; this is what produces the list. It is the piece that makes
"a node dies and the training does not stop" true rather than arranged, because
the list being short is the only thing that happens.

**Pull, not push.** Every node serves its own report and fetches everyone
else's, each fetch on its own socket with its own deadline. A collective — any
collective, on any medium — needs every participant to arrive, and turns one
absent node into a group-wide failure after a timeout measured in the training
group's patience. Here an absent node is one connection that does not answer,
the other fetches are untouched, and the round closes with whoever replied.
That difference is the whole issue.

**What it costs, said plainly.** Each node downloads ``(N-1) × S`` bytes, where
``S`` is one delta. A ring all-reduce moves ``2(N-1)/N × S`` — so at two nodes
these are identical, at four this is twice the traffic, and it grows from
there. The trade is deliberate and it is not permanent: a ring is bandwidth
optimal *and* is exactly the shape that has to be rebuilt when a link in it
goes away. Fault tolerance first on a link where a node vanishing is the normal
case, then the hierarchical topology of GPU-113's point 6, which cuts the
traffic by region rather than by rewriting this.

**Rounds rendezvous here, and that is on purpose.** A node asking for round 5
from a peer still finishing round 4 does not get a refusal, it waits — up to
the deadline it brought with it. Under a wall-clock round the peer is about to
publish, and the alternative to waiting is every node needing to already know
when the others will be ready, which is the thing there is no channel for.

**Reused rather than rebuilt.** The listener address (``RingLink._advertise``,
and the ten seconds a hostname cost on a box with a Hyper-V interface), the
job token (``RingLink._shared_secret``, GPU-112) and its derived
acknowledgement all come from the replication ring. Same question — "is this
one of us" — and answering it twice is how two answers end up differing.

**The bytes are moonclip's, and so is the format.** A report is written as a
moonclip snapshot and moved with the same Rust framing the replication ring
uses (``_core.prestage_send`` / ``prestage_receive``), manifest included. One
directory per round on both sides, so a transfer carries exactly one report and
nothing has to be skipped. Nothing here invents a way to write a tensor down;
:mod:`ravex._dist.report` says why that matters and what it buys.

**A deadline needs SO_RCVTIMEO, not settimeout, and this is not a style
choice.** ``socket.settimeout`` puts the descriptor into non-blocking mode with
the timeout enforced in Python — and the descriptor is then handed to Rust,
where every read comes straight back as "would block". ``RingLink`` knows this
and answers it with ``settimeout(None)``, which is why *the replication
transport has no deadline at all* once bytes start moving. That was tolerable
there. Here the deadline is the entire mechanism, so the socket stays blocking
and the bound is set on the kernel with ``SO_RCVTIMEO``/``SO_SNDTIMEO``:
measured on 2026-09-06, a stalled peer aborts the Rust transfer at 1.50 s
against a 1.5 s bound, where ``settimeout`` aborted it at 0.00 s having
transferred nothing.
"""

from __future__ import annotations

import logging
import os
import socket as _socket
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from ravex._dist import report as _report

logger = logging.getLogger("ravex")

#: Where a node advertises the port it serves its reports on.
ADDRESS_KEY = "ravex/gpu115/exchange/addr/%d"

#: How an operator says what address peers should dial. **Required whenever the
#: nodes are not on one network** — which is the case this whole subsystem
#: exists for. Nothing a process can ask its own kernel returns the address a
#: peer on another continent reaches it at: behind NAT the local interface
#: address is not it, and the hostname is not it either.
ADDRESS_ENV = "RAVEX_EXCHANGE_ADDRESS"

#: Greeting and answer, both fixed length so neither side ever reads a length a
#: stranger chose.
#:
#: The greeting carries a **kind** since GPU-121, because a node now serves two
#: different things and the round number alone cannot say which. Spelling it as
#: its own field rather than stealing a bit of the round number is deliberate:
#: a reserved bit inside an integer is a convention that has to be remembered
#: at four call sites, and the day someone forgets it the request still parses
#: and asks for the wrong tree.
REQUEST = struct.Struct("<8s32sIII")
RESPONSE = struct.Struct("<64sBQ")
REQUEST_MAGIC = b"RVXROUND"

#: What a greeting is asking for.
KIND_ROUND = 0  #: a round's delta report, the ordinary case
KIND_STATE = 1  #: the outer parameters themselves, for a node joining (GPU-121)

#: Response status bytes.
OK = 0
GONE = 1  #: the round asked for is behind the one this node has published

#: How long a served connection may hold a slot waiting for the local round.
#: Bounded so a peer that connects and then stalls cannot pin a handler for the
#: length of the round.
SERVE_PATIENCE = 600.0

#: Concurrent handlers. A node serves one report to each of its peers, so this
#: is a ceiling on the job's node count and not a tuning knob; past it the
#: listen backlog holds the rest, which is the right place for them to wait.
MAX_HANDLERS = 256

#: Bytes moved per socket call.
CHUNK = 1 << 20

#: Round directories a node keeps. A report is worth holding for about as long
#: as a peer might still be fetching it, and retention never deletes one that
#: is on a socket - see `DeltaExchange._serving`. Three, so a slow peer has a
#: round or two of slack before it is told the round is gone.
KEEP_ROUNDS = 3

#: How many outer-state snapshots a node keeps for joiners. Two, not three:
#: a joiner asks for one round by name and the only reason to hold the round
#: before it is a joiner that read the store just as the boundary passed. Past
#: that it is asking for a moment the run has left, and the answer is to read
#: the store again rather than to be handed history.
KEEP_STATE = 2


def close_round(loop, exchange: "DeltaExchange", peers, deadline: float):
    """Publish this node's report, collect the peers', take the outer step.

    The one function a training loop calls, and the seam between the two halves
    of GPU-113: :mod:`ravex._dist.outer` knows what to do with a list of
    contributions and nothing about a network, this knows how to get the list
    and nothing about momentum.

    The local contribution is added here rather than inside ``gather`` — which
    returns peers only — so that "did this node average itself in twice" has
    exactly one place to be answered.

    Returns whatever :meth:`ravex._dist.outer.OuterLoop.apply` returns, plus
    where the round's seconds went. The first half is the honest record of the
    round — how many nodes were in it, and how far apart their step counts
    were; the second half is GPU-117, and it is in the report a real run
    produces rather than only in a bench, because the seconds a round spends on
    the network is the number that decides H and until now the only place it
    was ever printed was a loopback run, where it was 0.0.

    Four spans, and they answer different questions:

    ``delta_seconds``
        subtracting the round's parameters. Compute, not network, and the one
        line here that scales with the model rather than with the link.
    ``publish_seconds``
        writing this node's report to its store, and reading back what the
        peers will see of it (:meth:`DeltaExchange.as_published`, free unless
        ``save_dtype`` is set). ``publish_wait_seconds`` is whatever that span
        was *not* writing the report — near zero since GPU-119 gave each round
        its own directory, and kept as a sentinel for the day something
        serialises a publish against a reader again.
    ``gather_seconds``
        the network, and the rendezvous: fetching every peer's report. Of
        which ``gather_wait_seconds`` is the part before the slowest peer's
        first byte — the dial, the round trip, and that peer still finishing
        the round it is in. The remainder is bytes moving, and a round that
        took minutes is a link to pay for or a peer to stop waiting for
        depending on which of the two it was.
    ``apply_seconds``
        combining and the outer step.
    """
    from ravex._dist.outer import Contribution

    started = time.monotonic()
    mine = loop.contribution()
    expected = _report.expectation(mine.delta)
    subtracted = time.monotonic()
    exchange.publish(mine.delta, loop.round_number, mine.steps)
    # What the peers will average, which under `save_dtype` is not what was
    # just handed over. See `DeltaExchange.as_published`.
    mine.delta = exchange.as_published(mine.delta, loop.round_number, expected)
    published = time.monotonic()
    reports = exchange.gather(peers, loop.round_number, expected, deadline)
    gathered = time.monotonic()
    report = loop.apply(
        [mine]
        + [
            Contribution(delta=r.delta, steps=r.steps, node=r.node)
            for r in reports
        ]
    )
    report.update(
        {
            "delta_seconds": subtracted - started,
            "publish_seconds": published - subtracted,
            "publish_wait_seconds": exchange.publish_wait,
            "gather_seconds": gathered - published,
            "gather_wait_seconds": exchange.gather_wait,
            "apply_seconds": time.monotonic() - gathered,
        }
    )
    return report


#: The round number the starting parameters are published under. Training
#: rounds begin at 1, so this one can never collide with a real round.
SEED_ROUND = 0


def adopt_outer_state(loop, exchange: "DeltaExchange", source: int, rank: int,
                      deadline: float) -> bool:
    """Make every node start from the *same* outer parameters.

    **This is not a nicety, and leaving it out is silently wrong.** An outer
    round applies the same averaged pseudo-gradient on every node, to whatever
    outer parameters that node is holding. So if two nodes start from different
    ones, the difference between them is never touched again by anything: it is
    a constant added to one node's model for the length of the run. The nodes
    look like they are converging — the loss falls, the rounds close over
    everybody — and they are training two models a fixed distance apart.

    That happens for a reason nobody would predict from the outside: the outer
    loop is built at the first optimizer step that has a model to build it
    from, which is *after* that step has moved the weights, and each node moved
    them with its own data. Measured on two ranks at 0.044 per parameter sum,
    constant across every round that followed.

    So one node's parameters are taken as the truth and the others adopt them.
    Not averaged — averaging is what the rounds do, and it would leave the same
    problem one iteration smaller. The same call is what a node joining a run
    already in progress needs, which is why ``source`` is a parameter rather
    than rank 0 written into the body.

    Returns whether this node ended up holding the shared state.
    """
    expected = _report.expectation(loop.outer)
    exchange.publish(loop.outer, SEED_ROUND, 0)
    if rank == source:
        # **The source has to adopt its own state too**, whenever a
        # ``save_dtype`` means the peers will read a cast of it. Otherwise the
        # one node that did not go through the wire holds parameters a
        # quantization apart from everyone else's — from the first instant,
        # never touched again, which is precisely the failure this function
        # exists to prevent, reintroduced by the thing that shrinks the round.
        # Found by the end-to-end test, which is where it would have to be
        # found: nothing raises, and the run trains.
        loop.outer = exchange.as_published(loop.outer, SEED_ROUND, expected)
        loop.write_back()
        return True

    reports = exchange.gather([source], SEED_ROUND, expected, deadline)
    if not reports:
        return False

    loop.outer = reports[0].delta
    loop.write_back()
    return True


def _rust_core():
    """Ravex's own transport, the same one the replication ring is framed by."""
    from ravex import _core

    return _core


def set_deadline(sock, seconds: float) -> None:
    """Bound every read and write on ``sock``, **keeping it blocking**.

    ``settimeout`` cannot be used for this. It makes the descriptor
    non-blocking and enforces the timeout in Python, and the descriptor is then
    handed to the Rust transport, where the first read returns "would block"
    immediately — measured at 0.00 s on 2026-09-06, having moved nothing.
    ``SO_RCVTIMEO``/``SO_SNDTIMEO`` is the kernel's own bound and leaves the
    socket blocking: the same test aborted at 1.50 s against a 1.5 s bound.

    Windows takes a DWORD of milliseconds where everything else takes a
    ``timeval``, which is the whole reason this is a function.
    """
    seconds = max(0.001, float(seconds))
    if sys.platform == "win32":
        value = struct.pack("<I", int(seconds * 1000))
    else:
        value = struct.pack("@qq", int(seconds), int((seconds % 1) * 1_000_000))
    for option in (_socket.SO_RCVTIMEO, _socket.SO_SNDTIMEO):
        sock.setsockopt(_socket.SOL_SOCKET, option, value)


def advertise(port: int, toward: Optional[str] = None) -> str:
    """The address to publish, and **never this host's name**.

    ``RingLink._advertise`` solves the same problem for the replication ring
    and is used here whenever it can answer, including the measurement behind
    it: on a Windows box with a Hyper-V interface, dialling the machine's own
    name took **10.06 s** because the name resolves to several addresses and
    the virtual ones swallow the attempt. Reproduced on 2026-09-06 on a
    different box at **16.1 s**.

    Where this differs is the fallback. ``RingLink`` ends at the hostname,
    reasoning that a name a cluster's DNS can resolve beats nothing. For a
    round with a deadline that reasoning inverts: sixteen seconds spent on an
    address that then refuses the connection is a round lost, and it is lost
    the same way every time. So the order here is

    1. ``RAVEX_EXCHANGE_ADDRESS``, which is the only one that can be right when
       the nodes are on different continents — behind NAT no local lookup
       returns the address a peer dials.
    2. The routing table, asked which local address would be used to reach
       ``toward`` — the ``ravex rendezvous`` server, when there is one — and
       then ``MASTER_ADDR``. Same trick as ``RingLink._advertise``: a UDP
       ``connect`` sends no packet, it only fills in the local address. The
       server comes first because it is the one address known to sit on the
       network the peers share: on a RunPod pair it is on ``podnet1``, where
       the route toward a public address leaves by the bridge interface no
       other pod can reach (GPU-129).
    3. The same question asked toward a public address, for a node that has
       neither.
    4. Loopback, with a warning that says plainly that no other machine will
       reach it.
    """
    configured = os.environ.get(ADDRESS_ENV)
    if configured:
        return configured if ":" in configured else "%s:%d" % (configured, port)

    for target in (toward, os.environ.get("MASTER_ADDR"), "8.8.8.8"):
        if not target:
            continue
        probe = None
        try:
            probe = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            probe.connect((target, 1))
            return "%s:%d" % (probe.getsockname()[0], port)
        except OSError:
            continue
        finally:
            if probe is not None:
                probe.close()

    logger.warning(
        "No address to advertise for the round exchange: falling back to "
        "loopback, which no other machine can reach. Set %s to the address "
        "peers should dial.",
        ADDRESS_ENV,
    )
    return "127.0.0.1:%d" % port


class DeltaExchange:
    """This node's server and its fetches, for the whole run.

    Built once and kept: the listener, the advertised address and the job token
    do not change between rounds, and re-establishing them every round would
    put a rendezvous on the critical path of something already paying for a
    network.
    """

    def __init__(
        self,
        rank: int,
        store,
        root: str,
        node: str = "",
        patience: float = 60.0,
        compression_level: int = 3,
        save_dtype=None,
        route_toward: Optional[str] = None,
    ):
        self.rank = int(rank)
        #: A host on the network the peers share, to pick the address this
        #: node advertises by. See :func:`advertise`.
        self.route_toward = route_toward
        self.node = node or str(rank)
        self.store = store
        self.patience = float(patience)
        self.listener: Optional[Any] = None
        self.secret: Optional[bytes] = None
        self.address: Optional[str] = None

        #: This node's reports and the peers', **one directory per round on
        #: both sides** — see :func:`ravex._dist.report.round_path` for why.
        #: The short version: what is served is then never what is being
        #: written, so no lock has to stand between a publish and a transfer.
        self.root = root
        self.mine_path = os.path.join(root, "mine")
        self.peers_path = os.path.join(root, "peers")
        os.makedirs(self.mine_path, exist_ok=True)
        os.makedirs(self.peers_path, exist_ok=True)

        #: The outer parameters, republished for a node that is joining
        #: (GPU-121). **Its own tree, and that is the point.** A round report
        #: is a delta and a joiner needs the parameters themselves, so the two
        #: are different content under the same round numbers; sharing a tree
        #: would put them in one retention window, where a join could push a
        #: round a peer still needs out of the far end. Nothing here is written
        #: unless somebody is actually joining.
        self.state_path = os.path.join(root, "state")
        self.peer_state_path = os.path.join(root, "peer-state")
        os.makedirs(self.state_path, exist_ok=True)
        os.makedirs(self.peer_state_path, exist_ok=True)
        #: Kept, and not only handed to the store: it decides whether this
        #: node's own contribution has to be read back before it can be
        #: averaged. See :meth:`as_published`.
        self.save_dtype = save_dtype
        self.compression_level = int(compression_level)

        #: The rounds currently going down a socket, by round number. The only
        #: thing that can still touch a directory somebody else is reading is
        #: retention, so retention is the only thing that has to ask — and it
        #: asks with a set lookup instead of a lock held across a transfer.
        self._serving: Dict[int, int] = {}

        #: Seconds the most recent :meth:`publish` spent waiting for anything.
        #:
        #: **It is zero by construction now, and that is the finding it
        #: records.** Until GPU-119 a lock stood between writing this node's
        #: report and serving it down a socket, because one store held every
        #: round and moonclip renames its manifest into place. Measured on a
        #: throttled link (`bench/round_link_cost.py`, 2026-09-07, a 15.8 MB
        #: report, 200 ms round trip), that lock cost:
        #:
        #: ==================== ============= ==============
        #: link                 publish       of which lock
        #: ==================== ============= ==============
        #: both at 7 MB/s       0.13 s        0.00 s
        #: one node throttled   0.83 s        **0.72 s**
        #: after a cut fetch    0.87 s        0.76 s
        #: ==================== ============= ==============
        #:
        #: With both nodes on one link neither can get ahead of the other, so
        #: the lock was never contended and the original reasoning held exactly
        #: there. It failed in the case this system is *for*: a peer whose
        #: uplink is slower, whose fetch is still draining when this node comes
        #: round again — worst round observed 2.34 s, against a transfer of
        #: 0.30 s. One directory per round removed the conflict instead of
        #: serialising it.
        #:
        #: The field stays because a number that is zero *for a reason* is
        #: worth keeping: it rides out in every round report, so the day
        #: something serialises a publish again, a real run says so.
        self.publish_wait = 0.0

        #: Seconds the most recent :meth:`gather` spent before a peer's first
        #: byte of report — the dial, the round trip, and the peer finishing
        #: the round it is still in. The rest of the gather is bytes moving,
        #: and telling the two apart is what says whether a slow round is a
        #: link to pay for or a peer to stop waiting for.
        self.gather_wait = 0.0
        self._waits: Dict[int, float] = {}

        self._round = -1
        #: The newest round each peer has actually taken from this node. Not
        #: bookkeeping: it is what :meth:`close` waits on, and the only thing
        #: that distinguishes "published" from "delivered".
        self._served: Dict[int, int] = {}
        self._lock = threading.Condition()
        self._handlers = threading.Semaphore(MAX_HANDLERS)
        self._stop = threading.Event()
        self._accepting: Optional[threading.Thread] = None

    # -- setup ------------------------------------------------------------

    def start(self) -> bool:
        """Bind, take the token, advertise, and start serving. False if not.

        False rather than an exception because a node that cannot be reached is
        not a broken node: it can still fetch from everyone else, and the
        caller's fallback — fewer contributors, or the collective road — is a
        decision about the round rather than about this object.
        """
        from ravex._dist.replication import RingLink

        try:
            deadline = time.monotonic() + self.patience
            self.secret = RingLink._shared_secret(self.store, self.rank, deadline)
            if self.secret is None:
                logger.warning(
                    "No job token on the rendezvous store after %.0fs; this "
                    "node cannot prove it belongs and will not serve.",
                    self.patience,
                )
                return False

            listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            listener.bind(("", 0))
            listener.listen(MAX_HANDLERS)
            listener.settimeout(0.5)
            self.listener = listener

            self.address = advertise(listener.getsockname()[1], self.route_toward)
            self.store.set(ADDRESS_KEY % self.rank, self.address.encode("utf-8"))
        except OSError as exc:
            logger.warning("Could not open the round exchange: %s", exc)
            self.close()
            return False

        self._accepting = threading.Thread(target=self._accept, daemon=True)
        self._accepting.start()
        return True

    def close(self, linger: float = 0.0, expect=None) -> None:
        """Stop serving. ``linger`` first, so a last report is not lost with it.

        **Leaving is not free, and this was measured rather than reasoned
        about.** A node that finishes its final round, applies its own outer
        step and shuts the listener down takes its last report with it: a peer
        still fetching that round gets nothing, closes the round with one fewer
        contributor, and the two models end the run *different*. That is
        exactly what the end-to-end test showed the first time it ran — rank 0
        exited, rank 1 logged "round 3 closed without rank 0", and the
        parameters diverged in the fifth decimal.

        So ``expect`` names the peers whose fetch this node waits for, and
        ``linger`` bounds the wait. Best effort on purpose: a peer that has
        itself died will never fetch, and outliving it is not a service to
        anybody. What the bound buys is that the ordinary case — everyone
        finishing within a round of each other — never loses a report, and the
        pathological one costs a known number of seconds.
        """
        if linger > 0 and expect:
            deadline = time.monotonic() + linger
            with self._lock:
                while time.monotonic() < deadline:
                    if all(self._served.get(peer, -1) >= self._round for peer in expect):
                        break
                    self._lock.wait(min(0.2, max(0.01, deadline - time.monotonic())))
            waiting = [p for p in expect if self._served.get(p, -1) < self._round]
            if waiting:
                logger.info(
                    "Leaving with round %d untaken by rank(s) %s after %.0fs.",
                    self._round,
                    ", ".join(str(peer) for peer in waiting),
                    linger,
                )

        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        listener, self.listener = self.listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    # -- publishing -------------------------------------------------------

    def publish(self, delta, round_number: int, steps: int) -> None:
        """Write this node's report and offer it to whoever asks.

        The round number is the snapshot's step, so a peer asks for a round by
        name rather than trusting whatever is newest — which matters because by
        the time this node answers it may well be a round further on.

        A few rounds are kept rather than only the current one, so that serving
        a peer can never race retention deleting the thing being served. Past
        that a peer is asking for a round it can no longer join, and the answer
        is to catch up from the current outer state rather than to be handed
        history one round at a time.

        **Nothing here waits on a reader** (GPU-119). The round goes into its
        own directory, so a peer draining an earlier one is reading files this
        call never touches; retention runs after, and skips whatever is on a
        socket right now.
        """
        entered = time.monotonic()
        began = time.monotonic()
        _report.publish_round(
            self.mine_path,
            round_number,
            delta,
            steps,
            self.node,
            compression_level=self.compression_level,
            save_dtype=self.save_dtype,
        )
        wrote = time.monotonic() - began

        with self._lock:
            self._round = max(self._round, int(round_number))
            self._lock.notify_all()
            protect = tuple(self._serving)
        _report.drop_rounds(self.mine_path, KEEP_ROUNDS, protect=protect)

        # Everything this call spent *not* writing the report. That is the
        # honest form of the old lock measurement: it is near zero because
        # nothing here blocks on a reader any more, and it would grow again the
        # day something did. A hardcoded zero could not say that.
        self.publish_wait = max(0.0, (time.monotonic() - entered) - wrote)

    def as_published(self, delta, round_number: int, expected):
        """This node's delta **as its peers will read it**, which is not always
        the one it wrote.

        With ``save_dtype`` set, the report on the wire is a cast of this
        node's delta and the peers combine *that*. If this node combined its
        own uncast one instead, every node would take a slightly different
        outer step from the same round — and that is not a rounding detail, it
        is the failure :func:`adopt_outer_state` exists to prevent, arriving by
        another door. The nodes' parameters separate by a quantization error
        per round, nothing ever brings them back together, and from the outside
        the run looks healthy: the loss falls and every round closes over
        everybody.

        Without ``save_dtype`` it is the identity and costs nothing, which is
        why the read is behind the check rather than done unconditionally.
        The read is of this node's own disk, not a network, and it is the only
        way to see exactly what the peers will see — the per-tensor scales of
        the float8 path are moonclip's, and reimplementing them here to guess
        at the answer would be a second way to quantize a tensor.

        **And with it, it costs nothing measurable either.**
        ``bench/round_link_cost.py`` at ``--save-dtype bf16``, 2026-09-07, put
        the publish at 0.34 s against 0.13 s uncast — and 0.24 s of that
        difference was the publish lock, which GPU-119 has since removed, not
        this read. The round it is part of spent 2.6 s on the network.
        """
        if not self.save_dtype:
            return delta
        try:
            return _report.read_round(self.mine_path, round_number, expected).delta
        except Exception as exc:
            logger.warning(
                "Could not read back this node's own round %d report (%s), so "
                "it is averaged uncast while the peers average the cast one. "
                "This node's parameters will part from theirs by the "
                "quantization error of one round.",
                round_number,
                exc,
            )
            return delta

    # -- fetching ---------------------------------------------------------

    def gather(
        self,
        peers: List[int],
        round_number: int,
        expected,
        deadline: float,
    ) -> List[_report.Report]:
        """Everyone else's report for this round, or as many as arrive in time.

        ``deadline`` is absolute (``time.monotonic``). What comes back is
        whoever answered — never padded, never waited past. The caller adds its
        own contribution; this returns peers only, so that a node cannot
        average itself in twice by a mistake made in one place.

        Leaves :attr:`gather_wait` behind: of the seconds this took, how many
        went on the *slowest peer's* rendezvous rather than on bytes. The two
        are different problems with the same symptom — a round that took
        minutes is a link to pay for or a peer to wait less for, and which one
        it is cannot be read off a total (GPU-117).
        """
        if not peers:
            self.gather_wait = 0.0
            return []

        answered: Dict[int, _report.Report] = {}
        self._waits = {}
        found = threading.Lock()
        threads = []
        for peer in peers:
            thread = threading.Thread(
                target=self._fetch_into,
                args=(peer, round_number, expected, deadline, answered, found),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)

        # The slowest peer's, not the sum: the fetches run at once, so what the
        # round waited for is the last peer to arrive.
        self.gather_wait = max(self._waits.values()) if self._waits else 0.0

        # Keyed by peer rather than by the name in the payload: who answered is
        # a fact this node knows, and a report's own ``node`` field is written
        # by the peer. Trusting the latter to identify the former is how one
        # machine gets counted twice.
        silent = [peer for peer in peers if peer not in answered]
        if silent:
            logger.warning(
                "Round %d closed without rank(s) %s. It goes on with %d of %d "
                "peer(s) - a node that did not answer did not contribute, "
                "which is an answer and not a failure.",
                round_number,
                ", ".join(str(peer) for peer in silent),
                len(answered),
                len(peers),
            )
        return [answered[peer] for peer in peers if peer in answered]

    def _fetch_into(self, peer, round_number, expected, deadline, answered, found):
        try:
            report = self.fetch(peer, round_number, expected, deadline)
        except Exception as exc:  # one peer's failure is not the round's
            logger.info("No report from rank %s this round: %s", peer, exc)
            return
        if report is None:
            return
        with found:
            answered[peer] = report

    def fetch(self, peer: int, round_number: int, expected, deadline: float,
              kind: int = KIND_ROUND):
        """One peer's report, or None if it did not arrive before ``deadline``.

        ``kind`` picks which of the two things this peer serves is being asked
        for: its delta for a round (:data:`KIND_ROUND`), or the outer
        parameters a joining node needs (:data:`KIND_STATE`, GPU-121). The two
        land in different trees on this side as well, so a join in flight can
        never evict a round a peer is still going to be asked for.
        """
        from ravex._dist.replication import RingLink

        address = self._address_of(peer, deadline)
        if address is None:
            return None

        host, _, port = address.rpartition(":")
        connection = None
        dialled = time.monotonic()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            connection = _socket.create_connection(
                (host, int(port)), timeout=min(remaining, 30.0)
            )
            connection.sendall(
                REQUEST.pack(
                    REQUEST_MAGIC,
                    self.secret or b"",
                    self.rank,
                    int(round_number) & 0xFFFFFFFF,
                    int(kind),
                )
            )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            connection.settimeout(remaining)
            head = _recv_exactly(connection, RESPONSE.size, deadline)
            # The peer answers the greeting only once it holds the round asked
            # for, so everything up to here is the dial, the round trip and the
            # peer's own round - and everything after it is bytes. See
            # `gather_wait`.
            self._waits[peer] = time.monotonic() - dialled
            acknowledgement, status, length = RESPONSE.unpack(head)

            expect = RingLink._acknowledgement(self.secret, self.rank)
            if acknowledgement != expect:
                logger.warning(
                    "Rank %s answered without the job token. Not reading a "
                    "delta from it.",
                    peer,
                )
                return None
            if status != OK:
                logger.info(
                    "Rank %s no longer holds round %d and cannot serve it.",
                    peer,
                    round_number,
                )
                return None

            # From here the descriptor belongs to the Rust transport, so the
            # bound has to be one the kernel enforces on a blocking socket -
            # see `set_deadline`. The socket-level `timeout` set by
            # `create_connection` is cleared for the same reason.
            connection.settimeout(None)
            set_deadline(connection, max(1.0, deadline - time.monotonic()))
            destination = (
                self._peer_state_round_path(peer, round_number)
                if kind == KIND_STATE
                else self._peer_round_path(peer, round_number)
            )
            _core = _rust_core()
            if not _core.prestage_receive(connection.fileno(), destination):
                return None
        except (OSError, _socket.timeout):
            return None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

        try:
            # Opened fresh: a manager reads its manifest when it is built, and
            # the manifest is exactly what just changed on disk.
            #
            # This is also the one place moonclip's recovery scan is load
            # bearing rather than a safety net. Snapshots appear in this
            # directory without *this* process having written them — the
            # transport put them there — which from the store's point of view
            # is indistinguishable from a run killed after writing and before
            # recording. It says so on stderr, and it is right to: the files
            # are complete and it adopts them, which is what makes a
            # prestaged store readable at all.
            theirs = _report.open_store(destination)
            report = _report.read(theirs, round_number, expected)
            if kind == KIND_STATE:
                _report.drop_rounds(self._peer_state_path(peer), KEEP_STATE)
            else:
                _report.drop_rounds(self._peer_path(peer), KEEP_ROUNDS)
        except _report.ReportError as exc:
            # Not "absent": absent is silence. This peer said something this
            # node will not act on, and that is worth a louder line, because it
            # is a version skew or a misconfiguration rather than a machine
            # that went away.
            logger.warning("Unusable report from rank %s: %s", peer, exc)
            return None
        except Exception as exc:
            logger.warning("Could not read rank %s's report: %s", peer, exc)
            return None
        return report

    def _hold(self, round_number: int) -> None:
        """Say a round is on a socket, so retention leaves it alone."""
        with self._lock:
            self._serving[round_number] = self._serving.get(round_number, 0) + 1

    def _release(self, round_number: int) -> None:
        with self._lock:
            left = self._serving.get(round_number, 0) - 1
            if left > 0:
                self._serving[round_number] = left
            else:
                self._serving.pop(round_number, None)

    def _peer_path(self, peer: int) -> str:
        path = os.path.join(self.peers_path, str(peer))
        os.makedirs(path, exist_ok=True)
        return path

    def _peer_state_path(self, peer: int) -> str:
        path = os.path.join(self.peer_state_path, str(peer))
        os.makedirs(path, exist_ok=True)
        return path

    def _peer_state_round_path(self, peer: int, round_number: int) -> str:
        path = _report.round_path(self._peer_state_path(peer), round_number)
        os.makedirs(path, exist_ok=True)
        return path

    def _peer_round_path(self, peer: int, round_number: int) -> str:
        """Where one peer's one round lands.

        Per round on this side too, and not only for symmetry: the sender now
        offers a directory holding a single round, so a destination carrying
        several would have the transport reconciling two different shapes. It
        also means a fetch that was cut in half leaves its own directory behind
        instead of a partial snapshot inside the one the next round needs -
        which is what made a starved round cost a resend of everything.
        """
        path = _report.round_path(self._peer_path(peer), round_number)
        os.makedirs(path, exist_ok=True)
        return path

    def _address_of(self, peer: int, deadline: float) -> Optional[str]:
        """Poll for a peer's advertisement. ``check``, never ``get``.

        ``get`` blocks to the store's own timeout rather than to this round's,
        which is how a deadline stops being a deadline.
        """
        key = ADDRESS_KEY % peer
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                if self.store.check([key]):
                    return self.store.get(key).decode("utf-8")
            except Exception:
                return None
            time.sleep(0.05)
        return None

    # -- serving ----------------------------------------------------------

    def _accept(self) -> None:
        while not self._stop.is_set():
            listener = self.listener
            if listener is None:
                return
            try:
                connection, _ = listener.accept()
            except (_socket.timeout, OSError):
                continue
            if not self._handlers.acquire(blocking=False):
                # Past MAX_HANDLERS the backlog is the right queue to wait in.
                try:
                    connection.close()
                except OSError:
                    pass
                continue
            threading.Thread(
                target=self._serve, args=(connection,), daemon=True
            ).start()

    def _serve(self, connection) -> None:
        from ravex._dist.replication import RingLink

        try:
            connection.settimeout(SERVE_PATIENCE)
            greeting = _recv_exactly(
                connection, REQUEST.size, time.monotonic() + SERVE_PATIENCE
            )
            magic, token, peer, wanted, kind = REQUEST.unpack(greeting)
            if magic != REQUEST_MAGIC or token != (self.secret or b""):
                # Silence rather than a message. Something that reached the
                # port without the token is not owed an explanation, and a
                # refusal that says which half was wrong is a hint.
                logger.warning("Refused a connection without the job token.")
                return

            if kind == KIND_STATE:
                self._serve_state(connection, peer, wanted)
                return

            if not self._await_round(wanted):
                connection.sendall(
                    RESPONSE.pack(RingLink._acknowledgement(self.secret, peer), GONE, 0)
                )
                return

            # Claimed *before* the OK goes out, so there is no window where the
            # peer has been promised a round that retention could then delete.
            # `_await_round` came first, so the directory is complete and this
            # only has to stop it from disappearing.
            self._hold(wanted)
            try:
                connection.sendall(
                    RESPONSE.pack(RingLink._acknowledgement(self.secret, peer), OK, 0)
                )
                # The descriptor goes to Rust from here, so the bound has to be
                # the kernel's - see `set_deadline`.
                connection.settimeout(None)
                set_deadline(connection, SERVE_PATIENCE)
                # No lock (GPU-119): `wanted` is a directory nothing writes to
                # any more, and the hold above is what keeps retention off it.
                _rust_core().prestage_send(
                    connection.fileno(),
                    _report.round_path(self.mine_path, wanted),
                    CHUNK,
                )
            finally:
                self._release(wanted)

            # Recorded only after the last byte is out. "Published" and
            # "delivered" are different facts, and `close` waits on this one.
            with self._lock:
                if wanted > self._served.get(peer, -1):
                    self._served[peer] = wanted
                self._lock.notify_all()
        except (OSError, _socket.timeout, struct.error):
            pass
        finally:
            self._handlers.release()
            try:
                connection.close()
            except OSError:
                pass

    def _serve_state(self, connection, peer: int, wanted: int) -> None:
        """Hand a joining node the outer parameters for round ``wanted``.

        Deliberately *not* a wait. A round fetch waits because the peer asking
        is a member whose round is coming; a state fetch is asked by a node
        that is not in the run yet, and the honest answer to "I do not have
        that round's state written down" is ``GONE`` rather than holding a
        handler for ten minutes on behalf of a stranger. The joiner reads the
        round it should ask for off the store and comes back.
        """
        from ravex._dist.replication import RingLink

        acknowledgement = RingLink._acknowledgement(self.secret, peer)
        if not _report.round_is_complete(self.state_path, wanted):
            connection.sendall(RESPONSE.pack(acknowledgement, GONE, 0))
            return

        connection.sendall(RESPONSE.pack(acknowledgement, OK, 0))
        connection.settimeout(None)
        set_deadline(connection, SERVE_PATIENCE)
        _rust_core().prestage_send(
            connection.fileno(),
            _report.round_path(self.state_path, wanted),
            CHUNK,
        )

    def publish_state(self, payload, round_number: int) -> None:
        """Write what a joining node needs, as of the start of ``round_number``.

        ``payload`` is built by :func:`ravex._dist.membership.state_payload`,
        and it is **not** only the outer parameters: the outer optimizer's
        momentum travels with them, because a node that arrives without it
        applies a different update to the same gradient from its first round.

        Called at a round boundary by every member while a join is pending, so
        a joiner can take the state from whichever of them answers first —
        which is what spares this protocol from having to elect a source, and
        therefore from having to notice that rank 0 is dead.

        ``steps`` is ``0`` on the wire: what is written here is not a round's
        work, it is where the run has got to, and a number that looked like
        step count would be averaged by anything that mistook this for a
        report.
        """
        _report.publish_round(
            self.state_path,
            round_number,
            payload,
            0,
            self.node,
            compression_level=self.compression_level,
            save_dtype=None,  # never a cast: a joiner adopting a quantized
                              # copy is a node a rounding apart from the rest,
                              # for the length of the run. See adopt_outer_state.
        )
        _report.drop_rounds(self.state_path, KEEP_STATE)

    def fetch_state(self, peers: List[int], round_number: int, expected,
                    deadline: float):
        """The outer parameters for ``round_number`` from the first peer that has them.

        Tried in order and one at a time, not in parallel: this runs once per
        join and what it wants is *a* copy, so the second request is worth
        making only when the first did not produce one. Every member publishes
        the same state at the same boundary, so "the first that answers" needs
        no tie-break — and a member that is gone is a connection that fails,
        which is the next name in the list rather than a special case.
        """
        for peer in peers:
            if time.monotonic() >= deadline:
                break
            try:
                report = self.fetch(
                    peer, round_number, expected, deadline, kind=KIND_STATE
                )
            except Exception as exc:
                logger.info("No outer state from rank %s: %s", peer, exc)
                continue
            if report is not None:
                return report
        return None

    def _await_round(self, wanted: int) -> bool:
        """Whether this node can serve ``wanted``, waiting for it if it is coming.

        A peer ahead of us waits; a peer asking for a round retention has
        already dropped gets ``GONE``. Which of the two it is cannot be decided
        by the asker, because the asker is the one without the information — so
        it is decided here, where the published round is known.

        Note what is *not* checked: whether ``wanted`` is the newest round.
        The whole store goes down the socket and the reader looks up the round
        it asked for by number, so a node that has moved on can still answer
        for a round it still holds. That is what makes a slightly slow peer a
        contributor rather than a casualty.
        """
        deadline = time.monotonic() + SERVE_PATIENCE
        while not self._stop.is_set():
            if _report.round_is_complete(self.mine_path, wanted):
                return True
            with self._lock:
                if wanted < self._round:
                    return False  # retention has dropped it
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lock.wait(min(remaining, 1.0))
        return False


def _recv_exactly(connection, count: int, deadline: float) -> bytes:
    """``count`` bytes or an error. Never a short read treated as a message.

    The deadline is rechecked per chunk rather than only set on the socket:
    a peer trickling one byte per timeout would otherwise hold a fetch open
    long past the round it belongs to.
    """
    if count == 0:
        return b""
    buffer = bytearray()
    while len(buffer) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OSError("deadline passed after %d of %d bytes" % (len(buffer), count))
        connection.settimeout(min(remaining, 30.0))
        block = connection.recv(min(CHUNK, count - len(buffer)))
        if not block:
            raise OSError("peer closed after %d of %d bytes" % (len(buffer), count))
        buffer += block
    return bytes(buffer)
