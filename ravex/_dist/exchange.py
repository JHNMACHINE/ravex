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
import hashlib
import json
import os
import socket as _socket
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ravex._dist import report as _report
from ravex._dist.membership import MembershipError

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
KIND_RELAY = 2  #: another node's report for a round, held by this one (GPU-140)

#: What follows a :data:`KIND_RELAY` greeting: whose report is wanted. Sent
#: after the greeting rather than folded into it, so the greeting keeps its
#: size and a request of either older kind reads exactly as it always did.
OWNER = struct.Struct("<I")

#: Where a round's contributors are decided (GPU-140): the job token's digest,
#: then the round. The digest scopes the key to one incarnation of the job -
#: rank 0 makes a new token every time it starts - so a job that restarts on a
#: store that outlived it does not find the sets its previous life decided.
ROUND_SET_KEY = "ravex/gpu140/set/%s/%d"

#: A node saying it has applied a round: job digest, round, rank. What
#: :meth:`DeltaExchange.close` waits on before it stops serving, because until
#: every member of the round has applied it, one of them may still need a
#: report relayed from this node.
APPLIED_KEY = "ravex/gpu140/applied/%s/%d/%d"

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

#: The longest a single ``recv`` waits before the deadline is looked at again.
#: A slice, not a limit: running out of one means "check the deadline and read
#: again", never "the peer is gone". See `_recv_exactly`.
RECV_SLICE = 30.0

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


class RoundSplitError(MembershipError):
    """This node cannot apply the round everyone else is applying (GPU-140).

    A :class:`MembershipError`, so the runtime stops instead of abandoning the
    round: abandoning means *not* taking the outer step the others took, which
    is the two-model outcome this exists to prevent. A node that stops is a
    departure, and a departure is safe; relaunched, it comes back as a joiner
    and takes the run's current state (GPU-121).
    """


def _job_digest(secret: Optional[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(secret or b"")
    digest.update(b"ravex-round-set")
    return digest.hexdigest()[:16]


def decide_round(store, secret, round_number: int, rank: int, collected,
                 deadline: float) -> Tuple[List[int], int]:
    """The round's contributors, as the store decided them, and who proposed.

    **Why a decision and not an observation.** Every node used to close the
    round over the reports *it* managed to fetch. That is only one model while
    every node fetches the same set, and it takes very little for them not to:
    a node killed between serving one peer and the next, or a link slow in one
    direction, and one node averages {A, B} while the other averages {B}. From
    then on they apply the same pseudo-gradients to different parameters, both
    losses fall, and nothing raises.

    So the first node to finish gathering proposes what it holds, with
    ``compare_set`` against an empty key: the store applies it atomically, the
    first proposal is the one that stays, and every later caller gets it back
    instead of writing its own. Every node then applies exactly that set. The
    proposer holds every report in it, which is what makes the set something
    the others can always recover - see :meth:`DeltaExchange.recover`.

    Retried on a store error until ``deadline``, then :class:`RoundSplitError`:
    a node that cannot learn the decision cannot know whether the others took
    the step, and guessing either way is the failure this closes.
    """
    key = ROUND_SET_KEY % (_job_digest(secret), round_number)
    proposal = json.dumps({"by": int(rank), "set": sorted(int(r) for r in collected)})
    while True:
        try:
            raw = store.compare_set(key, "", proposal)
            break
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RoundSplitError(
                    "could not read round %d's decision off the rendezvous "
                    "store (%s); stopping rather than guessing whether the "
                    "other nodes took the step" % (round_number, exc)
                ) from exc
            time.sleep(0.2)
    try:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        decided = json.loads(text)
        return [int(r) for r in decided["set"]], int(decided["by"])
    except Exception as exc:
        raise RoundSplitError(
            "round %d's decision on the store is unreadable (%r)" % (round_number, raw)
        ) from exc


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

    **The set that is averaged is decided, not observed (GPU-140).** After the
    gather, :func:`decide_round` fixes the round's contributors on the store,
    and this node applies exactly those: reports it holds beyond them are left
    out, reports it is missing are recovered from a node that has them
    (:meth:`DeltaExchange.recover`), and if one cannot be recovered it raises
    :class:`RoundSplitError` rather than apply a different average. Two more
    spans come with it: ``decide_seconds`` and ``recover_seconds``, the second
    zero unless this node's gather came up short of the decision.
    """
    from ravex._dist.outer import Contribution

    round_number = loop.round_number
    started = time.monotonic()
    # A round's whole budget, reused for the steps after the gather: the
    # decision and a recovery are each allowed as long as the gather was.
    budget = max(deadline - started, 1.0)
    mine = loop.contribution()
    expected = _report.expectation(mine.delta)
    subtracted = time.monotonic()
    offered = True
    try:
        exchange.publish(mine.delta, round_number, mine.steps)
        # What the peers will average, which under `save_dtype` is not what
        # was just handed over. See `DeltaExchange.as_published`.
        mine.delta = exchange.as_published(mine.delta, round_number, expected)
    except Exception as exc:
        # A full disk here used to fail the round on this node alone, which
        # then skipped the outer step its peers took. Now it only means this
        # node proposes itself to nobody: nobody can have fetched a report
        # that was never offered, so the decided set will not name it, and
        # this node still applies the others'.
        logger.warning("Could not publish round %d: %s", round_number, exc)
        offered = False
    published = time.monotonic()
    received = exchange.gather_by_rank(peers, round_number, expected, deadline)
    gathered = time.monotonic()

    collected = set(received) | ({exchange.rank} if offered else set())
    members, decided_by = decide_round(
        exchange.store, exchange.secret, round_number, exchange.rank, collected,
        time.monotonic() + budget,
    )
    decided = time.monotonic()

    if exchange.rank in members and not offered:
        raise RoundSplitError(
            "round %d was decided with this node's report in it, and this node "
            "never finished offering one" % round_number
        )
    missing = [r for r in members if r != exchange.rank and r not in received]
    recovery_deadline = time.monotonic() + budget
    for owner in missing:
        holders = [decided_by] + [r for r in members if r != decided_by]
        recovered = exchange.recover(
            owner, round_number, expected, holders, recovery_deadline
        )
        if recovered is None:
            raise RoundSplitError(
                "round %d was decided over ranks %s and this node could not "
                "get rank %d's report from anyone that holds it; stopping "
                "rather than averaging a different set. Relaunched, it rejoins "
                "the run from the current state."
                % (round_number, members, owner)
            )
        received[owner] = recovered
    recovered_at = time.monotonic()

    dropped = sorted(set(received) - set(members))
    if dropped:
        logger.info(
            "Round %d: leaving out rank(s) %s, which this node reached but the "
            "round was decided without.",
            round_number,
            ", ".join(str(r) for r in dropped),
        )

    contributions = ([mine] if exchange.rank in members else []) + [
        Contribution(delta=received[r].delta, steps=received[r].steps, node=received[r].node)
        for r in members
        if r != exchange.rank
    ]
    exchange.decided(round_number, members)
    if contributions:
        report = loop.apply(contributions)
        exchange.applied(round_number)
    else:
        # Decided empty: the proposer held nothing it could vouch for. Every
        # node reads the same empty set, so every node skips the same step,
        # which is the one case where not applying keeps them together.
        loop.abandon_round()
        report = {"round": round_number, "nodes": 0, "steps": []}
    report.update(
        {
            "decided_by": decided_by,
            "recovered": len(missing),
            "delta_seconds": subtracted - started,
            "publish_seconds": published - subtracted,
            "publish_wait_seconds": exchange.publish_wait,
            "gather_seconds": gathered - published,
            "gather_wait_seconds": exchange.gather_wait,
            "decide_seconds": decided - gathered,
            "recover_seconds": recovered_at - decided,
            "apply_seconds": time.monotonic() - recovered_at,
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
        #: The same, for other nodes' reports going out as relays (GPU-140),
        #: by (owner, round).
        self._relaying: Dict[Tuple[int, int], int] = {}
        #: The newest round this node has seen decided, and over whom. Who is
        #: still owed a relay when this node leaves - see :meth:`close`.
        self._last_decided: Optional[Tuple[int, List[int]]] = None

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

        if linger > 0 and self._last_decided is not None:
            self._linger_for_relays(time.monotonic() + linger)

        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        listener, self.listener = self.listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def decided(self, round_number: int, members: List[int]) -> None:
        """Remember the newest decided round, for :meth:`close`."""
        self._last_decided = (int(round_number), [int(m) for m in members])

    def applied(self, round_number: int) -> None:
        """Say on the store that this node has applied ``round_number``.

        Best effort: it only shortens how long a peer lingers on its way out.
        """
        try:
            self.store.set(
                APPLIED_KEY % (_job_digest(self.secret), int(round_number), self.rank),
                b"1",
            )
        except Exception as exc:
            logger.debug("Could not record round %d as applied: %s", round_number, exc)

    def _linger_for_relays(self, deadline: float) -> None:
        """Keep serving until every member of the last round has applied it.

        GPU-140. A member that did not reach some node gets that node's report
        relayed by one that did, and the ones that did include this node. The
        original linger waits only for this node's *own* report to be taken,
        so a node leaving after its last round could close the listener on a
        peer still asking it for a relay - which is exactly what the test for
        the relay did the first time it ran, on the run's last round.
        """
        round_number, members = self._last_decided
        digest = _job_digest(self.secret)
        waiting = [m for m in members if m != self.rank]
        while waiting and time.monotonic() < deadline:
            try:
                waiting = [
                    m for m in waiting
                    if not self.store.check([APPLIED_KEY % (digest, round_number, m)])
                ]
            except Exception:
                return
            if waiting:
                time.sleep(0.1)
        if waiting:
            logger.info(
                "Leaving before rank(s) %s applied round %d.",
                ", ".join(str(m) for m in waiting),
                round_number,
            )

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
        answered = self.gather_by_rank(peers, round_number, expected, deadline)
        return [answered[peer] for peer in peers if peer in answered]

    def gather_by_rank(
        self,
        peers: List[int],
        round_number: int,
        expected,
        deadline: float,
    ) -> Dict[int, _report.Report]:
        """:meth:`gather`, keyed by the rank each report was fetched from.

        What :func:`close_round` needs to hold a report against the round's
        decided set, which names ranks.
        """
        if not peers:
            self.gather_wait = 0.0
            return {}

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
        with found:
            answered = dict(answered)
        silent = [peer for peer in peers if peer not in answered]
        if silent:
            logger.warning(
                "Round %d: no report from rank(s) %s before the deadline; %d of "
                "%d peer(s) answered. A node that did not answer did not "
                "contribute, which is an answer and not a failure - once every "
                "node agrees on it, which the round's decision sees to.",
                round_number,
                ", ".join(str(peer) for peer in silent),
                len(answered),
                len(peers),
            )
        return answered

    def recover(self, owner: int, round_number: int, expected, holders,
                deadline: float) -> Optional[_report.Report]:
        """``owner``'s report for a round, from whichever holder still has it.

        For a node the round was decided with and this node did not reach
        (GPU-140). ``holders`` is asked in order and the first is the proposer,
        which held every report in the set when it proposed it: the owner
        itself may be the one that died, or the one whose link to this node was
        the problem in the first place. The owner is asked for its own round
        directly, everyone else for a relay. The deadline is split evenly, so a
        holder that died without a reset cannot take the whole of it.
        """
        asked = [h for h in holders if h != self.rank]
        if owner not in asked:
            asked.append(owner)
        # The gather's own fetch may have left half a directory behind, and a
        # second transfer into it would be reconciling two shapes.
        partial = _report.round_path(self._peer_path(owner), round_number)
        if os.path.isdir(partial) and not _report.round_is_complete(
            self._peer_path(owner), round_number
        ):
            import shutil

            shutil.rmtree(partial, ignore_errors=True)
        for index, holder in enumerate(asked):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            until = time.monotonic() + left / (len(asked) - index)
            try:
                if holder == owner:
                    report = self.fetch(owner, round_number, expected, until)
                else:
                    report = self.fetch(
                        holder, round_number, expected, until,
                        kind=KIND_RELAY, owner=owner,
                    )
            except Exception as exc:
                logger.info(
                    "Rank %s could not hand over rank %s's round %d: %s",
                    holder, owner, round_number, exc,
                )
                continue
            if report is not None:
                logger.info(
                    "Round %d: recovered rank %s's report from rank %s.",
                    round_number, owner, holder,
                )
                return report
        return None

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
              kind: int = KIND_ROUND, owner: Optional[int] = None):
        """One peer's report, or None if it did not arrive before ``deadline``.

        ``kind`` picks which of the two things this peer serves is being asked
        for: its delta for a round (:data:`KIND_ROUND`), or the outer
        parameters a joining node needs (:data:`KIND_STATE`, GPU-121). The two
        land in different trees on this side as well, so a join in flight can
        never evict a round a peer is still going to be asked for.

        :data:`KIND_RELAY` asks ``peer`` for ``owner``'s report rather than its
        own (GPU-140), and it lands where a direct fetch from ``owner`` would
        have: to everything after this, a relayed report is that node's report.
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
            if kind == KIND_RELAY:
                connection.sendall(OWNER.pack(int(owner)))

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
            source = owner if kind == KIND_RELAY else peer
            destination = (
                self._peer_state_round_path(peer, round_number)
                if kind == KIND_STATE
                else self._peer_round_path(source, round_number)
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
                # Marked servable here as well, once it has been read whole:
                # a report this node holds is one it can relay (GPU-140), and
                # "complete" has to mean the same thing on both sides.
                _report.mark_complete(destination)
                _report.drop_rounds(
                    self._peer_path(source), KEEP_ROUNDS,
                    protect=self._relaying_rounds(source),
                )
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
            if kind == KIND_RELAY:
                owner = OWNER.unpack(
                    _recv_exactly(connection, OWNER.size,
                                  time.monotonic() + SERVE_PATIENCE)
                )[0]
                self._serve_relay(connection, peer, owner, wanted)
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

    def _serve_relay(self, connection, peer: int, owner: int, wanted: int) -> None:
        """Hand ``peer`` the report ``owner`` sent this node for round ``wanted``.

        GPU-140: the round was decided over a set that includes ``owner``, and
        ``peer`` did not get ``owner``'s report itself. Not a wait, for the
        reason :meth:`_serve_state` is not one: either this node holds the
        report whole or it does not, and ``GONE`` lets the asker try the next
        holder instead of holding a handler on a report that is not coming.
        """
        from ravex._dist.replication import RingLink

        acknowledgement = RingLink._acknowledgement(self.secret, peer)
        root = self.mine_path if owner == self.rank else self._peer_path(owner)
        key = (owner, wanted)
        with self._lock:
            self._relaying[key] = self._relaying.get(key, 0) + 1
        try:
            if not _report.round_is_complete(root, wanted):
                connection.sendall(RESPONSE.pack(acknowledgement, GONE, 0))
                return
            connection.sendall(RESPONSE.pack(acknowledgement, OK, 0))
            connection.settimeout(None)
            set_deadline(connection, SERVE_PATIENCE)
            _rust_core().prestage_send(
                connection.fileno(), _report.round_path(root, wanted), CHUNK
            )
        finally:
            with self._lock:
                left = self._relaying.get(key, 0) - 1
                if left > 0:
                    self._relaying[key] = left
                else:
                    self._relaying.pop(key, None)

    def _relaying_rounds(self, owner: int):
        """Rounds of ``owner``'s this node is relaying right now, for retention."""
        with self._lock:
            return tuple(r for (o, r) in self._relaying if o == owner)

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

    **A slice running out is not the peer being absent** (2026-09-15, EU and
    US on two RunPod boxes). The greeting's answer is held back until the peer
    has the round, so this call is where a node ahead waits for one behind —
    up to the round's deadline. The slice used to be a timeout that raised,
    and `fetch` read the raise as "no report": the node in Europe closed round
    1 without the American one after exactly 30 s, while the American one,
    still in its first round for 56 s, then found the European report
    published and averaged it in. Two nodes averaging different sets from the
    first round is two models, and every later round closed over both of them
    as if nothing had happened.
    """
    if count == 0:
        return b""
    buffer = bytearray()
    while len(buffer) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OSError("deadline passed after %d of %d bytes" % (len(buffer), count))
        connection.settimeout(min(remaining, RECV_SLICE))
        try:
            block = connection.recv(min(CHUNK, count - len(buffer)))
        except _socket.timeout:
            continue
        if not block:
            raise OSError("peer closed after %d of %d bytes" % (len(buffer), count))
        buffer += block
    return bytes(buffer)
