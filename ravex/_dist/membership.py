"""Who is in a round, when the answer can change while the run is going.

GPU-121, under GPU-113. ``ravex._dist.exchange`` already makes "a node dies and
the training does not stop" true, and it does it without asking anyone: the
list of reports is short and the round closes over whoever answered. The other
half of GPU-113's point 4 — *«un nodo si aggiunge: il training non si ferma»* —
cannot be done the same way, and the asymmetry is worth stating because it is
the whole reason this module exists.

**A departure is safe to notice late; an arrival is not.** A node that stops
answering costs the others one deadline each and nothing else: every node
averages over the reports it got, and a node that contributed nothing is
absent from everybody's list at once. A node that *starts* answering is the
opposite. If this node counts the newcomer's delta in round 7 and its peer
does not, the two apply different averages to the same parameters and from
then on they hold two models a fixed distance apart — the loss falls on both,
every round closes over everybody, and nothing raises. That is the same
failure ``exchange.adopt_outer_state`` exists to prevent, arriving through a
different door.

So the rule this module enforces is not "read the store and see who is there".
It is:

    **Membership is a pure function of the round number.**

A joiner writes one key, once, holding the round it will first contribute to,
and never changes it. Every member reads those keys at every round boundary
and computes the peer set for round R as "the base ranks, plus every announced
joiner whose ``admit_round`` is at most R". Two members that read the store at
completely different moments still compute the *same* set for the same round,
which is the property a store lookup on its own does not have.

**The margin, and why it is two and not one.** A joiner does not pick "the next
round". It observes a round number and adds :data:`JOIN_MARGIN`, so there is a
full round — seconds to minutes of wall clock, not microseconds — between the
key being on the store and the first round it affects. Round numbers are in
lockstep by construction (``OuterLoop.abandon_round`` advances the counter
precisely so that a failed exchange does not desynchronise it, and the round
rendezvous in ``DeltaExchange`` makes a node ahead wait for one behind), so the
observed number means the same thing on every member.

**What is left, and it raises rather than diverging.** A member could still see
a join key for the first time *after* the round it was supposed to take effect
— which means it has been averaging a different set than its peers. There is no
repair for that from here: the models have already parted. It raises
:class:`MembershipError`. Refusing the joiner and carrying on would be worse,
because it is precisely the silent two-model outcome above, wearing the mask of
a node that declined something. Reaching this means a store read at a round
boundary took longer than a whole round, on a node that was therefore already
unwell.

**Monotone, and that is what makes a late read harmless.** The key is written
once and never rewritten or deleted while the run is going, so a member that
reads late reads exactly what a member that read early read. The registry only
ever grows; a joiner that gives up does not take its key back, it simply never
publishes a report and costs the others one deadline, which is the departure
case and is already safe.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("ravex")

#: One key per joining rank, holding the round it first contributes to.
JOIN_KEY = "ravex/gpu121/join/%d"

#: A candidate saying it exists, before it knows what round to ask for. It is
#: what makes members start republishing their outer parameters at all: nothing
#: is written for joiners on a run that has none.
WANTED_KEY = "ravex/gpu121/wanted/%d"

#: The round each member last wrote its outer parameters down for. Every member
#: writes the same number at the same boundary — rounds are in lockstep — so
#: they overwrite each other with what they already agree on.
STATE_AT_KEY = "ravex/gpu121/state-at"

#: One member's acknowledgement that it has accepted one joiner, and at which
#: round. **This is what makes the protocol safe rather than merely likely.**
#: A joiner publishes its first report only once every base member has written
#: one; short of that it stays silent, and a member that admitted it pays a
#: deadline for a round — which is the departure case, already safe — instead
#: of averaging a contributor its peer never saw.
ACK_KEY = "ravex/gpu121/ack/%d/%d"

#: A node number that has left the run on its own account, holding the round
#: it left at (GPU-142). Written once, never taken back: a node that comes back
#: comes back under a new number.
#:
#: **Why a departure can be declared at all now.** Membership used to have to
#: be identical on every node, because each node averaged whoever it reached
#: and two views of who was in meant two models. Since GPU-140 the round's set
#: is decided on the store, so views may differ without the averages doing so,
#: and the one place a departed number still mattered is the acknowledgement a
#: joiner waits for: a number that left will never give it.
LEFT_KEY = "ravex/gpu142/left/%d"

#: Rounds between a joiner observing the run and first contributing to it.
#: One would be a race with the members' own boundary read; two leaves a whole
#: round for a key already on the store to be noticed. It is not a tuning knob
#: for throughput — it is the width of the window this module is closing.
JOIN_MARGIN = 2

#: How many ranks past the launched world size are looked at for join keys.
#: Bounded because the scan happens at every round boundary and an unbounded
#: one would grow with nothing. A run that wants to more than double is asking
#: for a relaunch, not for a bigger number here.
JOIN_WINDOW = 16


class MembershipError(RuntimeError):
    """A membership change this node cannot apply without diverging.

    Its own class because the caller has to treat it differently from every
    other failure in a round: ``_close_outer_round`` swallows exceptions and
    carries on, which is right for a peer whose disk was full and wrong for
    this one. There is nothing to carry on *to* — the averages have already
    differed.
    """


def announce(store, rank: int, admit_round: int) -> None:
    """Say that ``rank`` will contribute from ``admit_round`` onwards.

    Written once by the joiner and never rewritten. Callers that retry a failed
    join announce under a *later* ``admit_round``, which is a new value on the
    same key and is the one case rewriting is correct — the old number is in
    the past by then, so no member can be applying it.
    """
    store.set(JOIN_KEY % rank, json.dumps(int(admit_round)).encode("utf-8"))


def announced(store, base_world: int, window: int = JOIN_WINDOW) -> Dict[int, int]:
    """Every announced joiner and the round it enters at.

    Ranks are looked for one at a time with ``check``, never ``get``: ``get``
    blocks until the key exists, which on a store that has no joiner is the
    round boundary waiting for something that will never be written.
    """
    found: Dict[int, int] = {}
    for rank in range(base_world, base_world + window):
        key = JOIN_KEY % rank
        try:
            if not store.check([key]):
                continue
            found[rank] = int(json.loads(store.get(key).decode("utf-8")))
        except Exception:
            # A store that cannot be read is not a membership decision. The
            # round goes on with the set this node already holds, which is the
            # set it computed from keys that are still there.
            break
    return found


def observe(store, base_world: int, window: int = JOIN_WINDOW) -> Optional[int]:
    """The highest ``admit_round`` already announced, or None.

    For a joiner **retrying**: the second announcement has to name a round
    later than the first, or a member still holding the first one sees no
    change and goes on waiting at a round the joiner has abandoned. Two
    different candidates naming the *same* round is not a problem and is not
    what this guards — ``combine`` divides by the contributions it got, so two
    arrivals in one round are two arrivals in one round.
    """
    seen = announced(store, base_world, window)
    return max(seen.values()) if seen else None


class Membership:
    """One node's view of who is in the run, evaluated per round.

    Holds the joiners it has accepted so that "have I seen this before" is
    answerable — which is what turns a key discovered too late into a raise
    instead of a silent divergence.
    """

    def __init__(self, store, rank: int, base_world: int,
                 window: int = JOIN_WINDOW,
                 ceiling: Optional[Callable[[], int]] = None):
        self.store = store
        self.rank = rank
        self.base_world = base_world
        self.window = window
        #: How many node numbers the rendezvous has handed out, when there is a
        #: rendezvous of Ravex's own to ask (GPU-129). See :meth:`span`.
        self.ceiling = ceiling
        self.accepted: Dict[int, int] = {}

        #: Joiners this node has already told the store it accepts. Kept so
        #: the acknowledgement is written once and not at every boundary for
        #: the rest of the run.
        self.acknowledged_joiners: set = set()

        #: Base numbers that declared they left (GPU-142). Only ever grows, so
        #: a number seen here is never asked about again.
        self.departed: set = set()

        #: For each candidate, the last round this node will write its outer
        #: parameters down for. A candidate that announces itself and then dies
        #: would otherwise cost every member a full model snapshot per round
        #: forever, and nothing would ever say why the run got slower.
        self._state_until: Dict[int, int] = {}

    def span(self) -> int:
        """How many ranks past the base to look at for join keys.

        With a rendezvous of Ravex's own the answer is exact: nobody announces
        without having taken a number first, so the counter bounds every rank
        that could have. Without one it is the fixed :data:`JOIN_WINDOW`.

        The difference is more than a ceiling. The fixed window limited the
        joins a run could take **over its whole life** — a node that crashes
        and comes back takes a new number — and every rank looked at is a
        store round trip at every boundary, twice. On a link between
        continents a round trip is a sizeable fraction of a second, and the
        window was sixteen of them asking about nodes that do not exist.
        """
        if self.ceiling is None:
            return self.window
        try:
            return max(0, int(self.ceiling()) - self.base_world)
        except Exception:
            return self.window

    def take_in_joined(self) -> None:
        """For a node that has just joined: every announcement made so far counts.

        A member meets each join key at a boundary before the round it names,
        and :meth:`refresh` raises when it meets one too late. A node that has
        just joined cannot have met the earlier ones in time — it was not
        running — and does not need to have: :func:`join` handed it the outer
        parameters published for its own admission round, which already carry
        everything those earlier joiners contributed. So they are taken in as
        known, and from here this node meets later ones the way members do.
        """
        for peer, admit in announced(self.store, self.base_world, self.span()).items():
            if peer != self.rank:
                self.accepted[peer] = admit

    def wants_state(self, round_number: int) -> bool:
        """Whether somebody is waiting for the outer parameters right now.

        The window is opened by a candidate's ``wanted`` key and closes on its
        own :data:`JOIN_MARGIN` + 2 rounds later — long enough for the round it
        can legally announce, plus the one it actually takes the state from,
        plus one. A candidate that vanishes mid-join therefore costs four
        snapshots and not a run's worth, and the last one is logged.
        """
        wanted = False
        for rank in range(self.base_world, self.base_world + self.span()):
            if rank == self.rank:
                continue
            try:
                if not self.store.check([WANTED_KEY % rank]):
                    continue
            except Exception:
                return False
            until = self._state_until.get(rank)
            if until is None:
                until = round_number + JOIN_MARGIN + 2
                self._state_until[rank] = until
                logger.info(
                    "Rank %d wants into the run; writing the outer parameters "
                    "down at every boundary through round %d.", rank, until,
                )
            if round_number <= until:
                wanted = True
            elif round_number == until + 1:
                logger.warning(
                    "Rank %d asked to join at round %d and never announced a "
                    "round to join at. No more outer-state snapshots for it; "
                    "it can ask again.", rank, until - JOIN_MARGIN - 2,
                )
        return wanted

    def peers_at(self, round_number: int) -> List[int]:
        """Every rank this node exchanges with for round ``round_number``.

        Call it at the round boundary, before the round is trained, and use
        what it returns for that round only. Calling it once and keeping the
        list is what this module exists to stop: it is the shape
        ``_build_outer_loop`` had, and a joiner appears in it nowhere.
        """
        self.refresh(round_number)
        joined = [
            peer
            for peer, admit in sorted(self.accepted.items())
            if admit <= round_number
        ]
        base = [peer for peer in range(self.base_world) if peer != self.rank]
        # Not a correctness matter since GPU-140 - a departed number that is
        # still asked just does not answer - but asking it every round costs a
        # dial and a warning line for the rest of the run.
        for peer in base:
            if peer not in self.departed and has_left(self.store, peer):
                self.departed.add(peer)
        return [
            peer for peer in base + joined
            if peer != self.rank and peer not in self.departed
        ]

    def refresh(self, round_number: int) -> None:
        """Take in any join keys written since the last look.

        Separate from :meth:`peers_at` because a joiner calls this too — it has
        to know about *other* joiners to compute its own admission round, and
        it has no peer set of its own to ask for yet.
        """
        for peer, admit in announced(self.store, self.base_world, self.span()).items():
            if peer == self.rank:
                continue
            known = self.accepted.get(peer)
            if known == admit:
                continue
            if known is None and admit <= round_number:
                raise MembershipError(
                    "rank %d announced that it joins at round %d and this node "
                    "is only seeing that now, at round %d. It has therefore "
                    "been averaging a different set of nodes than the peers "
                    "that saw it in time, which is two models and not a "
                    "degraded one - and there is nothing to repair from here. "
                    "A store read at a round boundary outlasting a whole "
                    "round is the condition that produces this."
                    % (peer, admit, round_number)
                )
            if known is not None and admit <= round_number:
                # A re-announcement whose round has already passed. The joiner
                # writes a later number when it retries, so this is a stale
                # read rather than a new decision, and the value already held
                # is the one every member agreed on.
                continue
            if known is not None:
                logger.info(
                    "Rank %d moved its join from round %d to round %d.",
                    peer, known, admit,
                )
            else:
                logger.info(
                    "Rank %d joins the run at round %d (now at round %d).",
                    peer, admit, round_number,
                )
            self.accepted[peer] = admit


def want_in(store, rank: int) -> None:
    """A candidate says it exists, before it can know what to ask for.

    Two keys and not one because the two facts arrive at different times: this
    one starts the members writing their outer parameters down, and only once
    they have — and said which round they wrote — can the candidate pick the
    round it joins at and :func:`announce` it.
    """
    store.set(WANTED_KEY % rank, b"1")


def gave_up(store, rank: int) -> None:
    """Stop the members writing state down for a candidate that is in, or out.

    Best effort, and the members do not depend on it: they stop on their own
    after :data:`JOIN_MARGIN` + 2 rounds. This only makes the common case cost
    one round of snapshots instead of four.
    """
    try:
        store.delete_key(WANTED_KEY % rank)
    except Exception:
        pass


def leave(store, rank: int, round_number: int) -> None:
    """Declare that ``rank`` has left the run, at ``round_number``. For good."""
    store.set(LEFT_KEY % rank, json.dumps(int(round_number)).encode("utf-8"))


def has_left(store, rank: int) -> bool:
    try:
        return bool(store.check([LEFT_KEY % rank]))
    except Exception:
        return False


def acknowledge(store, member: int, joiner: int, admit_round: int) -> None:
    """One member records that it has accepted one joiner, at which round."""
    store.set(
        ACK_KEY % (member, joiner), json.dumps(int(admit_round)).encode("utf-8")
    )


def acknowledged(store, joiner: int, members: List[int]) -> List[int]:
    """Which of ``members`` have *not* acknowledged ``joiner`` yet.

    Empty means every one of them has, which is the only condition under which
    a joiner may publish its first report. A member that is gone never
    acknowledges, so a run that has lost a node cannot take a new one until
    something declares the lost one gone — **a real limit, and named rather
    than worked around**: the alternative is the joiner deciding for itself
    that a silent member is dead.

    A member that declared its own departure (:func:`leave`, GPU-142) is not
    waited for. That is the declaration this limit was asking for, made by the
    only node that can make it without guessing. A member that died without
    making it still blocks joins.
    """
    missing = []
    for member in members:
        if has_left(store, member):
            continue
        try:
            if not store.check([ACK_KEY % (member, joiner)]):
                missing.append(member)
        except Exception:
            missing.append(member)
    return missing


def serve_joins(store, exchange, loop, membership: "Membership") -> None:
    """The members' half, called at every round boundary. Cheap when idle.

    Two jobs, in this order:

    1. **Write the outer parameters down** if a candidate is waiting for them,
       under the number of the round about to start. A joiner admitted at round
       A takes the state published for A, so it starts that round holding
       exactly what everybody else holds — which is the invariant the whole
       outer loop rests on, and taking state from any earlier round would break
       it by however many rounds it was behind.
    2. **Take in any announcement** and acknowledge it, so the joiner knows it
       is safe to publish.

    Costs one store scan per boundary on a run nobody is joining, and nothing
    else: no snapshot is written until somebody asks.
    """
    round_number = int(loop.round_number)
    membership.refresh(round_number)

    if membership.wants_state(round_number):
        exchange.publish_state(state_payload(loop), round_number)
        try:
            store.set(STATE_AT_KEY, json.dumps(round_number).encode("utf-8"))
        except Exception:
            pass

    for joiner, admit in membership.accepted.items():
        if joiner in membership.acknowledged_joiners:
            continue
        acknowledge(store, membership.rank, joiner, admit)
        membership.acknowledged_joiners.add(joiner)


def state_round(store) -> Optional[int]:
    """The round the members last wrote their outer parameters down for."""
    try:
        if not store.check([STATE_AT_KEY]):
            return None
        return int(json.loads(store.get(STATE_AT_KEY).decode("utf-8")))
    except Exception:
        return None


def join(store, exchange, loop, rank: int, base_world: int, deadline: float,
         poll: float = 0.05, window: int = JOIN_WINDOW) -> bool:
    """Take a node into a run that is already going. The joiner's whole half.

    Returns True once this node holds the run's outer parameters and is
    entitled to contribute from ``loop.round_number``; False if it ran out of
    ``deadline`` trying, having published nothing and told nobody it would.

    The order is the argument:

    1. **Say it exists**, which is what makes the members start writing their
       outer parameters down. Nothing is written for joiners on a run with
       none, so a run that never grows pays nothing for this.
    2. **Read what round they are on** and announce ``that + JOIN_MARGIN``.
       The margin is a whole round of slack for every member to notice the
       announcement, and it is what makes membership at a given round the same
       on all of them.
    3. **Wait for that round to arrive**, and take the outer parameters the
       members published *for it* — not for the round observed in step 2. A
       joiner adopting an older snapshot would start ``admit - observed``
       rounds behind everybody and stay exactly that far behind for the length
       of the run, which is ``adopt_outer_state``'s failure with a different
       cause.
    4. **Check every base member acknowledged** before publishing anything.
       This is the step that turns the remaining race into a wasted deadline:
       if one member never saw the announcement, this node stays silent, every
       member closes the round without it, and they all close the *same* round.

    A run that has already lost a member cannot take a new one, because the
    lost member never acknowledges. Named in :func:`acknowledged`.
    """
    members = [peer for peer in range(base_world) if peer != rank]
    want_in(store, rank)
    announced_at: Optional[int] = None

    while time.monotonic() < deadline:
        observed = state_round(store)
        if observed is None:
            time.sleep(poll)
            continue

        floor = observed + JOIN_MARGIN
        if announced_at is None or announced_at < observed:
            # Either the first go, or the round announced has been passed
            # while this node was still getting ready - which is a join that
            # did not happen rather than one that half happened, because
            # nothing was published under it.
            highest = observe(store, base_world, window) or -1
            announced_at = max(floor, highest + 1)
            announce(store, rank, announced_at)
            logger.info(
                "Asked to join at round %d (the run is at round %d).",
                announced_at, observed,
            )
            continue

        if observed < announced_at:
            time.sleep(poll)
            continue

        missing = acknowledged(store, rank, members)
        if missing:
            logger.warning(
                "Round %d arrived and rank(s) %s never acknowledged this "
                "node. Publishing nothing - every member closes this round "
                "without it, which is the same round for all of them - and "
                "asking again for a later one.",
                announced_at,
                ", ".join(str(member) for member in missing),
            )
            announced_at = observed  # forces a fresh, later announcement
            continue

        state = exchange.fetch_state(
            members, announced_at, _expectation(state_payload(loop)), deadline
        )
        if state is None:
            logger.warning(
                "No member served the outer parameters for round %d. Asking "
                "again for a later round.", announced_at,
            )
            announced_at = observed
            continue

        apply_state(loop, state.delta)
        loop.round_number = announced_at
        loop.steps_this_round = 0
        gave_up(store, rank)
        logger.info(
            "Joined the run at round %d, holding the outer parameters every "
            "member holds.", announced_at,
        )
        return True

    gave_up(store, rank)
    return False


def _expectation(outer):
    from ravex._dist import report as _report

    return _report.expectation(outer)


#: How the two halves of a joiner's state share one report. A parameter name
#: can contain dots and slashes — ``0.weight`` — so the separator is one that
#: cannot: ``::``.
OUTER_PREFIX = "outer::"
MOMENTUM_PREFIX = "momentum::"


def state_payload(loop) -> Dict[str, object]:
    """What a joining node has to be handed. **Not just the parameters.**

    The outer optimizer's momentum buffer goes with them, and leaving it out is
    not a small loss — it is the same defect GPU-110 found one floor up with
    Adam's moments, wearing different clothes. The buffer is what carries a
    direction several rounds agreed on through a round where one node dominated
    the average, so a node that arrives with an empty one applies a *different*
    update to the same gradient from its very first round. Every node then
    holds a different model, immediately, and nothing raises: the loss falls,
    the rounds close over everybody.

    Found exactly that way — the first version of this shipped ``loop.outer``
    alone and ``torch.equal`` failed on the joiner in the round it entered, by
    far more than a last-place difference.

    What is *not* shipped is ``lr``, ``momentum`` and ``nesterov``: those are
    configuration, and a joiner whose configuration differs from the run's has
    a problem this transfer would hide rather than fix.
    """
    import torch

    buffers = getattr(loop.optimizer, "buffers", {}) or {}
    payload: Dict[str, object] = {}
    for name, value in loop.outer.items():
        payload[OUTER_PREFIX + name] = value
        # **Both halves, always, even before a single outer step has been
        # taken.** A payload whose shape depends on how old the run is cannot
        # be described in advance, and the joiner has to describe it: the
        # report format checks what arrived against what was expected, and a
        # node that has never stepped expects half of what a running one
        # sends. Zeros are not a stand-in here either — they are exactly what
        # an absent buffer means, because the first step does
        # ``buffer.mul_(momentum).add_(grad)`` and a zero buffer gives ``grad``,
        # which is what the ``buffer is None`` branch produces.
        buffer = buffers.get(name)
        payload[MOMENTUM_PREFIX + name] = (
            buffer if buffer is not None else torch.zeros_like(value)
        )
    return payload


def apply_state(loop, payload: Dict[str, object]) -> None:
    """Put a :func:`state_payload` into this node's loop, and into its model."""
    outer = {}
    buffers = {}
    for key, value in payload.items():
        if key.startswith(OUTER_PREFIX):
            outer[key[len(OUTER_PREFIX):]] = value
        elif key.startswith(MOMENTUM_PREFIX):
            buffers[key[len(MOMENTUM_PREFIX):]] = value

    if not outer:
        raise MembershipError(
            "the state served for this join carries no outer parameters. "
            "Adopting the momentum alone would leave this node training its "
            "own weights inside everyone else's average."
        )
    loop.outer = outer
    loop.optimizer.buffers = buffers
    loop.write_back()
