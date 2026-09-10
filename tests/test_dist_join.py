"""GPU-121: a node that joins a run already in progress.

``test_dist_round.py`` proves the other half of GPU-113's point 4 — a node dies
and the round closes over whoever is left. This is the arrival, and the two are
not symmetric. A departure is safe to notice late: the node that went away is
missing from *everybody's* list of reports at once, because there is nothing to
miss. An arrival noticed by one node and not another is two nodes applying
different averages to the same parameters, for ever, with the loss falling on
both and nothing raising — which is why the interesting assertions here are not
about the newcomer training well, they are about all three holding **one
model**, checked with ``torch.equal`` and not ``allclose``.

Threads, one per node, for the same reason ``test_dist_round.py`` gives: a node
blocks inside ``close_round`` until its peers publish, so driving them in
sequence would be testing a shape the real thing never has.
"""

import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from ravex._dist import membership as _membership  # noqa: E402
from ravex._dist.exchange import DeltaExchange, close_round  # noqa: E402
from ravex._dist.outer import OuterLoop  # noqa: E402

from test_dist_exchange import FakeStore  # noqa: E402
from test_dist_round import a_model, a_problem, loss_of, train  # noqa: E402

BASE_WORLD = 2


class Plan:
    """How long the members keep going, decided by the node that joins.

    A fixed round count on the members is a race and not a test: with rounds
    this small they can finish the run before the joiner has announced
    anything, and then the join times out for a reason that has nothing to do
    with the protocol. So the members run until the joiner says it is done, and
    the joiner says so by naming the last round it will contribute to - which
    it knows the moment it is admitted, and which lets the members stop
    *without* paying a gather deadline waiting for a node that has finished.
    """

    def __init__(self, cap):
        self.cap = cap
        self.last_round = None


class Member(threading.Thread):
    """One of the nodes the run started with."""

    def __init__(self, rank, store, shard, plan, inner, root):
        super().__init__(daemon=True)
        self.rank = rank
        self.store = store
        self.shard = shard
        self.plan = plan
        self.inner = inner
        self.model = a_model()
        self.exchange = DeltaExchange(
            rank, store, root=os.path.join(root, "rank%d" % rank),
            node="n%d" % rank, patience=15.0,
        )
        self.membership = _membership.Membership(store, rank, BASE_WORLD)
        self.loop = None
        self.reports = []
        self.error = None

    def run(self):
        try:
            assert self.exchange.start()
            self.loop = OuterLoop(
                self.model, inner_steps=self.inner, node="n%d" % self.rank
            )
            optimizer = None
            for _ in range(self.plan.cap):
                last = self.plan.last_round
                if last is not None and self.loop.round_number > last:
                    break
                # At the boundary, before the round is trained: this is where a
                # joiner's state has to be written down and its announcement
                # taken in, because the number it is taken in *at* is what every
                # member has to agree on.
                _membership.serve_joins(
                    self.store, self.exchange, self.loop, self.membership
                )
                peers = self.membership.peers_at(self.loop.round_number)
                optimizer = train(
                    self.model, *self.shard, self.inner,
                    self.loop.round_number * self.inner, optimizer,
                )
                for _ in range(self.inner):
                    self.loop.record_step()
                self.reports.append(
                    close_round(
                        self.loop, self.exchange, peers, time.monotonic() + 30
                    )
                )
        except BaseException as exc:
            self.error = exc
        finally:
            self.exchange.close(linger=5)


class Joiner(threading.Thread):
    """The node that was not there when the run started."""

    def __init__(self, rank, store, shard, rounds, inner, root, plan, after=0.0):
        super().__init__(daemon=True)
        self.rank = rank
        self.store = store
        self.shard = shard
        self.rounds = rounds
        self.inner = inner
        self.after = after
        self.plan = plan
        # A different seed on purpose: if the join did not really transfer the
        # outer parameters, this node's own weights are what would be averaged
        # in, and every assertion about one model would fail loudly instead of
        # passing because the models happened to start alike.
        self.model = a_model(seed=17)
        self.exchange = DeltaExchange(
            rank, store, root=os.path.join(root, "rank%d" % rank),
            node="n%d" % rank, patience=15.0,
        )
        self.membership = _membership.Membership(store, rank, BASE_WORLD)
        self.loop = None
        self.joined_at = None
        self.reports = []
        self.error = None

    def run(self):
        try:
            time.sleep(self.after)
            assert self.exchange.start()
            self.loop = OuterLoop(
                self.model, inner_steps=self.inner, node="n%d" % self.rank
            )
            if not _membership.join(
                self.store, self.exchange, self.loop, self.rank,
                BASE_WORLD, time.monotonic() + 60,
            ):
                raise AssertionError("the join never completed")
            self.joined_at = self.loop.round_number
            self.plan.last_round = self.joined_at + self.rounds - 1

            optimizer = None
            for _ in range(self.rounds):
                peers = self.membership.peers_at(self.loop.round_number)
                optimizer = train(
                    self.model, *self.shard, self.inner,
                    self.loop.round_number * self.inner, optimizer,
                )
                for _ in range(self.inner):
                    self.loop.record_step()
                self.reports.append(
                    close_round(
                        self.loop, self.exchange, peers, time.monotonic() + 30
                    )
                )
        except BaseException as exc:
            self.error = exc
        finally:
            if self.plan.last_round is None:
                # A join that never happened still has to release the members,
                # or they run to the cap waiting for a round that is not coming.
                self.plan.last_round = -1
            self.exchange.close(linger=5)


def a_run(root, cap=40, inner=6, joiner_rounds=3, after=0.3):
    store = FakeStore()
    x, y = a_problem()
    third = len(x) // 3
    shards = [
        (x[i * third : (i + 1) * third], y[i * third : (i + 1) * third])
        for i in range(3)
    ]
    plan = Plan(cap)
    members = [
        Member(rank, store, shards[rank], plan, inner, root)
        for rank in range(BASE_WORLD)
    ]
    joiner = Joiner(
        BASE_WORLD, store, shards[BASE_WORLD], joiner_rounds, inner, root,
        plan, after=after,
    )

    for node in members + [joiner]:
        node.start()
    for node in members + [joiner]:
        node.join(240)
    for node in members + [joiner]:
        assert not node.is_alive(), "rank %d never finished" % node.rank
        if node.error is not None:
            raise node.error
    return members, joiner, (x, y)


def test_a_node_joins_a_running_run_and_its_delta_is_in_the_average(tmp_path):
    """The sentence GPU-113 asks for by name, with a real socket arriving.

    Three things have to be true at once, and only the third is hard: the
    newcomer contributes from a round it announced in advance; the members
    counted it from that same round and not one either side of it; and all
    three hold one model afterwards, bit for bit.
    """
    members, joiner, (x, y) = a_run(str(tmp_path))

    assert joiner.joined_at is not None
    admitted = joiner.joined_at

    # It joined mid-run, not at the start: a test where the newcomer happens to
    # arrive before round 1 proves nothing this issue is about.
    assert admitted >= 1, "the joiner was admitted before the run got going"

    # Every member counted three nodes from exactly the announced round, and
    # two before it. This is the assertion that fails if membership is read off
    # the store at whatever moment each node happens to look.
    for member in members:
        counts = {report["round"]: report["nodes"] for report in member.reports}
        for round_number, nodes in counts.items():
            if round_number < admitted:
                assert nodes == BASE_WORLD, (
                    "rank %d counted %d nodes in round %d, before the joiner "
                    "was admitted at %d"
                    % (member.rank, nodes, round_number, admitted)
                )
            elif round_number < admitted + len(joiner.reports):
                assert nodes == BASE_WORLD + 1, (
                    "rank %d counted %d nodes in round %d, where the joiner "
                    "was admitted and contributing"
                    % (member.rank, nodes, round_number)
                )

    # And the joiner's own rounds are the same rounds, by number.
    assert [report["round"] for report in joiner.reports] == list(
        range(admitted, admitted + len(joiner.reports))
    )


def test_every_node_including_the_one_that_joined_holds_one_model(tmp_path):
    """The invariant, with ``torch.equal`` — which is the point.

    ``allclose`` would pass on two nodes that are drifting apart slowly, and
    slowly is how this fails: an outer average assembled in a different order
    on each node differs in the last places, and the difference is applied to
    the parameters rather than cancelling, so it accumulates. ``combine``
    sorts by node name for exactly this reason.
    """
    members, joiner, _ = a_run(str(tmp_path))

    # Compared at the last round the joiner took part in, which is where all
    # three have applied the same averages.
    last = joiner.reports[-1]["round"]
    holders = [member for member in members] + [joiner]
    reference = holders[0].loop.outer

    for holder in holders[1:]:
        for name, value in reference.items():
            assert torch.equal(value, holder.loop.outer[name]), (
                "rank %d holds a different %r after round %d"
                % (holder.rank, name, last)
            )


def test_the_joiner_did_not_bring_its_own_weights_in(tmp_path):
    """A join that silently skipped the state transfer looks like this test
    passing on everything else and failing here.

    The joiner starts from a different seed. If it had contributed without
    adopting the run's outer parameters, its delta would carry that difference
    into the average once and never again — the classic constant offset — and
    the members' loss would be visibly worse than a run without it.
    """
    members, joiner, (x, y) = a_run(str(tmp_path))

    fresh = loss_of(a_model(), x, y)
    for node in list(members) + [joiner]:
        assert loss_of(node.model, x, y) < fresh / 2, (
            "rank %d did not converge, which is what a joiner that brought "
            "its own weights into the average looks like" % node.rank
        )


def test_a_joiner_publishes_nothing_until_every_member_acknowledged(tmp_path):
    """The step that turns the last race into a wasted deadline.

    One member is kept from ever acknowledging. The joiner must then stay
    silent rather than publish into a round only some members would count it
    in — so the join fails, and it fails without having written a report.
    """
    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    root = str(tmp_path)

    plan = Plan(40)
    members = [
        Member(rank, store, (x[:half], y[:half]), plan, 6, root)
        for rank in range(BASE_WORLD)
    ]
    # Rank 1 takes announcements in but never writes an acknowledgement: the
    # set of joiners it thinks it has already told the store about is primed
    # with this one, so `serve_joins` never writes the key.
    members[1].membership.acknowledged_joiners.add(BASE_WORLD)

    joiner = Joiner(
        BASE_WORLD, store, (x[half:], y[half:]), 2, 6, root, plan, after=0.4
    )
    joiner_deadline_hit = []

    original = _membership.join

    def brief_join(store_, exchange, loop, rank, base_world, deadline, **kw):
        # A short deadline, because what is under test is that it gives up
        # rather than that it waits well.
        result = original(
            store_, exchange, loop, rank, base_world, time.monotonic() + 8, **kw
        )
        joiner_deadline_hit.append(result)
        return result

    _membership.join = brief_join
    try:
        for node in members + [joiner]:
            node.start()
        for node in members + [joiner]:
            node.join(240)
    finally:
        _membership.join = original

    assert joiner_deadline_hit == [False], "the join should not have completed"
    assert joiner.reports == [], "the joiner published into a round anyway"

    # And the members carried on over themselves alone, agreeing with each
    # other about every round: never a round where one of them counted the
    # joiner in, which is the outcome the acknowledgement exists to rule out.
    for member in members:
        assert member.error is None
        assert member.reports, "rank %d closed no rounds at all" % member.rank
        assert [report["nodes"] for report in member.reports] == (
            [BASE_WORLD] * len(member.reports)
        )
