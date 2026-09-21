"""GPU-115: a whole round, end to end, over real sockets.

``tests/test_dist_outer.py`` proves the arithmetic with the network stubbed out
and ``tests/test_dist_exchange.py`` proves the transport with the training
stubbed out. This is the pair of them running together: several nodes, each in
its own thread with its own model, its own optimizer and its own shard, talking
over loopback and agreeing on one model at the end of every round.

Threads rather than a single loop driving both nodes, because a node blocks
inside ``close_round`` until its peers publish. Driving them in sequence would
be testing a shape the real thing never has — and the deadlock that shape
causes is exactly what the round rendezvous exists to prevent.
"""

import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from ravex._dist.exchange import DeltaExchange, close_round  # noqa: E402
from ravex._dist.outer import OuterLoop  # noqa: E402

from test_dist_exchange import FakeStore  # noqa: E402


def a_model(seed=3):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    )


def a_problem(n=512, seed=7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=generator)
    truth = torch.randn(4, 1, generator=generator)
    return x, x @ truth + 0.05 * torch.randn(n, 1, generator=generator)


def loss_of(model, x, y):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x), y).item()


def train(model, x, y, steps, start, optimizer):
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()
    for step in range(steps):
        begin = ((start + step) * 32) % len(x)
        optimizer.zero_grad()
        loss_fn(model(x[begin : begin + 32]), y[begin : begin + 32]).backward()
        optimizer.step()
    return optimizer


class Node(threading.Thread):
    """One participant: a model, a shard, a listener, and its own clock."""

    def __init__(self, rank, store, shard, rounds, inner, world, root,
                 dies_after=None, save_dtype=None, deadline=30):
        super().__init__(daemon=True)
        self.rank = rank
        self.store = store
        self.shard = shard
        self.rounds = rounds
        self.inner = inner
        self.peers = [r for r in range(world) if r != rank]
        self.dies_after = dies_after
        self.deadline = deadline
        self.model = a_model()
        self.exchange = DeltaExchange(
            rank, store, root=os.path.join(root, "rank%d" % rank),
            node="n%d" % rank, patience=15.0, save_dtype=save_dtype,
        )
        self.loop = None
        self.reports = []
        self.error = None

    def run(self):
        try:
            assert self.exchange.start()
            self.loop = OuterLoop(self.model, inner_steps=self.inner, node="n%d" % self.rank)
            optimizer = None
            for round_number in range(self.rounds):
                if self.dies_after is not None and round_number >= self.dies_after:
                    return  # the box went away, with no notice and no cleanup
                optimizer = train(
                    self.model, *self.shard, self.inner,
                    round_number * self.inner, optimizer,
                )
                for _ in range(self.inner):
                    self.loop.record_step()
                self.reports.append(
                    close_round(
                        self.loop, self.exchange, self.peers,
                        time.monotonic() + self.deadline,
                    )
                )
        except BaseException as exc:  # surfaced on the test's thread
            self.error = exc
        finally:
            # A node that stops serving the moment it is done takes its last
            # report with it - see DeltaExchange.close.
            self.exchange.close(linger=5, expect=self.peers)


def run_nodes(root, world=2, rounds=4, inner=15, dies_after=None):
    store = FakeStore()
    x, y = a_problem()
    size = len(x) // world
    nodes = [
        Node(
            rank, store,
            (x[rank * size : (rank + 1) * size], y[rank * size : (rank + 1) * size]),
            rounds, inner, world, root,
            dies_after=dies_after if rank == world - 1 else None,
        )
        for rank in range(world)
    ]
    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
    for node in nodes:
        assert not node.is_alive(), "rank %d never finished" % node.rank
        if node.error is not None:
            raise node.error
    return nodes, (x, y)


def test_two_nodes_train_one_model_over_sockets(tmp_path):
    """The end-to-end claim: real deltas, real sockets, one model at the end."""
    nodes, (x, y) = run_nodes(str(tmp_path), world=2, rounds=4, inner=15)

    assert loss_of(nodes[0].model, x, y) < loss_of(a_model(), x, y) / 2

    left = dict(nodes[0].model.named_parameters())
    for name, right in nodes[1].model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name

    for node in nodes:
        assert [r["nodes"] for r in node.reports] == [2, 2, 2, 2]


def test_three_nodes_all_see_each_other(tmp_path):
    nodes, _ = run_nodes(str(tmp_path), world=3, rounds=3, inner=10)
    for node in nodes:
        assert [r["nodes"] for r in node.reports] == [3, 3, 3]


def test_a_node_that_dies_mid_run_does_not_stop_the_others(tmp_path):
    """The sentence the whole issue exists for, with a real socket going away.

    Rank 2 stops after two rounds without saying anything - no leave, no
    cleanup, its advertised address still on the store. The survivors have to
    close the rounds after that on their own and keep converging.
    """
    nodes, (x, y) = run_nodes(str(tmp_path), world=3, rounds=4, inner=10, dies_after=2)
    survivors = nodes[:2]

    for node in survivors:
        assert len(node.reports) == 4, "a survivor stopped when the third left"
        assert [r["nodes"] for r in node.reports] == [3, 3, 2, 2]

    left = dict(survivors[0].model.named_parameters())
    for name, right in survivors[1].model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name
    assert loss_of(survivors[0].model, x, y) < loss_of(a_model(), x, y) / 2


def two_halves(world):
    x, y = a_problem()
    size = len(x) // world
    return [(x[r * size:(r + 1) * size], y[r * size:(r + 1) * size]) for r in range(world)]


def run_all(nodes):
    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
        assert not node.is_alive(), "rank %d never finished" % node.rank


def same_model(a, b):
    left = dict(a.model.named_parameters())
    for name, right in b.model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name


def failing_fetch(node, owner, times=None, stall=0.0):
    """Make ``node``'s direct fetches of ``owner``'s rounds run out of time.

    Through the real ``fetch``, handed a deadline already spent - the branch a
    slow link takes. ``stall`` first, so this node reaches the round's decision
    after the others have made it. Relays go through untouched.
    """
    original = node.exchange.fetch
    failed = []

    def fetch(peer, round_number, expected, deadline, kind=0, owner_=None, **kw):
        relay_owner = kw.pop("owner", owner_)
        if peer == owner and kind == 0 and (times is None or len(failed) < times):
            failed.append(round_number)
            time.sleep(stall)
            return original(peer, round_number, expected, time.monotonic(), kind)
        return original(peer, round_number, expected, deadline, kind, owner=relay_owner)

    node.exchange.fetch = fetch
    return failed


def test_a_one_way_timeout_between_live_nodes_does_not_split_the_model(tmp_path):
    """GPU-140 without anyone dying.

    The rule the round rested on was that a node which did not contribute is
    missing from *everybody's* list at once. That holds for a node that is
    gone; it does not hold for a link that is slow in one direction. Here
    rank 1's fetch of rank 0's first report runs out of time while rank 0
    fetches rank 1's normally, and both are alive for the whole run.

    Before the round's set was decided on the store, rank 0 averaged {0, 1},
    rank 1 averaged {1}, and they held two models from then on - both losses
    falling, nothing raised. Now both apply the one set the store decided.
    """
    store = FakeStore()
    nodes = [Node(r, store, shard, 3, 10, 2, str(tmp_path))
             for r, shard in enumerate(two_halves(2))]
    failed = failing_fetch(nodes[1], owner=0, times=1)

    run_all(nodes)
    for node in nodes:
        if node.error is not None:
            raise node.error

    assert failed, "the one-way timeout was never injected"
    assert [r["nodes"] for r in nodes[0].reports] == [r["nodes"] for r in nodes[1].reports]
    same_model(*nodes)


def test_a_report_the_owner_cannot_hand_over_is_relayed_by_one_that_holds_it(tmp_path):
    """The recovery half: rank 2 never reaches rank 0, rank 1 always does.

    Rank 2 stalls before giving up on rank 0, so the round is decided by a
    node that holds rank 0's report and names it. Rank 2 then has to average a
    report it cannot fetch from its owner, and gets it from a node that has it.
    """
    store = FakeStore()
    nodes = [Node(r, store, shard, 2, 10, 3, str(tmp_path))
             for r, shard in enumerate(two_halves(3))]
    failing_fetch(nodes[2], owner=0, stall=1.0)

    run_all(nodes)
    for node in nodes:
        if node.error is not None:
            raise node.error

    assert [r["nodes"] for r in nodes[2].reports] == [3, 3]
    assert sum(r["recovered"] for r in nodes[2].reports) == 2
    same_model(nodes[0], nodes[2])
    same_model(nodes[1], nodes[2])


def test_a_node_that_cannot_get_the_decided_set_stops_instead_of_diverging(tmp_path):
    """And when nobody can hand it over: out, loudly, rather than a second model.

    Two nodes, so there is no third to relay through, and rank 1's every
    direct fetch of rank 0 runs out of time. Rank 0 decides the round over
    both; rank 1 cannot average that set, and the only non-divergent move left
    to it is to leave - which is a departure, and a departure is safe.
    """
    from ravex._dist.exchange import RoundSplitError
    from ravex._dist.membership import MembershipError

    store = FakeStore()
    nodes = [Node(r, store, shard, 2, 10, 2, str(tmp_path), deadline=4)
             for r, shard in enumerate(two_halves(2))]
    failing_fetch(nodes[1], owner=0, stall=1.0)

    run_all(nodes)

    assert isinstance(nodes[1].error, RoundSplitError)
    assert isinstance(nodes[1].error, MembershipError), "the runtime must not swallow it"
    assert nodes[1].reports == [], "it applied a round it could not agree on"
    assert nodes[0].error is None
    assert nodes[0].reports[0]["nodes"] == 2


def test_nodes_at_different_speeds_still_agree_on_one_model(tmp_path):
    """The heterogeneous round, end to end: different step counts, one model."""
    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    fast = Node(0, store, (x[:half], y[:half]), 3, 20, 2, str(tmp_path))
    slow = Node(1, store, (x[half:], y[half:]), 3, 4, 2, str(tmp_path))

    for node in (fast, slow):
        node.start()
    for node in (fast, slow):
        node.join(180)
        if node.error is not None:
            raise node.error

    assert [r["steps"] for r in fast.reports][0] in ([20, 4], [4, 20])
    for report in fast.reports:
        assert report["slowest"] == 4 and report["fastest"] == 20

    left = dict(fast.model.named_parameters())
    for name, right in slow.model.named_parameters():
        assert torch.allclose(left[name], right, atol=1e-5), name


def test_nodes_that_start_from_different_weights_end_up_on_one_model(tmp_path):
    """The seed round, and the failure it exists for is a silent one.

    An outer round applies the same averaged pseudo-gradient to whatever outer
    parameters each node holds, so a difference in the *starting* ones is never
    touched again by anything: it is a constant added to one node's model for
    the whole run. The loss falls, every round closes over everybody, and the
    two nodes are training two models a fixed distance apart.

    Here they deliberately start from different seeds - which is the stronger
    version of what really happens, where each node's first optimizer step has
    already moved its weights with its own data before the loop is built.
    """
    from ravex._dist.exchange import SEED_ROUND, adopt_outer_state

    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    shards = [(x[:half], y[:half]), (x[half:], y[half:])]

    models = [a_model(seed=3), a_model(seed=11)]
    before = [float(p.detach().sum()) for p in models[0].parameters()]
    other = [float(p.detach().sum()) for p in models[1].parameters()]
    assert before != pytest.approx(other), "the two started the same"

    exchanges, loops, errors = [], [], []
    for rank in range(2):
        exchange = DeltaExchange(
            rank, store, root=os.path.join(str(tmp_path), "r%d" % rank),
            node="n%d" % rank, patience=15.0,
        )
        assert exchange.start()
        exchanges.append(exchange)
        loops.append(OuterLoop(models[rank], inner_steps=10, node="n%d" % rank))

    def seed(rank):
        try:
            assert adopt_outer_state(
                loops[rank], exchanges[rank], 0, rank, time.monotonic() + 30
            )
            loops[rank].round_number = SEED_ROUND + 1
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=seed, args=(rank,), daemon=True)
               for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    if errors:
        raise errors[0]

    try:
        left = dict(models[0].named_parameters())
        for name, right in models[1].named_parameters():
            assert torch.allclose(left[name], right, atol=1e-6), name

        # And one real round on top, to show the agreement survives a round
        # rather than only surviving the seeding.
        def one_round(rank):
            try:
                train(models[rank], *shards[rank], 10, 0, None)
                for _ in range(10):
                    loops[rank].record_step()
                close_round(loops[rank], exchanges[rank], [1 - rank],
                            time.monotonic() + 30)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=one_round, args=(rank,), daemon=True)
                   for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        if errors:
            raise errors[0]

        left = dict(models[0].named_parameters())
        for name, right in models[1].named_parameters():
            assert torch.allclose(left[name], right, atol=1e-5), name
    finally:
        for exchange in exchanges:
            exchange.close()


def test_a_cast_round_still_leaves_every_node_on_one_model(tmp_path):
    """``save_dtype`` shrinks the report without splitting the nodes.

    The wire form of a delta is a cast of it, and each node has to average the
    cast one — including its own — or every node takes a slightly different
    outer step from the same round. That failure is the one
    ``adopt_outer_state`` exists to prevent, arriving through the other door:
    nothing raises, the loss falls, every round closes over both nodes, and the
    models drift apart by a quantization error per round for the rest of the
    run. bf16 rather than fp8 so the tolerance below is a real bound and not a
    formality.
    """
    store = FakeStore()
    x, y = a_problem()
    half = len(x) // 2
    nodes = [
        Node(rank, store,
             (x[rank * half : (rank + 1) * half], y[rank * half : (rank + 1) * half]),
             4, 15, 2, str(tmp_path), save_dtype="bf16")
        for rank in range(2)
    ]

    for node in nodes:
        node.start()
    for node in nodes:
        node.join(180)
        if node.error is not None:
            raise node.error

    assert loss_of(nodes[0].model, x, y) < loss_of(a_model(), x, y) / 2

    left = dict(nodes[0].model.named_parameters())
    for name, right in nodes[1].model.named_parameters():
        assert torch.equal(left[name], right), name


def test_the_round_report_says_where_its_seconds_went(tmp_path):
    """GPU-117: the split is in the report a real run produces.

    Not a bench's business alone. Until this existed the only place a round's
    network cost was ever printed was a loopback run, where it was 0.0 — so the
    number the whole architecture is chosen around had never been looked at.
    """
    nodes, _ = run_nodes(str(tmp_path), world=2, rounds=2, inner=10)

    for report in nodes[0].reports:
        for key in ("delta_seconds", "publish_seconds", "publish_wait_seconds",
                    "gather_seconds", "gather_wait_seconds", "apply_seconds"):
            assert key in report, key
            assert report[key] >= 0.0, key
        # The wait is a part of the gather and not a second span next to it.
        assert report["gather_wait_seconds"] <= report["gather_seconds"] + 1e-6
        assert report["publish_wait_seconds"] <= report["publish_seconds"] + 1e-6


def test_the_seed_round_leaves_one_model_even_when_the_report_is_cast(tmp_path):
    """``save_dtype`` must not put the source node a quantization from the rest.

    The peers adopt what came down the wire, which under a cast is not what the
    source wrote. If the source keeps its own uncast copy, the one node that
    did not go through the wire starts a quantization away from everybody else
    — from the first instant, and never touched again, because an outer round
    only ever applies the *same* averaged gradient to whatever each node holds.
    Which is the exact failure ``adopt_outer_state`` exists to prevent,
    reintroduced by the setting that shrinks the round.

    ``torch.equal`` and not ``allclose``: the whole claim is that they are one
    model, and a tolerance is how the previous version of this passed.
    """
    from ravex._dist.exchange import adopt_outer_state

    store = FakeStore()
    models = [a_model(seed=3), a_model(seed=11)]
    exchanges, errors = [], []
    loops = []
    for rank in range(2):
        exchange = DeltaExchange(
            rank, store, root=os.path.join(str(tmp_path), "r%d" % rank),
            node="n%d" % rank, patience=15.0, save_dtype="bf16",
        )
        assert exchange.start()
        exchanges.append(exchange)
        loops.append(OuterLoop(models[rank], inner_steps=10, node="n%d" % rank))

    def seed(rank):
        try:
            assert adopt_outer_state(
                loops[rank], exchanges[rank], 0, rank, time.monotonic() + 30
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=seed, args=(rank,), daemon=True)
               for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    for exchange in exchanges:
        exchange.close()
    if errors:
        raise errors[0]

    for name, value in loops[0].outer.items():
        assert torch.equal(value, loops[1].outer[name]), name
    left = dict(models[0].named_parameters())
    for name, right in models[1].named_parameters():
        assert torch.equal(left[name], right), name
