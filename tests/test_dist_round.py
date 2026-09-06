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

    def __init__(self, rank, store, shard, rounds, inner, world, root, dies_after=None):
        super().__init__(daemon=True)
        self.rank = rank
        self.store = store
        self.shard = shard
        self.rounds = rounds
        self.inner = inner
        self.peers = [r for r in range(world) if r != rank]
        self.dies_after = dies_after
        self.model = a_model()
        self.exchange = DeltaExchange(
            rank, store, root=os.path.join(root, "rank%d" % rank),
            node="n%d" % rank, patience=15.0,
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
                        time.monotonic() + 30,
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
