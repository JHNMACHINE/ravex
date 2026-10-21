"""GPU-115: a whole round, end to end, over real sockets.

``tests/unit/test_dist_outer.py`` proves the arithmetic with the network stubbed out
and ``tests/unit/test_dist_exchange.py`` proves the transport with the training
stubbed out. This is the pair of them running together: several nodes, each in
its own thread with its own model, its own optimizer and its own shard, talking
over loopback and agreeing on one model at the end of every round.

Threads rather than a single loop driving both nodes, because a node blocks
inside ``close_round`` until its peers publish. Driving them in sequence would
be testing a shape the real thing never has — and the deadlock that shape
causes is exactly what the round rendezvous exists to prevent.
"""

import time

import pytest

torch = pytest.importorskip("torch")


from ravex._dist.exchange import close_round  # noqa: E402

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


def test_a_step_that_fails_after_the_decision_is_a_split_not_a_skip():
    """After the decision the others are applying the round: a node that
    cannot is off their model, and says so instead of carrying on."""
    from types import SimpleNamespace

    from ravex._dist.exchange import RoundSplitError

    class Loop:
        round_number = 1
        outer = {}

        def contribution(self):
            return SimpleNamespace(delta={}, steps=1)

        def apply(self, contributions):
            raise RuntimeError("out of memory")

    exchange = SimpleNamespace(
        rank=0, store=FakeStore(), secret=b"t", scope=b"t", publish_wait=0.0, gather_wait=0.0,
        publish=lambda *a: None, as_published=lambda delta, *a: delta,
        gather_by_rank=lambda *a: {}, decided=lambda *a: None, applied=lambda *a: None,
    )
    with pytest.raises(RoundSplitError, match="could not apply round 1"):
        close_round(Loop(), exchange, [], time.monotonic() + 1)


def test_a_rendezvous_gone_is_asked_once_not_until_it_is_given_up(monkeypatch):
    """GPU-229, seen on 2026-10-16: a client that lost its server fails every
    later call at once, each with a C++ warning and its backtrace. Asked every
    0.2 s until the server was given up, that was some 650 lines per node
    around the four that said what happened. One call now, and the same wait
    before the store is called gone."""
    from ravex._dist import exchange
    from ravex._dist.exchange import RoundSplitError, decide_round

    class DistNetworkError(RuntimeError):
        pass

    class Gone:
        calls = 0

        def compare_set(self, key, expected, desired):
            Gone.calls += 1
            raise DistNetworkError("Broken pipe")

    monkeypatch.setattr(exchange, "STORE_LOST_AFTER", 0.6)
    started = time.monotonic()
    with pytest.raises(RoundSplitError, match="stopped answering"):
        decide_round(Gone(), b"run", 3, 0, [0, 1], deadline=time.monotonic() + 30)
    assert Gone.calls == 1
    # Waited for the store to be given up, and not for the round's deadline.
    assert 0.5 <= time.monotonic() - started < 15


def test_a_deadline_before_the_store_is_given_up_still_ends_the_wait(monkeypatch):
    from ravex._dist import exchange
    from ravex._dist.exchange import RoundSplitError, decide_round

    class DistNetworkError(RuntimeError):
        pass

    class Gone:
        def compare_set(self, key, expected, desired):
            raise DistNetworkError("Broken pipe")

    monkeypatch.setattr(exchange, "STORE_LOST_AFTER", 30.0)
    started = time.monotonic()
    with pytest.raises(RoundSplitError, match="could not read round 3"):
        decide_round(Gone(), b"run", 3, 0, [0, 1], deadline=time.monotonic() + 0.5)
    assert time.monotonic() - started < 15
