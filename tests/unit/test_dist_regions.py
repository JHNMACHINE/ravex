"""GPU-143: a round over regions, the slow link carrying one aggregate per region.

Real sockets on loopback and a store made of a dictionary, as in
``test_dist_exchange.py`` and for the same reason: the claims are about what
the network does - who fetches from whom, what happens when a delegate is not
there - and a stub standing in for the network would be the thing deciding.

What has to hold:

* every node ends the round with the same parameters, **bit for bit**, and
  those equal the flat round's over the same deltas up to the order of the
  additions - the one thing a hierarchy changes;
* only delegates fetch on the delegates' exchange: the slow link carries an
  aggregate per region, never a node's delta;
* a region is in or out whole, and a node whose delegate cannot serve the
  result takes it from another region's.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("moonclip")


from ravex._dist.exchange import (  # noqa: E402
    DeltaExchange, announce_region,
)
from ravex._dist.outer import (  # noqa: E402
    Contribution, combine, combine_partials, partial,
)

from test_dist_exchange import FakeStore  # noqa: E402

REGIONS = {0: "eu", 1: "eu", 2: "us", 3: "us"}
STEPS = {0: 10, 1: 30, 2: 20, 3: 7}


def a_delta(rank):
    generator = torch.Generator().manual_seed(100 + rank)
    return {
        "weight": torch.randn(4, 6, generator=generator),
        "bias": torch.randn(4, generator=generator),
    }


@pytest.fixture
def store():
    return FakeStore()


class TestTheArithmetic:
    """No network: a region's partial and the combination of partials are the
    flat combination, re-associated."""

    @pytest.mark.parametrize("mode", ["mean", "step_weighted", "normalized"])
    def test_partials_combine_to_the_flat_answer(self, mode):
        contributions = {r: Contribution(delta=a_delta(r), steps=STEPS[r], node=str(r)) for r in REGIONS}
        flat = combine(list(contributions.values()), mode=mode)
        parts = []
        for region in ("eu", "us"):
            summed, tally = partial([c for r, c in contributions.items() if REGIONS[r] == region], mode)
            parts.append((region, summed, tally))
        hierarchical = combine_partials(parts, mode)
        for name in flat:
            assert torch.allclose(flat[name], hierarchical[name], rtol=1e-6, atol=1e-6)
        # And the order the regions arrive in changes nothing at all.
        again = combine_partials(list(reversed(parts)), mode)
        assert all(torch.equal(hierarchical[n], again[n]) for n in hierarchical)

    def test_a_round_nobody_stepped_in_is_refused(self):
        summed, tally = partial([Contribution(delta=a_delta(0), steps=0)], "step_weighted")
        with pytest.raises(ValueError, match="weighs zero"):
            combine_partials([("eu", summed, tally)], "step_weighted")


class TestARegionThatCannotMeet:
    """GPU-233: a node that cannot reach one of its own region stops the run
    before its first step; one it cannot reach in another region does not."""

    def _nodes(self, store, tmp_path, ranks):
        made = {}
        for rank in ranks:
            exchange = DeltaExchange(rank, store, root=os.path.join(str(tmp_path), "r%d" % rank),
                                     node=str(rank), patience=5.0)
            assert exchange.start()
            announce_region(store, exchange.scope_base or exchange.scope, rank, REGIONS[rank])
            made[rank] = exchange
        return made

    def _check(self, exchange, peers, region):
        from types import SimpleNamespace

        from ravex._runtime import RavexRuntime

        me = SimpleNamespace(config=SimpleNamespace(outer_region=region))
        RavexRuntime._say_who_is_unreachable(me, exchange, peers)

    def test_the_runtime_does_not_train_on_alone(self, monkeypatch):
        # `_outer_loop` turns every other failure into training locally; this
        # one has to reach the training loop.
        from ravex._dist.exchange import RegionUnreachable
        from ravex._runtime import RavexRuntime

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime._outer = None
        runtime.registry = type("Registry", (), {"models": [object()]})()

        def refuse(self):
            raise RegionUnreachable("eu", 0, {1: "10.0.0.2:1"})

        monkeypatch.setattr(RavexRuntime, "_build_outer_loop", refuse)
        with pytest.raises(RegionUnreachable):
            runtime._outer_loop()
