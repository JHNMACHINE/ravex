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
import threading
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("moonclip")

import torch.nn as nn  # noqa: E402

from ravex._dist import exchange as _exchange  # noqa: E402
from ravex._dist.exchange import (  # noqa: E402
    DELEGATES_NAMESPACE, REGION_NAMESPACE, RESULTS_NAMESPACE,
    DeltaExchange, announce_region, close_round_regions,
)
from ravex._dist.outer import (  # noqa: E402
    Contribution, OuterLoop, OuterOptimizer, combine, combine_partials, partial,
)

from test_dist_exchange import FakeStore  # noqa: E402

REGIONS = {0: "eu", 1: "eu", 2: "us", 3: "us"}
STEPS = {0: 10, 1: 30, 2: 20, 3: 7}


def build():
    torch.manual_seed(0)
    return nn.Linear(6, 4)


def a_delta(rank):
    generator = torch.Generator().manual_seed(100 + rank)
    return {
        "weight": torch.randn(4, 6, generator=generator),
        "bias": torch.randn(4, generator=generator),
    }


def a_loop(rank, mode="mean"):
    """An outer loop whose round already happened: its model moved by this
    rank's delta, in this rank's number of steps."""
    model = build()
    loop = OuterLoop(model, inner_steps=1, combine_mode=mode, node=str(rank))
    loop.round_number = 1
    delta = a_delta(rank)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.sub_(delta[name])
    loop.steps_this_round = STEPS[rank]
    return loop


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def opened(store, tmp_path):
    made = []

    def open_node(rank):
        # The node's first exchange, as the runtime opens it for joins: the
        # region exchanges take its token and scope rather than reading them
        # again (see ``DeltaExchange.inherit``).
        main = DeltaExchange(rank, store, root=os.path.join(str(tmp_path), "r%d" % rank, "main"),
                             node=str(rank), patience=5.0)
        assert main.start()
        made.append(main)
        roles = {
            "region": REGION_NAMESPACE % REGIONS[rank],
            "delegates": DELEGATES_NAMESPACE,
            "results": RESULTS_NAMESPACE,
        }
        exchanges = {}
        for role, namespace in roles.items():
            exchange = DeltaExchange(
                rank, store, root=os.path.join(str(tmp_path), "r%d" % rank, role),
                node=str(rank), patience=5.0, namespace=namespace, inherit=main,
            )
            assert exchange.start(), "rank %d could not open its %s exchange" % (rank, role)
            made.append(exchange)
            exchanges[role] = exchange
        announce_region(store, exchanges["region"].scope_base, rank, REGIONS[rank])
        return exchanges

    yield open_node
    for exchange in made:
        exchange.close()


def run_round(nodes, loops, ranks, deadline_seconds=8.0):
    """Every node of ``ranks`` closes the round at once; their reports."""
    reports, errors = {}, {}

    def close(rank):
        try:
            peers = [r for r in ranks if r != rank]
            reports[rank] = close_round_regions(
                loops[rank], REGIONS[rank], nodes[rank], peers,
                time.monotonic() + deadline_seconds,
            )
        except Exception as exc:  # pragma: no cover - shown by the assert below
            errors[rank] = exc

    threads = [threading.Thread(target=close, args=(rank,)) for rank in ranks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors, errors
    return reports


def flat_outer(ranks, mode="mean"):
    """What the flat round would make of the same deltas."""
    outer = {name: p.detach().clone().float() for name, p in build().named_parameters()}
    gradient = combine(
        [Contribution(delta=a_delta(r), steps=STEPS[r], node=str(r)) for r in ranks], mode=mode
    )
    OuterOptimizer().step(outer, gradient)
    return outer


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


class TestTheRound:
    def test_every_node_holds_the_same_model_and_the_flat_rounds(self, opened):
        nodes = {rank: opened(rank) for rank in REGIONS}
        loops = {rank: a_loop(rank) for rank in REGIONS}
        reports = run_round(nodes, loops, list(REGIONS))

        reference = loops[0].outer
        for rank in REGIONS:
            for name in reference:
                assert torch.equal(loops[rank].outer[name], reference[name]), (rank, name)
        expected = flat_outer(list(REGIONS))
        for name in reference:
            assert torch.allclose(reference[name], expected[name], rtol=1e-6, atol=1e-6)
        for rank, report in reports.items():
            assert report["members"] == [0, 1, 2, 3]
            assert report["regions"] == ["eu", "us"]
            assert report["delegate"] == (0 if REGIONS[rank] == "eu" else 2)
            assert loops[rank].round_number == 2
            assert loops[rank].model_steps == sum(STEPS.values())
            # What the regions cost is reported apart from the region's gather.
            assert report["between_seconds"] >= 0.0
            assert report["between_wait_seconds"] >= 0.0
            # Who fetches from which exchange, for the linger on the way out.
            assert report["region_members"] == [r for r in REGIONS if REGIONS[r] == REGIONS[rank]]
            assert report["delegates"] == ([0, 2] if rank in (0, 2) else [])

    def test_only_delegates_cross_regions(self, opened, monkeypatch):
        nodes = {rank: opened(rank) for rank in REGIONS}
        loops = {rank: a_loop(rank) for rank in REGIONS}
        fetches = []
        real = DeltaExchange.fetch

        def counting(self, peer, *args, **kwargs):
            fetches.append((self.namespace, self.rank, peer))
            return real(self, peer, *args, **kwargs)

        monkeypatch.setattr(DeltaExchange, "fetch", counting)
        run_round(nodes, loops, list(REGIONS))

        across = {(rank, peer) for namespace, rank, peer in fetches if namespace == DELEGATES_NAMESPACE}
        assert across == {(0, 2), (2, 0)}
        # Within a region, nobody fetches from the other one.
        for namespace, rank, peer in fetches:
            if namespace.startswith("region/"):
                assert REGIONS[rank] == REGIONS[peer]

    def test_a_missing_node_leaves_its_region_and_everyone_agrees(self, opened):
        present = [0, 1, 3]
        nodes = {rank: opened(rank) for rank in present}
        loops = {rank: a_loop(rank) for rank in present}
        # Rank 2 said where it was and never came: the others wait for it to
        # the deadline, and then the round closes without it.
        announce_region(nodes[0]["region"].store, nodes[0]["region"].scope_base, 2, "us")

        def close(rank, reports):
            peers = [r for r in REGIONS if r != rank]
            reports[rank] = close_round_regions(
                loops[rank], REGIONS[rank], nodes[rank], peers, time.monotonic() + 4.0
            )

        reports = {}
        threads = [threading.Thread(target=close, args=(r, reports)) for r in present]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)

        assert {r: reports[r]["members"] for r in present} == {r: [0, 1, 3] for r in present}
        assert reports[3]["delegate"] == 3
        for rank in present:
            for name in loops[0].outer:
                assert torch.equal(loops[rank].outer[name], loops[0].outer[name])
        expected = flat_outer(present)
        for name in expected:
            assert torch.allclose(loops[0].outer[name], expected[name], rtol=1e-6, atol=1e-6)

    def test_a_delegate_that_cannot_serve_the_result_is_routed_around(self, opened, monkeypatch, caplog):
        nodes = {rank: opened(rank) for rank in REGIONS}
        loops = {rank: a_loop(rank) for rank in REGIONS}
        # Rank 2 delegates "us" and takes part in everything except serving
        # the result: its region has to take it from "eu"'s delegate.
        monkeypatch.setattr(nodes[2]["results"], "publish", lambda *args, **kwargs: None)
        reports = run_round(nodes, loops, list(REGIONS), deadline_seconds=6.0)

        for rank in REGIONS:
            for name in loops[0].outer:
                assert torch.equal(loops[rank].outer[name], loops[0].outer[name]), rank
        assert reports[3]["members"] == [0, 1, 2, 3]
        assert "took it from rank 0 in another region" in caplog.text
