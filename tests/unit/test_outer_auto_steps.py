"""GPU-231: H moved by the link, the same on every node, proved on one process.

The rule is :func:`next_inner_steps`, a function of one round's clock; the
agreement is :func:`decide_inner_steps` on a store made of a dictionary, the
same fake ``test_dist_exchange`` uses; and where a new H lands is
:meth:`OuterLoop.set_inner_steps`. ``tests/distributed`` runs it through the
runtime on two ranks.
"""

import copy

import pytest

torch = pytest.importorskip("torch")

from ravex._config import RavexConfig  # noqa: E402
from ravex._dist.exchange import decide_inner_steps  # noqa: E402
from ravex._dist.membership import apply_state, state_payload  # noqa: E402
from ravex._dist.outer import OuterLoop, next_inner_steps  # noqa: E402

from test_dist_exchange import FakeStore  # noqa: E402
from test_dist_outer import tiny_model, train_locally, two_shards  # noqa: E402


def blocking(current, trained, network, steps=None, **bounds):
    return next_inner_steps(
        current,
        trained_seconds=trained,
        trained_steps=current if steps is None else steps,
        network_seconds=network,
        overlap=False,
        **bounds,
    )


def overlapped(current, trained, network, **bounds):
    return next_inner_steps(
        current, trained_seconds=trained, trained_steps=current,
        network_seconds=network, overlap=True, **bounds,
    )


# ─── the rule ───────────────────────────────────────────────────────


def test_blocking_lands_on_the_h_where_the_network_is_a_quarter_of_the_round():
    # 0.1 s a step, 30 s of network: a quarter of the round at H = 900.
    assert blocking(500, trained=50.0, network=30.0, high=5000) == 900
    # And there it stays.
    assert blocking(900, trained=90.0, network=30.0, high=5000) == 900


def test_a_network_that_leaves_room_brings_h_down():
    """A target, not a ceiling: fewer steps per round has been the better
    model wherever it was measured."""
    # 0.1 s a step, 3 s of network: a quarter at H = 90.
    assert blocking(100, trained=10.0, network=3.0) == 90


def test_one_round_moves_h_at_most_by_half_or_double():
    assert blocking(500, trained=50.0, network=600.0, high=5000) == 1000
    assert blocking(500, trained=50.0, network=0.5) == 250


def test_h_stays_inside_its_bounds():
    assert blocking(1500, trained=150.0, network=600.0, high=2000) == 2000
    assert blocking(40, trained=4.0, network=0.0, low=32) == 32


def test_noise_does_not_move_h():
    # Within a tenth of where it is: left alone.
    assert blocking(900, trained=90.0, network=32.0, high=5000) == 900


def test_overlap_wants_the_network_just_outlasting_the_training():
    """With the next round training meanwhile, the training waits only for
    what the network outlasts it by. A quarter of the round waiting is a
    network that takes 4/3 of a round's training."""
    # 0.1 s a step, 40 s of network: (40 - 0.1 H) / 40 = 1/4 at H = 300.
    assert overlapped(400, trained=40.0, network=40.0) == 300
    # A network the training hides entirely: H comes down.
    assert overlapped(400, trained=40.0, network=1.0) == 200


def test_a_round_that_measured_nothing_changes_nothing():
    assert blocking(500, trained=0.0, network=30.0) == 500
    assert blocking(500, trained=50.0, network=30.0, steps=0) == 500
    assert blocking(500, trained=50.0, network=30.0, share=1.0) == 500


# ─── where a new H lands ────────────────────────────────────────────


def test_a_blocking_round_takes_the_new_h_at_once():
    loop = OuterLoop(tiny_model(), inner_steps=4)
    loop.set_inner_steps(2, now=True)
    loop.record_step()
    loop.record_step()
    assert loop.round_is_over()


def test_with_overlap_the_new_h_waits_for_the_next_launch():
    """The round now open has been trained a different number of steps on
    each node by the time its predecessor lands; the next launch is the first
    point every node reaches with the same rounds landed."""
    loop = OuterLoop(tiny_model(), inner_steps=4)
    loop.record_step()
    loop.set_inner_steps(2, now=False)
    loop.record_step()
    assert not loop.round_is_over(), "the open round changed H mid-way"
    assert loop.inner_steps == 4
    for _ in range(2):
        loop.record_step()
    loop.launch()
    assert loop.inner_steps == 2 and loop.pending_inner_steps is None


def test_a_resume_goes_on_from_the_h_the_run_had_reached():
    loop = OuterLoop(tiny_model(), inner_steps=500)
    loop.set_inner_steps(120, now=True)
    loop.set_inner_steps(60, now=False)
    fresh = OuterLoop(tiny_model(), inner_steps=500)
    fresh.load_state_dict(loop.state_dict())
    assert (fresh.inner_steps, fresh.pending_inner_steps) == (120, 60)


def test_a_joiner_is_handed_the_h_the_run_had_reached():
    member = OuterLoop(tiny_model(), inner_steps=500)
    member.set_inner_steps(120, now=True)
    joiner = OuterLoop(tiny_model(seed=1), inner_steps=500)
    apply_state(joiner, state_payload(member))
    assert (joiner.inner_steps, joiner.pending_inner_steps) == (120, None)


def test_a_clock_round_is_not_given_a_count_by_a_joiner():
    member = OuterLoop(tiny_model(), round_seconds=10.0)
    joiner = OuterLoop(tiny_model(seed=1), round_seconds=10.0)
    apply_state(joiner, state_payload(member))
    assert joiner.inner_steps is None


# ─── the same H on every node ───────────────────────────────────────


def test_the_first_proposal_is_everyone_s():
    store = FakeStore()
    assert decide_inner_steps(store, b"job", 3, 0, 120, 0.0) == 120
    assert decide_inner_steps(store, b"job", 3, 1, 900, 0.0) == 120
    # Each round decides its own.
    assert decide_inner_steps(store, b"job", 4, 1, 900, 0.0) == 900
    # And each job.
    assert decide_inner_steps(store, b"other", 3, 1, 64, 0.0) == 64


def test_a_store_that_never_answers_raises_for_the_caller_to_keep_its_h():
    class Gone:
        def compare_set(self, key, expected, desired):
            raise RuntimeError("connection closed")

    with pytest.raises(RuntimeError):
        decide_inner_steps(Gone(), b"job", 3, 0, 120, 0.0)


def test_nodes_that_measured_differently_change_h_together_and_hold_one_model():
    """The issue's own test: each node proposes from its own clock - here
    node 1's network looks three times slower - and both take the store's
    answer at the same round. Their outer parameters stay bit for bit the
    same, and their rounds close at the same step."""
    shards, _ = two_shards()
    base = tiny_model(seed=3)
    models = [copy.deepcopy(base) for _ in shards]
    loops = [OuterLoop(m, inner_steps=64, node=str(i)) for i, m in enumerate(models)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.05) for m in models]
    store = FakeStore()
    seen = []

    for round_number in range(6):
        steps = [loop.inner_steps for loop in loops]
        assert steps[0] == steps[1], "round %d: H %s" % (round_number, steps)
        seen.append(steps[0])
        for index, (x, y) in enumerate(shards):
            train_locally(models[index], x, y, steps[index], optimizer=optimizers[index])
            for _ in range(steps[index]):
                loops[index].record_step()
        mine = [loop.contribution() for loop in loops]
        for loop in loops:
            loop.apply(mine)
        # Node 1 decides first in even rounds, node 0 in odd ones.
        order = [1, 0] if round_number % 2 == 0 else [0, 1]
        for index in order:
            proposal = next_inner_steps(
                loops[index].inner_steps,
                trained_seconds=0.01 * steps[index],
                trained_steps=steps[index],
                network_seconds=(3.0 if index == 1 else 1.0) * 0.05 * (round_number + 1),
                overlap=False,
                low=8,
                high=512,
            )
            decided = decide_inner_steps(store, b"job", round_number, index, proposal, 0.0)
            loops[index].set_inner_steps(decided, now=True)
        for name, value in loops[0].outer.items():
            assert torch.equal(value, loops[1].outer[name]), (
                "round %d: the nodes hold two models (%s)" % (round_number, name)
            )

    assert len(set(seen)) > 1, "H never moved, so nothing was tested: %s" % seen


# ─── configuration ──────────────────────────────────────────────────


def test_auto_in_place_of_a_number_turns_it_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    (tmp_path / "ravex.yaml").write_text("outer_inner_steps: auto\n", encoding="utf-8")
    config = RavexConfig.load()
    assert config.outer_auto_steps and config.outer_inner_steps == 500
    assert not config.problems


def test_auto_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.setenv("RAVEX_OUTER_INNER_STEPS", "auto")
    monkeypatch.setenv("RAVEX_OUTER_NETWORK_SHARE", "0.1")
    config = RavexConfig.load()
    assert config.outer_auto_steps and config.outer_network_share == 0.1


def test_auto_and_a_clock_round_do_not_go_together(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.setenv("RAVEX_OUTER_AUTO_STEPS", "1")
    monkeypatch.setenv("RAVEX_OUTER_ROUND_SECONDS", "60")
    config = RavexConfig.load()
    assert not config.outer_auto_steps
    assert any("outer_round_seconds" in problem for problem in config.problems)


def test_bad_bounds_and_shares_are_replaced_and_said(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.setenv("RAVEX_OUTER_MIN_INNER_STEPS", "100")
    monkeypatch.setenv("RAVEX_OUTER_MAX_INNER_STEPS", "10")
    monkeypatch.setenv("RAVEX_OUTER_NETWORK_SHARE", "1.5")
    config = RavexConfig.load()
    assert config.outer_max_inner_steps == config.outer_min_inner_steps == 100
    assert config.outer_network_share == 0.25
    assert len(config.problems) == 2
