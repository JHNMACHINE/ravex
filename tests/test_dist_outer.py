"""GPU-114: the outer loop, proved on one process.

Two kinds of test here, and they answer different questions.

The first kind is about the *rules*: the sign of the pseudo-gradient, what the
three combine modes actually compute, that a node missing from a round is not
an error, that the outer momentum survives a checkpoint. Those are arithmetic
and they are pinned exactly.

The second kind is the one the issue is finished by, and it is not a unit test
in spirit: **does a model trained by several simulated nodes exchanging deltas
every H steps actually converge**, and does it land near a single node that saw
the same data. That question cannot be answered by asserting a shape. It is
``test_two_nodes_converge_like_one`` and the ones under it, and they train a
real (tiny) model on a real (tiny) problem.

Nothing here touches ``torch.distributed``. The module under test has no
transport in it, which is what lets these run in one process with no
rendezvous — the same reason ``agreement`` is testable apart from
``collectives``.
"""

import copy

import pytest

torch = pytest.importorskip("torch")

from ravex._dist.outer import (  # noqa: E402
    COMBINE_MODES,
    Contribution,
    OuterLoop,
    OuterOptimizer,
    combine,
    float_buffers,
    pseudo_gradient,
    snapshot,
    trainable,
)


def tiny_model(seed=0, features=4, hidden=8):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(features, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )


def contribution(values, steps=1, node=""):
    """A contribution whose delta is one named tensor, for the arithmetic."""
    return Contribution(
        delta={"w": torch.tensor(values, dtype=torch.float32)}, steps=steps, node=node
    )


# --------------------------------------------------------------------------
# what the loop is responsible for


def test_frozen_parameters_are_not_the_loops_business():
    model = tiny_model()
    model[0].weight.requires_grad_(False)
    names = [name for name, _ in trainable(model)]
    assert "0.weight" not in names
    assert "0.bias" in names


def test_snapshot_is_fp32_and_a_copy():
    model = tiny_model().to(torch.bfloat16)
    taken = snapshot(model)
    assert all(t.dtype is torch.float32 for t in taken.values())

    with torch.no_grad():
        model[0].bias.add_(1.0)
    assert not torch.allclose(
        taken["0.bias"], model[0].bias.detach().to(torch.float32)
    )


def test_float_buffers_are_reported_not_averaged(caplog):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
    assert "0.running_mean" in float_buffers(model)

    with caplog.at_level("WARNING", logger="ravex"):
        OuterLoop(model, inner_steps=1)
    assert "not exchanged between rounds" in caplog.text


def test_a_buffer_no_checkpoint_carries_is_not_something_to_warn_about():
    """A causal attention mask is the common case, and it cannot drift.

    Non-persistent buffers are not in the state dict, which is the module's own
    way of saying they are derived rather than state. Warning about them means
    every transformer built the ordinary way gets a line about parameters
    parting company that names the one thing in the model that is identical on
    every node by construction.
    """

    class Masked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.register_buffer("mask", torch.zeros(3, 3), persistent=False)
            self.register_buffer("carried", torch.zeros(3))

    model = Masked()
    reported = float_buffers(model)
    assert "mask" not in reported
    assert "carried" in reported


# --------------------------------------------------------------------------
# the pseudo-gradient, whose sign is the whole trick


def test_pseudo_gradient_points_the_way_a_gradient_points():
    model = tiny_model()
    outer = snapshot(model)
    with torch.no_grad():
        model[0].bias.sub_(0.5)  # local training moved this parameter down

    delta = pseudo_gradient(outer, model)
    assert torch.allclose(delta["0.bias"], torch.full_like(delta["0.bias"], 0.5))

    # Descending it has to follow the model down, not away from it.
    OuterOptimizer(lr=1.0, momentum=0.0, nesterov=False).step(outer, delta)
    assert torch.allclose(outer["0.bias"], model[0].bias.detach().to(torch.float32))


def test_a_model_rebuilt_mid_round_raises_instead_of_updating_half():
    model = tiny_model()
    outer = snapshot(model)
    model[0].weight.requires_grad_(False)

    with pytest.raises(KeyError, match="no longer has"):
        pseudo_gradient(outer, model)


# --------------------------------------------------------------------------
# combine: the three modes, and what separates them


def test_mean_is_the_plain_average_over_whoever_showed_up():
    got = combine([contribution([2.0]), contribution([4.0])], mode="mean")
    assert got["w"].item() == pytest.approx(3.0)


def test_a_dead_node_is_an_absence_and_not_an_error():
    """The whole fault tolerance of GPU-113 is that this is a shorter list."""
    everyone = [contribution([2.0], node="a"), contribution([4.0], node="b")]
    survivors = everyone[:1]
    assert combine(survivors)["w"].item() == pytest.approx(2.0)


def test_step_weighted_counts_a_fast_node_quadratically():
    """Why it is not the default, stated as arithmetic instead of an opinion.

    Both nodes move 1.0 per step; one takes 3 steps and the other 1. Under the
    plain mean the answer is the average of what they did, 2.0. Under
    ``step_weighted`` the fast node is counted once in the size of its delta
    and again in its weight, and the answer is 2.5.
    """
    fast = contribution([3.0], steps=3)
    slow = contribution([1.0], steps=1)

    assert combine([fast, slow], mode="mean")["w"].item() == pytest.approx(2.0)
    assert combine([fast, slow], mode="step_weighted")["w"].item() == pytest.approx(2.5)


def test_normalized_gives_every_node_one_vote_at_the_scale_of_a_round():
    """Per-step movement averaged evenly, then scaled back up to a round.

    Same two nodes: per-step they both move 1.0, so one node one vote gives
    1.0 per step, and the average round was 2 steps long, so 2.0.
    """
    fast = contribution([3.0], steps=3)
    slow = contribution([1.0], steps=1)
    assert combine([fast, slow], mode="normalized")["w"].item() == pytest.approx(2.0)


def test_the_modes_agree_when_every_node_did_the_same_work():
    """They differ only in how they treat heterogeneity, and this pins that."""
    same = [contribution([2.0], steps=5), contribution([4.0], steps=5)]
    answers = [combine(same, mode=mode)["w"].item() for mode in COMBINE_MODES]
    assert answers == pytest.approx([3.0, 3.0, 3.0])


def test_an_empty_round_is_not_an_update_of_zero():
    with pytest.raises(ValueError, match="nobody reported"):
        combine([])


def test_contributions_covering_different_models_do_not_get_intersected():
    mine = contribution([1.0], node="a")
    theirs = Contribution(delta={"other": torch.zeros(1)}, steps=1, node="b")
    with pytest.raises(KeyError, match="different parameters"):
        combine([mine, theirs])


def test_the_same_parameters_in_a_different_order_are_the_same_parameters():
    """GPU-122, and it cost every round of a real run before it was found.

    A node's own delta is built in ``named_parameters()`` order and never
    round-trips; a peer's has been written to a store and read back, and comes
    back in the store's order. On a two-parameter model the two coincide, which
    is why every test here passed while a 96-parameter model abandoned **every**
    round - caught, warned about, and left with each node training alone as the
    loss kept falling.
    """
    mine = Contribution(
        delta={"a": torch.ones(1), "b": torch.ones(1) * 3},
        steps=1, node="0",
    )
    theirs = Contribution(
        delta={"b": torch.ones(1) * 5, "a": torch.ones(1) * 3},
        steps=1, node="1",
    )

    combined = combine([mine, theirs])

    # Averaged by name and not by position: getting this wrong would pair "a"
    # with "b" and produce numbers that look plausible.
    assert combined["a"].item() == pytest.approx(2.0)
    assert combined["b"].item() == pytest.approx(4.0)


def test_the_outer_state_can_be_kept_off_the_accelerator():
    """GPU-124, and the honest note about what this test can and cannot prove.

    A round report is a moonclip snapshot and moonclip reads host memory, so an
    outer state living on the accelerator is not publishable at all: with the
    model on CUDA the outer loop raised before the seed round, Ravex said so,
    and the run carried on training locally - two boxes holding two models
    rather than a degraded one. The cause was ``_build_outer_loop`` never
    passing ``device``, so the snapshot was born wherever the model was.

    **This test cannot reproduce that**, and neither can any other test in this
    repository: there is no GPU here, so ``device=None`` and ``device="cpu"``
    are the same thing and always were. That is exactly why nothing caught it.
    What it does pin is the property the fix relies on - the snapshot and the
    contribution follow ``device`` rather than the model - and the proof that
    it works on an accelerator is the rented pair of Blackwell boxes on
    2026-09-10: four rounds closed over two machines with ``device cuda``, at
    0.78-0.85 s of network each, indistinguishable from the same run on CPU.
    """
    model = torch.nn.Linear(4, 4)
    loop = OuterLoop(model, inner_steps=1, device="cpu", node="a")

    assert all(value.device.type == "cpu" for value in loop.outer.values())

    loop.record_step()
    delta = loop.contribution().delta
    assert delta, "a contribution over no parameters is not a contribution"
    assert all(value.device.type == "cpu" for value in delta.values())


def test_a_round_where_nobody_stepped_is_refused_under_the_weighted_modes():
    idle = [contribution([0.0], steps=0), contribution([0.0], steps=0)]
    with pytest.raises(ValueError, match="nobody took a step"):
        combine(idle, mode="step_weighted")


def test_an_unknown_mode_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown combine mode"):
        combine([contribution([1.0])], mode="weighted")


# --------------------------------------------------------------------------
# the outer optimizer


def test_momentum_carries_a_direction_between_rounds():
    outer = {"w": torch.zeros(1)}
    optimizer = OuterOptimizer(lr=1.0, momentum=0.9, nesterov=False)
    grad = {"w": torch.ones(1)}

    optimizer.step(outer, grad)
    after_one = outer["w"].item()
    optimizer.step(outer, grad)
    after_two = outer["w"].item()

    # The second identical round moves further than the first, which is the
    # buffer doing its job and not an accumulation artefact.
    assert (after_one - after_two) > abs(after_one) * 1.5


def test_without_momentum_two_identical_rounds_move_the_same():
    outer = {"w": torch.zeros(1)}
    optimizer = OuterOptimizer(lr=0.5, momentum=0.0, nesterov=False)
    optimizer.step(outer, {"w": torch.ones(1)})
    first = outer["w"].item()
    optimizer.step(outer, {"w": torch.ones(1)})
    assert (outer["w"].item() - first) == pytest.approx(first)


def test_nesterov_needs_a_momentum_to_look_ahead_with():
    with pytest.raises(ValueError, match="nesterov needs a momentum"):
        OuterOptimizer(momentum=0.0, nesterov=True)


def test_the_momentum_buffer_survives_a_checkpoint():
    """A resume that drops it looks healthy and re-derives it for free rounds."""
    original = OuterOptimizer(lr=0.7, momentum=0.9)
    outer = {"w": torch.zeros(1)}
    original.step(outer, {"w": torch.ones(1)})

    restored = OuterOptimizer(lr=0.1, momentum=0.1)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))

    assert restored.lr == pytest.approx(0.7)
    assert restored.rounds == 1
    kept, mine = {"w": torch.zeros(1)}, {"w": torch.zeros(1)}
    restored.step(kept, {"w": torch.ones(1)})
    original.step(mine, {"w": torch.ones(1)})
    assert kept["w"].item() == pytest.approx(mine["w"].item())


# --------------------------------------------------------------------------
# the loop itself


def test_a_round_needs_something_that_ends_it():
    with pytest.raises(ValueError, match="needs a round that ends"):
        OuterLoop(tiny_model())


def test_the_round_closes_on_the_count():
    loop = OuterLoop(tiny_model(), inner_steps=3)
    for _ in range(2):
        loop.record_step()
        assert not loop.round_is_over()
    loop.record_step()
    assert loop.round_is_over()


def test_the_round_closes_on_the_clock_whatever_the_count(monkeypatch):
    """The clock is what lets a slow node stop at the same moment as a fast one."""
    now = [1000.0]
    monkeypatch.setattr("ravex._dist.outer.time.monotonic", lambda: now[0])

    loop = OuterLoop(tiny_model(), round_seconds=10.0)
    loop.record_step()
    assert not loop.round_is_over()
    now[0] += 10.0
    assert loop.round_is_over()
    assert loop.contribution().steps == 1


def test_apply_writes_the_outer_parameters_into_the_live_model():
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=1, lr=1.0, momentum=0.0, nesterov=False)
    with torch.no_grad():
        model[0].bias.add_(1.0)
    loop.record_step()

    report = loop.apply([loop.contribution()])

    assert report["nodes"] == 1
    assert report["round"] == 0
    assert loop.round_number == 1
    assert loop.steps_this_round == 0
    # One node, lr 1.0, no momentum: the outer state lands exactly where the
    # local training went, and the model is left holding it.
    assert torch.allclose(model[0].bias, loop.outer["0.bias"])


def test_the_report_says_how_far_apart_the_nodes_were():
    loop = OuterLoop(tiny_model(), inner_steps=1)
    loop.record_step()
    mine = loop.contribution()
    theirs = Contribution(delta=copy.deepcopy(mine.delta), steps=17, node="b")
    report = loop.apply([mine, theirs])
    assert (report["slowest"], report["fastest"]) == (1, 17)


def test_a_bf16_model_keeps_fp32_outer_state():
    model = tiny_model().to(torch.bfloat16)
    loop = OuterLoop(model, inner_steps=1)
    loop.record_step()
    loop.apply([loop.contribution()])
    assert loop.outer["0.bias"].dtype is torch.float32
    assert model[0].bias.dtype is torch.bfloat16


def test_the_loop_resumes_where_it_stopped():
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=2, node="a")
    with torch.no_grad():
        model[0].bias.add_(0.3)
    loop.record_step()
    loop.apply([loop.contribution()])
    saved = copy.deepcopy(loop.state_dict())

    fresh = OuterLoop(tiny_model(seed=1), inner_steps=2)
    fresh.load_state_dict(saved)
    assert fresh.round_number == 1
    assert fresh.optimizer.rounds == 1
    assert torch.allclose(fresh.outer["0.bias"], loop.outer["0.bias"])


# --------------------------------------------------------------------------
# the question the issue is actually finished by


def a_problem(n=256, features=4, seed=7):
    """A linear target with noise: small, and it has a right answer."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, features, generator=generator)
    truth = torch.randn(features, 1, generator=generator)
    y = x @ truth + 0.05 * torch.randn(n, 1, generator=generator)
    return x, y


def train_locally(model, x, y, steps, lr=0.05, start=0, optimizer=None):
    """Plain SGD over a fixed shard, returning the optimizer for reuse."""
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    batch = 32
    for step in range(steps):
        begin = ((start + step) * batch) % len(x)
        xb, yb = x[begin : begin + batch], y[begin : begin + batch]
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()
    return optimizer


def final_loss(model, x, y):
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x), y).item()


def run_nodes(shards, rounds, inner_steps, combine_mode="mean", speeds=None):
    """Several nodes on one process, each with its own shard and optimizer.

    ``speeds`` scales each node's inner steps, which is how a heterogeneous
    round is simulated without actually being slow.
    """
    model = tiny_model(seed=3)
    loops = []
    models = []
    optimizers = []
    for index, _ in enumerate(shards):
        clone = copy.deepcopy(model)
        models.append(clone)
        optimizers.append(None)
        loops.append(
            OuterLoop(
                clone,
                inner_steps=inner_steps,
                combine_mode=combine_mode,
                node=str(index),
            )
        )

    speeds = speeds or [1.0] * len(shards)
    for round_number in range(rounds):
        contributions = []
        for index, (x, y) in enumerate(shards):
            steps = max(1, int(inner_steps * speeds[index]))
            optimizers[index] = train_locally(
                models[index],
                x,
                y,
                steps,
                start=round_number * steps,
                optimizer=optimizers[index],
            )
            for _ in range(steps):
                loops[index].record_step()
            contributions.append(loops[index].contribution())
        for loop in loops:
            loop.apply(contributions)
    return models, loops


def two_shards():
    x, y = a_problem(n=512)
    half = len(x) // 2
    return [(x[:half], y[:half]), (x[half:], y[half:])], (x, y)


def test_two_nodes_converge_like_one():
    """The claim the whole design rests on, measured rather than asserted.

    Two nodes, each seeing half the data, exchanging deltas every 20 steps for
    10 rounds. The comparison is a single node that took the same *total*
    number of optimizer steps over all the data — the honest baseline, because
    it is what the same compute buys without any network at all.
    """
    shards, (x, y) = two_shards()
    rounds, inner = 10, 20

    models, loops = run_nodes(shards, rounds=rounds, inner_steps=inner)
    together = final_loss(models[0], x, y)

    alone = tiny_model(seed=3)
    train_locally(alone, x, y, steps=rounds * inner)
    baseline = final_loss(alone, x, y)

    start = final_loss(tiny_model(seed=3), x, y)

    assert together < start / 2, (
        "the outer loop did not train: %.4f from a start of %.4f" % (together, start)
    )
    assert together < baseline * 2.0, (
        "two nodes landed far off a single node with the same step budget: "
        "%.4f against %.4f" % (together, baseline)
    )


def test_every_node_holds_the_same_model_after_a_round():
    """Not an optimisation — nodes that disagree are training two models."""
    shards, _ = two_shards()
    models, _ = run_nodes(shards, rounds=3, inner_steps=10)
    for name, left in models[0].named_parameters():
        right = dict(models[1].named_parameters())[name]
        assert torch.allclose(left, right, atol=1e-6), name


def test_a_node_five_times_slower_does_not_hold_the_others_back():
    """The point of the wall-clock round, checked on a loss and not a log line.

    One node does a fifth of the steps of the other, every round, which is what
    a box in another region on cheaper hardware looks like. The run has to
    still converge, and the fast node's contribution has to still count for
    more than the slow one's — which under the plain mean it does by itself,
    because its delta is bigger.
    """
    shards, (x, y) = two_shards()
    models, _ = run_nodes(
        shards, rounds=10, inner_steps=20, speeds=[1.0, 0.2]
    )
    start = final_loss(tiny_model(seed=3), x, y)
    assert final_loss(models[0], x, y) < start / 2


def test_a_node_that_dies_mid_run_does_not_stop_the_round():
    """No collective to hang in, so this is a shorter list and nothing else."""
    shards, (x, y) = two_shards()
    model = tiny_model(seed=3)
    survivor = copy.deepcopy(model)
    loop = OuterLoop(survivor, inner_steps=10, node="survivor")
    optimizer = None

    for round_number in range(6):
        optimizer = train_locally(
            survivor, *shards[0], 10, start=round_number * 10, optimizer=optimizer
        )
        for _ in range(10):
            loop.record_step()
        contributions = [loop.contribution()]
        if round_number < 3:
            # The other node was there for the first three rounds and then went
            # away, with no notice and no cleanup.
            other = Contribution(
                delta=copy.deepcopy(contributions[0].delta), steps=10, node="departed"
            )
            contributions.append(other)
        report = loop.apply(contributions)
        assert report["nodes"] == (2 if round_number < 3 else 1)

    assert final_loss(survivor, x, y) < final_loss(tiny_model(seed=3), x, y) / 2
