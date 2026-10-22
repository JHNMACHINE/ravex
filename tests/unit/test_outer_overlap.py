"""GPU-230: a round in flight while the next one trains, proved on one process.

The exchange is not here: a round in flight goes through the same
``close_round`` a blocking one does, and ``tests/distributed`` runs that over
sockets. What is here is what overlap adds - :meth:`OuterLoop.launch`,
:class:`RoundInFlight`, :meth:`OuterLoop.land` - and the three things it must
not break: every node holding the same outer parameters, a node's own training
counted once, and the run still converging.
"""

import copy

import pytest

torch = pytest.importorskip("torch")

from ravex._dist.outer import Contribution, OuterLoop, RoundInFlight  # noqa: E402

from test_dist_outer import (  # noqa: E402
    final_loss,
    tiny_model,
    train_locally,
    two_shards,
)


def nudge(model, by):
    """Some local training, made deterministic: every parameter moves by ``by``."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(by)


def params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_a_flight_landed_at_once_is_a_blocking_round():
    """Nothing trained between launch and land: the model is the outer
    parameters, bit for bit, as after ``apply``."""
    blocking_model, overlap_model = tiny_model(), tiny_model()
    blocking = OuterLoop(blocking_model, inner_steps=1, node="a")
    overlap = OuterLoop(overlap_model, inner_steps=1, node="a")
    for model in (blocking_model, overlap_model):
        nudge(model, 0.25)
    blocking.record_step()
    overlap.record_step()

    blocking.apply([blocking.contribution()])
    flight = overlap.launch()
    flight.apply([flight.contribution()])
    overlap.land(flight)

    for name, value in params(blocking_model).items():
        assert torch.equal(value, params(overlap_model)[name]), name
        assert torch.equal(blocking.outer[name], overlap.outer[name]), name
    assert overlap.round_number == blocking.round_number == 1


def test_land_keeps_what_the_model_trained_while_the_round_was_away():
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=1, node="a")
    nudge(model, 0.25)
    loop.record_step()
    flight = loop.launch()
    photograph = {name: value.clone() for name, value in flight.photograph.items()}

    nudge(model, -0.1)  # the next round's training, while this one travels
    live = params(model)
    flight.apply([flight.contribution()])
    loop.land(flight)

    for name, p in model.named_parameters():
        expected = loop.outer[name] + (live[name] - photograph[name])
        assert torch.allclose(p.detach(), expected, atol=1e-6), name


def test_the_training_in_flight_goes_into_the_next_delta_once():
    """After landing, the next delta is minus the training since the
    photograph - not that plus the round that just landed."""
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=1, node="a")
    nudge(model, 0.25)
    loop.record_step()
    flight = loop.launch()
    nudge(model, -0.1)
    flight.apply([flight.contribution()])
    loop.land(flight)

    delta = loop.contribution().delta
    for name, value in delta.items():
        assert torch.allclose(value, torch.full_like(value, 0.1), atol=1e-6), name


def test_the_counters_belong_to_the_round_being_trained():
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=3, node="a")
    for _ in range(3):
        loop.record_step()
    flight = loop.launch()
    assert loop.steps_this_round == 0
    assert flight.contribution().steps == 3

    loop.record_step()
    loop.record_step()
    flight.apply([flight.contribution()])
    # The outer step happened on another thread's behalf; the round open on
    # this one keeps its two steps.
    assert loop.steps_this_round == 2
    assert loop.round_number == 1
    assert loop.model_steps == 3


def test_nodes_that_trained_different_amounts_meanwhile_hold_one_outer_model():
    """The invariant `decide_round` protects (GPU-140), with overlap on: the
    live models differ by what each trained since its photograph, the outer
    parameters do not differ at all."""
    models = [tiny_model(seed=3), tiny_model(seed=3)]
    loops = [OuterLoop(m, inner_steps=1, node=str(i)) for i, m in enumerate(models)]
    for round_number in range(4):
        for index, model in enumerate(models):
            nudge(model, 0.01 * (index + 1) * (round_number + 1))
            loops[index].record_step()
        flights = [loop.launch() for loop in loops]
        # Different amounts of training while the round travels.
        nudge(models[0], 0.003)
        nudge(models[1], -0.02)
        contributions = [flight.contribution() for flight in flights]
        for flight in flights:
            flight.apply(contributions)
        for loop, flight in zip(loops, flights):
            loop.land(flight)
        for name in loops[0].outer:
            assert torch.equal(loops[0].outer[name], loops[1].outer[name]), (round_number, name)


def test_an_abandoned_flight_leaves_the_model_alone():
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=1, node="a")
    outer = {name: value.clone() for name, value in loop.outer.items()}
    nudge(model, 0.25)
    loop.record_step()
    flight = loop.launch()
    nudge(model, 0.1)
    live = params(model)
    loop.record_step()

    flight.abandon_round()
    loop.land(flight)

    assert loop.round_number == 1
    assert loop.steps_this_round == 1
    for name, value in params(model).items():
        assert torch.equal(value, live[name]), name
        assert torch.equal(loop.outer[name], outer[name]), name


def test_a_delta_that_could_not_be_taken_fails_inside_the_round():
    """Where `close_round` catches it and offers nothing, as for a blocking
    round - not at the launch, on the training thread."""
    model = tiny_model()
    loop = OuterLoop(model, inner_steps=1, node="a")
    loop.outer["gone"] = torch.zeros(1)
    loop.record_step()
    flight = loop.launch()
    with pytest.raises(KeyError):
        flight.contribution()
    flight.abandon_round()
    assert loop.round_number == 1


def test_a_flight_answers_what_close_round_asks_a_loop():
    loop = OuterLoop(tiny_model(), inner_steps=1, node="a", combine_mode="normalized")
    loop.record_step()
    flight = loop.launch()
    assert isinstance(flight, RoundInFlight)
    assert flight.round_number == 0
    assert flight.outer is loop.outer
    assert flight.combine_mode == "normalized"
    assert isinstance(flight.contribution(), Contribution)
    assert flight.with_residual({"x": 1}) == {"x": 1}


def run_overlapped(shards, rounds, inner_steps, flight_steps):
    """Two nodes, each landing its round ``flight_steps`` steps after launch."""
    base = tiny_model(seed=3)
    models = [copy.deepcopy(base) for _ in shards]
    loops = [OuterLoop(m, inner_steps=inner_steps, node=str(i)) for i, m in enumerate(models)]
    optimizers = [None] * len(shards)
    flights = None
    for round_number in range(rounds):
        for index, (x, y) in enumerate(shards):
            start = round_number * inner_steps
            optimizers[index] = train_locally(
                models[index], x, y, flight_steps if flights else inner_steps,
                start=start, optimizer=optimizers[index],
            )
        if flights:
            for loop, flight in zip(loops, flights):
                loop.land(flight)
            for index, (x, y) in enumerate(shards):
                optimizers[index] = train_locally(
                    models[index], x, y, inner_steps - flight_steps,
                    start=round_number * inner_steps + flight_steps,
                    optimizer=optimizers[index],
                )
        for loop in loops:
            for _ in range(inner_steps):
                loop.record_step()
        flights = [loop.launch() for loop in loops]
        contributions = [flight.contribution() for flight in flights]
        for flight in flights:
            flight.apply(contributions)
    for loop, flight in zip(loops, flights or []):
        loop.land(flight)
    return models, loops


def test_two_overlapped_nodes_still_converge():
    """Late by a quarter of a round, every round: still training, still near a
    single node with the same step budget."""
    shards, (x, y) = two_shards()
    rounds, inner = 10, 20
    models, _ = run_overlapped(shards, rounds, inner, flight_steps=5)
    together = final_loss(models[0], x, y)

    alone = tiny_model(seed=3)
    train_locally(alone, x, y, steps=rounds * inner)
    baseline = final_loss(alone, x, y)
    start = final_loss(tiny_model(seed=3), x, y)

    assert together < start / 2, "%.4f from a start of %.4f" % (together, start)
    assert together < baseline * 2.0, "%.4f against %.4f" % (together, baseline)
