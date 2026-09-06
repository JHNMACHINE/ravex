"""GPU-115: a round report through moonclip, and what a reader refuses.

This replaces a hand-rolled wire format that should not have been written. The
tests that survived the change are the ones about *refusing*, because those are
about this node's own rules rather than about an encoding: a report arrives from
another machine, and before any of it is materialised its names, shapes and
dtypes are checked against the ones this node already holds.

What got better in the move is where that check happens. ``describe()`` answers
from the manifest without touching a tensor, so a peer training a different
model is refused having cost a manifest read — not an allocation sized by a
number the peer chose.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("moonclip")

from ravex._dist import report as _report  # noqa: E402
from ravex._dist.report import (  # noqa: E402
    ReportError,
    expectation,
    open_store,
    read,
    write,
)


def a_delta():
    torch.manual_seed(0)
    return {"layer.weight": torch.randn(3, 4), "layer.bias": torch.randn(3)}


@pytest.fixture
def store(tmp_path):
    return open_store(str(tmp_path / "mine"))


def test_a_report_survives_the_round_trip_exactly(store):
    delta = a_delta()
    write(store, delta, round_number=7, steps=13, node="turin")

    report = read(store, 7, expectation(delta))

    assert report.round_number == 7
    assert report.steps == 13
    assert report.node == "turin"
    for name, tensor in delta.items():
        assert torch.equal(report.delta[name], tensor), name


def test_a_round_is_looked_up_by_number_and_not_by_being_newest(store):
    """A peer may be a round ahead by the time it answers, and still owe this one."""
    delta = a_delta()
    write(store, delta, round_number=4, steps=10, node="n0")
    write(store, {k: v * 2 for k, v in delta.items()}, round_number=5, steps=11, node="n0")

    fourth = read(store, 4, expectation(delta))
    assert fourth.steps == 10
    assert torch.equal(fourth.delta["layer.bias"], delta["layer.bias"])


def test_a_round_that_was_never_written_is_not_there(store):
    write(store, a_delta(), round_number=0, steps=1, node="n0")
    with pytest.raises(ReportError, match="no snapshot for round 3"):
        read(store, 3, expectation(a_delta()))


def test_several_rounds_are_kept_so_serving_cannot_race_retention(store):
    delta = a_delta()
    for round_number in range(3):
        write(store, delta, round_number, steps=1, node="n0")
    steps = {int(s["step"]) for s in store.list_snapshots()}
    assert {1, 2} <= steps, "a peer one round behind has nothing to fetch"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
def test_the_float_dtypes_a_delta_can_be_round_trip(store, dtype):
    delta = {"t": torch.arange(6).reshape(2, 3).to(dtype)}
    write(store, delta, 0, 1, "n0")
    report = read(store, 0, expectation(delta))
    assert torch.equal(report.delta["t"], delta["t"])


def test_an_empty_tensor_is_carried_rather_than_skipped(store):
    delta = {"t": torch.zeros(0), "u": torch.randn(2)}
    write(store, delta, 0, 1, "n0")
    report = read(store, 0, expectation(delta))
    assert report.delta["t"].shape == (0,)


# --------------------------------------------------------------------------
# what a reader refuses, which is the point of the module


def test_a_parameter_this_node_does_not_have_is_refused(store):
    theirs = {"layer.weight": torch.randn(3, 4), "sneaky": torch.randn(3)}
    write(store, theirs, 0, 1, "n0")
    with pytest.raises(ReportError, match="does not have"):
        read(store, 0, expectation(a_delta()))


def test_a_shape_that_disagrees_is_refused_before_a_tensor_is_touched(store):
    write(store, {"w": torch.randn(256, 256)}, 0, 1, "n0")
    with pytest.raises(ReportError, match="arrived as"):
        read(store, 0, expectation({"w": torch.randn(2, 2)}))


def test_a_dtype_that_disagrees_is_refused(store):
    write(store, {"w": torch.randn(4).to(torch.float16)}, 0, 1, "n0")
    with pytest.raises(ReportError, match="arrived as"):
        read(store, 0, expectation({"w": torch.randn(4)}))


def test_a_short_delta_is_refused_rather_than_averaged_as_if_whole(store):
    """Part of the model updated by fewer nodes than the rest, silently."""
    write(store, {"layer.weight": torch.randn(3, 4)}, 0, 1, "n0")
    with pytest.raises(ReportError, match="missing parameters"):
        read(store, 0, expectation(a_delta()))


# --------------------------------------------------------------------------
# the quantised round, which came with the format rather than being built


def bytes_under(root):
    return sum(
        os.path.getsize(os.path.join(where, name))
        for where, _, names in os.walk(root)
        for name in names
    )


def test_save_dtype_halves_a_round_and_the_reader_needs_no_changes(tmp_path):
    """GPU-113's point 5, and nothing in this module knows it happened.

    The peer casts to bf16 before the bytes leave; moonclip uncasts on load, so
    what comes back is fp32 of the right shape and the reader's check passes
    unchanged. What is lost is precision, which is the trade being made and is
    why this is a setting rather than the default.
    """
    delta = {"w": torch.randn(256, 256), "b": torch.randn(256)}

    plain = open_store(str(tmp_path / "plain"))
    write(plain, delta, 0, 1, "n0")
    plain.flush()

    small = open_store(str(tmp_path / "small"), save_dtype="bf16")
    write(small, delta, 0, 1, "n0")
    small.flush()

    assert bytes_under(str(tmp_path / "small")) < bytes_under(str(tmp_path / "plain")) / 1.8

    report = read(small, 0, expectation(delta))
    assert report.delta["w"].dtype is torch.float32
    assert report.delta["w"].shape == delta["w"].shape
    # Close, not equal: the bytes went through bf16 and came back.
    assert torch.allclose(report.delta["w"], delta["w"], atol=0.02)
    assert not torch.equal(report.delta["w"], delta["w"])


def test_the_stored_dtype_is_not_what_the_reader_checks(tmp_path):
    """A peer compressing its report is a setting, not a different model."""
    delta = {"w": torch.randn(64, 64)}
    small = open_store(str(tmp_path / "small"), save_dtype="bf16")
    write(small, delta, 0, 1, "n0")

    described = small.describe(_report.snapshot_of_round(small, 0))
    entry = described["tensors"][0]
    assert entry["stored_dtype"] == "bfloat16" and entry["dtype"] == "float32"

    read(small, 0, expectation(delta))  # accepted, on `dtype`
