"""Reading a torch distributed checkpoint — GPU-90.

Unlike the DeepSpeed tests beside these, the fixtures here are the real thing:
torch writes this format itself, so every checkpoint below was produced by
``dcp.save`` a few lines earlier rather than hand-built from a measured layout.
That is worth doing where it is possible, and it is possible exactly because
this format's writer ships with the reader.

The format matters beyond torch: ``megatron.core.dist_checkpointing`` is built
on it, so a Megatron checkpoint is one of these. There is no Megatron test here
because there would be nothing Megatron-specific in it.
"""

import pytest

import torch
import torch.nn as nn

dcp = pytest.importorskip("torch.distributed.checkpoint")

from ravex._interop.dcp import describe, missing_from, read, tensors  # noqa: E402
from ravex._interop.foreign import identify  # noqa: E402


@pytest.fixture
def saved(tmp_path):
    """A checkpoint of a model and its optimizer, mid-training."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.randn(2, 6)).sum().backward()
    optimizer.step()

    path = str(tmp_path / "ckpt")
    dcp.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict()},
        checkpoint_id=path,
    )
    return path, model, optimizer


# ── description, which must not cost a tensor ───────────────────────────────


def test_shapes_and_dtypes_without_reading_the_tensors(saved):
    path, model, _ = saved
    described = describe(path)

    for name, tensor in model.state_dict().items():
        entry = described["model." + name]
        assert entry["shape"] == tuple(tensor.shape)
        assert entry["dtype"] == str(tensor.dtype)


def test_the_optimizer_is_described_too(saved):
    """A description of the tensors is not a description of the checkpoint."""
    path, _, _ = saved
    described = describe(path)
    assert any(key.startswith("optim.state.") for key in described)
    assert any("exp_avg" in key for key in described)


# ── reading ─────────────────────────────────────────────────────────────────


def test_the_structure_comes_back_as_it_was_saved(saved):
    path, model, optimizer = saved
    state = read(path)

    assert set(state) == {"model", "optim"}
    for name, tensor in model.state_dict().items():
        assert torch.equal(state["model"][name], tensor), name


def test_the_optimizer_moments_come_back(saved):
    path, _, optimizer = saved
    state = read(path)

    want = optimizer.state_dict()["state"]
    got = state["optim"]["state"]
    assert len(got) == len(want)
    for index, entry in want.items():
        # DCP flattens integer keys to strings on the way out; either is fine
        # as long as the moments are there and equal.
        mine = got.get(index, got.get(str(index)))
        assert mine is not None, index
        assert torch.equal(mine["exp_avg"], entry["exp_avg"])
        assert torch.equal(mine["exp_avg_sq"], entry["exp_avg_sq"])


def test_nothing_the_metadata_promised_is_missing(saved):
    """The silent failure this guards: a read that skipped something looks complete."""
    path, _, _ = saved
    assert missing_from(read(path), describe(path)) == []


def test_a_missing_key_is_reported_by_name(saved):
    path, _, _ = saved
    state = read(path)
    del state["model"]["0.weight"]
    assert missing_from(state, describe(path)) == ["model.0.weight"]


# ── the flattening, which follows the format rather than taste ──────────────


def test_lists_are_indexed_by_position():
    """``optim.param_groups.0.lr`` is DCP's own convention, not a preference."""
    flat = tensors({"a": [torch.zeros(1), torch.ones(1)]})
    assert sorted(flat) == ["a.0", "a.1"]


def test_a_bare_tensor_flattens_to_its_prefix():
    assert list(tensors(torch.zeros(2), "x")) == ["x"]


def test_things_that_are_not_tensors_are_dropped_from_the_flattening():
    assert tensors({"lr": 0.01, "name": "adam"}) == {}


# ── and that the detector agrees this is what it is ─────────────────────────


def test_a_saved_checkpoint_identifies_as_dcp(saved):
    path, _, _ = saved
    found = identify(path)
    assert found.format == "dcp"
    assert found.world_size == 1
    assert any("Megatron" in note for note in found.notes)
