"""Turning a foreign checkpoint into a restorable state — GPU-90.

Most of what could go wrong in a conversion is not arithmetic — that lives in
:mod:`ravex._zero`, is checked against DeepSpeed's own reader, and is covered
next door. What goes wrong here is **names**: a wrapper prefix, a buffer the
model does not have, a head added since the checkpoint was written. Those
produce a state dict that loads *partially*, and a partially loaded model is
the failure this whole package is written against — it runs, it trains, and
the loss curve looks like a bad learning rate.

So these tests are mostly about :func:`ravex._convert.align`, and specifically
about it refusing to be clever.
"""

import pytest

import torch
import torch.nn as nn

from ravex._convert import (
    Alignment,
    CannotConvert,
    align,
    fit_param_groups,
    into_snapshot,
    rename,
    unify,
)
from ravex._foreign import Foreign

dcp = pytest.importorskip("torch.distributed.checkpoint")


# ── names ───────────────────────────────────────────────────────────────────


def test_names_that_already_match():
    found = align(["a.weight", "a.bias"], ["a.weight", "a.bias"])
    assert found.complete
    assert found.prefix is None
    assert found.mapping == {"a.weight": "a.weight", "a.bias": "a.bias"}


def test_a_wrapper_prefix_is_stripped_from_all_of_them():
    """`module.` in front of everything is one naming, not many."""
    found = align(
        ["module.a.weight", "module.a.bias"], ["a.weight", "a.bias"]
    )
    assert found.complete
    assert found.prefix == "module."
    assert found.mapping["module.a.weight"] == "a.weight"


def test_a_prefix_on_only_some_keys_is_not_stripped():
    """All or none.

    Stripping the ones that carry it would restore half the model under one
    naming and half under another, and report success. Leaving them alone
    reports two missing parameters, which is a thing someone can look at.
    """
    found = align(["module.a.weight", "a.bias"], ["a.weight", "a.bias"])
    assert not found.complete
    assert found.prefix is None
    assert found.missing == ["a.weight"]
    assert found.extra == ["module.a.weight"]


def test_nothing_is_matched_by_similarity():
    """`layers.0.weight` and `layers.1.weight` are not each other.

    A fuzzy match is worst exactly where names are most alike, which in a
    transformer is everywhere.
    """
    found = align(["layers.0.weight"], ["layers.1.weight"])
    assert found.mapping == {}
    assert found.missing == ["layers.1.weight"]


def test_a_model_that_grew_a_head_reports_what_is_missing():
    """A fine-tune with extra parameters is a real case, not a failure.

    The alignment says so and lets the caller decide; refusing here would
    refuse the ordinary way people use a converted checkpoint.
    """
    found = align(["body.weight"], ["body.weight", "head.weight"])
    assert not found.complete
    assert found.mapping == {"body.weight": "body.weight"}
    assert found.missing == ["head.weight"]
    assert found.extra == []


def test_a_checkpoint_with_buffers_the_model_lacks():
    found = align(["a.weight", "a.running_mean"], ["a.weight"])
    assert found.complete  # every live parameter is served
    assert found.extra == ["a.running_mean"]


def test_the_report_says_what_happened():
    found = align(["module.a.weight"], ["a.weight"])
    assert "1 parameter(s) matched" in found.report()
    assert "module." in found.report()

    torn = align(["x"], ["y"])
    assert "have nothing in the checkpoint" in torn.report()
    assert "no home in the model" in torn.report()


# ── renaming carries the optimizer too ──────────────────────────────────────


def test_renaming_moves_the_weights_the_state_and_the_group_membership():
    """Three places have to agree, which is why one function does all three."""
    unified = {
        "model": {"module.a.weight": torch.zeros(2)},
        "optimizer": {
            "state": {"module.a.weight": {"exp_avg": torch.ones(2)}},
            "param_groups": [{"lr": 0.1, "params": ["module.a.weight"]}],
        },
    }
    moved = rename(unified, align(["module.a.weight"], ["a.weight"]))

    assert list(moved["model"]) == ["a.weight"]
    assert list(moved["optimizer"]["state"]) == ["a.weight"]
    assert moved["optimizer"]["param_groups"][0]["params"] == ["a.weight"]
    assert moved["optimizer"]["param_groups"][0]["lr"] == 0.1


def test_renaming_drops_what_did_not_match_rather_than_carrying_it():
    unified = {
        "model": {"a": torch.zeros(1), "stale": torch.zeros(1)},
        "optimizer": {"state": {"stale": {}}, "param_groups": [{"params": ["stale"]}]},
    }
    moved = rename(unified, Alignment({"a": "a"}, [], ["stale"]))
    assert list(moved["model"]) == ["a"]
    assert moved["optimizer"]["state"] == {}
    assert moved["optimizer"]["param_groups"][0]["params"] == []


# ── the snapshot Ravex's own restore path takes ─────────────────────────────


def test_the_snapshot_is_a_gather_layout_group():
    """`gather` is not a stand-in: a foreign checkpoint read whole *is* that."""
    unified = {"model": {"a": torch.zeros(1)}, "optimizer": {}, "source": "somewhere"}
    snapshot = into_snapshot(unified, key="model_abc123", step=42)

    assert snapshot["step"] == 42
    assert set(snapshot["sharded"]) == {"model_abc123"}
    group = snapshot["sharded"]["model_abc123"]
    assert group["layout"] == "gather"
    assert group["parameters"] == ["a"]
    assert group["converted_from"] == "somewhere"


def test_the_step_falls_back_to_what_the_checkpoint_recorded():
    unified = {"model": {}, "optimizer": {}, "step": 9}
    assert into_snapshot(unified, key="k")["step"] == 9


# ── reading a real distributed checkpoint through unify ─────────────────────


@pytest.fixture
def dcp_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    path = str(tmp_path / "ckpt")
    dcp.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict()},
        checkpoint_id=path,
    )
    return path, model


def test_unify_finds_the_model_in_a_distributed_checkpoint(dcp_checkpoint):
    path, model = dcp_checkpoint
    unified = unify(Foreign("dcp", path), loader=None)

    assert set(unified["model"]) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(unified["model"][name], tensor)
    assert any("model" in note for note in unified["notes"])


def test_unify_handles_a_checkpoint_that_is_just_a_model(tmp_path):
    """`dcp.save(model.state_dict())` is an ordinary thing to have written."""
    model = nn.Linear(3, 2)
    path = str(tmp_path / "bare")
    dcp.save(model.state_dict(), checkpoint_id=path)

    unified = unify(Foreign("dcp", path), loader=None)
    assert set(unified["model"]) == set(model.state_dict())
    assert any("top level" in note for note in unified["notes"])


def test_unify_refuses_a_format_nothing_reads(tmp_path):
    with pytest.raises(CannotConvert, match="identified but not opened"):
        unify(Foreign("ravex", str(tmp_path)), loader=None)


# ── and the end to end shape, against a live model ──────────────────────────


def test_a_converted_snapshot_loads_into_a_plain_model(dcp_checkpoint):
    """The point of all of it: the converted state restores.

    A plain module rather than a sharded one, because what is being checked
    here is the shape of the hand-off — the scattering onto shards is
    `apply_sharded_state`'s job, is torch's code underneath, and is exercised
    by the resume tests that already exist.
    """
    path, original = dcp_checkpoint
    fresh = nn.Sequential(nn.Linear(4, 3))
    assert not torch.equal(fresh[0].weight, original[0].weight)

    unified = unify(Foreign("dcp", path), loader=None)
    alignment = align(list(unified["model"]), list(fresh.state_dict()))
    assert alignment.complete

    fresh.load_state_dict(rename(unified, alignment)["model"])
    for name, tensor in original.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor), name


# ── hyperparameters, which do not transfer ──────────────────────────────────


def test_a_hyperparameter_the_receiver_does_not_have_is_dropped():
    """DeepSpeed's Adam records `bias_correction`; torch's does not.

    Found end to end on 2026-08-31, and it is not a near miss: `set_state_dict`
    raises `KeyError: param_groups.0.bias_correction` from inside its own
    unflattening, several frames from anything the caller wrote.
    """
    unified = {
        "model": {},
        "optimizer": {
            "state": {},
            "param_groups": [
                {"lr": 0.001, "bias_correction": True, "params": ["a"]}
            ],
        },
    }
    fitted = fit_param_groups(unified, [{"lr": 0.1, "eps": 1e-8, "params": []}])

    group = fitted["optimizer"]["param_groups"][0]
    assert group == {"lr": 0.001, "params": ["a"]}
    assert any("bias_correction" in note for note in fitted["notes"])


def test_membership_always_survives():
    """`params` is which parameters are in the group, not a setting."""
    unified = {
        "model": {},
        "optimizer": {"state": {}, "param_groups": [{"params": ["a", "b"]}]},
    }
    fitted = fit_param_groups(unified, [{"lr": 0.1}])
    assert fitted["optimizer"]["param_groups"][0]["params"] == ["a", "b"]


def test_nothing_is_translated_into_a_hyperparameter_the_target_has():
    """A setting with no counterpart has no correct value in the target.

    Inventing one changes how the resumed run trains, quietly, in a way no
    check of the checkpoint would catch.
    """
    unified = {
        "model": {},
        "optimizer": {"state": {}, "param_groups": [{"weight_decay": 0.5}]},
    }
    fitted = fit_param_groups(unified, [{"lr": 0.1}])
    assert "weight_decay" not in fitted["optimizer"]["param_groups"][0]
    assert "lr" not in fitted["optimizer"]["param_groups"][0]


def test_allowed_may_be_given_as_bare_key_names():
    unified = {
        "model": {},
        "optimizer": {"state": {}, "param_groups": [{"lr": 0.1, "odd": 1}]},
    }
    fitted = fit_param_groups(unified, {"lr"})
    assert fitted["optimizer"]["param_groups"][0] == {"lr": 0.1}
