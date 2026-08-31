"""Recognising a checkpoint written by something else — GPU-90.

:mod:`ravex._foreign` reads directory names and never opens a file, so these
tests build their fixtures out of empty ones. That is not a shortcut: the cases
worth covering are the malformed ones — a shard missing, a `latest` pointing
nowhere, two layouts mixed — and producing those out of real frameworks means
sabotaging a real checkpoint, which tests the sabotage as much as the reader.

The layouts here are transcribed from artefacts actually produced on
2026-08-31 by ``integration/frameworks/make_zero.py`` (DeepSpeed 0.19.6, ZeRO
stages 1, 2 and 3) and by ``torch.distributed.checkpoint.save``. The file names
are not invented, which is the only reason a test made of empty files means
anything.
"""

import os

import pytest

from ravex._foreign import Foreign, confirm_stage, identify, summary


def tree(root, *names):
    """Create every named file, and any directory named with a trailing slash."""
    for name in names:
        path = os.path.join(root, name.rstrip("/"))
        if name.endswith("/"):
            os.makedirs(path, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
    return str(root)


# ── DeepSpeed ZeRO ──────────────────────────────────────────────────────────


def test_zero_stage_one_and_two_look_the_same_and_that_is_correct(tmp_path):
    """One shared model file, one optimizer file per rank.

    Stages 1 and 2 differ in what is partitioned *at run time* — gradients —
    and not in what reaches disk, so a layout that could tell them apart would
    be reporting something the directory does not contain.
    """
    root = tree(
        tmp_path / "z1",
        "step3/mp_rank_00_model_states.pt",
        "step3/zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "step3/zero_pp_rank_1_mp_rank_00_optim_states.pt",
        "zero_to_fp32.py",
    )
    with open(os.path.join(root, "latest"), "w", encoding="utf-8") as handle:
        handle.write("step3\n")

    found = identify(root)
    assert found.format == "deepspeed"
    assert found.world_size == 2
    assert found.stage is None  # the layout cannot say; `confirm_stage` can
    assert found.root.endswith("step3")
    assert "tag 'step3'" in found.notes[0]


def test_zero_stage_three_is_visible_in_the_layout(tmp_path):
    """Parameters partitioned too, so the model file is per rank and small."""
    root = tree(
        tmp_path / "z3",
        "step3/zero_pp_rank_0_mp_rank_00_model_states.pt",
        "step3/zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "step3/zero_pp_rank_1_mp_rank_00_model_states.pt",
        "step3/zero_pp_rank_1_mp_rank_00_optim_states.pt",
    )
    with open(os.path.join(root, "latest"), "w", encoding="utf-8") as handle:
        handle.write("step3")

    found = identify(root)
    assert found.format == "deepspeed"
    assert found.stage == 3
    assert found.world_size == 2


def test_a_tag_directory_on_its_own_is_still_recognised(tmp_path):
    """What someone copies out when they want one step, without the `latest`."""
    root = tree(
        tmp_path / "step7",
        "mp_rank_00_model_states.pt",
        "zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )
    found = identify(root)
    assert found.format == "deepspeed"
    assert found.world_size == 1
    assert found.root == root


def test_a_missing_shard_is_said_out_loud(tmp_path):
    """Ranks 0 and 2 present, 1 gone.

    Not refused here — this module reports, it does not decide — but a caller
    that reconstructs from ranks 0 and 2 as if they were 0 and 1 produces a
    tensor with a band of the wrong rows, which is the failure the whole
    resharding side of this project is written against.
    """
    root = tree(
        tmp_path / "torn",
        "mp_rank_00_model_states.pt",
        "zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "zero_pp_rank_2_mp_rank_00_optim_states.pt",
    )
    found = identify(root)
    assert found.format == "deepspeed"
    assert found.world_size == 2
    assert any("not a complete" in note for note in found.notes)


def test_optimizer_state_with_no_model_file_is_flagged(tmp_path):
    root = tree(tmp_path / "opt", "zero_pp_rank_0_mp_rank_00_optim_states.pt")
    found = identify(root)
    assert found.format == "deepspeed"
    assert any("optimizer state only" in note for note in found.notes)


# ── `latest`, which is a filename someone else wrote ────────────────────────


def test_latest_naming_something_that_is_not_there(tmp_path):
    root = str(tmp_path / "stale")
    os.makedirs(root)
    with open(os.path.join(root, "latest"), "w", encoding="utf-8") as handle:
        handle.write("step99")
    found = identify(root)
    assert found.format == "unknown"
    assert "step99" in found.notes[0]


@pytest.mark.parametrize("tag", ["../elsewhere", "/etc", "a/b", "", "..", "."])
def test_latest_does_not_get_to_leave_the_checkpoint(tmp_path, tag):
    """A tag is joined onto a path, so it is a name and not a route.

    DeepSpeed writes this file, but "DeepSpeed wrote it" is a property of a
    checkpoint someone vouches for, not of a directory that arrived.
    """
    root = str(tmp_path / "hostile")
    os.makedirs(root)
    with open(os.path.join(root, "latest"), "w", encoding="utf-8") as handle:
        handle.write(tag)
    found = identify(root)
    assert found.format == "unknown"


# ── the other formats ───────────────────────────────────────────────────────


def test_a_distributed_checkpoint_is_not_claimed_for_megatron(tmp_path):
    """`.metadata` plus `.distcp` is what Megatron-core writes *and* what plain torch writes.

    Naming it `dcp` rather than `megatron` is the finding from 2026-08-31:
    Megatron stopped having a manifest of its own, so the format cannot say
    which framework produced it and neither should this.
    """
    root = tree(tmp_path / "dcp", ".metadata", "__0_0.distcp", "__1_0.distcp")
    found = identify(root)
    assert found.format == "dcp"
    assert found.world_size == 2
    assert any("Megatron" in note for note in found.notes)


def test_ravexs_own_store_is_recognised_rather_than_converted(tmp_path):
    root = tree(tmp_path / "mine", "manifest.json", "snapshots/")
    assert identify(root).format == "ravex"


def test_ravexs_per_rank_store_is_recognised_too(tmp_path):
    root = tree(tmp_path / "mine", "rank_0/x", "rank_1/x", "rank_2/x")
    found = identify(root)
    assert found.format == "ravex"
    assert found.world_size == 3


def test_something_else_entirely(tmp_path):
    root = tree(tmp_path / "nope", "model.safetensors", "config.json")
    found = identify(root)
    assert found.format == "unknown"
    assert found.understood is False
    assert found.files == ["config.json", "model.safetensors"]


def test_a_directory_that_is_not_there(tmp_path):
    found = identify(str(tmp_path / "absent"))
    assert found.format == "unknown"
    assert "cannot be listed" in found.notes[0]


# ── what the files say, once one is opened ──────────────────────────────────


def test_the_recorded_stage_wins_over_the_layout(tmp_path):
    """The layout is an inference; the number DeepSpeed wrote is a statement.

    Manufactured here rather than found: a real checkpoint whose layout and
    recorded stage disagree would be a DeepSpeed bug, and waiting for one is
    not a test plan. What is being checked is which of the two the code
    believes, and that it says so.
    """
    root = tree(
        tmp_path / "odd",
        "zero_pp_rank_0_mp_rank_00_model_states.pt",
        "zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )
    found = identify(root)
    assert found.stage == 3  # from the layout

    settled = confirm_stage(
        found, lambda path: {"optimizer_state_dict": {"zero_stage": 2}}
    )
    assert settled.stage == 2
    assert any("says stage 2" in note for note in settled.notes)


def test_the_stage_is_filled_in_where_the_layout_could_not_say(tmp_path):
    root = tree(
        tmp_path / "s1",
        "mp_rank_00_model_states.pt",
        "zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )
    found = confirm_stage(
        identify(root), lambda path: {"optimizer_state_dict": {"zero_stage": 1}}
    )
    assert found.stage == 1
    assert not any("says stage" in note for note in found.notes)


def test_an_unreadable_file_leaves_the_layout_standing(tmp_path):
    """A checkpoint that cannot be opened is still a checkpoint that was identified."""

    def explode(path):
        raise OSError("nope")

    root = tree(
        tmp_path / "s3",
        "zero_pp_rank_0_mp_rank_00_model_states.pt",
        "zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )
    found = confirm_stage(identify(root), explode)
    assert found.stage == 3
    assert any("could not be read" in note for note in found.notes)


def test_confirm_stage_leaves_other_formats_alone(tmp_path):
    root = tree(tmp_path / "dcp", ".metadata", "__0_0.distcp")
    found = identify(root)
    assert confirm_stage(found, lambda path: 1 / 0) is found


# ── the line a log would carry ──────────────────────────────────────────────


def test_summary_reads_as_a_sentence():
    plain = Foreign("deepspeed", "/x", world_size=4, stage=3)
    assert summary(plain) == "deepspeed, ZeRO stage 3, 4 rank(s)"

    noted = Foreign("unknown", "/x", notes=["cannot be listed: no"])
    assert summary(noted) == "unknown (cannot be listed: no)"
