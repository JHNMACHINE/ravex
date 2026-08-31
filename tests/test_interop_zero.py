"""Rebuilding whole tensors out of a ZeRO checkpoint — GPU-90.

The authoritative check on :mod:`ravex._interop.zero` is not here. It is
``integration/frameworks/check_zero_oracle.py``, which compares the
reconstruction against ``zero_to_fp32.py`` — DeepSpeed's own reader, shipped
inside every checkpoint — over real artefacts. On 2026-08-31 that agreed
bit-exactly across twelve combinations: stages 1, 2 and 3, world sizes 2 and 3,
and a hidden size of 97 chosen because it does not divide by either.

What is here is what that cannot be: fast, dependency-free, and able to
construct the cases a real DeepSpeed run will not produce on demand — a shard
missing, two shards from different checkpoints, a stage nobody recognises. The
fixtures below are hand-built in the layout the real files were measured to
have, which is the only thing that makes a synthetic test of a foreign format
mean anything.

The stage-3 test is the one worth keeping honest. This module first assembled
stage 3 the same way as stage 1 and produced every shape correctly with every
value wrong — the parameters interleaved, off by up to 0.13 each. Shapes alone
do not catch it, so these tests check values.
"""

import math

import pytest

import torch

from ravex._interop.foreign import Foreign
from ravex._interop.zero import ZeroUnsupported, unshard


HIDDEN = 5
SHAPES = {"a.weight": [HIDDEN, HIDDEN], "a.bias": [HIDDEN], "b.weight": [3, HIDDEN]}


def whole_tensors():
    """The model this checkpoint is of, with every element distinguishable.

    Sequential values rather than random ones: if the reader joins two pieces
    in the wrong order the result is still the right shape and the right dtype,
    and only a value that says where it came from makes that visible.
    """
    at = 0
    out = {}
    for name, shape in SHAPES.items():
        size = math.prod(shape)
        out[name] = torch.arange(at, at + size, dtype=torch.float32).view(*shape)
        at += size
    return out


def flat_of(tensors):
    return torch.cat([t.flatten() for t in tensors.values()])


def model_file(step=7):
    return {"param_shapes": [dict(SHAPES)], "global_steps": step, "module": {}}


def moments_for(partition, step=3):
    """Adam's buffers over one rank's partition, distinguishable from it.

    Scaled rather than copied so a reader that hands back the parameters where
    the moments were asked for is caught: the shapes would be identical and
    only the values differ.
    """
    return {
        "step": step,
        "exp_avg": partition * 0.5,
        "exp_avg_sq": partition * 0.25,
    }


def shard_files(stage, ranks, tensors=None):
    """One optimizer shard per rank, cut the way the measured format cuts them.

    The asymmetry at stages 1 and 2 is the point of this fixture and is not an
    embellishment: DeepSpeed trims the **last** rank's parameter partition down
    to the real remainder while leaving that rank's moments at the full padded
    length. Measured on 2026-08-31 — three 97x97 layers over two ranks gave
    rank 1 a 14258-element parameter partition and 14260 elements of
    ``exp_avg``. A reader that pairs the two by length drops that rank.
    """
    tensors = tensors or whole_tensors()
    key = "fp32_flat_groups" if stage == 3 else "single_partition_of_fp32_groups"
    shards = []

    if stage == 3:
        # Each parameter partitioned on its own, ceil(numel / ranks) per rank.
        per_rank = [[] for _ in range(ranks)]
        for name, shape in SHAPES.items():
            size = math.prod(shape)
            stride = -(-size // ranks)
            flat = tensors[name].flatten()
            padded = torch.cat([flat, torch.zeros(stride * ranks - size)])
            for r in range(ranks):
                per_rank[r].append(padded[r * stride : (r + 1) * stride])
        for r in range(ranks):
            partition = torch.cat(per_rank[r])
            shards.append(
                {
                    "optimizer_state_dict": {
                        "zero_stage": 3,
                        key: [partition],
                        "optimizer_state_dict": {
                            "state": {0: moments_for(partition)},
                            "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999)}],
                        },
                    }
                }
            )
        return shards

    # Stages 1 and 2: the group flattened whole, padded, split evenly.
    flat = flat_of(tensors)
    stride = -(-flat.numel() // ranks)
    padding = stride * ranks - flat.numel()
    padded = torch.cat([flat, torch.zeros(padding)])
    for r in range(ranks):
        slot = padded[r * stride : (r + 1) * stride]
        # The moments keep the whole padded slot; the parameter partition of
        # the last rank does not.
        partition = slot[: stride - padding] if r == ranks - 1 else slot
        shards.append(
            {
                "optimizer_state_dict": {
                    "zero_stage": stage,
                    key: [partition],
                    "group_paddings": [padding],
                    "base_optimizer_state": {
                        "state": {0: moments_for(slot)},
                        "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999)}],
                    },
                }
            }
        )
    return shards


def checkpoint(stage, ranks, tmp_path, tensors=None):
    """A `Foreign` plus the loader that serves it, with nothing written to disk.

    The loader is a dict lookup rather than `torch.load`: these tests are about
    the arithmetic of reassembly, and writing the fixtures out only to read
    them back would test pickling.
    """
    files = {}
    names = []
    if stage == 3:
        for r, shard in enumerate(shard_files(stage, ranks, tensors)):
            name = "zero_pp_rank_%d_mp_rank_00_optim_states.pt" % r
            files[name] = shard
            names.append(name)
        model = "zero_pp_rank_0_mp_rank_00_model_states.pt"
    else:
        for r, shard in enumerate(shard_files(stage, ranks, tensors)):
            name = "zero_pp_rank_%d_mp_rank_00_optim_states.pt" % r
            files[name] = shard
            names.append(name)
        model = "mp_rank_00_model_states.pt"
    files[model] = model_file()
    names.append(model)

    import os

    found = Foreign("deepspeed", str(tmp_path), world_size=ranks, stage=stage,
                    files=sorted(names))
    return found, lambda path: files[os.path.basename(path)]


# ── the reconstruction ──────────────────────────────────────────────────────


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("ranks", [1, 2, 3, 4, 7])
def test_every_stage_and_rank_count_rebuilds_the_original(stage, ranks, tmp_path):
    """Seven ranks for three parameters is deliberate.

    More ranks than a tensor has rows is not a silly case: it is what a wide
    job looks like from the point of view of a bias vector, and it is where a
    stride computed as ``numel // ranks`` rather than as a ceiling silently
    loses the tail.
    """
    want = whole_tensors()
    found, loader = checkpoint(stage, ranks, tmp_path, want)

    got = unshard(found, loader)

    assert got["stage"] == stage
    assert got["world_size"] == ranks
    assert got["step"] == 7
    assert set(got["model"]) == set(want)
    for name, tensor in want.items():
        assert torch.equal(got["model"][name], tensor), name


def test_the_stage_comes_from_the_shards_not_from_the_detection(tmp_path):
    """`Foreign.stage` is an opinion formed outside; the shard is the record.

    Worth pinning down because the detector can be wrong by design — at stages
    1 and 2 the layout genuinely cannot tell which it is — so a reader that
    took its assembly from the detector would be taking it from a guess.
    """
    want = whole_tensors()
    found, loader = checkpoint(3, 2, tmp_path, want)
    found.stage = 1  # contradicted by every shard, and ignored

    got = unshard(found, loader)
    assert got["stage"] == 3
    for name, tensor in want.items():
        assert torch.equal(got["model"][name], tensor), name


def test_the_values_are_checked_not_only_the_shapes(tmp_path):
    """Assembling stage 3 the stage-1 way gives right shapes and wrong values.

    This is the bug the oracle caught on 2026-08-31, reproduced deliberately so
    that the test which would have caught it exists. A stage-3 layout whose
    shards *claim* stage 1 makes the reader take the wrong branch, and nothing
    about the sizes objects — every tensor comes back the right shape, holding
    somebody else's numbers.

    It also states a limit honestly: a checkpoint that misreports its own stage
    is reconstructed wrongly and silently. Detecting that would mean checking
    the layout against the record on every read, and the layout cannot separate
    stage 1 from stage 2, so the check would only ever catch this one case.
    """
    want = whole_tensors()
    found, loader = checkpoint(3, 2, tmp_path, want)
    original = loader

    def lying(path):
        blob = original(path)
        if "optim" in path:
            osd = dict(blob["optimizer_state_dict"])
            osd["zero_stage"] = 1
            blob = {"optimizer_state_dict": osd}
        return blob

    got = unshard(found, lying)["model"]
    assert all(tuple(got[n].shape) == tuple(want[n].shape) for n in want)
    assert any(not torch.equal(got[n], want[n]) for n in want)


# ── what it refuses, and why ────────────────────────────────────────────────


def test_a_missing_shard_is_refused_rather_than_joined(tmp_path):
    """Ranks 0 and 2 with 1 gone would concatenate the wrong pieces together."""
    found, loader = checkpoint(1, 3, tmp_path)
    found.files = [n for n in found.files if "rank_1_" not in n]
    with pytest.raises(ZeroUnsupported, match="not a complete"):
        unshard(found, loader)


def test_shards_that_disagree_about_the_stage(tmp_path):
    found, loader = checkpoint(1, 2, tmp_path)
    original = loader

    def confused(path):
        blob = original(path)
        if path.endswith("rank_1_mp_rank_00_optim_states.pt"):
            blob = dict(blob)
            blob["optimizer_state_dict"] = dict(blob["optimizer_state_dict"])
            blob["optimizer_state_dict"]["zero_stage"] = 3
        return blob

    with pytest.raises(ZeroUnsupported, match="not from one checkpoint"):
        unshard(found, confused)


def test_a_stage_this_reader_does_not_know(tmp_path):
    found, loader = checkpoint(1, 2, tmp_path)
    original = loader

    def future(path):
        blob = original(path)
        if "optim_states" in path:
            blob = dict(blob)
            blob["optimizer_state_dict"] = dict(blob["optimizer_state_dict"])
            blob["optimizer_state_dict"]["zero_stage"] = 4
        return blob

    with pytest.raises(ZeroUnsupported, match="ZeRO stage 4"):
        unshard(found, future)


def test_no_model_file_means_no_names(tmp_path):
    """Optimizer state alone cannot be turned back into named tensors."""
    found, loader = checkpoint(1, 2, tmp_path)
    found.files = [n for n in found.files if "model_states" not in n]
    with pytest.raises(ZeroUnsupported, match="parameter shapes are unknown"):
        unshard(found, loader)


def test_a_truncated_shard_is_refused(tmp_path):
    found, loader = checkpoint(1, 2, tmp_path)
    original = loader

    def short(path):
        blob = original(path)
        if "rank_0_" in path and "optim" in path:
            osd = dict(blob["optimizer_state_dict"])
            osd["single_partition_of_fp32_groups"] = [
                osd["single_partition_of_fp32_groups"][0][:-4]
            ]
            blob = {"optimizer_state_dict": osd}
        return blob

    with pytest.raises(ZeroUnsupported):
        unshard(found, short)


def test_recorded_padding_that_does_not_match_is_reported_not_refused(tmp_path):
    """`group_paddings` is noted when it disagrees, and never acted on.

    It was a refusal first. Then the field was measured across stages 1 and 2
    at two world sizes and two shapes and came back `[0]` every time —
    including where the partitions are visibly uneven — so it does not mean
    "the slack at the end", and gating on it would refuse checkpoints that
    reconstruct correctly. The dangerous direction, too few elements, is still
    a refusal; see the test below.
    """
    found, loader = checkpoint(1, 2, tmp_path)
    original = loader
    want = whole_tensors()

    def lying(path):
        blob = original(path)
        if "optim" in path:
            osd = dict(blob["optimizer_state_dict"])
            osd["group_paddings"] = [999]
            blob = {"optimizer_state_dict": osd}
        return blob

    got = unshard(found, lying)
    assert any("999" in note for note in got["notes"])
    for name, tensor in want.items():
        assert torch.equal(got["model"][name], tensor), name


def test_nothing_at_all(tmp_path):
    found = Foreign("deepspeed", str(tmp_path), files=[])
    with pytest.raises(ZeroUnsupported, match="no ZeRO optimizer shard"):
        unshard(found, lambda path: {})


# ── the optimizer moments ───────────────────────────────────────────────────


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("ranks", [1, 2, 3, 5])
def test_the_moments_come_back_on_the_right_parameters(stage, ranks, tmp_path):
    """Every rank's moments, reassembled and keyed by parameter name.

    The fixture makes ``exp_avg`` half the parameter value and ``exp_avg_sq`` a
    quarter, so a moment that lands on the wrong parameter is visible as a
    number rather than as a shape. Real checkpoints get a stronger check —
    ``integration/frameworks/check_zero_moments.py`` predicts one Adam step
    from a pair of consecutive checkpoints, which is a statement about the
    pairing that no shape can make. It agreed to 1.5e-08 across stages 1, 2
    and 3 at two world sizes on 2026-08-31.
    """
    want = whole_tensors()
    found, loader = checkpoint(stage, ranks, tmp_path, want)

    state = unshard(found, loader)["optimizer"]["state"]

    assert set(state) == set(want)
    for name, tensor in want.items():
        assert torch.allclose(state[name]["exp_avg"], tensor * 0.5), name
        assert torch.allclose(state[name]["exp_avg_sq"], tensor * 0.25), name


def test_a_trimmed_last_rank_does_not_lose_its_moments(tmp_path):
    """The bug this fixture exists for, named.

    At stages 1 and 2 the last rank's parameter partition is shorter than its
    moments. Pairing them by length drops that rank, and the reconstruction
    then comes up short by a whole partition — which the assembly notices, so
    the symptom is a refusal rather than a wrong answer, but a refusal on a
    perfectly good checkpoint.
    """
    ranks = 2
    found, loader = checkpoint(1, ranks, tmp_path)

    # The fixture is only interesting if it really is asymmetric.
    shards = shard_files(1, ranks)
    partition = shards[-1]["optimizer_state_dict"]["single_partition_of_fp32_groups"][0]
    moments = shards[-1]["optimizer_state_dict"]["base_optimizer_state"]["state"][0]
    assert partition.numel() < moments["exp_avg"].numel()

    got = unshard(found, loader)
    assert len(got["optimizer"]["state"]) == len(SHAPES)


def test_scalars_are_carried_per_parameter_not_partitioned(tmp_path):
    """``step`` is one number for the group, and belongs on every name in it."""
    found, loader = checkpoint(1, 2, tmp_path)
    state = unshard(found, loader)["optimizer"]["state"]
    assert all(entry["step"] == 3 for entry in state.values())


def test_the_hyperparameters_come_through(tmp_path):
    found, loader = checkpoint(3, 2, tmp_path)
    groups = unshard(found, loader)["optimizer"]["param_groups"]
    assert groups and groups[0]["lr"] == 0.001


def test_a_checkpoint_with_no_optimizer_state_still_gives_the_weights(tmp_path):
    """Weights without moments is a real checkpoint; weights missing is not."""
    found, loader = checkpoint(1, 2, tmp_path)
    original = loader

    def bare(path):
        blob = original(path)
        if "optim" in path:
            osd = dict(blob["optimizer_state_dict"])
            osd.pop("base_optimizer_state", None)
            blob = {"optimizer_state_dict": osd}
        return blob

    got = unshard(found, bare)
    assert got["optimizer"]["state"] == {}
    for name, tensor in whole_tensors().items():
        assert torch.equal(got["model"][name], tensor), name
