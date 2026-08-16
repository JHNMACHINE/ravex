"""Sharded models: the state is spread across ranks before it is saved.

With FSDP every rank holds a slice of each parameter and of the optimizer
moments that pair with it, so ``state_dict()`` returns a fragment. Ravex
gathers the whole thing through
``torch.distributed.checkpoint.state_dict`` — which means collecting a
checkpoint becomes a *collective*, and the rank gate has to move after it. A
rank that skipped the gather would leave the others waiting forever.

FSDP2 (``fully_shard``) is used because FSDP1 refuses to initialise without an
accelerator and there is no GPU here. Ravex detects them differently — DTensor
parameters against a wrapper class — but from that point on both go through the
same code.
"""

import glob
import subprocess
import sys

import pytest
import torch

from conftest import SCRIPTS, read_trace
from test_ddp import torchrun

pytest.importorskip("torch.distributed.fsdp", reason="no FSDP in this torch")

TOTAL_STEPS = 40
CRASH_AT = 20
RESUME_FROM = 17


def latest_checkpoint(directory):
    files = sorted(glob.glob(str(directory / "checkpoints" / "step_*.pt")))
    assert files, "no checkpoint was written"
    return torch.load(files[-1], map_location="cpu", weights_only=False)


def losses(trace):
    return [entry["loss"] for entry in trace]


@pytest.fixture
def fsdp_workspace(workspace):
    return lambda name: workspace(name, max_steps=TOTAL_STEPS)


def test_the_checkpoint_holds_whole_tensors_not_shards(fsdp_workspace):
    directory = fsdp_workspace("fsdp-shapes")
    result = torchrun(directory, "train_fsdp.py", epochs=3)
    assert result.returncode == 0, result.stderr

    state = latest_checkpoint(directory)

    assert state["sharded"], "the model should have been recognised as sharded"
    assert not state["models"], "a sharded model must not also be saved raw"
    assert not state["optimizers"], "its optimizer belongs to the sharded group"

    group = state["sharded"]["sharded_0"]

    # Linear(6, 12) over two ranks: each rank holds (6, 6). Saving that would
    # produce a checkpoint that only reloads on exactly two ranks — if at all.
    assert tuple(group["model"]["0.weight"].shape) == (12, 6)
    assert tuple(group["model"]["3.weight"].shape) == (1, 12)

    for name, value in group["model"].items():
        assert value.__class__ is torch.Tensor, f"{name} is still a {type(value)}"
        assert not value.is_meta, f"{name} was never materialised"


def test_a_killed_fsdp_run_resumes_on_every_rank(fsdp_workspace):
    reference = fsdp_workspace("fsdp-reference")
    assert torchrun(reference, "train_fsdp.py").returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) == TOTAL_STEPS

    run = fsdp_workspace("fsdp-interrupted")
    killed = torchrun(run, "train_fsdp.py", die_at=CRASH_AT)
    assert killed.returncode != 0, "expected the run to die"
    assert losses(read_trace(run)) == expected[:CRASH_AT]

    restarted = torchrun(run, "train_fsdp.py", trace_name="trace2.jsonl")
    assert restarted.returncode == 0, restarted.stderr

    after = losses(read_trace(run, "trace2.jsonl"))
    assert after == expected[RESUME_FROM - 1 :], (
        "a resumed sharded run must reproduce the uninterrupted one exactly"
    )


def test_no_final_checkpoint_is_attempted_at_shutdown(fsdp_workspace):
    """A gather at exit can hang; the periodic cadence is what you get.

    Ranks stop being in lockstep during shutdown — user code may already have
    destroyed the process group — and a collective nobody else joins blocks
    until something kills the process. Losing the last few steps is bounded;
    a hang is not.
    """
    directory = fsdp_workspace("fsdp-shutdown")
    # 3 epochs x 4 steps per rank = 12 steps, with checkpoints every 4.
    assert torchrun(directory, "train_fsdp.py", epochs=3).returncode == 0

    log = (directory / "ravex.log").read_text()
    assert "Skipping the final checkpoint" in log or "Checkpoint at step 12" in log
    assert "shutdown complete" in log, "shutdown still has to finish cleanly"


def test_a_sharded_run_resumes_in_an_unsharded_one(fsdp_workspace):
    """Eight GPUs in, one out - the claim the gathering is for.

    Reading the file is not enough: the resumed process has a plain
    `nn.Sequential` where the checkpoint has a sharded group, and the two have
    to find each other. If they do not, the run starts from random weights and
    says so only in a log line nobody reads.
    """
    directory = fsdp_workspace("fsdp-to-plain")

    killed = torchrun(directory, "train_fsdp.py", die_at=CRASH_AT)
    assert killed.returncode != 0

    saved = latest_checkpoint(directory)["sharded"]["sharded_0"]["model"]

    # A plain, single-process script. No torchrun, no mesh, no FSDP.
    plain = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "train_vanilla.py"),
            "--trace",
            "plain.jsonl",
            "--dump-weights",
            "restored.pt",
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert plain.returncode == 0, plain.stdout + plain.stderr

    log = (directory / "ravex.log").read_text()
    assert "Resumed at step" in log, "the plain run ignored the sharded checkpoint\n" + log

    restored = torch.load(directory / "restored.pt", map_location="cpu")
    assert set(restored) == set(saved), "parameter names do not line up"
    for name, value in restored.items():
        assert torch.equal(value, saved[name]), (
            f"{name} is not what the sharded run saved - the plain run started "
            f"from its own initialisation"
        )


# ─── per-rank checkpoints ──────────────────────────────────────────
#
# The other half of the trade. Gathering puts the whole state through rank 0 —
# 14.2 s and 18.1 GiB for a 1.48B model on 8 GPUs — and buys a checkpoint that
# resumes at any world size. Per-rank writes nothing but each rank's own shard,
# gathers nothing, and buys a checkpoint that only resumes at the world size
# that wrote it.


@pytest.fixture
def per_rank_workspace(workspace):
    return lambda name: workspace(
        name, max_steps=TOTAL_STEPS, sharded_checkpoints="per_rank"
    )


def rank_checkpoint(directory, rank):
    files = sorted(glob.glob(str(directory / "checkpoints" / f"rank_{rank}" / "step_*.pt")))
    assert files, f"rank {rank} wrote no checkpoint"
    return torch.load(files[-1], map_location="cpu", weights_only=False)


def test_every_rank_writes_its_own_shard_to_its_own_store(per_rank_workspace):
    directory = per_rank_workspace("fsdp-per-rank-shapes")
    result = torchrun(directory, "train_fsdp.py", epochs=3)
    assert result.returncode == 0, result.stderr

    for rank in (0, 1):
        state = rank_checkpoint(directory, rank)
        group = state["sharded"]["sharded_0"]
        assert group["layout"] == "per_rank"
        assert group["world_size"] == 2
        assert group["rank"] == rank

        # Linear(6, 12) over two ranks: (12, 6) whole, (6, 6) per rank. Storing
        # the shard is the entire point — the gathered path stores (12, 6) and
        # pays a collective and rank 0's memory for it.
        weight = group["model"]["0.weight"]
        assert weight["__ravex_shard__"] == 1
        assert tuple(weight["local"].shape) == (6, 6)
        assert weight["global_shape"] == [12, 6]

    # Nothing gathered means nothing for one rank to hold on behalf of the
    # others: two stores, not one shared one.
    assert not glob.glob(str(directory / "checkpoints" / "step_*.pt"))


def test_a_killed_per_rank_run_resumes_on_every_rank(per_rank_workspace):
    reference = per_rank_workspace("fsdp-per-rank-reference")
    assert torchrun(reference, "train_fsdp.py").returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) == TOTAL_STEPS

    run = per_rank_workspace("fsdp-per-rank-interrupted")
    killed = torchrun(run, "train_fsdp.py", die_at=CRASH_AT)
    assert killed.returncode != 0, "expected the run to die"
    assert losses(read_trace(run)) == expected[:CRASH_AT]

    restarted = torchrun(run, "train_fsdp.py", trace_name="trace2.jsonl")
    assert restarted.returncode == 0, restarted.stderr

    after = losses(read_trace(run, "trace2.jsonl"))
    assert after == expected[RESUME_FROM - 1 :], (
        "a resumed per-rank run must reproduce the uninterrupted one exactly — "
        "each rank puts back its own shard, and together they have to be the "
        "same model the gathered path would have restored"
    )


def test_a_per_rank_checkpoint_is_not_half_applied_at_another_world_size(
    per_rank_workspace,
):
    """The cost of not gathering, at the place someone will hit it.

    A per-rank checkpoint is a set of shards cut for one topology; on another
    one they compose into nothing. Two ranks wrote two stores, so of four ranks
    two find a checkpoint and two find none — and the wrong outcome is not "no
    resume", it is *half* a resume: two ranks restoring while two start from
    random weights, which trains happily and converges to nothing.

    Starting clean everywhere is the only correct answer, and the ranks have to
    reach it together.
    """
    directory = per_rank_workspace("fsdp-per-rank-reshard")
    assert torchrun(directory, "train_fsdp.py", epochs=3).returncode == 0
    assert rank_checkpoint(directory, 0), "the 2-rank run wrote nothing to resume from"

    wider = torchrun(
        directory, "train_fsdp.py", ranks=4, epochs=2, trace_name="wide.jsonl"
    )
    assert wider.returncode == 0, wider.stderr

    log = (directory / "ravex.log").read_text()
    assert "Resumed at step" not in log, (
        "some rank restored a checkpoint cut for a different world size\n" + log
    )
    assert "starting from scratch" in log


def test_an_fsdp_checkpoint_loads_into_a_plain_model(fsdp_workspace):
    directory = fsdp_workspace("fsdp-portable")
    assert torchrun(directory, "train_fsdp.py", epochs=3).returncode == 0

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "load_checkpoint.py"),
            "--checkpoints",
            str(directory / "checkpoints"),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # No FSDP, no mesh, no process group: eight GPUs wrote it, one CPU reads it.
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK "), result.stdout
