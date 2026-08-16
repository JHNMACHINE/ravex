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
