"""Two ranks, one checkpoint.

Distributed runs are where checkpointing is worth having and also where it is
easiest to get subtly wrong: only one rank should write, every rank must
resume, and the file that comes out has to be loadable by a plain single-
process script afterwards.

gloo on CPU, so this runs anywhere — the failure modes being tested are about
process topology, not about the device.
"""

import os
import subprocess
import sys

import pytest

from conftest import SCRIPTS, read_trace

TOTAL_STEPS = 40
CRASH_AT = 20
RESUME_FROM = 17  # see test_process_restart for how this falls out


def torchrun(directory, script, trace_name="trace.jsonl", ranks=2, **script_args):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={ranks}",
        "--master_port=29555",
        str(SCRIPTS / script),
        "--trace",
        trace_name,
    ]
    for key, value in script_args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]

    return subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=900,
        # Ignored on CPU; on CUDA it makes cuBLAS pick a deterministic
        # algorithm, and it has to be set before the child initialises CUDA.
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )


def losses(trace):
    return [entry["loss"] for entry in trace]


@pytest.fixture
def ddp_workspace(workspace):
    # Two ranks means half the dataset each, so an epoch is 4 steps rather
    # than 8. Keep the budget where the single-process tests have it.
    return lambda name: workspace(name, max_steps=TOTAL_STEPS)


def test_two_ranks_train_and_only_rank_zero_writes(ddp_workspace):
    directory = ddp_workspace("ddp-plain")
    result = torchrun(directory, "train_ddp.py", epochs=3)

    assert result.returncode == 0, result.stderr
    assert "rank 0 done" in result.stdout and "rank 1 done" in result.stdout

    checkpoints = sorted((directory / "checkpoints").glob("step_*.pt"))
    assert checkpoints, "rank 0 should have written checkpoints"

    log = (directory / "ravex.log").read_text()
    # Both ranks run the runtime — rank 1 has to resume even though it never
    # writes. The torchrun launcher process activates too (rank=0/1, before it
    # sets WORLD_SIZE); it never trains, and _ensure_backend keeps it from
    # opening any storage.
    assert "rank=0/2" in log and "rank=1/2" in log
    assert "Checkpoint at step" in log


def test_a_killed_ddp_run_resumes_on_every_rank(ddp_workspace):
    reference = ddp_workspace("ddp-reference")
    assert torchrun(reference, "train_ddp.py").returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) == TOTAL_STEPS

    run = ddp_workspace("ddp-interrupted")
    killed = torchrun(run, "train_ddp.py", die_at=CRASH_AT)
    assert killed.returncode != 0, "expected the run to die"

    before = losses(read_trace(run))
    assert before == expected[:CRASH_AT], "diverged before the crash"

    restarted = torchrun(run, "train_ddp.py", trace_name="trace2.jsonl")
    assert restarted.returncode == 0, restarted.stderr

    after = losses(read_trace(run, "trace2.jsonl"))
    # Exact equality here is the strong claim: it means rank 1 resumed too. A
    # rank that restarted from fresh weights would poison the all-reduce and
    # every loss after the first step would differ.
    assert after == expected[RESUME_FROM - 1 :]


def test_a_ddp_checkpoint_loads_into_a_plain_model(ddp_workspace):
    directory = ddp_workspace("ddp-portable")
    assert torchrun(directory, "train_ddp.py", epochs=3).returncode == 0

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

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK "), result.stdout
