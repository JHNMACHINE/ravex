"""The parts that only a GPU can prove.

Everything here is skipped without CUDA, which is why it lives apart from the
rest of the integration suite: the development machine has no GPU, so these run
on a rented box.

Three things are unverifiable on CPU:

- the AMP loss scale, because ``GradScaler`` never actually scales anything on
  CPU and so never overflows or adjusts;
- the CUDA RNG, which is a separate generator from the CPU one;
- FSDP1, which refuses to initialise without an accelerator.
"""

import subprocess
import sys

import pytest
import torch

from conftest import SCRIPTS, read_trace
from test_ddp import torchrun

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

TOTAL_STEPS = 40
CRASH_AT = 20
RESUME_FROM = 17


def run_single(directory, trace_name="trace.jsonl", **script_args):
    command = [
        sys.executable,
        str(SCRIPTS / "train_cuda.py"),
        "--trace",
        trace_name,
    ]
    for key, value in script_args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    return subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=900
    )


def losses(trace):
    return [entry["loss"] for entry in trace]


def needs_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs to shard across")


@pytest.fixture
def gpu_workspace(workspace):
    return lambda name: workspace(name, max_steps=TOTAL_STEPS)


# ─── AMP ────────────────────────────────────────────────────────────


def test_amp_resumes_with_the_same_losses_and_loss_scale(gpu_workspace):
    reference = gpu_workspace("amp-reference")
    result = run_single(reference, mode="amp")
    assert result.returncode == 0, result.stderr

    expected = read_trace(reference)
    assert len(expected) == TOTAL_STEPS

    run = gpu_workspace("amp-interrupted")
    killed = run_single(run, mode="amp", die_at=CRASH_AT)
    assert killed.returncode == -9

    restarted = run_single(run, trace_name="trace2.jsonl", mode="amp")
    assert restarted.returncode == 0, restarted.stderr
    after = read_trace(run, "trace2.jsonl")

    assert losses(after) == losses(expected)[RESUME_FROM - 1 :], (
        "fp16 training must resume bit-identically"
    )

    # The loss scale is tuned state. A resume that reset it to init_scale would
    # spend the next few hundred steps overflowing its way back.
    assert [e["scale"] for e in after] == [
        e["scale"] for e in expected[RESUME_FROM - 1 :]
    ]


def test_the_cuda_rng_state_is_restored(gpu_workspace):
    """Dropout on the GPU draws from the CUDA generator, not the CPU one.

    If only the CPU RNG were restored the weights would still look right and
    the losses would quietly diverge — which is exactly the failure this whole
    project exists to avoid.
    """
    reference = gpu_workspace("rng-reference")
    assert run_single(reference, mode="amp").returncode == 0
    expected = losses(read_trace(reference))

    run = gpu_workspace("rng-interrupted")
    run_single(run, mode="amp", die_at=CRASH_AT)
    assert run_single(run, trace_name="trace2.jsonl", mode="amp").returncode == 0

    after = losses(read_trace(run, "trace2.jsonl"))
    assert after[0] == expected[RESUME_FROM - 1], "first resumed step already differs"
    assert after == expected[RESUME_FROM - 1 :]


# ─── FSDP1 and NCCL ─────────────────────────────────────────────────


def test_fsdp1_state_is_gathered_whole(gpu_workspace):
    needs_two_gpus()
    directory = gpu_workspace("fsdp1-shapes")
    result = torchrun(directory, "train_cuda.py", epochs=3, mode="fsdp1")
    assert result.returncode == 0, result.stderr

    import glob

    files = sorted(glob.glob(str(directory / "checkpoints" / "step_*.pt")))
    assert files
    state = torch.load(files[-1], map_location="cpu", weights_only=False)

    assert state["sharded"], "FSDP1 should be recognised through its wrapper"
    assert not state["models"]
    group = state["sharded"]["sharded_0"]

    # Full shapes, on CPU, with no wrapper prefixes in the names.
    assert tuple(group["model"]["0.weight"].shape) == (12, 6)
    for name, value in group["model"].items():
        assert "_fsdp_wrapped_module" not in name
        assert value.device.type == "cpu", f"{name} was not offloaded"


def test_a_killed_fsdp1_run_resumes_on_every_rank(gpu_workspace):
    needs_two_gpus()
    reference = gpu_workspace("fsdp1-reference")
    assert torchrun(reference, "train_cuda.py", mode="fsdp1").returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) == TOTAL_STEPS

    run = gpu_workspace("fsdp1-interrupted")
    assert torchrun(run, "train_cuda.py", mode="fsdp1", die_at=CRASH_AT).returncode != 0

    restarted = torchrun(run, "train_cuda.py", trace_name="trace2.jsonl", mode="fsdp1")
    assert restarted.returncode == 0, restarted.stderr

    assert losses(read_trace(run, "trace2.jsonl")) == expected[RESUME_FROM - 1 :]


def test_ddp_over_nccl_resumes(gpu_workspace):
    needs_two_gpus()
    reference = gpu_workspace("nccl-reference")
    assert torchrun(reference, "train_cuda.py", mode="ddp").returncode == 0
    expected = losses(read_trace(reference))

    run = gpu_workspace("nccl-interrupted")
    torchrun(run, "train_cuda.py", mode="ddp", die_at=CRASH_AT)
    assert torchrun(run, "train_cuda.py", trace_name="trace2.jsonl", mode="ddp").returncode == 0

    assert losses(read_trace(run, "trace2.jsonl")) == expected[RESUME_FROM - 1 :]
