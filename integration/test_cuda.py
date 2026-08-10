"""The parts that only a GPU can prove.

Everything here is skipped without CUDA, which is why it lives apart from the
rest of the integration suite: the development machine has no GPU, so these run
on a rented box.

Three things are unverifiable on CPU:

- the AMP loss scale, because ``GradScaler`` never actually scales anything on
  CPU and so never overflows or adjusts;
- the CUDA RNG, which is a separate generator from the CPU one;
- FSDP1, which refuses to initialise without an accelerator.

Two things about comparing runs here that do not come up on CPU:

**Iterations are not steps.** Under AMP an overflowing gradient makes
``scaler.step()`` skip the optimizer. The iteration happened, the step did not,
nothing about the model changed, and Ravex - correctly - does not count it. So
the skipped lines are filtered out before two traces are lined up.

**Collectives are not bit-reproducible.** NCCL picks its reduction order at
runtime, so two identical distributed runs differ in the last few bits.
Single-GPU comparisons stay exact; distributed ones use a tolerance, still far
tighter than any real state-restoration bug could hide behind.
"""

import os
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

#: Makes cuBLAS pick a deterministic algorithm. Has to be set before CUDA
#: initialises, so it goes in the child's environment.
DETERMINISTIC_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

#: Tolerance for the steps *after* the first resumed one in a distributed run.
#: NCCL picks its reduction order at runtime, so two identical runs differ in
#: the last bits, and training amplifies that difference step by step. 1% still
#: catches every real failure: a rank that did not resume poisons the
#: all-reduce and its losses come out tens of percent away, not fractions.
COLLECTIVE_TOLERANCE = 1e-2


def run_single(directory, trace_name="trace.jsonl", **script_args):
    command = [sys.executable, str(SCRIPTS / "train_cuda.py"), "--trace", trace_name]
    for key, value in script_args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    return subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, **DETERMINISTIC_ENV},
    )


def stepped(trace):
    """Iterations that actually moved the optimizer, in order.

    Under AMP an overflow makes ``scaler.step()`` skip the optimizer, so the
    file has more lines than there were steps. Dropping the skipped ones lines
    the sequence back up with what Ravex counts.
    """
    return [entry for entry in trace if entry["stepped"]]


def assert_matches(actual, expected, first_step, exact=True):
    """Compare a resumed run against the tail of the uninterrupted one.

    The script's own counter restarts at 1 in the resumed process - it has no
    idea a previous one existed - so the alignment is positional: the resumed
    run's first step is the reference's step `first_step`.
    """
    tail = expected[first_step - 1 :]
    assert len(actual) == len(tail), (
        f"resumed run produced {len(actual)} steps, expected {len(tail)}"
    )

    for offset, (got_entry, want_entry) in enumerate(zip(actual, tail)):
        step = first_step + offset
        got, want = float(got_entry["loss"]), float(want_entry["loss"])

        # The first resumed step is a forward pass over the restored weights,
        # before any collective runs in that iteration. It is exact even under
        # NCCL, and it is the assertion that actually proves every rank came
        # back with the right state. Everything after it inherits whatever
        # ordering the collectives chose this time.
        if exact or offset == 0:
            assert got == want, f"step {step}: {got!r} != {want!r}"
        else:
            assert got == pytest.approx(want, rel=COLLECTIVE_TOLERANCE), (
                f"step {step}: {got!r} vs {want!r}"
            )


def needs_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs to shard across")


@pytest.fixture
def gpu_workspace(workspace):
    return lambda name: workspace(name, max_steps=TOTAL_STEPS)


# --- AMP ------------------------------------------------------------


def test_amp_resumes_with_the_same_losses_and_loss_scale(gpu_workspace):
    reference = gpu_workspace("amp-reference")
    result = run_single(reference, mode="amp")
    assert result.returncode == 0, result.stderr

    raw = read_trace(reference)
    expected = stepped(raw)
    assert len(expected) == TOTAL_STEPS
    assert any(not entry["stepped"] for entry in raw), (
        "fp16 should have overflowed at least once; without a skipped step this "
        "test is not exercising the scaler"
    )

    run = gpu_workspace("amp-interrupted")
    killed = run_single(run, mode="amp", die_at=CRASH_AT)
    assert killed.returncode == -9

    restarted = run_single(run, trace_name="trace2.jsonl", mode="amp")
    assert restarted.returncode == 0, restarted.stderr
    after = stepped(read_trace(run, "trace2.jsonl"))

    # Single GPU, no collectives: this can and must be exact.
    assert_matches(after, expected, RESUME_FROM, exact=True)

    # The loss scale is tuned state. A resume that reset it to init_scale would
    # spend the next few hundred steps overflowing its way back to where it was.
    for offset, (got, want) in enumerate(zip(after, expected[RESUME_FROM - 1 :])):
        assert got["scale"] == want["scale"], f"scale at step {RESUME_FROM + offset}"


def test_the_cuda_rng_state_is_restored(gpu_workspace):
    """Dropout on the GPU draws from the CUDA generator, not the CPU one.

    If only the CPU RNG came back, the weights would still look right and the
    losses would quietly diverge - the exact failure this project exists to
    prevent.
    """
    reference = gpu_workspace("rng-reference")
    assert run_single(reference, mode="amp").returncode == 0
    expected = stepped(read_trace(reference))

    run = gpu_workspace("rng-interrupted")
    run_single(run, mode="amp", die_at=CRASH_AT)
    assert run_single(run, trace_name="trace2.jsonl", mode="amp").returncode == 0
    after = stepped(read_trace(run, "trace2.jsonl"))

    assert after[0]["loss"] == expected[RESUME_FROM - 1]["loss"], (
        "the first resumed step already differs"
    )
    assert_matches(after, expected, RESUME_FROM, exact=True)


# --- FSDP1 and NCCL -------------------------------------------------


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
    expected = stepped(read_trace(reference))
    assert len(expected) == TOTAL_STEPS

    run = gpu_workspace("fsdp1-interrupted")
    assert torchrun(run, "train_cuda.py", mode="fsdp1", die_at=CRASH_AT).returncode != 0

    restarted = torchrun(run, "train_cuda.py", trace_name="trace2.jsonl", mode="fsdp1")
    assert restarted.returncode == 0, restarted.stderr

    after = stepped(read_trace(run, "trace2.jsonl"))
    assert_matches(after, expected, RESUME_FROM, exact=False)


def test_ddp_over_nccl_resumes(gpu_workspace):
    needs_two_gpus()
    reference = gpu_workspace("nccl-reference")
    assert torchrun(reference, "train_cuda.py", mode="ddp").returncode == 0
    expected = stepped(read_trace(reference))

    run = gpu_workspace("nccl-interrupted")
    torchrun(run, "train_cuda.py", mode="ddp", die_at=CRASH_AT)
    restarted = torchrun(run, "train_cuda.py", trace_name="trace2.jsonl", mode="ddp")
    assert restarted.returncode == 0, restarted.stderr

    after = stepped(read_trace(run, "trace2.jsonl"))
    assert_matches(after, expected, RESUME_FROM, exact=False)
