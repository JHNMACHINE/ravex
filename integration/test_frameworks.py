"""HuggingFace and Lightning, actually run.

"Ravex works with the whole PyTorch ecosystem" follows from where the hooks
are: HuggingFace, Lightning, Accelerate and DeepSpeed all end up calling
`torch.optim.Optimizer.step` and iterating a `torch.utils.data.DataLoader`. That
is a good argument. It is not evidence.

It is also, for one of those four, wrong — and the gap between the list in that
sentence and the two scripts below is where it hid. DeepSpeed was named in the
argument and never put under test, and when it finally was, on 2026-08-31, the
checkpoints it produced had no weights in them: ZeRO gives the base optimizer
flat partition buffers, so hooking `Optimizer.step` sees an optimizer that owns
nothing of the model. Fixed, and the evidence now lives in
`test_deepspeed.py` rather than in this docstring.

What these tests establish, and it is worth being precise about the difference:

*State restoration is exact.* Strip the per-step randomness and a killed run
resumes into a loss sequence identical to the uninterrupted one. Model,
optimizer, LR scheduler and step count all come back.

*Replay is not.* Leave the shuffling and dropout in, and the resumed run
continues correctly from the right state but sees a different draw. Both
frameworks own the loop: they iterate the dataloader on their own schedule and
consume the global RNG around it, so the epoch-start snapshot Ravex replays no
longer lines up. Training is sound; it is simply not the same sequence.

The plain-PyTorch, DDP and FSDP paths *are* bit-exact with randomness on - see
test_process_restart, test_ddp, test_fsdp. This is a framework limitation, not
a general one.
"""

import re
import subprocess
import sys

import pytest

from conftest import SCRIPTS, read_trace

TOTAL_STEPS = 40
CRASH_AT = 20

pytest.importorskip("transformers")
pytest.importorskip("lightning")

SCRIPT_IDS = {"train_hf.py": "huggingface", "train_lightning.py": "lightning"}
SCRIPTS_UNDER_TEST = list(SCRIPT_IDS)


def run(directory, script, trace_name="trace.jsonl", extra=(), **script_args):
    command = [sys.executable, str(SCRIPTS / script), "--trace", trace_name]
    for key, value in script_args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    command += list(extra)
    return subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=900
    )


def losses(trace):
    return [entry["loss"] for entry in trace]


@pytest.fixture
def framework_workspace(workspace):
    return lambda name: workspace(name, max_steps=TOTAL_STEPS, checkpoint_every=4)


@pytest.mark.parametrize("script", SCRIPTS_UNDER_TEST, ids=SCRIPT_IDS.values())
def test_ravex_intercepts_the_framework_at_all(framework_workspace, script):
    """The floor: does anything get checkpointed when the framework drives?"""
    directory = framework_workspace(f"{script}-plain")
    result = run(directory, script, epochs=3)

    assert result.returncode == 0, result.stdout + result.stderr

    log = (directory / "ravex.log").read_text()
    assert "Ravex active" in log

    written = sorted((directory / "checkpoints").glob("step_*.pt"))
    assert len(written) > 1, (
        "expected periodic checkpoints, not just one at exit - the framework's "
        "dataloader or optimizer wrapper is bypassing the batch-boundary hook\n"
        + log
    )


@pytest.mark.parametrize("script", SCRIPTS_UNDER_TEST, ids=SCRIPT_IDS.values())
def test_state_comes_back_exactly(framework_workspace, script):
    """With the randomness removed, a resumed run must be indistinguishable.

    This is the assertion that pins down model, optimizer, LR schedule and step
    count. Anything that failed to restore would show up here as a divergent
    loss, because nothing else varies between the two runs.
    """
    deterministic = ("--no-shuffle", "--no-dropout")

    reference = framework_workspace(f"{script}-exact-reference")
    result = run(reference, script, extra=deterministic)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = losses(read_trace(reference))
    assert len(expected) >= CRASH_AT

    run_dir = framework_workspace(f"{script}-exact-interrupted")
    killed = run(run_dir, script, die_at=CRASH_AT, extra=deterministic)
    assert killed.returncode != 0, "expected the run to be killed"
    assert losses(read_trace(run_dir)) == expected[: len(losses(read_trace(run_dir)))]

    restarted = run(run_dir, script, trace_name="trace2.jsonl", extra=deterministic)
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr

    after = losses(read_trace(run_dir, "trace2.jsonl"))
    assert after, "the resumed run trained nothing"
    assert after[0] in expected, "the resumed run did not rejoin the reference"

    start = expected.index(after[0])
    assert start > 0, "the resumed run started over instead of resuming"
    assert after == expected[start : start + len(after)], (
        f"diverged after rejoining at step {start + 1}"
    )


@pytest.mark.parametrize("script", SCRIPTS_UNDER_TEST, ids=SCRIPT_IDS.values())
def test_a_killed_run_resumes_rather_than_restarting(framework_workspace, script):
    """With shuffling and dropout on, the run continues - just not identically.

    Both frameworks iterate the dataloader on their own schedule and draw from
    the global RNG around the loop, so the epoch-start snapshot Ravex replays
    no longer lines up and the resumed run sees a different draw. What must
    still hold is that it picks up from the checkpointed state instead of
    starting from scratch, and that it respects the step budget.
    """
    reference = framework_workspace(f"{script}-reference")
    assert run(reference, script).returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) >= CRASH_AT

    run_dir = framework_workspace(f"{script}-interrupted")
    killed = run(run_dir, script, die_at=CRASH_AT)
    assert killed.returncode != 0

    before = losses(read_trace(run_dir))
    assert before == expected[: len(before)], "diverged before the crash"

    restarted = run(run_dir, script, trace_name="trace2.jsonl")
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr

    after = losses(read_trace(run_dir, "trace2.jsonl"))
    assert after, "the resumed run trained nothing"
    assert after[0] != expected[0], "the resumed run started from scratch"

    log = (run_dir / "ravex.log").read_text()
    assert "Resumed at step" in log, log

    # The budget is global. Without it a resumed script would run its own loop
    # bounds again from the top.
    assert len(before) + len(after) <= TOTAL_STEPS + CRASH_AT


#: Three epochs of eight batches: the framework's own budget, with Ravex's
#: `max_steps` switched off so that nothing but the framework stops the run.
BUDGET_EPOCHS = 3
BUDGET_STEPS = 8 * BUDGET_EPOCHS

#: The adapters that restore the counter the framework stops on.
ADAPTED = ["train_hf.py"]


@pytest.mark.parametrize("script", ADAPTED, ids=[SCRIPT_IDS[s] for s in ADAPTED])
def test_a_resumed_run_keeps_the_frameworks_step_budget(workspace, script):
    """The framework's loop, not Ravex's `max_steps`, decides when this ends.

    Which is the ordinary case: nobody sets `max_steps` in `ravex.yaml` when
    `num_train_epochs` or `max_epochs` already says how long to train. Ravex
    restores the weights, the moments and the dataset position underneath a
    loop that builds its step counter fresh — so until the adapters existed the
    resumed run trained its whole budget again from the checkpoint, and the
    looser assertion above could not tell.
    """
    directory = workspace(f"{script}-budget", max_steps=None)
    killed = run(directory, script, epochs=BUDGET_EPOCHS, die_at=10)
    assert killed.returncode != 0, "expected the run to be killed"

    restarted = run(directory, script, trace_name="trace2.jsonl", epochs=BUDGET_EPOCHS)
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr

    log = (directory / "ravex.log").read_text()
    resumed = re.search(r"Resumed at step (\d+)", log)
    assert resumed, log
    resumed_at = int(resumed.group(1))
    assert resumed_at > 0

    after = losses(read_trace(directory, "trace2.jsonl"))
    assert len(after) == BUDGET_STEPS - resumed_at, (
        f"resumed at step {resumed_at} of {BUDGET_STEPS} and trained "
        f"{len(after)} more"
    )
