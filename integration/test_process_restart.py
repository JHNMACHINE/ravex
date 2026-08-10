"""Kill a real process, start it again, check nothing was lost.

The in-process unit test simulates a crash with an exception. This one uses
SIGKILL on a separate interpreter, so the resumed run starts from an empty
process: no warm registry, no leftover patches, no cached config. Ravex has to
find its way back purely from what is on disk.
"""

import pytest

from conftest import read_trace, run_training

TOTAL_STEPS = 40
CHECKPOINT_EVERY = 4
CRASH_AT = 20

#: A hard kill loses everything after the last checkpoint that was actually
#: collected. The one due at step 20 would have been taken at the top of
#: iteration 21, which never ran — so step 16 is the recovery point and the
#: resumed run replays 17-20 before reaching new ground.
RESUME_FROM = 17


def losses(trace):
    return [entry["loss"] for entry in trace]


def test_the_autoloader_activates_without_being_asked(workspace):
    directory = workspace("plain")
    result = run_training(directory, epochs=2)

    assert result.returncode == 0, result.stderr
    # Nothing in the script mentions Ravex, yet:
    assert (directory / "checkpoints").is_dir()
    assert "Ravex active" in (directory / "ravex.log").read_text()


def test_ravex_stays_out_of_a_project_that_did_not_opt_in(workspace, tmp_path):
    directory = tmp_path / "no-config"
    directory.mkdir()

    result = run_training(directory, epochs=2)

    assert result.returncode == 0, result.stderr
    assert not (directory / "checkpoints").exists()
    assert not (directory / "ravex.log").exists()


def test_a_killed_run_resumes_where_it_stopped(workspace):
    reference = workspace("reference")
    assert run_training(reference).returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) == TOTAL_STEPS, "max_steps should stop the run at 40"

    # Same script, killed outright at step 20.
    run = workspace("interrupted")
    killed = run_training(run, die_at=CRASH_AT)
    assert killed.returncode == -9, "expected SIGKILL, not a clean exit"

    before = losses(read_trace(run))
    assert len(before) == CRASH_AT
    assert before == expected[:CRASH_AT], "diverged before the crash"

    # The process comes back. Same command, same directory, no code change.
    restarted = run_training(run, trace_name="trace2.jsonl")
    assert restarted.returncode == 0, restarted.stderr

    after = losses(read_trace(run, "trace2.jsonl"))
    assert after == expected[RESUME_FROM - 1 :], (
        "the resumed run must reproduce the tail of the uninterrupted one"
    )

    # And the budget is global, not per-process: 40 steps total across both
    # runs plus the replayed overlap, never 40 more.
    assert len(before) + len(after) == TOTAL_STEPS + (CRASH_AT - RESUME_FROM + 1)


def test_the_learning_rate_survives_the_restart(workspace):
    reference = workspace("lr-reference")
    run_training(reference)
    expected = [entry["lr"] for entry in read_trace(reference)]

    run = workspace("lr-interrupted")
    run_training(run, die_at=CRASH_AT)
    run_training(run, trace_name="trace2.jsonl")
    after = [entry["lr"] for entry in read_trace(run, "trace2.jsonl")]

    # StepLR decays at steps 10, 20, 30, 40. A resume that restored the model
    # but not the scheduler would restart the decay schedule from zero here.
    assert after == expected[RESUME_FROM - 1 :]


@pytest.mark.parametrize("workers", [2])
def test_resume_is_exact_with_dataloader_workers(workspace, workers):
    """With workers, the sampler runs ahead of the training loop.

    The recorded position has to be the loop's batch count, not the sampler's,
    or the resumed run restarts a prefetch-depth's worth of batches too late.
    """
    reference = workspace(f"workers{workers}-reference")
    assert run_training(reference, workers=workers).returncode == 0
    expected = losses(read_trace(reference))
    assert len(expected) == TOTAL_STEPS

    run = workspace(f"workers{workers}-interrupted")
    run_training(run, die_at=CRASH_AT, workers=workers)
    assert run_training(run, trace_name="trace2.jsonl", workers=workers).returncode == 0

    after = losses(read_trace(run, "trace2.jsonl"))
    assert after == expected[RESUME_FROM - 1 :]
