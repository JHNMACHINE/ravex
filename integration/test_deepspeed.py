"""DeepSpeed, actually run — the evidence `test_frameworks.py` names and lacks.

That file's own opening says it plainly: HuggingFace, Lightning, Accelerate and
DeepSpeed all end up calling `torch.optim.Optimizer.step`, "That is a good
argument. It is not evidence." — and then puts HuggingFace and Lightning under
test and leaves DeepSpeed in the argument. On 2026-08-31 the evidence was
gathered and contradicted it: Ravex wrote checkpoints for a DeepSpeed run
containing an optimizer, a dataloader and **no weights at all**, reporting a
successful handoff each time.

Two independent causes, and neither is DeepSpeed doing anything wrong:

- its engine is itself an `nn.Module` holding the user's model, so the model is
  dropped as "contained in something else observed" and the engine becomes the
  candidate;
- ZeRO hands the base optimizer flat partition buffers, so no module owns any
  optimized parameter and the engine is dropped as "not part of the training".

These tests are deliberately about **what is in the checkpoint**, not about
loss curves matching after a restart. The defect was that the file existed and
was empty of the one thing that matters, so that is what is asserted.
"""

import json
import subprocess
import sys

import pytest

import torch

from conftest import SCRIPTS

pytest.importorskip("deepspeed")

STAGES = [1, 2, 3]


def run(directory, stage, port):
    """DeepSpeed under torchrun: its `init_distributed` needs the launcher's env."""
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc-per-node=1", "--master-port=%d" % port,
        str(SCRIPTS / "train_deepspeed.py"),
        "--stage", str(stage),
    ]
    return subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=900
    )


def latest_checkpoint(directory):
    found = sorted((directory / "checkpoints").glob("*.pt"))
    return torch.load(found[-1], map_location="cpu", weights_only=False) if found else None


@pytest.mark.parametrize("stage", STAGES)
def test_the_checkpoint_actually_contains_the_model(workspace, stage):
    """The regression, stated as the thing a user would care about.

    Not "a checkpoint was written" — one always was. The weights are what a
    resume cannot do without, and they were the part that was missing.
    """
    directory = workspace("ds%d" % stage, backend="torch_save")
    result = run(directory, stage, 29610 + stage)
    assert result.returncode == 0, result.stderr[-2000:]

    state = latest_checkpoint(directory)
    assert state is not None, "no checkpoint was written at all"

    models = state.get("models") or {}
    assert models, "the checkpoint holds no model: this is the defect"

    saved = next(iter(models.values()))
    elements = sum(t.numel() for t in saved.values() if torch.is_tensor(t))
    assert elements > 0, "the model's tensors are empty"

    # `unwrap_model` follows the engine's `.module`, so the keys belong to the
    # user's model and not to the wrapper — a checkpoint keyed `module.0.weight`
    # would not load into a plain rerun of the same script.
    assert all(not name.startswith("module.") for name in saved), sorted(saved)[:4]


@pytest.mark.parametrize("stage", STAGES)
def test_a_checkpoint_is_written_at_every_stage(workspace, stage):
    """Stage 3 used to take the whole collection down with it.

    Its base optimizer's `state_dict()` raises a bare `KeyError` on a parameter
    id, which cost the checkpoint its weights too. One optimizer refusing to
    describe itself is now a logged skip.
    """
    directory = workspace("ds_written%d" % stage, backend="torch_save")
    result = run(directory, stage, 29620 + stage)
    assert result.returncode == 0, result.stderr[-2000:]

    log = (directory / "ravex.log").read_text(encoding="utf-8", errors="replace")
    assert "Checkpoint at step" in log
    assert "failed:" not in log, log[-1500:]


def test_the_run_itself_is_untouched(workspace):
    """Ravex is invisible: the losses are the ones the script would have had.

    Run twice with Ravex writing to two different places and no checkpoint to
    resume from either time; the sequences must be identical, or Ravex has
    changed the training rather than observed it.
    """
    first = workspace("ds_a", backend="torch_save", checkpoint_every=1000)
    second = workspace("ds_b", backend="torch_save", checkpoint_every=4)

    assert run(first, 1, 29631).returncode == 0
    assert run(second, 1, 29632).returncode == 0

    def losses(directory):
        lines = (directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["loss"] for line in lines]

    assert losses(first) == losses(second)
