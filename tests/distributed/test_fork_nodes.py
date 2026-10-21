"""A fork: a new run that starts from step N of another (GPU-149) - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_fork.py.
"""


import pytest

import ravex
import ravex.runs
from test_fork import train_to  # noqa: E402


class TestAParentStillTraining:
    """The issue's last condition: the parent carries on without noticing."""

    def test_a_fork_starts_while_the_parent_keeps_training(self, tmp_path, monkeypatch):
        # Two processes on one Moonclip store: the parent writing its manifest
        # at every checkpoint, the child opening it to read step 4 and pin it.
        # The pin must survive the parent's later writes.
        import subprocess
        import sys
        import time

        moonclip = pytest.importorskip("moonclip")
        parent = tmp_path / "base"
        code = f"""
import os, time, torch, ravex
os.environ["RAVEX_STORAGE_PATH"] = {str(parent)!r}

# keep_last=50: step 4 has to outlive the parent's retention until the child
# pins it, which is what this test is about; a full per checkpoint keeps five.
@ravex.train_loop(backend="moonclip", checkpoint_every=2, async_save=False, delta=False, keep_last=50)
def train():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    while ravex.step() < 40:
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        time.sleep(0.1)

train()
"""
        process = subprocess.Popen([sys.executable, "-c", code])
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                described = ravex.runs.describe(str(parent))
                if described and (described["status"].get("step") or 0) >= 6:
                    break
                time.sleep(0.2)
            else:
                raise AssertionError("the parent never reached step 6")

            monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "child"))
            seen = {}
            ravex.train_loop(
                backend="moonclip", checkpoint_every=2, async_save=False, delta=False, fork_from=str(parent), fork_step=4
            )(lambda: train_to(8, seen=seen))()
            assert seen["start"] == 4
            # Forked while the parent was still going.
            assert process.poll() is None
            assert process.wait(timeout=120) == 0
        finally:
            if process.poll() is None:
                process.kill()

        described = ravex.runs.describe(str(parent))
        assert described["status"]["state"] == "finished" and described["status"]["step"] == 40
        manager = moonclip.MoonclipManager(str(parent))
        if hasattr(manager, "pinned_steps"):
            assert 4 in manager.pinned_steps(), "the parent's own writes dropped the fork's pin"
