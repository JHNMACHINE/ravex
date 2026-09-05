import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent / "scripts"

__all__ = ["SCRIPTS", "read_trace", "run_training"]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="needs SIGKILL and torchrun; run in the container"
)


@pytest.fixture(scope="session", autouse=True)
def ravex_is_installed():
    """A real install, with the compiled core in it.

    This replaced a check that the ``.pth`` autoloader was in site-packages,
    which was the thing that made these tests mean anything until GPU-108. What
    makes them mean something now is narrower and easier to state: the training
    scripts import ravex and are decorated, so the suite needs an importable
    ravex — and, since GPU-105, one whose extension module actually built.
    """
    try:
        from ravex import _core  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"ravex is not importable in this interpreter: {exc}")


@pytest.fixture
def workspace(tmp_path):
    """A directory that behaves like a user's project.

    The ravex.yaml here is the only thing that turns Ravex on for processes
    started with this as their working directory.
    """

    def make(name, **overrides):
        directory = tmp_path / name
        directory.mkdir()
        settings = {
            "checkpoint_every": 4,
            "max_steps": 40,
            "backend": "torch_save",
            "keep_last": 5,
        }
        settings.update(overrides)

        lines = [f"{key}: {_yaml(value)}" for key, value in settings.items()]
        (directory / "ravex.yaml").write_text(
            "\n".join(lines)
            + textwrap.dedent(
                """
                storage:
                  type: local
                  path: ./checkpoints
                log_file: ./ravex.log
                log_level: INFO
                """
            ),
            encoding="utf-8",
        )
        return directory

    return make


def _yaml(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def run_training(directory, trace_name="trace.jsonl", **script_args):
    """Run the training script as a real, separate process."""
    command = [sys.executable, str(SCRIPTS / "train_vanilla.py"), "--trace", trace_name]
    for key, value in script_args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]

    return subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=600,
    )


def read_trace(directory, name="trace.jsonl"):
    path = Path(directory) / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
