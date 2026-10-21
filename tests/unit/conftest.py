"""The unit tests: one process, no nodes talking to each other (GPU-256).

What belongs here is logic that a single interpreter settles on its own - the
planner, the config, the registry, a resume against a store on local disk. A
test that starts another Python process, or plays several nodes or ranks over
real sockets with deadlines, belongs in ``tests/distributed``: those are the
tests that are only worth a verdict on Linux, and whose timing a loaded machine
can push over. Kept apart, a red here means the logic broke.

The process half of that rule is checked, not trusted. A test here that starts
a Python process fails, naming what it started, so a misplaced test is found
the first time it runs rather than by a CI job that went red for the wrong
reason. A tool the code under test asks the machine about - ``nvidia-smi`` for
the GPU's compute capability - is not a node, and is allowed. So is a fresh
interpreter started to see what an import pulls in, which is the one way to
know that ``ravex.metrics`` loads without the Rust core: a test doing that says
so with ``@pytest.mark.own_interpreter``.

The socket half is not checked: plenty of unit tests talk to a local server
(the S3 server, the metrics endpoint) without being a second node. That one is
the rule in ``CLAUDE.md``, and the measurement that placed every test is in
GPU-256.
"""

import multiprocessing.process
import os
import sys

import pytest

#: The test running now, and what it started. Module state because an audit
#: hook cannot be removed once added, so it is added once and told what to do.
_running = {"test": None, "started": []}


def _is_python(executable) -> bool:
    name = os.path.basename(str(executable)).lower()
    return name.startswith("python") or os.path.abspath(str(executable)) == os.path.abspath(sys.executable)


def _audit(event, args):
    if _running["test"] is None or event != "subprocess.Popen":
        return
    executable, argv = args[0], args[1]
    # A list on POSIX; on Windows the command line arrives already joined.
    if isinstance(argv, (list, tuple)):
        first, shown = (argv[0] if argv else None), " ".join(map(str, argv))
    else:
        first, shown = (str(argv).split()[0].strip("\"") if argv else None), str(argv)
    first = executable or first
    if first is not None and _is_python(first):
        _running["started"].append("subprocess: %s" % shown[:120])


sys.addaudithook(_audit)

# A process started by multiprocessing with the "spawn" context reaches the OS
# through `_posixsubprocess.fork_exec`, which raises no audit event, so the
# start itself is watched.
_original_start = multiprocessing.process.BaseProcess.start


def _start(self, *args, **kwargs):
    if _running["test"] is not None:
        _running["started"].append("multiprocessing: %s" % getattr(self, "name", "?"))
    return _original_start(self, *args, **kwargs)


multiprocessing.process.BaseProcess.start = _start


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _running["test"], _running["started"] = item.nodeid, []
    try:
        yield
    finally:
        _running["test"] = None
    if _running["started"] and item.get_closest_marker("own_interpreter") is None:
        pytest.fail(
            "%s started a Python process (%s). A test that starts processes is a "
            "distributed test: move it to tests/distributed (GPU-256)."
            % (item.nodeid, "; ".join(_running["started"])),
            pytrace=False,
        )
