"""The distributed tests: several nodes or ranks, as processes or over sockets (GPU-256).

Everything here starts another Python process - ``multiprocessing`` with the
spawn context, a ``ravex rendezvous`` server, a script run as a child - or
plays several nodes or ranks in threads that talk over real sockets with
deadlines: the rendezvous, the outer loop, failover, elastic membership,
replication.

**The verdict on these comes from Linux.** On Windows orphaned children outlive
a failed test and break the next ones, `TCPStore` has no libuv and connects
slowly through Docker Desktop's hosts entry; locally, run them in the CI
container (`ravex-ci:local`) with `MOONCLIP_FSYNC=0`. The timed ones - a node
that falls off, an election, a gated round - only CI decides: a loaded machine
moves them either way.

Each test is marked ``distributed``, for selecting or deselecting them by
marker as well as by directory.
"""

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(items):
    for item in items:
        if os.path.abspath(str(item.fspath)).startswith(_HERE + os.sep):
            item.add_marker(pytest.mark.distributed)
