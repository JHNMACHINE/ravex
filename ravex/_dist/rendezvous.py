"""A rendezvous of Ravex's own: nodes that start on their own, with no torchrun.

GPU-129, under GPU-113. Everything above this module — the round exchange, the
membership of GPU-121, the job token — talks to a key-value store and to
nothing else. Until now that store was torch's: the one ``init_process_group``
builds, which exists only once every rank has arrived, lives inside the
torchrun agent of whichever machine is the rendezvous endpoint, and dies with
that machine. For a run whose nodes are rented on different continents and
come and go, two of those are disqualifying:

* **Everybody has to be there at the start.** ``init_process_group`` is
  collective. A node that arrives later has no rank and no world size to be
  given, which is why GPU-121's join was reachable from a test and from nothing
  a user could launch.
* **The store dies with one node.** The exchange reads peer addresses from it
  on every fetch, so losing the endpoint's machine stops every round, not only
  that node's contribution.

So the store moves out of the training processes into one of its own:
``ravex rendezvous``. It is torch's ``TCPStore`` — the same object, with the
``wait`` and ``multi_get`` the agreement code already relies on — hosted by a
small process that trains nothing and can be kept alive somewhere that is not a
spot GPU. The nodes are plain ``python train.py``.

**Identity is a counter, and a number is never handed out twice.** A node takes
the next one with ``add``, which the server applies atomically. The first
``min_nodes`` are the base members: they wait for each other and start the run
together, as ranks did under torchrun. Every later number is a joiner and goes
through :func:`ravex._dist.membership.join`. A node that crashes and comes back
is a new node with a new number, which is the only reading under which
"membership is a function of the round number" survives a restart.

**What it does not do, said before anyone relies on it.**

* ``TCPStore`` has **no authentication**. Whoever reaches the port can take a
  number, and write any key. With ``RAVEX_JOB_TOKEN`` set on every node
  (GPU-134) that stranger still cannot put a delta into the average: the token
  is not on the store, and the exchange and the ring prove it without sending
  it. It can still slow a round down, or muddle the membership. Without the
  token, rank 0 leaves one on the store, and the port belongs on a private
  network.
* The server is **still a single point** — moved from a rented GPU to a process
  that is cheap to keep alive. If it restarts, the job's state on it is gone.
* A base member that dies **before** the run starts cannot be replaced by
  relaunching it: the relaunch takes the next number and arrives as a joiner,
  into a run that never started.
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from typing import Tuple

logger = logging.getLogger("ravex")

#: Where a node finds the server when the configuration does not say.
ENV = "RAVEX_RENDEZVOUS"

#: The port ``ravex rendezvous`` listens on unless told otherwise. Not torch's
#: 29500, so a rendezvous and a torchrun job can share a box.
DEFAULT_PORT = 29400

#: Every key a job writes lives under this, so one server holds many jobs —
#: which is what a platform hosting it needs, and costs the process nothing.
JOB_PREFIX = "ravex/job/%s/"

#: A job name goes into every key, so it is kept to characters that cannot
#: change what a key means.
_JOB_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

#: The counter node numbers come from.
NEXT_KEY = "ravex/gpu129/next"

#: How often a waiting base member looks again.
POLL = 0.05


def parse_address(value: str) -> Tuple[str, int]:
    """``host:port``, ``[v6]:port``, or a host alone for :data:`DEFAULT_PORT`."""
    text = str(value).strip()
    host, sep, port = text.rpartition(":")
    if not sep or (text.count(":") > 1 and not text.startswith("[")):
        # No colon, or a bare IPv6 address with no port.
        host, port = text, ""
    host = host.strip("[]")
    if not host:
        raise ValueError("rendezvous address %r has no host" % (value,))
    if not port:
        return host, DEFAULT_PORT
    try:
        number = int(port)
    except ValueError:
        raise ValueError(
            "rendezvous address %r has port %r, which is not a number" % (value, port)
        ) from None
    if not 0 < number < 65536:
        raise ValueError("rendezvous address %r has port %d, out of range" % (value, number))
    return host, number


def valid_job(name: object) -> bool:
    """Whether ``name`` can go into a key as it is."""
    return isinstance(name, str) and bool(_JOB_NAME.match(name))


def _tcp_store(host: str, port: int, is_master: bool, timeout: float):
    """torch's ``TCPStore``, on whichever implementation this torch has.

    libuv first, because it is torch's default and the one built for many
    clients. Windows CPU wheels are built without it and refuse the default
    with a message naming libuv — and ``USE_LIBUV=0``, which that message
    suggests, is not honoured when the argument is passed. The older
    implementation speaks the same protocol, so that is what they get.
    """
    import torch.distributed as dist

    kwargs = dict(
        world_size=None,
        is_master=is_master,
        timeout=datetime.timedelta(seconds=timeout),
        wait_for_workers=False,
    )
    try:
        return dist.TCPStore(host, port, **kwargs)
    except Exception as exc:
        if "libuv" not in str(exc):
            raise
    return dist.TCPStore(host, port, use_libuv=False, **kwargs)


def serve(host: str = "0.0.0.0", port: int = DEFAULT_PORT):
    """Open the server. It lives exactly as long as the returned object."""
    return _tcp_store(host, port, True, 300.0)


def connect(address: str, job: str, timeout: float = 300.0):
    """A client of the server at ``address``, confined to ``job``'s keys.

    ``timeout`` bounds reaching the server — a node may well be started before
    it — and any blocking read after that. Nothing above this module blocks on
    a read on purpose (it asks ``check`` before ``get``), so in practice it is
    the first.
    """
    import torch.distributed as dist

    if not valid_job(job):
        raise ValueError(
            "job name %r: use letters, digits, '.', '_' and '-', at most 128" % (job,)
        )
    host, port = parse_address(address)
    return dist.PrefixStore(JOB_PREFIX % job, _tcp_store(host, port, False, timeout))


def register(store) -> int:
    """This node's number: the next one, never handed out before."""
    return int(store.add(NEXT_KEY, 1)) - 1


def registered(store) -> int:
    """How many numbers have been handed out, including to nodes long gone."""
    return int(store.add(NEXT_KEY, 0))


def wait_for_base(store, min_nodes: int, deadline: float, poll: float = POLL) -> bool:
    """Wait until the first ``min_nodes`` nodes have registered. False at ``deadline``.

    Only base members wait. They are the ones that start the run, and its seed
    round is theirs: a base member that started training before the others had
    arrived would have nobody to take the starting parameters from, or to give
    them to.
    """
    said = False
    while True:
        count = registered(store)
        if count >= min_nodes:
            return True
        if time.monotonic() >= deadline:
            return False
        if not said:
            logger.info(
                "Waiting at the rendezvous for %d more node(s) to start the run.",
                min_nodes - count,
            )
            said = True
        time.sleep(poll)
