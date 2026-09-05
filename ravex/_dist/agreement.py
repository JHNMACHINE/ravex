"""Agreeing about small things without a collective.

Ravex asks its ranks to agree about a handful of tiny facts — the newest step
every rank holds, whether a resume succeeded everywhere, whether anyone caught
SIGTERM — and every one of those is an all-gather of a scalar followed by a
reduction done in Python on a list. Not one of them reduces a tensor. So what
this module provides is **one** primitive, :func:`all_gather_scalar`, and not a
general ``all_reduce``: writing dtypes, reduction ops and ring algorithms for
payloads of eight bytes would be most of a library for none of the reasons a
library like that exists.

**The medium is the rendezvous store, which is not a collective.** It needs no
registered process group, it has no device (so GPU-82 — CPU tensors handed to
the default NCCL group — cannot happen here), and it does not require everyone
to be present. ``elastic.announce_join``/``pending_join`` already use it for a
candidate that belongs to no group yet, and GPU-109 used it to address the
replication ring.

**What it changes when a rank does not answer.** A collective waits, and then
the whole group fails together, which is the behaviour that makes a missing
rank an outage rather than an answer. Here the question has a deadline, and
running out of it *is* an answer: for "did every rank succeed", a rank that
never said so did not, and that is exactly what the caller wants to know. So
the failure mode moves from a hang to a decision, and the log says which rank
never voted — which the collective never could.

**Measured, and the measurement bounded the claim** (`bench/agreement_cost.py`,
4 ranks, gloo on loopback). ``all_ranks_agree`` costs **1544 µs** over the
collectives and **849 µs** here: 1.8x, not the twenty that a bare store lookup
(70–98 µs) would suggest. The difference between those two numbers is the whole
point, and it is worth stating before someone quotes the wrong one: **a gather
cannot escape waiting for the slowest rank on any medium**. With 5 ms of skew
on one rank both roads pay it — 5.9 ms here against 6.9 ms there — because both
need an answer from everybody. What the store changes for a gather is the cost
of the mechanism, not the cost of waiting.

The twenty belongs to a different shape: a question that does *not* need
everyone's answer, like "has anyone asked for an emergency checkpoint", where
the common answer is the absence of a key and no rank waits for any other. That
is the next step of GPU-111 and it is not this one.

**Scalars only, on purpose.** ``bool``, ``int``, ``str`` and nothing else:
those round-trip through JSON exactly. The gathers that carry structures
(``gather_visible_stores``, ``agree_on_run_id``) stay on the collectives until
someone decides how a peer-written blob should be parsed, because that decision
is a security question and not an encoding one.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ravex")

#: ``name`` says which question, ``round`` which time it was asked, ``rank`` who
#: answered. Stamped with the issue so a store dump is readable by whoever finds
#: it.
KEY = "ravex/gpu111/agree/%s/%d/%d"

#: The fallback poll, for a store without ``wait``. It starts here and backs
#: off to :data:`SLOWEST_POLL`.
#:
#: **This is the slow road and the numbers say so.** Polling was the first
#: implementation and it measured *worse than the collective it replaced* — 1592
#: µs against 1088 — because a rank that arrives 100 µs late still costs the
#: others a whole sleep, and because reading four answers was four round trips.
#: `store.wait` blocks on the server and wakes on the write, and `multi_get`
#: reads them all at once; between them the call is three round trips and no
#: sleeping at all. See GPU-111 and `bench/agreement_cost.py`.
FASTEST_POLL = 0.0005
SLOWEST_POLL = 0.05

#: Rounds kept before the keys are dropped. Two, so a rank still reading round
#: *n* is never racing a delete of round *n*.
KEEP_ROUNDS = 2

_rounds: Dict[Any, int] = {}
_rounds_lock = threading.Lock()


def _next_round(name: str, rank: int) -> int:
    """How many times *this rank* has asked this question. Counted locally.

    Every rank reaches the same call sites in the same order — that is not an
    assumption this module adds, it is the invariant a collective already
    demands, and violating it costs a hang there and a wrong round here. The
    difference is that a wrong round answers ``False`` rather than waiting for
    a timeout that never comes.

    Counted per rank rather than per process, and that is not only for the
    tests that put several ranks in one interpreter: it is what the number
    means. A process is one rank in production, so the two readings coincide
    there — but only one of them stays true when they do not.
    """
    with _rounds_lock:
        current = _rounds.get((name, rank), 0)
        _rounds[(name, rank)] = current + 1
        return current


def reset_rounds() -> None:
    """Forget the round counters. For tests, and for a runtime being rebuilt."""
    with _rounds_lock:
        _rounds.clear()


def wanted_transport() -> str:
    """What the running configuration says: ``auto``, ``store``, or
    ``collectives``.

    Asked of the live runtime rather than by loading the config, because this
    is on the path of a replication round and `RavexConfig.load()` reads a file.
    No runtime means nothing is being checkpointed by this process, so the
    answer does not matter and ``auto`` is as good as any.
    """
    try:
        from ravex._runtime import get_runtime

        runtime = get_runtime(create=False)
        if runtime is not None:
            return str(runtime.config.agreement_transport)
    except Exception:
        pass
    return "auto"


def rendezvous_store():
    """The key-value store torch built for this job, or None.

    Private API, reached through one ``try``: this is the rendezvous every rank
    of a torchrun job already shares and there is no public way to ask for it.
    If a torch release moves it, everything here quietly goes back to the
    collectives instead of breaking.
    """
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None
        return dist.distributed_c10d._get_default_store()
    except Exception:
        return None


def default_patience() -> float:
    """How long the collective this replaces would have waited.

    Deliberately the same number rather than a shorter one of this module's
    own. ``all_ranks_agree`` gates a resume, and a rank that takes ninety
    seconds to read its checkpoint is slow, not absent; a tighter deadline here
    would turn that into "not everyone succeeded" and throw away a resume the
    collective would have waited for.
    """
    try:
        import torch.distributed as dist

        return float(dist.distributed_c10d.default_pg_timeout.total_seconds())
    except Exception:
        return 1800.0


def _forget(store, name: str, round_number: int, world_size: int) -> None:
    """Drop a round's keys, best effort, from one rank only.

    Without this a long run leaves a key per rank per round on the store
    forever — small, but unbounded, and the store is a service the training job
    shares. Only rank 0 deletes so the work is N and not N².
    """
    if round_number < 0:
        return
    for rank in range(world_size):
        try:
            store.delete_key(KEY % (name, round_number, rank))
        except Exception:
            return  # no delete_key on this store; leaving them is not a failure


def all_gather_scalar(
    name: str,
    value: Any,
    rank: int,
    world_size: int,
    store,
    patience: Optional[float] = None,
) -> Optional[List[Any]]:
    """Every rank's answer to one small question, or None if someone never said.

    Posting happens before reading, which is what makes this a rendezvous as
    well as a gather: a rank that has everyone's answer knows everyone arrived.

    ``None`` means the deadline passed with someone still silent. It is not an
    error and must not be turned into one by the caller — it is the honest
    "I do not know what rank 5 thinks", and what that means is the caller's to
    decide, because it differs by question. For "did everyone succeed" it means
    no; for "what is the newest step everyone holds" it means there is no
    answer and the run starts from scratch.
    """
    if not isinstance(value, (bool, int, str)):
        raise TypeError(
            "all_gather_scalar carries scalars only, not %s. A structure needs "
            "an encoding decided on purpose - see the module docstring"
            % type(value).__name__
        )
    if world_size <= 1:
        return [value]

    round_number = _next_round(name, rank)
    keys = [KEY % (name, round_number, other) for other in range(world_size)]
    store.set(keys[rank], json.dumps(value).encode("utf-8"))

    budget = patience if patience is not None else default_patience()
    blobs = _await_all(store, keys, budget)
    if blobs is None:
        missing = [
            other for other in range(world_size) if not store.check([keys[other]])
        ]
        logger.warning(
            "No answer from rank(s) %s to '%s' after %.0fs. Treating that as an "
            "answer rather than waiting on it: what it means for this question "
            "is the caller's to say",
            ", ".join(str(other) for other in missing) or "?",
            name,
            budget,
        )
        return None

    answers = [json.loads(blob.decode("utf-8")) for blob in blobs]
    if rank == 0:
        _forget(store, name, round_number - KEEP_ROUNDS, world_size)
    return answers


def _await_all(store, keys, budget: float) -> Optional[List[bytes]]:
    """Every key's value once they all exist, or None if the budget ran out.

    Two roads, and the first is the one that matters. ``wait`` blocks inside the
    store server and wakes on the write, so a rank that arrives 100 µs after the
    others costs 100 µs; ``multi_get`` then reads every answer in one round
    trip. Three round trips for the whole call, no sleeping.

    The polling fallback is for a store that has neither — and it is genuinely
    slower, not merely less elegant: it was the first implementation here, and
    the bench measured it at 1592 µs against the 1088 µs of the collective it
    was replacing. A sleep ladder pays for a rank being *slightly* late at the
    granularity of the ladder, and reading answers one at a time pays a round
    trip each.
    """
    if hasattr(store, "wait") and hasattr(store, "multi_get"):
        try:
            store.wait(keys, datetime.timedelta(seconds=budget))
        except Exception:
            return None
        try:
            return list(store.multi_get(keys))
        except Exception:
            return None

    deadline = time.monotonic() + budget
    wait = FASTEST_POLL
    while not store.check(keys):
        if time.monotonic() >= deadline:
            return None
        time.sleep(wait)
        wait = min(SLOWEST_POLL, wait * 2)
    return [store.get(key) for key in keys]
