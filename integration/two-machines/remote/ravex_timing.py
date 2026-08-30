"""Time the two costs GPU-81 asks for, without touching the repository.

Ravex logs *that* it replicated, never how long it took, and the whole question
in GPU-81 is whether the transfer fits inside the interval between replications.
So this wraps three seams from outside:

* ``MoonclipBackend.consolidate`` — the ``merge_now`` that collapses the delta
  chain before a copy leaves the machine. It runs on the training thread, it is
  a cost that did not exist before GPU-76, and it is measure 2.
* ``_replication.exchange_stores`` — the bytes on the wire, with the encoded
  size of what was sent, which gives measure 1 and an effective rate.
* the training step, so the interval between replications is measured in the
  same run rather than assumed from an older one. On a rented box the disk
  degrades as the session goes on, and a number from an hour ago is a number
  about a different machine.

One JSONL per rank, appended, flushed per line: a preemption mid-run must not
take the phases that already completed with it.
"""

import json
import os
import time

_OUT = os.environ.get("RAVEX_TIMING_OUT", "/root/out/timing.jsonl")


class _Sink:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rank = os.environ.get("RANK", "0")
        stem, ext = os.path.splitext(path)
        self._fh = open(f"{stem}.rank{rank}{ext}", "a", buffering=1)
        self.rank = int(rank)

    def write(self, **record):
        record["rank"] = self.rank
        record["t"] = time.time()
        self._fh.write(json.dumps(record) + "\n")
        os.fsync(self._fh.fileno())


_sink = None


def install():
    global _sink
    if _sink is not None:
        return
    _sink = _Sink(_OUT)

    from ravex import _backends, _replication, _runtime

    original_consolidate = _backends.MoonclipBackend.consolidate

    def consolidate(self):
        started = time.perf_counter()
        try:
            return original_consolidate(self)
        finally:
            _sink.write(kind="merge_now", seconds=time.perf_counter() - started)

    _backends.MoonclipBackend.consolidate = consolidate

    # Since GPU-97, `exchange_stores` skips files the peer already has, so
    # `encoded_size(source)` overstates what actually crosses the wire — it
    # is the number that used to be true when every round sent the whole
    # store, and reporting it unchanged here would hide the fix it exists to
    # measure. `exchange_stores` computes the real figure itself, as
    # `encoded_size(source, skip=...)`, right before sizing the transfer;
    # this hooks that same call to read it back rather than recomputing a
    # skip set this wrapper has no way to reproduce (it does not see what the
    # peer reported over the wire).
    original_encoded_size = _replication.encoded_size
    last_outgoing = {"value": None}

    def encoded_size(path, skip=None):
        value = original_encoded_size(path, skip=skip)
        if skip is not None:
            last_outgoing["value"] = value
        return value

    _replication.encoded_size = encoded_size

    original_exchange = _replication.exchange_stores

    def exchange_stores(source, destination, send_to, receive_from, **kwargs):
        last_outgoing["value"] = None
        started = time.perf_counter()
        ok = original_exchange(source, destination, send_to, receive_from, **kwargs)
        seconds = time.perf_counter() - started
        sent = last_outgoing["value"] or 0
        # The store's current size on disk, not "bytes pulled this round" —
        # a skipped file is already there and never crosses the wire, but it
        # is still part of what this replica now holds.
        received = _replication.encoded_size(destination) if destination else 0
        _sink.write(
            kind="exchange",
            seconds=seconds,
            sent_bytes=sent,
            received_bytes=received,
            send_to=send_to,
            receive_from=receive_from,
            ok=bool(ok),
        )
        return ok

    _replication.exchange_stores = exchange_stores

    # The step clock. `_replicate_if_due` is called once per step from the
    # runtime, so the gap between consecutive calls is the step time as the
    # replication sees it — including whatever the checkpoint handoff costs,
    # which is the interval that matters here.
    original_due = _runtime.RavexRuntime._replicate_if_due
    state = {"last": None, "step": None}

    def _replicate_if_due(self, step):
        now = time.perf_counter()
        if state["last"] is not None and state["step"] is not None:
            _sink.write(
                kind="step",
                step=step,
                seconds=(now - state["last"]) / max(step - state["step"], 1),
            )
        state["last"], state["step"] = now, step
        return original_due(self, step)

    _runtime.RavexRuntime._replicate_if_due = _replicate_if_due
