"""The cadence observation: what it says, and when it stays quiet."""

import logging

from ravex._config import RavexConfig
from ravex._runtime import RavexRuntime


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _runtime(every=2):
    """A runtime plus the records it warns with.

    The handler goes on *after* construction on purpose: `_setup_logging`
    clears the ravex logger's handlers and turns off propagation, so anything
    attached earlier — pytest's `caplog` included — is thrown away.
    """
    config = RavexConfig()
    config.checkpoint_every = every
    runtime = RavexRuntime(config)
    collector = _Collect()
    logging.getLogger("ravex").addHandler(collector)
    return runtime, collector.records


def _handoff(runtime, step, started, finished, phases):
    runtime._warn_if_cadence_is_expensive(step, started, finished, phases)


def test_the_first_checkpoint_says_nothing():
    """There is no interval yet, so there is no fraction to report."""
    runtime, warnings = _runtime()
    _handoff(runtime, 2, 0.0, 10.0, {"drain": 3.0, "collect": 5.0, "store": 2.0})
    assert warnings == []


def test_an_expensive_cadence_is_reported_once():
    runtime, warnings = _runtime(every=2)
    # First handoff only establishes the reference point.
    _handoff(runtime, 2, 0.0, 10.0, {"drain": 3.0, "collect": 1.6, "store": 1.0})
    # 2.6s of handoff in a 6s interval: 43%.
    _handoff(runtime, 4, 10.0, 16.0, {"drain": 3.0, "collect": 1.6, "store": 1.0})
    _handoff(runtime, 6, 16.0, 22.0, {"drain": 3.0, "collect": 1.6, "store": 1.0})

    assert len(warnings) == 1, "the observation is made once, not per checkpoint"
    message = warnings[0].getMessage()
    assert "checkpoint_every=2" in message
    assert "43%" in message


def test_drain_does_not_count_against_the_cadence():
    """`drain` is the training loop's own queued work, not a checkpoint cost.

    Counting it would make a cheap writer look expensive: at a wide cadence the
    drain is most of the handoff precisely because there is more training
    behind it.
    """
    runtime, warnings = _runtime(every=20)
    _handoff(runtime, 20, 0.0, 14.0, {"drain": 11.1, "collect": 1.6, "store": 1.0})
    # 2.6s of real cost in a 68s interval is 4%, even though the handoff
    # itself was 13.7s.
    _handoff(runtime, 40, 14.0, 82.0, {"drain": 11.1, "collect": 1.6, "store": 1.0})
    assert warnings == []
