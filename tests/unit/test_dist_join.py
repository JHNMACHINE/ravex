"""GPU-121: a node that joins a run already in progress.

``test_dist_round.py`` proves the other half of GPU-113's point 4 — a node dies
and the round closes over whoever is left. This is the arrival, and the two are
not symmetric. A departure is safe to notice late: the node that went away is
missing from *everybody's* list of reports at once, because there is nothing to
miss. An arrival noticed by one node and not another is two nodes applying
different averages to the same parameters, for ever, with the loss falling on
both and nothing raising — which is why the interesting assertions here are not
about the newcomer training well, they are about all three holding **one
model**, checked with ``torch.equal`` and not ``allclose``.

Threads, one per node, for the same reason ``test_dist_round.py`` gives: a node
blocks inside ``close_round`` until its peers publish, so driving them in
sequence would be testing a shape the real thing never has.
"""

import time

import pytest

torch = pytest.importorskip("torch")


from ravex._dist import membership as _membership  # noqa: E402

from test_dist_exchange import FakeStore  # noqa: E402

BASE_WORLD = 2


def test_a_run_every_member_has_left_is_not_waited_for():
    """A node rejoining a run whose nodes have all finished gives up at once.

    On rented machines it waited out its whole deadline - fifteen minutes,
    billed - for members that had ended and would never take it in (GPU-186).
    """
    store = FakeStore()
    for member in range(BASE_WORLD):
        _membership.leave(store, member, 7)
    began = time.monotonic()
    assert not _membership.join(store, None, None, BASE_WORLD, BASE_WORLD, time.monotonic() + 60)
    assert time.monotonic() - began < 5


def test_one_member_still_there_is_waited_for():
    store = FakeStore()
    for member in range(1, BASE_WORLD):
        _membership.leave(store, member, 7)
    began = time.monotonic()
    # Nobody serves the run's state, so it waits to its (short) deadline.
    assert not _membership.join(store, None, None, BASE_WORLD, BASE_WORLD, time.monotonic() + 2)
    assert time.monotonic() - began >= 2


def test_a_run_that_ended_while_this_node_was_out_finishes_the_training(tmp_path, monkeypatch):
    """`RunOver` ends the training as finished, not as a failure (GPU-186)."""
    import ravex

    monkeypatch.chdir(tmp_path)
    steps = []

    @ravex.train_loop(storage={"type": "local", "path": str(tmp_path / "store")}, checkpoint_every=10 ** 9)
    def train():
        steps.append(1)
        raise _membership.RunOver("the run ended while node 2 was out of it; every member has finished")

    assert train() is None
    assert steps == [1]


def test_a_script_catching_exceptions_does_not_swallow_run_over():
    """A BaseException, like KeyboardInterrupt: a script's own `except
    Exception` around its step must not catch it and train on alone."""
    assert not issubclass(_membership.RunOver, Exception)
