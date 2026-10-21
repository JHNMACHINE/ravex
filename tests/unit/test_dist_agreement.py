"""GPU-111: agreeing about one bit without a collective.

``all_ranks_agree`` keeps a resume all-or-nothing, and until now it did that by
pickling a Python bool through two collectives — the most expensive call in this
family, measured at 1025 µs against 41–56 µs for a store lookup, and 5.7 ms per
rank once one rank is 5 ms late (``bench/agreement_cost.py``).

The point of these tests is not the speed. It is that the answer is the *same
answer*, and that the one place the two roads differ is a difference in the
right direction: a rank that goes silent produces ``False`` here, where the
collective produced a group-wide timeout. "Not every rank succeeded" is exactly
what the caller asked, and it is better delivered as an answer than as an
outage.

Built against a store made of a dictionary, because what is under test is the
protocol — the rounds, the deadline, the reduction — and not torch's key-value
server. ``test_both_roads_give_the_same_answer`` is the one that runs two real
processes with a real ``TCPStore`` behind it.
"""

import threading
import time

import pytest

from ravex._dist import agreement
from ravex._dist.agreement import all_gather_scalar


@pytest.fixture(autouse=True)
def fresh_rounds():
    """Round numbers are per process, and a test is not a process."""
    agreement.reset_rounds()
    yield
    agreement.reset_rounds()


class FakeStore:
    """The four methods this module asks of a rendezvous store."""

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()
        self.deleted = []

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def check(self, keys):
        with self._lock:
            return all(key in self._data for key in keys)

    def get(self, key):
        with self._lock:
            return self._data[key]

    def delete_key(self, key):
        with self._lock:
            self.deleted.append(key)
            self._data.pop(key, None)


def gather_from(store, votes, world_size=None, patience=None, name="q"):
    """Every rank votes at once, from its own thread. Returns the answers."""
    world_size = world_size if world_size is not None else len(votes)
    answers = {}

    def vote(rank):
        answers[rank] = all_gather_scalar(
            name, votes[rank], rank, world_size, store, patience=patience
        )

    threads = [threading.Thread(target=vote, args=(rank,)) for rank in votes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return answers


def test_every_rank_sees_every_answer():
    store = FakeStore()
    answers = gather_from(store, {0: True, 1: False, 2: True})

    assert answers[0] == [True, False, True]
    # Same list on every rank, in rank order, which is what makes a local
    # `all()` a decision the whole job agrees on.
    assert answers[0] == answers[1] == answers[2]


def test_a_silent_rank_is_an_answer_and_not_a_wait():
    """The one place the two roads differ, and the reason to prefer this one.

    A collective waits for the group's timeout and then fails everybody. Here
    rank 2 never votes, the question has a deadline, and running out of it says
    "no" — which is what "did every rank succeed" was asking.
    """
    store = FakeStore()
    started = time.monotonic()
    answers = gather_from(store, {0: True, 1: True}, world_size=3, patience=0.5)
    elapsed = time.monotonic() - started

    assert answers[0] is None and answers[1] is None
    assert elapsed < 10, "it waited far past its own deadline"


def test_the_missing_rank_is_named(caplog):
    """A collective's timeout says the group failed. This says who."""
    import logging

    store = FakeStore()
    with caplog.at_level(logging.WARNING, logger="ravex"):
        gather_from(store, {0: True}, world_size=2, patience=0.2)

    assert "rank(s) 1" in caplog.text


def test_two_questions_in_a_row_do_not_read_each_other():
    """Rounds, which is the whole reason the key carries a number.

    Ranks reach the same call sites in the same order — the invariant a
    collective already demands — so a per-process counter is the same counter
    on every rank. Without it the second question would read the first's votes
    and answer the wrong thing, silently.
    """
    store = FakeStore()
    first = gather_from(store, {0: True, 1: True})
    second = gather_from(store, {0: True, 1: False})

    assert first[0] == [True, True]
    assert second[0] == [True, False]


def test_old_rounds_are_dropped():
    """Small keys, but one per rank per round, on a store the job shares."""
    store = FakeStore()
    for _ in range(4):
        gather_from(store, {0: True, 1: True})

    assert store.deleted, "nothing was ever cleaned up"
    # Only from rank 0, and only rounds old enough that nobody is still reading
    # them: the two most recent survive.
    assert all("/agree/q/" in key for key in store.deleted)
    assert not any(key.endswith("/3/0") or key.endswith("/3/1") for key in store.deleted)


def test_only_scalars_travel():
    """A structure needs an encoding decided on purpose, not defaulted into."""
    store = FakeStore()
    with pytest.raises(TypeError, match="scalars only"):
        all_gather_scalar("q", {"step": 3}, 0, 2, store)


def test_one_rank_asks_nobody():
    store = FakeStore()
    assert all_gather_scalar("q", True, 0, 1, store) == [True]


# ─── agree_on_step, the other gather of the family ─────────────────


def test_the_oldest_of_the_newest_steps_comes_back():
    """The reduction is ``min``, and it is done on the same list everywhere."""
    from ravex._dist import collectives

    store = FakeStore()
    answers = gather_from(store, {0: 8, 1: 6, 2: 11}, name="agree_on_step")

    assert answers[0] == [8, 6, 11]
    assert min(answers[0]) == 6 == collectives.NOTHING_TO_RESUME + 7


def test_a_silent_rank_means_nothing_to_resume(monkeypatch, caplog):
    """Not "resume from the minimum of whoever replied".

    A rank left out of that minimum restores a different moment in training
    than the others, which is the one outcome ``agree_on_step`` exists to
    prevent. So silence collapses the whole answer to
    ``NOTHING_TO_RESUME`` - the same vote a rank with an empty store casts -
    and every rank starts from scratch together.
    """
    from ravex._dist import collectives

    store = FakeStore()
    monkeypatch.setattr(agreement, "default_patience", lambda: 0.2)
    monkeypatch.setattr(agreement, "rendezvous_store", lambda: store)
    monkeypatch.setattr(agreement, "wanted_transport", lambda: "store")
    monkeypatch.setattr(collectives, "get_rank", lambda: 0)
    monkeypatch.setattr(collectives, "get_world_size", lambda: 2)

    # Rank 1 never writes its key.
    assert collectives._newest_step_over_store(8) == collectives.NOTHING_TO_RESUME
    assert "rank(s) 1" in caplog.text


def test_the_store_road_is_skipped_when_the_config_says_collectives(monkeypatch):
    """``None`` from the private half means "not asked", never "nobody said"."""
    from ravex._dist import collectives

    monkeypatch.setattr(agreement, "wanted_transport", lambda: "collectives")
    assert collectives._newest_step_over_store(8) is None


# ─── both roads, two real processes, a real store ───────────────────────────
