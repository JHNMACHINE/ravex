"""Resuming from per-rank stores.

The multi-rank behaviour is covered for real in ``integration/test_fsdp.py``,
under torchrun with two processes. What is here is the part that decides *which*
step to read — the one place where a wrong answer produces a model that trains
happily and means nothing.
"""

import ravex._distributed as distributed
from ravex._resume import ResumeManager


class FakeBackend:
    """A store holding whichever steps a test says this rank managed to write."""

    def __init__(self, steps):
        self.steps = sorted(steps)
        self.loaded = []

    def has_checkpoint(self):
        return bool(self.steps)

    def latest_step(self):
        return self.steps[-1] if self.steps else None

    def load_step(self, step):
        self.loaded.append(step)
        if step not in self.steps:
            return None
        return {"ravex_version": 1, "step": step, "models": {}, "sharded": {}}

    def load_latest(self):
        return self.load_step(self.steps[-1]) if self.steps else None


class FakeRegistry:
    def __init__(self):
        self.step_count = 0
        self.restored = None

    def restore_state(self, state, defer_rng=False):
        self.restored = state
        self.step_count = state["step"]


def test_a_rank_ahead_of_the_others_rewinds_to_the_agreed_step(monkeypatch):
    """This rank wrote step 12; some other rank was killed before it did.

    Reading 12 here and 8 elsewhere assembles one model out of two moments in
    training: every shard valid on its own, the whole thing wrong, and nothing
    downstream in a position to notice.
    """
    monkeypatch.setattr(distributed, "agree_on_step", lambda local: 8)
    monkeypatch.setattr(distributed, "all_ranks_agree", lambda ok: bool(ok))

    backend = FakeBackend([4, 8, 12])
    registry = FakeRegistry()

    assert ResumeManager(backend, registry).try_resume(per_rank=True) is True
    assert backend.loaded == [8], "the newest step this rank had was not the answer"
    assert registry.step_count == 8


def test_nothing_on_one_rank_means_nothing_anywhere(monkeypatch):
    """-1 travels through the agreement like any other value, so a rank with an
    empty store makes every rank start clean instead of leaving it behind."""
    monkeypatch.setattr(distributed, "agree_on_step", lambda local: -1)

    backend = FakeBackend([4, 8])
    registry = FakeRegistry()

    assert ResumeManager(backend, registry).try_resume(per_rank=True) is False
    assert backend.loaded == [], "a rank read a checkpoint the others do not have"
    assert registry.restored is None


def test_a_step_that_will_not_read_stops_every_rank(monkeypatch):
    """A resume half the ranks complete is worse than none: the ones that
    restored go on to collectives the others never join."""
    monkeypatch.setattr(distributed, "agree_on_step", lambda local: 8)
    monkeypatch.setattr(distributed, "all_ranks_agree", lambda ok: False)

    backend = FakeBackend([4, 8])
    registry = FakeRegistry()

    assert ResumeManager(backend, registry).try_resume(per_rank=True) is False
    assert registry.restored is None


def test_the_shared_store_path_is_untouched():
    """Without per_rank nothing agrees on anything — one store, one answer."""
    backend = FakeBackend([4, 8])
    registry = FakeRegistry()

    assert ResumeManager(backend, registry).try_resume() is True
    assert registry.step_count == 8


def _config_at(path):
    from ravex._config import RavexConfig

    config = RavexConfig()
    config.storage.type = "local"
    config.storage.path = str(path)
    return config


def test_a_copy_cut_off_part_way_is_named_rather_than_called_absent(tmp_path):
    """The third possibility, which the two in the message used to swallow.

    A transfer interrupted leaves the directory here and disqualified: bytes on
    disk, no completeness marker. Saying "no store anywhere" then accuses a
    machine that is powered on and holding the data. Found on 2026-08-21 by
    killing both ranks mid-copy on a 100 Mbps link, where that window is 42% of
    the cycle rather than a millisecond.
    """
    from ravex._replication import COMPLETE_MARKER, REPLICA_DIR
    from ravex._resume import _torn_copy_here

    config = _config_at(tmp_path)
    copy = tmp_path / REPLICA_DIR / "rank_1"
    copy.mkdir(parents=True)
    (copy / "manifest.json").write_text("{}", encoding="utf-8")

    said = _torn_copy_here(config, [1])
    assert "rank(s) 1" in said
    assert "interrupted" in said

    (copy / COMPLETE_MARKER).write_text("ok", encoding="utf-8")
    assert _torn_copy_here(config, [1]) == "", "a whole copy is not this case"


def test_a_rank_with_no_copy_here_adds_nothing(tmp_path):
    """Silence when this machine has nothing to add: the other ranks say what
    they see, and a machine inventing a third case it cannot see would be the
    same failure in the other direction."""
    from ravex._resume import _torn_copy_here

    assert _torn_copy_here(_config_at(tmp_path), [1]) == ""
