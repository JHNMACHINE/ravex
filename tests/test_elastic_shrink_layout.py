"""Which store layout a run uses when the world shrank under it — GPU-94.

One method decides both where a checkpoint is written and where a resume looks
for one, and for a while it answered the second question wrongly for exactly
one world size. A job that shrank to a single rank came back, found ``per_rank``
switched off by the world size alone, looked in ``checkpoints/`` instead of
``checkpoints/rank_<n>/``, and logged *No checkpoint found — starting from
scratch* with the shards sitting beside it. Everything about it looked healthy:
training resumed, the loss curve started over, and nothing said why.

The reshard planner had handled ``N -> 1`` the whole time — ``test_reshard.py``
calls it "the shrink taken to its limit" — but every test of it passes
``per_rank=True`` by hand, so the machinery was proven and the decision to use
it was not. These tests cover the decision.

Confirmed end to end in ``integration/elastic/probe.sh`` under Docker, where a
real ``torchrun`` elastic job loses an agent: at ``AGENTS=3`` the survivors
resumed even before the fix, at ``AGENTS=2`` they did not. That probe needs a
container and three processes; this file needs neither, which is why the
regression lives here.
"""

import os

import pytest

from ravex._config import RavexConfig
from ravex._runtime import RavexRuntime


class FakeRegistry:
    """Answers the one structural question the runtime asks it."""

    def __init__(self, layout: str = "per_rank") -> None:
        self._layout = layout

    def sharded_layout(self, requested: str) -> str:
        return self._layout if requested == "per_rank" else "gather"


def runtime_for(config, layout: str = "per_rank") -> RavexRuntime:
    runtime = RavexRuntime.__new__(RavexRuntime)
    runtime.config = config
    runtime.registry = FakeRegistry(layout)
    return runtime


@pytest.fixture
def config(tmp_path):
    resolved = RavexConfig()
    resolved.sharded_checkpoints = "per_rank"
    resolved.storage.type = "local"
    resolved.storage.path = str(tmp_path / "checkpoints")
    return resolved


def make_rank_stores(config, ranks) -> None:
    """The directories a wider run left behind, as `visible_rank_stores` reads them.

    Each one gets a file. An *empty* ``rank_<n>/`` is deliberately not a store
    as far as ``visible_rank_stores`` is concerned — a directory that exists
    and holds nothing is a half-made thing, not a checkpoint — so a fixture
    that only calls ``makedirs`` writes a test that fails for its own reasons.
    ``.ravex-owner`` is what a real run leaves there first.
    """
    for rank in ranks:
        store = os.path.join(config.storage.path, "rank_%d" % rank)
        os.makedirs(store, exist_ok=True)
        with open(os.path.join(store, ".ravex-owner"), "w", encoding="utf-8") as f:
            f.write('{"world_size": %d}' % len(list(ranks)))


@pytest.fixture
def world(monkeypatch):
    """Set the world size the runtime believes it is in."""

    def set_to(size: int) -> None:
        monkeypatch.setattr("ravex._runtime.get_world_size", lambda: size)

    return set_to


# ── the bug, stated as a test ───────────────────────────────────────────────


def test_a_lone_rank_continuing_a_wider_run_still_reads_per_rank(config, world):
    """The regression. Two ranks wrote, one came back: the shards are still theirs.

    This is what an elastic job looks like after losing every node but one,
    and answering False here is what threw a resume away.
    """
    make_rank_stores(config, [0, 1])
    world(1)
    assert runtime_for(config)._per_rank_active() is True


def test_a_lone_rank_with_no_history_writes_an_ordinary_store(config, world):
    """An ordinary single-rank run, unchanged.

    Nothing on disk says a wider run came before, so there is nothing to
    continue and no reason to invent a ``rank_0/`` directory for one rank.
    """
    world(1)
    assert runtime_for(config)._per_rank_active() is False


def test_more_than_one_rank_needs_nothing_on_disk(config, world):
    """The ordinary case does not consult the filesystem at all.

    A first run has written nothing yet and must still choose the per-rank
    layout, or rank 3 would write where the resume will not look — the same
    mismatch as the bug above, arrived at from the other side.
    """
    world(4)
    assert runtime_for(config)._per_rank_active() is True


# ── the answers that must not change ────────────────────────────────────────


def test_the_config_still_decides_first(config, world):
    config.sharded_checkpoints = "gather"
    make_rank_stores(config, [0, 1])
    world(1)
    assert runtime_for(config)._per_rank_active() is False


def test_a_model_whose_shards_are_not_dtensors_still_gathers(config, world):
    """``per_rank`` downgrades on structure, and stores on disk do not override that.

    If the live model cannot be written per rank, finding old per-rank stores
    is not a reason to try: the run would open a store it then never writes,
    which is the empty-store failure the layout rule exists to prevent.
    """
    make_rank_stores(config, [0, 1])
    world(1)
    assert runtime_for(config, layout="gather")._per_rank_active() is False

    world(4)
    assert runtime_for(config, layout="gather")._per_rank_active() is False


def test_remote_storage_is_not_asked_the_question(config, world, monkeypatch):
    """Every node sees the same bucket, so "what is here" carries no information.

    ``visible_rank_stores`` returns nothing for remote storage deliberately,
    and resharding refuses remote storage one layer down in any case — so a
    lone rank on a bucket takes the ordinary path rather than a per-rank one
    that nothing downstream would complete.
    """
    config.storage.type = "s3"
    config.storage.bucket = "somewhere"
    world(1)
    assert runtime_for(config)._per_rank_active() is False
