"""Per-rank stores across machines that share only a network.

With ``sharded_checkpoints: per_rank`` and local storage, each node holds the
stores for the ranks it happened to host. That is fine until the nodes come
back in a different order: rank 2 lands on the machine holding ``rank_0`` and
``rank_1``, finds nothing of its own, and the run starts over — exit code
zero, hours of checkpoints intact one machine away.

Reproduced on 2026-08-19 with two containers on a Docker network, one volume
each. These tests are the single-process version of what that showed, and they
exist so the *explanation* cannot quietly go back to being wrong: the message
used to blame a change in world size, in a run whose world size never changed.
"""

import logging

import pytest

from ravex._backends import visible_rank_stores
from ravex._config import RavexConfig
from ravex._resume import ResumeManager
from ravex._runtime import RavexRuntime


def store_for(base, rank, files=("step_000000000004.pt",)):
    directory = base / ("rank_%d" % rank)
    directory.mkdir(parents=True)
    for name in files:
        (directory / name).write_text("x", encoding="utf-8")
    return directory


def local_config(path):
    config = RavexConfig()
    config.storage.type = "local"
    config.storage.path = str(path)
    return config


class TestWhatThisMachineCanReach:
    def test_it_finds_the_rank_stores_that_are_here(self, tmp_path):
        store_for(tmp_path, 0)
        store_for(tmp_path, 1)

        assert visible_rank_stores(local_config(tmp_path)) == {0, 1}

    def test_an_empty_store_directory_does_not_count(self, tmp_path):
        """A directory a rank created and never wrote into is not a checkpoint.

        It is the ordinary state between ``makedirs`` and the first save, and
        counting it would report a reachable checkpoint that is not there.
        """
        store_for(tmp_path, 0)
        (tmp_path / "rank_1").mkdir()

        assert visible_rank_stores(local_config(tmp_path)) == {0}

    def test_anything_not_named_like_a_rank_is_ignored(self, tmp_path):
        store_for(tmp_path, 3)
        (tmp_path / "logs").mkdir()
        (tmp_path / "rank_").mkdir()
        (tmp_path / "rank_x").mkdir()

        assert visible_rank_stores(local_config(tmp_path)) == {3}

    def test_a_missing_directory_is_the_first_run_not_a_failure(self, tmp_path):
        assert visible_rank_stores(local_config(tmp_path / "nothing-here")) == set()

    def test_remote_storage_reports_nothing_on_purpose(self, tmp_path):
        """Every node sees the same bucket, so "what I can reach" says nothing.

        Returning the contents there would invite the caller to compare two
        identical sets and conclude something from it.
        """
        store_for(tmp_path, 0)
        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"

        assert visible_rank_stores(config) == set()


class Captured(logging.Handler):
    """Records straight off the ``ravex`` logger.

    Not ``caplog``: the runtime sets ``propagate = False`` so the user's root
    logger stays untouched, and once any test has activated Ravex the records
    never reach the root handler caplog installs. Attaching here is immune to
    that, and to the order tests happen to run in.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.text = ""

    def emit(self, record):
        self.text += record.getMessage() + "\n"


class TestWhyThereIsNothingToResume:
    """The three situations that share one exit, and must not share one story."""

    def explain(self, config):
        manager = ResumeManager(backend=None, registry=None, config=config)
        logger = logging.getLogger("ravex")
        captured = Captured()
        logger.addHandler(captured)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            manager._explain_nothing_to_resume(None)
        finally:
            logger.removeHandler(captured)
            logger.setLevel(previous)
        return captured.text

    def test_a_first_run_says_so_plainly(self, tmp_path, monkeypatch):
        _install(monkeypatch, seen=[set()], world=1)
        text = self.explain(local_config(tmp_path))

        assert "No checkpoint found on any node" in text
        assert "cannot reach" not in text

    def test_a_store_on_another_machine_is_named_as_that(
        self, tmp_path, monkeypatch
    ):
        """The case the Docker run reproduced.

        Four ranks, world size unchanged, every store present somewhere and
        none of them where its rank is now running.
        """
        _install(monkeypatch, seen=[{2, 3}, {2, 3}, {0, 1}, {0, 1}], world=4)
        text = self.explain(local_config(tmp_path))

        assert "this topology cannot reach it" in text
        assert "rank(s) 0, 1, 2, 3" in text
        assert "Nothing was lost" in text
        # The old message blamed this, in a run whose world size never moved.
        assert "world size" not in text

    def test_a_hole_in_the_middle_is_a_machine_that_is_gone(
        self, tmp_path, monkeypatch
    ):
        """Four ranks, stores for 0, 1 and 3, nothing for 2.

        Reproduced on 2026-08-19 by SIGKILLing one of four containers mid-run
        and bringing it back with an empty disk. Stores are written by a
        contiguous range from 0, so rank 3 being present while rank 2 is not
        cannot be a run that had fewer ranks — it is a store that existed and
        went away with its machine.

        Worth separating because the advice differs: a smaller previous run is
        nothing to act on, a lost machine says the checkpoint needed a second
        copy somewhere.
        """
        _install(monkeypatch, seen=[{0, 1, 3}] * 4, world=4)
        text = self.explain(local_config(tmp_path))

        assert "part of it is gone" in text
        assert "rank(s) 2" in text
        assert "not a smaller run" in text
        assert "cannot reach" not in text

    def test_missing_ranks_at_the_top_are_reported_as_ambiguous(
        self, tmp_path, monkeypatch
    ):
        """Four ranks, stores only for 0 and 1, and no way to tell why.

        A two-rank run writes exactly this, and so does a four-rank run that
        lost the machine holding the top half. Nothing on disk separates them,
        so the message must not pick one — which is the mistake the original
        wording made, and the one this whole file exists to prevent.
        """
        _install(monkeypatch, seen=[{0, 1}] * 4, world=4)
        text = self.explain(local_config(tmp_path))

        assert "rank(s) 2, 3" in text
        assert "the two look the same" in text
        assert "part of it is gone" not in text

    def test_stores_past_the_end_of_this_run_are_reported(
        self, tmp_path, monkeypatch
    ):
        """Stores up to rank 5 while this run has 3, and rank 2 has nothing.

        The leftover high ranks are evidence the reader wants, so the message
        carries them instead of leaving them to be discovered by hand.
        """
        _install(monkeypatch, seen=[{0, 1, 5}] * 3, world=3)
        text = self.explain(local_config(tmp_path))

        assert "part of it is gone" in text
        assert "rank(s) 2" in text

    def test_the_diagnosis_never_costs_the_run(self, tmp_path, monkeypatch):
        """A scan that raises must not turn a fresh start into a crash.

        This runs at startup on every rank; an exception here would take down
        a training job to explain why it had nothing to restore.
        """

        def explode(_config):
            raise OSError("the filesystem is having a day")

        monkeypatch.setattr("ravex._backends.visible_rank_stores", explode)
        _install(monkeypatch, seen=[set()], world=1)

        text = self.explain(local_config(tmp_path))
        assert "starting from scratch" in text


class TestTheWarningBeforeTheFirstCheckpoint:
    """Said at activation, where it can still prevent something.

    At the resume that fails, hours of checkpoints have already gone to the
    wrong machines. See ``_warn_if_split_across_machines``.
    """

    def warn(self, monkeypatch, config, world, local, local_rank=0):
        monkeypatch.setenv("WORLD_SIZE", str(world))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local))
        monkeypatch.setenv("LOCAL_RANK", str(local_rank))
        monkeypatch.setattr("ravex._distributed._dist", lambda: None)

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = config

        logger = logging.getLogger("ravex")
        captured = Captured()
        logger.addHandler(captured)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            runtime._warn_if_split_across_machines()
        finally:
            logger.removeHandler(captured)
            logger.setLevel(previous)
        return captured.text

    def test_local_storage_across_machines_is_warned_about(
        self, tmp_path, monkeypatch
    ):
        text = self.warn(monkeypatch, local_config(tmp_path), world=32, local=8)

        assert "spans 4 machines" in text
        assert "not at all if one machine is lost" in text

    def test_one_machine_is_not_warned_about(self, tmp_path, monkeypatch):
        """Eight ranks on one box is the ordinary case and needs no noise."""
        text = self.warn(monkeypatch, local_config(tmp_path), world=8, local=8)

        assert text == ""

    def test_remote_storage_is_not_warned_about(self, tmp_path, monkeypatch):
        """The case that works. Every node writes to the same bucket."""
        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"

        assert self.warn(monkeypatch, config, world=32, local=8) == ""

    def test_it_is_said_once_per_machine_not_once_per_rank(
        self, tmp_path, monkeypatch
    ):
        """It is a statement about a filesystem, not about a process.

        Eight ranks per node repeating it would put it on screen 32 times.
        """
        text = self.warn(monkeypatch, local_config(tmp_path), world=32, local=8,
                         local_rank=3)

        assert text == ""


def _install(monkeypatch, seen, world):
    """Stand in for the two collectives, which need a process group."""
    monkeypatch.setattr(
        "ravex._distributed.gather_visible_stores", lambda mine: [set(s) for s in seen]
    )
    monkeypatch.setattr("ravex._distributed.get_world_size", lambda: world)


@pytest.fixture(autouse=True)
def _no_process_group(monkeypatch):
    """These tests describe multi-node situations from a single process.

    The collectives are replaced per test; this only guarantees that a stray
    real one cannot be reached and block.
    """
    monkeypatch.setattr("ravex._distributed._dist", lambda: None)
