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
import os

import pytest

from ravex._backends import visible_rank_stores
from ravex._dist.collectives import agree_on_run_id, storage_is_shared
from ravex._dist.identity import (
    FROM_CONFIG,
    FROM_SCHEDULER,
    FROM_STORE,
    GENERATED,
    local_run_id,
    read_owner,
    scheduler_run_id,
    write_owner,
)
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

    def test_it_names_the_machine_that_wrote_the_stranded_stores(
        self, tmp_path, monkeypatch
    ):
        """Which box to put back, not merely that the placement moved."""
        _install(
            monkeypatch,
            seen=[
                {2: {"host": "node1"}, 3: {"host": "node1"}},
                {2: {"host": "node1"}, 3: {"host": "node1"}},
                {0: {"host": "node0"}, 1: {"host": "node0"}},
                {0: {"host": "node0"}, 1: {"host": "node0"}},
            ],
            world=4,
        )
        text = self.explain(local_config(tmp_path))

        assert "this topology cannot reach it" in text
        assert "written on node0, node1" in text

    def test_stores_that_never_recorded_an_owner_still_get_a_sentence(
        self, tmp_path, monkeypatch
    ):
        """A checkpoint from before owner records degrades, it does not break."""
        _install(monkeypatch, seen=[{2: {}, 3: {}}, {2: {}, 3: {}},
                                    {0: {}, 1: {}}, {0: {}, 1: {}}], world=4)
        text = self.explain(local_config(tmp_path))

        assert "this topology cannot reach it" in text
        assert "written on" not in text

    def test_two_runs_on_one_disk_are_called_out(self, tmp_path, monkeypatch):
        """The phase-4 situation: an accidental history beside the real one.

        A restart that began from scratch wrote into the same directory names,
        and from then on nothing on disk said the two were unrelated. The ids
        are what make it sayable.
        """
        _install(
            monkeypatch,
            seen=[
                {2: {"host": "n1", "run_id": "run-b"}},
                {2: {"host": "n1", "run_id": "run-b"}},
                {0: {"host": "n0", "run_id": "run-a"}},
                {0: {"host": "n0", "run_id": "run-a"}},
            ],
            world=4,
        )
        text = self.explain(local_config(tmp_path))

        assert "2 different runs" in text
        assert "run-a" in text and "run-b" in text

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


class TestTheSharedStorageProbe:
    """Asked of the filesystem, because the config cannot answer it.

    The collective half — every rank dropping a marker and looking for the
    others' — is covered end-to-end in Docker: two containers on separate
    volumes answer False, the same two on one volume answer True and then
    resume across a swapped placement. What is here is the rest.
    """

    def test_a_lone_process_sees_what_it_wrote(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)

        assert storage_is_shared(str(tmp_path / "checkpoints")) is True

    def test_it_leaves_no_marker_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)
        path = tmp_path / "checkpoints"

        storage_is_shared(str(path))

        assert list(path.iterdir()) == [], "the probe left its marker on disk"

    def test_a_directory_it_cannot_write_answers_no(self, tmp_path, monkeypatch):
        """False is the safe reading: it makes the caller assume the split.

        An unwritable checkpoint directory is a larger problem, and the backend
        reports it. This must not raise on the way to finding that out.
        """
        monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)

        def refuse(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr("os.makedirs", refuse)

        assert storage_is_shared(str(tmp_path / "nope")) is False


class TestWhatItSaysBeforeTheFirstCheckpoint:
    """Three situations, and the difference between them is asked, not assumed.

    A shared filesystem and a local disk look identical in the config: both are
    ``storage.type: local`` pointing at a directory that exists. The first
    version of this warning fired on both, which would have cried wolf at every
    cluster with an NFS mount.
    """

    def announce(
        self,
        monkeypatch,
        config,
        world,
        local,
        local_rank=0,
        shared=False,
        transport=(True, None),
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local))
        monkeypatch.setenv("LOCAL_RANK", str(local_rank))
        monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)

        probed = []

        def probe(path):
            probed.append(path)
            return shared

        monkeypatch.setattr("ravex._dist.collectives.storage_is_shared", probe)
        # These tests have no process group at all, so the real probe would
        # answer "no transport" and the announcement would be about that
        # instead of about the topology, which is what they are here for.
        monkeypatch.setattr(
            "ravex._dist.collectives.byte_transport_group", lambda: transport
        )

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = config
        runtime._storage_announced = False

        logger = logging.getLogger("ravex")
        captured = Captured()
        logger.addHandler(captured)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            runtime._announce_storage_topology()
        finally:
            logger.removeHandler(captured)
            logger.setLevel(previous)
        return captured.text, probed

    def test_split_storage_with_copies_running_is_not_alarming(
        self, tmp_path, monkeypatch
    ):
        """Split disks stop being a warning once the copies are going.

        Losing a machine costs an interval, not the run — proven on the bench
        on 2026-08-19, where a replaced node took its store back from a peer
        and all four ranks resumed. Saying "not at all if one machine is lost"
        after that would be false.
        """
        config = local_config(tmp_path)
        config.replicate_every = 10
        text, probed = self.announce(monkeypatch, config, world=32, local=8)

        assert "spans 4 machines" in text
        assert "every 10 checkpoints" in text
        assert "at most that much progress" in text
        assert "not at all if one machine is lost" not in text
        assert probed, "it spoke without checking the filesystem"

    def test_split_storage_with_copies_off_is_warned_about(
        self, tmp_path, monkeypatch
    ):
        config = local_config(tmp_path)
        config.replicate_every = 0
        text, _ = self.announce(monkeypatch, config, world=32, local=8)

        assert "not at all if one machine is lost" in text
        assert "replicate_every=0" in text

    def test_a_backend_that_cannot_carry_bytes_is_said_out_loud(
        self, tmp_path, monkeypatch
    ):
        """GPU-82, and the reason it went unnoticed for two days.

        Replication moves a store as bytes, and bytes live on the host. NCCL is
        a GPU collective library and refuses them: measured on 8x RTX 5060 Ti
        on 2026-08-21, every round failed with ``No backend type associated
        with device type cpu`` and **not one copy was made** — while this
        announcement went on promising that losing a machine cost an interval.

        A gloo subgroup fixes it, and where one cannot be opened the honest
        answer is to say replication is off **here**, once, rather than let
        every round fail a warning at a time into a log nobody reads.
        """
        config = local_config(tmp_path)
        config.replicate_every = 10
        text, _ = self.announce(
            monkeypatch, config, world=32, local=8, transport=(False, None)
        )

        assert "no way to move bytes between the ranks" in text
        assert "not at all if one machine is lost" in text
        assert "at most that much progress" not in text, (
            "it promised protection it does not have"
        )

    def test_copies_asked_for_but_impossible_says_which(self, tmp_path, monkeypatch):
        """Wanting copies and not being able to place them is the silent case.

        With ranks spread unevenly no peer can be shown to be on a different
        machine, so nothing is copied. A run that asked for protection and did
        not get it must not read the same as one that never asked.
        """
        config = local_config(tmp_path)
        config.replicate_every = 10
        text, _ = self.announce(monkeypatch, config, world=12, local=8)

        assert "not spread evenly" in text
        assert "not at all if one machine is lost" in text

    def test_a_shared_filesystem_is_not_warned_about(self, tmp_path, monkeypatch):
        """The requirement: if the condition is there, use it and say nothing alarming."""
        text, _ = self.announce(
            monkeypatch, local_config(tmp_path), world=32, local=8, shared=True
        )

        assert "shared across all 4 machines" in text
        assert "lost" not in text

    def test_one_machine_is_not_probed_or_warned_about(self, tmp_path, monkeypatch):
        """Eight ranks on one box is the ordinary case and needs no noise."""
        text, probed = self.announce(
            monkeypatch, local_config(tmp_path), world=8, local=8
        )

        assert text == ""
        assert probed == [], "a single-machine job paid for a collective probe"

    def test_remote_storage_is_not_probed_or_warned_about(self, tmp_path, monkeypatch):
        """The case that already works. Every node writes to the same bucket."""
        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"

        text, probed = self.announce(monkeypatch, config, world=32, local=8)

        assert text == ""
        assert probed == []

    def test_every_rank_probes_but_only_one_per_machine_speaks(
        self, tmp_path, monkeypatch
    ):
        """The probe is collective; the message is about a filesystem.

        Skipping the probe on non-speaking ranks would leave the speaker alone
        inside a collective, which is the hang this codebase has already paid
        for once.
        """
        text, probed = self.announce(
            monkeypatch, local_config(tmp_path), world=32, local=8, local_rank=3
        )

        assert text == ""
        assert probed, "a silent rank skipped the collective the speaker entered"

    def test_it_is_said_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "32")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)
        monkeypatch.setattr("ravex._dist.collectives.storage_is_shared", lambda p: False)

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = local_config(tmp_path)
        runtime._storage_announced = False

        logger = logging.getLogger("ravex")
        captured = Captured()
        logger.addHandler(captured)
        logger.setLevel(logging.INFO)
        try:
            runtime._announce_storage_topology()
            runtime._announce_storage_topology()
        finally:
            logger.removeHandler(captured)

        assert captured.text.count("spans 4 machines") == 1


def _install(monkeypatch, seen, world):
    """Stand in for the two collectives, which need a process group.

    ``seen`` is passed through untouched: entries may be plain rank sets, or
    the rank-to-owner-record mappings the real discovery returns, and coercing
    one into the other here would quietly drop the records the messages read.
    """
    monkeypatch.setattr(
        "ravex._dist.collectives.gather_visible_stores", lambda mine: list(seen)
    )
    monkeypatch.setattr("ravex._dist.collectives.get_world_size", lambda: world)


@pytest.fixture(autouse=True)
def _no_process_group(monkeypatch):
    """These tests describe multi-node situations from a single process.

    The collectives are replaced per test; this only guarantees that a stray
    real one cannot be reached and block.
    """
    monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)


# ─── the old-torch from_local fallback ──────────────────────────────


class _Mesh:
    """Just the `.size(mesh_dim)` that the shape arithmetic asks for."""

    def __init__(self, *sizes):
        self._sizes = sizes

    def size(self, mesh_dim):
        return self._sizes[mesh_dim]


class _Shard:
    def __init__(self, dim):
        self.dim = dim


class _Replicate:
    """No `dim`, because a replicated placement multiplies nothing."""


class TestInferredGlobalShape:
    """What `DTensor.from_local` works out when it is not told the shape.

    Torch is not a declared dependency, so a build whose `from_local` predates
    `shape=`/`stride=` is reachable and Ravex falls back to it. That fallback
    assumes an even split. These tests are the arithmetic that decides whether
    the assumption holds here — the check that turns a silently wrong global
    shape into a refusal.
    """

    def test_an_even_split_infers_the_real_shape(self):
        from ravex._dist.collectives import _inferred_global_shape

        # 12 rows over 4 ranks: every rank holds 3, and 3 x 4 is the truth.
        assert _inferred_global_shape(
            (3, 512), [_Shard(0)], _Mesh(4)
        ) == (12, 512)

    def test_an_uneven_split_is_caught_from_every_rank(self):
        from ravex._dist.collectives import _inferred_global_shape

        # 10 rows over 4 ranks: torch gives 3, 3, 3, 1. No rank infers 10, and
        # that is what keeps some ranks from raising while others proceed —
        # a split verdict inside a collective is worse than either answer.
        for local_rows in (3, 3, 3, 1):
            inferred = _inferred_global_shape((local_rows, 512), [_Shard(0)], _Mesh(4))
            assert inferred != (10, 512), (
                "rank holding %d rows inferred %s, which agrees with a global "
                "shape it should not" % (local_rows, inferred)
            )

    def test_replicated_placements_multiply_nothing(self):
        from ravex._dist.collectives import _inferred_global_shape

        assert _inferred_global_shape(
            (12, 512), [_Replicate()], _Mesh(4)
        ) == (12, 512)

    def test_a_two_dimensional_mesh_multiplies_each_axis_once(self):
        from ravex._dist.collectives import _inferred_global_shape

        # Sharded on dim 0 over a mesh axis of 2, and on dim 1 over one of 3.
        assert _inferred_global_shape(
            (4, 5), [_Shard(0), _Shard(1)], _Mesh(2, 3)
        ) == (8, 15)

    def test_the_same_dimension_sharded_twice_compounds(self):
        from ravex._dist.collectives import _inferred_global_shape

        # Both mesh axes cut dim 0, so the local rows stand for 2 x 3 of them.
        assert _inferred_global_shape(
            (4, 5), [_Shard(0), _Shard(0)], _Mesh(2, 3)
        ) == (24, 5)


class TestTheOldSignatureFallback:
    """`_rebuild_dtensor` against a torch whose `from_local` takes no shape.

    Driven with stand-ins rather than a real mesh: the branch only runs on a
    torch older than the one installed anywhere we can test, and the thing
    worth testing is the decision, not torch. `_inferred_global_shape` above
    covers the arithmetic; this covers whether anything consults it.
    """

    @staticmethod
    def _live(local_rows, global_rows, mesh_size):
        import torch

        class Mesh:
            def size(self, mesh_dim):
                return mesh_size

        class Shard:
            dim = 0

        class Live:
            device_mesh = Mesh()
            placements = [Shard()]
            shape = (global_rows, 4)

            def to_local(self):
                return torch.zeros(local_rows, 4)

            def stride(self):
                return (4, 1)

        return Live()

    @staticmethod
    def _old_torch(monkeypatch):
        """A DTensor whose `from_local` rejects shape=/stride=, as older ones do."""

        class OldDTensor:
            @staticmethod
            def from_local(local, mesh, placements, run_check=None, **kwargs):
                if kwargs:
                    raise TypeError("from_local() got an unexpected keyword argument")
                return ("rebuilt", tuple(local.shape))

        import ravex._dist.collectives as distributed

        monkeypatch.setattr(distributed, "_dtensor_class", lambda: OldDTensor)

    def test_an_even_split_still_rebuilds(self, monkeypatch):
        """12 rows over 4 ranks: the inferred shape is the real one, so the
        older signature loses nothing and the fallback is taken."""
        import torch

        from ravex._dist.collectives import _rebuild_dtensor

        self._old_torch(monkeypatch)
        rebuilt = _rebuild_dtensor(
            {"local": torch.zeros(3, 4)}, self._live(3, 12, 4)
        )
        assert rebuilt == ("rebuilt", (3, 4))

    def test_an_uneven_split_is_refused_rather_than_rebuilt_wrong(self, monkeypatch):
        """10 rows over 4 ranks. Without shape= the rebuild would claim 12,
        and every shard would look fine on its own."""
        import torch

        from ravex._dist.collectives import _rebuild_dtensor

        self._old_torch(monkeypatch)
        with pytest.raises(ValueError) as caught:
            _rebuild_dtensor({"local": torch.zeros(3, 4)}, self._live(3, 10, 4))

        message = str(caught.value)
        assert "(12, 4)" in message, "the wrong shape it would have produced"
        assert "(10, 4)" in message, "and the right one"
        assert "shape=" in message, "and why this torch cannot say so"


@pytest.fixture
def one_rank_group():
    """A real gloo group, so the collectives below are on the wire."""
    import torch.distributed as dist

    if dist.is_initialized():  # pragma: no cover - a leaked group from elsewhere
        dist.destroy_process_group()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29614")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


class TestWhenTorchCannotReachNumpy:
    """Torch is installed, NumPy is not, and the object collectives still work.

    ``dist.all_gather_object`` decodes what it gathered with
    ``tensor.numpy().tobytes()``, so on such an install every one of them dies
    with "Numpy is not available" — after the wire work, which makes it a crash
    and not something a caller could recover from. Ravex requires PyYAML and
    nothing else and torch does not require NumPy either, so this is a
    configuration people can and do run in; it is also what the unit job in CI
    installs, which is how it was found: three reshard tests, all of them at
    the first collective.
    """

    def test_the_probe_answers_no_when_the_conversion_raises(self, monkeypatch):
        """Asked of ``tensor.numpy()`` rather than of ``import numpy``.

        Torch initialises NumPy itself, and a version it was not built against
        imports perfectly well and then fails the conversion.
        """
        import torch

        from ravex._dist import collectives as distributed

        def refuse(self, *args, **kwargs):
            raise RuntimeError("Numpy is not available")

        monkeypatch.setattr(torch.Tensor, "numpy", refuse, raising=False)
        monkeypatch.setattr(distributed, "_torch_numpy", None)

        assert distributed._torch_can_reach_numpy() is False

    def test_objects_survive_the_round_trip(self, one_rank_group, monkeypatch):
        """The encode, the padding and the NumPy-free decode, end to end."""
        from ravex._dist import collectives as distributed

        monkeypatch.setattr(distributed, "_torch_numpy", False)

        assert distributed.gather_objects({"rank": 0, "held": [1, 2, 3]}) == [
            {"rank": 0, "held": [1, 2, 3]}
        ]

    def test_the_agreements_built_on_it_still_answer(
        self, one_rank_group, monkeypatch
    ):
        """The three callers a resume cannot get past: the step every rank
        holds, whether every rank restored, and the run id."""
        from ravex._dist import collectives as distributed
        from ravex._dist.identity import FROM_STORE

        monkeypatch.setattr(distributed, "_torch_numpy", False)

        assert distributed.agree_on_step(7) == 7
        assert distributed.all_ranks_agree(True) is True
        assert distributed.all_ranks_agree(False) is False
        assert distributed.agree_on_run_id(FROM_STORE, "run-abc") == "run-abc"


def _numpy_is_reachable():
    """Whether torch can convert a tensor to NumPy in this interpreter.

    Not every runner can produce "a rank with NumPy". The unit job in CI
    installs torch and nothing else on purpose — that is the configuration
    `_all_gather_object` exists for, and how the defect it fixes was found —
    so on those jobs both paths in a "mixed" pair would be the NumPy-free one
    and the test would be asserting something it cannot arrange.

    `dist.all_gather_object` decodes with `tensor.numpy().tobytes()`, so where
    this is False torch's own object collectives do not work at all and there
    is nothing to be compatible *with*.
    """
    try:
        import torch

        torch.zeros(1, dtype=torch.uint8).numpy()
        return True
    except Exception:
        return False


NEEDS_NUMPY = pytest.mark.skipif(
    not _numpy_is_reachable(),
    reason="torch cannot reach NumPy here, so there is no 'with NumPy' rank to pair",
)


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _object_gather_worker(rank, world_size, port, no_numpy, payload, out):
    """One rank of a real gloo group, gathering `payload`.

    Top level and argument-driven because spawn has to pickle it.
    """
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if no_numpy:
        os.environ["RAVEX_ASSUME_NO_NUMPY"] = "1"
    else:
        os.environ.pop("RAVEX_ASSUME_NO_NUMPY", None)

    try:
        import torch.distributed as dist

        from ravex._dist import collectives as distributed

        # The module-level cache is per process, and this process is fresh;
        # clearing it makes the env variable the only thing deciding.
        distributed._torch_numpy = None
        took_the_new_path = not distributed._torch_can_reach_numpy()

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            got = distributed._all_gather_object(dist, payload)
        finally:
            dist.destroy_process_group()
        out.put((rank, took_the_new_path, got))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _gather_across_two_ranks(no_numpy_by_rank, payloads):
    """Run a two-rank gather and return each rank's result, in rank order."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_object_gather_worker,
            args=(rank, 2, port, no_numpy_by_rank[rank], payloads[rank], out),
        )
        for rank in (0, 1)
    ]
    for p in procs:
        p.start()

    results = {}
    try:
        for _ in procs:
            rank, took_new, got = out.get(timeout=120)
            results[rank] = (took_new, got)
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():  # pragma: no cover - a hung collective
                p.terminate()
                p.join(timeout=10)
    return results


class TestTheObjectGatherOnTwoRanks:
    """What one rank cannot ask.

    With ``world_size=1`` the replacement in `_all_gather_object` barely runs:
    `widest` is always this rank's own length, so the padding — the part most
    likely to be wrong — is never exercised, and there is no second rank to
    disagree with about the wire. These use two real processes and payloads of
    deliberately different sizes.

    Two processes on one box, not two machines: the question here is the shape
    of what goes on the wire, which loopback answers honestly. What loopback
    cannot answer — bandwidth, a peer that disappears — is what the rented kit
    in `integration/two-machines` is for.
    """

    # Sizes chosen to differ by a lot, so a rank that padded to the wrong
    # length produces garbage rather than something that happens to fit.
    PAYLOADS = [{"rank": 0, "blob": "a"}, {"rank": 1, "blob": "b" * 5000}]

    def test_both_ranks_without_numpy_agree(self):
        results = _gather_across_two_ranks([True, True], self.PAYLOADS)
        assert set(results) == {0, 1}, results
        for rank, (took_new, got) in results.items():
            assert took_new is True, f"rank {rank} did not take the new path"
            assert got == self.PAYLOADS, f"rank {rank} gathered {got!r}"

    @NEEDS_NUMPY
    def test_a_rank_with_numpy_and_a_rank_without_still_meet(self):
        """The claim `_all_gather_object` makes in its own docstring, and the
        one nothing else checks: the replacement posts the same collectives, in
        the same order and with the same shapes, as torch's own — so a mixed
        pair still agrees.

        It is not hypothetical. Two machines rented from one provider can come
        up from different images, and the unit job in CI installs no NumPy
        while every other job does.
        """
        results = _gather_across_two_ranks([False, True], self.PAYLOADS)
        assert set(results) == {0, 1}, results
        assert results[0][0] is False, "rank 0 was supposed to keep torch's path"
        assert results[1][0] is True, "rank 1 was supposed to take the new one"
        for rank, (_, got) in results.items():
            assert got == self.PAYLOADS, f"rank {rank} gathered {got!r}"

    def test_the_env_override_forces_the_replacement(self, monkeypatch):
        """`RAVEX_ASSUME_NO_NUMPY` exists so the replacement is reachable at
        all: every image worth renting ships NumPy, so without it the code that
        replaces torch's object collectives would reach a release having never
        run on a network."""
        from ravex._dist import collectives as distributed

        monkeypatch.setattr(distributed, "_torch_numpy", None)
        monkeypatch.setenv("RAVEX_ASSUME_NO_NUMPY", "1")
        assert distributed._torch_can_reach_numpy() is False

    @NEEDS_NUMPY
    def test_without_the_override_the_probe_decides(self, monkeypatch):
        """Turned off, the answer comes from asking torch rather than from the
        variable — which is only observable where the answer is yes."""
        from ravex._dist import collectives as distributed

        monkeypatch.setattr(distributed, "_torch_numpy", None)
        monkeypatch.setenv("RAVEX_ASSUME_NO_NUMPY", "0")
        assert distributed._torch_can_reach_numpy() is True


def _object_broadcast_worker(rank, world_size, port, no_numpy, payload, src, out):
    """One rank of a real gloo group, receiving `payload` from `src`."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if no_numpy:
        os.environ["RAVEX_ASSUME_NO_NUMPY"] = "1"
    else:
        os.environ.pop("RAVEX_ASSUME_NO_NUMPY", None)

    try:
        import torch.distributed as dist

        from ravex._dist import collectives as distributed

        distributed._torch_numpy = None
        took_the_new_path = not distributed._torch_can_reach_numpy()

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            got = distributed._broadcast_object(
                dist, payload if rank == src else None, src
            )
        finally:
            dist.destroy_process_group()
        out.put((rank, took_the_new_path, got))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _broadcast_across_two_ranks(no_numpy_by_rank, payload, src=0):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_object_broadcast_worker,
            args=(rank, 2, port, no_numpy_by_rank[rank], payload, src, out),
        )
        for rank in (0, 1)
    ]
    for proc in procs:
        proc.start()
    results = {}
    try:
        for _ in procs:
            rank, took_new, got = out.get(timeout=120)
            if took_new is None:
                pytest.fail("rank %d failed: %s" % (rank, got))
            results[rank] = (took_new, got)
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - a hung collective
                proc.terminate()
                proc.join(timeout=10)
    return results


class TestTheObjectBroadcastOnTwoRanks:
    """The other object collective, and the one that had been left behind.

    `_all_gather_object` replaced torch's gather because it decodes with
    `tensor.numpy().tobytes()`; `elastic.share_captured` went on calling
    `dist.broadcast_object_list`, which decodes exactly the same way. The CI
    job that installs no NumPy found it on 2026-09-13, at the worst possible
    line: the broadcast that hands every rank the captured state after an
    elastic regroup. The wire work was done and then it raised, so the run had
    already paid for the transfer and got a crash instead of a model.
    """

    # Deliberately not a small scalar: the payload a regroup broadcasts is a
    # state dict, and a length handled wrongly shows up as garbage rather than
    # as something that happens to fit.
    PAYLOAD = ({"layer.weight": [0.5] * 2000, "layer.bias": [0.25]}, {"step": 41})

    def test_both_ranks_without_numpy_agree(self):
        results = _broadcast_across_two_ranks([True, True], self.PAYLOAD)
        assert set(results) == {0, 1}, results
        for rank, (took_new, got) in results.items():
            assert took_new is True, "rank %d did not take the new path" % rank
            assert got == self.PAYLOAD, "rank %d received %r" % (rank, got)

    @NEEDS_NUMPY
    def test_a_rank_with_numpy_and_a_rank_without_still_meet(self):
        """The same claim `_all_gather_object` makes, on this primitive.

        NumPy is a property of the process, not of the job, so the receiving
        rank can be on one path while the source is on the other. They agree
        only if the replacement posts the same two broadcasts, in the same
        order and with the same dtypes, as torch's own — a `long` holding the
        length and then the payload as `uint8`.

        The source here is the rank *with* NumPy, which is the interesting
        direction: it encodes through torch and the other decodes through the
        replacement.
        """
        results = _broadcast_across_two_ranks([False, True], self.PAYLOAD, src=0)
        assert results[0][0] is False, "rank 0 was supposed to keep torch's path"
        assert results[1][0] is True, "rank 1 was supposed to take the new one"
        for rank, (_took, got) in results.items():
            assert got == self.PAYLOAD, "rank %d received %r" % (rank, got)

    @NEEDS_NUMPY
    def test_and_the_other_direction_too(self):
        """Source without NumPy, receiver with. Encoding and decoding are
        different halves of the compatibility claim and only this arm exercises
        the replacement's encoder against torch's decoder."""
        results = _broadcast_across_two_ranks([True, False], self.PAYLOAD, src=0)
        assert results[0][0] is True
        assert results[1][0] is False
        for rank, (_took, got) in results.items():
            assert got == self.PAYLOAD, "rank %d received %r" % (rank, got)


def _drain_split_worker(rank, world_size, port, sharded, out):
    """One rank deciding whether to split `drain`, the way the runtime does."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    try:
        import torch.distributed as dist

        from ravex._dist.collectives import all_ranks_agree, barrier, get_world_size

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            # The runtime's own guard, not a copy of it: `sharded and
            # get_world_size() > 1`, unconditional since GPU-98 — no
            # environment variable left to read asymmetrically.
            split = sharded and get_world_size() > 1
            if split:
                barrier()
            # A second collective every rank posts. If the ranks had disagreed
            # about the barrier above, this is where the mismatch would show.
            settled = all_ranks_agree(True)
        finally:
            dist.destroy_process_group()
        out.put((rank, split, settled))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _decide_drain_split_across_two_ranks(sharded):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_drain_split_worker, args=(rank, 2, port, sharded, out))
        for rank in (0, 1)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, split, settled = out.get(timeout=120)
            results[rank] = (split, settled)
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():  # pragma: no cover - a hung collective
                p.terminate()
                p.join(timeout=10)
    return results


class TestTheDrainSplitGuardMatchesTheVerdict:
    """The skew barrier (GPU-98) puts a collective in the checkpoint path,
    gated by `sharded and get_world_size() > 1` — the same guard the
    checkpoint's own verdict collective carries, and for the same reason: a
    barrier some ranks enter and others do not is two different collectives
    posted on one process group, a hang until NCCL gives up rather than an
    error.

    This used to be `RAVEX_SPLIT_DRAIN`, an opt-in diagnostic agreed on by
    every rank rather than read locally, because the variable could be set
    asymmetrically. There is no variable left to set wrong now — the guard
    comes from `sharded` and `get_world_size()`, which read the same on every
    rank by construction — but that symmetry is exactly the kind of thing that
    is easy to believe and wrong to assume: this pins it on two real
    processes, where `world_size=1` would show nothing and a mock could not
    be trusted either way.
    """

    def test_both_ranks_split_on_a_sharded_run(self):
        results = _decide_drain_split_across_two_ranks(sharded=True)
        assert set(results) == {0, 1}, results
        for rank, (split, settled) in results.items():
            assert split is True, f"rank {rank}: {split!r}"
            assert settled is True, f"rank {rank} did not reach the end: {settled!r}"

    def test_neither_rank_splits_on_a_replicated_run(self):
        """Not sharded: the guard must keep both ranks out of the barrier,
        the same way it keeps the non-main ranks of a real replicated job out
        of the verdict collective (see
        `test_a_replicated_job_posts_no_collective_from_rank_zero_alone` in
        `test_patches.py`) — a barrier only rank 0 entered is a hang, not an
        error, and this is the multi-process pin for that guard.
        """
        results = _decide_drain_split_across_two_ranks(sharded=False)
        assert set(results) == {0, 1}, results
        for rank, (split, settled) in results.items():
            assert split is False, f"rank {rank} barriered alone: {split!r}"
            assert settled is True, f"rank {rank} did not reach the end: {settled!r}"
