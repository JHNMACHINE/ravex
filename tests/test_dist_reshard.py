"""Resuming a per-rank checkpoint onto a different number of ranks.

The planner in :mod:`ravex._dist.reshard` is pure arithmetic, so the tests for it
are exhaustive rather than illustrative. That is deliberate: the only shapes
where an old shard boundary lines up with a new one are exact halvings and
doublings, so a suite that tested 8 -> 4 and 4 -> 8 would pass with the general
mechanism unwritten. Every ``N -> M`` in a range is covered instead, against a
model of the tensor built out of row numbers, where a misplaced row is visible.

The end of the file is the real thing: four ranks' stores on disk, an FSDP2
model on one rank, and the assertion that what comes back is the tensor the
four ranks were holding between them.
"""

import logging
import os

import pytest

import torch

from ravex._dist.reshard import (
    ReshardUnsupported,
    check_covered,
    decode_placements,
    encode_placement,
    offsets_from_lengths,
    plan_reshard,
    shard_dim,
    sources_needed,
)


def chunked(total: int, parts: int):
    """How torch splits ``total`` rows over ``parts`` ranks.

    Reproduced here and *only* here: the planner never derives an offset, it
    is handed measurements. This exists to manufacture plausible measurements
    for the tests, including the uneven tails that are the interesting case.
    """
    size = -(-total // parts)  # ceil
    lengths = []
    left = total
    for _ in range(parts):
        take = min(size, max(left, 0))
        lengths.append(take)
        left -= take
    return lengths


class TestOffsets:
    def test_fence_posts(self):
        assert offsets_from_lengths([3, 3, 2]) == [0, 3, 6, 8]

    def test_empty_is_a_single_post(self):
        assert offsets_from_lengths([]) == [0]

    def test_a_rank_holding_nothing_is_not_an_error(self):
        # A world larger than the tensor: torch gives the tail ranks nothing,
        # and the plan has to survive a zero-length shard rather than divide
        # by it.
        assert offsets_from_lengths([1, 1, 0]) == [0, 1, 2, 2]


class TestPlanner:
    @pytest.mark.parametrize("total", [12, 13, 24, 31])
    def test_every_pair_reassembles_the_tensor(self, total):
        """Exhaustive over N -> M: the rows arrive where they belong.

        The tensor is modelled as the list ``[0, 1, ... total-1]``, so the
        assertion is not "the shapes match" — which a plan that shuffled rows
        would also satisfy — but that every row is in its place.
        """
        rows = list(range(total))
        for old_world in range(1, 9):
            for new_world in range(1, 9):
                old_lengths = chunked(total, old_world)
                new_lengths = chunked(total, new_world)
                old_at = offsets_from_lengths(old_lengths)
                new_at = offsets_from_lengths(new_lengths)
                old_shards = [
                    rows[old_at[q] : old_at[q + 1]] for q in range(old_world)
                ]

                plan = plan_reshard(old_lengths, new_lengths)
                assert len(plan) == new_world

                for r, pieces in enumerate(plan):
                    check_covered(pieces, new_lengths[r], "%d->%d" % (old_world, new_world))
                    built = []
                    for q, (start, stop), _ in pieces:
                        built.extend(old_shards[q][start:stop])
                    expected = rows[new_at[r] : new_at[r + 1]]
                    assert built == expected, (old_world, new_world, r)

    def test_four_to_three_stitches_every_shard_from_two(self):
        """The figure in the design, asserted.

        4 -> 3 over 12 rows is the smallest case where no new shard is any old
        shard: each is built from two, and one old shard feeds two new ones.
        Written out rather than swept into the loop above because it is the
        case that distinguishes a real planner from a shard permutation.
        """
        plan = plan_reshard([3, 3, 3, 3], [4, 4, 4])
        assert [len(pieces) for pieces in plan] == [2, 2, 2]
        # New rank 1 wants rows 4-7: the tail of old rank 1, the head of old 2.
        assert plan[1] == [(1, (1, 3), (0, 2)), (2, (0, 2), (2, 4))]

    def test_an_exact_halving_moves_whole_shards(self):
        """8 -> 4 is the easy case, and it is still supposed to be right."""
        plan = plan_reshard([2] * 8, [4] * 4)
        assert plan[0] == [(0, (0, 2), (0, 2)), (1, (0, 2), (2, 4))]

    def test_totals_that_disagree_are_refused(self):
        with pytest.raises(ValueError, match="not of this tensor"):
            plan_reshard([3, 3], [4, 4])

    def test_sources_needed_is_narrower_than_the_old_world(self):
        plan = plan_reshard([2] * 8, [4] * 4)
        assert sources_needed([plan[0]]) == {0, 1}
        assert sources_needed(plan) == set(range(8))


class TestCoverage:
    def test_a_gap_is_caught(self):
        with pytest.raises(ValueError, match="do not join up"):
            check_covered([(0, (0, 2), (0, 2)), (1, (0, 2), (3, 5))], 5, "t")

    def test_a_short_shard_is_caught(self):
        with pytest.raises(ValueError, match="cover 2 of 5"):
            check_covered([(0, (0, 2), (0, 2))], 5, "t")


class TestPlacements:
    """Torch's own objects in, plain data out, and the old strings still read."""

    def test_shard_and_replicate_round_trip(self):
        from torch.distributed.tensor import Replicate, Shard

        assert encode_placement(Shard(0)) == {"kind": "shard", "dim": 0}
        assert encode_placement(Shard(2)) == {"kind": "shard", "dim": 2}
        assert encode_placement(Replicate()) == {"kind": "replicate"}

    def test_partial_is_named_not_folded_into_replicate(self):
        from torch.distributed.tensor import Partial

        encoded = encode_placement(Partial())
        assert encoded["kind"] == "partial"

    @pytest.mark.parametrize(
        "text,expected",
        [
            # What `str(placement)` gives on torch 2.12 — which is what the
            # checkpoints written before this feature actually contain.
            ("S(0)", {"kind": "shard", "dim": 0}),
            ("S(3)", {"kind": "shard", "dim": 3}),
            ("R", {"kind": "replicate"}),
            # And what `repr` gives, which is what a reader would guess.
            ("Shard(dim=1)", {"kind": "shard", "dim": 1}),
            ("Replicate()", {"kind": "replicate"}),
        ],
    )
    def test_checkpoints_written_before_this_are_still_readable(self, text, expected):
        assert decode_placements([text]) == [expected]

    def test_legacy_partial_survives_as_partial(self):
        assert decode_placements(["P(sum)"])[0]["kind"] == "partial"
        assert decode_placements(["Partial(avg)"])[0]["kind"] == "partial"

    def test_an_unrecognised_string_is_carried_not_guessed(self):
        decoded = decode_placements(["Whatever(4)"])
        assert decoded == [{"kind": "unknown", "repr": "Whatever(4)"}]


class TestShardDim:
    def test_a_shard_names_its_dimension(self):
        assert shard_dim([{"kind": "shard", "dim": 1}]) == 1

    def test_replicated_is_none_not_an_error(self):
        assert shard_dim([{"kind": "replicate"}]) is None

    def test_a_two_dimensional_mesh_is_refused_by_name(self):
        with pytest.raises(ReshardUnsupported, match="2 mesh dimensions"):
            shard_dim([{"kind": "shard", "dim": 0}, {"kind": "shard", "dim": 1}])

    def test_a_partial_value_is_refused(self):
        with pytest.raises(ReshardUnsupported, match="Partial"):
            shard_dim([{"kind": "partial", "op": "sum"}])

    def test_an_unknown_placement_is_refused_with_what_it_was(self):
        with pytest.raises(ReshardUnsupported, match="Whatever"):
            shard_dim(decode_placements(["Whatever(4)"]))


# ─── the tree walkers ────────────────────────────────────────────────


def shard_node(local, dim=0, global_shape=None):
    """A shard as `_encode_shards` writes one, without needing a process group."""
    from ravex._dist.collectives import _SHARD_TAG

    return {
        _SHARD_TAG: 1,
        "local": local,
        "global_shape": list(global_shape or local.shape),
        "placements": [{"kind": "shard", "dim": dim}],
        "mesh_shape": [1],
    }


class TestTreeWalking:
    def test_extents_measure_the_sharded_dimension(self):
        from ravex._dist.collectives import shard_extents

        tree = {"a": shard_node(torch.zeros(3, 8)), "b": {"c": shard_node(torch.zeros(5, 2))}}
        assert shard_extents(tree) == {("a",): 3, ("b", "c"): 5}

    def test_replicated_tensors_are_left_out_of_the_plan(self):
        from ravex._dist.collectives import _SHARD_TAG, shard_extents

        tree = {
            "a": shard_node(torch.zeros(3, 8)),
            "r": {
                _SHARD_TAG: 1,
                "local": torch.zeros(4, 4),
                "global_shape": [4, 4],
                "placements": [{"kind": "replicate"}],
            },
        }
        assert shard_extents(tree) == {("a",): 3}

    def test_slices_are_copies_not_views(self):
        """A view would pin the whole old snapshot this pass exists to drop."""
        from ravex._dist.collectives import take_shard_slices

        source = torch.arange(24.0).reshape(6, 4)
        taken = take_shard_slices({"a": shard_node(source)}, {("a",): [(1, 3)]})
        piece = taken[("a",)][0]
        assert torch.equal(piece, source[1:3])
        piece[0, 0] = -1
        assert source[1, 0] == 4.0

    def test_non_shard_leaves_come_through_untouched(self):
        from ravex._dist.collectives import build_resharded_tree

        base = {"lr": 0.001, "steps": [1, 2], "w": shard_node(torch.zeros(2, 3))}
        live = {"lr": 0.9, "steps": [9], "w": shard_node(torch.zeros(4, 3))}
        built = build_resharded_tree(
            base, live, {("w",): [torch.ones(4, 3)]}
        )
        assert built["lr"] == 0.001
        assert built["steps"] == [1, 2]
        assert tuple(built["w"]["local"].shape) == (4, 3)

    def test_the_pieces_are_concatenated_in_order(self):
        from ravex._dist.collectives import build_resharded_tree

        base = {"w": shard_node(torch.zeros(2, 2))}
        live = {"w": shard_node(torch.zeros(4, 2))}
        pieces = [torch.full((1, 2), 7.0), torch.full((3, 2), 8.0)]
        built = build_resharded_tree(base, live, {("w",): pieces})
        assert built["w"]["local"][0, 0] == 7.0
        assert built["w"]["local"][1, 0] == 8.0

    def test_a_tensor_with_no_pieces_is_an_error_not_an_empty_shard(self):
        from ravex._dist.collectives import build_resharded_tree

        base = {"w": shard_node(torch.zeros(2, 2))}
        live = {"w": shard_node(torch.zeros(4, 2))}
        with pytest.raises(ValueError, match="nothing was read"):
            build_resharded_tree(base, live, {})


# ─── end to end ──────────────────────────────────────────────────────


class Captured(logging.Handler):
    """Records straight off the ``ravex`` logger.

    Not ``caplog``: the runtime sets ``propagate = False`` so the user's root
    logger stays untouched, and once any test in the session has activated
    Ravex the records never reach the root handler caplog installs. The twin of
    this class in ``test_dist_multinode.py`` is here for the same reason, and the
    duplication is cheaper than a shared import between test modules.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.text = ""

    def emit(self, record):
        self.text += record.getMessage() + "\n"


@pytest.fixture
def ravex_log():
    """Everything the ``ravex`` logger emits during the test."""
    logger = logging.getLogger("ravex")
    captured = Captured()
    logger.addHandler(captured)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield captured
    finally:
        logger.removeHandler(captured)
        logger.setLevel(previous)


@pytest.fixture
def one_rank_group():
    """A real gloo group of one, so FSDP2 gives real DTensors.

    World size one is not a toy here: it is the far end of a 4 -> 1 reshard,
    which is the shrink taken to its limit and the case where every old shard
    has to be stitched into a single tensor.
    """
    import torch.distributed as dist

    if dist.is_initialized():  # pragma: no cover - a leaked group from elsewhere
        dist.destroy_process_group()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29613")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def sharded_model():
    import torch.nn as nn
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 4))
    fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def take_a_step(model, optimizer):
    """One real step, so the optimizer has moments to reshard as well."""
    model(torch.randn(4, 12)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()


def split_tree(tree, parts):
    """Turn one rank's shards into ``parts`` ranks' shards of the same tensors.

    The inverse of what the reshard does, used to manufacture a checkpoint that
    a wider run would have written. Splitting with ``chunked`` rather than with
    ``torch.chunk`` keeps the fixture honest about uneven tails.
    """
    from ravex._dist.collectives import _SHARD_TAG, walk_shards

    import copy

    out = [copy.deepcopy(tree) for _ in range(parts)]
    for path, node in walk_shards(tree):
        from ravex._dist.collectives import node_at
        from ravex._dist.reshard import decode_placements, shard_dim

        dim = shard_dim(decode_placements(node["placements"]), "x")
        local = node["local"]
        if dim is None:
            continue
        lengths = chunked(local.shape[dim], parts)
        at = offsets_from_lengths(lengths)
        for q in range(parts):
            piece = local.narrow(dim, at[q], lengths[q]).clone()
            node_at(out[q], path)["local"] = piece
    return out


class TestEndToEnd:
    def test_four_ranks_resume_onto_one(self, one_rank_group, tmp_path, ravex_log):
        """The whole feature: 4 stores in, one correct model out."""
        from ravex._backends import get_backend, per_rank_store_path, store_config_at
        from ravex._config import RavexConfig, StorageConfig
        from ravex._dist.collectives import local_sharded_state
        from ravex._dist.identity import write_owner
        from ravex._registry import ObjectRegistry
        from ravex._resume import ResumeManager

        model, optimizer = sharded_model()
        take_a_step(model, optimizer)
        model_tree, optimizer_tree = local_sharded_state(model, [optimizer])

        # What the parameters were when the four-rank run wrote its stores.
        wanted = {
            name: param.to_local().detach().clone()
            for name, param in model.named_parameters()
        }

        config = RavexConfig(
            backend="torch_save",
            sharded_checkpoints="per_rank",
            reshard_on_resume=True,
            storage=StorageConfig(path=str(tmp_path / "checkpoints")),
        )

        old_world = 4
        model_parts = split_tree(model_tree, old_world)
        optimizer_parts = split_tree(optimizer_tree, old_world)
        for q in range(old_world):
            snapshot = {
                "ravex_version": 1,
                "step": 7,
                "models": {},
                "optimizers": {},
                "schedulers": {},
                "scalers": {},
                "dataloaders": {},
                "rng": {"torch": torch.zeros(8, dtype=torch.uint8)},
                "sharded": {
                    "sharded_0": {
                        "model": model_parts[q],
                        "optimizer": optimizer_parts[q],
                        "parameters": sorted(model_tree.keys()),
                        "layout": "per_rank",
                        "world_size": old_world,
                        "rank": q,
                    }
                },
            }
            store = get_backend(store_config_at(config, "rank_%d" % q))
            store.save(7, snapshot, {})
            store.close()
            write_owner(per_rank_store_path(config, q), "run-abc", q, old_world)

        # A fresh run, at world size 1, holding different numbers.
        fresh_model, fresh_optimizer = sharded_model()
        take_a_step(fresh_model, fresh_optimizer)
        for param in fresh_model.parameters():
            with torch.no_grad():
                param.to_local().fill_(0.0)

        registry = ObjectRegistry()
        registry.register_model(fresh_model)
        registry.register_optimizer(fresh_optimizer)
        assert registry.sharded_groups(), "the fixture model is not sharded"

        own = get_backend(config, per_rank=True)
        resume = ResumeManager(own, registry, config=config)

        restored = resume.try_resume(per_rank=True)
        own.close()

        assert restored, ravex_log.text
        assert registry.step_count == 7

        for name, param in fresh_model.named_parameters():
            assert torch.allclose(param.to_local(), wanted[name], atol=1e-6), name

        # The two things a resharded resume does not carry, said out loud.
        assert "are not restored" in ravex_log.text

    def test_a_missing_store_refuses_rather_than_leaving_a_hole(
        self, one_rank_group, tmp_path, ravex_log
    ):
        """The failure that must be loud, because its success is invisible.

        A reshard that carried on without rank 2 would produce tensors with a
        band of uninitialised rows: right shape, right dtype, wrong model, and
        nothing downstream able to tell. So the run starts from scratch and the
        message names the rank whose store is gone.
        """
        import shutil

        from ravex._backends import get_backend, per_rank_store_path, store_config_at
        from ravex._config import RavexConfig, StorageConfig
        from ravex._dist.collectives import local_sharded_state
        from ravex._dist.identity import write_owner
        from ravex._registry import ObjectRegistry
        from ravex._resume import ResumeManager

        model, optimizer = sharded_model()
        take_a_step(model, optimizer)
        model_tree, optimizer_tree = local_sharded_state(model, [optimizer])

        config = RavexConfig(
            backend="torch_save",
            sharded_checkpoints="per_rank",
            reshard_on_resume=True,
            storage=StorageConfig(path=str(tmp_path / "checkpoints")),
        )

        old_world = 4
        model_parts = split_tree(model_tree, old_world)
        optimizer_parts = split_tree(optimizer_tree, old_world)
        for q in range(old_world):
            snapshot = {
                "ravex_version": 1,
                "step": 3,
                "models": {},
                "optimizers": {},
                "schedulers": {},
                "scalers": {},
                "dataloaders": {},
                "sharded": {
                    "sharded_0": {
                        "model": model_parts[q],
                        "optimizer": optimizer_parts[q],
                        "layout": "per_rank",
                        "world_size": old_world,
                        "rank": q,
                    }
                },
            }
            store = get_backend(store_config_at(config, "rank_%d" % q))
            store.save(3, snapshot, {})
            store.close()
            write_owner(per_rank_store_path(config, q), "run-abc", q, old_world)

        # The machine that held rank 2 is gone, and nobody kept a copy.
        shutil.rmtree(per_rank_store_path(config, 2))

        fresh_model, fresh_optimizer = sharded_model()
        registry = ObjectRegistry()
        registry.register_model(fresh_model)
        registry.register_optimizer(fresh_optimizer)

        own = get_backend(config, per_rank=True)
        resume = ResumeManager(own, registry, config=config)
        restored = resume.try_resume(per_rank=True)
        own.close()

        assert not restored
        assert "rank(s) 2" in ravex_log.text
        assert "uninitialised" in ravex_log.text

    def test_the_mismatch_is_reported_even_with_the_feature_off(
        self, one_rank_group, tmp_path, ravex_log
    ):
        """Detection is unconditional; only acting on it is opt-in.

        The case this protects against is a launcher that started the wrong
        number of ranks: silence there is a run that trains on happily at a
        topology nobody asked for.
        """
        from ravex._backends import get_backend, per_rank_store_path, store_config_at
        from ravex._config import RavexConfig, StorageConfig
        from ravex._dist.identity import write_owner
        from ravex._registry import ObjectRegistry
        from ravex._resume import ResumeManager

        config = RavexConfig(
            backend="torch_save",
            sharded_checkpoints="per_rank",
            reshard_on_resume=False,
            storage=StorageConfig(path=str(tmp_path / "checkpoints")),
        )
        for q in range(4):
            store = get_backend(store_config_at(config, "rank_%d" % q))
            store.save(1, {"ravex_version": 1, "step": 1}, {})
            store.close()
            write_owner(per_rank_store_path(config, q), "run-abc", q, 4)

        registry = ObjectRegistry()
        own = get_backend(config, per_rank=True)
        resume = ResumeManager(own, registry, config=config)
        resume.try_resume(per_rank=True)
        own.close()

        assert "reshard_on_resume" in ravex_log.text
        assert "4-rank run and this one has 1" in ravex_log.text
