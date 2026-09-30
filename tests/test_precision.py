"""``@ravex.save_dtype``: precision declared on the module (GPU-137).

The criterion the issue closes on: a MoE with the experts annotated fp8 and the
rest bf16 writes a checkpoint where the experts carry fp8's error and the rest
bf16's — on a sharded run and on one that is not — with no ``save_dtype`` line
in the configuration, and renaming the attribute changes nothing.
"""

import logging
import os

import pytest
import torch
import torch.nn as nn

import ravex
from ravex import _precision
from ravex._config import RavexConfig
from ravex._precision import declared_rules
from ravex._registry import ObjectRegistry

try:
    import moonclip  # noqa: F401

    HAVE_MOONCLIP = True
except ImportError:  # pragma: no cover
    HAVE_MOONCLIP = False

needs_moonclip = pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")

#: Between the two errors the issue measured: bf16 4.8e-4, fp8 8.8e-3.
BF16_AT_MOST = 2e-3
FP8_AT_LEAST = 3e-3


@pytest.fixture(autouse=True)
def clean_declarations():
    _precision._instances.clear()
    _precision._classes.clear()
    _precision._said.clear()
    yield
    _precision._instances.clear()
    _precision._classes.clear()
    _precision._said.clear()


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(16, 16)


def moe(experts_attr="experts"):
    """A router and four experts, with the experts under a chosen name."""

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(16, 4)
            setattr(self, experts_attr, nn.ModuleList(Expert() for _ in range(4)))

        def forward(self, x):
            experts = getattr(self, experts_attr)
            weights = self.router(x).softmax(-1)
            return sum(weights[:, i : i + 1] * e.up(x) for i, e in enumerate(experts))

    torch.manual_seed(0)
    return MoE()


def registry_of(model, optimizer=None):
    registry = ObjectRegistry()
    registry.register_model(model)
    registry.register_optimizer(
        optimizer or torch.optim.SGD(model.parameters(), lr=0.1)
    )
    return registry


# ─── the API ────────────────────────────────────────────────────────


class TestDeclaring:
    def test_a_decorated_class_is_declared(self):
        @ravex.save_dtype("FP8")
        class Block(nn.Module):
            pass

        assert _precision.declared_dtype(Block()) == "fp8"

    def test_an_instance_is_declared_and_returned(self):
        layer = nn.Linear(2, 2)
        assert ravex.save_dtype(layer, "bf16") is layer
        assert _precision.declared_dtype(layer) == "bf16"
        assert _precision.declared_dtype(nn.Linear(2, 2)) is None

    def test_an_instance_beats_its_class(self):
        @ravex.save_dtype("fp8")
        class Block(nn.Module):
            pass

        kept = ravex.save_dtype(Block(), "none")
        assert _precision.declared_dtype(kept) == "none"
        assert _precision.declared_dtype(Block()) == "fp8"

    def test_a_subclass_inherits(self):
        @ravex.save_dtype("fp8")
        class Block(nn.Module):
            pass

        class Wider(Block):
            pass

        assert _precision.declared_dtype(Wider()) == "fp8"

    def test_a_bad_dtype_raises_where_it_is_written(self):
        with pytest.raises(ValueError, match="int8"):
            ravex.save_dtype("int8")
        with pytest.raises(ValueError, match="int8"):
            ravex.save_dtype(nn.Linear(2, 2), "int8")

    def test_a_bad_target_raises(self):
        with pytest.raises(TypeError):
            ravex.save_dtype(object(), "bf16")
        with pytest.raises(TypeError, match="needs a dtype"):
            ravex.save_dtype(nn.Linear(2, 2))

    def test_no_train_loop_is_needed(self):
        # A declaration at import time, long before any runtime exists.
        assert not ravex.is_active()
        ravex.save_dtype(nn.Linear(2, 2), "bf16")


# ─── resolution into globs ──────────────────────────────────────────


class TestRules:
    def test_the_deepest_annotation_comes_first(self):
        model = moe()
        ravex.save_dtype(model, "bf16")
        ravex.save_dtype(model.experts, "fp8")
        ravex.save_dtype(model.experts[2], "none")

        rules = list(declared_rules(registry_of(model)).items())
        key = ObjectRegistry.fingerprint_model(model)
        assert rules == [
            ("ravex/models/%s/experts.2.*" % key, "none"),
            ("ravex/models/%s/experts.*" % key, "fp8"),
            ("ravex/models/%s/*" % key, "bf16"),
        ]

    def test_a_class_annotation_reaches_every_instance(self):
        _precision.declare(Expert, "fp8")
        model = moe()
        rules = declared_rules(registry_of(model))
        assert len(rules) == 4
        assert all(p.endswith((".0.*", ".1.*", ".2.*", ".3.*")) for p in rules)

    def test_index_one_does_not_cover_index_ten(self):
        model = nn.Sequential(*[nn.Linear(2, 2) for _ in range(11)])
        ravex.save_dtype(model[1], "fp8")
        (pattern,) = declared_rules(registry_of(model))
        matcher = _precision._glob(pattern)
        prefix = pattern[: -len("1.*")]
        assert matcher.fullmatch(prefix + "1.weight")
        assert not matcher.fullmatch(prefix + "10.weight")


class TestTiers:
    """Specific config glob > annotation > default, and order within each."""

    def config(self, save_dtype, declared=None):
        config = RavexConfig()
        config.save_dtype = save_dtype
        config._normalize()
        config.declared_save_dtype = declared
        return config

    def test_without_annotations_the_scalar_stays_a_scalar(self):
        assert self.config("bf16").resolve_save_dtype() == "bf16"

    def test_the_scalar_becomes_a_trailing_catch_all(self):
        resolved = self.config("bf16", {"ravex/models/m/experts.*": "fp8"})
        assert list(resolved.resolve_save_dtype().items()) == [
            ("ravex/models/m/experts.*", "fp8"),
            ("*", "bf16"),
        ]

    def test_annotations_only(self):
        declared = {"ravex/models/m/experts.*": "fp8"}
        assert self.config(None, declared).resolve_save_dtype() == declared

    def test_a_naming_glob_beats_an_annotation_and_a_component_loses(self):
        config = self.config(
            {"model": "bf16", "*experts.0*": "fp32"},
            {"ravex/models/m/experts.*": "fp8"},
        )
        assert list(config.resolve_save_dtype().items()) == [
            ("*experts.0*", "fp32"),
            ("ravex/models/m/experts.*", "fp8"),
            ("ravex/models/*", "bf16"),
            ("ravex/sharded/*/model/*", "bf16"),
        ]

    def test_a_glob_after_a_component_is_no_longer_shadowed(self):
        """The order bug the issue opened with, fixed with no annotation at all."""
        resolved = self.config({"model": "bf16", "*expert*": "fp8"}).resolve_save_dtype()
        assert next(iter(resolved)) == "*expert*"

    def test_the_bare_catch_all_keeps_its_place(self):
        """``{model: none, "*": bf16}`` still means "everything but the weights"."""
        resolved = self.config({"model": "none", "*": "bf16"}).resolve_save_dtype()
        assert list(resolved)[-1] == "*"
        assert resolved["ravex/models/*"] == "none"


# ─── a declaration that does nothing says so ────────────────────────


class TestSilenceIsReported:
    @pytest.fixture
    def warnings(self, caplog):
        caplog.set_level(logging.WARNING, logger="ravex")
        return lambda: [r.getMessage() for r in caplog.records]

    def test_a_class_no_model_contains(self, warnings):
        @ravex.save_dtype("fp8")
        class Unused(nn.Module):
            pass

        declared_rules(registry_of(moe()))
        assert any("class Unused" in m for m in warnings())

    def test_an_instance_outside_every_model(self, warnings):
        stray = ravex.save_dtype(nn.Linear(2, 2), "fp8")
        declared_rules(registry_of(moe()))
        assert any("not part of any model" in m for m in warnings())
        del stray

    def test_a_module_with_no_tensors_of_its_own(self, warnings):
        model = moe()
        ravex.save_dtype(model.experts, "bf16")
        for expert in model.experts:
            ravex.save_dtype(expert, "fp8")
        declared_rules(registry_of(model))
        assert any("experts (ModuleList)" in m and "covers no tensor" in m for m in warnings())

    def test_an_annotation_a_config_glob_overrides(self, warnings):
        model = moe()
        ravex.save_dtype(model.router, "fp8")
        declared_rules(registry_of(model), ["*router*"])
        assert any("overridden" in m and "*router*" in m for m in warnings())

    def test_a_live_annotation_is_quiet(self, warnings):
        model = moe()
        ravex.save_dtype(model.experts, "fp8")
        declared_rules(registry_of(model), ["*router*"])
        assert warnings() == []


# ─── the criterion ──────────────────────────────────────────────────


def errors_by_part(saved, reference):
    """Largest error on the router, and on the experts, against ``reference``."""
    router, experts = 0.0, 0.0
    for name, want in reference.items():
        got = saved[name]
        if isinstance(got, dict):  # a per-rank shard node
            got = got["local"]
        if hasattr(got, "full_tensor"):
            got = got.full_tensor()
        err = (got.float() - want.float()).abs().max().item()
        if "experts" in name:
            experts = max(experts, err)
        else:
            router = max(router, err)
    return router, experts


def full_state(model):
    state = {}
    for name, value in model.state_dict().items():
        state[name.replace("moe_experts", "experts")] = (
            value.full_tensor() if hasattr(value, "full_tensor") else value
        ).detach().clone()
    return state


@needs_moonclip
@pytest.mark.parametrize("attr", ["experts", "moe_experts"])
def test_unsharded_run_through_train_loop(storage, attr):
    """A real run: annotations on the model, no save_dtype anywhere in config."""
    from ravex._backends import MoonclipBackend

    _precision.declare(Expert, "fp8")
    finished = {}

    @ravex.train_loop(backend="moonclip", async_save=False, checkpoint_every=2)
    def train():
        model = moe(attr)
        ravex.save_dtype(model, "bf16")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(4):
            model(torch.randn(8, 16)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        finished["state"] = full_state(model)

    train()

    config = RavexConfig()
    config.storage.path = str(storage)
    config._normalize()
    assert config.save_dtype is None
    loaded = MoonclipBackend(config).load_latest()
    (saved,) = loaded["models"].values()
    saved = {k.replace("moe_experts", "experts"): v for k, v in saved.items()}

    router, experts = errors_by_part(saved, finished["state"])
    assert 0 < router <= BF16_AT_MOST, router
    assert experts >= FP8_AT_LEAST, experts


@pytest.fixture
def one_rank_group():
    import torch.distributed as dist

    if dist.is_initialized():  # pragma: no cover
        dist.destroy_process_group()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@needs_moonclip
@pytest.mark.parametrize("layout", ["gather", "per_rank"])
@pytest.mark.parametrize("attr", ["experts", "moe_experts"])
def test_sharded_run(one_rank_group, tmp_path, layout, attr):
    """FSDP2, both layouts: the rules land on ``ravex/sharded/<key>/model/``."""
    from torch.distributed.fsdp import fully_shard

    from ravex._backends import MoonclipBackend

    _precision.declare(Expert, "fp8")
    model = moe(attr)
    ravex.save_dtype(model, "bf16")
    fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.randn(8, 16)).sum().backward()
    optimizer.step()

    registry = registry_of(model, optimizer)
    assert registry.sharded_groups(), "the model should count as sharded"

    config = RavexConfig()
    config.storage.path = str(tmp_path / "checkpoints")
    config.backend = "moonclip"
    config.async_save = False
    config._normalize()
    config.declared_save_dtype = declared_rules(registry)
    assert all(p.startswith("ravex/sharded/") for p in config.declared_save_dtype)

    state = registry.collect_state(sharded_layout=layout)
    backend = MoonclipBackend(config)
    backend.save(1, state, {})
    backend.flush()

    loaded = MoonclipBackend(config).load_latest()
    (group,) = loaded["sharded"].values()
    saved = {k.replace("moe_experts", "experts"): v for k, v in group["model"].items()}

    router, experts = errors_by_part(saved, full_state(model))
    assert 0 < router <= BF16_AT_MOST, router
    assert experts >= FP8_AT_LEAST, experts
