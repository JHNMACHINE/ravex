"""Converting a foreign checkpoint at resume time — GPU-90.

The conversion itself is covered in ``test_convert.py`` and, against real
DeepSpeed artefacts, in ``integration/frameworks/check_zero_convert.py``. What
is here is the decision *around* it: when Ravex converts, when it declines, and
what it says either way.

Three of those decisions carry the weight:

- **Detection is unconditional, acting on it is opt-in.** A foreign checkpoint
  where Ravex's own store belongs is usually a path pointing somewhere
  unintended, and converting it quietly turns that into a run trained on
  someone else's weights.
- **A DeepSpeed run resuming a DeepSpeed checkpoint is declined by name.** The
  engine loads its own checkpoints; doing it twice by two routes is how the
  optimizer ends up disagreeing with itself.
- **Ranks that see different things all take the ordinary path.** Applying a
  converted state is collective, so a rank converting alone waits for peers
  that went elsewhere.
"""

import logging

import pytest

import torch
import torch.nn as nn

from ravex._config import RavexConfig
from ravex._runtime import RavexRuntime

dcp = pytest.importorskip("torch.distributed.checkpoint")


class Captured(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        # `getMessage` already interpolates the args; doing it again turns
        # every message with a literal % in it into a TypeError.
        self.lines.append(record.getMessage())

    @property
    def text(self):
        return "\n".join(self.lines)


@pytest.fixture
def log():
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


class OneModel:
    """A registry holding a single plain model, which is all this path needs.

    ``optimizers`` and ``keyed_optimizers`` are here because the conversion
    reaches for them to restore Adam's moments — a plain-model resume that
    brought back the weights and dropped the moments would be a quieter
    version of the failure this whole feature is careful about.
    """

    def __init__(self, model=None, sharded=(), optimizers=(), key="model_1"):
        self._model = model
        self._sharded = list(sharded)
        self._optimizers = list(optimizers)
        self._key = key
        self.restored = None

    def sharded_groups(self):
        return self._sharded

    def keyed_models(self):
        return [(self._key, self._model)] if self._model is not None else []

    @property
    def optimizers(self):
        return self._optimizers

    def keyed_optimizers(self):
        return [("opt_1", opt) for opt in self._optimizers]

    def restore_state(self, state, defer_rng=False):
        self.restored = state


def runtime_for(tmp_path, registry, convert_foreign=False, remote=False):
    config = RavexConfig()
    config.storage.type = "s3" if remote else "local"
    if remote:
        config.storage.bucket = "somewhere"
    config.storage.path = str(tmp_path)
    config.convert_foreign = convert_foreign

    runtime = RavexRuntime.__new__(RavexRuntime)
    runtime.config = config
    runtime.registry = registry
    return runtime


@pytest.fixture
def foreign(tmp_path, monkeypatch):
    """A real distributed checkpoint of a small model, at the storage path."""
    monkeypatch.setattr("ravex._frameworks.detect_framework", lambda: "vanilla")
    torch.manual_seed(0)
    trained = nn.Sequential(nn.Linear(4, 3))
    trained[0].weight.data.fill_(7.0)
    dcp.save(trained.state_dict(), checkpoint_id=str(tmp_path))
    return trained


# ── when it does nothing ────────────────────────────────────────────────────


def test_a_ravex_store_is_left_alone(tmp_path, log):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    runtime = runtime_for(tmp_path, OneModel(nn.Linear(2, 2)), convert_foreign=True)
    assert runtime._convert_foreign(defer_rng=False) is False
    assert "written by something else" not in log.text


def test_remote_storage_is_not_inspected(tmp_path, log):
    """`identify` reads a directory, and a bucket is not one."""
    runtime = runtime_for(tmp_path, OneModel(nn.Linear(2, 2)), remote=True)
    assert runtime._convert_foreign(defer_rng=False) is False
    assert log.text == ""


# ── detection without action ────────────────────────────────────────────────


def test_a_foreign_checkpoint_is_reported_and_not_converted(tmp_path, foreign, log):
    """Off by default, and the message says how to turn it on and why it is off."""
    registry = OneModel(nn.Sequential(nn.Linear(4, 3)))
    runtime = runtime_for(tmp_path, registry, convert_foreign=False)

    assert runtime._convert_foreign(defer_rng=False) is False
    assert registry.restored is None
    assert "written by something else" in log.text
    assert "convert_foreign=true" in log.text
    assert "somewhere unintended" in log.text


def test_a_deepspeed_run_is_left_to_its_own_engine(tmp_path, monkeypatch, log):
    """The engine loads its own checkpoints; two routes disagree about the optimizer."""
    tag = tmp_path / "step1"
    tag.mkdir()
    (tag / "mp_rank_00_model_states.pt").write_text("", encoding="utf-8")
    (tag / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_text("", encoding="utf-8")
    (tmp_path / "latest").write_text("step1", encoding="utf-8")

    monkeypatch.setattr("ravex._frameworks.detect_framework", lambda: "deepspeed")
    runtime = runtime_for(tmp_path, OneModel(nn.Linear(2, 2)), convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is False
    assert "leaving it to the engine" in log.text
    assert "convert_foreign" not in log.text


def test_a_deepspeed_checkpoint_still_converts_for_a_plain_run(
    tmp_path, monkeypatch, log
):
    """The guard is about the pair, not about the format.

    Reaching the conversion and failing there on an empty fixture is the point:
    what is being checked is that the *decision* got that far.
    """
    tag = tmp_path / "step1"
    tag.mkdir()
    (tag / "mp_rank_00_model_states.pt").write_text("", encoding="utf-8")
    (tag / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_text("", encoding="utf-8")
    (tmp_path / "latest").write_text("step1", encoding="utf-8")

    monkeypatch.setattr("ravex._frameworks.detect_framework", lambda: "vanilla")
    runtime = runtime_for(tmp_path, OneModel(nn.Linear(2, 2)), convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is False
    assert "leaving it to the engine" not in log.text
    assert "Could not convert" in log.text


# ── choosing what to convert into ───────────────────────────────────────────


def test_several_groups_are_refused_rather_than_guessed(tmp_path, foreign, log):
    model = nn.Sequential(nn.Linear(4, 3))
    registry = OneModel(
        None, sharded=[("a", model, []), ("b", model, [])]
    )
    runtime = runtime_for(tmp_path, registry, convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is False
    assert "not something to guess at" in log.text


def test_no_model_at_all_is_refused(tmp_path, foreign, log):
    runtime = runtime_for(tmp_path, OneModel(None), convert_foreign=True)
    assert runtime._convert_foreign(defer_rng=False) is False
    assert "0 sharded group(s) and 0 plain model(s)" in log.text


# ── and the conversion actually happening ───────────────────────────────────


def test_a_plain_model_is_restored_from_a_foreign_checkpoint(tmp_path, foreign, log):
    """The whole point, end to end through the resume decision."""
    registry = OneModel(nn.Sequential(nn.Linear(4, 3)))
    runtime = runtime_for(tmp_path, registry, convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is True

    snapshot = registry.restored
    assert snapshot is not None
    assert set(snapshot["models"]) == {"model_1"}
    restored = snapshot["models"]["model_1"]
    for name, tensor in foreign.state_dict().items():
        assert torch.equal(restored[name], tensor), name

    assert "Converting from" in log.text
    assert "the RNG are not restored" in log.text


def test_a_name_mismatch_is_refused_rather_than_half_loaded(tmp_path, foreign, log):
    """A model of a different shape shares no parameter names with this one."""
    registry = OneModel(nn.ModuleDict({"other": nn.Linear(4, 3)}))
    runtime = runtime_for(tmp_path, registry, convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is False
    assert registry.restored is None
    assert "nothing to convert into it" in log.text


def test_the_optimizer_moments_come_across_on_the_plain_path(tmp_path, monkeypatch):
    """A resume that restores weights and drops Adam is the quiet failure.

    The plain path first did exactly that: the snapshot it built had no
    optimizer state in it at all, so a converted resume came up with the right
    weights and a freshly created optimizer, and nothing said so. Torch's
    plain ``load_state_dict`` wants positions rather than names, which is the
    only reason this needs its own step.
    """
    monkeypatch.setattr("ravex._frameworks.detect_framework", lambda: "vanilla")

    torch.manual_seed(0)
    trained = nn.Sequential(nn.Linear(4, 3))
    trainer = torch.optim.Adam(trained.parameters(), lr=0.01)
    trained(torch.randn(2, 4)).sum().backward()
    trainer.step()
    dcp.save(
        {"model": trained.state_dict(), "optim": trainer.state_dict()},
        checkpoint_id=str(tmp_path),
    )

    live = nn.Sequential(nn.Linear(4, 3))
    optimizer = torch.optim.Adam(live.parameters(), lr=0.01)
    registry = OneModel(live, optimizers=[optimizer])
    runtime = runtime_for(tmp_path, registry, convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is True

    saved = registry.restored["optimizers"]
    assert saved, "the optimizer state was dropped"
    state = next(iter(saved.values()))["state"]
    assert set(state) == {0, 1}, "keyed by position, one entry per parameter"
    assert "exp_avg" in state[0]


def test_moments_that_do_not_cover_the_model_are_not_half_restored(
    tmp_path, foreign, log
):
    """Partial is refused: the right length with the wrong pairing loads silently."""
    live = nn.Sequential(nn.Linear(4, 3))
    optimizer = torch.optim.Adam(live.parameters(), lr=0.01)
    registry = OneModel(live, optimizers=[optimizer])
    runtime = runtime_for(tmp_path, registry, convert_foreign=True)

    # `foreign` saved the weights only, so there are no moments to bring over.
    assert runtime._convert_foreign(defer_rng=False) is True
    assert registry.restored["optimizers"] == {}


def test_a_positional_optimizer_state_is_not_forced_onto_a_sharded_model(
    tmp_path, monkeypatch, log
):
    """`set_state_dict` matches by name; positions belong to the run that wrote them.

    The weights still come across — refusing the whole conversion over the
    optimizer would throw away the part that *is* unambiguous.
    """
    monkeypatch.setattr("ravex._frameworks.detect_framework", lambda: "vanilla")
    torch.manual_seed(0)
    trained = nn.Sequential(nn.Linear(4, 3))
    trainer = torch.optim.Adam(trained.parameters(), lr=0.01)
    trained(torch.randn(2, 4)).sum().backward()
    trainer.step()
    dcp.save(
        {"model": trained.state_dict(), "optim": trainer.state_dict()},
        checkpoint_id=str(tmp_path),
    )

    live = nn.Sequential(nn.Linear(4, 3))
    registry = OneModel(None, sharded=[("g", live, [])])
    runtime = runtime_for(tmp_path, registry, convert_foreign=True)

    assert runtime._convert_foreign(defer_rng=False) is True
    group = registry.restored["sharded"]["g"]
    assert group["optimizer"] == {"state": {}, "param_groups": []}
    assert "keyed by position" in log.text
    for name, tensor in trained.state_dict().items():
        assert torch.equal(group["model"][name], tensor), name
