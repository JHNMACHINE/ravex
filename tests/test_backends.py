import importlib.util

import pytest
import torch

from ravex._backends import TorchSaveBackend, get_backend
from ravex._config import RavexConfig

HAVE_MOONCLIP = importlib.util.find_spec("moonclip") is not None


def make_config(tmp_path, **overrides):
    config = RavexConfig()
    config.storage.path = str(tmp_path / "checkpoints")
    for key, value in overrides.items():
        setattr(config, key, value)
    config._normalize()
    return config


def sample_state(step=1):
    return {
        "ravex_version": 1,
        "step": step,
        "models": {"model_a": {"weight": torch.arange(12, dtype=torch.float32)}},
        "optimizers": {},
        "schedulers": {},
        "scalers": {},
        "dataloaders": {},
        "sharded": {},
    }


def test_torch_save_roundtrip(tmp_path):
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    assert not backend.has_checkpoint()

    backend.save(7, sample_state(7), {"step": "7"})
    backend.flush()

    assert backend.has_checkpoint()
    loaded = backend.load_latest()
    assert loaded["step"] == 7
    assert torch.equal(
        loaded["models"]["model_a"]["weight"],
        sample_state()["models"]["model_a"]["weight"],
    )
    backend.close()


def test_an_unknown_backend_falls_back_instead_of_raising(tmp_path):
    backend = get_backend(make_config(tmp_path, backend="nonsense"))
    assert isinstance(backend, TorchSaveBackend)
    backend.close()


@pytest.mark.parametrize("distributed_env", [False, True])
@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_moonclip_backend_writes_under_a_torchrun_environment(
    tmp_path, monkeypatch, distributed_env
):
    """Moonclip must be told this is a single-rank save, whatever the env says.

    MoonclipManager infers world_size from RANK/WORLD_SIZE when it is not given
    one. Under torchrun it would therefore reject the single-rank save API -
    "Multi-rank save requires explicit create_snapshot/save_rank/finalize
    flow" - and Ravex, doing what it promises, would disable itself and let
    training continue with no checkpoints at all.

    Ravex never needs that flow: sharded state is gathered before it reaches a
    backend, and exactly one rank writes. Without this test the failure only
    shows up on a real multi-GPU run, as a single line in a log file.
    """
    if distributed_env:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")

    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    backend.save(4, sample_state(4), {"step": "4"})
    backend.flush()

    assert backend.has_checkpoint()
    loaded = backend.load_latest()
    assert loaded["step"] == 4
    assert torch.equal(
        loaded["models"]["model_a"]["weight"],
        sample_state()["models"]["model_a"]["weight"],
    )
    backend.close()


# ─── where the handoff spends its time ──────────────────────────────


def test_the_handoff_says_which_phase_it_spent_its_time_in(tmp_path):
    """A handoff costs the training loop wall time, and the total alone never
    says where it went.

    The measured 10.6 s per checkpoint on 8× RTX 5060 Ti took three A/B runs to
    *not* explain — thread count, compression level and cadence each moved it by
    under a second. The phases are what turns the next such number into a
    diagnosis instead of another round of experiments.
    """
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    phases = backend.save(1, sample_state(1), {"step": "1"})
    backend.close()

    assert list(phases) == ["copy", "queue"]
    assert all(seconds >= 0 for seconds in phases.values())


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_the_moonclip_handoff_separates_flattening_from_storing(tmp_path):
    """The two halves answer different questions: `flatten` is Python walking
    the state tree, `store` is the shadow copy plus any writer still draining."""
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    phases = backend.save(1, sample_state(1), {"step": "1"})
    backend.close()

    assert list(phases) == ["flatten", "store"]
    assert all(seconds >= 0 for seconds in phases.values())


# ─── per-rank stores ────────────────────────────────────────────────


def test_per_rank_gives_each_rank_a_store_of_its_own(tmp_path, monkeypatch):
    """Pointed at one store, N ranks are N writers against one manifest, each
    reading it, adding itself and writing it back with nothing between them."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    config = make_config(tmp_path, backend="torch_save")
    config.storage.prefix = "run-a"

    shared = get_backend(config, per_rank=False)
    assert shared.directory == str(tmp_path / "checkpoints")
    shared.close()

    mine = get_backend(config, per_rank=True)
    assert mine.directory == str(tmp_path / "checkpoints" / "rank_3")
    mine.close()

    # The remote side has to split too, or eight ranks would sync eight
    # different local stores onto one prefix in the bucket.
    from ravex._backends import _per_rank_config

    assert _per_rank_config(config, True).storage.prefix == "run-a/rank_3"


def test_a_backend_can_be_asked_for_a_specific_step(tmp_path):
    """Per-rank resume needs a step every rank holds, which is not always the
    newest one any of them has."""
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    assert backend.latest_step() is None

    for step in (4, 8, 12):
        backend.save(step, sample_state(step), {"step": str(step)})
        backend.flush()

    assert backend.latest_step() == 12
    assert backend.load_step(8)["step"] == 8
    assert backend.load_step(9) is None
    backend.close()


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_moonclip_can_be_asked_for_a_specific_step(tmp_path):
    """The same question against the default backend, where a snapshot is
    addressed by id and the step is metadata rather than a filename."""
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    assert backend.latest_step() is None

    for step in (4, 8, 12):
        backend.save(step, sample_state(step), {"step": str(step)})
    backend.flush()

    assert backend.latest_step() == 12
    loaded = backend.load_step(8)
    assert loaded["step"] == 8
    assert torch.equal(
        loaded["models"]["model_a"]["weight"],
        sample_state()["models"]["model_a"]["weight"],
    )
    assert backend.load_step(9) is None
    backend.close()
