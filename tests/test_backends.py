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
