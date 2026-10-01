"""Exporting a Ravex checkpoint as a DeepSpeed universal checkpoint — GPU-145.

The authoritative check is not here. It is
``integration/frameworks/check_deepspeed_export.py``, which has DeepSpeed itself
resume the export at ZeRO stages 1, 2 and 3 and one to three ranks, reads back
what it holds through Ravex's ZeRO reader, and compares the next step against
the one DeepSpeed's own converter leads to. That needs Linux and DeepSpeed.

What is here is what that cannot be: fast, dependency-free, and pinned to the
decisions the format leaves to the writer - which tensors go where by name,
what is left out on purpose, and which ZeRO settings are *not* written so that
the resuming run's own configuration stands.
"""

import os

import pytest
import torch
import torch.nn as nn

import ravex
from ravex._interop import export
from ravex._interop.convert import CannotConvert


def build():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 1))


def trained(named=True, steps=3):
    """A model and an Adam that have taken ``steps`` steps."""
    model = build()
    optimizer = torch.optim.Adam(model.named_parameters() if named else model.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(8, 4)).sum().backward()
        optimizer.step()
    return model, optimizer


def plain_state(model, optimizer, step=3):
    return {
        "step": step,
        "models": {"model": model.state_dict()},
        "optimizers": {"optimizer": optimizer.state_dict()},
    }


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def tag_dir(out):
    with open(os.path.join(out, "latest_universal"), encoding="utf-8") as handle:
        return os.path.join(out, handle.read().strip())


class TestTheLayout:
    def test_every_parameter_and_moment_lands_under_its_name(self, tmp_path):
        model, optimizer = trained()
        out = str(tmp_path / "ds")
        notes = export.to_deepspeed(plain_state(model, optimizer), out)

        root = tag_dir(out)
        assert os.path.basename(root) == "global_step3_universal"
        by_position = optimizer.state_dict()["state"]
        for index, (name, param) in enumerate(model.named_parameters()):
            folder = os.path.join(root, "zero", name)
            fp32 = load(os.path.join(folder, "fp32.pt"))
            assert fp32["cat_dim"] == 0 and torch.equal(fp32["param"], param.detach())
            for buffer in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(load(os.path.join(folder, buffer + ".pt"))["param"], by_position[index][buffer])
            assert load(os.path.join(folder, "step.pt")) == 3
        assert any("param_names" in note for note in notes)

    def test_the_model_states_stage_3_also_looks_for(self, tmp_path):
        model, optimizer = trained()
        out = str(tmp_path / "ds")
        export.to_deepspeed(plain_state(model, optimizer), out)
        root = tag_dir(out)

        states = load(os.path.join(root, "mp_rank_00_model_states.pt"))
        assert set(states["module"]) == set(model.state_dict())
        assert list(states["param_shapes"][0]) == [name for name, _ in model.named_parameters()]
        assert states["global_steps"] == 3
        assert os.path.exists(os.path.join(root, "zero_pp_rank_0_mp_rank_00_model_states.pt"))

    def test_the_resuming_runs_own_zero_settings_are_left_alone(self, tmp_path):
        """DeepSpeed reads these with ``sd.get(key, its own)``: written, a
        clip_grad of 0 turned the resuming run's gradient clipping off."""
        from packaging.version import Version

        model, optimizer = trained()
        out = str(tmp_path / "ds")
        export.to_deepspeed(plain_state(model, optimizer), out)
        global_state = load(os.path.join(tag_dir(out), "zero", "optimizer_state.pt"))

        for key in ("clip_grad", "loss_scaler", "dynamic_loss_scale", "overflow"):
            assert key not in global_state
        # DeepSpeed parses it as a version and compares against it.
        Version(global_state["ds_version"])
        group = global_state["param_groups"][0]
        assert group["lr"] == 0.1 and "param_names" not in group
        assert global_state["optimizer_state_dict"]["state"][0]["step"] == 3


class TestWhereTheNamesComeFrom:
    def test_a_gathered_sharded_group_is_keyed_by_name_already(self, tmp_path):
        model, optimizer = trained()
        by_position = optimizer.state_dict()
        names = [name for name, _ in model.named_parameters()]
        state = {
            "step": 3,
            "sharded": {"model": {
                "layout": "gather",
                "model": model.state_dict(),
                "optimizer": {
                    "state": {names[i]: entry for i, entry in by_position["state"].items()},
                    "param_groups": [dict(by_position["param_groups"][0], params=names)],
                },
            }},
        }
        out = str(tmp_path / "ds")
        export.to_deepspeed(state, out)
        got = load(os.path.join(tag_dir(out), "zero", names[0], "exp_avg.pt"))["param"]
        assert torch.equal(got, by_position["state"][0]["exp_avg"])

    def test_by_position_without_names_the_moments_are_left_out_and_said_so(self, tmp_path):
        model, optimizer = trained(named=False)
        out = str(tmp_path / "ds")
        notes = export.to_deepspeed(plain_state(model, optimizer), out)

        folder = os.path.join(tag_dir(out), "zero", "0.weight")
        assert sorted(os.listdir(folder)) == ["fp32.pt", "step.pt"]
        assert any("fresh optimizer" in note for note in notes)

    def test_an_optimizer_naming_a_parameter_the_model_lacks_is_refused(self, tmp_path):
        model, optimizer = trained()
        state = plain_state(model, optimizer)
        state["optimizers"]["optimizer"]["param_groups"][0]["param_names"][0] = "elsewhere.weight"
        with pytest.raises(CannotConvert, match="elsewhere.weight"):
            export.to_deepspeed(state, str(tmp_path / "ds"))

    def test_without_an_optimizer_buffers_are_told_from_weights_by_dtype(self, tmp_path):
        model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
        out = str(tmp_path / "ds")
        export.to_deepspeed({"step": 1, "models": {"model": model.state_dict()}}, out)
        states = load(os.path.join(tag_dir(out), "mp_rank_00_model_states.pt"))
        # The integer counter is a buffer; a float running statistic cannot be
        # told from a weight without an optimizer to ask, and is said to be one.
        assert "1.num_batches_tracked" in states["buffer_names"]
        assert "1.running_mean" in states["param_shapes"][0]


class TestTheCommand:
    def test_it_exports_a_real_store(self, storage, tmp_path, capsys):
        from ravex._cli import main

        @ravex.train_loop(backend="torch_save", checkpoint_every=3)
        def train():
            model, optimizer = build(), None
            optimizer = torch.optim.Adam(model.named_parameters(), lr=0.1)
            for _ in range(6):
                model(torch.randn(8, 4)).sum().backward()
                optimizer.step()
                optimizer.zero_grad()

        train()
        out = tmp_path / "ds"
        code = main([
            "export", "--storage", str(storage), "--backend", "torch_save",
            "--out", str(out), "--to", "deepspeed",
        ])
        said = capsys.readouterr()

        assert code == 0, said.err
        assert "exported step 6" in said.out and "as deepspeed" in said.out
        assert "load_universal" in said.out
        root = tag_dir(str(out))
        assert os.path.exists(os.path.join(root, "zero", "2.weight", "exp_avg_sq.pt"))
