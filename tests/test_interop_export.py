"""Exporting a Ravex checkpoint as a torch distributed checkpoint — GPU-90.

The reading half of GPU-90 is tested against the readers of other frameworks'
formats. The writing half is tested against Ravex's own reader of DCP, and that
is not circular: ``ravex._interop.dcp.read`` is torch's loader with a stable
entry point in front of it, so what it reads is what a DCP consumer reads. The
strongest test here goes all the way round — train, export, and resume a fresh
run from the export through ``convert_foreign`` — because an export is only
worth something if the thing on the other side can start from it.
"""

import pytest
import torch
import torch.nn as nn

import ravex
from ravex._interop import dcp as dcp_reader
from ravex._interop import export
from ravex._interop.convert import CannotConvert, _unify_dcp

pytest.importorskip("torch.distributed.checkpoint")

BACKENDS = ["torch_save", pytest.param("moonclip", id="moonclip")]


def build():
    return nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 1))


def train_a_store(backend, steps=6):
    """Six Adam steps, checkpointed at 3 and 6. Returns the final weights."""

    @ravex.train_loop(backend=backend, checkpoint_every=3)
    def train():
        torch.manual_seed(0)
        model = build()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(steps):
            model(torch.randn(8, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        return {name: t.detach().clone() for name, t in model.state_dict().items()}

    return train()


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "moonclip":
        pytest.importorskip("moonclip")
    return request.param


class TestWhatComesOut:
    def test_the_weights_read_back_out_of_dcp_exactly(self, storage, tmp_path, backend):
        final = train_a_store(backend)
        out = str(tmp_path / "dcp")

        export.to_dcp(export.load_store(str(storage), backend=backend), out)
        read = dcp_reader.read(out)

        assert set(read["model"]) == set(final)
        for name, tensor in final.items():
            assert torch.equal(read["model"][name], tensor), name
        # Everything the directory's own metadata promises was read.
        assert dcp_reader.missing_from(read, dcp_reader.describe(out)) == []
        assert read["ravex"]["step"] == 6

    def test_a_plain_models_optimizer_leaves_keyed_by_position_and_says_so(
        self, storage, tmp_path
    ):
        train_a_store("torch_save")
        out = str(tmp_path / "dcp")

        notes = export.to_dcp(export.load_store(str(storage), backend="torch_save"), out)
        unified = _unify_dcp(dcp_reader.read(out))

        assert unified["optimizer"]["state"], "the moments were not exported"
        assert unified["optimizer_keyed_by"] == "position"
        assert any("keyed by position" in note for note in notes)

    def test_a_gathered_sharded_group_leaves_keyed_by_name(self, tmp_path):
        moment = torch.full((2, 2), 0.5)
        state = {
            "step": 5,
            "sharded": {
                "group": {
                    "model": {"layer.weight": torch.ones(2, 2)},
                    "optimizer": {
                        "state": {"layer.weight": {"exp_avg": moment}},
                        "param_groups": [{"lr": 0.1, "params": ["layer.weight"]}],
                    },
                    "layout": "gather",
                }
            },
        }
        out = str(tmp_path / "dcp")

        export.to_dcp(state, out)
        unified = _unify_dcp(dcp_reader.read(out))

        assert unified["optimizer_keyed_by"] == "name"
        assert torch.equal(unified["optimizer"]["state"]["layer.weight"]["exp_avg"], moment)


class TestTheOtherSide:
    def test_a_fresh_run_resumes_from_the_export(self, storage, tmp_path, monkeypatch):
        """Train, export, and start again from the export as a foreign checkpoint.

        The resume lands on the first optimizer step (there is no DataLoader),
        after that step's update — so right after it the model must hold the
        exported weights, not the fresh ones and not one step past them.
        """
        final = train_a_store("torch_save")
        out = tmp_path / "dcp"
        export.to_dcp(export.load_store(str(storage), backend="torch_save"), str(out))

        monkeypatch.setenv("RAVEX_STORAGE_PATH", str(out))
        monkeypatch.setattr("ravex._frameworks.detect_framework", lambda: "vanilla")
        resumed = {}

        @ravex.train_loop(backend="torch_save", checkpoint_every=100, convert_foreign=True)
        def resume():
            torch.manual_seed(1)
            model = build()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            model(torch.randn(8, 4)).sum().backward()
            optimizer.step()
            resumed.update(
                {name: t.detach().clone() for name, t in model.state_dict().items()}
            )

        resume()
        for name, tensor in final.items():
            assert torch.equal(resumed[name], tensor), name


class TestRefusals:
    def test_a_per_rank_shard_is_refused_with_the_reason(self, tmp_path):
        state = {
            "sharded": {
                "group": {
                    "model": {"w": torch.zeros(2)},
                    "optimizer": {},
                    "layout": "per_rank",
                    "rank": 1,
                    "world_size": 4,
                }
            }
        }
        with pytest.raises(CannotConvert, match="rank 1 of 4"):
            export.to_dcp(state, str(tmp_path / "out"))
        assert not (tmp_path / "out").exists(), "wrote something before refusing"

    def test_several_models_need_a_name(self, tmp_path):
        state = {
            "models": {"a": {"w": torch.zeros(1)}, "b": {"w": torch.ones(1)}},
            "optimizers": {},
        }
        with pytest.raises(CannotConvert, match="holds 2 models"):
            export.to_dcp(state, str(tmp_path / "refused"))

        out = str(tmp_path / "b")
        export.to_dcp(state, out, key="b")
        assert torch.equal(dcp_reader.read(out)["model"]["w"], torch.ones(1))

    def test_a_name_that_is_not_there_lists_what_is(self, tmp_path):
        state = {"models": {"a": {"w": torch.zeros(1)}}, "optimizers": {}}
        with pytest.raises(CannotConvert, match="it holds a"):
            export.to_dcp(state, str(tmp_path / "out"), key="nope")

    def test_an_empty_store_is_refused_rather_than_exported_as_nothing(self, tmp_path):
        with pytest.raises(CannotConvert, match="no checkpoint"):
            export.load_store(str(tmp_path / "empty"), backend="torch_save")


class TestTheCommand:
    def test_it_exports_and_says_how_exactly(self, storage, tmp_path, capsys):
        from ravex._cli import main

        train_a_store("torch_save")
        out = tmp_path / "dcp"
        code = main(
            ["export", "--storage", str(storage), "--backend", "torch_save", "--out", str(out)]
        )
        said = capsys.readouterr()

        assert code == 0, said.err
        assert "exported step 6" in said.out
        assert "keyed by position" in said.out
        assert (out / ".metadata").exists()

    def test_it_will_not_write_into_a_directory_with_something_in_it(
        self, storage, tmp_path, capsys
    ):
        from ravex._cli import main

        train_a_store("torch_save")
        out = tmp_path / "dcp"
        out.mkdir()
        (out / "leftover").write_text("from an earlier export")

        code = main(
            ["export", "--storage", str(storage), "--backend", "torch_save", "--out", str(out)]
        )
        assert code == 2
        assert "not empty" in capsys.readouterr().err
        assert sorted(p.name for p in out.iterdir()) == ["leftover"]

    def test_an_empty_store_is_an_error_not_an_empty_export(self, tmp_path, capsys):
        from ravex._cli import main

        code = main(
            [
                "export",
                "--storage", str(tmp_path / "nothing"),
                "--backend", "torch_save",
                "--out", str(tmp_path / "dcp"),
            ]
        )
        assert code == 2
        assert "no checkpoint" in capsys.readouterr().err
