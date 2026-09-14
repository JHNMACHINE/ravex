"""The framework adapters that fill the GPU-69 seam.

What they exist to restore is not decoration. A framework that owns the loop
also owns the counter it stops on, and Ravex putting the weights, the moments
and the dataset position back underneath a fresh one gives a resumed run that
starts its step budget over — measured on 2026-09-14 at 28 steps for a 20-step
run, with nothing in the log to say so.

Two kinds of test, on purpose. The fakes below need neither transformers nor
Lightning, so the decisions an adapter makes — which object is the trainer,
what is progress and what is configuration, when to warn — are checked in the
job that does not install them. The real runs at the bottom are skipped there
and are the evidence that those decisions meet the framework they are about;
``integration/test_frameworks.py`` repeats that across a real SIGKILL.
"""

import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import ravex
import ravex._frameworks as frameworks


@pytest.fixture
def ravex_log(caplog, monkeypatch):
    """Ravex's records, exactly once each, whatever test ran before.

    A runtime sets ``propagate = False`` on the ``ravex`` logger and nothing
    sets it back, so whether ``caplog`` sees anything would depend on test
    order. So the handler is attached directly — and propagation is held off
    for the test, or in a process where no runtime has run yet every record
    arrives twice, once through each path.
    """
    logger = logging.getLogger("ravex")
    monkeypatch.setattr(logger, "propagate", False)
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="ravex")
    yield caplog
    logger.removeHandler(caplog.handler)


# ─── HuggingFace, against a stand-in ────────────────────────────────


def fake_trainer_state(**overrides):
    state = SimpleNamespace(
        global_step=0,
        epoch=0.0,
        log_history=[],
        best_metric=None,
        best_global_step=None,
        best_model_checkpoint=None,
        total_flos=0.0,
        num_input_tokens_seen=0,
        max_steps=20,
        num_train_epochs=5,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


@pytest.fixture
def fake_transformers(monkeypatch):
    """A ``transformers.trainer`` module whose ``Trainer`` records nothing."""
    module = types.ModuleType("transformers.trainer")

    class Trainer:
        def __init__(self, model=None, save_strategy="no", **state):
            self.model = model
            self.args = SimpleNamespace(save_strategy=save_strategy)
            self.state = fake_trainer_state(**state)

    module.Trainer = Trainer
    monkeypatch.setitem(sys.modules, "transformers.trainer", module)
    return module


def tracking(monkeypatch, *models):
    monkeypatch.setattr(
        frameworks, "_runtime_registry", lambda: SimpleNamespace(models=list(models))
    )


class TestHuggingFaceFindsItsTrainer:
    def test_a_plain_loop_over_a_hub_model_has_nothing_to_collect(self, monkeypatch):
        """``transformers`` imported, ``Trainer`` never: detection still says
        huggingface, and there is no loop state that is not already Ravex's."""
        monkeypatch.delitem(sys.modules, "transformers.trainer", raising=False)
        assert frameworks.HuggingFaceAdapter().collect_extra_state() == {}

    def test_one_trainer_is_the_trainer(self, fake_transformers):
        trainer = fake_transformers.Trainer(global_step=7)
        collected = frameworks.HuggingFaceAdapter().collect_extra_state()
        assert collected["trainer_state"]["global_step"] == 7
        del trainer

    def test_a_subclass_is_found_too(self, fake_transformers):
        """``SequentialTrainer(Trainer)`` is how most scripts customise it."""

        class Custom(fake_transformers.Trainer):
            pass

        trainer = Custom(global_step=3)
        collected = frameworks.HuggingFaceAdapter().collect_extra_state()
        assert collected["trainer_state"]["global_step"] == 3
        del trainer

    def test_of_several_the_one_training_a_tracked_model_wins(
        self, fake_transformers, monkeypatch
    ):
        model = torch.nn.Linear(2, 1)
        tracking(monkeypatch, model)
        evaluator = fake_transformers.Trainer(model=torch.nn.Linear(2, 1), global_step=99)
        trainer = fake_transformers.Trainer(model=model, global_step=5)

        collected = frameworks.HuggingFaceAdapter().collect_extra_state()
        assert collected["trainer_state"]["global_step"] == 5
        del evaluator, trainer

    def test_several_it_cannot_tell_apart_collect_nothing_and_say_so(
        self, fake_transformers, monkeypatch, ravex_log
    ):
        """Guessing would restore one run's step counter into another's loop."""
        tracking(monkeypatch)
        first = fake_transformers.Trainer(global_step=1)
        second = fake_transformers.Trainer(global_step=2)

        assert frameworks.HuggingFaceAdapter().collect_extra_state() == {}
        assert "could not tell which one is training" in ravex_log.text
        del first, second


class TestHuggingFaceState:
    def test_progress_is_collected_and_configuration_is_not(self, fake_transformers):
        trainer = fake_transformers.Trainer(global_step=12, epoch=2.5, best_metric=0.3)
        progress = frameworks.HuggingFaceAdapter().collect_extra_state()["trainer_state"]

        assert progress["global_step"] == 12
        assert progress["epoch"] == 2.5
        assert progress["best_metric"] == 0.3
        # Recomputed by `Trainer` from the resuming run's own arguments.
        assert "max_steps" not in progress
        assert "num_train_epochs" not in progress
        del trainer

    def test_what_is_collected_does_not_move_with_the_loop(self, fake_transformers):
        """The writer is a background thread; `log_history` keeps growing."""
        trainer = fake_transformers.Trainer(log_history=[{"loss": 1.0}])
        progress = frameworks.HuggingFaceAdapter().collect_extra_state()["trainer_state"]
        trainer.state.log_history.append({"loss": 0.5})

        assert progress["log_history"] == [{"loss": 1.0}]

    def test_restore_puts_progress_back_and_leaves_the_budget_alone(
        self, fake_transformers
    ):
        trainer = fake_transformers.Trainer(max_steps=30)
        frameworks.HuggingFaceAdapter().restore_extra_state(
            {"trainer_state": {"global_step": 8, "log_history": [{"loss": 2.0}]}}
        )

        assert trainer.state.global_step == 8
        assert trainer.state.log_history == [{"loss": 2.0}]
        assert trainer.state.max_steps == 30

    def test_a_best_checkpoint_that_no_longer_exists_is_forgotten(
        self, fake_transformers, tmp_path
    ):
        """What ``Trainer`` does itself when its own resume meets one."""
        trainer = fake_transformers.Trainer()
        frameworks.HuggingFaceAdapter().restore_extra_state(
            {"trainer_state": {"best_model_checkpoint": str(tmp_path / "gone")}}
        )
        assert trainer.state.best_model_checkpoint is None

    def test_progress_with_no_trainer_to_take_it_is_named(
        self, monkeypatch, ravex_log
    ):
        monkeypatch.delitem(sys.modules, "transformers.trainer", raising=False)
        frameworks.HuggingFaceAdapter().restore_extra_state(
            {"trainer_state": {"global_step": 8}}
        )
        assert "no Trainer was found" in ravex_log.text


class TestHuggingFaceDuplicateCheckpoints:
    def test_a_trainer_that_saves_too_is_warned_about_once(
        self, fake_transformers, ravex_log
    ):
        strategy = SimpleNamespace(value="steps")
        trainer = fake_transformers.Trainer(save_strategy=strategy)
        adapter = frameworks.HuggingFaceAdapter()
        adapter.collect_extra_state()
        adapter.collect_extra_state()

        assert ravex_log.text.count("checkpointing too") == 1
        assert 'save_strategy="no"' in ravex_log.text
        del trainer

    def test_save_strategy_no_is_silent(self, fake_transformers, ravex_log):
        trainer = fake_transformers.Trainer(save_strategy=SimpleNamespace(value="no"))
        frameworks.HuggingFaceAdapter().collect_extra_state()
        assert "checkpointing too" not in ravex_log.text
        del trainer


# ─── HuggingFace, for real ──────────────────────────────────────────


class Crash(RuntimeError):
    """Stands in for the machine going away, inside the framework's loop."""


def run_hf_trainer(workdir, die_at=0, epochs=5):
    """A decorated ``Trainer`` run: 8 rows, batch 2, so 4 steps an epoch.

    Returns the steps this call trained and the ``TrainerState`` it ended on.
    """
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    trained = []

    class Rows(Dataset):
        def __init__(self):
            generator = torch.Generator().manual_seed(0)
            self.x = torch.randn(8, 2, generator=generator)
            self.y = torch.randn(8, 1, generator=generator)

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return {"x": self.x[index], "labels": self.y[index]}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x, labels=None):
            if self.training:
                trained.append(1)
            return {"loss": ((self.linear(x) - labels) ** 2).mean()}

    class DieAt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if die_at and state.global_step >= die_at:
                raise Crash(state.global_step)

    @ravex.train_loop(backend="torch_save", checkpoint_every=4)
    def train():
        arguments = TrainingArguments(
            output_dir=str(workdir / "hf-output"),
            per_device_train_batch_size=2,
            num_train_epochs=epochs,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
            dataloader_num_workers=0,
        )
        trainer = Trainer(model=Model(), args=arguments, train_dataset=Rows(), callbacks=[DieAt()])
        try:
            trainer.train()
        except Crash:
            pass
        return trainer.state

    state = train()
    return len(trained), state


class TestHuggingFaceTrainerForReal:
    @pytest.fixture(autouse=True)
    def detected_as_huggingface(self, monkeypatch):
        """Detection reads ``sys.modules``, and whether an earlier test in this
        process imported Lightning is not something these tests should depend
        on. Detection itself is tested on its own, below."""
        monkeypatch.setattr(frameworks, "detect_framework", lambda: "huggingface")

    def test_a_resumed_run_keeps_its_step_budget(self, storage, tmp_path):
        pytest.importorskip("transformers")
        pytest.importorskip("accelerate")

        trained, _ = run_hf_trainer(tmp_path, die_at=10)
        assert trained == 10

        trained, state = run_hf_trainer(tmp_path)
        assert state.global_step == 20
        assert trained == 10, "the resumed run started its step budget over"

    def test_without_the_adapter_the_budget_starts_over(
        self, storage, tmp_path, monkeypatch
    ):
        """The premise, pinned. If this ever trains exactly what was left, the
        adapter is no longer what keeps the budget and the reasoning above needs
        rereading.

        "More than was left" rather than a number. With the weights, the
        optimizer and the dataset position restored underneath a `TrainerState`
        counting from zero, this measured 17 steps for the 10 that remained;
        the exact figure is how the loop's fresh counter meets the restored
        position, and none of it is what this test is about.
        """
        pytest.importorskip("transformers")
        pytest.importorskip("accelerate")
        monkeypatch.setattr(
            frameworks, "get_adapter", lambda framework: frameworks.FrameworkAdapter()
        )

        run_hf_trainer(tmp_path, die_at=10)
        trained, _ = run_hf_trainer(tmp_path)
        assert trained > 10, "without the adapter the budget no longer starts over"


# ─── Detection ──────────────────────────────────────────────────────


class TestDetection:
    def test_lightning_wins_when_both_are_imported(self, monkeypatch):
        """A Lightning run fine-tuning a hub model imports transformers too, and
        Lightning is the one that owns the loop."""
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
        monkeypatch.setitem(sys.modules, "lightning", types.ModuleType("lightning"))
        assert frameworks.detect_framework() == "lightning"

    def test_transformers_alone_is_huggingface(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
        monkeypatch.delitem(sys.modules, "lightning", raising=False)
        monkeypatch.delitem(sys.modules, "pytorch_lightning", raising=False)
        assert frameworks.detect_framework() == "huggingface"

    def test_the_old_import_name_is_lightning_too(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "lightning", raising=False)
        monkeypatch.setitem(
            sys.modules, "pytorch_lightning", types.ModuleType("pytorch_lightning")
        )
        assert frameworks.detect_framework() == "lightning"


# ─── Lightning, against a stand-in ──────────────────────────────────


class FakeLoop:
    def __init__(self, state):
        self.state = state
        self.loaded = []

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.loaded.append(state)


def fake_lightning_trainer(checkpoint_callback=None, **state):
    return SimpleNamespace(
        fit_loop=FakeLoop(state), checkpoint_callback=checkpoint_callback
    )


def fake_lightning_module(trainer):
    """A module with a ``trainer``, which is all the adapter looks at."""
    module = torch.nn.Linear(2, 1)
    module.trainer = trainer
    return module


class TestLightningFindsItsTrainer:
    def test_through_the_module_it_is_fitting(self, monkeypatch):
        tracking(monkeypatch, fake_lightning_module(fake_lightning_trainer(step=4)))
        assert frameworks.LightningAdapter().collect_extra_state() == {
            "fit_loop": {"step": 4}
        }

    def test_a_level_down_inside_a_strategy_wrapper(self, monkeypatch):
        trainer = fake_lightning_trainer(step=4)
        tracking(monkeypatch, torch.nn.Sequential(fake_lightning_module(trainer)))
        assert frameworks.LightningAdapter().collect_extra_state()["fit_loop"] == {
            "step": 4
        }

    def test_a_plain_model_has_nothing_to_collect(self, monkeypatch):
        tracking(monkeypatch, torch.nn.Linear(2, 1))
        assert frameworks.LightningAdapter().collect_extra_state() == {}

    def test_a_shim_with_no_loops_is_not_a_trainer(self, monkeypatch):
        """What Fabric attaches in a Trainer's place."""
        tracking(monkeypatch, fake_lightning_module(SimpleNamespace()))
        assert frameworks.LightningAdapter().collect_extra_state() == {}

    def test_what_is_collected_does_not_move_with_the_loop(self, monkeypatch):
        trainer = fake_lightning_trainer(progress={"completed": 1})
        tracking(monkeypatch, fake_lightning_module(trainer))
        collected = frameworks.LightningAdapter().collect_extra_state()
        trainer.fit_loop.state["progress"]["completed"] = 2
        assert collected["fit_loop"]["progress"] == {"completed": 1}


class TestLightningState:
    def test_restore_hands_the_loop_its_own_state(self, monkeypatch):
        trainer = fake_lightning_trainer()
        tracking(monkeypatch, fake_lightning_module(trainer))
        frameworks.LightningAdapter().restore_extra_state({"fit_loop": {"step": 8}})
        assert trainer.fit_loop.loaded == [{"step": 8}]

    def test_loop_state_with_no_trainer_to_take_it_is_named(
        self, monkeypatch, ravex_log
    ):
        tracking(monkeypatch)
        frameworks.LightningAdapter().restore_extra_state({"fit_loop": {"step": 8}})
        assert "no Lightning Trainer was found" in ravex_log.text

    def test_model_checkpoint_on_is_warned_about_once(self, monkeypatch, ravex_log):
        trainer = fake_lightning_trainer(checkpoint_callback=object())
        tracking(monkeypatch, fake_lightning_module(trainer))
        adapter = frameworks.LightningAdapter()
        adapter.collect_extra_state()
        adapter.collect_extra_state()

        assert ravex_log.text.count("checkpointing too") == 1
        assert "enable_checkpointing=False" in ravex_log.text

    def test_model_checkpoint_off_is_silent(self, monkeypatch, ravex_log):
        tracking(monkeypatch, fake_lightning_module(fake_lightning_trainer()))
        frameworks.LightningAdapter().collect_extra_state()
        assert "checkpointing too" not in ravex_log.text


# ─── Lightning, for real ────────────────────────────────────────────


def run_lightning(die_at=0, epochs=5):
    """A decorated ``Trainer.fit``: 8 rows, batch 2, so 4 steps an epoch.

    Returns the steps this call trained, and ``global_step`` and
    ``current_epoch`` where it ended.
    """
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    trained = []

    class Regressor(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def training_step(self, batch, index):
            trained.append(1)
            x, y = batch
            return ((self.linear(x) - y) ** 2).mean()

        def on_train_batch_end(self, *args):
            if die_at and self.trainer.global_step >= die_at:
                raise Crash(self.trainer.global_step)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    @ravex.train_loop(backend="torch_save", checkpoint_every=4)
    def train():
        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(
            TensorDataset(
                torch.randn(8, 2, generator=generator),
                torch.randn(8, 1, generator=generator),
            ),
            batch_size=2,
        )
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        try:
            trainer.fit(Regressor(), loader)
        except Crash:
            pass
        return trainer.global_step, trainer.current_epoch

    global_step, epoch = train()
    return len(trained), global_step, epoch


class TestLightningForReal:
    @pytest.fixture(autouse=True)
    def detected_as_lightning(self, monkeypatch):
        monkeypatch.setattr(frameworks, "detect_framework", lambda: "lightning")

    def test_a_resumed_run_keeps_its_step_budget_and_its_epoch(self, storage):
        pytest.importorskip("lightning")

        trained, _, _ = run_lightning(die_at=10)
        assert trained == 10

        trained, global_step, epoch = run_lightning()
        assert (global_step, epoch) == (20, 5)
        assert trained == 10, "the resumed run started its step budget over"

    def test_without_the_adapter_the_budget_starts_over(self, storage, monkeypatch):
        pytest.importorskip("lightning")
        monkeypatch.setattr(
            frameworks, "get_adapter", lambda framework: frameworks.FrameworkAdapter()
        )

        run_lightning(die_at=10)
        trained, _, _ = run_lightning()
        assert trained > 10, "without the adapter the budget no longer starts over"
