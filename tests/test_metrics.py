"""`ravex.log_metrics` and `ravex.metrics.read` (GPU-147).

The case this file exists for is the resume. A run that dies after its last
checkpoint comes back at that checkpoint, and whatever it logged in between
describes a model that no longer exists. The reader has to hand back the
history of the model the store holds, not the concatenation of every execution
- that concatenation has two values for the same step, and a chart drawn from
it shows a loss that jumps backwards for no reason anyone can find.
"""

import json
import math
import os

import pytest
import torch

import ravex
import ravex.metrics
from ravex import _metrics


def loop(steps, log, start=0):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    for _ in range(steps):
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        log(ravex.step(), model)


class TestLogging:
    def test_scalars_land_on_the_optimizer_step(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(5, lambda step, _m: ravex.log_metrics({"train/loss": torch.tensor(step * 1.5)}))

        train()
        loss = ravex.metrics.read(str(storage))["scalars"]["train/loss"]
        assert loss["step"] == [1, 2, 3, 4, 5]
        assert loss["value"] == [1.5, 3.0, 4.5, 6.0, 7.5]

    def test_an_explicit_step_wins(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(3, lambda step, _m: None)
            ravex.log_metrics({"eval/loss": 0.25}, step=1000)

        train()
        series = ravex.metrics.read(str(storage))["scalars"]["eval/loss"]
        assert series["step"] == [1000]
        assert series["value"] == [0.25]

    def test_a_tensor_is_a_histogram(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(1, lambda _s, _m: ravex.log_metrics({"w": torch.arange(100.0)}))

        train()
        (entry,) = ravex.metrics.read(str(storage))["histograms"]["w"]
        assert entry["step"] == 1
        assert entry["min"] == 0.0 and entry["max"] == 99.0
        assert len(entry["counts"]) == _metrics.HISTOGRAM_BINS
        assert sum(entry["counts"]) == 100
        assert entry["nonfinite"] == 0

    def test_a_nan_gets_its_own_count_and_leaves_the_range_alone(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            values = torch.tensor([1.0, 2.0, float("nan"), float("inf"), 3.0])
            ravex.log_metrics({"w": values, "loss": float("nan")}, step=1)

        train()
        history = ravex.metrics.read(str(storage))
        (entry,) = history["histograms"]["w"]
        assert (entry["min"], entry["max"]) == (1.0, 3.0)
        assert entry["nonfinite"] == 2
        assert sum(entry["counts"]) == 3
        assert math.isnan(history["scalars"]["loss"]["value"][0])

    def test_nan_is_written_as_json_a_browser_can_parse(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            ravex.log_metrics({"loss": float("inf")}, step=1)

        train()
        (header,) = ravex.metrics.segments(str(storage))
        with open(header["file"], encoding="utf-8") as handle:
            text = handle.read()
        assert "Infinity" not in text and "NaN" not in text
        assert '"inf"' in text

    def test_the_decorator_logs_what_the_function_returns(self, storage):
        @ravex.log_metrics
        def evaluate():
            return {"eval/acc": 0.5}

        @ravex.log_metrics(step=77)
        def evaluate_at():
            return {"eval/acc": 0.75}

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(2, lambda _s, _m: None)
            assert evaluate() == {"eval/acc": 0.5}
            evaluate_at()

        train()
        series = ravex.metrics.read(str(storage))["scalars"]["eval/acc"]
        assert series["step"] == [2, 77]
        assert series["value"] == [0.5, 0.75]

    def test_a_value_that_cannot_be_charted_raises_where_it_was_logged(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            with pytest.raises(TypeError, match="notes"):
                ravex.log_metrics({"notes": "hello"})
            with pytest.raises(TypeError):
                ravex.log_metrics([1.0, 2.0])

        train()

    def test_outside_a_run_it_is_dropped_not_raised(self, storage):
        ravex.log_metrics({"loss": 1.0})
        assert not (storage / "metrics").exists()

    def test_metrics_false_writes_nothing(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, metrics=False)
        def train():
            loop(3, lambda _s, _m: ravex.log_metrics({"loss": 1.0}))

        train()
        assert not (storage / "metrics").exists()

    def test_a_run_that_never_steps_keeps_what_it_logged(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            ravex.log_metrics({"eval/baseline": 0.1})

        train()
        assert ravex.metrics.read(str(storage))["scalars"]["eval/baseline"]["value"] == [0.1]


class TestAutomatic:
    def test_learning_rate_and_time_per_step(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, metrics_every=2)
        def train():
            loop(6, lambda _s, _m: None)

        train()
        scalars = ravex.metrics.read(str(storage))["scalars"]
        assert scalars["ravex/lr"]["step"] == [2, 4, 6]
        assert scalars["ravex/lr"]["value"] == [0.1, 0.1, 0.1]
        assert scalars["ravex/step_seconds"]["step"] == [2, 4, 6]
        assert all(value >= 0 for value in scalars["ravex/step_seconds"]["value"])

    def test_checkpoint_duration(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=2)
        def train():
            loop(4, lambda _s, _m: None)

        train()
        seconds = ravex.metrics.read(str(storage))["scalars"]["ravex/checkpoint_seconds"]
        assert seconds["step"] == [2, 4]

    def test_system_samples_go_to_the_machine_and_are_never_cut(self, storage, monkeypatch):
        monkeypatch.setattr(_metrics, "system_samplers", lambda: [lambda: {"sys/cpu_percent": 12.5}])

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, system_metrics_every=0.05)
        def train():
            import time

            loop(1, lambda _s, _m: None)
            time.sleep(0.3)

        train()
        system = ravex.metrics.read(str(storage))["system"]
        (host,) = system
        assert system[host]["sys/cpu_percent"]["value"][0] == 12.5


class TestTimeline:
    def test_a_resume_drops_what_the_abandoned_execution_logged(self, storage):
        """Killed at 8 with the last checkpoint at 5: steps 6-8 never happened."""

        @ravex.train_loop(backend="torch_save", checkpoint_every=5, checkpoint_on_exit=False)
        def first():
            loop(8, lambda step, _m: ravex.log_metrics({"loss": float(step)}))
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            first()

        @ravex.train_loop(backend="torch_save", checkpoint_every=5)
        def second():
            assert ravex.step() == 0  # nothing restored until the loop starts
            loop(5, lambda step, _m: ravex.log_metrics({"loss": step + 100.0}))

        second()
        history = ravex.metrics.read(str(storage))
        loss = history["scalars"]["loss"]
        assert loss["step"] == list(range(1, 11))
        assert loss["value"] == [1.0, 2.0, 3.0, 4.0, 5.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        first_segment, second_segment = history["segments"]
        assert second_segment["start_step"] == 5
        assert first_segment["cut_at"] == 5
        assert second_segment["cut_at"] is None

    def test_a_fresh_start_over_old_metrics_replaces_them(self, storage):
        """`resume: false` over a store with history: the history is not this run's."""

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def first():
            loop(3, lambda step, _m: ravex.log_metrics({"loss": float(step)}))

        first()

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, resume=False)
        def second():
            loop(2, lambda step, _m: ravex.log_metrics({"loss": step * 10.0}))

        second()
        loss = ravex.metrics.read(str(storage))["scalars"]["loss"]
        assert loss["step"] == [1, 2]
        assert loss["value"] == [10.0, 20.0]

    def test_a_torn_last_line_is_skipped(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(2, lambda step, _m: ravex.log_metrics({"loss": float(step)}))

        train()
        (header,) = ravex.metrics.segments(str(storage))
        with open(header["file"], "a", encoding="utf-8") as handle:
            handle.write('{"step": 3, "time": 1, "val')
        assert ravex.metrics.read(str(storage))["scalars"]["loss"]["step"] == [1, 2]

    def test_an_empty_store_reads_as_empty(self, tmp_path):
        history = ravex.metrics.read(str(tmp_path / "nothing"))
        assert history == {"scalars": {}, "histograms": {}, "system": {}, "segments": []}


class TestHistogramOnTheDevice:
    def test_a_large_tensor_is_sampled_not_indexed_whole(self, monkeypatch):
        monkeypatch.setattr(_metrics, "HISTOGRAM_MAX_ELEMENTS", 1000)
        prepared = _metrics.prepare("w", torch.ones(10_000))
        resolved = _metrics._resolve(prepared)["histogram"]
        assert resolved["stride"] == 10
        assert sum(resolved["counts"]) == 1000

    def test_a_constant_tensor_is_one_bar(self):
        resolved = _metrics._resolve(_metrics.prepare("w", torch.full((10,), 3.0)))["histogram"]
        assert resolved["min"] == resolved["max"] == 3.0
        assert resolved["counts"][0] == 10

    def test_a_scalar_is_cloned_not_aliased(self):
        total = torch.tensor(1.0)
        prepared = _metrics.prepare("total", total)
        total += 5
        assert _metrics._resolve(prepared) == 1.0

    def test_a_record_is_one_json_line(self, tmp_path):
        writer = _metrics.MetricsWriter(str(tmp_path), rank=0, start_step=0)
        writer.put(1, 0.0, {"a": _metrics.Scalar(2.0)})
        writer.close()
        (name,) = os.listdir(tmp_path / "metrics")
        lines = (tmp_path / "metrics" / name).read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[1]) == {"step": 1, "time": 0.0, "values": {"a": 2.0}}
