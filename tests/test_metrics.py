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


def chunk_files(storage):
    """Every finished chunk of every segment, in order."""
    return sorted((storage / "metrics").glob("*/*.jsonl"))


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
        text = "".join(path.read_text(encoding="utf-8") for path in chunk_files(storage))
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

    def test_a_run_shorter_than_the_interval_still_samples_once(self, storage, monkeypatch):
        """Otherwise a quick run has an empty system chart, which reads as a fault."""
        monkeypatch.setattr(_metrics, "system_samplers", lambda: [lambda: {"sys/cpu_percent": 3.0}])

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, system_metrics_every=3600)
        def train():
            loop(1, lambda _s, _m: None)

        train()
        system = ravex.metrics.read(str(storage))["system"]
        (host,) = system
        assert system[host]["sys/cpu_percent"]["value"] == [3.0]

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

    def test_a_chunk_still_being_written_is_not_read(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(2, lambda step, _m: ravex.log_metrics({"loss": float(step)}))

        train()
        (segment,) = (storage / "metrics").iterdir()
        (segment / "000099.jsonl.tmp").write_text(
            '{"step": 3, "time": 1, "values": {"loss": 3.0}}\n', encoding="utf-8"
        )
        assert ravex.metrics.read(str(storage))["scalars"]["loss"]["step"] == [1, 2]

    def test_a_torn_line_is_skipped(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            loop(2, lambda step, _m: ravex.log_metrics({"loss": float(step)}))

        train()
        last = chunk_files(storage)[-1]
        with open(last, "a", encoding="utf-8") as handle:
            handle.write('{"step": 3, "time": 1, "val')
        assert ravex.metrics.read(str(storage))["scalars"]["loss"]["step"] == [1, 2]

    def test_a_segment_without_its_header_cuts_nothing(self):
        """A header that has not reached the bucket yet: no resume step to cut at."""
        chunks = {
            "a/000000.jsonl": json.dumps({"segment": "a", "rank": 0, "start_step": 0, "time": 1.0}),
            "a/000001.jsonl": json.dumps({"step": 5, "time": 1.0, "values": {"loss": 5.0}}),
            "b/000001.jsonl": json.dumps({"step": 3, "time": 2.0, "values": {"loss": 30.0}}),
        }
        loss = ravex.metrics.resolve(chunks)["scalars"]["loss"]
        assert loss["step"] == [5]

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
        (segment,) = os.listdir(tmp_path / "metrics")
        chunk = tmp_path / "metrics" / segment / "000001.jsonl"
        (line,) = chunk.read_text(encoding="utf-8").splitlines()
        assert json.loads(line) == {"step": 1, "time": 0.0, "values": {"a": 2.0}}


class TestChunks:
    """Why a segment is a directory of files written once (GPU-152).

    Moonclip's sync skips a file the bucket already holds by name. A file that
    kept growing would go up once and never again, so a dashboard reading the
    bucket would see the first second of every run and nothing after.
    """

    def test_every_chunk_is_handed_to_the_uploader_once_it_is_whole(self, tmp_path):
        seen = []

        def uploader(relative):
            # Called after the rename: the file is already there, complete.
            assert (tmp_path / relative).is_file()
            seen.append(relative)

        writer = _metrics.MetricsWriter(
            str(tmp_path), rank=0, start_step=0, chunk_every=0.0, uploader=uploader
        )
        writer.put(1, 0.0, {"a": _metrics.Scalar(1.0)})
        writer.put(2, 0.0, {"a": _metrics.Scalar(2.0)})
        writer.close()
        segment = "metrics/" + writer.segment
        assert seen[0] == segment + "/000000.jsonl"
        assert len(seen) >= 2
        assert all(path.startswith(segment + "/") and path.endswith(".jsonl") for path in seen)
        assert len(set(seen)) == len(seen)

    def test_a_chunk_is_closed_on_the_clock_not_only_at_the_end(self, tmp_path):
        import time

        writer = _metrics.MetricsWriter(str(tmp_path), rank=0, start_step=0, chunk_every=0.2)
        writer.put(1, 0.0, {"a": _metrics.Scalar(1.0)})
        time.sleep(1.5)
        # Still open, and the record is already readable.
        history = ravex.metrics.read(str(tmp_path))
        writer.close()
        assert history["scalars"]["a"]["step"] == [1]

    def test_an_upload_that_fails_does_not_stop_the_local_copy(self, tmp_path):
        def uploader(_relative):
            raise RuntimeError("syncer is gone")

        writer = _metrics.MetricsWriter(
            str(tmp_path), rank=0, start_step=0, chunk_every=0.0, uploader=uploader
        )
        writer.put(1, 0.0, {"a": _metrics.Scalar(1.0)})
        writer.close()
        assert ravex.metrics.read(str(tmp_path))["scalars"]["a"]["value"] == [1.0]

    def test_resolve_reads_bucket_keys_the_way_read_reads_a_directory(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=5, checkpoint_on_exit=False)
        def first():
            loop(8, lambda step, _m: ravex.log_metrics({"loss": float(step)}))
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            first()

        @ravex.train_loop(backend="torch_save", checkpoint_every=5)
        def second():
            loop(3, lambda step, _m: ravex.log_metrics({"loss": step + 100.0}))

        second()
        # As a Worker would list them: full object keys, forward slashes.
        root = storage / "metrics"
        chunks = {
            "run-7/metrics/" + path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in chunk_files(storage)
        }
        # The prefix above the store is the caller's to strip.
        chunks = {key[len("run-7/"):]: text for key, text in chunks.items()}
        from_bucket = ravex.metrics.resolve(chunks)
        from_disk = ravex.metrics.read(str(storage))
        assert from_bucket["scalars"] == from_disk["scalars"]
        assert from_bucket["scalars"]["loss"]["step"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_the_reader_imports_without_the_rust_core():
    """The dashboard runs on Python Workers, which cannot load a compiled module."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['ravex._core'] = None; sys.modules['torch'] = None\n"
        "import ravex.metrics\n"
        "print(ravex.metrics.resolve({})['scalars'])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "{}"
