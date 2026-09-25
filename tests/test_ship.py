"""A run sent to the platform's backend while it trains (GPU-156).

The backend here is a small HTTP server in the test process that records what
it is sent, and can be told to be down. What these tests hold Ravex to is the
issue's two rules: training never waits for the network, and a backend that
was unreachable costs nothing once it answers again.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

import ravex
import ravex.runs
from ravex._metrics import METRICS_DIR
from ravex._ship import LEDGER, Shipper


class FakeBackend:
    def __init__(self):
        self.down = False
        self.runs = {}
        self.batches = []
        self.checkpoints = {}
        self.lock = threading.Lock()
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if backend.down:
                    self.send_response(503)
                    self.end_headers()
                    return
                parts = self.path.strip("/").split("/")
                with backend.lock:
                    if parts == ["api", "ingest", "runs"]:
                        backend.runs[body["run_id"]] = body
                        status = 200
                    elif parts[3] not in backend.runs:
                        status = 404
                    elif parts[4] == "batches":
                        backend.batches.append((parts[3], body))
                        status = 200
                    else:
                        backend.checkpoints[parts[3]] = [c["step"] for c in body["checkpoints"]]
                        status = 200
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def points(self, run_id, name):
        with self.lock:
            out = []
            for batch_run, body in self.batches:
                if batch_run != run_id:
                    continue
                for record in body["records"]:
                    if name in record["values"]:
                        out.append(record["step"])
            return out

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def backend():
    server = FakeBackend()
    yield server
    server.close()


def tiny(steps, log=True):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    for _ in range(steps):
        ravex.batch_boundary()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if log:
            ravex.log_metrics({"train/loss": loss})


def shipped(storage):
    names = set()
    root = os.path.join(str(storage), METRICS_DIR)
    for segment in os.listdir(root):
        ledger = os.path.join(root, segment, LEDGER)
        if os.path.exists(ledger):
            with open(ledger, encoding="utf-8") as handle:
                names.update("%s/%s" % (segment, line.strip()) for line in handle if line.strip())
    return names


def written(storage):
    root = os.path.join(str(storage), METRICS_DIR)
    return {
        "%s/%s" % (segment, name)
        for segment in os.listdir(root)
        for name in os.listdir(os.path.join(root, segment))
        if name.endswith(".jsonl")
    }


class TestSending:
    def test_a_run_arrives_whole(self, storage, backend):
        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=2,
            metrics_chunk_every=0,
            system_metrics_every=0,
            metrics_endpoint=backend.url,
        )
        def train():
            tiny(5)

        train()
        run_id = ravex.runs.describe(str(storage))["run_id"]
        document = backend.runs[run_id]
        assert document["status"]["state"] == "finished"
        assert document["store_uri"] == os.path.abspath(str(storage))
        assert backend.points(run_id, "train/loss") == [1, 2, 3, 4, 5]
        assert backend.checkpoints[run_id] == [2, 4, 5]
        # Every chunk written is recorded as confirmed, so a restart resends nothing.
        assert shipped(storage) == written(storage)

    def test_the_token_never_reaches_the_document(self, storage, backend):
        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=10_000,
            metrics_endpoint=backend.url,
            metrics_token="s3cret",
        )
        def train():
            tiny(1)

        train()
        with open(os.path.join(str(storage), "run.json"), encoding="utf-8") as handle:
            assert "s3cret" not in handle.read()
        (document,) = backend.runs.values()
        assert "s3cret" not in json.dumps(document)

    def test_a_crash_is_not_reported_as_finished(self, storage, backend):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, metrics_endpoint=backend.url)
        def train():
            tiny(2)
            raise RuntimeError("the loss went NaN and the script gave up")

        with pytest.raises(RuntimeError):
            train()
        (document,) = backend.runs.values()
        assert document["status"]["state"] == "failed"
        assert ravex.runs.describe(str(storage))["status"]["state"] == "failed"

    def test_without_an_endpoint_nothing_is_sent(self, storage, backend):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            tiny(2)

        train()
        assert backend.runs == {} and backend.batches == []


class TestUnreachable:
    def test_training_does_not_wait_for_a_backend_that_is_down(self, storage, backend):
        backend.down = True

        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=10_000,
            metrics_chunk_every=0,
            system_metrics_every=0,
            metrics_endpoint=backend.url,
        )
        def train():
            started = time.monotonic()
            tiny(20)
            return time.monotonic() - started

        # The loop itself: a request timing out inside a step would show here.
        assert train() < 5.0
        assert backend.batches == []
        # And nothing was lost: every chunk is still on disk, none marked sent.
        assert written(storage) and not shipped(storage)

    def test_what_was_left_unsent_goes_first_next_time(self, storage, backend):
        backend.down = True

        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=5,
            metrics_chunk_every=0,
            system_metrics_every=0,
            metrics_endpoint=backend.url,
        )
        def first():
            tiny(5)

        first()
        run_id = ravex.runs.describe(str(storage))["run_id"]
        assert backend.points(run_id, "train/loss") == []

        backend.down = False

        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=5,
            metrics_chunk_every=0,
            system_metrics_every=0,
            metrics_endpoint=backend.url,
        )
        def second():
            tiny(3)

        second()
        # The first execution's points arrived, once, and the second's after them.
        assert backend.points(run_id, "train/loss") == [1, 2, 3, 4, 5, 6, 7, 8]
        assert shipped(storage) == written(storage)

    def test_a_run_that_ended_while_the_backend_was_down_is_sent_by_ravex_ship(self, storage, backend):
        # What the platform's first end-to-end run hit: the backend went down
        # mid-run and came back as the run finished. Closing makes one more
        # attempt and leaves the rest on disk "for the next execution" - which a
        # finished run never gets, so the backend kept a run stuck at `running`
        # with four fifths of its points missing. `ravex ship` is that next
        # execution without the training.
        from ravex._cli import main

        backend.down = True

        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=10_000,
            metrics_chunk_every=0,
            system_metrics_every=0,
            metrics_endpoint=backend.url,
        )
        def train():
            tiny(6)

        train()
        run_id = ravex.runs.describe(str(storage))["run_id"]
        assert backend.points(run_id, "train/loss") == []

        backend.down = False
        assert main(["ship", "--storage", str(storage), "--endpoint", backend.url, "--wait", "10"]) == 0
        assert backend.points(run_id, "train/loss") == [1, 2, 3, 4, 5, 6]
        assert backend.runs[run_id]["status"]["state"] == "finished"
        # And what a resume can pick: the final checkpoint, read off the store.
        assert backend.checkpoints[run_id] == [6]
        assert shipped(storage) == written(storage)

        # Again, with nothing left: nothing is sent twice.
        assert main(["ship", "--storage", str(storage), "--endpoint", backend.url]) == 0
        assert backend.points(run_id, "train/loss") == [1, 2, 3, 4, 5, 6]

    def test_ravex_ship_says_so_when_the_backend_is_still_down(self, storage, backend, capsys):
        from ravex._cli import main

        backend.down = True

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, metrics_endpoint=backend.url)
        def train():
            tiny(2)

        train()
        assert main(["ship", "--storage", str(storage), "--endpoint", backend.url, "--wait", "2"]) == 1
        assert "did not reach" in capsys.readouterr().err
        # Nothing marked as sent, so the next attempt still has all of it.
        assert not shipped(storage)

    def test_ravex_ship_refuses_a_directory_that_is_not_a_store(self, tmp_path, backend):
        from ravex._cli import main

        assert main(["ship", "--storage", str(tmp_path), "--endpoint", backend.url]) == 2
        assert backend.runs == {}

    def test_a_backend_that_comes_back_gets_the_backlog(self, tmp_path, backend):
        store = tmp_path / "store"
        segment = store / METRICS_DIR / "0000000000001-host-1-r0-abcdef12"
        segment.mkdir(parents=True)
        (store / "run.json").write_text(json.dumps({"run_id": "r-1", "name": "n"}))
        header = {"segment": segment.name, "format": 1, "rank": 0, "start_step": 0, "time": 1.0}
        (segment / "000000.jsonl").write_text(json.dumps(header) + "\n")
        (segment / "000001.jsonl").write_text(
            "\n".join(json.dumps({"step": s, "time": 1.0, "values": {"x": s}}) for s in (1, 2)) + "\n"
        )

        backend.down = True
        shipper = Shipper(backend.url, str(store), recover=True)
        time.sleep(1.5)
        assert backend.batches == []
        backend.down = False
        shipper.close(timeout=40)
        assert backend.points("r-1", "x") == [1, 2]
        assert shipper.pending == 0


class TestTheStoresAddress:
    """What `ravex ship` calls a store it did not open itself (GPU-159).

    The agent ships a finished run from its local copy; for a run in a bucket,
    the address the platform keeps has to stay the bucket's, or a resume on
    another machine is sent to a path that went with the first one.
    """

    def write_run(self, store, storage):
        os.makedirs(store, exist_ok=True)
        with open(os.path.join(store, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"run_id": "r-1", "config": {"storage": storage}}, handle)

    def test_a_store_in_a_bucket_keeps_its_bucket_address(self, tmp_path):
        from ravex._ship import recorded_store_uri

        store = str(tmp_path / "harold-001")
        self.write_run(store, {"type": "r2", "bucket": "gpuzero", "prefix": "runs/harold-001", "path": store})
        assert recorded_store_uri(store) == "s3://gpuzero/runs/harold-001"
        assert Shipper("http://127.0.0.1:9", store).store_uri == "s3://gpuzero/runs/harold-001"

    def test_a_local_store_is_its_path(self, tmp_path):
        from ravex._ship import recorded_store_uri

        store = str(tmp_path / "local")
        self.write_run(store, {"type": "local", "path": store})
        assert recorded_store_uri(store) is None
        assert Shipper("http://127.0.0.1:9", store).store_uri == os.path.abspath(store)

    def test_an_address_given_wins(self, tmp_path):
        store = str(tmp_path / "harold-001")
        self.write_run(store, {"type": "r2", "bucket": "gpuzero", "prefix": "runs/harold-001"})
        assert Shipper("http://127.0.0.1:9", store, store_uri="s3://other/x").store_uri == "s3://other/x"
