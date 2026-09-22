"""Metrics sent to a backend, from a queue that lives on disk (GPU-156).

Ravex is a client of the platform, not its database: with
``metrics_endpoint`` set, the run document, its status, every metrics chunk
and the list of checkpoints are posted to it. Without one, nothing here runs
and a run is exactly what it was before - a store on a disk.

Two rules this file is built around.

**The training loop never waits for the network.** Everything is handed to a
thread of its own. What the loop does is note a path, the same call that
already queues a chunk for the bucket.

**An unreachable backend does not cost the metrics.** The chunks
:class:`ravex._metrics.MetricsWriter` writes *are* the queue: a chunk is on
disk before anybody tries to send it, and a segment's ``.shipped`` ledger
records which ones the backend has confirmed. A backend down for an hour means
a backlog that drains when it comes back; a process that dies with chunks
unsent leaves them for the next execution on the same store, which sends them
first. Nothing is deleted - the chunks are also what :func:`ravex.metrics.read`
and the bucket hold - so the queue cannot outgrow what the run already keeps.

**A resend is harmless.** Every batch is identified by (run, segment, chunk),
all chosen here, and the backend upserts on them: a timeout that arrives after
the backend did write the batch leads to the same rows written twice, not to
twice the points.

Only :mod:`urllib`: Ravex depends on PyYAML and nothing else, and one POST a
few seconds apart does not need a client library.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from ravex._metrics import HEADER_CHUNK, METRICS_DIR

logger = logging.getLogger("ravex")

#: Beside a segment's chunks: the names the backend has confirmed, one a line.
LEDGER = ".shipped"

#: Seconds a request may take before it counts as failed.
TIMEOUT = 10.0

#: Longest wait between two attempts while the backend is unreachable.
MAX_BACKOFF = 30.0

#: How long closing waits for a backlog that is still going out. A backend that
#: does not answer at all gets one more attempt, not this long: the rest stays
#: on disk for the next execution, and a run that has finished training should
#: not sit there because the dashboard is down.
CLOSE_SECONDS = 15.0


class _Rejected(Exception):
    """The backend answered, and the answer is no. Retrying will not change it."""


class _Unknown(Exception):
    """The backend does not know this run yet."""


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_lines(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                out.append(record)
    return out


class Shipper:
    """Posts a store's run document, metrics chunks and checkpoints to a backend.

    ``notify`` takes a path relative to the store - the same argument the
    bucket uploader takes, so the runtime hands both the same call:
    ``run.json`` or ``status.json`` mark the run document as changed, and
    ``metrics/<segment>/<chunk>.jsonl`` queues a chunk.

    ``recover`` queues what earlier executions on this store left unsent.
    Only one process per store should ask for it, or two would send the same
    backlog - harmless, since the backend is idempotent, but twice the work.
    """

    def __init__(
        self,
        endpoint: str,
        store: str,
        *,
        token: Optional[str] = None,
        store_uri: Optional[str] = None,
        recover: bool = False,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.store = store
        self.token = token or None
        self.store_uri = store_uri or os.path.abspath(store)
        self._chunks: Deque[str] = deque()
        self._run_dirty = False
        self._checkpoints: Optional[List[int]] = None
        self._headers: Dict[str, Dict[str, Any]] = {}
        self._condition = threading.Condition()
        self._closing = False
        self._deadline: Optional[float] = None
        self._unreachable_since: Optional[float] = None
        if recover:
            self._recover()
        self._thread = threading.Thread(target=self._run, name="ravex-ship", daemon=True)
        self._thread.start()

    # ─── what the runtime calls ───────────────────────────────────────

    def notify(self, relative: str) -> None:
        relative = relative.replace(os.sep, "/")
        with self._condition:
            if relative.startswith(METRICS_DIR + "/"):
                self._chunks.append(relative)
            else:
                self._run_dirty = True
            self._condition.notify()

    def checkpoints(self, steps: List[int]) -> None:
        """The steps the store holds now. Only the latest list is sent."""
        with self._condition:
            self._checkpoints = sorted(int(step) for step in steps)
            self._condition.notify()

    @property
    def pending(self) -> int:
        with self._condition:
            return len(self._chunks)

    def close(self, timeout: float = CLOSE_SECONDS) -> None:
        with self._condition:
            self._closing = True
            self._deadline = time.monotonic() + timeout
            self._condition.notify()
        self._thread.join(timeout + TIMEOUT)
        left = self.pending
        if left:
            logger.warning(
                "%d metrics chunk(s) did not reach %s; they stay in %s and go "
                "first the next time a run opens this store",
                left,
                self.endpoint,
                os.path.join(self.store, METRICS_DIR),
            )

    # ─── recovery ────────────────────────────────────────────────────

    def _recover(self) -> None:
        root = os.path.join(self.store, METRICS_DIR)
        try:
            segments = sorted(os.listdir(root))
        except OSError:
            return
        found = 0
        for segment in segments:
            directory = os.path.join(root, segment)
            if not os.path.isdir(directory):
                continue
            shipped = self._ledger(directory)
            try:
                names = sorted(n for n in os.listdir(directory) if n.endswith(".jsonl"))
            except OSError:
                continue
            for name in names:
                if name not in shipped:
                    self._chunks.append("%s/%s/%s" % (METRICS_DIR, segment, name))
                    found += 1
        if found:
            logger.info("%d metrics chunk(s) from earlier executions were never sent; sending them first", found)
        # Whatever the backend knew of this run may be stale, or nothing.
        self._run_dirty = True

    @staticmethod
    def _ledger(directory: str) -> Set[str]:
        try:
            with open(os.path.join(directory, LEDGER), "r", encoding="utf-8") as handle:
                return {line.strip() for line in handle if line.strip()}
        except OSError:
            return set()

    def _record_shipped(self, relative: str) -> None:
        directory, name = relative.rsplit("/", 1)
        path = os.path.join(self.store, *directory.split("/"), LEDGER)
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(name + "\n")
        except OSError as exc:
            # The chunk is sent; without the line it is sent again after a
            # restart, which the backend absorbs.
            logger.debug("Cannot record %s as sent: %s", relative, exc)

    # ─── the thread ──────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = 0.0
        while True:
            with self._condition:
                if backoff:
                    wait = backoff
                    if self._deadline is not None:
                        wait = min(wait, max(0.0, self._deadline - time.monotonic()))
                    self._condition.wait(wait)
                elif not self._has_work() and not self._closing:
                    self._condition.wait(1.0)
                if self._closing and (
                    not self._has_work()
                    or (self._deadline is not None and time.monotonic() >= self._deadline)
                ):
                    return
            try:
                self._send_all()
            except (_Unknown, OSError, urllib.error.URLError, ValueError) as exc:
                self._unreachable(exc)
                with self._condition:
                    if self._closing:
                        # One last try was made after the close. Waiting out a
                        # backend that is down would only hold the process
                        # open; the rest is on disk for the next execution.
                        return
                backoff = min(MAX_BACKOFF, max(1.0, backoff * 2))
                continue
            if self._unreachable_since is not None:
                logger.info(
                    "Backend %s reachable again after %.0fs; the backlog is going out",
                    self.endpoint,
                    time.monotonic() - self._unreachable_since,
                )
                self._unreachable_since = None
            backoff = 0.0

    def _has_work(self) -> bool:
        return bool(self._chunks) or self._run_dirty or self._checkpoints is not None

    def _unreachable(self, exc: BaseException) -> None:
        if self._unreachable_since is None:
            self._unreachable_since = time.monotonic()
            logger.warning(
                "Cannot reach the backend at %s (%s). Training goes on; the "
                "metrics wait on disk and go out when it answers",
                self.endpoint,
                exc,
            )

    def _send_all(self) -> None:
        with self._condition:
            run_dirty = self._run_dirty
            self._run_dirty = False
        if run_dirty:
            try:
                self._send_run()
            except BaseException:
                with self._condition:
                    self._run_dirty = True
                raise

        while True:
            with self._condition:
                if not self._chunks:
                    break
                relative = self._chunks[0]
            try:
                self._send_chunk(relative)
            except _Unknown:
                # Rank 0 has not described the run yet, or the backend lost
                # it. Describe it, then try the chunk again.
                with self._condition:
                    self._run_dirty = True
                raise
            except _Rejected as exc:
                logger.warning("The backend refused %s and it will not be resent: %s", relative, exc)
            self._record_shipped(relative)
            with self._condition:
                if self._chunks and self._chunks[0] == relative:
                    self._chunks.popleft()

        with self._condition:
            steps = self._checkpoints
            self._checkpoints = None
        if steps is not None:
            try:
                self._send_checkpoints(steps)
            except BaseException:
                with self._condition:
                    if self._checkpoints is None:
                        self._checkpoints = steps
                raise

    # ─── what goes over the wire ─────────────────────────────────────

    def _run_id(self) -> str:
        record = _read_json(os.path.join(self.store, "run.json"))
        run_id = record.get("run_id") if record else None
        if not run_id:
            # Not written yet: rank 0 writes it at the first step. The same
            # answer as a backend that does not know the run.
            raise _Unknown("run.json is not in %s yet" % self.store)
        return str(run_id)

    def _send_run(self) -> None:
        record = _read_json(os.path.join(self.store, "run.json"))
        if record is None:
            raise _Unknown("run.json is not in %s yet" % self.store)
        document = dict(record)
        document["status"] = _read_json(os.path.join(self.store, "status.json")) or {}
        document["store_uri"] = self.store_uri
        self._post("/api/ingest/runs", document)

    def _header(self, segment: str) -> Dict[str, Any]:
        header = self._headers.get(segment)
        if header is None:
            path = os.path.join(self.store, METRICS_DIR, segment, HEADER_CHUNK)
            lines = _read_lines(path)
            if not lines:
                raise ValueError("segment %s has no header" % segment)
            header = lines[0]
            self._headers[segment] = header
        return header

    def _send_chunk(self, relative: str) -> None:
        _, segment, name = relative.split("/", 2)
        header = self._header(segment)
        path = os.path.join(self.store, METRICS_DIR, segment, name)
        try:
            records = [] if name == HEADER_CHUNK else _read_lines(path)
        except FileNotFoundError:
            # Removed by hand, or by a reset of the store. Nothing to send.
            return
        run_id = urllib.parse.quote(self._run_id(), safe="")
        self._post(
            "/api/ingest/runs/%s/batches" % run_id,
            {"execution": header, "records": records, "batch": relative},
        )

    def _send_checkpoints(self, steps: List[int]) -> None:
        run_id = urllib.parse.quote(self._run_id(), safe="")
        self._post(
            "/api/ingest/runs/%s/checkpoints" % run_id,
            {"checkpoints": [{"step": step} for step in steps]},
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                text = response.read().decode("utf-8") or "{}"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 404:
                raise _Unknown(detail) from None
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise _Rejected("%d %s" % (exc.code, detail)) from None
            raise
        try:
            answer = json.loads(text)
        except ValueError:
            return {}
        return answer if isinstance(answer, dict) else {}


def combine(*uploaders: Any) -> Any:
    """One uploader that calls every one given, or None if none was."""
    present = [uploader for uploader in uploaders if uploader is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    def both(relative: str) -> None:
        failure: Optional[BaseException] = None
        for uploader in present:
            try:
                uploader(relative)
            except Exception as exc:  # the others still get their turn
                failure = exc
        if failure is not None:
            raise failure

    return both


def describe_store(storage: Any) -> Tuple[str, str]:
    """(local path, uri) of a store: where its files are, and what to call it."""
    path = os.path.abspath(storage.path)
    if getattr(storage, "is_remote", False) and getattr(storage, "bucket", None):
        prefix = (getattr(storage, "prefix", "") or "").strip("/")
        return path, "s3://%s/%s" % (storage.bucket, prefix) if prefix else "s3://%s" % storage.bucket
    return path, path
