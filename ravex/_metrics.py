"""Metrics a run logs, written into its store (GPU-147).

``ravex.log_metrics({"train/loss": loss})`` is the whole API, and what it asks
of the training loop is nothing that waits on the GPU. Reading a CUDA tensor
back to the host synchronises the stream: a logger that calls ``.item()`` on
the training thread turns every logged step into a stall, and the stall shows
up as a slower model, not as a slow logger. So the calling thread only queues
work the device does asynchronously — a ``detach().clone()`` for a scalar, a
min, a max and a ``bincount`` for a histogram — and a writer thread of its own
brings the results home and appends them to disk.

**Where they go.** ``<storage.path>/metrics/<segment>/``, one directory per
process per execution, written as numbered chunks (see :class:`MetricsWriter`).
A *segment* starts with a header that says which step it resumed from, and that header is what makes a run's history readable across a
resume: a run killed at step 700 with its last checkpoint at 500 comes back at
500, and the points its first execution logged between 501 and 700 belong to a
timeline that no longer exists. :func:`ravex.metrics.read` keeps each segment
only up to the step the next one resumed from. The reading rule lives in Ravex
rather than in whatever draws the chart, because it is a fact about resumes.

**Who writes.** Rank 0, for the metrics the training script logs: every rank
computes the same loss under data parallelism, or a meaningless partial one
under anything else, and eight copies of either help nobody. System metrics are
per machine, so the first process on each machine samples its own.

**A remote store** (GPU-152). With ``storage.type: s3`` the chunks are written
into the staging directory, and each one is handed to Moonclip's
``sync_prefix`` as it lands, so the bucket holds the metrics while the run is
still going. That is what a dashboard that cannot see the machine reads.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import socket
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("ravex")

#: The directory inside the store.
METRICS_DIR = "metrics"

#: Written into every segment header. A reader that meets a newer one says so
#: instead of guessing.
FORMAT_VERSION = 1

#: Bins per histogram. Fixed rather than configurable: a chart comparing two
#: runs needs them to agree, and 64 is enough to see a distribution's shape.
HISTOGRAM_BINS = 64

#: Beyond this many elements a histogram is taken over an evenly strided
#: sample. Binning needs an index per element, and for a billion-parameter
#: tensor that index would be 8 GB of device memory spent on a chart.
HISTOGRAM_MAX_ELEMENTS = 1 << 22

#: Records waiting for the writer. Past this the logger is outrunning the disk,
#: and dropping with a warning is better than growing until the host runs out
#: of memory holding tensors nobody will read in time.
QUEUE_LIMIT = 100_000

#: How long the writer waits before writing what it has. Also the most a crash
#: can lose.
FLUSH_SECONDS = 1.0


class Scalar:
    """A value on its way to becoming a float. ``pending`` may be a device tensor."""

    __slots__ = ("pending",)

    def __init__(self, pending: Any) -> None:
        self.pending = pending


class Histogram:
    """A distribution, reduced on the device where the tensor lives."""

    __slots__ = ("low", "high", "counts", "nonfinite", "stride")

    def __init__(self, low: Any, high: Any, counts: Any, nonfinite: Any, stride: int) -> None:
        self.low = low
        self.high = high
        self.counts = counts
        self.nonfinite = nonfinite
        self.stride = stride


def _is_tensor(value: Any) -> bool:
    # Without importing torch: Ravex does not depend on it at import time, and
    # a script that logs plain floats should not pay for it here.
    return type(value).__module__.startswith("torch") and hasattr(value, "detach")


def prepare(key: str, value: Any) -> Any:
    """Turn a logged value into a :class:`Scalar` or a :class:`Histogram`.

    Runs on the training thread, so it queues device work and never reads
    anything back. Raises ``TypeError`` for a value it cannot chart: a mistake
    in the script is better reported on its first call than turned into a
    metric that silently never appears.
    """
    if isinstance(value, bool):
        return Scalar(float(value))
    if isinstance(value, (int, float)):
        return Scalar(float(value))
    if _is_tensor(value):
        numel = value.numel()
        if numel == 0:
            raise ValueError("metric %r is an empty tensor; there is nothing to log" % key)
        if numel == 1:
            # `clone`, not just `detach`: a running total updated in place
            # (`total += loss`) would otherwise be read by the writer after the
            # next update, and log a value from a step that had not happened.
            return Scalar(value.detach().reshape(()).clone())
        return _histogram(value)
    if hasattr(value, "item") and getattr(value, "size", None) == 1:
        # A numpy scalar or a one-element array.
        return Scalar(float(value.item()))
    if hasattr(value, "__array__") or isinstance(value, (list, tuple)):
        import torch

        try:
            tensor = torch.as_tensor(value, dtype=torch.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "metric %r is a sequence that is not numeric: %s" % (key, exc)
            ) from None
        return prepare(key, tensor)
    raise TypeError(
        "metric %r is a %s; log_metrics takes numbers, tensors, and sequences "
        "or arrays of numbers" % (key, type(value).__name__)
    )


def _histogram(tensor: Any) -> Histogram:
    import torch

    flat = tensor.detach().reshape(-1)
    stride = 1
    if flat.numel() > HISTOGRAM_MAX_ELEMENTS:
        stride = -(-flat.numel() // HISTOGRAM_MAX_ELEMENTS)
        flat = flat[::stride]
    flat = flat.float()
    finite = torch.isfinite(flat)
    # The range over finite values only. Non-finite elements get a bin of
    # their own, past the last one, and are counted separately: a NaN that
    # stretched the range would flatten the whole histogram into one bar.
    low = torch.where(finite, flat, torch.full_like(flat, math.inf)).amin()
    high = torch.where(finite, flat, torch.full_like(flat, -math.inf)).amax()
    span = high - low
    span = torch.where(span > 0, span, torch.ones_like(span))
    position = torch.nan_to_num((flat - low) / span * HISTOGRAM_BINS, nan=0.0, posinf=0.0, neginf=0.0)
    index = position.floor().clamp(0, HISTOGRAM_BINS - 1).long()
    index = torch.where(finite, index, torch.full_like(index, HISTOGRAM_BINS))
    counts = torch.bincount(index, minlength=HISTOGRAM_BINS + 1)
    return Histogram(low, high, counts[:HISTOGRAM_BINS], counts[HISTOGRAM_BINS], stride)


def _number(value: float) -> Any:
    """A float as JSON can carry it. NaN and infinity become strings.

    Python's ``json`` writes them bare, which no browser's ``JSON.parse``
    accepts — and a loss that went NaN is exactly the point someone opens the
    chart to look at.
    """
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _resolve(value: Any) -> Any:
    """Bring a prepared value home. On the writer thread: this is where it waits."""
    if isinstance(value, Scalar):
        pending = value.pending
        return _number(float(pending.item() if hasattr(pending, "item") else pending))
    low = float(value.low.item())
    high = float(value.high.item())
    counts = [int(c) for c in value.counts.tolist()]
    nonfinite = int(value.nonfinite.item())
    histogram: Dict[str, Any] = {
        # Empty of finite values: the range is (inf, -inf) and means nothing.
        "min": low if math.isfinite(low) else None,
        "max": high if math.isfinite(high) else None,
        "counts": counts,
        "nonfinite": nonfinite,
    }
    if value.stride > 1:
        histogram["stride"] = value.stride
    return {"histogram": histogram}


def segment_id(started: float, host: str, pid: int, rank: int) -> str:
    """Sortable by start time, and unique even within one millisecond.

    The random tail is not decoration. Two runs of a decorated function in one
    process can start in the same millisecond - the test suite does it - and
    without it they got the same name, so the second execution was appended to
    the first one's file, under the first one's header, and its resume point
    was never seen by the reader.
    """
    safe_host = "".join(c if c.isalnum() or c in "-_." else "_" for c in host)
    return "%013d-%s-%d-r%d-%s" % (
        int(started * 1000),
        safe_host,
        pid,
        rank,
        uuid.uuid4().hex[:8],
    )


#: Name of a segment's first chunk, which holds only its header.
HEADER_CHUNK = "000000.jsonl"


def chunk_name(sequence: int) -> str:
    return "%06d.jsonl" % sequence


class MetricsWriter:
    """Writes one segment as a series of chunks, from a thread of its own.

    **Chunks, not one growing file**, because of where they end up. A bucket
    has no append, and Moonclip's sync skips a file the remote already holds
    by name: a file that kept growing would go up once, at whatever length it
    had, and never again. So a segment is a directory, ``000000.jsonl`` holds
    its header, and every ``chunk_every`` seconds what has been logged since
    becomes the next numbered file, written once and never touched after.

    Each chunk appears atomically - written beside its final name and renamed
    - so neither a reader nor the sync ever sees half of one. Once it is in
    place ``uploader`` is told its path relative to the store, which with a
    remote store is Moonclip's ``sync_prefix``.

    What that costs: a reader sees a record up to ``chunk_every`` seconds
    after it was logged, and a crash loses at most as much.
    """

    def __init__(
        self,
        directory: str,
        rank: int,
        start_step: int,
        chunk_every: float = 15.0,
        uploader: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.started = time.time()
        self.host = socket.gethostname()
        self.rank = rank
        self.segment = segment_id(self.started, self.host, os.getpid(), rank)
        self.directory = directory
        self.relative = METRICS_DIR + "/" + self.segment
        self.path = os.path.join(directory, METRICS_DIR, self.segment)
        self.chunk_every = max(float(chunk_every), 0.0)
        self._uploader = uploader
        self._header = {
            "segment": self.segment,
            "format": FORMAT_VERSION,
            "rank": rank,
            "host": self.host,
            "pid": os.getpid(),
            "start_step": start_step,
            "time": self.started,
        }
        self._sequence = 0
        self._queue: "queue.Queue[Optional[Tuple[int, float, Dict[str, Any]]]]" = queue.Queue(
            maxsize=QUEUE_LIMIT
        )
        self._dropped = 0
        self._failed = False
        self._upload_failed = False
        self._thread = threading.Thread(target=self._run, name="ravex-metrics", daemon=True)
        self._thread.start()

    def put(self, step: int, when: float, values: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((step, when, values))
        except queue.Full:
            if self._dropped == 0:
                logger.warning(
                    "Metrics are being logged faster than they can be written; "
                    "dropping them until the writer catches up"
                )
            self._dropped += 1

    def close(self, timeout: float = 30.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("Metrics writer did not finish within %.0fs", timeout)
        if self._dropped:
            logger.warning("%d metric record(s) were dropped", self._dropped)

    def _write_chunk(self, name: str, lines: List[str]) -> None:
        final = os.path.join(self.path, name)
        partial = final + ".tmp"
        try:
            # "x": a chunk is never written twice. Finding one already there
            # means two executions think they are the same one.
            with open(partial, "x", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            os.replace(partial, final)
        except OSError as exc:
            logger.warning("Cannot write metrics to %s: %s", final, exc)
            self._failed = True
            return
        if self._uploader is None:
            return
        try:
            self._uploader(self.relative + "/" + name)
        except Exception as exc:
            if not self._upload_failed:
                self._upload_failed = True
                logger.warning(
                    "Metrics are written locally but cannot be queued for the "
                    "remote store: %s",
                    exc,
                )

    def _run(self) -> None:
        try:
            os.makedirs(self.path, exist_ok=True)
        except OSError as exc:
            logger.warning("Cannot write metrics to %s: %s", self.path, exc)
            self._failed = True
        if not self._failed:
            self._write_chunk(HEADER_CHUNK, [json.dumps(self._header)])

        pending: List[str] = []
        opened = time.monotonic()
        done = False
        while not done:
            wait = max(0.05, min(FLUSH_SECONDS, opened + self.chunk_every - time.monotonic()))
            batch: List[Optional[Tuple[int, float, Dict[str, Any]]]] = []
            try:
                batch.append(self._queue.get(timeout=wait))
                while True:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            for record in batch:
                if record is None:
                    done = True
                    continue
                if self._failed:
                    continue
                step, when, values = record
                try:
                    resolved = {key: _resolve(value) for key, value in values.items()}
                except Exception as exc:  # a device error, a freed tensor
                    logger.warning("Dropping metrics at step %d: %s", step, exc)
                    continue
                pending.append(json.dumps({"step": step, "time": when, "values": resolved}))
            due = time.monotonic() - opened >= self.chunk_every
            if pending and not self._failed and (done or due):
                self._sequence += 1
                self._write_chunk(chunk_name(self._sequence), pending)
                pending = []
            if due:
                opened = time.monotonic()


# ─── system metrics ──────────────────────────────────────────────────


def _nvml_sampler() -> Optional[Callable[[], Dict[str, float]]]:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
        ]
    except Exception:
        return None

    def sample() -> Dict[str, float]:
        out: Dict[str, float] = {}
        for index, handle in enumerate(handles):
            prefix = "sys/gpu%d/" % index
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                out[prefix + "utilization"] = float(util.gpu)
                out[prefix + "memory_used_mb"] = memory.used / 2**20
                out[prefix + "memory_percent"] = 100.0 * memory.used / max(memory.total, 1)
                out[prefix + "temperature"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
                out[prefix + "power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                continue
        return out

    return sample


def _torch_cuda_sampler() -> Optional[Callable[[], Dict[str, float]]]:
    # Memory only, and only this process's: utilisation needs NVML. Better than
    # nothing on an image without `nvidia-ml-py`, and says so by what it lacks.
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        count = torch.cuda.device_count()
    except Exception:
        return None

    def sample() -> Dict[str, float]:
        out: Dict[str, float] = {}
        for index in range(count):
            prefix = "sys/gpu%d/" % index
            out[prefix + "memory_allocated_mb"] = torch.cuda.memory_allocated(index) / 2**20
            out[prefix + "memory_reserved_mb"] = torch.cuda.memory_reserved(index) / 2**20
        return out

    return sample


def _host_sampler() -> Optional[Callable[[], Dict[str, float]]]:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    process = psutil.Process()
    psutil.cpu_percent(None)  # the first call only starts the interval

    def sample() -> Dict[str, float]:
        return {
            "sys/cpu_percent": float(psutil.cpu_percent(None)),
            "sys/ram_percent": float(psutil.virtual_memory().percent),
            "sys/process_rss_mb": process.memory_info().rss / 2**20,
        }

    return sample


def system_samplers() -> List[Callable[[], Dict[str, float]]]:
    gpu = _nvml_sampler() or _torch_cuda_sampler()
    return [sampler for sampler in (gpu, _host_sampler()) if sampler is not None]


class SystemSampler:
    """Samples the machine every ``every`` seconds into a writer."""

    def __init__(
        self,
        writer: MetricsWriter,
        every: float,
        current_step: Callable[[], int],
        samplers: Optional[List[Callable[[], Dict[str, float]]]] = None,
    ) -> None:
        self._writer = writer
        self._every = every
        self._current_step = current_step
        self._samplers = system_samplers() if samplers is None else samplers
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ravex-sysmetrics", daemon=True)
        if self._samplers:
            self._thread.start()

    def _run(self) -> None:
        # At once, then on the interval. A run shorter than `every` would
        # otherwise record nothing about the machine it ran on, and a chart
        # with no points reads as a broken sampler rather than a quick run.
        self.sample_once()
        while not self._stop.wait(self._every):
            self.sample_once()

    def sample_once(self) -> None:
        values: Dict[str, Any] = {}
        for sampler in self._samplers:
            try:
                values.update(sampler())
            except Exception as exc:
                logger.debug("System metrics sampler failed: %s", exc)
        if values:
            self._writer.put(
                self._current_step(),
                time.time(),
                {key: Scalar(value) for key, value in values.items()},
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(5.0)


# ─── the session a runtime holds ─────────────────────────────────────


def _environment_rank(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


class MetricsSession:
    """What the runtime holds: pending records until the segment begins, then a writer.

    The segment cannot begin at activation. Its header records the step the run
    resumed from, and that is only known once the resume has happened — at the
    first dataloader iteration or the first ``optimizer.step()``, which is also
    when the process group is up and this process knows its rank. Anything
    logged before then waits here.
    """

    def __init__(self, directory: str, system_every: float, chunk_every: float = 15.0) -> None:
        self.directory = directory
        self.system_every = system_every
        self.chunk_every = chunk_every
        self._pending: List[Tuple[int, float, Dict[str, Any]]] = []
        self._writer: Optional[MetricsWriter] = None
        self._sampler: Optional[SystemSampler] = None
        self._begun = False
        self._accepts_user = True
        self._lock = threading.Lock()

    @property
    def begun(self) -> bool:
        return self._begun

    def begin(
        self,
        start_step: int,
        current_step: Callable[[], int],
        uploader: Optional[Callable[[str], None]] = None,
    ) -> None:
        with self._lock:
            if self._begun:
                return
            self._begun = True
            from ravex._dist.collectives import get_rank

            rank = get_rank()
            local_rank = _environment_rank("LOCAL_RANK")
            self._accepts_user = rank == 0
            samples_machine = self.system_every > 0 and (local_rank or 0) == 0
            if not (self._accepts_user or samples_machine):
                self._pending.clear()
                return
            self._writer = MetricsWriter(
                self.directory, rank, start_step, self.chunk_every, uploader
            )
            if self._accepts_user:
                for record in self._pending:
                    self._writer.put(*record)
            self._pending.clear()
            if samples_machine:
                self._sampler = SystemSampler(self._writer, self.system_every, current_step)

    def log(self, step: int, values: Dict[str, Any]) -> None:
        record = (step, time.time(), values)
        with self._lock:
            if not self._begun:
                self._pending.append(record)
                return
            if self._accepts_user and self._writer is not None:
                self._writer.put(*record)

    def close(self, current_step: int) -> None:
        if not self._begun:
            # A run that never stepped still logged something worth keeping,
            # an evaluation before training, say.
            self.begin(current_step, lambda: current_step)
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def prepare_all(values: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(
            "log_metrics takes a mapping of name to value, got %s" % type(values).__name__
        )
    prepared: Dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError("metric names are non-empty strings, got %r" % (key,))
        prepared[key] = prepare(key, value)
    return prepared
