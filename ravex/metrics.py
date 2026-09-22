"""Read back what a run logged with :func:`ravex.log_metrics` (GPU-147).

::

    import ravex.metrics

    history = ravex.metrics.read("./checkpoints")
    loss = history["scalars"]["train/loss"]      # {"step": [...], "time": [...], "value": [...]}
    weights = history["histograms"]["weights"]   # [{"step", "time", "min", "max", "counts", ...}]
    gpu = history["system"]["node-a"]["sys/gpu0/utilization"]

The path is the run's ``storage.path``, the same one its checkpoints are in.

**The timeline rule.** A run is written in segments, one per execution, and
each segment's header says the step it resumed from. When a run resumes at
step 500 after reaching 700, the points its earlier execution logged after 500
describe a history that was abandoned: the model that produced them no longer
exists. :func:`read` keeps every segment only up to the step the next one
resumed from, so the series it returns is the history of the model the store
holds — one value per step, as if the run had never been interrupted.

System metrics are exempt. They describe machines, not the model: a GPU that
was at 100% for the 200 steps that were thrown away really was at 100%, and
that is worth seeing when someone asks what a crash cost.

Values that are NaN or infinite come back as floats. On disk they are strings
(``"nan"``), because a browser's JSON parser refuses them bare.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ravex._metrics import FORMAT_VERSION, METRICS_DIR

__all__ = ["read", "segments"]

logger = logging.getLogger("ravex")

_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _value(raw: Any) -> Any:
    if isinstance(raw, str):
        return _NONFINITE.get(raw, raw)
    return raw


def _lines(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                # The last line of a process killed mid-write. Anything else
                # torn is worth a word, because it means the file was edited.
                logger.debug("Skipping unreadable line %d of %s", number, path)


def segments(path: str) -> List[Dict[str, Any]]:
    """Every segment's header, oldest first, with the file it came from as ``file``."""
    directory = os.path.join(path, METRICS_DIR)
    try:
        names = sorted(n for n in os.listdir(directory) if n.endswith(".jsonl"))
    except FileNotFoundError:
        return []
    headers = []
    for name in names:
        file = os.path.join(directory, name)
        try:
            first = next(_lines(file), None)
        except OSError:
            continue
        if not isinstance(first, dict) or "segment" not in first:
            continue
        if first.get("format", 0) > FORMAT_VERSION:
            logger.warning(
                "%s was written by a newer Ravex (format %s); reading what this "
                "version understands",
                file,
                first.get("format"),
            )
        headers.append(dict(first, file=file))
    headers.sort(key=lambda header: (header.get("time", 0.0), header["segment"]))
    return headers


def _series() -> Dict[str, List[Any]]:
    return {"step": [], "time": [], "value": []}


def read(path: str) -> Dict[str, Any]:
    """Everything logged into the store at ``path``, resolved into one timeline.

    Returns a dict with four keys:

    ``scalars``
        ``{name: {"step": [...], "time": [...], "value": [...]}}``, sorted by
        step.
    ``histograms``
        ``{name: [{"step", "time", "min", "max", "counts", "nonfinite"}, ...]}``.
        ``counts`` are ``len(counts)`` equal bins between ``min`` and ``max``;
        ``stride`` is present when the histogram was taken over every
        ``stride``-th element of a large tensor.
    ``system``
        ``{host: {name: {"step", "time", "value"}}}``, sorted by time and never
        cut at a resume.
    ``segments``
        The headers, oldest first, each with ``cut_at``: the step past which
        its points were dropped, or ``None`` for the segment that is current.
    """
    headers = segments(path)
    primary = [header for header in headers if header.get("rank", 0) == 0]
    cut_at: Dict[str, Optional[int]] = {}
    for index, header in enumerate(primary):
        following = primary[index + 1] if index + 1 < len(primary) else None
        cut_at[header["segment"]] = None if following is None else int(following["start_step"])

    scalars: Dict[str, Dict[str, List[Any]]] = {}
    histograms: Dict[str, List[Dict[str, Any]]] = {}
    system: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    # Collected with the segment's position, so a step logged by two
    # segments - an evaluation at the resume step, say - keeps the later one.
    scalar_points: Dict[str, List[Tuple[int, int, float, Any]]] = {}
    histogram_points: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}

    for order, header in enumerate(headers):
        limit = cut_at.get(header["segment"])
        is_primary = header.get("rank", 0) == 0
        host = str(header.get("host", "unknown"))
        try:
            records = list(_lines(header["file"]))
        except OSError as exc:
            logger.warning("Cannot read %s: %s", header["file"], exc)
            continue
        for record in records[1:]:
            step = record.get("step")
            when = record.get("time")
            values = record.get("values")
            if not isinstance(step, int) or not isinstance(values, dict):
                continue
            for name, raw in values.items():
                if name.startswith("sys/"):
                    series = system.setdefault(host, {}).setdefault(name, _series())
                    series["step"].append(step)
                    series["time"].append(when)
                    series["value"].append(_value(raw))
                    continue
                if not is_primary or (limit is not None and step > limit):
                    continue
                if isinstance(raw, dict) and "histogram" in raw:
                    entry = dict(raw["histogram"], step=step, time=when)
                    histogram_points.setdefault(name, []).append((step, order, entry))
                else:
                    scalar_points.setdefault(name, []).append((step, order, when, _value(raw)))

    for name, points in scalar_points.items():
        points.sort(key=lambda point: (point[0], point[1]))
        series = scalars[name] = _series()
        for step, _order, when, value in points:
            if series["step"] and series["step"][-1] == step:
                series["time"][-1] = when
                series["value"][-1] = value
                continue
            series["step"].append(step)
            series["time"].append(when)
            series["value"].append(value)

    for name, entries in histogram_points.items():
        entries.sort(key=lambda point: (point[0], point[1]))
        resolved: List[Dict[str, Any]] = []
        for step, _order, entry in entries:
            if resolved and resolved[-1]["step"] == step:
                resolved[-1] = entry
            else:
                resolved.append(entry)
        histograms[name] = resolved

    for per_host in system.values():
        for series in per_host.values():
            order = sorted(range(len(series["time"])), key=lambda i: series["time"][i])
            for key in ("step", "time", "value"):
                series[key] = [series[key][i] for i in order]

    for header in headers:
        header["cut_at"] = cut_at.get(header["segment"])
    return {
        "scalars": scalars,
        "histograms": histograms,
        "system": system,
        "segments": headers,
    }
