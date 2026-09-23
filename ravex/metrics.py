"""Read back what a run logged with :func:`ravex.log_metrics` (GPU-147).

::

    import ravex.metrics

    history = ravex.metrics.read("./checkpoints")
    loss = history["scalars"]["train/loss"]      # {"step": [...], "time": [...], "value": [...]}
    weights = history["histograms"]["weights"]   # [{"step", "time", "min", "max", "counts", ...}]
    gpu = history["system"]["node-a"]["sys/gpu0/utilization"]

The path is the run's ``storage.path``, the same one its checkpoints are in.

**Reading from somewhere that is not a directory.** :func:`read` does the I/O
and :func:`resolve` does the rest. A reader that gets the files some other
way - a Worker listing an R2 bucket, asynchronously - fetches every object
under ``metrics/`` and hands ``resolve`` a mapping from each object's path
below ``metrics/`` (``"<segment>/000003.jsonl"``) to its text. The timeline
comes out the same either way, which is the point of keeping the rule here.
This module and :mod:`ravex._metrics` import nothing compiled, so they load
where Ravex's Rust core cannot.

**The timeline rule.** A run is written in segments, one per execution, and
each segment's header says the step it resumed from. When a run resumes at
step 500 after reaching 700, the points its earlier execution logged after 500
describe a history that was abandoned: the model that produced them no longer
exists. :func:`resolve` keeps every segment only up to the step the next one
resumed from, so the series it returns is the history of the model the store
holds - one value per step, as if the run had never been interrupted.

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
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from ravex._metrics import FORMAT_VERSION, HEADER_CHUNK, METRICS_DIR

__all__ = ["read", "resolve", "segments"]

logger = logging.getLogger("ravex")

_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _value(raw: Any) -> Any:
    if isinstance(raw, str):
        return _NONFINITE.get(raw, raw)
    return raw


def _records(text: str, where: str) -> Iterator[Dict[str, Any]]:
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            # Chunks appear whole, by rename, so this is a file somebody
            # edited, or one copied while it was being written.
            logger.debug("Skipping unreadable line %d of %s", number, where)
            continue
        if isinstance(record, dict):
            yield record


def _local_chunks(path: str) -> Dict[str, str]:
    """Every chunk under ``<path>/metrics``, keyed by its path below that."""
    root = os.path.join(path, METRICS_DIR)
    chunks: Dict[str, str] = {}
    try:
        segment_names = os.listdir(root)
    except FileNotFoundError:
        return chunks
    for segment in segment_names:
        directory = os.path.join(root, segment)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.endswith(".jsonl"):
                continue  # a chunk still being written ends in `.tmp`
            try:
                with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
                    chunks[segment + "/" + name] = handle.read()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", os.path.join(directory, name), exc)
    return chunks


def _grouped(chunks: Mapping[str, str]) -> Dict[str, List[Tuple[str, str]]]:
    by_segment: Dict[str, List[Tuple[str, str]]] = {}
    for key, text in chunks.items():
        key = key.replace("\\", "/").strip("/")
        if key.startswith(METRICS_DIR + "/"):
            key = key[len(METRICS_DIR) + 1 :]
        segment, _, name = key.rpartition("/")
        if not segment or not name.endswith(".jsonl"):
            continue
        by_segment.setdefault(segment, []).append((name, text))
    for entries in by_segment.values():
        entries.sort()
    return by_segment


def _headers(by_segment: Mapping[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
    headers = []
    for segment, entries in by_segment.items():
        first = entries[0]
        if first[0] != HEADER_CHUNK:
            # The header has not arrived, or was lost. Without it there is no
            # resume step, and guessing one could cut another segment wrongly.
            logger.debug("Segment %s has no header chunk; skipped", segment)
            continue
        header = next(_records(first[1], segment + "/" + first[0]), None)
        if header is None or "segment" not in header:
            continue
        if header.get("format", 0) > FORMAT_VERSION:
            logger.warning(
                "Segment %s was written by a newer Ravex (format %s); reading "
                "what this version understands",
                segment,
                header.get("format"),
            )
        headers.append(dict(header, chunks=len(entries) - 1))
    headers.sort(key=lambda header: (header.get("time", 0.0), header["segment"]))
    return headers


def segments(path: str) -> List[Dict[str, Any]]:
    """Every segment's header in the store at ``path``, oldest first."""
    return _headers(_grouped(_local_chunks(path)))


def _series() -> Dict[str, List[Any]]:
    return {"step": [], "time": [], "value": []}


def read(path: str, inherited: bool = True) -> Dict[str, Any]:
    """Everything logged into the store at ``path``, resolved into one timeline.

    See :func:`resolve` for what comes back.

    **A fork continues its parent's line** (GPU-149). When the store's
    ``run.json`` names a parent - a run started with ``fork_from`` - each of
    its series begins with the parent's points up to the fork step, read the
    same way and so through the parent's own parents, and the answer has a
    ``parent`` key, ``{"store": ..., "step": ...}``: where the line changes
    hands, which is where a chart draws the branch. ``inherited=False`` is the
    run's own points only.

    System metrics are not inherited: they describe the parent's machines, not
    this run's. A parent store that is not where the fork recorded it is
    skipped with a warning rather than failing the read.
    """
    history = resolve(_local_chunks(path))
    if inherited:
        _inherit(path, history, {os.path.abspath(path)})
    return history


def _parent_of(path: str) -> Optional[Tuple[str, int]]:
    """The store this run forked from, and the step, from its run.json."""
    try:
        with open(os.path.join(path, "run.json"), encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    parent = record.get("parent") or {}
    source = (record.get("config") or {}).get("fork_from")
    if parent.get("step") is None or not source:
        return None
    return str(source), int(parent["step"])


def _inherit(path: str, history: Dict[str, Any], seen: set) -> None:
    found = _parent_of(path)
    if found is None:
        return
    source, step = found
    if os.path.abspath(source) in seen:
        return
    if not os.path.isdir(source):
        logger.warning(
            "%s forked from %s, which is not there; reading its own points only", path, source
        )
        return
    seen.add(os.path.abspath(source))
    before = resolve(_local_chunks(source))
    _inherit(source, before, seen)
    for key, series in before["scalars"].items():
        keep = [i for i, at in enumerate(series["step"]) if at <= step]
        own = history["scalars"].get(key, _series())
        merged = _series()
        for field in ("step", "time", "value"):
            merged[field] = [series[field][i] for i in keep] + [
                value for at, value in zip(own["step"], own[field]) if at > step
            ]
        history["scalars"][key] = merged
    for key, entries in before["histograms"].items():
        own_entries = [entry for entry in history["histograms"].get(key, []) if entry["step"] > step]
        history["histograms"][key] = [entry for entry in entries if entry["step"] <= step] + own_entries
    history["parent"] = {"store": source, "step": step}


def resolve(chunks: Mapping[str, str]) -> Dict[str, Any]:
    """Resolve metric chunks, however they were fetched, into one timeline.

    ``chunks`` maps each chunk's path below ``metrics/`` - ``"<segment>/<n>.jsonl"``,
    with or without the leading ``metrics/`` - to its text.

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
    by_segment = _grouped(chunks)
    headers = _headers(by_segment)
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
        segment = header["segment"]
        limit = cut_at.get(segment)
        is_primary = header.get("rank", 0) == 0
        host = str(header.get("host", "unknown"))
        for name, text in by_segment[segment][1:]:
            for record in _records(text, segment + "/" + name):
                step = record.get("step")
                when = record.get("time")
                values = record.get("values")
                if not isinstance(step, int) or not isinstance(values, dict):
                    continue
                for key, raw in values.items():
                    if key.startswith("sys/"):
                        series = system.setdefault(host, {}).setdefault(key, _series())
                        series["step"].append(step)
                        series["time"].append(when)
                        series["value"].append(_value(raw))
                        continue
                    if not is_primary or (limit is not None and step > limit):
                        continue
                    if isinstance(raw, dict) and "histogram" in raw:
                        entry = dict(raw["histogram"], step=step, time=when)
                        histogram_points.setdefault(key, []).append((step, order, entry))
                    else:
                        scalar_points.setdefault(key, []).append((step, order, when, _value(raw)))

    for key, points in scalar_points.items():
        points.sort(key=lambda point: (point[0], point[1]))
        series = scalars[key] = _series()
        for step, _order, when, value in points:
            if series["step"] and series["step"][-1] == step:
                series["time"][-1] = when
                series["value"][-1] = value
                continue
            series["step"].append(step)
            series["time"].append(when)
            series["value"].append(value)

    for key, entries in histogram_points.items():
        entries.sort(key=lambda point: (point[0], point[1]))
        resolved: List[Dict[str, Any]] = []
        for step, _order, entry in entries:
            if resolved and resolved[-1]["step"] == step:
                resolved[-1] = entry
            else:
                resolved.append(entry)
        histograms[key] = resolved

    for per_host in system.values():
        for series in per_host.values():
            order = sorted(range(len(series["time"])), key=lambda i: series["time"][i])
            for field in ("step", "time", "value"):
                series[field] = [series[field][i] for i in order]

    for header in headers:
        header["cut_at"] = cut_at.get(header["segment"])
    return {
        "scalars": scalars,
        "histograms": histograms,
        "system": system,
        "segments": headers,
    }
