"""A round report, written and read by moonclip.

GPU-115. :mod:`ravex._dist.outer` produces a dict of tensors per round and
:mod:`ravex._dist.exchange` moves it between machines; this is what turns it
into something on disk and back, and **the format is moonclip's**.

The first version of this module was a hand-rolled wire format — lengths, dtype
codes, raw little-endian bytes. It worked and it was wrong, for the reason this
repo keeps re-deriving: it was a *second* way to write tensors down, next to the
one the project already owns, and two formats mean two answers to every question
about precision, compression and what happens on a version skew.

Using the real one is not merely tidier, and the reason it pays is not the one
that was expected. Measured on an 8.39 MB delta of the shape a real one has —
small, sign-mixed values, eight 512x512 matrices:

======================== ========= =========
configuration            on disk   factor
======================== ========= =========
uncompressed             8.40 MB   1.00x
zstd level 3             7.77 MB   1.08x
zstd level 9             7.76 MB   1.08x
zstd 3 + ``save_dtype``  3.28 MB   2.56x
  bf16
zstd 3 + ``save_dtype``  1.73 MB   4.84x
  fp8
======================== ========= =========

**So compression buys almost nothing, and that was worth finding out.** Float
mantissas are close to incompressible, and nine levels of zstd argue with three
about one percent. Anyone reaching for "we get compression for free" as the
argument here should stop at 1.08x.

**What does pay is `save_dtype`, and it is GPU-113's point 5 already built.**
Halving the payload halves the H a model needs to amortise a round, and it is a
constructor argument rather than a project — per-tensor scales for the float8
targets, automatic uncast on load, all of it in Rust. The 4.84x of fp8 comes
with a few percent of relative error on every element, which moonclip is
explicit about being wrong for a checkpoint a run resumes from. An outer delta
is not that: it is averaged across nodes and then scaled by an outer learning
rate.

**And it survives that.** Measured, 2026-09-07, `bench/outer_convergence.py`:
a byte-level transformer on this repository's prose, two nodes on contiguous
shards, 2048 local steps each. Held-out loss with the cast and without it sits
inside ±0.005 at every H tried — smaller than the spread between neighbouring
arms, and fp8 lands marginally *ahead* at H=64, which is how you know it is
noise. The specific worry, that quantization error is not independent between
rounds and so accumulates instead of averaging out, does not show either: at
H=8 there are 256 rounds to accumulate over and the column is still flat.
A small model, so this is a direction and not a scaling law — but the reduction
is available and the argument for holding it back is now the measured one
rather than the cautious one.

**Deltas against a base, and a manifest that skips what the peer already has.**
Both are the replication path's machinery, and both apply unchanged.

**And it is still not pickle**, which was the one real argument the hand-rolled
format had. A moonclip snapshot is shapes, dtypes and compressed bytes; nothing
in reading one can be talked into executing anything.

**What stays ours is the check, and it got cheaper.** A payload arrives from
another machine, so before any of it is materialised its names, shapes and
dtypes are compared against the ones this node already holds — it has them,
because every node computes a delta over the same model. ``describe()`` answers
that from the manifest without touching a tensor, so a peer training something
else is refused having cost a manifest read rather than an allocation.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger("ravex")

#: Written last inside a round's directory, and its presence is the whole
#: definition of "this round can be served". A directory that exists without it
#: is one a publish is still filling in.
ROUND_OK = ".ravex-round-ok"

#: Round directories under a node's report root.
ROUND_PREFIX = "round-"

#: Metadata keys on a round snapshot. moonclip carries a ``Dict[str, str]``
#: alongside the tensors, which is exactly the shape of what a report needs to
#: say about itself, so there is no second channel for it.
STEPS = "ravex.steps"
NODE = "ravex.node"


class ReportError(ValueError):
    """A snapshot this node will not act on, and the message says why.

    Its own class because the caller's response differs from the response to a
    network failure: a peer that sends nothing is absent and the round closes
    without it, while a peer whose snapshot does not match this model is a
    version skew or a misconfiguration, and that deserves a louder line than
    silence does.
    """


class Report:
    """One node's round: what it did, and the delta it did it with."""

    __slots__ = ("delta", "steps", "node", "round_number")

    def __init__(self, delta, steps, node, round_number):
        self.delta = delta
        self.steps = steps
        self.node = node
        self.round_number = round_number

    def __repr__(self) -> str:
        return "Report(node=%r, round=%d, steps=%d, tensors=%d)" % (
            self.node,
            self.round_number,
            self.steps,
            len(self.delta),
        )


def open_store(
    root: str,
    *,
    compression_level: int = 3,
    save_dtype=None,
    keep_rounds: int = 3,
):
    """A moonclip manager tuned for round reports rather than for checkpoints.

    Three settings differ from a checkpoint store, and each is a consequence of
    what a round report is.

    ``full_every_steps=1`` — **every round is a full snapshot.** A delta
    snapshot is only readable next to the base it was written against, and the
    peer reading this one may have joined a round ago, or have pruned on its
    own schedule. The bandwidth a delta would save is real and it is GPU-113's
    point 5 to go measure; shipping a report that decodes only if the receiver
    happens to hold the right base is not the way to get it.

    ``keep_rounds`` — a report is worth keeping for about as long as a peer
    might still be fetching it, which is a round or two. Three, so that
    retention deleting an old round can never race a peer still reading it.

    ``keep_base_in_memory=False`` — with no deltas there is no base to keep,
    and the default would hold a second copy of the delta in host memory for
    nothing. On a box rented for how cheap it is, that copy is the difference
    between a model fitting and not.
    """
    import moonclip

    return moonclip.MoonclipManager(
        storage_root=root,
        compression_level=compression_level,
        save_dtype=save_dtype,
        full_every_steps=1,
        max_full_snapshots=max(2, keep_rounds),
        max_deltas_per_full=0,
        keep_base_in_memory=False,
        async_save=False,
    )


def round_path(root: str, round_number: int) -> str:
    """Where one round's report lives. **One directory per round, and that is
    the point** (GPU-119).

    Until 0.1.0 every round went into one store, and a lock kept a publish from
    writing it while a peer was reading it down a socket — because moonclip
    renames its manifest into place, and on Windows a rename onto a file
    another handle holds fails. The lock was correct and it was expensive: with
    one node's uplink slower than the other's, a publish blocked **2.2-2.3 s**
    waiting for a fetch to drain, against a round whose own transfer was
    0.30 s.

    A directory per round removes the conflict rather than serialising it: what
    is served is never what is being written. A published round is immutable
    from the moment its marker lands, so a send needs no lock at all, and
    retention deletes whole directories instead of pruning a store somebody may
    be reading.

    It costs nothing on the wire. The single store held ``keep_rounds``
    snapshots and the transport's skip list is what kept only the newest one
    crossing; one round per directory sends exactly that same one snapshot,
    with no skip list to negotiate. Measured on an 8 MB delta: 7.4 MB per round
    either way, and +3 ms for the extra manager.

    **And the lock cost more than the issue that filed it thought**, because
    that number was taken at two nodes. ``_serve`` held it across the whole
    send, so two peers fetching from one node were serialised against each
    other — which does not show at two nodes, where there is only ever one
    fetcher. Same bench, three nodes, 7 MB/s, 200 ms round trip:

    ================ ============ ============ =========== ===========
    arm              publish, was of which lock publish, is lock, is
    ================ ============ ============ =========== ===========
    both at 7 MB/s   2.49 s       **2.40 s**   0.13 s      **0.00 s**
    one node slower  1.07 s       0.98 s       0.10 s      0.00 s
    ================ ============ ============ =========== ===========

    So the lock's cost grew with the node count and not only with an asymmetric
    link. Publish and network together went from 7.66 s to 5.20 s per round at
    three nodes. What did *not* change is a two-node round: there the link is
    the bound either way, and the seconds the lock used to hide simply show up
    as an honest wait on the peer.
    """
    return os.path.join(root, "%s%d" % (ROUND_PREFIX, int(round_number)))


def publish_round(
    root: str,
    round_number: int,
    delta: Dict[str, Any],
    steps: int,
    node: str,
    *,
    compression_level: int = 3,
    save_dtype=None,
) -> str:
    """Write one round into its own directory, and mark it servable last.

    The marker is written after the store, never before: a peer that finds the
    directory without it is looking at a publish in progress, and
    :func:`round_is_complete` is what keeps it from being served half-written.
    """
    path = round_path(root, round_number)
    os.makedirs(path, exist_ok=True)
    store = open_store(
        path, compression_level=compression_level, save_dtype=save_dtype
    )
    write(store, delta, round_number, steps, node)
    with open(os.path.join(path, ROUND_OK), "wb"):
        pass
    return path


def mark_complete(path: str) -> None:
    """Mark a round directory servable. Idempotent."""
    with open(os.path.join(path, ROUND_OK), "wb"):
        pass


def round_is_complete(root: str, round_number: int) -> bool:
    return os.path.exists(os.path.join(round_path(root, round_number), ROUND_OK))


def read_round(root: str, round_number: int, expected) -> "Report":
    """One round's report, out of the directory it arrived in."""
    return read(open_store(round_path(root, round_number)), round_number, expected)


def rounds_present(root: str) -> Iterable[int]:
    """The round numbers this root holds, complete or not."""
    try:
        names = os.listdir(root)
    except OSError:
        return []
    found = []
    for name in names:
        if not name.startswith(ROUND_PREFIX):
            continue
        try:
            found.append(int(name[len(ROUND_PREFIX):]))
        except ValueError:
            continue
    return sorted(found)


def drop_rounds(root: str, keep: int, protect: Iterable[int] = ()) -> None:
    """Delete round directories past the newest ``keep``, except ``protect``.

    ``protect`` is the rounds currently going down a socket. Retention is the
    only thing left that can touch a directory somebody else is reading, so it
    is the only thing that has to ask — and asking is a set lookup rather than
    a lock held across a transfer, which is the whole trade this design makes.
    """
    held = sorted(rounds_present(root))
    if len(held) <= keep:
        return
    keeping = set(held[-keep:]) | set(protect)
    for number in held:
        if number in keeping:
            continue
        try:
            shutil.rmtree(round_path(root, number), ignore_errors=True)
        except OSError as exc:  # pragma: no cover - best effort
            logger.debug("Could not drop round %d: %s", number, exc)


def write(store, delta: Dict[str, Any], round_number: int, steps: int, node: str) -> str:
    """Put this node's report in the store, under ``round_number`` as the step.

    The round number *is* the step, so a peer looks a report up by the round it
    is asking about rather than by trusting whatever happens to be newest —
    which matters because a peer may well be a round ahead by the time it
    answers.
    """
    return store.save_tensors(
        int(round_number),
        delta,
        metadata={STEPS: str(int(steps)), NODE: str(node)},
    )


def snapshot_of_round(store, round_number: int) -> Optional[str]:
    """The snapshot id holding ``round_number``, or None if it is not there.

    None covers both "not written yet" and "retention dropped it", and the
    caller cannot tell them apart from here — nor does it need to: either way
    this store cannot answer for that round.
    """
    try:
        for snapshot in store.list_snapshots():
            if int(snapshot.get("step", -1)) == int(round_number):
                return snapshot.get("id")
    except Exception as exc:
        logger.debug("Could not list snapshots: %s", exc)
    return None


def expectation(delta: Dict[str, Any]) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """What a reader will accept, taken from the delta it already holds.

    Every node in a round computes a delta over the same model, so this node's
    own is an exact description of what a peer's must look like. Dtypes as the
    strings moonclip reports, so the comparison happens in one vocabulary.
    """
    return {
        name: (str(t.dtype).replace("torch.", ""), tuple(t.shape))
        for name, t in delta.items()
    }


def read(store, round_number: int, expected) -> Report:
    """One peer's report for ``round_number``, checked before it is materialised.

    Raises :class:`ReportError` for anything that does not match what this node
    holds. The order matters: ``describe`` reads the manifest and no tensor
    data at all, so a snapshot from a different model costs a manifest read
    rather than however many gigabytes it claimed.
    """
    import torch

    snapshot_id = snapshot_of_round(store, round_number)
    if snapshot_id is None:
        raise ReportError("no snapshot for round %d in this store" % round_number)

    described = store.describe(snapshot_id)
    tensors = {entry["name"]: entry for entry in described.get("tensors", [])}

    if len(tensors) != len(expected):
        raise ReportError(
            "report carries %d tensor(s) and this node holds %d. A delta "
            "missing parameters would update part of the model with fewer "
            "nodes than the rest" % (len(tensors), len(expected))
        )
    for name, entry in tensors.items():
        if name not in expected:
            raise ReportError(
                "report carries a parameter this node does not have (%r). Two "
                "nodes in one round have to be training one model" % name[:64]
            )
        want_dtype, want_shape = expected[name]
        shape = tuple(int(dimension) for dimension in entry.get("shape", ()))
        if shape != want_shape:
            raise ReportError(
                "%s arrived as %s and this node holds %s"
                % (name, list(shape), list(want_shape))
            )
        # ``dtype`` is what a load hands back; ``stored_dtype`` is what is on
        # disk, and they differ exactly when save_dtype cast the tensor. The
        # first is the one that has to match — a peer compressing its report to
        # bf16 on the wire is a setting, not a different model.
        if str(entry.get("dtype")) != want_dtype:
            raise ReportError(
                "%s arrived as %s and this node holds %s"
                % (name, entry.get("dtype"), want_dtype)
            )

    raw = store.load(snapshot_id)
    delta = {}
    for name, entry in tensors.items():
        want_dtype, want_shape = expected[name]
        buffer = raw.get(name)
        if buffer is None:
            raise ReportError("%s was described but not in the snapshot" % name)
        dtype = getattr(torch, want_dtype)
        if not want_shape or all(want_shape):
            expect_bytes = torch.empty(0, dtype=dtype).element_size()
            for dimension in want_shape:
                expect_bytes *= dimension
        else:
            expect_bytes = 0
        if len(buffer) != expect_bytes:
            raise ReportError(
                "%s is %d bytes and its shape needs %d"
                % (name, len(buffer), expect_bytes)
            )
        if expect_bytes == 0:
            delta[name] = torch.empty(want_shape, dtype=dtype)
            continue
        try:
            flat = torch.frombuffer(buffer, dtype=torch.uint8).view(dtype)
            delta[name] = flat.reshape(want_shape).clone()
        except (RuntimeError, ValueError) as exc:
            raise ReportError("%s could not be read: %s" % (name, exc)) from exc

    metadata = described.get("metadata") or {}
    return Report(
        delta=delta,
        steps=int(metadata.get(STEPS, 0) or 0),
        node=str(metadata.get(NODE, "")),
        round_number=int(described.get("step", round_number)),
    )
