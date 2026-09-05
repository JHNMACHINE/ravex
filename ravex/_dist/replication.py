"""Sending each rank's checkpoint to a peer on another machine.

The third rung of the ladder in GPU-59. With storage that is neither remote nor
shared, a per-rank checkpoint lives only on the machine that wrote it, and
losing that machine loses those shards for good — they are disjoint slices no
other rank can rebuild. Demonstrated on 2026-08-19: four nodes, one killed
mid-run, 2260 steps unrecoverable on three intact disks.

The fix has to happen *before* the failure. Nothing at resume can recover bytes
that exist nowhere, which is why a fetch-on-resume design was rejected: it
repairs a placement that moved, not a machine that is gone.

**What this buys, stated exactly:** survival of losing any one machine, at the
cost of losing at most one replication interval of progress. Not "never loses
anything" — that is not on offer here, and durability past the end of the run
is the user's, by decision on 2026-08-19.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: Replicas live here, deliberately outside the ``rank_<n>`` namespace that
#: :func:`ravex._backends.visible_rank_stores` scans. A replica found under
#: that name would be read as a store this machine owns, which is the one
#: mistake that would turn redundancy into confusion.
REPLICA_DIR = "replica"


def machine_count(world_size: int, local_size: int) -> Optional[int]:
    """How many machines this job spans, or None when it cannot be told.

    Only uniform layouts can be worked out from these two numbers. With eight
    ranks on one node and four on another, ``local_size`` describes the node
    asking and says nothing about the others, so rank *n* cannot be mapped to a
    machine and no peer can be shown to be on a different one.
    """
    if local_size <= 0 or world_size <= 0:
        return None
    if world_size % local_size != 0:
        return None
    return world_size // local_size


def replication_ring(
    rank: int, world_size: int, local_size: int
) -> Optional[Tuple[int, int]]:
    """``(send_to, receive_from)`` for this rank, or None if it cannot be safe.

    A ring with a stride of ``local_size``: rank *r* sends to
    ``r + local_size``, which under a uniform layout is the same position on
    the next machine along. Every rank sends exactly one copy and receives
    exactly one, so the traffic is one shard per rank per interval no matter
    how many machines there are.

    **None is a real answer and means the replication must not run.** Three
    situations produce it, and in each one going ahead would write a copy onto
    the machine that already holds the original — all the cost of replication
    and none of the protection, which is worse than declining, because it looks
    like protection:

    * one machine, where there is no second machine to copy to;
    * a layout this cannot map to machines (see :func:`machine_count`);
    * nonsense inputs.

    The caller is expected to say why it is off rather than fall silent: a run
    that believes it is protected and is not is the failure this whole module
    exists to prevent.
    """
    machines = machine_count(world_size, local_size)
    if machines is None or machines < 2:
        return None
    if not 0 <= rank < world_size:
        return None

    send_to = (rank + local_size) % world_size
    receive_from = (rank - local_size) % world_size
    return send_to, receive_from


def machine_of(rank: int, local_size: int) -> Optional[int]:
    """Which machine a rank sits on under a uniform layout, else None."""
    if local_size <= 0 or rank < 0:
        return None
    return rank // local_size


# ─── moving a store, without moving it twice ────────────────────────

import os
import shutil

from ravex import _core

#: Bytes per chunk on the wire. Bounded on purpose: a real shard is over a
#: gigabyte, and a whole-store buffer would be that allocation on the training
#: process at checkpoint time — on top of the shadow copy and whatever the
#: allocator is already holding. Host memory pressure right there is measured,
#: not hypothetical; it is what GPU-54 turned out to be.
CHUNK = _core.CHUNK

# ─── the framing, which is Rust since GPU-109 ───────────────────────────────
#
# `store_files`, the two manifest encodings, `encoded_size`, `encode_store` and
# `StoreWriter` were written here until GPU-109 and are now names over
# `ravex._core`. **The format did not change and could not**: a replica written
# by one version is read back by another, and `exchange_stores` below still
# moves its bytes over torch collectives through this same framing. It is
# little-endian throughout — a `<I` count, then `<HQ` plus the name plus a `<B`
# flag per entry, then the bodies — and `src/transport.rs` is where it lives
# now.
#
# Why they moved is measured rather than assumed, which is the rule GPU-105 set
# itself and did not follow. Two numbers:
#
# * The receiving side buffered every chunk and then did `del pending[:take]`,
#   a memmove of the tail at every file boundary. The transfer fell from
#   529 MB/s to 113 MB/s as the chunk grew from 4 to 64 MiB — GPU-84, and the
#   wrong way round for a block transfer. The Rust holds a cursor instead.
# * Reading the disk and writing the socket took turns. Measured 2026-09-05
#   with `bench/transport_ceiling.py`: 1.2 GB/s reading and framing, 4.5 GB/s
#   on the wire, 0.99 GB/s for the two of them alternating. The Rust reads on
#   its own thread, bounded to three chunks ahead.
#
# The names keep the leading underscores they had. This module's surface is
# unchanged, `ravex/_dist/elastic.py` imports two of them, and the test suite
# that covers them was written against the Python and has not been touched.

store_files = _core.store_files
_encode_manifest = _core.encode_manifest
_parse_manifest = _core.parse_manifest
encoded_size = _core.encoded_size
StoreWriter = _core.StoreWriter


def encode_store(path: str, chunk: int = CHUNK, skip: "set[str] | None" = None):
    """Yield a store as a byte stream, reading from disk as it goes.

    An iterator rather than a buffer, and read file by file rather than packed
    into a temporary archive first: the archive would be a second full copy on
    a disk that is already the binding constraint on these machines — the
    reason ``measure_handoff.sh`` has space guards at all.

    ``skip`` names files whose bytes must not be read or sent — the peer
    already holds them, byte for byte, and said so before this call. They are
    still listed in the header, flagged, so the receiver's manifest and pruning
    stay exactly as complete as a full transfer's; only the body is shorter.

    A function wrapping the class rather than the class itself, so the
    signature, the defaults and this docstring stay where a reader of this
    module will look for them.
    """
    return _core.StoreEncoder(path, chunk, skip)


#: Written last, deleted first. A replica directory is only trustworthy while
#: this is present: everything else about it — the files, their names, even a
#: manifest — is equally true of a copy that was cut off half way.
COMPLETE_MARKER = _core.COMPLETE_MARKER


def replica_is_complete(path: str) -> bool:
    """Whether a replica directory holds a whole store rather than a torn one.

    The case this exists for is not hypothetical: a transfer is interrupted
    when the peer dies, which is the very failure replication is here to
    survive. A half-written copy that looks like a copy would turn a recovery
    into a wrong model.
    """
    return os.path.exists(os.path.join(path, COMPLETE_MARKER))


def unmark_replica(path: str) -> None:
    """Drop the marker from a directory that has stopped being a copy.

    Both promotions end here: the one that rebuilds a store from a copy on
    this machine's own disk, and the one that takes it back off a peer. They
    used to end differently — :func:`promote_copy` removed the marker and the
    network path left it — because the code that writes it, `StoreWriter`, is
    building a copy as far as it knows and cannot tell the two destinations
    apart. The same logical act left two different states on disk depending on
    which way the bytes travelled, which is the sort of difference that
    survives until someone rents a machine and looks.

    What the marker asserts is that a *copy* is whole. About a store the rank
    is about to start rewriting, that is true for one instant and false ever
    after, and it is left exactly where a later reader — code or person — would
    take it at its word.

    Removed after the store is whole and never before. While the transfer is
    still running, the absence of this file is the only thing separating a
    half-built store from a finished one.
    """
    try:
        os.remove(os.path.join(path, COMPLETE_MARKER))
    except OSError:
        pass


def fixed_chunks(blocks, size: int):
    """Re-cut a stream of ragged blocks into exact ``size`` pieces.

    ``encode_store`` yields a header and then whatever came off each file, so
    its blocks do not line up with anything. The wire needs them to: both ends
    compute the chunk count from the total length, and a short chunk in the
    middle would put them out of step for the rest of the transfer.
    """
    buffer = bytearray()
    for block in blocks:
        buffer += block
        while len(buffer) >= size:
            yield bytes(buffer[:size])
            del buffer[:size]
    if buffer:
        yield bytes(buffer)


def recovery_roles(rank, send_to, receive_from, needs, holds):
    """What this rank does in a recovery pass: ``(should_send, should_receive)``.

    ``needs[r]`` is whether rank *r* came up without its own store — a machine
    that was replaced. ``holds[r]`` is whether rank *r* is holding a **whole**
    copy of the rank it received from, which under the ring is the rank that
    sends to it.

    Recovery runs the ring backwards: a copy travels back to the rank it
    belongs to. So this rank receives from the peer it normally sends to, and
    sends to the peer it normally receives from.

    Both sides work this out from the same gathered lists, which is what makes
    the pairing exact. A send posted without a matching receive would sit there
    until the job died — and this runs at startup, before anyone is watching.
    """
    if not (0 <= rank < len(needs)) or len(holds) != len(needs):
        return False, False

    should_receive = bool(needs[rank]) and bool(holds[send_to])
    should_send = bool(needs[receive_from]) and bool(holds[rank])
    return should_send, should_receive


def local_recoveries(needs, has_own_copy, copy_runs, store_runs):
    """Which ranks can rebuild themselves from a copy already on their disk.

    The ring addresses a peer **by rank**, but a copy travels **with the disk**
    it was written to. Shift the nodes by one position and every rank comes up
    sitting on the copy of its own store: all of them present, complete, and
    none reachable by asking a peer — because the peer that used to hold your
    copy is now elsewhere holding somebody else's. Seen on the bench on
    2026-08-20, six intact copies buying nothing.

    A copy under your own feet costs nothing to use: no pairing, no transfer,
    no collective. What it does need is proof that it belongs to the history
    this job is continuing. Promoting a copy left by an older run would leave
    the job resuming half its shards from one training history and half from
    another, which is a wrong model — worse than starting from scratch, and
    silent.

    So the history has to be **unambiguous**, and there are two ways to know it:

    * some ranks still have their own store, and those stores name it;
    * none does — the full reshuffle — and then the copies are all there is,
      so they have to agree among themselves.

    Anything less settled promotes nobody. Two histories on one disk is exactly
    the case that must not be guessed at, and it is not hypothetical: it
    happened on 2026-08-19.

    Every rank computes this from the same gathered lists and so reaches the
    same answer, which is what keeps the network pairing that follows exact
    even though no one announces whether their own promotion worked.
    """
    size = len(needs)
    if any(len(other) != size for other in (has_own_copy, copy_runs, store_runs)):
        return [False] * size

    # A copy with no owner record cannot be attributed to anything, and an
    # unattributable copy is the one case this function exists to refuse.
    candidates = {
        rank
        for rank in range(size)
        if needs[rank] and has_own_copy[rank] and copy_runs[rank]
    }
    if not candidates:
        return [False] * size

    history = {
        store_runs[rank]
        for rank in range(size)
        if not needs[rank] and store_runs[rank]
    }
    if not history:
        history = {copy_runs[rank] for rank in candidates}

    if len(history) != 1:
        return [False] * size

    wanted = next(iter(history))
    return [rank in candidates and copy_runs[rank] == wanted for rank in range(size)]


def promote_copy(copy_path: str, store_path: str) -> bool:
    """Turn a copy already on this disk into this rank's own store.

    A plain local copy: the bytes are here, and nothing has to be asked of
    anyone. Built in a hidden sibling and renamed into place, so a process that
    dies half way leaves something the store scans skip rather than a directory
    that looks like a store and is not — ``rank_<n>`` is what everything
    scans for, and a leading dot is not.

    The completeness marker stays behind — see :func:`unmark_replica`, which
    the recovery over the network calls for the same reason. The owner record
    does come across: it is what lets the resume recognise which history it is
    continuing, and dropping it would undo the check that allowed this
    promotion.
    """
    if not replica_is_complete(copy_path):
        return False

    parent = os.path.dirname(store_path) or "."
    staging = os.path.join(parent, "." + os.path.basename(store_path) + ".incoming")

    try:
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(parent, exist_ok=True)
        shutil.copytree(copy_path, staging)

        unmark_replica(staging)

        # The caller only gets here when this rank has no usable store, so what
        # is being removed is a missing or empty directory — and `rename` onto
        # a non-empty one fails on POSIX and on Windows alike.
        shutil.rmtree(store_path, ignore_errors=True)
        os.rename(staging, store_path)
        return True
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        return False


def _wire_tensor(block: bytes):
    """A uint8 tensor over ``block``'s own memory, without copying it.

    ``torch.frombuffer`` warns when the buffer is read-only, because a tensor
    that cannot be written to is usually a mistake. Here it is the point: this
    one is handed to ``isend`` and never touched again, and the immutability is
    what makes skipping the copy safe.
    """
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(block, dtype=torch.uint8)


def exchange_stores(
    source: "str | None",
    destination: "str | None",
    send_to: int,
    receive_from: int,
    chunk: int = CHUNK,
    group=None,
) -> bool:
    """Trade one store for another around the ring. **Collective in effect.**

    Either direction may be switched off by passing ``None`` for its path, and
    the recovery pass in :func:`fetch_missing_stores` needs that: there, only
    the ranks that lost a store receive, and only the ranks holding what they
    lost send. Both ends work out the same pairs beforehand, so a send is never
    posted without a receive waiting for it.

    Blocking sends would stall the whole ring — every rank sitting in ``send``
    waiting for a receiver who is itself sitting in ``send`` — so the send is
    posted non-blocking and waited on beside the receive.

    Returns whether the incoming store arrived whole, and ``True`` when nothing
    was expected. The caller must not read that as the answer for everybody: a
    round is a recovery point only if every rank succeeded, which is a separate
    agreement.

    Before any store bytes move, the receiving side reports what its
    destination already holds — file names and sizes, nothing read — and the
    sending side uses that to skip files that would arrive unchanged. Safe on
    names alone, no hash, because the files a store is made of are immutable
    once written; see GPU-97. This travels opposite to the size and body
    exchange further down (the receiver is the one with something to say
    here, not the sender), so it is its own pair of messages rather than
    riding along with those.
    """
    import torch
    import torch.distributed as dist

    sending = source is not None
    receiving = destination is not None
    if not sending and not receiving:
        return True

    existing_blob = _encode_manifest(store_files(destination)) if receiving else b""
    peer_existing: "dict[str, int]" = {}
    incoming_blob_len = 0

    handle = None
    if receiving:
        blob_len = torch.tensor([len(existing_blob)], dtype=torch.int64)
        handle = dist.isend(blob_len, dst=receive_from, group=group)
    if sending:
        blob_len_in = torch.zeros(1, dtype=torch.int64)
        dist.recv(blob_len_in, src=send_to, group=group)
        incoming_blob_len = int(blob_len_in[0].item())
    if handle is not None:
        handle.wait()

    handle = None
    if receiving and existing_blob:
        blob_tensor = _wire_tensor(existing_blob)
        handle = dist.isend(blob_tensor, dst=receive_from, group=group)
    if sending and incoming_blob_len:
        blob_buffer = torch.empty(incoming_blob_len, dtype=torch.uint8)
        dist.recv(blob_buffer, src=send_to, group=group)
        peer_existing = dict(_parse_manifest(blob_buffer.numpy().tobytes()))
    if handle is not None:
        handle.wait()

    skip_names = (
        {
            relative
            for relative, size in store_files(source)
            if peer_existing.get(relative) == size
        }
        if sending
        else set()
    )

    outgoing = encoded_size(source, skip=skip_names) if sending else 0
    incoming = 0

    # Sizes first, so the receiver allocates before anything large moves.
    # The wait is deferred past the recv rather than folded into the block
    # above: two ranks both sitting in a blocking send is how the ring stalls.
    handle = None
    if sending:
        size = torch.tensor([outgoing], dtype=torch.int64)
        handle = dist.isend(size, dst=send_to, group=group)
    if receiving:
        size_in = torch.zeros(1, dtype=torch.int64)
        dist.recv(size_in, src=receive_from, group=group)
        incoming = int(size_in[0].item())
    if handle is not None:
        handle.wait()

    mine = (
        fixed_chunks(encode_store(source, chunk=chunk, skip=skip_names), chunk)
        if sending
        else None
    )
    writer = StoreWriter(destination) if receiving else None

    my_chunks = -(-outgoing // chunk) if outgoing else 0
    their_chunks = -(-incoming // chunk) if incoming else 0

    try:
        for index in range(max(my_chunks, their_chunks)):
            handle = None
            outgoing_chunk = None
            # `my_chunks` is zero unless there is something to send, so it
            # already implies `mine`. Said out loud because the implication
            # runs through four assignments, and a reader — or a checker —
            # should not have to reconstruct it to know this is safe.
            if mine is not None and index < my_chunks:
                block = next(mine)
                # No copy on the way out. `fixed_chunks` yields a fresh
                # immutable `bytes` per chunk, so nothing can rewrite this
                # memory while the send is in flight — which is the only thing
                # the copy was buying. The tensor is kept in a local until
                # `wait()` below rather than left to the temporary's lifetime:
                # an isend must own its buffer until it completes.
                outgoing_chunk = _wire_tensor(block)
                handle = dist.isend(outgoing_chunk, dst=send_to, group=group)

            if writer is not None and index < their_chunks:
                expected = min(chunk, incoming - index * chunk)
                buffer = torch.empty(expected, dtype=torch.uint8)
                dist.recv(buffer, src=receive_from, group=group)
                # `.numpy()` shares the tensor's memory; `tobytes()` used to
                # copy it. A fresh buffer is allocated every iteration, so the
                # writer is never handed memory that is about to be reused.
                writer.feed(buffer.numpy())

            if handle is not None:
                handle.wait()

        if writer is None:
            return True
        writer.close()
        return writer.commit()
    finally:
        if writer is not None:
            writer.close()
