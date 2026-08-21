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
import struct

#: Bytes per chunk on the wire. Bounded on purpose: a real shard is over a
#: gigabyte, and a whole-store buffer would be that allocation on the training
#: process at checkpoint time — on top of the shadow copy and whatever the
#: allocator is already holding. Host memory pressure right there is measured,
#: not hypothetical; it is what GPU-54 turned out to be.
CHUNK = 4 * 1024 * 1024

#: ``<path length><size>`` ahead of each file, then the bytes.
_ENTRY = struct.Struct("<HQ")
_COUNT = struct.Struct("<I")


def store_files(path: str) -> "list[tuple[str, int]]":
    """Every file under a store, relative to it, with its size. Sorted.

    Sorted so both ends agree on order without exchanging it, and so a failure
    part-way through leaves a prefix rather than a scatter.
    """
    found = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            absolute = os.path.join(root, name)
            try:
                size = os.path.getsize(absolute)
            except OSError:
                continue
            found.append((os.path.relpath(absolute, path).replace(os.sep, "/"), size))
    found.sort()
    return found


def encode_store(path: str, chunk: int = CHUNK):
    """Yield a store as a byte stream, reading from disk as it goes.

    A generator rather than a buffer, and read file by file rather than
    packed into a temporary archive first: the archive would be a second full
    copy on a disk that is already the binding constraint on these machines —
    the reason ``measure_handoff.sh`` has space guards at all.
    """
    entries = store_files(path)
    header = bytearray(_COUNT.pack(len(entries)))
    for relative, size in entries:
        encoded = relative.encode("utf-8")
        header += _ENTRY.pack(len(encoded), size) + encoded
    yield bytes(header)

    for relative, size in entries:
        absolute = os.path.join(path, *relative.split("/"))
        sent = 0
        with open(absolute, "rb") as handle:
            while sent < size:
                block = handle.read(min(chunk, size - sent))
                if not block:
                    # Truncated under us. Pad so the receiver's framing stays
                    # aligned; the manifest hash will condemn the file later,
                    # which is a better failure than a desynchronised stream.
                    block = bytes(size - sent)
                sent += len(block)
                yield block


#: Written last, deleted first. A replica directory is only trustworthy while
#: this is present: everything else about it — the files, their names, even a
#: manifest — is equally true of a copy that was cut off half way.
COMPLETE_MARKER = ".ravex-replica-ok"


def replica_is_complete(path: str) -> bool:
    """Whether a replica directory holds a whole store rather than a torn one.

    The case this exists for is not hypothetical: a transfer is interrupted
    when the peer dies, which is the very failure replication is here to
    survive. A half-written copy that looks like a copy would turn a recovery
    into a wrong model.
    """
    return os.path.exists(os.path.join(path, COMPLETE_MARKER))


class StoreWriter:
    """Turns the stream back into files, writing as the bytes arrive.

    Streaming rather than buffering the whole store and unpacking at the end,
    for the same reason: the buffer would be a second copy of the store on the
    receiving disk, which then also has to hold the store itself.

    Written in place rather than staged in a sibling directory and swapped.
    Staging would be safer to reason about but doubles the replica's footprint
    for the length of every transfer, and disk is the binding constraint on
    these machines. Safety comes instead from :data:`COMPLETE_MARKER`, which is
    removed before the first byte lands and written only once the last one has.
    """

    def __init__(self, path: str):
        self.path = path
        # Removed first: from here until `commit`, this directory is not a
        # replica of anything and must not be read as one.
        try:
            os.remove(os.path.join(path, COMPLETE_MARKER))
        except OSError:
            pass
        self._pending = bytearray()
        self._entries: "list[tuple[str, int]] | None" = None
        self._index = 0
        self._left = 0
        self._handle = None

    def feed(self, block: bytes) -> None:
        self._pending += block
        while True:
            if self._entries is None:
                if not self._read_header():
                    return
            if not self._write_body():
                return

    def _read_header(self) -> bool:
        if len(self._pending) < _COUNT.size:
            return False
        (count,) = _COUNT.unpack_from(self._pending, 0)
        offset = _COUNT.size
        entries = []
        for _ in range(count):
            if len(self._pending) < offset + _ENTRY.size:
                return False
            name_len, size = _ENTRY.unpack_from(self._pending, offset)
            offset += _ENTRY.size
            if len(self._pending) < offset + name_len:
                return False
            entries.append(
                (self._pending[offset : offset + name_len].decode("utf-8"), size)
            )
            offset += name_len
        del self._pending[:offset]
        self._entries = entries
        return True

    def _write_body(self) -> bool:
        assert self._entries is not None
        while True:
            if self._handle is None:
                if self._index >= len(self._entries):
                    return False
                relative, size = self._entries[self._index]
                target = os.path.join(self.path, *relative.split("/"))
                os.makedirs(os.path.dirname(target) or self.path, exist_ok=True)
                self._handle = open(target, "wb")
                self._left = size
                if size == 0:
                    self._finish_file()
                    continue
            if not self._pending:
                return False
            take = min(self._left, len(self._pending))
            self._handle.write(self._pending[:take])
            del self._pending[:take]
            self._left -= take
            if self._left == 0:
                self._finish_file()

    def _finish_file(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._index += 1

    @property
    def complete(self) -> bool:
        """Whether every file the header promised has been written whole."""
        return (
            self._entries is not None
            and self._index >= len(self._entries)
            and self._handle is None
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def commit(self) -> bool:
        """Drop what the source no longer has, then mark the copy good.

        Without the pruning the replica only grows: the source is pruned by
        retention and the copy is not, so a long run leaves a copy holding
        every step ever sent while the original holds three. Seen on the bench
        on 2026-08-19 — 194 files against 3 — and on machines where the disk is
        the binding constraint that is not untidiness, it is the failure.
        """
        if not self.complete:
            return False

        assert self._entries is not None
        wanted = {relative for relative, _ in self._entries}
        for relative, _ in store_files(self.path):
            if relative == COMPLETE_MARKER or relative in wanted:
                continue
            try:
                os.remove(os.path.join(self.path, *relative.split("/")))
            except OSError:
                pass

        try:
            with open(
                os.path.join(self.path, COMPLETE_MARKER), "w", encoding="utf-8"
            ) as handle:
                handle.write("ok")
        except OSError:
            return False
        return True


def encoded_size(path: str) -> int:
    """Bytes :func:`encode_store` will produce, without producing them.

    Needed before the first byte moves: the receiving side allocates buffers
    from this, and both ends have to agree on how many chunks there will be.
    """
    entries = store_files(path)
    total = _COUNT.size
    for relative, size in entries:
        total += _ENTRY.size + len(relative.encode("utf-8")) + size
    return total


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

    The completeness marker stays behind. It says something true about a copy,
    and what is being built here is not one. The owner record does come across:
    it is what lets the resume recognise which history it is continuing, and
    dropping it would undo the check that allowed this promotion.
    """
    if not replica_is_complete(copy_path):
        return False

    parent = os.path.dirname(store_path) or "."
    staging = os.path.join(parent, "." + os.path.basename(store_path) + ".incoming")

    try:
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(parent, exist_ok=True)
        shutil.copytree(copy_path, staging)

        marker = os.path.join(staging, COMPLETE_MARKER)
        if os.path.exists(marker):
            os.remove(marker)

        # The caller only gets here when this rank has no usable store, so what
        # is being removed is a missing or empty directory — and `rename` onto
        # a non-empty one fails on POSIX and on Windows alike.
        shutil.rmtree(store_path, ignore_errors=True)
        os.rename(staging, store_path)
        return True
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        return False


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
    """
    import torch
    import torch.distributed as dist

    sending = source is not None
    receiving = destination is not None
    if not sending and not receiving:
        return True

    outgoing = encoded_size(source) if sending else 0
    incoming = 0

    # Sizes first, so the receiver allocates before anything large moves.
    if sending:
        size = torch.tensor([outgoing], dtype=torch.int64)
        handle = dist.isend(size, dst=send_to, group=group)
    if receiving:
        size_in = torch.zeros(1, dtype=torch.int64)
        dist.recv(size_in, src=receive_from, group=group)
        incoming = int(size_in[0].item())
    if sending:
        handle.wait()

    mine = fixed_chunks(encode_store(source, chunk=chunk), chunk) if sending else None
    writer = StoreWriter(destination) if receiving else None

    my_chunks = -(-outgoing // chunk) if outgoing else 0
    their_chunks = -(-incoming // chunk) if incoming else 0

    try:
        for index in range(max(my_chunks, their_chunks)):
            handle = None
            if index < my_chunks:
                block = next(mine)
                handle = dist.isend(
                    torch.frombuffer(bytearray(block), dtype=torch.uint8),
                    dst=send_to,
                    group=group,
                )

            if index < their_chunks:
                expected = min(chunk, incoming - index * chunk)
                buffer = torch.empty(expected, dtype=torch.uint8)
                dist.recv(buffer, src=receive_from, group=group)
                writer.feed(buffer.numpy().tobytes())

            if handle is not None:
                handle.wait()

        if writer is None:
            return True
        writer.close()
        return writer.commit()
    finally:
        if writer is not None:
            writer.close()
