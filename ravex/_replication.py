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


def exchange_stores(
    source: str,
    destination: str,
    send_to: int,
    receive_from: int,
    chunk: int = CHUNK,
    group=None,
) -> bool:
    """Trade one store for another around the ring. **Collective in effect.**

    Every rank sends its own store to ``send_to`` and receives one from
    ``receive_from``, at the same time. Blocking sends would deadlock the whole
    ring — each rank would sit in ``send`` waiting for a receiver that is
    itself sitting in ``send`` — so the send is posted non-blocking and waited
    on beside the receive.

    Returns whether the store arrived whole. The caller must not treat that as
    the answer for everybody: a round is only a recovery point if *every* rank
    succeeded, which is a separate agreement.
    """
    import torch
    import torch.distributed as dist

    outgoing = encoded_size(source)
    lengths = torch.zeros(2, dtype=torch.int64)
    lengths[0] = outgoing

    # Sizes first, so the receiver can allocate before anything large moves.
    sending = dist.isend(lengths[0:1].clone(), dst=send_to, group=group)
    dist.recv(lengths[1:2], src=receive_from, group=group)
    sending.wait()
    incoming = int(lengths[1].item())

    mine = fixed_chunks(encode_store(source, chunk=chunk), chunk)
    writer = StoreWriter(destination)

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
        writer.close()
        return writer.commit()
    finally:
        writer.close()
