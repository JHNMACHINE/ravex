"""Moving a store between machines: the framing, and the socket under it.

Three callers. The replication ring (:mod:`ravex._dist.replication`) sends
each rank's store to a peer on the next machine, so losing a machine does not
lose its shards (GPU-59). A node joining a run is handed the store it needs
before it joins (:mod:`ravex._dist.elastic`, :mod:`ravex._dist.exchange`), in
passes over one connection. And ``exchange_stores`` moves the same framing
over torch collectives instead of a socket.

**Python again, since GPU-267.** From GPU-109 to 0.7.8 this was
``src/transport.rs``; Ravex is now Python only, because the run on one or more
machines is Ravex's and the files are Moonclip's, and neither half needed a
second compiled core. The two things the Rust bought are kept, because they
were measured rather than assumed:

* **The receiving buffer has a cursor.** The Python before GPU-109 did
  ``del pending[:take]``, a memmove of the tail at every file boundary, and the
  transfer fell from 529 MB/s to 113 MB/s as the chunk grew (GPU-84). Here the
  head is an offset, and the buffer is compacted only when the dead prefix is
  worth reclaiming. The common case never touches it: a chunk that falls
  inside an open file is written straight from the caller's memory.
* **The disk and the socket overlap.** Reading and framing ran at 1.2 GB/s,
  the wire at 4.5 GB/s, and the two of them taking turns at 0.99 GB/s
  (2026-09-05, ``bench/transport_ceiling.py``). :func:`send_store` reads on a
  thread of its own, at most :data:`AHEAD` chunks in front of the socket. Both
  a file read and ``sendall`` release the GIL, so a thread is enough.

**The wire format is not new and must not drift.** A replica written by one
version is read back by another, and nodes of different versions talk during
an upgrade. Little-endian throughout::

    header:  u32 count
             per entry: u16 name_len, u64 size, name bytes, u8 already_there
    bodies:  the bytes of every entry whose flag is 0, in header order

A separate, flagless form of the same entry list - :func:`encode_manifest` -
is what a receiver sends *first* to say what it already holds, so the sender
can leave those files out. That one is never itself skippable.

Three failures, and the split is the one the callers make:
``ConnectionError`` when the peer hung up with bytes still owed - a machine
that went away is the ordinary case replication exists to survive; ``OSError``
for a disk or a socket that said no; ``ValueError`` for a stream that is not
the format, which is a bug in whoever wrote it.
"""

from __future__ import annotations

import os
import queue
import struct
import threading
from typing import Iterator, List, Optional, Set, Tuple

#: Bytes per chunk on the wire. Bounded on purpose: a real shard is over a
#: gigabyte, and a whole-store buffer would be that allocation on the training
#: process at checkpoint time.
CHUNK = 4 * 1024 * 1024

#: Written last, deleted first. A replica directory is only trustworthy while
#: this is present: everything else about it - the files, their names, even a
#: manifest - is equally true of a copy that was cut off half way. The name is
#: on disks already, so it stays.
COMPLETE_MARKER = ".ravex-replica-ok"

#: How many chunks the reader thread may run ahead of the socket. One is
#: enough for the disk and the wire to stop taking turns; three is the memory
#: this costs - 12 MiB at the default chunk - bounded because the process
#: holding it is the one training.
AHEAD = 3

#: What one ``recv`` asks for on the receiving side. Larger than a page and
#: smaller than a chunk: each one is a syscall, and the bytes go straight to
#: the file when a file is open.
RECEIVE = 1 << 20

_COUNT = struct.Struct("<I")
#: ``<name length><size>`` ahead of each name.
_ENTRY = struct.Struct("<HQ")
#: After each entry in the transfer header: 1 when the peer already has the
#: file and no bytes for it follow, 0 when they do.
_FLAG = struct.Struct("<B")
#: Big-endian, unlike every other integer here, because that is what the
#: greeting of the pre-staging pair has always packed.
_PREFIX = struct.Struct(">Q")

#: Above this many dead bytes at the front of the pending buffer, and when they
#: are more than half of it, the buffer is compacted.
_COMPACT_AT = 1 << 20

Entry = Tuple[str, int]


# ─── the file list ──────────────────────────────────────────────────────────


def store_files(path: str) -> List[Entry]:
    """Every file under a store, relative to it, with its size. Sorted.

    Sorted so both ends agree on order without exchanging it, and so a failure
    part-way through leaves a prefix rather than a scatter. Names are joined
    with ``/`` on every platform, because they travel.

    A file whose size cannot be read is left out rather than reported: the
    list describes what is readable here, and a directory being walked while a
    peer prunes it is ordinary. So is a name that is not UTF-8, which cannot
    cross the wire, where the format says the bytes are UTF-8.
    """
    found = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            absolute = os.path.join(root, name)
            relative = os.path.relpath(absolute, path).replace(os.sep, "/")
            try:
                relative.encode("utf-8")
                size = os.path.getsize(absolute)
            except (OSError, UnicodeEncodeError):
                continue
            found.append((relative, size))
    found.sort()
    return found


def _on_disk(root: str, relative: str) -> str:
    return os.path.join(root, *relative.split("/"))


# ─── the two encodings of an entry list ─────────────────────────────────────


def encode_manifest(entries: List[Entry]) -> bytes:
    """``(name, size)`` pairs as bytes, no flag: what one side already holds."""
    blob = bytearray(_COUNT.pack(len(entries)))
    for relative, size in entries:
        name = relative.encode("utf-8")
        blob += _ENTRY.pack(len(name), size)
        blob += name
    return bytes(blob)


def parse_manifest(blob: bytes) -> List[Entry]:
    """The inverse of :func:`encode_manifest`. A short blob is a ``ValueError``,
    never a guess at what the missing bytes would have said."""
    if len(blob) < _COUNT.size:
        raise ValueError("manifest shorter than its own count")
    (count,) = _COUNT.unpack_from(blob, 0)
    offset = _COUNT.size
    entries = []
    for _ in range(count):
        if len(blob) < offset + _ENTRY.size:
            raise ValueError("manifest ended inside an entry")
        name_len, size = _ENTRY.unpack_from(blob, offset)
        offset += _ENTRY.size
        if len(blob) < offset + name_len:
            raise ValueError("manifest ended inside a name")
        try:
            name = bytes(blob[offset : offset + name_len]).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("manifest carried a name that is not UTF-8") from None
        offset += name_len
        entries.append((name, size))
    return entries


def _encode_header(entries: List[Entry], skip: Set[str]) -> bytes:
    """The transfer header: the entries, each with a flag saying whether its
    bytes follow."""
    blob = bytearray(_COUNT.pack(len(entries)))
    for relative, size in entries:
        name = relative.encode("utf-8")
        blob += _ENTRY.pack(len(name), size)
        blob += name
        blob += _FLAG.pack(1 if relative in skip else 0)
    return bytes(blob)


def encoded_size(path: str, skip: Optional[Set[str]] = None) -> int:
    """Bytes a transfer of ``path`` will produce, without producing them.

    Must be called with the same ``skip`` the transfer gets, or the two ends
    disagree about the stream's length.
    """
    skip = skip or set()
    total = _COUNT.size
    for relative, size in store_files(path):
        total += _ENTRY.size + _FLAG.size + len(relative.encode("utf-8"))
        if relative not in skip:
            total += size
    return total


# ─── the sending half ───────────────────────────────────────────────────────


def encode_store(path: str, chunk: int = CHUNK, skip: Optional[Set[str]] = None) -> Iterator[bytes]:
    """Yield a store as a byte stream, reading from disk as it goes.

    An iterator rather than a buffer, and read file by file rather than packed
    into a temporary archive first: the archive would be a second full copy on
    a disk that is already the binding constraint on these machines.

    ``skip`` names files whose bytes must not be read or sent - the peer
    already holds them, byte for byte, and said so before this call. They are
    still listed in the header, flagged, so the receiver's manifest and pruning
    stay exactly as complete as a full transfer's; only the body is shorter.

    The blocks are ragged on purpose - a header, then whatever came off each
    file - because ``exchange_stores`` re-cuts them with ``fixed_chunks`` and
    its tests are written against those boundaries.
    """
    skip = skip or set()
    chunk = max(1, chunk)
    entries = store_files(path)
    yield _encode_header(entries, skip)

    for relative, size in entries:
        if relative in skip or size == 0:
            continue
        left = size
        with open(_on_disk(path, relative), "rb") as handle:
            while left:
                block = handle.read(min(chunk, left))
                if not block:
                    # Truncated under us. Padded, so the framing stays aligned
                    # and the file is condemned later: a better failure than a
                    # desynchronised stream.
                    block = bytes(min(chunk, left))
                left -= len(block)
                yield block


def send_store(sock, path: str, chunk: int = CHUNK, skip: Optional[Set[str]] = None) -> int:
    """Push a store down ``sock``, reading the disk on another thread.

    The overlap is the reason this function exists rather than a loop at the
    call site: the transfer pays for the slower of the disk and the socket, not
    for both in series. Returns the bytes written, which is what a caller
    reports and a test asserts on: the point of the skip list is that a second
    round is smaller than the first, and this is the number that says so.
    """
    blocks: "queue.Queue[object]" = queue.Queue(maxsize=AHEAD)
    stop = threading.Event()
    done = object()

    def offer(item: object) -> bool:
        """Queue ``item`` unless the sender has gone; whether it was queued.
        Never a plain ``put``: a sender that failed takes nothing more, and a
        reader waiting on a full queue would keep the join below waiting too."""
        while not stop.is_set():
            try:
                blocks.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def read() -> None:
        try:
            for block in encode_store(path, chunk, skip):
                if not offer(block):
                    return  # the socket gave up; stop reading for nobody
            offer(done)
        except BaseException as exc:  # re-raised on the sending thread
            offer(exc)

    reader = threading.Thread(target=read, name="ravex-store-reader", daemon=True)
    reader.start()
    written = 0
    try:
        while True:
            block = blocks.get()
            if block is done:
                return written
            if isinstance(block, BaseException):
                raise block
            sock.sendall(block)
            written += len(block)  # type: ignore[arg-type]
    finally:
        # Set before the join, which is what lets a reader blocked on a full
        # queue see that nobody will take what it holds.
        stop.set()
        reader.join()


# ─── the receiving half ─────────────────────────────────────────────────────


class StoreWriter:
    """Turns the stream back into files, writing as the bytes arrive.

    Streaming rather than buffering the whole store: the buffer would be a
    second copy of the store on the receiving disk, which then also has to hold
    the store itself.

    Written in place rather than staged in a sibling directory and swapped,
    because disk is the binding constraint on these machines. Safety comes
    instead from :data:`COMPLETE_MARKER`, removed before the first byte lands
    and written only once the last one has.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        # Removed first: from here until `commit`, this directory is not a
        # replica of anything and must not be read as one.
        try:
            os.remove(os.path.join(path, COMPLETE_MARKER))
        except OSError:
            pass
        self._pending = bytearray()
        #: Where the unread part of `_pending` starts: the cursor that replaced
        #: `del pending[:take]` (GPU-84).
        self._at = 0
        self._entries: Optional[List[Tuple[str, int, bool]]] = None
        self._index = 0
        self._left = 0
        self._handle = None
        # Set when a file flagged "already here" turns out not to be, on disk,
        # what the sender was told it was. `complete` must not go true over
        # that: a wrong local file is exactly as unusable as a missing one.
        self._broken = False

    def _buffered(self) -> int:
        return len(self._pending) - self._at

    def _take_front(self, count: int) -> None:
        self._at += count
        if self._at == len(self._pending):
            self._pending.clear()
            self._at = 0
        elif self._at > _COMPACT_AT and self._at * 2 > len(self._pending):
            del self._pending[: self._at]
            self._at = 0

    def feed(self, block) -> None:
        """Take one chunk off the wire. Anything with a contiguous buffer will do.

        ``exchange_stores`` hands this a numpy view over a torch tensor and
        :func:`receive_store` a view of its own buffer; neither is copied when
        the chunk falls inside an open file, which is the normal case because
        the files in a store are large and the chunks are not. Headers and
        file boundaries fall through to the buffered path.
        """
        try:
            view = memoryview(block).cast("B")
        except TypeError:
            raise ValueError("feed() needs a contiguous buffer; this one is strided") from None

        while view and not self._buffered() and self._entries is not None and self._handle is not None:
            take = min(self._left, len(view))
            self._handle.write(view[:take])
            self._left -= take
            view = view[take:]
            if self._left == 0:
                self._finish_file()

        # Falls through even with nothing left over, and that is not a
        # formality: a zero-length file is created by *reaching* it, not by
        # writing to it, so a chunk that ends exactly on a file boundary must
        # still hand control back to `_write_body` or a trailing empty file is
        # never made.
        if view:
            self._pending += view

        while True:
            if self._entries is None and not self._read_header():
                return
            if not self._write_body():
                return

    def _read_header(self) -> bool:
        blob = memoryview(self._pending)[self._at :]
        try:
            if len(blob) < _COUNT.size:
                return False
            (count,) = _COUNT.unpack_from(blob, 0)
            offset = _COUNT.size
            entries = []
            for _ in range(count):
                if len(blob) < offset + _ENTRY.size + _FLAG.size:
                    return False
                name_len, size = _ENTRY.unpack_from(blob, offset)
                offset += _ENTRY.size
                if len(blob) < offset + name_len + _FLAG.size:
                    return False
                try:
                    name = bytes(blob[offset : offset + name_len]).decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError("a file name on the wire is not UTF-8") from None
                offset += name_len
                (flag,) = _FLAG.unpack_from(blob, offset)
                offset += _FLAG.size
                entries.append((name, size, bool(flag)))
        finally:
            # A view left open would stop `_take_front` from resizing the
            # buffer it looks into.
            blob.release()
        self._take_front(offset)
        self._entries = entries
        return True

    def _verify_skip(self, relative: str, size: int) -> None:
        """A file the sender did not send because we said we already had it.

        Said moments earlier, by us - but checked again rather than trusted,
        because a wrong file marked complete is the one failure this whole
        module exists to rule out.
        """
        try:
            ok = os.path.getsize(_on_disk(self.path, relative)) == size
        except OSError:
            ok = False
        if not ok:
            self._broken = True

    def _write_body(self) -> bool:
        assert self._entries is not None
        while True:
            if self._handle is None:
                if self._index >= len(self._entries):
                    return False
                relative, size, already_there = self._entries[self._index]
                if already_there:
                    self._verify_skip(relative, size)
                    self._index += 1
                    continue
                target = _on_disk(self.path, relative)
                os.makedirs(os.path.dirname(target) or self.path, exist_ok=True)
                self._handle = open(target, "wb")
                self._left = size
                if size == 0:
                    self._finish_file()
                    continue
            if not self._buffered():
                return False
            take = min(self._left, self._buffered())
            with memoryview(self._pending) as pending:
                self._handle.write(pending[self._at : self._at + take])
            self._take_front(take)
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
        """Whether every file the header promised is present here, whole."""
        return (
            self._entries is not None
            and self._index >= len(self._entries)
            and self._handle is None
            and not self._broken
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def commit(self) -> bool:
        """Drop what the source no longer has, then mark the copy good.

        Without the pruning the replica only grows: the source is pruned by
        retention and the copy is not, so a long run leaves a copy holding every
        step ever sent while the original holds three (194 files against 3 on
        the bench, 2026-08-19).
        """
        if not self.complete:
            return False
        assert self._entries is not None
        wanted = {relative for relative, _, _ in self._entries}
        for relative, _ in store_files(self.path):
            if relative == COMPLETE_MARKER or relative in wanted:
                continue
            try:
                os.remove(_on_disk(self.path, relative))
            except OSError:
                pass
        try:
            with open(os.path.join(self.path, COMPLETE_MARKER), "wb") as handle:
                handle.write(b"ok")
        except OSError:
            return False
        return True


def receive_store(sock, destination: str) -> bool:
    """Read a store off ``sock`` into ``destination``, until the header says it
    is whole or the peer stops talking. Whether the copy was committed.

    The loop ends on ``complete`` alone: the header already says how much to
    expect, so no length prefix or connection close is needed, and the
    connection is left open for whatever round comes next. An empty read still
    ends the loop rather than spinning, but that is the belt, not the buckle.

    Single-threaded, unlike :func:`send_store`, and the asymmetry is
    deliberate: to overlap the socket with the disk here, the reading side
    would have to know when to stop reading *before* the writing side has
    parsed the header, and a reader that guesses wrong blocks forever on a
    chunk that is never sent.
    """
    writer = StoreWriter(destination)
    buffer = bytearray(RECEIVE)
    view = memoryview(buffer)
    try:
        while not writer.complete:
            count = sock.recv_into(buffer)
            if count == 0:
                break
            writer.feed(view[:count])
        writer.close()
        return writer.commit()
    finally:
        writer.close()


# ─── the pre-staging pair, over a connection someone else owns ──────────────


def _recv_exactly(sock, want: int) -> bytes:
    buffer = bytearray(want)
    view = memoryview(buffer)
    filled = 0
    while filled < want:
        count = sock.recv_into(view[filled:])
        if count == 0:
            raise ConnectionError(
                "peer closed while %d of %d bytes were still expected" % (want - filled, want)
            )
        filled += count
    return bytes(buffer)


def prestage_send(sock, source: str, chunk: int = CHUNK) -> int:
    """Push ``source`` to whatever :func:`prestage_receive` holds at the other
    end of ``sock``, leaving out the files the peer says it already has.
    Returns the bytes written.

    Does not shut the connection down afterwards, on purpose: the header
    already tells the receiver how many files of what size to expect, which is
    what lets it end its read loop without a message boundary from the
    transport. Leaving the connection open is what makes several rounds over
    one socket possible - a coarse pass while the old ranks are still far from
    the cutover, then smaller and smaller deltas, without a new handshake for
    each.
    """
    (manifest_len,) = _PREFIX.unpack(_recv_exactly(sock, _PREFIX.size))
    peer = dict(parse_manifest(_recv_exactly(sock, manifest_len))) if manifest_len else {}
    # Names alone, no hash, and that is safe for the reason GPU-97 gives: the
    # files a store is made of are immutable once written.
    skip = {relative for relative, size in store_files(source) if peer.get(relative) == size}
    return send_store(sock, source, chunk, skip)


def prestage_receive(sock, destination: str) -> bool:
    """The other end of :func:`prestage_send`: say what is already here, then
    take what comes back. Whether the copy was committed."""
    blob = encode_manifest(store_files(destination))
    sock.sendall(_PREFIX.pack(len(blob)) + blob)
    return receive_store(sock, destination)
