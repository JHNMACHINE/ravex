"""The store transport (``ravex._dist.transport``) on its own.

These were the Rust tests of ``src/transport.rs`` until GPU-267 brought the
transport back to Python; ``test_dist_replication.py`` covers it as the ring
uses it. A socket here is either a fake over two buffers, where a real one
would only add a thread, or a ``socketpair``, where the point is what a socket
does to a stream.
"""

import os
import socket
import threading

import pytest

from ravex._dist import transport
from ravex._dist.transport import COMPLETE_MARKER


class Wire:
    """A socket over buffers: what is sent is kept, what is received is given."""

    def __init__(self, incoming=b""):
        self.sent = bytearray()
        self.incoming = memoryview(bytes(incoming))

    def sendall(self, data):
        self.sent += data

    def recv_into(self, buffer):
        count = min(len(buffer), len(self.incoming))
        buffer[:count] = self.incoming[:count]
        self.incoming = self.incoming[count:]
        return count


def write(root, relative, data):
    path = os.path.join(root, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def read(root, relative):
    with open(os.path.join(root, *relative.split("/")), "rb") as handle:
        return handle.read()


def round_trip(source, destination, skip=None, chunk=64):
    out = Wire()
    transport.send_store(out, str(source), chunk, skip)
    return transport.receive_store(Wire(out.sent), str(destination))


def test_a_store_arrives_whole(tmp_path):
    write(tmp_path / "s", "a.bin", b"\x07" * 300)
    write(tmp_path / "s", "nested/b.bin", b"\x09" * 5)

    assert round_trip(tmp_path / "s", tmp_path / "d")
    assert read(tmp_path / "d", "a.bin") == b"\x07" * 300
    assert read(tmp_path / "d", "nested/b.bin") == b"\x09" * 5
    assert (tmp_path / "d" / COMPLETE_MARKER).exists()


def test_an_empty_file_still_arrives(tmp_path):
    """The case the buffered path exists for: a zero-length file is created by
    being reached, not by being written to."""
    write(tmp_path / "s", "a.bin", b"\x01" * 128)
    write(tmp_path / "s", "z-empty.bin", b"")

    assert round_trip(tmp_path / "s", tmp_path / "d")
    assert read(tmp_path / "d", "z-empty.bin") == b""


def test_a_skipped_file_is_not_sent_but_is_still_promised(tmp_path):
    write(tmp_path / "s", "kept.bin", b"\x03" * 64)
    write(tmp_path / "s", "already.bin", b"\x04" * 64)
    write(tmp_path / "d", "already.bin", b"\x04" * 64)
    skip = {"already.bin"}

    out = Wire()
    sent = transport.send_store(out, str(tmp_path / "s"), 16, skip)

    assert sent == len(out.sent) == transport.encoded_size(str(tmp_path / "s"), skip)
    assert transport.receive_store(Wire(out.sent), str(tmp_path / "d"))
    assert read(tmp_path / "d", "already.bin") == b"\x04" * 64


def test_a_skip_the_receiver_cannot_honour_is_not_marked_complete(tmp_path):
    """The sender was told the file was here and left it out; it is not here.
    Anything but a refusal turns a recovery into a wrong model."""
    write(tmp_path / "s", "missing.bin", b"\x05" * 32)

    assert not round_trip(tmp_path / "s", tmp_path / "d", skip={"missing.bin"}, chunk=16)
    assert not (tmp_path / "d" / COMPLETE_MARKER).exists()


def test_what_the_source_dropped_is_pruned_from_the_copy(tmp_path):
    write(tmp_path / "s", "current.bin", b"\x01" * 16)
    write(tmp_path / "d", "step_00000001.bin", b"\x02" * 16)

    assert round_trip(tmp_path / "s", tmp_path / "d")
    assert not (tmp_path / "d" / "step_00000001.bin").exists()
    assert (tmp_path / "d" / "current.bin").exists()


def test_the_marker_is_gone_while_the_copy_is_half_written(tmp_path):
    (tmp_path / COMPLETE_MARKER).write_bytes(b"ok")

    writer = transport.StoreWriter(str(tmp_path))
    assert not (tmp_path / COMPLETE_MARKER).exists()
    assert not writer.complete
    assert not writer.commit()


def test_a_stream_split_at_every_byte_parses_the_same(tmp_path):
    """The framing has to survive arriving one byte at a time, because a
    socket is entitled to deliver it that way."""
    write(tmp_path / "s", "a.bin", b"\x06" * 70)
    write(tmp_path / "s", "b/c.bin", b"\x08" * 3)
    out = Wire()
    transport.send_store(out, str(tmp_path / "s"), 1 << 20)

    writer = transport.StoreWriter(str(tmp_path / "d"))
    for at in range(len(out.sent)):
        writer.feed(out.sent[at : at + 1])
    writer.close()
    assert writer.commit()
    assert read(tmp_path / "d", "a.bin") == b"\x06" * 70
    assert read(tmp_path / "d", "b/c.bin") == b"\x08" * 3


def test_a_large_file_in_small_pieces_compacts_the_buffer(tmp_path):
    """Past the compaction threshold the cursor is reset rather than the tail
    moved at every piece (GPU-84): what arrives must still be the file."""
    data = os.urandom(3 * (1 << 20) + 17)
    write(tmp_path / "s", "big.bin", data)
    out = Wire()
    transport.send_store(out, str(tmp_path / "s"))

    writer = transport.StoreWriter(str(tmp_path / "d"))
    # The header alone first, then the body in pieces that never line up with
    # it, so everything goes through the buffered path.
    writer.feed(out.sent[:3])
    for at in range(3, len(out.sent), 65_537):
        writer.feed(out.sent[at : at + 65_537])
    writer.close()
    assert writer.commit()
    assert read(tmp_path / "d", "big.bin") == data


def test_a_strided_buffer_is_refused(tmp_path):
    writer = transport.StoreWriter(str(tmp_path))
    with pytest.raises(ValueError, match="contiguous"):
        writer.feed(memoryview(b"abcdef")[::2])


def test_the_manifest_round_trips():
    entries = [("a.bin", 12), ("nested/b.bin", 0)]
    assert transport.parse_manifest(transport.encode_manifest(entries)) == entries


def test_a_truncated_manifest_is_an_error_not_a_guess():
    blob = transport.encode_manifest([("a.bin", 12)])
    with pytest.raises(ValueError):
        transport.parse_manifest(blob[:-2])


def test_encoded_size_matches_what_is_produced(tmp_path):
    write(tmp_path / "s", "a.bin", b"\x01" * 1000)
    write(tmp_path / "s", "deep/nested/b.bin", b"\x02" * 7)
    out = Wire()
    transport.send_store(out, str(tmp_path / "s"), 128)
    assert len(out.sent) == transport.encoded_size(str(tmp_path / "s"))


def test_the_prestage_pair_moves_a_store_over_a_socket(tmp_path):
    """A real socket rather than two buffers: the pair exists because of what a
    socket does to a stream."""
    write(tmp_path / "s", "a.bin", b"\x02" * 5000)
    write(tmp_path / "s", "b/c.bin", b"\x03" * 17)
    # Already there, byte for byte, so this round must not send it again.
    write(tmp_path / "d", "a.bin", b"\x02" * 5000)

    left, right = socket.socketpair()
    sent = {}
    try:
        sender = threading.Thread(
            target=lambda: sent.update(n=transport.prestage_send(left, str(tmp_path / "s")))
        )
        sender.start()
        ok = transport.prestage_receive(right, str(tmp_path / "d"))
        sender.join(10)
    finally:
        left.close()
        right.close()

    assert ok
    assert read(tmp_path / "d", "b/c.bin") == b"\x03" * 17
    assert sent["n"] == transport.encoded_size(str(tmp_path / "s"), {"a.bin"})
    assert (tmp_path / "d" / COMPLETE_MARKER).exists()


def test_a_peer_that_hangs_up_mid_greeting_is_named_not_guessed_at(tmp_path):
    write(tmp_path / "s", "a.bin", b"\x01" * 8)
    with pytest.raises(ConnectionError, match="peer closed"):
        transport.prestage_send(Wire(b"\x00\x00\x00"), str(tmp_path / "s"))


def test_a_send_that_fails_stops_the_reader(tmp_path):
    """A socket that gives up must not leave the reader thread holding a chunk
    nobody will take, or the send never returns."""
    for n in range(8):
        write(tmp_path / "s", "f%d.bin" % n, b"\x00" * 4096)

    class Broken(Wire):
        def sendall(self, data):
            raise ConnectionResetError("gone")

    finished = threading.Event()

    def send():
        try:
            transport.send_store(Broken(), str(tmp_path / "s"), 1024)
        except ConnectionResetError:
            finished.set()

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    thread.join(10)
    assert finished.is_set()
