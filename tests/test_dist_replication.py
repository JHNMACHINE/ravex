"""Choosing a peer to hold a copy, and telling a copy from an original.

The parts of GPU-76 that are pure arithmetic and filesystem layout. What they
protect is a claim: that the copy lands on a *different machine*. A ring that
quietly wraps onto the machine it started from costs the full price of
replication and delivers none of the protection, and — worse — the run would
report itself protected.

The failure being defended against was demonstrated on 2026-08-19: four nodes,
one SIGKILLed mid-run, 2260 steps unrecoverable on three intact disks.
"""

import os

from ravex._backends import replica_store_path, visible_replica_stores
from ravex._config import RavexConfig
from ravex._dist.replication import (
    COMPLETE_MARKER,
    REPLICA_DIR,
    StoreWriter,
    encode_store,
    encoded_size,
    local_recoveries,
    machine_count,
    machine_of,
    promote_copy,
    recovery_roles,
    replica_is_complete,
    replication_ring,
    store_files,
)


def local_config(path):
    config = RavexConfig()
    config.storage.type = "local"
    config.storage.path = str(path)
    return config


class TestCountingMachines:
    def test_a_uniform_layout_divides(self):
        assert machine_count(world_size=32, local_size=8) == 4

    def test_one_machine_is_one_machine(self):
        assert machine_count(world_size=8, local_size=8) == 1

    def test_an_uneven_layout_cannot_be_told(self):
        """Eight ranks here and four somewhere else is not eight everywhere.

        ``local_size`` describes the node asking, so a world that is not a
        multiple of it says nothing about where the other ranks sit.
        """
        assert machine_count(world_size=12, local_size=8) is None

    def test_nonsense_is_not_a_layout(self):
        assert machine_count(world_size=8, local_size=0) is None
        assert machine_count(world_size=0, local_size=8) is None


class TestTheRing:
    def test_every_rank_sends_one_copy_and_receives_one(self):
        """Traffic is one shard per rank per interval, whatever the size."""
        world, local = 16, 4
        sends = [replication_ring(r, world, local)[0] for r in range(world)]

        assert sorted(sends) == list(range(world)), "the ring is not a permutation"

    def test_the_copy_lands_on_another_machine(self):
        """The whole point, and the thing an off-by-one would silently break."""
        world, local = 16, 4
        for rank in range(world):
            send_to, _ = replication_ring(rank, world, local)
            assert machine_of(send_to, local) != machine_of(rank, local), (
                "rank %d copies to rank %d, which is on its own machine"
                % (rank, send_to)
            )

    def test_receiving_is_the_mirror_of_sending(self):
        world, local = 16, 4
        for rank in range(world):
            _, receive_from = replication_ring(rank, world, local)
            assert replication_ring(receive_from, world, local)[0] == rank

    def test_two_machines_still_form_a_ring(self):
        """The smallest case that protects anything: they hold each other."""
        assert replication_ring(0, world_size=4, local_size=2) == (2, 2)
        assert replication_ring(2, world_size=4, local_size=2) == (0, 0)

    def test_one_machine_declines(self):
        """Not an error — there is no second machine, so there is no copy.

        Going ahead would write the copy onto the disk that holds the original.
        """
        assert replication_ring(0, world_size=8, local_size=8) is None

    def test_an_uneven_layout_declines(self):
        """Refusing beats a copy that cannot be shown to have moved machines."""
        assert replication_ring(0, world_size=12, local_size=8) is None

    def test_a_rank_outside_the_world_declines(self):
        assert replication_ring(9, world_size=8, local_size=2) is None


class TestTellingACopyFromAnOriginal:
    def test_replicas_live_outside_the_rank_namespace(self, tmp_path):
        """Everything that scans ``rank_<n>`` is asking what this machine owns.

        A copy answering that question would be read as an original, which is
        the one mistake that turns redundancy into confusion.
        """
        path = replica_store_path(local_config(tmp_path), 3)

        assert REPLICA_DIR in path
        assert path.endswith("rank_3")

    def test_a_held_copy_is_reported(self, tmp_path):
        replica = tmp_path / REPLICA_DIR / "rank_5"
        replica.mkdir(parents=True)
        (replica / "step_000000000008.pt").write_text("x", encoding="utf-8")

        assert visible_replica_stores(local_config(tmp_path)) == {5}

    def test_an_empty_copy_directory_does_not_count(self, tmp_path):
        (tmp_path / REPLICA_DIR / "rank_5").mkdir(parents=True)

        assert visible_replica_stores(local_config(tmp_path)) == set()

    def test_own_stores_are_not_reported_as_copies(self, tmp_path):
        """The two answers must not bleed into each other."""
        own = tmp_path / "rank_0"
        own.mkdir()
        (own / "step_000000000004.pt").write_text("x", encoding="utf-8")

        assert visible_replica_stores(local_config(tmp_path)) == set()

    def test_nothing_held_is_not_a_failure(self, tmp_path):
        assert visible_replica_stores(local_config(tmp_path)) == set()

    def test_remote_storage_holds_no_copies(self, tmp_path):
        """One bucket every node reads; copying into it protects nothing."""
        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"

        assert visible_replica_stores(config) == set()


def build_store(path, files):
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return path


def move(source, destination, chunk=64):
    """Stream a store across, one chunk at a time, as the wire would."""
    writer = StoreWriter(str(destination))
    for block in encode_store(str(source), chunk=chunk):
        writer.feed(block)
    writer.close()
    return writer


def move_incremental(source, destination, chunk=64):
    """Stream a store across, skipping what the destination already has.

    What `exchange_stores` does over the wire, done in one process: the
    destination's own `store_files` stands in for the manifest it would
    report, the source keeps only the names that match on both name and
    size, and only those bytes travel.
    """
    existing = {relative: size for relative, size in store_files(str(destination))}
    skip = {
        relative
        for relative, size in store_files(str(source))
        if existing.get(relative) == size
    }
    writer = StoreWriter(str(destination))
    for block in encode_store(str(source), chunk=chunk, skip=skip):
        writer.feed(block)
    writer.close()
    return writer


class TestMovingAStore:
    """Streamed both ways, because neither end can afford a second copy.

    A whole-store buffer would be a gigabyte-scale allocation on the training
    process at checkpoint time — the pressure GPU-54 turned out to be — and a
    temporary archive would be a second copy on a disk that is already the
    binding constraint on these machines.
    """

    def test_a_store_arrives_byte_for_byte(self, tmp_path):
        files = {
            "manifest.json": b'{"snapshots": []}',
            "snapshots/abc/rank_0.pack": bytes(range(256)) * 7,
            "nested/deeper/thing.bin": b"x" * 1000,
        }
        source = build_store(tmp_path / "src", files)

        writer = move(source, tmp_path / "dst")

        assert writer.complete
        for relative, content in files.items():
            assert (tmp_path / "dst" / relative).read_bytes() == content

    def test_it_survives_a_chunk_size_that_splits_everything(self, tmp_path):
        """Framing must not depend on chunks lining up with file boundaries."""
        files = {"a.bin": b"0123456789" * 13, "b/c.bin": b"abcdef" * 21}
        source = build_store(tmp_path / "src", files)

        writer = move(source, tmp_path / "dst", chunk=7)

        assert writer.complete
        for relative, content in files.items():
            assert (tmp_path / "dst" / relative).read_bytes() == content

    def test_one_enormous_chunk_is_also_fine(self, tmp_path):
        files = {"a.bin": b"y" * 5000}
        source = build_store(tmp_path / "src", files)

        writer = move(source, tmp_path / "dst", chunk=10_000_000)

        assert writer.complete
        assert (tmp_path / "dst" / "a.bin").read_bytes() == files["a.bin"]

    def test_an_empty_file_still_arrives(self, tmp_path):
        """Zero-length entries are a framing trap: no body follows the header."""
        source = build_store(tmp_path / "src", {"empty": b"", "after.bin": b"z" * 40})

        writer = move(source, tmp_path / "dst")

        assert writer.complete
        assert (tmp_path / "dst" / "empty").read_bytes() == b""
        assert (tmp_path / "dst" / "after.bin").read_bytes() == b"z" * 40

    def test_it_takes_any_buffer_and_not_only_bytes(self, tmp_path):
        """The wire hands over borrowed memory, and copying it was the cost.

        `exchange_stores` passes the received tensor's own memory straight in,
        so `feed` has to accept anything with a buffer. It used to be handed
        `bytes` only, which meant a `tobytes()` copy of every chunk — most of
        what the receiving side spent its time on. Pinned here because the
        signature invites a well-meant narrowing back to `bytes`, and the
        breakage would only show up on a real transfer.
        """
        files = {"big.bin": bytes(range(256)) * 40, "small.bin": b"q" * 3}
        source = build_store(tmp_path / "src", files)

        writer = StoreWriter(str(tmp_path / "dst"))
        for index, block in enumerate(encode_store(str(source), chunk=64)):
            # Alternating, so neither kind is only ever seen mid-file.
            writer.feed(memoryview(block) if index % 2 else bytearray(block))
        writer.close()

        assert writer.complete
        for relative, content in files.items():
            assert (tmp_path / "dst" / relative).read_bytes() == content

    def test_an_empty_store_completes_without_sending_a_body(self, tmp_path):
        source = (tmp_path / "src")
        source.mkdir()

        writer = move(source, tmp_path / "dst")

        assert writer.complete

    def test_a_half_delivered_store_does_not_claim_to_be_whole(self, tmp_path):
        """`complete` is what the caller trusts before swapping a replica in.

        A transfer cut short — a peer that died mid-round is exactly the case
        this feature exists for — must be visibly partial, not silently short.
        """
        source = build_store(tmp_path / "src", {"a.bin": b"q" * 500})

        writer = StoreWriter(str(tmp_path / "dst"))
        blocks = list(encode_store(str(source), chunk=64))
        for block in blocks[:-1]:
            writer.feed(block)
        writer.close()

        assert not writer.complete

    def test_the_file_list_is_ordered_the_same_on_both_sides(self, tmp_path):
        """Sorted, so the ends agree without exchanging the order."""
        source = build_store(
            tmp_path / "src", {"z.bin": b"1", "a.bin": b"2", "m/b.bin": b"3"}
        )

        names = [name for name, _ in store_files(str(source))]

        assert names == sorted(names)
        assert names == ["a.bin", "m/b.bin", "z.bin"]


class TestAReplicaThatCanBeTrusted:
    """A directory full of files is not evidence of a whole copy.

    The transfer is interrupted when the peer dies — which is the very failure
    this feature exists to survive — so "looks like a copy" and "is a copy"
    have to be different questions.
    """

    def test_a_finished_copy_is_marked(self, tmp_path):
        source = build_store(tmp_path / "src", {"a.bin": b"a" * 100})
        destination = tmp_path / "dst"

        writer = move(source, destination)
        assert writer.commit()

        assert replica_is_complete(str(destination))

    def test_a_torn_copy_is_not_marked(self, tmp_path):
        source = build_store(tmp_path / "src", {"a.bin": b"a" * 500})
        destination = tmp_path / "dst"

        writer = StoreWriter(str(destination))
        for block in list(encode_store(str(source), chunk=64))[:-1]:
            writer.feed(block)
        writer.close()

        assert not writer.commit()
        assert not replica_is_complete(str(destination))

    def test_a_new_transfer_unmarks_before_it_writes(self, tmp_path):
        """The old mark must not vouch for the new bytes.

        Between the first byte and the last, the directory is a mixture of two
        stores and is a replica of neither.
        """
        source = build_store(tmp_path / "src", {"a.bin": b"a" * 100})
        destination = tmp_path / "dst"
        move(source, destination).commit()
        assert replica_is_complete(str(destination))

        StoreWriter(str(destination))

        assert not replica_is_complete(str(destination))

    def test_what_the_source_pruned_is_pruned_in_the_copy(self, tmp_path):
        """Found on the bench, 2026-08-19: 194 files in the copy against 3.

        Retention prunes the source and used to leave the copy untouched, so
        the copy grew for the length of the run. On machines where the disk is
        the binding constraint that is the failure, not untidiness.
        """
        source = tmp_path / "src"
        destination = tmp_path / "dst"

        build_store(source, {"step_1.pt": b"1", "step_2.pt": b"2"})
        move(source, destination).commit()
        assert {p.name for p in destination.iterdir()} == {
            "step_1.pt",
            "step_2.pt",
            COMPLETE_MARKER,
        }

        # Retention drops the oldest and the next transfer carries the rest.
        (source / "step_1.pt").unlink()
        build_store(source, {"step_3.pt": b"3"})
        move(source, destination).commit()

        assert {p.name for p in destination.iterdir()} == {
            "step_2.pt",
            "step_3.pt",
            COMPLETE_MARKER,
        }

    def test_a_torn_transfer_leaves_the_previous_copy_unvouched(self, tmp_path):
        """It may not claim the old copy is still good: it is not there any more."""
        source = tmp_path / "src"
        destination = tmp_path / "dst"
        build_store(source, {"a.bin": b"a" * 100})
        move(source, destination).commit()

        bigger = build_store(tmp_path / "src2", {"a.bin": b"b" * 900})
        writer = StoreWriter(str(destination))
        for block in list(encode_store(str(bigger), chunk=64))[:-1]:
            writer.feed(block)
        writer.close()

        assert not replica_is_complete(str(destination))


class TestIncrementalTransfer:
    """GPU-97: a checkpoint's unchanged pack files must not cross the wire twice.

    Safe because moonclip's store files are immutable once written — a name
    that matches on both ends, at the same size, is the same bytes, with no
    hash and no content read needed to know it.
    """

    def test_an_unchanged_file_is_not_resent(self, tmp_path):
        source = tmp_path / "src"
        destination = tmp_path / "dst"
        build_store(source, {"pack.bin": b"p" * 3000, "manifest.json": b"{}"})
        move(source, destination).commit()

        build_store(source, {"manifest.json": b'{"v": 2}'})
        full = encoded_size(str(source))

        writer = move_incremental(source, destination)

        assert writer.commit()
        assert (destination / "pack.bin").read_bytes() == b"p" * 3000
        assert (destination / "manifest.json").read_bytes() == b'{"v": 2}'
        # pack.bin's 3000 bytes did not have to travel a second time.
        assert encoded_size(str(source), skip={"pack.bin"}) < full

    def test_a_name_match_with_a_different_size_still_travels(self, tmp_path):
        """Immutability is the assumption, not something this code can verify
        by itself. Matching on size too is the cheap half of the safety net,
        and this is what it catches: same name, not the same bytes."""
        source = tmp_path / "src"
        destination = tmp_path / "dst"
        build_store(source, {"a.bin": b"x" * 100})
        move(source, destination).commit()

        (source / "a.bin").write_bytes(b"y" * 250)

        writer = move_incremental(source, destination)

        assert writer.commit()
        assert (destination / "a.bin").read_bytes() == b"y" * 250

    def test_a_torn_incremental_transfer_is_not_marked(self, tmp_path):
        """The same guarantee `test_a_torn_copy_is_not_marked` checks for a
        full transfer, with a skipped file sitting in the same header."""
        source = tmp_path / "src"
        destination = tmp_path / "dst"
        build_store(source, {"old.bin": b"a" * 100, "new.bin": b"b" * 500})
        move(source, destination).commit()

        build_store(source, {"new.bin": b"c" * 500})

        writer = StoreWriter(str(destination))
        blocks = list(encode_store(str(source), chunk=64, skip={"old.bin"}))
        for block in blocks[:-1]:
            writer.feed(block)
        writer.close()

        assert not writer.commit()
        assert not replica_is_complete(str(destination))

    def test_pruning_still_reaches_a_skipped_copy(self, tmp_path):
        """The same guarantee `test_what_the_source_pruned_is_pruned_in_the_copy`
        checks for a full transfer: retention drops a step, the incremental
        copy must drop it too, even though the surviving file was skipped
        rather than resent."""
        source = tmp_path / "src"
        destination = tmp_path / "dst"

        build_store(source, {"step_1.pt": b"1", "step_2.pt": b"2"})
        move_incremental(source, destination).commit()
        assert {p.name for p in destination.iterdir()} == {
            "step_1.pt",
            "step_2.pt",
            COMPLETE_MARKER,
        }

        # Retention drops the oldest; step_2.pt is unchanged and skipped.
        (source / "step_1.pt").unlink()
        build_store(source, {"step_3.pt": b"3"})
        move_incremental(source, destination).commit()

        assert {p.name for p in destination.iterdir()} == {
            "step_2.pt",
            "step_3.pt",
            COMPLETE_MARKER,
        }

    def test_skip_does_not_disturb_the_agreed_order(self, tmp_path):
        """The order comes from `store_files` alone; a skip flag must not
        reshuffle it, or a torn transfer would leave a scatter instead of a
        prefix."""
        source = build_store(
            tmp_path / "src",
            {"z.bin": b"1" * 10, "a.bin": b"2" * 10, "m/b.bin": b"3" * 10},
        )
        expected = [name for name, _ in store_files(str(source))]

        writer = StoreWriter(str(tmp_path / "dst"))
        header = next(encode_store(str(source), skip={"a.bin", "z.bin"}))
        writer.feed(header)

        assert [relative for relative, _, _ in writer._entries] == expected


class TestWhoSendsWhatBackToWhom:
    """Recovery runs the ring backwards, and the pairs must match exactly.

    A send posted with no receive waiting hangs at startup, before anyone is
    watching. Both ends decide from the same two lists, so this is where that
    agreement is checked.
    """

    def roles(self, world, local, needs, holds):
        out = {}
        for rank in range(world):
            send_to, receive_from = replication_ring(rank, world, local)
            out[rank] = recovery_roles(rank, send_to, receive_from, needs, holds)
        return out

    def test_nobody_moves_when_every_rank_has_its_store(self):
        world, local = 4, 1
        roles = self.roles(world, local, [False] * 4, [True] * 4)

        assert all(role == (False, False) for role in roles.values())

    def test_one_lost_machine_makes_exactly_one_pair(self):
        """Rank 2's machine was replaced. Rank 3 has been holding its copy."""
        world, local = 4, 1
        needs = [False, False, True, False]
        roles = self.roles(world, local, needs, [True] * 4)

        assert roles[2] == (False, True), "the replaced rank must receive"
        assert roles[3] == (True, False), "its copy holder must send"
        assert roles[0] == (False, False)
        assert roles[1] == (False, False)

    def test_every_send_has_a_receive_and_the_other_way(self):
        """The property that matters; checked over every pattern of loss."""
        import itertools

        world, local = 4, 1
        for pattern in itertools.product([False, True], repeat=world):
            roles = self.roles(world, local, list(pattern), [True] * world)
            for rank in range(world):
                send_to, receive_from = replication_ring(rank, world, local)
                sends, _ = roles[rank]
                _, peer_receives = roles[receive_from]
                assert sends == peer_receives, (
                    "with %s rank %d sends=%s but rank %d receives=%s"
                    % (list(pattern), rank, sends, receive_from, peer_receives)
                )

    def test_a_torn_copy_is_not_offered(self):
        """Rank 2 lost its store and rank 3's copy of it is incomplete.

        Nothing moves: half a store is not a store, and sending it would put a
        checkpoint that cannot be loaded where a missing one used to be.
        """
        world, local = 4, 1
        needs = [False, False, True, False]
        holds = [True, True, True, False]
        roles = self.roles(world, local, needs, holds)

        assert roles[2] == (False, False)
        assert roles[3] == (False, False)

    def test_two_machines_lost_at_once_recover_independently(self):
        world, local = 4, 1
        needs = [True, False, True, False]
        roles = self.roles(world, local, needs, [True] * 4)

        assert roles[0][1] and roles[2][1], "both replaced ranks receive"
        assert roles[1][0] and roles[3][0], "both holders send"

    def test_a_rank_can_both_send_and_receive_in_one_pass(self):
        """Neighbours replaced together: rank 1 needs one and owes one.

        Both happen in the same exchange, which is why the send is posted
        non-blocking rather than completed before the receive begins.
        """
        world, local = 4, 1
        needs = [True, True, False, False]
        roles = self.roles(world, local, needs, [True] * 4)

        assert roles[1] == (True, True)

    def test_nonsense_input_moves_nothing(self):
        assert recovery_roles(0, 1, 3, [], []) == (False, False)
        assert recovery_roles(0, 1, 3, [True, True], [True]) == (False, False)


class TestFetchingBackFromTheBucket:
    """The way back that did not exist until 2026-08-19.

    With a remote configured, a rank whose disk was replaced used to come up
    empty while its data sat in the bucket — and because a resume is agreed at
    the oldest step every rank holds, it took the whole job back to zero.
    Measured on a six-node bench, not reasoned about.
    """

    def runtime(self, monkeypatch, config, backend):
        from ravex._runtime import RavexRuntime

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = config
        runtime._backend = backend
        return runtime

    def capture(self, runtime):
        import logging

        logger = logging.getLogger("ravex")
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            runtime._restore_from_remote_if_empty()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        return "\n".join(records)

    def test_an_empty_local_store_is_refilled(self, tmp_path, monkeypatch):
        class Backend:
            def __init__(self):
                self.asked = False

            def has_checkpoint(self):
                return False

            def restore_from_remote(self):
                self.asked = True
                return True

        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"
        backend = Backend()

        text = self.capture(self.runtime(monkeypatch, config, backend))

        assert backend.asked
        assert "fetched one back from remote storage" in text

    def test_a_store_that_is_there_is_left_alone(self, tmp_path, monkeypatch):
        """A present store is the one this run is writing.

        Overwriting it from the bucket would undo work rather than recover it.
        """

        class Backend:
            def __init__(self):
                self.asked = False

            def has_checkpoint(self):
                return True

            def restore_from_remote(self):
                self.asked = True
                return True

        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"
        backend = Backend()

        self.capture(self.runtime(monkeypatch, config, backend))

        assert not backend.asked, "it pulled over a store that was already here"

    def test_local_storage_is_never_asked(self, tmp_path, monkeypatch):
        """There is no bucket to ask, and the peer ring covers this case."""

        class Backend:
            def __init__(self):
                self.asked = False

            def has_checkpoint(self):
                return False

            def restore_from_remote(self):
                self.asked = True
                return True

        backend = Backend()
        self.capture(self.runtime(monkeypatch, local_config(tmp_path), backend))

        assert not backend.asked

    def test_a_backend_that_raises_does_not_take_the_run_down(
        self, tmp_path, monkeypatch
    ):
        """Failing to fetch means starting from scratch, which is survivable.

        Crashing at startup because a download failed is not.
        """

        class Backend:
            def has_checkpoint(self):
                return False

            def restore_from_remote(self):
                raise OSError("the bucket is having a day")

        config = local_config(tmp_path)
        config.storage.type = "s3"
        config.storage.bucket = "checkpoints"

        text = self.capture(self.runtime(monkeypatch, config, Backend()))

        assert "Could not fetch" in text


# --- a copy that came back on this machine's own disk ---------------


def a_complete_copy(base, rank, run_id, step=20):
    """A replica directory the way a finished transfer leaves one."""
    import json

    path = replica_store_path(local_config(base), rank)
    os.makedirs(os.path.join(path, "snapshots"), exist_ok=True)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"step": step}, handle)
    with open(os.path.join(path, "snapshots", "pack"), "wb") as handle:
        handle.write(b"shard bytes")
    with open(os.path.join(path, ".ravex-owner"), "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "rank": rank, "world_size": 6}, handle)
    with open(os.path.join(path, COMPLETE_MARKER), "w", encoding="utf-8") as handle:
        handle.write("ok")
    return path


def a_store(base, rank, run_id):
    """A rank's own store, with the record that says whose history it is."""
    import json

    path = os.path.join(str(base), "rank_%d" % rank)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"step": 20}, handle)
    with open(os.path.join(path, ".ravex-owner"), "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "rank": rank, "world_size": 6}, handle)
    return path


class TestDecidingWhoCanPromoteACopy:
    """GPU-79: after a reshuffle every copy is present and none is reachable.

    The ring addresses a peer by rank; a copy travels with the disk. Move the
    nodes round by one and each rank is sitting on the copy of its own store,
    while the peer that used to hold it is elsewhere holding somebody else's.
    """

    def test_a_full_reshuffle_lets_everyone_rebuild_locally(self):
        """Nobody has a store, every copy is here, and they agree whose."""
        assert local_recoveries(
            needs=[True] * 6,
            has_own_copy=[True] * 6,
            copy_runs=["run-a"] * 6,
            store_runs=[None] * 6,
        ) == [True] * 6

    def test_the_surviving_stores_name_the_history(self):
        """One rank lost its disk; the others still hold theirs."""
        assert local_recoveries(
            needs=[False, False, True],
            has_own_copy=[False, False, True],
            copy_runs=[None, None, "run-a"],
            store_runs=["run-a", "run-a", None],
        ) == [False, False, True]

    def test_a_copy_from_another_run_is_refused(self):
        """The case this check exists for.

        Promoting it would resume half the shards from one training history
        and half from another - a wrong model, and a silent one. Starting from
        scratch is the better of the two.
        """
        assert local_recoveries(
            needs=[False, False, True],
            has_own_copy=[False, False, True],
            copy_runs=[None, None, "run-from-last-week"],
            store_runs=["run-a", "run-a", None],
        ) == [False, False, False]

    def test_two_histories_among_the_stores_promote_nobody(self):
        """Not hypothetical: two runs left their stores side by side on
        2026-08-19. Which one is being continued is not a thing to guess."""
        assert local_recoveries(
            needs=[False, False, True],
            has_own_copy=[False, False, True],
            copy_runs=[None, None, "run-a"],
            store_runs=["run-a", "run-b", None],
        ) == [False, False, False]

    def test_reshuffled_copies_that_disagree_promote_nobody(self):
        """No store left to arbitrate, and the copies do not agree either."""
        assert local_recoveries(
            needs=[True] * 3,
            has_own_copy=[True] * 3,
            copy_runs=["run-a", "run-a", "run-b"],
            store_runs=[None] * 3,
        ) == [False] * 3

    def test_a_copy_with_no_owner_record_is_not_promoted(self):
        """Unattributable, so it cannot be shown to belong to this history."""
        assert local_recoveries(
            needs=[False, True],
            has_own_copy=[False, True],
            copy_runs=[None, None],
            store_runs=["run-a", None],
        ) == [False, False]

    def test_a_torn_copy_is_not_promoted(self):
        assert local_recoveries(
            needs=[False, True],
            has_own_copy=[False, False],
            copy_runs=[None, "run-a"],
            store_runs=["run-a", None],
        ) == [False, False]

    def test_a_rank_that_has_its_store_promotes_nothing(self):
        """A present store is the one this run is writing over."""
        assert local_recoveries(
            needs=[False, False],
            has_own_copy=[True, True],
            copy_runs=["run-a", "run-a"],
            store_runs=["run-a", "run-a"],
        ) == [False, False]

    def test_nonsense_input_promotes_nothing(self):
        assert local_recoveries([], [], [], []) == []
        assert local_recoveries([True, True], [True], ["a", "a"], [None, None]) == [
            False,
            False,
        ]


class TestPromotingACopy:
    def test_a_copy_becomes_a_store(self, tmp_path):
        copy = a_complete_copy(tmp_path, 0, "run-a")
        store = os.path.join(str(tmp_path), "rank_0")

        assert promote_copy(copy, store)
        assert os.path.exists(os.path.join(store, "manifest.json"))
        assert os.path.exists(os.path.join(store, "snapshots", "pack"))

    def test_the_owner_record_comes_across(self, tmp_path):
        """It is what lets the resume know which history it is continuing -
        and dropping it would undo the check that allowed the promotion."""
        from ravex._dist.identity import run_id_at

        copy = a_complete_copy(tmp_path, 0, "run-a")
        store = os.path.join(str(tmp_path), "rank_0")

        promote_copy(copy, store)

        assert run_id_at(store) == "run-a"

    def test_the_completeness_marker_stays_behind(self, tmp_path):
        """It says something true about a copy, and this is not one."""
        copy = a_complete_copy(tmp_path, 0, "run-a")
        store = os.path.join(str(tmp_path), "rank_0")

        promote_copy(copy, store)

        assert not os.path.exists(os.path.join(store, COMPLETE_MARKER))

    def test_the_copy_is_left_where_it_was(self, tmp_path):
        """Promotion is a copy, not a move: this machine still holds a copy of
        this rank, and the next replication interval expects to find it."""
        copy = a_complete_copy(tmp_path, 0, "run-a")
        store = os.path.join(str(tmp_path), "rank_0")

        promote_copy(copy, store)

        assert replica_is_complete(copy)

    def test_a_torn_copy_is_refused(self, tmp_path):
        copy = a_complete_copy(tmp_path, 0, "run-a")
        os.remove(os.path.join(copy, COMPLETE_MARKER))
        store = os.path.join(str(tmp_path), "rank_0")

        assert not promote_copy(copy, store)
        assert not os.path.exists(store)

    def test_nothing_is_staged_when_it_is_over(self, tmp_path):
        """A leftover staging directory is neither one thing nor the other. It
        is hidden so the scans skip it, but it should not be there at all once
        the promotion has finished."""
        copy = a_complete_copy(tmp_path, 0, "run-a")
        promote_copy(copy, os.path.join(str(tmp_path), "rank_0"))

        leftovers = [n for n in os.listdir(str(tmp_path)) if n.endswith(".incoming")]
        assert leftovers == []

    def test_an_empty_directory_in_the_way_is_replaced(self, tmp_path):
        """`visible_rank_stores` calls an empty directory "no store", so this
        is a rank that needs promoting with a husk where its store goes."""
        copy = a_complete_copy(tmp_path, 0, "run-a")
        store = os.path.join(str(tmp_path), "rank_0")
        os.makedirs(store, exist_ok=True)

        assert promote_copy(copy, store)
        assert os.path.exists(os.path.join(store, "manifest.json"))


class TestBothRoadsHomeLeaveTheSameThing:
    """GPU-85: a store recovered over the wire stayed marked as a copy.

    Two ways exist for a rank to get its store back — promote a copy off this
    machine's own disk, or take one back from a peer — and until this test
    nothing compared them. `StoreWriter.commit` writes the completeness marker
    into whatever directory it filled, because from where it stands it is
    always building a copy; `promote_copy` removes it deliberately. So the
    local road ended clean and the network road ended with a file claiming a
    live store was a finished copy: true for one instant, false from the first
    save onward, and left where a later reader would believe it.

    Found on the first pair of real machines (two RunPod pods, 2026-08-21),
    which is what a divergence costs when no test holds the two side by side.
    """

    def runtime(self, config):
        from ravex._runtime import RavexRuntime

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = config
        runtime._backend = None
        runtime._resume_manager = None
        runtime._storage_split = True
        runtime._byte_transport_ok = True
        runtime._byte_group = None
        runtime._per_rank_active = lambda: True
        runtime._ensure_backend = lambda: True
        return runtime

    def a_peer_sends_the_store_back(self, monkeypatch, tmp_path, rank=0, world=6):
        """Drive the recovery down the network road and hand back the store.

        The stand-in for `exchange_stores` does what the real one does to the
        destination directory and nothing else: it fills it through
        `StoreWriter`, which is the code that writes the marker. Patching the
        marker away here instead would test the mock.
        """
        import ravex._dist.collectives as distributed
        import ravex._dist.replication as replication
        import ravex._runtime as runtime_module
        from ravex._dist.replication import StoreWriter, encode_store

        source = a_store(tmp_path / "elsewhere", rank, "run-a")

        def hand_it_over(src, destination, *args, **kwargs):
            assert destination is not None, "this rank was supposed to receive"
            writer = StoreWriter(destination)
            for block in encode_store(source):
                writer.feed(block)
            writer.close()
            return writer.commit()

        monkeypatch.setattr(runtime_module, "get_rank", lambda: rank)
        monkeypatch.setattr(runtime_module, "get_world_size", lambda: world)
        monkeypatch.setattr(distributed, "local_world_size", lambda: 1)
        monkeypatch.setattr(replication, "exchange_stores", hand_it_over)

        # This rank lost its machine and has no copy of itself on this disk, so
        # the local promotion cannot fire. Every peer is whole and holding the
        # copy the ring gave it, which is what makes one of them able to send.
        def gather(value):
            mine = (True, True, False, None, None)
            theirs = (False, True, False, None, "run-a")
            return [mine if r == rank else theirs for r in range(world)]

        monkeypatch.setattr(distributed, "gather_objects", gather)

        runtime = self.runtime(local_config(tmp_path))
        runtime._recover_missing_stores()
        return os.path.join(str(tmp_path), "rank_%d" % rank)

    def test_the_store_arrives(self, tmp_path, monkeypatch):
        """The recovery itself works, and did before this fix. Asserted so a
        later failure here is not mistaken for the marker regressing."""
        store = self.a_peer_sends_the_store_back(monkeypatch, tmp_path)

        assert os.path.exists(os.path.join(store, "manifest.json"))

    def test_what_came_over_the_wire_is_not_left_marked_as_a_copy(
        self, tmp_path, monkeypatch
    ):
        store = self.a_peer_sends_the_store_back(monkeypatch, tmp_path)

        assert not os.path.exists(os.path.join(store, COMPLETE_MARKER))

    def test_the_two_roads_end_the_same_way(self, tmp_path, monkeypatch):
        """The point of the issue, said as one assertion.

        Same logical act, two transports, and the directory left behind should
        not be able to say which one was taken.
        """
        over_the_wire = self.a_peer_sends_the_store_back(monkeypatch, tmp_path)

        off_the_disk = os.path.join(str(tmp_path), "local", "rank_0")
        promote_copy(a_complete_copy(tmp_path / "local", 0, "run-a"), off_the_disk)

        def marked(path):
            return os.path.exists(os.path.join(path, COMPLETE_MARKER))

        assert marked(over_the_wire) == marked(off_the_disk)

    def test_the_marker_is_gone_only_once_the_store_is_whole(
        self, tmp_path, monkeypatch
    ):
        """It is not dead weight during the transfer, and must not be dropped
        early: while the bytes are still arriving, its absence is the only
        thing separating a half-built store from a finished one. A transfer
        that fails leaves the directory unmarked, which is what says so."""
        import ravex._dist.collectives as distributed
        import ravex._dist.replication as replication
        import ravex._runtime as runtime_module
        from ravex._dist.replication import StoreWriter, encode_store

        source = a_store(tmp_path / "elsewhere", 0, "run-a")
        seen = []

        def cut_off_half_way(src, destination, *args, **kwargs):
            writer = StoreWriter(destination)
            for block in encode_store(source):
                seen.append(os.path.exists(os.path.join(destination, COMPLETE_MARKER)))
                writer.feed(block)
                break
            writer.close()
            return False

        monkeypatch.setattr(runtime_module, "get_rank", lambda: 0)
        monkeypatch.setattr(runtime_module, "get_world_size", lambda: 6)
        monkeypatch.setattr(distributed, "local_world_size", lambda: 1)
        monkeypatch.setattr(replication, "exchange_stores", cut_off_half_way)
        monkeypatch.setattr(
            distributed,
            "gather_objects",
            lambda value: [
                (True, True, False, None, None)
                if r == 0
                else (False, True, False, None, "run-a")
                for r in range(6)
            ],
        )

        runtime = self.runtime(local_config(tmp_path))
        runtime._recover_missing_stores()

        store = os.path.join(str(tmp_path), "rank_0")
        assert seen and not any(seen), "the marker was standing during the transfer"
        assert not os.path.exists(os.path.join(store, COMPLETE_MARKER))


class TestTheRuntimeRebuildsFromItsOwnDisk:
    """The wiring, which is where GPU-79 actually lived.

    Every piece above was correct on its own. What was missing was the runtime
    ever looking at a copy named after itself.
    """

    def runtime(self, config):
        from ravex._runtime import RavexRuntime

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = config
        runtime._backend = None
        runtime._resume_manager = None
        runtime._storage_split = True
        # A gloo job, where the default group already carries host bytes.
        runtime._byte_transport_ok = True
        runtime._byte_group = None
        runtime._per_rank_active = lambda: True
        runtime._ensure_backend = lambda: True
        return runtime

    def run(self, monkeypatch, runtime, rank=0, world=6, gather=None):
        import logging

        import ravex._dist.collectives as distributed
        import ravex._dist.replication as replication
        import ravex._runtime as runtime_module

        monkeypatch.setattr(runtime_module, "get_rank", lambda: rank)
        monkeypatch.setattr(runtime_module, "get_world_size", lambda: world)
        monkeypatch.setattr(distributed, "local_world_size", lambda: 1)
        # Every rank in the same position, which is what a reshuffle is.
        monkeypatch.setattr(
            distributed, "gather_objects", gather or (lambda value: [value] * world)
        )

        def refuse(*args, **kwargs):
            raise AssertionError("it went to the network for bytes already here")

        monkeypatch.setattr(replication, "exchange_stores", refuse)

        logger = logging.getLogger("ravex")
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            runtime._recover_missing_stores()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        return chr(10).join(records)

    def test_the_copy_under_this_rank_is_used(self, tmp_path, monkeypatch):
        """Six intact copies used to buy nothing. This is that, fixed."""
        a_complete_copy(tmp_path, 0, "run-a")
        # A reshuffle leaves another rank's store on this disk as well.
        a_store(tmp_path, 1, "run-a")

        text = self.run(monkeypatch, self.runtime(local_config(tmp_path)))

        assert "nothing had to be fetched" in text
        assert os.path.exists(os.path.join(str(tmp_path), "rank_0", "manifest.json"))

    def test_a_copy_from_another_run_is_left_alone(self, tmp_path, monkeypatch):
        """One rank still holds its own store, so that store names the
        history - and this copy is not from it."""
        a_complete_copy(tmp_path, 0, "run-from-last-week")

        def uneven(value):
            others = (False, False, False, None, "run-a")
            return [tuple(value)] + [others] * 5

        text = self.run(
            monkeypatch, self.runtime(local_config(tmp_path)), gather=uneven
        )

        store = os.path.join(str(tmp_path), "rank_0", "manifest.json")
        assert not os.path.exists(store)
        assert "nothing had to be fetched" not in text

    def test_a_rank_that_has_its_store_is_left_alone(self, tmp_path, monkeypatch):
        a_complete_copy(tmp_path, 0, "run-a")
        store = a_store(tmp_path, 0, "run-a")
        manifest = os.path.join(store, "manifest.json")
        before = open(manifest, encoding="utf-8").read()

        self.run(monkeypatch, self.runtime(local_config(tmp_path)))

        assert open(manifest, encoding="utf-8").read() == before
