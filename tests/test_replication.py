"""Choosing a peer to hold a copy, and telling a copy from an original.

The parts of GPU-76 that are pure arithmetic and filesystem layout. What they
protect is a claim: that the copy lands on a *different machine*. A ring that
quietly wraps onto the machine it started from costs the full price of
replication and delivers none of the protection, and — worse — the run would
report itself protected.

The failure being defended against was demonstrated on 2026-08-19: four nodes,
one SIGKILLed mid-run, 2260 steps unrecoverable on three intact disks.
"""

from ravex._backends import replica_store_path, visible_replica_stores
from ravex._config import RavexConfig
from ravex._replication import (
    COMPLETE_MARKER,
    REPLICA_DIR,
    StoreWriter,
    encode_store,
    machine_count,
    machine_of,
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
