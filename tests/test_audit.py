"""The audit trail: the chain, the fingerprints, the digest, and a whole run.

What has to be right regardless of who calls it comes first — above all that
:func:`verify` catches every kind of edit a person could make to the file,
because an audit log whose tampering check misses a deleted line is a log that
says "intact" about whatever someone chose to leave in it. Then whole runs, and
the Moonclip one checks something beyond the entries: that every checkpoint is
still in the store afterwards. The first version of this feature read
Moonclip's manifest file and, on Windows, cost the save being written; a test
that only looked at the audit log would have called that a pass.
"""

import json
from types import SimpleNamespace

import pytest
import torch

import ravex
from ravex import _audit as audit
from ravex._config import RavexConfig


def lines_of(path):
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def log_with_three(tmp_path):
    path = tmp_path / audit.AUDIT_FILE
    log = audit.AuditLog(str(path))
    for step in (4, 8, 12):
        log.append({"step": step, "fingerprint": "f%02d" % step})
    return path


class TestChain:
    def test_an_untouched_log_verifies(self, log_with_three):
        assert audit.verify(str(log_with_three)) == []

    def test_the_first_entry_follows_genesis(self, log_with_three):
        first = json.loads(lines_of(log_with_three)[0])
        assert first["previous_sha256"] == audit.GENESIS

    def test_reopening_continues_the_chain(self, log_with_three):
        audit.AuditLog(str(log_with_three)).append({"step": 16})
        assert audit.verify(str(log_with_three)) == []

    def test_an_edited_entry_is_named(self, log_with_three):
        lines = lines_of(log_with_three)
        entry = json.loads(lines[1])
        entry["fingerprint"] = "forged"
        lines[1] = json.dumps(entry, sort_keys=True)
        write_lines(log_with_three, lines)

        problems = audit.verify(str(log_with_three))
        assert any("line 2 (step 8) was modified" in p for p in problems)

    def test_a_deleted_entry_is_named_where_the_chain_breaks(self, log_with_three):
        lines = lines_of(log_with_three)
        del lines[1]
        write_lines(log_with_three, lines)

        problems = audit.verify(str(log_with_three))
        assert problems == [
            "line 2 (step 12) does not follow the line before it - an entry "
            "was removed, inserted or reordered here"
        ]

    def test_reordered_entries_are_caught(self, log_with_three):
        lines = lines_of(log_with_three)
        lines[1], lines[2] = lines[2], lines[1]
        write_lines(log_with_three, lines)
        assert audit.verify(str(log_with_three))

    def test_an_edit_that_recomputes_its_own_hash_still_breaks_the_next_link(
        self, log_with_three
    ):
        """The forger who read this module: the edited line verifies on its
        own, and the line after it no longer follows it."""
        lines = lines_of(log_with_three)
        entry = json.loads(lines[1])
        entry["fingerprint"] = "forged"
        entry["entry_sha256"] = audit._entry_hash(entry)
        lines[1] = json.dumps(entry, sort_keys=True)
        write_lines(log_with_three, lines)

        problems = audit.verify(str(log_with_three))
        assert any("line 3 (step 12) does not follow" in p for p in problems)

    def test_find_takes_a_prefix(self, log_with_three):
        (found,) = audit.find(str(log_with_three), "F08")
        assert found["step"] == 8


def described(name, hashed, storage="full", dtype="float32", shape=(2, 4)):
    """One tensor as Moonclip's ``describe()`` reports it."""
    tensor = {"name": name, "shape": list(shape), "dtype": dtype, "storage": storage}
    if hashed is not None:
        tensor["hash_raw"] = hashed
    return tensor


class TestFingerprints:
    def test_a_file_fingerprint_is_its_sha256(self, tmp_path):
        import hashlib

        path = tmp_path / "blob"
        path.write_bytes(b"x" * (audit._CHUNK + 17))
        assert audit.file_fingerprint(str(path)) == hashlib.sha256(
            b"x" * (audit._CHUNK + 17)
        ).hexdigest()

    def test_the_order_tensors_are_described_in_does_not_matter(self):
        a, b = described("a", "11"), described("b", "22")
        assert audit.tensor_fingerprint([a, b]) == audit.tensor_fingerprint([b, a])

    def test_how_a_tensor_is_stored_does_not_matter(self):
        """A delta and a full copy of the same values are the same checkpoint."""
        assert audit.tensor_fingerprint([described("a", "11", storage="full")]) == (
            audit.tensor_fingerprint([described("a", "11", storage="delta")])
        )

    def test_a_changed_value_or_shape_changes_it(self):
        base = audit.tensor_fingerprint([described("a", "11")])
        assert audit.tensor_fingerprint([described("a", "12")]) != base
        assert audit.tensor_fingerprint([described("a", "11", shape=(4, 2))]) != base

    def test_a_tensor_without_a_hash_makes_it_none_not_a_partial_answer(self):
        """A Moonclip that predates reporting ``hash_raw``."""
        assert audit.tensor_fingerprint([described("a", "11"), described("b", None)]) is None
        assert audit.tensor_fingerprint([]) is None


class TestConfigDigest:
    def test_credentials_do_not_enter_it(self):
        one, two = RavexConfig(), RavexConfig()
        one.storage.secret_key = "hunter2"
        two.storage.secret_key = "correct horse"
        assert audit.config_digest(one) == audit.config_digest(two)

    def test_a_setting_that_matters_does(self):
        one, two = RavexConfig(), RavexConfig()
        two.checkpoint_every = one.checkpoint_every + 1
        assert audit.config_digest(one) != audit.config_digest(two)


class TestTrail:
    def test_a_step_is_written_only_once_the_next_save_returns(self, tmp_path):
        fingerprinted = []

        def fingerprint(step):
            fingerprinted.append(step)
            return "fp%d" % step, "test"

        trail = audit.AuditTrail(str(tmp_path), RavexConfig(), fingerprint)
        trail.saved(4, {"step": "4"})
        trail._executor.submit(lambda: None).result()
        assert fingerprinted == [], "fingerprinted a checkpoint still being written"

        trail.saved(8, {"step": "8"})
        trail.close()

        entries = audit.read_entries(str(tmp_path / audit.AUDIT_FILE))
        assert [e["step"] for e in entries] == [4, 8]
        assert [e["fingerprint"] for e in entries] == ["fp4", "fp8"]
        assert audit.verify(str(tmp_path / audit.AUDIT_FILE)) == []

    def test_a_fingerprint_that_fails_still_leaves_an_entry(self, tmp_path):
        def fingerprint(step):
            raise OSError("gone")

        trail = audit.AuditTrail(str(tmp_path), RavexConfig(), fingerprint)
        trail.saved(4, {})
        trail.close()

        (entry,) = audit.read_entries(str(tmp_path / audit.AUDIT_FILE))
        assert entry["fingerprint"] is None
        assert entry["fingerprint_kind"] == "unavailable"


class TestMoonclipBackendFingerprint:
    """The backend's half, against a stand-in manager: which snapshot, and what
    an older Moonclip without ``hash_raw`` gets."""

    @staticmethod
    def backend(snapshots, tensors):
        from ravex._backends import MoonclipBackend

        backend = MoonclipBackend.__new__(MoonclipBackend)
        backend._moonclip = SimpleNamespace(__version__="0.1.0")
        backend._manager = SimpleNamespace(
            list_snapshots=lambda: snapshots,
            describe=lambda snapshot_id: {"tensors": tensors[snapshot_id]},
        )
        return backend

    def test_the_snapshot_at_that_step_is_the_one_described(self):
        backend = self.backend(
            [{"step": 3, "id": "a"}, {"step": 6, "id": "b"}],
            {"a": [described("w", "11")], "b": [described("w", "22")]},
        )
        assert backend.fingerprint(6) == (
            audit.tensor_fingerprint([described("w", "22")]),
            "sha256:tensor-xxh3",
        )

    def test_a_step_retention_already_merged_away_is_unavailable(self):
        backend = self.backend([{"step": 6, "id": "b"}], {"b": [described("w", "22")]})
        assert backend.fingerprint(3) == (None, "unavailable")

    def test_an_older_moonclip_is_unavailable_and_said_once(self, caplog, monkeypatch):
        import logging

        logger = logging.getLogger("ravex")
        monkeypatch.setattr(logger, "propagate", False)
        logger.addHandler(caplog.handler)
        try:
            backend = self.backend(
                [{"step": 3, "id": "a"}, {"step": 6, "id": "b"}],
                {"a": [described("w", None)], "b": [described("w", None)]},
            )
            assert backend.fingerprint(3) == (None, "unavailable")
            assert backend.fingerprint(6) == (None, "unavailable")
        finally:
            logger.removeHandler(caplog.handler)
        assert caplog.text.count("does not report tensor hashes") == 1


def audited_run(backend, keep_last=5, steps=6, audit_log=True, seed=0):
    """Six steps checkpointed at 3 and 6, with the audit log as asked."""

    @ravex.train_loop(
        backend=backend, checkpoint_every=3, keep_last=keep_last, audit_log=audit_log
    )
    def train():
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(steps):
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

    train()


def open_moonclip_store(path):
    from ravex._backends import MoonclipBackend

    config = RavexConfig()
    config.storage.path = str(path)
    config.backend = "moonclip"
    config._normalize()
    return MoonclipBackend(config)


class TestAWholeRun:
    def test_off_by_default_writes_nothing(self, storage):
        audited_run("torch_save", audit_log=False)
        assert not (storage / audit.AUDIT_FILE).exists()

    def test_every_torch_save_checkpoint_is_recorded_with_its_files_hash(self, storage):
        audited_run("torch_save")
        path = str(storage / audit.AUDIT_FILE)
        entries = audit.read_entries(path)

        assert [entry["step"] for entry in entries] == [3, 6]
        for entry in entries:
            written = storage / ("step_%012d.pt" % entry["step"])
            assert entry["fingerprint"] == audit.file_fingerprint(str(written))
            assert entry["fingerprint_kind"] == "sha256:file"
            assert entry["metadata"]["step"] == str(entry["step"])
        assert audit.verify(path) == []

    def test_a_file_pruned_straight_away_was_still_hashed_first(self, storage):
        """``keep_last: 1`` deletes step 3's file the moment step 6 lands."""
        audited_run("torch_save", keep_last=1)
        entries = audit.read_entries(str(storage / audit.AUDIT_FILE))

        assert len(list(storage.glob("step_*.pt"))) == 1
        assert [entry["step"] for entry in entries] == [3, 6]
        assert all(entry["fingerprint"] for entry in entries)

    def test_a_moonclip_run_loses_no_checkpoint_and_records_what_moonclip_reports(
        self, storage
    ):
        pytest.importorskip("moonclip")
        audited_run("moonclip")

        store = open_moonclip_store(storage)
        try:
            snapshots = {s["step"]: s["id"] for s in store._manager.list_snapshots()}
            assert {3, 6} <= set(snapshots), "the audit trail cost a checkpoint"

            entries = audit.read_entries(str(storage / audit.AUDIT_FILE))
            assert [entry["step"] for entry in entries] == [3, 6]
            for entry in entries:
                tensors = store._manager.describe(snapshots[entry["step"]])["tensors"]
                expected = audit.tensor_fingerprint(tensors)
                assert entry["fingerprint"] == expected
                assert entry["fingerprint_kind"] == (
                    "sha256:tensor-xxh3" if expected else "unavailable"
                )
        finally:
            store.close()

    def test_the_same_training_fingerprints_the_same_and_different_training_does_not(
        self, tmp_path, monkeypatch
    ):
        """What makes a fingerprint worth writing down: it identifies content."""
        pytest.importorskip("moonclip")

        def fingerprints(name, seed):
            monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / name))
            audited_run("moonclip", seed=seed)
            entries = audit.read_entries(str(tmp_path / name / audit.AUDIT_FILE))
            return [entry["fingerprint"] for entry in entries]

        first = fingerprints("first", seed=0)
        if first[0] is None:
            pytest.skip("the installed Moonclip does not report tensor hashes")
        assert fingerprints("again", seed=0) == first
        assert fingerprints("other", seed=1) != first


class TestTheCommand:
    def test_verify_says_intact_and_prints_the_hash_to_keep(self, storage, capsys):
        from ravex._cli import main

        audited_run("torch_save")
        assert main(["audit", "verify", "--storage", str(storage)]) == 0

        said = capsys.readouterr().out
        last = audit.read_entries(str(storage / audit.AUDIT_FILE))[-1]["entry_sha256"]
        assert "intact: 2 entries, steps 3 to 6" in said
        assert last in said

    def test_verify_fails_on_a_log_with_a_line_removed(self, storage, capsys):
        from ravex._cli import main

        audited_run("torch_save")
        path = storage / audit.AUDIT_FILE
        lines = lines_of(path)
        del lines[0]
        write_lines(path, lines)

        assert main(["audit", "verify", "--storage", str(storage)]) == 1
        assert "does not follow" in capsys.readouterr().out

    def test_find_answers_a_prefix_with_the_step(self, storage, capsys):
        from ravex._cli import main

        audited_run("torch_save")
        first = audit.read_entries(str(storage / audit.AUDIT_FILE))[0]

        assert main(["audit", "find", first["fingerprint"][:12], "--storage", str(storage)]) == 0
        assert '"step": 3' in capsys.readouterr().out
        assert main(["audit", "find", "ffffffffffff", "--storage", str(storage)]) == 1

    def test_list_is_one_line_per_checkpoint(self, storage, capsys):
        from ravex._cli import main

        audited_run("torch_save")
        assert main(["audit", "list", "--storage", str(storage)]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 2 and "sha256:file" in lines[0]

    def test_no_log_is_an_error_that_names_the_setting(self, tmp_path, capsys):
        from ravex._cli import main

        assert main(["audit", "verify", "--storage", str(tmp_path)]) == 2
        assert "audit_log: true" in capsys.readouterr().err
