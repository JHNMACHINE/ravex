"""A resume whose newest checkpoint will not read (GPU-241).

The unit tests in ``test_resume.py`` decide which step comes next with a fake
store. This one corrupts a real Moonclip pack - the way a full disk or a bad
copy leaves one - and checks that the resumed run still reproduces the
uninterrupted one from the step before, and that the run says so.
"""

import importlib.util
import json
import os

import pytest
import torch

import ravex

from test_e2e_vanilla import CRASH_AT, TOTAL_STEPS, train, weights

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("moonclip") is None, reason="moonclip not installed"
)

OPTIONS = dict(backend="moonclip", checkpoint_every=4, max_steps=TOTAL_STEPS)


def _snapshots(path):
    """``[(step, id, is_delta)]``, oldest first, read through the API.

    Asked of a manager rather than read out of ``manifest.json``, and only
    once the run that wrote the store has stopped: a reader holding Moonclip's
    files while a manager writes is what loses checkpoints on Windows. The
    manager is dropped before the pack is touched.
    """
    import moonclip

    manager = moonclip.MoonclipManager(storage_root=str(path))
    found = [(int(s["step"]), s["id"], bool(s["is_delta"])) for s in manager.list_snapshots()]
    manager.flush()
    del manager
    return sorted(found)


def _corrupt(path, snapshot_id):
    """Every byte of the snapshot's pack inverted, padding included.

    The whole file and not one byte: a pack is padded to its block size, and
    a flip that lands in the padding is one nothing ever reads.
    """
    pack = os.path.join(str(path), "snapshots", snapshot_id, "rank_0.pack")
    with open(pack, "rb") as handle:
        data = handle.read()
    with open(pack, "wb") as handle:
        handle.write(bytes(byte ^ 0xFF for byte in data))


def test_a_corrupt_newest_pack_resumes_from_the_one_before(tmp_path, monkeypatch):
    # ── the run to reproduce ────────────────────────────────────────
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "reference"))
    ravex._activate(**OPTIONS)
    reference_model, reference = train()
    reference_weights = weights(reference_model)
    ravex.deactivate()

    # ── killed at step 20, then the newest checkpoint is damaged ────
    store = tmp_path / "interrupted"
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(store))
    ravex._activate(checkpoint_on_exit=False, **OPTIONS)
    train(die_at=CRASH_AT)
    ravex.deactivate()

    snapshots = _snapshots(store)
    newest_step, newest_id, newest_is_delta = snapshots[-1]
    assert newest_is_delta, "the test means to break one delta, not the window's base"
    _corrupt(store, newest_id)

    # ── the same script again ───────────────────────────────────────
    ravex._activate(**OPTIONS)
    resumed_model, second_half = train()
    ravex.deactivate()

    # Resumed from the snapshot before the broken one, so the run replays one
    # more step than an ordinary resume would - identically.
    previous_step = snapshots[-2][0]
    assert min(second_half) == previous_step + 1
    for step, loss in second_half.items():
        assert loss == reference[step], f"loss at step {step} differs after the fallback"
    for key, value in weights(resumed_model).items():
        assert torch.equal(value, reference_weights[key]), f"{key} differs"

    with open(store / "status.json", encoding="utf-8") as handle:
        status = json.load(handle)
    assert status["notice"] == (
        "The latest checkpoint, step %d, could not be read; resumed from step %d instead."
        % (newest_step, previous_step)
    )

