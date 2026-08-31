"""GPU-92: the coordinated emergency checkpoint on SIGTERM, for sharded models
split across more than one machine.

Two things matter more than the rest here, and get top billing:

* the detection collective failing must degrade to "no emergency this round",
  never propagate, never hang (see ``TestTheDetectionRoundDegradesCleanly``);
* one rank raising its flag has to reach the *other* rank, not just settle
  something locally (see ``TestOneRankIsEnoughToPullEveryoneIn``).

The rest — config plumbing, the signal handler's own minimality, the metadata
flag, the isolated timeout actually firing when a peer disappears mid-round —
is exercised for completeness, at lower stakes.
"""

import os
import signal
import time

import pytest

from ravex._config import RavexConfig
from ravex._runtime import RavexRuntime


# ─── unit level: the fallback discipline, no real torch.distributed ────


class TestTheDetectionRoundDegradesCleanly:
    """A broken emergency channel must behave exactly like no channel at all.

    Nothing here reaches a real process group - the point is that
    `_check_emergency_signal` itself never lets a failure escape, regardless
    of what failed inside it. See CHANGELOG / docs/configuration.md for why
    this matters: the channel adds a new collective to a job that did not
    have one on this path before, and the one invariant that must hold no
    matter what is that a job with the channel off and a job whose channel
    just broke behave identically from here.
    """

    def _bare_runtime(self):
        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = RavexConfig()
        runtime._emergency_group = None
        runtime._emergency_active = True
        runtime._emergency_requested = False
        runtime.registry = type("R", (), {"step_count": 0})()
        return runtime

    def test_a_group_that_cannot_be_built_turns_coordination_off(self, monkeypatch):
        """`emergency_group` reporting `(False, None)` — no gloo, no process
        group, `new_group` itself failing — must not raise, must not attempt
        a checkpoint, and must stop retrying every cadence."""
        runtime = self._bare_runtime()
        calls = []
        runtime.checkpoint = lambda *a, **k: calls.append((a, k)) or True

        monkeypatch.setattr(
            "ravex._distributed.emergency_group", lambda timeout: (False, None)
        )

        runtime._check_emergency_signal()

        assert calls == [], "an unusable group must never lead to a checkpoint attempt"
        assert runtime._emergency_active is False, (
            "must remember the failure instead of rebuilding the group every "
            "cadence for the rest of the run"
        )

    def test_the_detection_collective_raising_is_swallowed(self, monkeypatch):
        """The case this whole mechanism has to survive: the rank that raised
        the flag is gone by the time the others get to the collective, and the
        isolated group's own short timeout fires as a real exception. That
        exception must never reach the training loop."""
        runtime = self._bare_runtime()
        runtime._emergency_group = "already-built"
        runtime._emergency_requested = True
        calls = []
        runtime.checkpoint = lambda *a, **k: calls.append((a, k)) or True

        def boom(local_flag, group):
            raise TimeoutError("simulated: the group's own short timeout fired")

        monkeypatch.setattr("ravex._distributed.emergency_signalled", boom)

        runtime._check_emergency_signal()  # must return, not raise

        assert calls == [], "a failed detection round must not trigger a save"
        # Unlike the group-build failure, a transient collective failure does
        # not permanently disable the channel - the network blip that caused
        # it may not recur, and the next cadence gets its own attempt.
        assert runtime._emergency_active is True

    def test_no_signal_means_no_checkpoint(self, monkeypatch):
        """The ordinary case, paid for at every cadence: the round completes,
        nobody raised a flag, nothing else happens."""
        runtime = self._bare_runtime()
        runtime._emergency_group = "already-built"
        calls = []
        runtime.checkpoint = lambda *a, **k: calls.append((a, k)) or True

        monkeypatch.setattr(
            "ravex._distributed.emergency_signalled", lambda local, group: False
        )

        runtime._check_emergency_signal()

        assert calls == []

    def test_a_signal_triggers_exactly_one_coordinated_checkpoint(self, monkeypatch):
        runtime = self._bare_runtime()
        runtime._emergency_group = "already-built"
        runtime._emergency_requested = True
        calls = []
        runtime.checkpoint = lambda *a, **k: calls.append((a, k)) or True

        monkeypatch.setattr(
            "ravex._distributed.emergency_signalled", lambda local, group: True
        )
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
        monkeypatch.setattr(signal, "signal", lambda sig, handler: None)

        runtime._check_emergency_signal()

        assert calls == [((), {"final": True, "emergency": True})]
        assert killed, "the rank that actually raised the flag must terminate"

    def test_the_preempted_rank_flushes_before_it_dies(self, monkeypatch):
        """GPU-100. The ordering is the whole feature, not a nicety.

        `checkpoint()` hands off to a background writer; killing the process
        before that writer finishes destroys the checkpoint it just claimed
        to have written. Measured on two real machines on 2026-08-31: the
        preempted rank logged "Emergency checkpoint written at step 9" and
        its store held steps 4 and 8. With `per_rank` every rank's shard is
        needed, so the survivor's step 9 was unusable alone and the resume
        fell back to step 8 - the coordinated save bought nothing, and said
        nothing.

        Asserting on the *order* rather than merely that flush was called:
        a flush after the kill is not a flush.
        """
        runtime = self._bare_runtime()
        runtime._emergency_group = "already-built"
        runtime._emergency_requested = True
        runtime.checkpoint = lambda *a, **k: True

        order = []

        class RecordingBackend:
            def flush(self):
                order.append("flush")

        runtime._backend = RecordingBackend()

        monkeypatch.setattr(
            "ravex._distributed.emergency_signalled", lambda local, group: True
        )
        monkeypatch.setattr(os, "kill", lambda pid, sig: order.append("kill"))
        monkeypatch.setattr(signal, "signal", lambda sig, handler: None)

        runtime._check_emergency_signal()

        assert order == ["flush", "kill"], (
            "the preempted rank must wait for the write to land before "
            "terminating; %r" % (order,)
        )

    def test_a_flush_that_fails_still_lets_the_rank_die(self, monkeypatch):
        """The budget is not ours: SIGTERM gives ~10s and then SIGKILL, which
        no handler can catch. A rank that raised out of the flush - or sat in
        it forever - would trade a partial checkpoint for a hang, which is
        worse. It says so and then goes.
        """
        runtime = self._bare_runtime()
        runtime._emergency_group = "already-built"
        runtime._emergency_requested = True
        runtime.checkpoint = lambda *a, **k: True

        class ExplodingBackend:
            def flush(self):
                raise OSError("disk went away")

        runtime._backend = ExplodingBackend()

        monkeypatch.setattr(
            "ravex._distributed.emergency_signalled", lambda local, group: True
        )
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
        monkeypatch.setattr(signal, "signal", lambda sig, handler: None)

        runtime._check_emergency_signal()

        assert killed, "a failed flush must not stop the rank from terminating"

    def test_a_signal_from_another_rank_does_not_kill_this_one(self, monkeypatch):
        """The other half of the same invariant: this rank enters the save
        because *someone* signalled, but it was not the one preempted, so it
        must go back to training rather than terminate itself."""
        runtime = self._bare_runtime()
        runtime._emergency_group = "already-built"
        runtime._emergency_requested = False  # not this rank
        runtime.checkpoint = lambda *a, **k: True

        monkeypatch.setattr(
            "ravex._distributed.emergency_signalled", lambda local, group: True
        )
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

        runtime._check_emergency_signal()

        assert killed == []


class TestEmergencyCoordinationIsDecidedOnce:
    def test_off_by_default_for_a_replicated_job(self, monkeypatch):
        """No sharded models: DDP already gets a reliable local emergency
        checkpoint today (see docs/configuration.md), so the channel that
        exists only to coordinate a sharded save has nothing to do."""
        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = RavexConfig()
        runtime._emergency_active = None
        runtime.registry = type("R", (), {"has_sharded_models": lambda self: False})()

        assert runtime._emergency_coordination_active() is False

    def test_off_on_a_single_machine_even_when_sharded(self, monkeypatch):
        """Every local process gets SIGTERM from the same source at
        effectively the same instant - nothing to coordinate across a
        network channel for."""
        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = RavexConfig()
        runtime._emergency_active = None
        runtime.registry = type("R", (), {"has_sharded_models": lambda self: True})()

        monkeypatch.setattr(
            "ravex._distributed.spans_several_machines", lambda: False
        )

        assert runtime._emergency_coordination_active() is False

    def test_the_config_switch_turns_it_off_independent_of_handle_sigterm(self):
        """`emergency_coordination` is a separate knob from `handle_sigterm`
        on purpose - see the field's own comment in `_config.py`."""
        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = RavexConfig()
        runtime.config.handle_sigterm = True
        runtime.config.emergency_coordination = False
        runtime._emergency_active = None
        runtime.registry = type("R", (), {"has_sharded_models": lambda self: True})()

        assert runtime._emergency_coordination_active() is False

    def test_decided_once_and_cached(self, monkeypatch):
        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = RavexConfig()
        runtime._emergency_active = None
        calls = []

        def has_sharded():
            calls.append(1)
            return True

        runtime.registry = type("R", (), {"has_sharded_models": lambda self: has_sharded()})()
        monkeypatch.setattr("ravex._distributed.spans_several_machines", lambda: True)

        first = runtime._emergency_coordination_active()
        second = runtime._emergency_coordination_active()

        assert first is True and second is True
        assert len(calls) == 1, "must not re-derive this every time it is asked"


@pytest.mark.skipif(
    os.name == "nt", reason="Ravex does not install a SIGTERM handler on Windows"
)
def test_the_signal_handler_only_raises_a_flag(monkeypatch):
    """The handler must do the one thing that is safe inside a signal context
    and nothing else - no collective, no I/O, no call into `checkpoint`. See
    the docstring on `_install_signal_handler` for why: this used to call
    `shutdown()` directly from here, which risks entering a second collective
    from inside a context that may have interrupted a first one already in
    flight."""
    captured = {}
    monkeypatch.setattr(
        signal, "signal", lambda sig, fn: captured.__setitem__(sig, fn)
    )
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)

    config = RavexConfig()
    config.handle_sigterm = True
    runtime = RavexRuntime(config)

    runtime._install_signal_handler()

    assert signal.SIGTERM in captured
    assert runtime._emergency_requested is False

    captured[signal.SIGTERM](signal.SIGTERM, None)

    assert runtime._emergency_requested is True


# ─── the metadata flag ──────────────────────────────────────────────────


def test_emergency_checkpoints_are_flagged_in_the_manifest(tmp_path, monkeypatch):
    """`emergency=True` reaches the same generic metadata map `final=True`
    already does - no backend or manifest change needed, see CHANGELOG."""
    import torch

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("RAVEX_BACKEND", "torch_save")
    config = RavexConfig.load()
    runtime = RavexRuntime(config)

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runtime.registry.register_model(model)
    runtime.registry.register_optimizer(optimizer)
    runtime.registry.step_count = 3

    assert runtime.checkpoint(final=True, emergency=True)
    runtime._backend.flush()

    saved = torch.load(
        tmp_path / "checkpoints" / "step_000000000003.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["_ravex_metadata"]["emergency"] == "true"
    assert saved["_ravex_metadata"]["final"] == "true"

    runtime._backend.close()


# ─── config plumbing ────────────────────────────────────────────────────


def test_emergency_defaults():
    config = RavexConfig()
    assert config.emergency_coordination is True
    assert config.emergency_check_every == 1
    assert config.emergency_timeout == 20


def test_emergency_env_overrides(monkeypatch):
    monkeypatch.setenv("RAVEX_EMERGENCY_COORDINATION", "0")
    monkeypatch.setenv("RAVEX_EMERGENCY_CHECK_EVERY", "5")
    monkeypatch.setenv("RAVEX_EMERGENCY_TIMEOUT", "45")

    config = RavexConfig.load()

    assert config.emergency_coordination is False
    assert config.emergency_check_every == 5
    assert config.emergency_timeout == 45


def test_emergency_numeric_fields_are_clamped(monkeypatch):
    monkeypatch.setenv("RAVEX_EMERGENCY_CHECK_EVERY", "0")
    monkeypatch.setenv("RAVEX_EMERGENCY_TIMEOUT", "-5")

    config = RavexConfig.load()

    assert config.emergency_check_every == 1
    assert config.emergency_timeout == 1


# ─── multi-rank: real torch.distributed, gloo, spawned processes ───────


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _emergency_worker(rank, world_size, port, local_flag, timeout_seconds, out):
    """One rank running the real detection primitive, start to finish."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    try:
        import torch.distributed as dist

        from ravex._distributed import emergency_group, emergency_signalled

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            usable, group = emergency_group(timeout_seconds)
            signalled = emergency_signalled(local_flag, group) if usable else None
        finally:
            dist.destroy_process_group()
        out.put((rank, usable, signalled))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _run_emergency_across_two_ranks(flag_by_rank, timeout_seconds=20):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_emergency_worker,
            args=(rank, 2, port, flag_by_rank[rank], timeout_seconds, out),
        )
        for rank in (0, 1)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, usable, signalled = out.get(timeout=120)
            results[rank] = (usable, signalled)
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():  # pragma: no cover - a hung collective
                p.terminate()
                p.join(timeout=10)
    return results


class TestOneRankIsEnoughToPullEveryoneIn:
    """The reason this channel exists at all: a SIGTERM landing on one rank
    has to be visible to a rank that never received anything, since it is the
    survivor - not the one that caught the signal - whose participation the
    coordinated save actually depends on."""

    def test_only_rank_zero_signals_and_both_detect_it(self):
        results = _run_emergency_across_two_ranks({0: True, 1: False})

        assert set(results) == {0, 1}, results
        for rank, (usable, signalled) in results.items():
            assert usable is True, f"rank {rank}: group was not usable"
            assert signalled is True, (
                f"rank {rank} did not see the other rank's flag: {signalled!r}"
            )

    def test_only_rank_one_signals_and_both_detect_it(self):
        """The mirror case, so this is not an artefact of rank ordering."""
        results = _run_emergency_across_two_ranks({0: False, 1: True})

        for rank, (usable, signalled) in results.items():
            assert usable is True
            assert signalled is True, f"rank {rank}: {signalled!r}"

    def test_neither_signals_and_neither_saves(self):
        results = _run_emergency_across_two_ranks({0: False, 1: False})

        for rank, (usable, signalled) in results.items():
            assert usable is True
            assert signalled is False, f"rank {rank}: {signalled!r}"

    def test_both_signal_and_both_detect_it(self):
        results = _run_emergency_across_two_ranks({0: True, 1: True})

        for rank, (usable, signalled) in results.items():
            assert usable is True
            assert signalled is True


def _emergency_partial_worker(rank, world_size, port, timeout_seconds, out):
    """Both ranks build the channel together; only rank 0 then waits on the
    detection round, while rank 1 vanishes without joining it - the process
    exits right after the group exists, the way a killed spot instance would
    disappear mid-round rather than declining to participate."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    try:
        import torch.distributed as dist

        from ravex._distributed import emergency_group, emergency_signalled

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        usable, group = emergency_group(timeout_seconds)

        if rank != 0:
            out.put((rank, "left-without-joining", 0.0))
            return  # no destroy_process_group: simulates vanishing, not exiting cleanly

        started = time.perf_counter()
        try:
            emergency_signalled(True, group)
            out.put((rank, "returned", time.perf_counter() - started))
        except Exception as exc:
            out.put((rank, type(exc).__name__, time.perf_counter() - started))
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        out.put((rank, "error", f"{type(exc).__name__}: {exc}"))


class TestTheIsolatedTimeoutActuallyBounds:
    """Constraint: if the preempted rank disappears before the others reach
    the collective, the result must be the channel's own short timeout, not
    the 30-minute default the main process group runs under. This is the one
    claim in the whole design that a mock cannot stand in for - it depends on
    what `dist.new_group(timeout=...)` actually does, not on what this
    project's code around it assumes it does.
    """

    def test_a_vanished_peer_is_bounded_by_the_short_timeout_not_the_default(self):
        timeout_seconds = 3
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        port = _free_port()
        procs = [
            ctx.Process(
                target=_emergency_partial_worker,
                args=(rank, 2, port, timeout_seconds, out),
            )
            for rank in (0, 1)
        ]
        for p in procs:
            p.start()
        try:
            # Generous outer bound: this must fail the test loudly, by timing
            # out `Queue.get`, rather than hang the suite, if the isolation
            # does not hold and rank 0 falls back to the default timeout.
            results = {}
            for _ in procs:
                rank, outcome, elapsed = out.get(timeout=90)
                results[rank] = (outcome, elapsed)
        finally:
            for p in procs:
                p.join(timeout=15)
                if p.is_alive():  # pragma: no cover - the failure this test guards against
                    p.terminate()
                    p.join(timeout=10)

        outcome, elapsed = results[0]
        assert outcome != "returned", (
            "rank 0 should not see a clean answer when its only peer vanished "
            "before joining the round"
        )
        assert elapsed < timeout_seconds + 30, (
            f"rank 0 took {elapsed:.1f}s to notice the peer was gone - the "
            f"isolated {timeout_seconds}s timeout does not appear to be in "
            "effect (the main group's default is 30 *minutes*)"
        )
