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
        runtime.registry = type(
            "R",
            (),
            # `sharded_groups` since GPU-125: the group build asks which ranks
            # share a model before it decides how wide to make itself.
            {"step_count": 0, "sharded_groups": staticmethod(lambda: [])},
        )()
        return runtime

    def test_a_group_that_cannot_be_built_turns_coordination_off(self, monkeypatch):
        """`emergency_group` reporting `(False, None)` — no gloo, no process
        group, `new_group` itself failing — must not raise, must not attempt
        a checkpoint, and must stop retrying every cadence."""
        runtime = self._bare_runtime()
        calls = []
        runtime.checkpoint = lambda *a, **k: calls.append((a, k)) or True

        monkeypatch.setattr(
            "ravex._dist.collectives.emergency_group",
            lambda timeout, partition=None: (False, None),
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

        monkeypatch.setattr("ravex._dist.collectives.emergency_signalled", boom)

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
            "ravex._dist.collectives.emergency_signalled", lambda local, group: False
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
            "ravex._dist.collectives.emergency_signalled", lambda local, group: True
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
            "ravex._dist.collectives.emergency_signalled", lambda local, group: True
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
            "ravex._dist.collectives.emergency_signalled", lambda local, group: True
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
            "ravex._dist.collectives.emergency_signalled", lambda local, group: True
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
            "ravex._dist.collectives.spans_several_machines", lambda: False
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
        monkeypatch.setattr("ravex._dist.collectives.spans_several_machines", lambda: True)

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

        from ravex._dist.collectives import emergency_group, emergency_signalled

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

        from ravex._dist.collectives import emergency_group, emergency_signalled

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


# ─── GPU-111: the announced round, and what it costs when nothing happens ──


class FakeStore:
    """A store with the three methods this road uses, and nothing else."""

    def __init__(self):
        self.values = {}
        self.checks = 0

    def set(self, key, value):
        self.values[key] = value

    def check(self, keys):
        self.checks += 1
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]


class TestTheAnnouncedRoundIsAStepNumber:
    """The alarm says *when*, not *whether*, and that is the whole protocol.

    A bool tells each rank that somebody wants to save, and each of them
    learns it at a different moment; acting on it immediately is how some
    ranks enter a collective save that the others reach three steps later or
    never. A step number is learned at different moments and acted on at the
    same one.
    """

    def test_the_round_is_rounded_up_to_a_step_the_others_look_at(self):
        from ravex._dist.agreement import announce_emergency, announced_emergency

        store = FakeStore()
        # Cadence 10: the other ranks look at steps 10, 20, 30... so an alarm
        # raised at 23 has to name 40 and not 25, which nobody attends.
        assert announce_emergency(store, 23, cadence=10, lead=2) == 40
        assert announced_emergency(store) == 40

    def test_the_lead_is_counted_in_checks_and_not_in_steps(self):
        from ravex._dist.agreement import announce_emergency

        store = FakeStore()
        assert announce_emergency(store, 100, cadence=1, lead=2) == 102
        store.values.clear()
        assert announce_emergency(store, 100, cadence=5, lead=2) == 110

    def test_no_alarm_reads_as_no_alarm_rather_than_as_an_error(self):
        from ravex._dist.agreement import announced_emergency

        assert announced_emergency(FakeStore()) is None

    def test_a_store_that_refuses_the_write_is_not_an_emergency_either(self):
        """Every failure on this path has to look like a job that never had
        the channel — the one invariant this whole file is built on."""
        from ravex._dist.agreement import announce_emergency

        class Refuses(FakeStore):
            def set(self, key, value):
                raise OSError("store is gone")

        assert announce_emergency(Refuses(), 10, cadence=1) is None


def _announced_worker(rank, world, port, plan, out):
    """One rank running the real `_check_emergency_signal` over a real store.

    Built with `__new__` rather than through a decorated training loop: what
    is under test is the protocol, not FSDP. The barrier at the top of each
    step stands in for the gradient all-reduce, which is what makes a step
    number mean the same thing on every rank — and is exactly the assumption
    the design rests on. Where it does not hold, see GPU-125.
    """
    import time
    from unittest import mock

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    try:
        import torch.distributed as dist

        import ravex._dist.collectives as collectives
        import ravex._runtime as runtime_module
        from ravex._config import RavexConfig
        from ravex._runtime import RavexRuntime

        dist.init_process_group("gloo", rank=rank, world_size=world)

        runtime = RavexRuntime.__new__(RavexRuntime)
        runtime.config = RavexConfig()
        runtime.config.emergency_check_every = plan["cadence"]
        runtime.config.emergency_timeout = 20
        runtime.config.emergency_transport = plan["transport"]
        runtime._emergency_group = None
        runtime._emergency_active = True
        runtime._emergency_requested = False
        runtime._emergency_store = None
        runtime._emergency_round = None
        runtime._emergency_settled = False
        runtime.registry = type(
            "R",
            (),
            # `sharded_groups` since GPU-125: the group build asks which ranks
            # share a model before it decides how wide to make itself.
            {"step_count": 0, "sharded_groups": staticmethod(lambda: [])},
        )()

        saved = []
        runtime.checkpoint = lambda *a, **k: (
            saved.append(runtime.registry.step_count) or True
        )
        runtime._flush_before_dying = lambda: None

        # How many times the detection collective was actually posted, which is
        # the number this whole change is about: in a run nobody preempts it
        # used to be one per step.
        posted = []
        real_signalled = collectives.emergency_signalled
        collectives.emergency_signalled = lambda flag, group: (
            posted.append(1) or real_signalled(flag, group)
        )

        # The preempted rank reaches `os.kill` at the end of the real path.
        # Patched rather than routed around, so what runs here is the code
        # that ships; this process is about to exit anyway.
        with mock.patch.object(runtime_module.os, "kill"), mock.patch.object(
            runtime_module.signal, "signal"
        ):
            for step in range(1, plan["steps"] + 1):
                dist.barrier()
                if rank == plan["slow_rank"]:
                    time.sleep(plan["skew_seconds"])
                runtime.registry.step_count = step
                if rank == plan["preempt_rank"] and step == plan["preempt_at"]:
                    runtime._emergency_requested = True
                if step % plan["cadence"] == 0:
                    runtime._check_emergency_signal()

        collectives.emergency_signalled = real_signalled
        dist.destroy_process_group()
        out.put((rank, saved, len(posted), runtime._emergency_round))
    except BaseException:  # reported, never a silent empty queue
        import traceback

        out.put((rank, None, None, traceback.format_exc()))


def _run_announced(plan, world=4):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_announced_worker, args=(rank, world, port, plan, out))
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    results = {}
    try:
        for _ in procs:
            rank, saved, posted, announced = out.get(timeout=180)
            if saved is None:
                pytest.fail("rank %d failed: %s" % (rank, announced))
            results[rank] = (saved, posted, announced)
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - a hung collective
                proc.terminate()
                proc.join(timeout=10)
    return results


BASE_PLAN = {
    "steps": 20,
    "cadence": 1,
    "preempt_rank": 2,
    "preempt_at": 8,
    "slow_rank": 1,
    "skew_seconds": 0.005,
    "transport": "auto",
}


class TestTheAnnouncedRoundUnderDelay:
    """The bench GPU-111 asked for before any of this was allowed to ship.

    Four ranks, one of them 5 ms late on every step on purpose, because a
    straggler is the normal case on the hardware this project runs on and not
    the edge one. Two questions, which are the two halves of the change: does
    it still agree, and does a run that is *not* being preempted stop paying.
    """

    def test_every_rank_saves_at_the_same_step_with_one_rank_running_late(self):
        results = _run_announced(dict(BASE_PLAN))

        announced = {row[2] for row in results.values()}
        assert announced == {BASE_PLAN["preempt_at"] + 2}, (
            "the ranks did not agree on when to save: %s" % announced
        )

        for rank, (saved, _posted, _round) in results.items():
            assert saved == [BASE_PLAN["preempt_at"] + 2], (
                "rank %d saved at %s, not once at the announced step"
                % (rank, saved)
            )

    def test_the_collective_is_posted_once_and_not_once_per_step(self):
        """The measurement, turned into an assertion.

        Twenty steps used to be twenty `all_reduce`s on every rank. What is
        left is one, at the announced step, where it is what proves everybody
        arrived before any of them enters a sharded save alone.
        """
        results = _run_announced(dict(BASE_PLAN))

        for rank, (_saved, posted, _round) in results.items():
            assert posted == 1, (
                "rank %d posted the detection collective %d time(s) over %d "
                "steps; the point of the announced round is that it posts it "
                "at the announced step and nowhere else"
                % (rank, posted, BASE_PLAN["steps"])
            )

    def test_a_run_nobody_preempts_posts_no_collective_at_all(self):
        """The common case, which is every step of almost every run."""
        results = _run_announced(dict(BASE_PLAN, preempt_rank=-1))

        for rank, (saved, posted, announced) in results.items():
            assert posted == 0, (
                "rank %d posted %d collective(s) with nothing to detect"
                % (rank, posted)
            )
            assert saved == [], "rank %d saved without an emergency: %s" % (
                rank,
                saved,
            )
            assert announced is None

    def test_the_road_back_still_asks_every_step_and_still_agrees(self):
        """`emergency_transport: collectives` is what this channel used to be.

        Kept for a job that would rather pay a collective per step than have
        its preemption path depend on a key-value store, and worth a test of
        its own for a second reason: it is what says the counts above are
        measuring the announced round and not the harness. Same plan, same
        ranks, same straggler — twenty posts instead of one.
        """
        results = _run_announced(dict(BASE_PLAN, transport="collectives"))

        for rank, (saved, posted, announced) in results.items():
            assert posted == BASE_PLAN["steps"], (
                "rank %d posted %d time(s) on the collective road, not once "
                "per step" % (rank, posted)
            )
            assert announced is None, (
                "rank %d announced a round on the road that does not announce"
                % rank
            )
            # From the step the flag went up onwards: the collective road has
            # no lead, so detection is the very next check.
            assert saved and saved[0] == BASE_PLAN["preempt_at"], (
                "rank %d saved at %s, not at the step the flag went up"
                % (rank, saved)
            )


# ─── GPU-125: how wide the detection channel is allowed to be ──────────────


def _partition_worker(rank, world, port, plan, out):
    """One rank deciding how wide its SIGTERM channel should be.

    Two machines are simulated with `GROUP_RANK`, which is what torchrun sets
    and what `machine_key` reads. `shard_group_ranks` is replaced rather than
    driven through real FSDP: what is under test is the partition and the
    group built from it, and standing up FSDP on gloo to obtain a process
    group whose ranks are already known would be testing torch.
    """
    import time

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GROUP_RANK"] = str(rank // plan["per_node"])
    os.environ["LOCAL_WORLD_SIZE"] = str(plan["per_node"])

    try:
        import torch.distributed as dist

        import ravex._dist.collectives as collectives

        dist.init_process_group("gloo", rank=rank, world_size=world)

        node = rank // plan["per_node"]
        if plan["sharding"] == "per_node":
            collectives.shard_group_ranks = lambda model: [
                other for other in range(world) if other // plan["per_node"] == node
            ]
        else:
            collectives.shard_group_ranks = lambda model: list(range(world))

        partition = collectives.emergency_partition(object())
        usable, group = collectives.emergency_group(plan["timeout"], partition)
        members = (
            dist.get_process_group_ranks(group)
            if usable and group is not None
            else None
        )

        # The property the whole issue is about: with the channel confined to
        # a machine, a machine that stops taking part costs the other machine
        # nothing. Node 1 simply walks away here, which is what a node running
        # its own inner steps looks like from node 0.
        answered, elapsed = None, None
        if usable and plan["detect"]:
            if node == 0:
                began = time.perf_counter()
                try:
                    answered = collectives.emergency_signalled(rank == 0, group)
                except Exception as exc:
                    answered = "%s: %s" % (type(exc).__name__, exc)
                elapsed = time.perf_counter() - began
            else:
                time.sleep(plan["walk_away_seconds"])

        dist.destroy_process_group()
        out.put((rank, partition, members, answered, elapsed))
    except BaseException:
        import traceback

        out.put((rank, None, None, traceback.format_exc(), None))


def _run_partition(plan, world=4):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_partition_worker, args=(rank, world, port, plan, out))
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    results = {}
    try:
        for _ in procs:
            rank, partition, members, answered, elapsed = out.get(timeout=180)
            if members is None and partition is None and isinstance(answered, str):
                pytest.fail("rank %d failed: %s" % (rank, answered))
            results[rank] = (partition, members, answered, elapsed)
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - a hung collective
                proc.terminate()
                proc.join(timeout=10)
    return results


class TestTheChannelIsNoWiderThanWhatItProtects:
    """GPU-125. The save this channel coordinates is collective on the
    *model's* group — `get_state_dict` is handed no `process_group` — so the
    ranks that must enter it together are the ranks holding shards of the same
    model. The channel was one group over every rank of the job, which put
    every rank on every other machine into a per-step collective it had no
    stake in.
    """

    def test_sharding_that_stays_on_a_machine_gets_a_group_per_machine(self):
        results = _run_partition(
            {"per_node": 2, "sharding": "per_node", "detect": False,
             "walk_away_seconds": 0.0, "timeout": 20}
        )

        for rank, (partition, members, _answered, _elapsed) in results.items():
            assert partition == [[0, 1], [2, 3]], (
                "rank %d partitioned the job as %s" % (rank, partition)
            )
            assert members == [0, 1] if rank < 2 else members == [2, 3], (
                "rank %d landed in the group %s" % (rank, members)
            )

    def test_sharding_that_crosses_machines_keeps_the_whole_job(self):
        """Not a fallback, and not a regression to be fixed later: when the
        shards themselves span machines, the save's own collective spans them,
        so the detection has to as well. The defect was never the wide group —
        it was the wide group when the sharding was not."""
        results = _run_partition(
            {"per_node": 2, "sharding": "global", "detect": False,
             "walk_away_seconds": 0.0, "timeout": 20}
        )

        for rank, (partition, members, _answered, _elapsed) in results.items():
            assert partition is None, "rank %d split a job it should not" % rank
            assert members == [0, 1, 2, 3], "rank %d: %s" % (rank, members)

    def test_a_machine_that_walks_away_does_not_cost_the_other_one(self):
        """The reason this matters, run rather than argued.

        Node 1 stops taking part, which is what a node doing its own inner
        steps between outer rounds looks like from node 0. With the channel
        confined to a machine, node 0's detection round completes on its own
        in milliseconds. On the group this channel used to build it would have
        waited for node 1 and then failed the whole group on a timeout — once
        per step.
        """
        results = _run_partition(
            {"per_node": 2, "sharding": "per_node", "detect": True,
             "walk_away_seconds": 3.0, "timeout": 20}
        )

        for rank in (0, 1):
            _partition, _members, answered, elapsed = results[rank]
            assert answered is True, (
                "rank %d did not get an answer while the other machine was "
                "away: %r" % (rank, answered)
            )
            assert elapsed < 1.0, (
                "rank %d took %.2fs, so it was waiting for the machine that "
                "walked away" % (rank, elapsed)
            )

    def test_and_the_wide_group_really_does_wait_for_it(self):
        """The negative control, without which the test above proves nothing.

        Same two machines, same machine walking away, and the only difference
        is that the sharding crosses machines so the channel is one group over
        the job — which is what it was for every job before GPU-125. Node 0
        now waits for node 1 and comes out on the group's timeout, shortened
        here to two seconds so the control is cheap to keep.

        Note what this costs where it is wrong: it is per step.
        """
        results = _run_partition(
            {"per_node": 2, "sharding": "global", "detect": True,
             "walk_away_seconds": 3.0, "timeout": 2}
        )

        for rank in (0, 1):
            _partition, _members, answered, elapsed = results[rank]
            assert elapsed is not None and elapsed > 1.0, (
                "rank %d answered in %.3fs on a group containing a machine "
                "that was not there, so this control is not controlling for "
                "anything" % (rank, elapsed or 0.0)
            )
