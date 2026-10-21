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


class TestTheSaveHasToFitTheNotice:
    """GPU-132: a coordinated save that cannot land before SIGKILL is not tried.

    Measured across two continents under synchronous FSDP: 95 s from the
    announcement to the checkpoint on disk, against ten seconds of notice.
    Attempting it spends the whole notice on a save that will be killed, when
    the same seconds could have made the last periodic checkpoint durable.
    """

    def _preempted(self, monkeypatch, notice, cost):
        runtime = TestTheDetectionRoundDegradesCleanly()._bare_runtime()
        runtime.config.preemption_notice = notice
        runtime._last_checkpoint_cost = cost
        runtime._last_saved_step = 40
        runtime._emergency_group = "already-built"
        runtime._emergency_requested = True

        events = []
        runtime.checkpoint = lambda *a, **k: events.append("save") or True

        class Backend:
            def flush(self):
                events.append("flush")

        runtime._backend = Backend()
        monkeypatch.setattr(
            "ravex._dist.collectives.emergency_signalled",
            lambda local, group: events.append("collective") or True,
        )
        monkeypatch.setattr(
            "ravex._dist.agreement.announce_emergency",
            lambda *a, **k: events.append("announce") or 0,
        )
        monkeypatch.setattr(os, "kill", lambda pid, sig: events.append("kill"))
        monkeypatch.setattr(signal, "signal", lambda sig, handler: None)
        return runtime, events

    def test_too_slow_means_flush_and_go_joining_nothing(self, monkeypatch, caplog):
        runtime, events = self._preempted(monkeypatch, notice=10.0, cost=130.0)

        runtime._check_emergency_signal()

        # Neither the announcement nor the collective: a "no" in the
        # collective could be outvoted by another preempted rank's "yes" and
        # leave the survivors in a sharded save without this one.
        assert events == ["flush", "kill"]
        assert any("not attempting it" in r.getMessage() for r in caplog.records)

    def test_on_the_store_road_nothing_is_announced(self, monkeypatch):
        runtime, events = self._preempted(monkeypatch, notice=10.0, cost=130.0)
        runtime._emergency_store = object()  # a rendezvous store is there

        runtime._check_emergency_signal()

        assert "announce" not in events
        assert events == ["flush", "kill"]

    def test_fast_enough_is_attempted(self, monkeypatch):
        runtime, events = self._preempted(monkeypatch, notice=30.0, cost=4.0)

        runtime._check_emergency_signal()

        assert events == ["collective", "save", "flush", "kill"]

    def test_without_a_notice_it_is_always_attempted_and_the_estimate_said(
        self, monkeypatch, caplog
    ):
        runtime, events = self._preempted(monkeypatch, notice=None, cost=130.0)

        runtime._check_emergency_signal()

        assert "save" in events
        assert any(
            "took 130s" in r.getMessage() and "unset" in r.getMessage()
            for r in caplog.records
        )

    def test_with_no_checkpoint_timed_yet_it_is_attempted(self, monkeypatch):
        runtime, events = self._preempted(monkeypatch, notice=10.0, cost=None)

        runtime._check_emergency_signal()

        assert "save" in events

    def test_decided_once(self, monkeypatch):
        runtime, _ = self._preempted(monkeypatch, notice=10.0, cost=130.0)
        assert runtime._emergency_is_declined() is True
        runtime._last_checkpoint_cost = 1.0
        assert runtime._emergency_is_declined() is True

    def test_a_rank_that_was_not_preempted_never_asks(self, monkeypatch):
        runtime, events = self._preempted(monkeypatch, notice=10.0, cost=130.0)
        runtime._emergency_requested = False

        runtime._check_emergency_signal()

        assert runtime._emergency_declined is None
        assert "kill" not in events


def test_a_checkpoint_records_what_it_cost(storage):
    """The estimate exists after an ordinary periodic checkpoint."""
    import torch

    import ravex
    from ravex._runtime import get_runtime

    seen = {}

    @ravex.train_loop(backend="torch_save", checkpoint_every=2)
    def train():
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        ravex.batch_boundary()
        seen["cost"] = get_runtime(create=False)._last_checkpoint_cost

    train()
    assert seen["cost"] is not None and seen["cost"] >= 0


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


def test_preemption_notice(monkeypatch):
    assert RavexConfig().preemption_notice is None

    monkeypatch.setenv("RAVEX_PREEMPTION_NOTICE", "30")
    config = RavexConfig.load()
    assert config.preemption_notice == 30.0
    assert not config.problems

    for bad in ("0", "-2", "soon", "nan"):
        monkeypatch.setenv("RAVEX_PREEMPTION_NOTICE", bad)
        config = RavexConfig.load()
        assert config.preemption_notice is None, bad
        assert any("preemption_notice" in p for p in config.problems), bad


def test_emergency_numeric_fields_are_clamped(monkeypatch):
    monkeypatch.setenv("RAVEX_EMERGENCY_CHECK_EVERY", "0")
    monkeypatch.setenv("RAVEX_EMERGENCY_TIMEOUT", "-5")

    config = RavexConfig.load()

    assert config.emergency_check_every == 1
    assert config.emergency_timeout == 1


# ─── multi-rank: real torch.distributed, gloo, spawned processes ───────


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


# ─── GPU-125: how wide the detection channel is allowed to be ──────────────
