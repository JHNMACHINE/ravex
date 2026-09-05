"""The adapter seam, now that something calls it (GPU-69).

``FrameworkAdapter`` declared three methods for months and nothing invoked any
of them: ``get_adapter()`` had no caller, and the only thing the runtime did
with a detected framework was write its name into the checkpoint's metadata. So
the seam was three signatures and an intention.

What is tested here is the wiring, not a framework. No adapter Ravex ships
overrides these methods yet — HuggingFace's ``Trainer.state`` and Lightning's
loop counters are the reason the seam exists, and they come next. The wiring is
worth testing on its own because its interesting cases are all *absences*: a
checkpoint written before any of this existed, a checkpoint written under
another framework, an adapter that raises in the middle of a collective. Each
of those has a right answer, and none of them is "lose the weights".
"""

import pytest
import torch

import ravex
import ravex._frameworks as frameworks
from ravex._backends import TorchSaveBackend
from ravex._config import RavexConfig

EXTRA = {"epoch": 3, "global_step": 12}


class RecordingAdapter(frameworks.FrameworkAdapter):
    """An adapter with something to say, and a record of what it was asked."""

    def __init__(self, extra=EXTRA, intercept=True, raise_on=()):
        self.extra = extra
        self.intercept = intercept
        self.raise_on = raise_on
        self.collected = 0
        self.restored = []

    def _maybe_fail(self, where):
        if where in self.raise_on:
            raise RuntimeError("the %s side of this adapter is broken" % where)

    def should_intercept_step(self):
        self._maybe_fail("intercept")
        return self.intercept

    def collect_extra_state(self):
        self._maybe_fail("collect")
        self.collected += 1
        return dict(self.extra)

    def restore_extra_state(self, state):
        self._maybe_fail("restore")
        self.restored.append(state)


@pytest.fixture
def install(monkeypatch):
    """Put an adapter in front of whatever framework this interpreter looks like.

    Patching ``get_adapter`` rather than ``_ADAPTERS`` because that is the seam
    the runtime actually goes through, and because it keeps the test honest
    about *when*: the runtime imports it inside ``activate()``, so an adapter
    installed here is in place before the first step and before the resume.
    """

    def install(**kwargs):
        adapter = RecordingAdapter(**kwargs)
        monkeypatch.setattr(frameworks, "get_adapter", lambda framework: adapter)
        return adapter

    return install


def stderr_of(capsys):
    """Ravex's warnings, which never reach `caplog`.

    ``RavexRuntime._setup_logging`` sets ``propagate = False`` on purpose — a
    library that reroutes the training script's own logging the moment it
    attaches is a library people stop attaching — and hands its records to a
    stderr handler of its own. So stderr is where a test has to look.
    """
    return capsys.readouterr().err


def written(storage):
    """The latest checkpoint, read back the way a resume would read it."""
    config = RavexConfig()
    config.storage.path = str(storage)
    config.backend = "torch_save"
    config._normalize()
    return TorchSaveBackend(config).load_latest()


def tiny_loop(steps=2):
    """Two optimizer steps and nothing Ravex-shaped in sight."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


def run(steps=2, seen=None):
    """A decorated run, reporting the step it finished on."""

    @ravex.train_loop(backend="torch_save", checkpoint_every=1)
    def train():
        tiny_loop(steps)
        if seen is not None:
            seen["step"] = ravex.step()

    train()


class TestCollection:
    def test_an_adapter_with_nothing_to_say_writes_no_key(self, storage):
        """The default adapter, which is every adapter today.

        A checkpoint that grows an empty `extra` for every run would put the
        key in every reader's way for nothing, and would make "this framework
        contributed state" indistinguishable from "this framework was asked".
        """
        run()
        assert "extra" not in written(storage)

    def test_extra_state_travels_under_its_own_key(self, install, storage):
        adapter = install()
        run()

        state = written(storage)
        assert state["extra"] == {
            "framework": frameworks.detect_framework(),
            "state": EXTRA,
        }
        assert adapter.collected >= 1
        # The point of the separate key: what the adapter handed over is opaque,
        # and the trees a reshard walks have formats that a framework's own
        # bookkeeping would violate.
        assert "epoch" not in state["models"]
        assert "epoch" not in state["sharded"]

    def test_a_collect_that_raises_does_not_cost_the_checkpoint(
        self, install, storage, capsys
    ):
        """The constraint that shaped this: collection is inside the collective.

        An adapter raising here used to be able to take the weights with it —
        and on a sharded run it would take them on one rank while the others
        wrote theirs, which is worse than losing the checkpoint outright.
        """
        install(raise_on=("collect",))
        run()
        said = stderr_of(capsys)

        state = written(storage)
        assert state["models"], "the weights went down with the adapter"
        assert "extra" not in state
        assert "could not collect" in said


class TestRestoration:
    def test_it_comes_back_on_the_next_run(self, install, storage):
        install()
        run()

        reader = install(extra={})
        seen = {}
        run(seen=seen)

        assert reader.restored == [EXTRA]
        assert seen["step"] == 4, "the resume itself did not happen"

    def test_a_checkpoint_without_extra_state_resumes_and_asks_nothing(
        self, install, storage
    ):
        """Read compatibility, which is most of what this test file is for.

        Every checkpoint written before this seam existed is this case, and so
        is every checkpoint written by the adapters Ravex ships today.
        """
        install(extra={})
        run()

        reader = install()
        seen = {}
        run(seen=seen)

        assert reader.restored == []
        assert seen["step"] == 4

    def test_state_from_another_framework_is_skipped_and_named(
        self, install, storage, monkeypatch, capsys
    ):
        """A Lightning checkpoint resumed under HuggingFace.

        Handing Lightning's loop counters to a `Trainer` adapter would be a
        crash at best. Ignoring them silently would be a resume that quietly
        lost half its state. Saying so and resuming everything else is the only
        one of the three a user can act on.
        """
        monkeypatch.setattr(frameworks, "detect_framework", lambda: "lightning")
        install()
        run()

        monkeypatch.setattr(frameworks, "detect_framework", lambda: "huggingface")
        reader = install()
        seen = {}
        run(seen=seen)
        said = stderr_of(capsys)

        assert reader.restored == []
        assert "lightning" in said and "huggingface" in said
        assert seen["step"] == 4, "the rest of the checkpoint was lost with it"

    def test_a_restore_that_raises_does_not_cost_the_resume(
        self, install, storage, capsys
    ):
        install()
        run()

        install(raise_on=("restore",))
        seen = {}
        run(seen=seen)

        assert seen["step"] == 4
        assert "could not restore" in stderr_of(capsys)


class TestStepInterception:
    def test_the_question_is_asked_once_at_activation(self, install, storage):
        """Not per step. `on_step` is the hottest path Ravex has."""
        adapter = install()
        asked = []
        adapter.should_intercept_step = lambda: asked.append(1) or True

        run(steps=5)
        assert asked == [1]

    def test_an_adapter_can_take_over_step_counting(self, install, storage, capsys):
        """False means Ravex stops counting — and says so, loudly.

        Nothing advances the counter in its place today, so the run
        checkpoints nothing. That is the declared meaning of the answer rather
        than an accident, and it is why the runtime warns instead of trusting
        it quietly.
        """
        install(intercept=False)
        seen = {}
        run(steps=3, seen=seen)

        assert seen["step"] == 0
        assert "taken over step counting" in stderr_of(capsys)

    def test_a_broken_answer_counts_steps_anyway(self, install, storage, capsys):
        """An adapter that cannot answer must not silently stop checkpointing.

        The failure mode this guards against is the quiet one: a run that
        trains for hours, writes nothing, and looks fine until it is preempted.
        """
        install(raise_on=("intercept",))
        seen = {}
        run(steps=3, seen=seen)

        assert seen["step"] == 3
        assert "could not say whether to count" in stderr_of(capsys)
        assert ravex.is_active() is False, "activation should have survived it"
