"""The decorator, which is now the only way in.

``tests/test_patches.py`` covers what happens *while* Ravex is attached — the
registration, the step counting, the cadence. This file covers the thing that
replaced the autoloader: the boundary itself. When does the runtime exist, when
does it stop existing, and what happens at each edge.

The edges are where an entry point earns its keep. An autoloader had no exit at
all — the runtime lived until the interpreter did, and a final checkpoint was an
``atexit`` hook hoping to run. A decorator has a ``finally``, which is a much
stronger promise, and these tests are what makes it one.
"""

import pytest

import torch

import ravex
from ravex._runtime import get_runtime


def tiny_loop(steps: int = 3):
    """A model and an optimizer, stepped, with nothing Ravex-shaped in sight."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, optimizer


class TestTheBoundary:
    def test_the_runtime_exists_only_inside(self, storage):
        seen = {}

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            seen["inside"] = ravex.is_active()

        assert get_runtime(create=False) is None, "active before it was called"
        train()
        assert seen["inside"] is True
        assert get_runtime(create=False) is None, (
            "the runtime outlived the function; on a long-lived process that is "
            "a patched torch nobody asked for"
        )

    def test_pytorch_is_unpatched_on_the_way_out(self, storage):
        # `nn.Module.train` rather than `Optimizer.step`: the step hook is
        # attached per optimizer instance, not by replacing the class method,
        # so `Optimizer.step` is the same function throughout and would make
        # this test pass without patching anything at all.
        original_train = torch.nn.Module.train

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            assert torch.nn.Module.train is not original_train

        train()
        assert torch.nn.Module.train is original_train

    def test_an_exception_still_tears_down(self, storage):
        """The `finally`, which is the whole reason this is a decorator.

        A training run that dies is the case Ravex exists for. If the teardown
        only ran on the happy path, the checkpoint that matters most — the one
        after the failure — is the one that would not be written.
        """

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            tiny_loop()
            raise RuntimeError("the run died")

        with pytest.raises(RuntimeError, match="the run died"):
            train()

        assert get_runtime(create=False) is None
        assert not hasattr(torch.nn.Module, "_ravex_patched")

    def test_it_can_be_called_twice(self, storage):
        """A second call starts clean rather than finding the first run's state.

        This is the property the autoloader could not have had, and it is not
        theoretical: a sweep that calls its training function once per
        hyperparameter is the ordinary way people use a decorated entry point.
        """
        steps = []

        # `resume=False` and no exit checkpoint, so this asks about the
        # teardown and nothing else. With the defaults the second call *does*
        # continue the first — that is Ravex working, and it is pinned in
        # `test_a_second_call_resumes_the_first` below.
        @ravex.train_loop(
            backend="torch_save",
            checkpoint_every=10_000,
            checkpoint_on_exit=False,
            resume=False,
        )
        def train():
            tiny_loop(steps=2)
            steps.append(ravex.step())

        train()
        train()
        assert steps == [2, 2], (
            "the second run continued the first one's step count, so the "
            "patches or the registry survived the teardown: %s" % steps
        )

    def test_a_second_call_resumes_the_first(self, storage):
        """Two calls in one process are two runs, and the second resumes.

        Not obvious, and worth pinning rather than discovering: the decorator's
        exit writes a final checkpoint, and its next entry finds that checkpoint
        under the same storage path and picks it up — exactly as a rerun of the
        script would. So a sweep that calls its training function once per
        hyperparameter, with one storage path, continues one run rather than
        starting several. Point `storage.path` at something per-trial, or set
        `resume=False`, if that is not what was meant.
        """
        steps = []

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            tiny_loop(steps=2)
            steps.append(ravex.step())

        train()
        train()
        assert steps == [2, 4], (
            "the second call did not pick up the checkpoint the first one "
            "wrote on the way out: %s" % steps
        )


class TestTheSignature:
    def test_it_works_bare(self, storage):
        @ravex.train_loop
        def train():
            return ravex.is_active()

        assert train() is True

    def test_it_keeps_the_wrapped_function_identity(self):
        @ravex.train_loop(backend="torch_save")
        def train_the_model():
            """A docstring worth keeping."""

        assert train_the_model.__name__ == "train_the_model"
        assert train_the_model.__doc__ == "A docstring worth keeping."

    def test_arguments_and_return_value_pass_through(self, storage):
        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train(a, b, *, c):
            return a + b + c

        assert train(1, 2, c=3) == 6

    def test_an_unknown_option_is_refused_by_name(self, storage):
        """Not ignored. A silently-dropped `checkpoint_evry` is a run that
        checkpoints on the default cadence and never says why."""

        @ravex.train_loop(checkpoint_evry=50)
        def train():  # pragma: no cover - never entered
            raise AssertionError("should not have been called")

        with pytest.raises(TypeError, match="checkpoint_evry"):
            train()

    def test_preemption_handler_sets_the_config_option(self, storage):
        seen = {}

        @ravex.train_loop(backend="torch_save", preemption_handler=False)
        def train():
            seen["handle_sigterm"] = get_runtime(create=False).config.handle_sigterm

        train()
        assert seen["handle_sigterm"] is False

    def test_elastic_refuses_rather_than_pretending(self):
        """`elastic=True` has nothing to turn on, so it says so.

        The runtime does not call `ravex._dist.elastic` at all — only the tests
        and the two-machine harness do. Accepting the flag and doing nothing
        would be a job that looks like it survives a resize and does not, which
        is the failure this codebase is written against. It lands with GPU-110.
        """
        with pytest.raises(NotImplementedError, match="GPU-110"):

            @ravex.train_loop(elastic=True)
            def train():  # pragma: no cover - the decorator raises first
                pass

    def test_nesting_is_refused(self, storage):
        @ravex.train_loop(backend="torch_save")
        def inner():  # pragma: no cover - never reached
            pass

        @ravex.train_loop(backend="torch_save")
        def outer():
            inner()

        with pytest.raises(RuntimeError, match="already running"):
            outer()


class TestTrack:
    def test_it_registers_what_it_is_given(self, storage):
        seen = {}

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            ravex.track(model, optimizer)
            seen["status"] = ravex.status()

        train()
        assert seen["status"]["models"] >= 1
        assert seen["status"]["optimizers"] >= 1

    def test_a_tracked_optimizer_still_counts_its_steps(self, storage):
        """Registering by hand must not cost the step hook.

        The automatic path attaches it in the patched `__init__`; an optimizer
        handed over afterwards has to get the same treatment, or the run
        silently never reaches a cadence and never checkpoints.
        """
        seen = {}

        @ravex.train_loop(backend="torch_save", checkpoint_every=10_000)
        def train():
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            ravex.track(model, optimizer)
            for _ in range(3):
                model(torch.randn(2, 4)).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            seen["step"] = ravex.step()

        train()
        assert seen["step"] == 3

    def test_outside_a_train_loop_it_raises(self):
        model = torch.nn.Linear(4, 2)
        with pytest.raises(RuntimeError, match="outside a @ravex.train_loop"):
            ravex.track(model)


class TestWhatTheAutoloaderTookWithIt:
    def test_importing_ravex_starts_nothing(self):
        """The one promise the `.pth` made that still has to hold.

        It used to be defended by `_bootstrap` refusing to do anything without a
        ravex.yaml. Now it is defended by there being nothing to defend: an
        import installs no patches and builds no runtime, because only the
        decorator does.
        """
        assert get_runtime(create=False) is None
        assert not hasattr(torch.nn.Module, "_ravex_patched")

    def test_the_removed_names_are_really_gone(self):
        """A leftover `activate` would be the worst outcome of this change:
        two entry points, one of which has no exit."""
        for name in ("activate", "enable", "disable"):
            assert not hasattr(ravex, name), name


class TestTheBoundaryHandedOverByHand:
    """GPU-123: what a loop with no DataLoader can do about the boundary.

    Ravex takes the top of the training iteration from the DataLoader iterator
    it wraps. A loop over tensors that are already batched has no iterator to
    wrap, and everything the runtime defers to that moment has to go somewhere
    else: the checkpoint falls back to mid-step, and the outer round — which
    cannot fall back, because it writes parameters — simply never happened.

    `ravex.batch_boundary()` is that moment, handed over by hand.
    """

    def test_calling_it_outside_a_run_does_nothing(self):
        """It is the kind of call that ends up in a library's own loop, run
        both under Ravex and not. Raising there would make it un-callable."""
        assert get_runtime(create=False) is None
        ravex.batch_boundary()

    def test_the_checkpoint_lands_after_the_scheduler_rather_than_inside_the_step(
        self, storage
    ):
        """The cost of the mid-step fallback, and that handing over removes it.

        A checkpoint taken inside the `optimizer.step()` hook is one
        `scheduler.step()` short of the state the loop is actually in, so a
        resumed run trains with a learning rate from the previous step and
        every step after it is wrong by a little. With an LR that halves every
        step that is not a rounding difference — it is the whole schedule off
        by one.
        """
        LR, GAMMA = 0.1, 0.5

        def saved_learning_rate(hand_over):
            @ravex.train_loop(
                backend="torch_save",
                checkpoint_every=2,
                resume=False,
                checkpoint_on_exit=False,
            )
            def train():
                torch.manual_seed(0)
                model = torch.nn.Linear(4, 2)
                optimizer = torch.optim.SGD(model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=GAMMA
                )
                # Three, not four: a fourth step makes a second checkpoint
                # due and `load_latest` would answer about that one instead.
                for _ in range(3):
                    if hand_over:
                        ravex.batch_boundary()
                    model(torch.randn(2, 4)).sum().backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                ravex.flush()
                state = get_runtime().backend.load_latest()
                # The checkpoint due at step 2, wherever it was taken. Which
                # `scheduler.step()` it landed after is the whole question.
                assert state["step"] == 2, state["step"]
                (schedule,) = state["schedulers"].values()
                return schedule["_last_lr"][0]

            return train()

        assert saved_learning_rate(hand_over=True) == pytest.approx(LR * GAMMA**2), (
            "the boundary was handed over and the checkpoint still landed "
            "inside optimizer.step(), one scheduler step stale"
        )
        assert saved_learning_rate(hand_over=False) == pytest.approx(LR * GAMMA), (
            "the mid-step fallback no longer costs a stale learning rate, so "
            "the reason for handing the boundary over is gone and this test "
            "is measuring nothing"
        )
