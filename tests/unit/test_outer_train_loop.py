"""GPU-116: the outer loop reached the way a user reaches it.

``tests/unit/test_dist_round.py`` drives ``close_round`` by hand. These drive
``@ravex.train_loop``, which is the only way anyone outside this repository
will ever get to it — and the difference is not cosmetic: the wiring under test
here is *when* a round closes, which is a question about the training loop and
not about the round.

Two processes, launched the way torchrun launches them, so the rendezvous store
is a real ``TCPStore`` and the exchange is real sockets. The training function
they run mentions neither rounds nor deltas nor peers.
"""


import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("moonclip")


def test_the_outer_loop_stays_off_unless_it_is_asked_for(tmp_path, monkeypatch):
    """It changes what the run trains, so it never turns itself on."""
    import ravex
    from ravex._runtime import get_runtime

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)

    @ravex.train_loop(enabled=True, resume=False, checkpoint_on_exit=False)
    def train():
        runtime = get_runtime()
        assert runtime.config.outer_loop is False
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.randn(2, 3)).sum().backward()
            optimizer.step()
        return runtime

    runtime = train()
    assert runtime._outer is None, "an outer loop was built without being asked"


def test_without_a_rendezvous_the_run_carries_on_locally(tmp_path, monkeypatch):
    """Asked for, unavailable, and said once rather than every step.

    A single process has no process group, so there is no store to take peer
    addresses from. The refusal has to be loud — a job that expected peers and
    got none is a different run, not a slower one — and it has to happen once.
    """
    import logging

    import ravex
    from ravex._runtime import get_runtime

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    # Not caplog: activating a runtime reconfigures the "ravex" logger, and a
    # handler attached afterwards is the only one that sees what it emits.
    said = []

    class Collect(logging.Handler):
        def emit(self, record):
            said.append(record.getMessage())

    if True:

        @ravex.train_loop(
            outer_loop=True,
            outer_inner_steps=2,
            enabled=True,
            resume=False,
            checkpoint_on_exit=False,
        )
        def train():
            runtime = get_runtime()
            logging.getLogger("ravex").addHandler(Collect())
            # Built inside, so the patches register it - which is the ordinary
            # case and the one where "no model yet" must not be a refusal.
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(6):
                optimizer.zero_grad()
                model(torch.randn(2, 3)).sum().backward()
                optimizer.step()
                ravex.batch_boundary()
            return runtime

        runtime = train()

    assert runtime._outer is False, "the failure was not remembered"
    refusals = [line for line in said if "Could not start the outer loop" in line]
    assert len(refusals) == 1, "the refusal was repeated: %d times" % len(refusals)
    assert "init_process_group" in refusals[0]
