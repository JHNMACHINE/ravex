"""GPU-116: the outer loop reached the way a user reaches it.

``tests/test_dist_round.py`` drives ``close_round`` by hand. These drive
``@ravex.train_loop``, which is the only way anyone outside this repository
will ever get to it — and the difference is not cosmetic: the wiring under test
here is *when* a round closes, which is a question about the training loop and
not about the round.

Two processes, launched the way torchrun launches them, so the rendezvous store
is a real ``TCPStore`` and the exchange is real sockets. The training function
they run mentions neither rounds nor deltas nor peers.
"""

import multiprocessing as mp
import os
import socket
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("moonclip")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def a_node(rank, world, port, root, rounds, inner, queue):
    """One rank: init the group, then a training loop that knows nothing."""
    try:
        import torch.distributed as dist

        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world),
            LOCAL_RANK="0",
        )
        dist.init_process_group("gloo", rank=rank, world_size=world)

        import logging

        import ravex

        said = []

        class Collect(logging.Handler):
            def emit(self, record):
                said.append(record.getMessage())

        os.makedirs(os.path.join(root, "cwd%d" % rank), exist_ok=True)
        os.chdir(os.path.join(root, "cwd%d" % rank))

        generator = torch.Generator().manual_seed(7 + rank)
        x = torch.randn(256, 4, generator=generator)
        y = x @ torch.randn(4, 1, generator=torch.Generator().manual_seed(7))

        @ravex.train_loop(
            outer_loop=True,
            outer_inner_steps=inner,
            outer_root=os.path.join(root, "rank%d" % rank),
            outer_deadline=60,
            enabled=True,
            resume=False,
            checkpoint_every=10**9,
            checkpoint_on_exit=False,
        )
        def train():
            logging.getLogger("ravex").addHandler(Collect())
            # Built inside the decorated function, which is where Ravex can
            # see it. A model constructed before activation is invisible to
            # the patches and has to be named with ravex.track().
            torch.manual_seed(3)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            loss_fn = torch.nn.MSELoss()
            before = loss_fn(model(x), y).item()
            for step in range(rounds * inner):
                begin = (step * 32) % len(x)
                optimizer.zero_grad()
                loss_fn(model(x[begin : begin + 32]), y[begin : begin + 32]).backward()
                optimizer.step()
                # The batch boundary a dataloader would give for free. This
                # loop has no dataloader, which is exactly the case worth
                # testing: a round must not close inside optimizer.step().
                ravex._runtime.get_runtime().on_batch_boundary()
            return (
                before,
                loss_fn(model(x), y).item(),
                [float(p.detach().sum()) for p in model.parameters()],
            )

        before, after, sums = train()
        queue.put(
            (
                rank,
                before,
                after,
                sums,
                [line for line in said if "Outer round" in line],
            )
        )
    except BaseException as exc:  # reported rather than a silent empty queue
        import traceback

        queue.put((rank, None, None, None, traceback.format_exc()))
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def run_two(tmp_path, rounds=3, inner=8):
    context = mp.get_context("spawn")
    queue = context.Queue()
    port = free_port()
    workers = [
        context.Process(
            target=a_node,
            args=(rank, 2, port, str(tmp_path), rounds, inner, queue),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    results = {}
    for _ in workers:
        rank, before, after, sums, said = queue.get(timeout=300)
        if before is None:
            pytest.fail("rank %d failed:\n%s" % (rank, said))
        results[rank] = (before, after, sums, said)
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0, "rank exited with %s" % worker.exitcode
    return results


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
def test_a_training_loop_that_knows_nothing_trains_with_a_peer(tmp_path):
    """The whole point of GPU-116, and the assertion is the second half.

    That it trains is necessary. That the two ranks end holding **the same
    model** is what says the rounds actually happened: each rank sees its own
    slice of the data, so without an exchange they would end up somewhere
    different, and the loss would drop anyway.
    """
    results = run_two(tmp_path)

    for rank, (before, after, _, _said) in results.items():
        assert after < before / 2, "rank %d did not train: %.4f -> %.4f" % (
            rank,
            before,
            after,
        )

    left, right = results[0][2], results[1][2]
    assert left == pytest.approx(right, abs=1e-4), (
        "the two ranks ended with different models, so the rounds did not "
        "reach them: %s against %s\nrank0 rounds: %s\nrank1 rounds: %s"
        % (left, right, results[0][3], results[1][3])
    )


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
                runtime.on_batch_boundary()
            return runtime

        runtime = train()

    assert runtime._outer is False, "the failure was not remembered"
    refusals = [line for line in said if "Could not start the outer loop" in line]
    assert len(refusals) == 1, "the refusal was repeated: %d times" % len(refusals)
    assert "init_process_group" in refusals[0]
