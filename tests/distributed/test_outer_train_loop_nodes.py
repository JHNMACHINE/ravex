"""GPU-116: the outer loop reached the way a user reaches it - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_outer_train_loop.py.
"""

import multiprocessing as mp
import os
import sys

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("moonclip")


def _store_server():
    """A rendezvous store for one spawned group, held by the test's own process.

    On port 0, with the port read back from the store. The probe it replaces
    found a free port, closed it and handed the number to a rank to listen on;
    under `pytest -n` another worker can take it in between, and that rank dies
    with EADDRINUSE (GPU-191, as in test_emergency_checkpoint and
    test_dist_multinode). A server that is already listening has no such
    window. The caller keeps it alive until its ranks are done.
    """
    from ravex._dist.rendezvous import _tcp_store

    return _tcp_store("127.0.0.1", 0, True, 120.0)


def _store_client(port):
    """The store `_store_server` opened, from inside a rank."""
    from ravex._dist.rendezvous import _tcp_store

    return _tcp_store("127.0.0.1", port, False, 120.0)


def a_node(rank, world, port, root, rounds, inner, queue, boundary="public", log=False, resume=False, barrier=None,
           overlap=False, auto=False):
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
        dist.init_process_group("gloo", store=_store_client(port), rank=rank, world_size=world)

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
            # No `outer_root`: the default is under test too, beside the
            # store in `checkpoints.rounds` and never inside it (GPU-240).
            outer_deadline=60,
            enabled=True,
            # A second run in the same directories resumes the first's
            # checkpoints, one per rank, written as the first ended (GPU-212).
            resume=resume,
            checkpoint_every=10**9,
            checkpoint_on_exit=True,
            outer_overlap=overlap,
            # GPU-231: bounds a run of a few dozen steps can travel inside.
            outer_auto_steps=auto,
            outer_min_inner_steps=2,
            outer_max_inner_steps=32,
            # Inside every round when the metrics are under test (GPU-274):
            # the default 10 would skip the first round of 8 steps.
            histograms_every=inner if log else 200,
            metrics_every=inner if log else 10,
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
                # `model.forward` runs no module hooks: the one loop Ravex
                # cannot take the top of the iteration from (GPU-276).
                forward = model.forward if boundary == "unseen" else model
                loss_fn(forward(x[begin : begin + 32]), y[begin : begin + 32]).backward()
                optimizer.step()
                if log:
                    # A different constant per rank, so the model's value can
                    # only be the mean over both (GPU-135).
                    ravex.log_metrics({"train/loss": 1.0 + 2.0 * rank})
                if boundary == "public":
                    # The batch boundary a dataloader would give for free.
                    # This loop has no dataloader, which is exactly the case
                    # worth testing: a round must not close inside
                    # optimizer.step(), so somebody has to hand over the top
                    # of the iteration. `boundary="forward"` leaves it to the
                    # next forward of the model (GPU-276); `"unseen"` gives
                    # Ravex neither (GPU-123).
                    ravex.batch_boundary()
            if barrier is not None:
                # Nobody leaves before everybody has trained. A node that
                # finished its steps and shut down takes the round it serves
                # with it, and a peer that started late - a spawned
                # interpreter importing torch on a loaded runner - then
                # cannot take the run's starting state and trains alone,
                # which is not the case any of these tests is about. Seen in
                # CI once the multi-node tests ran on their own workers
                # (GPU-256). Not the process group's barrier: Ravex uses that
                # group too, and two collectives meeting out of order hang.
                barrier.wait(timeout=240)
            if auto:
                # The outer parameters as the training ended (GPU-231): the
                # model also holds the steps trained after the last round,
                # which differ per rank once H no longer divides the loop's
                # length. A round still in flight is landed first, while the
                # peer is there to finish it; the runtime is gone by the time
                # the decorator returns.
                from ravex._runtime import get_runtime

                runtime = get_runtime(create=False)
                runtime._land_round(wait=True)
                outer.extend(float(value.sum()) for _, value in sorted(runtime._outer.outer.items()))
            return before, model, loss_fn

        outer = []
        # Measured once the run has ended rather than inside it: with
        # `overlap` the last round lands at the shutdown (GPU-230), and a
        # blocking run's model is the same either side of it.
        before, model, loss_fn = train()
        after = loss_fn(model(x), y).item()
        sums = [float(p.detach().sum()) for p in model.parameters()]
        queue.put(
            (
                rank,
                before,
                after,
                sums,
                list(said),
                outer,
            )
        )
    except BaseException as exc:  # reported rather than a silent empty queue
        import traceback

        queue.put((rank, None, None, None, traceback.format_exc(), None))
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def rounds_of(said):
    """The log lines a closed round leaves behind."""
    return [line for line in said if "Outer round" in line and "took" in line]


def run_two(tmp_path, rounds=3, inner=8, boundary="public", log=False, resume=False, overlap=False,
            auto=False):
    context = mp.get_context("spawn")
    queue = context.Queue()
    barrier = context.Barrier(2)
    server = _store_server()
    port = server.port
    workers = [
        context.Process(
            target=a_node,
            args=(rank, 2, port, str(tmp_path), rounds, inner, queue, boundary, log, resume, barrier,
                  overlap, auto),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    results = {}
    for _ in workers:
        rank, before, after, sums, said, outer = queue.get(timeout=300)
        if before is None:
            pytest.fail("rank %d failed:\n%s" % (rank, said))
        results[rank] = (before, after, sums, said, outer)
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

    for rank in range(2):
        # GPU-240: the reports are staged beside the store, so the sync of
        # the store to a bucket carries checkpoints and nothing else.
        cwd = tmp_path / ("cwd%d" % rank)
        assert (cwd / "checkpoints").is_dir(), sorted(os.listdir(cwd))
        assert not (cwd / "checkpoints" / "rounds").exists(), sorted(os.listdir(cwd / "checkpoints"))
        assert (cwd / "checkpoints.rounds").is_dir(), sorted(os.listdir(cwd))

    for rank, (before, after, _, _said, _) in results.items():
        assert after < before / 2, "rank %d did not train: %.4f -> %.4f" % (
            rank,
            before,
            after,
        )

    left, right = results[0][2], results[1][2]
    assert left == pytest.approx(right, abs=1e-4), (
        "the two ranks ended with different models, so the rounds did not "
        "reach them: %s against %s\nrank0 rounds: %s\nrank1 rounds: %s"
        % (left, right, rounds_of(results[0][3]), rounds_of(results[1][3]))
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
def test_rounds_in_flight_still_leave_the_ranks_on_one_model(tmp_path):
    """GPU-230: with `outer_overlap` each round is launched and the next one
    trains while it travels. Every round still lands on both ranks - the
    last at the shutdown, with nothing trained after it - so they end on the
    same model, and the log says each landed rather than blocked."""
    results = run_two(tmp_path, rounds=4, overlap=True)

    for rank, (before, after, _, said, _) in results.items():
        assert after < before / 2, "rank %d did not train: %.4f -> %.4f" % (rank, before, after)
        landed = [line for line in said if "landed" in line]
        assert len(landed) == 4, "rank %d: %s" % (rank, said)

    left, right = results[0][2], results[1][2]
    assert left == pytest.approx(right, abs=1e-4), (
        "the two ranks ended with different models: %s against %s\nrank0: %s\nrank1: %s"
        % (left, right, rounds_of(results[0][3]), rounds_of(results[1][3]))
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
@pytest.mark.parametrize("overlap", [False, True], ids=["blocking", "overlap"])
def test_ranks_that_let_the_link_set_h_change_it_together(tmp_path, overlap):
    """GPU-231 through the runtime: each rank proposes an H from its own
    clock after every round, the store keeps one, and both take it at the
    same round. The rounds the two ranks close then pair up one for one with
    the same number of local steps, and they end on one model."""
    results = run_two(tmp_path, rounds=6, inner=8, overlap=overlap, auto=True)

    def closed(said):
        return [line for line in said if line.startswith("Outer round") and "closed over" in line]

    def moved(said):
        return [line for line in said if line.startswith("Outer rounds go from")]

    for rank, (before, after, _, said, _) in results.items():
        assert after < before, "rank %d did not train: %.4f -> %.4f" % (rank, before, after)
    left, right = results[0][3], results[1][3]
    assert moved(left), "H never moved, so nothing was tested:\n%s" % "\n".join(left)
    assert [line.split(":")[0] for line in moved(left)] == [line.split(":")[0] for line in moved(right)]
    assert closed(left) == closed(right), "the ranks closed different rounds:\n%s\n%s" % (
        "\n".join(closed(left)), "\n".join(closed(right)),
    )
    assert results[0][4], "rank 0 reported no outer parameters"
    assert results[0][4] == pytest.approx(results[1][4], abs=1e-5)


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
def test_a_multi_node_run_logs_one_series_for_the_model(tmp_path):
    """GPU-135: one run, the model's metrics - not one series per node.

    Each rank logs its own constant loss every step. What reaches the store is
    one point per round, on the model's axis (every step of every node), at
    the mean over both nodes, written by one node only.
    """
    import ravex.metrics

    run_two(tmp_path, rounds=3, inner=8, log=True)

    writer = ravex.metrics.read(str(tmp_path / "cwd0" / "checkpoints"))["scalars"]
    loss = writer["train/loss"]
    assert loss["step"] == [16, 32, 48]
    assert loss["value"] == pytest.approx([2.0, 2.0, 2.0])
    assert writer["train/outer/nodes"]["value"] == [2.0, 2.0, 2.0]
    assert writer["train/outer/steps"]["value"] == [16.0, 16.0, 16.0]

    other = ravex.metrics.read(str(tmp_path / "cwd1" / "checkpoints"))["scalars"]
    assert "train/loss" not in other and "train/outer/round" not in other

    # GPU-274: what Ravex takes on its own goes the same way - the gradient
    # norm averaged over the nodes, and the writer's own histograms on the
    # model's axis instead of being dropped.
    assert writer["train/grad_norm"]["step"] == [16, 32, 48]
    histograms = ravex.metrics.read(str(tmp_path / "cwd0" / "checkpoints"))["histograms"]
    assert [entry["step"] for entry in histograms["train/weights/0.weight"]] == [16, 32, 48]
    assert "train/grads/0.weight" in histograms

    # GPU-232: and beside the model's series, each node's own line, every
    # step it logged, on its own steps, in its own store.
    for rank in (0, 1):
        nodes = ravex.metrics.read(str(tmp_path / ("cwd%d" % rank) / "checkpoints"))["nodes"]
        (own,) = nodes.values()
        assert own["train/loss"]["step"] == list(range(1, 25))
        assert own["train/loss"]["value"] == [1.0 + 2.0 * rank] * 24


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
def test_a_loop_that_hands_over_nothing_trains_with_its_peer(tmp_path):
    """GPU-276: no DataLoader and no `ravex.batch_boundary()`, and the rounds
    still close - at the next forward of the model after the step that made
    them due - so the two ranks end on one model, as with the call."""
    results = run_two(tmp_path, boundary="forward")

    for rank, (before, after, _, said, _) in results.items():
        assert after < before / 2, "rank %d did not train: %.4f -> %.4f" % (rank, before, after)
        assert not [line for line in said if "ravex.batch_boundary()" in line], said
    left, right = results[0][2], results[1][2]
    assert left == pytest.approx(right, abs=1e-4), (
        "the two ranks ended with different models: %s against %s\nrank0: %s\nrank1: %s"
        % (left, right, rounds_of(results[0][3]), rounds_of(results[1][3]))
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
def test_a_loop_that_hands_over_no_boundary_is_told_so(tmp_path):
    """GPU-123: the same loop, with nobody handing over the top of it.

    Since GPU-276 the next forward of the model is that boundary, so the loop
    here runs `model.forward` directly, which no hook sees: what is left of the
    case is a loop whose iteration Ravex cannot see start at all.

    Until now this failure was invisible. A round comes due inside
    `optimizer.step()` and is flagged rather than closed - deliberately,
    because closing it there would write the outer parameters into the model
    underneath an optimizer mid-step - and the flag comes down at the batch
    boundary. Without a DataLoader and without `ravex.batch_boundary()` that
    boundary never arrives: the flag stays up, no round closes, and each node
    trains its own model to the end of the run with not one line in the log
    to say so.

    So the assertion is in two halves. The run still trains **wrong**, which
    is the honest outcome of a loop that never hands over its boundary and is
    not this issue's to fix. What changed is that it now names the missing
    call, once. And since GPU-276 the end of the run is a boundary of its own,
    so the round left due closes there: one round, at the exit, and none
    before it.
    """
    results = run_two(tmp_path, boundary="unseen")

    for rank, (_before, _after, _sums, said, _) in results.items():
        closed = rounds_of(said)
        assert len(closed) == 1 and closed[0].startswith("Outer round 1 "), (
            "rank %d closed rounds the loop never gave it a moment for, or not "
            "the one left due at the exit: %s" % (rank, closed)
        )
        complaints = [line for line in said if "ravex.batch_boundary()" in line]
        assert len(complaints) == 1, (
            "rank %d said it %d time(s), and once is the whole contract: a "
            "warning repeated every step is noise a real run scrolls past. "
            "Everything it said: %s" % (rank, len(complaints), said)
        )
        assert "no delta will be exchanged" in complaints[0], (
            "the warning does not say what the silence costs: %s" % complaints[0]
        )


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn + gloo is slow on macOS")
def test_a_resumed_run_carries_on_from_the_round_it_stopped_at(tmp_path):
    """GPU-212: the outer loop's state is saved beside the checkpoints. A run
    resumed goes on from the round it was in, with the momentum it had,
    rather than from round 0 with none - and its two nodes still hold one
    model."""
    moonclip = pytest.importorskip("moonclip")
    run_two(tmp_path, rounds=3)

    # Once a round and apart, not in every checkpoint: with a delta per step
    # that was two copies of the model per step, for a state that moves only
    # when a round closes, and the rounds slowed until a rejoin missed them.
    store = tmp_path / "cwd0" / "checkpoints"
    names = [t["name"] for t in moonclip.MoonclipManager(storage_root=str(store)).describe_latest()["tensors"]]
    assert not [name for name in names if "outer_loop" in name], names
    saved = moonclip.MoonclipManager(storage_root=str(store / "outer_loop")).list_snapshots()
    assert len(saved) == 4, "the start and one per round: %s" % [s["step"] for s in saved]

    results = run_two(tmp_path, rounds=1, resume=True)
    # Under one process group only rank 0 writes the checkpoint, so it is the
    # one that resumes; rank 1 starts from nothing and takes rank 0's state.
    assert any("Outer loop resumed at round 4" in line for line in results[0][3]), results[0][3]
    for rank in (0, 1):
        said = results[rank][3]
        assert any("Outer round 4" in line for line in rounds_of(said)), (rank, rounds_of(said))
    assert results[0][2] == pytest.approx(results[1][2])
