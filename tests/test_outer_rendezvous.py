"""GPU-129: nodes started on their own, meeting at a rendezvous of Ravex's own.

``test_outer_train_loop.py`` launches its nodes the way torchrun does, with a
process group and a rank. The nodes here have neither: no
``init_process_group``, no ``RANK``, no ``MASTER_ADDR`` — a plain process that
knows one address, which is how a rented box on another continent starts. If
anything underneath still reached for torch's process group, these are the
tests where it shows.

The two end-to-end tests are the two things the store of torch could not do:
a node that arrives after the start, and the first node dying without taking
the store with it. Both assert **one model** at the end, bit for bit, because
that is the property that fails silently: nodes that stopped exchanging still
train, and their loss still falls.
"""

import hashlib
import multiprocessing as mp
import os
import socket
import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from ravex._dist import rendezvous  # noqa: E402
from ravex._dist.membership import Membership, announce  # noqa: E402

#: The test's own key, on a job of its own beside the one under test: the
#: round after which every node stops.
LAST_KEY = "test/last-round"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server():
    port = free_port()
    store = rendezvous.serve("127.0.0.1", port)
    yield "127.0.0.1:%d" % port
    del store


# ─── the pieces ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10.0.0.5:29401", ("10.0.0.5", 29401)),
        ("box.runpod.internal", ("box.runpod.internal", rendezvous.DEFAULT_PORT)),
        ("[::1]:7000", ("::1", 7000)),
        ("::1", ("::1", rendezvous.DEFAULT_PORT)),
    ],
)
def test_an_address_is_a_host_and_a_port(text, expected):
    assert rendezvous.parse_address(text) == expected


@pytest.mark.parametrize("text", [":29400", "host:port", "host:70000", ""])
def test_a_bad_address_is_refused_with_its_reason(text):
    with pytest.raises(ValueError):
        rendezvous.parse_address(text)


def test_node_numbers_are_never_handed_out_twice(server):
    """Twenty nodes arriving at once, each on its own connection.

    The counter is the whole of identity here, so two nodes holding one number
    would be two nodes publishing under one address key — each overwriting the
    other, and every peer averaging whichever wrote last.
    """
    numbers = []
    lock = threading.Lock()

    def arrive():
        number = rendezvous.register(rendezvous.connect(server, "job"))
        with lock:
            numbers.append(number)

    threads = [threading.Thread(target=arrive) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert sorted(numbers) == list(range(20))


def test_jobs_on_one_server_count_on_their_own(server):
    first = rendezvous.connect(server, "first")
    second = rendezvous.connect(server, "second")
    assert [
        rendezvous.register(first),
        rendezvous.register(first),
        rendezvous.register(second),
    ] == [0, 1, 0]


def test_a_base_member_waits_for_the_others_and_not_past_its_deadline(server):
    store = rendezvous.connect(server, "job")
    rendezvous.register(store)

    started = time.monotonic()
    assert not rendezvous.wait_for_base(store, 2, time.monotonic() + 0.3)
    assert time.monotonic() - started < 5, "the deadline was not a deadline"

    arrival = threading.Timer(
        0.2, lambda: rendezvous.register(rendezvous.connect(server, "job"))
    )
    arrival.start()
    try:
        assert rendezvous.wait_for_base(store, 2, time.monotonic() + 30)
    finally:
        arrival.join()


def test_join_keys_are_looked_for_as_far_as_the_counter_goes(server):
    """The fixed window limited the joins a run could take over its whole life.

    A node that crashes and comes back takes a new number, so a run that has
    seen seventeen arrivals — restarts included — has a joiner past rank 18
    that the window of sixteen never looks at: it announces, nobody reads it,
    and it gives up having been refused by nobody.
    """
    store = rendezvous.connect(server, "job")
    for _ in range(40):
        rendezvous.register(store)
    announce(store, 37, admit_round=9)

    windowed = Membership(store, 0, 2)
    windowed.refresh(1)
    assert 37 not in windowed.accepted, "the premise of this test moved"

    counted = Membership(
        store, 0, 2, ceiling=lambda: rendezvous.registered(store)
    )
    counted.refresh(1)
    assert counted.accepted == {37: 9}


def test_the_rendezvous_is_configured_from_the_environment(monkeypatch, tmp_path):
    from ravex._config import RavexConfig

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.setenv("RAVEX_RENDEZVOUS", "10.0.0.5:29401")
    monkeypatch.setenv("RAVEX_OUTER_JOB", "run-7")
    monkeypatch.setenv("RAVEX_OUTER_MIN_NODES", "3")

    config = RavexConfig.load()
    assert (config.outer_rendezvous, config.outer_job, config.outer_min_nodes) == (
        "10.0.0.5:29401",
        "run-7",
        3,
    )
    assert not config.problems


def test_a_bad_rendezvous_setting_is_reported_rather_than_raised(monkeypatch, tmp_path):
    from ravex._config import RavexConfig

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAVEX_CONFIG", raising=False)
    monkeypatch.setenv("RAVEX_RENDEZVOUS", "host:port")
    monkeypatch.setenv("RAVEX_OUTER_JOB", "a/b")

    config = RavexConfig.load()
    assert config.outer_rendezvous is None
    assert config.outer_job == "default"
    said = " | ".join(config.problems)
    assert "outer_rendezvous" in said and "outer_job" in said, said


# ─── whole nodes ───────────────────────────────────────────────────────


def outer_digest(outer):
    """A fingerprint of the outer parameters, without NumPy.

    `.numpy().tobytes()` was the obvious spelling and it cost this file every
    run of the one job where it executes. Ravex depends on PyYAML and nothing
    else, torch does not require NumPy, and the Moonclip job's image has none
    — that install is a supported configuration the package goes out of its
    way to keep working (GPU-126), and these tests only run where Moonclip is
    installed, which is that image. So the helper raised *"Numpy is not
    available"*, the node reported no result, and the failure read as the
    rendezvous not finishing.

    `bytes(tensor.tolist())` is the NumPy-free spelling `collectives.py`
    already uses; the `uint8` view is what makes it the same bytes for a
    parameter that is not `uint8` to begin with.
    """
    digest = hashlib.sha256()
    for name in sorted(outer):
        digest.update(name.encode("utf-8"))
        raw = outer[name].detach().cpu().contiguous().flatten().view(torch.uint8)
        digest.update(bytes(raw.tolist()))
    return digest.hexdigest()


def die_after_serving_one(exchange, round_number):
    """Serve ``round_number`` to the first peer that asks, then vanish.

    The window GPU-140 is about, made deterministic: one survivor holds this
    node's last report and the other never gets it. The first delivery runs to
    its last byte (``_release`` is called after the transport returns); a
    second request that arrives meanwhile is held until the process is gone,
    and one that arrives after finds nobody.
    """
    serving = threading.Lock()
    original_hold, original_release = exchange._hold, exchange._release

    def hold(wanted):
        if wanted == round_number and not serving.acquire(blocking=False):
            threading.Event().wait()  # the process exits before this returns
        original_hold(wanted)

    def release(wanted):
        original_release(wanted)
        if wanted == round_number:
            os._exit(0)

    exchange._hold, exchange._release = hold, release


def a_node(address, job, min_nodes, root, name, role, queue):
    """One node, started the way a rented box starts one."""
    try:
        for variable in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(variable, None)

        import logging

        import ravex
        from ravex._runtime import get_runtime

        said = []

        class Collect(logging.Handler):
            def emit(self, record):
                said.append(record.getMessage())

        workdir = os.path.join(root, name)
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
        control = rendezvous.connect(address, job + ".control")

        seed = sum(map(ord, name))
        x = torch.randn(256, 4, generator=torch.Generator().manual_seed(seed))
        y = x @ torch.randn(4, 1, generator=torch.Generator().manual_seed(7))

        @ravex.train_loop(
            outer_loop=True,
            outer_rendezvous=address,
            outer_job=job,
            outer_min_nodes=min_nodes,
            # Not 4, and the number is the point. A joiner announces itself
            # `JOIN_MARGIN` rounds ahead, so the whole exchange - the joiner
            # reading which round the run is on, then writing its key, then
            # every member reading that key at a round boundary - has two
            # rounds to complete. On real hardware a round is seconds and that
            # is a wide window.
            #
            # With four inner steps on a model this small it was 156 ms. The
            # CI runner closed **191 rounds** while the third process was
            # still importing torch, about 78 ms each, and the round trip did
            # not fit: `MembershipError`, rank 2 announcing for round 191 and
            # a node first seeing it at 191. Ravex was right to stop - a node
            # that averaged a different set is two models, and it says so -
            # but what it caught was this test running the outer loop faster
            # than the machine answers a question about it.
            #
            # 256 costs seven seconds in this file, measured, and makes the
            # window about 64 times what failed. Before changing it down,
            # note that what matters is not the step count but that a round
            # outlasts a store round trip on the slowest box this runs on.
            outer_inner_steps=256,
            outer_root=os.path.join(workdir, "rounds"),
            outer_deadline=60,
            enabled=True,
            resume=False,
            checkpoint_every=10**9,
            checkpoint_on_exit=False,
        )
        def train():
            logging.getLogger("ravex").setLevel(logging.INFO)
            logging.getLogger("ravex").addHandler(Collect())
            # A different start on every node, on purpose: one model at the
            # end then means the seed round and the join really handed the
            # parameters over, rather than the nodes starting alike.
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            loss_fn = torch.nn.MSELoss()
            runtime = get_runtime()
            armed = False
            for step in range(20000):
                begin = (step * 32) % len(x)
                optimizer.zero_grad()
                loss_fn(model(x[begin : begin + 32]), y[begin : begin + 32]).backward()
                optimizer.step()
                ravex.batch_boundary()
                outer = runtime._outer
                if not outer:
                    continue
                if role == "joiner" and not control.check([LAST_KEY]):
                    control.set(LAST_KEY, str(outer.round_number + 3).encode())
                if (
                    role == "first-dies"
                    and runtime._membership.rank == 0
                    and outer.round_number == 3
                    and not armed
                ):
                    # The way a preempted box goes: no goodbye, no final
                    # checkpoint, its address still on the store - and in the
                    # worst moment, having handed its round 3 report to one
                    # survivor and not the other.
                    #
                    # Until GPU-140 that left the two survivors averaging
                    # different sets and holding two models for the rest of
                    # the run, silently. The CI runner landed in this window
                    # by chance on 2026-09-20; this puts the test in it every
                    # time. The round's set is now decided on the store and a
                    # survivor missing a report gets it relayed by the one
                    # that has it, so they still end on one model.
                    die_after_serving_one(runtime._exchange, 3)
                    armed = True
                if control.check([LAST_KEY]) and outer.round_number > int(
                    control.get(LAST_KEY)
                ):
                    return outer_digest(outer.outer), outer.round_number
            return None, None

        digest, last = train()
        grouped = torch.distributed.is_available() and torch.distributed.is_initialized()
        queue.put((name, digest, last, said, grouped))
    except BaseException:
        import traceback

        queue.put((name, None, None, traceback.format_exc(), None))


def collect(queue, count):
    results = {}
    for _ in range(count):
        name, digest, last, said, grouped = queue.get(timeout=300)
        if digest is None:
            detail = said if isinstance(said, str) else "\n".join(said)
            pytest.fail("node %s did not finish:\n%s" % (name, detail))
        assert not grouped, "node %s built a torch process group" % name
        results[name] = (digest, last, said)
    return results


def rounds_over(said, nodes):
    return [line for line in said if "Outer round" in line and "over %d node(s)" % nodes in line]


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
def test_nodes_started_on_their_own_train_one_model_and_a_later_one_joins(server, tmp_path):
    """GPU-121's join, reached for the first time from a process a user launches."""
    # Here and not at the top of the module: a round report is a Moonclip
    # snapshot, but the pieces above need no engine, and the CI job without
    # Moonclip should still run them.
    pytest.importorskip("moonclip")
    context = mp.get_context("spawn")
    queue = context.Queue()
    job = "join"
    members = [
        context.Process(
            target=a_node, args=(server, job, 2, str(tmp_path), name, "member", queue)
        )
        for name in ("a", "b")
    ]
    for member in members:
        member.start()

    # The joiner starts only once both base members hold their numbers: started
    # together, it could win the race to number 0 and become a base member.
    watch = rendezvous.connect(server, job)
    deadline = time.monotonic() + 120
    while rendezvous.registered(watch) < 2:
        assert time.monotonic() < deadline, "the base members never reached the rendezvous"
        time.sleep(0.1)
    joiner = context.Process(
        target=a_node, args=(server, job, 2, str(tmp_path), "c", "joiner", queue)
    )
    joiner.start()

    try:
        results = collect(queue, 3)
    finally:
        for process in members + [joiner]:
            process.join(60)

    digests = {name: digest for name, (digest, _, _) in results.items()}
    assert len(set(digests.values())) == 1, (
        "the nodes ended holding different models: %s" % digests
    )
    assert any("Joined the run at round" in line for line in results["c"][2]), results["c"][2]
    for name, (_, _, said) in results.items():
        assert rounds_over(said, 3), "node %s never closed a round with the joiner in it" % name


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
def test_the_first_node_dying_does_not_take_the_run_with_it(server, tmp_path):
    """Under torchrun the store lives with the first node's agent, and the
    exchange reads peer addresses from it on every fetch. Here the store is the
    server's, so node 0 going away costs the run one contributor and nothing
    else.

    Node 0 leaves without a goodbye, in the middle of serving its last round:
    one survivor has its report and the other does not (GPU-140). They must
    still end on one model."""
    pytest.importorskip("moonclip")
    context = mp.get_context("spawn")
    queue = context.Queue()
    job = "dies"
    rendezvous.connect(server, job + ".control").set(LAST_KEY, b"7")
    nodes = [
        context.Process(
            target=a_node, args=(server, job, 3, str(tmp_path), name, "first-dies", queue)
        )
        for name in ("a", "b", "c")
    ]
    for node in nodes:
        node.start()
    try:
        results = collect(queue, 2)
    finally:
        for node in nodes:
            node.join(60)

    digests = {name: digest for name, (digest, _, _) in results.items()}
    assert len(set(digests.values())) == 1, (
        "the survivors ended holding different models: %s" % digests
    )
    for name, (_, last, said) in results.items():
        assert last == 8, "node %s stopped at round %s" % (name, last)
        assert rounds_over(said, 3), "node %s never had all three nodes" % name
        assert rounds_over(said, 2), "node %s never closed a round without node 0" % name
