"""GPU-129: nodes started on their own, meeting at a rendezvous of Ravex's own - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_outer_rendezvous.py.
"""

import hashlib
import multiprocessing as mp
import os
import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from ravex._dist import rendezvous  # noqa: E402

# Fixtures, asked for by parameter: a linter sees them unused, pytest does not.
from test_outer_rendezvous import server  # noqa: E402, F401, F811

#: The test's own key, on a job of its own beside the one under test: the
#: round after which every node stops.
LAST_KEY = "test/last-round"

#: Where each node writes what Ravex said to it, in its own directory.
SAID_FILE = "said.log"


@pytest.fixture
def gated_server(monkeypatch):
    """A server with a token, and this process's nodes holding the same one.

    Set in the environment rather than passed, so processes spawned from here
    inherit it the way a node started by the platform would.
    """
    monkeypatch.setenv(rendezvous.TOKEN_ENV, "the-job-token")
    gated = rendezvous.serve("127.0.0.1", 0)
    yield "127.0.0.1:%d" % gated.port
    gated.close()


def test_a_gated_server_hands_numbers_only_to_the_token(gated_server):
    """GPU-134: a stranger at the port never reaches the store.

    Not the store refusing it - the store never sees it. So nothing it could
    do there is possible: no number taken, no key read, no key written.
    """
    import socket as _socket

    first = rendezvous.connect(gated_server, "job", timeout=10)
    assert rendezvous.register(first) == 0

    with pytest.raises(Exception):
        rendezvous.connect(gated_server, "job", timeout=2, token="a guess")

    # Straight at the port, the way a store client or a scanner would come.
    host, port = rendezvous.parse_address(gated_server)
    raw = _socket.create_connection((host, port), timeout=5)
    try:
        opening = raw.recv(64)
        assert opening.startswith(rendezvous.GATE_MAGIC), opening
        raw.sendall(b"\x00" * (rendezvous.NONCE_BYTES + rendezvous.PROOF_CHARS))
        raw.settimeout(5)
        assert raw.recv(64) == b"", "the gate answered a connection without the token"
    finally:
        raw.close()

    assert rendezvous.register(rendezvous.connect(gated_server, "job", timeout=10)) == 1


def test_a_node_without_the_token_does_not_get_in(gated_server):
    with pytest.raises(Exception):
        rendezvous.connect(gated_server, "job", timeout=2, token="")
    assert rendezvous.registered(rendezvous.connect(gated_server, "job", timeout=10)) == 0


# ─── the pieces ────────────────────────────────────────────────────────


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


def split_off_next_round(exchange):
    """Make this node lose every report of the next round, and every relay.

    It still publishes its own, so the others decide the round over all three
    and apply it; this node cannot, and the only thing left that keeps it on
    their model is to leave and come back (GPU-142).

    It really does fetch the others' reports first, and only then throws them
    away. A proposer names itself only once every peer it names has taken its
    report, so a node that never fetched would leave the others unable to
    propose themselves - and then it is this node that decides, over itself
    alone, and nobody has to leave. The two seconds make the others decide
    first.
    """
    original = exchange.gather_by_rank

    def gather_by_rank(peers, round_number, expected, deadline):
        original(peers, round_number, expected, deadline)
        time.sleep(2.0)
        return {}

    exchange.gather_by_rank = gather_by_rank
    exchange.recover = lambda *args, **kwargs: None


def a_node(address, job, min_nodes, root, name, role, queue, control_at=None,
           member=None, kill=None, deadline=60, overlap=False):
    """One node, started the way a rented box starts one.

    ``control_at`` is where the test's own keys live when ``address`` may die
    under the run; ``member`` its place in ``address``'s list; ``kill`` the
    pid of the run's server, which a node in the role ``server-dies`` kills
    at round 3 (GPU-229)."""
    try:
        for variable in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(variable, None)

        import logging

        import ravex
        from ravex._runtime import get_runtime

        said = []
        workdir = os.path.join(root, name)
        os.makedirs(workdir, exist_ok=True)
        # Also on disk, line by line: a node that never finishes never puts
        # `said` on the queue, and its INFO lines are the only account of
        # where it stopped (GPU-260). `collect` prints them.
        said_file = open(os.path.join(workdir, SAID_FILE), "a", buffering=1)

        class Collect(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                said.append(message)
                said_file.write("%.3f %s\n" % (time.monotonic(), message))

        os.chdir(workdir)
        control = rendezvous.connect(control_at or address, job + ".control")

        seed = sum(map(ord, name))
        x = torch.randn(256, 4, generator=torch.Generator().manual_seed(seed))
        y = x @ torch.randn(4, 1, generator=torch.Generator().manual_seed(7))

        @ravex.train_loop(
            outer_loop=True,
            outer_rendezvous=address,
            outer_job=job,
            outer_min_nodes=min_nodes,
            outer_member=member,
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
            outer_deadline=deadline,
            enabled=True,
            resume=False,
            checkpoint_every=10**9,
            checkpoint_on_exit=False,
            outer_overlap=overlap,
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
                if (
                    role == "server-dies"
                    and kill is not None
                    and outer.round_number == 3
                    and not armed
                    and control.add("test/killed", 1) == 1
                ):
                    # The server goes the way a machine does: at once, mid
                    # run, with every node's state on it.
                    os.kill(kill, 9)
                    armed = True
                if (
                    role == "splits"
                    and runtime._membership.rank == 2
                    and outer.round_number == 3
                    and not armed
                ):
                    split_off_next_round(runtime._exchange)
                    armed = True
                if not control.check([LAST_KEY]):
                    continue
                last = int(control.get(LAST_KEY))
                flight = runtime._flight
                if flight is not None and flight.flight.round_number == last:
                    # With `overlap` (GPU-230) the round number moves on the
                    # round's own thread and a round lands at whichever
                    # boundary finds it back, so nodes reading "the first
                    # boundary past the last round" read different rounds.
                    # The last one is landed here, at the boundary right
                    # after its launch, so every node reads the same.
                    runtime._land_round(wait=True)
                if runtime._flight is None and outer.round_number > last:
                    return outer_digest(outer.outer), outer.round_number
            return None, None

        digest, last = train()
        grouped = torch.distributed.is_available() and torch.distributed.is_initialized()
        queue.put((name, digest, last, said, grouped))
    except BaseException:
        import traceback

        queue.put((name, None, None, traceback.format_exc(), None))


def collect(queue, count, root=None, names=()):
    """Every node's result, or a failure that says where the missing ones were.

    Given ``root`` and ``names``, a timeout ends with the last lines each
    node that did not answer wrote to its file: until GPU-260 it ended in a
    bare ``_queue.Empty``, and the INFO lines that tell a joiner that never
    got in from rounds gone slow were lost with the process."""
    import queue as _queue

    results = {}
    for _ in range(count):
        try:
            name, digest, last, said, grouped = queue.get(timeout=300)
        except _queue.Empty:
            missing = [name for name in names if name not in results]
            pytest.fail(
                "no result in 300 s from %s\n%s"
                % (missing or "some node", "\n".join(
                    tail_of(root, name) for name in missing
                ))
            )
        if digest is None:
            detail = said if isinstance(said, str) else "\n".join(said)
            if root is not None:
                detail += "\n" + tail_of(root, name)
            pytest.fail("node %s did not finish:\n%s" % (name, detail))
        assert not grouped, "node %s built a torch process group" % name
        results[name] = (digest, last, said)
    return results


def tail_of(root, name, lines=40):
    path = os.path.join(root, name, SAID_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            said = handle.read().splitlines()
    except OSError:
        return "--- %s: nothing written" % name
    return "--- %s, last %d of %d lines:\n%s" % (
        name, min(lines, len(said)), len(said), "\n".join(said[-lines:])
    )


def end_nodes(processes, patience=60):
    """Join every node, and kill the ones still going after ``patience``.

    A test that failed leaves nodes training with nobody to stop them, and on
    Windows a child does not die with its parent: they spun on, thousands of
    CPU seconds each, and every timed test after the first failure in a run
    failed too (2026-10-16)."""
    for process in processes:
        process.join(patience)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(10)


def rounds_over(said, nodes):
    return [line for line in said if "Outer round" in line and "over %d node(s)" % nodes in line]


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
def test_nodes_started_on_their_own_train_one_model_and_a_later_one_joins(server, tmp_path):
    """GPU-121's join, reached for the first time from a process a user launches."""
    join_a_run(server, tmp_path)


def test_the_same_run_through_a_gated_rendezvous(gated_server, tmp_path):
    """GPU-134: with a token, the whole of it - numbers, rounds, the join -
    goes through the gate and the proofs, and ends in one model."""
    join_a_run(gated_server, tmp_path)


def join_a_run(server, tmp_path):
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
        results = collect(queue, 3, str(tmp_path), "abc")
    finally:
        end_nodes(members + [joiner])

    digests = {name: digest for name, (digest, _, _) in results.items()}
    assert len(set(digests.values())) == 1, (
        "the nodes ended holding different models: %s (at rounds %s)"
        % (digests, {name: last for name, (_, last, _) in results.items()})
    )
    assert any("Joined the run at round" in line for line in results["c"][2]), results["c"][2]
    for name, (_, _, said) in results.items():
        assert rounds_over(said, 3), "node %s never closed a round with the joiner in it" % name


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
@pytest.mark.parametrize("overlap", [False, True], ids=["blocking", "overlap"])
def test_the_first_node_dying_does_not_take_the_run_with_it(server, tmp_path, overlap):
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
    job = "dies-overlap" if overlap else "dies"
    rendezvous.connect(server, job + ".control").set(LAST_KEY, b"7")
    nodes = [
        context.Process(
            target=a_node, args=(server, job, 3, str(tmp_path), name, "first-dies", queue),
            kwargs=dict(overlap=overlap),
        )
        for name in ("a", "b", "c")
    ]
    for node in nodes:
        node.start()
    try:
        results = collect(queue, 2, str(tmp_path), "abc")
    finally:
        end_nodes(nodes)

    digests = {name: digest for name, (digest, _, _) in results.items()}
    assert len(set(digests.values())) == 1, (
        "the survivors ended holding different models: %s" % digests
    )
    for name, (_, last, said) in results.items():
        assert last == 8, "node %s stopped at round %s" % (name, last)
        assert rounds_over(said, 3), "node %s never had all three nodes" % name
        assert rounds_over(said, 2), "node %s never closed a round without node 0" % name


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
def test_a_node_that_falls_off_the_model_rejoins_without_a_relaunch(server, tmp_path):
    """GPU-142, end to end: node 2 cannot average round 3, which the others
    decided over all three. It declares its number gone, takes a new one, comes
    back in through the join, and the run ends on one model - with the
    training script never having stopped."""
    pytest.importorskip("moonclip")
    context = mp.get_context("spawn")
    queue = context.Queue()
    job = "splits"
    rendezvous.connect(server, job + ".control").set(LAST_KEY, b"9")
    nodes = [
        context.Process(
            target=a_node, args=(server, job, 3, str(tmp_path), name, "splits", queue)
        )
        for name in ("a", "b", "c")
    ]
    for node in nodes:
        node.start()
    try:
        results = collect(queue, 3, str(tmp_path), "abc")
    finally:
        end_nodes(nodes)

    digests = {name: digest for name, (digest, _, _) in results.items()}
    assert len(set(digests.values())) == 1, (
        "the nodes ended holding different models: %s (at rounds %s)"
        % (digests, {name: last for name, (_, last, _) in results.items()})
    )
    rejoined = [
        name for name, (_, _, said) in results.items()
        if any("Rejoined the run as node" in line for line in said)
    ]
    assert len(rejoined) == 1, results


# ─── the rendezvous itself dies (GPU-229) ──────────────────────────────


def free_ports(count):
    """Ports nothing listens on now. Probed and released, which GPU-175 warns
    can race another worker; a candidate's port has to be known before any
    server is on it, and that is what is being tested."""
    import socket as _socket

    held = []
    try:
        for _ in range(count):
            sock = _socket.socket()
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        return [sock.getsockname()[1] for sock in held]
    finally:
        for sock in held:
            sock.close()


def a_server(port):
    """``ravex rendezvous`` in a process of its own, as smith starts it."""
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "-m", "ravex._cli", "rendezvous", "--host", "127.0.0.1",
         "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert b"listening" in process.stdout.readline(), "the server did not start"
    return process


def stop_servers_on(ports):
    """Kill whatever ``ravex rendezvous`` was started on ``ports``: an elected
    node starts one that outlives it on purpose, and a test that failed never
    got to read its pid from the node's log."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % entry, "rb") as handle:
                words = handle.read().split(b"\0")
        except OSError:
            continue
        if b"rendezvous" in words and any(
            str(port).encode() in words for port in ports
        ):
            try:
                os.kill(int(entry), 9)
            except OSError:
                pass


def hosted_pids(said):
    import re

    return [
        int(found.group(1))
        for line in said
        for found in [re.search(r"rendezvous from this machine at .*, pid (\d+)", line)]
        if found
    ]


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
@pytest.mark.parametrize("overlap", [False, True], ids=["blocking", "overlap"])
def test_the_run_elects_a_new_rendezvous_when_its_own_dies(server, tmp_path, overlap):
    """The server is killed at round 3, every node's state on it. The nodes
    elect one among themselves, start epoch 1 there, and end on one model -
    and a node started after the failover finds the run where it is now,
    though the first address it is given is the dead one."""
    pytest.importorskip("moonclip")
    context = mp.get_context("spawn")
    queue = context.Queue()
    job = "failover-overlap" if overlap else "failover"
    ports = free_ports(4)
    addresses = ",".join("127.0.0.1:%d" % port for port in ports)
    dying = a_server(ports[0])
    control = rendezvous.connect(server, job + ".control")
    control.set(LAST_KEY, b"14")
    nodes = [
        context.Process(
            target=a_node,
            args=(addresses, job, 3, str(tmp_path), name, "server-dies", queue),
            # A node that arrives after the epoch's quorum joins it, and a
            # join takes a few rounds; on a loaded runner a minute is not
            # always that.
            kwargs=dict(control_at=server, member=place, kill=dying.pid, deadline=180,
                        overlap=overlap),
        )
        for place, name in ((1, "a"), (2, "b"), (3, "c"))
    ]
    for node in nodes:
        node.start()
    joiner = None
    results = {}
    try:
        # The joiner starts once the run is on epoch 1 somewhere.
        deadline = time.monotonic() + 240
        while True:
            assert time.monotonic() < deadline, "the run never moved to epoch 1"
            try:
                found, epoch, _ = rendezvous.current(
                    addresses.split(","), job, time.monotonic() + 1, timeout=10
                )
                if epoch >= 1:
                    break
            except TimeoutError:
                pass
            time.sleep(0.5)
        joiner = context.Process(
            target=a_node,
            args=(addresses, job, 3, str(tmp_path), "d", "joiner", queue),
            kwargs=dict(control_at=server, deadline=180, overlap=overlap),
        )
        joiner.start()
        results = collect(queue, 4, str(tmp_path), "abcd")
    finally:
        end_nodes(nodes + ([joiner] if joiner else []))
        dying.kill()
        dying.wait()
        # The servers the elected nodes started outlive them on purpose. Each
        # node said its server's pid; /proc finds one whose node never got to
        # say it, on Linux - elsewhere, said is all there is, and without this
        # every run left one listening on Windows.
        for pid in (pid for _, _, said in results.values() for pid in hosted_pids(said)):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        if sys.platform.startswith("linux"):
            stop_servers_on(ports)

    digests = {name: digest for name, (digest, _, _) in results.items()}
    assert len(set(digests.values())) == 1, (
        "the nodes ended holding different models: %s (at rounds %s)"
        % (digests, {name: last for name, (_, last, _) in results.items()})
    )
    # Two candidates may open a server at once - each counts its turn from
    # when it noticed the loss, not from a clock they share - and the one
    # placed later closes its own on finding the other. One is left.
    hosts = [
        name for name, (_, _, said) in results.items()
        if hosted_pids(said) and not any("closing this one" in line for line in said)
    ]
    assert len(hosts) == 1, "%d nodes hosted epoch 1: %s" % (len(hosts), hosts)
    # Each survivor is in epoch 1 either as one of the nodes that started it
    # or, arriving after its quorum, through the join.
    for name in ("a", "b", "c"):
        said = results[name][2]
        assert any(
            "Carried on in epoch 1" in line or "Joined epoch 1" in line for line in said
        ), (name, said)
    assert sum(
        any("Carried on in epoch 1" in line for line in results[name][2])
        for name in ("a", "b", "c")
    ) >= 2, "epoch 1 opened without its quorum"
    assert any("Joined the run at round" in line for line in results["d"][2]), results["d"][2]
    for name, (_, _, said) in results.items():
        assert rounds_over(said, 4), "node %s never closed a round with the joiner" % name
