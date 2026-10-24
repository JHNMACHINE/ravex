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

import os
import threading
import time

import pytest


from ravex._dist import rendezvous  # noqa: E402
from ravex._dist.membership import Membership, announce  # noqa: E402


@pytest.fixture
def server():
    # Port 0, and the port read back from the server: a port probed free and
    # then handed over could be taken by another worker's server in between,
    # and under `pytest -n` it was - EADDRINUSE at the listen (GPU-175).
    store = rendezvous.serve("127.0.0.1", 0)
    yield "127.0.0.1:%d" % store.port
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


def test_round_reports_are_staged_beside_the_store_and_not_in_it(tmp_path):
    """GPU-240: inside the store, the closing sync carried every report up to
    the bucket and a restore brought them down again. Every way a node opens
    its exchange - the start, a rejoin, an epoch after a failover - stages
    outside the directory the checkpoints sync from."""
    from ravex._config import RavexConfig
    from ravex._runtime import RavexRuntime

    runtime = RavexRuntime.__new__(RavexRuntime)
    runtime.config = RavexConfig()
    runtime.config.storage.path = str(tmp_path / "store") + os.sep
    runtime._epoch = 0
    store = str(tmp_path / "store")

    for epoch in (0, 2):
        runtime._epoch = epoch
        root = os.path.abspath(runtime._round_root(1))
        assert os.path.commonpath([root, store]) != store, root
        assert root.startswith(store + ".rounds"), root
    assert os.path.abspath(runtime._rounds_base()) == store + ".rounds"

    # Given explicitly, the place is the caller's, as before.
    runtime.config.outer_root = str(tmp_path / "elsewhere")
    assert runtime._round_root(1).startswith(runtime.config.outer_root)


# ─── whole nodes ───────────────────────────────────────────────────────


# ─── the rendezvous itself dies (GPU-229) ──────────────────────────────


def test_a_rendezvous_dying_while_it_is_joined_is_passed_over(monkeypatch):
    """The run's first server answers the probe and dies before its store is
    opened - the moment a failover starts. Whoever asks then finds epoch 1 on
    the next address, rather than the dead server's connection error: that
    error failed the election test every time on a machine where a connect is
    slow, and would fail a node arriving at that moment the same way."""
    dead, elected = "127.0.0.1:1", "127.0.0.1:2"
    alive = {dead: True, elected: True}

    class Record:
        """The job's record on a server: epoch 1 is recorded on the elected
        one only once the first server is gone, as a failover does it."""

        def __init__(self, address):
            self.address = address

        def check(self, keys):
            return self.address == elected and not alive[dead]

        def get(self, key):
            return b"1"

    asked = []

    def connect(address, job, timeout):
        if not alive[address]:
            raise RuntimeError("The client socket has timed out")
        asked.append(address)
        if address == dead and asked.count(dead) == 2:
            # Its record was read; asked for its store, it is gone.
            alive[dead] = False
            raise RuntimeError("The client socket has timed out")
        if job == "run" and (address != dead or asked.count(dead) == 1):
            return Record(address)
        return ("store", address, job)

    monkeypatch.setattr(rendezvous, "answers", lambda address, timeout=0: alive[address])
    monkeypatch.setattr(rendezvous, "connect", connect)
    monkeypatch.setattr(rendezvous.time, "sleep", lambda seconds: None)
    store, epoch, address = rendezvous.current([dead, elected], "run", time.monotonic() + 30, timeout=5)
    assert (epoch, address) == (1, elected)
    assert store == ("store", elected, rendezvous.epoch_job("run", 1))
