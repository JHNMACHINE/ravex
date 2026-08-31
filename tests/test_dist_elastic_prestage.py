"""GPU-94 step 4: pre-staging a joining rank's copy of the checkpoint while
the old ranks keep training - over a plain socket, not torch.distributed.

**Why not torch.distributed for this.** The first design tried was reusing
``ravex._dist.replication.exchange_stores`` on a small, independent
``ProcessGroupGloo`` built by hand between one old rank and the candidate,
so the real training group would never be touched. The independent group
works fine for a *collective* - see
:func:`test_a_raw_side_group_does_not_disturb_the_live_default_group`, which
keeps that finding on record even though production code does not use it -
but ``exchange_stores`` moves bytes with ``dist.isend``/``dist.recv``, and
those refuse a group that was never registered through
``torch.distributed.new_group``. The private escape hatch
(``_register_process_group``) does not help either - it rejects the same
``ProcessGroupGloo`` instance with a pybind11 type-mismatch. And the public
``new_group()`` needs every rank of the *current* default group to call it,
which a not-yet-member candidate cannot do. Full account in
``ravex._dist.elastic``'s module docstring. So :func:`ravex._dist.elastic.prestage_send`
/ :func:`ravex._dist.elastic.prestage_receive` move bytes over a plain, connected
TCP socket instead, reusing ``ravex._dist.replication``'s manifest/skip/
``StoreWriter`` machinery directly - none of that depends on a process group,
only the transport ``exchange_stores`` wires it to does.

**What the incremental skip actually buys**, measured on the real transport:
:func:`test_the_second_round_sends_only_what_changed` runs two prestage
rounds over a real socket, with a real training step (``all_reduce`` on the
untouched default group) interleaved between them. Round 1 has nothing to
skip. Between the rounds, one file changes, one is added, one is left
alone. The byte count is measured by wrapping the real ``encode_store``
generator ravex already uses - the assertion is that the wire saw less, not
that it matches an independently-derived expectation.

**What this does not prove.** Loopback, one box, small files. Nothing here
says how long a real round costs at 7-12 MB/s over a real inter-machine
link (see the design notes for GPU-94) - only that the amount of data that
has to make that trip shrinks correctly, and that moving it over a bare
socket does not disturb the real training group's own collectives.
"""

import datetime
import multiprocessing as mp
import os
import socket
import traceback

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_file(path, content: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


def _read_file(path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


# ─── on record: a raw side *group* is fine for a collective, just not for
# ─── exchange_stores - see the module docstring for why production code
# ─── does not use this path.


def _coexistence_worker(rank, main_port, side_port, out):
    try:
        import torch
        import torch.distributed as dist

        if rank in (0, 1):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(main_port)
            dist.init_process_group("gloo", rank=rank, world_size=2)
            before = torch.tensor([1.0])
            dist.all_reduce(before)
            out.put((rank, "default_before", before.item()))

        if rank in (0, 2):
            side_rank = 0 if rank == 0 else 1
            side_store = dist.TCPStore(
                "127.0.0.1", side_port, world_size=2,
                is_master=(rank == 0), use_libuv=False,
            )
            side_pg = dist.ProcessGroupGloo(
                side_store, side_rank, 2, datetime.timedelta(seconds=10)
            )
            side_t = torch.tensor([float(rank)])
            dist.all_reduce(side_t, group=side_pg)
            out.put((rank, "side_group", side_t.item()))

        if rank in (0, 1):
            after = torch.tensor([1.0])
            dist.all_reduce(after)
            out.put((rank, "default_after", after.item()))
            dist.destroy_process_group()
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def test_a_raw_side_group_does_not_disturb_the_live_default_group():
    ctx = mp.get_context("spawn")
    main_port = _free_port()
    side_port = _free_port()
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_coexistence_worker, args=(r, main_port, side_port, out))
        for r in (0, 1, 2)
    ]
    for p in procs:
        p.start()
    messages = []
    killed = []
    try:
        for _ in range(6):
            try:
                messages.append(out.get(timeout=30))
            except Exception:
                break
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                killed.append(p.pid)
                p.terminate()

    assert killed == [], "a rank hung - the two groups did not coexist cleanly"
    by_rank_phase = {(r, phase): value for r, phase, value in messages}
    assert by_rank_phase.get((0, "default_before")) == 2.0
    assert by_rank_phase.get((1, "default_before")) == 2.0
    assert by_rank_phase.get((0, "default_after")) == 2.0
    assert by_rank_phase.get((1, "default_after")) == 2.0
    assert by_rank_phase.get((0, "side_group")) == 2.0
    assert by_rank_phase.get((2, "side_group")) == 2.0


# ─── the real pre-staging path: a plain socket, GPU-97's skip on top ────


def _prestage_worker(rank, main_port, socket_port, source_dir, dest_dir, out):
    try:
        import socket as socket_module
        import torch
        import torch.distributed as dist

        import ravex._dist.replication as replication
        from ravex._dist.elastic import prestage_send, prestage_receive

        sent_bytes = {"round1": 0, "round2": 0}
        current_round = {"n": 1}
        real_encode_store = replication.encode_store

        def counting_encode_store(path, chunk=replication.CHUNK, skip=None):
            for block in real_encode_store(path, chunk=chunk, skip=skip):
                sent_bytes["round%d" % current_round["n"]] += len(block)
                yield block

        replication.encode_store = counting_encode_store

        if rank in (0, 1):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(main_port)
            dist.init_process_group("gloo", rank=rank, world_size=2)
            dist.all_reduce(torch.tensor([1.0]))  # a real training step

        sock = None
        if rank == 2:
            # The candidate listens - in a real join this address is exactly
            # what `ravex._dist.elastic.announce_join` would have advertised.
            server = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
            server.bind(("127.0.0.1", socket_port))
            server.listen(1)
            out.put((rank, "listening", None))
        if rank == 0:
            # Give rank 2 a moment to bind before connecting.
            import time
            time.sleep(1.0)
            sock = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
            sock.connect(("127.0.0.1", socket_port))

        if rank == 2:
            conn, _ = server.accept()
            sock = conn

        # Round 1: nothing at the destination yet - full transfer.
        current_round["n"] = 1
        if rank == 0:
            prestage_send(sock, source_dir)
        elif rank == 2:
            prestage_receive(sock, dest_dir)

        if rank in (0, 1):
            # The old ranks keep training in between the two rounds - the
            # whole point of pre-staging in the background.
            dist.all_reduce(torch.tensor([1.0]))

        if rank == 0:
            # a.bin untouched, b.bin changes, c.bin is new.
            _write_file(os.path.join(source_dir, "b.bin"), b"B" * 4096 + b"-v2")
            _write_file(os.path.join(source_dir, "c.bin"), b"C" * 2048)

        current_round["n"] = 2
        if rank == 0:
            prestage_send(sock, source_dir)
        elif rank == 2:
            prestage_receive(sock, dest_dir)

        if rank in (0, 1):
            dist.all_reduce(torch.tensor([1.0]))
            dist.destroy_process_group()

        if sock is not None:
            sock.close()

        if rank == 0:
            out.put((rank, "bytes", dict(sent_bytes)))
        elif rank == 1:
            out.put((rank, "ok", None))
        else:
            out.put((rank, "ok", None))
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def test_the_second_round_sends_only_what_changed(tmp_path):
    source_dir = str(tmp_path / "source")
    dest_dir = str(tmp_path / "dest")

    a_content = b"A" * 100_000  # left alone across both rounds
    b_content_v1 = b"B" * 4096
    _write_file(os.path.join(source_dir, "a.bin"), a_content)
    _write_file(os.path.join(source_dir, "b.bin"), b_content_v1)

    ctx = mp.get_context("spawn")
    main_port = _free_port()
    socket_port = _free_port()
    out = ctx.Queue()
    procs = [
        ctx.Process(
            target=_prestage_worker,
            args=(r, main_port, socket_port, source_dir, dest_dir, out),
        )
        for r in (0, 1, 2)
    ]
    for p in procs:
        p.start()

    results = {}
    killed = []
    try:
        for _ in range(4):  # rank2: listening + ok, rank0: bytes, rank1: nothing
            try:
                rank, status, payload = out.get(timeout=30)
            except Exception:
                break
            if status != "listening":
                results[rank] = (status, payload)
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                killed.append(p.pid)
                p.terminate()

    assert killed == [], results
    for rank, (status, payload) in results.items():
        assert status in ("bytes", "ok"), f"rank {rank}: {payload}"

    sent = results[0][1]

    assert sent["round1"] >= len(a_content) + len(b_content_v1)
    assert sent["round2"] < len(a_content), (
        "round 2 sent %d bytes - at least as much as a.bin alone (%d), which "
        "means the unchanged file was retransmitted instead of skipped"
        % (sent["round2"], len(a_content))
    )

    assert _read_file(os.path.join(dest_dir, "a.bin")) == a_content
    assert _read_file(os.path.join(dest_dir, "b.bin")) == b"B" * 4096 + b"-v2"
    assert _read_file(os.path.join(dest_dir, "c.bin")) == b"C" * 2048
