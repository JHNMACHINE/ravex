"""GPU-109: a replication round over Ravex's own sockets, next to the old one.

``exchange_stores`` has two roads now. The bytes can go over ``dist.isend`` /
``dist.recv`` as they always have, or over a :class:`~ravex._dist.replication.RingLink`
— one TCP connection to the ring successor, one from the predecessor, both
opened once and reused — with the framing and the loop in the Rust core.

**What has to be true for that to be a road and not a fork.** Both must leave
the receiving disk in the same state, down to the file list and the completion
marker, because everything downstream reads a replica without knowing or caring
which way it arrived: ``replica_is_complete``, ``promote_copy``, the recovery
pass in ``_recover_missing_stores``. So the test that matters is not "the socket
road works" — it is "the two roads leave the same thing", and that is what is
asserted here, in one process pair, on the same store, in the same run.

**And the reverse direction, which only the socket road has to think about.**
The recovery round sends a copy *back* to the rank it belongs to, against the
ring. On the collective road that is just two different rank numbers. On the
socket road the connection a rank made and the one it accepted are different
objects, and on a two-rank job the successor and the predecessor are the same
peer — so the direction cannot be recovered from the ranks and is passed in.
Getting it backwards would mean both ends pushing into each other and neither
reading, which is a hang, and :func:`test_the_ring_runs_backwards_for_a_recovery`
is what says it does not.

**What this does not prove.** Loopback, one box, two ranks, small files. It says
the road is the same road; it says nothing about what it costs between two real
machines, where the link has been measured at 12 MB/s and is the whole story
(GPU-96). `bench/transport_ceiling.py` is where the cost question lives.
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


def _manifest(path):
    """The directory as the rest of Ravex reads it: names and sizes, sorted."""
    from ravex._dist.replication import store_files

    return sorted(store_files(path))


def _both_roads_worker(rank, port, root, out):
    """Rank 0 sends its store to rank 1, once each way, and the two are compared.

    One process pair, both roads, so a difference cannot be blamed on a
    different store or a different moment.
    """
    try:
        import torch.distributed as dist

        from ravex._dist.replication import RingLink, exchange_stores

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60)
        )

        source = os.path.join(root, "rank0-store")
        over_sockets = os.path.join(root, "replica-sockets")
        over_collectives = os.path.join(root, "replica-collectives")
        if rank == 0:
            _write_file(os.path.join(source, "manifest.json"), b'{"step": 7}')
            _write_file(os.path.join(source, "step_7", "shard.bin"), b"S" * 40_000)
            _write_file(os.path.join(source, "empty.bin"), b"")
        else:
            os.makedirs(over_sockets, exist_ok=True)
            os.makedirs(over_collectives, exist_ok=True)
        dist.barrier()

        store = dist.distributed_c10d._get_default_store()
        # A two-rank ring: each rank's successor and predecessor are the same
        # peer, which is the case the direction argument exists for.
        send_to = receive_from = 1 - rank
        link = RingLink.connect(rank, send_to, receive_from, store, timeout_seconds=30)
        if link is None:
            out.put((rank, "no-link", None))
            return

        # Road one: the sockets.
        ok_sockets = exchange_stores(
            source if rank == 0 else None,
            over_sockets if rank == 1 else None,
            send_to=send_to,
            receive_from=receive_from,
            link=link,
        )
        dist.barrier()

        # Road two: the collectives, same store, same pair, no link.
        ok_collectives = exchange_stores(
            source if rank == 0 else None,
            over_collectives if rank == 1 else None,
            send_to=send_to,
            receive_from=receive_from,
        )
        dist.barrier()

        payload = {"ok_sockets": bool(ok_sockets), "ok_collectives": bool(ok_collectives)}
        if rank == 1:
            payload["sockets"] = _manifest(over_sockets)
            payload["collectives"] = _manifest(over_collectives)
            payload["shard"] = os.path.getsize(
                os.path.join(over_sockets, "step_7", "shard.bin")
            )
        else:
            payload["source"] = _manifest(source)

        link.close()
        dist.destroy_process_group()
        out.put((rank, "done", payload))
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def _backwards_worker(rank, port, root, out):
    """The recovery direction: rank 1 hands the copy it holds back to rank 0."""
    try:
        import torch.distributed as dist

        from ravex._dist.replication import RingLink, exchange_stores

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60)
        )

        held = os.path.join(root, "copy-of-rank0")
        recovered = os.path.join(root, "rank0-store")
        if rank == 1:
            _write_file(os.path.join(held, "manifest.json"), b'{"step": 7}')
            _write_file(os.path.join(held, "step_7", "shard.bin"), b"R" * 20_000)
        else:
            os.makedirs(recovered, exist_ok=True)
        dist.barrier()

        store = dist.distributed_c10d._get_default_store()
        send_to = receive_from = 1 - rank
        link = RingLink.connect(rank, send_to, receive_from, store, timeout_seconds=30)
        if link is None:
            out.put((rank, "no-link", None))
            return

        # Exactly the call `_recover_missing_stores` makes: the ranks are
        # swapped and the direction is declared.
        ok = exchange_stores(
            held if rank == 1 else None,
            recovered if rank == 0 else None,
            send_to=receive_from,
            receive_from=send_to,
            link=link,
            forward=False,
        )
        dist.barrier()

        payload = {"ok": bool(ok)}
        if rank == 0:
            payload["recovered"] = _manifest(recovered)

        link.close()
        dist.destroy_process_group()
        out.put((rank, "done", payload))
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def _run(worker, tmp_path):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=worker, args=(rank, port, str(tmp_path), out))
        for rank in (0, 1)
    ]
    for process in procs:
        process.start()

    results = {}
    killed = []
    try:
        for _ in range(2):
            try:
                rank, status, payload = out.get(timeout=120)
            except Exception:
                break
            results[rank] = (status, payload)
    finally:
        for process in procs:
            process.join(timeout=15)
            if process.is_alive():
                killed.append(process.pid)
                process.terminate()

    assert killed == [], "a rank hung: %r" % (results,)
    for rank, (status, payload) in results.items():
        if status == "EXC":
            pytest.fail("rank %d: %s" % (rank, payload))
        if status == "no-link":
            pytest.skip("no socket ring could be built on this host")
    assert set(results) == {0, 1}, results
    return {rank: payload for rank, (_status, payload) in results.items()}


def test_both_roads_leave_the_same_replica(tmp_path):
    results = _run(_both_roads_worker, tmp_path)

    assert results[0]["ok_sockets"] is True
    assert results[1]["ok_sockets"] is True
    assert results[1]["sockets"], "the socket road left an empty directory"

    # The point of the test. Not "the socket road produced something
    # plausible" - the same file list, the same sizes, the same marker, as the
    # road everything downstream was written against.
    assert results[1]["sockets"] == results[1]["collectives"]
    assert results[1]["shard"] == 40_000

    names = {name for name, _size in results[1]["sockets"]}
    assert "empty.bin" in names, "a zero-length file is created by being reached"
    assert ".ravex-replica-ok" in names


def test_the_ring_runs_backwards_for_a_recovery(tmp_path):
    results = _run(_backwards_worker, tmp_path)

    assert results[0]["ok"] is True
    names = {name for name, _size in results[0]["recovered"]}
    assert "manifest.json" in names
    assert "step_7/shard.bin" in names
