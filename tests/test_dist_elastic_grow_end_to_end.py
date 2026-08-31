"""GPU-94 step 5: the whole grow path, end to end, on one box - every piece
from steps 1-4 composed into a single 2 -> 3 rank scenario, gloo/CPU.

The scenario, in order: rank 2 announces itself on the persistent store
(step 3) while ranks 0 and 1 are mid-training on a real FSDP2 model. Rank 0
polls and finds it, and ranks 0/1 agree over the isolated topology channel
(step 3) that a join is happening. Rank 0 pre-stages its checkpoint to rank
2 over a plain socket in two rounds with a real training step interleaved
(step 4) - not toy bytes, an actual ``torch.save`` of the trained
parameters' full tensors, read back with ``torch.load`` on the other end.
At the agreed step, all three tear down and rejoin at world_size=3 on the
same persistent store, one generation later (step 1). Each rank rebuilds a
fresh module from the values it has - ranks 0/1 from what is already in
memory, rank 2 from what pre-staging actually delivered to its disk - and
calls ``fully_shard()`` fresh under the new three-rank mesh (step 2).

**What "end to end" means here and what it does not.** Every transition is
the real mechanism from its own step's test, not a stand-in: real
``announce_join``/``pending_join``, a real isolated-subgroup
``topology_decision``, real ``prestage_send``/``prestage_receive`` moving an
actual file rank 2 never had, a real ``destroy_process_group`` +
``init_process_group`` at a new world_size, a real second ``fully_shard()``
after a real first one. What is not real: the model (a 3-layer toy, CPU,
gloo), the "training" (one ``backward()`` per step, not an optimizer with
real moments carried across the regroup - see the design notes on what
reshard-on-resume already does and does not carry across a topology
change), and the network (loopback - see every earlier step's tests for
why that specifically does not stand in for the 7-12 MB/s inter-machine
link this is ultimately for).

**Three real bugs this test caught while being written, none of them in
steps 1-4's own mechanisms** - each one is exactly the kind of assumption
this composition was for surfacing, and each is now guarded against
explicitly rather than fixed silently:

1. **``DTensor.full_tensor()`` is a collective and has to be called by
   every rank of the group, even the ranks that throw the result away.**
   The first draft called it on rank 0 alone (only rank 0 needed the
   values, to write them to disk). That desynchronised
   ``torch.distributed``'s internal per-process group-naming counter
   between ranks 0 and 1 - invisibly, with no error at the call site -
   and it only surfaced later as an unrelated-looking timeout inside
   ``topology_decision``'s ``new_group()`` call, with an error message
   naming a rendezvous key neither rank could explain. Now every
   ``full_tensor()`` call here is made by both ranks together.
2. **Never call ``full_tensor()`` on a module whose process group has
   already been destroyed.** After the ``destroy_process_group()`` /
   ``init_process_group()`` regroup, the old ``model``'s ``DTensor``
   parameters are still meshed against the world_size=2 group that no
   longer exists. Calling ``full_tensor()`` on it then does not raise
   cleanly - it corrupts, surfacing as a baffling
   ``RuntimeError: narrow unexpectedly changed concrete size``. The values
   captured *before* the destroy (``state2``) are the last-known-good
   ones and are what ranks 0/1 use as their ground truth after the
   regroup - not a fresh call on a dead mesh.
3. **Store file names have to be unique per snapshot, never reused.**
   The first draft overwrote one ``state.pt`` path on every round.
   ``ravex._dist.replication``'s incremental skip (GPU-97) matches files by
   name and size alone, deliberately, on the assumption that store files
   are immutable once written - overwriting one with same-shape,
   different-value tensors produces the same size, so the skip silently
   treated genuinely new data as unchanged and rank 2 kept a stale copy
   with no error anywhere. Exactly the "a hole is worse than a failure"
   case ``ravex._dist.reshard`` already argues for elsewhere - reached here
   from the opposite direction, by a test violating an invariant real
   ravex checkpoints already respect (every step gets its own path).
   Fixed by giving each round its own file name, the way a real store
   would.
"""

import datetime
import multiprocessing as mp
import os
import socket
import time
import traceback

import pytest
import torch


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _toy_module():
    import torch.nn as nn

    return nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 4))


def _grow_worker(rank, store_port, socket_port, source_dir, dest_dir, out):
    try:
        import torch.distributed as dist
        from torch.distributed.fsdp import fully_shard

        from ravex._dist.elastic import (
            announce_join,
            generation_store,
            pending_join,
            prestage_receive,
            prestage_send,
            topology_decision,
        )

        is_rank0 = rank == 0
        if is_rank0:
            base_store = dist.TCPStore(
                "127.0.0.1", store_port, world_size=None, is_master=True, use_libuv=False
            )
        else:
            # Ranks 1 and 2 both need rank 0's server up first - the test
            # gives it a moment below rather than a formal barrier, since no
            # group exists yet for rank 2 to be barriered on.
            time.sleep(0.5)
            base_store = dist.TCPStore(
                "127.0.0.1", store_port, world_size=None, is_master=False, use_libuv=False
            )

        # ─── ranks 0/1: the original training group, generation 0 ──────
        if rank in (0, 1):
            gen0 = generation_store(base_store, 0)
            dist.init_process_group("gloo", store=gen0, rank=rank, world_size=2)

            torch.manual_seed(0)
            model = _toy_module()
            fully_shard(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model(torch.randn(4, 12)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # full_tensor() redistributes a sharded DTensor - a collective -
            # so every rank must call it together even though only rank 0
            # needs the result. Debug note (GPU-94): calling it on rank 0
            # alone desynchronised torch's internal process-group counter
            # between the ranks, which only surfaced later as a timeout in
            # an unrelated new_group() call.
            with torch.no_grad():
                state = {
                    name: p.full_tensor().detach().clone()
                    for name, p in model.named_parameters()
                }
            if is_rank0:
                os.makedirs(source_dir, exist_ok=True)
                # A distinct name per snapshot, not an overwrite - store files
                # are assumed immutable once written (see GPU-97's skip:
                # matched by name and size alone, no hash). Overwriting
                # "state.pt" in place produced a same-size, different-content
                # file that the skip logic silently treated as unchanged -
                # exactly the "hole" this design is supposed to refuse, and a
                # real ravex store never does this because every checkpoint
                # step already gets its own path.
                torch.save(state, os.path.join(source_dir, "state_round1.pt"))

        # ─── rank 2: announce itself while 0/1 are mid-training ─────────
        if rank == 2:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("127.0.0.1", socket_port))
            server.listen(1)
            announce_join(base_store, candidate_rank=2, address="127.0.0.1:%d" % socket_port)

        # ─── rank 0: discover the candidate ──────────────────────────────
        candidate_address = None
        if is_rank0:
            for _ in range(100):  # bounded - no silent infinite poll
                candidate_address = pending_join(base_store, candidate_rank=2)
                if candidate_address is not None:
                    break
                time.sleep(0.05)
            if candidate_address is None:
                raise TimeoutError("rank 2 never announced itself")

        # ─── ranks 0/1: agree a join is happening (step 3's channel) ────
        if rank in (0, 1):
            local_view = {"wants_join": is_rank0 and candidate_address is not None}
            decision = topology_decision(local_view, timeout_seconds=10)
            agreed = any(view.get("wants_join") for view in decision)
            out.put((rank, "decision", agreed))
            if not agreed:
                raise AssertionError("ranks did not agree a join was happening")

        # ─── pre-stage round 1: full transfer, rank 0 <-> rank 2 ────────
        sock = None
        if is_rank0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", socket_port))
            prestage_send(sock, source_dir)
        elif rank == 2:
            conn, _ = server.accept()
            sock = conn
            prestage_receive(sock, dest_dir)

        # ─── ranks 0/1 keep training in between - the point of pre-staging ──
        if rank in (0, 1):
            model(torch.randn(4, 12)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():  # collective - see the note on the first full_tensor() above
                state2 = {
                    name: p.full_tensor().detach().clone()
                    for name, p in model.named_parameters()
                }
            if is_rank0:
                torch.save(state2, os.path.join(source_dir, "state_round2.pt"))

        # ─── pre-stage round 2: only the delta ──────────────────────────
        if is_rank0:
            prestage_send(sock, source_dir)
        elif rank == 2:
            prestage_receive(sock, dest_dir)

        if rank in (0, 1):
            dist.destroy_process_group()
        if sock is not None:
            sock.close()

        # ─── cutover: regroup at world_size=3, one generation later ─────
        gen1 = generation_store(base_store, 1)
        dist.init_process_group(
            "gloo", store=gen1, rank=rank, world_size=3,
            timeout=datetime.timedelta(seconds=15),
        )

        if rank in (0, 1):
            # `model`'s DTensors are meshed against the world_size=2 group
            # just destroyed above - calling full_tensor() on it now redistributes
            # against a dead process group and corrupts silently (surfaced here as
            # "narrow unexpectedly changed concrete size", not as a clean error).
            # `state2`, captured before the destroy, is the last-known-good values.
            ground_truth = state2
        else:
            # The newest snapshot the candidate actually received - round 2's
            # delta only makes sense on top of round 1 having arrived first.
            loaded = torch.load(
                os.path.join(dest_dir, "state_round2.pt"), weights_only=True
            )
            ground_truth = loaded

        fresh = _toy_module()
        with torch.no_grad():
            for name, param in fresh.named_parameters():
                param.copy_(ground_truth[name])
        fully_shard(fresh)

        after = {
            name: param.full_tensor().detach().clone()
            for name, param in fresh.named_parameters()
        }
        # Ranks 0/1 compare against their own in-memory ground truth (always
        # correct by construction). Rank 2 compares against what pre-staging
        # actually wrote to disk - the real claim being tested.
        mismatches = [
            name for name in ground_truth if not torch.equal(after[name], ground_truth[name])
        ]

        # Prove it trains for real under the new three-rank mesh.
        fresh(torch.randn(4, 12)).sum().backward()

        out.put((rank, "ok", mismatches))
        dist.destroy_process_group()
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def test_a_candidate_grows_the_group_without_a_process_restart(tmp_path):
    source_dir = str(tmp_path / "source_store")
    dest_dir = str(tmp_path / "dest_store")

    ctx = mp.get_context("spawn")
    store_port = _free_port()
    socket_port = _free_port()
    out = ctx.Queue()
    procs = [
        ctx.Process(
            target=_grow_worker,
            args=(r, store_port, socket_port, source_dir, dest_dir, out),
        )
        for r in (0, 1, 2)
    ]
    for p in procs:
        p.start()

    results = {}
    decisions = {}
    debug_msgs = []
    killed = []

    def _drain():
        while True:
            try:
                rank, kind, payload = out.get(timeout=1)
            except Exception:
                return
            if kind == "decision":
                decisions[rank] = payload
            elif kind == "debug":
                debug_msgs.append((rank, payload))
            else:
                results[rank] = (kind, payload)

    try:
        # Wait for the processes to actually finish before draining: pulling
        # from the queue while a process is mid-put, then joining it right
        # after, risks losing whatever the feeder thread had not flushed yet.
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                killed.append(p.pid)
                p.terminate()
                p.join(timeout=5)
        _drain()
    finally:
        for p in procs:
            if p.is_alive():  # pragma: no cover - already handled above
                p.terminate()

    assert killed == [], f"a rank hung: results so far {results}, decisions {decisions}"
    assert decisions == {0: True, 1: True}, (
        f"ranks 0/1 did not both agree a join was happening: {decisions}"
    )
    assert set(results) == {0, 1, 2}, results
    for rank, (status, payload) in results.items():
        assert status == "ok", f"rank {rank}: {payload}"
        assert payload == [], (
            f"rank {rank} mismatched parameters after growing: {payload}; debug: {debug_msgs}"
        )
