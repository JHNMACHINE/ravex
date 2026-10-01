"""GPU-94 step 1: can a rank that never existed join a live gloo group,
without any of the original processes leaving their OS process?

Elastic-without-restart needs a rendezvous point that survives
``destroy_process_group()`` + ``init_process_group()`` at a *different*
world_size, inside the same process, with a genuinely new process attaching
for the first time. That is not what ``env://`` rendezvous gives you - the
default store it builds is scoped to one ``init_process_group`` call. This
tests the alternative: a raw ``TCPStore`` created once, held by the test's own
process for the whole test, and passed explicitly as ``store=`` to every
``init_process_group`` call on every rank, original or joining.

**What this does not prove.** Three processes on one box, gloo, loopback. No
bandwidth, no real network, and nothing here touches NCCL - GPU-89's own
resharding is FSDP2/DTensor, and DTensor's process group is typically NCCL on
GPU. Whether the *destroy+reinit* pattern this test exercises is safe for a
live NCCL communicator (not just gloo) is exactly the question that needs real
GPUs to answer - see the design notes for GPU-94. This test answers a narrower
question: does torch's public API even support the rendezvous shape elastic-
without-restart needs, at all. It does, but only one way of the two an
implementer would reasonably try - see below.

**What actually failed first**, so the negative test is not manufactured
after the fact: reusing the same raw ``TCPStore`` directly, for both the
world_size=2 call and the later world_size=3 call, was the first thing tried
because it looked like it should just work - a store is a store. It mostly
does not, but "mostly" is the finding, not a hedge: this failure is
**intermittent, on both platforms**, which is worse than a clean break
because a run can pass by luck and look fixed. On Windows, 5 repeated runs
failed 5/5 with a deterministic error from gloo's own transport layer
(``sizeof(impl_) == bytes.size(). 136 vs 68``, in
``gloo/transport/uv/address.cc``) - but a 6th run, later, succeeded cleanly
on every rank with no code change at all. Cross-checked in a Linux container
(this box has no Linux any other way) to rule out a Windows-only artifact:
it fails there too, differently - "Connection reset by peer" from
``gloo/transport/tcp/pair.cc``, or an outright hang with every process still
alive past a 10 s timeout - and also not on every run. Both platforms need
the same fix: wrap the shared store in a fresh ``PrefixStore`` per
rendezvous "generation" (``gen0`` for the world_size=2 call, ``gen1`` for
world_size=3), so the two calls never read each other's leftover handshake
keys off the same store.

The naive version is not run here. It used to be, repeated five times and
required to break at least once; in CI torch got through all five (GPU-175),
which made the suite fail on a race going our way. Because the failure is
intermittent it cannot be asserted, only described - which is what the
paragraph above is for. The test below pins the mechanism the fix relies on
instead: each generation reads only its own keys.
"""

import datetime
import multiprocessing as mp
import traceback

import pytest

PHASE2_TIMEOUT = datetime.timedelta(seconds=10)
GET_TIMEOUT = 20


def _rejoin_worker(rank, port, out):
    """One rank of: world_size 2 -> destroy -> world_size 3, rank 2 is new.

    Top-level and argument-driven because ``spawn`` has to pickle it - the
    same constraint ``test_dist_multinode.py``'s own worker functions document.
    """
    try:
        import torch
        import torch.distributed as dist

        from ravex._dist.elastic import generation_store

        store = dist.TCPStore(
            "127.0.0.1", port, world_size=None, is_master=False, use_libuv=False
        )

        if rank in (0, 1):
            gen0 = generation_store(store, 0)
            dist.init_process_group("gloo", store=gen0, rank=rank, world_size=2)
            t = torch.tensor([1.0])
            dist.all_reduce(t)
            if t.item() != 2.0:
                raise AssertionError("phase1 all_reduce wrong: %r" % t.item())
            dist.destroy_process_group()
            out.put((rank, "phase1", "ok"))

        # Phase 2: all three, whenever each of them gets here.
        # init_process_group's store-based rendezvous blocks until world_size
        # parties show up, so rank 2 can walk straight in without waiting on
        # any hand-off from 0/1.
        gen1 = generation_store(store, 1)
        dist.init_process_group(
            "gloo", store=gen1, rank=rank, world_size=3, timeout=PHASE2_TIMEOUT
        )
        t2 = torch.tensor([1.0])
        dist.all_reduce(t2)
        out.put((rank, "phase2", t2.item()))
        dist.destroy_process_group()
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def _run_three_rank_rejoin():
    """Spawn 0 and 1 at world_size=2, then all three at world_size=3.

    Returns a dict of rank -> list of (phase, value) messages received before
    the collection timeout, plus whether any process had to be killed rather
    than exiting on its own - a hang is the failure mode this whole design
    exists to avoid (see GPU-92's rationale for a short, independent
    timeout), so it is reported as data, not swallowed as cleanup.
    """
    import torch.distributed as dist

    # The TCPStore server, held here: it must outlive both generations, which
    # is the entire point - the rendezvous point cannot be torn down with the
    # process group. On port 0 and already listening when the ranks are
    # spawned, because a port probed free and handed to a rank to listen on
    # can be taken by another worker in between (GPU-175).
    server = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, use_libuv=False
    )
    port = server.port
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_rejoin_worker, args=(r, port, out))
        for r in (0, 1, 2)
    ]
    for p in procs:
        p.start()

    messages = {0: [], 1: [], 2: []}
    killed = []
    try:
        for _ in range(5):  # phase1 x2 (ranks 0,1) + phase2 x3 (all ranks)
            try:
                rank, phase, value = out.get(timeout=GET_TIMEOUT)
            except Exception:
                break
            messages[rank].append((phase, value))
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                killed.append(p.pid)
                p.terminate()
                p.join(timeout=5)
    return messages, killed


class TestARankThatNeverExistedCanJoin:
    """The core question step 1 was meant to answer, isolated from FSDP2, from
    checkpoints, from ravex entirely: can torch's public distributed API even
    do the rendezvous shape a live, no-restart topology change needs.
    """

    def test_generation_prefixed_store_lets_a_new_rank_join_cleanly(self):
        messages, killed = _run_three_rank_rejoin()

        assert killed == [], (
            "a process had to be killed rather than exiting on its own - "
            "that is a hang, not a clean failure, and it is exactly the "
            "failure mode the naive version in the module docstring produces"
        )
        assert messages[0] == [("phase1", "ok"), ("phase2", 3.0)]
        assert messages[1] == [("phase1", "ok"), ("phase2", 3.0)]
        # Rank 2 was never part of phase 1 - it has no phase1 message at all,
        # and its only result is the phase-2 collective it joined fresh.
        assert messages[2] == [("phase2", 3.0)]

    def test_each_generation_reads_only_its_own_handshake_keys(self):
        """What the ``PrefixStore`` above is for, checked without a race.

        This used to be the negative case run for real: the bare store reused
        across generations, five times over, passing if any attempt broke. It
        bet on a race, and in CI torch won it five times running (GPU-175) -
        a test that goes red when the bug fails to show up says nothing about
        our code. The failure itself is written up in the module docstring.

        What is pinned instead is the mechanism the fix relies on, in one
        process and deterministically: a rendezvous leaves its handshake keys
        behind on the shared store, and the next generation's view of that
        store cannot see them, while the same generation number reaches the
        same keys again. It goes through ``ravex._dist.elastic.generation_store``,
        the function the runtime calls, not a copy of it.
        """
        import torch
        import torch.distributed as dist

        from ravex._dist.elastic import generation_store

        store = dist.TCPStore(
            "127.0.0.1", 0, world_size=None, is_master=True, use_libuv=False
        )
        before = store.num_keys()

        gen0 = generation_store(store, 0)
        dist.init_process_group("gloo", store=gen0, rank=0, world_size=1)
        t = torch.tensor([1.0])
        dist.all_reduce(t)
        dist.destroy_process_group()
        gen0.set("leftover", "from generation 0")

        # The cause, as it stands: the first rendezvous is gone, its keys are
        # not. A bare store reused at another world_size would read them.
        assert store.num_keys() > before

        gen1 = generation_store(store, 1)
        assert not gen1.check(["leftover"])
        assert generation_store(store, 0).check(["leftover"])
        assert generation_store(store, 0).get("leftover") == b"from generation 0"

        # And generation 1 still rendezvous on the same store, around them.
        dist.init_process_group("gloo", store=gen1, rank=0, world_size=1)
        t = torch.tensor([1.0])
        dist.all_reduce(t)
        dist.destroy_process_group()
        assert t.item() == 1.0
