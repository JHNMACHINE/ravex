"""GPU-94 step 1: can a rank that never existed join a live gloo group,
without any of the original processes leaving their OS process?

Elastic-without-restart needs a rendezvous point that survives
``destroy_process_group()`` + ``init_process_group()`` at a *different*
world_size, inside the same process, with a genuinely new process attaching
for the first time. That is not what ``env://`` rendezvous gives you - the
default store it builds is scoped to one ``init_process_group`` call. This
tests the alternative: a raw ``TCPStore`` created once, held by rank 0 for the
whole test, and passed explicitly as ``store=`` to every
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
keys off the same store. Because the failure is intermittent, the negative
test below repeats the naive attempt several times and requires at least one
failure, rather than asserting it fails outright - a single clean pass does
not mean the bug is gone, only that this particular run got lucky.
"""

import datetime
import multiprocessing as mp
import socket
import traceback

import pytest

PHASE2_TIMEOUT = datetime.timedelta(seconds=10)
GET_TIMEOUT = 20


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rejoin_worker(rank, is_rank0, port, join_evt, out, use_generations):
    """One rank of: world_size 2 -> destroy -> world_size 3, rank 2 is new.

    Top-level and argument-driven because ``spawn`` has to pickle it - the
    same constraint ``test_dist_multinode.py``'s own worker functions document.
    """
    try:
        import torch
        import torch.distributed as dist

        if is_rank0:
            # Rank 0 owns the TCPStore server. It must outlive both
            # generations, which is the entire point: the rendezvous point
            # cannot be torn down with the process group.
            store = dist.TCPStore(
                "127.0.0.1", port, world_size=None, is_master=True, use_libuv=False
            )
            join_evt.set()
        else:
            join_evt.wait(timeout=15)
            store = dist.TCPStore(
                "127.0.0.1", port, world_size=None, is_master=False, use_libuv=False
            )

        if rank in (0, 1):
            gen0 = dist.PrefixStore("gen0", store) if use_generations else store
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
        gen1 = dist.PrefixStore("gen1", store) if use_generations else store
        dist.init_process_group(
            "gloo", store=gen1, rank=rank, world_size=3, timeout=PHASE2_TIMEOUT
        )
        t2 = torch.tensor([1.0])
        dist.all_reduce(t2)
        out.put((rank, "phase2", t2.item()))
        dist.destroy_process_group()
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


def _run_three_rank_rejoin(use_generations):
    """Spawn 0 and 1 at world_size=2, then all three at world_size=3.

    Returns a dict of rank -> list of (phase, value) messages received before
    the collection timeout, plus whether any process had to be killed rather
    than exiting on its own - a hang is the failure mode this whole design
    exists to avoid (see GPU-92's rationale for a short, independent
    timeout), so it is reported as data, not swallowed as cleanup.
    """
    ctx = mp.get_context("spawn")
    port = _free_port()
    join_evt = ctx.Event()
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_rejoin_worker, args=(r, r == 0, port, join_evt, out, use_generations))
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
        messages, killed = _run_three_rank_rejoin(use_generations=True)

        assert killed == [], (
            "a process had to be killed rather than exiting on its own - "
            "that is a hang, not a clean failure, and it is exactly the "
            "failure mode the naive version below produces"
        )
        assert messages[0] == [("phase1", "ok"), ("phase2", 3.0)]
        assert messages[1] == [("phase1", "ok"), ("phase2", 3.0)]
        # Rank 2 was never part of phase 1 - it has no phase1 message at all,
        # and its only result is the phase-2 collective it joined fresh.
        assert messages[2] == [("phase2", 3.0)]

    def test_reusing_the_bare_store_across_generations_is_not_reliable(self):
        """The negative case, kept rather than deleted once the fix was
        found - the same discipline as GPU-98's rule 5: this is what proves
        the ``PrefixStore`` above is load-bearing and not decoration.

        Reusing the same raw store directly for both the world_size=2 and
        world_size=3 ``init_process_group`` calls is the version that looked
        like it should work and was tried first. Phase 1 always succeeds -
        the two original ranks agree on world_size=2 with a store they are
        both new to. Phase 2 is where it breaks, because the store still
        holds handshake keys from phase 1 that a fresh rendezvous at a
        different world_size was never meant to see - but it does not break
        *every* time (see the module docstring), which is why this repeats
        the attempt several times rather than asserting failure on a single
        run: on this box (Windows) 5 repeats failed 5/5 with a deterministic
        ``RuntimeError`` from gloo's own transport layer before a later,
        unrelated run passed cleanly with no code change. Cross-checked in a
        Linux container the failure shape is different again - "Connection
        reset by peer", or every process still alive past the 10 s phase-2
        timeout - and also not on every run there either. What is pinned
        here is the one thing constant across both platforms and both
        failure shapes: over several repeats, at least one must fail to
        deliver three correct results. A test that required failure on
        every single run would itself be wrong about what was actually
        observed.
        """
        attempts = 5
        any_failure = False
        for _ in range(attempts):
            messages, killed = _run_three_rank_rejoin(use_generations=False)
            phase2_by_rank = {
                rank: [v for phase, v in msgs if phase == "phase2"]
                for rank, msgs in messages.items()
            }
            all_correct = all(values == [3.0] for values in phase2_by_rank.values())
            if killed or not all_correct:
                any_failure = True
                break

        assert any_failure, (
            "reusing the bare store across generations succeeded cleanly on "
            "every rank across %d repeats - if torch has started tolerating "
            "this reliably, the PrefixStore-per-generation workaround (in "
            "ravex._dist.elastic.generation_store and in this test) is no longer "
            "needed and should be revisited, not left in out of caution"
            % attempts
        )
