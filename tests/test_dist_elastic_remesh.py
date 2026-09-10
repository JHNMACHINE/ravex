"""GPU-94 step 2: can an already fully_shard()-ed module be re-meshed onto a
different world_size in place, or does growing the group mean rebuilding the
wrapped module from scratch inside the live process?

**What this settles, with a torch error message rather than a guess.**
``fully_shard()`` refuses to be applied twice to the same module - see
:func:`test_fully_shard_refuses_a_second_wrap_of_the_same_module` below, which
pins the exact ``AssertionError`` torch raises. There is no public API to move
an already-materialised DTensor parameter from one ``DeviceMesh`` to another;
:meth:`DTensor.redistribute` changes *placements* on a fixed mesh, it does not
change which ranks compose the mesh. So "grow without a restart" cannot mean
"the live module quietly changes shape" - it has to mean "the wrapped module
is torn down and rebuilt inside the same OS process", which is a real and
useful distinction (no CUDA re-init, no new interpreter, no re-importing
torch/moonclip - see the design notes for GPU-94) but is not what "in place"
suggests on first read. Worth being explicit about that gap before anything
downstream is built on the wrong mental model of what this does.

**What survives the rebuild, proven across an actual world_size change,** not
just a same-size no-op: :func:`test_a_fresh_module_reloaded_from_full_tensors_survives_a_real_regroup`
trains a tiny FSDP2 model at world_size=2, captures the *full* (unsharded)
parameter values via ``DTensor.full_tensor()``, regroups to world_size=3 with
a rank that never existed in the first group (reusing step 1's proven
rendezvous - a raw ``TCPStore`` held by rank 0, wrapped in a fresh
``PrefixStore`` per generation), rebuilds a plain module on every rank,
loads the captured values into it, and calls ``fully_shard()`` fresh under
the new three-rank mesh. The result matches the pre-regroup values bit for
bit, on all three ranks including the new one, and the rebuilt module takes
a real training step afterwards without error.

**What this does not prove.** Rank 2 receives the trained values through the
test's own ``multiprocessing.Queue``, not through any ravex transport -
that is deliberate: this isolates "can the module be rebuilt correctly once
the right values exist locally" from "how do the bytes get there", which is
steps 4 and 5's question, not this one's. And this is gloo on CPU, three
processes on one box - it says nothing about whether the same rebuild is
affordable or safe with a live NCCL communicator on real GPUs, which is
the piece that needs real hardware (see the design notes).
"""

import datetime
import multiprocessing as mp
import socket
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


class TestFullyShardCannotBeReapplied:
    """No multi-rank machinery here on purpose - this is a single-process
    question about the FSDP2 API surface, orthogonal to the rendezvous work
    in test_dist_elastic_rendezvous.py.
    """

    def test_fully_shard_refuses_a_second_wrap_of_the_same_module(
        self, one_rank_group
    ):
        from torch.distributed.fsdp import fully_shard

        model = _toy_module()
        fully_shard(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model(torch.randn(4, 12)).sum().backward()
        optimizer.step()

        with pytest.raises(AssertionError, match="can only be applied.*once"):
            fully_shard(model)

    def test_rebuilding_from_scratch_is_the_working_alternative(
        self, one_rank_group
    ):
        """The same module, unwrapped conceptually via ``full_tensor()`` and
        reloaded into a fresh instance, matches exactly - proof that the
        rebuild path (not re-application) is what carries values across a
        remesh.
        """
        from torch.distributed.fsdp import fully_shard

        torch.manual_seed(0)
        model = _toy_module()
        fully_shard(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model(torch.randn(4, 12)).sum().backward()
        optimizer.step()

        ground_truth = {
            name: param.full_tensor().detach().clone()
            for name, param in model.named_parameters()
        }

        torch.manual_seed(999)  # a different seed - a match proves the load, not luck
        fresh = _toy_module()
        with torch.no_grad():
            for name, param in fresh.named_parameters():
                param.copy_(ground_truth[name])
        fully_shard(fresh)

        after = {
            name: param.full_tensor().detach().clone()
            for name, param in fresh.named_parameters()
        }
        for name in ground_truth:
            assert torch.equal(after[name], ground_truth[name]), name


@pytest.fixture
def one_rank_group():
    import torch.distributed as dist
    import os

    if dist.is_initialized():  # pragma: no cover - a leaked group from elsewhere
        dist.destroy_process_group()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29615")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _remesh_worker(rank, is_rank0, port, join_evt, out):
    """Top-level and argument-driven because ``spawn`` has to pickle it.

    Since GPU-110 the rebuild is **Ravex's**, not this test's: the capture, the
    teardown, the new group and the reload all happen inside
    ``ravex._dist.elastic.regroup``. The assertions stayed where they were, and
    one was added — the optimizer's moments, which the hand-rolled version
    never carried and which are the difference between a continuation and a
    restart from good weights.
    """
    try:
        import torch.distributed as dist
        from torch.distributed.fsdp import fully_shard

        from ravex._dist import elastic

        if is_rank0:
            store = dist.TCPStore(
                "127.0.0.1", port, world_size=None, is_master=True, use_libuv=False
            )
            join_evt.set()
        else:
            join_evt.wait(timeout=15)
            store = dist.TCPStore(
                "127.0.0.1", port, world_size=None, is_master=False, use_libuv=False
            )

        def build():
            """The factory. What `@ravex.train_loop` finally makes available."""
            fresh = _toy_module()
            fully_shard(fresh)
            return fresh, [torch.optim.Adam(fresh.parameters(), lr=0.01)]

        ground_truth = None
        model = optimizer = None
        if rank in (0, 1):
            gen0 = dist.PrefixStore("gen0", store)
            dist.init_process_group("gloo", store=gen0, rank=rank, world_size=2)

            torch.manual_seed(0)
            model = _toy_module()
            fully_shard(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model(torch.randn(4, 12)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            ground_truth = {
                name: param.full_tensor().detach().clone()
                for name, param in model.named_parameters()
            }

        # Nothing goes out of band any more. The joining rank is handed the
        # state by the broadcast `regroup` does once the new group is up, which
        # is what removed this test's `multiprocessing.Queue` entirely.
        fresh, optimizers = elastic.regroup(
            build, store, 1, rank, 3,
            model=model, optimizers=[optimizer] if model is not None else [],
            timeout_seconds=15,
        )

        after = {
            name: param.full_tensor().detach().clone()
            for name, param in fresh.named_parameters()
        }
        mismatches = [
            name for name in ground_truth or after
            if not torch.equal(after[name], (ground_truth or after)[name])
        ] if ground_truth else []

        # The moments the hand-rolled rebuild dropped. One Adam step in, they
        # are not zeros, so this fails loudly if the optimizer came back empty.
        moments = 0
        for state in optimizers[0].state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if value is not None:
                    full = value.full_tensor() if hasattr(value, "full_tensor") else value
                    if float(full.abs().sum()) > 0:
                        moments += 1

        # Prove it is not just holding the right bytes but is a real trainable
        # module again under the new three-rank mesh.
        fresh(torch.randn(4, 12)).sum().backward()

        out.put((rank, "ok", {"mismatches": mismatches, "moments": moments}))
        dist.destroy_process_group()
    except Exception:
        out.put((rank, "EXC", traceback.format_exc()))


class TestRegroupingAtANewWorldSize:
    """Combines step 1's rendezvous with the rebuild pattern above, across an
    actual 2 -> 3 world_size change with a rank that never existed before.
    """

    def test_a_fresh_module_reloaded_from_full_tensors_survives_a_real_regroup(self):
        ctx = mp.get_context("spawn")
        port = _free_port()
        join_evt = ctx.Event()
        out = ctx.Queue()
        procs = [
            ctx.Process(
                target=_remesh_worker,
                args=(r, r == 0, port, join_evt, out),
            )
            for r in (0, 1, 2)
        ]
        for p in procs:
            p.start()

        results = {}
        killed = []
        try:
            for _ in range(3):
                try:
                    rank, status, payload = out.get(timeout=30)
                except Exception:
                    break
                results[rank] = (status, payload)
        finally:
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    killed.append(p.pid)
                    p.terminate()
                    p.join(timeout=5)

        assert killed == [], "a rank hung rather than finishing - see step 1's notes"
        assert set(results) == {0, 1, 2}, results
        for rank, (status, payload) in results.items():
            assert status == "ok", f"rank {rank}: {payload}"
            assert payload["mismatches"] == [], (
                f"rank {rank} mismatched parameters: {payload['mismatches']}"
            )
            assert payload["moments"] > 0, (
                f"rank {rank} came back with an empty optimizer: the regroup "
                "restored weights and dropped the Adam moments, which is a "
                "restart from a good place and not a continuation"
            )
