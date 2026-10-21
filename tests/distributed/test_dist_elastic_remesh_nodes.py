"""GPU-94 step 2: can an already fully_shard()-ed module be re-meshed onto a - the tests that run several nodes or ranks (GPU-256).

They start processes, or play nodes that talk over real sockets with
deadlines; see tests/distributed/conftest.py. The rest of this subject,
and the helpers both halves use, are in tests/unit/test_dist_elastic_remesh.py.
"""

import multiprocessing as mp
import traceback

import torch
from test_dist_elastic_remesh import _store_server, _toy_module  # noqa: E402


def _store_client(port):
    """The store `_store_server` opened, from inside a rank."""
    from ravex._dist.rendezvous import _tcp_store

    return _tcp_store("127.0.0.1", port, False, 120.0)


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

        # The test holds the server (`_store_server`); rank 0 still says when
        # it is connected, so the others join after it as before.
        if is_rank0:
            store = _store_client(port)
            join_evt.set()
        else:
            join_evt.wait(timeout=15)
            store = _store_client(port)

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
        server = _store_server()
        port = server.port
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
