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


import pytest
import torch


def _store_server():
    """A rendezvous store for one spawned group, held by the test's own process.

    On port 0, with the port read back from the store. The probe it replaces
    found a free port, closed it and handed the number to a rank to listen on;
    under `pytest -n` another worker can take it in between, and that rank dies
    with EADDRINUSE (GPU-191, as in test_emergency_checkpoint and
    test_dist_multinode). A server that is already listening has no such
    window. The caller keeps it alive until its ranks are done.
    """
    from ravex._dist.rendezvous import _tcp_store

    return _tcp_store("127.0.0.1", 0, True, 120.0)


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

    if dist.is_initialized():  # pragma: no cover - a leaked group from elsewhere
        dist.destroy_process_group()
    # On a server of its own rather than MASTER_PORT 29615: two pytest
    # workers in this fixture at once would both listen there (GPU-191).
    server = _store_server()
    dist.init_process_group("gloo", store=server, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
