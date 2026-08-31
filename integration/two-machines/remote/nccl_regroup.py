"""GPU-94 on real GPUs: can a live NCCL communicator be torn down and rebuilt
at a different world size, in-process, with a model already in VRAM?

This is the one question the CPU/gloo suite in ``tests/test_elastic_*.py``
cannot answer, and the design notes for GPU-94 say so in as many words: gloo
never touches a CUDA communicator, so a green local suite says nothing about
whether ``destroy_process_group()`` + ``init_process_group()`` is safe for
NCCL while a model occupies the device.

Launched by ``torchrun`` with ``WORLD`` ranks. Two knobs decide the shape:

    START   ranks taking part in generation 0 (ranks >= START are "new" and
            sit out phase 1 entirely, exactly like a candidate that has just
            been provisioned)
    TARGET  ranks taking part in generation 1

So ``START=1 TARGET=2`` is a grow, ``START=2 TARGET=1`` is a shrink, and
``START=2 TARGET=4`` across two boxes is a grow that crosses the network.

What it checks, in order:

1. Phase 1 trains a real FSDP2 model on NCCL and records every parameter's
   full (unsharded) values.
2. The group is destroyed and rebuilt at ``TARGET`` on a *fresh*
   ``PrefixStore`` generation over one persistent ``TCPStore``. The store is
   held by rank 0 for the whole run and is deliberately not the one torchrun
   set up - torchrun's own rendezvous is scoped to one
   ``init_process_group`` call.
3. Each rank rebuilds a fresh module, loads the recorded values, and calls
   ``fully_shard()`` again under the new mesh - the only route available,
   since ``fully_shard()`` refuses a second application to the same module
   and no public API re-meshes a materialised DTensor.
4. The rebuilt model is verified bit-exact against what phase 1 held, and
   then takes a real training step under the new topology.
5. VRAM is sampled around the regroup, because a communicator that is
   destroyed but does not release its device memory would still "work"
   while quietly making every subsequent regroup more expensive.

Two rules carried over from the local suite, both found the hard way there
and both fatal here if broken:

* ``full_tensor()`` is a collective - every rank calls it, including the
  ranks that throw the result away. Calling it on one rank alone
  desynchronises torch's internal group-naming counter and surfaces much
  later as an unrelated timeout.
* Never read a DTensor whose process group has already been destroyed. The
  values needed after a regroup must be captured *before* it.
"""

import datetime
import json
import os
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard


def log(msg):
    print("[rank %s] %s" % (os.environ.get("RANK", "?"), msg), flush=True)


def vram():
    """(allocated, reserved) MiB on this rank's device."""
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    return (
        torch.cuda.memory_allocated() / 2**20,
        torch.cuda.memory_reserved() / 2**20,
    )


def build_model(hidden, layers, device):
    torch.manual_seed(0)
    blocks = []
    for _ in range(layers):
        blocks.append(nn.Linear(hidden, hidden))
        blocks.append(nn.ReLU())
    blocks.append(nn.Linear(hidden, 8))
    return nn.Sequential(*blocks).to(device)


def train_step(model, optimizer, hidden, device):
    model(torch.randn(8, hidden, device=device)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    start = int(os.environ.get("START", world))
    target = int(os.environ.get("TARGET", world))
    hidden = int(os.environ.get("HIDDEN", "1024"))
    layers = int(os.environ.get("LAYERS", "4"))

    master_addr = os.environ["MASTER_ADDR"]
    # Deliberately not torchrun's own port: that store belongs to torchrun's
    # rendezvous. This one has to outlive both generations.
    store_port = int(os.environ.get("STORE_PORT", str(int(os.environ["MASTER_PORT"]) + 7)))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    result = {
        "rank": rank,
        "world": world,
        "start": start,
        "target": target,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(local_rank),
    }

    # ── the persistent rendezvous point ────────────────────────────────
    is_store_master = rank == 0
    if not is_store_master:
        time.sleep(2)  # let rank 0 bind first; no group exists to barrier on
    base_store = dist.TCPStore(
        master_addr, store_port, world_size=None, is_master=is_store_master
    )
    log("store ready on %s:%d (master=%s)" % (master_addr, store_port, is_store_master))

    model = optimizer = None
    recorded = None

    # ── generation 0 ───────────────────────────────────────────────────
    if rank < start:
        gen0 = dist.PrefixStore("gpu94/gen0", base_store)
        t0 = time.time()
        dist.init_process_group(
            "nccl", store=gen0, rank=rank, world_size=start,
            timeout=datetime.timedelta(seconds=120),
        )
        result["gen0_init_seconds"] = round(time.time() - t0, 3)
        log("gen0 up at world_size=%d in %.2fs" % (start, result["gen0_init_seconds"]))

        model = build_model(hidden, layers, device)
        fully_shard(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_step(model, optimizer, hidden, device)
        train_step(model, optimizer, hidden, device)

        result["vram_after_gen0"] = vram()

        # Collective: every rank of this group, even those that will drop out.
        with torch.no_grad():
            recorded = {
                name: p.full_tensor().detach().clone().cpu()
                for name, p in model.named_parameters()
            }
        log("recorded %d tensors from gen0" % len(recorded))

        # Everything that needs the old mesh is done. From here the old
        # DTensors must not be read again - see the module docstring.
        t0 = time.time()
        dist.destroy_process_group()
        result["gen0_destroy_seconds"] = round(time.time() - t0, 3)
        del model, optimizer
        model = optimizer = None
        torch.cuda.empty_cache()
        result["vram_after_destroy"] = vram()
        log("gen0 destroyed in %.2fs, vram now %s" % (
            result["gen0_destroy_seconds"], result["vram_after_destroy"]))

    # Ranks that sat out generation 0 need the values from somewhere, and
    # that is `ravex._elastic.prestage_send`/`prestage_receive` over a plain
    # socket - the real production path from GPU-94 step 4, exercised here on
    # real hardware and (in the cross-box shape) over the real network.
    #
    # Not the rendezvous store: a TCPStore payload is capped at 8 MiB and a
    # real checkpoint is orders of magnitude past that. Tried first and it
    # failed exactly there ("Invalid payload size. size: 16829701, max:
    # 8388608"), which is a decent argument on its own for why the transport
    # is a socket rather than the store everything else coordinates on.
    from ravex._elastic import prestage_receive, prestage_send

    joiners = max(0, target - start)
    handoff_port = int(os.environ.get("HANDOFF_PORT", str(store_port + 1)))

    if joiners:
        if rank == 0:
            source_dir = os.path.join(
                os.environ.get("KIT_ROOT", "/root"), "gpu94_handoff_src"
            )
            os.makedirs(source_dir, exist_ok=True)
            # One file per snapshot, never overwritten: the incremental skip
            # matches on name and size alone (GPU-97), so reusing a path for
            # different values is how a stale copy passes for a fresh one.
            torch.save(recorded, os.path.join(source_dir, "gen0_values.pt"))

            payload_bytes = sum(
                os.path.getsize(os.path.join(source_dir, f))
                for f in os.listdir(source_dir)
            )
            result["payload_bytes"] = payload_bytes

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", handoff_port))
            server.listen(joiners)
            base_store.set("gpu94/handoff_ready", b"1")
            log("serving the handoff to %d joiner(s) on :%d (%.1f MiB)"
                % (joiners, handoff_port, payload_bytes / 2**20))
            served = []
            for _ in range(joiners):
                conn, peer = server.accept()
                t0 = time.time()
                prestage_send(conn, source_dir)
                seconds = time.time() - t0
                served.append({
                    "peer": peer[0],
                    "seconds": round(seconds, 3),
                    "mib_per_s": round((payload_bytes / 2**20) / seconds, 2) if seconds else None,
                })
                log("prestaged to %s in %.2fs (%.1f MiB/s)" % (
                    peer[0], seconds, (payload_bytes / 2**20) / seconds if seconds else 0))
                conn.close()
            server.close()
            result["served"] = served

        # Only the ranks that sat out generation 0. Getting this wrong is not
        # abstract: an earlier version said `elif rank < target`, which also
        # caught the gen0 ranks other than rank 0 - they queued up for a
        # handoff they did not need, ate the joiners' slots, and the real
        # joiner that never got served died on a connection reset while the
        # log cheerfully reported two successful transfers.
        elif start <= rank < target:
            base_store.get("gpu94/handoff_ready")  # blocks until rank 0 listens
            dest_dir = os.path.join(
                os.environ.get("KIT_ROOT", "/root"), "gpu94_handoff_dst_%d" % rank
            )
            os.makedirs(dest_dir, exist_ok=True)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((master_addr, handoff_port))
            t0 = time.time()
            ok = prestage_receive(sock, dest_dir)
            result["prestage_seconds"] = round(time.time() - t0, 3)
            result["prestage_complete"] = ok
            sock.close()
            recorded = torch.load(
                os.path.join(dest_dir, "gen0_values.pt"), weights_only=True
            )
            log("received %d tensors in %.2fs (complete=%s)" % (
                len(recorded), result["prestage_seconds"], ok))

    if rank >= target:
        log("not part of the new topology - leaving")
        result["left"] = True
        print("RESULT " + json.dumps(result), flush=True)
        return

    # ── generation 1: the regroup under test ───────────────────────────
    gen1 = dist.PrefixStore("gpu94/gen1", base_store)
    t0 = time.time()
    dist.init_process_group(
        "nccl", store=gen1, rank=rank, world_size=target,
        timeout=datetime.timedelta(seconds=120),
    )
    result["gen1_init_seconds"] = round(time.time() - t0, 3)
    log("gen1 up at world_size=%d in %.2fs" % (target, result["gen1_init_seconds"]))

    # A real collective on the new communicator, before anything else, so a
    # broken regroup fails here rather than somewhere subtler.
    probe = torch.tensor([1.0], device=device)
    dist.all_reduce(probe)
    result["probe_all_reduce"] = probe.item()
    result["probe_expected"] = float(target)

    fresh = build_model(hidden, layers, device)
    with torch.no_grad():
        for name, param in fresh.named_parameters():
            param.copy_(recorded[name].to(device))
    fully_shard(fresh)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.01)

    with torch.no_grad():
        after = {
            name: p.full_tensor().detach().clone().cpu()
            for name, p in fresh.named_parameters()
        }

    mismatched = [n for n in recorded if not torch.equal(after[n], recorded[n])]
    result["mismatched"] = mismatched
    result["tensors_checked"] = len(recorded)

    # It has to be a trainable model again, not just the right bytes.
    train_step(fresh, fresh_optimizer, hidden, device)
    result["trained_after_regroup"] = True
    result["vram_after_gen1"] = vram()

    dist.destroy_process_group()
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print("RESULT " + json.dumps({
            "rank": os.environ.get("RANK"),
            "error": "%s: %s" % (type(exc).__name__, exc),
        }), flush=True)
        sys.exit(1)
