"""Training scripts that need a real GPU, in one file. Still no mention of Ravex.

Three modes, selected with --mode:

``amp``
    Single GPU, ``autocast(float16)`` plus a live ``GradScaler``. This is the
    only way to exercise the loss scale for real: on CPU the scaler is inert,
    so gradients never overflow and the scale never moves.

``fsdp1``
    ``FullyShardedDataParallel`` under torchrun. FSDP1 refuses to initialise
    without an accelerator, so this path cannot run anywhere else.

``ddp``
    DDP over NCCL rather than gloo.

Dropout runs on the GPU in every mode, so its masks come from the CUDA RNG —
which means an exact loss match after a restart also proves the CUDA generator
state was restored, not just the weights.
"""

import argparse
import json
import os
import signal

import ravex
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

SAMPLES = 64
BATCH = 8
SEED = 0


def build_model():
    return nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Dropout(0.2), nn.Linear(12, 1))


@ravex.train_loop()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument(
        "--mode", choices=("amp", "bf16", "ddp", "fsdp1"), required=True
    )
    parser.add_argument("--die-at", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this script needs CUDA")

    # A plain MLP is deterministic under these settings, so the comparison
    # against an uninterrupted run can be exact rather than approximate.
    # CUBLAS_WORKSPACE_CONFIG has to be in the environment before CUDA starts,
    # so the caller sets it.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    distributed = args.mode in ("ddp", "fsdp1")
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
    else:
        local_rank, rank = 0, 0
        torch.cuda.set_device(0)

    device = torch.device("cuda", local_rank)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = build_model().to(device)

    if args.mode == "bf16":
        # Parameters themselves in bfloat16, not just autocast around fp32
        # weights: that is what puts bf16 tensors into the checkpoint, and
        # Moonclip has a dedicated path for them because numpy cannot
        # represent the dtype at all.
        model = model.to(torch.bfloat16)

    if args.mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[local_rank])
    elif args.mode == "fsdp1":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = FSDP(model, device_id=local_rank, use_orig_params=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mode == "amp"))

    features = torch.randn(SAMPLES, 6)
    targets = features.sum(dim=1, keepdim=True)
    dataset = TensorDataset(features, targets)
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=SEED)
        loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model.train()
    trace = open(args.trace, "a", buffering=1) if rank == 0 else None

    optimizer_steps = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if args.mode == "bf16":
                x, y = x.to(torch.bfloat16), y.to(torch.bfloat16)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(args.mode == "amp")):
                loss = ((model(x) - y) ** 2).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # An overflowing gradient makes scaler.step() skip the optimizer
            # and back the scale off. The iteration happened; the step did not,
            # and nothing about the model changed. The scale only ever drops on
            # a skip, so comparing it across update() tells the two apart.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
            scheduler.step()

            if stepped:
                optimizer_steps += 1

            if trace is not None:
                trace.write(
                    json.dumps(
                        {
                            "step": optimizer_steps,
                            "stepped": stepped,
                            "loss": repr(loss.item()),
                            "lr": repr(optimizer.param_groups[0]["lr"]),
                            "scale": repr(scaler.get_scale()),
                        }
                    )
                    + "\n"
                )

            if args.die_at and optimizer_steps >= args.die_at:
                if trace is not None:
                    trace.flush()
                    os.fsync(trace.fileno())
                os.kill(os.getpid(), signal.SIGKILL)

    if trace is not None:
        trace.close()
    if distributed:
        dist.destroy_process_group()
    print(f"rank {rank} done after {optimizer_steps} optimizer steps")


if __name__ == "__main__":
    main()
