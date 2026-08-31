"""What a ZeRO-3 run leaves reachable on the module and the optimizer.

Two questions, and both have to be answered before deciding what Ravex should
do about stage 3:

1. **Are the model's parameters still there?** ZeRO-3 partitions parameters as
   well as optimizer state, so a checkpoint taken from the live module may be
   holding empty shells.
2. **Can the base optimizer's ``state_dict()`` even be called?** Ravex calls it
   on every optimizer it has seen, and at stage 3 that raised a bare ``KeyError``
   on a parameter id — which fails the whole checkpoint, every interval, with a
   traceback that says nothing about ZeRO.

Run under torchrun: DeepSpeed's ``init_distributed`` falls back to MPI without
the launcher's environment variables.
"""

import torch
import torch.nn as nn


def main() -> None:
    import deepspeed

    deepspeed.init_distributed(dist_backend="gloo")

    for stage in (1, 2, 3):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config={
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 1,
                "optimizer": {"type": "Adam", "params": {"lr": 0.01}},
                "zero_optimization": {"stage": stage},
                "fp16": {"enabled": False},
                "bf16": {"enabled": False},
            },
        )

        # After real steps, not straight after `initialize`: ZeRO-3 gathers
        # parameters during the forward, so what `.numel()` reports depends on
        # when it is asked. A checkpoint is taken at a step boundary, so that
        # is where this has to look.
        import torch.distributed as dist
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(torch.randn(8, 8)), batch_size=2)
        for (batch,) in loader:
            engine.backward(engine(batch).sum())
            engine.step()

        sizes = [(n, p.numel()) for n, p in engine.module.named_parameters()]
        true_sizes = [("0.weight", 64), ("0.bias", 8), ("1.weight", 32), ("1.bias", 4)]
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print("   true sizes:      %s" % true_sizes)
        print("   rank %d observes: %s" % (rank, sizes))
        empty = sum(1 for _, n in sizes if n == 0)
        print("stage %d: %d parameter(s), %d empty  %s"
              % (stage, len(sizes), empty, sizes[:2]))

        base = engine.optimizer.optimizer if hasattr(engine.optimizer, "optimizer") \
            else engine.optimizer
        try:
            state = base.state_dict()
            print("         base optimizer state_dict(): %d state entries"
                  % len(state.get("state", {})))
        except Exception as exc:
            print("         base optimizer state_dict(): %s: %s"
                  % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
