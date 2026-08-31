"""What Ravex's registry actually sees inside a DeepSpeed run — GPU-90 follow-up.

``ravex/_frameworks.py`` states that intercepting PyTorch is enough, so
"anything built on top of PyTorch — HuggingFace ``Trainer``, Lightning,
Accelerate — is already covered". DeepSpeed is the case that tests the claim
rather than illustrating it, and this asks the registry directly instead of
reading the checkpoint and inferring backwards.

Run it under torchrun, which is what DeepSpeed's ``init_distributed`` needs
before it will fall back from MPI::

    torchrun --nproc-per-node=1 probe_deepspeed_registry.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    import argparse

    import deepspeed

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="gloo")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.01}},
            "zero_optimization": {"stage": args.stage},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
        },
    )
    engine.module.train()

    loader = DataLoader(TensorDataset(torch.randn(16, 64)), batch_size=2)
    for (batch,) in loader:
        engine.backward(engine(batch).sum())
        engine.step()

    from ravex._runtime import get_runtime

    registry = get_runtime().registry
    module_params = {id(p) for p in model.parameters()}
    owned = {
        id(p)
        for optimizer in registry.optimizers
        for group in optimizer.param_groups
        for p in group.get("params", [])
    }

    print("the engine is itself an nn.Module: %s" % isinstance(engine, nn.Module))
    print("engine.module is the user's model: %s" % (engine.module is model))
    print("root models the registry found: %s"
          % [type(m).__name__ for m in registry.models])
    print("optimizers registered: %s"
          % [type(o).__name__ for o in registry.optimizers])
    print("module parameters: %d" % len(module_params))
    print("parameters any optimizer owns: %d" % len(owned))
    print("of the module's, owned: %d" % len(owned & module_params))
    print("root parameter sizes: %s"
          % [p.numel() for m in registry.models for p in m.parameters()][:6])
    print("parameters_are_partitioned_away: %s"
          % registry.parameters_are_partitioned_away())


if __name__ == "__main__":
    main()
