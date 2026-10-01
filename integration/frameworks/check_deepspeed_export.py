"""Does DeepSpeed resume what ``ravex export --to deepspeed`` writes? — GPU-145.

The writer is checked against the thing that reads its output, in a closed
loop, rather than against this module's reading of DeepSpeed's format.

**The round trip.**

1. a model trains a few Adam steps in plain torch, and its state is put in the
   shape a Ravex checkpoint holds;
2. :func:`ravex._interop.export.to_deepspeed` writes it as a universal
   checkpoint;
3. DeepSpeed loads it with ``load_universal`` at this world size and ZeRO
   stage, and saves it again as an ordinary ZeRO checkpoint;
4. Ravex's ZeRO reader - itself checked bit for bit against DeepSpeed's own
   ``zero_to_fp32`` (``check_zero_oracle.py``) - rebuilds every weight and
   moment by name from that, and they must equal what went in.

That is what shapes cannot check: a moment on the wrong parameter has the right
shape and plausible values. Equality by name is the claim.

**The next step, against DeepSpeed's own converter.** A resumed checkpoint is
worth something only if training carries on as it would have; the step count
and the bias correction have to come across, not only the tensors. Comparing
DeepSpeed's next step against torch's would not show it: measured here, the
two Adams part by 0.0057 on the *first* step from the same weights, with no
checkpoint anywhere. So the next step is compared against DeepSpeed itself:

5. a DeepSpeed run trains three steps and saves; DeepSpeed's own
   ``ds_to_universal`` makes a universal checkpoint of it, and an engine
   resumed from that takes a step;
6. the same checkpoint, read by Ravex and written by ``to_deepspeed``, resumes
   another engine that takes the same step;
7. the two must hold identical parameters (``torch.equal``).

    torchrun --nproc-per-node=2 check_deepspeed_export.py --stage 2 --out /tmp/x

``--positional`` builds the optimizer from ``model.parameters()``, whose state
carries no names: the export must then leave the moments out, and DeepSpeed
must still resume the weights.
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

HIDDEN = 97  # divides by neither 2 nor 3, so every partition is uneven
LR = 0.01


def build():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 5))


def batches(count):
    generator = torch.Generator().manual_seed(1)
    return [torch.randn(4, HIDDEN, generator=generator) for _ in range(count)]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--out", required=True)
    parser.add_argument("--positional", action="store_true")
    args = parser.parse_args()

    import deepspeed

    from ravex._interop.export import to_deepspeed
    from ravex._interop.foreign import confirm_stage, identify
    from ravex._interop.zero import unshard

    deepspeed.init_distributed(dist_backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    data = batches(4)
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        # Adam, not AdamW: torch.optim.Adam below has no decoupled decay.
        # DeepSpeed's own Adam, not ``torch_adam``: the universal loader puts
        # the step back as an int, as DeepSpeed's own converter writes it, and
        # torch's Adam wants a tensor there - true of any universal checkpoint.
        "optimizer": {"type": "Adam", "params": {"lr": LR, "adam_w_mode": False}},
        "zero_optimization": {"stage": args.stage},
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
    }
    universal = dict(config, checkpoint={"load_universal": True})

    def engine_from(directory):
        model = build()
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=universal)
        engine.load_checkpoint(directory, load_optimizer_states=not args.positional, load_lr_scheduler_states=False)
        return engine

    def read_back(directory, tag):
        return unshard(confirm_stage(identify(os.path.join(directory, tag)), load), load)

    failures = []

    # 1-4: torch -> ravex state -> universal -> DeepSpeed -> ZeRO -> ravex reader.
    reference = build()
    params = reference.parameters() if args.positional else reference.named_parameters()
    optimizer = torch.optim.Adam(params, lr=LR)
    for x in data[:3]:
        optimizer.zero_grad()
        reference(x).sum().backward()
        optimizer.step()
    exported = os.path.join(args.out, "exported")
    if rank == 0:
        state = {
            "step": 3,
            "models": {"model": reference.state_dict()},
            "optimizers": {"optimizer": optimizer.state_dict()},
        }
        for note in to_deepspeed(state, exported):
            print("export:", note, flush=True)
    dist.barrier()
    engine = engine_from(exported)
    back = os.path.join(args.out, "back")
    engine.save_checkpoint(back, tag="back")
    dist.barrier()
    if rank == 0:
        got = read_back(back, "back")
        names = dict(reference.named_parameters())
        for name, tensor in names.items():
            if not torch.equal(got["model"][name], tensor.detach()):
                failures.append("weight %s differs by %g" % (name, (got["model"][name] - tensor).abs().max()))
        if not args.positional:
            by_position = optimizer.state_dict()["state"]
            for index, name in enumerate(names):
                for buffer in ("exp_avg", "exp_avg_sq"):
                    have = got["optimizer"]["state"].get(name, {}).get(buffer)
                    if have is None or not torch.equal(have, by_position[index][buffer]):
                        failures.append("%s of %s did not come back" % (buffer, name))

    # 5-7: the next step, against DeepSpeed's own converter.
    if not args.positional:
        native = build()
        trained, _, _, _ = deepspeed.initialize(model=native, model_parameters=native.parameters(), config=config)
        for x in data[:3]:
            trained.backward(trained(x).sum())
            trained.step()
        theirs_dir, ours_dir = os.path.join(args.out, "native"), os.path.join(args.out, "ours")
        trained.save_checkpoint(theirs_dir, tag="step3")
        dist.barrier()
        if rank == 0:
            subprocess.run(
                [sys.executable, "-m", "deepspeed.checkpoint.ds_to_universal",
                 "--input_folder", os.path.join(theirs_dir, "step3"),
                 "--output_folder", os.path.join(theirs_dir, "step3_universal"),
                 "--inject_missing_state"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(os.path.join(theirs_dir, "latest_universal"), "w") as handle:
                handle.write("step3_universal")
            # The same checkpoint as Ravex sees it: weights and moments by name.
            got = read_back(theirs_dir, "step3")
            groups = got["optimizer"]["param_groups"] or [{}]
            state = {
                "step": got["step"],
                "sharded": {"model": {
                    "layout": "gather",
                    "model": got["model"],
                    "optimizer": {"state": got["optimizer"]["state"], "param_groups": groups},
                }},
            }
            to_deepspeed(state, ours_dir)
        dist.barrier()
        results = []
        for directory in (theirs_dir, ours_dir):
            resumed = engine_from(directory)
            resumed.backward(resumed(data[3]).sum())
            resumed.step()
            results.append({n: p.detach().clone() for n, p in resumed.module.named_parameters()})
        for name in results[0]:
            if not torch.equal(results[0][name], results[1][name]):
                failures.append(
                    "next step from ravex's export differs from DeepSpeed's own converter on %s by %g"
                    % (name, (results[0][name] - results[1][name]).abs().max())
                )

    report = [None] * world
    dist.all_gather_object(report, failures)
    if rank == 0:
        merged = sorted(set(f for part in report for f in part))
        label = "stage %d, %d rank(s)%s" % (args.stage, world, ", positional optimizer" if args.positional else "")
        if merged:
            print("FAIL %s:" % label)
            for line in merged[:12]:
                print("   ", line)
        elif args.positional:
            print("ok   %s: weights equal by name after the round trip, moments left out as said" % label)
        else:
            print("ok   %s: weights and moments equal by name, and the next step equals DeepSpeed's own converter's" % label)
    dist.barrier()
    return 1 if any(report) else 0


if __name__ == "__main__":
    sys.exit(main())
