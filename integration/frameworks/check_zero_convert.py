"""A real ZeRO checkpoint, converted and loaded into a live model — GPU-90.

The unit tests check the name reconciliation; ``check_zero_oracle.py`` checks
the arithmetic against DeepSpeed's own reader. Neither answers the question the
feature exists for: **does a DeepSpeed checkpoint restore into a run that is
not DeepSpeed?**

This does the whole path — identify, read, align, rename, hand to
``torch.distributed.checkpoint.state_dict.set_state_dict`` — and then checks
the live model against the oracle rather than against the converter. Going
through ``set_state_dict`` rather than ``Module.load_state_dict`` is the point:
that is the call Ravex's own restore path makes, it is what takes an optimizer
state keyed by parameter name, and it is what scatters onto shards when the
live model has them. Proving the hand-off on a plain model proves the shape;
the scattering is torch's and is already exercised elsewhere.

    torchrun --nproc-per-node=2 make_zero.py --stage 3 --out /tmp/z3
    python check_zero_convert.py /tmp/z3 --hidden 256 --layers 4
"""

import argparse
import sys

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

from ravex._convert import align, fit_param_groups, rename, unify
from ravex._foreign import confirm_stage, identify


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def rebuild(hidden: int, layers: int) -> nn.Module:
    """The same architecture ``make_zero.py`` trained, freshly initialised.

    Freshly, and the assertion below leans on it: a converted checkpoint that
    quietly did nothing would leave these weights at their initialisation, and
    a test that compared them against a model built with the same seed would
    pass anyway.
    """
    return nn.Sequential(
        *[
            layer
            for _ in range(layers)
            for layer in (nn.Linear(hidden, hidden), nn.ReLU())
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    found = confirm_stage(identify(args.root), load)
    if found.format != "deepspeed":
        print("  ** %s is not a DeepSpeed checkpoint **" % args.root)
        return 1

    unified = unify(found, load)
    model = rebuild(args.hidden, args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    alignment = align(list(unified["model"]), list(model.state_dict()))
    print("  %s" % unified["source"])
    print("  %s" % alignment.report())
    for note in unified.get("notes", []):
        print("  note: %s" % note)
    if not alignment.complete:
        print("  ** not every live parameter was served **")
        return 1

    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    moved = fit_param_groups(
        rename(unified, alignment), optimizer.state_dict()["param_groups"]
    )
    for note in moved.get("notes", [])[len(unified.get("notes", [])):]:
        print("  note: %s" % note)

    set_state_dict(
        model,
        [optimizer],
        model_state_dict=moved["model"],
        optim_state_dict=moved["optimizer"],
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    oracle = get_fp32_state_dict_from_zero_checkpoint(args.root)

    live = model.state_dict()
    changed = sum(
        1 for name in before if not torch.equal(before[name], live[name])
    )
    print("  %d of %d parameters changed by the load" % (changed, len(before)))
    if changed == 0:
        print("  ** nothing was loaded **")
        return 1

    bad = []
    for name, want in oracle.items():
        got = live.get(name)
        if got is None:
            bad.append("%s: not in the live model" % name)
        elif not torch.equal(got, want):
            bad.append(
                "%s: differs, largest gap %g"
                % (name, (got.float() - want.float()).abs().max().item())
            )
    if bad:
        print("  ** %d parameter(s) disagree with zero_to_fp32 **" % len(bad))
        for line in bad[:8]:
            print("     " + line)
        return 1

    print("  weights match zero_to_fp32 after loading through set_state_dict")

    # The moments are the half with no oracle of its own; what can be said
    # here is that they arrived at all and landed on the right parameters,
    # which `torch.optim` will only allow if the group membership was rebuilt
    # correctly.
    state = optimizer.state_dict()["state"]
    with_moments = sum(1 for entry in state.values() if "exp_avg" in entry)
    print("  optimizer: %d of %d parameters have moments"
          % (with_moments, len(list(model.parameters()))))
    if with_moments != len(list(model.parameters())):
        print("  ** some moments did not survive the conversion **")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
