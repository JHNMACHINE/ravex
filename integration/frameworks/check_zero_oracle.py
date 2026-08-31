"""Check Ravex's ZeRO reader against DeepSpeed's own — GPU-90.

``ravex._interop.zero.unshard`` is a reimplementation of somebody else's format, and a
reimplementation is worth having only when something independent can say it is
right. DeepSpeed writes ``zero_to_fp32.py`` into every checkpoint precisely so
the reconstruction can be done without it; comparing the two is therefore not
a test of Ravex against Ravex.

Compares element by element, not by summary statistic. A mean or a norm agrees
in exactly the cases where a reader is subtly wrong — an off-by-one in an
offset moves values between neighbouring parameters and leaves every aggregate
untouched.

    torchrun --nproc-per-node=2 make_zero.py --stage 1 --out /tmp/z1
    python check_zero_oracle.py /tmp/z1
"""

import argparse
import sys

import torch

from ravex._interop.foreign import confirm_stage, identify, summary
from ravex._interop.zero import unshard


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    worst = 0
    for root in args.roots:
        found = confirm_stage(identify(root), load)
        print("\n%s\n  identified: %s" % (root, summary(found)))
        if found.format != "deepspeed":
            print("  ** not a DeepSpeed checkpoint, skipped **")
            worst = max(worst, 1)
            continue

        mine = unshard(found, load)["model"]
        theirs = get_fp32_state_dict_from_zero_checkpoint(root)

        if set(mine) != set(theirs):
            only_mine = sorted(set(mine) - set(theirs))
            only_theirs = sorted(set(theirs) - set(mine))
            print("  ** different parameters **")
            print("     only ravex:     %s" % (only_mine or "-"))
            print("     only deepspeed: %s" % (only_theirs or "-"))
            worst = max(worst, 1)
            continue

        bad = []
        for name in sorted(theirs):
            a, b = mine[name], theirs[name]
            if tuple(a.shape) != tuple(b.shape):
                bad.append("%s: shape %s vs %s" % (name, tuple(a.shape), tuple(b.shape)))
            elif not torch.equal(a, b):
                gap = (a.float() - b.float()).abs().max().item()
                bad.append("%s: differs, largest gap %g" % (name, gap))

        if bad:
            print("  ** %d parameter(s) disagree **" % len(bad))
            for line in bad[:10]:
                print("     " + line)
            worst = max(worst, 1)
        else:
            total = sum(t.numel() for t in theirs.values())
            print("  bit-exact against zero_to_fp32: %d parameters, %d elements"
                  % (len(theirs), total))

    return worst


if __name__ == "__main__":
    sys.exit(main())
