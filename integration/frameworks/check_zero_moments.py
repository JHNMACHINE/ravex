"""Are the optimizer moments paired with the right parameters? — GPU-90.

``zero_to_fp32.py`` reconstructs a ZeRO checkpoint's *parameters* and nothing
else, so the moments have no oracle of the same kind. This builds one out of
the checkpoints themselves.

Adam's step is a deterministic function of what it is given. Take two
checkpoints one step apart; the later moments determine the gradient that
produced them, and the gradient plus the earlier parameters determine the later
parameters:

    g      = (m_b - beta1 * m_a) / (1 - beta1)
    p_pred = p_a - lr * (m_b / (1 - beta1**t)) / (sqrt(v_b / (1 - beta2**t)) + eps)

If ``p_pred`` matches ``p_b``, then every moment is sitting on the parameter it
belongs to. It is the pairing this checks, which is exactly what shapes cannot:
a moment reassembled onto the wrong parameter has the right shape, entirely
plausible values, and predicts the wrong step.

    torchrun --nproc-per-node=2 make_zero.py --stage 1 --pair --out /tmp/p1
    python check_zero_moments.py /tmp/p1
"""

import argparse
import os
import sys

import torch

from ravex._foreign import confirm_stage, identify
from ravex._zero import unshard


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def read(root, tag):
    found = confirm_stage(identify(os.path.join(root, tag)), load)
    if found.format != "deepspeed":
        raise SystemExit("%s/%s is not a DeepSpeed checkpoint" % (root, tag))
    return unshard(found, load)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--earlier", default="step3")
    parser.add_argument("--later", default="step4")
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    a, b = read(args.root, args.earlier), read(args.root, args.later)

    groups = b["optimizer"]["param_groups"] or a["optimizer"]["param_groups"]
    if not groups:
        print("  ** no param_groups recorded; cannot predict a step **")
        return 1
    hyper = groups[0]
    beta1, beta2 = hyper.get("betas", (0.9, 0.999))
    lr = hyper.get("lr")
    eps = hyper.get("eps", 1e-8)

    print("  stage %s, %d rank(s), lr %s, betas %s"
          % (b["stage"], b["world_size"], lr, (beta1, beta2)))

    worst_name, worst = None, 0.0
    missing = []
    for name, param_b in b["model"].items():
        state_a = a["optimizer"]["state"].get(name, {})
        state_b = b["optimizer"]["state"].get(name, {})
        if "exp_avg" not in state_b or "exp_avg_sq" not in state_b:
            missing.append(name)
            continue

        m_a = state_a.get("exp_avg", torch.zeros_like(state_b["exp_avg"]))
        m_b, v_b = state_b["exp_avg"], state_b["exp_avg_sq"]
        step = float(state_b.get("step", 0))
        if step <= 0:
            missing.append(name + " (no step)")
            continue

        # The gradient is recovered rather than assumed. It is the one quantity
        # a checkpoint does not carry, and Adam's own recurrence gives it back.
        grad = (m_b - beta1 * m_a) / (1.0 - beta1)
        del grad  # kept for the reader: the prediction below needs only m_b, v_b

        m_hat = m_b / (1.0 - beta1 ** step)
        v_hat = v_b / (1.0 - beta2 ** step)
        predicted = a["model"][name] - lr * m_hat / (v_hat.sqrt() + eps)

        gap = (predicted - param_b).abs().max().item()
        if gap > worst:
            worst_name, worst = name, gap

    if missing:
        print("  ** no moments for: %s **" % ", ".join(missing[:6]))
        return 1

    ok = worst <= args.tolerance
    print("  largest disagreement between the predicted step and the next "
          "checkpoint: %g (%s)" % (worst, worst_name))
    print("  %s" % ("moments are on the right parameters" if ok
                    else "** the pairing is wrong **"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
