"""What is actually inside a DeepSpeed ZeRO checkpoint — groundwork for GPU-90.

``make_zero.py`` produces the files; this reads them. The distinction matters:
an adapter is written against the *contents*, and a listing of file names only
tells you how many there are. What decides the design is which keys carry the
parameters, whether they are flat or per-tensor, and what has to be read from
more than one file to be understood at all.

Deliberately a describer, not a converter. It prints a shape and never asserts
one, so it can be pointed at a checkpoint from a version or a stage this
project has not seen and still be informative — which is the whole reason it
exists rather than the knowledge living in someone's head.
"""

import argparse
import json
import os

import torch


def summarise(value, depth: int = 0, max_depth: int = 4):
    """One value as a printable shape, recursing into containers but not far."""
    if isinstance(value, torch.Tensor):
        return "tensor%s %s" % (tuple(value.shape), value.dtype)
    if isinstance(value, dict):
        if depth >= max_depth:
            return "{...%d keys}" % len(value)
        return {str(k): summarise(v, depth + 1, max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if depth >= max_depth:
            return "[...%d]" % len(value)
        if len(value) > 6:
            head = [summarise(v, depth + 1, max_depth) for v in value[:3]]
            return head + ["... %d more" % (len(value) - 3)]
        return [summarise(v, depth + 1, max_depth) for v in value]
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="do not elide long dicts")
    args = parser.parse_args()

    for base, _, names in os.walk(args.root):
        for name in sorted(names):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(base, name)
            print("\n" + "=" * 70)
            print(os.path.relpath(path, args.root).replace(os.sep, "/"))
            print("=" * 70)
            # `weights_only=False` on purpose: a ZeRO checkpoint carries the
            # engine's own bookkeeping objects beside the tensors, and refusing
            # to unpickle them would hide exactly the part an adapter has to
            # understand. These files are produced by `make_zero.py` two steps
            # earlier in the same pipeline; nothing untrusted reaches here.
            blob = torch.load(path, map_location="cpu", weights_only=False)
            summary = summarise(blob, max_depth=args.depth if not args.full else 9)
            print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
