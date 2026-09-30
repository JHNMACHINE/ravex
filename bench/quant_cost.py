"""What quantizing an outer delta costs, per path (GPU-139).

The issue's last number: the kernel's seconds per round against the network it
saves. Three paths over the same tensor — the torch path on the CPU (where a
node without a GPU quantizes), the torch path on CUDA, and the TileLang kernel —
and whether the kernel's bytes are the torch path's.

    python bench/quant_cost.py
    python bench/quant_cost.py --elements 1000000000   # a 1B-parameter delta

Timed with a synchronize on each side, the first call excluded (compilation).
With the kernel present it first checks the kernel's bytes against the torch
path's on awkward shapes and dtypes, and says CHECK OK or CHECK FAILED - the
same comparison as ``tests/test_dist_quant_kernel.py``, for a node that has the
bench but not the tests.
"""

import argparse
import time

import torch

from ravex._dist import quant_tilelang
from ravex._dist.quant import FORMATS, encode_torch


def timed(fn, repeat):
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeat, out


def check():
    """The kernel against the torch path on shapes that exercise the padding."""
    failures = []
    for name, fmt in FORMATS.items():
        for shape in [(1,), (31,), (4096, 4096), (3, 1000, 7), (0,)]:
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(0)
                x = (torch.randn(shape) * torch.logspace(-5, 1, shape[-1] or 1)).to(dtype)
                want = encode_torch(x, fmt)
                got = quant_tilelang.encode(x.cuda(), fmt)
                ok = got is not None and all(torch.equal(a, b) for a, b in zip(got, want))
                if not ok:
                    failures.append("%s %s %s" % (name, shape, dtype))
    print("CHECK %s%s" % ("OK" if not failures else "FAILED: ",
                          "; ".join(failures)), flush=True)
    return not failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--link-mbps", type=float, default=7.0,
                        help="MB/s of the link the saving is measured against")
    args = parser.parse_args()

    x = torch.randn(args.elements)
    gpu = torch.cuda.is_available()
    print("%d elements (%.0f MB fp32); cuda=%s; tilelang=%s%s"
          % (args.elements, args.elements * 4 / 1e6, gpu, quant_tilelang.available(),
             " (%s)" % torch.cuda.get_device_name() if gpu else ""))

    if quant_tilelang.available():
        check()

    for name, fmt in FORMATS.items():
        cpu, want = timed(lambda: encode_torch(x, fmt), 1)
        line = "%-10s torch/cpu %7.3fs" % (name, cpu)
        if gpu:
            xc = x.cuda()
            cuda, _ = timed(lambda: encode_torch(xc, fmt), args.repeat)
            line += "  torch/cuda %7.4fs" % cuda
            if quant_tilelang.available():
                kernel, got = timed(lambda: quant_tilelang.encode(xc, fmt), args.repeat)
                same = got is not None and all(torch.equal(a, b) for a, b in zip(got, want))
                line += "  tilelang %7.4fs  same bytes: %s" % (kernel, same)
        wire = args.elements * 4 / 1e6 * (1 - (fmt.bits + 8 / fmt.group) / 32)
        line += "  saves %.0f MB = %.1fs at %g MB/s" % (wire, wire / args.link_mbps, args.link_mbps)
        print(line, flush=True)


if __name__ == "__main__":
    main()
