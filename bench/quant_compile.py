"""Compile the TileLang quantization kernels for named GPUs, without one (GPU-139).

Lowering TileLang to CUDA C and compiling that with nvcc need no device, so a
kernel's errors can be found on any machine with TileLang and a CUDA toolkit —
smith's GPU image is one — before a node is rented to run it. What this cannot
check is the kernel running: that is ``bench/quant_cost.py``, on a GPU.

    python bench/quant_compile.py                    # sm_80 sm_89 sm_90a sm_100a sm_120a
    python bench/quant_compile.py sm_89 --source     # and print the CUDA

Measured 2026-09-30 in smith's image (TileLang 0.1.15, nvcc 12.8): both formats
compile for all five. Hopper and Blackwell need the ``a`` targets, because
TileLang warp-specializes the pipeline there; a real device's ``auto`` target
picks them itself.
"""

import argparse
import sys

import tilelang

from ravex._dist import quant_tilelang
from ravex._dist.quant import FORMATS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", nargs="*",
                        default=["sm_80", "sm_89", "sm_90a", "sm_100a", "sm_120a"])
    parser.add_argument("--source", action="store_true", help="print the CUDA C")
    args = parser.parse_args()

    print("tilelang", tilelang.__version__)
    failed = 0
    for arch in args.arch:
        for name, fmt in FORMATS.items():
            program = quant_tilelang.program(fmt.bits, fmt.group, "float32")
            try:
                kernel = tilelang.compile(
                    program, target={"kind": "cuda", "arch": arch},
                    execution_backend="tvm_ffi",
                )
            except Exception as exc:
                failed += 1
                errors = [l for l in str(exc).splitlines() if "error" in l.lower()]
                print("FAIL     %-8s %-9s %s" % (arch, name, " | ".join(errors[:3])[:600]))
                continue
            print("COMPILED %-8s %-9s" % (arch, name), flush=True)
            if args.source:
                print(kernel.get_kernel_source())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
