"""The block quantization of :mod:`ravex._dist.quant`, as TileLang kernels (GPU-139).

The kernels came from our own reference (the GPU-139 thread) and keep its
shape: a CTA per 32 rows of blocks, an ``amax`` reduction per block, a
power-of-two scale through IEEE-754 bit manipulation rather than ``log2`` and
``ceil``, and the cast done by the hardware. Two things were changed, both
so that a kernel and :func:`ravex._dist.quant.encode` produce **the same bytes**
— the contract that makes the torch path an oracle rather than an estimate:

* the floor on ``amax`` is ``maxval * 2**-126`` for both formats. The reference
  used ``1e-4`` for fp8, which gives an all-zero block a different exponent
  than the torch path does, and a peer reading either would decode zeros
  either way — but the bytes would differ, and "almost the same bytes" is not
  something a test can hold a kernel to;
* the fp8 scale is written as its exponent byte (e8m0), like the fp4 one,
  instead of a float32: a quarter of the size, and the wire format is one.

**Optional in every sense.** TileLang is not a dependency of Ravex: a compiler
inside a library that otherwise needs PyYAML is a weight to opt into, not to
impose (the issue's first objection). :func:`available` answers whether this
process can use the kernels, once; everything else goes through the torch
path, which is also where a failed kernel call falls back to.

The tensor is handled as a flat row of blocks, like the torch path: zero-padded
into rows of ``ROW`` elements and a whole number of 32-row tiles, run, and the
padding cut off again. Padding blocks are all zeros, so they change nothing
the reader sees.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

logger = logging.getLogger("ravex")

#: Elements per row handed to a kernel: a multiple of both group sizes.
ROW = 1024
#: Rows per tile, as in the reference: the kernels need whole tiles.
BLK_M = 32

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
FP32 = "float32"

#: Set by the first failed kernel call: from then on this process quantizes on
#: the torch path, rather than failing and logging once per round.
_failed = False


@functools.lru_cache(maxsize=1)
def _tilelang():
    """``(tilelang, T)`` or None, decided once per process."""
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:  # ImportError, or a broken CUDA toolchain
        logger.debug("TileLang is not usable here (%s); using the torch path", exc)
        return None
    return tilelang, T


def available() -> bool:
    """Whether the kernels can run in this process at all."""
    if _failed:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    return _tilelang() is not None


def program(bits: int, group: int, in_dtype: str, N: int = ROW):
    """The TileLang program for one format, before compilation.

    Separate from :func:`_kernel` so that it can be compiled for a named target
    on a machine without a GPU - lowering and ``nvcc`` need none - which is how
    a kernel's errors are found before a node is rented for it.
    """
    _, T = _tilelang()

    def fast_log2_ceil(x):
        """ceil(log2(x)) from the float's bits: exponent, plus one if any
        mantissa bit is set. Exact at powers of two, where log2+ceil may not be."""
        bits_x = T.reinterpret("uint32", x)
        exp_x = (bits_x >> 23) & 0xFF
        man_bits = bits_x & ((1 << 23) - 1)
        return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

    def fast_pow2(x):
        return T.reinterpret("float32", (x + 127) << 23)

    maxval = 448.0 if bits == 8 else 6.0
    maxval_inv = 1.0 / maxval
    floor = maxval * 2.0 ** -126
    code_dtype = FP8 if bits == 8 else FP4
    blk_m = BLK_M

    M = T.symbolic("M")

    @T.prim_func
    def quant(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), code_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group)), FE8M0],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group), threads=128) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, group), in_dtype)
            x_local = T.alloc_fragment((blk_m, group), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), FP32)
            s_local = T.alloc_fragment((blk_m,), FP32)
            y_local = T.alloc_fragment((blk_m, group), code_dtype)
            y_shared = T.alloc_shared((blk_m, group), code_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    # Keep the scale of an all-zero block a number: the same
                    # floor as the torch path, see the module doc.
                    amax_local[i] = T.max(amax_local[i], floor)
                    s_local[i] = fast_pow2(fast_log2_ceil(amax_local[i] * maxval_inv))
                for i, j in T.Parallel(blk_m, group):
                    y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], -maxval, maxval)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(FE8M0, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group])

    return quant


@functools.lru_cache(maxsize=None)
def _kernel(bits: int, group: int, in_dtype: str, target: str = "auto"):
    tilelang, _ = _tilelang()
    return tilelang.compile(program(bits, group, in_dtype), target=target)


def encode(tensor, fmt) -> Optional[Tuple["object", "object"]]:
    """``(codes, exponents)`` as :func:`ravex._dist.quant.encode` lays them
    out, on the CPU — or None, and the caller takes the torch path."""
    import torch

    if not tensor.is_cuda or not available():
        return None
    try:
        numel = tensor.numel()
        blocks = -(-numel // fmt.group) if numel else 0
        tile = ROW * BLK_M
        padded = max(tile, -(-numel // tile) * tile)
        flat = torch.zeros(padded, dtype=torch.float32, device=tensor.device)
        flat[:numel] = tensor.detach().reshape(-1).to(torch.float32)
        rows = padded // ROW

        kernel = _kernel(fmt.bits, fmt.group, "float32")
        if fmt.bits == 8:
            codes = torch.empty(rows, ROW, dtype=torch.float8_e4m3fn, device=tensor.device)
        else:
            codes = torch.empty(rows, ROW // 2, dtype=torch.float4_e2m1fn_x2, device=tensor.device)
        scales = torch.empty(rows, ROW // fmt.group, dtype=torch.float8_e8m0fnu, device=tensor.device)
        kernel(flat.view(rows, ROW), codes, scales)

        code_bytes = blocks * fmt.group * fmt.bits // 8
        codes = codes.view(torch.uint8).reshape(-1)[:code_bytes]
        exponents = scales.view(torch.uint8).reshape(-1)[:blocks]
        return codes.cpu().contiguous(), exponents.cpu().contiguous()
    except Exception as exc:
        global _failed
        _failed = True
        logger.warning(
            "The TileLang %s kernel failed (%s: %s); quantizing on the torch "
            "path for the rest of this process",
            fmt.name, type(exc).__name__, exc,
        )
        return None
