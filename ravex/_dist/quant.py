"""Block-scaled quantization of the outer delta, for the wire (GPU-139).

``outer_save_dtype`` used to mean one thing: a cast that Moonclip applies on
the CPU, with **one scale per tensor**. That keeps the range and loses the
dynamics inside a tensor — measured on 2026-09-20 with
``bench/resume_precision.py``, fp8 at one scale per tensor zeroes 0.71% of the
smallest entries of a second moment — and it is what makes fp4 impossible
rather than merely worse: sixteen levels spread over a whole tensor's range is
not a delta any more.

This is the other way: **one scale per block** of 128 elements (fp8) or 32
(fp4), the microscaling layout the MX formats use, so a block of small values
gets a small scale of its own.

Two formats, both named after what goes on the wire:

``fp8_block``
    float8 e4m3 codes, one byte each, and a power-of-two scale per 128.
    8.06 bits per element.
``fp4_block``
    float4 e2m1 codes, two per byte, and a power-of-two scale per 32 — MXFP4.
    4.25 bits per element: 7.5x under fp32 and 3.8x under bf16, which is the
    difference between H of ~3200 and ~800 for a 1B model on 7 MB/s (GPU-113).

**Scales are powers of two, stored as one exponent byte** (e8m0, bias 127).
Dividing by a power of two is exact, so the only rounding is the element's own,
and an exponent byte costs a quarter of a float32 scale. The exponent is
``ceil(log2(amax / max))``, so the largest element of a block always fits.

**Rounding is to nearest, ties to even** — what the hardware cast does, so a
kernel and this code agree byte for byte where they both run.

**This module is the floor, not a fallback in the pejorative sense.** The issue
decided that the kernels are TileLang, and that every kernel has a pure-torch
path beside it: where the kernel does not run — an old GPU, no GPU, no
compiler — and as the oracle a kernel is tested against. It also settles the
protocol question the issue raised: since every node can produce either format
here, on any device, **the format on the wire is one per job**, set in the
config, and nobody negotiates it per node. A faster path is an optimisation a
node may or may not have; it never changes what it sends.

**Quantized once.** The codes go to Moonclip as plain ``uint8`` tensors with no
``save_dtype`` of Moonclip's own on top: a per-tensor scale applied over values
that already carry per-block ones would be a second quantization of something
Moonclip knows nothing about (the issue's "one cast per tensor").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class Format:
    name: str
    #: Elements that share one scale.
    group: int
    #: The largest magnitude a code can hold.
    maxval: float
    #: Bits per code.
    bits: int


FORMATS = {
    "fp8_block": Format("fp8_block", group=128, maxval=448.0, bits=8),
    "fp4_block": Format("fp4_block", group=32, maxval=6.0, bits=4),
}

#: The magnitudes an e2m1 code can hold, by its three low bits.
_FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
#: Midpoints between neighbouring magnitudes: where rounding changes code.
_FP4_BOUNDS = (_FP4_VALUES[1:] + _FP4_VALUES[:-1]) / 2

_EXPONENT_BIAS = 127


def is_block(name) -> bool:
    return str(name) in FORMATS


def wire_bits(fmt: Format) -> float:
    """Bits per element on the wire, the scale included."""
    return fmt.bits + 8.0 / fmt.group


def _blocks(tensor: torch.Tensor, fmt: Format) -> torch.Tensor:
    """The tensor as float32 rows of ``group``, zero-padded at the end."""
    flat = tensor.detach().reshape(-1).to(torch.float32)
    padding = (-flat.numel()) % fmt.group
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    return flat.view(-1, fmt.group)


def _exponents(blocks: torch.Tensor, fmt: Format) -> torch.Tensor:
    """``ceil(log2(amax / maxval))`` per block, as int32.

    Through ``frexp`` rather than ``log2``: ``r = m * 2**e`` with ``m`` in
    [0.5, 1), so ``ceil(log2 r)`` is ``e`` unless ``m`` is exactly 0.5, where
    ``r`` is a power of two and the answer is ``e - 1``. Exact, where a
    floating ``log2`` followed by ``ceil`` is not guaranteed to be at a power
    of two. An all-zero block gets the smallest exponent, so its scale is
    still a number and its codes are zeros — frozen parameters, whose delta is
    zero, are exactly these.
    """
    amax = blocks.abs().amax(dim=1)
    amax = torch.clamp(amax, min=fmt.maxval * 2.0 ** -126)
    mantissa, exponent = torch.frexp(amax * (1.0 / fmt.maxval))
    exponent = exponent - (mantissa == 0.5).to(exponent.dtype)
    return exponent.clamp(-_EXPONENT_BIAS, _EXPONENT_BIAS).to(torch.int32)


def _fp4_codes(scaled: torch.Tensor) -> torch.Tensor:
    """e2m1 codes, 0..15, sign in bit 3. Ties go to the even code."""
    magnitude = scaled.abs().clamp(max=6.0)
    bounds = _FP4_BOUNDS.to(scaled.device)
    index = torch.bucketize(magnitude, bounds)  # a tie lands on the lower code
    at_tie = magnitude == bounds[index.clamp(max=bounds.numel() - 1)]
    index = torch.where(at_tie & (index % 2 == 1), index + 1, index)
    sign = ((scaled < 0) & (index > 0)).to(torch.int64)
    return (index | (sign << 3)).to(torch.uint8)


def encode(tensor: torch.Tensor, fmt: Format) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(codes, exponents)``, both flat ``uint8``.

    ``codes`` is one byte per element for fp8 and one per two for fp4, over
    the tensor padded to whole blocks; ``exponents`` is one byte per block.
    """
    blocks = _blocks(tensor, fmt)
    exponent = _exponents(blocks, fmt)
    scaled = blocks * torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), -exponent)[:, None]

    if fmt.bits == 8:
        codes = scaled.clamp(-fmt.maxval, fmt.maxval).to(torch.float8_e4m3fn)
        codes = codes.view(torch.uint8).reshape(-1)
    else:
        nibbles = _fp4_codes(scaled).reshape(-1)
        codes = nibbles[0::2] | (nibbles[1::2] << 4)
    return codes.contiguous(), (exponent + _EXPONENT_BIAS).to(torch.uint8)


def decode(
    codes: torch.Tensor,
    exponents: torch.Tensor,
    fmt: Format,
    shape,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The tensor a peer reconstructs from ``encode``'s output."""
    numel = 1
    for dimension in shape:
        numel *= int(dimension)

    if fmt.bits == 8:
        values = codes.view(torch.float8_e4m3fn).to(torch.float32)
    else:
        low = codes & 0x0F
        high = codes >> 4
        nibbles = torch.stack([low, high], dim=1).reshape(-1).to(torch.int64)
        magnitude = _FP4_VALUES.to(codes.device)[nibbles & 0x7]
        values = torch.where((nibbles & 0x8) != 0, -magnitude, magnitude)

    exponent = exponents.to(torch.int32) - _EXPONENT_BIAS
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    values = values.view(-1, fmt.group) * scale[:, None]
    return values.reshape(-1)[:numel].reshape(tuple(shape)).to(dtype)


def encoded_lengths(numel: int, fmt: Format) -> Tuple[int, int]:
    """How many code bytes and exponent bytes a tensor of ``numel`` encodes to."""
    blocks = -(-numel // fmt.group) if numel else 0
    return blocks * fmt.group * fmt.bits // 8, blocks


def roundtrip(tensor: torch.Tensor, fmt: Format) -> torch.Tensor:
    """What a peer will read of ``tensor``: encode, then decode."""
    codes, exponents = encode(tensor, fmt)
    return decode(codes, exponents, fmt, tensor.shape, tensor.dtype)
