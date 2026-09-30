"""The TileLang kernels against the torch path, byte for byte (GPU-139).

The comparison needs a CUDA device and TileLang, and skips without them — which
on a CPU box is always. The dispatch around it does not, and is tested here with
a stand-in kernel: which path runs, and what happens when the kernel fails.
"""

import pytest
import torch

from ravex._dist import quant, quant_tilelang
from ravex._dist.quant import FORMATS, encode, encode_torch

on_gpu = pytest.mark.skipif(
    not quant_tilelang.available(), reason="needs CUDA and TileLang"
)


@on_gpu
@pytest.mark.parametrize("fmt", list(FORMATS.values()), ids=list(FORMATS))
@pytest.mark.parametrize("shape", [(1,), (31,), (4096, 4096), (3, 1000, 7), (0,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_kernel_writes_the_bytes_the_torch_path_writes(fmt, shape, dtype):
    torch.manual_seed(0)
    x = (torch.randn(shape) * torch.logspace(-5, 1, shape[-1] or 1)).to(dtype)
    want_codes, want_exponents = encode_torch(x, fmt)
    got = quant_tilelang.encode(x.cuda(), fmt)
    assert got is not None, "the kernel declined a CUDA tensor"
    codes, exponents = got
    assert torch.equal(exponents, want_exponents)
    assert torch.equal(codes, want_codes)


@on_gpu
def test_the_torch_path_on_cuda_agrees_with_itself_on_cpu():
    x = torch.randn(10_000)
    for fmt in FORMATS.values():
        on_cuda = encode_torch(x.cuda(), fmt)
        on_cpu = encode_torch(x, fmt)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(on_cuda, on_cpu))


class TestTheDispatch:
    def test_a_cpu_tensor_never_asks_the_kernel(self, monkeypatch):
        asked = []
        monkeypatch.setattr(quant_tilelang, "encode", lambda t, f: asked.append(1))
        encode(torch.randn(64), FORMATS["fp4_block"])
        assert asked == []

    def test_a_kernel_answer_is_used_as_is(self, monkeypatch):
        sentinel = (torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8))
        monkeypatch.setattr(quant_tilelang, "encode", lambda t, f: sentinel)
        fake = torch.randn(64)
        monkeypatch.setattr(type(fake), "is_cuda", property(lambda self: True))
        try:
            assert quant.encode(fake, FORMATS["fp4_block"]) is sentinel
        finally:
            monkeypatch.undo()

    def test_a_declined_kernel_falls_back_to_torch(self, monkeypatch):
        monkeypatch.setattr(quant_tilelang, "encode", lambda t, f: None)
        x = torch.randn(64)
        monkeypatch.setattr(type(x), "is_cuda", property(lambda self: True))
        try:
            got = quant.encode(x, FORMATS["fp8_block"])
        finally:
            monkeypatch.undo()
        want = encode_torch(x, FORMATS["fp8_block"])
        assert all(torch.equal(a, b) for a, b in zip(got, want))

    def test_one_failure_turns_the_kernel_off(self, monkeypatch):
        monkeypatch.setattr(quant_tilelang, "_failed", False)
        monkeypatch.setattr(quant_tilelang, "available", lambda: not quant_tilelang._failed)

        def boom(*a, **k):
            raise RuntimeError("no kernel image for this GPU")

        monkeypatch.setattr(quant_tilelang, "_kernel", boom)

        class Cuda:
            is_cuda = True
            device = "cuda"

            def numel(self):
                return 10

        assert quant_tilelang.encode(Cuda(), FORMATS["fp4_block"]) is None
        assert quant_tilelang._failed is True
