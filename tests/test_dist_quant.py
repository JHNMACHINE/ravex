"""Block-scaled quantization of the outer delta (GPU-139)."""

import pytest
import torch

from ravex._dist import quant, report
from ravex._dist.quant import FORMATS, decode, encode, encoded_lengths, roundtrip

FP4 = FORMATS["fp4_block"]
FP8 = FORMATS["fp8_block"]

try:
    import moonclip  # noqa: F401

    HAVE_MOONCLIP = True
except ImportError:  # pragma: no cover
    HAVE_MOONCLIP = False

needs_moonclip = pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")


class TestTheFormats:
    def test_fp4_rounds_to_nearest_with_ties_to_even(self):
        # One block whose amax is exactly 6, so the scale is 1 and every value
        # below is what the code sees. Each tie sits between two codes, and the
        # even one - the one the hardware cast picks - must win.
        values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, 6.0]
        block = torch.tensor(values + [0.0] * (32 - len(values)))
        got = roundtrip(block, FP4)[: len(values)].tolist()
        assert got == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.0, -4.0, 6.0]

    def test_a_small_negative_value_rounds_to_negative_zero(self):
        """What the hardware cast writes, so what the kernels write too."""
        block = torch.tensor([6.0, -0.1, 0.1, -0.0] + [0.0] * 28)
        codes, _ = encode(block, FP4)
        nibbles = [codes[0] & 0xF, codes[0] >> 4, codes[1] & 0xF, codes[1] >> 4]
        assert [int(n) for n in nibbles] == [7, 8, 0, 8]

    def test_fp8_agrees_with_the_torch_cast_at_the_block_scale(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128)
        got = roundtrip(x, FP8)
        for row in range(4):
            amax = x[row].abs().max()
            mantissa, exponent = torch.frexp(amax / 448.0)
            exponent = exponent - (mantissa == 0.5).int()
            scale = torch.ldexp(torch.tensor(1.0), exponent)
            want = (x[row] / scale).to(torch.float8_e4m3fn).float() * scale
            assert torch.equal(got[row], want)

    def test_the_largest_element_of_a_block_always_fits(self):
        torch.manual_seed(1)
        for fmt in FORMATS.values():
            x = torch.randn(10, fmt.group) * torch.logspace(-8, 3, 10)[:, None]
            codes, exponents = encode(x, fmt)
            scales = torch.ldexp(torch.ones(10), exponents.int() - 127)
            assert (x.abs().amax(dim=1) / scales <= fmt.maxval).all()

    def test_a_scale_per_block_keeps_small_blocks_alive(self):
        """The point of the issue: one scale per tensor zeroes a small block."""
        small = torch.full((32,), 1e-4)
        large = torch.full((32,), 1.0)
        x = torch.cat([small, large])
        got = roundtrip(x, FP4)
        assert torch.allclose(got[:32], small, rtol=0.25)

        # The same values under one scale for the whole tensor, which the
        # large block decides: 1e-4 lands far below the smallest code and the
        # small half is simply gone.
        whole = FORMATS["fp4_block"].__class__("whole", group=64, maxval=6.0, bits=4)
        assert (roundtrip(x, whole)[:32] == 0).all()

    def test_zero_blocks_stay_zero_and_their_scale_is_finite(self):
        for fmt in FORMATS.values():
            codes, exponents = encode(torch.zeros(fmt.group * 2), fmt)
            assert (codes == 0).all()
            assert (exponents > 0).all()
            assert torch.equal(decode(codes, exponents, fmt, (fmt.group * 2,), torch.float32),
                               torch.zeros(fmt.group * 2))

    @pytest.mark.parametrize("shape", [(0,), (1,), (3, 5), (33,), (2, 3, 7)])
    @pytest.mark.parametrize("fmt", list(FORMATS.values()), ids=list(FORMATS))
    def test_shape_dtype_and_length(self, shape, fmt):
        x = torch.randn(shape, dtype=torch.bfloat16)
        codes, exponents = encode(x, fmt)
        numel = x.numel()
        assert (codes.numel(), exponents.numel()) == encoded_lengths(numel, fmt)
        back = decode(codes, exponents, fmt, x.shape, x.dtype)
        assert back.shape == x.shape and back.dtype == torch.bfloat16

    def test_bits_on_the_wire(self):
        assert quant.wire_bits(FP4) == 4.25
        assert quant.wire_bits(FP8) == 8.0625


@needs_moonclip
class TestTheReport:
    def delta(self):
        torch.manual_seed(2)
        return {"w": torch.randn(64, 33), "b": torch.randn(7), "empty": torch.zeros(0)}

    @pytest.mark.parametrize("name", list(FORMATS))
    def test_a_peer_reads_exactly_the_local_roundtrip(self, tmp_path, name):
        delta = self.delta()
        report.publish_round(str(tmp_path), 1, delta, 5, "n0", save_dtype=name)
        got = report.read_round(str(tmp_path), 1, report.expectation(delta))
        assert got.steps == 5
        for key, value in delta.items():
            assert torch.equal(got.delta[key], roundtrip(value, FORMATS[name])), key

    def test_moonclip_does_not_cast_the_codes_again(self, tmp_path):
        delta = self.delta()
        path = report.publish_round(str(tmp_path), 1, delta, 5, "n0", save_dtype="fp4_block")
        store = report.open_store(path, save_dtype="fp4_block")
        described = store.describe(report.snapshot_of_round(store, 1))
        stored = {e["name"]: e.get("stored_dtype", e.get("dtype")) for e in described["tensors"]}
        assert set(stored.values()) == {"uint8"}
        assert described["metadata"][report.QUANT] == "fp4_block"

    def test_a_different_model_is_refused_from_the_manifest(self, tmp_path):
        delta = self.delta()
        report.publish_round(str(tmp_path), 1, delta, 5, "n0", save_dtype="fp4_block")
        other = dict(delta, w=torch.randn(64, 34))
        with pytest.raises(report.ReportError, match="needs"):
            report.read_round(str(tmp_path), 1, report.expectation(other))
        with pytest.raises(report.ReportError, match="tensor"):
            report.read_round(str(tmp_path), 1, report.expectation({"w": delta["w"]}))

    def test_an_unknown_format_is_refused_by_name(self, tmp_path):
        delta = self.delta()
        path = report.round_path(str(tmp_path), 1)
        store = report.open_store(path)
        report.write(store, delta, 1, 5, "n0", quantize="fp4_block")
        # Rewrite the same round claiming a format this version does not know.
        codes = {}
        for key, value in delta.items():
            c, e = encode(value, FP4)
            codes[key], codes[key + report.EXPONENTS] = c, e
        store.save_tensors(2, codes, metadata={report.QUANT: "fp2_block"})
        with pytest.raises(report.ReportError, match="fp2_block"):
            report.read(store, 2, report.expectation(delta))


def test_config_accepts_the_block_formats(monkeypatch):
    from ravex._config import RavexConfig

    monkeypatch.setenv("RAVEX_OUTER_SAVE_DTYPE", "FP4_Block")
    config = RavexConfig.load()
    assert config.outer_save_dtype == "fp4_block" and not config.problems

    monkeypatch.setenv("RAVEX_OUTER_SAVE_DTYPE", "fp3_block")
    config = RavexConfig.load()
    assert config.outer_save_dtype is None
    assert any("fp4_block" in p for p in config.problems)


def test_error_feedback_carries_what_was_dropped():
    from ravex._dist.outer import OuterLoop

    model = torch.nn.Linear(4, 4)
    loop = OuterLoop(model, inner_steps=1)
    sent = {"weight": torch.full((4, 4), 1.3), "bias": torch.full((4,), 0.2)}
    published = {k: roundtrip(v, FP4) for k, v in sent.items()}
    loop.keep_residual(sent, published)
    carried = loop.with_residual({k: torch.zeros_like(v) for k, v in sent.items()})
    for key in sent:
        assert torch.allclose(carried[key], sent[key] - published[key])
    assert "residual" not in loop.state_dict()
