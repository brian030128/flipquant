"""Bitwise agreement of flipquant.formats with the reference quantizers.

The reference repo is located via $REFERENCE_ROOT (default: its checkout on the cluster); tests that need
it are skipped when it is absent.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

from flipquant import formats as F

REFERENCE = Path(os.environ.get("REFERENCE_ROOT", "/home/u4320956/NVFP4-RaZeR"))
needs_reference = pytest.mark.skipif(not (REFERENCE / "quantize" / "quantizer.py").exists(),
                                     reason="reference quantizers not found ($REFERENCE_ROOT)")


def _ref():
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    from quantize import quantizer
    from quantize.causal_four_over_six import quantize_rows
    from quantize.fast_act import quant_per_document
    return quantizer, quantize_rows, quant_per_document


def _weights(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(96, 256, generator=g) * 0.02
    w[3, 5] = 0.9                                        # outlier -> tiny block scales elsewhere
    w[7, :16] = 0.0                                      # all-zero block
    w[9] = torch.randn(256, generator=g).sign() * 0.01   # tie-heavy row
    heavy = torch.distributions.StudentT(2.0).sample((64, 256)) * 0.01
    return [w.bfloat16(), heavy.bfloat16()]


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@needs_reference
@pytest.mark.parametrize("seed", [0, 1])
def test_weight_quantizers_bitwise(seed):
    q, _, _ = _ref()
    for w in _weights(seed):
        assert _same(F.fake_quant(w, "nvfp4"), q.quant_nvfp4(w, 4, 16))
        assert _same(F.fake_quant(w, "fourover6"), q.quant_nvfp4_4over6(w, 4, 16))
        ref_a = q.quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip="a1", elect="always")
        assert _same(F.fake_quant(w, "e0m3"), ref_a)


@needs_reference
def test_activation_quantizers_bitwise():
    _, rows, per_doc = _ref()
    x = (torch.randn(3, 40, 256) * torch.logspace(-2, 1, 256)).bfloat16()
    assert _same(F.fake_quant(x, "fourover6", "row"), rows(x))
    assert _same(F.fake_quant(x, "fourover6", "document"), per_doc(x))
    assert _same(F.fake_quant(x[0], "fourover6", "document"), per_doc(x[0]))


def test_pack_roundtrip_and_mixed_decode():
    w = torch.randn(64, 128).bfloat16()
    a, b = F.e0m3(w), F.fourover6(w)
    assert torch.equal(a.gscale, b.gscale)
    assert torch.equal(F.unpack_nibbles(F.pack_nibbles(a.codes)), a.codes)
    assert torch.equal(a.scale_bytes().view(torch.float8_e4m3fn).float(), a.scale)
    fmt = torch.zeros(64, 8, dtype=torch.bool)
    fmt[::2] = True
    wide = fmt.repeat_interleave(16, 1)
    mixed = F.FP4Codes(torch.where(wide, a.codes, b.codes), torch.where(fmt, a.scale, b.scale), a.gscale, fmt)
    assert _same(mixed.dequantize(), torch.where(wide, a.dequantize(), b.dequantize()))


def test_row_gscale_shapes():
    x = torch.randn(2, 5, 64)
    assert F.fourover6(x, "row").row_gscale().shape == (10,)
    assert F.fourover6(x, "document").row_gscale().shape == (10,)
    assert F.fourover6(x, "tensor").row_gscale().shape == (10,)


def test_nvfp4_half_ulp_below_a_tie_rounds_up():
    """quant_nvfp4 rounds with FP32 ``floor(|y| * 2 + 0.5)``: a |y| half an FP32 ulp below 0.25 gives 2y + 0.5 halfway
    below 1.0, which rounds to 1.0, so the element becomes 0.5, not 0 (nvfp4 reproduces it for bitwise agreement)."""
    y = torch.tensor([0.25 - 2.0 ** -26, -(0.25 - 2.0 ** -26), 0.25 - 2.0 ** -25])
    assert F._round_e2m1_nvfp4(y).tolist() == [1, 1, 0]
    assert F._round_e2m1(y, "up").tolist() == [0, 0, 0]         # the exact rounding differs here


@needs_reference
def test_nvfp4_near_ties_match_quant_nvfp4():
    """Random weights with elements half an ulp below a tie (seed 1 has five): bitwise equal to quant_nvfp4."""
    quantizer, _, _ = _ref()
    w = (torch.randn(512, 256, generator=torch.Generator().manual_seed(1)) * 0.02).bfloat16()
    b, gs = F._grouped(w, "tensor")
    ws = b / gs
    y = ws / F._e4m3(ws.abs().amax(-1, keepdim=True) / F.E2M1_MAX)
    assert int((F._round_e2m1(y, "up") != F._round_e2m1_nvfp4(y)).sum()) > 0      # the case is present
    assert _same(F.fake_quant(w, "nvfp4"), quantizer.quant_nvfp4(w, 4, 16))
