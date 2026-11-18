"""The sm120 C-API weights-on-B library (``sm120-capi-b8``, 8x64 granule) against the emulated backend, and the
C-API backend names. The kernel tests skip unless an sm120 GPU and the built libraries are present."""
import pytest
import torch
from torch import nn

from flipquant import maps, native
from flipquant.linear import FlipLinear, QuantSpec


def _lib(name):
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and native.get_backend(name)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


needs_b8 = pytest.mark.skipif(not _lib("sm120-capi-b8"), reason="needs an sm120 GPU and libflipquant_sm120_b8.so")
needs_both = pytest.mark.skipif(not (_lib("sm120-capi-b8") and _lib("sm120-capi")),
                                reason="needs an sm120 GPU and both sm120 libraries")


def test_capi_names():
    """The C-API libraries are FlipLinear backends only by name on sm_120: 'sm120' is the optimized path
    (models/sm120.py), 'auto' resolves only on sm_100."""
    with pytest.raises(RuntimeError, match="sm120-capi"):
        native.get_backend("sm120")
    assert native.CAPI_ARCH["sm120-capi-b8"] == "sm120b8" and native.CAPI_LIB["sm120b8"] == "libflipquant_sm120_b8.so"


def _layer(o, k, unit, backend, mask, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(k, o, bias=False, device="cuda", dtype=torch.bfloat16)
    lin.weight.data = (torch.randn(o, k, device="cuda") * 0.05).bfloat16()
    return FlipLinear.from_linear(lin, QuantSpec(unit=unit, mode="native", native_backend=backend), mask=mask)


@needs_b8
@pytest.mark.parametrize("o,k,m", [(40, 192, 7), (264, 320, 130), (4480, 4480, 1), (1024, 4096, 300)])
def test_sm120_b8_matches_emulate_at_8x64(o, k, m):
    """Random per-tile formats at 8x64 (rows that are not 16-aligned included): same decoded weight,
    and the kernel output equals the emulated FP32 product rounded to BF16."""
    mask = torch.rand(maps.tile_shape(o, k, "8x64"), generator=torch.Generator().manual_seed(1)) < 0.5
    b8, em = _layer(o, k, "8x64", "sm120-capi-b8", mask), _layer(o, k, "8x64", "emulate", mask)
    assert b8._backend.granule == (8, 64)
    assert torch.equal(b8.effective_weight().view(torch.int16), em.effective_weight().view(torch.int16))
    x = torch.randn(m, k, device="cuda").bfloat16()
    yb, ye = b8(x), em(x)
    mism = (yb.view(torch.int16) != ye.view(torch.int16)).float().mean().item()
    rel = ((yb.float() - ye.float()).norm() / ye.float().norm()).item()
    assert rel < 2e-3 and mism < 0.05, (rel, mism)


@needs_both
def test_three_backends_on_one_16x64_map():
    o, k = 512, 1024
    mask = torch.rand(maps.tile_shape(o, k, "16x64"), generator=torch.Generator().manual_seed(2)) < 0.3
    layers = {b: _layer(o, k, "16x64", b, mask) for b in ("sm120-capi", "sm120-capi-b8", "emulate")}
    w = {b: l.effective_weight() for b, l in layers.items()}
    assert torch.equal(w["sm120-capi"], w["emulate"]) and torch.equal(w["sm120-capi-b8"], w["emulate"])
    x = torch.randn(64, k, device="cuda").bfloat16()
    y = {b: l(x).float() for b, l in layers.items()}
    for b in ("sm120-capi", "sm120-capi-b8"):
        assert ((y[b] - y["emulate"]).norm() / y["emulate"].norm()).item() < 2e-3, b
