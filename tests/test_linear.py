import pytest
import torch
from torch import nn

from flipquant import formats as F
from flipquant import maps
from flipquant.linear import FlipLinear, QuantSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _linear(o=48, k=192, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(k, o, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = (torch.randn(o, k, device=DEV) * 0.05).bfloat16()
    return lin


def _mask(lin, unit="8x64", seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(maps.tile_shape(*lin.weight.shape, unit), generator=g) > 0.5


def test_fake_weight_matches_where_of_candidates():
    lin = _linear()
    m = _mask(lin)
    layer = FlipLinear.from_linear(lin, QuantSpec(mode="fake"), mask=m)
    a, b = F.fake_quant(lin.weight, "e0m3"), F.fake_quant(lin.weight, "fourover6")
    expect = torch.where(maps.expand(m.to(DEV), 8, 64, 48), a, b)
    assert torch.equal(layer.weight.view(torch.int16), expect.view(torch.int16))


@pytest.mark.parametrize("scope", ["row", "document"])
def test_native_emulate_matches_fake(scope):
    lin = _linear()
    m = _mask(lin)
    fake = FlipLinear.from_linear(lin, QuantSpec(mode="fake", act_scope=scope), mask=m)
    nat = FlipLinear.from_linear(lin, QuantSpec(mode="native", native_backend="emulate", act_scope=scope), mask=m)
    x = torch.randn(2, 5, 192, device=DEV).bfloat16()
    yf, yn = fake(x), nat(x)
    rel = ((yf.float() - yn.float()).norm() / yf.float().norm()).item()
    assert rel < 1e-2, rel
    assert torch.equal(nat.effective_weight().view(torch.int16), fake.weight.view(torch.int16))


def test_set_map_and_uniform_formats():
    lin = _linear()
    layer = FlipLinear.from_linear(lin, QuantSpec(mode="fake"), trainable=True)
    assert torch.equal(layer.weight, F.fake_quant(lin.weight, "fourover6"))
    layer.set_map(torch.ones(6, 3, dtype=torch.bool))
    assert torch.equal(layer.weight, F.fake_quant(lin.weight, "e0m3"))
    for w in ("nvfp4", "fourover6", "e0m3"):
        nat = FlipLinear.from_linear(lin, QuantSpec(weight=w, mode="native", native_backend="emulate"))
        assert torch.equal(nat.effective_weight(), F.fake_quant(lin.weight, w))


@pytest.mark.parametrize("mode", ["fake", "native"])
def test_map_gradient_matches_hook_formula(mode):
    lin = _linear()
    m = _mask(lin)
    layer = FlipLinear.from_linear(lin, QuantSpec(mode=mode, native_backend="emulate"), mask=m, trainable=True)
    layer.map_grad = torch.zeros(6, 3, device=DEV)
    x = torch.randn(3, 7, 192, device=DEV).bfloat16().requires_grad_()
    y = layer(x)
    dy = torch.randn_like(y)
    y.backward(dy)
    xq = F.fake_quant(x.detach(), "fourover6", "row")
    a, b = F.fake_quant(lin.weight, "e0m3"), F.fake_quant(lin.weight, "fourover6")
    g = dy.reshape(-1, 48).float().T @ xq.reshape(-1, 192).float()
    expect = maps.reduce(g * (a.float() - b.float()), 8, 64)
    assert torch.allclose(layer.map_grad, expect, rtol=1e-4, atol=1e-6)
    w = torch.where(maps.expand(m.to(DEV), 8, 64, 48), a, b)
    assert torch.allclose(x.grad.float(), (dy.bfloat16() @ w).float())


def test_map_granule_checks():
    m = torch.zeros(4, 3, dtype=torch.bool)
    assert maps.to_granule(m, "16x64", (8, 64), 64).shape == (8, 3)
    with pytest.raises(ValueError):
        maps.to_granule(torch.zeros(64, 12, dtype=torch.bool), "1x16", (8, 64), 64)


def test_static_scope_equals_tensor_scope_at_its_scale():
    x = torch.randn(3, 7, 192, device=DEV).bfloat16()
    gs = x.float().abs().amax() / F.GLOBAL_QMAX
    a, b = F.nvfp4(x, "tensor"), F.nvfp4(x, "static", gs)
    assert torch.equal(a.dequantize(), b.dequantize())
    clipped = F.nvfp4(x, "static", gs / 4).dequantize().float()   # beyond 6*448*gs saturates
    assert clipped.abs().max() <= float(gs / 4 * F.GLOBAL_QMAX) * (1 + 1e-6)


def test_modelopt_given_codes():
    """A checkpoint's NVFP4 codes are used as is (weight 'modelopt'), and as candidate B of mixfp4:
    the all-B map reproduces them exactly and A is E0M3 under the given global scale."""
    lin = _linear()
    given = F.nvfp4(lin.weight.float() * 1.01)           # some other encoding than our own quantizer's
    act_gs = torch.tensor(0.01)
    spec = QuantSpec(weight="modelopt", act="nvfp4", act_scope="static")
    layer = FlipLinear.from_linear(lin, spec, given=given, act_gscale=act_gs)
    assert torch.equal(layer.weight, given.dequantize())
    x = torch.randn(2, 5, 192, device=DEV).bfloat16()
    xq = F.nvfp4(x, "static", act_gs.to(DEV)).dequantize()
    assert torch.equal(layer(x), torch.nn.functional.linear(xq, given.dequantize()))
    mix = FlipLinear.from_linear(lin, QuantSpec(act="nvfp4", act_scope="static"), given=given,
                                 act_gscale=act_gs, trainable=True)
    assert torch.equal(mix.weight, given.dequantize())
    mix.set_map(torch.ones(6, 3, dtype=torch.bool))
    assert torch.equal(mix.weight, F.e0m3(lin.weight, "static", given.gscale).dequantize())
    with pytest.raises(ValueError):
        FlipLinear.from_linear(lin, QuantSpec(weight="modelopt"))


def test_fp8_linear_matches_per_tensor_w8a8():
    from flipquant.modelopt import FP8Linear
    torch.manual_seed(0)
    w = torch.randn(48, 192, device=DEV)
    ws, xs = w.abs().amax() / 448, torch.tensor(0.002, device=DEV)
    layer = FP8Linear((w / ws).to(torch.float8_e4m3fn), ws, xs)
    x = (torch.randn(2, 5, 192, device=DEV) * 3).bfloat16()       # some entries saturate at 448 * xs
    xq = ((x.float() / xs).clamp(-448, 448).to(torch.float8_e4m3fn).float() * xs).bfloat16()
    wq = ((w / ws).to(torch.float8_e4m3fn).float() * ws).bfloat16()
    assert torch.equal(layer(x), torch.nn.functional.linear(xq, wq))


@pytest.mark.parametrize("weight", ["mixfp4", "fourover6"])
def test_fake_weight_cache_off_is_bitwise_the_cache(weight):
    """--fake-weight-cache off decodes the weight on demand: no BF16 weight buffer, and the weight,
    the forward, the input gradient and the map gradient are bitwise those of the cached layer, also
    after set_map; set_soft (an interpolated weight) is refused."""
    lin = _linear()
    m = _mask(lin, "16x64")
    layers = {}
    for cache in (True, False):
        spec = QuantSpec(weight=weight, unit="16x64", mode="fake", fake_weight_cache=cache)
        layers[cache] = FlipLinear.from_linear(lin, spec, mask=m if weight == "mixfp4" else None,
                                               trainable=weight == "mixfp4")
    on, off = layers[True], layers[False]
    assert "weight" not in dict(off.named_buffers()) and off._candidates
    assert "fake_weight_cache" not in on.spec.as_dict() and off.spec.as_dict()["fake_weight_cache"] is False
    same = lambda a, b: torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert same(on.effective_weight(), off.effective_weight())
    x = torch.randn(3, 7, 192, device=DEV).bfloat16()
    assert same(on(x), off(x))
    if weight != "mixfp4":
        return
    grads = {}
    for cache, layer in layers.items():
        layer.map_grad = torch.zeros(m.shape, device=DEV)
        xg = x.clone().requires_grad_()
        y = layer(xg)
        y.backward(torch.ones_like(y))
        grads[cache] = (layer.map_grad.clone(), xg.grad.clone())
        layer.map_grad = None
    assert torch.equal(grads[True][0], grads[False][0]) and same(grads[True][1], grads[False][1])
    flip = ~m
    on.set_map(flip)
    off.set_map(flip)
    assert same(on.effective_weight(), off.effective_weight()) and same(on(x), off(x))
    with pytest.raises(RuntimeError):
        off.set_soft(torch.full(m.shape, 0.5))


def test_sigmoid_needs_the_weight_cache():
    from calibration import train_map_legacy as train_map
    with pytest.raises(SystemExit, match="fake-weight-cache on"):
        train_map.main(["--model", "qwen3-1.7b", "--param", "sigmoid", "--fake-weight-cache", "off",
                        "--out", "/nonexistent/never-created"])
