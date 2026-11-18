"""FourOverSix's tensor-scale constant: ``formats.fourover6(..., tensor_qmax)``, ``QuantSpec.fo6_tensor_qmax``,
``models.sm120.install(..., fo6_tensor_qmax)`` and ``--fo6-tensor-qmax`` (NVFP4-RaZeR paper-eval part R).

* The default is unchanged: ``tensor_qmax=448`` given explicitly equals the default bit for bit for every scope
  (the default's agreement with the vendored reference quantizers is test_formats.py's, unchanged).
* 256 is the original FourOverSix (arXiv:2512.02010, Sec. 3): ``fourover6(..., tensor_qmax=256)`` equals
  :func:`original_fourover6`, written here from the paper's definition, for the tensor and row scopes; with 256 no
  block scale exceeds 384 and no 4-candidate is clamped, while with 448 a block whose max exceeds 2/3 of the amax has
  its 4-candidate clamped at 448.
* Fake mode: a FourOverSix ``FlipLinear`` with ``fo6_tensor_qmax=256`` quantizes its weight and its per-token
  activations with 256; anything but the FourOverSix baseline is refused.
* On an sm_120 GPU with the kernel builds ($SM120_BUILD_DIR):
  - NativeLinear's unfused branch with the quantizer's reciprocal left at fp32(1/2688) (``TensorQmaxNativeLinear``)
    equals the fused NativeLinear bit for bit, so the 256 path differs from the default only by the constant;
  - with fp32(1/1536) the quantizer's scale bytes and per-token global scales equal ``fourover6(x, "row", 256)``'s bit
    for bit, and its codes decode (fake order) to the same values;
  - an install with ``fo6_tensor_qmax=256`` packs every weight from those codes (pack_codes checks the decode bit for
    bit) and its layers' outputs match fake mode at 256, not at 448 (by value: the GEMM's accumulation order differs).
"""
import pytest
import torch
from torch import nn

from flipquant import formats as F
from flipquant.linear import FlipLinear, QuantSpec
from models import sm120 as NS
from models.registry import ModelSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SCOPES = ("tensor", "row", "document", "static")


def sample(shape, seed=0, device=DEV):
    """Heavy-tailed BF16 values: some blocks' maxima exceed 2/3 of the amax, others are far below it."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * torch.exp(1.5 * torch.randn(shape, generator=g)).clamp(max=50.0)
    return x.to(torch.bfloat16).to(device)


def original_fourover6(x, scope, tensor_qmax=256.0):
    """FourOverSix as arXiv:2512.02010 (Sec. 3) defines it, independently of flipquant.formats: the tensor scale
    ``ts = amax / (6 * tensor_qmax)`` (one per tensor, or per row); per 16-block of ``v = x / ts`` with max ``m``, the
    candidate E4M3 scales ``e4m3(m / 6)`` and ``e4m3(m / 4)``; each candidate rounds ``v / s`` to the nearest E2M1
    level (a midpoint goes to the smaller magnitude); the block takes 4 iff its squared error is strictly lower;
    the value is ``(code * s) * ts`` in BF16. Returns the values and the chosen scales."""
    k = x.shape[-1]
    b = x.float().reshape(-1, k // 16, 16)
    amax = b.abs().amax() if scope == "tensor" else b.abs().amax((1, 2), keepdim=True)
    ts = (amax / (6.0 * tensor_qmax)).clamp_min(torch.finfo(torch.float32).tiny)
    v = b / ts
    m = v.abs().amax(-1, keepdim=True)
    levels = v.new_tensor(LEVELS)
    mids = (levels[:-1] + levels[1:]) / 2
    best = None
    for q in (6.0, 4.0):
        s = (m / q).clamp(2.0 ** -9, 448.0).to(torch.float8_e4m3fn).float()
        code = levels[torch.bucketize((v / s).abs().contiguous(), mids, right=False)] * torch.where(v < 0, -1.0, 1.0)
        err = (code * s - v).square().sum(-1, keepdim=True)
        if best is None:
            best = (code, s, err)
        else:
            take4 = err < best[2]
            best = (torch.where(take4, code, best[0]), torch.where(take4, s, best[1]), None)
    return ((best[0] * best[1]) * ts).to(torch.bfloat16).reshape(x.shape), best[1].reshape(*x.shape[:-1], k // 16)


def _same(a: F.FP4Codes, b: F.FP4Codes):
    return (torch.equal(a.codes, b.codes) and torch.equal(a.scale.view(torch.int32), b.scale.view(torch.int32))
            and torch.equal(a.gscale.float().view(torch.int32), b.gscale.float().view(torch.int32)))


@pytest.mark.parametrize("scope", SCOPES)
def test_default_is_448(scope):
    x = sample((3, 40, 256), seed=1)
    g = None if scope != "static" else x.float().abs().amax() / 2000.0
    assert _same(F.fourover6(x, scope, g), F.fourover6(x, scope, g, tensor_qmax=448.0))
    assert F.global_qmax() == F.global_qmax(448.0) == F.GLOBAL_QMAX == 2688.0
    assert F.global_qmax(256.0) == 1536.0
    for bad in (0.0, -1.0, 449.0):
        with pytest.raises(ValueError):
            F.global_qmax(bad)


@pytest.mark.parametrize("scope", ["tensor", "row"])
def test_256_is_the_original_fourover6(scope):
    for seed in range(3):
        x = sample((64, 512), seed=seed)
        got = F.fourover6(x, scope, tensor_qmax=256.0)
        want, scales = original_fourover6(x, scope)
        assert torch.equal(got.dequantize(torch.bfloat16), want)
        assert torch.equal(got.scale, scales)
        # and it is not the 448 quantizer
        assert not torch.equal(F.fourover6(x, scope).dequantize(torch.bfloat16), want)


def test_every_block_can_take_four_with_256():
    x = sample((128, 1024), seed=5)
    for scope in ("tensor", "row"):
        b = x.float().reshape(-1, 1024 // 16, 16)
        amax = b.abs().amax() if scope == "tensor" else b.abs().amax((1, 2), keepdim=True)
        for q, clamped_any in ((256.0, False), (448.0, True)):
            peak = (b / (amax / F.global_qmax(q))).abs().amax(-1)
            clamped = peak / 4.0 > F.E4M3_MAX
            assert bool(clamped.any()) == clamped_any, (scope, q)
            if q == 448.0:     # exactly the blocks whose max exceeds 2/3 of the amax (up to float rounding at 2/3)
                big = b.abs().amax(-1) > (2.0 / 3.0) * amax.reshape(-1, 1)[: b.shape[0]] if scope == "row" else \
                    b.abs().amax(-1) > (2.0 / 3.0) * amax
                assert torch.equal(clamped, big)
        assert float(F.fourover6(x, scope, tensor_qmax=256.0).scale.max()) <= 384.0


def test_fake_layer_uses_256_for_weight_and_activations():
    torch.manual_seed(0)
    lin = nn.Linear(256, 96, bias=False).to(DEV).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(sample((96, 256), seed=7))
    x = sample((2, 9, 256), seed=8)
    spec = QuantSpec(weight="fourover6", act="fourover6", act_scope="row", mode="fake", fo6_tensor_qmax=256.0)
    layer = FlipLinear.from_linear(lin, spec)
    want = torch.nn.functional.linear(F.fourover6(x, "row", tensor_qmax=256.0).dequantize(torch.bfloat16),
                                      F.fourover6(lin.weight, tensor_qmax=256.0).dequantize(torch.bfloat16))
    with torch.no_grad():
        assert torch.equal(layer(x), want)
    assert spec.as_dict()["fo6_tensor_qmax"] == 256.0
    assert "fo6_tensor_qmax" not in QuantSpec(weight="fourover6").as_dict()      # the default leaves reports as they were


def test_refused_outside_the_fourover6_baseline():
    for weight, act in (("mixfp4", "fourover6"), ("nvfp4", "nvfp4"), ("fourover6", "nvfp4"), ("e0m3", "fourover6")):
        with pytest.raises(ValueError):
            QuantSpec(weight=weight, act=act, fo6_tensor_qmax=256.0)
    with pytest.raises(ValueError):
        QuantSpec(weight="fourover6", act="fourover6", fo6_tensor_qmax=500.0)
    spec = ModelSpec("tiny-qwen3", "tiny-qwen3")
    for kw in (dict(weight="mixfp4", tiles={}, unit="8x64"), dict(weight="nvfp4", act="nvfp4"),
               dict(weight="fourover6", activation_quantizer="nvfp4_rows")):
        with pytest.raises(ValueError):
            NS.install(None, spec, fo6_tensor_qmax=256.0, **{"act": "fourover6", **kw})


# ------------------------------------------------------------------------------------------------ sm_120 GPU

def _sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an sm_120 GPU")
    try:
        NS.resolve_kernel("auto", "fourover6", None, None)
    except (NS.package().lib.LibraryError, NS.KernelSetError) as e:
        pytest.skip(f"the FourOverSix route of 'auto' is not built in {NS.build_dir()}: {e}")


def _layer(n, k, seed, bias=True):
    lin = nn.Linear(k, n, bias=bias).cuda().to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(sample((n, k), seed=seed, device="cuda"))
        if bias:
            lin.bias.copy_(sample((n,), seed=seed + 1, device="cuda") * 0.1)
    return lin


SHAPES = [(256, 512), (512, 256), (1024, 2048), (2048, 1024)]
TOKENS = (1, 5, 33, 128, 300)


@pytest.mark.parametrize("n,k", SHAPES)
def test_unfused_route_at_448_equals_fused(n, k):
    _sm120()
    R = NS.package()
    kern, _, _ = NS.resolve_kernel("auto", "fourover6", None, None)
    lin = _layer(n, k, seed=n + k)
    pw = R.artifact.pack_module("t", lin.weight.data, lin.bias, "four_over_six", None, None,
                                expected=F.fake_quant(lin.weight.data, "fourover6"))
    fused = R.linear.NativeLinear(pw, kern, "four_over_six_rows", name="t")
    unfused = NS.tensor_qmax_native()(pw, kern, "four_over_six_rows", name="t", inv_qmax=R.quant_act.INV_2688)
    assert fused.fused and not unfused.fused and NS.inv_qmax(448.0) == R.quant_act.INV_2688
    with torch.no_grad():
        for t in TOKENS:
            x = sample((t, k), seed=t, device="cuda")
            a, b = fused(x), unfused(x)
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (n, k, t)
    assert R.quant_act.INV_2688 == NS.inv_qmax(448.0)          # the context restored the vendored constant


@pytest.mark.parametrize("t,k", [(1, 256), (37, 512), (130, 2048), (300, 5120)])
def test_native_quantizer_at_256(t, k):
    _sm120()
    R = NS.package()
    QA, N, L = R.quant_act, R.numerics, R.lib
    x = sample((t, k), seed=t + k, device="cuda")
    with NS.quantizer_inv_qmax(NS.inv_qmax(256.0)):
        packed, sf, gs = QA.quantize(x, "four_over_six_rows")
    want = F.fourover6(x, "row", tensor_qmax=256.0)
    rows = torch.arange(t, device="cuda")[:, None]
    kb = torch.arange(k // 16, device="cuda")[None, :]
    sbytes = sf[L.sf_offset_formula(rows, kb, k)]
    assert torch.equal(sbytes, want.scale_bytes())
    assert torch.equal(gs.view(torch.int32), want.row_gscale().view(torch.int32))
    got = N.decode_fake_order(N.unpack_nibbles(packed), sbytes, gs[:, None])
    assert torch.equal(got, want.dequantize(torch.bfloat16))
    assert torch.equal(got, original_fourover6(x, "row")[0])
    assert float(want.scale.max()) <= 384.0


def test_install_256_matches_fake_at_256():
    _sm120()
    from transformers import Qwen3Config, Qwen3ForCausalLM
    spec = ModelSpec("tiny-qwen3", "tiny-qwen3")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=64, max_position_embeddings=512, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).cuda().eval().requires_grad_(False)
    weights = {n: m.weight.detach().clone() for n, m in NS.quantizable_linears(model, spec).items()}
    rep = NS.install(model, spec, weight="fourover6", act="fourover6", fo6_tensor_qmax=256.0)
    assert rep["modules"] == 14 and rep["fo6_tensor_qmax"]["tensor_qmax"] == 256.0
    cls = NS.tensor_qmax_native()
    native = NS.native_modules(model)
    assert len(native) == 14 and all(isinstance(m, cls) and m.inv_qmax == NS.inv_qmax(256.0) for m in native.values())
    errs = {}
    with torch.no_grad():
        for name, nl in native.items():
            x = sample((40, nl.in_features), seed=len(name), device="cuda")
            y = nl(x).float()
            for q in (256.0, 448.0):
                ref = torch.nn.functional.linear(F.fourover6(x, "row", tensor_qmax=q).dequantize(torch.bfloat16),
                                                 F.fourover6(weights[name], tensor_qmax=q).dequantize(torch.bfloat16))
                errs[(name, q)] = float((y - ref.float()).norm() / ref.float().norm())
            assert errs[(name, 256.0)] < 1e-2, errs
            assert errs[(name, 448.0)] > 3 * errs[(name, 256.0)], errs
