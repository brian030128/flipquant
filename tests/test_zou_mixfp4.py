"""MixFP4 of Zou et al.: flipquant/zou_mixfp4.py against the reference implementation, verified (the paper has no
code), its E1M2 candidate against the repo's E0M3, and the coarse-tile, activation and layer checks.

External references (each test that needs one is skipped without it):
- ``$REFERENCE_ROOT``: the reference implementation at ``tm-opt`` (74058af), whose ``quantize/adaptive_formats.py`` is the reference
  implementation of experiment A;
- real weights: Llama-3.1-8B (gated) from the Hugging Face cache, the 11 modules of the reference implementation's A1 check
  (layers 0, 15 and 31). ``FLIPQUANT_REAL_WEIGHTS=0`` skips them. They run on the GPU when there is one (as the
  PPL runs do) and on the CPU otherwise.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

from flipquant import formats as F
from flipquant import zou_mixfp4 as zou
from flipquant.linear import FlipLinear, QuantSpec

REFERENCE = Path(os.environ.get("REFERENCE_ROOT", "/home/u4320956/NVFP4-RaZeR"))
needs_reference = pytest.mark.skipif(not (REFERENCE / "quantize" / "adaptive_formats.py").exists(),
                                 reason="the reference implementation tm-opt not found ($REFERENCE_ROOT)")
LLAMA = ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
A1_MODULES = ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj",
              "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
              "model.layers.0.mlp.down_proj", "model.layers.15.self_attn.q_proj", "model.layers.15.mlp.down_proj",
              "model.layers.31.self_attn.v_proj", "model.layers.31.mlp.down_proj")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                    b.contiguous().view(torch.uint8))


def _synthetic(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(40, 256, generator=g) * 0.02           # 40 rows: a partial last tile at 16x64 / 256x64
    w[3, 5] = 0.9                                          # outlier -> tiny block scales, underflow candidates
    w[7, :16] = 0.0                                        # all-zero block
    w[9] = torch.randn(256, generator=g).sign() * 0.01     # tie-heavy row
    w[11, 16:32] = 1e-7                                    # a nonzero block whose scale underflows
    heavy = torch.distributions.StudentT(2.0).sample((24, 128)) * 0.01
    return [w.bfloat16(), heavy.bfloat16()]


_REAL = {}


def _real_weights():
    if os.environ.get("FLIPQUANT_REAL_WEIGHTS", "1") == "0":
        pytest.skip("FLIPQUANT_REAL_WEIGHTS=0")
    if not _REAL:
        try:
            import json

            from huggingface_hub import snapshot_download
            from safetensors import safe_open
            snap = Path(snapshot_download(LLAMA[0], revision=LLAMA[1], local_files_only=True,
                                          allow_patterns=["*.json", "*.safetensors"]))
            index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Llama-3.1-8B not in the local Hugging Face cache: {type(e).__name__}")
        for name in A1_MODULES:
            with safe_open(str(snap / index[name + ".weight"]), "pt") as f:
                _REAL[name] = f.get_tensor(name + ".weight")
    return {n: w.to(DEVICE) for n, w in _REAL.items()}


# --------------------------------------------------------------------------------------------
# 1. against the reference implementation tm-opt (quantize/adaptive_formats.py)


def _reference_af():
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    from quantize import adaptive_formats
    return adaptive_formats


#: (our weight, unit, e2m1) -> the reference rule
ARMS = [("zou_mixfp4", u, "own", "zou") for u in zou.UNITS] + \
       [("zou_mixfp4", u, "fourover6", "zoufo6") for u in zou.UNITS] + [("zou_mixfp4_e2m1", "1x16", "own", "e2m1zou")]


def _check_reference(w, AF):
    for weight, unit, e2m1, rule in ARMS:
        mine, st = zou.quantize_weight(w, unit, e2m1, base=weight == "zou_mixfp4_e2m1")
        theirs, rst = AF.quantize(w, rule, zou.parse_unit(unit))
        assert _same(mine, theirs), (weight, unit, e2m1)
        for k in ("blocks", "uniform_blocks", "zero_scale_blocks"):
            assert st[k] == rst[k], (weight, unit, e2m1, k)
        assert st["e2m1_out_of_grid_elements"] == 0 or e2m1 == "own"
        assert st["sq_error"] == rst["sq_error"], (weight, unit, e2m1)
        assert st.get("mixing") == rst.get("mixing"), (weight, unit, e2m1)


@needs_reference
@pytest.mark.parametrize("seed", [0, 1])
def test_reference_synthetic(seed):
    AF = _reference_af()
    for w in _synthetic(seed):
        _check_reference(w.to(DEVICE), AF)


@needs_reference
def test_reference_real_weights():
    AF = _reference_af()
    for w in _real_weights().values():
        _check_reference(w, AF)


# --------------------------------------------------------------------------------------------
# 2. Zou's E1M2 candidate is the repo's E0M3 (flipquant.formats.e0m3), element for element


def _check_e1m2(w):
    _, d_int, _, _, _ = zou.candidates(w)
    assert _same(d_int.to(w.dtype), F.fake_quant(w, "e0m3"))


@pytest.mark.parametrize("seed", [0, 1])
def test_e1m2_equals_repo_e0m3(seed):
    for w in _synthetic(seed):
        _check_e1m2(w.to(DEVICE))


def test_e1m2_equals_repo_e0m3_real_weights():
    for w in _real_weights().values():
        _check_e1m2(w)


# --------------------------------------------------------------------------------------------
# 3. the coarse-tile rule at 1x16 is the per-block rule; tiles are uniform


@pytest.mark.parametrize("e2m1", zou.E2M1_CANDIDATES)
def test_tile_rule_at_1x16_is_block_rule(e2m1):
    for w in _synthetic(0):
        _, _, e_fp, e_int, _ = zou.candidates(w, e2m1)
        assert torch.equal(zou.select_tiles(e_fp, e_int, 1, 16), zou.select(e_fp, e_int, "1x16"))


def test_tile_rule_at_1x16_real_weights():
    for w in _real_weights().values():
        for e2m1 in zou.E2M1_CANDIDATES:
            _, _, e_fp, e_int, _ = zou.candidates(w, e2m1)
            assert torch.equal(zou.select_tiles(e_fp, e_int, 1, 16), zou.select(e_fp, e_int, "1x16"))


@pytest.mark.parametrize("unit", zou.UNITS[1:])
def test_tiles_are_uniform_and_elect_by_summed_error(unit):
    tr, tc = zou.parse_unit(unit)
    w = _synthetic(1)[0]
    d_fp, d_int, e_fp, e_int, _ = zou.candidates(w)
    uni = zou.select(e_fp, e_int, unit)
    bc = tc // 16
    for r0 in range(0, w.shape[0], tr):
        for c0 in range(0, uni.shape[1], bc):
            block = uni[r0:r0 + tr, c0:c0 + bc]
            assert block.all() or not block.any()
            want = e_int[r0:r0 + tr, c0:c0 + bc].double().sum() <= e_fp[r0:r0 + tr, c0:c0 + bc].double().sum()
            if abs(float(e_int[r0:r0 + tr, c0:c0 + bc].sum() - e_fp[r0:r0 + tr, c0:c0 + bc].sum())) > 1e-12:
                assert bool(block[0, 0]) == bool(want)


def test_ties_go_to_e1m2():
    e = torch.ones(2, 4)
    assert zou.select(e, e.clone(), "1x16").all()
    assert zou.select(e, e.clone(), "8x64").all()
    # a block both candidates represent exactly ({0, +-bmax}: code 6 at scale bmax / 6, code 7 at scale bmax / 7),
    # so the errors tie at 0 and the block goes to E1M2 (Algorithm 1's else)
    w = torch.tensor([[6.0, -6.0] + [0.0] * 14], dtype=torch.bfloat16)
    uni, _, _, deq = zou.codes(w)
    assert bool(uni[0]) and torch.equal(deq.reshape(w.shape).to(w.dtype), w)


# --------------------------------------------------------------------------------------------
# 4. activations


def test_act_hand_computed():
    """One token of 16 values, worked by hand (Algorithm 1 with s32 of the token).

    x = [7, 6, 5, 4, 3, 2, 1, 0.5, 0, ...]: amax 7, s32 = 7 / 2688, x8 = x * 384 = 2688, 2304, ..., 192; bmax8 = 2688.
      E2M1: s = e4m3(2688 / 6) = e4m3(448) = 448, x8 / s = 6, 5.143, 4.286, 3.429, 2.571, 1.714, 0.857, 0.429
            -> 6, 6, 4, 3, 3, 1.5, 1, 0.5; error (x8 units) 448^2 * (0 + 0.73 + 0.08 + 0.18 + 0.18 + 0.05 + 0.02 + 0.005)
            = 448^2 * 1.26.
      E1M2: s = e4m3(2688 / 7) = e4m3(384) = 384, x8 / s = x = 7, 6, 5, 4, 3, 2, 1, 0.5 -> 7, ..., 1, 0 (half to even);
            error 384^2 * 0.25, lower. So the block is E1M2: q * 384 * s32 = q * 384 * 7 / 2688 = q."""
    x = torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0.5] + [0] * 8], dtype=torch.float32)
    q = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0] + [0] * 8, dtype=torch.float32)
    want = (q * torch.tensor(384.0)) * (torch.tensor(7.0) / 2688)
    stats = zou.ActStats()
    out = zou.quantize_act(x, stats)
    assert torch.equal(out, want.unsqueeze(0))
    assert torch.allclose(out, q.unsqueeze(0), atol=1e-6)
    assert stats.as_dict()["uniform_blocks"] == 1
    # an E2M1 block: one outlier among small values, y = [6, 0.5, ...]: x8 = y * 448; E2M1 at s = 448 is exact; E1M2 at
    # s = e4m3(384) rounds 0.5 * 448 / 384 = 0.583 -> 1 (worse)
    y = torch.tensor([[6.0] + [0.5] * 15], dtype=torch.float32)
    stats = zou.ActStats()
    out = zou.quantize_act(y, stats)
    assert torch.equal(out, (y * torch.tensor(448.0)) * (torch.tensor(6.0) / 2688))   # q * s * s32, s32 inexact
    assert torch.allclose(out, y, rtol=1e-6)
    assert stats.as_dict()["uniform_blocks"] == 0
    # a zero token is zero
    assert torch.equal(zou.quantize_act(torch.zeros(2, 32)), torch.zeros(2, 32))


def test_act_per_token_equals_per_row_tensor_rule():
    """Each token is quantized as its own tensor: row r of the result is Zou's rule on x[r:r+1] at 1x16."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(3, 5, 64, generator=g) * torch.logspace(-3, 1, 5).reshape(1, 5, 1)
    x[0, 1] = 0.0                                                # an all-zero token -> 0
    x[2, 3, :16] *= 1e-6                                         # tiny blocks in one token
    x = x.bfloat16()
    stats = zou.ActStats()
    out = zou.quantize_act(x, stats)
    assert out.dtype == x.dtype and out.shape == x.shape
    flat, oflat = x.reshape(-1, 64), out.reshape(-1, 64)
    n_int = 0
    for r in range(flat.shape[0]):
        q, st = zou.quantize_weight(flat[r:r + 1], "1x16")
        assert _same(oflat[r:r + 1], q), r
        n_int += st["uniform_blocks"]
    assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1]))
    d = stats.as_dict()
    assert d["blocks"] == flat.numel() // 16 and d["uniform_blocks"] == n_int
    assert d["mixing"]["16x64"]["tiles"] == 1


def test_act_chunking_is_exact(monkeypatch):
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(37, 128, generator=g) * 0.1).bfloat16()
    full = zou.quantize_act(x)
    monkeypatch.setattr(zou, "ACT_CHUNK_ROWS", 5)
    assert _same(zou.quantize_act(x), full)


@pytest.mark.parametrize("act", ["nvfp4", "fourover6"])
@pytest.mark.parametrize("scope", ["row", "document"])
def test_act_baselines_equal_fliplinear(act, scope):
    """--act-method nvfp4 / fourover6 = FlipLinear fake with --act nvfp4 / fourover6 (--act-scope row), bitwise."""
    g = torch.Generator().manual_seed(5)
    lin = torch.nn.Linear(128, 48, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(48, 128, generator=g) * 0.05)
        lin.bias.copy_(torch.randn(48, generator=g))
    lin = lin.bfloat16().to(DEVICE)
    x = (torch.randn(2, 9, 128, generator=g) * 3).bfloat16().to(DEVICE)
    ref = FlipLinear.from_linear(lin, QuantSpec(weight="fourover6", act=act, act_scope=scope, mode="fake"))
    mine = zou.ZouMixFP4Linear(ref.weight.clone(), lin.bias, act, scope)
    with torch.no_grad():
        assert _same(mine(x), ref(x))


def test_install_report_and_layer():
    g = torch.Generator().manual_seed(6)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16, bias=False)).bfloat16()
    with torch.no_grad():
        for m in model:
            m.weight.copy_(torch.randn_like(m.weight.float(), dtype=torch.float32) * 0.1)
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    want = {n: zou.quantize_weight(m.weight.detach(), "1x16")[0] for n, m in linears.items()}

    def set_module(root, name, new):
        setattr(root, name, new)
    layers, rep = zou.install(model, linears, unit="1x16", e2m1="own", base=False, act="own", set_module=set_module)
    for n, layer in layers.items():
        assert _same(layer.weight, want[n])
    x = (torch.randn(4, 64, generator=g)).bfloat16()
    with torch.no_grad():
        model(x)
    d = rep.as_dict()
    assert d["method"]["unit"] == "1x16" and d["method"]["activation"] == "own"
    assert 0 <= d["weight"]["uniform_fraction"] <= 1 and d["weight"]["mixing"]["8x64"]["all"]["tiles"] > 0
    assert d["activation"]["by_projection"]["all"]["blocks"] == 4 * (64 + 32) // 16
