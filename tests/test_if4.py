"""IF4 (Cook et al.): flipquant/if4.py against the official code, against the reference implementation, verified, and
the coarse-tile, activation and layer checks.

External references (each test that needs one is skipped without it):
- ``$FOUROVERSIX_ROOT``: a checkout of github.com/mit-han-lab/fouroversix at dadfad6901d473a734fe71e0b082e70ee993e23a
  (loaded by path, without installing the package);
- ``$REFERENCE_ROOT``: the reference implementation at ``tm-opt`` (74058af), whose ``quantize/adaptive_formats.py`` is the reference
  implementation of experiment A;
- real weights: Llama-3.1-8B (gated) from the Hugging Face cache, the 11 modules of the reference implementation's A1 check
  (layers 0, 15 and 31). ``FLIPQUANT_REAL_WEIGHTS=0`` skips them. They run on the GPU when there is one (as the
  PPL runs do) and on the CPU otherwise.
"""
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from flipquant import formats as F
from flipquant import if4
from flipquant.linear import FlipLinear, QuantSpec

FOUROVERSIX = Path(os.environ.get("FOUROVERSIX_ROOT", "/share1/saves/JaaaaaA_l/refs/fouroversix"))
REFERENCE = Path(os.environ.get("REFERENCE_ROOT", "/home/u4320956/NVFP4-RaZeR"))
OFFICIAL_COMMIT = "dadfad6901d473a734fe71e0b082e70ee993e23a"
needs_official = pytest.mark.skipif(not (FOUROVERSIX / "src/fouroversix/quantize/pytorch/reference.py").exists(),
                                    reason="fouroversix checkout not found ($FOUROVERSIX_ROOT)")
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
# 1. against the official code


def _load_official(root):
    """fouroversix's pure-PyTorch reference, loaded by path under stub packages (as the reference implementation check_formats.py)."""
    src = Path(root) / "src" / "fouroversix"
    saved = {k: v for k, v in sys.modules.items() if k == "fouroversix" or k.startswith("fouroversix.")}

    def pkg(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    def mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    try:
        pkg("fouroversix", src)
        pkg("fouroversix.kernels", src / "kernels")
        pkg("fouroversix.quantize", src / "quantize")
        pkg("fouroversix.quantize.pytorch", src / "quantize" / "pytorch")
        mod("fouroversix.kernels.constants", src / "kernels" / "constants.py")
        utils = mod("fouroversix.utils", src / "utils.py")
        mod("fouroversix.quantize.utils", src / "quantize" / "utils.py")
        for sub in ("fp4", "fp6", "int4"):
            mod(f"fouroversix.quantize.pytorch.{sub}", src / "quantize" / "pytorch" / f"{sub}.py")
        ref = mod("fouroversix.quantize.pytorch.reference", src / "quantize" / "pytorch" / "reference.py")
    finally:
        for k in [k for k in sys.modules if k == "fouroversix" or k.startswith("fouroversix.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return ref, utils


def _official(ref, U, w):
    """The official IF4 of a weight: choice, code values, scale bytes and dequantized values (select_intfloat's own
    formulas applied to its outputs)."""
    x = w.float().reshape(-1, 16)
    amax = w.float().abs().max()
    _, fake, scales = ref.intfloat_block_scaled_quantization(x, amax, fp4_format=U.DataType.if4,
                                                             scale_rule=U.ScaleRule.mse,
                                                             round_style=U.RoundStyle.nearest)
    raw = scales.view(torch.uint8)
    is_int = raw >= 128
    s = (raw & 0x7F).view(torch.float8_e4m3fn).to(torch.float32)
    fake = fake.to(torch.float32)
    d_fp = (fake * s.unsqueeze(1) * amax) / (6 * 448 * 1)
    d_int = ((fake * s.unsqueeze(1) * amax) * 0.8571428571) / (6 * 448 * 1)
    return is_int, fake, raw, torch.where(is_int.unsqueeze(1), d_int, d_fp)


def _check_official(w, ref, U):
    o_int, o_codes, o_raw, o_deq = _official(ref, U, w)
    m_int, m_codes, m_raw, m_deq = if4.codes(w)
    assert torch.equal(o_int, m_int), "per-block choice"
    assert _same(o_codes, m_codes), "codes"
    assert _same(o_raw, m_raw), "scale bytes (with the INT4 sign bit)"
    assert _same(o_deq, m_deq), "dequantized values"
    # and the installed weight is that, in BF16
    q, st = if4.quantize_weight(w, "1x16")
    assert _same(q, o_deq.reshape(w.shape).to(w.dtype))
    assert st["uniform_blocks"] == int(o_int.sum())


@needs_official
def test_official_commit():
    head = subprocess.run(["git", "-C", str(FOUROVERSIX), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    assert head.stdout.strip() == OFFICIAL_COMMIT


@needs_official
@pytest.mark.parametrize("seed", [0, 1])
def test_official_synthetic(seed):
    ref, U = _load_official(FOUROVERSIX)
    for w in _synthetic(seed):
        _check_official(w.to(DEVICE), ref, U)


@needs_official
def test_official_real_weights():
    ref, U = _load_official(FOUROVERSIX)
    for name, w in _real_weights().items():
        _check_official(w, ref, U)


# --------------------------------------------------------------------------------------------
# 2. against the reference implementation tm-opt (quantize/adaptive_formats.py)


def _reference_af():
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    from quantize import adaptive_formats
    return adaptive_formats


#: (our weight, unit, e2m1) -> the reference rule
ARMS = [("if4", u, "own", "if4") for u in if4.UNITS] + [("if4", u, "fourover6", "if4fo6") for u in if4.UNITS] + \
       [("if4_e2m1", "1x16", "own", "e2m1")]


def _check_reference(w, AF):
    for weight, unit, e2m1, rule in ARMS:
        mine, st = if4.quantize_weight(w, unit, e2m1, base=weight == "if4_e2m1")
        theirs, rst = AF.quantize(w, rule, if4.parse_unit(unit))
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
# 3. the coarse-tile rule at 1x16 is the per-block rule; tiles are uniform


@pytest.mark.parametrize("e2m1", if4.E2M1_CANDIDATES)
def test_tile_rule_at_1x16_is_block_rule(e2m1):
    for w in _synthetic(0):
        _, _, e_fp, e_int, _ = if4.candidates(w, e2m1)
        assert torch.equal(if4.select_tiles(e_fp, e_int, 1, 16), if4.select(e_fp, e_int, "1x16"))


def test_tile_rule_at_1x16_real_weights():
    for w in _real_weights().values():
        for e2m1 in if4.E2M1_CANDIDATES:
            _, _, e_fp, e_int, _ = if4.candidates(w, e2m1)
            assert torch.equal(if4.select_tiles(e_fp, e_int, 1, 16), if4.select(e_fp, e_int, "1x16"))


@pytest.mark.parametrize("unit", if4.UNITS[1:])
def test_tiles_are_uniform_and_elect_by_summed_error(unit):
    tr, tc = if4.parse_unit(unit)
    w = _synthetic(1)[0]
    d_fp, d_int, e_fp, e_int, _ = if4.candidates(w)
    uni = if4.select(e_fp, e_int, unit)
    bc = tc // 16
    for r0 in range(0, w.shape[0], tr):
        for c0 in range(0, uni.shape[1], bc):
            block = uni[r0:r0 + tr, c0:c0 + bc]
            assert block.all() or not block.any()
            want = e_int[r0:r0 + tr, c0:c0 + bc].double().sum() < e_fp[r0:r0 + tr, c0:c0 + bc].double().sum()
            if abs(float(e_int[r0:r0 + tr, c0:c0 + bc].sum() - e_fp[r0:r0 + tr, c0:c0 + bc].sum())) > 1e-12:
                assert bool(block[0, 0]) == bool(want)


def test_ties_keep_fp():
    e = torch.ones(2, 4)
    assert not if4.select(e, e.clone(), "1x16").any()
    assert not if4.select(e, e.clone(), "8x64").any()


# --------------------------------------------------------------------------------------------
# 4. activations


def test_act_hand_computed():
    """One token of 16 values, worked by hand.

    x = [7, 6, 5, 4, 3, 2, 1, 0.5, 0, ...]: amax 7, so alpha = 7 / 2688, the block scale is e4m3(7 / 6 * 2688 / 7)
    = e4m3(448) = 448, and x_b = x / (alpha * 448) = x * 6 / 7 = 6, 5.143, 4.286, 3.429, 2.571, 1.714, 0.857, 0.429.
      FP4 (E2M1, RNE): 6, 6, 4, 3, 3, 1.5, 1, 0.5, dequantized x 7/6: 7, 7, 4.67, 3.5, 3.5, 1.75, 1.17, 0.58;
            squared error 0 + 1 + 0.11 + 0.25 + 0.25 + 0.06 + 0.03 + 0.007 = 1.71.
      INT4: x_b * 1.16666666 = 7, 6, 5, 4, 3, 2, 1, 0.49999998 -> 7, 6, 5, 4, 3, 2, 1, 0; dequantized
            q * 448 * alpha * 0.8571428571 = q (to 1e-7); squared error 0.25 (the 0.5).
    INT4 has the lower error, so the block is INT4."""
    x = torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0.5] + [0] * 8], dtype=torch.float32)
    # by hand: the INT4 values times s * alpha * 6/7 with s = 448, alpha = 7 / 2688
    q_int = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0] + [0] * 8, dtype=torch.float32)
    amax = torch.tensor(7.0)
    want = ((q_int * torch.tensor(448.0) * amax) * if4.IF4_INT_EXPANSION) / 2688
    out = if4.quantize_act(x)
    assert torch.equal(out, want.unsqueeze(0))
    assert torch.allclose(out, torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0] + [0] * 8], dtype=torch.float32), atol=1e-5)
    # and an FP4 block: one outlier among small values, x = [6, 0.5, 0.5, ...] (x_b = 6, 0.5, ...: FP4 exact; INT4
    # rounds 0.5 * 7 / 6 = 0.583 -> 1, dequantized 6 / 7 = 0.857: worse)
    y = torch.tensor([[6.0] + [0.5] * 15], dtype=torch.float32)
    stats = if4.ActStats()
    out = if4.quantize_act(y, stats)
    assert torch.equal(out, y)
    assert stats.as_dict()["uniform_blocks"] == 0


def test_act_per_token_equals_per_row_tensor_rule():
    """Each token is quantized as its own tensor: row r of the result is IF4 of x[r:r+1] at 1x16."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(3, 5, 64, generator=g) * torch.logspace(-3, 1, 5).reshape(1, 5, 1)
    x[0, 1] = 0.0                                                # an all-zero token -> 0
    x[2, 3, :16] *= 1e-6                                         # tiny blocks in one token
    x = x.bfloat16()
    stats = if4.ActStats()
    out = if4.quantize_act(x, stats)
    assert out.dtype == x.dtype and out.shape == x.shape
    flat, oflat = x.reshape(-1, 64), out.reshape(-1, 64)
    n_int = 0
    for r in range(flat.shape[0]):
        q, st = if4.quantize_weight(flat[r:r + 1], "1x16")
        assert _same(oflat[r:r + 1], q), r
        n_int += st["uniform_blocks"]
    assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1]))
    d = stats.as_dict()
    assert d["blocks"] == flat.numel() // 16 and d["uniform_blocks"] == n_int
    assert d["mixing"]["16x64"]["tiles"] == 1


def test_act_chunking_is_exact(monkeypatch):
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(37, 128, generator=g) * 0.1).bfloat16()
    full = if4.quantize_act(x)
    monkeypatch.setattr(if4, "ACT_CHUNK_ROWS", 5)
    assert _same(if4.quantize_act(x), full)


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
    mine = if4.IF4Linear(ref.weight.clone(), lin.bias, act, scope)
    with torch.no_grad():
        assert _same(mine(x), ref(x))


def test_install_report_and_layer():
    g = torch.Generator().manual_seed(6)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16, bias=False)).bfloat16()
    with torch.no_grad():
        for m in model:
            m.weight.copy_(torch.randn_like(m.weight.float(), dtype=torch.float32) * 0.1)
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    want = {n: if4.quantize_weight(m.weight.detach(), "1x16")[0] for n, m in linears.items()}

    def set_module(root, name, new):
        setattr(root, name, new)
    layers, rep = if4.install(model, linears, unit="1x16", e2m1="own", base=False, act="own", set_module=set_module)
    for n, layer in layers.items():
        assert _same(layer.weight, want[n])
    x = (torch.randn(4, 64, generator=g)).bfloat16()
    with torch.no_grad():
        model(x)
    d = rep.as_dict()
    assert d["method"]["unit"] == "1x16" and d["method"]["activation"] == "own"
    assert 0 <= d["weight"]["uniform_fraction"] <= 1 and d["weight"]["mixing"]["8x64"]["all"]["tiles"] > 0
    assert d["activation"]["by_projection"]["all"]["blocks"] == 4 * (64 + 32) // 16
