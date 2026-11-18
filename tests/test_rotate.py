"""Block-diagonal Hadamard rotation (flipquant/rotate.py) and its use in FlipLinear / the CLI."""
import argparse
import os

import pytest
import torch
from torch import nn

from flipquant import formats as F
from flipquant import maps, rotate
from flipquant.data import eval_windows
from flipquant.linear import FlipLinear, QuantSpec
from models.cli import add_model_args, build
from models.registry import ModelSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("n", [1, 2, 16, 64, 128])
def test_sylvester_is_orthonormal_and_symmetric(n):
    h = rotate.sylvester(n).double()
    assert torch.allclose(h @ h.T, torch.eye(n, dtype=torch.float64), atol=1e-6)
    assert torch.equal(h, h.T)


def test_block_size_rule():
    assert rotate.block_size(4480) == 16             # default 16: one scale group
    assert rotate.block_size(15680) == 16
    assert rotate.block_size(4480, 128) == 128       # 2^7 * 35
    assert rotate.block_size(15680, 128) == 64       # 2^6 * 245 (Nemotron down_proj)
    assert rotate.block_size(17408, 128) == 128      # capped
    assert rotate.block_size(4096, 256) == 256
    with pytest.raises(ValueError):
        rotate.block_size(4096, 96)


def test_hadamard_is_a_blockwise_orthogonal_map():
    r = rotate.Hadamard(320, 64)
    x = torch.randn(3, 5, 320, device=DEV)
    y = r(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5)          # orthogonal
    assert torch.allclose(r(y), x, atol=1e-5)                                  # H H = I (symmetric)
    full = torch.block_diag(*[rotate.sylvester(64)] * 5).to(DEV)
    assert torch.allclose(y, x @ full, atol=1e-5)


def test_rotated_linear_is_equivalent():
    torch.manual_seed(0)
    lin = nn.Linear(192, 40, bias=True, device=DEV)
    rl = rotate.RotatedLinear.from_linear(lin)
    assert rl.rotation.block == 16 and isinstance(rl, nn.Linear)
    assert rotate.RotatedLinear.from_linear(lin, 128).rotation.block == 64
    x = torch.randn(4, 192, device=DEV)
    assert torch.allclose(rl(x), lin(x), atol=1e-5)
    assert not torch.allclose(rl.weight, lin.weight)


@pytest.mark.parametrize("weight", ["fourover6", "nvfp4", "mixfp4"])
def test_rotated_flip_linear_fake_vs_native_and_map_gradient(weight):
    """Candidates are quantized from W H and the input is rotated before activation quantization."""
    torch.manual_seed(0)
    lin = nn.Linear(256, 48, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = (torch.randn(48, 256, device=DEV) * 0.05).bfloat16()
    rl = rotate.RotatedLinear.from_linear(lin)
    mask = torch.rand(maps.tile_shape(48, 256, "16x64")) < 0.5 if weight == "mixfp4" else None
    layers = {}
    for mode in ("fake", "native"):
        spec = QuantSpec(weight=weight, unit="16x64", mode=mode, native_backend="emulate")
        layers[mode] = FlipLinear.from_linear(rl, spec, mask=mask, trainable=weight == "mixfp4")
        layers[mode].rotation = rl.rotation
    x = torch.randn(2, 7, 256, device=DEV).bfloat16()
    expect = torch.nn.functional.linear(F.fake_quant(rl.rotation(x), "fourover6", "row"),
                                        layers["fake"].weight)
    assert torch.equal(layers["fake"](x), expect)
    rel = ((layers["fake"](x).float() - layers["native"](x).float()).norm() / expect.float().norm()).item()
    assert rel < 1e-2, rel
    if weight == "mixfp4":                       # map gradient in the rotated space
        layer = layers["fake"]
        layer.map_grad = torch.zeros(mask.shape, device=DEV)
        xg = x.clone().requires_grad_()
        y = layer(xg)
        dy = torch.randn_like(y)
        y.backward(dy)
        xq = F.fake_quant(rl.rotation(x), "fourover6", "row")
        g = dy.reshape(-1, 48).float().T @ xq.reshape(-1, 256).float()
        assert torch.allclose(layer.map_grad, maps.reduce(g * layer.delta(), 16, 64), rtol=1e-4, atol=1e-6)
        assert torch.allclose(xg.grad.float(), rl.rotation((dy.bfloat16() @ layer.weight).float()), atol=1e-2)


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=96, hidden_size=128, intermediate_size=320, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=64, max_position_embeddings=64,
                      initializer_range=0.2)
    return Qwen3ForCausalLM(cfg).to(DEV, torch.float32).eval().requires_grad_(False)


def test_rotate_model_equivalence_tiny_fp32():
    """In FP32 the rotated model is the same function (the BF16 check is the Qwen3-1.7B test below)."""
    model = _tiny_qwen3()
    ids = torch.randint(0, 96, (2, 24), device=DEV)
    ref = model(input_ids=ids).logits.float()
    rots = rotate.rotate_model(model, ModelSpec("toy", "toy"), max_block=128)
    assert rotate.summary(rots)["blocks"] == {"64": 2, "128": 12}      # down_proj: K = 320 = 64 * 5
    out = model(input_ids=ids).logits.float()
    assert ((out - ref).norm() / ref.norm()).item() < 1e-5


def _cached(hf_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(hf_id, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not (torch.cuda.is_available() and _cached("Qwen/Qwen3-1.7B")),
                    reason="needs a GPU and Qwen/Qwen3-1.7B in the HF cache")
def test_qwen3_1p7b_bf16_rotation_matches_unrotated_logits():
    """``--mode bf16 --rotate hadamard`` vs ``--mode bf16`` on WikiText-2 windows: same function up to
    BF16 rounding of ``W H`` and ``x H``."""
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    args = ap.parse_args(["--model", "qwen3-1.7b", "--mode", "bf16"])
    model, tok, _, _, _ = build(args)
    windows, _ = eval_windows(tok, "wiki", 2048)
    ids = torch.cat([w[:, :512] for w in windows[:4]]).cuda()
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.float()
        args = ap.parse_args(["--model", "qwen3-1.7b", "--mode", "bf16", "--rotate", "hadamard"])
        del model
        torch.cuda.empty_cache()
        model, _, _, _, info = build(args)
        assert info["rotate"]["modules"] == 196
        out = model(input_ids=ids, use_cache=False).logits.float()
    rel = ((out - ref).norm() / ref.norm()).item()
    nll = lambda lg: torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))
    top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"rel logits {rel:.3e}  top1 agreement {top1:.4f}  NLL {nll(ref).item():.5f} -> {nll(out).item():.5f}")
    assert rel < 3e-2 and top1 > 0.97 and abs(nll(out).item() - nll(ref).item()) < 1e-2


def test_map_rotation_must_match_kind_and_block():
    from models.cli import map_rotation_mismatch
    assert map_rotation_mismatch({}, "none", 16) is None
    assert map_rotation_mismatch({"rotate": "hadamard", "rotate_block": 16}, "hadamard", 16) is None
    assert "rotate_block=128" in map_rotation_mismatch({"rotate": "hadamard", "rotate_block": 128}, "hadamard", 16)
    assert "rotate='hadamard'" in map_rotation_mismatch({"rotate": "hadamard", "rotate_block": 16}, "none", 16)
    assert "rotate='none'" in map_rotation_mismatch({}, "hadamard", 16)


def test_tiny_fp32_equivalence_at_block_16():
    from models.registry import ModelSpec as MS
    model = _tiny_qwen3()
    ids = torch.randint(0, 96, (2, 24), device=DEV)
    ref = model(input_ids=ids).logits.float()
    rots = rotate.rotate_model(model, MS("toy", "toy"))
    assert rotate.summary(rots)["blocks"] == {"16": 14}
    assert ((model(input_ids=ids).logits.float() - ref).norm() / ref.norm()).item() < 1e-5
