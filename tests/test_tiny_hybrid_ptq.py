"""The real hybrid architectures (Nemotron-H: Mamba-2 / MLP / attention; Qwen3.8: Gated-DeltaNet /
attention) through GPTQ and the Hadamard rotation, on randomly initialized tiny versions
(models/tiny.py). Skipped without a GPU or when the config / tokenizer is not in the HF cache."""
import pytest
import torch

from flipquant import gptq, maps, rotate
from flipquant.linear import FlipLinear, QuantSpec
from models import get, load, quantizable_linears
from models.tiny import meta_linear_shapes, remap_map

MODELS = {"nemotron-nano-9b-v2": ("maps/nemotron-nano-9b-v2/flipquant_8x64.pt", 3),
          "qwen3.8-27b": ("maps/qwen3.8-27b/flipquant_8x64.pt", 2)}
HYBRID = {"nemotron-nano-9b-v2": ("mixer.in_proj", "mixer.out_proj"),
          "qwen3.8-27b": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                          "linear_attn.in_proj_b", "linear_attn.out_proj")}


def _cached(key, *files):
    try:
        from huggingface_hub import try_to_load_from_cache
        spec = get(key)
        return all(isinstance(try_to_load_from_cache(spec.hf_id, f, revision=spec.revision), str) for f in files)
    except Exception:
        return False


def _needs(key):
    ok = torch.cuda.is_available() and _cached(key, "config.json", "tokenizer_config.json")
    return pytest.mark.skipif(not ok, reason=f"{key}: no GPU, or config/tokenizer not in the HF cache")


PARAMS = [pytest.param(k, marks=_needs(k)) for k in MODELS]


def _windows(n=3, t=64):
    return [torch.randint(0, 1000, (1, t), generator=torch.Generator().manual_seed(i)) for i in range(n)]


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits.float()


@pytest.mark.parametrize("key", PARAMS)
@pytest.mark.parametrize("weight", ["fourover6", "mixfp4"])
def test_tiny_hybrid_gptq_all_layer_types(key, weight, tmp_path):
    """Sequential GPTQ walks every layer type, lowers each hybrid linear's calibration output error,
    and its codes are read identically by fake and native (emulate) layers (bitwise weights)."""
    windows = _windows()
    ids = torch.cat(windows).cuda()
    out = {}
    for mode in ("fake", "native"):
        model, tok, spec = load(key, tiny_layers=MODELS[key][1])
        tiles = None
        if weight == "mixfp4":
            tiles = remap_map(maps.load(MODELS[key][0])[0], model.flipquant_source["tiny_kept_layers"])
        names = list(quantizable_linears(model, spec))
        assert all(any(f in n for n in names) for f in HYBRID[key])
        qspec = QuantSpec(weight=weight, act="fourover6", act_scope="document", unit="8x64", mode=mode,
                          native_backend="emulate")
        cache = tmp_path / f"{mode}.pt"
        layers, info = gptq.quantize_model(model, spec, qspec, windows, maps=tiles, batch=1, cache=cache,
                                           log=lambda *a, **k: None)
        assert sorted(layers) == sorted(names)
        assert all(isinstance(model.get_submodule(n), FlipLinear) for n in names)
        ratios = {s["name"]: s["output_error_vs_rtn"] for s in info["per_linear"]}
        for f in HYBRID[key]:                                         # GPTQ did work on the hybrid layers
            assert all(r < 1 for n, r in ratios.items() if f in n), (f, ratios)
        out[mode] = _logits(model, ids)
        assert torch.isfinite(out[mode]).all()
        if mode == "fake":                                            # same codes, native layers
            codes = gptq._load(cache, gptq.cache_key(spec.key, qspec, tiles, "none", 0.01, 128, len(windows), False, "mathcode",
                                                         "quantized", gptq.windows_sha256(windows)))
            nspec = QuantSpec(weight=weight, act="fourover6", unit="8x64", mode="native", native_backend="emulate")
            for n, layer in layers.items():
                nat = FlipLinear.from_codes(layer, nspec, codes[n], mask=None if tiles is None else tiles[n])
                assert torch.equal(nat.effective_weight().view(torch.int16), layer.weight.view(torch.int16)), n
        cache.unlink()                      # GB-sized even for tiny models; pytest keeps old tmp dirs
        del model, layers
        torch.cuda.empty_cache()
    rel = ((out["fake"] - out["native"]).norm() / out["fake"].norm()).item()
    print(f"{key} {weight}: GPTQ fake vs native(emulate) logits rel {rel:.3e} (random-init model: chaotic)")


@pytest.mark.parametrize("key", PARAMS)
def test_tiny_hybrid_hadamard_bf16_equivalence(key):
    ids = torch.cat(_windows(2, 128)).cuda()
    model, tok, spec = load(key, tiny_layers=MODELS[key][1])
    ref = _logits(model, ids)
    rots = rotate.rotate_model(model, spec)
    assert len(rots) == len(quantizable_linears(model, spec))
    out = _logits(model, ids)
    rel = ((out - ref).norm() / ref.norm()).item()
    top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"{key}: bf16 rotated vs unrotated logits rel {rel:.3e}, top-1 agreement {top1:.4f}")
    assert torch.isfinite(out).all() and rel < 3e-2 and top1 > 0.9      # BF16 vs FP32 here: ~1-2%, 0.93-0.95


@pytest.mark.parametrize("key", [pytest.param(k, marks=pytest.mark.skipif(not _cached(k, "config.json"),
                                                                            reason="config not cached"))
                                 for k in MODELS])
def test_full_config_kernel_granules(key):
    """Every quantized linear of the full model (meta device) tiles at 16x64 and 256x64 and maps onto
    the sm120 (16x64) and sm100 (8x64) granules, and its Hadamard block divides in_features."""
    for n, (o, k) in meta_linear_shapes(get(key)).items():
        for u in ("16x64", "256x64"):
            m = torch.ones(maps.tile_shape(o, k, u), dtype=torch.bool)
            for granule in ((16, 64), (8, 64)):
                assert maps.to_granule(m, u, granule, o).shape == (o // granule[0], k // granule[1]), (n, u)
        assert k % rotate.block_size(k) == 0
