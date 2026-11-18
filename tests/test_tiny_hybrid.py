"""The real hybrid architectures (Nemotron-H: Mamba-2 / MLP / attention; Qwen3.8: Gated-DeltaNet /
attention) through the map-ablation code: full-config map checks on the meta device, and the three
baseline maps on randomly initialized tiny versions (models/tiny.py). Skipped when the config or
tokenizer is not in the HF cache (no network needed), and the tiny-model tests without a GPU."""
import pytest
import torch

from calibration.baseline_maps import (act_scores, check_match, input_gram_blocks, oneshot_gradients, random_map,
                                       top_k)
from calibration.train_map_legacy import teacher_logprobs
from flipquant import maps
from flipquant.linear import QuantSpec
from models import get, load, quantize, quantizable_linears
from models.tiny import meta_linear_shapes, remap_map

MODELS = {"nemotron-nano-9b-v2": ("maps/nemotron-nano-9b-v2/flipquant_8x64.pt", 3),
          "qwen3.8-27b": ("maps/qwen3.8-27b/flipquant_8x64.pt", 2)}
#: Layer types whose linears must be reached (module-name fragments).
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


def _needs(key, gpu=False):
    ok = _cached(key, "config.json") and (not gpu or (torch.cuda.is_available() and
                                                       _cached(key, "tokenizer_config.json")))
    return pytest.mark.skipif(not ok, reason=f"{key}: config/tokenizer not in the HF cache" +
                              (" or no GPU" if gpu else ""))


@pytest.mark.parametrize("key", [pytest.param(k, marks=_needs(k)) for k in MODELS])
def test_full_config_matches_ours_map_and_granules(key):
    """Full model on the meta device: the 'ours' 8x64 map covers exactly the quantized linears with
    the right tile counts; 16x64 / 256x64 tile shapes and kernel granules work for every layer."""
    shapes = meta_linear_shapes(get(key))
    tiles, unit, _ = maps.load(MODELS[key][0])
    assert unit == "8x64"
    check_match(tiles, shapes, unit)
    for n, (o, k) in shapes.items():
        assert tiles[n].numel() == maps.tile_shape(o, k, "8x64")[0] * maps.tile_shape(o, k, "8x64")[1]
        for u in ("16x64", "256x64"):
            ts = maps.tile_shape(o, k, u)
            m = torch.zeros(ts, dtype=torch.bool)
            m[-1, -1] = True                                       # the ragged corner tile, if any
            for granule in ((16, 64), (8, 64)):                    # sm120, sm100
                fb = maps.to_granule(m, u, granule, o)
                assert fb.shape == (o // granule[0], k // granule[1]) and fb[-1, -1] == 0, (n, u, granule)


def _tiny(key):
    model, tok, spec = load(key, tiny_layers=MODELS[key][1])
    kept = model.flipquant_source["tiny_kept_layers"]
    tiles, _, _ = maps.load(MODELS[key][0])
    match = remap_map(tiles, kept)
    lins = quantizable_linears(model, spec)
    check_match(match, {n: (l.out_features, l.in_features) for n, l in lins.items()}, "8x64")
    windows = [torch.randint(0, 1000, (1, 64), generator=torch.Generator().manual_seed(i)) for i in range(2)]
    return model, spec, match, lins, windows


@pytest.mark.parametrize("key", [pytest.param(k, marks=_needs(k, gpu=True)) for k in MODELS])
def test_tiny_hybrid_baseline_maps(key):
    model, spec, match, lins, windows = _tiny(key)
    assert all(any(f in n for n in lins) for f in HYBRID[key])
    k = {n: int(m.sum()) for n, m in match.items()}
    for n, m in random_map(match, 0).items():
        assert int(m.sum()) == k[n]
    grams = input_gram_blocks(model, lins, windows, "cuda", 1, 64)
    for n, lin in lins.items():
        assert float(grams[n].diagonal(dim1=1, dim2=2).sum()) > 0, n          # the hook saw this input
        s = act_scores(lin, grams[n], "8x64")
        assert torch.isfinite(s).all() and int(top_k(s, k[n]).sum()) == k[n]
    del grams
    teacher = teacher_logprobs(model, windows, "cuda")
    qspec = QuantSpec(weight="mixfp4", act="fourover6", act_scope="row", unit="8x64", mode="native",
                      native_backend="emulate")
    layers = {}
    for n in match:
        layers.update(quantize(model, spec, qspec, trainable=True, names=[n]))
    grads, kls = oneshot_gradients(model, layers, windows, teacher, "cuda", 1)
    assert all(torch.isfinite(torch.tensor(kls)))
    for n, g in grads.items():
        assert torch.isfinite(g).all() and g.abs().sum() > 0, n                # map_grad reached this linear
        assert int(top_k(-g, k[n]).sum()) == k[n]
