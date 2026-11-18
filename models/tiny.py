"""Shrunken, randomly initialized versions of the registry models (``--tiny-layers N``), for testing the
code paths on the real hybrid layer types without the real weights.

The config is read exactly as :func:`models.loader.load` reads it (same hf_id, revision and
``config_overrides``); only the number of decoder layers changes. Every width (hidden, intermediate,
heads, Mamba / Gated-DeltaNet dimensions, vocabulary) is kept, so every linear has the real model's
shape and tile layout. The per-layer type list (``layer_types``; Nemotron-H's ``layers_block_type``,
which transformers derives from ``hybrid_override_pattern``) is cut to the kept layers: the first
layer of each type, in model order, then the earliest remaining layers up to ``N``. Kept layer ``j``
is original layer ``kept[j]`` (:func:`remap_map` renames a full-model map accordingly). Weights are
random (``torch.manual_seed(seed)``, the model's own initializer); the tokenizer is the real one.

:func:`meta_linear_shapes` instantiates a config (full or tiny) on the meta device and returns the
quantizable linears' shapes, with the repo's own selection (:func:`models.loader.quantizable_linears`).
"""

from __future__ import annotations

import re

import torch
import transformers


def model_class(spec):
    if spec.loader == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    return transformers.AutoModelForCausalLM


def load_config(spec, revision=None):
    config = transformers.AutoConfig.from_pretrained(spec.hf_id, revision=revision or spec.revision)
    for k, v in spec.config_overrides.items():
        setattr(config, k, v)
    return config


def shrink_config(config, layers: int) -> list[int]:
    """Cut the decoder to ``layers`` layers in place, keeping every layer type. Returns the kept
    original layer indices."""
    text = getattr(config, "text_config", config)
    types = list(text.layer_types)
    kept = []
    for t in dict.fromkeys(types):
        kept.append(types.index(t))
    if layers < len(kept):
        raise ValueError(f"{layers} layers cannot hold the {len(kept)} layer types {list(dict.fromkeys(types))}")
    for i in range(len(types)):
        if len(kept) >= layers:
            break
        if i not in kept:
            kept.append(i)
    kept.sort()
    text.layer_types = [types[i] for i in kept]
    if getattr(text, "num_hidden_layers", None) is not None:
        text.num_hidden_layers = len(kept)
    return kept


def _instantiate(cls, config, dtype, attn):
    if cls is transformers.AutoModelForCausalLM:
        return cls.from_config(config, dtype=dtype, attn_implementation=attn)
    return cls._from_config(config, dtype=dtype, attn_implementation=attn)


def build(spec, revision, layers: int, *, dtype=torch.bfloat16, device="cuda", attn="sdpa", seed=0):
    """``(model, config, kept)``: the tiny model, randomly initialized on ``device``."""
    config = load_config(spec, revision)
    kept = shrink_config(config, layers)
    torch.manual_seed(seed)
    with torch.device(device):
        model = _instantiate(model_class(spec), config, dtype, attn)
    return model, config, kept


def meta_linear_shapes(spec, revision=None, tiny_layers: int | None = None) -> dict[str, tuple[int, int]]:
    """``{name: (out, in)}`` of the quantizable linears, from the config alone (meta device)."""
    from accelerate import init_empty_weights
    from .loader import quantizable_linears
    config = load_config(spec, revision)
    if tiny_layers:
        shrink_config(config, tiny_layers)
    with init_empty_weights():
        model = _instantiate(model_class(spec), config, None, "sdpa")
    return {n: (lin.out_features, lin.in_features) for n, lin in quantizable_linears(model, spec).items()}


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def remap_map(tiles: dict, kept: list[int]) -> dict:
    """A full-model map restricted to the kept layers, renamed to the tiny model's layer indices."""
    new = {orig: j for j, orig in enumerate(kept)}
    out = {}
    for name, m in tiles.items():
        hit = _LAYER.search(name)
        if hit and int(hit.group(1)) in new:
            out[name[:hit.start()] + f".layers.{new[int(hit.group(1))]}." + name[hit.end():]] = m
    return out


def coarsen(tiles: dict, factor: int) -> dict:
    """Row-wise OR of ``factor`` consecutive tile rows (e.g. an 8x64 map -> 16x64 with factor 2)."""
    import torch
    out = {}
    for n, m in tiles.items():
        pad = (-m.shape[0]) % factor
        if pad:
            m = torch.cat([m, torch.zeros(pad, m.shape[1], dtype=torch.bool)])
        out[n] = m.view(-1, factor, m.shape[1]).any(1)
    return out


def main(argv=None):
    """Tests and smoke runs only.

    ``python -m models.tiny remap --model M --tiny-layers N --map full.pt --out tiny.pt``: a full-model
    map restricted and renamed to the tiny model.

    ``python -m models.tiny smoke-maps --tiny-layers N --out DIR [--models a,b]``: a stand-in for the
    FlipQuant maps directory (``DIR/<model>/flipquant_{8x64,16x64,256x64}.pt``) for the tiny models: each of the
    repo's main-table maps (TM-OPT+TC) restricted to the kept layers, with its metadata (and a note)."""
    import argparse
    import json
    from pathlib import Path
    from flipquant import maps as maps_
    from .registry import get
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("remap", "smoke-maps"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--models", default="nemotron-nano-9b-v2,qwen3.8-27b")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tiny-layers", type=int, required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.command == "remap":
        spec = get(args.model)
        kept = shrink_config(load_config(spec, args.revision), args.tiny_layers)
        tiles, unit, meta = maps_.load(args.map)
        tiny = remap_map(tiles, kept)
        maps_.save(args.out, tiny, unit, dict(meta, tiny_layers=args.tiny_layers, tiny_kept_layers=kept,
                                              note="SMOKE ONLY: full-model map restricted to the tiny model"))
        Path(args.out).with_suffix(".json").write_text(json.dumps(dict(status="complete", kept=kept, **maps_.summary(tiny))))
        print(f"remapped {args.map} -> {args.out}: layers {kept}, {maps_.summary(tiny)}")
        return
    for model in args.models.split(","):
        spec = get(model)
        kept = shrink_config(load_config(spec, args.revision), args.tiny_layers)
        d = Path(args.out) / spec.key
        d.mkdir(parents=True, exist_ok=True)
        for u in ("8x64", "16x64", "256x64"):
            tiles, unit, meta = maps_.load(repo / "maps" / spec.key / f"flipquant_{u}.pt")
            tiny = remap_map(tiles, kept)
            meta = dict(meta, tiny_layers=args.tiny_layers, tiny_kept_layers=kept,
                        note="SMOKE ONLY: the main-table map restricted to the tiny model's layers")
            maps_.save(d / f"flipquant_{u}.pt", tiny, unit, meta)
            print(f"{d / f'flipquant_{u}.pt'}: {maps_.summary(tiny)}")


if __name__ == "__main__":
    main()
