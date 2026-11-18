"""Load a registered model and put our kernels into it.

    model, tok, spec = load("qwen3-8b")
    layers = quantize(model, spec, QuantSpec(weight="mixfp4", mode="native"), maps=tiles)

``quantize`` replaces each selected ``nn.Linear`` in place with a :class:`flipquant.linear.FlipLinear`
(fake quant or native kernel), so every forward -- PPL, calibration, generation -- runs through it.
"""

from __future__ import annotations

import hashlib
import logging

import torch
import transformers
from packaging.version import Version
from torch import nn

from flipquant.linear import FlipLinear, QuantSpec
from flipquant import maps as maps_

from .registry import ModelSpec, get

log = logging.getLogger(__name__)


def load(key: str, *, device_map="cuda", dtype=torch.bfloat16, revision: str | None = None, attn: str = "sdpa",
         tiny_layers: int | None = None):
    """Returns ``(model, tokenizer, spec)``. The model is in eval mode with gradients off.

    ``tiny_layers`` (tests only): the same architecture cut to that many decoder layers, randomly
    initialized, with the real tokenizer (models/tiny.py)."""
    spec = get(key) if isinstance(key, str) else key
    if Version(transformers.__version__) < Version(spec.min_transformers):
        raise RuntimeError(f"{spec.key} needs transformers >= {spec.min_transformers}, "
                           f"found {transformers.__version__}")
    rev = revision or spec.revision
    kwargs = dict(revision=rev, dtype=dtype, attn_implementation=attn, device_map=device_map)
    if spec.config_overrides:
        config = transformers.AutoConfig.from_pretrained(spec.hf_id, revision=rev)
        for k, v in spec.config_overrides.items():
            setattr(config, k, v)
        kwargs["config"] = config
    if spec.loader == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        cls = transformers.AutoModelForCausalLM
    if tiny_layers:
        from . import tiny
        model, _, kept = tiny.build(spec, revision, tiny_layers, dtype=dtype, attn=attn,
                                    device="cuda" if str(device_map) in ("cuda", "balanced", "auto") else device_map)
    else:
        model = cls.from_pretrained(spec.hf_id, **kwargs)
    model.eval().requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.flipquant_source = dict(hf_id=spec.hf_id, revision=getattr(model.config, "_commit_hash", None) or rev,
                                  transformers=transformers.__version__, torch=str(torch.__version__))
    if tiny_layers:
        model.flipquant_source.update(tiny_layers=tiny_layers, tiny_kept_layers=kept, random_init=True)
    return model, tok, spec


def quantizable_linears(model: nn.Module, spec: ModelSpec) -> dict[str, nn.Linear]:
    """Every text-model ``nn.Linear`` except the output head, in module order."""
    head = model.get_output_embeddings()
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or m is head or name.endswith("lm_head"):
            continue
        if spec.include and spec.include not in name:
            continue
        if m.in_features % 16:
            log.warning("skipping %s: in_features %d is not a multiple of 16", name, m.in_features)
            continue
        out[name] = m
    return out


def weight_sha256(w: torch.Tensor) -> str:
    return hashlib.sha256(w.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _set_module(model: nn.Module, name: str, new: nn.Module):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, new)


@torch.no_grad()
def apply_nvfp4_checkpoint(model: nn.Module, spec: ModelSpec,
                           checkpoint: tuple[str, str | None] | None = None) -> dict:
    """Turn ``model`` into the vendor's ModelOpt release (``spec.nvfp4_checkpoint``, or ``checkpoint``
    = ``(repo id or local export dir, revision)``, e.g. our GPTQ export) up to its NVFP4
    linears: every BF16 tensor of the checkpoint (norms, embeddings, conv1d, excluded linears -- QAD may
    have changed them) is copied in, FP8 linears become :class:`flipquant.modelopt.FP8Linear`, and an
    NVFP4 output head is installed as is. The NVFP4 body linears keep their original BF16 weights here
    (the E0M3 candidate of ``mixfp4`` is quantized from them); pass the returned ``given`` to
    :func:`quantize`. Checkpoint tensors of modules the model lacks (e.g. Qwen's ``mtp*`` draft head)
    must match the checkpoint's ``exclude_modules``.
    Returns ``{"given": {module: {"codes", "input_scale"}}, "info": {...}}``."""
    from fnmatch import fnmatch

    from flipquant.modelopt import FP8Linear, load_checkpoint
    checkpoint = checkpoint or spec.nvfp4_checkpoint
    if checkpoint is None:
        raise ValueError(f"{spec.key} has no NVFP4 checkpoint in the registry")
    ckpt = load_checkpoint(*checkpoint)

    def rename(key):
        for old, new in spec.checkpoint_renames:
            if key.startswith(old):
                return new + key[len(old):]
        return key
    for part in ("dense", "quantized", "fp8"):
        ckpt[part] = {rename(k): v for k, v in ckpt[part].items()}
    excluded = ckpt["config"]["quantization"].get("exclude_modules", [])
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    changed, skipped = 0, []
    for key, value in ckpt["dense"].items():
        if key not in params:
            if any(fnmatch(key, p) for p in excluded):
                skipped.append(key)
                continue
            raise KeyError(f"checkpoint tensor {key} has no counterpart in {spec.key}")
        p = params[key]
        if p.shape != value.shape:
            raise ValueError(f"{key}: checkpoint {tuple(value.shape)} != model {tuple(p.shape)}")
        value = value.to(p.device, p.dtype)
        changed += int(not torch.equal(p, value))
        p.copy_(value)
    for name, t in ckpt["fp8"].items():
        lin = model.get_submodule(name)
        if not isinstance(lin, nn.Linear) or tuple(lin.weight.shape) != tuple(t["weight"].shape):
            raise ValueError(f"FP8 checkpoint module {name} does not match {type(lin).__name__} in {spec.key}")
        dev = lin.weight.device
        _set_module(model, name, FP8Linear(t["weight"].to(dev), t["weight_scale"].to(dev),
                                           t["input_scale"].to(dev), lin.bias, lin.weight.dtype, name))
    linears = quantizable_linears(model, spec)
    head = model.get_output_embeddings()
    given, head_name = {}, None
    for name, g in ckpt["quantized"].items():
        if name in linears:
            given[name] = g
        elif model.get_submodule(name) is head:
            head_name = name
            qspec = QuantSpec(weight="modelopt", act="nvfp4", act_scope="static", mode="fake")
            _set_module(model, name, FlipLinear.from_linear(head, qspec, given=g["codes"],
                                                            act_gscale=g["input_scale"], name=name))
        else:
            raise KeyError(f"NVFP4 checkpoint module {name} is neither a quantizable linear nor the head")
    info = dict(source=ckpt["source"], quantized_modules=len(given), fp8_modules=len(ckpt["fp8"]),
                nvfp4_head=head_name,
                bf16_linears=sorted(set(linears) - set(given) - set(ckpt["fp8"])),
                dense_tensors=len(ckpt["dense"]) - len(skipped), dense_tensors_changed=changed,
                skipped_tensors=len(skipped))
    log.warning("ModelOpt checkpoint %s: %d NVFP4 linears, %d FP8 linears, NVFP4 head %s, %d BF16 linears, "
                "%d/%d dense tensors differ from %s, %d excluded tensors skipped", ckpt["source"]["hf_id"],
                len(given), len(ckpt["fp8"]), head_name, len(info["bf16_linears"]), changed,
                info["dense_tensors"], spec.hf_id, len(skipped))
    return dict(given=given, info=info)


@torch.no_grad()
def quantize(model: nn.Module, spec: ModelSpec, qspec: QuantSpec, *, maps: dict | None = None,
             trainable: bool = False, names: list[str] | None = None,
             given: dict | None = None) -> dict[str, FlipLinear]:
    """Swap every quantizable linear for a :class:`FlipLinear`. Returns ``{name: layer}``.

    ``maps`` (``{name: bool tile map}``, see :mod:`flipquant.maps`) is required for a fresh
    ``mixfp4`` evaluation unless ``trainable`` (calibration starts from all-FourOverSix); a
    missing entry is an error rather than a silent default.

    ``given`` (from :func:`apply_nvfp4_checkpoint`) restricts quantization to the checkpoint's
    modules and supplies their static activation scales, and their NVFP4 codes for ``modelopt`` /
    ``mixfp4`` (a uniform format quantizes the original weight instead).
    """
    linears = quantizable_linears(model, spec)
    if given is not None:
        names = [n for n in linears if n in given and (names is None or n in names)]
    if names is not None:
        linears = {n: linears[n] for n in names}
    if qspec.weight == "mixfp4" and maps is None and not trainable:
        raise ValueError("mixfp4 needs a map (--map); use --weight fourover6 / e0m3 for the uniform ends")
    layers = {}
    for name in list(linears):
        lin = linears.pop(name)        # drop our reference, so each BF16 weight is freed once replaced
        mask = None
        if qspec.weight == "mixfp4" and maps is not None:
            if name not in maps:
                raise KeyError(f"map has no entry for {name}")
            mask = maps[name]
            expect = maps_.tile_shape(lin.out_features, lin.in_features, qspec.unit)
            if tuple(mask.shape) != expect:
                raise ValueError(f"{name}: map shape {tuple(mask.shape)} != {expect} at unit {qspec.unit}")
        g = given[name] if given is not None else {}
        codes = g.get("codes") if qspec.weight in ("modelopt", "mixfp4") else None
        act_gscale = g.get("input_scale") if qspec.act_scope == "static" else None
        layer = FlipLinear.from_linear(lin, qspec, mask=mask, trainable=trainable, name=name,
                                       given=codes, act_gscale=act_gscale)
        _set_module(model, name, layer)
        layers[name] = layer
        del lin
    torch.cuda.empty_cache()
    return layers
