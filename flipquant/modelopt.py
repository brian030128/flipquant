"""Read an NVIDIA ModelOpt checkpoint (``hf_quant_config.json``: ``quant_algo = NVFP4``, or
``MIXED_PRECISION`` with NVFP4 and FP8 layers) so a published model can be evaluated exactly as it is
deployed, and its NVFP4 codes used as the E2M1 candidate of a MixFP4 map.

Per NVFP4 linear ``<m>`` the checkpoint holds

  <m>.weight          uint8 ``[N, K/2]``  E2M1 nibbles, element ``2j`` in the low nibble
  <m>.weight_scale    float8_e4m3fn ``[N, K/16]`` block scales
  <m>.weight_scale_2  float32 scalar: weight global scale (``amax / (6 * 448)``)
  <m>.input_scale     float32 scalar: static activation global scale (``amax / (6 * 448)``)

which is the nibble layout, nibble encoding and scale order of :mod:`flipquant.formats`:
``W = (e2m1[code] * weight_scale) * weight_scale_2``. At run time activations are quantized to NVFP4
with dynamic E4M3 block scales under the fixed ``input_scale`` (vLLM / TensorRT-LLM), which is
``formats.nvfp4(x, "static", input_scale)``.

Per FP8 linear (per-tensor static W8A8): ``<m>.weight`` float8_e4m3fn ``[N, K]``, ``<m>.weight_scale``
and ``<m>.input_scale`` float32 scalars; ``W = weight * weight_scale`` and activations are
``e4m3(x / input_scale) * input_scale`` (saturating), emulated by :class:`FP8Linear`.

Every other tensor (norms, embeddings, conv1d, unquantized linears) is stored in BF16.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import formats as fmt

_QUANT_SUFFIXES = (".weight_scale_2", ".weight_scale", ".input_scale")


def load_checkpoint(repo: str, revision: str | None = None) -> dict:
    """``{"quantized": {module: {"codes": FP4Codes, "input_scale"}}, "fp8": {module: {"weight",
    "weight_scale", "input_scale"}}, "dense": {key: tensor}, "config": hf_quant_config, "source": {...}}``.
    ``repo`` is a HuggingFace repo id or a local export directory. Tensors are on the CPU."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    if Path(repo).is_dir():
        root = Path(repo)
    else:
        root = Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]))
    config = json.loads((root / "hf_quant_config.json").read_text())
    q = config["quantization"]
    if q["quant_algo"] == "NVFP4":
        algos = {q["quant_algo"]}
        groups = {q.get("group_size", fmt.GROUP)}
    elif q["quant_algo"] == "MIXED_PRECISION":
        algos = {v["quant_algo"] for v in q["quantized_layers"].values()}
        groups = {v["group_size"] for v in q["quantized_layers"].values() if v["quant_algo"] == "NVFP4"}
    else:
        algos, groups = {q["quant_algo"]}, set()
    if not algos <= {"NVFP4", "FP8"} or not groups <= {fmt.GROUP}:
        raise ValueError(f"{repo}: only NVFP4 (group 16) and per-tensor FP8 are supported, found {algos}")
    tensors = {}
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                tensors[k] = h.get_tensor(k)
    quantized, fp8 = {}, {}
    for m in sorted(k[:-len(".weight_scale_2")] for k in tensors if k.endswith(".weight_scale_2")):
        packed = tensors.pop(m + ".weight")
        if packed.dtype != torch.uint8:
            raise ValueError(f"{m}.weight: expected packed uint8, found {packed.dtype}")
        scale = tensors.pop(m + ".weight_scale")
        codes = fmt.FP4Codes(fmt.unpack_nibbles(packed), scale.view(torch.float8_e4m3fn).float(),
                             tensors.pop(m + ".weight_scale_2").float().reshape(()), fmt.FMT_E2M1)
        quantized[m] = dict(codes=codes, input_scale=tensors.pop(m + ".input_scale").float().reshape(()))
    for m in sorted(k[:-len(".weight_scale")] for k in tensors if k.endswith(".weight_scale")):
        w = tensors.pop(m + ".weight")
        if w.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{m}.weight: expected float8_e4m3fn, found {w.dtype}")
        ws = tensors.pop(m + ".weight_scale")
        if ws.numel() != 1:
            raise ValueError(f"{m}.weight_scale: only per-tensor FP8 is supported, found {tuple(ws.shape)}")
        fp8[m] = dict(weight=w, weight_scale=ws.float().reshape(()),
                      input_scale=tensors.pop(m + ".input_scale").float().reshape(()))
    leftover = [k for k in tensors if k.endswith(_QUANT_SUFFIXES)]
    if leftover:
        raise ValueError(f"{repo}: unrecognized quantization tensors: {leftover[:4]}")
    return dict(quantized=quantized, fp8=fp8, dense=tensors, config=config,
                source=dict(hf_id=repo, revision=revision, producer=config.get("producer")))


class FP8Linear(nn.Module):
    """Fake-quant per-tensor static FP8 (E4M3) W8A8 linear, as a ModelOpt FP8 layer runs in vLLM."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, input_scale: torch.Tensor,
                 bias: torch.Tensor | None = None, dtype=torch.bfloat16, name: str = ""):
        super().__init__()
        self.name = name
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", (weight.float() * weight_scale.float()).to(dtype))
        self.register_buffer("input_scale", input_scale.float().reshape(()))
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        s = self.input_scale.to(x.device)
        xq = ((x.float() / s).clamp(-fmt.E4M3_MAX, fmt.E4M3_MAX).to(torch.float8_e4m3fn).float() * s).to(x.dtype)
        y = F.linear(xq, self.weight)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, w=fp8/tensor, a=fp8/static"
