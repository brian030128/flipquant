"""Block-diagonal Hadamard rotation of the quantized linears (``--rotate hadamard``).

For a linear ``y = x W^T`` with ``W`` ``[O, K]``, ``H`` is block-diagonal along K with orthonormal
Sylvester-Hadamard blocks ``H_b / sqrt(b)`` (symmetric, so ``H^T = H`` and ``H H^T = I``). The weight
is rotated once, offline, ``W' = W H``, and every input online, ``x' = x H`` (``H^T x`` for a column
vector), *before* activation quantization; ``x' W'^T = x H H^T W^T = x W^T``, so in BF16 the model is
unchanged up to rounding. The FP4 candidates (A = E0M3, B = FourOverSix), the map and GPTQ all live in
the rotated space: the quantizers see ``W'`` and ``x'``.

Block size ``b`` = the largest power of two dividing K, capped at ``max_block`` (default 16):
  * 16 is the NVFP4 / FourOverSix scale group, so each rotated block is exactly one scale group: the
    rotation never mixes values across scale groups, across the 64-wide map tiles or across the
    E0M3/E2M1 format boundary of a MixFP4 map, and every K here (a multiple of 16) gets b = 16;
  * it is also what measured best (Qwen3-1.7B, W4A4, 32 windows, dNLL vs no rotation in 1e-3 nats):
    FourOverSix +33 / +172 / +165 (wiki) for b = 16 / 64 / 128, NVFP4 +228 / +167 / +337 -- larger
    blocks Gaussianize values the per-16 scales already handle and hurt more; no block helped;
  * a Sylvester block needs a power of two, and blocks cannot straddle a K that is not a multiple of
    ``b``; ``--rotate-block`` up to 128 remains available (the online cost is ``O(K b)`` per token).
A map trained in the rotated space records ``rotate`` and ``rotate_block`` and is only accepted by
an evaluation with the same two settings.

The online rotation is a torch matmul in FP32 (``x.float() @ H_b`` per block), cast back to the input
dtype; no fused kernel.

``RotatedLinear`` is an ``nn.Linear`` (so :func:`models.quantize` picks it up unchanged) whose weight
is ``W'`` and whose forward rotates its input; it is the BF16 model of ``--mode bf16 --rotate
hadamard``. :class:`flipquant.linear.FlipLinear` takes the same ``rotation`` module (``layer.rotation``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

ROTATIONS = ("none", "hadamard")
_CACHE: dict = {}


def sylvester(n: int, device=None) -> torch.Tensor:
    """Orthonormal ``n x n`` Sylvester-Hadamard matrix (FP32, symmetric), ``n`` a power of two."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size {n} is not a power of two")
    key = (n, str(torch.device(device or "cpu")))
    if key not in _CACHE:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _CACHE[key] = (h / math.sqrt(n)).float().to(device)
    return _CACHE[key]


def block_size(features: int, max_block: int = 16) -> int:
    """Largest power of two dividing ``features``, capped at ``max_block`` (a power of two)."""
    if max_block < 1 or max_block & (max_block - 1):
        raise ValueError(f"max block {max_block} is not a power of two")
    return min(features & -features, max_block)


class Hadamard(nn.Module):
    """``x -> x H`` along the last dimension, ``H`` = blockdiag of ``block x block`` Hadamards."""

    def __init__(self, features: int, block: int):
        super().__init__()
        if features % block:
            raise ValueError(f"{features} features are not a multiple of the Hadamard block {block}")
        self.features, self.block = features, block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = sylvester(self.block, x.device)
        y = x.float().reshape(*x.shape[:-1], self.features // self.block, self.block) @ h
        return y.reshape(x.shape).to(x.dtype)

    def extra_repr(self):
        return f"features={self.features}, block={self.block}"


class RotatedLinear(nn.Linear):
    """BF16 linear in the rotated basis: weight ``W H``, forward ``F.linear(x H, W H, bias)``."""

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, max_block: int = 16) -> "RotatedLinear":
        w = linear.weight
        new = cls(linear.in_features, linear.out_features, bias=False, device="meta")
        new.rotation = Hadamard(linear.in_features, block_size(linear.in_features, max_block))
        new.weight = nn.Parameter(new.rotation(w.float()).to(w.dtype), requires_grad=False)
        new.bias = linear.bias
        return new

    def forward(self, x):
        return F.linear(self.rotation(x), self.weight, self.bias)


@torch.no_grad()
def rotate_model(model: nn.Module, spec, max_block: int = 16) -> dict[str, Hadamard]:
    """Replace every quantizable linear of ``model`` by its :class:`RotatedLinear` (in place).
    Returns ``{name: rotation}`` for :func:`attach`."""
    from models.loader import _set_module, quantizable_linears
    rotations = {}
    linears = quantizable_linears(model, spec)
    for name in list(linears):
        lin = linears.pop(name)                 # no other reference: the BF16 weight goes with it
        if isinstance(lin, RotatedLinear):
            raise ValueError(f"{name} is already rotated")
        new = RotatedLinear.from_linear(lin, max_block)
        _set_module(model, name, new)
        rotations[name] = new.rotation
        del lin
    torch.cuda.empty_cache()
    return rotations


def attach(layers: dict, rotations: dict[str, Hadamard]) -> None:
    """Give each quantized layer (built from a :class:`RotatedLinear`) its online input rotation."""
    for name, layer in layers.items():
        layer.rotation = rotations[name]


def summary(rotations: dict[str, Hadamard]) -> dict:
    blocks: dict[int, int] = {}
    for r in rotations.values():
        blocks[r.block] = blocks.get(r.block, 0) + 1
    return {"modules": len(rotations), "blocks": {str(b): n for b, n in sorted(blocks.items())}}
