"""NVFP4 / FourOverSix / E0M3 quantizers that return *codes*, not just dequantized values.

Every quantizer here produces an :class:`FP4Codes`: 4-bit sign-magnitude nibbles, one E4M3 scale
per 16 elements along K, and one FP32 global scale (per tensor, per document, or per row). The same
object feeds both execution modes:

  * fake quant  -- :meth:`FP4Codes.dequantize` gives BF16 values, bitwise identical to the
                   the reference quantizers (``quantize/quantizer.py``):
                     ``quant_nvfp4``                               -> ``nvfp4(w)``
                     ``quant_nvfp4_4over6``                        -> ``fourover6(w)``
                     ``quant_mix_4_6(clip='a1', elect='always')``  -> ``e0m3(w)``
                     ``causal_four_over_six.quantize_rows``        -> ``fourover6(x, "row")``
                     ``fast_act.quant_per_document``               -> ``fourover6(x, "document")``
                   tests/test_formats.py checks each of these bitwise.
  * native      -- :func:`pack_nibbles` gives the ``[rows, K/2]`` byte layout the CUDA kernels read,
                   and ``scale`` is already E4M3-representable, so :meth:`FP4Codes.scale_bytes` is exact.

Nibble encoding (shared with kernels/capi): ``code = (x < 0) << 3 | magnitude_index``. The index
reads as E2M1 {0,.5,1,1.5,2,3,4,6} or E0M3 {0..7} depending on the format of its region.

The order of floating-point operations is kept exactly as in the reference, since the goal is
bitwise agreement: ``global = amax / (6 * 448)`` (E0M3 shares it, so both candidates of one weight
have one global scale), ``block_scale = e4m3(clamp(block_max / qmax))`` for E2M1 but
``block_max * (1/7)`` for E0M3, and ``dequant = (value * block_scale) * global``.

FourOverSix's tensor-scale constant: ``fourover6(..., tensor_qmax=448)`` divides the amax by ``6 * tensor_qmax``.
448 (the default) is the NVFP4 convention this repository, its maps and RaZeR's reimplementation use. The original
FourOverSix (arXiv:2512.02010, Sec. 3) uses 256, the largest E4M3 value whose 6/4 multiple (384) is again an E4M3
value, so that every block can map its max onto 4; with 448 a block whose max exceeds 2/3 of the amax cannot (its
4-candidate scale is clamped at 448). The E4M3 clamp of the block scales is 448 either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP = 16
E4M3_MAX = 448.0
E4M3_MIN = 2.0 ** -9
E2M1_MAX = 6.0
E0M3_MAX = 7.0
GLOBAL_QMAX = E2M1_MAX * E4M3_MAX
TINY = torch.finfo(torch.float32).tiny

#: Format ids.  Same enum as the sm_100 UMMA instruction descriptor and kernels/capi.
FMT_E0M3 = 0
FMT_E2M1 = 1

_MAGNITUDES = {"e2m1": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
               "e0m3": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)}
_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def lut(fmt: int, device) -> torch.Tensor:
    """16-entry decode table indexed by nibble (``sign << 3 | magnitude``)."""
    name = "e0m3" if fmt == FMT_E0M3 else "e2m1"
    device = torch.device(device)

    def build():
        mags = torch.tensor(_MAGNITUDES[name], dtype=torch.float32, device=device)
        return torch.cat([mags, -mags])
    return _cached(("lut", name, device), build)


def _e2m1_mids(device) -> torch.Tensor:
    device = torch.device(device)

    def build():
        mags = torch.tensor(_MAGNITUDES["e2m1"], dtype=torch.float32, device=device)
        return (mags[:-1] + mags[1:]) / 2
    return _cached(("mids", device), build)


@dataclass
class FP4Codes:
    """A quantized ``[..., K]`` tensor.

    codes   uint8 ``[..., K]``       sign-magnitude nibble per element (unpacked)
    scale   float32 ``[..., K/16]``  E4M3-representable block scales
    gscale  float32, broadcastable against ``[..., K]``
    fmt     ``FMT_E2M1`` / ``FMT_E0M3`` for a uniform tensor, or a bool tensor ``[..., K/16]``
            that is True where the block decodes as E0M3
    """

    codes: torch.Tensor
    scale: torch.Tensor
    gscale: torch.Tensor
    fmt: object = FMT_E2M1

    def values(self) -> torch.Tensor:
        """Element values before any scale is applied (float32)."""
        c = self.codes.long()
        if isinstance(self.fmt, int):
            return lut(self.fmt, c.device)[c]
        e0m3 = self.fmt.bool().repeat_interleave(GROUP, -1)
        return torch.where(e0m3, lut(FMT_E0M3, c.device)[c], lut(FMT_E2M1, c.device)[c])

    def dequantize(self, dtype=torch.bfloat16) -> torch.Tensor:
        v = self.values()
        blocks = v.reshape(*v.shape[:-1], -1, GROUP) * self.scale.unsqueeze(-1)
        return (blocks.reshape(v.shape) * self.gscale).to(dtype)

    def scale_bytes(self) -> torch.Tensor:
        """Raw E4M3 bytes of the block scales, bit 7 clear (the scales are positive)."""
        return self.scale.to(torch.float8_e4m3fn).view(torch.uint8)

    def row_gscale(self) -> torch.Tensor:
        """The global scale expanded to one FP32 value per row, ``[rows]`` for a 2-D tensor."""
        rows = self.codes.shape[:-1]
        return self.gscale.float().expand(*rows, 1).reshape(-1)


# --------------------------------------------------------------------------------------------
# helpers


def _round_e2m1(y: torch.Tensor, ties: str) -> torch.Tensor:
    """Magnitude index 0..7 of the E2M1 value nearest ``|y|`` (saturating at 6).

    ``ties="down"`` is FourOverSix's midpoint rule (``x <= mid`` goes down); ``ties="up"`` is
    ``quant_nvfp4``'s ``floor(|mantissa| + 0.5)``, which sends ties up.
    """
    return torch.bucketize(y.abs().contiguous(), _e2m1_mids(y.device), right=(ties == "up"))


def _round_e2m1_nvfp4(y: torch.Tensor) -> torch.Tensor:
    """Magnitude index 0..7 of ``quant_nvfp4``'s rounding of ``|y|``, op for op: a per-element exponent
    ``e = max(floor(log2 |y|), 0)``, the mantissa ``floor(|y| / 2**e * 2 + 0.5)`` in FP32, saturating at 6.

    It differs from ``_round_e2m1(y, "up")`` (an exact nearest-with-ties-up) where ``|y| / 2**e * 2 + 0.5`` is within
    half an FP32 ulp below an integer: the FP32 addition rounds up there, so ``quant_nvfp4`` rounds up, e.g. ``y`` half an
    ulp below 0.25 gives 0.5, not 0. Kept for bitwise agreement with the reference quantizers.
    """
    a = y.abs()
    e = torch.floor(torch.log2(a + (a == 0).to(a.dtype))).clamp(min=0)
    q = (torch.floor(a / (2 ** e) * 2 + 0.5) * (2 ** e) / 2).clamp(max=E2M1_MAX)
    return torch.bucketize(q.contiguous(), _e2m1_mids(y.device), right=True)


def _round_e0m3(y: torch.Tensor) -> torch.Tensor:
    return y.round().clamp(-E0M3_MAX, E0M3_MAX).abs().to(torch.long)


def _sign(y: torch.Tensor) -> torch.Tensor:
    return (y < 0).to(torch.long) << 3


def _e4m3(s: torch.Tensor) -> torch.Tensor:
    return s.clamp(min=E4M3_MIN, max=E4M3_MAX).to(torch.float8_e4m3fn).float()


SCOPES = ("tensor", "document", "row", "static")


def _grouped(x: torch.Tensor, scope: str, gscale: torch.Tensor | None = None, qmax: float = GLOBAL_QMAX):
    """Reshape to ``[D, R, G, 16]`` and compute the global scale ``amax / qmax`` for ``scope``.

    ``document`` treats dim 0 of a >=3-D input as the document (batch) axis; a 2-D input is one
    document.  ``row`` is one scale per row of the flattened ``[-1, K]`` view (per token).
    ``static`` uses the given ``gscale`` (a calibrated scalar, e.g. a ModelOpt ``input_scale``);
    values beyond ``gscale * 6 * 448`` saturate.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scale scope {scope!r}; choose from {SCOPES}")
    k = x.shape[-1]
    if k % GROUP:
        raise ValueError(f"last dimension {k} must be a multiple of {GROUP}")
    if scope == "static":
        if gscale is None:
            raise ValueError("scope 'static' needs a fixed gscale")
        return x.float().reshape(1, -1, k // GROUP, GROUP), gscale.float().reshape(())
    docs = x.shape[0] if (scope == "document" and x.dim() >= 3) else 1
    b = x.float().reshape(docs, -1, k // GROUP, GROUP)
    if scope == "tensor":
        amax = b.abs().amax()
    elif scope == "document":
        amax = b.abs().amax(dim=(1, 2, 3), keepdim=True)
    else:
        amax = b.abs().amax(dim=(2, 3), keepdim=True)
    return b, (amax / qmax).clamp(min=TINY)


def _result(x, codes, scale, gs, fmt) -> FP4Codes:
    lead, k = x.shape[:-1], x.shape[-1]
    if gs.dim():                                   # [D,1,1,1] or [D,R,1,1] -> broadcast vs [..., K]
        if gs.shape[1] == 1:                       # per document
            gs = gs.reshape(gs.shape[0], *([1] * (x.dim() - 1))) if x.dim() >= 3 else gs.reshape(())
        else:                                      # per row
            gs = gs.reshape(*lead, 1)
    return FP4Codes(codes.to(torch.uint8).reshape(*lead, k), scale.reshape(*lead, k // GROUP), gs, fmt)


# --------------------------------------------------------------------------------------------
# quantizers


@torch.no_grad()
def nvfp4(x: torch.Tensor, scope: str = "tensor", gscale: torch.Tensor | None = None) -> FP4Codes:
    """Plain NVFP4: block scale ``block_max / 6``, ties up, rounded with ``quant_nvfp4``'s FP32 arithmetic."""
    b, gs = _grouped(x, scope, gscale)
    ws = b / gs
    scale = _e4m3(ws.abs().amax(-1, keepdim=True) / E2M1_MAX)
    y = ws / scale
    return _result(x, _round_e2m1_nvfp4(y) | _sign(y), scale.squeeze(-1), gs, FMT_E2M1)


def global_qmax(tensor_qmax: float = E4M3_MAX) -> float:
    """FourOverSix's global-scale divisor ``6 * tensor_qmax`` (module docstring): ``GLOBAL_QMAX`` itself for 448."""
    if not 0.0 < tensor_qmax <= E4M3_MAX:
        raise ValueError(f"tensor_qmax {tensor_qmax}: must be in (0, {E4M3_MAX}]")
    return GLOBAL_QMAX if tensor_qmax == E4M3_MAX else E2M1_MAX * float(tensor_qmax)


@torch.no_grad()
def fourover6(x: torch.Tensor, scope: str = "tensor", gscale: torch.Tensor | None = None,
              tensor_qmax: float = E4M3_MAX) -> FP4Codes:
    """FourOverSix (arXiv:2512.02010): per 16-block, map the block max onto 6 or onto 4 and keep
    the lower squared error, 4 only if strictly lower (``quant_nvfp4_4over6``). The global scale is
    ``amax / (6 * tensor_qmax)``: 448 by default, 256 in the original paper (module docstring)."""
    b, gs = _grouped(x, scope, gscale, global_qmax(tensor_qmax))
    ws = b / gs
    peak = ws.abs().amax(-1, keepdim=True)
    table = lut(FMT_E2M1, x.device)
    best = None
    for qmax in (6.0, 4.0):
        scale = _e4m3(peak / qmax)
        y = ws / scale
        codes = _round_e2m1(y, "down") | _sign(y)
        err = (table[codes] * scale - ws).square().sum(-1, keepdim=True)
        if best is None:
            best = (codes, scale, err)
        else:
            use4 = err < best[2]
            best = (torch.where(use4, codes, best[0]), torch.where(use4, scale, best[1]), None)
    return _result(x, best[0], best[1].squeeze(-1), gs, FMT_E2M1)


@torch.no_grad()
def e0m3(x: torch.Tensor, scope: str = "tensor", gscale: torch.Tensor | None = None) -> FP4Codes:
    """Sign-magnitude INT4 with the block max on 7: MixFP4's candidate A,
    ``quant_mix_4_6(clip='a1', elect='always')``."""
    b, gs = _grouped(x, scope, gscale)
    ws = b / gs
    scale = _e4m3(ws.abs().amax(-1, keepdim=True) * (1.0 / E0M3_MAX))
    y = ws / scale
    return _result(x, _round_e0m3(y) | _sign(y), scale.squeeze(-1), gs, FMT_E0M3)


QUANTIZERS = {"nvfp4": nvfp4, "fourover6": fourover6, "e0m3": e0m3}


def fake_quant(x: torch.Tensor, fmt: str = "fourover6", scope: str = "tensor") -> torch.Tensor:
    """Quantize then dequantize; returns BF16 like the reference quantizers."""
    return QUANTIZERS[fmt](x, scope).dequantize(torch.bfloat16).reshape(x.shape)


# --------------------------------------------------------------------------------------------
# packing for the native kernels


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """``[..., K]`` nibbles -> ``[..., K/2]`` bytes, element ``2j`` in the low nibble."""
    c = codes.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0xF, packed >> 4), -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
