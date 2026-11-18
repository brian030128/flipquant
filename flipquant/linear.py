"""``FlipLinear``: the drop-in replacement for ``nn.Linear`` in both execution modes.

``mode="fake"``    weights are held dequantized in BF16 and activations are quantize-dequantized
                   in torch; the matmul is an ordinary BF16 GEMM.  Runs anywhere (H100/H200), and
                   is bitwise the reference protocol (in-place weight overwrite + activation
                   pre-hooks).
``mode="native"``  weights are held packed (nibbles + E4M3 scales + per-granule format bytes),
                   activations are quantized to codes, and the product runs on the Blackwell FP4
                   tensor core through :mod:`flipquant.native` (B200 = sm100, Pro 6000 = sm120).
                   The ``emulate`` backend runs the same buffers through a torch reference.

Weight formats: ``nvfp4`` | ``fourover6`` | ``e0m3`` (uniform) and ``mixfp4`` (per-tile choice
between A = E0M3 and B = FourOverSix E2M1, driven by a map from :mod:`flipquant.maps`).
``modelopt`` is a given NVFP4 encoding (a published checkpoint, :mod:`flipquant.modelopt`) used as
is; with ``given`` codes, ``mixfp4`` uses them as candidate B and quantizes A = E0M3 under their
global scale. ``act_gscale`` gives the layer a static activation scale (``act_scope="static"``).

:meth:`FlipLinear.from_codes` builds a frozen layer from precomputed codes (GPTQ, flipquant/gptq.py),
used as they are in both modes. ``layer.rotation`` (default None) is an optional online input rotation
(flipquant/rotate.py) applied before activation quantization.

For calibration a layer is built with ``trainable=True``: it keeps both packed candidates so the map
can be changed in place (:meth:`set_map`), and while ``map_grad`` is set, the backward pass adds
``reduce((dy^T x_q) * (A - B))`` -- dLoss/dm for every tile -- exactly as the reference's
``run_train_map.py`` backward hook does. The backward is the same in both modes (``dx = dy @ W``
with the dequantized current weight), so native calibration trains on native forward numerics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import formats as fmt
from . import maps as maps_
from .native import get_backend

WEIGHT_FORMATS = ("nvfp4", "fourover6", "e0m3", "mixfp4", "modelopt")
ACT_FORMATS = ("none", "nvfp4", "fourover6")


@dataclass
class QuantSpec:
    weight: str = "mixfp4"
    act: str = "fourover6"
    #: Activation global-scale scope: ``row`` (per token; causal, used for training and generation),
    #: ``document`` (per sequence; the reference PPL protocol), ``tensor``, or ``static`` (the layer's
    #: calibrated ``act_gscale``, e.g. a ModelOpt ``input_scale``).
    act_scope: str = "row"
    unit: str = "8x64"
    mode: str = "fake"
    native_backend: str = "auto"
    #: Fake mode: keep the dequantized BF16 weight resident (``True``, the original behaviour), or keep
    #: only the packed candidates and decode the weight -- the same value -- whenever a forward or
    #: backward needs it (``False``: ~2 bytes per weight less, e.g. ~45 GiB for Qwen3.8-27B).
    fake_weight_cache: bool = True
    #: FourOverSix's tensor-scale constant (``formats.fourover6``'s ``tensor_qmax``): 448, the default and every
    #: map's, or 256, the original FourOverSix. Anything but 448 is the FourOverSix baseline only (weight and
    #: activations ``fourover6``).
    fo6_tensor_qmax: float = fmt.E4M3_MAX

    def __post_init__(self):
        assert self.weight in WEIGHT_FORMATS, self.weight
        assert self.act in ACT_FORMATS, self.act
        assert self.act_scope in fmt.SCOPES, self.act_scope
        assert self.mode in ("fake", "native"), self.mode
        if self.mode == "native" and self.act == "none":
            raise ValueError("native kernels are W4A4; weight-only (act=none) exists only in fake mode")
        fmt.global_qmax(self.fo6_tensor_qmax)
        if self.fo6_tensor_qmax != fmt.E4M3_MAX and (self.weight, self.act) != ("fourover6", "fourover6"):
            raise ValueError(f"fo6_tensor_qmax {self.fo6_tensor_qmax}: the FourOverSix baseline only (weight and act "
                             f"fourover6), not weight {self.weight!r} / act {self.act!r}")

    def as_dict(self):
        d = asdict(self)
        if d["fake_weight_cache"]:          # the default: reports stay as they were
            del d["fake_weight_cache"]
        if d["fo6_tensor_qmax"] == fmt.E4M3_MAX:
            del d["fo6_tensor_qmax"]
        return d


class _Packed:
    """One weight candidate as the kernels store it: nibbles [N, K/2] + E4M3 bytes [N, K/16]."""

    def __init__(self, codes: fmt.FP4Codes):
        self.q = fmt.pack_nibbles(codes.codes)
        self.s = codes.scale_bytes().contiguous()
        self.fmt = codes.fmt

    def to(self, device):
        self.q, self.s = self.q.to(device), self.s.to(device)
        return self

    def codes(self, gscale) -> fmt.FP4Codes:
        return fmt.FP4Codes(fmt.unpack_nibbles(self.q), self.s.view(torch.float8_e4m3fn).float(), gscale, self.fmt)


class _FlipFunction(torch.autograd.Function):
    """Forward through the layer's (fake or native) product; backward ``dx = dy @ W`` plus the
    per-tile map gradient.  Activation quantization is straight-through."""

    @staticmethod
    def forward(ctx, x, layer):
        y, xq = layer._product(x, keep_input=layer.map_grad is not None)
        ctx.layer = layer
        ctx.save_for_backward(xq if xq is not None else x.new_empty(0))
        return y

    @staticmethod
    def backward(ctx, dy):
        layer = ctx.layer
        (xq,) = ctx.saved_tensors
        w = layer.effective_weight()
        dx = dy.to(w.dtype) @ w
        if layer.map_grad is not None:
            rows, cols = maps_.parse_unit(layer.spec.unit)
            g = dy.detach().reshape(-1, dy.shape[-1]).float().T @ xq.reshape(-1, xq.shape[-1]).float()
            layer.map_grad += maps_.reduce(g * layer.delta(), rows, cols)
        return dx, None


class FlipLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, spec: QuantSpec, bias=None, name: str = ""):
        super().__init__()
        if in_features % fmt.GROUP:
            raise ValueError(f"{name}: in_features={in_features} is not a multiple of {fmt.GROUP}")
        self.in_features, self.out_features = in_features, out_features
        self.spec, self.name = spec, name
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.map_grad: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self._candidates: dict[str, _Packed] = {}
        self._backend = None
        #: Optional online input rotation (flipquant/rotate.py), applied before activation quantization.
        self.rotation: nn.Module | None = None
        self.register_buffer("act_gscale", None)

    # ---------------------------------------------------------------------------------------
    # construction

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, spec: QuantSpec, mask: torch.Tensor | None = None,
                    trainable: bool = False, name: str = "", given: fmt.FP4Codes | None = None,
                    act_gscale: torch.Tensor | None = None) -> "FlipLinear":
        w = linear.weight.detach()
        layer = cls(linear.in_features, linear.out_features, spec, linear.bias, name)
        layer.to(w.device)
        dtype = w.dtype
        if (spec.weight == "modelopt") != (given is not None) and spec.weight != "mixfp4":
            raise ValueError(f"{name}: given codes go with weight 'modelopt' or 'mixfp4', not {spec.weight!r}")
        if (spec.act_scope == "static") != (act_gscale is not None):
            raise ValueError(f"{name}: act_scope 'static' and act_gscale go together")
        if act_gscale is not None:
            layer.act_gscale = act_gscale.float().reshape(()).to(w.device)
        if given is not None:
            given = fmt.FP4Codes(given.codes.to(w.device), given.scale.to(w.device),
                                 given.gscale.float().reshape(()).to(w.device), given.fmt)
            if tuple(given.codes.shape) != tuple(w.shape):
                raise ValueError(f"{name}: given codes {tuple(given.codes.shape)} != weight {tuple(w.shape)}")
        if spec.weight == "mixfp4" and given is not None:
            a, b = fmt.e0m3(w, "static", given.gscale), given
            layer._candidates = {"a": _Packed(a), "b": _Packed(b)}
            gscale = b.gscale
            if mask is None:
                mask = torch.zeros(maps_.tile_shape(*w.shape, spec.unit), dtype=torch.bool)
        elif spec.weight == "mixfp4":
            a, b = fmt.e0m3(w), fmt.fourover6(w)
            layer._candidates = {"a": _Packed(a), "b": _Packed(b)}
            gscale = b.gscale
            if mask is None:
                mask = torch.zeros(maps_.tile_shape(*w.shape, spec.unit), dtype=torch.bool)
        else:
            if spec.weight == "modelopt":
                only = given
            elif spec.weight == "fourover6":
                only = fmt.fourover6(w, tensor_qmax=spec.fo6_tensor_qmax)
            else:
                only = fmt.QUANTIZERS[spec.weight](w)
            layer._candidates = {"b": _Packed(only)}
            gscale = only.gscale
        layer.register_buffer("gscale", gscale.float().reshape(()))
        layer.dtype_ = dtype
        if spec.mode == "fake":
            if spec.fake_weight_cache:
                layer.register_buffer("weight", torch.empty_like(w))
        else:
            layer._backend = get_backend(spec.native_backend)
        layer._install(mask)
        if not trainable:
            if spec.mode == "native" or spec.fake_weight_cache:   # keep only what the forward reads
                layer._candidates = {}
        return layer

    @classmethod
    @torch.no_grad()
    def from_codes(cls, linear: nn.Linear, spec: QuantSpec, codes: fmt.FP4Codes, mask: torch.Tensor | None = None,
                   name: str = "", act_gscale: torch.Tensor | None = None) -> "FlipLinear":
        """A frozen layer whose weight is the given codes, used as they are (e.g. GPTQ,
        flipquant/gptq.py): nothing is re-quantized, so fake and native both read exactly these
        nibbles, E4M3 block scales and global scale. ``linear`` supplies shape, device, dtype, bias.

        Uniform weights (``nvfp4`` / ``fourover6`` / ``modelopt``: ``codes.fmt == FMT_E2M1``; ``e0m3``:
        ``FMT_E0M3``). ``mixfp4``: ``codes.fmt`` is the per-16-block bool (True = E0M3) and must be
        ``mask`` (the tile map) expanded; the codes then serve as both candidates and ``mask`` selects
        which decode each tile gets, exactly as for :meth:`from_linear`.
        """
        w = linear.weight
        layer = cls(linear.in_features, linear.out_features, spec, linear.bias, name)
        layer.to(w.device)
        if tuple(codes.codes.shape) != tuple(w.shape):
            raise ValueError(f"{name}: codes {tuple(codes.codes.shape)} != weight {tuple(w.shape)}")
        if (spec.act_scope == "static") != (act_gscale is not None):
            raise ValueError(f"{name}: act_scope 'static' and act_gscale go together")
        if act_gscale is not None:
            layer.act_gscale = act_gscale.float().reshape(()).to(w.device)
        gscale = codes.gscale.float().reshape(()).to(w.device)
        c, s = codes.codes.to(w.device), codes.scale.to(w.device)
        if spec.weight == "mixfp4":
            if mask is None:
                raise ValueError(f"{name}: mixfp4 codes need their tile map")
            rows, cols = maps_.parse_unit(spec.unit)
            per_block = maps_.expand(mask.to(w.device).bool(), rows, cols // fmt.GROUP, linear.out_features)
            if isinstance(codes.fmt, int) or not torch.equal(codes.fmt.to(w.device).bool(), per_block):
                raise ValueError(f"{name}: the codes' per-block formats are not the map's")
            layer._candidates = {"a": _Packed(fmt.FP4Codes(c, s, gscale, fmt.FMT_E0M3)),
                                 "b": _Packed(fmt.FP4Codes(c, s, gscale, fmt.FMT_E2M1))}
        else:
            want = fmt.FMT_E0M3 if spec.weight == "e0m3" else fmt.FMT_E2M1
            if not isinstance(codes.fmt, int) or codes.fmt != want:
                raise ValueError(f"{name}: {spec.weight} codes must be uniformly format {want}")
            layer._candidates = {"b": _Packed(fmt.FP4Codes(c, s, gscale, want))}
            mask = None
        layer.register_buffer("gscale", gscale)
        layer.dtype_ = w.dtype
        if spec.mode == "fake":
            if spec.fake_weight_cache:
                layer.register_buffer("weight", torch.empty_like(w))
        else:
            layer._backend = get_backend(spec.native_backend)
        layer._install(mask)
        if spec.mode == "native" or spec.fake_weight_cache:  # cache off: the codes are the weight
            layer._candidates = {}
        return layer

    # ---------------------------------------------------------------------------------------
    # weights

    def _decoded(self, key) -> torch.Tensor:
        return self._candidates[key].codes(self.gscale).dequantize(self.dtype_)

    def _selection(self, mask, per_row_cols):
        """Tile mask expanded to ``[N, K / per_row_cols]`` (bytes or scale groups)."""
        rows, cols = maps_.parse_unit(self.spec.unit)
        return maps_.expand(mask.to(self.gscale.device), rows, cols // per_row_cols, self.out_features)

    @torch.no_grad()
    def _install(self, mask):
        spec = self.spec
        self.mask = None if mask is None else mask.to(self.gscale.device).bool()
        if spec.mode == "fake":
            if not spec.fake_weight_cache:          # decoded on demand by _fake_weight()
                return
            if spec.weight == "mixfp4":
                rows, cols = maps_.parse_unit(spec.unit)
                sel = maps_.expand(self.mask, rows, cols, self.out_features)
                self.weight.copy_(torch.where(sel, self._decoded("a"), self._decoded("b")))
            else:
                self.weight.copy_(self._decoded("b"))
            return
        # native: select packed bytes per tile, and the per-granule format bytes
        b = self._candidates["b"]
        n8, k64 = self._backend.granule
        if spec.weight == "mixfp4":
            a = self._candidates["a"]
            q = torch.where(self._selection(self.mask, 2), a.q, b.q)
            s = torch.where(self._selection(self.mask, fmt.GROUP), a.s, b.s)
            fb = maps_.to_granule(self.mask, spec.unit, (n8, k64), self.out_features)
        else:
            q, s = b.q, b.s
            value = fmt.FMT_E0M3 if spec.weight == "e0m3" else fmt.FMT_E2M1
            if self.out_features % n8 or self.in_features % k64:
                raise ValueError(f"{self.name}: shape {self.out_features}x{self.in_features} does not "
                                 f"tile the {n8}x{k64} kernel granule")
            fb = torch.full((self.out_features // n8, self.in_features // k64), value, dtype=torch.uint8,
                            device=q.device)
        for key, value in (("q", q), ("s", s), ("fmt_b", fb)):
            if key in self._buffers:
                self._buffers[key] = value.contiguous()
            else:
                self.register_buffer(key, value.contiguous())

    def set_map(self, mask: torch.Tensor):
        """Switch the per-tile format choice in place (needs ``trainable=True`` construction)."""
        assert self.spec.weight == "mixfp4" and self._candidates, "layer was not built trainable"
        self._install(mask)

    @torch.no_grad()
    def set_soft(self, prob: torch.Tensor):
        """Fake mode only: ``W = B + expand(prob) * (A - B)`` (sigmoid-relaxed calibration)."""
        assert self.spec.mode == "fake"
        if not self.spec.fake_weight_cache:
            raise RuntimeError("set_soft writes an interpolated weight that the packed candidates cannot "
                               "represent; it needs the fake weight cache (--fake-weight-cache on)")
        rows, cols = maps_.parse_unit(self.spec.unit)
        m = maps_.expand(prob.to(self.gscale.device), rows, cols, self.out_features)
        b = self._decoded("b")
        self.weight.copy_((b.float() + m * (self._decoded("a").float() - b.float())).to(b.dtype))

    @torch.no_grad()
    def _fake_weight(self) -> torch.Tensor:
        """The fake-mode BF16 weight: the cache, or (cache off) decoded from the packed candidates and
        the current map -- exactly what ``_install`` would have written into the cache."""
        if self.spec.fake_weight_cache:
            return self.weight
        if self.spec.weight == "mixfp4":
            rows, cols = maps_.parse_unit(self.spec.unit)
            sel = maps_.expand(self.mask, rows, cols, self.out_features)
            return torch.where(sel, self._decoded("a"), self._decoded("b"))
        return self._decoded("b")

    def effective_weight(self) -> torch.Tensor:
        if self.spec.mode == "fake":
            return self._fake_weight()
        if self._candidates:
            if self.spec.weight != "mixfp4":
                return self._decoded("b")
            rows, cols = maps_.parse_unit(self.spec.unit)
            sel = maps_.expand(self.mask, rows, cols, self.out_features)
            return torch.where(sel, self._decoded("a"), self._decoded("b"))
        # frozen native layer: decode what the kernel reads
        from .native import EmulatedBackend
        w = EmulatedBackend._decode(self.q, self.s, self.fmt_b, self._backend.granule)
        return (w * self.gscale).to(self.dtype_)

    def delta(self) -> torch.Tensor:
        return self._decoded("a").float() - self._decoded("b").float()

    # ---------------------------------------------------------------------------------------
    # forward

    def _quant_act(self, x):
        if self.spec.act == "fourover6":
            return fmt.fourover6(x, self.spec.act_scope, self.act_gscale, tensor_qmax=self.spec.fo6_tensor_qmax)
        return fmt.QUANTIZERS[self.spec.act](x, self.spec.act_scope, self.act_gscale)

    def _product(self, x, keep_input=False):
        """``(y, x_q)`` without autograd.  ``x_q`` is the dequantized activation if requested."""
        spec = self.spec
        lead = x.shape[:-1]
        if spec.mode == "fake":
            xq = x if spec.act == "none" else self._quant_act(x).dequantize(x.dtype).reshape(x.shape)
            y = F.linear(xq, self._fake_weight())
            return y, (xq if keep_input else None)
        codes = self._quant_act(x)
        c2 = codes.codes.reshape(-1, self.in_features)
        a = fmt.pack_nibbles(c2)
        sa = codes.scale_bytes().reshape(-1, self.in_features // fmt.GROUP).contiguous()
        d = self._backend.gemm(a, sa, self.q, self.s, self.fmt_b, fmt.FMT_E2M1)
        rescale = codes.row_gscale().to(d.device) * self.gscale
        y = (d.float() * rescale[:, None]).to(x.dtype).reshape(*lead, self.out_features)
        xq = codes.dequantize(x.dtype).reshape(x.shape) if keep_input else None
        return y, xq

    def forward(self, x):
        if self.rotation is not None:
            x = self.rotation(x)
        if torch.is_grad_enabled() and x.requires_grad:
            y = _FlipFunction.apply(x, self)
        else:
            y, _ = self._product(x)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def extra_repr(self):
        s = self.spec
        frac = f", e0m3={float(self.mask.float().mean()):.3f}" if self.mask is not None else ""
        return (f"{self.in_features}->{self.out_features}, w={s.weight}, a={s.act}/{s.act_scope}, "
                f"mode={s.mode}{'/' + self._backend.name if self._backend else ''}{frac}")
