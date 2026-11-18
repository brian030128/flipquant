"""The tuned SM100 (B200 / GB200) native path: every quantizable Linear runs the tuned 256x256x256 2-SM CUTLASS
kernel of ``kernels/src/fq_sm100_linear.cu`` (``libflipquant_sm100_linear.so``) -- the stock NVFP4 kernel for NVFP4 and
FourOverSix weights, the deployed per-tile kernel (one B format per 256x64 tile, a 1-bit map) for FlipQuant.

This is the SM100 counterpart of :mod:`models.sm120` (``--native-backend sm100-tuned``); the older ``sm100`` backend of
:mod:`flipquant.native` runs the N8xK64 C API kernel instead. It reuses the vendored, architecture-independent pieces of
``mixfp4_sm120``: the weight quantizers and packing (checked bit for bit against the fake-quant weights), the Triton
per-token activation quantizer, and the scale-factor placement -- CUTLASS's SM1xx block-scaled layout, the same atom on
SM100 (checked against the library's own layout function at load time).

Per call: the Triton quantizer (FP32 global scale per token, E2M1 codes, UE4M3 scales in the SFA layout), then one GEMM
with the fused epilogue ``D[t, n] = bf16((gs_x[t] * gs_w) * acc + bias[n])`` (weights on B). q/k/v and gate/up reuse
one quantized input. No allocation besides torch's caching allocator, so a forward is CUDA-graph capturable.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
from torch import nn

from flipquant import formats as F
from flipquant import maps as M

from . import sm120 as S120
from .loader import quantizable_linears

#: --weight -> the weight quantizer of mixfp4_sm120.numerics
KIND = {"nvfp4": "nvfp4", "fourover6": "four_over_six", "mixfp4": "map"}
#: --act -> the activation quantizer of mixfp4_sm120.quant_act
ACT = {"nvfp4": "nvfp4_rows", "fourover6": "four_over_six_rows"}
UNIT = (256, 64)

_LIB = None


def library_path() -> Path:
    """``$FLIPQUANT_SM100_LINEAR_LIB`` (the .so or its directory), else ``$FLIPQUANT_KERNEL_LIB``."""
    for var in ("FLIPQUANT_SM100_LINEAR_LIB", "FLIPQUANT_KERNEL_LIB"):
        value = os.environ.get(var)
        if value:
            p = Path(value)
            return p / "libflipquant_sm100_linear.so" if p.is_dir() else p
    raise RuntimeError("set FLIPQUANT_SM100_LINEAR_LIB to libflipquant_sm100_linear.so "
                       "(kernels: TARGETS=flipquant_sm100_linear scripts/build_sm100.sh)")


def library():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(library_path()))
        p, i, f, z = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_size_t
        for variant in ("stock", "tile"):
            gemm = getattr(lib, f"fq_sm100_{variant}_gemm")
            gemm.argtypes = [p, p, p, p, p, p, f, p, p, i, i, i, p, z, p]
            gemm.restype = i
            ws = getattr(lib, f"fq_sm100_{variant}_workspace_size")
            ws.argtypes = [i, i, i]
            ws.restype = z
        lib.fq_sm100_sf_bytes.argtypes = [i, i]
        lib.fq_sm100_sf_bytes.restype = z
        lib.fq_sm100_sf_offset.argtypes = [i, i, i, i]
        lib.fq_sm100_sf_offset.restype = ctypes.c_int64
        lib.fq_sm100_quant_rows.argtypes = [p, ctypes.c_int64, i, i, i, p, p, p, p]
        lib.fq_sm100_quant_rows.restype = i
        _check_sf_layout(lib)
        _LIB = lib
    return _LIB


def _check_sf_layout(lib):
    """The vendored placement formula must be CUTLASS's SM100 layout (rows and K tails included)."""
    lin = S120.package().lib
    for rows, k in ((1, 64), (48, 4480), (130, 1344), (2048, 4096), (4480, 15680)):
        if lin.sf_buffer_size(rows, k) != lib.fq_sm100_sf_bytes(rows, k):
            raise RuntimeError(f"SF buffer size differs at rows={rows} k={k}")
        for r in sorted({0, 31, 32, 127, 128, rows - 1}):
            if r >= rows:
                continue
            for j in sorted({0, 3, 4, k // 16 - 1}):
                if lin.sf_offset_formula(r, j, k) != lib.fq_sm100_sf_offset(rows, k, r, j):
                    raise RuntimeError(f"SF offset differs at rows={rows} k={k} r={r} j={j}")


def map_words(tiles: torch.Tensor, k: int) -> torch.Tensor:
    """A [ceil(N/256), K/64] bool map (True = E0M3) -> the kernel's words: int32 [rows, ceil(K/2048)], bit j of word w
    = K64 block 32w + j."""
    rows, cols = tiles.shape
    if cols != k // 64:
        raise ValueError(f"map has {cols} K64 blocks, the weight {k // 64}")
    words = (cols + 31) // 32
    bits = torch.zeros((rows, words * 32), dtype=torch.int64, device=tiles.device)
    bits[:, :cols] = tiles.to(torch.int64)
    packed = (bits.view(rows, words, 32) << torch.arange(32, device=tiles.device)).sum(-1)
    return torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32).contiguous()


class Sm100Linear(nn.Module):
    """An nn.Linear replacement on the tuned SM100 kernels. ``calls`` / ``tokens`` count native GEMMs."""

    _last = {}   # the last quantized input per (device, stream, activation quantizer): q/k/v and gate/up share it

    def __init__(self, pw, act_kind: str, tiles: torch.Tensor | None, name: str = ""):
        super().__init__()
        R = S120.package()
        n, k = pw.shape
        if k % 64:
            raise ValueError(f"{name}: in_features {k} is not a multiple of 64")
        self.name, self.out_features, self.in_features, self.act_kind = name, n, k, act_kind
        self.variant = "tile" if tiles is not None else "stock"
        self.register_buffer("packed", pw.packed.contiguous(), persistent=False)
        # SM100 reads the format from the map, not from bit 7 of the scale byte (SM120's E0M3 tag).
        self.register_buffer("sf", R.linear.place_scales(pw.scales & 0x7F, k), persistent=False)
        self.register_buffer("map", None if tiles is None else map_words(tiles.to(pw.packed.device), k),
                             persistent=False)
        self.global_scale = float(pw.global_scale)
        self.register_buffer("bias_bf16", None if pw.bias is None else pw.bias.contiguous(), persistent=False)
        self.calls = self.tokens = self.quant_reused = 0
        self.kernel_set = None        # one tuned tile; no per-shape kernel table on SM100
        self._lib = library()
        self._gemm = getattr(self._lib, f"fq_sm100_{self.variant}_gemm")
        self._workspace = getattr(self._lib, f"fq_sm100_{self.variant}_workspace_size")

    def _quantize(self, x, x2d):
        """q/k/v (and gate/up) receive the same input tensor ``x``: quantize it once. The entry keeps ``x`` alive, so
        its storage cannot be reused under the same identity, and an in-place update changes its version."""
        R = S120.package()
        key = (x.device, torch.cuda.current_stream().cuda_stream, self.act_kind, self.in_features)
        hit = Sm100Linear._last.get(key)
        if hit is not None and hit[0] is x and hit[1] == x._version:
            self.quant_reused += 1
            return hit[2]
        q = quantize_rows(x2d, self.act_kind)
        Sm100Linear._last[key] = (x, x._version, q)
        return q

    def forward(self, x):
        shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        t = x2d.shape[0]
        packed_a, sfa, gs_x = self._quantize(x, x2d)
        d = torch.empty((t, self.out_features), dtype=torch.bfloat16, device=x.device)
        ws_bytes = self._workspace(t, self.out_features, self.in_features)
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=x.device) if ws_bytes else None
        status = self._gemm(packed_a.data_ptr(), sfa.data_ptr(), self.packed.data_ptr(), self.sf.data_ptr(),
                            None if self.map is None else self.map.data_ptr(), gs_x.data_ptr(), self.global_scale,
                            None if self.bias_bf16 is None else self.bias_bf16.data_ptr(), d.data_ptr(),
                            t, self.out_features, self.in_features, None if ws is None else ws.data_ptr(), ws_bytes,
                            torch.cuda.current_stream().cuda_stream)
        if status != 0:
            raise RuntimeError(f"{self.name}: fq_sm100_{self.variant}_gemm failed ({status}) at T={t}")
        self.calls += 1
        self.tokens += t
        return d.view(*shape[:-1], self.out_features)

    def extra_repr(self):
        return f"{self.in_features} -> {self.out_features}, {self.variant} kernel, act={self.act_kind}"


#: activation quantizer kinds -> fq_sm100_quant_rows modes (as mixfp4_sm120.quant_act.MODES); the ``_fast`` kinds are
#: the same rules on fq_sm100_quant_rows_fast (kernels/src/quant_act_fast.cuh: not bit-identical to the simulation)
MODES = {"nvfp4_rows": 0, "four_over_six_rows": 1, "nvfp4_rows_fast": 0, "four_over_six_rows_fast": 1}


def _fast_quantizer(lib):
    fn = getattr(lib, "_fq_quant_fast", None)
    if fn is None:
        fn = lib.fq_sm100_quant_rows_fast
        p, i = ctypes.c_void_p, ctypes.c_int
        fn.argtypes = [p, ctypes.c_int64, i, i, i, p, p, p, p]
        fn.restype = i
        lib._fq_quant_fast = fn
    return fn


def quantize_rows(x2d: torch.Tensor, act_kind: str):
    """bf16 [T, K] (unit stride along K) -> (packed [T, K/2] uint8, SFA buffer uint8, gs [T] fp32) with SM120's CUDA
    quantizer from the library (``FLIPQUANT_SM100_TRITON_QUANT=1``: the Triton one, bit-identical and slower), or for
    a ``_fast`` kind the fast quantizer."""
    fast = act_kind.endswith("_fast")
    if not fast and os.environ.get("FLIPQUANT_SM100_TRITON_QUANT") == "1":
        return S120.package().quant_act.quantize(x2d, act_kind)
    if x2d.stride(1) != 1:
        x2d = x2d.contiguous()
    t, k = x2d.shape
    lib = library()
    packed = torch.empty((t, k // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty(lib.fq_sm100_sf_bytes(t, k), dtype=torch.uint8, device=x2d.device)
    gs = torch.empty(t, dtype=torch.float32, device=x2d.device)
    quant = _fast_quantizer(lib) if fast else lib.fq_sm100_quant_rows
    status = quant(x2d.data_ptr(), x2d.stride(0), t, k, MODES[act_kind], packed.data_ptr(), sf.data_ptr(),
                   gs.data_ptr(), torch.cuda.current_stream().cuda_stream)
    if status != 0:
        raise RuntimeError(f"fq_sm100_quant_rows{'_fast' if fast else ''} failed ({status}) for {t}x{k}")
    return packed, sf, gs


class _TileProbe:
    """The per-tile kernel in the interface of kernels/sm120/eval/ownership_check.py (``weight_operand`` 1 = weights on
    B): the activation operand it passes is the exact identity, and with unit global scales and no bias the GEMM
    returns every weight element as the kernel decoded it. The format comes from the 256x64 map words, as in
    deployment; the tag bit 7 of the scales SM120 reads is cleared."""

    weight_operand = 1

    def __init__(self, tiles: torch.Tensor, k: int):
        self.map = map_words(tiles, k)
        self.lib = library()

    def gemm(self, a, sfa, b, sfb, t, n, k):
        sfb = sfb & 0x7F
        gs_x = torch.ones(t, dtype=torch.float32, device=a.device)
        d = torch.empty((t, n), dtype=torch.bfloat16, device=a.device)
        ws_bytes = self.lib.fq_sm100_tile_workspace_size(t, n, k)
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=a.device) if ws_bytes else None
        status = self.lib.fq_sm100_tile_gemm(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(),
                                             self.map.data_ptr(), gs_x.data_ptr(), 1.0, None, d.data_ptr(), t, n, k,
                                             None if ws is None else ws.data_ptr(), ws_bytes,
                                             torch.cuda.current_stream().cuda_stream)
        if status != 0:
            raise RuntimeError(f"fq_sm100_tile_gemm failed ({status}) in the ownership check at T={t}")
        return d


def clear_quant_cache():
    Sm100Linear._last.clear()


def _set_module(model, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


@torch.no_grad()
def install(model: nn.Module, spec, *, weight: str = "mixfp4", tiles: dict | None = None, unit=None,
            act: str = "fourover6", activation_quantizer: str | None = None, check: bool = True,
            ownership: bool = False) -> dict:
    """Replace every quantizable Linear by :class:`Sm100Linear`. mixfp4 needs a 256x64 map (the deployed unit).

    ``ownership`` (mixfp4): every weight element is decoded by the per-tile kernel itself (an identity activation,
    :class:`_TileProbe`) and compared with its stored value under the map's format, and the format the kernel executed
    with the map's tile (models.sm120's check, kernels/sm120/eval/ownership_check.py). The counts are recorded per module
    in ``report["ownership"]`` and summed in ``report["ownership_total"]``; a mismatch does not stop the install."""
    R = S120.package()
    if weight not in KIND:
        raise ValueError(f"weight {weight!r}: one of {sorted(KIND)}")
    if act not in ACT or (act == "nvfp4") != (weight == "nvfp4"):
        raise ValueError("activations: fourover6 with mixfp4 / fourover6 weights, nvfp4 with nvfp4 weights")
    act_kind = ACT[act] if activation_quantizer is None else activation_quantizer
    if act_kind not in MODES:
        raise ValueError(f"activation quantizer {act_kind!r}: one of {sorted(MODES)}")
    if weight == "mixfp4":
        if tiles is None or unit is None or tuple(M.parse_unit(unit)) != UNIT:
            raise ValueError("mixfp4 on SM100 runs the deployed 256x64 map (--map ...flipquant_256x64.pt)")
    linears = quantizable_linears(model, spec)
    if weight == "mixfp4":
        missing, extra = sorted(set(linears) - set(tiles)), sorted(set(tiles) - set(linears))
        if missing or extra:
            raise ValueError(f"map does not cover the quantizable linears: missing {missing[:3]}, extra {extra[:3]}")
    library()
    report = dict(backend="sm100-tuned", library=str(library_path()), weight=weight, unit="256x64" if tiles else None,
                  kernel="per-tile 256x64" if weight == "mixfp4" else "stock", activation_quantizer=act_kind,
                  activation_quantizer_override=act_kind != ACT[act], modules=0, e0m3_tiles=0, checked=check,
                  device_bytes=dict(packed=0, scales_placed=0, map=0, bias=0))
    ownership = ownership and weight == "mixfp4"
    if ownership:
        report["ownership"] = {}
    for name, lin in list(linears.items()):
        w = lin.weight.data
        n, k = w.shape
        mask = rows = None
        if weight == "mixfp4":
            tile = tiles[name].to(w.device).to(torch.bool)
            if tuple(tile.shape) != ((n + 255) // 256, k // 64):
                raise ValueError(f"{name}: map {tuple(tile.shape)} for a {n}x{k} weight")
            rows = tile.repeat_interleave(256, 0)[:n]          # one row per output channel: partial last tile
        expected = None
        if check:
            expected = F.fake_quant(w, "nvfp4" if weight == "nvfp4" else "fourover6")
            if weight == "mixfp4" and bool(tile.any()):
                sel = M.expand(tile, *UNIT, n)
                expected = torch.where(sel, F.fake_quant(w, "e0m3"), expected)
        pw = R.artifact.pack_module(name, w, lin.bias, KIND[weight], rows, (1, 64) if rows is not None else None,
                                    expected=expected)
        if ownership:
            own = S120._ownership().check_module(_TileProbe(tile, k), pw, rows, chunk=1024)
            own["e0m3_tiles"] = int(tile.sum())                   # 256x64 tiles (the mask passed is per row)
            report["ownership"][name] = own
            if "negative_control" not in report and bool(tile.any()):
                # the kernel given an all-E2M1 map for this module must be caught
                neg = S120._ownership().check_module(_TileProbe(torch.zeros_like(tile), k), pw, rows, chunk=1024)
                report["negative_control"] = dict(module=name, map="all E2M1", value_mismatches=neg["value_mismatches"],
                                                  format_mismatches=neg["format_mismatches"],
                                                  detected=not neg["exact"] and neg["format_mismatches"] > 0)
        nl = Sm100Linear(pw, act_kind, tile if weight == "mixfp4" else None, name=name)
        _set_module(model, name, nl)
        report["modules"] += 1
        report["e0m3_tiles"] += int(tile.sum()) if weight == "mixfp4" else 0
        report["device_bytes"]["packed"] += nl.packed.numel()
        report["device_bytes"]["scales_placed"] += nl.sf.numel()
        report["device_bytes"]["map"] += 0 if nl.map is None else nl.map.numel() * 4
        report["device_bytes"]["bias"] += 0 if nl.bias_bf16 is None else nl.bias_bf16.numel() * 2
        del lin
    linears.clear()
    torch.cuda.empty_cache()
    if ownership:
        rows_ = report["ownership"].values()
        report["ownership_total"] = {key: sum(r[key] for r in rows_) for key in (
            "value_mismatches", "elements", "format_mismatches", "informative_elements", "e0m3_elements_observed",
            "e0m3_tiles")}
        report["ownership_total"].update(modules=len(rows_), all_exact=all(r["exact"] for r in rows_),
                                         failing_modules=sorted(n for n, r in report["ownership"].items()
                                                                if not r["exact"] or r["format_mismatches"]))
    return report


def native_modules(model: nn.Module) -> dict:
    return {n: m for n, m in model.named_modules() if isinstance(m, Sm100Linear)}


def coverage(model: nn.Module) -> dict:
    nat = native_modules(model)
    return dict(native=len(nat), native_called=sum(1 for m in nat.values() if m.calls > 0),
                calls=sum(m.calls for m in nat.values()), tokens=sum(m.tokens for m in nat.values()))


def reset_counters(model: nn.Module) -> None:
    for m in native_modules(model).values():
        m.calls = m.tokens = m.quant_reused = 0
