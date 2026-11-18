"""The sm_120 deployment path (RTX PRO 6000 Blackwell, RTX 5090): the FlipQuant paper's native kernels, optimized,
behind the model registry and the tile maps. The kernels are vendored in ``kernels/sm120`` (``VENDORED.json``,
``PORT_CHANGES.md``). ``--mode native`` on an sm_120 GPU runs this path (``--native-backend auto``, the default, or
``sm120``).

``install(model, spec, ...)`` replaces every :func:`models.loader.quantizable_linears` module by
``mixfp4_sm120.linear.NativeLinear``, which:

* holds the weight **pre-packed once**: FP4 nibbles plus UE4M3 scales whose bit 7 tags the scale block's format
  (set = E0M3), stored in the kernel's swizzled scale-factor layout. At install time the packed weight is checked bit
  for bit against this repository's own fake-quant weight (``flipquant.formats.fake_quant``). With ``ownership=True``
  (``--ownership-check``) every element is also decoded by the kernel itself and must equal its stored value under the
  map's format (maps only: the baselines' packed weights are covered by the bitwise check);
* picks a **kernel per call** from a width-selecting set, a per-GPU tile table (``kernels/sm120/configs``) and its
  per-call scheduler rows;
* runs the **fused activation quantizer** (per-token FourOverSix, or per-token NVFP4) and the **fused GEMM epilogue**
  (per-token and global scales, bias, BF16 output ``[T, out]`` in one rounding). It allocates nothing per call, so a
  forward can be captured in a CUDA graph.

Kernel sets (``kernel=``, ``--kernel-set``):

=================  ======================================================================================================
``auto``           The default: the optimized sets. 8x64 maps run on ``mixed_wB_ko`` (weights on B), 16x64 maps on
                   ``mixed_ko``, 256x64 maps on ``mixed256_ko``, and FourOverSix / NVFP4 weights on ``stock_ko``, each
                   with its optimized dispatch variant (``OPTIMIZED_DEFINES``). A set that is not built, or built without
                   its dispatch variant, is an error: run ``kernels/sm120/build_deploy.sh``. Nothing falls back.
``auto_stock_wB``  The FourOverSix / NVFP4 baselines with the weights on B (``stock_wB_ko``), the reference of the 8x64
                   maps.
``paper``          The paper's kernels: ``n8k64_wB`` for 8x64, ``mixed`` (paper table) for 16x64 and 256x64, ``stock``
                   for the baselines.
other              A ``mixfp4_sm120.select.FAMILIES`` set (e.g. ``mixed256``) or a ``mixfp4_sm120.configs.CONFIGS``
                   build name.
=================  ======================================================================================================

**Build directory:** ``build_root=`` (``--kernel-build-dir``), else ``$SM120_BUILD_DIR``, else ``kernels/sm120/
build_deploy``, which ``kernels/sm120/build_deploy.sh`` writes: the optimized sets and the paper's builds. The install
report records the directory and, in ``routing``, the set each route resolved to.

**The optimized dispatch variants are build variants** (the defines in ``OPTIMIZED_DEFINES``): the frequency-aware
dispatch (``MIXFP4_DISPATCH_FREQ=1``) for ``mixed_ko`` and ``mixed_wB_ko``; the uniform-branch dispatch
(``MIXFP4_UNIFORM_DISPATCH=1``) at ``mixed_ko``'s widths 64 and 128 and at every width of ``mixed256_ko``; the pipelined
flag read (``MIXFP4_PIPE_FLAGS=1``) at ``mixed_wB_ko``'s widths 64, '128x64' and 128. Every set computes the same
outputs bit for bit.

Only per-token activations are supported (``act_scope="row"``, the paper's convention); the activation format is
FourOverSix for maps and FourOverSix / NVFP4 baselines, or NVFP4 with NVFP4 weights if ``act="nvfp4"``.

**FourOverSix's tensor-scale constant** (``fo6_tensor_qmax``, ``--fo6-tensor-qmax``; ``flipquant.formats``): 448 (the
default) runs everything above unchanged. Another value (256: the original FourOverSix) is for the FourOverSix
baseline only: the weights are ``formats.fourover6(w, tensor_qmax=Q)``'s codes, packed by :func:`pack_codes`, and the
layers are :func:`tensor_qmax_native`'s ``TensorQmaxNativeLinear``, whose activations go through NativeLinear's unfused
branch, i.e. the vendored Triton per-token quantizer (the fused CUDA quantizer's fallback, bit-identical to it) with
the global-scale reciprocal fp32(1 / (6 Q)) in place of its fp32(1 / 2688). The fused CUDA quantizer has 1/2688
compiled into the kernel libraries, so it serves only 448; the GEMM is the same build's.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path

import numpy as np

import torch
from torch import nn

from flipquant import formats as F
from flipquant import maps as M

from .loader import _set_module, quantizable_linears
from .registry import ModelSpec

KERNELS = Path(__file__).resolve().parents[1] / "kernels" / "sm120"
DEPLOY_DIR = KERNELS / "build_deploy"            # kernels/sm120/build_deploy.sh's default output
KIND = {"mixfp4": "map", "fourover6": "four_over_six", "nvfp4": "nvfp4"}
ACT = {"fourover6": "four_over_six_rows", "nvfp4": "nvfp4_rows"}
BUILD_HINT = "build them with: bash kernels/sm120/build_deploy.sh [DIR] (default kernels/sm120/build_deploy)"


class KernelSetError(ValueError):
    """A requested kernel set is not built, or not built as required."""


def _why(e) -> str:
    """The vendored loader's reason, without its build hint (it names the source repository's layout)."""
    return re.sub(r"; run: python \S*build\.py.*$", "", str(e))


def package():
    """The vendored runtime package ``mixfp4_sm120`` (kernels/sm120)."""
    if str(KERNELS) not in sys.path:
        sys.path.insert(0, str(KERNELS))
    import mixfp4_sm120
    from mixfp4_sm120 import artifact, lib, linear, select  # noqa: F401
    return mixfp4_sm120


def _ownership():
    """kernels/sm120/eval/ownership_check.py: every weight element decoded by the kernel itself (an identity activation
    operand) must equal its stored value, and the format the tensor core executed must be the map's."""
    if str(KERNELS / "eval") not in sys.path:
        sys.path.insert(0, str(KERNELS / "eval"))
    import ownership_check
    return ownership_check


def build_dir(build_root=None) -> Path:
    """The build directory: ``build_root``, else ``$SM120_BUILD_DIR``, else ``kernels/sm120/build_deploy``."""
    if build_root is not None:
        return Path(build_root)
    env = os.environ.get("SM120_BUILD_DIR")
    return Path(env) if env else DEPLOY_DIR


FREQ, UNIFORM, PIPE = "MIXFP4_DISPATCH_FREQ", "MIXFP4_UNIFORM_DISPATCH", "MIXFP4_PIPE_FLAGS"
# The optimized sets' dispatch variants, per width (kernels/sm120/build_deploy.sh)
OPTIMIZED_DEFINES = {
    "mixed_ko": {16: {FREQ: 1}, 32: {FREQ: 1}, 64: {FREQ: 1, UNIFORM: 1}, 128: {FREQ: 1, UNIFORM: 1}},
    "mixed_wB_ko": {16: {FREQ: 1}, 32: {FREQ: 1}, 64: {FREQ: 1, PIPE: 1}, "128x64": {FREQ: 1, PIPE: 1},
                    128: {FREQ: 1, PIPE: 1}},
    "mixed256_ko": {16: {UNIFORM: 1}, 32: {UNIFORM: 1}, 64: {UNIFORM: 1}, 128: {UNIFORM: 1}},
}


def _optimized(family, root):
    """An optimized set, built with its dispatch variants; anything else is a :class:`KernelSetError`."""
    R = package()
    try:
        kset = R.select.KernelSet(family, build_root=root)
    except R.lib.LibraryError as e:
        raise KernelSetError(f"the optimized kernel set {family!r} is not built in {root} ({_why(e)}); {BUILD_HINT}") from None
    want = OPTIMIZED_DEFINES.get(family)
    if want is not None:
        have = {w: (k.manifest.get("extra_defines") or {}) for w, k in kset.kernels.items()}
        if have != want:
            raise KernelSetError(f"{family} in {root} is not built with its optimized dispatch variants (defines per "
                                 f"width {have}, expected {want}); {BUILD_HINT}")
    return kset


def _explicit(name, root):
    """A named set (``mixfp4_sm120.select.FAMILIES``) or single build, from ``root``; missing is an error."""
    R = package()
    try:
        if name in R.select.FAMILIES:
            return R.select.KernelSet(name, build_root=root)
        return R.lib.Kernel.load(name, build_root=root)
    except R.lib.LibraryError as e:
        raise KernelSetError(f"kernel {name!r} is not built in {root} ({_why(e)}); {BUILD_HINT}") from None


def resolve_kernel(kernel: str, weight: str, unit, build_root=None):
    """``(kernel or kernel set, label, routing note)`` for a weight format and map unit. ``auto`` and
    ``auto_stock_wB`` resolve to the optimized sets and refuse anything else; ``paper`` to the paper's kernels; any
    other name is loaded as given. Raises :class:`KernelSetError` if the resolved set is not built as required."""
    root = build_dir(build_root)
    rows = M.parse_unit(unit)[0] if unit else None
    what = f"{M.unit_name(unit)} map" if unit else f"{weight} weights"
    if kernel == "auto_stock_wB":
        if weight == "mixfp4":
            raise ValueError("kernel 'auto_stock_wB' runs the FourOverSix / NVFP4 baselines; maps use 'auto' or 'paper'")
        family = "stock_wB_ko"
    elif kernel == "auto":
        family = ("stock_ko" if weight != "mixfp4" else "mixed_wB_ko" if rows == 8 else
                  "mixed256_ko" if rows % 128 == 0 else "mixed_ko")
    else:
        family = None
    if family is not None:
        kern = _optimized(family, root)
        variants = " with its optimized dispatch variants" if family in OPTIMIZED_DEFINES else ""
        return kern, family, f"{kernel}: {what} -> {family}{variants}, from {root}"
    if kernel == "paper":
        name = "stock" if weight != "mixfp4" else "n8k64_wB" if rows == 8 else "mixed"
        return _explicit(name, root), name, f"paper: {what} -> {name} (the paper's kernels), from {root}"
    return _explicit(kernel, root), kernel, f"{kernel}: as requested, from {root}"


def _packing(mask, unit, out_features):
    """The map as the packer and the kernels read it: 8x64 and 16x64 as they are; coarser units as their 16x64
    granules (as the reference exporter does: a 256x64 tile becomes its 16x64 granules, the last tile of a
    matrix whose rows are not a multiple of 256 keeping only its real rows)."""
    rows, cols = M.parse_unit(unit)
    if cols != 64 or rows % 8:
        raise ValueError(f"unit {rows}x{cols} has no sm120 kernel (8x64, 16x64 or a multiple of 16 rows x 64)")
    if rows in (8, 16):
        return mask, (rows, cols)
    if rows % 16:
        raise ValueError(f"unit {rows}x{cols}: rows must be 8 or a multiple of 16")
    return mask.repeat_interleave(rows // 16, 0)[:-(-out_features // 16)], (16, cols)


# ------------------------------------------------------------------------------------------------------------------
# This branch's additions, outside the vendored package (kernels/sm120 is unchanged): precomputed FP4 codes
# (GPTQ) packed directly, and the online Hadamard rotation of --rotate hadamard.

_ROTATED = None
_LAST_ROT: dict = {}


def pack_codes(name: str, codes, bias, weight: str, mask=None, type_block=None, device=None):
    """the vendored ``PackedWeight`` from precomputed codes (``flipquant.formats.FP4Codes``, e.g. GPTQ's), without
    re-quantizing anything. The encodings are the same: nibble = sign << 3 | magnitude index (E2M1 0, 0.5, ..., 6 or
    E0M3 0..7), element 2j in the low nibble; scale byte = the E4M3 block scale with bit 7 set on E0M3 blocks; one FP32
    global scale. Checked here, in place of pack_module's comparison with the RTN fake-quant weight (which GPTQ codes
    differ from by design):

    * the format flags: E0M3 exactly on the map's tiles (``mask`` at ``type_block``) for mixfp4, none otherwise;
    * the weight the kernel will decode (``numerics.decode_fake_order``: bf16((codebook * scale) * gs), the fake
      quantizers' order) equals ``codes.dequantize()``, the weight the fake path computes from the same codes, bit
      for bit."""
    R = package()
    N = R.numerics
    nib = codes.codes.to(device=device or codes.codes.device, dtype=torch.uint8)
    o, k = nib.shape
    if isinstance(codes.fmt, int):
        if codes.fmt != F.FMT_E2M1:
            raise ValueError(f"{name}: uniform E0M3 codes have no sm120 packing")
        flags = torch.zeros((o, k // N.SCALE_BLOCK), dtype=torch.bool, device=nib.device)
    else:
        flags = codes.fmt.to(nib.device).bool()
    if weight == "mixfp4" and mask is not None and bool(mask.any()):
        want = N.expand_mask(mask.to(nib.device), type_block, (o, k))
    else:
        want = torch.zeros_like(flags)
    if not torch.equal(flags, want):
        raise ValueError(f"{name}: the codes' E0M3 blocks ({int(flags.sum())}) are not the map's ({int(want.sum())})")
    sbytes = N.scale_bytes(codes.scale.to(nib.device).float(), flags)
    gs = codes.gscale.to(nib.device).float().reshape(())
    got = N.decode_fake_order(nib, sbytes, gs)
    expected = F.FP4Codes(nib, codes.scale.to(nib.device), codes.gscale.to(nib.device),
                          codes.fmt if isinstance(codes.fmt, int) else codes.fmt.to(nib.device)).dequantize(torch.bfloat16)
    if not torch.equal(got, expected):
        raise R.artifact.ArtifactError(f"{name}: the packed codes decode differently from the codes in "
                                       f"{int((got != expected).sum())} elements")
    return R.artifact.PackedWeight(name, N.pack_nibbles(nib), sbytes, float(gs),
                                   None if bias is None else bias.detach().to(torch.bfloat16).contiguous(),
                                   tuple(type_block) if type_block else None,
                                   int(mask.sum()) if mask is not None else 0)


def _rotate_shared(rotation, x):
    """``rotation(x)`` (flipquant.rotate.Hadamard: FP32 matmul per block, cast back), computed once per input tensor:
    q/k/v (and gate/up) receive the same input, so they get the same rotated tensor, and NativeLinear's shared
    activation quantization (its ``_LAST_QUANT``) then applies to them as without rotation. The key holds the input
    (so its address cannot be reused while cached) and its version counter, as ``_LAST_QUANT`` does."""
    key = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x._version, rotation.features, rotation.block,
           x.device, torch.cuda.is_current_stream_capturing() if x.is_cuda else False)
    last = _LAST_ROT.get(x.device)
    if last is not None and last[0] == key:
        return last[2]
    y = rotation(x)
    _LAST_ROT[x.device] = (key, x, y)
    return y


def rotated_native():
    """``RotatedNativeLinear``: the vendored NativeLinear with the online rotation ``x -> x H`` applied in Python
    before ``NativeLinear.forward``, i.e. before the single ctypes call that launches the fused activation quantizer
    and the GEMM; the quantizer therefore sees ``x H``, as the fake path's FlipLinear does."""
    global _ROTATED
    if _ROTATED is None:
        R = package()

        class RotatedNativeLinear(R.linear.NativeLinear):
            # ``rotation`` (a flipquant.rotate.Hadamard) is set after construction and registered as a submodule;
            # no class attribute of that name, which would shadow it

            def forward(self, x):
                return super().forward(_rotate_shared(self.rotation, x))

            def extra_repr(self):
                return super().extra_repr() + f", rotation={self.rotation.extra_repr()}"

        _ROTATED = RotatedNativeLinear
    return _ROTATED


def inv_qmax(tensor_qmax: float) -> float:
    """fp32 ``1 / (6 * tensor_qmax)``, computed as the vendored quantizer computes its ``INV_2688``."""
    return float(np.float32(1.0) / np.float32(F.global_qmax(tensor_qmax)))


@contextlib.contextmanager
def quantizer_inv_qmax(value: float):
    """The vendored Triton per-token quantizer (``mixfp4_sm120.quant_act.quantize``) with the global-scale reciprocal
    ``value`` while the block runs: ``quantize`` passes the module's ``INV_2688`` to the kernel as a constexpr at each
    call, so this changes that one number of the launch and nothing else."""
    QA = package().quant_act
    old = QA.INV_2688
    QA.INV_2688 = value
    try:
        yield
    finally:
        QA.INV_2688 = old


_TENSOR_QMAX = None


def tensor_qmax_native():
    """``NativeLinear`` for FourOverSix with another tensor-scale constant (module docstring). Construct it as
    NativeLinear, then set ``inv_qmax`` (:func:`inv_qmax`); ``fused`` is off, so the vendored forward takes its unfused
    branch (``quant_act.quantize``, then ``kernel.gemm`` with the same scales and bias), and the quantizer runs with
    ``inv_qmax``."""
    global _TENSOR_QMAX
    if _TENSOR_QMAX is None:
        R = package()

        class TensorQmaxNativeLinear(R.linear.NativeLinear):
            def __init__(self, *args, inv_qmax: float, **kwargs):
                super().__init__(*args, **kwargs)
                self.inv_qmax = inv_qmax
                self.fused = False

            def forward(self, x):
                with quantizer_inv_qmax(self.inv_qmax):
                    return super().forward(x)

            def extra_repr(self):
                return super().extra_repr() + f", inv_qmax={self.inv_qmax!r} (unfused quantizer)"

        _TENSOR_QMAX = TensorQmaxNativeLinear
    return _TENSOR_QMAX


@torch.no_grad()
def install(model: nn.Module, spec: ModelSpec, *, weight: str = "mixfp4", tiles: dict | None = None, unit=None,
            kernel: str = "auto", act: str = "fourover6", act_scope: str = "row", build_root=None,
            check: bool = True, ownership: bool = False, activation_quantizer: str | None = None,
            codes: dict | None = None, linears: dict | None = None, rotations: dict | None = None,
            any_act: bool = False, fo6_tensor_qmax: float = F.E4M3_MAX) -> dict:
    """Replace every quantizable Linear by the vendored NativeLinear. Returns the install report: ``kernel`` (the set
    a route resolved to), ``routing`` (how, and from which build directory), ``build_dir``, ``kernel_set`` (the loaded
    builds and their tile table).

    ``activation_quantizer`` ('nvfp4_rows' / 'four_over_six_rows') overrides the activation quantizer, for latency
    measurements only. The maps were calibrated with FourOverSix activations, so an overridden install is not an
    evaluated configuration; the report records it.

    This branch's additions (all off by default):

    * ``codes`` ``{name: FP4Codes}``: precomputed codes (GPTQ) packed by :func:`pack_codes` instead of quantizing the
      weight; ``linears`` then names the modules to replace (e.g. the FlipLinears GPTQ left behind). The ownership
      check runs on every mixfp4 module of a codes install.
    * ``rotations`` ``{name: flipquant.rotate.Hadamard}``: those modules become :func:`rotated_native`'s
      ``RotatedNativeLinear`` (the weight packed is the module's, i.e. ``W H`` for a RotatedLinear or for GPTQ codes
      computed on one).
    * ``any_act``: allow NVFP4 weights with FourOverSix activations (Table 4's GPTQ setting) on the stock kernels.
    * ``fo6_tensor_qmax``: FourOverSix's tensor-scale constant (module docstring); anything but 448 is the FourOverSix
      baseline (weight and act ``fourover6``) only, with no codes, rotations or quantizer override."""
    R = package()
    if act_scope != "row":
        raise ValueError("the sm120 deployment path quantizes activations per token: use act_scope='row'")
    if weight not in KIND:
        raise ValueError(f"weight {weight!r}: one of {sorted(KIND)}")
    if act not in ACT or ((act == "nvfp4") != (weight == "nvfp4") and not (any_act and weight == "nvfp4")):
        raise ValueError("activations: fourover6 with mixfp4 / fourover6 weights, nvfp4 with nvfp4 weights")
    if weight == "mixfp4" and (tiles is None or unit is None):
        raise ValueError("mixfp4 needs a map (tiles and unit)")
    act_kind = ACT[act] if activation_quantizer is None else activation_quantizer
    if act_kind not in ACT.values():
        raise ValueError(f"activation quantizer {act_kind!r}: one of {sorted(ACT.values())}")
    tq = fo6_tensor_qmax != F.E4M3_MAX
    if tq and (weight, act) != ("fourover6", "fourover6"):
        raise ValueError(f"fo6_tensor_qmax {fo6_tensor_qmax}: the FourOverSix baseline only (weight and act fourover6)")
    if tq and (codes is not None or rotations is not None or activation_quantizer is not None):
        raise ValueError("fo6_tensor_qmax: not with given codes, rotations or an activation-quantizer override")
    inv = inv_qmax(fo6_tensor_qmax) if tq else None
    kern, label, routing = resolve_kernel(kernel, weight, unit if weight == "mixfp4" else None, build_root)
    probe = kern.primary if isinstance(kern, R.select.KernelSet) else kern     # the ownership check's kernel
    linears = quantizable_linears(model, spec) if linears is None else dict(linears)
    if codes is not None and set(codes) != set(linears):
        raise ValueError(f"codes for {len(codes)} modules, {len(linears)} to install")
    if rotations is not None and not set(rotations) <= set(linears):
        raise ValueError("rotations for modules that are not installed")
    if weight == "mixfp4":
        missing = sorted(set(linears) - set(tiles))
        extra = sorted(set(tiles) - set(linears))
        if missing or extra:
            raise ValueError(f"map does not cover the quantizable linears: missing {missing[:3]}, extra {extra[:3]}")
    report = dict(backend="sm120", kernel=label, routing=routing, build_dir=str(build_dir(build_root)), weight=weight,
                  unit=M.unit_name(unit) if unit else None,
                  activation_quantizer=act_kind, activation_quantizer_override=act_kind != ACT[act],
                  modules=0, e0m3_tiles=0, checked=check, ownership={},
                  kernel_set=kern.describe() if isinstance(kern, R.select.KernelSet) else dict(kernel=kern.cfg.name,
                                                                                            sha256=kern.sha256),
                  device_bytes=dict(packed=0, scales_placed=0, bias=0))
    if codes is not None:
        report["codes"] = "precomputed (pack_codes: flags and decode checked against the codes)"
    if tq:
        report["fo6_tensor_qmax"] = dict(
            tensor_qmax=fo6_tensor_qmax, global_scale=f"amax / {F.global_qmax(fo6_tensor_qmax):g}", inv_qmax=inv,
            weights="formats.fourover6(w, tensor_qmax) codes, packed by pack_codes (decode checked bit for bit)",
            activations="TensorQmaxNativeLinear: the vendored Triton per-token quantizer with inv_qmax, then the GEMM "
                        "(NativeLinear's unfused branch)")
    if rotations is not None:
        report["rotation"] = dict(modules=len(rotations), blocks=sorted({r.block for r in rotations.values()}),
                                  where="RotatedNativeLinear.forward, before NativeLinear's fused activation quantizer")
    for name, lin in linears.items():
        w = None if codes is not None else lin.weight.data
        dev = w.device if w is not None else next(t.device for t in (*lin.parameters(), *lin.buffers()))
        mask = tb = None
        if weight == "mixfp4":
            rows, cols = M.parse_unit(unit)
            mask, tb = _packing(tiles[name].to(dev), unit, lin.out_features)
        if codes is not None:
            pw = pack_codes(name, codes[name], lin.bias, weight, mask, tb, device=dev)
        elif tq:
            pw = pack_codes(name, F.fourover6(w, tensor_qmax=fo6_tensor_qmax), lin.bias, weight, device=dev)
        else:
            expected = None
            if check:
                if weight == "nvfp4":
                    expected = F.fake_quant(w, "nvfp4")
                else:
                    expected = F.fake_quant(w, "fourover6")
                    if weight == "mixfp4" and bool(tiles[name].any()):
                        sel = M.expand(tiles[name].to(w.device), rows, cols, lin.out_features)
                        expected = torch.where(sel, F.fake_quant(w, "e0m3"), expected)
            pw = R.artifact.pack_module(name, w, lin.bias, KIND[weight], mask, tb, expected=expected)
        if (ownership or codes is not None) and weight == "mixfp4":
            own = _ownership().check_module(probe, pw, mask, chunk=1024)
            if not own["exact"] or own["format_mismatches"]:
                raise RuntimeError(f"{name}: ownership check failed: {own}")
            report["ownership"][name] = own
        if rotations is not None and name in rotations:
            nl = rotated_native()(pw, kern, act_kind, name=name)
            nl.rotation = rotations[name]
        elif tq:
            nl = tensor_qmax_native()(pw, kern, act_kind, name=name, inv_qmax=inv)
        else:
            nl = R.linear.NativeLinear(pw, kern, act_kind, name=name)
        _set_module(model, name, nl)
        report["modules"] += 1
        report["e0m3_tiles"] += pw.e0m3_tiles
        report["device_bytes"]["packed"] += nl.packed.numel()
        report["device_bytes"]["scales_placed"] += nl.sf.numel()
        report["device_bytes"]["bias"] += 0 if nl.bias_bf16 is None else nl.bias_bf16.numel() * 2
        del lin
    linears.clear()
    torch.cuda.empty_cache()
    return report


def native_modules(model: nn.Module) -> dict:
    R = package()
    return {n: m for n, m in model.named_modules() if isinstance(m, R.linear.NativeLinear)}


def coverage(model: nn.Module) -> dict:
    """Which NativeLinears ran since the counters were reset."""
    nat = native_modules(model)
    return dict(native=len(nat), native_called=sum(1 for m in nat.values() if m.calls > 0),
                calls=sum(m.calls for m in nat.values()), tokens=sum(m.tokens for m in nat.values()))


def reset_counters(model: nn.Module) -> None:
    for m in native_modules(model).values():
        m.calls = 0
        m.tokens = 0
