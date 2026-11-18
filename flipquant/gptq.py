"""GPTQ on a fixed FP4 grid (``--ptq gptq``), for NVFP4, FourOverSix and MixFP4 weights.

**The grid is fixed by RTN of the original weight, and GPTQ only re-rounds on it.** Every E4M3 block
scale (one per 16 elements), the per-tensor global scale and -- for FourOverSix -- each block's 4-or-6
choice (which *is* its block scale: block max / 4 or / 6) come from the format's own RTN quantizer
(:mod:`flipquant.formats`) applied to the unmodified weight, and are not recomputed. GPTQ then walks
the columns, rounds each updated column to the nearest point of that column's fixed grid with the
format's own rounding rule (E2M1: FourOverSix ties down, NVFP4 ties up; E0M3: round-half-even, clamp
at 7), and spreads the rounding error over the not-yet-quantized columns. Why fixed:
  * the codes run unchanged on the native kernels: a re-quantization of GPTQ's dequantized output would
    recompute the scales (block max of the new values) and land on different codes;
  * scale groups (16), map tiles (e.g. 16x64) and FourOverSix's choices stay those of RTN, so the
    GPTQ rows of the table differ from the RTN rows only in rounding;
  * with no correlation between inputs (diagonal Hessian) GPTQ returns exactly the RTN codes
    (tests/test_gptq.py).
MixFP4: the map is learned with RTN candidates (calibration/train_map.py) and then fixed; each tile is
GPTQ-rounded on the grid of its format (E0M3 tile: candidate A's scales and E0M3 values; E2M1 tile:
FourOverSix's).

Algorithm per linear (Frantar et al. 2023), with the defaults of NVIDIA ModelOpt 0.47's GPTQ
(``modelopt.torch.quantization.utils.calib_utils``: ``update_hessian``, ``compute_hessian_inverse``,
``gptq_blockwise_update``; the recipe of calibration/gptq_modelopt.py): ``H = X^T X / n`` over the
calibration tokens, where ``X`` is the input the weight multiplies -- after the Hadamard rotation if any,
and by default (``hessian_input="quantized"``) after the activation quantizer, per token, as ModelOpt's
Hessian hook applies the module's ``input_quantizer``; ``"bf16"`` takes the input before activation
quantization. Dead columns are ModelOpt's: an all-zero weight column gets a zero row and column in ``H``
and ``H_ii = 1``; then dampening ``0.01 * mean(diag H)``, the upper Cholesky factor of ``H^-1`` (identity
if ``H`` is not positive definite, as ModelOpt), lazy batch updates in blocks of 128 columns, columns in K
order (no act-order; ``act_order=True`` is the option).

Model-level (:func:`quantize_model`): standard sequential GPTQ over the decoder layers. The calibration
windows (``flipquant.data.calibration_sets``: 128 x 512 math+code, the map's fit set) go through the
BF16 model once to record each layer's call arguments (masks, positions, ...; hybrid models give
Mamba / linear-attention layers different ones); then layer by layer: the Hessians of all its
quantizable linears are collected in one pass over the layer's inputs (with this layer still BF16;
not "true sequential" inside a layer), each linear is GPTQ'd and replaced by
:meth:`FlipLinear.from_codes`, and the *quantized* layer is re-run to give the next layer's inputs.
The quantized layers run as they will be evaluated (weight format, activation format, fake or native)
except that activations use the per-token (``row``) scope during calibration, as in train_map, so the
codes do not depend on the evaluation's activation scope. Every quantizable linear of the model --
Mamba ``in_proj`` / ``out_proj`` and Gated-DeltaNet projections included -- must sit inside a decoder
layer; anything else is an error.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from . import formats as fmt
from . import maps as maps_
from .linear import FlipLinear, QuantSpec

PTQ_METHODS = ("rtn", "gptq")


@dataclass
class Grid:
    """The fixed quantization grid of one ``[O, K]`` weight."""

    scale: torch.Tensor          #: [O, K/16] fp32, E4M3-representable block scales
    gscale: torch.Tensor         #: fp32 scalar
    e0m3: torch.Tensor | None    #: [O, K/16] bool, True = E0M3 block; None = uniform E2M1
    ties: str                    #: E2M1 rounding: "down" (FourOverSix) or "up" (NVFP4)

    @property
    def fmt(self):
        return fmt.FMT_E2M1 if self.e0m3 is None else self.e0m3

    def quantize(self, w: torch.Tensor, k0: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Round columns ``k0 .. k0 + w.shape[1]`` (``w`` fp32 ``[O, n]``, within one 16-group or
        not) to the grid. Returns ``(codes uint8, dequantized fp32)``, with the quantizers' exact
        operation order: ``y = (w / g) / s``, ``value = (v * s) * g``."""
        n = w.shape[1]
        groups = torch.arange(k0, k0 + n, device=w.device) // fmt.GROUP
        s = self.scale[:, groups]
        y = (w / self.gscale) / s
        sign = fmt._sign(y)
        mag = fmt._round_e2m1(y, self.ties)
        values = fmt.lut(fmt.FMT_E2M1, w.device)
        if self.e0m3 is not None:
            e0 = self.e0m3[:, groups]
            mag = torch.where(e0, fmt._round_e0m3(y), mag)
            codes = mag | sign
            v = torch.where(e0, fmt.lut(fmt.FMT_E0M3, w.device)[codes], values[codes])
        else:
            codes = mag | sign
            v = values[codes]
        return codes.to(torch.uint8), (v * s) * self.gscale


@torch.no_grad()
def rtn_grid(w: torch.Tensor, weight: str, mask: torch.Tensor | None = None, unit=None) -> tuple[Grid, fmt.FP4Codes]:
    """The fixed grid of ``weight`` for ``w``, and its RTN codes -- from the same quantizers
    :meth:`FlipLinear.from_linear` uses (MixFP4: A = ``e0m3(w)``, B = ``fourover6(w)``, per tile)."""
    if weight == "nvfp4":
        c = fmt.nvfp4(w)
        return Grid(c.scale, c.gscale, None, "up"), c
    if weight == "fourover6":
        c = fmt.fourover6(w)
        return Grid(c.scale, c.gscale, None, "down"), c
    if weight == "mixfp4":
        if mask is None:
            raise ValueError("mixfp4 needs its tile map")
        rows, cols = maps_.parse_unit(unit)
        a, b = fmt.e0m3(w), fmt.fourover6(w)
        e0 = maps_.expand(mask.to(w.device).bool(), rows, cols // fmt.GROUP, w.shape[0])
        wide = e0.repeat_interleave(fmt.GROUP, 1)
        scale = torch.where(e0, a.scale, b.scale)
        codes = fmt.FP4Codes(torch.where(wide, a.codes, b.codes), scale, b.gscale, e0)
        return Grid(scale, b.gscale, e0, "down"), codes
    raise ValueError(f"GPTQ supports nvfp4 / fourover6 / mixfp4 weights, not {weight!r}")


@torch.no_grad()
def gptq(w: torch.Tensor, hessian: torch.Tensor, grid: Grid, block: int = 128, damp: float = 0.01,
         act_order: bool = False) -> fmt.FP4Codes:
    """GPTQ-round ``w`` ``[O, K]`` on the fixed ``grid`` under ``hessian`` ``[K, K]`` (= X^T X / n).

    ``act_order``: process columns in decreasing ``diag(H)`` order. The grid is fixed per element, so
    a permuted column is still rounded on ITS OWN scale group and tile format ("static groups" by
    construction): scales, global scale, 4/6 choices and the map are untouched, only the order in
    which rounding errors are compensated changes."""
    W = w.float().clone()
    o, k = W.shape
    H = hessian.float().clone()
    dead = (W == 0).all(0)                                        # ModelOpt's dead columns: all-zero weights
    H[dead, :] = 0
    H[:, dead] = 0
    H[dead, dead] = 1
    perm = torch.arange(k, device=W.device)
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True, stable=True)
        W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(k, device=H.device)
    try:
        L = torch.linalg.cholesky(H.double())
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).float()
    except torch.linalg.LinAlgError:                              # ModelOpt: not positive definite -> identity
        Hinv = torch.eye(k, device=W.device)
    codes = torch.empty(o, k, dtype=torch.uint8, device=W.device)
    order = perm.tolist()
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = W1[:, i:i + 1]
            j = order[i1 + i]                                     # the column's original index
            c, q = grid.quantize(col, j)
            codes[:, j] = c[:, 0]
            err = (col - q) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            E1[:, i:i + 1] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return fmt.FP4Codes(codes, grid.scale, grid.gscale, grid.fmt)


def output_error(w: torch.Tensor, codes: fmt.FP4Codes, x: torch.Tensor) -> float:
    """``||(W - Q) X^T||_F^2`` with ``Q`` the fp32 dequantized codes."""
    q = codes.dequantize(torch.float32)
    return float(((w.float() - q) @ x.float().T).square().sum())


def hessian_error(w: torch.Tensor, codes: fmt.FP4Codes, hessian: torch.Tensor) -> float:
    """``tr((W - Q) H (W - Q)^T)``: :func:`output_error` per token, from ``H = X^T X / n``."""
    d = w.float() - codes.dequantize(torch.float32)
    return float(((d @ hessian.float()) * d).sum())


# ------------------------------------------------------------------------------------------------
# model level


class _Stop(Exception):
    pass


def decoder_layers(model: nn.Module, linears: dict) -> tuple[str, nn.ModuleList]:
    """The ``nn.ModuleList`` of decoder layers: the one holding the most quantizable linears, which
    must be all of them."""
    best = ("", None, -1)
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m):
            n = sum(1 for lin in linears if lin.startswith(name + "."))
            if n > best[2]:
                best = (name, m, n)
    name, layers, n = best
    if layers is None or n != len(linears):
        outside = [lin for lin in linears if not lin.startswith(name + ".")]
        raise ValueError(f"GPTQ needs every quantizable linear inside the decoder layers {name!r}; "
                         f"outside: {outside[:5]}")
    return name, layers


@torch.no_grad()
def _capture(model, layers: nn.ModuleList, windows, device, batch: int):
    """Inputs of layer 0 and every layer's other call arguments, per calibration batch."""
    inputs, calls = [], [[] for _ in layers]

    def pre(i):
        def fn(mod, args, kwargs):
            kwargs = dict(kwargs)
            hs = args[0] if args else kwargs.pop("hidden_states")
            if i == 0:
                inputs.append(hs.detach())
            calls[i].append((tuple(args[1:]), kwargs))
            if i == len(layers) - 1:
                raise _Stop
        return fn
    hooks = [layer.register_forward_pre_hook(pre(i), with_kwargs=True) for i, layer in enumerate(layers)]
    try:
        for start in range(0, len(windows), batch):
            ids = torch.cat(windows[start:start + batch]).to(device)
            try:
                model(input_ids=ids, use_cache=False)
            except _Stop:
                pass
    finally:
        for h in hooks:
            h.remove()
    return inputs, calls


def _device(module):
    for t in module.parameters():
        return t.device
    for t in module.buffers():
        return t.device
    return None


def _to(obj, device):
    """Tensors inside (nested) tuples / lists / dicts moved to ``device``; everything else as is."""
    if device is None:
        return obj
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def _run(layer, inputs, calls):
    """Run ``layer`` on every recorded batch. Inputs and call arguments are moved to the layer's own
    device explicitly: with ``--device-map balanced`` the layers live on different GPUs (accelerate's
    per-layer hooks would move them too; this does not rely on that)."""
    outs = []
    dev = _device(layer)
    for hs, (args, kwargs) in zip(inputs, calls):
        y = layer(_to(hs, dev), *_to(args, dev), **_to(kwargs, dev))
        outs.append((y[0] if isinstance(y, tuple) else y).detach())
    return outs


def _hessians(layer, linears: dict, inputs, calls, act: str | None = None) -> dict[str, torch.Tensor]:
    """``X^T X / n`` per linear; ``act``: the activation format whose per-token fake quantization ``X`` goes
    through first (None: the input as it is)."""
    sums, counts, hooks = {}, {}, []

    def hook(name):
        def fn(mod, args):
            x = args[0]
            rot = getattr(mod, "rotation", None)
            if rot is not None:
                x = rot(x)
            if act is not None:
                x = fmt.fake_quant(x, act, "row")
            x = x.detach().reshape(-1, mod.in_features).float()
            if name in sums:
                sums[name].addmm_(x.T, x)
            else:
                sums[name] = x.T @ x
            counts[name] = counts.get(name, 0) + x.shape[0]
        return fn
    for name, lin in linears.items():
        hooks.append(lin.register_forward_pre_hook(hook(name)))
    try:
        _run(layer, inputs, calls)
    finally:
        for h in hooks:
            h.remove()
    missing = set(linears) - set(sums)
    if missing:
        raise RuntimeError(f"no calibration inputs reached {sorted(missing)}")
    return {n: sums[n] / counts[n] for n in sums}


def cache_key(spec_key: str, qspec: QuantSpec, maps: dict | None, rotate: str, damp: float, block: int,
              windows: int, act_order: bool = False, calib: str = "mathcode", hessian_input: str | None = None,
              data_sha256: str | None = None) -> dict:
    digest = None
    if maps is not None:
        h = hashlib.sha256()
        for n in sorted(maps):
            h.update(n.encode())
            h.update(maps[n].cpu().numpy().tobytes())
        digest = h.hexdigest()
    key = dict(model=spec_key, weight=qspec.weight, act=qspec.act, unit=qspec.unit if maps is not None else None,
               mode=qspec.mode, map_sha256=digest, rotate=rotate, damp=damp, block=block, windows=windows)
    if act_order:                         # only when set, so caches written before these options stay valid
        key["act_order"] = True
    if calib != "mathcode":
        key["calib"] = calib
    if hessian_input is not None:         # the ModelOpt-aligned rules (dead columns, Hessian input), and the data
        key.update(rules="modelopt-0.47", hessian_input=hessian_input, data_sha256=data_sha256)
    return key


def codes_sha256(codes: dict) -> str:
    """One digest of a model's GPTQ codes as the kernels store them (per module, in name order: the name, the packed
    nibbles, the E4M3 scale bytes, the global scale and the per-block formats), for comparing code sets."""
    h = hashlib.sha256()
    for n in sorted(codes):
        p = _packed(codes[n])
        h.update(n.encode())
        for t in (p["q"], p["s"], p["g"]):
            h.update(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(p["fmt"].numpy().tobytes() if torch.is_tensor(p["fmt"]) else str(p["fmt"]).encode())
    return h.hexdigest()


def windows_sha256(windows) -> str:
    """The calibration data's identity in the cache key: sha256 of the windows' token ids, in order."""
    return hashlib.sha256(b"".join(w.cpu().numpy().tobytes() for w in windows)).hexdigest()


def _packed(c: fmt.FP4Codes) -> dict:
    """Host copy of one weight's codes as the kernels store them (nibbles, E4M3 bytes)."""
    return dict(q=fmt.pack_nibbles(c.codes).cpu(), s=c.scale_bytes().cpu(), g=c.gscale.float().cpu(),
                fmt=c.fmt.cpu() if torch.is_tensor(c.fmt) else int(c.fmt))


def _save(path: Path, packed: dict, key: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": "flipquant-gptq/1", "key": key, "codes": packed}, tmp)
    tmp.replace(path)


def _load(path: Path, key: dict) -> dict[str, fmt.FP4Codes]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("format") != "flipquant-gptq/1" or obj["key"] != key:
        raise SystemExit(f"{path} holds GPTQ codes for {obj.get('key')}, not {key}")
    return {n: fmt.FP4Codes(fmt.unpack_nibbles(c["q"]), c["s"].view(torch.float8_e4m3fn).float(), c["g"],
                            c["fmt"]) for n, c in obj["codes"].items()}


def load_codes_file(path: Path) -> tuple[dict, dict]:
    """A GPTQ codes file (:func:`_save`'s format, any key): ``(key, {name: {"q", "s", "g", "fmt"}})``. Read in full, not
    memory-mapped: on a network (FUSE) mount, page faults on a mapped file read about 2 MB/s, a full read about 85 MB/s."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("format") != "flipquant-gptq/1":
        raise SystemExit(f"{path} is not a flipquant-gptq/1 codes file")
    return obj["key"], obj["codes"]


def entries_sha256(entries: dict) -> str:
    """:func:`codes_sha256` of a codes file's stored entries (the same bytes, so the same digest as the loaded codes)."""
    h = hashlib.sha256()
    for n in sorted(entries):
        c = entries[n]
        h.update(n.encode())
        for t in (c["q"], c["s"], c["g"].float()):
            h.update(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(c["fmt"].numpy().tobytes() if torch.is_tensor(c["fmt"]) else str(int(c["fmt"])).encode())
    return h.hexdigest()


#: the GPTQ settings two candidate files must share with the run that composes them (every key field except the
#: weight format and its map)
COMPOSE_SHARED = ("model", "act", "mode", "rotate", "damp", "block", "windows", "act_order", "calib", "rules",
                  "hessian_input", "data_sha256", "propagate")


def compose_candidates(paths, key: dict, maps: dict, unit) -> tuple[dict[str, fmt.FP4Codes], dict]:
    """MixFP4 codes from two GPTQ runs, selected per tile by ``maps`` (``--gptq-candidates E2M1 E0M3``): an E0M3 tile
    takes the E0M3 file's codes and E4M3 scales, an E2M1 tile the E2M1 file's -- the per-tile selection of
    :func:`rtn_grid`'s mixfp4 branch, with GPTQ codes in place of RTN ones. The E2M1 file must hold FourOverSix codes
    (``--weight fourover6``, no map), the E0M3 file mixfp4 codes with every block E0M3 (``--weight mixfp4`` with an
    all-E0M3 map), both with this run's GPTQ settings (:data:`COMPOSE_SHARED` of ``key``), and every module's two
    global scales must be equal (one tensor, one global scale). Returns ``(codes, info)``."""
    paths = [Path(p) for p in paths]
    (kb, eb), (ka, ea) = (load_codes_file(p) for p in paths)
    for label, k, p in (("E2M1", kb, paths[0]), ("E0M3", ka, paths[1])):
        diff = {f: (k.get(f), key.get(f)) for f in COMPOSE_SHARED if k.get(f) != key.get(f)}
        if diff:
            raise SystemExit(f"--gptq-candidates: the {label} file {p} was made with other GPTQ settings "
                             f"(file, this run): {diff}")
    if kb.get("weight") != "fourover6" or kb.get("map_sha256") is not None:
        raise SystemExit(f"--gptq-candidates: the E2M1 file must hold FourOverSix codes without a map, not "
                         f"{kb.get('weight')} (map {kb.get('map_sha256')})")
    if ka.get("weight") != "mixfp4":
        raise SystemExit(f"--gptq-candidates: the E0M3 file must hold mixfp4 codes (all-E0M3 map), not {ka.get('weight')}")
    for label, e in (("E2M1", eb), ("E0M3", ea)):
        if set(e) != set(maps):
            raise SystemExit(f"--gptq-candidates: the {label} file covers {len(e)} modules, the map {len(maps)}")
    rows, cols = maps_.parse_unit(unit)
    out = {}
    for n in sorted(maps):
        b, a = eb[n], ea[n]
        if torch.is_tensor(b["fmt"]) or int(b["fmt"]) != fmt.FMT_E2M1:
            raise SystemExit(f"--gptq-candidates: {n}: the E2M1 file's codes are not uniform E2M1")
        if not (torch.is_tensor(a["fmt"]) and bool(a["fmt"].all())):
            raise SystemExit(f"--gptq-candidates: {n}: the E0M3 file has E2M1 blocks")
        if not torch.equal(a["g"].float(), b["g"].float()):
            raise SystemExit(f"--gptq-candidates: {n}: the two files' global scales differ")
        qa, qb = fmt.unpack_nibbles(a["q"]), fmt.unpack_nibbles(b["q"])
        if qa.shape != qb.shape:
            raise SystemExit(f"--gptq-candidates: {n}: shapes {tuple(qa.shape)} / {tuple(qb.shape)}")
        e0 = maps_.expand(maps[n].bool(), rows, cols // fmt.GROUP, qb.shape[0])
        wide = e0.repeat_interleave(fmt.GROUP, 1)
        sa, sb = (x["s"].view(torch.float8_e4m3fn).float() for x in (a, b))
        out[n] = fmt.FP4Codes(torch.where(wide, qa, qb), torch.where(e0, sa, sb), b["g"], e0)
    info = {label: dict(path=str(p), key=k, codes_sha256=entries_sha256(e), bytes=p.stat().st_size)
            for label, p, k, e in (("e2m1", paths[0], kb, eb), ("e0m3", paths[1], ka, ea))}
    return out, info


@torch.no_grad()
def quantize_model(model: nn.Module, spec, qspec: QuantSpec, windows, *, maps: dict | None = None,
                   damp: float = 0.01, block: int = 128, batch: int = 8, cache: Path | None = None,
                   rotate: str = "none", act_order: bool = False, calib: str = "mathcode",
                   hessian_input: str = "quantized", codes_out: dict | None = None,
                   propagate: str = "quantized", candidates=None, log=print) -> tuple[dict[str, FlipLinear], dict]:
    """Sequential GPTQ of every quantizable linear (see the module docstring); each is replaced in
    place by a frozen :class:`FlipLinear` built from its codes (keeping a :class:`RotatedLinear`'s
    rotation). With ``cache``, codes are read from it if it exists (and must match this setting),
    else computed and written. ``codes_out``: filled with ``{name: FP4Codes}`` (e.g. for the sm120 path).
    ``candidates`` (mixfp4 only; opt-in): ``(E2M1 file, E0M3 file)``, two GPTQ codes files whose codes are composed
    per tile by ``maps`` (:func:`compose_candidates`) instead of running GPTQ. Returns ``(layers, info)``."""
    from models.loader import _set_module, quantizable_linears
    started = time.time()
    if qspec.weight not in ("nvfp4", "fourover6", "mixfp4"):
        raise ValueError(f"--ptq gptq supports --weight nvfp4 / fourover6 / mixfp4, not {qspec.weight}")
    if (qspec.weight == "mixfp4") != (maps is not None):
        raise ValueError("--ptq gptq: mixfp4 needs a map, and only mixfp4 takes one")
    linears = quantizable_linears(model, spec)
    if hessian_input not in ("quantized", "bf16"):
        raise ValueError(f"hessian_input {hessian_input!r}")
    if propagate not in ("quantized", "bf16"):
        raise ValueError(f"propagate {propagate!r}")
    if propagate == "bf16" and hessian_input != "bf16":
        raise ValueError("propagate='bf16' runs calibration without activation quantization; the Hessians then come "
                         "from unquantized inputs: use hessian_input='bf16'")
    # paper-sm120-runs: propagate='bf16' = weight-only GPTQ's calibration (every activation quantizer off: the layers
    # already done run with quantized weights and BF16 activations), so the codes do not depend on the activation
    # format the model is later evaluated with; the layers get the evaluation spec back afterwards
    calib_spec = dataclasses.replace(qspec, act="none") if propagate == "bf16" else qspec
    key = cache_key(spec.key, calib_spec, maps, rotate, damp, block, len(windows), act_order, calib, hessian_input,
                    windows_sha256(windows))
    if propagate == "bf16":                    # only when set, so caches written before this option stay valid
        key["propagate"] = "bf16"
    info = dict(method="gptq", grid="rtn-fixed", **key)
    all_codes = {}

    def install(name, lin, codes):
        # host copies: a 27B model's unpacked codes are ~1.25 bytes per weight, which would otherwise stay on the GPU
        # next to the layers (the sm120 install moves each module's codes back itself; codes_sha256 reads them here)
        host = fmt.FP4Codes(codes.codes.cpu(), codes.scale.cpu(), codes.gscale.cpu(),
                            codes.fmt.cpu() if torch.is_tensor(codes.fmt) else codes.fmt)
        if codes_out is not None:
            codes_out[name] = host
        all_codes[name] = host
        layer = FlipLinear.from_codes(lin, calib_spec, codes, mask=None if maps is None else maps[name], name=name)
        layer.rotation = getattr(lin, "rotation", None)
        _set_module(model, name, layer)
        return layer

    if candidates is not None:
        if qspec.weight != "mixfp4" or cache is not None:
            raise ValueError("candidates compose a mixfp4 map's tiles from two codes files (no cache)")
        codes, composed = compose_candidates(candidates, key, maps, qspec.unit)
        if set(codes) != set(linears):
            raise SystemExit(f"--gptq-candidates: codes for {len(codes)} modules, the model quantizes {len(linears)}")
        layers = {}
        for n in list(linears):
            layers[n] = install(n, linears.pop(n), codes.pop(n))
        if propagate == "bf16":
            for layer in layers.values():
                layer.spec = qspec
        info.update(grid="rtn-fixed, composed per tile from two GPTQ runs", composed=composed, loaded=True,
                    seconds=time.time() - started, codes_sha256=codes_sha256(all_codes))
        torch.cuda.empty_cache()
        return layers, info

    if cache is not None and Path(cache).exists():
        codes = _load(Path(cache), key)
        if set(codes) != set(linears):
            raise SystemExit(f"{cache}: codes for {len(codes)} modules, the model quantizes {len(linears)}")
        layers = {}
        for n in list(linears):               # release each BF16 weight as its layer is installed (as below)
            layers[n] = install(n, linears.pop(n), codes[n])
        if propagate == "bf16":
            for layer in layers.values():
                layer.spec = qspec                  # the evaluation's activations
        info.update(cache=str(cache), loaded=True, seconds=time.time() - started, codes_sha256=codes_sha256(all_codes))
        torch.cuda.empty_cache()
        return layers, info

    prefix, blocks = decoder_layers(model, linears)
    device = model.get_input_embeddings().weight.device
    eval_scope, qspec.act_scope = qspec.act_scope, "row"
    try:
        inputs, calls = _capture(model, blocks, windows, device, batch)
        layers, saved, stats = {}, {}, []
        for i, layer in enumerate(blocks):
            t0 = time.time()
            mine = {n: lin for n, lin in linears.items() if n.startswith(f"{prefix}.{i}.")}
            hessians = _hessians(layer, mine, inputs, calls[i],
                                 qspec.act if hessian_input == "quantized" else None) if mine else {}
            count = len(mine)
            for name in list(mine):
                lin = mine.pop(name)
                w = lin.weight
                grid, rtn = rtn_grid(w, qspec.weight, None if maps is None else maps[name], qspec.unit)
                h = hessians.pop(name)
                codes = gptq(w, h, grid, block, damp, act_order)
                ratio = hessian_error(w, codes, h) / max(hessian_error(w, rtn, h), 1e-30)
                layers[name] = install(name, lin, codes)
                linears.pop(name)                                  # release the BF16 weight
                if cache is not None:
                    saved[name] = _packed(codes)
                changed = float((codes.codes != rtn.codes).float().mean())
                stats.append(dict(name=name, codes_changed=changed, output_error_vs_rtn=ratio))
                del lin, w, grid, rtn, codes, h
            inputs = _run(layer, inputs, calls[i])
            mine_stats = stats[len(stats) - count:]
            mean_ratio = sum(st["output_error_vs_rtn"] for st in mine_stats) / max(count, 1)
            log(f"GPTQ layer {i + 1}/{len(blocks)}: {count} linears, calibration output error "
                f"{mean_ratio:.3f} x RTN, {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()
    finally:
        qspec.act_scope = eval_scope
    if propagate == "bf16":
        for layer in layers.values():
            layer.spec = qspec                      # the evaluation's activations
    info["codes_sha256"] = codes_sha256(all_codes)
    info.update(seconds=time.time() - started, per_linear=stats,
                mean_codes_changed=sum(s["codes_changed"] for s in stats) / max(len(stats), 1),
                mean_output_error_vs_rtn=sum(s["output_error_vs_rtn"] for s in stats) / max(len(stats), 1))
    if cache is not None:
        _save(Path(cache), saved, key)
        info.update(cache=str(cache), loaded=False)
    return layers, info
