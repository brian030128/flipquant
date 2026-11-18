"""IF4 (Cook et al., "Adaptive Block-Scaled Data Types", arXiv 2603.28765) in fake quant: weights, per-tile
coarsening, and the per-token activation quantizer.

**The format** (official code github.com/mit-han-lab/fouroversix @ dadfad69, ``src/fouroversix/quantize/pytorch/
reference.py``: ``compute_nv_scale_factors``, ``fake_quantize_to_e2m1``, ``fake_quantize_to_int4``,
``select_intfloat``; constants in ``kernels/constants.py``), operation by operation as the reference implementation's ``tm-opt``
74058af ``quantize/adaptive_formats.py`` (rule ``if4``) implements it:

- tensor scale ``alpha = amax / (6 * 448)``; ONE block scale for both candidates, ``s = e4m3(bmax / 6 * (6 * 448 /
  amax))``, no clamp (a block whose scale underflows to 0 is zero, and counted); ``x_b = x / (alpha * s)``;
- FP4 candidate: E2M1, round to nearest even (the official piecewise ``torch.round``);
- INT4 candidate: ``round(clamp(x_b * 1.16666666, -7, 7))`` (half to even), dequantized ``x 0.8571428571`` (the
  official truncated 7/6 and 6/7);
- per 16-block, the lower sum of squared errors in the original units; INT4 only when strictly lower (ties keep FP4).

**+ FourOverSix** (``e2m1="fourover6"``; the reference rule ``if4fo6``, our variant, not Cook et al.'s): the FP
candidate is :func:`flipquant.formats.fourover6` (per block the max/6 or max/4 scale), the INT4 candidate is IF4's
above; both candidates' errors are taken on the weights as installed (BF16), ties keep the FP candidate.

**The E2M1 base** (``base=True``; the reference rule ``e2m1``): IF4's FP candidate everywhere, i.e. NVFP4 weights in
IF4's operation order. FourOverSix is the base of the FourOverSix variant.

**Tiles** (``unit`` ``RxC``, C a multiple of 16): per candidate, the squared errors of all elements of the tile are
summed, and the tile takes the smaller sum, with IF4's tie rule; every 16-block inside keeps that candidate's scale
rule. Rows are padded with zeros to a whole tile. ``1x16`` is the original per-block rule.

**Activations** (``--act-method own``): the same per-block IF4 rule on every forward, with one tensor scale per token
(``amax`` of the row; the paper writes one per tensor, see the README), always at 1x16. A row whose ``amax`` is 0
quantizes to 0, as the official code does for an all-zero tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import formats as fmt

GROUP = 16
IF4_INT_EXPANSION = 0.8571428571       # 6 / 7, fouroversix kernels/constants.py
IF4_INT_EXPANSION_RCP = 1.16666666     # 7 / 6
QMAX = 6 * 448
UNITS = ("1x16", "8x64", "16x64", "256x64")
E2M1_CANDIDATES = ("own", "fourover6")
MIXING_TILES = ((8, 64), (16, 64), (256, 64))


def parse_unit(unit: str) -> tuple[int, int]:
    r, c = (int(v) for v in unit.lower().split("x"))
    if c % GROUP or r < 1:
        raise ValueError(f"unit {unit!r}: columns must be a multiple of {GROUP}")
    return r, c


def e2m1_rne(x: torch.Tensor) -> torch.Tensor:
    """fouroversix ``fake_quantize_to_e2m1`` (round_style nearest): E2M1 values, round to nearest even."""
    a = x.abs()
    step1 = torch.round(2 * a) / 2
    step2 = torch.round(a)
    step3 = 2 * torch.round(a / 2)
    m1 = a < 2
    m2 = a < 4
    return x.sign() * (step1 * m1 + step2 * (~m1) * m2 + step3 * (~m1) * (~m2))


def _e4m3(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn).to(torch.float32)


def _if4_blocks(x: torch.Tensor, amax: torch.Tensor):
    """IF4's two candidates of the 16-blocks ``x`` [n, 16] (float32) under the tensor amax ``amax`` (a 0-d tensor, or
    [n, 1] for one amax per block's row). Returns ``(d_fp, d_int, e_fp, e_int, s, zero)``: dequantized candidates
    [n, 16], squared error per block [n], the block scale [n], the count of nonzero blocks whose scale underflowed,
    and the code values of both candidates (E2M1 values / integers) [n, 16].
    For a 0-d ``amax`` this is the official code's operation order exactly (the reference ``candidates(w, 'if4')``)."""
    per_row = amax.dim() > 0
    bmax = x.abs().max(dim=-1).values
    if per_row:
        a = amax.reshape(-1)
        ok = a != 0
        a_safe = torch.where(ok, a, torch.ones_like(a))
        encode = torch.tensor(QMAX, dtype=torch.float32, device=x.device) / a_safe
        s = torch.where(ok, _e4m3(bmax / torch.tensor(6, dtype=torch.float32, device=x.device) * encode),
                        torch.zeros_like(bmax))
        decode = 1 / (torch.tensor(QMAX, dtype=torch.float32, device=x.device) / a_safe)
        a_col = a.unsqueeze(1)
    else:
        if amax == 0:
            s = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        else:
            encode = torch.tensor(QMAX, dtype=torch.float32, device=x.device) / amax
            s = _e4m3(bmax / torch.tensor(6, dtype=torch.float32, device=x.device) * encode)
        decode = 1 / (torch.tensor(QMAX, dtype=torch.float32, device=x.device) / amax)
        a_col = amax
    dcol = decode.unsqueeze(1) if per_row else decode
    xb = torch.where(s.unsqueeze(1) != 0, x * (1 / (dcol * s.unsqueeze(1))), 0)
    q_fp = e2m1_rne(xb)
    q_int = (xb * IF4_INT_EXPANSION_RCP).clamp(min=-7, max=7).round()
    d_fp = (q_fp * s.unsqueeze(1) * a_col) / QMAX
    d_int = ((q_int * s.unsqueeze(1) * a_col) * IF4_INT_EXPANSION) / QMAX
    e_fp = ((d_fp - x) ** 2).sum(dim=-1)
    e_int = ((d_int - x) ** 2).sum(dim=-1)
    zero = ((s == 0) & (bmax != 0)).sum()
    return d_fp, d_int, e_fp, e_int, s, zero, q_fp, q_int


def candidates(w: torch.Tensor, e2m1: str = "own", base: bool = False):
    """The two candidates of every 16-block of a 2-D weight [rows, cols] (cols % 16 == 0): ``(d_fp, d_int, e_fp,
    e_int, zero)`` -- dequantized FP / INT4 candidates [rows, cols] and their squared errors per block
    [rows, cols / 16] (float32), and the count of underflowed nonzero blocks. ``base``: the INT4 candidate is never
    chosen (its error is +inf), i.e. IF4's E2M1 base (``e2m1="own"``) or FourOverSix (``e2m1="fourover6"``)."""
    return _candidates(w, e2m1, base)[:5]


def _candidates(w, e2m1, base):
    """:func:`candidates` plus, per block, the FP candidate's elements outside the E2M1 grid (|code| > 6: the
    official E2M1 rounding does not saturate, so a block scale that rounds down far, i.e. a subnormal E4M3 scale,
    can give 8; FourOverSix saturates)."""
    if e2m1 not in E2M1_CANDIDATES:
        raise ValueError(f"e2m1 candidate {e2m1!r}; choose from {E2M1_CANDIDATES}")
    rows, cols = w.shape
    x = w.float().reshape(-1, GROUP)
    d_fp, d_int, e_fp, e_int, _, zero, q_fp, _ = _if4_blocks(x, w.float().abs().max())
    over = (q_fp.abs() > 6).sum(dim=-1)
    if e2m1 == "fourover6":
        over = torch.zeros_like(over)
        d_fp = fmt.fake_quant(w, "fourover6").float().reshape(-1, GROUP)
        d_int = d_int.to(w.dtype).float()                   # both candidates as installed
        e_fp = ((d_fp - x) ** 2).sum(dim=-1)
        e_int = ((d_int - x) ** 2).sum(dim=-1)
    if base:
        d_int, e_int = d_fp, torch.full_like(e_fp, float("inf"))
    return (d_fp.reshape(rows, cols), d_int.reshape(rows, cols), e_fp.reshape(rows, cols // GROUP),
            e_int.reshape(rows, cols // GROUP), int(zero), over.reshape(rows, cols // GROUP))


def _uniform_wins(e_int, e_fp):
    return e_int < e_fp                    # IF4: INT4 only when strictly lower; ties keep FP


def select(e_fp: torch.Tensor, e_int: torch.Tensor, unit: str = "1x16") -> torch.Tensor:
    """Blocks [rows, cols / 16] that take the INT4 candidate: per block at 1x16, else elected per tile from the summed
    squared errors (rows padded with zeros), ties as IF4."""
    tr, tc = parse_unit(unit)
    if (tr, tc) == (1, GROUP):
        return _uniform_wins(e_int, e_fp)
    return select_tiles(e_fp, e_int, tr, tc)


def select_tiles(e_fp: torch.Tensor, e_int: torch.Tensor, tr: int, tc: int) -> torch.Tensor:
    """The coarse rule of :func:`select` for a ``tr x tc`` tile, also at 1x16 (where it must equal the per-block
    rule; tests/test_if4.py)."""
    rows, kb = e_fp.shape
    bc = tc // GROUP
    if kb % bc:
        raise ValueError(f"{kb * GROUP} columns do not tile {tr}x{tc}")
    pad = (-rows) % tr

    def tiles(e):
        e = F.pad(e, (0, 0, 0, pad))
        return e.reshape((rows + pad) // tr, tr, kb // bc, bc).sum(dim=(1, 3))
    pick = _uniform_wins(tiles(e_int), tiles(e_fp))
    return pick.repeat_interleave(tr, 0)[:rows].repeat_interleave(bc, 1)


def mixing(uni: torch.Tensor, tiles=MIXING_TILES) -> dict:
    """Within-tile mixing of per-block choices ``uni`` [rows, cols / 16] (True = INT4), as the reference ``mixing()``:
    per tile shape, the tiles, the tiles whose blocks disagree, and the sum over those of the minority share
    (min(INT4, FP4) / blocks). Padding rows are left out of the counts. Returns ints/floats."""
    return {k: {f: (int(v) if f != "minority_sum" else float(v)) for f, v in d.items()}
            for k, d in mixing_counts(uni, tiles).items()}


def mixing_counts(uni: torch.Tensor, tiles=MIXING_TILES) -> dict:
    """:func:`mixing` as 0-d tensors on ``uni``'s device (no host sync)."""
    rows, kb = uni.shape
    out = {}
    for tr, tc in tiles:
        bc = tc // GROUP
        if kb % bc:
            continue
        pad = (-rows) % tr
        u = F.pad(uni.float(), (0, 0, 0, pad)).reshape((rows + pad) // tr, tr, kb // bc, bc)
        real = F.pad(torch.ones_like(uni, dtype=torch.float32), (0, 0, 0, pad)).reshape(u.shape)
        n_uni, n = u.sum(dim=(1, 3)), real.sum(dim=(1, 3))
        mixed = (n_uni > 0) & (n_uni < n)
        minority = torch.minimum(n_uni, n - n_uni) / n
        out[f"{tr}x{tc}"] = dict(tiles=n.numel(), mixed=mixed.sum(), minority_sum=minority[mixed].sum())
    return out


@torch.no_grad()
def codes(w: torch.Tensor):
    """IF4 (``e2m1="own"``, 1x16) as the official code stores it: ``(is_int [blocks], code values [blocks, 16] (E2M1
    values or integers), scale bytes [blocks] with the sign bit set where INT4, dequantized [blocks, 16] float32)``."""
    x = w.float().reshape(-1, GROUP)
    d_fp, d_int, e_fp, e_int, s, _, q_fp, q_int = _if4_blocks(x, w.float().abs().max())
    is_int = _uniform_wins(e_int, e_fp)
    raw = s.to(torch.float8_e4m3fn).view(torch.uint8)
    raw = torch.where(is_int, raw + 128, raw).to(torch.uint8)
    return (is_int, torch.where(is_int.unsqueeze(1), q_int, q_fp), raw,
            torch.where(is_int.unsqueeze(1), d_int, d_fp))


@torch.no_grad()
def quantize_weight(w: torch.Tensor, unit: str = "1x16", e2m1: str = "own", base: bool = False):
    """-> (dequantized weight in ``w``'s dtype, statistics). The statistics: blocks, INT4 blocks, underflowed blocks,
    the total squared error of the installed weight against ``w`` (float64), and, at 1x16, the within-tile mixing of
    the per-block choices at 8x64, 16x64 and 256x64."""
    d_fp, d_int, e_fp, e_int, zero, over = _candidates(w, e2m1, base)
    uni = select(e_fp, e_int, unit)
    out = torch.where(uni.repeat_interleave(GROUP, 1), d_int, d_fp).to(w.dtype)
    stats = dict(blocks=uni.numel(), uniform_blocks=int(uni.sum()), zero_scale_blocks=zero,
                 sq_error=float(((out.double() - w.double()) ** 2).sum()),
                 e2m1_out_of_grid_elements=int(over[~uni].sum()))
    if parse_unit(unit) == (1, GROUP):
        stats["mixing"] = mixing(uni)
    return out, stats


# --------------------------------------------------------------------------------------------
# activations


ACT_METHODS = ("own", "nvfp4", "fourover6")
ACT_MIXING_TILES = ((16, 64),)           # the A operand's MMA tile: 16 tokens x 64 K
ACT_CHUNK_ROWS = 4096


def _act_rows(x2: torch.Tensor):
    """IF4 per token on a [rows, K] float32 view: (dequantized float32, INT4 choice [rows, K / 16], underflowed
    blocks, chosen E2M1 elements outside the grid)."""
    rows, k = x2.shape
    amax = x2.abs().amax(dim=-1, keepdim=True)
    blocks = x2.reshape(-1, GROUP)
    d_fp, d_int, e_fp, e_int, _, zero, q_fp, _ = _if4_blocks(blocks, amax.repeat_interleave(k // GROUP, 0))
    uni = _uniform_wins(e_int, e_fp)
    out = torch.where(uni.unsqueeze(1), d_int, d_fp)
    over = ((q_fp.abs() > 6) & ~uni.unsqueeze(1)).sum()
    return out.reshape(rows, k), uni.reshape(rows, k // GROUP), zero, over


@torch.no_grad()
def quantize_act(x: torch.Tensor, stats: "ActStats | None" = None) -> torch.Tensor:
    """IF4 activations, one tensor scale per token (row of the flattened ``[-1, K]`` view), per 16-block choice;
    returns the dequantized tensor in ``x``'s dtype. Rows are processed in chunks (row-wise, so exact)."""
    k = x.shape[-1]
    if k % GROUP:
        raise ValueError(f"last dimension {k} must be a multiple of {GROUP}")
    x2 = x.reshape(-1, k)
    out = torch.empty_like(x2)
    for i in range(0, x2.shape[0], ACT_CHUNK_ROWS):
        d, uni, zero, over = _act_rows(x2[i:i + ACT_CHUNK_ROWS].float())
        out[i:i + ACT_CHUNK_ROWS] = d.to(x.dtype)
        if stats is not None:
            stats.add(uni, zero, over)
    return out.reshape(x.shape)


class ActStats:
    """Running counts of a layer's activation choices, kept on the GPU (no host sync per forward)."""

    def __init__(self):
        self.blocks = 0
        self.t = None
        self.tiles = None
        self.mix_keys = []

    def add(self, uni, zero, over):
        mix = mixing_counts(uni, ACT_MIXING_TILES)
        vals = [uni.sum(), zero, over] + [v for d in mix.values() for v in (d["mixed"], d["minority_sum"])]
        tiles = [d["tiles"] for d in mix.values()]
        vec = torch.stack([torch.as_tensor(v, device=uni.device).double() for v in vals])
        self.t = vec if self.t is None else self.t + vec
        self.blocks += uni.numel()
        self.tiles = tiles if self.tiles is None else [a + b for a, b in zip(self.tiles, tiles)]
        self.mix_keys = list(mix)

    def as_dict(self):
        if self.t is None:
            return dict(blocks=0, uniform_blocks=0, zero_scale_blocks=0, e2m1_out_of_grid_elements=0, mixing={})
        v = self.t.tolist()                      # the only host sync
        mix = {key: dict(tiles=n, mixed=int(v[3 + 2 * i]), minority_sum=v[4 + 2 * i])
               for i, (key, n) in enumerate(zip(self.mix_keys, self.tiles))}
        return dict(blocks=self.blocks, uniform_blocks=int(v[0]), zero_scale_blocks=int(v[1]),
                    e2m1_out_of_grid_elements=int(v[2]), mixing=mix)


# --------------------------------------------------------------------------------------------
# the layer


class IF4Linear(nn.Module):
    """Fake-quant linear: a fixed dequantized BF16 weight, and the activation quantizer of ``act``:

    - ``own``: :func:`quantize_act` (IF4 per token);
    - ``nvfp4`` / ``fourover6``: :mod:`flipquant.formats` at ``act_scope`` (``row`` = per token), exactly as
      ``FlipLinear`` in fake mode with the same ``--act`` / ``--act-scope`` (same operations, so bitwise equal);
    - ``none``: weight only.
    """

    def __init__(self, weight: torch.Tensor, bias, act: str, act_scope: str = "row", name: str = ""):
        super().__init__()
        if act not in ACT_METHODS + ("none",):
            raise ValueError(f"act {act!r}")
        if act == "own" and act_scope != "row":
            raise ValueError("IF4 activations are per token (act_scope row)")
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.act, self.act_scope, self.name = act, act_scope, name
        self.register_buffer("weight", weight)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.act_stats = ActStats() if act == "own" else None

    def quant_act(self, x):
        if self.act == "own":
            return quantize_act(x, self.act_stats)
        if self.act == "none":
            return x
        return fmt.QUANTIZERS[self.act](x, self.act_scope, None).dequantize(x.dtype).reshape(x.shape)

    def forward(self, x):
        y = F.linear(self.quant_act(x), self.weight)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, w=if4, a={self.act}/{self.act_scope}"


class Report:
    """The ``adaptive`` record of a run: the weight statistics (fixed at install) and the activation statistics
    (read from the layers whenever the report is written; ``models.cli.save_json`` calls :meth:`as_dict`)."""

    def __init__(self, method: dict, weight: dict, layers: dict):
        self.method, self.weight, self.layers = method, weight, layers

    def activation(self) -> dict | None:
        per = {}
        for name, layer in self.layers.items():
            if layer.act_stats is None:
                return None
            d = layer.act_stats.as_dict()
            for key in ("all", name.rsplit(".", 1)[-1]):
                agg = per.setdefault(key, dict(blocks=0, uniform_blocks=0, zero_scale_blocks=0,
                                               e2m1_out_of_grid_elements=0, mixing={}))
                for f in ("blocks", "uniform_blocks", "zero_scale_blocks", "e2m1_out_of_grid_elements"):
                    agg[f] += d[f]
                for t, m in d["mixing"].items():
                    a = agg["mixing"].setdefault(t, dict(tiles=0, mixed=0, minority_sum=0.0))
                    for f in a:
                        a[f] += m[f]
        for agg in per.values():
            agg["uniform_fraction"] = agg["uniform_blocks"] / agg["blocks"] if agg["blocks"] else None
            for m in agg["mixing"].values():
                m["mixed_fraction"] = m["mixed"] / m["tiles"] if m["tiles"] else None
                m["mean_minority_share"] = m["minority_sum"] / m["mixed"] if m["mixed"] else None
        return dict(by_projection=per, uniform_fraction=per.get("all", {}).get("uniform_fraction"),
                    note="counted over every forward of the run (all tokens of all calls)")

    def as_dict(self):
        return dict(method=self.method, weight=self.weight, activation=self.activation())


def _aggregate_weight(stats: dict, fo6_sq: dict) -> dict:
    total, per_proj, mix = dict(blocks=0, uniform=0), {}, {}
    zero, over, sq, sq_fo6 = 0, 0, 0.0, 0.0
    for n, st in stats.items():
        over += st["e2m1_out_of_grid_elements"]
        proj = per_proj.setdefault(n.rsplit(".", 1)[-1], dict(blocks=0, uniform=0))
        for d in (total, proj):
            d["blocks"] += st["blocks"]
            d["uniform"] += st["uniform_blocks"]
        zero += st["zero_scale_blocks"]
        sq += st["sq_error"]
        sq_fo6 += fo6_sq[n]
        for mtile, x in st.get("mixing", {}).items():
            for key in ("all", n.rsplit(".", 1)[-1]):
                agg = mix.setdefault(mtile, {}).setdefault(key, dict(tiles=0, mixed=0, minority_sum=0.0))
                for k2 in agg:
                    agg[k2] += x[k2]
    frac = lambda d: d["uniform"] / d["blocks"]  # noqa: E731
    return dict(uniform_fraction=frac(total), uniform_fraction_by_projection={k: frac(v) for k, v in per_proj.items()},
                blocks=total["blocks"], uniform_blocks=total["uniform"], zero_scale_blocks=zero,
                e2m1_out_of_grid_elements=over, weight_sq_error=sq, fourover6_weight_sq_error=sq_fo6, weight_error_vs_fourover6=sq / sq_fo6,
                mixing={t: {k: dict(v, mixed_fraction=v["mixed"] / v["tiles"],
                                    mean_minority_share=v["minority_sum"] / v["mixed"] if v["mixed"] else None)
                            for k, v in by.items()} for t, by in mix.items()} or None)


@torch.no_grad()
def install(model: nn.Module, linears: dict, *, unit: str, e2m1: str, base: bool, act: str, act_scope: str = "row",
            set_module) -> tuple:
    """Replace every linear of ``linears`` ({name: nn.Linear}, module order) by an :class:`IF4Linear`
    (``linears`` is emptied as it goes, so the peak memory is the model plus one layer).
    Returns ``(layers, Report)``.
    ``installed_weight_sha256`` hashes (name, installed BF16 bytes) in module order, as
    the reference ``run_ppl_deploy.py`` does, so it is comparable with its records."""
    import hashlib
    parse_unit(unit)
    h = hashlib.sha256()
    layers, stats, fo6_sq = {}, {}, {}
    for name in list(linears):
        lin = linears.pop(name)            # drop the caller's reference, so each original weight is freed once replaced
        w = lin.weight.detach()
        q, st = quantize_weight(w, unit, e2m1, base)
        fo6 = fmt.fake_quant(w, "fourover6")
        fo6_sq[name] = float(((fo6.double() - w.double()) ** 2).sum())
        h.update(name.encode())
        h.update(q.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        layer = IF4Linear(q, lin.bias, act, act_scope, name)
        set_module(model, name, layer)
        layers[name] = layer
        stats[name] = st
        del lin, fo6
    torch.cuda.empty_cache()
    method = dict(method="IF4 (Cook et al., arXiv 2603.28765)" + (" E2M1 base" if base else ""),
                  weight=("if4_e2m1" if base else "if4"), unit=unit, e2m1_candidate=e2m1, activation=act,
                  activation_unit="1x16" if act == "own" else None,
                  activation_scale=None if act == "none" else ("per token (row)" if act_scope == "row" else act_scope),
                  installed_weight_sha256=h.hexdigest())
    return layers, Report(method, _aggregate_weight(stats, fo6_sq), layers)
