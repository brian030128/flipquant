"""The share of all-E2M1 (warp, K-tile) dispatch patterns in the FlipQuant maps, FLOP-weighted (Appendix D,
optimization (i): the frequency-ordered dispatch tests the all-E2M1 pattern first). CPU only: the map files, no GPU.

    python -m experiments.paper_1k.dispatch_share [--kinds flipquant_256,flipquant_1k] [--out results/paper_1k/aux]

The definition is kernel-opt's experiments/kernel_opt/dispatch_ceiling.py (NVFP4-RaZeR, branch kernel-opt), the analysis
behind the paper's number, ported to flipquant's map files:
- a K-tile is 128 K = two 64-K tile columns; a warp's dispatch pattern per K-tile covers its weight granules x the two
  64-K blocks; pattern 0 (all E2M1) is the one the deployed dispatch tests first;
- weights on A (n16k64_wA, 4x2 warps; 16x64 maps, and 256x64 maps at 16-row granules): a CTA's weight panel is 128 rows
  = 8 granules of 16 rows; warp w owns granules w and w + 4 (4 bits, 16 arms). The deployed 256x64 kernel's 32-row
  granule (two 16-row blocks 64 rows apart, 4 arms) is exactly that pair, and a 256-row tile covers both, so its
  pattern-0 share is the same number ("g32 4-arm");
- weights on B (n8k64_wB, 1x8 warps; 8x64 maps): a panel is 128 columns = 16 granules of 8; warp w owns g(w) and
  g(w) + 8, g(w) = 2 (w % 2) + (w // 2) % 2 + 4 (w // 4). The 64-column builds (wB64): 8 granules, g(w) and g(w) + 4;
- per module the share of pattern-0 (warp, K-tile) pairs; aggregated over a forward's quantized linears weighted by
  out x in (their share of the GEMM work at any token count), and as the plain mean over modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import artifacts as A
from . import config as C

UNITS = ("16x64", "256x64", "8x64")
WA128 = (8, [(w, w + 4) for w in range(4)])
WB128 = (16, [(x, x + 8) for x in (2 * (w % 2) + (w // 2) % 2 + 4 * (w // 4) for w in range(8))])
WB64 = (8, [(x, x + 4) for x in (2 * (w % 2) + (w // 2) % 2 for w in range(4))])
LAYOUTS = {"16x64": {"wA128": WA128}, "256x64": {"wA128": WA128}, "8x64": {"wB128": WB128, "wB64": WB64}}
EXPAND = {"16x64": 1, "256x64": 16, "8x64": 1}          # map rows -> kernel granule rows (256 = 16 x 16)


def pattern0(mask: np.ndarray, panel: int, pairs) -> tuple[int, int]:
    """(all-E2M1 (warp, K-tile) patterns, all patterns) of one module's granule mask [granule rows, 64-K columns]
    (dispatch_ceiling.metrics' pattern part; rows padded to a whole panel, a trailing odd 64-K column dropped)."""
    r = (-mask.shape[0]) % panel
    m = np.concatenate([mask, np.zeros((r, mask.shape[1]), bool)]) if r else mask
    npan, kt = m.shape[0] // panel, m.shape[1] // 2
    mk = m[:, :2 * kt].reshape(npan, panel, kt, 2)
    p0 = sum(int(((mk[:, a].sum(axis=2) + mk[:, b].sum(axis=2)) == 0).sum()) for a, b in pairs)
    return p0, npan * len(pairs) * kt


def share(tiles: dict, unit: str) -> dict:
    """{layout: dict(flop_weighted, plain, min_module, modules, e0m3_share)} of one map."""
    out = {}
    rows_per = {"16x64": 16, "256x64": 256, "8x64": 8}[unit]
    for lname, (panel, pairs) in LAYOUTS[unit].items():
        w = s = plain = 0.0
        mins, n, e, t = 1.0, 0, 0, 0
        for name, tile in tiles.items():
            mask = tile.numpy().astype(bool)
            e, t = e + int(mask.sum()), t + mask.size
            flops = mask.shape[0] * rows_per * mask.shape[1] * 64
            g = np.repeat(mask, EXPAND[unit], axis=0)
            p0, tot = pattern0(g, panel, pairs)
            f = p0 / tot
            w, s, plain, n, mins = w + flops, s + flops * f, plain + f, n + 1, min(mins, f)
        out[lname] = dict(flop_weighted=s / w, plain=plain / n, min_module=mins, modules=n, e0m3_share=e / t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", default="flipquant_256,flipquant_1k")
    ap.add_argument("--out", type=Path, default=C.REPO / "results" / "paper_1k" / "aux")
    a = ap.parse_args(argv)
    from flipquant import maps as M
    res = {}
    for kind in a.kinds.split(","):
        for m in C.MODELS:
            for u in UNITS:
                tiles, unit, _ = M.load(A.fetch(m, kind, u))
                assert M.unit_name(unit) == u, (m, kind, unit)
                res.setdefault(kind, {}).setdefault(m, {})[u] = share(tiles, u)
                print(kind, m, u, {k: round(100 * v["flop_weighted"], 2) for k, v in res[kind][m][u].items()}, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dispatch_share.json").write_text(json.dumps(res, indent=1) + "\n")
    (a.out / "dispatch_share.md").write_text(markdown(res))
    return res


def _rng(res, kind, models=C.MODELS, units=UNITS):
    v = [100 * res[kind][m][u][next(iter(LAYOUTS[u]))]["flop_weighted"] for m in models for u in units]
    return min(v), max(v)


def markdown(res: dict) -> str:
    kinds = [k for k in ("flipquant_256", "flipquant_1k") if k in res]
    label = {"flipquant_256": "256-window release maps", "flipquant_1k": "1k maps"}
    md = ["# The all-E2M1 dispatch-pattern share (Appendix D, optimization (i)), 256 and 1k maps", "",
          "Share of (warp, K-tile) dispatch patterns that are all E2M1, FLOP-weighted over each model's quantized "
          "linears (out x in), by kernel-opt's `dispatch_ceiling.py` definition (`experiments/paper_1k/dispatch_share.py`"
          ", CPU only, on the pinned Hugging Face maps). 16x64 and 256x64: n16k64_wA (a warp's two 16-row granules, "
          "rows w and w + 4 of a 128-row panel; for 256x64 maps this equals the deployed 32-row-granule 4-arm "
          "kernel's share); 8x64: n8k64_wB (128-column panels; the 64-column builds give the same to 0.01 point). In "
          "brackets: the maps' E0M3 tile share.", ""]
    for kind in kinds:
        md += [f"## The {label[kind]}", "", "| model | 16x64 | 256x64 | 8x64 |", "|---|---:|---:|---:|"]
        for m in C.MODELS:
            cells = []
            for u in UNITS:
                x = res[kind][m][u][next(iter(LAYOUTS[u]))]
                cells.append(f"{100 * x['flop_weighted']:.1f} % ({100 * x['e0m3_share']:.1f} %)")
            md.append(f"| {C.TITLE[m]} | " + " | ".join(cells) + " |")
        lo, hi = _rng(res, kind)
        per = "; ".join(f"{u} {_rng(res, kind, units=(u,))[0]:.1f}–{_rng(res, kind, units=(u,))[1]:.1f} %" for u in UNITS)
        no1b = _rng(res, kind, models=C.MODELS[1:])
        md += ["", f"Range over the six models and three units: **{lo:.1f}–{hi:.1f} %** ({per}); without "
               f"Llama-3.2-1B-Instruct {no1b[0]:.1f}–{no1b[1]:.1f} %.", ""]
    if "flipquant_256" in res:
        lo, hi = _rng(res, "flipquant_256")
        md += ["## Check against the paper", "",
               f"The release maps give {lo:.1f}–{hi:.1f} %, the paper's \"79–99 %\" (its lower end is "
               "Llama-3.2-1B-Instruct at 256x64). The source note of that sentence (2026-10-12) lists narrower ranges "
               "per unit (16x64 95.4–98.3, 256x64 92.5–98.2, 8x64 96.5–98.6 %) for six release maps it does not name; "
               "they do not reach Llama-3.2-1B-Instruct's 78.9 %, so that set was not these six models. Without "
               "Llama-3.2-1B-Instruct the maps here give 16x64 "
               f"{_rng(res, 'flipquant_256', C.MODELS[1:], ('16x64',))[0]:.1f}–"
               f"{_rng(res, 'flipquant_256', C.MODELS[1:], ('16x64',))[1]:.1f}, 256x64 "
               f"{_rng(res, 'flipquant_256', C.MODELS[1:], ('256x64',))[0]:.1f}–"
               f"{_rng(res, 'flipquant_256', C.MODELS[1:], ('256x64',))[1]:.1f}, 8x64 "
               f"{_rng(res, 'flipquant_256', C.MODELS[1:], ('8x64',))[0]:.1f}–"
               f"{_rng(res, 'flipquant_256', C.MODELS[1:], ('8x64',))[1]:.1f} %, close to the note's ranges but not equal "
               "to them.", ""]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
