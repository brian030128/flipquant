"""Item 8 (Appendix D.4's sentence: the real tags against random placement) at the 1k calibration (part AC item 8).

The D.4 measurement: `evaluation.gemm_time` on Mistral-7B-Instruct-v0.3 with its three maps and, in the same process, a
map of each unit whose E0M3 tiles are placed at random with the same count per module (random_maps.py); the o_proj,
gate_proj and down_proj projections; per cell the overhead over stock_ko of the real tags minus that of the random tags,
in points (the typical module's tags; the densest module's reported too), and gemm_time's per-cell bitwise check.

T: the paper's source ran six T values that are not recorded; this item runs the D.5 set (gemm_time's default, T = 1,
4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192), a superset of the usual six (1, 16, 128, 512, 2048, 8192).
A registered deviation from the plan's "C3v / amendment 19 method" (NVFP4-RaZeR paper-eval PROTOCOL.md, part AC item 8):
the paper's sentence now rests on this real-vs-random comparison (its source note of 2026-10-13), which replaced C3v.

- **Step 0** (gate `item8-step0`): the release maps. Pass: every typical cell within 1 point of random placement, the
  paper's statement (published ranges: 8x64 -0.53..+0.43, 16x64 -0.37..+0.94, 256x64 -0.28..+0.43 points, on another
  RTX PRO 6000 board, with other random draws).
- **The 1k run:** the 1k maps (their shares are larger: e.g. 16x64 5.3 % against 1.1 %).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import artifacts as A
from . import config as C

REF = json.loads((C.HERE / "reference" / "item8.json").read_text())
MODEL = "mistral-7b"
RUNS = {"256": "flipquant_256", "1k": "flipquant_1k"}
PROJ = ("o_proj", "gate_proj", "down_proj")
TOKENS = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)      # D.5's set (gemm_time's default)
POLICIES = ("stock_ko", "8x64", "16x64", "256x64", "8x64_random", "16x64_random", "256x64_random")


def random_cmd(real, out):
    return [C.py(MODEL), "-m", "experiments.paper_1k.random_maps", "--map", real, "--seed", 0, "--out", out]


def gemm_cmd(real: dict, rand: dict, out):
    maps = ",".join([f"{u}={real[u]}" for u in C.UNITS] + [f"{u}_random={rand[u]}" for u in C.UNITS])
    return [C.py(MODEL), "-m", "evaluation.gemm_time", "--model", MODEL, "--maps", maps, "--policies", ",".join(POLICIES),
            "--projections", ",".join(PROJ), "--kernel-build-dir", C.build_dir(), "--out", out]


def _rec(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if r.get("status") == "complete" else None


def cells(rec) -> dict:
    """{tags: {unit: {proj: {T: real - random (points)}}}} from one gemm_time record, both over its stock_ko."""
    us = {(r["config"], r["proj"], r["tokens"]): r["gemm_us"] for r in rec["rows"]}
    out = {}
    for tag in ("typical", "densest"):
        for u in C.UNITS:
            for p in PROJ:
                for t in TOKENS:
                    s = us[("stock_ko", p, t)]
                    real, rand = us[(f"{u}_{tag}", p, t)], us[(f"{u}_random_{tag}", p, t)]
                    out.setdefault(tag, {}).setdefault(u, {}).setdefault(p, {})[t] = 100 * (real - rand) / s
    return out


def shares(rec) -> dict:
    """The typical module's E0M3 share per unit and projection (granules / tiles of the module)."""
    out = {}
    for p, v in rec["projections"].items():
        for u in C.UNITS:
            ops = v["operands"]
            out.setdefault(u, {})[p] = dict(real=ops[f"{u}_typical"]["e0m3_granules"],
                                            random=ops[f"{u}_random_typical"]["e0m3_granules"])
    return out


def run(R, out: Path, dry_run=False, data_root=None):
    d = out / "item8"
    get = A.local if dry_run else A.fetch
    for name, kind in RUNS.items():
        real = {u: get(MODEL, kind, u) for u in C.UNITS}
        rand = {}
        for u in C.UNITS:
            o = d / name / f"random_{u}.pt"
            R.must(R.run(f"item8_random_{name}_{u}", random_cmd(real[u], o), o.with_suffix(".json"), gpu=False,
                         extra_env={"CUDA_VISIBLE_DEVICES": ""}), f"random map {name} {u}")
            rand[u] = o
        o = d / name / "gemm_time.json"
        R.must(R.run(f"item8_gemm_{name}", gemm_cmd(real, rand, o), o), f"gemm_time {name}")
        if name == "256" and not dry_run:
            c = cells(_rec(o))["typical"]
            worst = max(abs(v) for u in c.values() for p in u.values() for v in p.values())
            res = dict(worst_abs_points=worst, criterion_points=1.0, passed=worst <= 1.0,
                       ranges={u: [min(v for p in c[u].values() for v in p.values()),
                                   max(v for p in c[u].values() for v in p.values())] for u in C.UNITS},
                       published_ranges=REF["published_256"]["ranges"])
            res["accepted"] = not res["passed"] and "item8-step0" in R.accepted()
            (d / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
            R.log(f"item8 step 0: worst |real - random| {worst:.2f} points")
            R.gate("item8-step0", res["passed"], f"the release maps' real tags are {worst:.2f} points from random "
                                                 f"placement (the paper: within 1 point)")
    R.log("item8 done")


def report(out: Path, dest: Path):
    d = out / "item8"
    rec = dict(step0=None, runs={})
    s0 = d / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None
    md = ["# Item 8: Appendix D.4 (the real tags against random placement) at the 1k calibration", "",
          "`evaluation.gemm_time` on Mistral-7B-Instruct-v0.3 with its maps and, in the same process, a map of each unit "
          "with the same E0M3 count per module at random tiles; o_proj, gate_proj, down_proj; per cell the real tags' "
          "overhead over stock_ko minus the random tags', in points. Typical module (densest in the second table). Run "
          "from flipquant alone (`python -m experiments.paper_1k.run item8`).", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0 (the release maps)", "", f"Worst |real − random| over the typical cells: {s['worst_abs_points']:.2f} "
               f"points (criterion: within 1) — **{'pass' if s['passed'] else 'FAIL'}**"
               + (" (accepted by the runner's user)" if s.get("accepted") else "") + ". Ranges per unit: "
               + "; ".join(f"{u} {s['ranges'][u][0]:+.2f}..{s['ranges'][u][1]:+.2f} (published "
                           f"{s['published_ranges'][u][0]:+.2f}..{s['published_ranges'][u][1]:+.2f})" for u in C.UNITS)
               + ".", ""]
    for name in RUNS:
        r = _rec(d / name / "gemm_time.json")
        if r is None:
            md += [f"## The {name} maps", "", "TBD.", ""]
            continue
        c, sh = cells(r), shares(r)
        bw = r["checks"]["bitwise"]
        rec["runs"][name] = dict(cells=c, shares=sh, bitwise=dict(checked=len(bw), equal=sum(b["equal"] for b in bw)))
        md += [f"## The {name} maps", "", f"Bitwise checks (each isolated launch against the module's fused forward): "
               f"{rec['runs'][name]['bitwise']['equal']} of {rec['runs'][name]['bitwise']['checked']} equal.", ""]
        for tag in ("typical", "densest"):
            md += [f"## The {name} maps, {tag} module: real − random (points)", "",
                   "| unit | projection | E0M3 granules | " + " | ".join(f"T={t}" for t in TOKENS) + " |",
                   "|---|---|---:|" + "---:|" * len(TOKENS)]
            for u in C.UNITS:
                for p in PROJ:
                    md.append(f"| {u} | {p} | {sh[u][p]['real']:,} | " + " | ".join(f"{c[tag][u][p][t]:+.2f}" for t in TOKENS)
                              + " |")
            md.append("")
        rng = {u: [min(v for p in c["typical"][u].values() for v in p.values()),
                   max(v for p in c["typical"][u].values() for v in p.values())] for u in C.UNITS}
        rec["runs"][name]["ranges"] = rng
        md += ["Typical-module ranges: " + "; ".join(f"{u} {rng[u][0]:+.2f}..{rng[u][1]:+.2f}" for u in C.UNITS) + ".", ""]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "REAL_VS_RANDOM.md").write_text("\n".join(md) + "\n")
    (dest / "item8.json").write_text(json.dumps(rec, indent=1) + "\n")
    return f"{dest}/REAL_VS_RANDOM.md"
