"""Item 5 (Table D.5, `tab:app-gemm`) at the 1k calibration: the per-forward GEMM overhead of the deployed kernels over
the stock kernel, with each projection's tags taken from the 1k maps (part AC item 5).

Per model (the paper's six), one `evaluation.gemm_time` process with the three 1k maps: CUPTI GEMM time on the installed
NativeLinears, cold weights (a 512 MiB L2 flush, rotated weight copies), 3 rotated rounds of 30 calls, T = 1 ... 8192;
per projection the tags of its median-share module (typical) and of its max-share module (densest); the deployed tile
table and its fallbacks. `evaluation.gemm_time_report` then gives the six-model medians in Table D.5's layout.

Step 0: Phi-4 with the 256-window release maps, the paper's validation model. Pass: per tile, the median over its T >=
128 cells (typical tags) within 2 points of the published run's (6.72 / 0.75 / 0.91 % for 8x64 / 16x64 / 256x64, the
paper's source note for Table D.5; the user's relative criterion of 2026-10-13). The published run was on another RTX
PRO 6000 board; every cell's difference is reported.
"""
from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path

from . import artifacts as A
from . import config as C

REF = json.loads((C.HERE / "reference" / "item5.json").read_text())


def gemm_cmd(model, maps: dict, out):
    """evaluation.gemm_time with its defaults (the paper's method), the three maps and this repository's build."""
    return [C.py(model), "-m", "evaluation.gemm_time", "--model", model, "--maps",
            ",".join(f"{u}={maps[u]}" for u in C.UNITS), "--kernel-build-dir", C.build_dir(), "--out", out]


def _rec(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if r.get("status") == "complete" else None


def ge128(rec) -> dict:
    """Per tile and tags, the median over the T >= 128 cells of a record (gemm_time_report.overhead)."""
    from evaluation.gemm_time_report import TAGS, UNITS, overhead
    ts, o = overhead(rec)
    return {f"{u}_{v}": statistics.median(o[f"{u}_{v}"][t] for t in ts if t >= 128) for u in UNITS for v in TAGS}


def run(R, out: Path, dry_run=False, data_root=None):
    d = out / "item5"
    get = A.local if dry_run else A.fetch
    s0 = REF["step0"]
    m0 = s0["model"]
    o = d / "step0" / f"{m0}_256.json"
    R.must(R.run("item5_step0_gemm_256", gemm_cmd(m0, {u: get(m0, "flipquant_256", u) for u in C.UNITS}, o), o),
           "step 0")
    if not dry_run:
        got = ge128(_rec(o))
        diff = {u: got[f"{u}_typical"] - s0["ge128_typical"][u] for u in C.UNITS}
        res = dict(model=m0, ge128=got, published_typical=s0["ge128_typical"], diff_points=diff,
                   criterion_points=s0["criterion_points"],
                   passed=all(abs(v) <= s0["criterion_points"] for v in diff.values()))
        res["accepted"] = not res["passed"] and "item5-step0" in R.accepted()
        (d / "step0" / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
        typ = ", ".join(f"{u} {got[u + '_typical']:+.2f}" for u in C.UNITS)
        R.log(f"item5 step 0: T>=128 typical {typ}; passed {res['passed']}")
        R.gate("item5-step0", res["passed"], f"Phi-4's T >= 128 overheads differ from the published run's by more than "
                                             f"{s0['criterion_points']} points ({diff})")
    for model in C.MODELS:
        o = d / f"{model}.json"
        R.must(R.run(f"item5_gemm_{model}", gemm_cmd(model, {u: get(model, "flipquant_1k", u) for u in C.UNITS}, o), o),
               f"gemm_time {model}")
    R.log("item5 done")


def report(out: Path, dest: Path):
    from evaluation import gemm_time_report as G
    d = out / "item5"
    recs = [d / f"{m}.json" for m in C.MODELS if _rec(d / f"{m}.json")]
    dest.mkdir(parents=True, exist_ok=True)
    md = ["# Item 5: Table D.5 (`tab:app-gemm`) at the 1k calibration", "",
          "Per-forward GEMM overhead of the deployed kernels over the stock kernel (`evaluation.gemm_time`, its defaults: "
          "CUPTI time, cold weights, 3 rotated rounds of 30 calls), with each projection's tags from the 1k maps; "
          "`evaluation.gemm_time_report`'s tables below. Run from flipquant alone (`python -m experiments.paper_1k.run "
          "item5`).", ""]
    s0 = d / "step0" / "step0.json"
    rec = dict(step0=json.loads(s0.read_text()) if s0.exists() else None, published_256=REF["published_256"])
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", "Phi-4 with the 256-window release maps, against the published run (another RTX PRO 6000 "
               "board): the median over the T >= 128 cells, typical tags, % — "
               + "; ".join(f"{u}: {s['ge128'][u + '_typical']:+.2f} (published {s['published_typical'][u]:+.2f}, "
                           f"{s['diff_points'][u]:+.2f} points)" for u in C.UNITS)
               + f". Criterion: within {s['criterion_points']} points per tile: {'pass' if s['passed'] else 'FAIL'}.", ""]
    if recs:
        with tempfile.TemporaryDirectory() as tmp:
            G.main([*map(str, recs), "--out-dir", tmp])
            tables = (Path(tmp) / "tables.md").read_text()
            summ = json.loads((Path(tmp) / "summary.json").read_text())
        summ.pop("records", None)                      # local paths
        rec["summary"] = summ
        md += ["## The six models with the 1k maps", "", tables.replace("# Per-forward GEMM overhead over stock_ko "
                                                                         "(evaluation.gemm_time)", "").strip(), ""]
        md += ["## Against the published table (the 256-window release maps, another board)", "",
               "Each cell: the six-model median at 1k, then the published value; typical tags.", "",
               "| T | 8x64 | 16x64 | 256x64 |", "|---:|---:|---:|---:|"]
        for t in [*map(str, summ["tokens"]), "T>=128"]:
            pub = REF["published_256"].get(t)
            if pub is None:
                continue
            md.append(f"| {t} | " + " | ".join(f"{summ['table'][f'{u}_typical'][t]:+.2f} ({pub[u]['typical']:+.2f})"
                                                for u in C.UNITS) + " |")
    else:
        md += ["The panel has not run yet.", ""]
    (dest / "GEMM.md").write_text("\n".join(md) + "\n")
    (dest / "item5.json").write_text(json.dumps(rec, indent=1) + "\n")
    return f"{dest}/GEMM.md"
