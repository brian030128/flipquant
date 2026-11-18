"""Item 9 (Tables D.2 `tab:app-ktile` and D.3 `tab:app-c2`, RTX PRO 6000 rows) at the 1k calibration (part AC item 9).

The harness is experiments/paper_1k/ktile.py: amendment 20's cold method (iso_cold and b2b_cold) and the C2 split, at
4096^3, in one session of 10 repeats, all configurations interleaved, with the tags of Mistral-7B's layer 0 o_proj from
the release maps (step 0) and from the 1k maps. The RTX 5090 rows of Table D.2 are unchanged.

Jobs, in order:
1. `ktile build` (CPU): the seven builds beyond build_deploy.sh (the two per-MMA builds, the no-dispatch ceilings, the
   first kernels' no-dispatch builds with and without the site-0 tags). Gate `item9-sass`: each build's patched and
   unpatched SASS equals its reference (kernel-opt's manifests, CUDA 13.1).
2. `ktile census` (CPU): the static SASS census of the five D.2 kernels. Gate `item9-census`: equal to amendment 20's.
3. `ktile check` (GPU): each per-MMA build's output equals its per-K-tile build's bit for bit, for the release and the 1k
   tags, all-E2M1, all-E0M3 and 30 % random tags. Gate `item9-check`.
4. `ktile time` (GPU): the session. Step 0 is its release-tag half against the paper's values (another RTX PRO 6000
   board): the offsets are reported, not gated.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import artifacts as A
from . import config as C
from . import ktile as KT

MODEL = "mistral-7b"


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def run(R, out: Path, dry_run=False, data_root=None):
    d = out / "item9"
    get = A.local if dry_run else A.fetch
    m256 = ",".join(str(get(MODEL, "flipquant_256", u)) for u in C.UNITS)
    m1k = ",".join(str(get(MODEL, "flipquant_1k", u)) for u in C.UNITS)
    ktile = [C.py(MODEL), "-m", "experiments.paper_1k.ktile"]
    gate = KT.extra_build_dir() / "item9_sass_gate.json"
    R.must(R.run("item9_build", [*ktile, "build"], gate, gpu=False, extra_env={"CUDA_VISIBLE_DEVICES": ""}), "the builds")
    if not dry_run:
        g = _json(gate)
        R.gate("item9-sass", g["passed"], f"an item-9 build's SASS differs from its reference ({gate})")
    o = d / "census.json"
    R.must(R.run("item9_census", [*ktile, "census", "--out", o], o, gpu=False, extra_env={"CUDA_VISIBLE_DEVICES": ""}),
           "the census")
    if not dry_run:
        R.gate("item9-census", _json(o)["passed"], f"the static SASS census differs from amendment 20's ({o})")
    o = d / "check.json"
    R.must(R.run("item9_check", [*ktile, "check", "--maps256", m256, "--maps1k", m1k, "--out", o], o), "the check")
    if not dry_run:
        R.gate("item9-check", _json(o)["passed"], f"a per-MMA output differs from its per-K-tile build's ({o})")
    o = d / "time.json"
    R.must(R.run("item9_time", [*ktile, "time", "--maps256", m256, "--maps1k", m1k, "--repeats", 10, "--out", o], o),
           "the session")
    R.log("item9 done")


# ---------------------------------------------------------------------------------------------- the report
def _pm(v, d=1, sign=True):
    """mean ± SD; sign=False for absolute values (µs)."""
    return f"{v[0]:{'+' if sign else ''}.{d}f} ± {v[1]:.{d}f}"


def report(out: Path, dest: Path):
    d = out / "item9"
    rec = dict(gates={}, step0=None, runs={})
    gate = _json(KT.extra_build_dir() / "item9_sass_gate.json")
    census, check, time_ = _json(d / "census.json"), _json(d / "check.json"), _json(d / "time.json")
    rec["gates"] = dict(sass=gate and gate["passed"], census=census and census["passed"], check=check and check["passed"],
                        timed_operands=time_ and time_.get("checks"))
    pub = KT.REF["published"]
    md = ["# Item 9: Tables D.2 (`tab:app-ktile`) and D.3 (`tab:app-c2`), RTX PRO 6000 rows, at the 1k calibration", "",
          "4096³, one session of 10 repeats (each the median of 3 rotated rounds), all configurations interleaved; iso_cold "
          "(512 MiB flush and rotated cold weights per call) and b2b_cold (20 back-to-back calls over the rotating copies); "
          "CUPTI kernel times; mean ± SD over the repeats. Tags: layer 0 o_proj of Mistral-7B's release maps (step 0) and "
          "1k maps. Run from flipquant alone (`python -m experiments.paper_1k.run item9`).", "",
          f"Gates: builds' SASS {rec['gates']['sass']}; census {rec['gates']['census']}; per-MMA == per-K-tile "
          f"{rec['gates']['check']}. The seven extra builds' SASS gate (item9-sass) is the first check of those builds "
          "from flipquant's sources: they could not be built on the machine where item 9 was ported (its GPU driver was "
          "hung). There, on the CPU, the re-vendored generator gave byte-identical blobs for all 47 existing "
          "configurations, the per-MMA blobs equalled kernel-opt's build_KT, and the ported census reproduced amendment "
          "20's.", ""]
    if not (time_ and time_.get("status") == "complete"):
        md += ["The session has not run yet.", ""]
    else:
        mmas = {k: v["omma_per_steady_iteration"]["max"] for k, v in (census or {}).get("kernels", {}).items()}
        for tags, title in (("real256", "release-map tags (step 0)"), ("real1k", "1k-map tags")):
            s = KT.summarize(time_, tags)
            rec["runs"][tags] = s
            iso, b2b = s["iso_cold"], s["b2b_cold"]
            md += [f"## Table D.2 with the {title}", "",
                   "| tile | branch | µs | TFLOP/s | Iso. | B2B | MMAs | speedup |", "|---|---|---:|---:|---:|---:|---:|---:|",
                   f"| stock NVFP4 | | {_pm(iso['d2']['stock']['us'], 2, sign=False)} | {iso['d2']['stock']['tflops'][0]:.1f} | -- | -- | "
                   f"{mmas.get('stock_ko', '')} | |"]
            for u, n in (("8x64", "8"), ("16x64", "16")):
                for kind, label in (("permma", "per MMA"), ("ktile", "per K-tile")):
                    a, b = iso["d2"][u][kind][tags], b2b["d2"][u][kind][tags]
                    md.append(f"| {u} | {label} | {_pm(a['us'], 2, sign=False)} | {a['tflops'][0]:.1f} | {_pm(a['vs_stock'])} % | "
                              f"{_pm(b['vs_stock'])} % | {mmas.get(f'{kind}{n}', '')} | "
                              + (f"{_pm(iso['d2'][u]['speedup'], 3)}x" if kind == "ktile" else "") + " |")
            md += ["", "With all tiles E2M1 (iso_cold): " + "; ".join(
                f"{u} per MMA {_pm(iso['d2'][u]['permma']['e2m1']['vs_stock'])} %, per K-tile "
                f"{_pm(iso['d2'][u]['ktile']['e2m1']['vs_stock'])} %" for u in ("8x64", "16x64")) + ".", "",
                   f"## Table D.3 with the {title} (iso_cold, %)", "", "| part | 8x64 | 16x64 | 256x64 |", "|---|---:|---:|---:|"]
            for side, label in (("before", "Before optimization"), ("after", "After optimization (deployed)")):
                md.append(f"| *{label}* | | | |")
                for part, pl in (("tiles", "Tiles"), ("tag_copies", "Tag copies"), ("dispatch", "Dispatch"),
                                 ("e0m3", "E0M3 tiles"), ("total", "Total")):
                    if part == "tag_copies" and side == "after":
                        md.append("| Tag copies | -- | -- | -- |")
                        continue
                    md.append(f"| {pl} | " + " | ".join(_pm(iso["parts"][side][u][part]) for u in KT.UNITS) + " |")
            md += ["", "All tiles E0M3 against all E2M1 (iso_cold, points): before " + ", ".join(
                f"{u} {_pm(iso['all_e0m3']['before'][u])}" for u in KT.UNITS) + "; after " + ", ".join(
                f"{u} {_pm(iso['all_e0m3']['after'][u])}" for u in KT.UNITS) + ".", ""]
        # step 0: the release-tag half against the paper's values
        s, p2, p3 = rec["runs"]["real256"]["iso_cold"], pub["d2"], pub["d3"]
        off = dict(d2={}, d3={})
        off["d2"]["stock_us"] = s["d2"]["stock"]["us"][0] - p2["stock"]["us"]
        for u in ("8x64", "16x64"):
            for kind in ("permma", "ktile"):
                off["d2"][f"{u} {kind} iso"] = s["d2"][u][kind]["real256"]["vs_stock"][0] - p2[u][kind]["iso"]
            off["d2"][f"{u} speedup"] = s["d2"][u]["speedup"][0] - p2["speedup"][u]
        for side in ("before", "after"):
            for u in KT.UNITS:
                for part, v in p3[side][u].items():
                    off["d3"][f"{side} {u} {part}"] = s["parts"][side][u][part][0] - v[0]
        rec["step0"] = dict(offsets=off, note="this session's release-tag values minus the paper's (another board); reported")
        md += ["## Step 0: the release-tag half against the paper's values (another RTX PRO 6000 board)", "",
               f"Stock: {s['d2']['stock']['us'][0]:.2f} µs here, {p2['stock']['us']:.2f} µs in the paper ("
               f"{off['d2']['stock_us']:+.2f} µs).", "", "| quantity | offset (points; speedup: x) |", "|---|---:|"]
        md += [f"| D.2 {k} | {v:+.3f} |" for k, v in off["d2"].items() if k != "stock_us"]
        md += [f"| D.3 {k} | {v:+.2f} |" for k, v in off["d3"].items()]
        md.append("")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "KTILE.md").write_text("\n".join(md) + "\n")
    (dest / "item9.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return f"{dest}/KTILE.md"
