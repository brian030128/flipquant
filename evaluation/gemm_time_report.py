"""Tables from evaluation.gemm_time records: the per-forward GEMM overhead of each map unit over stock_ko, in the format
of the FlipQuant paper's Table 15 (Appendix D.5).

    python -m evaluation.gemm_time_report results/gemm_time/*.json --out-dir results/gemm_time

Per (model, unit, module choice, T): overhead = 100 x (per-forward GEMM time of the map on its set / that of FourOverSix
weights on stock_ko - 1), with both sums from the same record (same process, same rotation). Writes ``tables.md`` and
``summary.json``:
- the Table-15 layout: rows T, columns 8x64 / 16x64 / 256x64, each cell the median over the models, typical (densest);
  the last row the median over all (model, T) cells with T >= 128;
- the same per model, with the per-forward times (µs);
- stock_wB_ko over stock_ko, for reference;
- the (projection, set, T) calls whose width or scheduler row came from the tile table's fallback.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

UNITS = ("8x64", "16x64", "256x64")
TAGS = ("typical", "densest")


def overhead(rec):
    s = rec["per_forward_gemm_us"]
    ts = sorted({int(k.split("@")[1]) for k in s})
    out = {}
    for u in UNITS:
        for v in TAGS:
            c = f"{u}_{v}"
            out[c] = {t: 100 * (s[f"{c}@{t}"] / s[f"stock_ko@{t}"] - 1) for t in ts
                      if f"{c}@{t}" in s and f"stock_ko@{t}" in s}
    out["stock_wB_ko"] = {t: 100 * (s[f"stock_wB_ko@{t}"] / s[f"stock_ko@{t}"] - 1) for t in ts
                          if f"stock_wB_ko@{t}" in s and f"stock_ko@{t}" in s}
    return ts, out


def fallbacks(rec):
    out = {}
    for r in rec["rows"]:
        if r.get("set") is None:
            continue
        # a scheduler-row fallback counts only for a set whose table has scheduler rows ((0, 1) is the others' design)
        miss = [w for w, ok in (("width", r.get("width_from_table")),
                                ("schedule", r.get("schedule_from_table") or not r.get("set_has_schedule_rows", True)))
                if not ok]
        if miss:
            key = (r["proj"], f"{r['out']}x{r['inp']}", r["set"])
            out.setdefault(key, {}).setdefault(",".join(miss), []).append(r["tokens"])
    return [dict(proj=p, shape=s, set=st, **{k: sorted(set(v)) for k, v in m.items()}) for (p, s, st), m in out.items()]


def fmt(a, b):
    return f"{a:+.2f} ({b:+.2f})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    recs = [json.loads(p.read_text()) for p in args.records]
    for p, r in zip(args.records, recs):
        if r.get("status") != "complete":
            raise SystemExit(f"{p}: status {r.get('status')}")
    models = [r["model"] for r in recs]
    per = {}
    tokens = None
    for r in recs:
        ts, o = overhead(r)
        tokens = ts if tokens is None else [t for t in tokens if t in ts]
        per[r["model"]] = o
    table = {}
    for u in UNITS:
        for v in TAGS:
            c = f"{u}_{v}"
            table[c] = {t: statistics.median(per[m][c][t] for m in models) for t in tokens}
            cells = [per[m][c][t] for m in models for t in tokens if t >= 128]
            table[c]["T>=128"] = statistics.median(cells)
    L = ["# Per-forward GEMM overhead over stock_ko (evaluation.gemm_time)", "",
         f"Models: {', '.join(models)}. Each cell: 100 x (map on its set / FourOverSix on stock_ko - 1), %, median over "
         "the models; typical module tags, densest in parentheses. Last row: the median over all (model, T) cells with "
         "T >= 128.", ""]
    L += ["GPU per record: " + "; ".join(
        f"{r['model']}: {r['gpu'].get('device')}, driver {r['gpu'].get('driver_version')}, power limit "
        f"{r.get('power_limit_w')} W, busy SM clock median {r.get('sampler', {}).get('busy_sm_mhz_median')} MHz "
        f"(min {r.get('sampler', {}).get('busy_sm_mhz_min')}), sw power-cap samples "
        f"{r.get('sampler', {}).get('sw_power_cap_samples')}" for r in recs) + ".", ""]
    L += ["| T | 8x64 | 16x64 | 256x64 |", "|---:|---:|---:|---:|"]
    for t in tokens:
        L.append(f"| {t} | " + " | ".join(fmt(table[f'{u}_typical'][t], table[f'{u}_densest'][t]) for u in UNITS) + " |")
    L.append("| median, T >= 128 | " + " | ".join(fmt(table[f'{u}_typical']['T>=128'], table[f'{u}_densest']['T>=128'])
                                              for u in UNITS) + " |")
    L += ["", "## Per model", ""]
    for r in recs:
        m = r["model"]
        s = r["per_forward_gemm_us"]
        o = per[m]
        mods = sum(p["modules"] for p in r["projections"].values())
        L += [f"### {m}", "",
              f"{len(r['projections'])} projections, {mods} NativeLinears timed per forward ("
              + ", ".join(f"{p} {v['shape'][0]}x{v['shape'][1]} x{v['modules']}" for p, v in r["projections"].items())
              + "). Maps: " + ", ".join(f"{u} {v['sha256'][:12]}" for u, v in r["maps"].items()) + ". "
              + f"GPU: {r['gpu'].get('device')}, driver {r['gpu'].get('driver_version')}, power limit "
              f"{r.get('power_limit_w')} W, busy SM clock median {r.get('sampler', {}).get('busy_sm_mhz_median')} MHz. "
              + "Linear modules not on the NVFP4 path (excluded): "
              + (", ".join(sorted({n.rsplit('.', 1)[-1] for v in r.get('policies_linear_not_native', {}).values()
                                   for n in v})) or "none") + ".", "",
              "| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |", "|---:|---:|---:|---:|---:|---:|"]
        for t in tokens:
            L.append(f"| {t} | {s[f'stock_ko@{t}']:,.1f} | " + " | ".join(
                fmt(o[f'{u}_typical'][t], o[f'{u}_densest'][t]) for u in UNITS) + f" | {o['stock_wB_ko'][t]:+.2f} |")
        L.append("| median, T >= 128 | | " + " | ".join(fmt(
            statistics.median(o[f'{u}_typical'][t] for t in tokens if t >= 128),
            statistics.median(o[f'{u}_densest'][t] for t in tokens if t >= 128)) for u in UNITS) + " | |")
        L.append("")
    L += ["## Tile-table fallbacks", "",
          "Calls whose width (``fallback_width``) or scheduler row ((0, 1)) did not come from the deployed tile table "
          "(scheduler rows only for the sets whose table has them).", "",
          "| model | projection | shape | set | fallback | T |", "|---|---|---|---|---|---|"]
    fb = {}
    for r in recs:
        fb[r["model"]] = fallbacks(r)
        for f in fb[r["model"]]:
            for kind in ("width,schedule", "width", "schedule"):
                if kind in f:
                    L.append(f"| {r['model']} | {f['proj']} | {f['shape']} | {f['set']} | {kind} | "
                             f"{', '.join(map(str, f[kind]))} |")
    L.append("")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables.md").write_text("\n".join(L) + "\n")
    json.dump(dict(models=models, tokens=tokens, table={c: {str(k): v for k, v in d.items()} for c, d in table.items()},
                   per_model={m: {c: {str(t): v for t, v in d.items()} for c, d in o.items()} for m, o in per.items()},
                   per_forward_gemm_us={r["model"]: r["per_forward_gemm_us"] for r in recs}, fallbacks=fb,
                   records=[str(p) for p in args.records],
                   flipquant={r["model"]: r["flipquant"] for r in recs}),
              open(args.out_dir / "summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
