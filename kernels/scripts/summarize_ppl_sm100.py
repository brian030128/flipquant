#!/usr/bin/env python3
"""Collect kernels/scripts/ppl_native_vs_sim_sm100.py's runs (03_native_vs_simulated/<model>/{native,simulated}.json)
into the D.6 table and the D.7 ownership counts.

D.6, per (model, corpus): native and simulated PPL; dNLL = mean over windows of nll_native - nll_simulated (paired,
x 1e-3 nats/token; every window's nll is its token-mean cross-entropy and the windows have equal length); 2 SE with
SE = sd / sqrt(n) (sample sd); agree = |dNLL| <= 2 SE. The windows are checked to be the same ones (count, dataset
metadata). Also writes the per-window NLL of each run, <model>/{native,simulated}_nll.csv.

D.7, per model: the native install's ownership check (models/sm100.py): mismatched values over the weight elements
checked, format mismatches over the informative elements, and the negative control. Written to 04_ownership/.

    python kernels/scripts/summarize_ppl_sm100.py [kernels/docs/results/sm100_paper]
"""
import json
import math
from pathlib import Path
import sys

ORDER = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b-instruct", "nemotron-nano-9b-v2", "phi-4",
         "qwen3.8-27b"]
DATASETS = {"wiki": "WikiText-2", "c4": "C4"}


def mean_se(values):
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def main(root):
    root = Path(root)
    d6, d7 = root / "03_native_vs_simulated", root / "04_ownership"
    d7.mkdir(exist_ok=True)
    cells, owners = [], []
    for model in ORDER:
        if not (d6 / model / "native.json").exists() or not (d6 / model / "simulated.json").exists():
            print(f"{model}: missing runs")
            continue
        run = json.loads((d6 / model / "run.json").read_text())
        rep = {r: json.loads((d6 / model / f"{r}.json").read_text()) for r in ("native", "simulated")}
        for r, d in rep.items():
            with open(d6 / model / f"{r}_nll.csv", "w") as f:
                f.write("dataset,window,nll\n")
                for name in DATASETS:
                    for i, v in enumerate(d["results"][name]["nll"]):
                        f.write(f"{name},{i},{v!r}\n")
        for name, label in DATASETS.items():
            a, b = rep["native"]["results"][name], rep["simulated"]["results"][name]
            if a["windows"] != b["windows"] or a["data"] != b["data"] or a["use_cache"] != b["use_cache"]:
                raise SystemExit(f"{model} {name}: the runs evaluated different windows")
            diff = [x - y for x, y in zip(a["nll"], b["nll"])]
            m, se = mean_se(diff)
            cells.append(dict(model=model, dataset=name, windows=a["windows"], ppl_native=a["ppl"], ppl_simulated=b["ppl"],
                              dnll_e3=1e3 * m, two_se_e3=2e3 * se, agree=abs(m) <= 2 * se,
                              identical_windows=sum(1 for v in diff if v == 0), max_abs_window_e3=1e3 * max(map(abs, diff))))
        nat = rep["native"]["native"]
        tot, neg = nat["ownership_total"], nat.get("negative_control")
        owners.append(dict(model=model, map_revision=run["map"]["revision"], map_sha256=run["map"]["sha256"],
                           kernel=nat["kernel"], library=nat["library"], **tot, negative_control=neg,
                           source=rep["native"].get("source")))
    out = dict(cells=cells, agree=sum(c["agree"] for c in cells), of=len(cells))
    (d6 / "table.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["| model | corpus | windows | native PPL | simulated PPL | ΔNLL (×10⁻³ nats/token) | 2 SE (×10⁻³) | agree |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['model']} | {DATASETS[c['dataset']]} | {c['windows']} | {c['ppl_native']:.4f} | "
                     f"{c['ppl_simulated']:.4f} | {c['dnll_e3']:+.3f} | {c['two_se_e3']:.3f} | "
                     f"{'yes' if c['agree'] else 'no'} |")
    lines += ["", f"**Agree within 2 SE: {out['agree']} of {out['of']} cells.**"]
    (d6 / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    (d7 / "ownership.json").write_text(json.dumps(owners, indent=1) + "\n")
    lines = ["| model | value mismatches / weight elements | format mismatches / informative elements | E0M3 tiles | "
             "E0M3 elements observed | negative control (all-E2M1 map) |", "|---|---|---|---|---|---|"]
    for o in owners:
        neg = o["negative_control"]
        negtext = (f"{neg['module']}: {neg['value_mismatches']:,} value / {neg['format_mismatches']:,} format "
                   f"mismatches, {'detected' if neg['detected'] else 'NOT detected'}") if neg else "—"
        lines.append(f"| {o['model']} | {o['value_mismatches']:,} / {o['elements']:,} | {o['format_mismatches']:,} / "
                     f"{o['informative_elements']:,} | {o['e0m3_tiles']:,} | {o['e0m3_elements_observed']:,} | {negtext} |")
    if owners:
        lines += ["", f"**Total: {sum(o['value_mismatches'] for o in owners):,} mismatches over "
                      f"{sum(o['elements'] for o in owners):,} weight elements** ({len(owners)} models, "
                      f"{sum(o['modules'] for o in owners)} modules); format mismatches "
                      f"{sum(o['format_mismatches'] for o in owners):,}."]
    (d7 / "ownership.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fast_table(root, d6)


def paired(a, b):
    if a["windows"] != b["windows"] or a["data"] != b["data"] or a["use_cache"] != b["use_cache"]:
        raise SystemExit("the runs evaluated different windows")
    m, se = mean_se([x - y for x, y in zip(a["nll"], b["nll"])])
    return 1e3 * m, 2e3 * se


def fast_table(root, d6):
    """The fast activation quantizer (native_fast.json, 10_fast_quantizer): its PPL against the simulation and against
    the exact native run, paired per window."""
    out, rows = root / "10_fast_quantizer", []
    for model in ORDER:
        if not (d6 / model / "native_fast.json").exists():
            continue
        rep = {r: json.loads((d6 / model / f"{r}.json").read_text()) for r in ("native_fast", "native", "simulated")}
        assert rep["native_fast"]["native"]["activation_quantizer"].endswith("_fast")
        for name in DATASETS:
            f, n, s = (rep[r]["results"][name] for r in ("native_fast", "native", "simulated"))
            ds, ss = paired(f, s)
            dn, sn = paired(f, n)
            rows.append(dict(model=model, dataset=name, windows=f["windows"], ppl_fast=f["ppl"], ppl_native=n["ppl"],
                             ppl_simulated=s["ppl"], dnll_vs_sim_e3=ds, two_se_vs_sim_e3=ss, dnll_vs_native_e3=dn,
                             two_se_vs_native_e3=sn))
    if not rows:
        return
    out.mkdir(exist_ok=True)
    (out / "ppl.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = ["| model | corpus | native PPL, fast quantizer | native PPL, exact | simulated PPL | ΔNLL fast − simulated "
             "(×10⁻³) ± 2 SE | ΔNLL fast − exact native (×10⁻³) ± 2 SE |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {DATASETS[r['dataset']]} | {r['ppl_fast']:.4f} | {r['ppl_native']:.4f} | "
                     f"{r['ppl_simulated']:.4f} | {r['dnll_vs_sim_e3']:+.3f} ± {r['two_se_vs_sim_e3']:.3f} | "
                     f"{r['dnll_vs_native_e3']:+.3f} ± {r['two_se_vs_native_e3']:.3f} |")
    agree = sum(abs(r["dnll_vs_sim_e3"]) <= r["two_se_vs_sim_e3"] for r in rows)
    lines += ["", f"Fast-quantizer native vs simulated within 2 SE: {agree} of {len(rows)} cells."]
    (out / "ppl.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kernels/docs/results/sm100_paper")
