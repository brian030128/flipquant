"""Item 6 (Table A.2, `tab:app-native`) at the 1k calibration: native minus simulated perplexity of the 1k maps on
Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2 and Phi-4 at 8x64 / 16x64 /
256x64 (NVFP4-RaZeR paper-eval part C2 with amendment 9, at 1k; part AC item 6). Qwen3.8-27B is not in the table: its
simulation does not fit one 96 GB GPU.

Per model and tile, both on this GPU and in one env (n16k64; n16k64-fast for Nemotron-Nano-9B-v2, on both sides):
- native: part AA's command (`evaluation.ppl --mode native --kernel-set auto --weight mixfp4 --act fourover6 --map <1k
  map> --paper-convention`);
- simulated: part C2's command (`evaluation.ppl --mode fake --weight mixfp4 --act fourover6 --map <1k map>
  --paper-convention`).
Per corpus the paired ΔNLL native − simulated ± 2 SE; a cell agrees when |ΔNLL| ≤ 2 SE; the count of the 30 cells.

Step 0: Mistral-7B 16x64 with the 256-window release map, both ways, against part C2's simulated record and the
release evaluation's native record (reference/item6.json), each side two-tier (experiments/paper_1k/compare.py):
PASS-exact when both are equal per window bit for bit (then the cell is the published one); PASS-equivalent when not,
but each side's paired ΔNLL against its record is within 2 SE on both corpora; FAIL otherwise (the item stops).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import artifacts as A
from . import compare as K
from . import config as C

REF = json.loads((C.HERE / "reference" / "item6.json").read_text())
MODELS = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b", "nemotron-nano-9b-v2", "phi4-14b"]
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))


def native_cmd(model, map_path, out):
    """Part AA's native PPL (run_calib_size: ppl_cmd(quant_fq(map, ppl=True)))."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", C.REV[model], "--mode", "native",
            "--kernel-build-dir", C.build_dir(), "--kernel-set", "auto", "--weight", "mixfp4", "--act", "fourover6",
            "--map", map_path, "--paper-convention", "--out", out]


def fake_cmd(model, map_path, out):
    """Part C2's simulated PPL (run_gpu.fakeppl)."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--mode", "fake", "--weight", "mixfp4", "--act",
            "fourover6", "--map", map_path, "--paper-convention", "--out", out]


def _nll(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return {c: r["results"][c]["nll"] for c, _ in CORPORA} if "resources" in r else None


def run(R, out: Path, dry_run=False, data_root=None):
    d = out / "item6"
    get = A.local if dry_run else A.fetch
    # ---- step 0: one 256 cell, both ways
    s0 = REF["step0"]
    m0, u0 = s0["model"], s0["unit"]
    rel = get(m0, "flipquant_256", u0)
    on, of = d / "step0" / f"{m0}_{u0}_native_256.json", d / "step0" / f"{m0}_{u0}_fake_256.json"
    R.must(R.run("item6_step0_native_256", native_cmd(m0, rel, on), on), "step 0 native")
    R.must(R.run("item6_step0_fake_256", fake_cmd(m0, rel, of), of), "step 0 fake")
    if not dry_run:
        nat, fake = _nll(on), _nll(of)
        tn, tf = K.tier(nat, s0["native_nll"]), K.tier(fake, s0["fake_nll"])
        res = dict(model=m0, unit=u0, native=tn, fake=tf, result=K.overall(tn, tf), passed=K.passed(tn, tf),
                   cell={c: paired(nat[c], fake[c]) for c, _ in CORPORA}, published=s0["published"])
        res["accepted"] = not res["passed"] and "item6-step0" in R.accepted()
        (d / "step0" / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
        R.log(f"item6 step 0: {res['result']} (native {tn['tier']}, max |dNLL| {tn['max_abs']:.3g}; simulated "
              f"{tf['tier']}, max |dNLL| {tf['max_abs']:.3g})")
        R.gate("item6-step0", res["passed"], f"the 256 cell is not reproduced within 2 SE (native {tn['tier']}, simulated "
                                             f"{tf['tier']}; {d / 'step0' / 'step0.json'})")
    # ---- the panel
    for model in MODELS:
        for u in C.UNITS:
            mp = get(model, "flipquant_1k", u)
            o = d / model / f"{u}_native.json"
            R.must(R.run(f"item6_native_{model}_{u}", native_cmd(model, mp, o), o), f"native {model} {u}")
            o = d / model / f"{u}_fake.json"
            R.must(R.run(f"item6_fake_{model}_{u}", fake_cmd(model, mp, o), o), f"fake {model} {u}")
    R.log("item6 done")


# ---------------------------------------------------------------------------------------------- the report
def paired(x, ref):
    assert len(x) == len(ref)
    d = [a - b for a, b in zip(x, ref)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta_milli=1e3 * m, two_se_milli=2e3 * se, n=n, agree=abs(m) <= 2 * se)


def report(out: Path, dest: Path):
    d = out / "item6"
    rec = dict(cells={}, agree=0, cells_n=0, native_vs_part_aa={}, step0=None)
    s0 = d / "step0" / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None
    md = ["# Item 6: Table A.2 (`tab:app-native`) at the 1k calibration", "",
          "Native minus simulated perplexity of the 1k maps, both on this GPU and in one env per model (n16k64; "
          "n16k64-fast for Nemotron-Nano-9B-v2), on the same windows: native = this repository's SM120 kernels "
          "(--kernel-set auto); simulated = `--mode fake`, the same weights, maps and per-token FourOverSix "
          "activations. Paired ΔNLL native − simulated, x 1e-3 nats/token, ± 2 SE; § = |ΔNLL| beyond 2 SE. In "
          "parentheses: the paper's 256-window cell. Run from flipquant alone (`python -m experiments.paper_1k.run "
          "item6`).", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", f"Mistral-7B 16x64 with the 256-window release map: **{s['result']}** — simulated vs "
               f"part C2's record: {s['fake']['tier']} (max per-window |ΔNLL| {s['fake']['max_abs']:.3g}); native vs the "
               f"release evaluation's: {s['native']['tier']} (max {s['native']['max_abs']:.3g}); the cell "
               + "; ".join(f"{c}: {s['cell'][c]['delta_milli']:+.2f} ± {s['cell'][c]['two_se_milli']:.2f} (published "
                           f"{s['published'][c]['delta_milli']:+.2f} ± {s['published'][c]['two_se_milli']:.2f})"
                           for c, _ in CORPORA) + ".", ""]
    md += ["## Native − simulated", "", "| model | tile | WikiText-2 | C4 |", "|---|---|---:|---:|"]
    tex = ["% tab:app-native at 1k: native - simulated paired dNLL (x1e-3 nats/token, +-2SE) of the 1k maps on the RTX "
           "PRO 6000; S = beyond 2SE"]
    for model in MODELS:
        for u in C.UNITS:
            nat, fake = _nll(d / model / f"{u}_native.json"), _nll(d / model / f"{u}_fake.json")
            pub = REF["published_256"][model][u]
            if nat is None or fake is None:
                md.append(f"| {C.TITLE[model]} | {u} | TBD | TBD |")
                continue
            row = {c: paired(nat[c], fake[c]) for c, _ in CORPORA}
            rec["cells"].setdefault(model, {})[u] = row
            rec["agree"] += sum(row[c]["agree"] for c, _ in CORPORA)
            rec["cells_n"] += 2
            rec["native_vs_part_aa"].setdefault(model, {})[u] = K.tier(nat, REF["native_1k_nll"][model][u])
            md.append(f"| {C.TITLE[model]} | {u} | " + " | ".join(
                f"{row[c]['delta_milli']:+.2f} ± {row[c]['two_se_milli']:.2f}{'' if row[c]['agree'] else ' §'} "
                f"({pub[c]['delta_milli']:+.2f} ± {pub[c]['two_se_milli']:.2f})" for c, _ in CORPORA) + " |")
            ut = "$" + u.replace("x", "{\\times}") + "$"
            tex.append(f"{C.TITLE[model]} & {ut} & " + " & ".join(
                f"${row[c]['delta_milli']:+.2f} \\pm {row[c]['two_se_milli']:.2f}$" + ("" if row[c]["agree"] else "$^{\\S}$")
                for c, _ in CORPORA) + " \\\\")
    md += ["", f"**{rec['agree']} of {rec['cells_n']} cells agree within 2 SE** (the paper at 256: 28 of 30).", "",
           "## The native runs against part AA's records", "",
           "Reported, not gated. exact: equal per window bit for bit; equivalent: the paired ΔNLL within 2 SE on both "
           "corpora; differs: otherwise.", "",
           "| model | tile | tier | max abs ΔNLL | ΔNLL x 1e-3 (WikiText-2; C4) |", "|---|---|---|---:|---|"]
    for model, x in rec["native_vs_part_aa"].items():
        for u, v in x.items():
            md.append(f"| {C.TITLE[model]} | {u} | {v['tier']} | {v['max_abs']:.3g} | " + "; ".join(
                f"{1e3 * v['paired'][c]['delta']:+.3f} ± {1e3 * v['paired'][c]['two_se']:.3f}" for c, _ in CORPORA) + " |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "NATIVE.md").write_text("\n".join(md) + "\n")
    (dest / "item6.json").write_text(json.dumps(rec, indent=1) + "\n")
    (dest / "table_app_native.tex").write_text("\n".join(tex) + "\n")
    return f"{dest}/NATIVE.md"
