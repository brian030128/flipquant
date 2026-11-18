"""Item 2 (Table 4 `tab:latency` SM120 rows and the memory paragraph): what remains is the memory probe.

The latency panel (210 prefill jobs) is done and published (results/paper_1k/item2/, NVFP4-RaZeR paper-eval f60666e,
flipquant 33bad82). Do not re-run it: this module refuses to. The memory probe is part B's (mem_probe.py: flipquant's
loader and install, then the installed bytes, the allocation after the load and the peak during one 1x2048 prefill),
on the 1k maps, for the six models and seven policies:

1. Step 0, the anchor: Mistral-7B FlipQuant 16x64 with the 256-window release map. Pass: installed and whole-model
   bytes equal part B's record, after-load and 1x2048-peak allocations within 0.02 GiB (reference/item2.json).
2. The panel: 6 models x 7 policies (bf16, nvfp4, fo6, fo6-wB, fq-8x64 / 16x64 / 256x64), one process each.

Records: <out>/item2/memory_anchor/mistral-7b/fq-16x64.json and <out>/item2/memory/<model>/<policy>.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import artifacts as A
from . import config as C

REF = json.loads((C.HERE / "reference" / "item2.json").read_text())
PROBE = C.REPO / "experiments" / "paper_1k" / "mem_probe.py"
GIB = 2 ** 30


def maps_1k(model, fetch=True):
    get = A.fetch if fetch else A.local
    return {u: get(model, "flipquant_1k", u) for u in C.UNITS}


def maps_release(model, fetch=True):
    get = A.fetch if fetch else A.local
    return {u: get(model, "flipquant_256", u) for u in C.UNITS}


def probe_cmd(model, pol, maps, out, build=None):
    """run_paper1k.memory_job's command (part AC item 2), in this repository."""
    return [C.py(model), PROBE, "--model", model, *C.policy_args(pol, maps, build), "--label", pol, "--out", out]


def anchor_check(rec_path) -> dict:
    new = json.loads(Path(rec_path).read_text())
    old = REF["anchor"]["expected"]
    tol = REF["anchor"]["tol_gib"]
    res = dict(installed_equal=new["quantized_linears"]["installed_bytes"] == old["quantized_linears"]["installed_bytes"],
               model_equal=new["model_bytes"] == old["model_bytes"],
               after_load_diff_gib=(new["allocated_after_load"] - old["allocated_after_load"]) / GIB,
               peak_diff_gib=(new["peak_prefill_1x2048"] - old["peak_prefill_1x2048"]) / GIB, tol_gib=tol)
    res["passed"] = (res["installed_equal"] and res["model_equal"] and abs(res["after_load_diff_gib"]) <= tol
                     and abs(res["peak_diff_gib"]) <= tol)
    return res


def run(R, out: Path, dry_run=False, data_root=None):
    """Step 0 (the anchor), then the 42-job panel; stops on a failed job or a failed anchor."""
    d = out / "item2"
    fetch = not dry_run
    rec = d / "memory_anchor" / "mistral-7b" / "fq-16x64.json"
    R.must(R.run("item2_memory_anchor_mistral-7b_fq-16x64", probe_cmd("mistral-7b", "fq-16x64", maps_release("mistral-7b", fetch),
                                                                 rec), rec), "item 2 memory anchor")
    if not dry_run:
        res = anchor_check(rec)
        res["accepted"] = not res["passed"] and "item2-anchor" in R.accepted()
        (d / "memory_anchor_check.json").write_text(json.dumps(res, indent=1) + "\n")
        R.log(f"item2 memory anchor check: {res}")
        R.gate("item2-anchor", res["passed"], f"the memory anchor differs from part B's record: {res}")
    for model in C.MODELS:
        maps = maps_1k(model, fetch)
        for pol in C.POLICIES:
            rec = d / "memory" / model / f"{pol}.json"
            R.must(R.run(f"item2_memory_{model}_{pol}", probe_cmd(model, pol, maps, rec), rec), f"item 2 memory {model} {pol}")
    R.log("item2 memory done")


# ---------------------------------------------------------------------------------------------- the report
def report(out: Path, dest: Path):
    """MEMORY.md and memory.json in dest (results/paper_1k/item2/), B's columns and the paper's memory claims; the
    latency files there are left as published."""
    d = out / "item2"
    a = json.loads((d / "memory_anchor_check.json").read_text())
    names = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "fo6-wB": "FourOverSix (wB)",
             "fq-8x64": "FlipQuant 8x64", "fq-16x64": "FlipQuant 16x64", "fq-256x64": "FlipQuant 256x64"}
    fqs = ["fq-8x64", "fq-16x64", "fq-256x64"]
    mem = {}
    for m in C.MODELS:
        for p in C.POLICIES:
            f = d / "memory" / m / f"{p}.json"
            if f.exists():
                r = json.loads(f.read_text())
                if r.get("status") == "complete":
                    q = r["quantized_linears"]
                    mem.setdefault(m, {})[p] = dict(model_bytes=r["model_bytes"], installed_bytes=q["installed_bytes"],
                                                    bf16_bytes=q["bf16_bytes"], installed_over_bf16=q["installed_over_bf16"],
                                                    allocated_after_load=r["allocated_after_load"],
                                                    peak_prefill_1x2048=r["peak_prefill_1x2048"],
                                                    gpu={k: v for k, v in (r.get("gpu") or {}).items() if k != "host"})
    n = sum(len(v) for v in mem.values())
    total = len(C.MODELS) * len(C.POLICIES)
    L = ["# Item 2: memory on the RTX PRO 6000 at 1k (the paper's memory paragraph)", "",
         "Part B's probe (`experiments/paper_1k/mem_probe.py`: flipquant's loader and install, this repository's SM120 "
         "build) with the 1k maps (the Calib1k repositories), fo6-wB added to part B's six policies.", "",
         "- **Weights:** every parameter and buffer on the GPU after the load and install.",
         "- **After load:** torch.cuda.memory_allocated.",
         "- **Peak:** torch.cuda.max_memory_allocated during one 1x2048 eager prefill.", ""]
    if n < total:
        L += [f"**PENDING:** {n} of the {total} panel jobs are done.", ""]
    L += ["## Step 0: the anchor (Mistral, FlipQuant 16x64 with the 256-window release map, against part B's record)", "",
          f"- Installed quantized linears equal: {a['installed_equal']}; whole model equal: {a['model_equal']}.",
          f"- After load Δ {a['after_load_diff_gib']:+.4f} GiB; 1x2048 peak Δ {a['peak_diff_gib']:+.4f} GiB (tolerance "
          f"±{a['tol_gib']} GiB). **{'PASS' if a['passed'] else 'FAIL'}.**", "",
          "## The panel", "", "| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |",
          "|---|---|---:|---:|---:|---:|---:|"]
    claims = {}
    for m in C.MODELS:
        mm = mem.get(m, {})
        for p in C.POLICIES:
            if p in mm and "bf16" in mm:
                r = mm[p]
                L.append(f"| {C.TITLE[m]} | {p} | {r['model_bytes'] / GIB:.2f} GiB | {r['installed_over_bf16']:.4f} | "
                         f"{r['model_bytes'] / mm['bf16']['model_bytes']:.3f} | {r['allocated_after_load'] / GIB:.2f} GiB | "
                         f"{r['peak_prefill_1x2048'] / GIB:.2f} GiB |")
        if all(p in mm for p in C.POLICIES):
            c = dict(installed_equal={}, model_equal={}, after_load_max_diff_gib=0.0, peak_max_diff_gib=0.0)
            for fq in fqs:
                for ref in ("nvfp4", "fo6", "fo6-wB"):
                    x, y = mm[fq], mm[ref]
                    c["installed_equal"][f"{fq} vs {ref}"] = x["installed_bytes"] == y["installed_bytes"]
                    c["model_equal"][f"{fq} vs {ref}"] = x["model_bytes"] == y["model_bytes"]
                    c["after_load_max_diff_gib"] = max(c["after_load_max_diff_gib"],
                                                      abs(x["allocated_after_load"] - y["allocated_after_load"]) / GIB)
                    c["peak_max_diff_gib"] = max(c["peak_max_diff_gib"],
                                                 abs(x["peak_prefill_1x2048"] - y["peak_prefill_1x2048"]) / GIB)
            c["byte_equal"] = all(c["installed_equal"].values()) and all(c["model_equal"].values())
            c["within_0p02_gib"] = c["after_load_max_diff_gib"] <= 0.02 and c["peak_max_diff_gib"] <= 0.02
            c["bf16_over_model"] = mm["bf16"]["model_bytes"] / mm["fq-16x64"]["model_bytes"]
            pub = REF["published_256"].get(m, {})
            c["vs_published_256"] = {p: dict(bytes_equal=pub[p]["model_bytes"] == mm[p]["model_bytes"],
                                             after_load_diff_gib=(mm[p]["allocated_after_load"] - pub[p]["allocated_after_load"]) / GIB,
                                             peak_diff_gib=(mm[p]["peak_prefill_1x2048"] - pub[p]["peak_prefill_1x2048"]) / GIB)
                                     for p in C.POLICIES if p in pub}
            claims[m] = c
    L += ["", "## The paper's memory claims, per model", ""]
    for m in C.MODELS:
        c = claims.get(m)
        if not c:
            L.append(f"- {C.TITLE[m]}: incomplete")
            continue
        L.append(f"- **{C.TITLE[m]}:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and "
                 f"FourOverSix (wB): **{'yes' if c['byte_equal'] else 'NO'}**; after load and at the 1x2048 peak at most "
                 f"{c['after_load_max_diff_gib']:.4f} / {c['peak_max_diff_gib']:.4f} GiB apart (within 0.02 GiB: "
                 f"**{'yes' if c['within_0p02_gib'] else 'NO'}**); BF16 / FlipQuant whole model {c['bf16_over_model']:.2f}x.")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "MEMORY.md").write_text("\n".join(L) + "\n")
    (dest / "memory.json").write_text(json.dumps(dict(anchor=a, memory=mem, claims=claims, done=n, of=total), indent=1) + "\n")
    return n, total
