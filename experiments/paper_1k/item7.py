"""Item 7 (Appendix D.6, reading the weights back) at the 1k calibration: `--ownership-check` of the 18 1k maps (six
models x three tiles) and a negative control (NVFP4-RaZeR paper-eval part C1, at 1k; part AC item 7).

- The readback (part C1's command, `evaluation.ppl --mode native --weight mixfp4 --act fourover6 --map <1k map>
  --paper-convention --ownership-check --limit 1`): with the map installed, every quantized linear runs through the
  native kernel on an identity activation, and each decoded weight element must equal its stored value under the map's
  format (the install raises on any mismatch). Per map: modules, weight elements, informative elements, E0M3 tiles,
  E0M3 elements observed, exact, format mismatches.
- The negative control, per model on its 16x64 1k map (experiments/paper_1k/ownership_control.py): the same check must
  fail when an all-E2M1 packing is checked against the map, and the map's packing against an all-E2M1 map.

Step 0: part C1's readback of Mistral-7B's 16x64 release map: its per-module rows equal C1's (reference/item7.json).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import artifacts as A
from . import config as C

REF = json.loads((C.HERE / "reference" / "item7.json").read_text())


def readback_cmd(model, map_path, out):
    """Part C1's command (run_gpu.ownership)."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--mode", "native", "--kernel-build-dir",
            C.build_dir(), "--weight", "mixfp4", "--act", "fourover6", "--map", map_path, "--paper-convention",
            "--ownership-check", "--limit", 1, "--out", out]


def control_cmd(model, map_path, out):
    return [C.py(model), "-m", "experiments.paper_1k.ownership_control", "--model", model, "--map", map_path,
            "--kernel-build-dir", C.build_dir(), "--out", out]


def rows_of(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r["native"]["ownership"] if "resources" in r else None


def rows_digest(own: dict) -> str:
    return hashlib.sha256(json.dumps(own, sort_keys=True).encode()).hexdigest()


def run(R, out: Path, dry_run=False, data_root=None):
    d = out / "item7"
    get = A.local if dry_run else A.fetch
    s0 = REF["step0"]
    o = d / "step0" / f"{s0['model']}_{s0['unit']}_256.json"
    R.must(R.run("item7_step0_readback_256", readback_cmd(s0["model"], get(s0["model"], "flipquant_256", s0["unit"]), o), o),
           "step 0")
    if not dry_run:
        own = rows_of(o)
        res = dict(model=s0["model"], unit=s0["unit"], modules=len(own), rows_sha256=rows_digest(own),
                   rows_equal=rows_digest(own) == s0["rows_sha256"], exact=all(v["exact"] for v in own.values()),
                   mismatches=sum(v["format_mismatches"] for v in own.values()))
        res["accepted"] = not res["rows_equal"] and "item7-step0" in R.accepted()
        (d / "step0" / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
        R.log(f"item7 step 0: rows equal C1's {res['rows_equal']}")
        R.gate("item7-step0", res["rows_equal"], "the readback of the 256 release map differs from part C1's per-module rows "
                                                 f"(exact {res['exact']}, {res['mismatches']} format mismatches)")
    for model in C.MODELS:
        for u in C.UNITS:
            o = d / "readback" / f"{model}_{u}.json"
            R.must(R.run(f"item7_readback_{model}_{u}", readback_cmd(model, get(model, "flipquant_1k", u), o), o),
                   f"readback {model} {u}")
    for model in C.MODELS:
        o = d / "control" / f"{model}_16x64.json"
        R.must(R.run(f"item7_control_{model}", control_cmd(model, get(model, "flipquant_1k", "16x64"), o), o),
               f"negative control {model}")
        if not dry_run and not json.loads(o.read_text())["passed"]:      # a finished record that failed: never skip it
            R.stop(f"negative control {model}: the check did not detect the wrong tags; see {o}")
    R.log("item7 done")


def portable(control: dict) -> dict:
    """A control record without this machine's paths (the published report must carry none): the map as its Hugging
    Face repository and file, the kernel routing without the build directory."""
    import re
    r = dict(control)
    m = re.search(r"models--([^/]+?)--(.+?)/snapshots/([0-9a-f]+)/([^/]+)$", str(r.get("map", "")))
    r["map"] = f"{m.group(1)}/{m.group(2)}@{m.group(3)[:12]}/{m.group(4)}" if m else Path(str(r.get("map", ""))).name
    if isinstance(r.get("routing"), str):
        r["routing"] = r["routing"].split(", from /")[0]
    return r


def report(out: Path, dest: Path):
    d = out / "item7"
    rec = dict(maps={}, controls={}, step0=None, totals={})
    s0 = d / "step0" / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None
    md = ["# Item 7: Appendix D.6 (reading the weights back) at the 1k calibration", "",
          "`evaluation.ppl --ownership-check` of the 18 1k maps on this repository's SM120 build (part C1's command): "
          "every quantized linear runs through the native kernel on an identity activation, and every decoded weight "
          "element must equal its stored value under the map's format. Weight elements: the quantized linears' out x in "
          "(the checkpoints' safetensors headers). Informative elements: those whose two formats decode differently "
          "(every nibble but ±0). Run from flipquant alone (`python -m experiments.paper_1k.run item7`).", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", f"Part C1's readback of Mistral-7B's 16x64 release map: per-module rows equal C1's: "
               f"{s['rows_equal']} ({s['modules']} modules, exact {s['exact']}, {s['mismatches']} format mismatches).", ""]
    md += ["## The readback", "", "| model | tile | modules | weight elements checked | informative elements | E0M3 tiles "
           "| E0M3 elements observed | exact | format mismatches |", "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    tot = dict(maps=0, elements=0, informative=0, exact=True, mismatches=0)
    for model in C.MODELS:
        for u in C.UNITS:
            own = rows_of(d / "readback" / f"{model}_{u}.json")
            if own is None:
                md.append(f"| {C.TITLE[model]} | {u} | TBD | | | | | | |")
                continue
            el = REF["weight_elements"][model]
            o = dict(modules=len(own), modules_expected=el["modules"],
                     elements=el["weight_elements"] if len(own) == el["modules"] else None,
                     informative=sum(v["informative_elements"] for v in own.values()),
                     e0m3_tiles=sum(v["e0m3_tiles"] for v in own.values()),
                     e0m3_elements=sum(v["e0m3_elements_observed"] for v in own.values()),
                     exact=all(v["exact"] for v in own.values()),
                     mismatches=sum(v["format_mismatches"] for v in own.values()))
            rec["maps"].setdefault(model, {})[u] = o
            tot["maps"] += 1
            tot["elements"] += o["elements"] or 0
            tot["informative"] += o["informative"]
            tot["exact"] &= o["exact"]
            tot["mismatches"] += o["mismatches"]
            md.append(f"| {C.TITLE[model]} | {u} | {o['modules']} of {o['modules_expected']} | "
                      f"{'—' if o['elements'] is None else format(o['elements'], ',')} | {o['informative']:,} | "
                      f"{o['e0m3_tiles']:,} | {o['e0m3_elements']:,} | {o['exact']} | {o['mismatches']} |")
    rec["totals"] = tot
    md += ["", f"**{tot['maps']} maps, {tot['elements']:,} weight elements checked: exact {tot['exact']}, "
           f"{tot['mismatches']} format mismatches** (the paper at 256: 181.9 billion elements, exact, 0).", "",
           "## The negative control", "",
           "Per model, the first quantized linear with an E0M3 tile in its 16x64 1k map, packed with the map and with an "
           "all-E2M1 map, through the same check (`ownership_control.py`). The check must fail on both wrong pairs.", "",
           "| model | module | map vs map (exact, mismatches) | all-E2M1 deployed vs map: mismatches | map deployed vs "
           "all-E2M1: mismatches | detected |", "|---|---|---|---:|---:|---|"]
    for model in C.MODELS:
        f = d / "control" / f"{model}_16x64.json"
        if not f.exists():
            md.append(f"| {C.TITLE[model]} | TBD | | | | |")
            continue
        r = portable(json.loads(f.read_text()))
        rec["controls"][model] = r
        x = r["rows"]
        md.append(f"| {C.TITLE[model]} | `{r['module']}` | {x['map_deployed_vs_map']['exact']}, "
                  f"{x['map_deployed_vs_map']['format_mismatches']} | {x['all_e2m1_deployed_vs_map']['format_mismatches']:,} "
                  f"| {x['map_deployed_vs_all_e2m1']['format_mismatches']:,} | {r['passed']} |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "READBACK.md").write_text("\n".join(md) + "\n")
    (dest / "item7.json").write_text(json.dumps(rec, indent=1) + "\n")
    return f"{dest}/READBACK.md"
