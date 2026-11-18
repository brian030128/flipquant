"""Item 13 (Appendix F.1, the tile-gradient check, Figure F.1 / Table F.1) at the 1k calibration (part AC item 13).

The original scripts were never pushed. `calibration/gradcheck.py` re-implements the check from the paper's text and its
source notes, and has not run before; its step 0 is its validation against the published 256-sequence results.

Per run (Mistral-7B, 16x64): g_u from `calibration.tmopt_ext oneshot` (the trainer's tile gradient at the all-E2M1 map,
one pass over the fit windows), then `calibration.gradcheck`: thirty groups of 1,000 tiles from ten bins of g_u's rank,
the k most negative tiles for k = 10 ... 3e5, and the paper's map, each flipped from the all-E2M1 map and measured as the
paired change of the trainer's bucketed KL on in-sample (the fit windows) and held-out sequences; Spearman correlations
of g_u and two one-layer scores with the measured decrease, with 95 % bootstrap intervals.

- **Step 0 (a), a smoke run** (`--smoke 16`: 16 sequences per set, two groups, top-10 and the map) at 256 windows: the
  pipeline runs end to end and its checks pass.
- **Step 0 (b), the 256 run** (the release map, the release one-shot gradient, held-out `512:64:128`: the extension
  rule's math and code windows 65-192, as published). Gate `item13-step0` (reference/item13.json `step0_criteria`).
- **The 1k run:** the 1k map, the one-shot gradient over the 1,024 windows (in-sample: all of them), and a NEW held-out
  set outside the 1k calibration set (the old one lies inside it): `2048:448:128`, the extension rule's math and code
  windows 449-576 of the 2,048-window draw, 128 of each, disjoint from the 1,024 by the nesting (checked by token hash).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import artifacts as A
from . import config as C

REF = json.loads((C.HERE / "reference" / "item13.json").read_text())
MODEL, UNIT = "mistral-7b", "16x64"
RUNS = {"256": dict(windows=256, heldout="512:64:128", kind="flipquant_256"),
        "1k": dict(windows=1024, heldout="2048:448:128", kind="flipquant_1k")}
SMOKE = 16


def oneshot_cmd(match, data_root, windows, out):
    """Item 3's one-shot command (part G's, with --fit-windows for 1k)."""
    return [C.py(MODEL), "-m", "calibration.tmopt_ext", "oneshot", "--model", MODEL, "--unit", UNIT, "--match", match,
            "--data-root", data_root, *(["--fit-windows", windows] if windows != 256 else []), "--out", out]


def gradcheck_cmd(oneshot, map_path, data_root, run, out, smoke=None):
    return [C.py(MODEL), "-m", "calibration.gradcheck", "--model", MODEL, "--unit", UNIT, "--oneshot", oneshot, "--map",
            map_path, "--fit-windows", run["windows"], "--heldout", run["heldout"], "--data-root", data_root,
            *(["--smoke", smoke] if smoke else []), "--out", out]


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def step0_check(rep: dict) -> dict:
    """The 256 run against the published results: (1)-(4) of reference/item13.json's step0_criteria."""
    pub = REF["published_256"]
    ck = rep["checks"]
    sets, ana = rep["sets"], rep["analysis"]
    m = sets["map"]
    out = dict(
        own_checks=bool(ck["fit_windows_equal_trainer"] and ck["heldout"]["disjoint"] and ck["heldout"]["distinct"]
                        and ck["steps"]["steps"] == ck["steps"]["expected"]),
        baseline_kl=ck["baseline_kl"]["rel_diff"] is not None and ck["baseline_kl"]["rel_diff"] <= 1e-6,
        map={w: abs(1e3 * m[w]["mean"] - pub["map_milli"][w]["mean"]) <= pub["map_milli"][w]["two_se"]
             for w in ("heldout", "insample")},
        spearman_g_u={w: pub["spearman"]["g_u"][w]["ci95"][0] <= ana["spearman"]["g_u"][w]["rho"]
                      <= pub["spearman"]["g_u"][w]["ci95"][1] for w in ("heldout", "insample")})
    out["passed"] = out["own_checks"] and out["baseline_kl"] and all(out["map"].values()) and all(out["spearman_g_u"].values())
    return out


def run(R, out: Path, dry_run=False, data_root: Path | None = None):
    if data_root is None:
        R.stop("--data-root: the calibration data root of `setup records`")
    d = out / "item13"
    get = A.local if dry_run else A.fetch
    for name, r in RUNS.items():
        mp = get(MODEL, r["kind"], UNIT)
        one = d / name / "oneshot.pt"
        R.must(R.run(f"item13_oneshot_{name}", oneshot_cmd(mp, data_root, r["windows"], one), one.with_suffix(".json")),
               f"one-shot {name}")
        if name == "256":
            # step 0 (a): the smoke run
            o = d / "256" / "smoke.json"
            R.must(R.run("item13_step0_smoke", gradcheck_cmd(one, mp, data_root, r, o, SMOKE), o), "step 0 smoke")
            if not dry_run:
                rep = _json(o)
                ck = rep["checks"]
                ok = (rep["status"] == "complete" and ck["fit_windows_equal_trainer"] and ck["heldout"]["disjoint"]
                      and ck["steps"]["steps"] == ck["steps"]["expected"])
                if not ok:
                    R.stop(f"step 0 smoke: the re-implementation's checks failed ({ck}); see {o}")
        o = d / name / "gradcheck.json"
        R.must(R.run(f"item13_gradcheck_{name}", gradcheck_cmd(one, mp, data_root, r, o), o), f"gradcheck {name}")
        if name == "256" and not dry_run:
            res = step0_check(_json(o))
            res["accepted"] = not res["passed"] and "item13-step0" in R.accepted()
            (d / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
            R.log(f"item13 step 0: {res}")
            R.gate("item13-step0", res["passed"], f"the 256 run does not reproduce the published F.1 results ({res})")
    R.log("item13 done")


# ---------------------------------------------------------------------------------------------- the report
def summary(rep: dict) -> dict:
    sets, ana = rep["sets"], rep["analysis"]
    topk = {s["k"]: s for s in sets.values() if s.get("kind") == "topk"}
    return dict(spearman=ana["spearman"], beyond_2se=ana["beyond_2se"], middle_bins_pooled=ana["middle_bins_pooled"],
                topk={k: dict(predicted_milli=1e3 * s["predicted"],
                              **{w: dict(mean_milli=1e3 * s[w]["mean"], two_se_milli=1e3 * s[w]["two_se"], ratio=s[w]["ratio"])
                                 for w in ("heldout", "insample")}) for k, s in sorted(topk.items())},
                map=dict(tiles=sets["map"]["tiles"], predicted_milli=1e3 * sets["map"]["predicted"],
                         **{w: dict(mean_milli=1e3 * sets["map"][w]["mean"], two_se_milli=1e3 * sets["map"][w]["two_se"],
                                    ratio=sets["map"][w]["ratio"]) for w in ("heldout", "insample")}),
                map_ranks=ana["map_ranks"], baseline=rep["baseline"], checks=rep["checks"], tiles=rep["tiles"])


def report(out: Path, dest: Path):
    d = out / "item13"
    rec = dict(step0=_json(d / "step0.json"), runs={})
    md = ["# Item 13: Appendix F.1 (the tile-gradient check) at the 1k calibration (Mistral-7B, 16x64)", "",
          "`calibration/gradcheck.py`, a re-implementation of F.1 from the paper's text (the original scripts were never "
          "pushed), validated by its 256 run against the published results (step 0). KL changes x 1e-3 nats, paired "
          "over the sequences, ± 2 SE. Run from flipquant alone (`python -m experiments.paper_1k.run item13`).", ""]
    pub = REF["published_256"]
    for name in RUNS:
        rep = _json(d / name / "gradcheck.json")
        if rep is None or rep.get("status") != "complete":
            md += [f"## The {name} run", "", "TBD.", ""]
            continue
        s = rec["runs"][name] = summary(rep)
        md += [f"## The {name} run" + (" (step 0)" if name == "256" else ""), "",
               f"{s['tiles']['total']:,} tiles; the map flips {s['map']['tiles']:,}. In-sample: the {RUNS[name]['windows']} "
               f"fit windows; held-out: `{RUNS[name]['heldout']}` ({s['checks']['heldout']['windows']} windows, disjoint "
               f"{s['checks']['heldout']['disjoint']}). Baseline KL against the trainer's epoch 0: relative difference "
               f"{s['checks']['baseline_kl']['rel_diff']}.", "",
               "| score | held-out | in-sample |" + (" published held-out | published in-sample |" if name == "256" else ""),
               "|---|---|---|" + ("---|---|" if name == "256" else "")]
        for score in ("g_u", "act", "weight_error"):
            row = f"| {score} | " + " | ".join(
                f"{s['spearman'][score][w]['rho']:+.2f} [{s['spearman'][score][w]['ci95'][0]:+.2f}, "
                f"{s['spearman'][score][w]['ci95'][1]:+.2f}]" for w in ("heldout", "insample")) + " |"
            if name == "256":
                row += " " + " | ".join(f"{pub['spearman'][score][w]['rho']:+.2f} [{pub['spearman'][score][w]['ci95'][0]:+.2f}, "
                                        f"{pub['spearman'][score][w]['ci95'][1]:+.2f}]" for w in ("heldout", "insample")) + " |"
            md.append(row)
        b = s["beyond_2se"]
        md += ["", f"Groups beyond 2 SE: held-out {b['heldout']['groups']} ({b['heldout']['agree_with_sign_of_g']} with g_u's "
               f"sign), in-sample {b['insample']['groups']} ({b['insample']['agree_with_sign_of_g']}). Middle bins pooled: "
               + "; ".join(f"{w} {1e3 * s['middle_bins_pooled'][w]['mean']:+.2f} ± {1e3 * s['middle_bins_pooled'][w]['two_se']:.2f}"
                           for w in ("insample", "heldout")) + ".", "",
               "| k | predicted | held-out | ratio | in-sample | ratio |", "|---:|---:|---:|---:|---:|---:|"]
        for k, t in s["topk"].items():
            md.append(f"| {k:,} | {t['predicted_milli']:+.3f} | {t['heldout']['mean_milli']:+.3f} ± {t['heldout']['two_se_milli']:.3f} "
                      f"| {t['heldout']['ratio']:+.2f} | {t['insample']['mean_milli']:+.3f} ± {t['insample']['two_se_milli']:.3f} "
                      f"| {t['insample']['ratio']:+.2f} |")
        mm = s["map"]
        mr = s["map_ranks"]
        md += ["", f"The map: predicted {mm['predicted_milli']:+.2f}; measured held-out {mm['heldout']['mean_milli']:+.2f} ± "
               f"{mm['heldout']['two_se_milli']:.2f} (ratio {mm['heldout']['ratio']:.2f}), in-sample "
               f"{mm['insample']['mean_milli']:+.2f} ± {mm['insample']['two_se_milli']:.2f} (ratio {mm['insample']['ratio']:.2f}). "
               f"Its tiles under g_u: {mr['in_top_same_size']:,} in the top {mm['tiles']:,}, median rank "
               f"{100 * mr['median_rank']:.1f} %, {mr['positive_g']:,} with g_u > 0.", ""]
    if rec["step0"]:
        s0 = rec["step0"]
        md += ["## Step 0", "", f"The 256 run against the published results: own checks {s0['own_checks']}; baseline KL "
               f"{s0['baseline_kl']}; the map within the published 2 SE {s0['map']}; g_u's Spearman inside the published "
               f"intervals {s0['spearman_g_u']} — **{'pass' if s0['passed'] else 'FAIL'}**"
               + (" (accepted by the runner's user)" if s0.get("accepted") else "") + ".", ""]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "GRADCHECK.md").write_text("\n".join(md) + "\n")
    (dest / "item13.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return f"{dest}/GRADCHECK.md"
