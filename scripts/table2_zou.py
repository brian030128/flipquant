"""Table 2 (paper §2.3): MixFP4 (Zou et al.)'s per-block rule coarsened to hardware tiles, with FourOverSix per-token
activations.

One ``evaluation.ppl`` process per (model, arm), WikiText-2 and C4 on the protocol windows, then the paired analysis:

  bf16                 --mode bf16
  nvfp4                NVFP4 weights, NVFP4 per-token activations (the paper's NVFP4 row)
  fo6                  FourOverSix weights (--paper-convention: FourOverSix per-token activations)
  base                 Zou's E2M1 base (--weight zou_mixfp4_e2m1), the reference of Zou's own gain
  zou-<unit>           --weight zou_mixfp4 --unit <unit>                   unit in 1x16, 8x64, 16x64, 256x64
  zoufo6-<unit>        --weight zou_mixfp4 --unit <unit> --e2m1 fourover6  (+ FourOverSix; its base is fo6)
  flipquant-<unit>     --weight mixfp4 --map <maps>/<model>/flipquant_<unit>.pt (8x64, 16x64, 256x64)

In the tables the arms are named as in the paper: "MixFP4" (Zou et al.), "MixFP4 + FourOverSix", "FlipQuant" (ours).
Every quantized arm except nvfp4 uses ``--paper-convention`` (per-token FourOverSix activations): Table 2 compares the
weight formats only. Reports go to ``<out>/<model>/<arm>.json``; the analysis to ``<out>/table2.json`` and
``<out>/table2.md``: PPL, paired ΔNLL ± 2 SE (evaluation.significance) against FourOverSix, against the rule's own
E2M1 base, and against FlipQuant at the same tile, plus the uniform-format share, the within-tile mixing of the 1x16
choices and the weight error relative to FourOverSix.

  python scripts/table2_zou.py --models llama3.1-8b --out runs/table2 \\
      --launch "python /path/to/gpu_run.py --log {log} --" --jobs 3
  python scripts/table2_zou.py --models llama3.1-8b --out runs/table2 --analyze-only

``--launch`` wraps each command (``{log}`` is replaced by the arm's log path); without it the commands run here, one
at a time. ``--maps`` is the directory of FlipQuant maps (default: the repository's ``maps/``); a model without maps
gets no FlipQuant rows; with maps, ``scripts/map_weight_error.py`` adds FlipQuant's weight error relative to
FourOverSix (``<out>/<model>/flipquant_weight_error.json``, job ``flipquant-werr``). ``--revision`` pins the model
revision of every run; ``--device-map balanced`` spreads a large model over the GPUs ``--launch`` leases.
``--act-method own`` gives the method's arms (its E2M1 base, every unit, the + FourOverSix variant) the method's own
activations, per token at 1x16 on every forward (only the weight rule is coarsened); the references and FlipQuant keep
FourOverSix per-token activations. ``--no-fo6-variant`` leaves out the + FourOverSix arms. Use a separate ``--out`` per
activation setting: the analysis pairs whatever reports the directory holds. ``--reference`` compares the PPLs (and,
for the Zou arms, the installed-weight sha256) with
a reference experiment-A record directory (``results/paper_extra/A/ppl/<model>``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.significance import load_nll, paired  # noqa: E402

METHOD = "zou"                   # arm prefix (the reference's experiment A's name)
WEIGHT = "zou_mixfp4"            # --weight (mixfp4 is FlipQuant's own format)
FO6_VARIANT = "zoufo6"
BASE_WEIGHT = "zou_mixfp4_e2m1"
UNITS = ("1x16", "8x64", "16x64", "256x64")
MAP_UNITS = ("8x64", "16x64", "256x64")
WEIGHT_ERROR = "flipquant_weight_error.json"   # scripts/map_weight_error.py, per model
#: our arm -> the reference's experiment A policy
REFERENCE_NAMES = dict(bf16="bf16", nvfp4="nvfp4", fo6="fo6", base="e2m1z",
                       **{f"{METHOD}-{u}": f"zou-{u}" for u in UNITS},
                       **{f"{FO6_VARIANT}-{u}": f"zoufo6-{u}" for u in UNITS},
                       **{f"flipquant-{u}": f"tc-{u}" for u in MAP_UNITS})
REFERENCE_MODELS = {"llama3.1-8b": "llama8b", "phi4-14b": "phi4"}


#: the names in the tables: MixFP4 = Zou et al.'s method, FlipQuant = ours (the paper's names)
METHOD_LABEL = "MixFP4"


def label(arm: str) -> str:
    fixed = dict(bf16="BF16", nvfp4="NVFP4 (NVFP4 act.)", fo6="FourOverSix", base=f"{METHOD_LABEL} E2M1 base")
    if arm in fixed:
        return fixed[arm]
    kind, unit = arm.split("-", 1)
    return {METHOD: METHOD_LABEL, FO6_VARIANT: f"{METHOD_LABEL} + FourOverSix", "flipquant": "FlipQuant"}[kind] + " " + unit


def arms(model: str, maps: Path, act_method: str | None = None, fo6_variant: bool = True) -> dict[str, list[str]]:
    """The arms and their ``evaluation.ppl`` flags. ``act_method="own"``: the method's arms (its E2M1 base, every unit
    and the + FourOverSix variant) quantize the activations with the method's own rule (``--act-method own``: per
    token, per 16-block, on every forward; the weight rule alone is coarsened). The references and FlipQuant keep
    FourOverSix per-token activations (NVFP4: NVFP4 activations)."""
    paper = ["--mode", "fake", "--paper-convention"]
    method = ["--mode", "fake", "--act-method", "own"] if act_method == "own" else paper
    out = {"bf16": ["--mode", "bf16"],
           "nvfp4": ["--mode", "fake", "--weight", "nvfp4", "--act", "nvfp4", "--act-scope", "row"],
           "fo6": paper + ["--weight", "fourover6"],
           "base": method + ["--weight", BASE_WEIGHT]}
    for u in UNITS:
        out[f"{METHOD}-{u}"] = method + ["--weight", WEIGHT, "--unit", u]
    for u in UNITS if fo6_variant else ():
        out[f"{FO6_VARIANT}-{u}"] = method + ["--weight", WEIGHT, "--unit", u, "--e2m1", "fourover6"]
    for u in MAP_UNITS:
        m = maps / model / f"flipquant_{u}.pt"
        if m.exists():
            out[f"flipquant-{u}"] = paper + ["--weight", "mixfp4", "--map", str(m)]
    return out


def complete(path: Path) -> bool:
    try:
        r = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return "resources" in r and all(c in r.get("results", {}) for c in ("wiki", "c4"))


def run_all(args):
    jobs = []
    for model in args.models:
        for arm, flags in arms(model, args.maps, args.act_method, not args.no_fo6_variant).items():
            if args.arms and arm not in args.arms:
                continue
            out = args.out / model / f"{arm}.json"
            if complete(out) and not args.force:
                continue
            cmd = [args.python, "-m", "evaluation.ppl", "--model", model, *flags, "--out", str(out)]
            if args.revision:
                cmd += ["--revision", args.revision]
            if args.device_map:
                cmd += ["--device-map", args.device_map]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            log = args.out / "logs" / f"{model}_{arm}.log"
            jobs.append((model, arm, cmd, log, out))
        werr = args.out / model / WEIGHT_ERROR
        has_maps = any((args.maps / model / f"flipquant_{u}.pt").exists() for u in MAP_UNITS)
        if has_maps and (not args.arms or "flipquant-werr" in args.arms) and (args.force or not werr.exists()):
            cmd = [args.python, str(REPO / "scripts" / "map_weight_error.py"), "--model", model, "--maps", str(args.maps),
                   "--out", str(werr)] + (["--revision", args.revision] if args.revision else []) + \
                  (["--device-map", args.device_map] if args.device_map else [])
            jobs.append((model, "flipquant-werr", cmd, args.out / "logs" / f"{model}_flipquant-werr.log", werr))
    print(f"{len(jobs)} runs", flush=True)

    def one(job):
        model, arm, cmd, log, out = job
        log.parent.mkdir(parents=True, exist_ok=True)
        full = (shlex.split(args.launch.replace("{log}", str(log))) + cmd) if args.launch else cmd
        with (args.out / "commands.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} START {model} {arm} {shlex.join(full)}\n")
        if args.dry_run:
            return model, arm, 0
        if args.launch:
            rc = subprocess.run(full, cwd=REPO).returncode
        else:
            with log.open("w") as f:
                rc = subprocess.run(full, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        with (args.out / "commands.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} END {model} {arm} rc={rc}\n")
        print(f"{model} {arm}: rc={rc}", flush=True)
        return model, arm, rc

    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as pool:
        results = list(pool.map(one, jobs))
    failed = [(m, a) for m, a, rc in results if rc != 0]
    if failed:
        raise SystemExit(f"failed runs: {failed}")


# --------------------------------------------------------------------------------------------
# analysis


def _cell(p):
    return f"{p['delta']:+.4f} ± {2 * p['se']:.4f}{' *' if p['significant'] else ''}"


def analyze_model(model: str, d: Path, reference: Path | None) -> dict:
    reps = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if complete(p)}
    nll = {k: load_nll(d / f"{k}.json") for k in reps}
    corpora = ("wiki", "c4")
    row = {}
    for arm, r in reps.items():
        e = dict(ppl={c: r["results"][c]["ppl"] for c in corpora})
        ad = r.get("adaptive")
        if ad:
            w = ad["weight"]
            e.update(uniform_fraction=w["uniform_fraction"], uniform_fraction_by_projection=w["uniform_fraction_by_projection"],
                     weight_sq_error=w["weight_sq_error"], weight_error_vs_fo6=w["weight_error_vs_fourover6"],
                     zero_scale_blocks=w["zero_scale_blocks"], e2m1_out_of_grid_elements=w["e2m1_out_of_grid_elements"],
                     installed_weight_sha256=ad["method"]["installed_weight_sha256"])
            if w.get("mixing"):
                e["mixing"] = {t: v["all"] for t, v in w["mixing"].items()}
                e["mixing_by_projection"] = w["mixing"]
        if "map" in r:
            e["uniform_fraction"] = r["map"].get("e0m3_fraction", r["map"].get("fraction"))
            e["map"] = dict(path=r["map"].get("path"), revision=(r["map"].get("meta") or {}).get("revision"))
        e["model_revision"] = (r.get("source") or {}).get("revision")
        q = r.get("quant") or {}
        e["activation"] = "bf16" if r.get("mode") == "bf16" else (
            f"{METHOD_LABEL} own (per token, 1x16)" if q.get("act_method") == "own" else f"{q.get('act')} ({q.get('act_scope')})")
        e["label"] = label(arm)
        row[arm] = e

    werr = d / WEIGHT_ERROR
    if werr.exists():                       # FlipQuant's weight error (scripts/map_weight_error.py)
        for unit, x in json.loads(werr.read_text())["units"].items():
            if f"flipquant-{unit}" in row:
                row[f"flipquant-{unit}"].update(weight_sq_error=x["weight_sq_error"],
                                                weight_error_vs_fo6=x["weight_error_vs_fourover6"],
                                                e0m3_element_fraction=x["e0m3_element_fraction"])

    def cmp(a, b):
        if a in nll and b in nll:
            return {c: paired(nll[a][c]["nll"], nll[b][c]["nll"]) for c in corpora}
        return None
    for arm, e in row.items():
        e["vs_fo6"] = cmp(arm, "fo6") if arm != "fo6" else None
        if arm.startswith(METHOD + "-"):
            e["base"] = "base"
        elif arm.startswith(FO6_VARIANT + "-"):
            e["base"] = "fo6"
        if e.get("base"):
            e["vs_base"] = cmp(arm, e["base"])
            unit = arm.split("-", 1)[1]
            e["vs_flipquant"] = cmp(arm, f"flipquant-{unit}") if unit != "1x16" else None
            e["vs_1x16"] = cmp(arm, arm.split("-")[0] + "-1x16") if unit != "1x16" else None
    out = dict(model=model, arms=row)
    if reference is not None:
        out["reference"] = compare_reference(model, row, nll, reference)
    return out


def compare_reference(model, row, nll, reference: Path) -> dict:
    ref_dir = reference / REFERENCE_MODELS.get(model, model)
    res = {}
    for arm, e in row.items():
        p = ref_dir / f"{REFERENCE_NAMES.get(arm, arm)}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        ev = next(iter(r["evaluations"].values()))
        theirs = ev["evaluation"]
        x = dict(reference=str(p))
        for c in ("wiki", "c4"):
            a, b = nll[arm][c]["nll"], theirs[c]["nll"]
            x[c] = dict(ppl=e["ppl"][c], reference_ppl=theirs[c]["ppl"],
                        ppl_equal=e["ppl"][c] == theirs[c]["ppl"], windows_equal_nll=sum(u == v for u, v in zip(a, b)),
                        windows=len(a), max_abs_nll_diff=max(abs(u - v) for u, v in zip(a, b)),
                        mean_nll_diff=sum(u - v for u, v in zip(a, b)) / len(a))
        if "installed_weight_sha256" in e and "installed_weight_sha256" in ev:
            x["installed_weight_sha256_equal"] = e["installed_weight_sha256"] == ev["installed_weight_sha256"]
        if "format" in ev and "uniform_fraction" in e:
            x["uniform_fraction_equal"] = e["uniform_fraction"] == ev["format"]["uniform_fraction"]
        res[arm] = x
    return res


def markdown(analysis: list[dict]) -> str:
    lines = []
    for a in analysis:
        rows = a["arms"]
        own = sorted({e["activation"] for k, e in rows.items() if k == "base" or k.split("-")[0] in (METHOD, FO6_VARIANT)})
        if not own or own == ["fourover6 (row)"]:
            acts = "FourOverSix per-token activations everywhere except the NVFP4 row (NVFP4 activations)."
        else:
            acts = (f"{METHOD_LABEL} rows and its E2M1 base: {' / '.join(own)} activations (only the weight rule is "
                    "coarsened); FourOverSix and FlipQuant rows: FourOverSix per-token activations; NVFP4 row: NVFP4 "
                    "activations.")
        lines += [f"## {a['model']}", "", "PPL and paired ΔNLL (nats per token = Δ log PPL, ± 2 SE, * = |Δ| > 2 SE). "
                  + acts, "",
                  "| method | PPL wiki | PPL c4 | vs FO6 wiki | vs FO6 c4 | vs own base wiki | vs own base c4 | "
                  "vs FlipQuant wiki | vs FlipQuant c4 | uniform share | weight err / FO6 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        order = ["bf16", "nvfp4", "fo6", "base"] + [f"{METHOD}-{u}" for u in UNITS] + \
                [f"{FO6_VARIANT}-{u}" for u in UNITS] + [f"flipquant-{u}" for u in MAP_UNITS]
        for arm in order:
            if arm not in rows:
                continue
            e = rows[arm]

            def c(key, corpus):
                v = e.get(key)
                return _cell(v[corpus]) if v else "—"
            uf = e.get("uniform_fraction")
            we = e.get("weight_error_vs_fo6")
            lines.append(f"| {label(arm)} | {e['ppl']['wiki']:.4f} | {e['ppl']['c4']:.4f} | {c('vs_fo6', 'wiki')} | "
                         f"{c('vs_fo6', 'c4')} | {c('vs_base', 'wiki')} | {c('vs_base', 'c4')} | "
                         f"{c('vs_flipquant', 'wiki')} | {c('vs_flipquant', 'c4')} | "
                         f"{'—' if uf is None else f'{uf:.4f}'} | {'—' if we is None else f'{we:.4f}'} |")
        mix = [(arm, rows[arm]["mixing"]) for arm in (f"{METHOD}-1x16", f"{FO6_VARIANT}-1x16") if rows.get(arm, {}).get("mixing")]
        if mix:
            lines += ["", "Within-tile mixing of the 1x16 choices (share of tiles holding both formats; mean minority "
                      "share inside those tiles):", "", "| method | tile | mixed tiles | mean minority share |",
                      "|---|---|---:|---:|"]
            for arm, m in mix:
                for t, v in m.items():
                    share = v["mean_minority_share"]
                    lines.append(f"| {label(arm)} | {t} | {v['mixed_fraction']:.4f} | "
                                 f"{'—' if share is None else f'{share:.4f}'} |")
        deg = sorted(((arm, rows[arm]["vs_1x16"]) for arm in rows if rows[arm].get("vs_1x16")),
                     key=lambda av: (av[0].split("-")[0] != METHOD, UNITS.index(av[0].split("-")[1])))
        if deg:
            lines += ["", "Degradation from coarsening, R@g − R@1x16:", "", "| method | wiki | c4 |", "|---|---:|---:|"]
            lines += [f"| {label(arm)} | {_cell(v['wiki'])} | {_cell(v['c4'])} |" for arm, v in deg]
        if a.get("reference"):
            lines += ["", "Against the reference's experiment A (same windows; its runs were on an RTX PRO 6000):", "",
                      "| method | PPL wiki (ours / A) | PPL c4 (ours / A) | windows with equal NLL (wiki, c4) | "
                      "max |ΔNLL| per window | installed weights equal |", "|---|---|---|---|---:|---|"]
            for arm, x in a["reference"].items():
                lines.append(f"| {label(arm)} | {x['wiki']['ppl']:.4f} / {x['wiki']['reference_ppl']:.4f} | "
                             f"{x['c4']['ppl']:.4f} / {x['c4']['reference_ppl']:.4f} | "
                             f"{x['wiki']['windows_equal_nll']}/{x['wiki']['windows']}, "
                             f"{x['c4']['windows_equal_nll']}/{x['c4']['windows']} | "
                             f"{max(x['wiki']['max_abs_nll_diff'], x['c4']['max_abs_nll_diff']):.2e} | "
                             f"{x.get('installed_weight_sha256_equal', '—')} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, type=lambda s: s.split(","), help="comma list of registry keys")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", type=lambda s: s.split(","), default=None, help="only these arms (default all)")
    ap.add_argument("--maps", type=Path, default=REPO / "maps", help="FlipQuant maps: <maps>/<model>/flipquant_<unit>.pt")
    ap.add_argument("--revision", default=None, help="model revision for every run (e.g. the maps' meta revision)")
    ap.add_argument("--act-method", choices=("own",), default=None,
                    help="own: the method's arms use its own activations (--act-method own); default FourOverSix per token")
    ap.add_argument("--no-fo6-variant", action="store_true", help="leave out the + FourOverSix variant's arms")
    ap.add_argument("--device-map", default=None, help="passed to every run, e.g. balanced for a multi-GPU model")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--launch", default=None, help="command prefix for each run, {log} = its log file")
    ap.add_argument("--jobs", type=int, default=1, help="runs at once (each through --launch)")
    ap.add_argument("--limit", type=int, default=None, help="windows per corpus (smoke tests)")
    ap.add_argument("--force", action="store_true", help="rerun finished arms")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--reference", type=Path, default=None,
                    help="the reference's results/paper_extra/A/ppl (compare PPLs and installed weights)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run_all(args)
        if args.dry_run:
            return
    analysis = [analyze_model(m, args.out / m, args.reference) for m in args.models if (args.out / m).exists()]
    (args.out / "table2.json").write_text(json.dumps(analysis, indent=1) + "\n")
    text = markdown(analysis)
    (args.out / "table2.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
