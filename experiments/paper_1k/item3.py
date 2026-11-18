"""Item 3 (Table 5, `tab:ablation`) at the 1k calibration: how the map is chosen, on Nemotron-Nano-9B-v2 and
Qwen3.8-27B at 8x64 / 16x64 / 256x64 (NVFP4-RaZeR paper-eval part G, re-registered in part AC item 3).

Every baseline flips exactly the 1k map's E0M3 tiles per layer (`--match`), with statistics from the 1k windows:
random (seeds 0 / 1 / 2), activation-weighted (`baseline_maps --method act`, the BF16 inputs on the 1k windows),
one-shot gradient (`tmopt_ext oneshot --fit-windows 1024`, the trainer's tile gradient at the all-E2M1 start).
Each map's native WikiText-2 / C4 perplexity; the FlipQuant rows are the 1k maps; FourOverSix is the reference.

Step 0, in order (any failure stops the item):
  (a) the 1k smoke: `tmopt_ext oneshot --tiny-layers 3 --fit-windows 1024 --oneshot-steps 2` on random-init 3-layer
      models (Nemotron: GPU teacher; Qwen3.8-27B: --teacher-host). Pass: complete, 2 steps, 1,024 fit windows, the fit
      extension's 896 token hashes equal the paper's 1k set (windows/<model>_1024.json), the teacher on the host
      exactly for Qwen3.8-27B;
  (b) G's Nemotron 16x64 one-shot map rebuilt at 256 windows: its tiles' digest equals reference/item3.json's.
Within the item, FourOverSix and the 1k FlipQuant maps are re-measured, and their per-window NLLs are compared with
the paper's records (reference/item3.json), reported, not gated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import artifacts as A
from . import compare as K
from . import config as C

REF = json.loads((C.HERE / "reference" / "item3.json").read_text())
MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
MAPS = ("random0", "random1", "random2", "act", "oneshot")
TINY = 3


def tiles_digest(path) -> dict:
    """A digest of a map's tiles (module names, shapes, bits; the metadata ignored)."""
    import torch
    from flipquant import maps as M
    tiles, unit, _ = M.load(path)
    h = hashlib.sha256()
    for n in sorted(tiles):
        t = tiles[n].to(torch.bool).contiguous()
        h.update(n.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return dict(map_unit=M.unit_name(unit), modules=len(tiles), e0m3_tiles=int(sum(int(t.sum()) for t in tiles.values())),
                digest=h.hexdigest())


def window_hashes(model, n=1024):
    return json.loads((C.HERE / "windows" / f"{model}_{n}.json").read_text())["token_sha256"]


def ext_hashes(report) -> list:
    return [x["token_sha256"] for x in json.loads(Path(report).read_text())["fit_extension"]["records"]]


def native_ppl(model, map_path, out):
    """run_sm120.native_ppl: G's PPL command."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--mode", "native", "--kernel-build-dir", C.build_dir(),
            "--kernel-set", "auto", "--weight", "mixfp4", "--act", "fourover6", "--map", map_path, "--paper-convention",
            "--out", out]


def fo6_ppl(model, out):
    """FourOverSix's native PPL as part AA re-measured it (run_calib_size: ppl_cmd(quant_fo6(ppl=True))); its per-window
    NLLs equal part F's."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", C.REV[model], "--mode", "native",
            "--kernel-build-dir", C.build_dir(), "--kernel-set", "auto", "--weight", "fourover6", "--act", "fourover6",
            "--paper-convention", "--out", out]


def fq_ppl(model, map_path, out):
    """A FlipQuant map's native PPL as part AA measured the 1k maps (run_calib_size: ppl_cmd(quant_fq(map, ppl=True)))."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", C.REV[model], "--mode", "native",
            "--kernel-build-dir", C.build_dir(), "--kernel-set", "auto", "--weight", "mixfp4", "--act", "fourover6",
            "--map", map_path, "--paper-convention", "--out", out]


def host(model):
    return ["--teacher-host"] if model == "qwen3.8-27b" else []


def run(R, out: Path, dry_run=False, data_root: Path | None = None):
    if data_root is None:
        R.stop("--data-root: the calibration data root of `setup records`")
    d = out / "item3"
    fetch = not dry_run
    get = A.fetch if fetch else A.local
    if not dry_run:
        d.mkdir(parents=True, exist_ok=True)
    ext_file = {m: out / "windows" / f"{m}_1024.json" for m in MODELS}
    if not dry_run:
        for m in MODELS:
            if not ext_file[m].exists() or json.loads(ext_file[m].read_text())["token_sha256"] != window_hashes(m):
                R.stop(f"{ext_file[m]}: run `setup windows` first (its token hashes must be the paper's)")
    # ---- step 0 (a): the 1k smoke on random-init models
    sm = d / "smoke"
    for model in MODELS:
        key = C.TRAINER_KEY[model]
        root = sm / "data_root"
        kdir = root / f"tiny{TINY}" / key
        if not dry_run:
            kdir.mkdir(parents=True, exist_ok=True)
            for x in sorted((data_root / key).iterdir()):          # the development records, linked (read only)
                if x.is_dir() and x.name != "calibration" and not (kdir / x.name).exists():
                    (kdir / x.name).symlink_to(x.resolve())
        R.must(R.run(f"item3_smoke_record_{model}", [C.py(model), "-m", "calibration.tmopt_ext", "record", "--model", model,
                                                    "--tiny-layers", TINY, "--data-root", root],
                     kdir / "calibration" / "report.json"), "smoke record")
        tiny_map = sm / f"{model}_tiny{TINY}_16x64.pt"
        R.must(R.run(f"item3_smoke_remap_{model}", [C.py(model), "-m", "models.tiny", "remap", "--model", model,
                                                   "--tiny-layers", TINY, "--map", get(model, "flipquant_1k", "16x64"),
                                                   "--out", tiny_map], tiny_map, gpu=False), "smoke remap")
        o = sm / f"{model}_oneshot_1024.pt"
        R.must(R.run(f"item3_smoke_oneshot_{model}", [C.py(model), "-m", "calibration.tmopt_ext", "oneshot", "--model", model,
                                                     "--unit", "16x64", "--tiny-layers", TINY, "--fit-windows", 1024,
                                                     "--oneshot-steps", 2, "--match", tiny_map, "--data-root", root,
                                                     "--out", o, *host(model)], o.with_suffix(".json")), "smoke oneshot")
        if not dry_run:
            rep = json.loads(o.with_suffix(".json").read_text())
            trainer = json.loads((o.with_suffix(".run") / "run" / "report.json").read_text())
            got = ext_hashes(o.with_suffix(".run") / "run" / "report.json")
            r = dict(status=rep["status"], steps=rep["meta"]["oneshot"]["steps"], fit_windows=rep["meta"].get("fit_windows"),
                     trainer_fit_windows=trainer["args"].get("fit_windows"), extension_windows=len(got),
                     extension_equal_paper=got == window_hashes(model)[128:],
                     teacher_devices=(trainer.get("teacher_storage") or {}).get("devices"))
            r["passed"] = (r["status"] == "complete" and r["steps"] == 2 and r["fit_windows"] == 1024
                           and r["trainer_fit_windows"] == 1024 and len(got) == 896 and r["extension_equal_paper"]
                           and (r["teacher_devices"] == ["cpu"]) == (model == "qwen3.8-27b"))
            (d / f"step0_smoke_{model}.json").write_text(json.dumps(r, indent=1) + "\n")
            R.log(f"item3 step 0 smoke {model}: {'PASS' if r['passed'] else 'FAIL'} {r}")
            if not r["passed"]:
                R.stop(f"step 0 smoke {model} failed: {r}")
    # ---- step 0 (b): G's 256 one-shot map rebuilt
    s0 = REF["step0_oneshot_256"]
    o = d / "step0" / f"{s0['model']}_{s0['unit']}_oneshot_256.pt"
    R.must(R.run("item3_step0_rebuild_oneshot_256", [C.py(s0["model"]), "-m", "calibration.tmopt_ext", "oneshot", "--model",
                                                     s0["model"], "--unit", s0["unit"], "--match",
                                                     get(s0["model"], "flipquant_256", s0["unit"]), "--data-root", data_root,
                                                     "--out", o], o.with_suffix(".json")), "step 0 rebuild")
    if not dry_run:
        got = tiles_digest(o)
        ok = got["digest"] == s0["digest"]
        grads = hashlib.sha256((o.with_suffix(".run") / "oneshot_grads.pt").read_bytes()).hexdigest()
        (d / "step0_rebuild.json").write_text(json.dumps(dict(
            got=got, want=s0, tiles_equal=ok, grads_file_sha256=grads, grads_file_equal=grads == s0["grads_file_sha256"],
            accepted=not ok and "item3-rebuild" in R.accepted()), indent=1) + "\n")
        R.log(f"item3 step 0 rebuild: tiles equal {ok}, gradient file equal {grads == s0['grads_file_sha256']}")
        R.gate("item3-rebuild", ok, f"the rebuilt 256 one-shot map's tiles differ from part G's ({got['e0m3_tiles']} vs "
                                    f"{s0['e0m3_tiles']} E0M3 tiles)")
    # ---- the panel
    for model in MODELS:
        fo = d / model / "ppl_fo6.json"
        R.must(R.run(f"item3_ppl_{model}_fo6", fo6_ppl(model, fo), fo), f"fo6 ppl {model}")
        for u in C.UNITS:
            match, dd = get(model, "flipquant_1k", u), d / model / u
            for s in (0, 1, 2):
                o = dd / f"random{s}.pt"
                R.must(R.run(f"item3_map_{model}_{u}_random{s}", [C.py(model), "-m", "calibration.baseline_maps", "--method",
                                                                 "random", "--seed", s, "--model", model, "--match", match,
                                                                 "--out", o], o.with_suffix(".json")), f"map {model} {u} random{s}")
            o = dd / "act.pt"
            R.must(R.run(f"item3_map_{model}_{u}_act", [C.py(model), "-m", "calibration.baseline_maps", "--method", "act",
                                                       "--model", model, "--match", match, "--calib-record",
                                                       data_root / C.TRAINER_KEY[model] / "calibration" / "report.json",
                                                       "--calib-extension", ext_file[model], "--batch",
                                                       1 if model == "qwen3.8-27b" else 8, "--out", o],
                         o.with_suffix(".json")), f"map {model} {u} act")
            o = dd / "oneshot.pt"
            R.must(R.run(f"item3_map_{model}_{u}_oneshot", [C.py(model), "-m", "calibration.tmopt_ext", "oneshot", "--model",
                                                           model, "--unit", u, "--match", match, "--data-root", data_root,
                                                           "--fit-windows", 1024, *host(model), "--out", o],
                         o.with_suffix(".json")), f"map {model} {u} oneshot")
            if not dry_run:
                bad = map_checks(model, u, d, match)
                if bad:
                    R.stop(f"{model} {u}: {bad}")
            for m in MAPS:
                o = dd / f"ppl_{m}.json"
                R.must(R.run(f"item3_ppl_{model}_{u}_{m}", native_ppl(model, dd / f"{m}.pt", o), o), f"ppl {model} {u} {m}")
            o = dd / "ppl_flipquant.json"
            R.must(R.run(f"item3_ppl_{model}_{u}_flipquant", fq_ppl(model, match, o), o), f"ppl {model} {u} flipquant")
    R.log("item3 done")


def map_checks(model, u, d, match) -> list:
    """Every baseline has the 1k map's modules, shapes and per-layer E0M3 counts; act used the 1k windows; the one-shot
    trainer's fit extension is the paper's."""
    from flipquant import maps as M
    ref, _, _ = M.load(match)
    bad = []
    for m in MAPS:
        t, _, _ = M.load(d / model / u / f"{m}.pt")
        if set(t) != set(ref) or any(int(t[n].sum()) != int(ref[n].sum()) or t[n].shape != ref[n].shape for n in ref):
            bad.append(f"{m}: per-layer E0M3 counts or shapes differ from the 1k map")
    act = json.loads((d / model / u / "act.json").read_text())
    want = window_hashes(model)
    if act["data"]["fit_windows"] != 1024 or act["data_release"]["token_sha256"] != want:
        bad.append("act: not the 1k windows")
    if ext_hashes(d / model / u / "oneshot.run" / "run" / "report.json") != want[128:]:
        bad.append("oneshot: the fit extension is not the paper's 1k set")
    return bad


# ---------------------------------------------------------------------------------------------- the report
ROWS = [("random", "Random"), ("act", "Activation-weighted"), ("oneshot", "One-shot gradient"), ("flipquant", "FlipQuant")]
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))


def _rec(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if "resources" in r else None


def _nll(path):
    r = _rec(path)
    return {c: r["results"][c]["nll"] for c, _ in CORPORA} if r else None


def paired(x, ref):
    assert len(x) == len(ref)
    d = [a - b for a, b in zip(x, ref)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta_milli=1e3 * m, two_se_milli=2e3 * se, n=n, significant=abs(m) > 2 * se)


def cell(p):
    return f"{p['delta_milli']:+.2f} ± {p['two_se_milli']:.2f}{'*' if p['significant'] else ''}"


def ppl(nll):
    return math.exp(sum(nll) / len(nll))


def report(out: Path, dest: Path):
    d = out / "item3"

    def nll(model, u, row):
        """{corpus: per-window NLL}; Random = the per-window mean of the three seeds."""
        if row == "random":
            recs = [_nll(d / model / u / f"ppl_random{s}.json") for s in (0, 1, 2)]
        else:
            recs = [_nll(d / model / u / f"ppl_{row}.json")]
        if any(r is None for r in recs):
            return None
        return {c: [sum(x) / len(x) for x in zip(*[r[c] for r in recs])] for c, _ in CORPORA}

    rec = dict(cells={}, ppl={}, maps={}, fq_vs={}, vs_256={}, agreement={}, step0={})
    for name in [f"step0_smoke_{m}.json" for m in MODELS] + ["step0_rebuild.json"]:
        if (d / name).exists():
            rec["step0"][name[:-5]] = json.loads((d / name).read_text())
    md = ["# Item 3: Table 5 (`tab:ablation`) at the 1k calibration (Nemotron-Nano-9B-v2, Qwen3.8-27B)", "",
          "Part G's protocol at the 1k calibration (NVFP4-RaZeR paper-eval PROTOCOL.md, part AC item 3), run from "
          "flipquant alone (`python -m experiments.paper_1k.run item3`). Native: this repository's SM120 build, "
          "--kernel-set auto, paper convention. Every baseline flips exactly the 1k map's k_l tiles per layer. The "
          "activation-weighted and one-shot statistics use the 1k windows (the record's 128 + the extension's 896); "
          "Qwen3.8-27B's one-shot keeps the top-1000 teacher on the host. Cells: paired ΔNLL vs FourOverSix, x 1e-3 "
          "nats/token, ± 2 SE (* = beyond 2 SE); Random = the per-window mean of seeds 0 / 1 / 2. FourOverSix and the "
          "1k FlipQuant maps are re-measured in this session (their agreement with the paper's records is below).", ""]
    sm = {k[len("step0_smoke_"):]: v for k, v in rec["step0"].items() if k.startswith("step0_smoke_")}
    rb = rec["step0"].get("step0_rebuild", {})
    if sm or rb:
        md += ["## Step 0", ""]
        if sm:
            md.append("- **Smoke at 1k** (random-init 3-layer models, 2 optimizer steps, `--fit-windows 1024`): " + "; ".join(
                f"{m}: {'pass' if r['passed'] else 'FAIL'} (extension {r['extension_windows']} windows, token hashes "
                f"equal the paper's: {r['extension_equal_paper']}; teacher on {', '.join(r['teacher_devices'] or [])})"
                for m, r in sm.items()) + ".")
        if rb:
            md.append(f"- **The 256 rebuild** (G's Nemotron 16x64 one-shot map on this code): tiles bit for bit "
                      f"{'equal' if rb['tiles_equal'] else 'DIFFERENT'}"
                      + (" — **accepted by the runner's user**" if rb.get("accepted") else "")
                      + f" ({rb['got']['modules']} modules, {rb['got']['e0m3_tiles']} E0M3 tiles); the summed-gradient "
                      f"file {'equal' if rb['grads_file_equal'] else 'differs (reported, not gated)'}.")
        md.append("")
    for c, cname in CORPORA:
        md += [f"## {cname}", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
        lines = [f"% tab:ablation at 1k ({cname}): paired dNLL vs FourOverSix, x1e-3 nats/token, +-2SE; native sm120 "
                 "(kernel-set auto), per-token FourOverSix activations; 1,024 calibration windows"]
        for model in MODELS:
            fo6 = _nll(d / model / "ppl_fo6.json")
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{C.TITLE[model]}}}}} \\\\")
            for row, label in ROWS:
                cells, tcells = [], []
                for u in C.UNITS:
                    x = nll(model, u, row)
                    if x is None or fo6 is None:
                        cells.append("TBD")
                        tcells.append("TBD")
                        continue
                    p = paired(x[c], fo6[c])
                    rec["cells"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = p
                    rec["ppl"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = ppl(x[c])
                    old = REF["published_256"].get(model, {}).get(row, {}).get(u, {}).get(c)
                    if old:
                        rec["vs_256"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = dict(
                            now=p["delta_milli"], published=old["delta_milli"], diff=p["delta_milli"] - old["delta_milli"])
                    cells.append(cell(p))
                    tcells.append(f"${p['delta_milli']:.1f}$\\,{{\\scriptsize$\\pm${p['two_se_milli']:.1f}}}")
                md.append(f"| {C.TITLE[model]} | {label} | " + " | ".join(cells) + " |")
                lines.append(f"{label} & " + " & ".join(tcells) + " \\\\")
        md.append("")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"table_ablation_{c}.tex").write_text("\n".join(lines) + "\n")
    md += ["## Against the published 256-window table (part G)", "",
           "Each cell's ΔNLL vs FourOverSix (x 1e-3) at 1k, then (in parentheses) at 256, WikiText-2 / C4. The 256 "
           "values are the paper's (their FourOverSix reference is part F's record).", "",
           "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
    for model in MODELS:
        for row, label in ROWS:
            v = rec["vs_256"].get(model, {}).get(row, {})
            md.append(f"| {C.TITLE[model]} | {label} | " + " | ".join(
                " / ".join(f"{v[u][c]['now']:+.1f} ({v[u][c]['published']:+.1f})" for c, _ in CORPORA)
                if u in v and len(v[u]) == 2 else "TBD" for u in C.UNITS) + " |")
    md += ["", "## PPL per cell (WikiText-2 / C4)", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
    for model in MODELS:
        fo6 = _nll(d / model / "ppl_fo6.json")
        if fo6:
            md.append(f"| {C.TITLE[model]} | FourOverSix | " + " | ".join(
                " / ".join(f"{ppl(fo6[c]):.4f}" for c, _ in CORPORA) for _ in C.UNITS) + " |")
        for row, label in ROWS:
            pp = rec["ppl"].get(model, {}).get(row, {})
            md.append(f"| {C.TITLE[model]} | {label} | " + " | ".join(
                " / ".join(f"{pp[u][c]:.4f}" for c, _ in CORPORA) if u in pp and len(pp[u]) == 2 else "TBD"
                for u in C.UNITS) + " |")
    md += ["", "## FlipQuant vs each baseline (paired ΔNLL FlipQuant − baseline, x 1e-3, ± 2 SE; negative = FlipQuant "
           "better; WikiText-2; C4)", "", "| model | unit | vs Random | vs Activation-weighted | vs One-shot |",
           "|---|---|---:|---:|---:|"]
    for model in MODELS:
        for u in C.UNITS:
            fq = nll(model, u, "flipquant")
            if fq is None:
                continue
            cells = []
            for row in ("random", "act", "oneshot"):
                b = nll(model, u, row)
                if b is None:
                    cells.append("TBD")
                    continue
                ps = [paired(fq[c], b[c]) for c, _ in CORPORA]
                rec["fq_vs"].setdefault(model, {}).setdefault(u, {})[row] = dict(zip(("wiki", "c4"), ps))
                cells.append("; ".join(cell(q) for q in ps))
            md.append(f"| {C.TITLE[model]} | {u} | " + " | ".join(cells) + " |")
    for model in MODELS:
        for u in C.UNITS:
            for m in ("act", "oneshot"):
                f = d / model / u / f"{m}.json"
                if not f.exists():
                    continue
                r = json.loads(f.read_text())
                info = dict(status=r.get("status"))
                if m == "act":
                    info.update(windows=(r.get("data") or {}).get("fit_windows"),
                                layers_short_of_positive=r.get("layers_short_of_positive"))
                else:
                    meta = r.get("meta", {})
                    info.update(steps=(meta.get("oneshot") or {}).get("steps"), fit_windows=meta.get("fit_windows"),
                                layers_short_of_negative=r.get("layers_short_of_negative"),
                                non_release=meta.get("non_release"),
                                settings={k: (meta.get("settings") or {}).get(k) for k in ("fit_windows", "teacher_topk",
                                                                                          "epochs", "lr", "init_logit")})
                rec["maps"].setdefault(model, {}).setdefault(u, {})[m] = info
    md += ["", "## The maps' records", "", "| model | unit | act windows | act layers short of positives | one-shot steps | "
           "one-shot fit windows | one-shot layers short of negatives |", "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for u in C.UNITS:
            mm = rec["maps"].get(model, {}).get(u, {})
            a, o = mm.get("act", {}), mm.get("oneshot", {})
            md.append(f"| {C.TITLE[model]} | {u} | {a.get('windows')} | {a.get('layers_short_of_positive')} | "
                      f"{o.get('steps')} | {o.get('fit_windows')} | {o.get('layers_short_of_negative')} |")
    # the re-measured FourOverSix / FlipQuant rows against the paper's records (reported, not gated)
    md += ["", "## The re-measured rows against the paper's records", "",
           "FourOverSix: part F's record (= part AA's re-measurement, per window); FlipQuant: part AA's 1k maps' records. "
           "Reported, not gated. exact: equal per window bit for bit; equivalent: the paired ΔNLL within 2 SE on both "
           "corpora; differs: otherwise.", "",
           "| model | row | tier | max abs ΔNLL | PPL now (paper) WikiText-2 / C4 |",
           "|---|---|---|---:|---|"]
    for model in MODELS:
        pairs = [("FourOverSix", _nll(d / model / "ppl_fo6.json"), REF["fo6_nll"][model])]
        pairs += [(f"FlipQuant {u}", _nll(d / model / u / "ppl_flipquant.json"), REF["flipquant_1k_nll"][model][u])
                  for u in C.UNITS]
        for label, now, ref in pairs:
            if now is None:
                continue
            t = K.tier(now, ref)
            rec["agreement"].setdefault(model, {})[label] = dict(t, ppl_now={c: ppl(now[c]) for c, _ in CORPORA},
                                                                 ppl_paper={c: ppl(ref[c]) for c, _ in CORPORA})
            md.append(f"| {C.TITLE[model]} | {label} | {t['tier']} | {t['max_abs']:.3g} | " + " / ".join(
                f"{ppl(now[c]):.4f} ({ppl(ref[c]):.4f})" for c, _ in CORPORA) + " |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ABLATION.md").write_text("\n".join(md) + "\n")
    (dest / "item3.json").write_text(json.dumps(rec, indent=1) + "\n")
    return f"{dest}/ABLATION.md"
