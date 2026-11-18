"""Item 12 (Table C.5, `tab:app-robust`) at the 1k calibration: calibration choices on Mistral-7B at 16x64 (the paper's
Appendix C.4, flipquant calib-robustness 932e01b, at 1,024 sequences; part AC item 12).

Each run is part AA's 1k calibration command (`calibration.train_map --epochs 5 --teacher-topk 1000 --fit-windows 1024`,
the n16k64 env) with one change, and the rest of the paper's settings:
- seed 1 and seed 2 (`--seed`: the per-epoch order of the calibration sequences);
- non-deterministic kernels (`--no-deterministic`);
- other calibration data, 1,024 sequences of 512 tokens each (`--fit-source`): generic web text (FineWeb-Edu),
  instruction data (UltraChat), the WikiText-2 training split. Before its training, each source's windows pass the
  13-gram audit against every evaluation text, with its positive control (experiments/paper_1k/audit.py; gate
  `item12-audit`); after it, the trained windows must be the audited ones.
Each map, the 1k paper map and FourOverSix are evaluated natively (part AA's PPL command). Reported per run: the paired
ΔNLL against the 1k paper map ± 2 SE, the E0M3 share, the Jaccard index of the E0M3 tiles with the paper map's.

Step 0 (gate `item12-step0`): a seed-0 recalibration at 256 windows (part AA's step-0 command) reproduces the paper's
release map bit for bit, as the paper states for the environment of these runs. A map digest: an exact gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import artifacts as A
from . import compare as K
from . import config as C
from .item3 import tiles_digest

REF = json.loads((C.HERE / "reference" / "item12.json").read_text())
MODEL, UNIT = "mistral-7b", "16x64"
ROWS = {"seed1": ["--seed", 1], "seed2": ["--seed", 2], "nondet": ["--no-deterministic"],
        "fineweb-edu": ["--fit-source", "fineweb-edu"], "ultrachat": ["--fit-source", "ultrachat"],
        "wikitext2-train": ["--fit-source", "wikitext2-train"]}
LABEL = {"seed1": "Seed 1", "seed2": "Seed 2", "nondet": "Non-deterministic", "fineweb-edu": "Web text (FineWeb-Edu)",
         "ultrachat": "Instruction data (UltraChat)", "wikitext2-train": "WikiText-2 train"}
SOURCES = ("fineweb-edu", "ultrachat", "wikitext2-train")
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))


def train_cmd(extra, data_root, out, windows=1024):
    """Part AA's calibration command (run_calib_size.train_map_cmd) with the run's change before --out."""
    return [C.py_release(), "-m", "calibration.train_map", "--model", MODEL, "--unit", UNIT, "--epochs", 5,
            "--teacher-topk", 1000, "--fit-windows", windows, "--data-root", data_root, *extra, "--out", out]


def ppl_cmd(quant, out):
    return [C.py(MODEL), "-m", "evaluation.ppl", "--model", MODEL, "--revision", C.REV[MODEL], "--mode", "native",
            "--kernel-build-dir", C.build_dir(), "--kernel-set", "auto", *quant, "--paper-convention", "--out", out]


def fq(map_path):
    return ["--weight", "mixfp4", "--act", "fourover6", "--map", map_path]


FO6 = ["--weight", "fourover6", "--act", "fourover6"]


def train(R, name, cmd, o: Path, what: str):
    """One calibration into a fresh directory; an interrupted one stops the item (the driver deletes nothing)."""
    if (o / "reproduction.json").exists():
        return
    if o.exists():
        R.stop(f"{o} exists without reproduction.json (an interrupted calibration): move it aside, then run again")
    R.must(R.run(name, cmd, o / "reproduction.json", make_parent=False), what)


def run(R, out: Path, dry_run=False, data_root: Path | None = None):
    if data_root is None:
        R.stop("--data-root: the calibration data root of `setup records`")
    d = out / "item12"
    get = A.local if dry_run else A.fetch
    # ---- step 0: a seed-0 recalibration at 256 windows reproduces the release map
    o = d / "step0" / "fq16_256"
    cmd = train_cmd([], data_root, o, windows=256)
    if dry_run:
        R.run("item12_step0_train_256", cmd, o / "reproduction.json")
    else:
        train(R, "item12_step0_train_256", cmd, o, "step 0")
        got = tiles_digest(o / "map.pt")
        ok = got["digest"] == REF["step0"]["digest"]
        (d / "step0" / "step0.json").write_text(json.dumps(dict(got=got, want=REF["step0"], tiles_equal=ok,
                                                                accepted=not ok and "item12-step0" in R.accepted()),
                                                           indent=1) + "\n")
        R.gate("item12-step0", ok, f"the seed-0 256 recalibration's tiles differ from the release map's "
                                   f"({got['e0m3_tiles']} vs {REF['step0']['e0m3_tiles']} E0M3 tiles)")
    # ---- the audits of the three fit sources (CPU), before any training on them
    for s in SOURCES:
        o = d / "audit" / f"{s}.json"
        rc = R.run(f"item12_audit_{s}", audit_cmd(s, data_root, o), o, gpu=False)
        if dry_run:
            continue
        rec = _json(o)                  # the audit exits 1 on a hit with a complete record: the gate decides
        if rec is None or rec.get("status") != "complete":
            R.must(rc or 1, f"audit {s}")
        rec["accepted"] = not rec["passed"] and "item12-audit" in R.accepted()
        o.write_text(json.dumps(rec, indent=1) + "\n")
        R.gate("item12-audit", rec["passed"], f"{s}: {len(rec['hits'])} shared 13-grams with evaluation texts, control "
                                              f"ok {rec['control']['ok']} ({o})")
    # ---- the six calibrations
    for row, extra in ROWS.items():
        o = d / row
        cmd = train_cmd(extra, data_root, o)
        if dry_run:
            R.run(f"item12_train_{row}", cmd, o / "reproduction.json")
            continue
        train(R, f"item12_train_{row}", cmd, o, f"train {row}")
        if row in SOURCES:
            trained = [r["token_sha256"] for r in json.loads((o / "run" / "report.json").read_text())["fit_source"]["records"]]
            audited = json.loads((d / "audit" / f"{row}.json").read_text())["token_list_sha256"]
            if hashlib.sha256("".join(trained).encode()).hexdigest() != audited:
                R.stop(f"{row}: the trained windows are not the audited ones")
    # ---- PPL: the 1k paper map, FourOverSix, the six maps
    o = d / "ppl" / "paper_1k.json"
    R.must(R.run("item12_ppl_paper_1k", ppl_cmd(fq(get(MODEL, "flipquant_1k", UNIT)), o), o), "ppl paper map")
    o = d / "ppl" / "fo6.json"
    R.must(R.run("item12_ppl_fo6", ppl_cmd(FO6, o), o), "ppl fo6")
    for row in ROWS:
        o = d / "ppl" / f"{row}.json"
        R.must(R.run(f"item12_ppl_{row}", ppl_cmd(fq(d / row / "map.pt"), o), o), f"ppl {row}")
    if not dry_run:                     # the E0M3 shares and the overlaps with the paper map (CPU)
        pm = tiles(get(MODEL, "flipquant_1k", UNIT))
        ov = dict(paper=overlap(pm, pm), **{row: overlap(tiles(d / row / "map.pt"), pm) for row in ROWS})
        (d / "overlap.json").write_text(json.dumps(ov, indent=1) + "\n")
    R.log("item12 done")


def audit_cmd(source, data_root, out):
    return [C.py_release(), "-m", "experiments.paper_1k.audit", "--model", MODEL, "--fit-source", source,
            "--fit-windows", 1024, "--data-root", data_root, "--out", out]


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------- the report
def _nll(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return {c: r["results"][c]["nll"] for c, _ in CORPORA} if "resources" in r else None


def tiles(path):
    from flipquant import maps as M
    t, _, _ = M.load(path)
    return t


def overlap(a, b) -> dict:
    """E0M3 share of a; Jaccard index of the E0M3 tiles of a and b."""
    inter = sum(int((a[n] & b[n]).sum()) for n in b)
    union = sum(int((a[n] | b[n]).sum()) for n in b)
    total = sum(a[n].numel() for n in a)
    return dict(e0m3_pct=100 * sum(int(a[n].sum()) for n in a) / total, jaccard=inter / union if union else None)


def cell(p):
    sig = abs(p["delta"]) > p["two_se"]
    return f"{1e3 * p['delta']:+.2f} ± {1e3 * p['two_se']:.2f}{'*' if sig else ''}"


def report(out: Path, dest: Path):
    d = out / "item12"
    rec = dict(rows={}, step0=None, audits={}, fo6=None)
    s0 = d / "step0" / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None
    md = ["# Item 12: Table C.5 (`tab:app-robust`) at the 1k calibration (Mistral-7B, 16x64)", "",
          "Part AA's 1k calibration with one change per row (flipquant's calib-robustness options), evaluated natively "
          "with part AA's PPL command. Paired ΔNLL against the 1k paper map, x 1e-3 nats/token ± 2 SE (* beyond 2 SE; "
          "negative is better); E0M3: tiles flipped (%); Jaccard: E0M3 tiles shared with the paper map. In "
          "parentheses: the paper's 256-sequence row. Run from flipquant alone (`python -m experiments.paper_1k.run "
          "item12`).", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", f"A seed-0 recalibration at 256 windows: tiles bit for bit {'equal to' if s['tiles_equal'] else 'DIFFERENT from'} "
               f"the release map's" + (" — accepted by the runner's user" if s.get("accepted") else "")
               + f" ({s['got']['e0m3_tiles']} E0M3 tiles).", ""]
    for s in SOURCES:
        f = d / "audit" / f"{s}.json"
        if f.exists():
            a = json.loads(f.read_text())
            rec["audits"][s] = {k: a.get(k) for k in ("passed", "windows", "scanned_13grams", "windows_with_hits",
                                                      "control", "accepted", "eval_sets", "c4_documents")}
            rec["audits"][s]["hits"] = len(a.get("hits", []))
    ref = _nll(d / "ppl" / "paper_1k.json")
    fo6 = _nll(d / "ppl" / "fo6.json")
    ov = _json(d / "overlap.json") or {}
    rec["overlap"] = ov
    md += ["## Table C.5 at 1k", "", "| calibration | WikiText-2 | C4 | E0M3 (%) | Jaccard |", "|---|---:|---:|---:|---:|"]
    pub = REF["published_256"]
    if "paper" in ov:
        md.append(f"| Paper's map (1k) | 0 | 0 | {ov['paper']['e0m3_pct']:.2f} ({pub['paper']['e0m3_pct']:.2f}) | 1 |")
    if ref and fo6:
        rec["fo6"] = {c: K.paired(fo6[c], ref[c]) for c, _ in CORPORA}
    for row in ROWS:
        now = _nll(d / "ppl" / f"{row}.json")
        p = pub[row]
        if now is None or ref is None:
            md.append(f"| {LABEL[row]} | TBD | TBD | | |")
            continue
        r = dict(paired={c: K.paired(now[c], ref[c]) for c, _ in CORPORA}, **ov.get(row, {}))
        rec["rows"][row] = r
        md.append(f"| {LABEL[row]} | " + " | ".join(f"{cell(r['paired'][c])} ({p[c]})" for c, _ in CORPORA)
                  + f" | {r.get('e0m3_pct', float('nan')):.2f} ({p['e0m3_pct']:.2f}) | "
                  + (f"{r['jaccard']:.3f}" if r.get("jaccard") is not None else "—")
                  + (f" ({p['jaccard']:.3f})" if p.get("jaccard") is not None else "") + " |")
    if rec["fo6"]:
        md += ["", "FourOverSix against the 1k paper map: " + "; ".join(f"{name} {cell(rec['fo6'][c])}" for c, name in CORPORA)
               + f" (at 256: +{pub['fo6']['wiki']} / +{pub['fo6']['c4']}).", ""]
    md += ["", "## The 13-gram audits of the three fit sources", "",
           "| source | windows | 13-grams scanned | windows with hits | control (spliced items found) | passed |",
           "|---|---:|---:|---|---|---|"]
    for s, a in rec["audits"].items():
        c = a.get("control") or {}
        md.append(f"| {s} | {a['windows']} | {a['scanned_13grams']:,} | {a['windows_with_hits'] or 'none'} | "
                  f"{c.get('spliced_detected')} of {c.get('picked')} | {a['passed']}"
                  + (" (accepted)" if a.get("accepted") else "") + " |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ROBUST.md").write_text("\n".join(md) + "\n")
    (dest / "item12.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return f"{dest}/ROBUST.md"
