"""Item 4 (Table F.3, `tab:ptq`) at the 1k calibration: GPTQ, GPTQ candidates and the Hadamard basis on
Nemotron-Nano-9B-v2 and Qwen3.8-27B at 16x64 (NVFP4-RaZeR paper-eval part O, "round 3", and part H's Hadamard rows,
with the 1,024 windows; part AC item 4).

Per model:
- GPTQ with activation quantization on during calibration (`--ptq gptq --gptq-propagate quantized --gptq-hessian-input
  quantized`, the fixed RTN grid, block 128, damp 0.01) on the 1k windows (the record's 128, then the extension's 896):
  NVFP4, FourOverSix, FlipQuant 16x64 with the 1k map fixed; then the E0M3 candidate (`calibration.gptq_codes` with an
  all-E0M3 map);
- two maps trained at 1k: on the GPTQ candidates (`tmopt_ext train --candidates-e2m1 / --candidates-e0m3`, the n16k64
  env as round 3) and in the block-16 Hadamard basis (`--rotate hadamard --rotate-block 16`, as part H);
- native PPL: the three GPTQ rows, the retrained map on the GPTQ candidates, the re-check (the 1k RTN map on the 1k GPTQ
  candidates, no retraining), the Hadamard map;
- GSM8K (part N: lm-eval gsm8k_llama, chat template, thinking off, batch 64): the three GPTQ rows, the GPTQ-candidate
  map, the Hadamard map.
The rows that do not depend on the calibration windows (BF16, RTN NVFP4 / FourOverSix, Hadamard NVFP4 / FourOverSix)
and RTN FlipQuant at 1k are re-measured for PPL (their records' exact commands) and compared with the paper's per-window
NLLs (reported); their GSM8K results are the paper's (reference/item4.json, per problem).

Step 0: round 3's Nemotron GPTQ NVFP4 job at 256 windows (its exact command; the 256 windows from `setup windows`):
codes_sha256 must equal round 3's (reference/item4.json); the data hash and the per-window NLLs are reported.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from . import artifacts as A
from . import compare as K
from . import config as C
from .jobs import done

REF = json.loads((C.HERE / "reference" / "item4.json").read_text())
MODELS = ["nemotron-nano-9b-v2", "qwen3.8-27b"]
UNIT = "16x64"
#: round 3's rows (run_ptq_round3.ROWS / PPL_CONV); "@1k" is the 1k 16x64 map
ROWS = {"nvfp4": ["--weight", "nvfp4", "--act", "nvfp4"],
        "fo6": ["--weight", "fourover6", "--act", "fourover6"],
        "fq-16x64": ["--weight", "mixfp4", "--act", "fourover6", "--map", "@1k"]}
PPL_CONV = {"nvfp4": ["--act-scope", "row"], "fo6": ["--paper-convention"], "fq-16x64": ["--paper-convention"]}
ROT = ["--rotate", "hadamard", "--rotate-block", "16"]
TASK = "gsm8k_llama"
#: thinking off through lm-eval's own switches (part N)
THINK_OFF = {"qwen3.8-27b": ["--enable-thinking", "false"], "nemotron-nano-9b-v2": ["--system-instruction", "/no_think"]}
#: one GPTQ code file, rounded up (round 3: 4.3-4.8 / 13.7-15.2 GB), and the margin a GPTQ job needs on top
FILE_GB = {"nemotron-nano-9b-v2": 5, "qwen3.8-27b": 16}
MARGIN_GB = 10
FILTERS = ("strict_match", "flexible_extract")


def all_e0m3(src, dst, sidecar):
    """Every tile of ``src``'s modules E0M3 (round 3's all_e0m3_map, from the 1k map): the E0M3 candidate's grid for
    flipquant's mixfp4 GPTQ. Its tiles digest (names and bytes, what GPTQ's cache key hashes) equals round 3's map's."""
    import torch
    from flipquant import maps as M
    tiles, unit, meta = M.load(src)
    M.save(dst, {n: torch.ones_like(t, dtype=torch.bool) for n, t in tiles.items()}, unit,
           dict(method="all-E0M3 (round 3: the E0M3 candidate's grid)", model=meta.get("model"),
                registry_key=meta.get("registry_key"), revision=meta.get("revision"), rotate="none", modules_from=str(src)))
    back, unit2, _ = M.load(dst)
    assert unit2 == unit and set(back) == set(tiles)
    assert all(bool(back[n].all()) and back[n].shape == tiles[n].shape for n in tiles)
    n = sum(t.numel() for t in back.values())
    Path(sidecar).write_text(json.dumps(dict(status="complete", modules=len(back), tiles=n)) + "\n")
    print("ALL-E0M3", len(back), "modules", n, "tiles", flush=True)


def native():
    return ["--mode", "native", "--kernel-build-dir", C.build_dir(), "--kernel-set", "auto"]


def codes_dir(out: Path) -> Path:
    """The GPTQ code files (about 77 GB in all); PAPER1K_CODES puts them on another disk."""
    return Path(os.environ.get("PAPER1K_CODES") or out / "item4" / "codes")


def gptq_flags(model, data_root, ext, cache=None, candidates=None):
    """Round 3's GPTQ flags (run_ptq_round3.gptq_flags): activation quantization on during calibration, the release
    record's 128 windows, then the extension's (``ext``: a `setup windows` file)."""
    f = ["--ptq", "gptq", "--gptq-propagate", "quantized", "--gptq-hessian-input", "quantized", "--gptq-calib", "release",
         "--gptq-calib-record", data_root / C.TRAINER_KEY[model] / "calibration" / "report.json",
         "--gptq-calib-extension", ext, "--gptq-block", 128, "--gptq-damp", 0.01,
         "--gptq-batch", 1 if model == "qwen3.8-27b" else 8]
    if cache is not None:
        f += ["--gptq-cache", cache]
    if candidates is not None:
        f += ["--gptq-candidates", *candidates]
    return f


def ppl_cmd(model, quant, out):
    """run_sm120.ppl_cmd: evaluation.ppl with the release revision."""
    return [C.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", C.REV[model], *quant, "--out", out]


def gsm8k_cmd(model, quant, batch, out):
    """Part N's command (run_ptq_round3.gsm8k_cmd)."""
    return [C.py(model), "-u", "-m", "evaluation.downstream", "--model", model, "--revision", C.REV[model], *quant,
            "--tasks", TASK, "--apply-chat-template", *THINK_OFF[model], "--batch-size", batch, "--gen-lengths",
            "--samples-out", Path(out).with_suffix(""), "--out", out]


def reused_ppl(model, map1k, out_dir):
    """The reused rows' PPL, each with its record's exact command (part F, part H, part AA)."""
    py, B = C.py(model), C.build_dir()
    ppl = [py, "-m", "evaluation.ppl", "--model", model]
    rows = {"bf16": [*ppl, "--mode", "bf16", "--paper-convention"],
            "rtn_nvfp4": [*ppl, "--mode", "native", "--kernel-build-dir", B, "--weight", "nvfp4", "--act", "nvfp4",
                          "--act-scope", "row"],
            "rtn_fo6": [*ppl, "--mode", "native", "--kernel-build-dir", B, "--weight", "fourover6", "--act", "fourover6",
                        "--paper-convention"],
            "rtn_fq-16x64": [*ppl, "--revision", C.REV[model], *native(), "--weight", "mixfp4", "--act", "fourover6",
                             "--map", map1k, "--paper-convention"],
            "hadamard_nvfp4": [*ppl, "--revision", C.REV[model], *native(), "--weight", "nvfp4", "--act", "nvfp4",
                               "--act-scope", "row", *ROT],
            "hadamard_fo6": [*ppl, "--revision", C.REV[model], *native(), "--weight", "fourover6", "--act", "fourover6",
                             "--paper-convention", *ROT]}
    return {k: [*v, "--out", out_dir / f"{k}.json"] for k, v in rows.items()}


def host(model):
    return ["--teacher-host"] if model == "qwen3.8-27b" else []


def window_hashes(model, n=1024):
    return json.loads((C.HERE / "windows" / f"{model}_{n}.json").read_text())["token_sha256"]


def ext_hashes(report) -> list:
    return [x["token_sha256"] for x in json.loads(Path(report).read_text())["fit_extension"]["records"]]


def disk_ok(R, root: Path, model):
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    if free < FILE_GB[model] + MARGIN_GB:
        R.stop(f"{free:.0f} GB free under {root}; a {model} GPTQ code file needs {FILE_GB[model]} GB + {MARGIN_GB} GB "
               "margin (PAPER1K_CODES puts the code files on another disk)")


def run(R, out: Path, dry_run=False, data_root: Path | None = None):
    if data_root is None:
        R.stop("--data-root: the calibration data root of `setup records`")
    d, cdir = out / "item4", codes_dir(out)
    get = A.local if dry_run else A.fetch
    win = {(m, n): out / "windows" / f"{m}_{n}.json" for m in MODELS for n in (256, 1024)}
    if not dry_run:
        d.mkdir(parents=True, exist_ok=True)
        for (m, n), f in win.items():
            if n == 1024 or m == REF["step0"]["model"]:
                if not f.exists() or json.loads(f.read_text())["token_sha256"] != window_hashes(m, n):
                    R.stop(f"{f}: run `setup windows` first (its token hashes must be the paper's)")

    def gptq_job(name, model, cmd, rec, what):
        if not dry_run and not done(rec):
            disk_ok(R, cdir, model)
        R.must(R.run(name, cmd, rec), what)

    # ---- step 0: round 3's Nemotron GPTQ NVFP4 at 256 windows
    s0 = REF["step0"]
    m0 = s0["model"]
    o = d / "step0" / f"{m0}_gptq_nvfp4_256.json"
    q = [*native(), *ROWS["nvfp4"], *PPL_CONV["nvfp4"],
         *gptq_flags(m0, data_root, win[(m0, 256)], cdir / "step0" / m0 / "codes_nvfp4_256.pt")]
    gptq_job("item4_step0_gptq_nvfp4_256", m0, ppl_cmd(m0, q, o), o, "step 0")
    if not dry_run:
        new = json.loads(o.read_text())
        res = dict(model=m0, row="gptq_nvfp4", windows=new["ptq"]["windows"], codes_sha256=new["ptq"]["codes_sha256"],
                   round3_codes_sha256=s0["codes_sha256"], codes_equal=new["ptq"]["codes_sha256"] == s0["codes_sha256"],
                   data_equal=new["ptq"]["data_sha256"] == s0["data_sha256"],
                   nll=K.tier({c: new["results"][c]["nll"] for c in ("wiki", "c4")}, s0["nll"]))
        res["accepted"] = not res["codes_equal"] and "item4-step0" in R.accepted()
        (d / "step0" / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
        R.log(f"item4 step 0: codes equal {res['codes_equal']}, data equal {res['data_equal']}, NLL {res['nll']['tier']}")
        R.gate("item4-step0", res["codes_equal"], f"the GPTQ NVFP4 codes differ from round 3's ({res['codes_sha256'][:12]} "
                                                  f"vs {s0['codes_sha256'][:12]}; data equal {res['data_equal']})")
    # ---- the panel
    for model in MODELS:
        dm, cm, ext = d / model, cdir / model, win[(model, 1024)]
        map1k = get(model, "flipquant_1k", UNIT)
        # the reused rows' PPL, re-measured
        for name, cmd in reused_ppl(model, map1k, dm / "reused").items():
            o = dm / "reused" / f"{name}.json"
            R.must(R.run(f"item4_ppl_{model}_{name}", cmd, o), f"ppl {model} {name}")
        # GPTQ (act.) at 1k: NVFP4, FourOverSix, FlipQuant 16x64 with the 1k map fixed; then the E0M3 candidate
        for row in ROWS:
            o = dm / f"gptq_{row}.json"
            fmt = [str(map1k) if a == "@1k" else a for a in ROWS[row]]
            q = [*native(), *fmt, *PPL_CONV[row], *gptq_flags(model, data_root, ext, cm / f"codes_{row}.pt")]
            gptq_job(f"item4_gptq_{model}_{row}", model, ppl_cmd(model, q, o), o, f"gptq {model} {row}")
        amap = dm / f"all_e0m3_{UNIT}.pt"
        R.must(R.run(f"item4_all_e0m3_{model}", [C.py(model), "-m", "experiments.paper_1k.item4", "all-e0m3", map1k, amap,
                                                amap.with_suffix(".json")],
                     amap.with_suffix(".json"), gpu=False, extra_env={"CUDA_VISIBLE_DEVICES": ""}), f"all-E0M3 map {model}")
        o = dm / "gptq_e0m3.json"
        cmd = [C.py(model), "-m", "calibration.gptq_codes", "--model", model, "--revision", C.REV[model], "--weight",
               "mixfp4", "--act", "fourover6", "--act-scope", "row", "--map", amap,
               *gptq_flags(model, data_root, ext, cm / "codes_e0m3.pt"), "--out", o]
        gptq_job(f"item4_e0m3_{model}", model, cmd, o, f"e0m3 codes {model}")
        # the maps at 1k: on the GPTQ candidates (round 3, the n16k64 env) and in the Hadamard basis (part H)
        cand = [cm / "codes_fo6.pt", cm / "codes_e0m3.pt"]
        gmap = dm / f"fq-{UNIT}_gptqcand.pt"
        R.must(R.run(f"item4_train_{model}_gptqcand", [C.py_release(), "-m", "calibration.tmopt_ext", "train", "--model",
                                                      model, "--unit", UNIT, "--data-root", data_root, "--fit-windows",
                                                      1024, *host(model), "--candidates-e2m1", cand[0],
                                                      "--candidates-e0m3", cand[1], "--out", gmap],
                     gmap.with_suffix(".json")), f"train {model} gptqcand")
        rmap = dm / f"fq-{UNIT}_hadamard.pt"
        R.must(R.run(f"item4_train_{model}_hadamard", [C.py(model), "-m", "calibration.tmopt_ext", "train", "--model",
                                                      model, "--unit", UNIT, *ROT, "--data-root", data_root,
                                                      "--fit-windows", 1024, *host(model), "--out", rmap],
                     rmap.with_suffix(".json")), f"train {model} hadamard")
        if not dry_run:
            for mp in (gmap, rmap):
                if ext_hashes(mp.with_suffix(".run") / "run" / "report.json") != window_hashes(model)[128:]:
                    R.stop(f"{model}: {mp.name}'s fit extension is not the paper's 1k set")
        # PPL: the retrained map on the GPTQ candidates, the re-check (the 1k RTN map on them), the Hadamard map
        rows = {"gptqcand_fq-16x64": [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", gmap,
                                      "--paper-convention", *gptq_flags(model, data_root, ext, candidates=cand)],
                "recheck_rtnmap_gptqcand": [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", map1k,
                                            "--paper-convention", *gptq_flags(model, data_root, ext, candidates=cand)],
                "hadamard_fq-16x64": [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", rmap,
                                      "--paper-convention", *ROT]}
        for name, q in rows.items():
            o = dm / f"{name}.json"
            R.must(R.run(f"item4_ppl_{model}_{name}", ppl_cmd(model, q, o), o), f"ppl {model} {name}")
        # GSM8K, part N's protocol at batch 64 (round 3's O4 command)
        gsm = {f"gptq_{r}": [*native(), *[str(map1k) if a == "@1k" else a for a in ROWS[r]], "--act-scope", "row",
                             *gptq_flags(model, data_root, ext, cm / f"codes_{r}.pt")] for r in ROWS}
        gsm["gptqcand_fq-16x64"] = [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", gmap, "--act-scope",
                                    "row", *gptq_flags(model, data_root, ext, candidates=cand)]
        gsm["hadamard_fq-16x64"] = [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", rmap, "--act-scope",
                                    "row", *ROT]
        for name, q in gsm.items():
            o = d / "gsm8k" / model / f"{name}.json"
            R.must(R.run(f"item4_gsm8k_{model}_{name}", gsm8k_cmd(model, q, 64, o), o), f"gsm8k {model} {name}")
        R.log(f"item4 {model} done")
    fastenv_repeat(R, d, data_root, dry_run)
    R.log("item4 done")


#: the auxiliary repeats of part H's 256 Hadamard training per model (Nemotron twice: repeat 2 separates run-to-run
#: nondeterminism from the code history, the coordinator's request of 2026-10-26)
REPEATS = {"nemotron-nano-9b-v2": 2, "qwen3.8-27b": 1}


def first_diff(a: list, b: list):
    """The first epoch whose train_kl differs (None: all equal)."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def fastenv_conclusion(v: dict) -> str:
    reps = v["repeats"]
    if len(reps) >= 2:
        if all(r["tiles"]["digest"] == reps[0]["tiles"]["digest"] for r in reps[1:]):
            return ("deterministic on this board and code: the repeats are bit for bit equal"
                    + ("" if reps[0]["equal_part_h"] else "; the difference to part H comes from the code or session history"))
        return "not bit-reproducible run to run: the repeats of one command differ on this board and code"
    return ("reproduces part H's map bit for bit" if reps[0]["equal_part_h"] else
            "differs from part H's map (one repeat: run-to-run nondeterminism and the code history are not separated)")


def fastenv_repeat(R, d: Path, data_root, dry_run=False):
    """Auxiliary, not a gate: part H's exact 256 Hadamard training command (its job log's, n16k64-fast) once more per
    model (Nemotron twice), into item4/aux/; each map's tiles against part H's (reference/item4.json
    'hadamard_256_part_h'), the first epoch whose train_kl differs, and the conclusion: whether the fast recurrent
    kernels' training reproduces bit for bit on this board. Written to item4/fastenv_repeat.json."""
    from .item3 import tiles_digest
    res = {}
    for m in MODELS:
        want = REF["hadamard_256_part_h"][m]
        reps = []
        for k in range(1, REPEATS[m] + 1):
            sfx = "" if k == 1 else f"_r{k}"
            o = d / "aux" / f"h256_{m}{sfx}" / f"fq-{UNIT}_hadamard.pt"
            R.must(R.run(f"item4_aux_h256_{m}{sfx}", [C.py(m), "-m", "calibration.tmopt_ext", "train", "--model", m,
                                                      "--unit", UNIT, *ROT, "--data-root", data_root, "--out", o],
                         o.with_suffix(".json")), f"auxiliary repeat {k} {m}")
            if dry_run:
                continue
            got = tiles_digest(o)
            rep = _load_json(o.with_suffix(".run") / "run" / "report.json") or {}
            kl = [e["train_kl"] for e in rep.get("epochs", [])]
            reps.append(dict(repeat=k, tiles=got, equal_part_h=got["digest"] == want["digest"], epoch_train_kl=kl,
                             first_epoch_differs_from_part_h=first_diff(kl, want["epoch_train_kl"]),
                             env=job_env(R.out / "logs" / f"item4_aux_h256_{m}{sfx}.log"),
                             training_seconds=rep.get("training_seconds")))
            R.log(f"AUX item4 fast-env repeat {k} {m}: tiles equal to part H's {reps[-1]['equal_part_h']} "
                  f"({got['e0m3_tiles']} vs {want['e0m3_tiles']} E0M3 tiles); not a gate")
        if dry_run:
            continue
        v = dict(part_h=want, repeats=reps,
                 first_epoch_repeats_differ=first_diff(reps[0]["epoch_train_kl"], reps[1]["epoch_train_kl"])
                 if len(reps) > 1 else None)
        v["conclusion"] = fastenv_conclusion(v)
        res[m] = v
    if not dry_run:
        (d / "fastenv_repeat.json").write_text(json.dumps(dict(status="complete", models=res), indent=1) + "\n")

# ---------------------------------------------------------------------------------------------- the report
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))
#: (key, method, format, measured here, the 256 counterpart in reference/item4.json's published_256)
TABLE = [("bf16", "BF16", "--", "reused", None), ("rtn_nvfp4", "RTN", "NVFP4", "reused", None),
         ("rtn_fo6", "RTN", "FourOverSix", "reused", None),
         ("rtn_fq-16x64", "RTN", "FlipQuant (16x64)", "reused", "rtn_fq-16x64"),
         ("gptq_nvfp4", "GPTQ (act.)", "NVFP4", "new", "gptq_nvfp4"),
         ("gptq_fo6", "GPTQ (act.)", "FourOverSix", "new", "gptq_fo6"),
         ("gptq_fq-16x64", "GPTQ (act.)", "FlipQuant (16x64)", "new", "gptq_fq-16x64"),
         ("gptqcand_fq-16x64", "GPTQ cand. + map", "FlipQuant (16x64)", "new", "gptqcand_fq-16x64"),
         ("hadamard_nvfp4", "Hadamard", "NVFP4", "reused", None), ("hadamard_fo6", "Hadamard", "FourOverSix", "reused", None),
         ("hadamard_fq-16x64", "Hadamard", "FlipQuant (16x64)", "new", "hadamard_fq-16x64")]
PAIRS = [("gptq_nvfp4", "gptq_fo6"), ("gptq_fq-16x64", "gptq_fo6"), ("gptqcand_fq-16x64", "gptq_fo6"),
         ("gptqcand_fq-16x64", "gptq_fq-16x64"), ("gptq_fq-16x64", "rtn_fq-16x64"), ("gptqcand_fq-16x64", "rtn_fq-16x64"),
         ("hadamard_fq-16x64", "hadamard_fo6"), ("hadamard_fq-16x64", "rtn_fq-16x64"), ("rtn_fq-16x64", "rtn_fo6")]
THINK = ("<think>", "</think>")


def _load(path, key):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if (key in r if key == "resources" else r.get("status") == "complete") else None


def ppl_rec(path):
    return _load(path, "resources")


def gsm_rec(path):
    return _load(path, "status")


def ppl_of(nll):
    return math.exp(sum(nll) / len(nll))


def paired(x, ref):
    assert len(x) == len(ref)
    d = [a - b for a, b in zip(x, ref)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta_milli=1e3 * m, two_se_milli=2e3 * se, n=n, significant=abs(m) > 2 * se)


def cell(p):
    return f"{p['delta_milli']:+.2f} ± {p['two_se_milli']:.2f}{'*' if p['significant'] else ''}"


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_gsm(a: dict, b: dict) -> dict:
    """A vs B per problem (report_gsm8k.paired): Δ, 2 SE, A-only / B-only, McNemar's exact p."""
    ids = sorted(set(a) & set(b), key=int)
    d = [int(a[i]) - int(b[i]) for i in ids]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    bb, cc = sum(x == 1 for x in d), sum(x == -1 for x in d)
    return dict(n=n, delta=m, two_se=2 * se, a_only=bb, b_only=cc, mcnemar_p=mcnemar_p(bb, cc), significant=abs(m) > 2 * se)


def fmt_pp(p):
    return (f"{100 * p['delta']:+.2f} ± {100 * p['two_se']:.2f}{'*' if p['significant'] else ''} "
            f"({p['a_only']} / {p['b_only']}, p {p['mcnemar_p']:.3g})")


def gsm_problems(rep) -> dict:
    """{filter: {doc: correct}} of a GSM8K record."""
    ex = rep["tasks"][TASK]["examples"]
    return {f: {doc: bool(e[f]["exact_match"]) for doc, e in ex.items() if f in e} for f in FILTERS}


def gsm_reused(model, key) -> dict:
    r = REF["reused_gsm8k"][model][key]
    return {f: dict(zip(REF["doc_ids"], (ch == "1" for ch in r[f]))) for f in FILTERS}


def acc(problems: dict) -> dict:
    out = {}
    for f in FILTERS:
        c = problems[f]
        p = sum(c.values()) / len(c)
        out[f] = dict(accuracy=p, two_se=2 * math.sqrt(p * (1 - p) / len(c)), correct=sum(c.values()), n=len(c))
    return out


#: the published routes (reference/commands_item4.json: round 3's GPTQ-candidate trainings ran in n16k64, part H's
#: Hadamard trainings in n16k64-fast, both from their job logs' first lines)
PUBLISHED_ENV = {"gptqcand": "n16k64", "hadamard": "n16k64-fast"}
ENV_TEXT = {"n16k64": "n16k64 (the recurrent layers' fallback kernels)",
            "n16k64-fast": "n16k64-fast (mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, fla 0.5.2)"}


def job_env(log: Path) -> str | None:
    """n16k64-fast or n16k64, from a job log's first line (the executed command; its interpreter)."""
    try:
        py = log.read_text().split("\n", 1)[0].split(" ", 1)[0]
    except OSError:
        return None
    return "n16k64-fast" if "fast" in py else "n16k64"


def environments(out: Path, rec: dict) -> list:
    """Which environment each 1k training ran in, against the published route, with the trainer's time; and what is
    known of the fast kernels' run-to-run reproducibility (an optional repeat record, item4/fastenv_repeat.json)."""
    d = out / "item4"
    md = ["", "## The trainings' environments", "",
          "| model | training | environment (this run) | published route (256) | training seconds |", "|---|---|---|---|---:|"]
    for m in MODELS:
        for kind in ("gptqcand", "hadamard"):
            env = job_env(out / "logs" / f"item4_train_{m}_{kind}.log")
            rep = _load_json(d / m / f"fq-{UNIT}_{kind}.run" / "run" / "report.json")
            secs = (rep or {}).get("training_seconds")
            rec.setdefault("environments", {}).setdefault(m, {})[kind] = dict(env=env, published=PUBLISHED_ENV[kind],
                                                                              training_seconds=secs)
            md.append(f"| {C.TITLE[m]} | {kind} | {ENV_TEXT.get(env, 'not run')} | {ENV_TEXT[PUBLISHED_ENV[kind]]} | "
                      + (f"{secs:,.0f}" if secs else "—") + " |")
    rep = (_load_json(d / "fastenv_repeat.json") or {}).get("models")
    rec["fastenv_repeat"] = rep
    md += ["", "The GPTQ-candidate trainings keep round 3's n16k64 env (the hybrid models' recurrent layers on their "
           "fallback, as Appendix C states for the calibrations); the Hadamard trainings keep part H's n16k64-fast env "
           "(the fast recurrent kernels), so F.3's Hadamard maps are calibrated on the fast kernels (a paper-text point "
           "for F.3). The trainer's deterministic setting covers torch's own operations, not the custom kernels of "
           "mamba-ssm, causal-conv1d or fla."]
    if rep:
        md += ["", "**Auxiliary check (not a gate): part H's exact 256 Hadamard training once more on this board** "
               "(Nemotron twice), each map's tiles against part H's map (`item4/fastenv_repeat.json`):", "",
               "| model | run | environment | E0M3 tiles | tiles digest | equal to part H | first epoch whose train_kl "
               "differs from part H's | training seconds |", "|---|---|---|---:|---|---|---:|---:|"]
        for m, v in rep.items():
            h = v["part_h"]
            md.append(f"| {C.TITLE[m]} | part H (2026-10-08) | n16k64-fast | {h['e0m3_tiles']:,} | `{h['digest'][:16]}` | — | — | "
                      f"{h['training_seconds']:,.0f} |")
            for r in v["repeats"]:
                fe = r["first_epoch_differs_from_part_h"]
                md.append(f"| {C.TITLE[m]} | repeat {r['repeat']} | {r.get('env')} | {r['tiles']['e0m3_tiles']:,} | "
                          f"`{r['tiles']['digest'][:16]}` | {'yes' if r['equal_part_h'] else '**no**'} | "
                          f"{'none' if fe is None else fe} | {(r.get('training_seconds') or 0):,.0f} |")
        md.append("")
        for m, v in rep.items():
            fr = v.get("first_epoch_repeats_differ")
            md.append(f"- **{C.TITLE[m]}:** {v['conclusion']}"
                      + (f" (the repeats' train_kl first differs at epoch {fr})" if fr is not None else
                         " (the repeats' train_kl equal in every epoch)" if len(v["repeats"]) > 1 else "") + ".")
    else:
        md += ["", "Whether the fast kernels' trainings reproduce run to run: not measured yet (the auxiliary repeat "
               "has not run)."]
    return md


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def report(out: Path, dest: Path):
    d = out / "item4"
    rec = dict(rows={}, recheck={}, gsm8k_paired={}, checks={}, reused_agreement={}, step0=None)
    s0 = d / "step0" / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None
    md = ["# Item 4: Table F.3 (`tab:ptq`) at the 1k calibration (Nemotron-Nano-9B-v2, Qwen3.8-27B; 16x64)", "",
          "Round 3's protocol (NVFP4-RaZeR paper-eval PROTOCOL.md part O) and part H's Hadamard rows with the 1k "
          "windows, run from flipquant alone (`python -m experiments.paper_1k.run item4`). Native: this repository's "
          "SM120 build, --kernel-set auto. GPTQ with activation quantization on during calibration on the 1,024 "
          "windows (the record's 128 + the extension's 896); the GPTQ-candidate map and the Hadamard map trained at "
          "1k (Qwen3.8-27B with the teacher on the host). ΔNLL: paired per window vs RTN FourOverSix (re-measured in "
          "this session), x 1e-3 ± 2 SE (* beyond 2 SE). GSM8K: part N's protocol (lm-eval 0.4.11 gsm8k_llama, chat "
          "template, greedy, thinking off, 1,319 problems) at batch 64; strict-match ± 2 SE; paired per problem, A − B "
          "in pp ± 2 SE (A-only / B-only, McNemar's exact p).",
          "",
          "Reused rows (they do not depend on the calibration windows, or are part AA's 1k RTN map): their PPL is "
          "re-measured here with their records' commands (agreement with the paper's per-window NLLs below); their "
          "GSM8K results are the paper's (BF16: round 3's O0, Qwen3.8-27B at batch 48; RTN / Hadamard NVFP4 and "
          "FourOverSix: part N; RTN FlipQuant 1k: part AA), so GSM8K pairs with them compare this board with the "
          "paper's.", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", f"Round 3's Nemotron GPTQ NVFP4 job (256 windows) on this code and board: codes_sha256 "
               f"{'equal to' if s['codes_equal'] else 'DIFFERENT from'} round 3's (`{s['codes_sha256'][:16]}`)"
               + (" — **accepted by the runner's user**" if s.get("accepted") else "")
               + f"; data hash equal: {s['data_equal']}; per-window NLL against round 3's: {s['nll']['tier']} (max "
               f"|ΔNLL| {s['nll']['max_abs']:.3g}).", ""]
    md += ["## The table", "", "| model | method | format | WikiText-2 | C4 | ΔNLL wiki vs RTN FO6 | ΔNLL C4 vs RTN FO6 | "
           "GSM8K strict (%) | 256: WikiText-2 / C4 / GSM8K |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    tex = ["% tab:ptq at the 1k calibration (part AC item 4): WikiText-2 / C4 PPL (native sm120, 16x64) and GSM8K "
           "strict-match (%), lm-eval gsm8k_llama, thinking off; GPTQ with activation quantization on during calibration; "
           "1,024 calibration windows"]
    probs = {}
    for m in MODELS:
        dm = d / m
        ref = ppl_rec(dm / "reused" / "rtn_fo6.json")
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{C.TITLE[m]}}}}} \\\\")
        for key, method, fmt, kind, old in TABLE:
            r = ppl_rec(dm / "reused" / f"{key}.json") if kind == "reused" else ppl_rec(dm / f"{key}.json")
            if kind == "reused":
                g = acc(gsm_reused(m, key))
                probs[(m, key)] = gsm_reused(m, key)
                gcodes = None
            else:
                rep = gsm_rec(d / "gsm8k" / m / f"{key}.json")
                g = acc(gsm_problems(rep)) if rep else None
                if rep:
                    probs[(m, key)] = gsm_problems(rep)
                    ex = rep["tasks"][TASK]["examples"]
                    g["think_markers"] = sum(1 for e in ex.values() if any(t in (e.get("raw") or "") for t in THINK))
                gcodes = (((rep or {}).get("policy") or {}).get("ptq") or {}).get("codes_sha256")
            row = dict(method=method, format=fmt, measured="this session" if kind == "new" else "PPL re-measured, GSM8K the paper's")
            if r is not None:
                row["ppl"] = {c: ppl_of(r["results"][c]["nll"]) for c, _ in CORPORA}
                row["vs_rtn_fo6"] = ({c: paired(r["results"][c]["nll"], ref["results"][c]["nll"]) for c, _ in CORPORA}
                                     if ref and key != "rtn_fo6" else None)
                pt = r.get("ptq") or {}
                row["ptq"] = {k: pt.get(k) for k in ("method", "weight", "act", "windows", "hessian_input", "propagate",
                                                     "calib", "data_sha256", "codes_sha256", "map_sha256", "rotate")}
            if g:
                row["gsm8k"] = g
                row["gsm8k_codes_sha256"] = gcodes
            if old:
                row["published_256"] = REF["published_256"][m][old]
            rec["rows"].setdefault(m, {})[key] = row
            pp, vs, o = row.get("ppl"), row.get("vs_rtn_fo6") or {}, row.get("published_256")
            md.append(f"| {C.TITLE[m]} | {method} | {fmt} | " + (f"{pp['wiki']:.4f} | {pp['c4']:.4f}" if pp else "TBD | TBD")
                      + " | " + (" | ".join(cell(vs[c]) for c, _ in CORPORA) if vs else "— | —")
                      + " | " + (f"{100 * g['strict_match']['accuracy']:.2f} ± {100 * g['strict_match']['two_se']:.2f}"
                                 if g else "TBD")
                      + " | " + (f"{o['ppl']['wiki']:.4f} / {o['ppl']['c4']:.4f} / {100 * o['gsm8k_strict']:.2f}"
                                 if o else "unchanged") + " |")
            fmt_tex = fmt.replace("16x64", "16$\\times$64")
            tex.append(f"{method} & {fmt_tex} & "
                       + (f"{pp['wiki']:.2f} & {pp['c4']:.2f}" if pp else "TBD & TBD") + " & "
                       + (f"{100 * g['strict_match']['accuracy']:.1f}" if g else "TBD") + " \\\\")
        # the re-check against the retrained map
        rc, rt = ppl_rec(dm / "recheck_rtnmap_gptqcand.json"), ppl_rec(dm / "gptqcand_fq-16x64.json")
        if rc is not None:
            rec["recheck"][m] = dict(ppl={c: ppl_of(rc["results"][c]["nll"]) for c, _ in CORPORA},
                                     vs_rtn_fo6={c: paired(rc["results"][c]["nll"], ref["results"][c]["nll"])
                                                 for c, _ in CORPORA} if ref else None,
                                     retrained_minus_recheck={c: paired(rt["results"][c]["nll"], rc["results"][c]["nll"])
                                                              for c, _ in CORPORA} if rt else None)
        # the reused rows' re-measured PPL against the paper's per-window NLLs
        for key, *_ in [t for t in TABLE if t[3] == "reused"]:
            r = ppl_rec(dm / "reused" / f"{key}.json")
            if r is None:
                continue
            rec["reused_agreement"].setdefault(m, {})[key] = K.tier(
                {c: r["results"][c]["nll"] for c, _ in CORPORA}, REF["reused_ppl"][m][key])
    md += ["", "## The re-check: the 1k RTN map on the 1k GPTQ candidates (no retraining)", "",
           "| model | WikiText-2 | C4 | ΔNLL vs RTN FO6 (wiki; C4) | retrained map − re-check (wiki; C4) |",
           "|---|---:|---:|---:|---:|"]
    for m in MODELS:
        x = rec["recheck"].get(m)
        if not x:
            md.append(f"| {C.TITLE[m]} | TBD | TBD | | |")
            continue
        md.append(f"| {C.TITLE[m]} | {x['ppl']['wiki']:.4f} | {x['ppl']['c4']:.4f} | "
                  + ("; ".join(cell(x['vs_rtn_fo6'][c]) for c, _ in CORPORA) if x["vs_rtn_fo6"] else "—") + " | "
                  + ("; ".join(cell(x['retrained_minus_recheck'][c]) for c, _ in CORPORA)
                     if x["retrained_minus_recheck"] else "—") + " |")
    md += ["", "## GSM8K, paired per problem", "", "| model | A vs B | strict-match Δ (pp) | flexible-extract Δ (pp) |",
           "|---|---|---:|---:|"]
    for m in MODELS:
        for a, b in PAIRS:
            if (m, a) in probs and (m, b) in probs:
                pg = {f: paired_gsm(probs[(m, a)][f], probs[(m, b)][f]) for f in FILTERS}
                rec["gsm8k_paired"].setdefault(m, {})[f"{a} vs {b}"] = pg
                md.append(f"| {C.TITLE[m]} | {a} vs {b} | {fmt_pp(pg['strict_match'])} | {fmt_pp(pg['flexible_extract'])} |")
    for m in MODELS:
        rows = rec["rows"].get(m, {})
        g = [rows[k]["ptq"] for k in ("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64") if rows.get(k, {}).get("ptq")]
        if g:
            rec["checks"][f"{m}: GPTQ quantized Hessian and propagation, 1,024 windows, one data hash"] = (
                all(x["hessian_input"] == "quantized" and x["windows"] == 1024 and x["calib"] == "release" for x in g)
                and len({x["data_sha256"] for x in g}) == 1)
        for k in ("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64"):
            row = rows.get(k, {})
            if row.get("gsm8k") and row.get("ptq"):
                rec["checks"][f"{m}: {k} GSM8K codes = PPL codes"] = row["gsm8k_codes_sha256"] == row["ptq"]["codes_sha256"]
        for k, v in rec["reused_agreement"].get(m, {}).items():
            rec["checks"][f"{m}: {k} PPL against the paper's record (reported): exact / equivalent / differs"] = v["tier"]
    md += environments(out, rec)
    md += ["", "## Checks", "", "| check | passed |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rec["checks"].items()]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "item4.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (dest / "REPORT.md").write_text("\n".join(md) + "\n")
    (dest / "table_ptq.tex").write_text("\n".join(tex) + "\n")
    return f"{dest}/REPORT.md"


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["all-e0m3"] and len(sys.argv) == 5:
        all_e0m3(*sys.argv[2:])
    else:
        sys.exit("usage: python -m experiments.paper_1k.item4 all-e0m3 SRC_MAP DST_MAP SIDECAR_JSON")
