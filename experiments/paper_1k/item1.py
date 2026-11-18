"""Item 1 (the MMLU column of Table 3 and Tables E.1-E.3) at the 1k calibration: MMLU 5-shot of the 1k rows on the
paper's six models (part AC item 1; the colleague's spec: lm-eval 0.4.11, 5-shot, no chat template, acc x 100).

Rows per model: FlipQuant 8x64 / 16x64 / 256x64 (the 1k maps) and GPTQ‡-1k (its public Calib1k artifact),
each with part AA's evaluation flags (run_calib_size.quant_zs) and `evaluation.downstream --tasks mmlu` (5-shot, batch 16,
Qwen3.8-27B 8: the tool's defaults). The other rows (BF16, NVFP4, FourOverSix, IF4, MixFP4) do not depend on the
calibration windows and keep the paper's values. Table 3's Avg. of a 1k row is the mean of this MMLU and part AA's six
zero-shot tasks of the same artifact (reference/item1.json).

Step 0: Llama-3.2-1B-Instruct FlipQuant 16x64 with the 256-window release map, against the published 41.40 (gate
`item1-step0`, two-tier: exact when it rounds to 41.40; equivalent within 2 SE of it, the binomial SE over the 14,042
questions, because the published run's per-question records are not available).

Registered deviations (2026-10-26, after Nemotron's first row ran out of GPU memory at batch 16 on the porting
machine's 96 GB RTX PRO 6000):
- (A) environment: with PAPER1K_ALLOC_CONF=expandable_segments:True (jobs.Runner) the remaining jobs get that
  PYTORCH_CUDA_ALLOC_CONF; the command and batch are unchanged. The report marks those cells ¤.
- Consistency rows under a deviation: before a model's rows, FlipQuant 16x64 with the 256-window release map under the
  same deviation, against the published 256 cell (gate `item1-consistency`, the step-0 two-tier rule): Nemotron under
  (A), and every model whose batch is reduced.
- (B) command, the fallback: PAPER1K_ITEM1_BATCH="<model>=<N>,..." passes --batch-size N for that model only (16 -> 8;
  Qwen3.8-27B 8 -> 4), when it still runs out of memory under (A). The report marks those cells ‡b.

Model selection (2026-10-27, the user's order after the GPU reset): PAPER1K_ITEM1_MODELS="<model>,..." runs only those
models' rows, in that order (step 0 first as always); unset, the paper's model order. It changes no command.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import artifacts as A
from . import compare as K
from . import config as C

REF = json.loads((C.HERE / "reference" / "item1.json").read_text())
ROWS = ("fq-8x64", "fq-16x64", "fq-256x64", "gptq")
LABEL = {"fq-8x64": "FlipQuant 8x64", "fq-16x64": "FlipQuant 16x64", "fq-256x64": "FlipQuant 256x64",
         "gptq": "GPTQ‡"}
ZS = ("piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "lambada_openai")


def native():
    return ["--mode", "native", "--kernel-build-dir", C.build_dir(), "--kernel-set", "auto"]


def quant(model, row, get, data_root, out):
    """Part AA's evaluation flags of a row (run_calib_size.quant_zs; GPTQ‡: the codes file read as --gptq-cache)."""
    if row.startswith("fq-"):
        return [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", get(model, "flipquant_1k", row[3:])]
    return [*native(), "--weight", "nvfp4", "--act", "fourover6", "--any-act", "--ptq", "gptq", "--gptq-propagate", "bf16",
            "--gptq-hessian-input", "bf16", "--gptq-calib", "release", "--gptq-calib-record",
            data_root / C.TRAINER_KEY[model] / "calibration" / "report.json", "--gptq-calib-extension",
            out / "windows" / f"{model}_1024.json", "--gptq-block", 128, "--gptq-damp", 0.01,
            "--gptq-batch", 1 if model == "qwen3.8-27b" else 8, "--gptq-cache", get(model, "gptq_1k")]


#: the tool's default batch (evaluation.downstream BATCH)
DEFAULT_BATCH = {"qwen3.8-27b": 8}
#: deviation (A)'s consistency rows: the models whose remaining rows run under PAPER1K_ALLOC_CONF and get one
ALLOC_CONSISTENCY = ("nemotron-nano-9b-v2",)


def batch_overrides() -> dict:
    """Deviation (B): PAPER1K_ITEM1_BATCH="<model>=<N>,..." (the fallback batch per model; empty: the tool's default)."""
    out = {}
    for part in filter(None, (x.strip() for x in __import__("os").environ.get("PAPER1K_ITEM1_BATCH", "").split(","))):
        m, n = part.split("=")
        if m not in C.MODELS:
            raise SystemExit(f"PAPER1K_ITEM1_BATCH: unknown model {m!r}")
        out[m] = int(n)
    return out


def models_selected() -> list:
    """PAPER1K_ITEM1_MODELS="<model>,...": the models to run, in that order (unset: the paper's order)."""
    sel = [m.strip() for m in __import__("os").environ.get("PAPER1K_ITEM1_MODELS", "").split(",") if m.strip()]
    bad = [m for m in sel if m not in C.MODELS]
    if bad:
        raise SystemExit(f"PAPER1K_ITEM1_MODELS: unknown models {bad}")
    return sel or list(C.MODELS)


def mmlu_cmd(model, q, out, batch=None):
    return [C.py(model), "-m", "evaluation.downstream", "--model", model, "--revision", C.REV[model], *q, "--tasks", "mmlu",
            *(["--batch-size", batch] if batch else []), "--out", out]


def consistency(R, d, model, get, batch, dry_run):
    """A consistency row under a deviation: FlipQuant 16x64 with the 256-window release map, against the published 256
    cell (gate item1-consistency, the step-0 two-tier rule)."""
    pub = REF["published_256"][model]["fq-16x64"]["mmlu"]
    tag = f"_b{batch}" if batch else ""
    o = d / "consistency" / f"{model}_fq-16x64_256{tag}.json"
    q = [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", get(model, "flipquant_256", "16x64")]
    R.must(R.run(f"item1_consistency_{model}_fq-16x64_256{tag}", mmlu_cmd(model, q, o, batch), o), f"consistency {model}")
    if dry_run:
        return
    t = K.accuracy_tier(accuracy(_rec(o)), pub, REF["questions"])
    res = dict(model=model, row="fq-16x64", windows=256, batch=batch or DEFAULT_BATCH.get(model, 16),
               env=R.deviations(), **t, passed=t["tier"] in ("exact", "equivalent"))
    res["accepted"] = not res["passed"] and "item1-consistency" in R.accepted()
    o.with_suffix(".check.json").write_text(json.dumps(res, indent=1) + "\n")
    R.log(f"item1 consistency {model}{tag}: MMLU {t['accuracy_pct']:.2f} vs published {pub:.2f}: {t['tier']} "
          f"({t['delta_questions']:+.0f} questions)")
    R.gate("item1-consistency", res["passed"], f"{model} FlipQuant 16x64 (256){tag} MMLU {t['accuracy_pct']:.2f}, "
                                               f"published {pub:.2f}, beyond 2 SE ({t['two_se_pct']:.2f})")


def _rec(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if r.get("status") == "complete" else None


def accuracy(rec) -> float:
    return rec["tasks"]["mmlu"]["metrics"]["acc,none"]


def run(R, out: Path, dry_run=False, data_root: Path | None = None):
    if data_root is None:
        R.stop("--data-root: the calibration data root of `setup records` (the GPTQ‡ rows read its records)")
    d = out / "item1"
    get = A.local if dry_run else A.fetch
    if not dry_run:
        for m in C.MODELS:
            f = out / "windows" / f"{m}_1024.json"
            want = json.loads((C.HERE / "windows" / f"{m}_1024.json").read_text())["token_sha256"]
            if not f.exists() or json.loads(f.read_text())["token_sha256"] != want:
                R.stop(f"{f}: run `setup windows` first (its token hashes must be the paper's)")
    s0 = REF["step0"]
    o = d / "step0" / f"{s0['model']}_{s0['row']}_256.json"
    q = [*native(), "--weight", "mixfp4", "--act", "fourover6", "--map", get(s0["model"], "flipquant_256", "16x64")]
    R.must(R.run("item1_step0_mmlu_256", mmlu_cmd(s0["model"], q, o), o), "step 0")
    if not dry_run:
        t = K.accuracy_tier(accuracy(_rec(o)), s0["mmlu"], REF["questions"])
        res = dict(model=s0["model"], row=s0["row"], **t, passed=t["tier"] in ("exact", "equivalent"))
        res["accepted"] = not res["passed"] and "item1-step0" in R.accepted()
        (d / "step0" / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
        R.log(f"item1 step 0: MMLU {t['accuracy_pct']:.2f} vs published {s0['mmlu']:.2f}: {t['tier']} "
              f"({t['delta_questions']:+.0f} questions)")
        R.gate("item1-step0", res["passed"], f"Llama-3.2-1B-Instruct FlipQuant 16x64 (256) MMLU {t['accuracy_pct']:.2f}, "
                                             f"published {s0['mmlu']:.2f}, beyond 2 SE ({t['two_se_pct']:.2f})")
    bo = batch_overrides()
    for model in models_selected():
        pending = [row for row in ROWS if _rec(d / model / f"{row}.json") is None]
        if pending and ((R.deviations() and model in ALLOC_CONSISTENCY) or model in bo):
            consistency(R, d, model, get, bo.get(model), dry_run)
        for row in ROWS:
            o = d / model / f"{row}.json"
            R.must(R.run(f"item1_mmlu_{model}_{row}", mmlu_cmd(model, quant(model, row, get, data_root, out), o,
                                                               bo.get(model)), o), f"mmlu {model} {row}")
    R.log("item1 done")


# ---------------------------------------------------------------------------------------------- the report
def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    d = [int(a[i]) - int(b[i]) for i in ids]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    bb, cc = sum(x == 1 for x in d), sum(x == -1 for x in d)
    return dict(n=n, delta_pp=100 * m, two_se_pp=200 * se, a_only=bb, b_only=cc, mcnemar_p=mcnemar_p(bb, cc),
                significant=abs(m) > 2 * se)


def _env(out: Path, job: str) -> dict:
    """The opt-in environment deviations a job ran with (jobs.Runner's logs/<job>.env.json; none: {})."""
    try:
        return json.loads((out / "logs" / f"{job}.env.json").read_text())
    except (OSError, ValueError):
        return {}


def report(out: Path, dest: Path):
    d = out / "item1"
    rec = dict(rows={}, paired={}, step0=None)
    s0 = d / "step0" / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None
    md = ["# Item 1: the MMLU column (Table 3, Tables E.1-E.3) at the 1k calibration", "",
          "MMLU 5-shot (lm-eval 0.4.11, no chat template, acc x 100, 14,042 questions; `evaluation.downstream --tasks "
          "mmlu`, its default batch) of the 1k rows, native on this repository's SM120 build, with part AA's evaluation "
          "flags. ± 2 SE: binomial. In parentheses: the paper's 256-window value. Avg.: Table 3's mean of seven tasks, this "
          "MMLU and part AA's six zero-shot tasks of the same artifact. The other rows keep the paper's values. Run from "
          "flipquant alone (`python -m experiments.paper_1k.run item1`).", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", f"Llama-3.2-1B-Instruct FlipQuant 16x64 with the 256-window release map: MMLU "
               f"{s['accuracy_pct']:.2f} against the published {s['published_pct']:.2f} — **{s['tier']}** "
               f"({s['delta_questions']:+.0f} questions; 2 SE = {s['two_se_pct']:.2f} pp)"
               + (", accepted by the runner's user" if s.get("accepted") else "") + ".", ""]
    cons = []
    for f in sorted((d / "consistency").glob("*.check.json")) if (d / "consistency").exists() else []:
        cons.append(json.loads(f.read_text()))
    rec["consistency"] = cons
    md += ["## Deviations (registered 2026-10-26)", "",
           "Nemotron-Nano-9B-v2's first row ran out of GPU memory at the tool's batch 16 on this 96 GB board (lm-eval's "
           "float32 log-softmax over a padded batch of 5-shot prompts and a 131k vocabulary; 28.5 GiB of the process's "
           "memory reserved but unallocated). The published 256 MMLU column was run on another RTX PRO 6000 pod (also "
           "96 GB) with the colleague's spec; how it fitted there is not recorded here.", "",
           "- **¤ (environment):** the job ran with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (PAPER1K_ALLOC_CONF); "
           "the command and batch are unchanged.",
           "- **‡b (command):** the job ran at a reduced batch (`--batch-size`), the fallback for a model that still ran "
           "out of memory under ¤.", ""]
    if cons:
        md += ["Consistency rows (FlipQuant 16x64 with the 256-window release map under the deviation, against the "
               "published 256 cell; gate `item1-consistency`, the step-0 two-tier rule):", "",
               "| model | batch | environment | MMLU (%) | published | tier | Δ questions |", "|---|---:|---|---:|---:|---|---:|"]
        for c in cons:
            md.append(f"| {C.TITLE[c['model']]} | {c['batch']} | {'¤' if c['env'] else '—'} | {c['accuracy_pct']:.2f} | "
                      f"{c['published_pct']:.2f} | {c['tier']} | {c['delta_questions']:+.0f} |")
        md.append("")
    md += ["## MMLU and Avg. at 1k", "", "| model | row | MMLU (%) | Avg. (%) |", "|---|---|---:|---:|"]
    tex = ["% MMLU (5-shot, acc x 100) and the seven-task Avg. of the 1k rows (part AC item 1)"]
    probs = {}
    for model in C.MODELS:
        for row in ROWS:
            r = _rec(d / model / f"{row}.json")
            pub = REF["published_256"][model][row]
            if r is None:
                md.append(f"| {C.TITLE[model]} | {LABEL[row]} | TBD ({pub['mmlu']:.2f}) | TBD ({pub['avg']:.2f}) |")
                continue
            acc = accuracy(r)
            ex = r["tasks"]["mmlu"]["examples"]
            probs[(model, row)] = ex
            n = len(ex)
            zs = REF["zeroshot_1k"][model][row]
            avg = 100 * (sum(zs[t] for t in ZS) + acc) / 7
            rec["rows"].setdefault(model, {})[row] = dict(mmlu_pct=100 * acc, two_se_pct=200 * math.sqrt(acc * (1 - acc) / n),
                                                          questions=n, avg_pct=avg, published_256=pub,
                                                          sample_digest=r["tasks"]["mmlu"].get("sample_digest"))
            x = rec["rows"][model][row]
            x["batch"] = r.get("batch_size")
            x["env"] = _env(out, f"item1_mmlu_{model}_{row}")
            mark = ("¤" if x["env"] else "") + ("‡b" if x["batch"] and x["batch"] != DEFAULT_BATCH.get(model, 16) else "")
            x["marks"] = mark
            md.append(f"| {C.TITLE[model]} | {LABEL[row]} | {x['mmlu_pct']:.2f} ± {x['two_se_pct']:.2f}{mark} "
                      f"({pub['mmlu']:.2f}) | {avg:.2f}{mark} ({pub['avg']:.2f}) |")
            tex.append(f"{C.TITLE[model]} & {LABEL[row]} & {x['mmlu_pct']:.2f} & {avg:.2f} \\\\")
    md += ["", "## Paired per question, at 1k (A − B, pp ± 2 SE; A-only / B-only; McNemar's exact p)", "",
           "| model | A vs B | Δ |", "|---|---|---:|"]
    for model in C.MODELS:
        for a, b in (("fq-16x64", "gptq"), ("fq-8x64", "fq-16x64"), ("fq-256x64", "fq-16x64")):
            if (model, a) in probs and (model, b) in probs:
                p = paired(probs[(model, a)], probs[(model, b)])
                rec["paired"].setdefault(model, {})[f"{a} vs {b}"] = p
                md.append(f"| {C.TITLE[model]} | {LABEL[a]} vs {LABEL[b]} | {p['delta_pp']:+.2f} ± {p['two_se_pp']:.2f}"
                          f"{'*' if p['significant'] else ''} ({p['a_only']} / {p['b_only']}, p {p['mcnemar_p']:.3g}) |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "MMLU.md").write_text("\n".join(md) + "\n")
    (dest / "item1.json").write_text(json.dumps(rec, indent=1) + "\n")
    (dest / "table_mmlu.tex").write_text("\n".join(tex) + "\n")
    return f"{dest}/MMLU.md"
