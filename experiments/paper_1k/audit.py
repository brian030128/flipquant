"""Item 12's 13-gram audit of a robustness run's calibration windows against every evaluation text, with its positive
control (NVFP4-RaZeR paper-eval part AA's calib_size_audit.py and calib_size_audit_control.py, ported). CPU.

    python -m experiments.paper_1k.audit --model mistral-7b --fit-source fineweb-edu --fit-windows 1024 \\
        --data-root paper1k_data --out audit.json [--trainer-report RUN/run/report.json]

- **Windows:** drawn exactly as `calibration.train_map --fit-source S --fit-windows N` draws them: the trainer's own
  `fit_sources.windows` with the extension rule's skip set, run in the trainer's directory (as calibration.fit_windows
  runs the trainer's data lines). With --trainer-report, their token sha256 list must equal the trained run's record.
- **Evaluation texts** (supersets of what is scored): WikiText-2 raw test (the whole split); the C4 validation documents
  of the model's PPL windows (whole); the six zero-shot tasks' evaluated splits at their pinned revisions, every string
  field of each row; GSM8K main test (question and answer) and gsm8k_llama's eight few-shot exemplars.
- **13-grams:** lowercase, every character outside [a-z0-9] separates words; word 13-grams within each text. A hit is a
  window that shares at least one 13-gram with an evaluation text; every hit is listed.
- **Positive control:** a zero-hit audit is evidence only if the audit finds evaluation text that is there. Thirty
  seeded windows are rescanned as they are (zero hits), then with one evaluation item spliced into the middle of each:
  a GSM8K test problem, a HellaSwag context with its endings, a WikiText-2 test paragraph, a LAMBADA passage and an
  ARC-Challenge question. Every splice must be found and attributed to its own evaluation set.
- **Pass** (``status`` complete, ``passed``): no hit, and the control detects every splice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "calibration" / "tmopt"
N = 13
WORD = re.compile(r"[a-z0-9]+")
#: task: (repo, config, revision, split, rows) -- the zero-shot tasks' evaluated splits (part X's dataset loads)
ZS = {"piqa": ("baber/piqa", None, "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", "validation", 1838),
      "hellaswag": ("Rowan/hellaswag", None, "218ec52e09a7e7462a5400043bb9a69a41d06b76", "validation", 10042),
      "winogrande": ("allenai/winogrande", "winogrande_xl", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "validation", 1267),
      "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453", "test", 2376),
      "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "test", 1172),
      "lambada_openai": ("EleutherAI/lambada_openai", "default", "900124bf3b8235c6daf21033af9948b3f07346c4", "test", 5153)}
GSM8K = ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866", "test", 1319)


def words(text):
    return WORD.findall(text.lower())


def grams(ws):
    return (" ".join(ws[i:i + N]) for i in range(len(ws) - N + 1))


def strings(obj):
    """Every string leaf of a dataset row, in field order."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in strings(v)]
    return []


class Index:
    def __init__(self):
        self.first, self.stats = {}, {}

    def add(self, name, item, text):
        ws = words(text)
        st = self.stats.setdefault(name, dict(items=0, words=0, grams=0))
        st["items"] += 1
        st["words"] += len(ws)
        for g in grams(ws):
            st["grams"] += 1
            self.first.setdefault(g, (name, item))


def eval_index(tok):
    """The evaluation texts' 13-grams (part AA's eval_index, for one tokenizer's C4 windows)."""
    from datasets import load_dataset
    import lm_eval
    import yaml
    from flipquant import data as D
    idx = Index()
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=D.WIKI_REVISION, split="test")
    idx.add("wikitext2_test", "test", "\n\n".join(wiki["text"]))
    c4 = load_dataset("allenai/c4", revision=D.C4_REVISION, data_files={"validation": D.C4_FILE}, split="validation")
    docs = sorted({d["index"] for d in D.eval_windows(tok, "c4", 2048)[1]["documents"]})
    for i in docs:
        idx.add("c4_validation_eval", i, c4[i]["text"])
    for task, (repo, config, rev, split, n) in ZS.items():
        ds = load_dataset(repo, config, revision=rev, split=split)
        assert len(ds) == n, (task, len(ds))
        for i, row in enumerate(ds):
            idx.add(task, i, " ".join(strings(row)))
    repo, config, rev, split, n = GSM8K
    ds = load_dataset(repo, config, revision=rev, split=split)
    assert len(ds) == n
    for i, row in enumerate(ds):
        idx.add("gsm8k_test", i, row["question"] + " " + row["answer"])
    cfg = yaml.safe_load((Path(lm_eval.__file__).parent / "tasks" / "llama3" / "instruct" / "gsm8k" / "gsm8k.yaml")
                         .read_text())
    shots = cfg["fewshot_config"]["samples"]
    assert len(shots) == 8
    for i, s in enumerate(shots):
        idx.add("gsm8k_llama_fewshot", i, s["question"] + " " + s["target"])
    return idx, dict(c4_documents=len(docs))


def splices(seed=7):
    """The control's five evaluation items (part AA's calib_size_audit_control.splices)."""
    from datasets import load_dataset
    from flipquant import data as D
    rng = random.Random(seed)
    out = []
    g = load_dataset(*GSM8K[:2], revision=GSM8K[2], split=GSM8K[3])
    i = rng.randrange(len(g))
    out.append(("gsm8k_test", i, g[i]["question"] + " " + g[i]["answer"]))
    repo, cfg, rev, split, _ = ZS["hellaswag"]
    h = load_dataset(repo, cfg, revision=rev, split=split)
    i = rng.randrange(len(h))
    out.append(("hellaswag", i, " ".join(strings(h[i]))))
    repo, cfg, rev, split, _ = ZS["lambada_openai"]
    lam = load_dataset(repo, cfg, revision=rev, split=split)
    i = rng.randrange(len(lam))
    out.append(("lambada_openai", i, lam[i]["text"]))
    repo, cfg, rev, split, _ = ZS["arc_challenge"]
    arc = load_dataset(repo, cfg, revision=rev, split=split)
    i = rng.randrange(len(arc))
    out.append(("arc_challenge", i, " ".join(strings(arc[i]))))
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=D.WIKI_REVISION, split="test")
    paras = [t for t in wiki["text"] if len(words(t)) > 60]
    out.append(("wikitext2_test", "test", rng.choice(paras)))
    return out


def audit(texts, idx, records=None) -> dict:
    """Hits of the windows' texts, and the positive control on thirty of them."""
    hits, by = [], {}
    for i, text in enumerate(texts):
        shared = {}
        for g in grams(words(text)):
            if g in idx.first:
                shared.setdefault(idx.first[g], []).append(g)
        for (name, item), gs in shared.items():
            by.setdefault(name, set()).add(i)
            r = (records or [{}] * len(texts))[i]
            hits.append(dict(window=i, document_sha256=r.get("document_sha256"), chunk=r.get("chunk"),
                             offset=r.get("offset"), eval_set=name, item=item, shared_grams=len(gs), first_gram=gs[0]))
    items = splices()
    pick = random.Random(11).sample(range(len(texts)), min(30, len(texts)))
    plain = sum(1 for i in pick if any(g in idx.first for g in grams(words(texts[i]))))
    detect = {}
    for s, item, text in items:
        ok = 0
        for i in pick:
            ws = words(texts[i])
            mid = len(ws) // 2
            found = {idx.first[g] for g in grams(ws[:mid] + words(text) + ws[mid:]) if g in idx.first}
            ok += (s, item) in found
        detect[s] = ok
    control = dict(items=[dict(eval_set=s, item=i, words=len(words(t))) for s, i, t in items], picked=len(pick),
                   plain_windows_with_hits=plain, spliced_detected=detect,
                   ok=plain == 0 and all(v == len(pick) for v in detect.values()))
    return dict(windows=len(texts), scanned_13grams=sum(max(0, len(words(t)) - N + 1) for t in texts),
                hits=hits, windows_with_hits={k: len(v) for k, v in by.items()}, control=control,
                passed=not hits and control["ok"])


def child(argv) -> None:
    """In calibration/tmopt (cwd and path first, as the trainer runs): draw the windows, audit them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--fit-source", required=True)
    ap.add_argument("--fit-windows", type=int, required=True)
    ap.add_argument("--trainer-report", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sys.path[0:1] = [str(ENGINE)]
    sys.path.append(str(ROOT))
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    import fit_extension
    import fit_sources
    import run_multiround as RM
    from calibration.tmopt_common import RUNTIME_MODELS
    if a.key in RUNTIME_MODELS:          # as calibration/tmopt_launch.py --register-model does for the trainer
        from calibration.tmopt_launch import register_model
        register_model(SimpleNamespace(MODELS=RM.MODELS), ["--model", a.key], a.key)
    t0 = time.time()
    calibration, development = RM.data_paths(a.key, a.data_root)
    prior = json.loads((calibration / "report.json").read_text())
    tok = AutoTokenizer.from_pretrained(prior["source"], revision=prior["revision"])
    skip = fit_extension.skip_documents(prior, development, RM.PUBLISHED.get(a.key))
    ids, rec = fit_sources.windows(tok, a.fit_source, a.fit_windows, skip)
    tokens = [r["token_sha256"] for r in rec["records"]]
    trained = None
    if a.trainer_report is not None:
        trained = [r["token_sha256"] for r in json.loads(a.trainer_report.read_text())["fit_source"]["records"]]
        if trained != tokens:
            raise SystemExit(f"the drawn windows are not the trained run's ({a.trainer_report})")
    texts = [tok.decode(w[0].tolist(), skip_special_tokens=True) for w in ids]
    idx, info = eval_index(tok)
    res = audit(texts, idx, rec["records"])
    out = dict(status="complete", model_key=a.key, fit_source=a.fit_source, fit_windows=a.fit_windows,
               tokenizer=dict(source=prior["source"], revision=prior["revision"]),
               token_list_sha256=hashlib.sha256("".join(tokens).encode()).hexdigest(),
               equal_to_trainer_report=None if trained is None else True, n=N,
               normalization="lowercase; characters outside [a-z0-9] separate words; word 13-grams within each text",
               eval_sets=idx.stats, **info, **res, seconds=time.time() - t0)
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"AUDIT {a.key} {a.fit_source} {a.fit_windows}: {len(res['hits'])} hits, control ok {res['control']['ok']} "
          f"-> {'PASS' if res['passed'] else 'FAIL'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fit-source", required=True, choices=("fineweb-edu", "ultrachat", "wikitext2-train"))
    ap.add_argument("--fit-windows", type=int, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--trainer-report", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from calibration import tmopt_common as RC
    _, key = RC.resolve(a.model)
    a.out = a.out.resolve()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, Path(__file__).resolve(), "--child", "--key", key, "--data-root", Path(a.data_root).resolve(),
           "--fit-source", a.fit_source, "--fit-windows", a.fit_windows, "--out", a.out]
    if a.trainer_report is not None:
        cmd += ["--trainer-report", Path(a.trainer_report).resolve()]
    r = subprocess.run([str(c) for c in cmd], cwd=ENGINE, env=RC.env({"CUDA_VISIBLE_DEVICES": ""}))
    if r.returncode != 0:
        raise SystemExit(f"the audit failed (rc {r.returncode})")
    sys.exit(0 if json.loads(a.out.read_text())["passed"] else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2:])
    else:
        main()
