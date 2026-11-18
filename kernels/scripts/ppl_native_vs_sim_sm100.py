#!/usr/bin/env python3
"""HackMD round 2 #3 / #4 (paper D.6, D.7) on SM100: WikiText-2 and C4 perplexity of the release 256x64 map run natively
(the per-tile kernel, ``--native-backend sm100-tuned``) and simulated (fake quant: the same weights, map and per-token
FourOverSix activations), with the ownership check of every quantized Linear in the native run.

Per model, two ``python -m evaluation.ppl`` processes with the paper's convention (``--paper-convention``: per-token
FourOverSix activations) on the reference windows (Appendix A.4: 2048-token windows, all of WikiText-2's test split
and 256 of C4's): ``native.json`` (``--mode native --native-backend sm100-tuned --ownership-check``) and
``simulated.json`` (``--mode fake``). Both run offline after one online step that downloads the model and builds the
evaluation windows (cached by flipquant/data.py). The summary is kernels/scripts/summarize_ppl_sm100.py.

    python kernels/scripts/ppl_native_vs_sim_sm100.py --out DIR --models llama3.2-1b-instruct phi-4

Runs on the compute node with the job's Python stack ($PYTHON); this driver needs only the standard library.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from prefill_rounds_sm100 import MODELS, fetch_map, prefetch_model  # noqa: E402

RUNS = {"native": ["--mode", "native", "--native-backend", "sm100-tuned", "--ownership-check"],
        "simulated": ["--mode", "fake"],
        # the fast activation quantizer (kernels/src/quant_act_fast.cuh, not bit-identical); not run by default
        "native_fast": ["--mode", "native", "--native-backend", "sm100-tuned", "--sm100-quant", "fast"]}
DEFAULT_RUNS = ["native", "simulated"]


def build_windows(python, key):
    """The evaluation windows, built online once (the datasets) with the tokenizer read from the prefetched snapshot
    (offline: the Mistral repo's tokenizer.model fails every download)."""
    code = ("from transformers import AutoTokenizer; from models.registry import get; from flipquant.data import "
            "eval_windows; s = get(%r); tok = AutoTokenizer.from_pretrained(s.hf_id, revision=s.revision, "
            "local_files_only=True); print({d: len(eval_windows(tok, d, 2048)[0]) for d in ('wiki', 'c4')})" % key)
    return subprocess.run([python, "-c", code], check=True, text=True, capture_output=True).stdout.strip()


def run_model(python, model, out_dir, runs):
    key, repo = MODELS[model]
    map_info = fetch_map(python, repo)
    snapshot = prefetch_model(python, key)
    windows = build_windows(python, key)
    print(f"[{model}] map {map_info['revision'][:10]} sha256 {map_info['sha256'][:16]}, windows {windows}", flush=True)
    offline = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for run in runs:
        out = model_dir / f"{run}.json"
        cmd = [python, "-m", "evaluation.ppl", "--model", key, *RUNS[run], "--weight", "mixfp4", "--act", "fourover6",
               "--map", map_info["path"], "--paper-convention", "--datasets", "wiki,c4", "--out", str(out)]
        print(f"[{model}] {run}: {' '.join(cmd[1:])}", flush=True)
        done = subprocess.run(cmd, text=True, capture_output=True, env=offline)
        status[run] = done.returncode
        (model_dir / f"{run}.log").write_text(done.stdout[-20000:] + done.stderr[-20000:])
        print(f"[{model}] {run} exit {done.returncode}: " +
              " | ".join(line for line in done.stdout.splitlines() if line.startswith("PPL ")), flush=True)
    job = os.environ.get("SLURM_JOB_ID")
    meta = dict(model=model, registry_key=key, snapshot=snapshot, map=map_info, windows=windows, exit=status, job=job)
    path = model_dir / "run.json"
    if path.exists():                       # a later run of other --runs: keep the earlier ones' record
        old = json.loads(path.read_text())
        meta = dict(old, exit={**old.get("exit", {}), **status},
                    jobs={**old.get("jobs", {r: old.get("job") for r in old.get("exit", {})}), **{r: job for r in status}})
    path.write_text(json.dumps(meta, indent=1) + "\n")
    return all(code == 0 for code in status.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, choices=list(RUNS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    python = os.environ["PYTHON"]
    ok = [run_model(python, model, args.out, args.runs) for model in args.models]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
