#!/usr/bin/env python3
"""HackMD #2 / paper Table 5 on SM100: end-to-end prefill latency at 1x2048 tokens of BF16, NVFP4, FourOverSix and
FlipQuant 256x64, on the tuned SM100 kernels (--native-backend sm100-tuned).

Protocol (the authors' requirement): 5 rounds; every (model, method, round) in its own process
(``python -m evaluation.latency prefill``); the method order shuffled in every round with
``random.Random(20260928 + round)``; 7 timed repeats per process; latency = the CUDA-graph median (ms). Ratios are
paired within rounds: per round method / BF16 (and FlipQuant / FourOverSix), then the median over rounds.

Runs on the compute node with the job's Python stack ($PYTHON, kernels/scripts/sm100_pyenv.sh); this driver itself
needs only the standard library.
"""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

SEED = 20260928
METHODS = ("bf16", "nvfp4", "fourover6", "flipquant")
#: models whose eager forward is not bitwise reproducible (Qwen3.8-27B: flash-linear-attention, see
#: 02_prefill_latency/graph_diag): their graph is checked by evaluation.latency --graph-check report
GRAPH_CHECK = {"qwen3.8-27b": "report"}
#: paper model -> (registry key, release 256x64 map repo)
MODELS = {
    "llama3.2-1b-instruct": ("llama3.2-1b-instruct", "edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant"),
    "llama3.1-8b-instruct": ("llama3.1-8b-instruct", "edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant"),
    "mistral-7b-instruct": ("mistral-7b", "edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant"),
    "nemotron-nano-9b-v2": ("nemotron-nano-9b-v2", "edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant"),
    "phi-4": ("phi4-14b", "edgeai-lab/phi-4-FlipQuant"),
    "qwen3.8-27b": ("qwen3.8-27b", "edgeai-lab/Qwen3.8-27B-FlipQuant"),
}


def method_args(method, map_path, quant="exact"):
    if method == "bf16":
        return ["--mode", "bf16"]
    native = ["--mode", "native", "--native-backend", "sm100-tuned", "--sm100-quant", quant]
    if method == "nvfp4":
        return native + ["--weight", "nvfp4", "--act", "nvfp4"]
    if method == "fourover6":
        return native + ["--weight", "fourover6", "--act", "fourover6"]
    return native + ["--weight", "mixfp4", "--act", "fourover6", "--map", map_path]


#: release map repo -> its pinned revision: the revisions of round 1 (01_gemm_overhead) and, for Llama-3.2-1B, the
#: authors' (the one SM120's D.5 used)
MAP_REVISIONS = {
    "edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant": "1896c62f9e45dc734fc0213468362d61ec23e0a8",
    "edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant": "8ab5916e1cb18c610db5d1e973272f04199a7422",
    "edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant": "62a727be895efc77293a1b14453d02f9b90e6a67",
    "edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant": "24c446d656e0c86e3bb947efff0484c870e87267",
    "edgeai-lab/phi-4-FlipQuant": "548b81db1128e81344ae5a5a3c7cc8cbaa48dc11",
    "edgeai-lab/Qwen3.8-27B-FlipQuant": "74c52705f65e3b12a45270755f4fc44d42a15a78",
}


def fetch_map(python, repo):
    rev = MAP_REVISIONS[repo]
    code = ("import hashlib, json; from huggingface_hub import hf_hub_download; "
            f"rev = {rev!r}; p = hf_hub_download({repo!r}, 'flipquant_256x64.pt', revision=rev); "
            "print(json.dumps(dict(path=p, revision=rev, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest())))")
    out = subprocess.run([python, "-c", code], check=True, text=True, capture_output=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def prefetch_model(python, key):
    """Download the model once, before the rounds (which then run offline). Skips the files the loader never reads:
    duplicate original checkpoints, and tokenizer.model, whose Hub metadata in the Mistral repo disagrees with its
    contents (every download fails the size check; the fast tokenizer.json is used instead)."""
    code = ("from huggingface_hub import snapshot_download; from models.registry import get; s = get(%r); "
            "print(snapshot_download(s.hf_id, revision=s.revision, "
            "ignore_patterns=['original/*', 'consolidated*', '*.pth', 'tokenizer.model']))" % key)
    return subprocess.run([python, "-c", code], check=True, text=True, capture_output=True).stdout.strip()


def run_model(python, model, out_dir, rounds, reps, quant="exact"):
    key, repo = MODELS[model]
    map_info = fetch_map(python, repo) if repo else None
    snapshot = prefetch_model(python, key)
    offline = dict(os.environ, HF_HUB_OFFLINE="1")
    methods = [m for m in METHODS if m != "flipquant" or map_info]
    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    orders = {}
    for r in range(rounds):
        order = list(methods)
        random.Random(SEED + r).shuffle(order)
        orders[r] = order
        for method in order:
            out = model_dir / f"round{r}_{method}.json"
            cmd = [python, "-m", "evaluation.latency", "prefill", "--model", key,
                   *method_args(method, map_info["path"] if map_info else None, quant),
                   "--shapes", "1x2048", "--reps", str(reps), "--round", str(r), "--label", method, "--out", str(out),
                   "--graph-check", GRAPH_CHECK.get(model, "bitwise")]
            print(f"[{model}] round {r} {method}", flush=True)
            done = subprocess.run(cmd, text=True, capture_output=True, env=offline)
            if done.returncode != 0:
                (model_dir / f"round{r}_{method}.err").write_text(done.stdout[-4000:] + done.stderr[-8000:])
                print(f"[{model}] round {r} {method} FAILED (exit {done.returncode}); see the .err file", flush=True)
    return dict(model=model, registry_key=key, snapshot=snapshot, map=map_info, methods=methods, orders=orders,
                activation_quantizer=quant)


def aggregate(run, out_dir):
    model_dir = out_dir / run["model"]
    rows = {m: {} for m in run["methods"]}
    meta = {}
    for method in run["methods"]:
        for r in run["orders"]:
            f = model_dir / f"round{r}_{method}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            g = d.get("graph", {}).get("1x2048", {})
            if "ms" in g and (g.get("logits_equal_eager") or d.get("graph_check") == "report"):
                rows[method][r] = g["ms"]
            meta.setdefault(method, dict(source=d.get("policy", {}).get("source"),
                                         weights_gib=d.get("weights_gib"), quant=d.get("policy", {}).get("quant")))
    median = {m: statistics.median(v.values()) if v else None for m, v in rows.items()}

    def paired(a, b):
        common = sorted(set(rows.get(a, {})) & set(rows.get(b, {})))
        return statistics.median(rows[a][r] / rows[b][r] for r in common) if common else None

    ratios = {f"{m}/bf16": paired(m, "bf16") for m in run["methods"] if m != "bf16"}
    if "flipquant" in rows:
        ratios["flipquant/fourover6"] = paired("flipquant", "fourover6")
    return dict(**run, latency_ms_per_round=rows, latency_ms_median=median, paired_ratio_median=ratios, meta=meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--quant", choices=("exact", "fast"), default="exact",
                    help="the FP4 methods' activation quantizer (--sm100-quant); exact = bit-identical to the simulation")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild <model>/summary.json from the round files of an earlier run")
    args = ap.parse_args()
    python = os.environ.get("PYTHON")
    results = {}
    for model in args.models:
        if args.aggregate_only:
            old = json.loads((args.out / model / "summary.json").read_text())
            run = {k: old[k] for k in ("model", "registry_key", "snapshot", "map", "methods", "orders")}
            run["activation_quantizer"] = old.get("activation_quantizer", "exact")
        else:
            run = run_model(python, model, args.out, args.rounds, args.reps, args.quant)
        results[model] = aggregate(run, args.out)
        (args.out / model / "summary.json").write_text(json.dumps(results[model], indent=1) + "\n")
        r = results[model]
        print(json.dumps(dict(model=model, ms=r["latency_ms_median"], ratios=r["paired_ratio_median"])), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
