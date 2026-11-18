"""Perplexity on WikiText-2 and C4 -- the reference protocol, in fake-quant or native mode.

Windows are exactly the reference implementation's (flipquant/data.py). PPL = exp(mean over windows of the per-window
token-mean cross-entropy), which with equal-length windows is the token-level PPL. The default
activation scope is ``document`` (one FP32 activation scale per sequence), the convention of the
reference PPL numbers; ``--act-scope row`` gives the causal per-token convention used for
generation.

``--paper-convention`` is the FlipQuant paper's PPL convention, made explicit: per-token (``row``)
FourOverSix activations, the convention of the paper's PPL tables (the sm120 NativeLinear,
convention (c)). It only sets ``--act-scope row`` and records the convention in the report; the default
is unchanged. Compare runs with the same convention only.

  # H100/H200: fake quant of a trained map
  python -m evaluation.ppl --model qwen3-8b --mode fake --weight mixfp4 --map runs/qwen3-8b/map.pt
  # B200 / RTX Pro 6000: the same map on the native kernel
  python -m evaluation.ppl --model qwen3-8b --mode native --weight mixfp4 --map runs/qwen3-8b/map.pt
  # baselines
  python -m evaluation.ppl --model qwen3-8b --mode bf16
  python -m evaluation.ppl --model qwen3-8b --weight fourover6
  # the paper's convention (per-token FourOverSix activations)
  python -m evaluation.ppl --model qwen3-8b --mode native --weight mixfp4 --map runs/qwen3-8b/map.pt --paper-convention
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flipquant.data import eval_windows
from models.cli import add_model_args, build, resolve_act_method, resources, save_json


#: The reference evaluates WikiText-2 with ``use_cache=True`` and C4 with ``use_cache=False``; kept as the
#: default for fidelity, though on Llama-3.2-1B-Instruct + the reference's ins1b_8x64_ste map it does not change
#: PPL (wiki 14.5060 with the cache on and off, Slurm 461671/461672). What does matter is the CE
#: reduction below: with it, that replay reproduces the reference report exactly (wiki 14.5060, c4 19.6532)
#: under the reference's transformers 4.57.3; under 5.16.1 wiki is identical and c4 is 19.6574 (+0.02%).
PROTOCOL_USE_CACHE = {"wiki": True, "c4": False}


@torch.no_grad()
def perplexity(model, windows, batch: int = 1, limit: int | None = None, use_cache: bool = False) -> dict:
    device = model.get_input_embeddings().weight.device
    windows = windows[:limit] if limit else windows
    nll = []
    for i in range(0, len(windows), batch):
        ids = torch.cat(windows[i:i + batch]).to(device)
        logits = model(input_ids=ids, use_cache=use_cache).logits
        target = ids[:, 1:].to(logits.device)
        # Exactly the reference's reduction (flattened mean per window). The mathematically equal
        # transpose(1, 2) + reduction="none" form differs by ~2e-4 nats per window.
        for b in range(ids.shape[0]):
            nll.append(float(F.cross_entropy(logits[b, :-1].float(), target[b])))
        del logits
    return dict(ppl=math.exp(sum(nll) / len(nll)), nll=nll, windows=len(nll))


PAPER_CONVENTION = "FlipQuant paper: per-token (row) FourOverSix activations"


def apply_paper_convention(args, argv) -> None:
    """``--paper-convention``: ``--act-scope row`` with FourOverSix activations; an explicit conflicting choice
    is an error rather than silently overridden."""
    args.convention = None
    if not args.paper_convention:
        return
    if getattr(args, "act_method", None) == "own":
        raise SystemExit("--paper-convention is FourOverSix activations; --act-method own is per token already, drop it")
    explicit = any(a == "--act-scope" or a.startswith("--act-scope=") for a in argv)
    if explicit and args.act_scope != "row":
        raise SystemExit("--paper-convention uses per-token activations; drop --act-scope or set it to row")
    if args.act != "fourover6" or args.nvidia_nvfp4:
        raise SystemExit("--paper-convention is defined for FourOverSix activations (--act fourover6, no --nvidia-nvfp4)")
    args.act_scope, args.convention = "row", PAPER_CONVENTION


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, act_scope="document")
    ap.add_argument("--datasets", default="wiki,c4")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N windows (smoke tests)")
    ap.add_argument("--use-cache", choices=("protocol", "on", "off"), default="protocol",
                    help="protocol = the reference's (wiki on, c4 off); see PROTOCOL_USE_CACHE")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    ap.add_argument("--paper-convention", action="store_true",
                    help="the FlipQuant paper's PPL convention: per-token (row) FourOverSix activations")
    args = ap.parse_args(argv)
    resolve_act_method(args, sys.argv[1:] if argv is None else argv)
    apply_paper_convention(args, sys.argv[1:] if argv is None else argv)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok, spec, layers, info = build(args)
    if spec.batch1_only and args.batch != 1:
        raise SystemExit(f"{spec.key}: batched forward differs from batch 1; use --batch 1")
    report = dict(info, args={k: str(v) for k, v in vars(args).items()}, results={}, convention=args.convention)
    for name in args.datasets.split(","):
        windows, meta = eval_windows(tok, name, args.seqlen)
        cache = PROTOCOL_USE_CACHE[name] if args.use_cache == "protocol" else args.use_cache == "on"
        r = perplexity(model, windows, args.batch, args.limit, use_cache=cache)
        r["use_cache"] = cache
        r["data"] = {k: v for k, v in meta.items() if k != "token_sha256"}
        report["results"][name] = r
        print(f"PPL {spec.key} {args.mode} w={args.weight} {name}: {r['ppl']:.4f} ({r['windows']} windows)", flush=True)
        if args.out:
            save_json(args.out, report)
    report["resources"] = resources(started)
    if args.out:
        save_json(args.out, report)
    return report


if __name__ == "__main__":
    main()
