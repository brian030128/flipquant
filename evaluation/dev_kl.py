"""Development-set KL to the BF16 model of fake-quant configurations (map fidelity on held-out text).

The development windows are the model's calibration development sets (``calibration.tmopt_common``; Qwen3.8-27B: the
paper's three sets, 192 windows of 512 tokens, never used for fitting and skipped by the ``--fit-windows`` extension).
The BF16 teacher's full-vocabulary log-probabilities are computed once (kept in host memory as BF16, as the trainer's
development monitor does); each configuration is then built from a fresh model with ``models.cli.apply_quant`` -- the
PPL / accuracy path, per-token FourOverSix activations -- and scored per window: KL(teacher || model) and CE, both
the mean over the window's predicted tokens. The report has each configuration's mean and 2 SE over windows, and the
paired difference to the first configuration.

  python -m evaluation.dev_kl --model qwen3.8-27b --configs fourover6 mixfp4:maps/qwen3.8-27b/flipquant_8x64.pt \\
      --data-root $FLIPQUANT_CACHE/tmopt_data --out results/qwen3.8-27b/dev_kl.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calibration.tmopt_common import DRAWN_DEVELOPMENT, MODEL_KEYS, QWEN27B_DEVELOPMENT
from models.cli import add_model_args, apply_quant, resources, save_json
from models.loader import load
from models.registry import get


def development_windows(key: str, root: Path) -> list[torch.Tensor]:
    """``key``: the trainer's model key (tmopt_common.MODEL_KEYS)."""
    names = QWEN27B_DEVELOPMENT if key == "qwen27b" else DRAWN_DEVELOPMENT
    out = []
    for name in names:
        rows = torch.load(root / key / name / "fresh.pt", map_location="cpu", weights_only=True)
        out += [r["ids"] for r in rows]
    return out


def mean_2se(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, 2 * math.sqrt(var / len(xs))


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, modes=("fake",))
    ap.add_argument("--configs", nargs="+", required=True,
                    help="fourover6 | mixfp4:<map path> (each built from a fresh model, per-token FourOverSix acts)")
    ap.add_argument("--data-root", type=Path, required=True, help="calibration records (development sets) root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = time.time()
    key = MODEL_KEYS[get(args.model).key]
    windows = development_windows(key, args.data_root)
    print(f"{len(windows)} development windows of {windows[0].shape[-1]} tokens", flush=True)

    model, tok, spec = load(args.model, revision=args.revision, device_map=args.device_map, attn=args.attn)
    dev = model.get_input_embeddings().weight.device
    teacher, bf16_ce = [], []
    for ids in windows:
        lp = model(input_ids=ids.to(dev), use_cache=False).logits[:, :-1].float().log_softmax(-1)
        bf16_ce.append(F.nll_loss(lp[0], ids[0, 1:].to(dev)).item())
        teacher.append(lp.bfloat16().cpu())
    del model
    torch.cuda.empty_cache()

    results = {"bf16": dict(ce=mean_2se(bf16_ce), kl=(0.0, 0.0), per_window_ce=bf16_ce)}
    first = None
    for cfg in args.configs:
        weight, _, path = cfg.partition(":")
        args.weight, args.map = weight, (Path(path) if path else None)
        model, _, spec = load(args.model, revision=args.revision, device_map=args.device_map, attn=args.attn)
        _, info = apply_quant(model, spec, args)
        kl, ce = [], []
        for ids, t in zip(windows, teacher):
            lp = model(input_ids=ids.to(dev), use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = t.to(lp.device).float()
            kl.append((t.exp() * (t - lp)).sum(-1).mean().item())
            ce.append(F.nll_loss(lp[0], ids[0, 1:].to(lp.device)).item())
            del lp, t
        entry = dict(kl=mean_2se(kl), ce=mean_2se(ce), per_window_kl=kl, per_window_ce=ce,
                     quant=info.get("quant"), map=info.get("map", {}).get("path"))
        if first is None:
            first = cfg
        else:
            base = results[first]
            entry["kl_minus_first"] = mean_2se([a - b for a, b in zip(kl, base["per_window_kl"])])
            entry["ce_minus_first"] = mean_2se([a - b for a, b in zip(ce, base["per_window_ce"])])
        results[cfg] = entry
        print(f"{cfg}: KL {entry['kl'][0]:.5f} +- {entry['kl'][1]:.5f}  CE {entry['ce'][0]:.5f}"
              + (f"  dKL vs {first} {entry['kl_minus_first'][0]:+.5f} +- {entry['kl_minus_first'][1]:.5f}"
                 if "kl_minus_first" in entry else ""), flush=True)
        del model
        torch.cuda.empty_cache()
    report = dict(model=spec.key, windows=len(windows), seqlen=int(windows[0].shape[-1]), data_root=str(args.data_root),
                  development_sets=list(QWEN27B_DEVELOPMENT if key == "qwen27b" else DRAWN_DEVELOPMENT),
                  results=results, args={k: str(v) for k, v in vars(args).items()}, resources=resources(started))
    save_json(args.out, report)
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
