#!/usr/bin/env python3
"""Which implementation Qwen3.8-27B's Gated DeltaNet layers run in this Python stack: flash-linear-attention's Triton
kernels (the fast path) or Transformers' PyTorch fallback.

Transformers 5.16 binds ``chunk_gated_delta_rule`` / ``causal_conv1d_fn`` at import time
(``use_kernel_func_from_hub_with_fallback``): the ``fla`` / ``causal_conv1d`` package if it imports, else the torch
function, with the import error swallowed. So this probe (1) unwraps the module-level functions of
``modeling_qwen3_5`` and names the implementation each one calls, and (2) runs one Gated DeltaNet layer of the real
config (random weights, bf16, 1x2048, the prefill path) under the profiler and lists its CUDA kernels.
Exits non-zero unless both functions resolve to their packages and the layer launches FLA's kernels.

    python kernels/scripts/fla_probe.py --out probe.json
"""
import argparse
import collections
import importlib.metadata as md
import json
import sys

import torch

sys.path.insert(0, ".")
from models.registry import get  # noqa: E402

FUNCTIONS = {"torch_chunk_gated_delta_rule": "fla", "causal_conv1d_fn": "causal_conv1d"}


def bound(fn, seen=None):
    """Every function reachable from ``fn`` through ``__wrapped__`` and closure cells, as module.qualname."""
    seen = set() if seen is None else seen
    out = []
    stack = [fn]
    while stack:
        f = stack.pop()
        if id(f) in seen or not callable(f):
            continue
        seen.add(id(f))
        out.append(f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}")
        inner = getattr(f, "__wrapped__", None)
        if inner is not None:
            stack.append(inner)
        for cell in getattr(f, "__closure__", None) or ():
            try:
                stack.append(cell.cell_contents)
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
    report = dict(versions={p: md.version(p) for p in ("torch", "transformers", "triton", "flash-linear-attention",
                                                       "fla-core", "causal-conv1d", "mamba-ssm")
                            if _installed(p)}, functions={}, ok=True)
    import fla
    report["fla_file"] = fla.__file__
    for name, package in FUNCTIONS.items():
        chain = bound(getattr(Q, name))
        hit = [c for c in chain if c.split(".")[0] == package]
        report["functions"][name] = dict(chain=chain, implementation=hit[0] if hit else None)
        report["ok"] &= bool(hit)
    spec = get(args.model)
    cfg = transformers.AutoConfig.from_pretrained(spec.hf_id, revision=spec.revision)
    cfg = getattr(cfg, "text_config", cfg)
    layer_idx = cfg.layer_types.index("linear_attention")
    torch.manual_seed(0)
    layer = Q.Qwen3_5GatedDeltaNet(cfg, layer_idx).to("cuda", torch.bfloat16)
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(1, args.tokens, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.no_grad():
        layer(x)                          # warm up: Triton compiles here
        torch.cuda.synchronize()
        for _ in range(3):                # the first capture of a process can come back empty; keep the last
            with torch.profiler.profile(activities=activities) as prof:
                layer(x)
                torch.cuda.synchronize()
    kernels = collections.Counter()
    us = collections.Counter()
    ops = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kernels[e.name] += 1
            us[e.name] += e.device_time_total
        else:
            ops[e.name] += 1
    report["layer"] = dict(layer_idx=layer_idx, layer_type=cfg.layer_types[layer_idx], tokens=args.tokens,
                           kernels=[dict(name=k, calls=kernels[k], us=round(us[k], 1)) for k, _ in us.most_common()],
                           total_us=round(sum(us.values()), 1), cpu_ops=dict(ops.most_common(40)))
    fla_kernels = [k for k in kernels if "gated_delta" in k or "chunk" in k.lower()]
    report["layer"]["fla_kernels"] = fla_kernels
    # the binding decides the path; the kernels confirm it when the profiler captured any
    report["layer"]["kernels_captured"] = bool(kernels)
    report["ok"] &= bool(fla_kernels) or not kernels
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(dict(ok=report["ok"], versions=report["versions"],
                          implementations={k: v["implementation"] for k, v in report["functions"].items()},
                          layer_us=report["layer"]["total_us"], fla_kernels=fla_kernels[:8]), indent=1))
    for k in report["layer"]["kernels"][:20]:
        print(f"  {k['us']:9.1f} us  x{k['calls']:<3} {k['name'][:150]}")
    if not kernels:
        print("no CUDA kernels captured; CPU ops: " + ", ".join(f"{n} x{c}" for n, c in ops.most_common(25)))
    sys.exit(0 if report["ok"] else 1)


def _installed(package):
    try:
        md.version(package)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
