#!/usr/bin/env python3
"""Collect the SM100 prefill rounds (kernels/docs/results/sm100_paper/02_prefill_latency/<model>/) into one table:
checks every run (graph logits == eager, every quantized Linear ran natively) and writes table.json and table.md."""
import json
from pathlib import Path
import statistics
import sys

ORDER = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b-instruct", "nemotron-nano-9b-v2", "phi-4",
         "qwen3.8-27b"]
METHODS = ["bf16", "nvfp4", "fourover6", "flipquant"]


def main(root):
    root = Path(root)
    table, checks = {}, []
    for model in ORDER:
        s = json.loads((root / model / "summary.json").read_text())
        runs = 0
        for method in s["methods"]:
            for r in s["orders"]:
                d = json.loads((root / model / f"round{r}_{method}.json").read_text())
                g = d["graph"]["1x2048"]
                cov = d.get("coverage")
                ok = (bool(g.get("logits_equal_eager")) or d.get("graph_check") == "report") and d.get("status") == "complete" and (
                    method == "bf16" or (cov and cov["native"] == cov["native_called"] > 0))
                checks.append(dict(model=model, method=method, round=int(r), ok=ok))
                runs += 1
                if method != "bf16" and r in ("0", 0):
                    src = d["policy"]["source"]
                    s.setdefault("native", {})[method] = dict(modules=d["policy"]["native"]["modules"],
                                                              e0m3_tiles=d["policy"]["native"]["e0m3_tiles"],
                                                              quant_reuse=d.get("quant_reuse"))
                if r in ("0", 0):
                    s.setdefault("source", d["policy"]["source"])
                    s.setdefault("gpu", d.get("gpu"))
        table[model] = dict(ms=s["latency_ms_median"], ratios=s["paired_ratio_median"],
                            per_round=s["latency_ms_per_round"], runs=runs, source=s.get("source"),
                            map=s["map"], snapshot=s.get("snapshot"), native=s.get("native"), orders=s["orders"],
                            gpu=s.get("gpu"))
    bad = [c for c in checks if not c["ok"]]
    out = dict(checks=dict(runs=len(checks), failed=bad), models=table)
    (root / "table.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["| model | BF16 | NVFP4 | FourOverSix | FlipQuant 256×64 | NVFP4 / BF16 | FourOverSix / BF16 | "
             "FlipQuant / BF16 | FlipQuant / FourOverSix |", "|---" * 9 + "|"]
    for model in ORDER:
        t = table[model]
        ms = " | ".join("—" if t["ms"].get(m) is None else f"{t['ms'][m]:.2f}" for m in METHODS)
        ratio = lambda k: "—" if t["ratios"].get(k) is None else f"{t['ratios'][k]:.3f}"  # noqa: E731
        lines.append(f"| {model} | {ms} | {ratio('nvfp4/bf16')} | {ratio('fourover6/bf16')} | "
                     f"{ratio('flipquant/bf16')} | {ratio('flipquant/fourover6')} |")
    (root / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"runs {len(checks)}, failed checks {len(bad)}: {bad[:5]}")
    for model in ORDER:
        t = table[model]
        print(model, (t["source"] or {}).get("revision"), (t["map"] or {}).get("revision", "-")[:10],
              (t["map"] or {}).get("sha256", "-")[:16], json.dumps(t["native"])[:200] if t["native"] else "")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kernels/docs/results/sm100_paper/02_prefill_latency")
