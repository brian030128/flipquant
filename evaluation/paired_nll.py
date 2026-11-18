"""Paired per-window NLL difference between evaluation/ppl.py reports: a map vs a baseline (e.g.
FourOverSix) on the same windows.

For window ``i`` of a dataset, ``d_i = nll_i(run) - nll_i(base)`` (each ``nll_i`` is that window's
token-mean cross-entropy; windows have equal length, so the mean of ``d_i`` is a per-token difference
in nats). Reported: ``mean(d)`` and ``SE = sd(d) / sqrt(n)`` (sample sd, n - 1), both x 1e-3, as
``mean +- 2SE``.

Several runs (e.g. random maps with seeds 0, 1, 2) are averaged *per window* first,
``dbar_i = mean_s d_{s,i}``, and the mean and SE are taken over ``dbar_i``. This keeps the pairing:
the seeds share windows, so their differences are positively correlated and combining per-seed SEs
as if independent (``sqrt(sum SE_s^2) / S``) would understate the uncertainty. That SE is conditional
on the drawn seeds; the spread between seeds is reported separately (``seed_sd`` = sd of the per-seed
means, ``seed_se = seed_sd / sqrt(S)``).

The reports must have evaluated identical windows in the same order: same model and source revision,
same dataset metadata (WikiText-2 revision/token count; the C4 document indices and crop offsets,
which fix the window order), same ``--seqlen``, ``--limit``, ``--use-cache``, and the same number of
windows. They must also share the execution mode, activation format and scope, and native backend
(``--allow-config-mismatch`` to override, e.g. to pair fake vs native on purpose).

  python -m evaluation.paired_nll --base results/nemotron/fo6_ppl.json \\
      --runs results/nemotron/random0_ppl.json results/nemotron/random1_ppl.json results/nemotron/random2_ppl.json \\
      --label random --out results/nemotron/paired_random_8x64.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: Report fields that must agree for the windows to be the same ones, in the same order.
WINDOW_KEYS = ("seqlen", "limit", "use_cache")
#: Quantization settings that must agree for the comparison to isolate the weights.
CONFIG_KEYS = ("mode", "act", "act_scope")


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def _check_same_windows(base: dict, run: dict, name: str, base_path, run_path, check_config: bool):
    def fail(what, a, b):
        raise SystemExit(f"{run_path} vs {base_path} [{name}]: {what} differs: {b!r} vs {a!r}")
    if base.get("model") != run.get("model"):
        fail("model", base.get("model"), run.get("model"))
    for key in ("hf_id", "revision"):
        if base.get("source", {}).get(key) != run.get("source", {}).get(key):
            fail(f"source.{key}", base["source"].get(key), run["source"].get(key))
    for key in WINDOW_KEYS:
        if base["args"].get(key) != run["args"].get(key):
            fail(f"--{key.replace('_', '-')}", base["args"].get(key), run["args"].get(key))
    rb, rr = base["results"][name], run["results"][name]
    if rb["data"] != rr["data"]:
        fail("dataset metadata (revision / documents / offsets)", rb["data"], rr["data"])
    if rb["use_cache"] != rr["use_cache"]:
        fail("use_cache", rb["use_cache"], rr["use_cache"])
    if len(rb["nll"]) != len(rr["nll"]) or rb["windows"] != rr["windows"]:
        fail("number of windows", len(rb["nll"]), len(rr["nll"]))
    if check_config:
        qb, qr = base.get("quant", {}), run.get("quant", {})
        for key in CONFIG_KEYS:
            if qb.get(key) != qr.get(key):
                fail(f"quant.{key}", qb.get(key), qr.get(key))
        if base.get("native_backend") != run.get("native_backend"):
            fail("native_backend", base.get("native_backend"), run.get("native_backend"))


def paired(base: dict, runs: list[dict], name: str) -> dict:
    """Per-window paired differences of ``runs`` (averaged per window) against ``base``."""
    nb = base["results"][name]["nll"]
    diffs = [[r["results"][name]["nll"][i] - nb[i] for i in range(len(nb))] for r in runs]
    per_run = [mean_se(d) for d in diffs]
    dbar = [sum(col) / len(col) for col in zip(*diffs)]
    m, se = mean_se(dbar)
    out = dict(windows=len(nb), runs=len(runs), delta_nll=m, se=se,
               delta_nll_milli=m * 1e3, two_se_milli=2 * se * 1e3,
               ppl_base=base["results"][name]["ppl"],
               ppl_runs=[r["results"][name]["ppl"] for r in runs],
               per_run=[dict(delta_nll_milli=a * 1e3, two_se_milli=2 * b * 1e3) for a, b in per_run])
    if len(runs) > 1:
        _, seed_se = mean_se([a for a, _ in per_run])
        seed_sd = seed_se * math.sqrt(len(runs))
        out.update(seed_sd_milli=seed_sd * 1e3, seed_se_milli=seed_se * 1e3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="baseline ppl.py report (e.g. --weight fourover6)")
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="ppl.py report(s) of one table cell; several (e.g. random seeds) are averaged per window")
    ap.add_argument("--datasets", default=None, help="comma list (default: every dataset in --base)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--allow-config-mismatch", action="store_true",
                    help="do not require equal mode / activation format / scope / native backend")
    ap.add_argument("--out", type=Path, default=None, help="JSON output")
    args = ap.parse_args(argv)
    base = json.loads(args.base.read_text())
    runs = [json.loads(p.read_text()) for p in args.runs]
    names = args.datasets.split(",") if args.datasets else list(base["results"])
    report = dict(label=args.label, base=str(args.base), runs=[str(p) for p in args.runs], results={})
    for name in names:
        for path, run in zip(args.runs, runs):
            if name not in run["results"]:
                raise SystemExit(f"{path} has no {name} results")
            _check_same_windows(base, run, name, args.base, path, not args.allow_config_mismatch)
        r = paired(base, runs, name)
        report["results"][name] = r
        extra = f"  (seed sd {r['seed_sd_milli']:.2f})" if "seed_sd_milli" in r else ""
        print(f"PAIRED {args.label or ''} {name}: dNLL = {r['delta_nll_milli']:+.2f} +- {r['two_se_milli']:.2f} "
              f"x1e-3 nats/token over {r['windows']} windows, {r['runs']} run(s){extra}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
