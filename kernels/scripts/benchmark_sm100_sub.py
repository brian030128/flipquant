#!/usr/bin/env python3
"""Paper Table D.6: GEMM time on SM100 against MMA width, in the fastest kernel we have.

Every width runs the tuned 256x256x256 2-SM CUTLASS kernel (TMA, five-stage pipeline, warp
specialization, persistent scheduler, 128x64 epilogue) with each 256-wide MMA issued as 256/N
narrower 2-SM MMAs and one B format per N x 64 tile from a map (mixed_sub_n<N>_sm100). N=256 is
the per-tile kernel; the stock uniform-format kernel (mixed_nvfp4_gemm_sm100) is the reference.
N=8 is not expressible: 2-SM MXF4 MMAs need N to be a multiple of 16.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def run(command):
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    ms = re.search(r"Avg latency: ([0-9.]+) ms", output)
    tf = re.search(r"Throughput: ([0-9.]+) TFLOP/s", output)
    if not ms or not tf:
        raise ValueError(output)
    return float(ms[1]), float(tf[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 4096])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--family", action="append", required=True, metavar="PREFIX:W,W,...",
                        help="binaries <PREFIX>_n<W>_sm100, e.g. mixed_sub:64,128,256 or "
                             "mixed_subpad1sm:8,16,32,64,128,256; repeatable")
    args = parser.parse_args()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
                          "-i", "0"], text=True, capture_output=True).stdout.strip()
    cases = []
    for size in args.sizes:
        cases.append(dict(kernel="stock", width=256, pattern="uniform-e2m1", size=size,
                          command=[str(args.build / "mixed_nvfp4_gemm_sm100")]))
        for family in args.family:
            prefix, widths = family.split(":")
            for w in map(int, filter(None, widths.split(","))):
                for pattern in ["random", "e2m1"]:
                    cases.append(dict(kernel=prefix, width=w, pattern=pattern, size=size,
                                      command=[str(args.build / f"{prefix}_n{w}_sm100"),
                                               f"--pattern={pattern}"]))
    for case in cases:
        case["samples"] = []
    # Interleave repetitions so clock and thermal drift hit every kernel alike.
    for _ in range(args.reps):
        for case in cases:
            command = case["command"] + [*map(str, [case["size"]] * 3), "--no-check",
                                         f"--iters={args.iters}"]
            ms, tf = run(command)
            case["samples"].append({"ms": ms, "tflops": tf})
    for case in cases:
        case["median_ms"] = statistics.median(s["ms"] for s in case["samples"])
        case["median_tflops"] = statistics.median(s["tflops"] for s in case["samples"])
    for case in cases:
        stock = next(c for c in cases if c["kernel"] == "stock" and c["size"] == case["size"])
        base = next((c for c in cases if c["kernel"] == case["kernel"] and c["width"] == 256
                     and c["size"] == case["size"] and c["pattern"] == case["pattern"]), stock)
        case["time_vs_stock"] = case["median_ms"] / stock["median_ms"]
        case["time_vs_n256"] = case["median_ms"] / base["median_ms"]
        print(f"{case['size']:5}^3 {case['kernel']:16} N={case['width']:3} {case['pattern']:12} "
              f"{case['median_tflops']:8.1f} TFLOP/s  {case['time_vs_n256']:.3f}x vs N=256  "
              f"{case['time_vs_stock']:.3f}x vs stock", flush=True)
    report = {"account": os.environ.get("SLURM_JOB_ACCOUNT"), "job": os.environ.get("SLURM_JOB_ID"),
              "node": os.uname().nodename, "gpu": gpu, "repetitions": args.reps,
              "iters": args.iters, "timing": "graph" if os.environ.get("MIXFP4_BENCH_GRAPH") == "1"
              else "launch", "cases": [{k: v for k, v in c.items() if k != "command"} for c in cases]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
