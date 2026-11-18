"""GPTQ codes of one Table 9 row, computed once and stored (``--gptq-cache``) for every evaluation that uses them.

    python -m calibration.gptq_codes --model M --weight mixfp4 --act fourover6 --map maps/M/flipquant_16x64.pt \\
        --gptq-cache runs/ptq_combo/gptq/M/flipquant.pt --out runs/ptq_combo/gptq/M/flipquant.json

The model is loaded and quantized exactly as ``evaluation.ppl --ptq gptq --gptq-cache ...`` would (models/cli.py,
flipquant/gptq.py: ModelOpt-aligned defaults, the calibration record's windows, fake-quant layers per token during
the sequential calibration); then the run stops: no evaluation. The evaluations (``--native-backend sm120``) find the
cache under the same key and pack its codes as they are (models/sm120.py ``pack_codes``). ``--out`` gets the report:
the cache key, the file's sha256, per-linear statistics, time; ``status: complete`` last.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from models.cli import add_model_args, build, resources, save_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--out", type=Path, required=True, help="report (.json)")
    args = ap.parse_args(argv)
    if args.gptq_cache is None:
        raise SystemExit("--gptq-cache is the output: give its path")
    args.ptq, args.mode, args.act_scope = "gptq", "fake", "row"
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok, spec, layers, info = build(args)
    if not Path(args.gptq_cache).exists():
        raise SystemExit(f"GPTQ ran but wrote no {args.gptq_cache}")
    with open(args.gptq_cache, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    ptq = dict(info.get("ptq", {}))
    report = dict(model=spec.key, source=info.get("source"), quant=info.get("quant"), rotate=info.get("rotate"),
                  map=info.get("map", {}).get("path"), gptq=ptq, cache=str(args.gptq_cache), cache_sha256=digest,
                  cache_bytes=Path(args.gptq_cache).stat().st_size, resources=resources(started), status="complete")
    save_json(args.out, report)
    print(f"GPTQ codes {spec.key} {args.weight}: {len(layers)} linears -> {args.gptq_cache} "
          f"({report['cache_bytes'] / 2 ** 30:.2f} GiB, {'loaded' if ptq.get('loaded') else 'computed'})", flush=True)


if __name__ == "__main__":
    main()
