#!/usr/bin/env python3
"""Whole-model validation of the sm120 path (models/sm120.py) against the reference implementation itself.

    python kernels/sm120/validate_models.py --reference-root /path/to/reference \\
        --reference-build DIR --reference-build-paper DIR --artifacts DIR --model phi4-14b --reference-key phi4 \\
        [--map-root DIR] [--hf-id ID --revision REV] [--parallel] --out results/validation_phi4.json

For each case (a map unit or a baseline, on a route) the same model, inputs and weights run twice, each in its own
process:
* **ours:** this repository. `models.loader.load`, then `models.sm120.install` with the case's route (`auto`, `paper`
  or `auto_stock_wB`) on this repository's builds (`--ours-build`, default `kernels/sm120/build_deploy`, the
  deployment directory of build_deploy.sh). The route must resolve to the case's set.
  - The map is `maps/<model>/flipquant_<unit>.pt`, or with `--map-root` the reference's `<root>/<reference key>_<unit>/
    map.pt`.
  - `--hf-id` / `--revision` load the reference's exact checkpoint where the registry's differs (e.g. its base
    Mistral-7B-v0.3 against the registry's Instruct model).
* **theirs:** the reference implementation (`--reference-root`, the source of kernels/sm120). Its
  `sm120/eval/common.load_model`, then `mixfp4_sm120.model.install` from its exported artifact
  (`<artifacts>/<reference key>_tc_<unit>`, `_fo6`, `_nvfp4`) on the same set.
  - The optimized sets come from its deployment directory (`--reference-build`).
  - The paper sets come from `--reference-build-paper`.

The logits of every input shape must be equal bit for bit. Both sides read the same tiles: the reference's artifacts are
exported from the paper maps.
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (case label, weight, unit, route, the set the reference's resolve_kernel gives (its kernel-opt ae389a5), artifact suffix)
CASES = [("16x64-paper", "mixfp4", "16x64", "paper", "mixed", "tc_16x64"),
         ("16x64-auto", "mixfp4", "16x64", "auto", "mixed_ko", "tc_16x64"),
         ("8x64-paper", "mixfp4", "8x64", "paper", "n8k64_wB", "tc_8x64"),
         ("8x64-auto", "mixfp4", "8x64", "auto", "mixed_wB_ko", "tc_8x64"),
         ("256x64-paper", "mixfp4", "256x64", "paper", "mixed", "tc_256x64"),
         ("256x64-auto", "mixfp4", "256x64", "auto", "mixed256_ko", "tc_256x64"),
         ("fo6-paper", "fourover6", None, "paper", "stock", "fo6"),
         ("fo6-auto", "fourover6", None, "auto", "stock_ko", "fo6"),
         ("fo6-auto_stock_wB", "fourover6", None, "auto_stock_wB", "stock_wB_ko", "fo6"),
         ("nvfp4-paper", "nvfp4", None, "paper", "stock", "nvfp4"),
         ("nvfp4-auto", "nvfp4", None, "auto", "stock_ko", "nvfp4"),
         ("nvfp4-auto_stock_wB", "nvfp4", None, "auto_stock_wB", "stock_wB_ko", "nvfp4")]
SHAPES = [(1, 1), (1, 16), (1, 64), (1, 200), (2, 512)]


def ours(args, case, ids_path, out):
    import torch
    sys.path.insert(0, str(ROOT))
    from flipquant import maps as M
    from models import sm120 as RZ
    from models.loader import load
    from models.registry import get
    label, weight, unit, route, expected, _ = case
    spec = get(args.model)
    if args.hf_id or args.revision:
        spec = dataclasses.replace(spec, hf_id=args.hf_id or spec.hf_id, revision=args.revision or spec.revision)
    model, _, spec = load(spec)
    tiles = None
    if weight == "mixfp4":
        if args.map_root:
            tiles, unit, _ = M.load(Path(args.map_root) / f"{args.reference_key}_{unit}" / "map.pt", unit)
        else:
            tiles, unit, _ = M.load(ROOT / "maps" / args.model / f"flipquant_{unit}.pt")
    rep = RZ.install(model, spec, weight=weight, tiles=tiles, unit=unit, kernel=route,
                     act="nvfp4" if weight == "nvfp4" else "fourover6")
    if rep["kernel"] != expected:
        raise SystemExit(f"{label}: route {route!r} resolved to {rep['kernel']!r}, not {expected!r} ({rep['routing']})")
    ids = torch.load(ids_path)
    with torch.no_grad():
        logits = [model(input_ids=x.cuda(), use_cache=False).logits.cpu() for x in ids]
    torch.save(dict(logits=logits, kernel=rep["kernel_set"], routing=rep["routing"], modules=rep["modules"],
                    source=getattr(model, "flipquant_source", None)), out)


def theirs(args, case, ids_path, out):
    import torch
    root = Path(args.reference_root)
    sys.path[:0] = [str(root / "sm120"), str(root / "sm120" / "eval"), str(root)]
    import common as C
    from mixfp4_sm120 import model as RM
    from mixfp4_sm120.lib import Kernel
    from mixfp4_sm120.select import FAMILIES, KernelSet
    label, weight, unit, route, expected, suffix = case
    model = C.load_model(args.reference_key)
    kern = KernelSet(expected) if expected in FAMILIES else Kernel.load(expected)
    rep = RM.install(model, Path(args.artifacts) / f"{args.reference_key}_{suffix}", kernel=kern,
                     loader=C.MODELS[args.reference_key]["loader"])     # its scope (e.g. Qwen3.5: the text model only)
    ids = torch.load(ids_path)
    with torch.no_grad():
        logits = [model(input_ids=x.cuda(), use_cache=False).logits.cpu() for x in ids]
    torch.save(dict(logits=logits, kernel=rep.kernel_set, routing=None, modules=len(rep.native), source=None), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--reference-build", required=True, help="the reference's deployment directory (the optimized sets)")
    ap.add_argument("--reference-build-paper", default=None, help="the reference's paper builds (default <root>/sm120/build)")
    ap.add_argument("--ours-build", default=str(ROOT / "kernels" / "sm120" / "build_deploy"),
                    help="this repository's build directory (default the deployment directory build_deploy)")
    ap.add_argument("--artifacts", required=True, help="the reference's exported artifacts of the same maps")
    ap.add_argument("--model", default="phi4-14b", help="this repository's registry key")
    ap.add_argument("--reference-key", default="phi4", help="the reference's model key")
    ap.add_argument("--map-root", default=None, help="the reference's paper maps (<root>/<reference key>_<unit>/map.pt)")
    ap.add_argument("--hf-id", default=None, help="ours: load this checkpoint instead of the registry's")
    ap.add_argument("--revision", default=None, help="ours: and this revision")
    ap.add_argument("--parallel", action="store_true", help="run the two sides of a case at the same time")
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    ap.add_argument("--work", type=Path, default=ROOT / "runs" / "model_validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--role", choices=("driver", "ours", "theirs"), default="driver")
    ap.add_argument("--case", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--result", default=None)
    args = ap.parse_args()
    if args.role != "driver":
        case = next(c for c in CASES if c[0] == args.case)
        (ours if args.role == "ours" else theirs)(args, case, args.ids, args.result)
        return
    import torch
    args.work.mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig
    sys.path.insert(0, str(ROOT))
    from models.registry import get
    spec = get(args.model)
    config = AutoConfig.from_pretrained(args.hf_id or spec.hf_id, revision=args.revision or spec.revision)
    vocab = config.get_text_config().vocab_size        # the text model's (Qwen3.5-family configs nest it)
    g = torch.Generator().manual_seed(0)
    ids = [torch.randint(0, vocab, shape, generator=g) for shape in SHAPES]
    ids_path = args.work / f"ids_{args.reference_key}.pt"
    torch.save(ids, ids_path)
    paper_build = args.reference_build_paper or str(Path(args.reference_root) / "sm120" / "build")
    report = dict(model=args.model, reference_key=args.reference_key, hf_id=args.hf_id, revision=args.revision,
                  map_root=args.map_root, shapes=SHAPES, ours_build=args.ours_build, reference_build=args.reference_build,
                  reference_build_paper=paper_build, cases={})
    for case in [c for c in CASES if c[0] in args.cases.split(",")]:
        files, procs = {}, {}
        t0 = time.time()
        for role in ("ours", "theirs"):
            res = args.work / f"{args.reference_key}_{case[0]}_{role}.pt"
            env = dict(os.environ)
            env["SM120_BUILD_DIR"] = (args.ours_build if role == "ours" else paper_build if case[3] == "paper"
                                      else args.reference_build)
            cmd = [sys.executable, __file__, "--role", role, "--case", case[0], "--ids", str(ids_path), "--result",
                   str(res), "--reference-root", args.reference_root, "--reference-build", args.reference_build, "--artifacts",
                   args.artifacts, "--model", args.model, "--reference-key", args.reference_key, "--out", str(args.out)]
            for opt in ("map_root", "hf_id", "revision"):
                if getattr(args, opt):
                    cmd += [f"--{opt.replace('_', '-')}", getattr(args, opt)]
            files[role] = res
            procs[role] = subprocess.Popen(cmd, env=env, cwd=ROOT if role == "ours" else args.reference_root)
            if not args.parallel and procs[role].wait() != 0:
                raise SystemExit(f"{case[0]} {role} failed")
        for role, p in procs.items():
            if p.wait() != 0:
                raise SystemExit(f"{case[0]} {role} failed")
        a, b = torch.load(files["ours"]), torch.load(files["theirs"])
        equal = [bool(torch.equal(x.view(torch.int16), y.view(torch.int16))) for x, y in zip(a["logits"], b["logits"])]
        report["cases"][case[0]] = dict(equal=equal, route=case[3], set=case[4], modules=[a["modules"], b["modules"]],
                                        routing=a["routing"], kernels=[a["kernel"], b["kernel"]], source=a["source"],
                                        seconds=round(time.time() - t0))
        print(f"{case[0]}: logits equal per shape {equal} ({time.time() - t0:.0f} s; {a['routing']})", flush=True)
        for f in files.values():
            f.unlink()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    report["all_equal"] = all(all(c["equal"]) for c in report["cases"].values())
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ALL EQUAL" if report["all_equal"] else "DIFFERENCES FOUND", flush=True)
    sys.exit(0 if report["all_equal"] else 1)


if __name__ == "__main__":
    main()
