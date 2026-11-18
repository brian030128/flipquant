"""The fit set of ``calibration.train_map --fit-windows N``, drawn without training (NVFP4-RaZeR paper-eval part AA).

    python -m calibration.fit_windows --model phi4-14b --fit-windows 2048 --data-root <root> --out windows_2048.json

The trainer draws its fit set itself, after loading the model: the paper's 128 windows (the calibration record), then
N - 128 by the extension rule (``calibration/tmopt/fit_extension.py``). This tool runs the same lines of
``run_train_map.py`` (the record, ``math_code_data``, ``fit_extension.skip_documents`` / ``extend`` with the trainer's
development records and published C4 documents) in the trainer's directory and environment, on the CPU, and writes:
- ``fit_extension``: the rule's record, as the trainer's ``report.json`` holds it (every extra window's source,
  document sha256, offset and token sha256). The file is a ``--calib-extension`` for the release-set readers
  (``flipquant.data.calibration_release``, e.g. GPTQ's ``--gptq-calib-extension``), so GPTQ can use the
  N-window set before or without a FlipQuant run;
- ``token_sha256``: every window's token hash in the trainer's order (the 128, then the extension);
- the record, the tokenizer and this checkout.
The extension rule draws each source's windows from one seeded stream in file order, so the N-window set contains
every smaller one: the first (n - 128) / 2 windows of each source are the n-window set's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "calibration" / "tmopt"


def child(argv) -> None:
    """In calibration/tmopt (cwd and path first, as ``python run_train_map.py``): the trainer's data-load lines."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--fit-windows", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sys.path[0:1] = [str(ENGINE)]
    sys.path.append(str(ROOT))
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    import fit_extension
    import run_multiround as RM
    from run_conditional_format import sha
    from run_math_code_calibration import math_code_data
    from calibration.tmopt_common import RUNTIME_MODELS
    if args.key in RUNTIME_MODELS:          # as calibration/tmopt_launch.py --register-model does for the trainer
        from calibration.tmopt_launch import register_model
        register_model(SimpleNamespace(MODELS=RM.MODELS), ["--model", args.key], args.key)
    started = time.time()
    calibration, development = RM.data_paths(args.key, args.data_root)
    prior = json.loads((calibration / "report.json").read_text())
    tok = AutoTokenizer.from_pretrained(prior["source"], revision=prior["revision"])
    fit, _ = math_code_data(tok, prior["fit"])
    fit = [b for source in ("math", "code") for b in fit[source]]
    base = [sha(w) for w in fit]
    if base != [h for s in ("math", "code") for h in prior["fit"][s]["token_sha256"]]:
        raise SystemExit("the 128 fit windows do not match the calibration record's token hashes")
    skip = fit_extension.skip_documents(prior, development, RM.PUBLISHED.get(args.key))
    extra, ext = fit_extension.extend(tok, prior, args.fit_windows, skip)
    tokens = base + [r["token_sha256"] for r in ext["records"]]
    assert tokens[128:] == [sha(w) for w in extra] and len(set(tokens)) == len(tokens) == args.fit_windows
    out = dict(model_key=args.key, fit_windows=args.fit_windows, data_root=str(args.data_root),
               calibration_record=str(calibration / "report.json"), revision=prior["revision"],
               development=[str(d) for d in development], published=RM.PUBLISHED.get(args.key),
               tokenizer=dict(source=prior["source"], revision=prior["revision"], cls=type(tok).__name__),
               fit_extension=ext, token_sha256=tokens, seconds=time.time() - started)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"WINDOWS {args.key}: {len(tokens)} windows ({len(extra)} by the extension rule) -> {args.out}", flush=True)


def main(argv=None) -> None:
    from calibration import tmopt_common as RC
    from calibration.train_map import checkout
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"registry key or alias, one of {sorted(RC.MODEL_KEYS)}")
    ap.add_argument("--fit-windows", type=int, required=True, metavar="N", help="an even number above 128")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="calibration records (default $FLIPQUANT_CACHE/tmopt_data), as calibration.train_map")
    ap.add_argument("--out", type=Path, required=True, help="the JSON to write (must not exist)")
    args = ap.parse_args(argv)
    if args.fit_windows <= 128 or args.fit_windows % 2:
        raise SystemExit("--fit-windows: an even number above 128 (128 is the calibration record itself)")
    args.out = args.out.resolve()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    spec, key = RC.resolve(args.model)
    root = RC.data_root(args.data_root)
    RC.prepare(key, root)
    RC.prepare_development(key, root)
    code = checkout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, Path(__file__).resolve(), "--child", "--key", key, "--data-root", root,
           "--fit-windows", args.fit_windows, "--out", args.out]
    r = subprocess.run([str(c) for c in cmd], cwd=ENGINE, env=RC.env({"CUDA_VISIBLE_DEVICES": ""}))
    if r.returncode != 0 or not args.out.exists():
        raise SystemExit(f"drawing the fit windows failed (rc {r.returncode})")
    out = json.loads(args.out.read_text())
    out.update(model=spec.key, flipquant=code)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2:])
    else:
        main()
