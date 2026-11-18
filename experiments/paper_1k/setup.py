"""Setup of the paper's 1k reruns, before any item (RUNBOOK.md section 1). Each step prints PASS / FAIL; run them in order.

    python -m experiments.paper_1k.setup env          # the interpreters' package versions against env/*.yml
    python -m experiments.paper_1k.setup kernels      # build the SM120 kernels (CPU) and check their SASS
    python -m experiments.paper_1k.setup hf           # Hugging Face access: the gated Llama models, the public artifacts
    python -m experiments.paper_1k.setup fetch        # download every pinned artifact and check its sha256
    python -m experiments.paper_1k.setup records --data-root DIR    # calibration records, against reference/records.json
    python -m experiments.paper_1k.setup windows --data-root DIR --out DIR   # 256 / 1k / 2k windows, against windows/*.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import artifacts as A
from . import config as C

PINS = {"torch": "2.9.0+cu128", "transformers": "5.16.1", "lm_eval": "0.4.11", "accelerate": "1.13.0",
        "datasets": "4.8.5", "safetensors": "0.8.0", "triton": "3.5.0", "huggingface_hub": "1.31.0"}
PINS_FAST = dict(PINS, mamba_ssm="2.3.2.post1", causal_conv1d="1.7.0", fla="0.5.2")
GATED = ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B-Instruct")


def say(ok, what):
    print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
    return ok


def env_check():
    code = ("import importlib, json, sys\nout = {}\nfor m in sys.argv[1].split(','):\n"
            "    try:\n        mod = importlib.import_module(m)\n        out[m] = getattr(mod, '__version__', 'present')\n"
            "    except Exception as e:\n        out[m] = 'MISSING: ' + type(e).__name__\nprint(json.dumps(out))")
    ok = True
    for label, py, pins in (("eval (PAPER1K_PY)", C.py(C.MODELS[0]), PINS), ("fast (PAPER1K_PY_FAST)", C.py(C.HYBRID[0]), PINS_FAST)):
        r = subprocess.run([py, "-c", code, ",".join(pins)], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        bad = {k: (got.get(k), v) for k, v in pins.items() if str(got.get(k)) != v}
        ok &= say(not bad, f"{label} {py}: " + ("all pinned versions" if not bad else f"differs: {bad}"))
    return ok


def kernels(build: Path):
    """kernels/sm120/build_deploy.sh (CPU; CUDA_HOME = a CUDA 13.1.115 toolkit), then the SASS gate."""
    if not os.environ.get("CUDA_HOME"):
        return say(False, "CUDA_HOME is not set (a CUDA 13.1.115 toolkit; env/mixfp4-cuda131.yml)")
    r = subprocess.run(["bash", str(C.REPO / "kernels" / "sm120" / "build_deploy.sh"), str(build)], cwd=C.REPO)
    if not say(r.returncode == 0, f"build_deploy.sh -> {build}"):
        return False
    return sass_gate(build)


def sass_gate(build: Path):
    """tests/test_sm120_native.py's build tests: every build's patched and unpatched SASS equal the reference's
    (SASS_REFERENCE.json; the deployment builds = the paper's build_V), and 'auto' refuses unoptimized builds. The
    second loads the libraries, which initializes CUDA: the GPU stays visible (never run this on a hung driver)."""
    r = subprocess.run([C.py(C.MODELS[0]), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_sm120_native.py",
                        "-k", "sass_matches_reference or refuses_unoptimized"], cwd=C.REPO,
                       env=dict(os.environ, SM120_BUILD_DIR=str(build)))
    return say(r.returncode == 0, f"SASS of every build in {build} equals the reference")


def hf_check():
    from huggingface_hub import HfApi
    ok = True
    api = HfApi()
    for rid in GATED:
        try:
            api.model_info(rid)
            ok &= say(True, f"access to the gated {rid}")
        except Exception as e:
            ok &= say(False, f"{rid}: {type(e).__name__} (log in with `hf auth login` and accept the model's license)")
    anon = HfApi(token=False)
    for rid, r in A.MANIFEST["repos"].items():
        try:
            info = anon.model_info(rid)
            ok &= say(info.sha is not None, f"{rid} reachable (pinned {r['revision'][:12]})")
        except Exception as e:
            ok &= say(False, f"{rid}: {type(e).__name__}")
    return ok


def fetch_all():
    ok = True
    for rid, r in A.MANIFEST["repos"].items():
        kind = r["kind"]
        units = C.UNITS if kind.startswith("flipquant") else (None,)
        for u in units:
            try:
                path = A.fetch(r["model"], kind, u)
                ok &= say(True, f"{rid} {A.file_name(r['model'], kind, u)} sha256 = pinned ({path})")
            except SystemExit as e:
                ok &= say(False, str(e))
    return ok


def records(data_root: Path):
    """`calibration.tmopt_ext record` per model, then its fields against reference/records.json."""
    import hashlib
    ref = json.loads((C.HERE / "reference" / "records.json").read_text())["records"]
    ok = True
    for m in C.MODELS:
        r = subprocess.run([C.py(m), "-m", "calibration.tmopt_ext", "record", "--model", m, "--data-root", str(data_root)],
                           cwd=C.REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if not say(r.returncode == 0, f"tmopt_ext record {m}"):
            ok = False
            continue
        rec = json.loads((data_root / C.TRAINER_KEY[m] / "calibration" / "report.json").read_text())
        e = ref[m]
        digest = hashlib.sha256("".join(f"{n}:{v['source_sha256']}\n" for n, v in rec["matrices"].items()).encode()).hexdigest()
        same = (rec["revision"] == e["revision"] and digest == e["matrices_digest"]
                and all(rec["fit"][s]["token_sha256"] == e["fit"][s]["token_sha256"] for s in ("math", "code")))
        ok &= say(same, f"{m}: record revision, fit token hashes and matrices digest equal the paper-eval record's")
    return ok


#: attempts per windows file: a draw streams public datasets, and on the porting machine 3 of 13 draws aborted at
#: interpreter exit (rc -6, after a cut-off download and its retry); the token hashes are checked whatever the attempt
DRAW_ATTEMPTS = 3


def _drawn(dst: Path) -> bool:
    """A windows file is finished when fit_windows' wrapper completed it (its last write adds 'flipquant'). The child
    writes the file first: a draw that fails after that (phase A: a child aborted at exit, rc -6, after a dataset
    download retry) leaves a file without it, which is drawn again."""
    try:
        return "flipquant" in json.loads(dst.read_text())
    except (OSError, ValueError):
        return False


def _aside(dst: Path):
    """fit_windows refuses an existing --out: an unfinished file is moved aside (<name>.unfinished<k>.json), never
    deleted."""
    if dst.exists():
        k = 1
        while dst.with_name(f"{dst.stem}.unfinished{k}.json").exists():
            k += 1
        dst.rename(dst.with_name(f"{dst.stem}.unfinished{k}.json"))
        print(f"moved the unfinished {dst.name} aside: {dst.stem}.unfinished{k}.json", flush=True)


def windows(data_root: Path, out: Path):
    """calibration.fit_windows for 256, 1,024 and 2,048 windows per model; every window's token hash must equal the
    paper's (256: the earlier tables' set, which step 0s reuse; 1k: the paper's set; 2k: items 11 and 12)."""
    ok = True
    for m in C.MODELS:
        for n in (256, 1024, 2048):
            dst = out / "windows" / f"{m}_{n}.json"
            for attempt in range(1, DRAW_ATTEMPTS + 1):
                if _drawn(dst):
                    break
                dst.parent.mkdir(parents=True, exist_ok=True)
                _aside(dst)
                r = subprocess.run([C.py(m), "-m", "calibration.fit_windows", "--model", m, "--fit-windows", str(n),
                                    "--data-root", str(data_root), "--out", str(dst)], cwd=C.REPO,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
                if r.returncode:
                    print(f"fit_windows {m} {n}: attempt {attempt} of {DRAW_ATTEMPTS} failed (rc {r.returncode})", flush=True)
            if not _drawn(dst):
                ok &= say(False, f"fit_windows {m} {n}")
                continue
            got = json.loads(dst.read_text())["token_sha256"]
            want = json.loads((C.HERE / "windows" / f"{m}_{n}.json").read_text())["token_sha256"]
            ok &= say(got == want, f"{m} {n} windows: {len(got)} token hashes equal the paper's")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("env", "kernels", "sass", "hf", "fetch", "records", "windows"))
    ap.add_argument("--data-root", type=Path, default=None, help="records / windows: the calibration data root")
    ap.add_argument("--out", type=Path, default=Path("paper1k_runs"), help="windows: the run directory")
    ap.add_argument("--build", type=Path, default=None, help="kernels / sass: the build directory (default PAPER1K_BUILD "
                                                             "or kernels/sm120/build_deploy)")
    a = ap.parse_args(argv)
    build = a.build or C.build_dir()
    ok = {"env": env_check, "kernels": lambda: kernels(build), "sass": lambda: sass_gate(build), "hf": hf_check,
          "fetch": fetch_all, "records": lambda: records(a.data_root), "windows": lambda: windows(a.data_root, a.out)}[a.step]()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
