"""Acceptance tests of the trainer's candidates hook (``calibration.tmopt_ext train --candidates-e2m1 / --candidates-e0m3``,
calibration/tmopt_launch.py; NVFP4-RaZeR paper-eval part O, test (b)): GPU, Qwen3-1.7B at 16x64, 2 epochs with the
release settings otherwise, the calibration records under ``$FLIPQUANT_TMOPT_DATA`` (the release runs' data root).

Run with ``FLIPQUANT_ACCEPTANCE=1`` (the module sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8``); outputs go to
``$FLIPQUANT_ACCEPTANCE_DIR`` (default: a pytest temporary directory) and ``summary_candidates.json`` there.

(g)  ``--candidates-* rtn`` (flipquant's RTN codes through the hook) gives calibration.train_map's map.pt bit for bit;
(g2) RTN codes FILES (flipquant-gptq/1, FourOverSix and all-E0M3 mixfp4 codes of every module) give it bit for bit too.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ON = os.environ.get("FLIPQUANT_ACCEPTANCE") == "1"
pytestmark = pytest.mark.skipif(not ON, reason="acceptance tests: set FLIPQUANT_ACCEPTANCE=1 (GPU, Qwen3-1.7B, its "
                                               "calibration record under $FLIPQUANT_TMOPT_DATA)")
MODEL, UNIT, EPOCHS = "qwen3-1.7b", "16x64", 2
DATA = os.environ.get("FLIPQUANT_TMOPT_DATA", "/home/dev/n16k64_campaign/fqrel/tmopt_data")
SUMMARY = {}


def _run(args, log, cwd=REPO):
    env = dict(os.environ, PYTHONPATH=str(cwd) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, *map(str, args)], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert rc == 0, f"{args[:3]} failed, see {log}"


def _sha(p):
    from calibration.tmopt_common import sha256
    return sha256(p)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = Path(os.environ.get("FLIPQUANT_ACCEPTANCE_DIR") or tmp_path_factory.mktemp("acceptance_candidates"))
    d.mkdir(parents=True, exist_ok=True)
    yield d
    prev = json.loads((d / "summary_candidates.json").read_text()) if (d / "summary_candidates.json").exists() else {}
    (d / "summary_candidates.json").write_text(json.dumps(dict(prev, **SUMMARY), indent=1) + "\n")


@pytest.fixture(scope="module")
def direct(work):
    out = work / "direct"
    if not (out / "run" / "map.pt").exists():
        import shutil
        shutil.rmtree(out, ignore_errors=True)
        _run(["-m", "calibration.train_map", "--model", MODEL, "--unit", UNIT, "--epochs", EPOCHS, "--fit-windows", 256,
              "--teacher-topk", 1000, "--data-root", DATA, "--out", out], work / "direct.log")
    return out / "run" / "map.pt"


def _hooked(work, name, e2m1, e0m3):
    out = work / f"{name}.pt"
    if not out.with_suffix(".json").exists():
        _run(["-m", "calibration.tmopt_ext", "train", "--model", MODEL, "--unit", UNIT, "--epochs", EPOCHS,
              "--data-root", DATA, "--candidates-e2m1", e2m1, "--candidates-e0m3", e0m3, "--out", out],
             work / f"{name}.log")
    return out


RTN_FILES = r"""
import sys, torch
from pathlib import Path
from flipquant import gptq, maps
from models.cli import load
from models.loader import quantizable_linears
model, tok, spec = load(sys.argv[1], device_map="cuda")
lin = quantizable_linears(model, spec)
b, a = {}, {}
for n, m in lin.items():
    w = m.weight.detach()
    b[n] = gptq._packed(gptq.rtn_grid(w, "fourover6")[1])
    ones = torch.ones(maps.tile_shape(*w.shape, "16x64"), dtype=torch.bool)
    a[n] = gptq._packed(gptq.rtn_grid(w, "mixfp4", ones, "16x64")[1])
gptq._save(Path(sys.argv[2]), b, dict(model=spec.key, weight="fourover6", ptq="rtn"))
gptq._save(Path(sys.argv[3]), a, dict(model=spec.key, weight="mixfp4", ptq="rtn", map="all-E0M3"))
print("RTN FILES", len(lin))
"""


def test_g_rtn_is_the_trainer(work, direct):
    out = _hooked(work, "g_rtn", "rtn", "rtn")
    a, b = _sha(direct), _sha(out.with_suffix(".run") / "run" / "map.pt")
    hook = json.loads((out.with_suffix(".run") / "run" / "candidates_hook.json").read_text())
    SUMMARY["g_rtn"] = dict(direct_map_sha256=a, hooked_map_sha256=b, equal=a == b, modules=hook["modules"],
                            lean_store=hook["lean_store"], other_calls=hook["other_calls"])
    assert a == b and hook["lean_store"] and not hook["missing_or_repeated"]


def test_g2_rtn_files_are_the_trainer(work, direct):
    fb, fa = work / "rtn_e2m1.pt", work / "rtn_e0m3.pt"
    if not (fb.exists() and fa.exists()):
        _run(["-c", RTN_FILES, MODEL, fb, fa], work / "rtn_files.log")
    out = _hooked(work, "g2_files", fb, fa)
    a, b = _sha(direct), _sha(out.with_suffix(".run") / "run" / "map.pt")
    hook = json.loads((out.with_suffix(".run") / "run" / "candidates_hook.json").read_text())
    SUMMARY["g2_files"] = dict(direct_map_sha256=a, hooked_map_sha256=b, equal=a == b,
                               mean_codes_changed_vs_rtn=hook["mean_codes_changed_vs_rtn"])
    assert a == b and hook["mean_codes_changed_vs_rtn"] == {"e2m1": 0.0, "e0m3": 0.0}
