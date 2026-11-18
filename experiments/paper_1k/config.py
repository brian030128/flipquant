"""The paper's reruns at the 1,024-window calibration (results/paper_1k): the settings every item shares.

Everything here is fixed by the registrations the reruns follow (NVFP4-RaZeR paper-eval PROTOCOL.md, part AC and the
parts each item reruns); the per-item modules only add their own settings. Nothing points outside this repository:
interpreters come from the environment (PAPER1K_PY / PAPER1K_PY_FAST), kernels from this repository's build, artifacts
from the public Hugging Face repositories pinned in artifacts.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # the flipquant checkout
HERE = Path(__file__).resolve().parent

#: the six paper models, in the paper's order (registry keys)
MODELS = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b", "nemotron-nano-9b-v2", "phi4-14b", "qwen3.8-27b"]
TITLE = {"llama3.2-1b-instruct": "Llama-3.2-1B-Instruct", "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
         "mistral-7b": "Mistral-7B-Instruct-v0.3", "nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2", "phi4-14b": "Phi-4",
         "qwen3.8-27b": "Qwen3.8-27B"}
#: the models whose recurrent layers need the fast kernels (mamba-ssm, causal-conv1d, fla): the n16k64-fast environment
HYBRID = ("nemotron-nano-9b-v2", "qwen3.8-27b")
UNITS = ("8x64", "16x64", "256x64")
#: the trainer's model keys (calibration.tmopt_common.resolve)
TRAINER_KEY = {"llama3.2-1b-instruct": "llama1b_ins", "llama3.1-8b-instruct": "llama8b_ins", "mistral-7b": "mistral7b_ins",
               "nemotron-nano-9b-v2": "nemotron9b", "phi4-14b": "phi4", "qwen3.8-27b": "qwen27b"}
#: the release / Calib1k repositories' base names (edgeai-lab/<base>-FlipQuant[-Calib1k], ...)
HF_BASE = {"llama3.2-1b-instruct": "Llama-3.2-1B-Instruct", "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
           "mistral-7b": "Mistral-7B-Instruct-v0.3", "nemotron-nano-9b-v2": "NVIDIA-Nemotron-Nano-9B-v2",
           "phi4-14b": "phi-4", "qwen3.8-27b": "Qwen3.8-27B"}
HF_ORG = "edgeai-lab"
#: the models' revisions as the commands pass them (--revision; the same as models/registry.py's pins)
REV = {"llama3.2-1b-instruct": "9213176726f574b556790deb65791e0c5aa438b6",
       "llama3.1-8b-instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
       "mistral-7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",
       "nemotron-nano-9b-v2": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
       "phi4-14b": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
       "qwen3.8-27b": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}

# ------------------------------------------------------------------------------------- the native (sm120) policies
SHAPES = "1x128,1x256,1x512,1x1024,1x2048,1x4096,1x8192,4x2048"
SEED = 20260928                                     # part A: round r shuffles the policy list with SEED + r
#: run_gpu.POLICIES of the paper-eval runs; "{BUILD}" is the kernel build directory, "@<unit>" a map of that unit
NATIVE = ["--mode", "native", "--kernel-build-dir", "{BUILD}"]
POLICIES = {"bf16": ["--mode", "bf16"],
            "nvfp4": NATIVE + ["--weight", "nvfp4", "--act", "nvfp4"],
            "fo6": NATIVE + ["--weight", "fourover6", "--act", "fourover6"],
            "fo6-wB": NATIVE + ["--weight", "fourover6", "--act", "fourover6", "--kernel-set", "auto_stock_wB"],
            **{f"fq-{u}": NATIVE + ["--weight", "mixfp4", "--act", "fourover6", "--map", f"@{u}"] for u in UNITS}}
#: the kernel family each native policy must install (run_gpu.FAMILY)
FAMILY = {"nvfp4": "stock_ko", "fo6": "stock_ko", "fo6-wB": "stock_wB_ko", "fq-8x64": "mixed_wB_ko",
          "fq-16x64": "mixed_ko", "fq-256x64": "mixed256_ko"}
#: the environment variables the runs drop (run_gpu.ENV_DROP)
ENV_DROP = ("PYTHONPATH", "SM120_BUILD_DIR", "PYTORCH_CUDA_ALLOC_CONF")


def py(model: str) -> str:
    """The interpreter of a model's environment: n16k64-fast for the hybrid models, n16k64 otherwise."""
    if model in HYBRID:
        return os.environ.get("PAPER1K_PY_FAST") or os.environ.get("PAPER1K_PY") or sys.executable
    return os.environ.get("PAPER1K_PY") or sys.executable


def py_release() -> str:
    """The n16k64 interpreter for every model: the env of the release and 1k map trainings (their records), which the
    trainings with GPTQ candidates and the calibration-cost runs keep."""
    return os.environ.get("PAPER1K_PY") or sys.executable


def build_dir() -> Path:
    """The SM120 kernel build directory (kernels/sm120/build_deploy.sh's output; $PAPER1K_BUILD overrides)."""
    return Path(os.environ.get("PAPER1K_BUILD") or REPO / "kernels" / "sm120" / "build_deploy")


def policy_args(pol: str, maps: dict | None = None, build: Path | None = None) -> list[str]:
    """POLICIES[pol] with the build directory and the unit's map path filled in."""
    out = []
    for a in POLICIES[pol]:
        if a == "{BUILD}":
            out.append(str(build or build_dir()))
        elif a.startswith("@"):
            out.append(str(maps[a[1:]]))
        else:
            out.append(a)
    return out
