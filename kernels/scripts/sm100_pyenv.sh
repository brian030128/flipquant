#!/bin/bash
# Source inside a GB200 (aarch64) job: builds the repository's Python stack in node-local /tmp and exports PYTHON.
# torch 2.9.0 (cu130 aarch64 wheel), transformers 5.16.1, Triton 3.5 -- the versions of scripts/slurm/run.sbatch,
# on a uv-managed CPython 3.11 (the node's /usr python3.11 lacks the headers Triton compiles against).
#   SM100_TORCH_BACKEND   cu130 (default) or cu128
#   FLIPQUANT_EXTRA_PIP   further packages
set -euo pipefail
PYENV_ROOT="/tmp/$USER/fq_pyenv_$SLURM_JOB_ID"
mkdir -p "$PYENV_ROOT/tmp"
export UV_CACHE_DIR="$PYENV_ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$PYENV_ROOT/uv-python" TMPDIR="$PYENV_ROOT/tmp"
export HF_HOME="${HF_HOME:-/tmp/$USER/hf_$SLURM_JOB_ID}"       # node-local: the /work quota can be full
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PYENV_ROOT/triton" PYTHONPYCACHEPREFIX="$PYENV_ROOT/pycache"
/usr/bin/python3 -m venv "$PYENV_ROOT/boot"
"$PYENV_ROOT/boot/bin/pip" -q install uv
"$PYENV_ROOT/boot/bin/uv" venv -q -p 3.11 --python-preference only-managed "$PYENV_ROOT/venv"
"$PYENV_ROOT/boot/bin/uv" pip install -q -p "$PYENV_ROOT/venv/bin/python" \
  --torch-backend="${SM100_TORCH_BACKEND:-cu130}" torch==2.9.0 transformers==5.16.1 "datasets>=4.4" \
  "accelerate>=1.7" sentencepiece protobuf packaging ${FLIPQUANT_EXTRA_PIP:-}
export PYTHON="$PYENV_ROOT/venv/bin/python"
"$PYTHON" -c "import torch, transformers, triton; print('torch', torch.__version__, 'transformers', transformers.__version__, 'triton', triton.__version__)"
