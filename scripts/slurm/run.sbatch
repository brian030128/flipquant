#!/bin/bash
#SBATCH --job-name=flipquant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=190G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#SBATCH --error=scripts/slurm/logs/%x_%j.err
# Generic runner: sbatch [slurm opts] scripts/slurm/run.sbatch <python -m module> [args...]
#
# Pick the hardware on the command line (the cluster's partition names differ per GPU type):
#   H100/H200 (calibration; fake-quant evaluation):
#     sbatch -A <acct> -p <partition> --gres=gpu:1 scripts/slurm/run.sbatch \
#         calibration.train_map --model mistral-7b --unit 16x64 --fit-windows 1024 --epochs 5 --teacher-topk 1000 \
#         --out runs/mistral-7b_16x64
#   B200 / GB200 (sm_100), native:
#     sbatch -A <acct> -p <b200-partition> --gres=gpu:1 scripts/slurm/run.sbatch \
#         evaluation.ppl --model mistral-7b --mode native --native-backend sm100-tuned --weight mixfp4 \
#         --map runs/mistral-7b_256x64/map.pt --paper-convention --out results/mistral-7b/native_ppl.json
#   RTX Pro 6000 (sm_120), native: same with the Pro 6000 partition.
#
# Environment (override by exporting before sbatch):
#   PYTHON          interpreter with torch>=2.9, transformers>=5.16, datasets (see requirements.txt);
#                   unset: a uv venv with the tested versions (Python 3.11, torch 2.9.0+cu128,
#                   transformers 5.16.1) is built in node-local /tmp (a few minutes; compute nodes reach PyPI)
#   HF_HOME         HuggingFace cache
#   FLIPQUANT_CACHE tokenized eval/calibration windows
#   FLIPQUANT_KERNEL_LIB  directory holding libflipquant_sm100.so / libflipquant_sm120.so (native mode on sm_100, and
#                   --native-backend sm120-capi)
#   SM120_BUILD_DIR native mode on sm_120 (the default sm120 path): the build directory of kernels/sm120/build_deploy.sh
#                   (default kernels/sm120/build_deploy)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run through sbatch}"
export HF_HOME="${HF_HOME:-/work/$USER/hf}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export TMPDIR="/tmp/$USER/flipquant_$SLURM_JOB_ID/tmp" PYTHONPYCACHEPREFIX="/tmp/$USER/flipquant_$SLURM_JOB_ID/pycache"
mkdir -p "$TMPDIR" scripts/slurm/logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || {
  echo "no usable GPU on $(hostname) -- resubmit with --exclude=$(hostname)" >&2; exit 3; }
if [[ -z "${PYTHON:-}" ]]; then
  ENV="/tmp/$USER/flipquant_$SLURM_JOB_ID/env"
  export UV_CACHE_DIR="$ENV/uv-cache" UV_PYTHON_INSTALL_DIR="$ENV/uv-python"
  /usr/bin/python3 -m venv "$ENV/boot" && "$ENV/boot/bin/pip" -q install uv
  "$ENV/boot/bin/uv" venv -q -p 3.11 "$ENV/venv"
  # FLIPQUANT_EXTRA_PIP: further packages, e.g. "nvidia-modelopt==0.47.0" for calibration.gptq_modelopt
  "$ENV/boot/bin/uv" pip install -q -p "$ENV/venv/bin/python" --torch-backend=cu128 torch==2.9.0 \
      transformers==5.16.1 "datasets>=4.4" "accelerate>=1.7" sentencepiece protobuf packaging pytest \
      ${FLIPQUANT_EXTRA_PIP:-}
  PYTHON="$ENV/venv/bin/python"
fi
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
MODULE="${1:?module, e.g. evaluation.ppl}"; shift
exec "$PYTHON" -m "$MODULE" "$@"
