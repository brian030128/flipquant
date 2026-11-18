#!/bin/bash
# HackMD #2 / Table 5 on one GB200 GPU: the 5-round prefill protocol (kernels/scripts/prefill_rounds_sm100.py) for the
# models given as arguments (default: all six; split them over jobs to stay within gb200-dev's 2 hours).
#   sbatch kernels/scripts/prefill_rounds_gb200.sbatch qwen3.8-27b          # from the repository root
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -J fq-prefill
#SBATCH -o kernels/docs/results/sm100_paper/02_prefill_latency/prefill_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=${PREFILL_OUT:-kernels/docs/results/sm100_paper/02_prefill_latency}     # QUANT=fast: the fast quantizer
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted models=${*:-all}"
nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm,power.limit --format=csv
df -h /tmp | tail -1
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
SRC="$PWD/kernels" JOBS=12 TARGETS=flipquant_sm100_linear bash kernels/scripts/build_sm100.sh 2>&1 \
  | grep -E 'error|Error|built into' | head -20
export FLIPQUANT_SM100_LINEAR_LIB="$BUILD/libflipquant_sm100_linear.so"
# Nemotron's Mamba-2, the causal convolutions and Qwen3.8's Gated DeltaNet on their fast paths (Appendix C.1).
export FLA_PROBE_OUT="$OUT/fla_probe_$SLURM_JOB_ID.json"
source kernels/scripts/sm100_fast_env.sh
export PYTHONPATH="$PWD"
if [ $# -gt 0 ]; then models=(--models "$@"); else models=(); fi
"$PYTHON" kernels/scripts/prefill_rounds_sm100.py --out "$OUT" --quant "${QUANT:-exact}" "${models[@]}"
rm -rf "$BUILD"
echo "prefill rounds done"
