#!/bin/bash
# HackMD round 2 #3 + #4 (paper D.6, D.7) on one GB200 GPU: native (per-tile kernel) vs simulated perplexity of the
# release 256x64 maps, with the native install's ownership check (kernels/scripts/ppl_native_vs_sim_sm100.py), for the
# models given as arguments (default: all six; split them over jobs to stay within gb200-dev's 2 hours).
#   sbatch kernels/scripts/ppl_native_vs_sim_gb200.sbatch phi-4 qwen3.8-27b       # from the repository root
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -J fq-ppl-native-sim
#SBATCH -o kernels/docs/results/sm100_paper/03_native_vs_simulated/ppl_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=kernels/docs/results/sm100_paper/03_native_vs_simulated
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted models=${*:-all}"
nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm,power.limit --format=csv
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
SRC="$PWD/kernels" JOBS=12 TARGETS=flipquant_sm100_linear bash kernels/scripts/build_sm100.sh 2>&1 \
  | grep -E 'error|Error|built into' | head -20
export FLIPQUANT_SM100_LINEAR_LIB="$BUILD/libflipquant_sm100_linear.so"
export FLA_PROBE_OUT="$OUT/fla_probe_$SLURM_JOB_ID.json"
source kernels/scripts/sm100_fast_env.sh
export PYTHONPATH="$PWD"
if [ $# -gt 0 ]; then models=(--models "$@"); else models=(); fi
"$PYTHON" kernels/scripts/ppl_native_vs_sim_sm100.py --out "$OUT" --runs ${RUNS:-native simulated} "${models[@]}"
rm -rf "$BUILD"
echo "ppl native vs simulated done"
