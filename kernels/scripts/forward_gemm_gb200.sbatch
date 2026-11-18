#!/bin/bash
# HackMD #7 + #1 on one GB200 GPU: check the 256x64 per-tile kernel on N/K tails, then time the
# per-forward GEMMs of every model with its release 256x64 map against the stock kernel (paper D.5).
#   sbatch kernels/scripts/forward_gemm_gb200.sbatch     # from the repository root
#   TAILS=0 sbatch kernels/scripts/forward_gemm_gb200.sbatch --models llama3.2-1b-instruct --merge   # add one model
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J fq-forward-gemm
#SBATCH -o kernels/docs/results/sm100_paper/01_gemm_overhead/sm100_forward_%j.log
set -euo pipefail
KERNELS="${KERNELS:-$SLURM_SUBMIT_DIR/kernels}"
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
OUT="$KERNELS/docs/results"
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git -C "$KERNELS" rev-parse --short HEAD) $(git -C "$KERNELS" status --porcelain | wc -l) uncommitted"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
targets="mixed_nvfp4_gemm_sm100 mixed_sub_n256_sm100"
SRC="$KERNELS" JOBS=12 TARGETS="$targets" bash "$KERNELS/scripts/build_sm100.sh" 2>&1 \
  | grep -E 'error|Error|built into|CUPTI|cupti' | head -40
grep -E 'CUDA_cupti_LIBRARY' "$BUILD/CMakeCache.txt" || true
# #7: exact reference on 256-aligned shapes and on N/K tails (N=48, K=4480, partial tiles).
if [ "${TAILS:-1}" = 1 ]; then
PREFIX=mixed_sub WIDTHS=256 PASSED_FILE="$BUILD/passed" bash "$KERNELS/scripts/check_sm100_sub.sh" \
  | tee "$OUT/sm100_paper/07_tails/sm100_tail_check.txt"
grep -q 256 "$BUILD/passed"
fi
# #1: per-forward GEMM time, stock vs per-tile with the median-E0M3 and most-E0M3 tags.
python3 "$KERNELS/scripts/forward_gemm_sm100.py" --build "$BUILD" --work "$OUT/sm100_paper/01_gemm_overhead/work" \
  --output "$OUT/sm100_paper/01_gemm_overhead/forward_gemm.json" "$@"
rm -rf "$BUILD"
echo "forward gemm done"
