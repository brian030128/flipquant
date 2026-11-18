#!/bin/bash
# Paper Table D.6, every MMA width: padded-SFB GEMMs (2-SM N=16..256, 1-SM N=8..256) next to the
# tuned kernel and its unpadded sub-MMA variants. Builds, checks, then benchmarks what passes.
#   sbatch kernels/scripts/pad_sweep_gb200.sbatch     # from the repository root
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J fq-pad-sweep
#SBATCH -o kernels/docs/results/sm100_paper/05_operand_width/sm100_pad_%j.log
set -euo pipefail
KERNELS="${KERNELS:-$SLURM_SUBMIT_DIR/kernels}"
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
OUT="$KERNELS/docs/results"
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git -C "$KERNELS" rev-parse --short HEAD) $(git -C "$KERNELS" status --porcelain | wc -l) uncommitted"
targets="mixed_nvfp4_gemm_sm100"
for w in 64 128 256; do targets+=" mixed_sub_n${w}_sm100"; done
for w in 16 32 64 128 256; do targets+=" mixed_subpad_n${w}_sm100 mixed_subpadlo_n${w}_sm100"; done
for w in 8 16 32 64 128 256; do targets+=" mixed_subpad1sm_n${w}_sm100 mixed_subpadlo1sm_n${w}_sm100"; done
SRC="$KERNELS" JOBS=12 TARGETS="$targets" bash "$KERNELS/scripts/build_sm100.sh" -- -k 2>&1 \
  | grep -E 'error|Error|built into|Linking' | head -80 || true
for bin in $targets; do [ -x "$BUILD/$bin" ] || echo "NOT BUILT: $bin"; done
# Use the builder's stage count where the extra scale factors still fit, else one stage fewer.
pad2=mixed_subpad; [ -x "$BUILD/mixed_subpad_n256_sm100" ] || pad2=mixed_subpadlo
pad1=mixed_subpad1sm; [ -x "$BUILD/mixed_subpad1sm_n256_sm100" ] || pad1=mixed_subpadlo1sm
echo "padded families: $pad2 $pad1"
PREFIX=$pad2 WIDTHS="16 32 64 128 256" PASSED_FILE="$BUILD/passed_pad" \
  bash "$KERNELS/scripts/check_sm100_sub.sh" | tee "$OUT/sm100_paper/05_operand_width/sm100_pad_check.txt"
PREFIX=$pad1 WIDTHS="8 16 32 64 128 256" PASSED_FILE="$BUILD/passed_pad1sm" \
  bash "$KERNELS/scripts/check_sm100_sub.sh" | tee "$OUT/sm100_paper/05_operand_width/sm100_pad1sm_check.txt"
for bin in mixed_nvfp4_gemm_sm100 ${pad2}_n256_sm100 ${pad1}_n256_sm100; do
  [ -x "$BUILD/$bin" ] && "$BUILD/$bin" 8192 8192 8192 --no-check --iters=5 | grep -E 'stages|Throughput' | sed "s/^/$bin: /"
done
w2=$(tr ' ' ',' < "$BUILD/passed_pad"); w1=$(tr ' ' ',' < "$BUILD/passed_pad1sm")
python3 "$KERNELS/scripts/benchmark_sm100_sub.py" --build "$BUILD" --sizes 8192 4096 \
  --family mixed_sub:64,128,256 --family "$pad2:$w2" --family "$pad1:$w1" \
  --output "$OUT/sm100_paper/05_operand_width/sm100_pad_launch.json"
rm -rf "$BUILD"
echo "pad sweep done"
