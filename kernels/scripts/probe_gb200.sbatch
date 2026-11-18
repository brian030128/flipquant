#!/bin/bash
# Scale-factor TMEM probes plus the 1-SM width sweep (N=8..256), on one GB200 GPU.
#   sbatch kernels/scripts/probe_gb200.sbatch     # from the repository root
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -J fq-sfb-probe
#SBATCH -o kernels/docs/results/sm100_paper/05_operand_width/probes/sm100_probe_%j.log
set -euo pipefail
KERNELS="${KERNELS:-$SLURM_SUBMIT_DIR/kernels}"
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
OUT="$KERNELS/docs/results"
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git -C "$KERNELS" rev-parse --short HEAD) $(git -C "$KERNELS" status --porcelain | wc -l) uncommitted"
PROBES="probe_n32_sfb2_sm100 probe_n32_sfb4_sm100 probe_n16_sfb2_sm100 probe_n16_sfb2_lane_sm100 probe_cp_dst1_sm100 probe_cp_dst2_sm100 probe_cp_src16_sm100 probe1sm_n8_sfb2_sm100 probe1sm_n8_sfb4_sm100 probe1sm_n8_sfb4_lane_sm100"
ONESM="mixed_sub1sm_n8_sm100 mixed_sub1sm_n16_sm100 mixed_sub1sm_n32_sm100 mixed_sub1sm_n64_sm100 mixed_sub1sm_n128_sm100 mixed_sub1sm_n256_sm100"
SRC="$KERNELS" JOBS=12 TARGETS="mixed_nvfp4_gemm_sm100 $PROBES $ONESM" \
  bash "$KERNELS/scripts/build_sm100.sh" -- -k 2>&1 | grep -vE '^\[ *[0-9]+%\] Building' | grep -E 'error|Error|built into|Linking' | head -60 || true
bash "$KERNELS/scripts/probe_sm100_sfb.sh" | tee "$OUT/sm100_paper/05_operand_width/probes/sm100_probe.txt"
PREFIX=mixed_sub1sm WIDTHS="8 16 32 64 128 256" PASSED_FILE="$BUILD/passed1sm" \
  bash "$KERNELS/scripts/check_sm100_sub.sh" | tee "$OUT/sm100_paper/05_operand_width/probes/sm100_sub1sm_check.txt"
widths=$(cat "$BUILD/passed1sm")
if [ -n "$widths" ]; then
  # shellcheck disable=SC2086
  python3 "$KERNELS/scripts/benchmark_sm100_sub.py" --build "$BUILD" --family "mixed_sub1sm:$(echo $widths | tr " " ",")" \
    --sizes 8192 --output "$OUT/sm100_paper/05_operand_width/probes/sm100_sub1sm_launch.json"
fi
rm -rf "$BUILD"
echo "probe job done"
