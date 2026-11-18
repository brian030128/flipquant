#!/bin/bash
# Run on one allocated sm_100 GPU. References cover tile tails, scale padding,
# all four codebooks, and a deliberately incorrect reference codebook.
set -euo pipefail
BUILD="${BUILD:-/work/u4320956/b200/build-$(uname -m)${SLURM_JOB_ID:+-$SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for cfg in 128x64x256_1sm_fast 128x128x256_1sm_fast \
           256x128x256_2sm_fast 256x256x256_2sm_fast \
           128x64x256_1sm_static 256x128x256_2sm_static; do
  for shape in "144 272 288" "384 640 768"; do
    read -r m n k <<< "$shape"
    "$BUILD/nvfp4_gemm_sm100" "$m" "$n" "$k" --config="$cfg" --check --iters=5
  done
done

for fa in e2m1 e0m3; do
  for fb in e2m1 e0m3; do
    "$BUILD/mixed_nvfp4_gemm_sm100" 384 640 768 --fmt-a="$fa" --fmt-b="$fb" --check --iters=5
  done
done

for operand in A B; do
  if output=$(env "MIXFP4_REF_FMT_$operand=e0m3" "$BUILD/mixed_nvfp4_gemm_sm100" \
      384 640 768 --fmt-a=e2m1 --fmt-b=e2m1 --check --iters=5 2>&1); then
    echo "ERROR: wrong-codebook negative control unexpectedly passed" >&2
    exit 1
  fi
  # An unrelated CUDA/launch error must not count as a successful negative control.
  [[ "$output" == *"Correctness (A=E2M1 B=E2M1): FAILED"* ]]
  echo "$output"
done
echo "SM100 correctness and negative controls passed."
