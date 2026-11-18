#!/bin/bash
set -euo pipefail
BUILD="${BUILD:-/work/u4320956/b200/build-$(uname -m)${SLURM_JOB_ID:+-$SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
for cfg in "16 64" "16 128" "32 64" "32 128" "32 256" \
           "64 64" "64 128" "64 256" "128 64" "128 128"; do
  read -r tn tk <<< "$cfg"
  for fa in e2m1 e0m3; do
    for pattern in checker random; do
      "$BUILD/mixed_n8k64_sm100" 144 136 320 --packed-inputs \
        --tile-n="$tn" --tile-k="$tk" --fmt-a="$fa" --pattern="$pattern" --iters=3
    done
  done
  "$BUILD/mixed_n8k64_sm100" 144 136 320 --pipeline \
    --tile-n="$tn" --tile-k="$tk" --pattern=random --iters=3
done
"$BUILD/mixed_n8k64_sm100" 257 264 1024 --packed-inputs --pattern=random --iters=3
echo "N8K64 pipeline correctness passed."
