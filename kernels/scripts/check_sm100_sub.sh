#!/bin/bash
# Run within a one-GPU SM100 allocation. Checks the MMA-width experiment (mixed_sub_n*_sm100) against
# the host reference, with negative controls that must fail. Prints one PASS/FAIL line per width and
# writes the passing widths to $PASSED_FILE (default: none).
set -uo pipefail
BUILD="${BUILD:?set BUILD}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
# PREFIX=mixed_sub / mixed_subpad (2-SM, N=16..256) or mixed_sub1sm / mixed_subpad1sm (1-SM, N=8..256).
PREFIX="${PREFIX:-mixed_sub}"
WIDTHS="${WIDTHS:-16 32 64 128 256}"
passed=()
for w in $WIDTHS; do
  bin="$BUILD/${PREFIX}_n${w}_sm100"
  ok=1
  for shape in "512 512 512" "768 1536 1280"; do
    for fa in e2m1 e0m3; do
      for pattern in random checker e2m1 e0m3; do
        out=$("$bin" $shape --fmt-a=$fa --pattern=$pattern --iters=3 2>&1) || { ok=0; echo "$out" | tail -3; }
        echo "N=$w shape=[$shape] A=$fa map=$pattern: $(echo "$out" | grep -m1 Correctness)"
      done
    done
  done
  # Negative controls: each must FAIL, or the check above proves nothing.
  if [ "$w" -lt 256 ] && [[ "$PREFIX" != *1sm ]]; then  # 1-SM needs no B permutation
    if MIXFP4_SUB_NO_PERMUTE=1 "$bin" 512 512 512 --pattern=e2m1 --iters=1 >/dev/null 2>&1; then
      echo "N=$w negative control (no B permutation) unexpectedly PASSED"; ok=0
    else echo "N=$w negative control (no B permutation): failed as required"; fi
  fi
  for ref in e2m1 e0m3; do
    if MIXFP4_REF_FMT_B=$ref "$bin" 512 512 512 --pattern=random --iters=1 >/dev/null 2>&1; then
      echo "N=$w negative control (reference decodes all B as $ref) unexpectedly PASSED"; ok=0
    else echo "N=$w negative control (reference decodes all B as $ref): failed as required"; fi
  done
  if [ "$w" = 256 ]; then
    # N/K tails at the deployed granularity: N=48 (Qwen3.8-27B), K=4480 = 17.5 K-tiles and
    # N=720 = 2.8 tiles (Nemotron-9B-like), odd M.
    for shape in "256 48 4480" "130 720 4480" "200 4480 1344"; do
      for pattern in random checker; do
        out=$("$bin" $shape --pattern=$pattern --iters=3 2>&1) || { ok=0; echo "$out" | tail -3; }
        echo "N=$w tail shape=[$shape] map=$pattern: $(echo "$out" | grep -m1 Correctness)"
      done
    done
  fi
  if [ $ok = 1 ]; then echo "N=$w: PASS"; passed+=("$w"); else echo "N=$w: FAIL"; fi
done
[ -n "${PASSED_FILE:-}" ] && echo "${passed[*]}" > "$PASSED_FILE"
echo "passing widths: ${passed[*]}"
