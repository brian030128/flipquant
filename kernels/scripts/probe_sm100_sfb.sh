#!/bin/bash
# Run within a one-GPU SM100 allocation. Asks the hardware which scale-factor TMEM placements it
# accepts for narrow block-scaled MMAs: each probe binary deliberately uses a wrong-but-legal-looking
# SFB base or SFB copy offset, so numerics are expected to be wrong; only "faulted" vs "ran" matters.
set -uo pipefail
BUILD="${BUILD:?set BUILD}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
probe() {  # name shape description
  local name=$1 shape=$2 what=$3 out verdict
  if [ ! -x "$BUILD/$name" ]; then printf '%-30s %-62s %s\n' "$name" "$what" "NOT BUILT"; return; fi
  out=$("$BUILD/$name" $shape --pattern=random --iters=1 2>&1)
  if echo "$out" | grep -qiE 'misaligned|illegal|error'; then
    verdict="FAULT  ($(echo "$out" | grep -m1 -iE 'misaligned|illegal|error' | sed 's/.*: //'))"
  else
    verdict="ran    ($(echo "$out" | grep -m1 -oE 'PASSED|FAILED' || echo no check))"
  fi
  printf '%-30s %-62s %s\n' "$name" "$what" "$verdict"
}
probe probe_n32_sfb2_sm100       "512 512 512" "2-SM N=32 MMA, SFB base at even columns"
probe probe_n32_sfb4_sm100       "512 512 512" "2-SM N=32 MMA, SFB base at multiples of 4 columns"
probe probe_n16_sfb2_sm100       "512 512 512" "2-SM N=16 MMA, SFB base at even columns, lane 0"
probe probe_n16_sfb2_lane_sm100  "512 512 512" "2-SM N=16 MMA, SFB base at even columns, lane 0/16"
probe probe_cp_dst1_sm100        "512 512 512" "2-SM SFB copy to TMEM column +1"
probe probe_cp_dst2_sm100        "512 512 512" "2-SM SFB copy to TMEM column +2"
probe probe_cp_src16_sm100       "512 512 512" "2-SM SFB copy from smem +256 B (16 lanes)"
probe probe1sm_n8_sfb2_sm100     "512 512 512" "1-SM N=8 MMA, SFB base at even columns, lane 0"
probe probe1sm_n8_sfb4_sm100     "512 512 512" "1-SM N=8 MMA, SFB base at multiples of 4, lane 0"
probe probe1sm_n8_sfb4_lane_sm100 "512 512 512" "1-SM N=8 MMA, SFB base at multiples of 4, lane 0/8/16/24"
echo "probes done"
