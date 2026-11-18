#!/bin/bash
# The deployment directory of the sm_120 native path: the optimized kernel sets with their dispatch variants, plus the
# paper's builds, so that --kernel-set auto (the default) and --kernel-set paper both run from it. models/sm120.py reads
# it by default (kernels/sm120/build_deploy; $SM120_BUILD_DIR or --kernel-build-dir point elsewhere).
# CPU only; one build.py process per configuration, in parallel.
#
#     CUDA_HOME=/path/to/cuda-13.1 bash kernels/sm120/build_deploy.sh [DIR]     # default kernels/sm120/build_deploy
#     CUTLASS_DIR=/path/to/cutlass ...  # a CUTLASS checkout at the pinned commit instead of kernels/3rdparty/cutlass
#
# Every build computes the same outputs bit for bit as the build of the same configuration without defines; the defines
# select the optimized dispatch variants (the frequency-aware dispatch, the uniform-branch dispatch, the pipelined flag
# read). tests/test_sm120_native.py checks each build's SASS against the reference builds (SASS_REFERENCE.json).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-$HERE/build_deploy}
PY=${PYTHON:-python}
# build.py (vendored) reads the CUTLASS path from its own variable; CUTLASS_DIR is this script's name for it
if [ -n "${CUTLASS_DIR:-}" ]; then export RAZER_CUTLASS=$CUTLASS_DIR; fi
F="--define MIXFP4_DISPATCH_FREQ=1"; U="--define MIXFP4_UNIFORM_DISPATCH=1"; P="--define MIXFP4_PIPE_FLAGS=1"
mkdir -p "$D/logs"
pids=()
b() { local c=$1; shift; SM120_BUILD_DIR=$D $PY "$HERE/build.py" "$@" --config $c > "$D/logs/$c.log" 2>&1 & pids+=($!); }
# 16x64 'mixed_ko': the frequency-aware dispatch; the uniform-branch dispatch at widths 64 and 128
for c in n16k64_wA_n16_t0 n16k64_wA_n32_t0; do b $c $F; done
for c in n16k64_wA_n64_t0 n16k64_wA_e64_t0; do b $c $F $U; done
# 8x64 'mixed_wB_ko': the frequency-aware dispatch; the pipelined flag read at widths 64, '128x64' and 128
for c in n8k64_wB_m16_t0 n8k64_wB_m32_t0; do b $c $F; done
for c in n8k64_wB_m64_t0 n8k64_wB_n64_t0 n8k64_wB_t0; do b $c $F $P; done
# 256x64 'mixed256_ko': the uniform-branch dispatch at every width
for c in n16k64_wA_g32_n16_t0 n16k64_wA_g32_n32_t0 n16k64_wA_g32_n64_t0 n16k64_wA_g32_e64_t0; do b $c $U; done
# no defines: 'mixed256', 'stock_ko', 'stock_wB_ko', and the paper's sets 'mixed', 'stock', n8k64_wB
for c in n16k64_wA_g32 n16k64_wA_g32_n64 n16k64_wA_g32_n32 n16k64_wA_g32_n16 stock_wA_n16 stock_wA_n32 stock_wA_n64 \
         stock_wA_e64 stock_wB_e64 n16k64_wA n16k64_wA_n16 n16k64_wA_n32 n16k64_wA_n64 stock_wA n8k64_wB stock_wB; do b $c; done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
grep -h '^\[' "$D"/logs/*.log
[ $rc -eq 0 ] || { echo "build_deploy: a build failed (logs in $D/logs)"; exit 1; }
echo "build_deploy: $(ls "$D"/*/manifest.json | wc -l) builds in $D"
