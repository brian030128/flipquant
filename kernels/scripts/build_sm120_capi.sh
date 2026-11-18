#!/bin/bash
# Configure + build the sm_120a ctypes C APIs: libflipquant_sm120.so (weights as A, 16x64) and
# libflipquant_sm120_b8.so (weights on B, 8x64), both SASS-patched, and their test drivers.
# See capi/README.md.
#
# What the flipquant_sm120 target does:
#   1. generates the weights-as-A (16-row x 64-K granule) MMA blob header into the build tree;
#   2. compiles libflipquant_sm120_unpatched.so (sm_120a SASS only, no PTX);
#   3. runs scripts/patch_mixed_nvfp4_gemm.py on the .so: requires exactly format sites {0,1}
#      with equal OMMA counts, installs E0M3 on site 1, re-disassembles to verify, and flips the
#      library's patch marker. Any mismatch fails the build and leaves no libflipquant_sm120.so.
#
# Only nvcc is needed to build (no GPU). Running needs a CC 12.0 GPU.
#   BUILD=/tmp/fq120 JOBS=16 scripts/build_sm120_capi.sh
set -euo pipefail

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD="${BUILD:-$SRC/build-capi-sm120-$(uname -m)}"
JOBS="${JOBS:-$(nproc)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TARGETS="${TARGETS:-flipquant_sm120 test_capi_sm120 flipquant_sm120_b8 test_capi_sm120_b8}"

mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DMIXFP4_ARCH=120a \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_HEADERS_ONLY=ON

# shellcheck disable=SC2086
cmake --build "$BUILD" --target $TARGETS -j "$JOBS"
echo "built into $BUILD"
echo "C API: $BUILD/libflipquant_sm120.so   (do NOT load libflipquant_sm120_unpatched.so)"
