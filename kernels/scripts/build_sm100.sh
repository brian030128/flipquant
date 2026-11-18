#!/bin/bash
# Configure + build the datacenter-Blackwell (sm_100a) targets, including the ctypes C API
# (libflipquant_sm100.so + test_capi_sm100; see capi/README.md).
#
# The build host's CPU architecture must match the machine that will load the .so: GB200 nodes
# are Grace (aarch64), x86 B200 nodes are x86_64, and the host half of every object file is
# cross-incompatible. The build tree is therefore arch-tagged so an x86 and an aarch64 tree can
# coexist. (Device code only needs nvcc; no GPU is required to build.)
#
#   TARGETS="flipquant_sm100 test_capi_sm100" scripts/build_sm100.sh   # just the C API
set -euo pipefail

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Arch-tagged AND job-tagged. Arch-tagged because Grace (aarch64) and the x86_64 login node
# cannot share object files; job-tagged because two jobs of this repo can land on the same node at
# the same time, and two concurrent cmake builds in one tree corrupt each other.
BUILD="${BUILD:-/work/u4320956/b200/build-$(uname -m)${SLURM_JOB_ID:+-$SLURM_JOB_ID}}"
JOBS="${JOBS:-$(nproc)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TARGETS="${TARGETS:-nvfp4_gemm_sm100 mixed_nvfp4_gemm_sm100 flipquant_sm100 test_capi_sm100}"

mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DMIXFP4_ARCH=100a \
  -DCUTLASS_NVCC_ARCHS=100a \
  -DCUTLASS_ENABLE_HEADERS_ONLY=ON

# shellcheck disable=SC2086
cmake --build "$BUILD" --target $TARGETS -j "$JOBS" "$@"
echo "built into $BUILD"
case " $TARGETS " in
  *" flipquant_sm100 "*) echo "C API: $BUILD/libflipquant_sm100.so" ;;
esac
