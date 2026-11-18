// libflipquant_sm100.so -- C ABI over the exact N8 x K64 mixed-format SM100 GEMM.
//
// Kernel: n8k64_pipeline<32, 128, /*Packed=*/true> from src/n8k64_pipeline.cuh, i.e. the
// "--packed-inputs" double-buffered pipeline of mixed_n8k64_sm100 at its default CTA 128x32x128.
// Every MMA is a real tcgen05 128x8x64 block-scaled MMA whose B (weight) format is written into
// the runtime instruction descriptor from fmt_b, so every N8 x K64 weight region is decoded under
// its own codebook -- exact, no branching, no cubin patching.
//
// Why this configuration: it is the fastest one the existing harness VALIDATED
// (scripts/check_sm100_n8_pipeline.sh: 51 cases, M/N tails and a partial outer K tile at
// 144x136x320, plus 257x264x1024, bit-exact against the BF16-rounded reference; memcheck and
// racecheck clean). It is ~5% of the wide kernel's
// throughput -- correctness first.
//
// Internal layout conversion (all in the caller's workspace, stream-ordered):
//   a  [M, K/2]  -> pa [K/32][M] x 16 B      (n8_pack_operands)
//   b  [N, K/2]  -> pb [K/32][N] x 16 B      (n8_pack_operands)
//   sa [M, K/16] -> spa [K/64][M] x 4 B      (n8_pack_scales)
//   sb [N, K/16] -> spb [K/64][N] x 4 B      (n8_pack_scales)
// fmt_b [N/8, K/64] and d [M, N] are consumed/produced directly.
#include <cstdint>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"

using namespace cute;
using F4 = cutlass::float_e2m1_t;
using SF = cutlass::float_ue4m3_t;
using BF16 = cutlass::bfloat16_t;
#include "n8k64_pipeline.cuh"

#include "fq_common.hpp"

namespace {

constexpr int kTileN = 32;    // validated default of --packed-inputs
constexpr int kTileK = 128;
constexpr int kGranRows = 8;  // weight-format granule: N8 x K64
constexpr int kGranCols = 64;
constexpr int kMaxGridY = 65535;

struct Sm100Workspace {
  uint8_t* pa; uint8_t* pb; uint32_t* spa; uint32_t* spb;
  size_t bytes;
};

Sm100Workspace carve(void* base, int M, int N, int K) {
  fq::Carve c(base);
  Sm100Workspace w{};
  w.pa  = c.take<uint8_t>(size_t(M) * size_t(K / 2));
  w.pb  = c.take<uint8_t>(size_t(N) * size_t(K / 2));
  w.spa = c.take<uint32_t>(size_t(M) * size_t(K / 16));
  w.spb = c.take<uint32_t>(size_t(N) * size_t(K / 16));
  w.bytes = c.offset;
  return w;
}

int validate_shape(int M, int N, int K) {
  if (M < 1 || N < 1 || K < 1) {
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm100: M, N, K must be >= 1 (got %d, %d, %d)", M, N, K);
  }
  if (N % kGranRows != 0 || K % kGranCols != 0) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm100: requires N %% 8 == 0 and K %% 64 == 0 (got N=%d, K=%d)", N, K);
  }
  // Grid-dimension limits of the pack kernels (y = rows/32) and the GEMM (y = N/32).
  if ((M + 31) / 32 > kMaxGridY || (N + 31) / 32 > kMaxGridY) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm100: M and N must be <= %d (got M=%d, N=%d)",
                    kMaxGridY * 32, M, N);
  }
  return FQ_OK;
}

}  // namespace

extern "C" FQ_API void fq_sm100_weight_granule(int* rows, int* cols) {
  if (rows) *rows = kGranRows;
  if (cols) *cols = kGranCols;
}

extern "C" FQ_API size_t fq_sm100_workspace_bytes(int M, int N, int K) {
  if (M < 1 || N < 1 || K < 64) return 0;
  return carve(nullptr, M, N, K).bytes;
}

extern "C" FQ_API int fq_sm100_mixfp4_gemm(const uint8_t* a, const uint8_t* sa,
                                           const uint8_t* b, const uint8_t* sb,
                                           const uint8_t* fmt_b, void* d_bf16,
                                           int M, int N, int K, int fmt_a,
                                           void* workspace, size_t workspace_bytes,
                                           fq_stream_t stream_handle) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  if (!a || !sa || !b || !sb || !fmt_b || !d_bf16) {
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm100: null tensor pointer");
  }
  if (fmt_a != 0 && fmt_a != 1) {
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm100: fmt_a must be 0 (E0M3) or 1 (E2M1), got %d", fmt_a);
  }
  if (int rc = validate_shape(M, N, K)) return rc;
  if (!fq::aligned(a, 16) || !fq::aligned(b, 16)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm100: a and b must be 16-byte aligned");
  }
  if (!fq::aligned(sa, 4) || !fq::aligned(sb, 4)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm100: sa and sb must be 4-byte aligned");
  }
  if (!fq::aligned(d_bf16, 2)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm100: d must be 2-byte aligned");
  }
  size_t const need = carve(nullptr, M, N, K).bytes;
  if (!workspace || workspace_bytes < need || !fq::aligned(workspace, fq::kWorkspaceAlign)) {
    return fq::fail(FQ_ERR_WORKSPACE, "fq_sm100: workspace must be >= %zu bytes and 256-byte aligned "
                    "(got %zu bytes at %p)", need, workspace_bytes, workspace);
  }

  int dev = 0, major = 0, minor = 0;
  if (int rc = fq::current_device_cc(&dev, &major, &minor)) return rc;
  if (major != 10 || minor != 0) {
    // sm_100a SASS is architecture-specific: it loads on CC 10.0 only (B200/GB200).
    return fq::fail(FQ_ERR_DEVICE, "fq_sm100: device %d is sm_%d%d; this library contains sm_100a code only",
                    dev, major, minor);
  }

  Sm100Workspace w = carve(workspace, M, N, K);
  int const kg64 = K / 64, kg32 = K / 32;

  n8_pack_scales<<<dim3((kg64 + 31) / 32, (M + 31) / 32), dim3(32, 8), 0, stream>>>(
      reinterpret_cast<uint32_t const*>(sa), w.spa, M, kg64);
  n8_pack_scales<<<dim3((kg64 + 31) / 32, (N + 31) / 32), dim3(32, 8), 0, stream>>>(
      reinterpret_cast<uint32_t const*>(sb), w.spb, N, kg64);
  n8_pack_operands<<<dim3((kg32 + 31) / 32, (M + 31) / 32), dim3(32, 8), 0, stream>>>(
      reinterpret_cast<uint4 const*>(a), reinterpret_cast<uint4*>(w.pa), M, kg32);
  n8_pack_operands<<<dim3((kg32 + 31) / 32, (N + 31) / 32), dim3(32, 8), 0, stream>>>(
      reinterpret_cast<uint4 const*>(b), reinterpret_cast<uint4*>(w.pb), N, kg32);
  if (int rc = fq::check_launch("fq_sm100 pack")) return rc;

  dim3 grid((M + 127) / 128, (N + kTileN - 1) / kTileN);
  n8k64_pipeline<kTileN, kTileK, true><<<grid, 128, 0, stream>>>(
      w.pa, w.pb, w.spa, w.spb, fmt_b, static_cast<BF16*>(d_bf16), M, N, K, fmt_a);
  return fq::check_launch("fq_sm100 n8k64_pipeline<32,128,packed>");
}
