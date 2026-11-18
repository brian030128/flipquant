// C ABI of the SM100 linear kernels used for native inference (end-to-end prefill latency, native
// perplexity, ownership checks): the tuned 256x256x256 2-SM CUTLASS blockscaled NVFP4 GEMM, with
//   FQ_SM100_TILE defined: the deployed per-tile kernel (one B format per 256x64 tile, 1-bit map),
//   otherwise:             the stock kernel (E2M1 weights, no map).
// The library compiles this file once per variant; each defines its own fq_sm100_<variant>_* symbols.
//
// D[T, N] (bf16, row-major) = bf16((row_scale[t] * scale) * (A[T, K] @ B[N, K]^T) + bias[n])
//   A: activations, NVFP4 codes [T, K/2] row-major, scale factors in CUTLASS's SFA layout for T rows;
//   B: weights, NVFP4 codes [N, K/2] (K-major), scale factors in CUTLASS's SFB layout for N rows;
//   row_scale: the per-token FP32 activation scale (nullptr = 1), scale: the weight's global scale,
//   bias: bf16 [N] (nullptr = none); one rounding, as the SM120 deployed epilogue.
// No allocation and no host synchronization: the call can be captured in a CUDA graph.
#include <cstdint>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "collective/dispatch_policy_sm100_mixed.hpp"
#ifdef FQ_SM100_TILE
#define MIXFP4_SUB_N 256
#endif
#include "collective/sm100_blockscaled_mma_mixed.hpp"
#include "sm100_gemm_launch.hpp"
#ifndef FQ_SM100_TILE
// SM120's deployed per-token activation quantizer (architecture-independent CUDA; bit-identical to the Triton one):
// FP32 global scale per token, E2M1 codes, UE4M3 scales written straight into the SM1xx scale-factor layout.
#include "quant_act.cuh"
#include "quant_act_fast.cuh"
#endif

using namespace cute;

namespace {

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD = cutlass::bfloat16_t;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// The tuned configuration: 256x256x256 per CTA pair.
using MmaTileShape = Shape<_256, _256, _256>;
using ClusterShape = Shape<_2, _1, _1>;
using CtaTileShape = Shape<_128, _256, _256>;  // per CTA of the pair

// (row_scale[m] * scale) * acc + bias[n], computed in FP32, rounded once to bf16.
namespace fusion = cutlass::epilogue::fusion;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
using RowScale = fusion::Sm90ColBroadcast<0, CtaTileShape, float, float, Stride<_1, _0, int64_t>, 1>;
using Scale = fusion::Sm90ScalarBroadcast<float>;
using Bias = fusion::Sm90RowBroadcast<0, CtaTileShape, ElementD, float, Stride<_0, _1, int64_t>, AlignmentD>;
using Scales = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, float, float, kRound>, RowScale, Scale>;
using Epilogue = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementD, float, kRound>,
                                 Scales, fusion::Sm90AccFetch, Bias>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape, Shape<_128, _64>,
    ElementAccumulator, float, void, LayoutDTag, AlignmentD, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Epilogue>::CollectiveOp;

using StdMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// The forked mainloop (per-MMA format from the map), as in mixed_nvfp4_gemm_sm100.cu.
template <class T> struct RebindToMixed;
template <int Stages, int SchedPipe, int AccumPipe, class Cluster, class Arch, class... Rest>
struct RebindToMixed<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<Stages, SchedPipe, AccumPipe, Cluster, Arch>,
    Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledMixed<Stages, SchedPipe, AccumPipe, Cluster, Arch>,
      Rest...>;
};
using CollectiveMainloop = typename RebindToMixed<StdMainloop>::type;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                        CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

typename Gemm::Arguments make_arguments(void const *a, void const *sfa, void const *b, void const *sfb,
                                        uint32_t const *map, float const *row_scale, float scale,
                                        void const *bias, void *d, int m, int n, int k) {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  typename Gemm::GemmKernel::MainloopArguments mainloop{};
  mainloop.ptr_A = static_cast<typename ElementA::DataType const *>(a);
  mainloop.dA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  mainloop.ptr_B = static_cast<typename ElementB::DataType const *>(b);
  mainloop.dB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  mainloop.ptr_SFA = static_cast<typename ElementA::ScaleFactorType const *>(sfa);
  mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  mainloop.ptr_SFB = static_cast<typename ElementB::ScaleFactorType const *>(sfb);
  mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
#ifdef FQ_SM100_TILE
  mainloop.sub_fmt = map;                     // one bit per 256x64 tile, 1 = E0M3
  mainloop.sub_fmt_stride = (k / 64 + 31) / 32;
#else
  (void)map;
#endif
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1}, mainloop,
                                     {{}, nullptr, stride_d, static_cast<ElementD *>(d), stride_d}};
  // thread: {scales: {row_scale, scale}, acc, bias}
  arguments.epilogue.thread = {
      {{row_scale, 1.0f, {}}, {{scale}}, {}},
      {},
      {static_cast<ElementD const *>(bias), ElementD(0), {}},
      {}};
  arguments.scheduler.max_swizzle_size = (m >= 8192 || n >= 8192 || k >= 8192) ? 2 : 0;
  return arguments;
}

}  // namespace

#ifdef FQ_SM100_TILE
#define FQ_NAME(x) fq_sm100_tile_##x
#else
#define FQ_NAME(x) fq_sm100_stock_##x
#endif

#define FQ_API __attribute__((visibility("default")))

extern "C" {

// Status codes: 0 ok, 1 cannot implement (shape/alignment), 2 workspace too small, 3 launch failed.
FQ_API size_t FQ_NAME(workspace_size)(int m, int n, int k) {
  return Gemm::get_workspace_size(make_arguments(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0f,
                                                 nullptr, nullptr, m, n, k));
}

FQ_API int FQ_NAME(gemm)(void const *a, void const *sfa, void const *b, void const *sfb, uint32_t const *map,
                  float const *row_scale, float scale, void const *bias, void *d, int m, int n, int k,
                  void *workspace, size_t workspace_bytes, cudaStream_t stream) {
  auto arguments = make_arguments(a, sfa, b, sfb, map, row_scale, scale, bias, d, m, n, k);
  if (m <= 0 || k % 64 != 0 || Gemm::can_implement(arguments) != cutlass::Status::kSuccess) { return 1; }
  if (Gemm::get_workspace_size(arguments) > workspace_bytes) { return 2; }
  Gemm gemm;
  if (gemm.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess) { return 1; }
  mixfp4::Sm100GemmLaunch<Gemm> launch(gemm);
  launch(stream);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 3;
}

#ifndef FQ_SM100_TILE
// Scale-factor layout (shared by both variants): bytes of an SFA/SFB buffer for `rows` rows of K, and
// the byte offset of row r's scale for K16 block j, straight from CUTLASS's layout function.
FQ_API size_t fq_sm100_sf_bytes(int rows, int k) {
  return size(filter_zeros(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, rows, k, 1))));
}

FQ_API int64_t fq_sm100_sf_offset(int rows, int k, int r, int j) {
  auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, rows, k, 1));
  return int64_t(layout(r, j * 16, 0));
}

// The map's word layout, for callers: uint32 words per 256-wide tile row.
FQ_API int fq_sm100_map_words(int k) { return (k / 64 + 31) / 32; }

// x [T, K] bf16 (row stride x_stride elements) -> packed [T, K/2] E2M1 codes, sf (SFA layout, padding zeroed), gs [T].
// mode 0 = NVFP4 rows, 1 = FourOverSix rows. Returns 0, or -1 for a bad shape, -5000 - cudaError.
FQ_API int fq_sm100_quant_rows(void const *x, int64_t x_stride, int t, int k, int mode, void *packed, void *sf,
                               void *gs, cudaStream_t stream) {
  if (k % 32 != 0 || t < 0) { return -1; }
  if (t == 0) { return 0; }
  cudaError_t err = sm120_quant::quant_rows(x, x_stride, t, k, mode, packed, sf, static_cast<float *>(gs), stream);
  return err == cudaSuccess ? 0 : -5000 - int(err);
}

// The same interface and output format with the fast quantizer (quant_act_fast.cuh): reciprocals and the hardware
// E2M1 conversion instead of the bit-exact arithmetic of the simulation.
FQ_API int fq_sm100_quant_rows_fast(void const *x, int64_t x_stride, int t, int k, int mode, void *packed, void *sf,
                                    void *gs, cudaStream_t stream) {
  if (k % 32 != 0 || t < 0) { return -1; }
  if (t == 0) { return 0; }
  cudaError_t err = fq_fast_quant::quant_rows(x, x_stride, t, k, mode, packed, sf, static_cast<float *>(gs), stream);
  return err == cudaSuccess ? 0 : -5000 - int(err);
}
#endif

}  // extern "C"
