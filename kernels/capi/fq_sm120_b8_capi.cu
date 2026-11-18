// libflipquant_sm120_b8.so -- C ABI over the mixed E0M3/E2M1 CUTLASS GEMM for sm_120a with the
// WEIGHTS ON CUTLASS OPERAND B at an 8-row x 64-K format granule.
//
// KERNEL CONFIGURATION (sm120/kernel/docs/mixed_nvfp4_report.md, "8x64 on B via the 1x8 arrangement" and "The
// 8-column x 64-K frontier"; build recipe in section 6):
//   CTA 128x128x128, 1x8 warp arrangement (-DMIXFP4_ATOM_M=1), extent-128 N permutation
//   (-DMIXFP4_PERM_N=128: Shape<_8,_2,_2,_4>:Stride<_1,_16,_8,_32>), x4 LDSM, epilogue tile 64x64,
//   generated blob path (-DMIXFP4_BLOB=1, header generated with MMA_M=8 MMA_N=2 A_ATOMS=8 B_ATOMS=1),
//   -DMIXFP4_JOINT_KB=1 -DMIXFP4_A_ALL_E2M1=1 -DMIXFP4_A_ATOMS_PER_GRANULE=8
//   -DMIXFP4_B_ATOMS_PER_GRANULE=1.
// Measured +15.1% over stock NVFP4 on the RTX PRO 6000 (4096^3) in the research harness.
//
// CONSEQUENCES FOR THIS API (contrast with fq_sm120_capi.cu, weights as A at 16x64):
//   * We compute D = X * W^T directly: CUTLASS M = M_logical (tokens, operand A, pinned to E2M1),
//     CUTLASS N = N_logical (weight rows, operand B), D [Mp, Np] row-major. No transpose; D is
//     written straight into d when no padding is needed, else into workspace and cropped.
//   * fmt_a must be 1 (E2M1): E0M3 activations are not compiled in (sites 1/3 do not exist).
//   * Granule placement: the kernel reads one flag per (warp, n-atom, k_block). Each n-atom is 8
//     columns of B; the N permutation only decides WHICH 8-aligned block of 8 a warp's atom gets
//     (for fixed outer indices the atom covers 16*i1 + 8*i2 + 32*i3 + 0..7). So a logical
//     [N/8, K/64] map needs no row permutation. Not assumed: the first call derives the map from
//     the kernel's own TiledMma (partition_B) and returns FQ_ERR_INTERNAL if it is not contiguous.
//   * The .so must be SASS-patched (scripts/patch_mixed_nvfp4_gemm.py --expect-sites 0,2
//     --mark-patched FQ_SM120B8_PATCH_STATE); unpatched it returns FQ_ERR_NOT_PATCHED.

#if !(defined(MIXFP4_BLOB) && MIXFP4_BLOB && defined(MIXFP4_JOINT_KB) && MIXFP4_JOINT_KB && \
      defined(MIXFP4_A_ALL_E2M1) && MIXFP4_A_ALL_E2M1 && defined(MIXFP4_ATOM_M) && MIXFP4_ATOM_M == 1 && \
      defined(MIXFP4_PERM_N) && MIXFP4_PERM_N == 128 && defined(MIXFP4_A_ATOMS_PER_GRANULE) && \
      MIXFP4_A_ATOMS_PER_GRANULE == 8 && defined(MIXFP4_B_ATOMS_PER_GRANULE) && \
      MIXFP4_B_ATOMS_PER_GRANULE == 1)
#error "fq_sm120_b8_capi.cu must be built with -DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KB=1 -DMIXFP4_A_ALL_E2M1=1 -DMIXFP4_ATOM_M=1 -DMIXFP4_PERM_N=128 -DMIXFP4_A_ATOMS_PER_GRANULE=8 -DMIXFP4_B_ATOMS_PER_GRANULE=1 (see CMakeLists.txt)"
#endif
#if defined(MIXFP4_TILE_N) || defined(MIXFP4_TILE_K) || defined(MIXFP4_ATOM_N) || \
    defined(MIXFP4_NO_DISPATCH) || defined(MIXFP4_FAKE_INDEX) || defined(MIXFP4_PTX) || defined(MIXFP4_JOINT_KA)
#error "fq_sm120_b8_capi.cu is pinned to the validated 1x8 / PERM_N=128 configuration"
#endif

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "collective/dispatch_policy_mixed.hpp"
#include "collective/mma_sm120_mixed.hpp"
#include "collective/sm120_blockscaled_mma_tma_mixed.hpp"

#include "fq_common.hpp"

using namespace cute;

// -------------------------------------------------------------------------------------------------
// Kernel type -- a copy of the type assembly in sm120/kernel/src/mixed_nvfp4_gemm.cu at MIXFP4_ATOM_M=1 (1x8),
// MIXFP4_PERM_N=128, MIXFP4_EPI_M=MIXFP4_EPI_N=64, x4 LDSM (the "b8x64" recipe). Keep in sync.
// -------------------------------------------------------------------------------------------------
namespace fq120b8 {

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using MixedAtomLayoutMNK = Layout<Shape<_1, _8, _1>>;
// 1x8: MMA_TILE is (128, 64), so the epilogue tile must be named; 64x64 measured best.
using MixedEpilogueTile = Shape<_64, _64>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    MixedEpilogueTile,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using StdMainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>;

using MixedDispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaledMixed<
    StdMainloopBuilder::DispatchPolicy::Stages,
    StdMainloopBuilder::DispatchPolicy::SchedulerPipelineStageCount,
    typename StdMainloopBuilder::DispatchPolicy::ClusterShape,
    typename StdMainloopBuilder::DispatchPolicy::Schedule>;

using MixedMmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS_Mixed;
using MixedPermTileN = Layout<Shape<_8, _2, _2, _4>, Stride<_1, _16, _8, _32>>;
using MixedTiledMma = decltype(cute::make_tiled_mma(
    MixedMmaOp{}, MixedAtomLayoutMNK{},
    Tile<typename StdMainloopBuilder::PermTileM, MixedPermTileN,
         typename StdMainloopBuilder::PermTileK>{}));

using MixedSmemCopyAtomsB = decltype(cute::make_tuple(
    cutlass::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename StdMainloopBuilder::SmemAllocTypeB>{},
    cute::get<1>(typename StdMainloopBuilder::SmemCopyAtomsB{})));

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    MixedDispatchPolicy, ThreadBlockShape,
    cute::tuple<typename StdMainloopBuilder::ElementA, typename StdMainloopBuilder::ElementSF>,
    typename StdMainloopBuilder::StridePairA,
    cute::tuple<typename StdMainloopBuilder::ElementB, typename StdMainloopBuilder::ElementSF>,
    typename StdMainloopBuilder::StridePairB,
    MixedTiledMma,
    typename StdMainloopBuilder::GmemTiledCopyPairA,
    typename StdMainloopBuilder::SmemLayoutAtomsA,
    typename StdMainloopBuilder::SmemCopyAtomsA,
    cute::identity,
    typename StdMainloopBuilder::GmemTiledCopyPairB,
    typename StdMainloopBuilder::SmemLayoutAtomsB,
    MixedSmemCopyAtomsB,
    cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr int kTileM = 128, kTileN = 128, kTileK = 128;
constexpr int kGranRows = 8 * MIXFP4_B_ATOMS_PER_GRANULE;   // 8 weight rows (one n8 atom)
constexpr int kGranCols = 64;                              // MIXFP4_JOINT_KB: one k_block
static_assert(kGranRows == 8, "weight granule is one n8 atom");

// -------------------------------------------------------------------------------------------------
// Patch marker. scripts/patch_mixed_nvfp4_gemm.py --mark-patched FQ_SM120B8_PATCH_STATE rewrites
// "UNPATCHED" to "PATCHED__" (same length) in the file bytes after a verified SASS patch. volatile
// so the compiler cannot constant-fold the comparison against the initializer.
// -------------------------------------------------------------------------------------------------
extern "C" {
__attribute__((used, visibility("hidden")))
volatile char fq_sm120b8_patch_state[] = "FQ_SM120B8_PATCH_STATE:UNPATCHED";
}

bool library_is_patched() {
  static char const kPatched[] = "FQ_SM120B8_PATCH_STATE:PATCHED__";
  for (size_t i = 0; i + 1 < sizeof(kPatched); ++i) {
    if (fq_sm120b8_patch_state[i] != kPatched[i]) return false;
  }
  return true;
}

// -------------------------------------------------------------------------------------------------
// Granule self-check: derive, from the kernel's own TiledMma, which CTA-tile columns of B (weight
// rows) each (warp, n-atom) covers -- the unit the kernel reads one format flag for -- and require
// it to be exactly the 8-aligned contiguous blocks this API's logical fmt_b map assumes.
// -------------------------------------------------------------------------------------------------
bool granule_map_is_contiguous(char* why, size_t why_len) {
  MixedTiledMma tiled_mma;
  int const mn_extent = int(size<1>(ThreadBlockShape{}));
  int const k_extent = int(size<2>(ThreadBlockShape{}));
  Tensor id = make_identity_tensor(make_shape(mn_extent, k_extent));
  int const num_threads = int(size(typename MixedTiledMma::ThrLayoutVMNK{}));
  std::map<std::pair<int, int>, std::set<int>> coverage;  // (warp, n-atom) -> columns of B
  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto frag = thr_mma.partition_B(id);
    int const num_atoms = int(size<1>(frag));
    for (int atom = 0; atom < num_atoms; ++atom) {
      auto& dst = coverage[{tid / 32, atom / MIXFP4_B_ATOMS_PER_GRANULE}];
      for (int v = 0; v < int(size<0>(frag)); ++v) {
        for (int kk = 0; kk < int(size<2>(frag)); ++kk) {
          int const mn = get<0>(frag(v, atom, kk));
          if (mn >= 0 && mn < mn_extent) dst.insert(mn);
        }
      }
    }
  }
  std::vector<int> covered_by(size_t(mn_extent), 0);
  for (auto const& kv : coverage) {
    auto const& rows = kv.second;
    if (rows.empty()) continue;
    int const lo = *rows.begin();
    if (int(rows.size()) != kGranRows || lo % kGranRows != 0 || *rows.rbegin() != lo + kGranRows - 1) {
      std::snprintf(why, why_len, "warp %d atom-group %d covers %zu rows starting at %d "
                    "(expected %d contiguous rows aligned to %d)", kv.first.first, kv.first.second,
                    rows.size(), lo, kGranRows, kGranRows);
      return false;
    }
    for (int r : rows) covered_by[size_t(r)] = 1;
  }
  for (int r = 0; r < mn_extent; ++r) {
    if (!covered_by[size_t(r)]) {
      std::snprintf(why, why_len, "CTA column %d of B is not covered by any MMA atom", r);
      return false;
    }
  }
  return true;
}

// -------------------------------------------------------------------------------------------------
// Device helpers
// -------------------------------------------------------------------------------------------------

// dst [rows_pad, cols16_pad] (16-byte units) <- src [rows, cols16], zero-filled outside.
__global__ void pad_copy_rows(uint4 const* __restrict__ src, int rows, int cols16,
                              uint4* __restrict__ dst, int rows_pad, int cols16_pad) {
  size_t const total = size_t(rows_pad) * size_t(cols16_pad);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    int const r = int(i / size_t(cols16_pad));
    int const c = int(i % size_t(cols16_pad));
    dst[i] = (r < rows && c < cols16) ? src[size_t(r) * size_t(cols16) + size_t(c)]
                                      : make_uint4(0, 0, 0, 0);
  }
}

// Scatter row-major [rows, kgroups] UE4M3 scale bytes into CUTLASS's interleaved SF layout for a
// (rows_pad x 16*kgroups_pad) operand. Bit 7 is cleared, then -- when fmt != nullptr (weights) --
// set for every scale byte whose (16-row, 64-K) granule is E0M3 (fmt == 0). Padding is zero and
// untagged. Every byte of the destination is written, so no memset is required.
template <class LayoutSF>
__global__ void build_scales(uint8_t const* __restrict__ src, int rows, int kgroups,
                             uint8_t* __restrict__ dst, LayoutSF layout,
                             int rows_pad, int kgroups_pad,
                             uint8_t const* __restrict__ fmt, int fmt_cols) {
  size_t const total = size_t(rows_pad) * size_t(kgroups_pad);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    int const r = int(i / size_t(kgroups_pad));
    int const g = int(i % size_t(kgroups_pad));
    uint8_t v = 0;
    if (r < rows && g < kgroups) {
      v = uint8_t(src[size_t(r) * size_t(kgroups) + size_t(g)] & 0x7fu);
      if (fmt != nullptr && fmt[size_t(r / kGranRows) * size_t(fmt_cols) + size_t(g / 4)] == 0) {
        v |= uint8_t(MIXFP4_FLAG_MASK);
      }
    }
    dst[layout(r, g * 16, 0)] = v;
  }
}

// d [M, N] row-major <- dp [M_pad, N_pad] row-major (bf16 as raw 16-bit).
__global__ void crop_out(uint16_t const* __restrict__ dp, int n_pad, uint16_t* __restrict__ d, int M, int N) {
  size_t const total = size_t(M) * size_t(N);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    int const m = int(i / size_t(N)), n = int(i % size_t(N));
    d[i] = dp[size_t(m) * size_t(n_pad) + size_t(n)];
  }
}

int grid_for(size_t total, int block) {
  size_t blocks = (total + size_t(block) - 1) / size_t(block);
  return int(blocks < 65536 ? (blocks == 0 ? 1 : blocks) : 65536);
}

// -------------------------------------------------------------------------------------------------
// Problem geometry + workspace layout
// -------------------------------------------------------------------------------------------------
struct Plan {
  int M, N, K;       // logical
  int Mk, Nk, Kk;    // CUTLASS problem: Mk = Mp (tokens), Nk = Np (weights), Kk = Kp
  bool copy_w, copy_x, copy_d;
  LayoutSFA layout_sfa;
  LayoutSFB layout_sfb;
  size_t sfa_bytes, sfb_bytes;
};

Plan make_plan(int M, int N, int K) {
  Plan p{};
  p.M = M; p.N = N; p.K = K;
  p.Mk = fq::round_up(M, kTileM);
  p.Nk = fq::round_up(N, kTileN);
  p.Kk = fq::round_up(K, kTileK);
  p.copy_w = (p.Nk != N) || (p.Kk != K);
  p.copy_x = (p.Mk != M) || (p.Kk != K);
  p.copy_d = (p.Mk != M) || (p.Nk != N);
  p.layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(p.Mk, p.Nk, p.Kk, 1));
  p.layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(p.Mk, p.Nk, p.Kk, 1));
  p.sfa_bytes = size_t(cute::cosize(p.layout_sfa));
  p.sfb_bytes = size_t(cute::cosize(p.layout_sfb));
  return p;
}

typename Gemm::Arguments make_args(Plan const& p, void const* A, void const* B, void const* SFA,
                                   void const* SFB, void* D, int device, int sm_count) {
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {p.Mk, p.Kk, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {p.Nk, p.Kk, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {p.Mk, p.Nk, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {p.Mk, p.Nk, 1});
  using DataA = typename ElementA::DataType;
  using DataB = typename ElementB::DataType;
  using SFType = typename ElementA::ScaleFactorType;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {p.Mk, p.Nk, p.Kk, 1},
      {static_cast<DataA const*>(A), stride_a, static_cast<DataB const*>(B), stride_b,
       static_cast<SFType const*>(SFA), p.layout_sfa, static_cast<SFType const*>(SFB), p.layout_sfb},
      // beta = 0: C is never read (is_source_needed() is false); point it at D so that no TMA
      // descriptor is ever built over a null pointer.
      {{1.0f, 0.0f}, static_cast<ElementC const*>(D), stride_c, static_cast<ElementD*>(D), stride_d}};
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count;
  return args;
}

struct Buffers {
  uint8_t* w; uint8_t* x; uint8_t* sfa; uint8_t* sfb; uint16_t* dt; uint8_t* cutlass_ws;
  size_t cutlass_ws_bytes;
  size_t bytes;
};

Buffers carve(void* base, Plan const& p, size_t cutlass_ws_bytes) {
  fq::Carve c(base);
  Buffers b{};
  b.w   = p.copy_w ? c.take<uint8_t>(size_t(p.Nk) * size_t(p.Kk / 2)) : nullptr;
  b.x   = p.copy_x ? c.take<uint8_t>(size_t(p.Mk) * size_t(p.Kk / 2)) : nullptr;
  b.sfa = c.take<uint8_t>(p.sfa_bytes);
  b.sfb = c.take<uint8_t>(p.sfb_bytes);
  b.dt  = p.copy_d ? c.take<uint16_t>(size_t(p.Mk) * size_t(p.Nk) * sizeof(uint16_t)) : nullptr;
  b.cutlass_ws_bytes = cutlass_ws_bytes;
  b.cutlass_ws = cutlass_ws_bytes ? c.take<uint8_t>(cutlass_ws_bytes) : nullptr;
  b.bytes = c.offset;
  return b;
}

size_t cutlass_workspace_bytes(Plan const& p, int device, int sm_count) {
  // Only shapes and hardware info matter for sizing; pointers are never dereferenced here. The
  // real SM count is used so that a scheduler whose workspace scales with it is sized correctly.
  auto args = make_args(p, nullptr, nullptr, nullptr, nullptr, nullptr, device, sm_count);
  return Gemm::get_workspace_size(args);
}

int validate_shape(int M, int N, int K) {
  if (M < 1 || N < 1 || K < 1) {
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm120b8: M, N, K must be >= 1 (got %d, %d, %d)", M, N, K);
  }
  if (N % kGranRows != 0 || K % kGranCols != 0) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120b8: requires N %% 8 == 0 and K %% 64 == 0 (got N=%d, K=%d)", N, K);
  }
  if (N > (1 << 30) || M > (1 << 30) || K > (1 << 30)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120b8: shape too large (M=%d, N=%d, K=%d)", M, N, K);
  }
  return FQ_OK;
}

struct DeviceInfo { int major = -1, minor = -1, sm_count = 0; };

int device_info(int* dev_out, DeviceInfo* out) {
  static std::mutex mu;
  static std::map<int, DeviceInfo> cache;
  int dev = 0, major = 0, minor = 0;
  if (int rc = fq::current_device_cc(&dev, &major, &minor)) return rc;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(dev);
  if (it == cache.end()) {
    DeviceInfo info;
    info.major = major; info.minor = minor;
    cudaError_t e = cudaDeviceGetAttribute(&info.sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (e != cudaSuccess) {
      return fq::fail(FQ_ERR_CUDA, "fq_sm120b8: cudaDeviceGetAttribute(SM count) failed: %s", cudaGetErrorString(e));
    }
    it = cache.emplace(dev, info).first;
  }
  *dev_out = dev;
  *out = it->second;
  return FQ_OK;
}

int self_check_once() {
  static std::once_flag once;
  static bool ok = false;
  static char why[512] = "";
  std::call_once(once, [] { ok = granule_map_is_contiguous(why, sizeof(why)); });
  if (!ok) {
    return fq::fail(FQ_ERR_INTERNAL, "fq_sm120b8: kernel format granule is not the contiguous %dx%d "
                    "block this API assumes: %s", kGranRows, kGranCols, why);
  }
  return FQ_OK;
}

}  // namespace fq120b8

extern "C" FQ_API void fq_sm120b8_weight_granule(int* rows, int* cols) {
  if (rows) *rows = fq120b8::kGranRows;
  if (cols) *cols = fq120b8::kGranCols;
}

extern "C" FQ_API size_t fq_sm120b8_workspace_bytes(int M, int N, int K) {
  if (M < 1 || N < 1 || K < 64 || N % fq120b8::kGranRows || K % fq120b8::kGranCols) return 0;
  int dev = 0;
  fq120b8::DeviceInfo info;
  if (fq120b8::device_info(&dev, &info) != FQ_OK) return 0;  // fq_last_error() says why
  auto const plan = fq120b8::make_plan(M, N, K);
  return fq120b8::carve(nullptr, plan, fq120b8::cutlass_workspace_bytes(plan, dev, info.sm_count)).bytes;
}

extern "C" FQ_API int fq_sm120b8_mixfp4_gemm(const uint8_t* a, const uint8_t* sa,
                                           const uint8_t* b, const uint8_t* sb,
                                           const uint8_t* fmt_b, void* d_bf16,
                                           int M, int N, int K, int fmt_a,
                                           void* workspace, size_t workspace_bytes,
                                           fq_stream_t stream_handle) {
  using namespace fq120b8;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

  if (!library_is_patched()) {
    return fq::fail(FQ_ERR_NOT_PATCHED, "fq_sm120b8: this is the UNPATCHED library -- every E0M3 site "
                    "would decode as E2M1. Load libflipquant_sm120_b8.so produced by the build's "
                    "patch step (scripts/patch_mixed_nvfp4_gemm.py --mark-patched).");
  }
  if (!a || !sa || !b || !sb || !fmt_b || !d_bf16) {
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm120b8: null tensor pointer");
  }
  if (fmt_a != 1) {
    return fq::fail(fmt_a == 0 ? FQ_ERR_UNSUPPORTED : FQ_ERR_INVALID_ARGUMENT,
                    "fq_sm120b8: fmt_a=%d not supported -- this build pins activations (CUTLASS "
                    "operand A) to E2M1 (MIXFP4_A_ALL_E2M1); only fmt_a=1 is implemented", fmt_a);
  }
  if (int rc = validate_shape(M, N, K)) return rc;
  if (!fq::aligned(a, 16) || !fq::aligned(b, 16)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120b8: a and b must be 16-byte aligned");
  }
  if (!fq::aligned(d_bf16, 2)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120b8: d must be 2-byte aligned");
  }
  if (int rc = self_check_once()) return rc;

  int dev = 0;
  DeviceInfo info;
  if (int rc = device_info(&dev, &info)) return rc;
  if (info.major != 12 || info.minor != 0) {
    return fq::fail(FQ_ERR_DEVICE, "fq_sm120b8: device %d is sm_%d%d; this library contains patched "
                    "sm_120a SASS only (no PTX)", dev, info.major, info.minor);
  }

  Plan const plan = make_plan(M, N, K);
  size_t const cws = cutlass_workspace_bytes(plan, dev, info.sm_count);
  size_t const need = carve(nullptr, plan, cws).bytes;
  if (!workspace || workspace_bytes < need || !fq::aligned(workspace, fq::kWorkspaceAlign)) {
    return fq::fail(FQ_ERR_WORKSPACE, "fq_sm120b8: workspace must be >= %zu bytes and 256-byte "
                    "aligned (got %zu bytes at %p)", need, workspace_bytes, workspace);
  }
  Buffers const w = carve(workspace, plan, cws);

  constexpr int kBlock = 256;
  // Operands: consume in place when already tile-shaped, else zero-pad into workspace.
  void const* A = a;  // activations -> CUTLASS A (E2M1)
  void const* B = b;  // weights -> CUTLASS B
  if (plan.copy_x) {
    size_t const total = size_t(plan.Mk) * size_t(plan.Kk / 32);
    pad_copy_rows<<<grid_for(total, kBlock), kBlock, 0, stream>>>(
        reinterpret_cast<uint4 const*>(a), M, K / 32, reinterpret_cast<uint4*>(w.x), plan.Mk, plan.Kk / 32);
    A = w.x;
  }
  if (plan.copy_w) {
    size_t const total = size_t(plan.Nk) * size_t(plan.Kk / 32);
    pad_copy_rows<<<grid_for(total, kBlock), kBlock, 0, stream>>>(
        reinterpret_cast<uint4 const*>(b), N, K / 32, reinterpret_cast<uint4*>(w.w), plan.Nk, plan.Kk / 32);
    B = w.w;
  }
  {
    size_t const total_a = size_t(plan.Mk) * size_t(plan.Kk / 16);
    build_scales<<<grid_for(total_a, kBlock), kBlock, 0, stream>>>(
        sa, M, K / 16, w.sfa, plan.layout_sfa, plan.Mk, plan.Kk / 16,
        static_cast<uint8_t const*>(nullptr), 0);
    size_t const total_b = size_t(plan.Nk) * size_t(plan.Kk / 16);
    build_scales<<<grid_for(total_b, kBlock), kBlock, 0, stream>>>(
        sb, N, K / 16, w.sfb, plan.layout_sfb, plan.Nk, plan.Kk / 16, fmt_b, K / 64);
  }
  if (int rc = fq::check_launch("fq_sm120b8 operand/scale preparation")) return rc;

  void* D = plan.copy_d ? static_cast<void*>(w.dt) : d_bf16;
  auto args = make_args(plan, A, B, w.sfa, w.sfb, D, dev, info.sm_count);
  Gemm gemm;
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    return fq::fail(FQ_ERR_INTERNAL, "fq_sm120b8: CUTLASS can_implement failed: %s", cutlassGetStatusString(st));
  }
  st = gemm.initialize(args, w.cutlass_ws, stream);
  if (st != cutlass::Status::kSuccess) {
    return fq::fail(FQ_ERR_INTERNAL, "fq_sm120b8: CUTLASS initialize failed: %s", cutlassGetStatusString(st));
  }
  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) {
    return fq::fail(FQ_ERR_CUDA, "fq_sm120b8: CUTLASS run failed: %s (%s)", cutlassGetStatusString(st),
                    cudaGetErrorString(cudaGetLastError()));
  }

  if (!plan.copy_d) return FQ_OK;
  size_t const total_d = size_t(M) * size_t(N);
  crop_out<<<grid_for(total_d, kBlock), kBlock, 0, stream>>>(w.dt, plan.Nk, static_cast<uint16_t*>(d_bf16), M, N);
  return fq::check_launch("fq_sm120b8 output crop");
}
