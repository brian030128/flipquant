// libflipquant_sm120.so -- C ABI over the mixed E0M3/E2M1 CUTLASS GEMM for sm_120a.
//
// KERNEL CONFIGURATION ("weights as A", sm120/kernel/docs/mixed_nvfp4_report.md section 6, commit 7b3ab34):
//   CTA 128x128x128, stock 4x2 warp arrangement, stock epilogue, generated blob path
//   (-DMIXFP4_BLOB=1, header generated with MMA_M=2 MMA_N=8 A_ATOMS=1 B_ATOMS=8),
//   -DMIXFP4_JOINT_KA=1 -DMIXFP4_B_ALL_E2M1=1 -DMIXFP4_A_ATOMS_PER_GRANULE=1
//   -DMIXFP4_B_ATOMS_PER_GRANULE=8.
// That puts the WEIGHTS on CUTLASS operand A with a 16-row x 64-K format granule (one m16n8k64
// A footprint) and pins operand B (the activations) to E2M1. Measured +5.8%..+7.2% over stock
// NVFP4 on the RTX PRO 6000 and "correct at 512^3, 1024^3, 2048^3 and 1024x2048x512".
//
// CONSEQUENCES FOR THIS API
//   * fmt_a (activation codebook) must be 1 (E2M1): E0M3 activations are not compiled into this
//     kernel (sites 2/3 do not exist). fmt_a == 0 returns FQ_ERR_UNSUPPORTED -- never a silent
//     E2M1 decode.
//   * We compute D^T = W * X^T: CUTLASS M = N_logical (weight rows), CUTLASS N = M_logical
//     (tokens). CUTLASS writes D^T [Np, Mp] row-major into workspace; a transpose kernel then
//     writes d [M, N] row-major.
//   * Everything is padded to tile multiples in workspace: Np = roundup(N,128),
//     Mp = roundup(M,128), Kp = roundup(K,128). Operands are only copied when padding is needed
//     (weights with N%128==0 && K%128==0 are consumed in place). Scale factors are ALWAYS
//     rebuilt: CUTLASS wants the Sm1xxBlkScaledConfig interleaved SF atom layout, and the weight
//     scale bytes must carry the format tag in bit 7 (1 = E0M3).
//   * Granule placement: the kernel reads one flag per (warp, m-atom, k_block). With PermTileM =
//     128 (identity) and the 4x2 arrangement, m-atom i of warp row w covers rows
//     16*(w + 4 i) .. +15 of the CTA tile -- i.e. 16 CONTIGUOUS, 16-aligned rows. So a logical
//     [N/16, K/64] map needs no row permutation. This is not assumed: the first call derives the
//     granule map from the very TiledMma the kernel uses (same method as build_granule_map in
//     sm120/kernel/src/mixed_nvfp4_gemm.cu) and refuses to run (FQ_ERR_INTERNAL) if it is not contiguous.
//   * The .so must be SASS-patched (scripts/patch_mixed_nvfp4_gemm.py --mark-patched ...). The
//     patcher rewrites the marker below; an unpatched library returns FQ_ERR_NOT_PATCHED instead
//     of silently computing E2M1 everywhere.

#if !(defined(MIXFP4_BLOB) && MIXFP4_BLOB && defined(MIXFP4_JOINT_KA) && MIXFP4_JOINT_KA && \
      defined(MIXFP4_B_ALL_E2M1) && MIXFP4_B_ALL_E2M1 && defined(MIXFP4_A_ATOMS_PER_GRANULE) && \
      MIXFP4_A_ATOMS_PER_GRANULE == 1 && defined(MIXFP4_B_ATOMS_PER_GRANULE) && \
      MIXFP4_B_ATOMS_PER_GRANULE == 8)
#error "fq_sm120_capi.cu must be built with -DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KA=1 -DMIXFP4_B_ALL_E2M1=1 -DMIXFP4_A_ATOMS_PER_GRANULE=1 -DMIXFP4_B_ATOMS_PER_GRANULE=8 (see CMakeLists.txt)"
#endif
#if defined(MIXFP4_TILE_N) || defined(MIXFP4_TILE_K) || defined(MIXFP4_ATOM_M) || \
    defined(MIXFP4_ATOM_N) || defined(MIXFP4_PERM_N) || defined(MIXFP4_NO_DISPATCH) || \
    defined(MIXFP4_FAKE_INDEX) || defined(MIXFP4_PTX)
#error "fq_sm120_capi.cu is pinned to the validated stock-arrangement configuration; do not override tile/arrangement/probe macros"
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
// Kernel type -- a verbatim copy of the type assembly in sm120/kernel/src/mixed_nvfp4_gemm.cu at its default
// arrangement macros (TILE 128x128x128, ATOM 4x2, EpilogueTileAuto, builder PermTileN, x4 LDSM).
// Keep in sync with that file; the driver there is where this configuration was validated.
// -------------------------------------------------------------------------------------------------
namespace fq120 {

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
using MixedAtomLayoutMNK = Layout<Shape<_4, _2, _1>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
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
using MixedPermTileN = typename StdMainloopBuilder::PermTileN;
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
constexpr int kGranRows = 16 * MIXFP4_A_ATOMS_PER_GRANULE;  // 16 weight rows
constexpr int kGranCols = 64;                               // MIXFP4_JOINT_KA: one k_block
static_assert(kGranRows == 16, "weight granule is one m16 atom");

// -------------------------------------------------------------------------------------------------
// Patch marker. scripts/patch_mixed_nvfp4_gemm.py --mark-patched FQ_SM120_PATCH_STATE rewrites
// "UNPATCHED" to "PATCHED__" (same length) in the file bytes after a verified SASS patch. volatile
// so the compiler cannot constant-fold the comparison against the initializer.
// -------------------------------------------------------------------------------------------------
extern "C" {
__attribute__((used, visibility("hidden")))
volatile char fq_sm120_patch_state[] = "FQ_SM120_PATCH_STATE:UNPATCHED";
}

bool library_is_patched() {
  static char const kPatched[] = "FQ_SM120_PATCH_STATE:PATCHED__";
  for (size_t i = 0; i + 1 < sizeof(kPatched); ++i) {
    if (fq_sm120_patch_state[i] != kPatched[i]) return false;
  }
  return true;
}

// -------------------------------------------------------------------------------------------------
// Granule self-check: derive, from the kernel's own TiledMma, which CTA-tile rows of A each
// (warp, m-atom) covers -- the unit the kernel reads one format flag for -- and require it to be
// exactly the 16-aligned contiguous blocks this API's logical fmt_b map assumes.
// -------------------------------------------------------------------------------------------------
bool granule_map_is_contiguous(char* why, size_t why_len) {
  MixedTiledMma tiled_mma;
  int const mn_extent = int(size<0>(ThreadBlockShape{}));
  int const k_extent = int(size<2>(ThreadBlockShape{}));
  Tensor id = make_identity_tensor(make_shape(mn_extent, k_extent));
  int const num_threads = int(size(typename MixedTiledMma::ThrLayoutVMNK{}));
  std::map<std::pair<int, int>, std::set<int>> coverage;  // (warp, m-atom) -> rows
  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto frag = thr_mma.partition_A(id);
    int const num_atoms = int(size<1>(frag));
    for (int atom = 0; atom < num_atoms; ++atom) {
      auto& dst = coverage[{tid / 32, atom / MIXFP4_A_ATOMS_PER_GRANULE}];
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
      std::snprintf(why, why_len, "CTA row %d is not covered by any MMA atom", r);
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

// d [M, N] row-major <- dt [N_pad, M_pad] row-major (bf16 as raw 16-bit).
__global__ void transpose_out(uint16_t const* __restrict__ dt, int m_pad,
                              uint16_t* __restrict__ d, int M, int N) {
  __shared__ uint16_t tile[32][33];
  int const m0 = int(blockIdx.x) * 32, n0 = int(blockIdx.y) * 32;
  for (int j = int(threadIdx.y); j < 32; j += int(blockDim.y)) {
    int const n = n0 + j, m = m0 + int(threadIdx.x);
    if (n < N && m < M) tile[j][threadIdx.x] = dt[size_t(n) * size_t(m_pad) + size_t(m)];
  }
  __syncthreads();
  for (int j = int(threadIdx.y); j < 32; j += int(blockDim.y)) {
    int const m = m0 + j, n = n0 + int(threadIdx.x);
    if (m < M && n < N) d[size_t(m) * size_t(N) + size_t(n)] = tile[threadIdx.x][j];
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
  int Mk, Nk, Kk;    // CUTLASS problem: Mk = Np (weights), Nk = Mp (tokens), Kk = Kp
  bool copy_w, copy_x;
  LayoutSFA layout_sfa;
  LayoutSFB layout_sfb;
  size_t sfa_bytes, sfb_bytes;
};

Plan make_plan(int M, int N, int K) {
  Plan p{};
  p.M = M; p.N = N; p.K = K;
  p.Mk = fq::round_up(N, kTileM);
  p.Nk = fq::round_up(M, kTileN);
  p.Kk = fq::round_up(K, kTileK);
  p.copy_w = (p.Mk != N) || (p.Kk != K);
  p.copy_x = (p.Nk != M) || (p.Kk != K);
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
  b.w   = p.copy_w ? c.take<uint8_t>(size_t(p.Mk) * size_t(p.Kk / 2)) : nullptr;
  b.x   = p.copy_x ? c.take<uint8_t>(size_t(p.Nk) * size_t(p.Kk / 2)) : nullptr;
  b.sfa = c.take<uint8_t>(p.sfa_bytes);
  b.sfb = c.take<uint8_t>(p.sfb_bytes);
  b.dt  = c.take<uint16_t>(size_t(p.Mk) * size_t(p.Nk) * sizeof(uint16_t));
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
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm120: M, N, K must be >= 1 (got %d, %d, %d)", M, N, K);
  }
  if (N % kGranRows != 0 || K % kGranCols != 0) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120: requires N %% 16 == 0 and K %% 64 == 0 (got N=%d, K=%d)", N, K);
  }
  // Transpose grid.y = Np/32 and int-indexed CUTLASS extents.
  if (N > 65535 * 32 || M > (1 << 30) || K > (1 << 30)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120: shape too large (M=%d, N=%d, K=%d)", M, N, K);
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
      return fq::fail(FQ_ERR_CUDA, "fq_sm120: cudaDeviceGetAttribute(SM count) failed: %s", cudaGetErrorString(e));
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
    return fq::fail(FQ_ERR_INTERNAL, "fq_sm120: kernel format granule is not the contiguous %dx%d "
                    "block this API assumes: %s", kGranRows, kGranCols, why);
  }
  return FQ_OK;
}

}  // namespace fq120

extern "C" FQ_API void fq_sm120_weight_granule(int* rows, int* cols) {
  if (rows) *rows = fq120::kGranRows;
  if (cols) *cols = fq120::kGranCols;
}

extern "C" FQ_API size_t fq_sm120_workspace_bytes(int M, int N, int K) {
  if (M < 1 || N < 1 || K < 64 || N % fq120::kGranRows || K % fq120::kGranCols) return 0;
  int dev = 0;
  fq120::DeviceInfo info;
  if (fq120::device_info(&dev, &info) != FQ_OK) return 0;  // fq_last_error() says why
  auto const plan = fq120::make_plan(M, N, K);
  return fq120::carve(nullptr, plan, fq120::cutlass_workspace_bytes(plan, dev, info.sm_count)).bytes;
}

extern "C" FQ_API int fq_sm120_mixfp4_gemm(const uint8_t* a, const uint8_t* sa,
                                           const uint8_t* b, const uint8_t* sb,
                                           const uint8_t* fmt_b, void* d_bf16,
                                           int M, int N, int K, int fmt_a,
                                           void* workspace, size_t workspace_bytes,
                                           fq_stream_t stream_handle) {
  using namespace fq120;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

  if (!library_is_patched()) {
    return fq::fail(FQ_ERR_NOT_PATCHED, "fq_sm120: this is the UNPATCHED library -- every E0M3 site "
                    "would decode as E2M1. Load libflipquant_sm120.so produced by the build's "
                    "patch step (scripts/patch_mixed_nvfp4_gemm.py --mark-patched).");
  }
  if (!a || !sa || !b || !sb || !fmt_b || !d_bf16) {
    return fq::fail(FQ_ERR_INVALID_ARGUMENT, "fq_sm120: null tensor pointer");
  }
  if (fmt_a != 1) {
    return fq::fail(fmt_a == 0 ? FQ_ERR_UNSUPPORTED : FQ_ERR_INVALID_ARGUMENT,
                    "fq_sm120: fmt_a=%d not supported -- this build pins activations (CUTLASS "
                    "operand B) to E2M1 (MIXFP4_B_ALL_E2M1); only fmt_a=1 is implemented", fmt_a);
  }
  if (int rc = validate_shape(M, N, K)) return rc;
  if (!fq::aligned(a, 16) || !fq::aligned(b, 16)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120: a and b must be 16-byte aligned");
  }
  if (!fq::aligned(d_bf16, 2)) {
    return fq::fail(FQ_ERR_UNSUPPORTED, "fq_sm120: d must be 2-byte aligned");
  }
  if (int rc = self_check_once()) return rc;

  int dev = 0;
  DeviceInfo info;
  if (int rc = device_info(&dev, &info)) return rc;
  if (info.major != 12 || info.minor != 0) {
    return fq::fail(FQ_ERR_DEVICE, "fq_sm120: device %d is sm_%d%d; this library contains patched "
                    "sm_120a SASS only (no PTX)", dev, info.major, info.minor);
  }

  Plan const plan = make_plan(M, N, K);
  size_t const cws = cutlass_workspace_bytes(plan, dev, info.sm_count);
  size_t const need = carve(nullptr, plan, cws).bytes;
  if (!workspace || workspace_bytes < need || !fq::aligned(workspace, fq::kWorkspaceAlign)) {
    return fq::fail(FQ_ERR_WORKSPACE, "fq_sm120: workspace must be >= %zu bytes and 256-byte "
                    "aligned (got %zu bytes at %p)", need, workspace_bytes, workspace);
  }
  Buffers const w = carve(workspace, plan, cws);

  constexpr int kBlock = 256;
  // Operands: consume in place when already tile-shaped, else zero-pad into workspace.
  void const* A = b;  // weights -> CUTLASS A
  void const* B = a;  // activations -> CUTLASS B
  if (plan.copy_w) {
    size_t const total = size_t(plan.Mk) * size_t(plan.Kk / 32);
    pad_copy_rows<<<grid_for(total, kBlock), kBlock, 0, stream>>>(
        reinterpret_cast<uint4 const*>(b), N, K / 32, reinterpret_cast<uint4*>(w.w), plan.Mk, plan.Kk / 32);
    A = w.w;
  }
  if (plan.copy_x) {
    size_t const total = size_t(plan.Nk) * size_t(plan.Kk / 32);
    pad_copy_rows<<<grid_for(total, kBlock), kBlock, 0, stream>>>(
        reinterpret_cast<uint4 const*>(a), M, K / 32, reinterpret_cast<uint4*>(w.x), plan.Nk, plan.Kk / 32);
    B = w.x;
  }
  {
    size_t const total_a = size_t(plan.Mk) * size_t(plan.Kk / 16);
    build_scales<<<grid_for(total_a, kBlock), kBlock, 0, stream>>>(
        sb, N, K / 16, w.sfa, plan.layout_sfa, plan.Mk, plan.Kk / 16, fmt_b, K / 64);
    size_t const total_b = size_t(plan.Nk) * size_t(plan.Kk / 16);
    build_scales<<<grid_for(total_b, kBlock), kBlock, 0, stream>>>(
        sa, M, K / 16, w.sfb, plan.layout_sfb, plan.Nk, plan.Kk / 16,
        static_cast<uint8_t const*>(nullptr), 0);
  }
  if (int rc = fq::check_launch("fq_sm120 operand/scale preparation")) return rc;

  auto args = make_args(plan, A, B, w.sfa, w.sfb, w.dt, dev, info.sm_count);
  Gemm gemm;
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    return fq::fail(FQ_ERR_INTERNAL, "fq_sm120: CUTLASS can_implement failed: %s", cutlassGetStatusString(st));
  }
  st = gemm.initialize(args, w.cutlass_ws, stream);
  if (st != cutlass::Status::kSuccess) {
    return fq::fail(FQ_ERR_INTERNAL, "fq_sm120: CUTLASS initialize failed: %s", cutlassGetStatusString(st));
  }
  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) {
    return fq::fail(FQ_ERR_CUDA, "fq_sm120: CUTLASS run failed: %s (%s)", cutlassGetStatusString(st),
                    cudaGetErrorString(cudaGetLastError()));
  }

  dim3 tgrid((M + 31) / 32, (N + 31) / 32);
  transpose_out<<<tgrid, dim3(32, 8), 0, stream>>>(w.dt, plan.Nk, static_cast<uint16_t*>(d_bf16), M, N);
  return fq::check_launch("fq_sm120 transpose");
}
