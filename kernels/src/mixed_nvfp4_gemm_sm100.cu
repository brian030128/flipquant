// Mixed E0M3/E2M1 block-scaled NVFP4 GEMM on datacenter Blackwell (sm_100a).
//
// This is the sm_100 counterpart of src/mixed_nvfp4_gemm.cu, and it is deliberately much smaller,
// because the hardware problem that dominated the sm_120 work does not exist here.
//
// On sm_120 the operand format is two bits of the *SASS opcode* of `mma.sync...block_scale`.
// Choosing it at runtime therefore meant branching between distinct instructions, ptxas
// if-converted those branches, and a predicated-off OMMA still consumed a tensor-pipe issue slot
// -- costing 2x before any of the mitigation work. PTX cannot even spell E0M3, so the binary had
// to be patched after compilation.
//
// On sm_100 the format lives in the UMMA *instruction descriptor*, which `tcgen05.mma` takes in a
// general-purpose register:
//
//     asm volatile("tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
//                  "[%0], %1, %2, %3, [%5], [%6], p;"
//                  :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
//                     "r"(uint32_t(idescE>>32)),    // <-- the format bits, in a register
//                     ...);
//                                     (cute/arch/mma_sm100_umma.hpp)
//
// So selecting E0M3 is a register write. No branch, no if-conversion, no wasted issue slot, and
// no cubin patching -- the same compiled instruction does either format. CuTe already models this:
// `UMMA::InstrDescriptorBlockScaled idesc_` is a non-static member of the MMA_Traits, and the
// mainloop already rewrites it at runtime for CUTLASS's own type-erased dtype feature.
//
// What this binary establishes, in order:
//   1. that format enum 0 on the mxf4nvf4 path is the evenly-spaced INT4/E0M3 codebook, by
//      checking the result against a host reference built on that codebook; and
//   2. what forcing it costs, against src/nvfp4_gemm_sm100.cu on the same shape.
//
// Uniform over the launch. Per-granule selection is the next step and rides on the same
// descriptor write.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "collective/dispatch_policy_sm100_mixed.hpp"
#include "collective/sm100_blockscaled_mma_mixed.hpp"

#include "helper.h"
#include "gemm_benchmark.hpp"
#include "sm100_gemm_launch.hpp"

using namespace cute;

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
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

#ifndef MIXFP4_TILE_M
#define MIXFP4_TILE_M 256
#endif
#ifndef MIXFP4_TILE_N
#define MIXFP4_TILE_N 256
#endif
#ifndef MIXFP4_CLUSTER_M
#define MIXFP4_CLUSTER_M 2
#endif
using MmaTileShape = Shape<Int<MIXFP4_TILE_M>, Int<MIXFP4_TILE_N>, _256>;
#ifndef MIXFP4_STAGE_DELTA
#define MIXFP4_STAGE_DELTA 0
#endif
#ifdef MIXFP4_SUB_N
// MMA-width experiment: the tuned 256x256x256 2-SM kernel, each 256-wide MMA issued as
// 256/MIXFP4_SUB_N narrower ones with one B format per MIXFP4_SUB_N x 64 tile from a map.
// 2-SM: the tuned 256x256 tile, cluster 2x1. 1-SM: a 128x256 tile, the only way to issue N=8.
static_assert(MIXFP4_TILE_N == 256 &&
              ((MIXFP4_TILE_M == 256 && MIXFP4_CLUSTER_M == 2) ||
               (MIXFP4_TILE_M == 128 && MIXFP4_CLUSTER_M == 1)),
              "the MMA-width experiment splits a 256-wide MMA");
static std::string g_sub_pattern = "random";

// The deployed map: one bit per 256x64 tile, 1 = E0M3 (the map file's convention), rows of
// ceil(K/2048) uint32 words, bit j of word w = K64 block 32w + j. `logical` holds one uint32 per
// (tile row, K64 block) with bit 0 = 1 for E2M1, `logical_stride` entries per row.
static std::vector<uint32_t> pack_tile_bits(std::vector<uint32_t> const &logical, int rows, int k,
                                            int logical_stride) {
  int const words = (k / 64 + 31) / 32;
  std::vector<uint32_t> packed(size_t(rows) * size_t(words), 0u);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < k / 64; ++c) {
      if ((logical[size_t(r) * logical_stride + c] & 1u) == 0u) {
        packed[size_t(r) * words + c / 32] |= 1u << (c % 32);
      }
    }
  }
  return packed;
}
#endif
using ClusterShape = Shape<Int<MIXFP4_CLUSTER_M>, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    Shape<_128, Int<(MIXFP4_TILE_N == 256 ? 64 : 32)>>,
    ElementAccumulator, ElementAccumulator,
    void, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cute::conditional_t<MIXFP4_TILE_M == 256,
        cutlass::epilogue::TmaWarpSpecialized2Sm,
        cutlass::epilogue::TmaWarpSpecialized1Sm>>::CollectiveOp;

// CUTLASS's own mainloop, using the same epilogue as the stock *_fast configuration. We never launch it;
// we only harvest the twenty-odd template arguments CollectiveBuilder derived, so that the mixed
// mainloop differs from the stock one in the dispatch policy and nothing else.
using StdMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// Swap the dispatch policy for our structurally identical tag, which is what routes the type to
// the forked mainloop specialization in collective/sm100_blockscaled_mma_mixed.hpp. Pattern
// matching on the whole parameter pack means a CUTLASS bump that adds a parameter is a compile
// error here rather than a silent divergence.
template <class T>
struct RebindToMixed;

template <int Stages, int SchedPipe, int AccumPipe, class Cluster, class Arch, class... Rest>
struct RebindToMixed<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<Stages, SchedPipe, AccumPipe,
                                                                  Cluster, Arch>,
    Rest...>> {
  // MIXFP4_STAGE_DELTA: fewer stages when the padded SFB no longer fits the builder's count.
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledMixed<Stages - MIXFP4_STAGE_DELTA,
                                                                         SchedPipe, AccumPipe, Cluster,
                                                                         Arch>,
      Rest...>;
};

using CollectiveMainloop = typename RebindToMixed<StdMainloop>::type;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

template <typename T>
auto make_iterator(T *ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename Element, typename Layout>
void initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  constexpr int bits = cutlass::sizeof_bits<Element>::value;
  double scope_max, scope_min;
  if constexpr (bits <= 6) {
    scope_max = 6; scope_min = -6;
  } else if constexpr (cute::is_same_v<Element, cutlass::float_ue4m3_t> ||
                       cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
    scope_max = 4; scope_min = 1;
  } else {
    scope_max = 4; scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, bits <= 6 ? 1 : 0);
}

// E2M1 and E0M3 index the *same* 4-bit nibble through different codebooks:
//   E2M1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
//   E0M3 magnitudes: 0, 1,   2, 3,   4, 5, 6, 7   (evenly spaced signed integers, i.e. INT4)
// both with a sign bit. Recovering the nibble from an E2M1-decoded value and re-reading it under
// the E0M3 codebook therefore reproduces what an E0M3 instruction computes, without needing raw
// sub-byte access. (Same construction as the sm_120 driver, where it is checked against hardware.)
static constexpr float kE2m1Magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static float reinterpret_e2m1_as_e0m3(float v) {
  float const mag = std::fabs(v);
  int nibble = 0;
  for (int i = 0; i < 8; ++i) {
    if (mag == kE2m1Magnitudes[i]) { nibble = i; break; }
  }
  return std::signbit(v) ? -float(nibble) : float(nibble);
}

static float decode_scale(uint8_t raw) {
  cutlass::float_ue4m3_t s{};
  s.raw() = uint8_t(raw & 0x7fu);
  return float(s);
}

static uint8_t parse_fmt(char const *s, uint8_t dflt) {
  if (s == nullptr) { return dflt; }
  std::string v = s;
  if (v == "e2m1" || v == "1") { return cutlass::gemm::collective::MixedFmt::kE2M1; }
  if (v == "e0m3" || v == "int4" || v == "0") { return cutlass::gemm::collective::MixedFmt::kE0M3; }
  std::cerr << "unknown format '" << v << "' (want e2m1 or e0m3)" << std::endl;
  std::exit(2);
}

static char const *fmt_name(uint8_t f) {
  return f == cutlass::gemm::collective::MixedFmt::kE0M3 ? "E0M3" : "E2M1";
}

int run(int m, int n, int k, uint8_t fmt_a, uint8_t fmt_b, int warmup_iters, int bench_iters,
        bool check, int raster, int swizzle) {
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_a = make_layout(make_shape(m, k, 1), stride_a);
  auto layout_b = make_layout(make_shape(n, k, 1), stride_b);
  auto layout_c = make_layout(make_shape(m, n, 1), stride_c);
  auto layout_d = make_layout(make_shape(m, n, 1), stride_d);
  LayoutSFA layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_a;
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfa;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_b;
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfb;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_c;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_d;

  block_a.reset(cutlass::make_Coord(size(layout_a)));
  block_b.reset(cutlass::make_Coord(size(layout_b)));
  block_c.reset(cutlass::make_Coord(size(layout_c)));
  block_d.reset(cutlass::make_Coord(size(layout_d)));
  block_sfa.reset(cutlass::make_Coord(size(filter_zeros(layout_sfa))));
  block_sfb.reset(cutlass::make_Coord(size(filter_zeros(layout_sfb))));

  if (check) {
    initialize_block(block_a.host_view(), 2021);
    initialize_block(block_b.host_view(), 2022);
    initialize_block(block_c.host_view(), 2023);
    initialize_block(block_sfa.host_view(), 2024);
    initialize_block(block_sfb.host_view(), 2025);

    block_a.sync_device();
    block_b.sync_device();
    block_c.sync_device();
    block_sfa.sync_device();
    block_sfb.sync_device();
  } else {
    CUDA_CHECK(cudaMemset(block_a.device_data(), 0x22, size_t(m) * k / 2));
    CUDA_CHECK(cudaMemset(block_b.device_data(), 0x22, size_t(n) * k / 2));
    CUDA_CHECK(cudaMemset(block_c.device_data(), 0, size_t(m) * n * sizeof(ElementC)));
    CUDA_CHECK(cudaMemset(block_sfa.device_data(), 0x3c, block_sfa.capacity()));
    CUDA_CHECK(cudaMemset(block_sfb.device_data(), 0x3c, block_sfb.capacity()));
  }

  float alpha = 1.0f, beta = 0.0f;

  typename Gemm::GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.ptr_A = block_a.device_data();
  mainloop_args.dA = stride_a;
  mainloop_args.ptr_B = block_b.device_data();
  mainloop_args.dB = stride_b;
  mainloop_args.ptr_SFA = block_sfa.device_data();
  mainloop_args.layout_SFA = layout_sfa;
  mainloop_args.ptr_SFB = block_sfb.device_data();
  mainloop_args.layout_SFB = layout_sfb;
  mainloop_args.fmt_a = fmt_a;
  mainloop_args.fmt_b = fmt_b;
#ifdef MIXFP4_SUB_N
  constexpr int kSubN = MIXFP4_SUB_N;
  // N and K tails: TMA zero-fills out-of-bounds B rows, K columns and scale factors (zero scales
  // make the padding contribute nothing) and the epilogue masks the output, so the per-tile kernel
  // (kSubN = 256) takes any N and any K that is a multiple of 64. Narrower sub-MMAs permute B rows
  // within whole 256-row tiles and pad the scale factors per tile, so they keep N % 256 == 0.
  if (k % 64 || (kSubN < 256 && n % 256)) {
    std::fprintf(stderr, "Need K %% 64 == 0, and N %% 256 == 0 below MMA width 256.\n");
    return 2;
  }
  // One uint32 per (256-wide N tile, K64 block): bit g = format of sub-MMA g (1 = E2M1), i.e. of
  // logical channels [g*kSubN, (g+1)*kSubN) of the tile.
  // Rows: ceil(N/256) tiles, the last possibly partial. Columns: one entry per K64 block, padded
  // to whole 256-deep K-tiles because the kernel loads four entries per K-tile.
  int const map_stride = (k + 255) / 256 * 4;
  std::vector<uint32_t> sub_map(size_t((n + 255) / 256) * size_t(map_stride));
  {
    std::mt19937 rng(2026);
    uint32_t const used = (256 / kSubN) >= 32 ? 0xFFFFFFFFu : ((1u << (256 / kSubN)) - 1u);
    for (size_t t = 0; t < sub_map.size(); ++t) {
      uint32_t bits = g_sub_pattern == "e2m1" ? ~0u : g_sub_pattern == "e0m3" ? 0u
                    : g_sub_pattern == "checker" ? ((t % 2) ? 0xAAAAAAAAu : 0x55555555u) : uint32_t(rng());
      sub_map[t] = bits & used;
    }
  }
  std::vector<uint32_t> const device_map = kSubN == 256
      ? pack_tile_bits(sub_map, (n + 255) / 256, k, map_stride) : sub_map;
  cutlass::device_memory::allocation<uint32_t> sub_map_dev(device_map.size());
  CUDA_CHECK(cudaMemcpy(sub_map_dev.get(), device_map.data(), device_map.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  mainloop_args.sub_fmt = sub_map_dev.get();
  mainloop_args.sub_fmt_stride = kSubN == 256 ? (k / 64 + 31) / 32 : map_stride;
  // Sub-MMA g of the 2-SM pair reads rows [g*kSubN/2, (g+1)*kSubN/2) of each CTA's 128-row B half
  // and writes accumulator columns [g*kSubN, (g+1)*kSubN): first the CTA-0 rows, then the CTA-1
  // rows. Storing logical channel g*kSubN + h*kSubN/2 + j at row h*128 + g*kSubN/2 + j makes every
  // output column, scale and map bit refer to its natural channel. A one-time offline weight
  // permutation; the identity at kSubN = 256. MIXFP4_SUB_NO_PERMUTE=1 skips it (negative control).
  // A 1-SM sub-MMA reads contiguous rows, so no permutation is needed there.
  auto logical_channel = [](int row) {
    if (MIXFP4_CLUSTER_M == 1) { return row; }
    int const tile = row / 256, h = (row % 256) / 128, l = row % 128;
    int const g = l / (kSubN / 2), j = l % (kSubN / 2);
    return tile * 256 + g * kSubN + h * (kSubN / 2) + j;
  };
#ifdef MIXFP4_SFB_PAD
  // Padded B scale factors: 512 virtual channels per 256-wide tile (see mixfp4_issue_sub_pad).
  // Logical channel t = g*kSubN + j of a tile goes to chunk c, position 2p, lane kSubN*q + j, where
  // g = c*2G + 2q + p and G = 32/kSubN lane groups per column. N >= 64 keeps t in place.
  auto virtual_channel = [](int t) {
    if (kSubN >= 64) { return t; }
    int const G = 32 / kSubN, g = t / kSubN, j = t % kSubN;
    int const c = g / (2 * G), r = g % (2 * G), q = r / 2, p = r % 2;
    return 128 * c + 32 * (2 * p) + kSubN * q + j;
  };
  LayoutSFB layout_sfb_pad = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, 2 * n, k, 1));
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfb_pad;
  block_sfb_pad.reset(cutlass::make_Coord(size(filter_zeros(layout_sfb_pad))));
  if (check) {
    std::memset(block_sfb_pad.host_data(), 0, block_sfb_pad.capacity());
    auto logical = make_tensor(block_sfb.host_data(), layout_sfb);
    auto padded = make_tensor(block_sfb_pad.host_data(), layout_sfb_pad);
    for (int j = 0; j < n; ++j) {
      int const v = (j / 256) * 512 + virtual_channel(j % 256);
      for (int kk = 0; kk < k; kk += 16) { padded(v, kk, 0) = logical(j, kk, 0); }
    }
    block_sfb_pad.sync_device();
  } else {
    CUDA_CHECK(cudaMemset(block_sfb_pad.device_data(), 0x3c, block_sfb_pad.capacity()));
  }
  mainloop_args.ptr_SFB = block_sfb_pad.device_data();
  mainloop_args.layout_SFB = layout_sfb_pad;
#endif
  cutlass::device_memory::allocation<uint8_t> permuted_b(check ? size_t(n) * size_t(k) / 2 : 0);
  if (check) {
    bool const permute = std::getenv("MIXFP4_SUB_NO_PERMUTE") == nullptr;
    if (!permute) { std::printf("  [negative control: B rows not permuted -- this SHOULD fail below N=256]\n"); }
    auto const *src = reinterpret_cast<uint8_t const *>(block_b.device_data());
    size_t const row_bytes = size_t(k) / 2;
    for (int row = 0; row < n; ++row) {
      CUDA_CHECK(cudaMemcpy(permuted_b.get() + size_t(row) * row_bytes,
                            src + size_t(permute ? logical_channel(row) : row) * row_bytes,
                            row_bytes, cudaMemcpyDeviceToDevice));
    }
    mainloop_args.ptr_B = reinterpret_cast<decltype(mainloop_args.ptr_B)>(permuted_b.get());
  }
#endif

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop_args,
      {{alpha, beta}, block_c.device_data(), stride_c, block_d.device_data(), stride_d}};

  using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
  arguments.scheduler.raster_order = raster == 1 ? Raster::AlongM
                                    : raster == 2 ? Raster::AlongN : Raster::Heuristic;
  arguments.scheduler.max_swizzle_size = swizzle;

  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  mixfp4::Sm100GemmLaunch<Gemm> launch(gemm_op);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  if (check) {
    // Decode both operands to scaled floats under the format the kernel was told to use, then do
    // a plain float GEMM. Decoding up front keeps the O(M*N*K) inner loop free of sub-byte and
    // scale-factor address arithmetic.
    auto tensor_a = make_tensor(make_iterator(block_a.host_data()), layout_a);
    auto tensor_sfa = make_tensor(block_sfa.host_data(), layout_sfa);
    auto tensor_b = make_tensor(make_iterator(block_b.host_data()), layout_b);
    auto tensor_sfb = make_tensor(block_sfb.host_data(), layout_sfb);
    auto tensor_c = make_tensor(make_iterator(block_c.host_data()), layout_c);

    std::vector<float> a_dec(size_t(m) * size_t(k));
    std::vector<float> b_dec(size_t(n) * size_t(k));
    // Negative control. "All four combinations PASSED" only means something if a WRONG codebook
    // makes the check fail -- otherwise the test could be passing because neither the kernel nor
    // the reference is honouring the format at all. MIXFP4_REF_FMT_A/B decouple the reference's
    // codebook from the kernel's so a deliberate mismatch can be demanded to fail.
    uint8_t const ref_fmt_a = parse_fmt(std::getenv("MIXFP4_REF_FMT_A"), fmt_a);
    uint8_t const ref_fmt_b = parse_fmt(std::getenv("MIXFP4_REF_FMT_B"), fmt_b);
    if (ref_fmt_a != fmt_a || ref_fmt_b != fmt_b) {
      std::printf("  [negative control: reference decodes A=%s B=%s while the kernel was told "
                  "A=%s B=%s -- this SHOULD fail]\n",
                  fmt_name(ref_fmt_a), fmt_name(ref_fmt_b), fmt_name(fmt_a), fmt_name(fmt_b));
    }
    bool const a_is_e0m3 = (ref_fmt_a == cutlass::gemm::collective::MixedFmt::kE0M3);
    bool const b_is_e0m3 = (ref_fmt_b == cutlass::gemm::collective::MixedFmt::kE0M3);
#ifdef MIXFP4_SUB_N
    // The map decides B's format unless MIXFP4_REF_FMT_B forces one codebook (negative control:
    // with a mixed map, decoding everything as E2M1 or E0M3 SHOULD fail).
    bool const ref_uses_map = std::getenv("MIXFP4_REF_FMT_B") == nullptr;
    auto b_e0m3_at = [&](int j, int kk) {
      if (!ref_uses_map) { return b_is_e0m3; }
      uint32_t const bits = sub_map[size_t(j / 256) * size_t(map_stride) + size_t(kk / 64)];
      return ((bits >> ((j % 256) / kSubN)) & 1u) == 0u;
    };
#else
    auto b_e0m3_at = [&](int, int) { return b_is_e0m3; };
#endif

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
      for (int kk = 0; kk < k; ++kk) {
        uint8_t const sf = tensor_sfa(i, kk, 0).raw();
        float v = float(static_cast<cutlass::float_e2m1_t>(tensor_a(i, kk, 0)));
        if (a_is_e0m3) { v = reinterpret_e2m1_as_e0m3(v); }
        a_dec[size_t(i) * size_t(k) + size_t(kk)] = v * decode_scale(sf);
      }
    }
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
      for (int kk = 0; kk < k; ++kk) {
        uint8_t const sf = tensor_sfb(j, kk, 0).raw();
        float v = float(static_cast<cutlass::float_e2m1_t>(tensor_b(j, kk, 0)));
        if (b_e0m3_at(j, kk)) { v = reinterpret_e2m1_as_e0m3(v); }
        b_dec[size_t(j) * size_t(k) + size_t(kk)] = v * decode_scale(sf);
      }
    }

    block_d.sync_host();
    auto tensor_d = make_tensor(make_iterator(block_d.host_data()), layout_d);

    // Relative Frobenius norm, not element-wise relative error: the GPU sums K in a different
    // order and rounds to bfloat16, so near-cancelling elements can show a large relative error
    // while the result is right. The norm ratio is insensitive to that but still moves by order 1
    // if an operand decodes under the wrong codebook.
    double num = 0.0, den = 0.0, max_abs_err = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : num, den) reduction(max : max_abs_err)
    for (int i = 0; i < m; ++i) {
      float const *ap = &a_dec[size_t(i) * size_t(k)];
      for (int j = 0; j < n; ++j) {
        float const *bp = &b_dec[size_t(j) * size_t(k)];
        float acc = 0.f;
        for (int kk = 0; kk < k; ++kk) { acc += ap[kk] * bp[kk]; }
        float const ref = alpha * acc +
                          beta * float(static_cast<cutlass::bfloat16_t>(tensor_c(i, j, 0)));
        float const got = float(static_cast<cutlass::bfloat16_t>(tensor_d(i, j, 0)));
        double const d = double(got) - double(ref);
        num += d * d;
        den += double(ref) * double(ref);
        max_abs_err = std::max(max_abs_err, std::fabs(d));
      }
    }
    double const rel = (den > 0.0) ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
    bool const passed = (rel < 2e-2);
    std::printf("Correctness (A=%s B=%s): %s  rel_fro=%.3e  max_abs=%.3e\n",
                fmt_name(fmt_a), fmt_name(fmt_b), passed ? "PASSED" : "FAILED", rel, max_abs_err);
    if (!passed) { return -1; }
  }

  std::vector<ElementD> checked_output;
  if (check) {
    checked_output.assign(block_d.host_data(), block_d.host_data() + size(layout_d));
  }
  double ms = mixfp4::benchmark_ms(
      [&](cudaStream_t stream) {
        launch(stream);
      },
      warmup_iters, bench_iters);
  if (check) {
    block_d.sync_host();
    if (!std::equal(checked_output.begin(), checked_output.end(), block_d.host_data())) {
      std::fprintf(stderr, "Correctness: FAILED after repeated launches\n");
      return 1;
    }
  }

  double const tflops =
      2.0 * double(m) * double(n) * double(k) / (ms / 1.0e3) / 1.0e12;

  std::printf("Problem size: %dx%dx%d  (tile %dx%dx256, cluster %dx1x1)\n", m, n, k,
              MIXFP4_TILE_M, MIXFP4_TILE_N, MIXFP4_CLUSTER_M);
#ifdef MIXFP4_SUB_N
  std::printf("MMA: %dx%dx64 %d-SM, %d per K64 block; B formats: one per %dx64 tile (%s map)\n",
              MIXFP4_TILE_M, MIXFP4_SUB_N, MIXFP4_CLUSTER_M, 256 / MIXFP4_SUB_N, MIXFP4_SUB_N,
              g_sub_pattern.c_str());
#endif
  std::printf("Pipeline stages: %d\n", int(CollectiveMainloop::DispatchPolicy::Stages));
  std::printf("Formats: A=%s B=%s\n", fmt_name(fmt_a), fmt_name(fmt_b));
  std::printf("Scheduler: raster=%d swizzle=%d; timing=%s; PDL=%s\n", raster, swizzle,
              mixfp4::benchmark_uses_graph() ? "graph" : "launch",
              mixfp4::sm100_uses_pdl() ? "on" : "off");
  std::printf("Avg latency: %f ms\n", ms);
  std::printf("Throughput: %.1f TFLOP/s (mixed sm100)\n", tflops);
  return 0;
}

// ---------------------------------------------------------------------------------------------
// Per-forward GEMM timing (paper Appendix D.5 on SM100; HackMD item #1).
//
// --forward=SPEC --out=JSONL [--tokens=1,4,...] [--reps=30] [--warmup=3] [--flush-mib=512]
// SPEC lines: "<projection> <tagset> <N> <K> <modules> <mapfile|->". For every line and token count
// T it times the GEMM D[T,N] = A[T,K] * W[N,K]^T (weights on B) as D.5 does: before each call a
// 512 MiB read evicts the L2 and the activations are read back in (as the untimed activation
// quantizer would leave them), the weights rotate over copies whose total exceeds four times the
// L2, and the time is the CUPTI kernel time (CUDA events if built without CUPTI). The per-tile
// build reads its B formats from <mapfile>: one bit per 256x64 tile (1 = E0M3), rows of
// ceil(K/2048) little-endian uint32 words, bit j of word w = K64 block 32w + j. The stock build
// ignores it. One JSON line per (line, T).
// ---------------------------------------------------------------------------------------------
#if defined(MIXFP4_HAVE_CUPTI)
#include <cupti.h>
#endif
#include <fstream>
#include <memory>
#include <sstream>

namespace forward_bench {

#if defined(MIXFP4_HAVE_CUPTI)
std::vector<std::pair<std::string, double>> g_kernels;  // (name, microseconds)

void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
  *size = size_t(16) << 20;
  *buffer = static_cast<uint8_t *>(std::aligned_alloc(8, *size));
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t *buffer, size_t, size_t valid) {
  CUpti_Activity *record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto const *kernel = reinterpret_cast<CUpti_ActivityKernel10 const *>(record);
      g_kernels.emplace_back(kernel->name ? kernel->name : "", double(kernel->end - kernel->start) * 1e-3);
    }
  }
  std::free(buffer);
}

#define CUPTI_CHECK(call)                                                                   \
  do {                                                                                      \
    CUptiResult status_ = (call);                                                           \
    if (status_ != CUPTI_SUCCESS) {                                                         \
      char const *message_ = nullptr;                                                       \
      cuptiGetResultString(status_, &message_);                                             \
      std::fprintf(stderr, "CUPTI error %s at %s:%d\n", message_, __FILE__, __LINE__);      \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)
#endif

// Reads n uint4 words; the sink write never happens but keeps the loads alive.
__global__ void fq_read_kernel(uint4 const *data, size_t n, unsigned *sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint4 v = data[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) { atomicAdd(sink, 1u); }
}

// Pseudo-random bytes (codes) or bytes in [lo, lo + span) (UE4M3 scales from 0.25 to ~1.9).
__global__ void fq_fill_kernel(uint8_t *data, size_t n, uint32_t seed, uint32_t lo, uint32_t span) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 0x9e3779b1u ^ seed;
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
    data[i] = span ? uint8_t(lo + h % span) : uint8_t(h);
  }
}

inline void read_bytes(void const *data, size_t bytes, unsigned *sink, cudaStream_t stream) {
  size_t const words = bytes / 16;
  if (words == 0) { return; }
  int const blocks = int(std::min<size_t>((words + 255) / 256, 4096));
  fq_read_kernel<<<blocks, 256, 0, stream>>>(static_cast<uint4 const *>(data), words, sink);
}

inline void fill(void *data, size_t bytes, uint32_t seed, uint32_t lo = 0, uint32_t span = 0) {
  int const blocks = int(std::min<size_t>((bytes + 255) / 256, 8192));
  fq_fill_kernel<<<blocks, 256>>>(static_cast<uint8_t *>(data), bytes, seed, lo, span);
  CUDA_CHECK(cudaGetLastError());
}

struct Line {
  std::string projection, tagset, mapfile;
  int n = 0, k = 0, modules = 0;
};

int run(char const *spec_path, char const *out_path, std::vector<int> const &tokens, int reps,
        int warmup, int flush_mib) {
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
#if (defined(MIXFP4_SUB_N) && MIXFP4_SUB_N != 256) || defined(MIXFP4_SFB_PAD) || MIXFP4_CLUSTER_M != 2
  std::fprintf(stderr, "--forward times the stock kernel or the deployed 256x64 per-tile kernel only\n");
  return 2;
#endif
  std::vector<Line> lines;
  {
    std::ifstream spec(spec_path);
    std::string text;
    while (std::getline(spec, text)) {
      if (text.empty() || text[0] == '#') { continue; }
      std::istringstream fields(text);
      Line line;
      fields >> line.projection >> line.tagset >> line.n >> line.k >> line.modules >> line.mapfile;
      if (!fields || line.n <= 0 || line.k <= 0 || line.k % 64) {
        std::fprintf(stderr, "bad spec line: %s\n", text.c_str());
        return 2;
      }
      lines.push_back(line);
    }
  }
  std::ofstream out(out_path, std::ios::app);
  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  size_t const l2_bytes = size_t(props.l2CacheSize);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  size_t const flush_bytes = size_t(flush_mib) << 20;
  void *flush = nullptr;
  unsigned *sink = nullptr;
  CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
  fill(flush, flush_bytes, 7);
#if defined(MIXFP4_HAVE_CUPTI)
  CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  char const *timer = "cupti";
#else
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  char const *timer = "cuda-events";
#endif
  for (Line const &line : lines) {
    int const n = line.n, k = line.k;
    // Tags of this projection (per-tile build only): one bit per 256x64 tile, 1 = E0M3.
    std::vector<uint32_t> map;
    int const map_stride = (k / 64 + 31) / 32;
    if (line.mapfile != "-") {
      std::ifstream file(line.mapfile, std::ios::binary);
      map.resize(size_t((n + 255) / 256) * size_t(map_stride));
      file.read(reinterpret_cast<char *>(map.data()), std::streamsize(map.size() * sizeof(uint32_t)));
      if (!file || file.peek() != EOF) {
        std::fprintf(stderr, "%s: expected %zu uint32 entries\n", line.mapfile.c_str(), map.size());
        return 2;
      }
    }
    uint32_t *map_dev = nullptr;
    if (!map.empty()) {
      CUDA_CHECK(cudaMalloc(&map_dev, map.size() * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(map_dev, map.data(), map.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    size_t e0m3_tiles = 0, tiles = 0;
    for (int r = 0; r < (n + 255) / 256; ++r) {
      for (int c = 0; c < k / 64; ++c) {
        ++tiles;
        e0m3_tiles += map.empty() ? 0u : (map[size_t(r) * map_stride + c / 32] >> (c % 32)) & 1u;
      }
    }
    for (int t : tokens) {
      int const m = t;
      StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
      StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
      StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
      StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
      LayoutSFA layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
      LayoutSFB layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
      size_t const a_bytes = size_t(m) * k / 2, b_bytes = size_t(n) * k / 2;
      size_t const sfa_bytes = size(filter_zeros(layout_sfa)), sfb_bytes = size(filter_zeros(layout_sfb));
      size_t const weight_bytes = b_bytes + sfb_bytes;
      // D.5: copies whose total exceeds four times the L2; only warmup + reps of them are ever used.
      int const copies = int(std::min<size_t>((4 * l2_bytes + weight_bytes - 1) / weight_bytes + 1,
                                              size_t(warmup + reps)));
      uint8_t *a = nullptr, *sfa = nullptr, *weights = nullptr;
      ElementD *d = nullptr;
      CUDA_CHECK(cudaMalloc(&a, a_bytes));
      CUDA_CHECK(cudaMalloc(&sfa, sfa_bytes));
      CUDA_CHECK(cudaMalloc(&weights, weight_bytes * copies));
      CUDA_CHECK(cudaMalloc(&d, size_t(m) * n * sizeof(ElementD)));
      fill(a, a_bytes, 11);
      fill(sfa, sfa_bytes, 12, 0x30, 16);
      for (int c = 0; c < copies; ++c) {
        fill(weights + c * weight_bytes, b_bytes, 100 + c);
        fill(weights + c * weight_bytes + b_bytes, sfb_bytes, 200 + c, 0x30, 16);
      }
      std::vector<std::unique_ptr<Gemm>> ops;
      std::vector<cutlass::device_memory::allocation<uint8_t>> workspaces;
      std::vector<std::unique_ptr<mixfp4::Sm100GemmLaunch<Gemm>>> launches;
      for (int c = 0; c < copies; ++c) {
        typename Gemm::GemmKernel::MainloopArguments mainloop_args{};
        mainloop_args.ptr_A = reinterpret_cast<decltype(mainloop_args.ptr_A)>(a);
        mainloop_args.dA = stride_a;
        mainloop_args.ptr_B = reinterpret_cast<decltype(mainloop_args.ptr_B)>(weights + c * weight_bytes);
        mainloop_args.dB = stride_b;
        mainloop_args.ptr_SFA = reinterpret_cast<decltype(mainloop_args.ptr_SFA)>(sfa);
        mainloop_args.layout_SFA = layout_sfa;
        mainloop_args.ptr_SFB = reinterpret_cast<decltype(mainloop_args.ptr_SFB)>(weights + c * weight_bytes + b_bytes);
        mainloop_args.layout_SFB = layout_sfb;
#ifdef MIXFP4_SUB_N
        mainloop_args.sub_fmt = map_dev;
        mainloop_args.sub_fmt_stride = map_stride;
#endif
        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1}, mainloop_args,
                                           {{1.0f, 0.0f}, d, stride_c, d, stride_d}};
        arguments.scheduler.max_swizzle_size = (m >= 8192 || n >= 8192 || k >= 8192) ? 2 : 0;
        ops.push_back(std::make_unique<Gemm>());
        CUTLASS_CHECK(ops.back()->can_implement(arguments));
        workspaces.emplace_back(Gemm::get_workspace_size(arguments));
        CUTLASS_CHECK(ops.back()->initialize(arguments, workspaces.back().get()));
        launches.push_back(std::make_unique<mixfp4::Sm100GemmLaunch<Gemm>>(*ops.back()));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
#if defined(MIXFP4_HAVE_CUPTI)
      CUPTI_CHECK(cuptiActivityFlushAll(1));
      g_kernels.clear();
#endif
      std::vector<double> samples;
      for (int i = 0; i < warmup + reps; ++i) {
        read_bytes(flush, flush_bytes, sink, stream);        // evict the weights from L2
        read_bytes(a, a_bytes, sink, stream);                // activations as the quantizer leaves them
        read_bytes(sfa, sfa_bytes, sink, stream);
#if defined(MIXFP4_HAVE_CUPTI)
        (*launches[i % copies])(stream);
#else
        CUDA_CHECK(cudaEventRecord(start, stream));
        (*launches[i % copies])(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (i >= warmup) { samples.push_back(double(ms) * 1e3); }
#endif
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaGetLastError());
#if defined(MIXFP4_HAVE_CUPTI)
      CUPTI_CHECK(cuptiActivityFlushAll(1));
      std::vector<double> gemm;
      for (auto const &[name, us] : g_kernels) {
        if (name.find("fq_") == std::string::npos) { gemm.push_back(us); }
      }
      if (int(gemm.size()) != warmup + reps) {
        std::fprintf(stderr, "expected %d GEMM kernel records, got %zu\n", warmup + reps, gemm.size());
        return 1;
      }
      samples.assign(gemm.begin() + warmup, gemm.end());
#endif
      std::vector<double> sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      double const median = sorted[sorted.size() / 2];
      out << "{\"projection\":\"" << line.projection << "\",\"tagset\":\"" << line.tagset << "\",\"kernel\":\""
#ifdef MIXFP4_SUB_N
          << "per-tile-256x64"
#else
          << "stock"
#endif
          << "\",\"n\":" << n << ",\"k\":" << k << ",\"modules\":" << line.modules << ",\"tokens\":" << t
          << ",\"e0m3_tiles\":" << e0m3_tiles << ",\"tiles\":" << tiles
          << ",\"map_bytes\":" << map.size() * sizeof(uint32_t)
          << ",\"copies\":" << copies << ",\"timer\":\"" << timer << "\",\"median_us\":" << median
          << ",\"samples_us\":[";
      for (size_t i = 0; i < samples.size(); ++i) { out << (i ? "," : "") << samples[i]; }
      out << "]}\n";
      out.flush();
      std::printf("%-14s %-8s %6dx%-6d T=%-5d %10.2f us\n", line.projection.c_str(), line.tagset.c_str(), n, k, t, median);
      CUDA_CHECK(cudaFree(a));
      CUDA_CHECK(cudaFree(sfa));
      CUDA_CHECK(cudaFree(weights));
      CUDA_CHECK(cudaFree(d));
    }
    if (map_dev) { CUDA_CHECK(cudaFree(map_dev)); }
  }
  CUDA_CHECK(cudaFree(flush));
  CUDA_CHECK(cudaFree(sink));
  return 0;
}

}  // namespace forward_bench

int main(int argc, char const **argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10) {
    std::cerr << "This kernel targets datacenter Blackwell (sm_100); detected compute capability "
              << props.major << "." << props.minor << std::endl;
    return 0;
  }

  int m = 4096, n = 4096, k = 4096;
  int warmup_iters = 10, bench_iters = 50;
  int raster = 0, swizzle = -1;
  bool check = (std::getenv("MIXFP4_SKIP_REF") == nullptr);
  uint8_t fmt_a = parse_fmt(std::getenv("MIXFP4_FMT_A"), cutlass::gemm::collective::MixedFmt::kE2M1);
  uint8_t fmt_b = parse_fmt(std::getenv("MIXFP4_FMT_B"), cutlass::gemm::collective::MixedFmt::kE2M1);

  std::string forward_spec, forward_out;
  std::vector<int> forward_tokens{1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  int forward_reps = 30, forward_warmup = 3, forward_flush_mib = 512;
  std::vector<char const *> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--fmt-a=", 0) == 0) { fmt_a = parse_fmt(a.c_str() + 8, fmt_a); }
    else if (a.rfind("--fmt-b=", 0) == 0) { fmt_b = parse_fmt(a.c_str() + 8, fmt_b); }
    else if (a == "--no-check") { check = false; }
    else if (a == "--check") { check = true; }
    else if (a.rfind("--iters=", 0) == 0) { bench_iters = std::atoi(a.c_str() + 8); }
    else if (a.rfind("--raster=", 0) == 0) { raster = std::atoi(a.c_str() + 9); }
    else if (a.rfind("--swizzle=", 0) == 0) { swizzle = std::atoi(a.c_str() + 10); }
    else if (a.rfind("--forward=", 0) == 0) { forward_spec = a.substr(10); }
    else if (a.rfind("--out=", 0) == 0) { forward_out = a.substr(6); }
    else if (a.rfind("--tokens=", 0) == 0) {
      forward_tokens.clear();
      std::stringstream list(a.substr(9));
      for (std::string item; std::getline(list, item, ',');) { forward_tokens.push_back(std::atoi(item.c_str())); }
    }
    else if (a.rfind("--reps=", 0) == 0) { forward_reps = std::atoi(a.c_str() + 7); }
    else if (a.rfind("--warmup=", 0) == 0) { forward_warmup = std::atoi(a.c_str() + 9); }
    else if (a.rfind("--flush-mib=", 0) == 0) { forward_flush_mib = std::atoi(a.c_str() + 12); }
#ifdef MIXFP4_SUB_N
    else if (a.rfind("--pattern=", 0) == 0) {
      g_sub_pattern = a.substr(10);
      if (g_sub_pattern != "random" && g_sub_pattern != "e2m1" && g_sub_pattern != "e0m3" &&
          g_sub_pattern != "checker") {
        std::fprintf(stderr, "--pattern must be random, e2m1, e0m3 or checker\n");
        return 2;
      }
    }
#endif
    else if (a.rfind("--", 0) == 0) {
      std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
      return 2;
    }
    else { pos.push_back(argv[i]); }
  }
  if (!forward_spec.empty()) {
    if (forward_out.empty() || forward_tokens.empty() || forward_reps < 1 || forward_warmup < 0) {
      std::fprintf(stderr, "--forward needs --out, a token list and positive --reps\n");
      return 2;
    }
    return forward_bench::run(forward_spec.c_str(), forward_out.c_str(), forward_tokens, forward_reps,
                              forward_warmup, forward_flush_mib);
  }
  if (pos.size() >= 3) { m = std::atoi(pos[0]); n = std::atoi(pos[1]); k = std::atoi(pos[2]); }

  if ((!pos.empty() && pos.size() != 3) || m <= 0 || n <= 0 || k <= 0 || bench_iters <= 0 ||
      raster < 0 || raster > 2 || swizzle < -1 || swizzle > 8 ||
      (swizzle > 0 && (swizzle & (swizzle - 1)))) {
    std::fprintf(stderr, "Invalid dimensions, iteration count or scheduler options.\n");
    return 2;
  }
  if (swizzle < 0) { swizzle = (m >= 8192 || n >= 8192 || k >= 8192) ? 2 : 0; }
  return run(m, n, k, fmt_a, fmt_b, warmup_iters, bench_iters, check, raster, swizzle);
}
