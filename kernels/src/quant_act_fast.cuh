// A fast per-token activation quantizer for SM100 (sm_100a): the same output as sm120/csrc/quant_act.cuh -- E2M1 codes
// [T, K/2], UE4M3 scales in the SM1xx SFA layout (padding zeroed), the FP32 per-token global scale gs = amax / 2688 --
// and the same rules (NVFP4: one scale per 16-block, e4m3(peak / 6); FourOverSix: the /6 and /4 candidates, the one with
// the lower squared error), but not bit-identical to the simulation:
//   * x / gs and s / scale are multiplies by reciprocals (one __frcp_rn per row and per scale), not IEEE divisions;
//   * the E2M1 codes come from the hardware conversion cvt.rn.satfinite.e2m1x2.f32 (round to nearest even: an exact
//     tie between two levels can go the other way than the reference's bucketize, which rounds ties down);
//   * the FourOverSix squared errors: the candidates' codes decoded by cvt.rn.f16x2.e2m1x2, fused multiply-adds into
//     four partial sums, not torch's separately rounded pairwise tree;
//   * 16-byte loads; the row amax as bf16x2 maxima (exact).
// The grid is quant_act.cuh's: one CTA per padded token row, a row split into K chunks when there are few rows.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace fq_fast_quant {

constexpr int kThreads = 128;

__device__ __forceinline__ float e4m3_round(float v) {
  v = fminf(fmaxf(v, 0.001953125f), 448.0f);
  __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3)));
}

// Eight floats -> eight E2M1 codes, element j in bits 4j..4j+3 (the packed layout of the GEMM's A operand).
__device__ __forceinline__ uint32_t e2m1x8(float const *v) {
  uint32_t out;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}

// Eight E2M1 codes (element j in bits 4j..4j+3) -> their values, with the hardware conversion (exact in f16).
__device__ __forceinline__ void e2m1x8_values(uint32_t w, float *v) {
  uint32_t h[4];
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(w));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 const f = __half22float2(*reinterpret_cast<__half2 const *>(&h[i]));
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

// Squared error of the 16 codes in (w0, w1) at scale `scale` against s (four independent partial sums).
__device__ __forceinline__ float sq_error(uint32_t w0, uint32_t w1, float scale, float const *s) {
  float q[16];
  e2m1x8_values(w0, q);
  e2m1x8_values(w1, q + 8);
  float e[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float const d = fmaf(q[j], scale, -s[j]);
    e[j & 3] = fmaf(d, d, e[j & 3]);
  }
  return (e[0] + e[1]) + (e[2] + e[3]);
}

__device__ __forceinline__ int64_t sf_offset(int64_t row, int kb, int k_atoms) {
  return ((row / 128) * k_atoms + kb / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4;
}

// 16 bf16 of block kb as floats; `vec`: the row is 16-byte aligned.
__device__ __forceinline__ void load16(__nv_bfloat16 const *xr, int kb, bool vec, float *s) {
  if (vec) {
    uint4 const *p = reinterpret_cast<uint4 const *>(xr + kb * 16);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 const u = __ldg(p + h);
      uint32_t const w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 const f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&w[i]));
        s[8 * h + 2 * i] = f.x;
        s[8 * h + 2 * i + 1] = f.y;
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) { s[j] = __bfloat162float(xr[kb * 16 + j]); }
  }
}

template <int MODE>
__global__ void __launch_bounds__(kThreads)
quant_rows_kernel(__nv_bfloat16 const *__restrict__ x, int64_t x_stride, uint8_t *__restrict__ packed,
                  uint8_t *__restrict__ sf, float *__restrict__ gs_out, int T, int K, int chunk_blocks, bool vec) {
  int64_t const row = blockIdx.x;
  int const KB = K / 16;
  int const KB_PAD = (KB + 3) / 4 * 4;
  int const k_atoms = KB_PAD / 4;
  int const kb_begin = blockIdx.y * chunk_blocks;
  int const kb_end_pad = min(kb_begin + chunk_blocks, KB_PAD);
  int const kb_end = min(kb_begin + chunk_blocks, KB);
  if (row >= T) {
    for (int kb = kb_begin + threadIdx.x; kb < kb_end_pad; kb += kThreads) { sf[sf_offset(row, kb, k_atoms)] = 0; }
    return;
  }
  __nv_bfloat16 const *xr = x + row * x_stride;
  // pass 1: row amax
  float amax = 0.f;
  if (vec) {
    __nv_bfloat162 m2 = __float2bfloat162_rn(0.f);
    for (int i = threadIdx.x * 8; i < K; i += kThreads * 8) {
      uint4 const u = __ldg(reinterpret_cast<uint4 const *>(xr + i));
      uint32_t const w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { m2 = __hmax2(m2, __habs2(*reinterpret_cast<__nv_bfloat162 const *>(&w[j]))); }
    }
    amax = fmaxf(__bfloat162float(m2.x), __bfloat162float(m2.y));
  } else {
    for (int i = threadIdx.x; i < K; i += kThreads) { amax = fmaxf(amax, fabsf(__bfloat162float(xr[i]))); }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
  __shared__ float red[kThreads / 32];
  if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = amax; }
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int w = 1; w < kThreads / 32; ++w) { amax = fmaxf(amax, red[w]); }
  float const gs = fmaxf(amax * (1.0f / 2688.0f), 1.17549435e-38f);
  float const inv_gs = __frcp_rn(gs);
  if (threadIdx.x == 0 && blockIdx.y == 0) { gs_out[row] = gs; }
  // pass 2: this CTA's chunk of 16-element blocks (the row is L1/L2-resident from pass 1)
  for (int kb = kb_begin + threadIdx.x; kb < kb_end; kb += kThreads) {
    float s[16];
    load16(xr, kb, vec, s);
    float peak = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      s[j] *= inv_gs;
      peak = fmaxf(peak, fabsf(s[j]));
    }
    float scale = e4m3_round(peak * (1.0f / 6.0f));
    float a[16];
    float r = __frcp_rn(scale);
#pragma unroll
    for (int j = 0; j < 16; ++j) { a[j] = s[j] * r; }
    uint32_t w0 = e2m1x8(a), w1 = e2m1x8(a + 8);
    if (MODE == 1) {
      float const sc4 = e4m3_round(peak * 0.25f);
      r = __frcp_rn(sc4);
#pragma unroll
      for (int j = 0; j < 16; ++j) { a[j] = s[j] * r; }
      uint32_t const v0 = e2m1x8(a), v1 = e2m1x8(a + 8);
      if (sq_error(v0, v1, sc4, s) < sq_error(w0, w1, scale, s)) {
        w0 = v0;
        w1 = v1;
        scale = sc4;
      }
    }
    *reinterpret_cast<uint2 *>(packed + row * (K / 2) + kb * 8) = make_uint2(w0, w1);
    sf[sf_offset(row, kb, k_atoms)] = uint8_t(__nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3));
  }
  for (int kb = max(KB, kb_begin) + threadIdx.x; kb < kb_end_pad; kb += kThreads) { sf[sf_offset(row, kb, k_atoms)] = 0; }
}

inline cudaError_t quant_rows(void const *x, int64_t x_stride, int T, int K, int mode, void *packed, void *sf,
                              float *gs, cudaStream_t stream) {
  int const rows_pad = (T + 127) / 128 * 128;
  int const kb_pad = (K / 16 + 3) / 4 * 4;
  static int num_sms = 0;
  if (num_sms == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    if (num_sms <= 0) { num_sms = 1; }
  }
  int const max_chunks = (kb_pad + kThreads - 1) / kThreads;
  int const want = (2 * num_sms + T - 1) / T;
  int const chunks = want < 1 ? 1 : (want > max_chunks ? max_chunks : want);
  int const chunk_blocks = (kb_pad + chunks - 1) / chunks;
  dim3 const grid(rows_pad, chunks);
  bool const vec = (reinterpret_cast<uintptr_t>(x) % 16 == 0) && (x_stride % 8 == 0);
  auto const *xb = static_cast<__nv_bfloat16 const *>(x);
  auto *p = static_cast<uint8_t *>(packed);
  auto *s = static_cast<uint8_t *>(sf);
  if (mode == 1) {
    quant_rows_kernel<1><<<grid, kThreads, 0, stream>>>(xb, x_stride, p, s, gs, T, K, chunk_blocks, vec);
  } else {
    quant_rows_kernel<0><<<grid, kThreads, 0, stream>>>(xb, x_stride, p, s, gs, T, K, chunk_blocks, vec);
  }
  return cudaGetLastError();
}

}  // namespace fq_fast_quant
