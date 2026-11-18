// Standalone correctness driver for the flipquant C API. Loads the shared library with dlopen --
// exactly as Python ctypes does -- calls it on random data and compares against a host reference
// that decodes every weight region under its own codebook (the same reference logic as
// src/mixed_n8k64_sm100.cu and sm120/kernel/src/mixed_nvfp4_gemm.cu: nibble = sign-magnitude index,
// E2M1 {0,.5,1,1.5,2,3,4,6} or E0M3 {0..7}, times the UE4M3 scale of its 16-element K group).
//
// Every case also runs a NEGATIVE CONTROL: the output is compared against a reference with every
// weight region's codebook flipped, and must be far from it. A check that passes against both
// would be vacuous (e.g. an unpatched sm120 library decoding everything as E2M1 on an all-E2M1
// pattern), so it fails the case.
//
// Build: see CMakeLists.txt targets test_capi_sm100 / test_capi_sm120.
// Run:   test_capi_smXXX [path/to/libflipquant_smXXX.so]   (default: next to the executable)
#pragma once

#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "flipquant_capi.h"

namespace fqtest {

#define FQT_CUDA(expr)                                                                   \
  do {                                                                                   \
    cudaError_t e_ = (expr);                                                             \
    if (e_ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_),      \
                   __FILE__, __LINE__, #expr);                                           \
      std::exit(2);                                                                      \
    }                                                                                    \
  } while (0)

using GemmFn = int (*)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
                       const uint8_t*, void*, int, int, int, int, void*, size_t, fq_stream_t);
using WsFn = size_t (*)(int, int, int);
using GranFn = void (*)(int*, int*);
using ErrFn = const char* (*)(void);

struct Api {
  GemmFn gemm = nullptr;
  WsFn workspace = nullptr;
  GranFn granule = nullptr;
  ErrFn last_error = nullptr;
};

inline std::string default_lib_path(char const* argv0, char const* libname) {
  std::string p(argv0);
  auto slash = p.find_last_of('/');
  std::string dir = slash == std::string::npos ? std::string(".") : p.substr(0, slash);
  return dir + "/" + libname;
}

inline Api load_api(std::string const& path, char const* arch) {
  void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", path.c_str(), dlerror());
    std::exit(2);
  }
  auto sym = [&](std::string const& name) {
    void* s = dlsym(h, name.c_str());
    if (!s) {
      std::fprintf(stderr, "missing export %s in %s\n", name.c_str(), path.c_str());
      std::exit(2);
    }
    return s;
  };
  Api api;
  std::string pre = std::string("fq_") + arch + "_";
  api.gemm = reinterpret_cast<GemmFn>(sym(pre + "mixfp4_gemm"));
  api.workspace = reinterpret_cast<WsFn>(sym(pre + "workspace_bytes"));
  api.granule = reinterpret_cast<GranFn>(sym(pre + "weight_granule"));
  api.last_error = reinterpret_cast<ErrFn>(sym("fq_last_error"));
  return api;
}

// ---- host decoding ------------------------------------------------------------------------------
inline float ue4m3(uint8_t raw) {
  raw &= 0x7f;
  int const e = raw >> 3, m = raw & 7;
  if (raw == 0x7f) return NAN;
  if (e == 0) return std::ldexp(float(m) / 8.f, -6);
  return std::ldexp(1.f + float(m) / 8.f, e - 7);
}
inline float decode(uint8_t packed, int k, int fmt) {
  int const nib = (packed >> ((k & 1) * 4)) & 15;
  static constexpr float e2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float const v = fmt ? e2[nib & 7] : float(nib & 7);
  return (nib & 8) ? -v : v;
}
inline float bf16_to_float(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t float_to_bf16_rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if (std::isnan(f)) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

enum class Pattern { kE2M1, kE0M3, kRandom, kChecker };
inline char const* pattern_name(Pattern p) {
  switch (p) {
    case Pattern::kE2M1: return "e2m1";
    case Pattern::kE0M3: return "e0m3";
    case Pattern::kRandom: return "random";
    default: return "checker";
  }
}

struct Case { int M, N, K, fmt_a; Pattern pattern; };

template <class T> T* to_device(std::vector<T> const& h) {
  T* d = nullptr;
  FQT_CUDA(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
  if (!h.empty()) FQT_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// Returns true on pass.
inline bool run_case(Api const& api, Case const& c, uint32_t seed, cudaStream_t stream) {
  int gr = 0, gc = 0;
  api.granule(&gr, &gc);
  int const M = c.M, N = c.N, K = c.K;
  std::mt19937 rng(seed);
  std::vector<uint8_t> a(size_t(M) * K / 2), b(size_t(N) * K / 2);
  std::vector<uint8_t> sa(size_t(M) * K / 16), sb(size_t(N) * K / 16);
  std::vector<uint8_t> fmt(size_t(N / gr) * (K / gc));
  for (auto& v : a) v = uint8_t(rng());
  for (auto& v : b) v = uint8_t(rng());
  for (auto& v : sa) v = uint8_t(0x30 + rng() % 16);  // same scale range as the research drivers
  for (auto& v : sb) v = uint8_t(0x30 + rng() % 16);
  for (int r = 0; r < N / gr; ++r) {
    for (int q = 0; q < K / gc; ++q) {
      uint8_t f = 1;
      switch (c.pattern) {
        case Pattern::kE2M1: f = 1; break;
        case Pattern::kE0M3: f = 0; break;
        case Pattern::kRandom: f = uint8_t(rng() & 1); break;
        case Pattern::kChecker: f = uint8_t((r + q) & 1); break;
      }
      fmt[size_t(r) * (K / gc) + q] = f;
    }
  }

  // Host reference, and the negative control with every weight codebook flipped.
  std::vector<float> ad(size_t(M) * K), bd(size_t(N) * K), bf(size_t(N) * K);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      ad[size_t(m) * K + k] = decode(a[size_t(m) * (K / 2) + k / 2], k, c.fmt_a) *
                              ue4m3(sa[size_t(m) * (K / 16) + k / 16]);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      int const f = fmt[size_t(n / gr) * (K / gc) + k / gc];
      float const s = ue4m3(sb[size_t(n) * (K / 16) + k / 16]);
      uint8_t const byte = b[size_t(n) * (K / 2) + k / 2];
      bd[size_t(n) * K + k] = decode(byte, k, f) * s;
      bf[size_t(n) * K + k] = decode(byte, k, 1 - f) * s;
    }

  uint8_t *da = to_device(a), *dsa = to_device(sa), *db = to_device(b), *dsb = to_device(sb),
          *dfmt = to_device(fmt);
  uint16_t* dd = nullptr;
  FQT_CUDA(cudaMalloc(&dd, size_t(M) * N * 2));
  FQT_CUDA(cudaMemsetAsync(dd, 0xff, size_t(M) * N * 2, stream));  // NaN: catches unwritten output
  size_t const ws_bytes = api.workspace(M, N, K);
  void* ws = nullptr;
  FQT_CUDA(cudaMalloc(&ws, std::max<size_t>(ws_bytes, 256)));

  int const rc = api.gemm(da, dsa, db, dsb, dfmt, dd, M, N, K, c.fmt_a, ws, ws_bytes,
                          reinterpret_cast<fq_stream_t>(stream));
  bool pass = rc == FQ_OK;
  std::vector<uint16_t> out(size_t(M) * N);
  if (rc != FQ_OK) {
    std::printf("  rc=%d: %s\n", rc, api.last_error());
  } else {
    FQT_CUDA(cudaStreamSynchronize(stream));
    FQT_CUDA(cudaMemcpy(out.data(), dd, out.size() * 2, cudaMemcpyDeviceToHost));
  }

  double err = 0, norm = 0, err_flip = 0;
  long exact = 0, nonfinite = 0;
  if (pass) {
#pragma omp parallel for reduction(+ : err, norm, err_flip, exact, nonfinite) schedule(static)
    for (int m = 0; m < M; ++m) {
      float const* ap = &ad[size_t(m) * K];
      for (int n = 0; n < N; ++n) {
        float const* bp = &bd[size_t(n) * K];
        float const* fp = &bf[size_t(n) * K];
        float ref = 0.f, reff = 0.f;
        for (int k = 0; k < K; ++k) { ref += ap[k] * bp[k]; reff += ap[k] * fp[k]; }
        uint16_t const got_raw = out[size_t(m) * N + n];
        float const got = bf16_to_float(got_raw);
        if (!std::isfinite(got)) { ++nonfinite; continue; }
        if (got_raw == float_to_bf16_rne(ref)) ++exact;
        err += double(got - ref) * double(got - ref);
        err_flip += double(got - reff) * double(got - reff);
        norm += double(ref) * double(ref);
      }
    }
  }
  double const rel = pass ? std::sqrt(err / std::max(norm, 1e-30)) : -1;
  double const rel_flip = pass ? std::sqrt(err_flip / std::max(norm, 1e-30)) : -1;
  if (pass) pass = nonfinite == 0 && norm > 0 && rel < 1e-2 && rel_flip > 5e-2;
  std::printf("%s  M=%-5d N=%-5d K=%-5d fmt_a=%s weights=%-7s rel_fro=%.3e  flipped-codebook "
              "rel_fro=%.3e  exact_bf16=%ld/%ld  nonfinite=%ld\n",
              pass ? "PASS" : "FAIL", M, N, K, c.fmt_a ? "e2m1" : "e0m3", pattern_name(c.pattern),
              rel, rel_flip, exact, long(M) * N, nonfinite);

  for (void* p : {static_cast<void*>(da), static_cast<void*>(dsa), static_cast<void*>(db),
                  static_cast<void*>(dsb), static_cast<void*>(dfmt), static_cast<void*>(dd), ws})
    FQT_CUDA(cudaFree(p));
  return pass;
}

// An argument check must return `expect` without touching the device.
inline bool expect_rc(Api const& api, char const* what, int expect, int M, int N, int K, int fmt_a,
                      size_t ws_bytes_override, cudaStream_t stream) {
  // Well-aligned dummy device buffers large enough for any shape used here.
  static void* big = nullptr;
  if (!big) FQT_CUDA(cudaMalloc(&big, size_t(64) << 20));
  auto* p = static_cast<uint8_t*>(big);
  size_t ws = ws_bytes_override;
  int const rc = api.gemm(p, p, p, p, p, p, M, N, K, fmt_a, p, ws, reinterpret_cast<fq_stream_t>(stream));
  bool const ok = rc == expect;
  std::printf("%s  reject %-34s rc=%d (want %d): %s\n", ok ? "PASS" : "FAIL", what, rc, expect,
              rc ? api.last_error() : "accepted");
  return ok;
}

}  // namespace fqtest
