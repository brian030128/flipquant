// Internal helpers shared by the per-arch C API translation units. Header-only; each .so gets its
// own copy (the only exported symbol from here is fq_last_error, defined once per library).
#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "flipquant_capi.h"

namespace fq {

// Per-thread message for fq_last_error(). Fixed-size so setting it can never throw or allocate.
inline char* error_buffer() {
  static thread_local char buf[1024] = "no error";
  return buf;
}

inline int fail(int code, char const* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(error_buffer(), 1024, fmt, ap);
  va_end(ap);
  return code;
}

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
inline int round_up(int v, int a) { return (v + a - 1) / a * a; }
inline bool aligned(void const* p, size_t a) { return (reinterpret_cast<uintptr_t>(p) % a) == 0; }

constexpr size_t kWorkspaceAlign = 256;

// Linear bump allocator over the caller's workspace; every sub-buffer is 256-byte aligned.
struct Carve {
  uint8_t* base;
  size_t offset = 0;
  explicit Carve(void* p) : base(static_cast<uint8_t*>(p)) {}
  template <class T> T* take(size_t bytes) {
    T* p = reinterpret_cast<T*>(base ? base + offset : nullptr);
    offset = align_up(offset + bytes, kWorkspaceAlign);
    return p;
  }
};

// Compute capability of the calling thread's current device, cached per device ordinal. Uses only
// attribute queries (no context-wide synchronisation).
inline int current_device_cc(int* device_out, int* major_out, int* minor_out) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) {
    return fail(FQ_ERR_CUDA, "cudaGetDevice failed: %s", cudaGetErrorString(e));
  }
  int major = 0, minor = 0;
  e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (e != cudaSuccess) {
    return fail(FQ_ERR_CUDA, "cudaDeviceGetAttribute failed: %s", cudaGetErrorString(e));
  }
  *device_out = dev; *major_out = major; *minor_out = minor;
  return FQ_OK;
}

inline int check_launch(char const* what) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    return fail(FQ_ERR_CUDA, "%s: kernel launch failed: %s (note: this can also be a sticky error "
                "left by earlier work on the device)", what, cudaGetErrorString(e));
  }
  return FQ_OK;
}

}  // namespace fq

extern "C" FQ_API const char* fq_last_error(void) { return fq::error_buffer(); }
