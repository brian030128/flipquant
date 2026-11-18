/*
 * flipquant C ABI over the mixfp4 research GEMM kernels (W4A4, per-region E0M3/E2M1 weights).
 *
 * Two shared libraries implement this header, one per tensor-core ISA; each exports only its own
 * arch-prefixed entry points plus fq_last_error():
 *
 *   libflipquant_sm100.so   datacenter Blackwell, sm_100a (B200 / GB200)    fq_sm100_*
 *   libflipquant_sm120.so   GeForce/workstation Blackwell, sm_120a           fq_sm120_*
 *                           (RTX 5090, RTX PRO 6000 Blackwell). Built UNPATCHED as
 *                           libflipquant_sm120_unpatched.so and then SASS-patched; only load the
 *                           patched file -- the unpatched one refuses to run (FQ_ERR_NOT_PATCHED).
 *
 * All pointer arguments are DEVICE pointers. Every call is stream-ordered on `stream`: no host
 * synchronisation, no device-wide synchronisation, no allocation. Scratch space comes from the
 * caller-provided `workspace` (size from fq_smXXX_workspace_bytes; 256-byte aligned).
 *
 * Logical layouts (identical for both archs):
 *   a     [M, K/2]   uint8, activations, row-major. Element k of a row is the LOW nibble of byte
 *                    k/2 when k is even, the HIGH nibble when k is odd. Nibble = sign-magnitude:
 *                    bit 3 = sign, bits 0-2 = magnitude index.
 *   sa    [M, K/16]  uint8, UE4M3 scale byte per 16 consecutive K (float8_e4m3fn bit pattern of a
 *                    positive value; bit 7 must be clear -- sm120 masks it off internally).
 *   b     [N, K/2]   uint8, weights, same packing as a.
 *   sb    [N, K/16]  uint8, weight scales, same as sa.
 *   fmt_b [N/R, K/C] uint8, row-major weight-format map over contiguous weight rows, where (R, C)
 *                    is the arch's granule (fq_smXXX_weight_granule). 0 = E0M3 (magnitudes
 *                    0..7), 1 = E2M1 (magnitudes 0,.5,1,1.5,2,3,4,6). Values other than 0/1 are
 *                    undefined behaviour.
 *   d     [M, N]     bf16, row-major. d[m][n] = sum_k dec(a[m][k]) * sa[m][k/16]
 *                                                 * dec(b[n][k]) * sb[n][k/16]
 *                    accumulated in fp32, rounded once to bf16. No global alpha is applied.
 *   fmt_a            activation codebook, uniform for the launch (0 = E0M3, 1 = E2M1).
 *
 * Constraints: M >= 1; N % R == 0; K % 64 == 0; a and b 16-byte aligned; sa and sb 4-byte
 * aligned; d 2-byte aligned; workspace 256-byte aligned. sm120 additionally supports fmt_a == 1
 * only (see capi/README.md).
 *
 * Return value: 0 on success, one of FQ_ERR_* otherwise; fq_last_error() then returns a
 * human-readable message for the calling thread (valid until that thread's next failing call).
 */
#ifndef FLIPQUANT_CAPI_H_
#define FLIPQUANT_CAPI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define FQ_API __attribute__((visibility("default")))
#else
#define FQ_API
#endif

/* Opaque stream handle; ABI-identical to cudaStream_t / CUstream (a pointer). */
typedef struct CUstream_st* fq_stream_t;

enum {
  FQ_OK                   = 0,
  FQ_ERR_INVALID_ARGUMENT = 1, /* null pointer, M<1, fmt_a out of range, ...              */
  FQ_ERR_UNSUPPORTED      = 2, /* shape/alignment/format this build does not implement      */
  FQ_ERR_WORKSPACE        = 3, /* workspace too small or misaligned                         */
  FQ_ERR_CUDA             = 4, /* a CUDA runtime call or kernel launch failed               */
  FQ_ERR_DEVICE           = 5, /* current device is not the arch this library was built for */
  FQ_ERR_NOT_PATCHED      = 6, /* sm120: SASS patcher has not been applied to this .so      */
  FQ_ERR_INTERNAL         = 7  /* CUTLASS can_implement/initialize failure, self-check fail  */
};

FQ_API const char* fq_last_error(void);

FQ_API int fq_sm100_mixfp4_gemm(const uint8_t* a, const uint8_t* sa,
                                const uint8_t* b, const uint8_t* sb,
                                const uint8_t* fmt_b, void* d_bf16,
                                int M, int N, int K, int fmt_a,
                                void* workspace, size_t workspace_bytes, fq_stream_t stream);
FQ_API size_t fq_sm100_workspace_bytes(int M, int N, int K);
FQ_API void fq_sm100_weight_granule(int* rows, int* cols);

FQ_API int fq_sm120_mixfp4_gemm(const uint8_t* a, const uint8_t* sa,
                                const uint8_t* b, const uint8_t* sb,
                                const uint8_t* fmt_b, void* d_bf16,
                                int M, int N, int K, int fmt_a,
                                void* workspace, size_t workspace_bytes, fq_stream_t stream);
FQ_API size_t fq_sm120_workspace_bytes(int M, int N, int K);
FQ_API void fq_sm120_weight_granule(int* rows, int* cols);

/* sm120 with the weights on CUTLASS operand B: 8-row x 64-K weight granule (libflipquant_sm120_b8.so). */
FQ_API int fq_sm120b8_mixfp4_gemm(const uint8_t* a, const uint8_t* sa,
                                  const uint8_t* b, const uint8_t* sb,
                                  const uint8_t* fmt_b, void* d_bf16,
                                  int M, int N, int K, int fmt_a,
                                  void* workspace, size_t workspace_bytes, fq_stream_t stream);
FQ_API size_t fq_sm120b8_workspace_bytes(int M, int N, int K);
FQ_API void fq_sm120b8_weight_granule(int* rows, int* cols);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLIPQUANT_CAPI_H_ */
