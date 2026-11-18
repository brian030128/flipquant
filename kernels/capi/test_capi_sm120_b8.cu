// Correctness driver for libflipquant_sm120_b8.so (weights on CUTLASS operand B, 8-row x 64-K
// format granule; RTX PRO 6000 Blackwell / RTX 5090, CC 12.0). Load the PATCHED library; the
// unpatched one must refuse (checked below if given).
//   test_capi_sm120_b8 [path/to/libflipquant_sm120_b8.so] [path/to/libflipquant_sm120_b8_unpatched.so]
// Exit status 0 iff every case passes.
#include "test_capi_common.hpp"

int main(int argc, char** argv) {
  using namespace fqtest;
  std::string const lib = argc > 1 ? argv[1] : default_lib_path(argv[0], "libflipquant_sm120_b8.so");
  Api api = load_api(lib, "sm120b8");
  int gr = 0, gc = 0;
  api.granule(&gr, &gc);
  std::printf("library: %s\nweight-format granule: %d rows x %d K\n", lib.c_str(), gr, gc);
  if (gr != 8 || gc != 64) { std::printf("FAIL unexpected granule\n"); return 1; }

  cudaStream_t stream;
  FQT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Shapes cover: everything tile-shaped (operands consumed in place, D written straight into d),
  // N a multiple of 8 but not 16 (the granule this library exists for), M/N padding and the output
  // crop, K % 128 == 64 (K padding), decode-style M=1, multiple CTA tiles in both directions.
  int const shapes[][3] = {{1, 8, 64},       {5, 24, 192},      {130, 264, 320},
                           {257, 256, 1024}, {128, 128, 128},   {256, 512, 1024},
                           {1, 4096, 512},   {384, 392, 2048},  {128, 136, 256}};
  bool ok = true;
  uint32_t seed = 2027;
  for (auto const& s : shapes) {
    for (Pattern p : {Pattern::kRandom, Pattern::kChecker, Pattern::kE2M1, Pattern::kE0M3}) {
      ok &= run_case(api, Case{s[0], s[1], s[2], 1, p}, seed++, stream);
    }
  }
  ok &= expect_rc(api, "fmt_a == 0 (E0M3 activations)", FQ_ERR_UNSUPPORTED, 16, 16, 64, 0,
                  size_t(64) << 20, stream);
  ok &= expect_rc(api, "N % 8 != 0", FQ_ERR_UNSUPPORTED, 16, 12, 64, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "K % 64 != 0", FQ_ERR_UNSUPPORTED, 16, 16, 96, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "M == 0", FQ_ERR_INVALID_ARGUMENT, 0, 16, 64, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "workspace too small", FQ_ERR_WORKSPACE, 16, 16, 64, 1, 16, stream);

  if (argc > 2) {
    Api raw = load_api(argv[2], "sm120b8");
    ok &= expect_rc(raw, "unpatched library", FQ_ERR_NOT_PATCHED, 16, 16, 64, 1,
                    size_t(64) << 20, stream);
  }

  FQT_CUDA(cudaStreamDestroy(stream));
  std::printf(ok ? "ALL PASSED\n" : "SOME CASES FAILED\n");
  return ok ? 0 : 1;
}
