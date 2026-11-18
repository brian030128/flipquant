// Correctness driver for libflipquant_sm120.so (run on an RTX PRO 6000 Blackwell / RTX 5090,
// CC 12.0). Load the PATCHED library; the unpatched one must refuse (checked below if given).
//   test_capi_sm120 [path/to/libflipquant_sm120.so] [path/to/libflipquant_sm120_unpatched.so]
// Exit status 0 iff every case passes.
#include "test_capi_common.hpp"

int main(int argc, char** argv) {
  using namespace fqtest;
  std::string const lib = argc > 1 ? argv[1] : default_lib_path(argv[0], "libflipquant_sm120.so");
  Api api = load_api(lib, "sm120");
  int gr = 0, gc = 0;
  api.granule(&gr, &gc);
  std::printf("library: %s\nweight-format granule: %d rows x %d K\n", lib.c_str(), gr, gc);
  if (gr != 16 || gc != 64) { std::printf("FAIL unexpected granule\n"); return 1; }

  cudaStream_t stream;
  FQT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Shapes cover: everything already tile-shaped (weights consumed in place), M/N padding,
  // K % 128 == 64 (K padding; the last 64-K granule is then the padded half of a k_tile),
  // single k_tile, decode-style M=1, and multiple CTA tiles in both directions.
  int const shapes[][3] = {{1, 16, 64},     {5, 48, 192},      {130, 272, 320},
                           {257, 256, 1024}, {128, 128, 128},   {256, 512, 1024},
                           {1, 4096, 512},   {384, 384, 2048}};
  bool ok = true;
  uint32_t seed = 2026;
  for (auto const& s : shapes) {
    for (Pattern p : {Pattern::kRandom, Pattern::kChecker, Pattern::kE2M1, Pattern::kE0M3}) {
      ok &= run_case(api, Case{s[0], s[1], s[2], 1, p}, seed++, stream);
    }
  }
  // E0M3 activations are not compiled into this build; they must be refused, not mis-decoded.
  ok &= expect_rc(api, "fmt_a == 0 (E0M3 activations)", FQ_ERR_UNSUPPORTED, 16, 16, 64, 0,
                  size_t(64) << 20, stream);
  ok &= expect_rc(api, "N % 16 != 0", FQ_ERR_UNSUPPORTED, 16, 8, 64, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "K % 64 != 0", FQ_ERR_UNSUPPORTED, 16, 16, 96, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "M == 0", FQ_ERR_INVALID_ARGUMENT, 0, 16, 64, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "workspace too small", FQ_ERR_WORKSPACE, 16, 16, 64, 1, 16, stream);

  if (argc > 2) {
    Api raw = load_api(argv[2], "sm120");
    ok &= expect_rc(raw, "unpatched library", FQ_ERR_NOT_PATCHED, 16, 16, 64, 1,
                    size_t(64) << 20, stream);
  }

  FQT_CUDA(cudaStreamDestroy(stream));
  std::printf(ok ? "ALL PASSED\n" : "SOME CASES FAILED\n");
  return ok ? 0 : 1;
}
