// Correctness driver for libflipquant_sm100.so (run on a B200/GB200, CC 10.0).
//   test_capi_sm100 [path/to/libflipquant_sm100.so]
// Exit status 0 iff every case passes.
#include "test_capi_common.hpp"

int main(int argc, char** argv) {
  using namespace fqtest;
  std::string const lib = argc > 1 ? argv[1] : default_lib_path(argv[0], "libflipquant_sm100.so");
  Api api = load_api(lib, "sm100");
  int gr = 0, gc = 0;
  api.granule(&gr, &gc);
  std::printf("library: %s\nweight-format granule: %d rows x %d K\n", lib.c_str(), gr, gc);
  if (gr != 8 || gc != 64) { std::printf("FAIL unexpected granule\n"); return 1; }

  cudaStream_t stream;
  FQT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Shapes: minimal, M/N tails, a partial outer K tile (K % 128 == 64), and the two shapes the
  // research harness validated the packed pipeline at (144x136x320, 257x264x1024).
  int const shapes[][3] = {{1, 8, 64},     {3, 40, 192},     {144, 136, 320},
                           {257, 264, 1024}, {128, 256, 512}, {1, 4096, 256},
                           {33, 72, 128}};
  bool ok = true;
  uint32_t seed = 2026;
  for (auto const& s : shapes) {
    for (int fa : {1, 0}) {
      for (Pattern p : {Pattern::kRandom, Pattern::kChecker, Pattern::kE2M1, Pattern::kE0M3}) {
        ok &= run_case(api, Case{s[0], s[1], s[2], fa, p}, seed++, stream);
      }
    }
  }
  ok &= expect_rc(api, "N % 8 != 0", FQ_ERR_UNSUPPORTED, 16, 12, 64, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "K % 64 != 0", FQ_ERR_UNSUPPORTED, 16, 16, 96, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "fmt_a == 2", FQ_ERR_INVALID_ARGUMENT, 16, 16, 64, 2, size_t(64) << 20, stream);
  ok &= expect_rc(api, "M == 0", FQ_ERR_INVALID_ARGUMENT, 0, 16, 64, 1, size_t(64) << 20, stream);
  ok &= expect_rc(api, "workspace too small", FQ_ERR_WORKSPACE, 16, 16, 64, 1, 16, stream);

  FQT_CUDA(cudaStreamDestroy(stream));
  std::printf(ok ? "ALL PASSED\n" : "SOME CASES FAILED\n");
  return ok ? 0 : 1;
}
