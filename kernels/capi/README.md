# flipquant C API: MixFP4 W4A4 GEMM through ctypes

This directory wraps the research kernels in `src/` behind a small `extern "C"` ABI, so a Python or
PyTorch model can call them with `ctypes`. There is one shared library per Blackwell tensor-core
ISA. The two kernels share almost no code:

| library | GPU | kernel | weight-format granule |
|---|---|---|---|
| `libflipquant_sm100.so` | B200 / GB200 (CC 10.0, `sm_100a`) | exact N8×K64 tcgen05 pipeline (`src/n8k64_pipeline.cuh`) | **8 rows × 64 K** |
| `libflipquant_sm120.so` | RTX PRO 6000 Blackwell / RTX 5090 (CC 12.0, `sm_120a`) | CUTLASS mixed mainloop, weights on operand A (`src/collective/sm120_blockscaled_mma_tma_mixed.hpp`) | **16 rows × 64 K** |
| `libflipquant_sm120_b8.so` | same GPUs | same mainloop, weights on operand B, 1×8 warp arrangement (`capi/fq_sm120_b8_capi.cu`) | **8 rows × 64 K** |

The public header is `flipquant_capi.h`. Python backend names (`flipquant/native.py`): `sm100`, `sm120`, `sm120-b8`.

## API

```c
int    fq_sm100_mixfp4_gemm(const uint8_t* a, const uint8_t* sa, const uint8_t* b, const uint8_t* sb,
                            const uint8_t* fmt_b, void* d_bf16, int M, int N, int K, int fmt_a,
                            void* workspace, size_t workspace_bytes, cudaStream_t stream);
size_t fq_sm100_workspace_bytes(int M, int N, int K);
void   fq_sm100_weight_granule(int* rows, int* cols);          /* -> 8, 64  */

int    fq_sm120_mixfp4_gemm(/* same signature */);
size_t fq_sm120_workspace_bytes(int M, int N, int K);          /* 0 if the shape is unsupported */
void   fq_sm120_weight_granule(int* rows, int* cols);          /* -> 16, 64 */

const char* fq_last_error(void);   /* per-thread message for the last failing call */
```

All tensor pointers are device pointers. A call enqueues kernels on `stream` and returns. It does
not synchronize the host or the device, and it does not allocate. The return value is 0 on
success and an `FQ_ERR_*` code otherwise:

| code | name | meaning |
|---|---|---|
| 1 | `INVALID_ARGUMENT` | null pointer, `M < 1`, or `fmt_a` not in {0, 1} |
| 2 | `UNSUPPORTED` | bad shape or alignment, or sm120 with `fmt_a = 0` |
| 3 | `WORKSPACE` | workspace too small or not 256-byte aligned |
| 4 | `CUDA` | launch or runtime error |
| 5 | `DEVICE` | wrong GPU architecture |
| 6 | `NOT_PATCHED` | sm120 library that was never SASS-patched |
| 7 | `INTERNAL` | CUTLASS `can_implement` or `initialize` failed, or the granule self-check failed |

### Logical layouts (both archs)

| tensor | shape / dtype | contents |
|---|---|---|
| `a` | `[M, K/2]` uint8, row-major | Activations. Element `k` is the low nibble of byte `k/2` when `k` is even, the high nibble when `k` is odd. A nibble is sign-magnitude: bit 3 is the sign, bits 0–2 index the magnitude. |
| `sa` | `[M, K/16]` uint8 | One UE4M3 scale per 16 K. This is the bit pattern of a positive `float8_e4m3fn`, so bit 7 is clear. |
| `b` | `[N, K/2]` uint8 | Weights, packed the same way as `a`. |
| `sb` | `[N, K/16]` uint8 | Weight scales. |
| `fmt_b` | `[N/R, K/C]` uint8, row-major | One byte per weight region of `R` contiguous rows × `C` K, where `(R, C)` comes from `fq_*_weight_granule`. 0 = E0M3 (magnitudes 0…7), 1 = E2M1 (magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6). Any other value is undefined. |
| `d` | `[M, N]` bf16, row-major | `d[m,n] = Σ_k dec(a[m,k])·sa[m,k/16] · dec(b[n,k])·sb[n,k/16]`, accumulated in fp32. No global alpha is applied; apply it in Python. |
| `fmt_a` | int | Activation codebook, uniform for the whole call. |

Shape and alignment constraints:
- `M ≥ 1`, `N % R == 0`, `K % 64 == 0`.
- `a` and `b` must be 16-byte aligned. On sm100, `sa` and `sb` must be 4-byte aligned.
- `workspace` must be 256-byte aligned. Fresh PyTorch CUDA allocations meet every alignment requirement.

### Python (ctypes) sketch

```python
import ctypes, torch
lib = ctypes.CDLL("build/libflipquant_sm120.so")      # or libflipquant_sm100.so
P, I, Z = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
lib.fq_sm120_mixfp4_gemm.argtypes = [P, P, P, P, P, P, I, I, I, I, P, Z, P]
lib.fq_sm120_workspace_bytes.argtypes = [I, I, I]; lib.fq_sm120_workspace_bytes.restype = Z
lib.fq_last_error.restype = ctypes.c_char_p

ws_bytes = lib.fq_sm120_workspace_bytes(M, N, K)
ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
d = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
rc = lib.fq_sm120_mixfp4_gemm(a.data_ptr(), sa.data_ptr(), b.data_ptr(), sb.data_ptr(),
                              fmt_b.data_ptr(), d.data_ptr(), M, N, K, 1,
                              ws.data_ptr(), ws_bytes, torch.cuda.current_stream().cuda_stream)
if rc: raise RuntimeError(lib.fq_last_error().decode())
d.mul_(alpha)
```

Both libraries link the CUDA runtime statically, which is CMake's default. That lets them load
next to any PyTorch build. Device pointers and `cudaStream_t` handles can be passed between
runtime instances in the same process.

## What each library does internally

### sm100 (`fq_sm100_capi.cu`)

The library runs the `--packed-inputs` double-buffered N8 pipeline at its default CTA of
128×32×128. Every MMA is a real `tcgen05` 128×8×64 block-scaled MMA. Each N8×K64 weight region's
byte from `fmt_b` is written into the B-format field of the runtime UMMA instruction descriptor,
so the result is exact. No branching and no cubin patching are involved.

The workspace holds the kernel's internal layouts:
- operands repacked to K32-chunk-major by `n8_pack_operands`;
- scales repacked to K64-major 32-bit words by `n8_pack_scales`.

`fmt_b` and `d` are used as-is. This configuration is correct but slow: about 400 TFLOP/s at 8192³,
roughly 5% of the wide kernel.

### sm120 (`fq_sm120_capi.cu`)

The library uses the "weights as A" configuration from `sm120/kernel/docs/mixed_nvfp4_report.md` §6:
- CTA 128×128×128 with the stock 4×2 warp arrangement.
- The generated blob path (`MMA_M=2 MMA_N=8 A_ATOMS=1 B_ATOMS=8`) with `MIXFP4_JOINT_KA=1 MIXFP4_B_ALL_E2M1=1 MIXFP4_A_ATOMS_PER_GRANULE=1 MIXFP4_B_ATOMS_PER_GRANULE=8`.

The report measures this configuration at +5.8% to +7.2% over stock NVFP4 on the RTX PRO 6000.

Each call does the following:

1. **Operand placement.** It computes Dᵀ = W·Xᵀ: CUTLASS M is the weight rows `N`, and CUTLASS N
   is the tokens `M`. CUTLASS writes Dᵀ `[Np, Mp]` into the workspace, and a transpose kernel
   produces `d [M, N]`.
2. **Padding.** `Np`, `Mp` and `Kp` are rounded up to 128. The weights are consumed in place when
   `N % 128 == 0` and `K % 128 == 0`, which covers typical LLM layers. Otherwise they are
   zero-padded into the workspace, and the activations are handled the same way.
3. **Scales and format tags.** Both scale arrays are scattered into CUTLASS's interleaved
   `Sm1xxBlkScaledConfig` SF layout. The weight scale bytes also get bit 7 set wherever the
   16×64 region is E0M3; this is the kernel's format flag, and the tensor core ignores it. Bit 7
   of the activation scales is cleared.
4. **Granule placement.** The builder's `PermTileM` is 128, an identity, and the warp arrangement
   is 4×2. So the unit the kernel flags, one warp's m-atom per k_block, is 16 contiguous
   16-aligned rows, and no row permutation is needed. This is not assumed: on its first call the
   library derives the granule map from the kernel's own `TiledMma`, the same way
   `build_granule_map` does, and returns `FQ_ERR_INTERNAL` if the map is not contiguous.
5. **Patching.** The CMake target `flipquant_sm120` runs `scripts/patch_mixed_nvfp4_gemm.py` on
   `libflipquant_sm120_unpatched.so`, using `--expect-sites 0,1 --mark-patched
   FQ_SM120_PATCH_STATE`. The patcher:
   - requires the site census to be exactly {0, 1} with equal OMMA counts;
   - sets E0M3 on site 1;
   - re-disassembles the result to verify it;
   - rewrites a marker string in the library's `.data` from `UNPATCHED` to `PATCHED__`;
   - writes `libflipquant_sm120.so` atomically, and only if verification passes.

   The unpatched library returns `FQ_ERR_NOT_PATCHED` instead of silently decoding everything as
   E2M1. The code is SASS only, with no PTX, so it cannot be JIT-compiled unpatched on another GPU.

**Limitation: activations are E2M1 only on sm120.** `MIXFP4_B_ALL_E2M1` removes the E0M3-on-B
sites from the kernel, so `fmt_a = 0` returns `FQ_ERR_UNSUPPORTED`. Supporting E0M3 activations
would need a second instantiation that pins B to E0M3, meaning sites {1, 3} would be patched.
That instantiation is not built.

### sm120-b8 (`fq_sm120_b8_capi.cu`): weights on B, 8×64

The 8-row granule (`sm120/kernel/docs/mixed_nvfp4_report.md`, "8×64 on B via the 1×8 arrangement" / "The
8-column × 64-K frontier"): `D = X·Wᵀ` with the activations on CUTLASS A (pinned to E2M1) and the
weights on B, where the m16n8k64 atom is 8 wide.

- CTA 128×128×128, 1×8 warp arrangement (`MIXFP4_ATOM_M=1`), `MIXFP4_PERM_N=128`
  (`Shape<_8,_2,_2,_4>:Stride<_1,_16,_8,_32>`), x4 LDSM, epilogue tile 64×64, blob generated with
  `MMA_M=8 MMA_N=2 A_ATOMS=8 B_ATOMS=1`, `-DMIXFP4_JOINT_KB=1 -DMIXFP4_A_ALL_E2M1=1
  -DMIXFP4_A_ATOMS_PER_GRANULE=8 -DMIXFP4_B_ATOMS_PER_GRANULE=1`. 16 joint arms per k_tile.
- No transpose: CUTLASS writes `D [Mp, Np]`, straight into `d` when `M % 128 == 0` and
  `N % 128 == 0`, else into the workspace and cropped. Requires `N % 8 == 0`, `K % 64 == 0`.
- The N permutation only reorders which 8-aligned block of 8 columns each warp's n-atom gets, so a
  logical `[N/8, K/64]` map needs no row permutation; the first call checks this against the
  kernel's own TiledMma (`partition_B`) and returns `FQ_ERR_INTERNAL` otherwise.
- Patch: exact site census `--expect-sites 0,2` (A is pinned, so only e2m1×e2m1 and e2m1×e0m3
  exist), marker `FQ_SM120B8_PATCH_STATE`; the unpatched library returns `FQ_ERR_NOT_PATCHED`.

Measured on an RTX 5090 (CUDA 13.1), research harness, 4096³: stock NVFP4 1430.5 TFLOP/s,
16×64 weights-as-A 1339.9 (+6.8%), 8×64 weights-on-B 1293.7 (+10.6%). The report's RTX PRO 6000
numbers for the same two configurations are +7.2% and +15.1%.

## Building

The build only needs `nvcc`. The host CPU architecture must match the machine that will load the
`.so`. Keep build trees outside the repo, and use at most `-j 16`, because CUTLASS translation
units are large.

### B200 on x86_64 (`sm_100a`)

```bash
module load cuda/13.0            # or export CUDA_HOME=/usr/local/cuda-13.x
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} BUILD=/path/to/build-sm100 JOBS=16 \
  TARGETS="flipquant_sm100 test_capi_sm100" bash scripts/build_sm100.sh
# -> $BUILD/libflipquant_sm100.so, $BUILD/test_capi_sm100
$BUILD/test_capi_sm100                 # on the B200: exits 0 iff all cases pass
```

### GB200 (aarch64)

GB200 nodes are Grace (aarch64) and the login node is x86_64. Build on a
GB200 compute node, for example with `srun -A <acct> -p gb200-dev -N1 --gres=gpu:1 --pty bash`.
Run the same command there; the default `BUILD` is arch-tagged.

### RTX PRO 6000 Blackwell / RTX 5090 (`sm_120a`)

```bash
CUDA_HOME=/usr/local/cuda-13.x BUILD=/path/to/build-sm120 JOBS=16 bash scripts/build_sm120_capi.sh
# -> $BUILD/libflipquant_sm120.so (patched), $BUILD/libflipquant_sm120_unpatched.so, $BUILD/test_capi_sm120
$BUILD/test_capi_sm120 $BUILD/libflipquant_sm120.so $BUILD/libflipquant_sm120_unpatched.so
# the 8x64 weights-on-B library and its driver are built by the same script:
$BUILD/test_capi_sm120_b8 $BUILD/libflipquant_sm120_b8.so $BUILD/libflipquant_sm120_b8_unpatched.so
```

The patch step needs `python3`, and it uses `cuobjdump` from the same toolkit
(`CUDAToolkit_BIN_DIR`). When `nvcc` supports `--no-compress`, it is added so that the fatbin
bytes the patcher edits are stored uncompressed. If the patcher reports "could not uniquely
locate the kernel text", suspect fatbin compression first.

## Test drivers

`test_capi_sm100.cu` and `test_capi_sm120.cu` share `test_capi_common.hpp`. Each driver loads its
library with `dlopen`, the same way ctypes does, and runs it on random data. It compares the
result with a host reference that decodes every weight region under its own codebook; this is the
same decode as the research drivers.

A case passes when all four checks hold:
- the relative Frobenius error is below 1e-2;
- the output is far from a reference with every weight codebook flipped (relative error above 5e-2);
- every output value is finite (the output is pre-filled with NaN);
- the return code is `FQ_OK`.

Both drivers also check that bad arguments are rejected. The sm120 driver checks that `fmt_a = 0`
is refused and, if given the unpatched library as a second argument, that it refuses to run.

The shapes cover:
- M = 1 (decode);
- M and N tails;
- `K % 128 == 64`, a partial outer K tile on sm100 and K padding on sm120;
- multiple CTA tiles;
- the two shapes the sm100 research harness validated (144×136×320 and 257×264×1024).

Each shape runs with random, checkerboard, all-E2M1 and all-E0M3 weight maps. On sm100 each also
runs with both activation codebooks.

## Validation status

**Validated upstream, before this wrapper existed:**
- sm100: `n8k64_pipeline<32,128,packed>` is bit-exact on the research harness across 51 cases
  (`scripts/check_sm100_n8_pipeline.sh`), and memcheck and racecheck are clean. Those runs
  used the harness's own launch code, not this library.
- sm120: the weights-as-A configuration was reported correct at 512³, 1024³, 2048³ and
  1024×2048×512 under `MIXFP4_TAG=randa` and `a` on the RTX PRO 6000. Every one of those shapes is
  a multiple of 128.

**Not validated.** Nothing in this directory has been run on real hardware. This includes:
- the wrapper's operand and scale repacking;
- the SF-layout scatter and bit-7 tagging;
- the padding path, including non-multiple-of-128 M, N and K on sm120, which the research runs
  never exercised;
- the Dᵀ transpose;
- the patch marker;
- stream-ordering under PyTorch.

Run `test_capi_sm100` on a B200 and `test_capi_sm120` on an RTX PRO 6000 before trusting either
library.

**Build status (x86_64, CUDA 13.0.88, CUTLASS e64a9136):**
- `libflipquant_sm100.so` compiles for `sm_100a` with no warnings. Its fatbin contains one
  `sm_100a` cubin and no PTX. The SASS contains `UTCOMMA.4X` and `UTCCP` instructions. The only
  dynamic exports are the four `fq_*` symbols, and cudart is linked statically.
- `libflipquant_sm120_unpatched.so` compiles for `sm_120a`. The patcher finds 1024 OMMAs split
  512/512 over sites {0, 1}, which matches `--expect-sites 0,1`. It solves the text base, patches
  512 OMMAs, and sets the marker.

**CUDA 13.0 `cuobjdump` cannot disassemble E0M3 OMMAs.** It prints the patched instructions with an
empty mnemonic. The patched encodings themselves are right: word 1 goes from `…43e04` to
`…47e04`, which is bit 14 (E0M3 on A). The patcher therefore falls back to a byte-exact
verification in that case:
- the native OMMAs must still disassemble as E2M1×E2M1;
- every patched address must carry exactly the written encoding, with nothing else
  undisassemblable.

It prints a WARNING when it uses this fallback. With CUDA 13.1 or later, pass
`-DCUDAToolkit_ROOT` pointing at 13.1 so the full opcode-text census runs.
