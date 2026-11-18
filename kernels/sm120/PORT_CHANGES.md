# kernels/sm120: NVFP4-RaZeR's SM120 deployment stack, vendored

**Source:** NVFP4-RaZeR `sm120/` on branch `kernel-opt` at 9580a2a (first vendored at a84ab6f; re-vendored on
2026-10-04 at ae389a5 with the 8x64 plan and kernel-opt amendments 17 and 18; re-vendored on 2026-10-25 at 9580a2a
for the paper's 1k reruns, item 9). `VENDORED.json` lists every file and its sha256 at vendoring.
- **9580a2a over ae389a5:** amendment 20's registration (4ec02b9) only: the per-MMA-branch option for the K-tile
  ablation (`-DMIXFP4_PER_MMA_BRANCH=1`, the blob generator's `BRANCH=a|b`, the patcher's `--predicate-aware`,
  `KernelConfig.allow_predicated`) and the two configurations `n16k64_wA_e64_t0_permma` / `n8k64_wB_t0_permma`. They
  are not deployment kernels. With the option unset every configuration's generated blob is byte-identical (checked
  for all 47 configurations with a blob), so the deployment builds' SASS (`SASS_REFERENCE.json`) is unchanged.
- **Not vendored:** `results/`, `maps/`, `eval/reference/` (data) and `third_party/cutlass`. This repository's
  submodule `kernels/3rdparty/cutlass` is the same commit (e64a9136), and `build.py` checks the pin.
- **The kernel itself:** NVFP4-RaZeR vendors the mixfp4 kernel from brian030128/mixfp4 at 7b3ab34, with
  compile-time-only local changes. `kernel/NOTICE.md`, `kernel/VENDORED.json` and `kernel/LOCAL_CHANGES.md` /
  `.patch` carry its provenance and license status unchanged: no upstream license file, used with the author's
  permission within the team, and an explicit license is needed before any public release.

**Changes made here, and nothing else:**

| file | change |
|---|---|
| `build.py` | CUTLASS is `kernels/3rdparty/cutlass` (this repository's submodule), or `$RAZER_CUTLASS` (`build_deploy.sh` sets it from `$CUTLASS_DIR`); the pinned-commit check is unchanged |
| `tests/conftest.py` | `$REFERENCE_ROOT` (an NVFP4-RaZeR checkout) is appended to `sys.path` for the reference quantizers |
| `tests/test_numerics.py` | skips when NVFP4-RaZeR's `quantize/` reference package is not importable (set `$REFERENCE_ROOT`) |

This repository adds `PORT_CHANGES.md`, `VENDORED.json`, `SASS_REFERENCE.json`, `build_deploy.sh`,
`validate_models.py` and `validation/`.

**What it contains:**
- the kernel sources and the SASS patcher (`kernel/`);
- the build script (`build.py`), which builds one library per configuration into `build/` (or `$SM120_BUILD_DIR`)
  and checks the SASS census;
- the runtime package `mixfp4_sm120` (ctypes binding, kernel sets and tile tables, NativeLinear, artifacts);
- the per-GPU tile tables (`configs/`):
  - the paper table;
  - the adopted `.ko.json`: the 16x64 4b widths and scheduler rows, the 8x64 widths and `stock_wB_ko`'s scheduler
    rows (amendments 11-12b), and the 256x64 widths and scheduler rows (amendment 18);
- the evaluation and benchmark scripts (`eval/`, `bench/`);
- the tests (`tests/`).

**Kernel families** (`mixfp4_sm120/select.py` FAMILIES; the `auto` routes of `models/sm120.py` in bold):

| family | builds | role |
|---|---|---|
| `mixed` | n16k64_wA (+ narrow n16 / n32 / n64) | the paper's 16x64 path (and 256x64 maps, as 16x64 granules) |
| `stock` | stock_wA (+ narrow) | the paper's baselines |
| `n8k64_wB` | single build | the paper's 8x64 kernel |
| **`mixed_ko`** | t0 builds, #4's epilogue tile at width 128 | the adopted 16x64 path (amendment 7) |
| **`mixed_wB_ko`** | the weights-on-B t0 builds, widths 16 / 32 / 64 / '128x64' / 128 | the adopted 8x64 path (amendments 11-12b) |
| **`mixed256_ko`** | A′'s g32 t0 builds, #4's epilogue tile at width 128 | the adopted 256x64 path (amendment 18) |
| **`stock_ko`** | stock with #4 | the adopted baselines |
| **`stock_wB_ko`** | stock_wB_e64 | the baselines with the weights on B, tuned alike (amendments 12-12b); route `auto_stock_wB` |
| `mixed256` | A′'s g32 builds | the previous 256x64 path (A′: 4 arms, 32-row granules), on the paper table |
| `mixed_wB` | n8k64_wB_m16 / m32 / m64 and the 128x64 build | kernel-opt optimization 1/1b, superseded by `mixed_wB_ko` |

The no-dispatch ceilings (`nodisp_ko`, `nodisp_wB_ko`, `nodisp256_ko`), the t0 sets and the diagnostics (nodisp, xor,
sk and the like) are in configs.py and select.py, but they are not deployment kernels.

**The adopted dispatch variants are build variants** (NVFP4-RaZeR docs/BUILD_AND_USE.md):

| set | define | widths |
|---|---|---|
| `mixed_ko`, `mixed_wB_ko` | #2's frequency-aware dispatch (`MIXFP4_DISPATCH_FREQ=1`) | all |
| `mixed_ko` | the uniform-branch dispatch (`MIXFP4_UNIFORM_DISPATCH=1`) | 64, 128 |
| `mixed_wB_ko` | the pipelined flag read (`MIXFP4_PIPE_FLAGS=1`) | 64, '128x64', 128 |
| `mixed256_ko` | the uniform-branch dispatch (`MIXFP4_UNIFORM_DISPATCH=1`) | all (and not #2's) |

- `build_deploy.sh` builds them into the deployment directory `build_deploy`: NVFP4-RaZeR's kernel-opt `build_V` (22
  builds), plus the paper sets' 7 builds, so that `--kernel-set paper` runs from the same directory.
- Every variant computes the same outputs bit for bit. `models/sm120.py`'s routing note says whether the loaded
  builds are the adopted variants.

**Verification:**
- `SASS_REFERENCE.json` holds the SASS of NVFP4-RaZeR's gated builds at kernel-opt ae389a5, registered with its
  amendment 18:
  - all 56 configurations without defines (`build_Vall`);
  - the 13 deployment builds with defines (`build_V`).
- Every build here must reproduce the reference of its configuration and defines (`tests/test_sm120_native.py`).
- The same test file checks the adapter (`models/sm120.py`) against NVFP4-RaZeR's own artifact path and package. It
  covers outputs and whole-model logits on a small model, every route, CUDA-graph replay and the ownership check.
- `validate_models.py` checks the paper's models against NVFP4-RaZeR's own install. The results are in `validation/`
  and are summarized in the README.

**Build**, CPU only, CUDA 13.1 (`build.py` checks the toolchain):

```bash
git submodule update --init kernels/3rdparty/cutlass
CUDA_HOME=/path/to/cuda-13.1 python kernels/sm120/build.py --all          # every configuration, into build/
CUDA_HOME=/path/to/cuda-13.1 bash kernels/sm120/build_deploy.sh            # the deployment directory build_deploy
```
