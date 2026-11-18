# evaluation/: the FlipQuant paper's evaluation harnesses, ported from NVFP4-RaZeR (port stage 4)

**Source:** NVFP4-RaZeR `kernel-opt` at a84ab6f, the commit `kernels/sm120` was first vendored from.
`kernels/sm120` has since been re-vendored from ae389a5. The harness sources listed here are unchanged
between the two commits.

| here | NVFP4-RaZeR | how |
|---|---|---|
| `lm_eval_datasets.py` | `lm_eval_datasets.py` | vendored unchanged (sha256 5290f5f6…): pins every lm-eval dataset to a commit |
| `downstream.py` | `run_lmeval_deploy.py` (harness, record format, coverage) + `experiments/paper/04_downstream.py` (settings) | ported |
| `decode_check.py` | `sm120/eval/decode_check.py` | ported |
| `latency.py prefill` | `experiments/paper/bench_prefill.py` (step 05) | ported |
| `latency.py decode` | `experiments/paper_extra/bench_decode.py` (Experiment D) | ported |
| `latency.py gemm` | `experiments/paper/bench_gemm_isolated.py` (deviation 2) + `gemm_isolated_tables.per_forward` | ported |
| `latency.py gemm-warm` | `experiments/paper/bench_gemm.py` (step 06, the paper's alternative GEMM method) | ported |

The prefill and decode timings call NVFP4-RaZeR's own `sm120/bench/model.py` (`prefill`, `decode_graph`,
`decode_eager_tokens`) and `sm120/bench/common.py`, as vendored in `kernels/sm120/bench`.

**What the ports change, and only this:**
- **Models and policies.** NVFP4-RaZeR loads its models by its own keys and data roots, and installs exported
  artifacts. Here, a model is a registry key (`models.loader.load`), and a policy is the usual `models.cli` flags: map,
  weight format, `--mode native`, `--kernel-set`. `models/sm120.py` packs the weights in memory with
  NVFP4-RaZeR's packer, and stage 2 validated that packing bit for bit against the artifact path.
- **Kernel names.** These map to the README's stage 2 table; every set computes the same outputs bit for bit.
  - NVFP4-RaZeR's `n8k64_wB`, `paper_mixed` and `paper_stock` are `--kernel-set paper`. The paper's runs used them;
    its `auto` and `auto_stock` meant the paper sets before kernel-opt's adoption.
  - Its current `auto`, `auto_stock`, `auto_wB` and `auto_256` (the adopted kernel-opt sets) are `--kernel-set auto`.
- **GEMM bench inputs.** The GEMM benches take the maps (`--maps 8x64=...`) instead of artifacts. Each module is
  packed with the same packer. The typical, worst and densest modules are chosen by the same rules, from the E0M3
  granule counts.
- **Per-forward sums.** They are computed into the GEMM record (`per_forward`), not rendered into step 07's tables.
- **The latency-only activation override** (`--act-override`) is `models.sm120.install(activation_quantizer=...)`. It
  is NVFP4-RaZeR's `model.install(activation_quantizer=...)`, and the install report records it.
- **One process per policy.** NVFP4-RaZeR's lm-eval harness ran several policies per process; this one runs one. The
  paper ran one per process too.

**Not ported:**
- the kernel micro-experiments of `experiments/paper_extra`: C2-lite (SASS census and the 4096³ GEMM timing) and C3
  (GEMM latency against the E0M3 share). Say so if they are wanted;
- Experiment A (IF4 / Zou MixFP4 per-block rules), which the user excluded from the port;
- step 07's table rendering (`gemm_isolated_tables.sections`, `07_tables.py`);
- the multi-round drivers of steps 04, 05 and 06b, and Experiment D (`D_decode.py`): process orchestration, round
  shuffling and resumption across processes;
- NVFP4-RaZeR's fake-quant lm-eval policies (`fake:map:...`): this repository's `--mode fake` covers them.

**Validation:** `evaluation/port_validation/` and the README's stage 4 section. These are small checks against
NVFP4-RaZeR's own harnesses in the same session, not full runs:
- lm-eval per-example log-likelihoods on limited subsets;
- the decode check;
- the prefill and decode graph checks;
- spot timings.
