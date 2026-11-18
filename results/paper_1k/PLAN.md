# Paper reruns at the 1,024-window calibration (part AC): plan and status

> **2026-10-25: hand-off.** The remaining items (item 2's memory probe, 3–9, 11–13 and 1) are run by another author on
> their own RTX PRO 6000, from this repository alone. The steps are in [RUNBOOK.md](RUNBOOK.md); which items are ready
> to run is in [README.md](README.md); what was ported from where is in [PORTING.md](PORTING.md). The "Status" column
> below is this plan's status on the machine where the earlier items ran (its GPU stopped responding on 2026-10-24).

The paper switches FlipQuant, FOCUS and GPTQ‡ from 256 to 1,024 calibration windows. These windows are nested,
256 ⊂ 1k ⊂ 2k, and are the windows of `results/calib_size`. This file lists every experiment that must be re-run
for that switch, in the order the runs go, with its scale, its time estimate and its current status.

- **Models:** the six paper models, Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
  Nemotron-Nano-9B-v2, Phi-4 and Qwen3.8-27B. Qwen3-1.7B and Qwen3-8B are not part of this plan.
- **Maps:** the 1k FlipQuant maps of `results/calib_size` (8x64, 16x64 and 256x64), the FOCUS-1k states and the
  GPTQ‡-1k codes. They are also public as `edgeai-lab/<Base>-{FlipQuant,FOCUS-NVFP4,GPTQ-BF16Calib-NVFP4}-Calib1k`.
- **Hardware:** one RTX PRO 6000 (SM120, driver 595.71.05, 500 W limit, no clock locking), with the jobs run
  one after another. Timing items run on an idle GPU.
- **Step 0:** every item first reproduces, or drift-checks, its existing 256-window record before it measures 1k.
  An item stops on any failed check.
- **Estimates:** taken from the run times of the earlier records of the same kind.
- **Reports:** each finished item's report is pushed to this branch, in `results/paper_1k/<item>/`.

## Must re-run (this GPU)

| # | Paper item | What | Scale | Estimate | Status |
|---|---|---|---|---|---|
| 2 | Table 4 (`tab:latency`), SM120 rows, and the memory paragraph of Section 5 | CUDA-graph prefill, shapes 1x128 to 1x8192 and 4x2048, 5 rounds, method order shuffled per round. The memory probe records installed bytes, allocated after load and the prefill peak. | 6 models × 7 policies (BF16, NVFP4, FO6, FO6-wB, FQ 8x64 / 16x64 / 256x64) | ~4.5 h | Latency DONE: 210 jobs (2026-10-24, 15:30–18:52 UTC), every registered check passes. Step 0: the paired FQ16 / FO6 ratio matched the published session (+0.75 vs +0.82 %), but the absolute times were 0.3–3.7 % faster (a cooler GPU); accepted as a recorded deviation, with a 10-minute warm-up before the panel. Memory: the anchor against the published record passed (bytes equal, 0.00 GiB apart) and Llama-1B BF16 finished; the probe was then interrupted by a GPU hang (see below) and resumes at Llama-1B NVFP4. |
| 3 | Table 5 (`tab:ablation`) | Random (3 seeds), activation-weighted and one-shot-gradient maps matched per layer to the 1k FlipQuant maps' E0M3 counts, with statistics from the 1k windows. Native WikiText-2 and C4 perplexity; ΔNLL against FO6, ±2 SE. | Nemotron-9B and Qwen3.8-27B × 3 tiles × 5 maps | 3–4 h | Registered. `tmopt_ext`'s opt-in `--fit-windows` / `--teacher-host` are committed (2739458) and CPU-tested: with their defaults the trainer command equals the previous code's in 20 of 20 cases. The GPU smoke at 1k and the window token-hash check are this item's step 0 and have not run yet. |
| 4 | Table F.3 (`tab:ptq`) | GPTQ with activation-quantized calibration (NVFP4 and FO6 codes) on the 1k windows. FlipQuant maps trained at 1k on the GPTQ candidates and on the block-16 Hadamard basis. Native perplexity and GSM8K (`gsm8k_llama`). Also the fixed RTN-map-on-GPTQ check. | Nemotron-9B and Qwen3.8-27B: GPTQ NVFP4 / FO6 / FlipQuant (1k map fixed) and the E0M3 candidate codes, two map trainings (GPTQ candidates; Hadamard), 3 PPL rows and 5 GSM8K rows each | ~24–30 h (each 1k training takes 2.6–2.8 h on these two models) | Pending; driver and report generator written, registration drafted |
| 5 | Table D.5 (`tab:app-gemm`) | Per-forward GEMM time (CUPTI, cold weights, 3 rotated rounds of 30 calls). For each projection, the tags of its median-share module and of its max-share module from the 1k maps. | 6 models × 3 tiles × T = 1 to 8192 | 1–3 h | Pending |
| 6 | Table A.2 (`tab:app-native`) | Simulated (fake-quant) perplexity of the 1k maps on the windows of the native 1k records; native − simulated paired ΔNLL ±2 SE and the agreement count. | 5 models × 3 tiles = 15 runs | ~1 h | Pending |
| 7 | Appendix D.6 (reading the weights back) | `--ownership-check` of every 1k map: element totals, format mismatches, negative control. | 18 maps | ~10 min | Pending |
| 8 | Appendix D.4 (one sentence) | The real tags of the Mistral-7B 1k maps against random placement at the same E0M3 share. | 1 model | ~15 min | Pending |
| 9 | Tables D.2 (`tab:app-ktile`) and D.3 (`tab:app-c2`), RTX PRO 6000 rows | Per-MMA branch vs per-K-tile dispatch, and the overhead split (tiles / tag copies / dispatch / E0M3 tiles, before and after the optimizations), at 4096³. The cold method: iso_cold and b2b_cold, the median of 3 rounds per repeat, mean ± SD over 10 repeats. One session. The tags are layer 0 `o_proj` of the Mistral-7B 1k maps. Step 0 with the 256-window tags measures the offset from the paper's values, which were taken on another RTX PRO 6000 board. The RTX 5090 rows are unchanged. | One session | < 1 h | Pending |

## Also run (optional in the paper; included by the user)

| # | Paper item | What | Estimate | Status |
|---|---|---|---|---|
| 11 | Tables C.2 / C.3 (`tab:app-calib-models`, `tab:app-calib-granularity`) | A clean re-measurement of the 1k calibration cost: 6 models × 3 tiles on an idle GPU, from launch to the written map, with the phases recorded. Every map must be bitwise equal to its `results/calib_size` map. Qwen3.8-27B uses `--teacher-host`, and the report says so. | ~19.5 h | cancelled by the user (2026-10-26); filled from part AA records |
| 12 | Table C.5 (`tab:app-robust`) | Mistral-7B 16x64 at 1k: seeds 1 and 2; non-deterministic kernels; FineWeb-Edu, UltraChat and WikiText-2 train with 1,024 sequences each, checked for shared 13-grams with the evaluation sets. Paired ΔNLL against the 1k paper map, the E0M3 share and the Jaccard index. | ~2 h | Pending |
| 13 | Appendix F.1 (Figure F.1, Table F.1) | The tile-gradient check, re-implemented from F.1 because the original code is not available. Step 0 reproduces the published 256-window results within noise. The 1k run uses a NEW held-out set outside the 1k calibration set, because the old held-out sequences lie inside it. | 1–2 h | Ported. The re-implementation (`calibration/gradcheck.py`) is on this branch and CPU-tested; it has not run on a GPU yet, and its 256 step 0 is its validation (RUNBOOK item 13). |

## Last

| # | Paper item | What | Scale | Estimate | Status |
|---|---|---|---|---|---|
| 1 | Table 3 (`tab:main-acc`) Avg. column and Tables E.1–E.3, MMLU column | MMLU 5-shot, lm-eval 0.4.11, no chat template, acc × 100. Step 0 reproduces a published 256-window cell exactly (e.g. Llama-1B FlipQuant 16x64 = 41.40). | (FlipQuant × 3 tiles + FOCUS-1k + GPTQ‡-1k) × 6 models = 30 runs | 8–11 h | Pending |

**Total on this GPU:** about 39–51 h for items 2–9 and 1 (item 2's latency panel is done), plus about 4 h for items 12–13 (item 11 was cancelled by the user on 2026-10-26; filled from part AA records).

## Not run here

- **#10, the GB200 / SM100 items (co-author):** Table 4's SM100 rows, Appendix D.7's per-forward GEMM overhead
  (median 1.7 %), Appendix A.4's GB200 native-vs-simulated check, and the SM100 weight read-back, all with the
  1k 256x64 maps.
- **Already measured, so filled from records without a new run:**
  - Table 2 (`tab:main-ppl`), Figure 1, the GSM8K column of Table 3, and the six zero-shot tasks of Tables E.1–E.3;
  - the FlipQuant rows of Table B.1;
  - Table C.4 (E0M3 share) and Table C.6 (the budget grid re-based on the 1k map), with Table C.7 unchanged;
  - Table F.2, re-based on the 1k map, with FOCUS 1024 × 1 taken from `results/calib_size`.
- **Not affected by the switch:**
  - the BF16, NVFP4, FO6, IF4 and MixFP4 rows;
  - Table B.2 (the rules' 1x16 choices);
  - Table D.4 (random placement);
  - the SM100 narrow-tile figure.

## Incident: GPU hang during item 2 (2026-10-24 18:53 UTC)

- **What failed:** the memory probe's Llama-1B NVFP4 job failed at model load with `cudaErrorDevicesUnavailable`.
  The job's threads stayed blocked in the driver. nvidia-smi then reported utilization, power and clocks as [N/A].
- **What was not affected:** the PCIe AER error counters are 0 and the device is in D0. The latency panel had
  finished cleanly 27 s earlier (18:52:48 UTC), and the memory anchor and Llama-1B BF16 finished after it, so
  those records are unaffected.
- **Cause:** not determinable from inside the container, because there is no kernel log or Xid. The symptoms
  point to a driver / GPU-firmware (GSP) hang, not to the workload.
- **Recovery:** the GPU needs a reset or a pod restart. After it, a CUDA health check runs first; then item 2
  resumes at the memory probe and the order above continues.

## Order

2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 12 → 13 → 1. Item 11: cancelled by the user (2026-10-26); filled from part AA records.
