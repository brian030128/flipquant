# Item 4: Table F.3 (`tab:ptq`) at the 1k calibration (Nemotron-Nano-9B-v2, Qwen3.8-27B; 16x64)

Round 3's protocol (NVFP4-RaZeR paper-eval PROTOCOL.md part O) and part H's Hadamard rows with the 1k windows, run from flipquant alone (`python -m experiments.paper_1k.run item4`). Native: this repository's SM120 build, --kernel-set auto. GPTQ with activation quantization on during calibration on the 1,024 windows (the record's 128 + the extension's 896); the GPTQ-candidate map and the Hadamard map trained at 1k (Qwen3.8-27B with the teacher on the host). ΔNLL: paired per window vs RTN FourOverSix (re-measured in this session), x 1e-3 ± 2 SE (* beyond 2 SE). GSM8K: part N's protocol (lm-eval 0.4.11 gsm8k_llama, chat template, greedy, thinking off, 1,319 problems) at batch 64; strict-match ± 2 SE; paired per problem, A − B in pp ± 2 SE (A-only / B-only, McNemar's exact p).

Reused rows (they do not depend on the calibration windows, or are part AA's 1k RTN map): their PPL is re-measured here with their records' commands (agreement with the paper's per-window NLLs below); their GSM8K results are the paper's (BF16: round 3's O0, Qwen3.8-27B at batch 48; RTN / Hadamard NVFP4 and FourOverSix: part N; RTN FlipQuant 1k: part AA), so GSM8K pairs with them compare this board with the paper's.

## Step 0

Round 3's Nemotron GPTQ NVFP4 job (256 windows) on this code and board: codes_sha256 equal to round 3's (`6a9c9d292d49a82e`); data hash equal: True; per-window NLL against round 3's: exact (max |ΔNLL| 0).

## The table

| model | method | format | WikiText-2 | C4 | ΔNLL wiki vs RTN FO6 | ΔNLL C4 vs RTN FO6 | GSM8K strict (%) | 256: WikiText-2 / C4 / GSM8K |
|---|---|---|---:|---:|---:|---:|---:|---|
| Nemotron-Nano-9B-v2 | BF16 | -- | 8.0863 | 11.1528 | -41.15 ± 1.64* | -30.96 ± 1.34* | 87.41 ± 1.83 | unchanged |
| Nemotron-Nano-9B-v2 | RTN | NVFP4 | 8.4535 | 11.5437 | +3.26 ± 1.51* | +3.50 ± 1.00* | 87.34 ± 1.83 | unchanged |
| Nemotron-Nano-9B-v2 | RTN | FourOverSix | 8.4260 | 11.5034 | — | — | 88.93 ± 1.73 | unchanged |
| Nemotron-Nano-9B-v2 | RTN | FlipQuant (16x64) | 8.3267 | 11.4397 | -11.86 ± 1.32* | -5.56 ± 0.84* | 88.63 ± 1.75 | 8.3532 / 11.4452 / 89.31 |
| Nemotron-Nano-9B-v2 | GPTQ (act.) | NVFP4 | 8.4323 | 11.5039 | +0.76 ± 1.49 | +0.04 ± 1.11 | 84.46 ± 2.00 | 8.4494 / 11.5223 / 83.78 |
| Nemotron-Nano-9B-v2 | GPTQ (act.) | FourOverSix | 8.3825 | 11.4646 | -5.18 ± 1.40* | -3.38 ± 0.94* | 86.88 ± 1.86 | 8.3950 / 11.4884 / 87.11 |
| Nemotron-Nano-9B-v2 | GPTQ (act.) | FlipQuant (16x64) | 8.3738 | 11.4604 | -6.21 ± 1.42* | -3.75 ± 0.99* | 84.99 ± 1.97 | 8.3936 / 11.4920 / 86.81 |
| Nemotron-Nano-9B-v2 | GPTQ cand. + map | FlipQuant (16x64) | 8.3234 | 11.4275 | -12.25 ± 1.39* | -6.62 ± 0.98* | 84.99 ± 1.97 | 8.3550 / 11.4483 / 87.64 |
| Nemotron-Nano-9B-v2 | Hadamard | NVFP4 | 8.5178 | 11.6449 | +10.84 ± 2.04* | +12.23 ± 1.32* | 84.91 ± 1.97 | unchanged |
| Nemotron-Nano-9B-v2 | Hadamard | FourOverSix | 8.4535 | 11.5580 | +3.26 ± 1.91* | +4.73 ± 1.17* | 88.17 ± 1.78 | unchanged |
| Nemotron-Nano-9B-v2 | Hadamard | FlipQuant (16x64) | 8.3357 | 11.4625 | -10.77 ± 1.86* | -3.57 ± 1.12* | 88.17 ± 1.78 | 8.3727 / 11.4953 / 88.93 |
| Qwen3.8-27B | BF16 | -- | 7.0523 | 9.8931 | -34.84 ± 4.75* | -29.60 ± 2.57* | 95.68 ± 1.12 | unchanged |
| Qwen3.8-27B | RTN | NVFP4 | 7.5740 | 10.2231 | +36.53 ± 8.12* | +3.22 ± 1.08* | 95.38 ± 1.16 | unchanged |
| Qwen3.8-27B | RTN | FourOverSix | 7.3023 | 10.1903 | — | — | 94.62 ± 1.24 | unchanged |
| Qwen3.8-27B | RTN | FlipQuant (16x64) | 7.0784 | 10.1224 | -31.15 ± 5.18* | -6.68 ± 1.07* | 95.91 ± 1.09 | 7.1859 / 10.1410 / 95.15 |
| Qwen3.8-27B | GPTQ (act.) | NVFP4 | 7.4409 | 10.1651 | +18.80 ± 5.84* | -2.48 ± 1.21* | 95.91 ± 1.09 | 7.3706 / 10.1799 / 96.06 |
| Qwen3.8-27B | GPTQ (act.) | FourOverSix | 7.3132 | 10.1299 | +1.49 ± 5.19 | -5.94 ± 1.00* | 96.21 ± 1.05 | 7.4282 / 10.1433 / 95.98 |
| Qwen3.8-27B | GPTQ (act.) | FlipQuant (16x64) | 7.2170 | 10.1306 | -11.76 ± 4.50* | -5.88 ± 1.07* | 96.66 ± 0.99 | 7.2270 / 10.1390 / 95.98 |
| Qwen3.8-27B | GPTQ cand. + map | FlipQuant (16x64) | 7.1132 | 10.1022 | -26.24 ± 5.27* | -8.68 ± 1.19* | 95.83 ± 1.10 | 7.3109 / 10.1261 / 96.06 |
| Qwen3.8-27B | Hadamard | NVFP4 | 7.6126 | 10.3425 | +41.61 ± 11.36* | +14.83 ± 2.14* | 95.75 ± 1.11 | unchanged |
| Qwen3.8-27B | Hadamard | FourOverSix | 7.7335 | 10.2574 | +57.36 ± 12.87* | +6.57 ± 1.46* | 96.21 ± 1.05 | unchanged |
| Qwen3.8-27B | Hadamard | FlipQuant (16x64) | 7.2051 | 10.1526 | -13.40 ± 5.04* | -3.70 ± 1.30* | 96.21 ± 1.05 | 7.3684 / 10.1923 / 95.75 |

## The re-check: the 1k RTN map on the 1k GPTQ candidates (no retraining)

| model | WikiText-2 | C4 | ΔNLL vs RTN FO6 (wiki; C4) | retrained map − re-check (wiki; C4) |
|---|---:|---:|---:|---:|
| Nemotron-Nano-9B-v2 | 8.3549 | 11.4517 | -8.47 ± 1.42*; -4.50 ± 0.97* | -3.78 ± 1.16*; -2.12 ± 0.80* |
| Qwen3.8-27B | 7.1642 | 10.1085 | -19.09 ± 4.43*; -8.06 ± 1.07* | -7.15 ± 2.92*; -0.62 ± 0.78 |

## GSM8K, paired per problem

| model | A vs B | strict-match Δ (pp) | flexible-extract Δ (pp) |
|---|---|---:|---:|
| Nemotron-Nano-9B-v2 | gptq_nvfp4 vs gptq_fo6 | -2.43 ± 1.76* (52 / 84, p 0.00762) | -2.58 ± 1.51* (33 / 67, p 0.000874) |
| Nemotron-Nano-9B-v2 | gptq_fq-16x64 vs gptq_fo6 | -1.90 ± 1.75* (54 / 79, p 0.037) | -0.23 ± 1.33 (37 / 40, p 0.82) |
| Nemotron-Nano-9B-v2 | gptqcand_fq-16x64 vs gptq_fo6 | -1.90 ± 1.71* (51 / 76, p 0.0328) | -0.83 ± 1.38 (36 / 47, p 0.272) |
| Nemotron-Nano-9B-v2 | gptqcand_fq-16x64 vs gptq_fq-16x64 | +0.00 ± 1.76 (67 / 67, p 1) | -0.61 ± 1.34 (35 / 43, p 0.428) |
| Nemotron-Nano-9B-v2 | gptq_fq-16x64 vs rtn_fq-16x64 | -3.64 ± 1.78* (46 / 94, p 6.11e-05) | -1.44 ± 1.40* (33 / 52, p 0.0503) |
| Nemotron-Nano-9B-v2 | gptqcand_fq-16x64 vs rtn_fq-16x64 | -3.64 ± 1.82* (49 / 97, p 8.77e-05) | -2.05 ± 1.44* (32 / 59, p 0.00611) |
| Nemotron-Nano-9B-v2 | hadamard_fq-16x64 vs hadamard_fo6 | +0.00 ± 1.62 (57 / 57, p 1) | +0.61 ± 1.49 (52 / 44, p 0.475) |
| Nemotron-Nano-9B-v2 | hadamard_fq-16x64 vs rtn_fq-16x64 | -0.45 ± 1.61 (53 / 59, p 0.637) | -1.59 ± 1.33* (28 / 49, p 0.022) |
| Nemotron-Nano-9B-v2 | rtn_fq-16x64 vs rtn_fo6 | -0.30 ± 1.58 (52 / 56, p 0.773) | +0.53 ± 1.30 (40 / 33, p 0.483) |
| Qwen3.8-27B | gptq_nvfp4 vs gptq_fo6 | -0.30 ± 0.86 (14 / 18, p 0.597) | -0.15 ± 0.86 (15 / 17, p 0.86) |
| Qwen3.8-27B | gptq_fq-16x64 vs gptq_fo6 | +0.45 ± 0.80 (17 / 11, p 0.345) | +0.45 ± 0.80 (17 / 11, p 0.345) |
| Qwen3.8-27B | gptqcand_fq-16x64 vs gptq_fo6 | -0.38 ± 0.87 (14 / 19, p 0.487) | -0.23 ± 0.84 (14 / 17, p 0.72) |
| Qwen3.8-27B | gptqcand_fq-16x64 vs gptq_fq-16x64 | -0.83 ± 0.84 (10 / 21, p 0.0708) | -0.68 ± 0.82 (10 / 19, p 0.136) |
| Qwen3.8-27B | gptq_fq-16x64 vs rtn_fq-16x64 | +0.76 ± 0.86 (21 / 11, p 0.11) | +0.38 ± 0.82 (17 / 12, p 0.458) |
| Qwen3.8-27B | gptqcand_fq-16x64 vs rtn_fq-16x64 | -0.08 ± 0.97 (20 / 21, p 1) | -0.30 ± 0.91 (16 / 20, p 0.618) |
| Qwen3.8-27B | hadamard_fq-16x64 vs hadamard_fo6 | +0.00 ± 0.86 (16 / 16, p 1) | -0.30 ± 0.86 (14 / 18, p 0.597) |
| Qwen3.8-27B | hadamard_fq-16x64 vs rtn_fq-16x64 | +0.30 ± 0.96 (22 / 18, p 0.636) | +0.00 ± 0.91 (18 / 18, p 1) |
| Qwen3.8-27B | rtn_fq-16x64 vs rtn_fo6 | +1.29 ± 1.04* (32 / 15, p 0.0186) | +1.06 ± 0.91* (25 / 11, p 0.0288) |

## The trainings' environments

| model | training | environment (this run) | published route (256) | training seconds |
|---|---|---|---|---:|
| Nemotron-Nano-9B-v2 | gptqcand | n16k64 (the recurrent layers' fallback kernels) | n16k64 (the recurrent layers' fallback kernels) | 9,529 |
| Nemotron-Nano-9B-v2 | hadamard | n16k64-fast (mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, fla 0.5.2) | n16k64-fast (mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, fla 0.5.2) | 1,771 |
| Qwen3.8-27B | gptqcand | n16k64 (the recurrent layers' fallback kernels) | n16k64 (the recurrent layers' fallback kernels) | 8,630 |
| Qwen3.8-27B | hadamard | n16k64-fast (mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, fla 0.5.2) | n16k64-fast (mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, fla 0.5.2) | 5,415 |

The GPTQ-candidate trainings keep round 3's n16k64 env (the hybrid models' recurrent layers on their fallback, as Appendix C states for the calibrations); the Hadamard trainings keep part H's n16k64-fast env (the fast recurrent kernels), so F.3's Hadamard maps are calibrated on the fast kernels (a paper-text point for F.3). The trainer's deterministic setting covers torch's own operations, not the custom kernels of mamba-ssm, causal-conv1d or fla.

**Auxiliary check (not a gate): part H's exact 256 Hadamard training once more on this board** (Nemotron twice), each map's tiles against part H's map (`item4/fastenv_repeat.json`):

| model | run | environment | E0M3 tiles | tiles digest | equal to part H | first epoch whose train_kl differs from part H's | training seconds |
|---|---|---|---:|---|---|---:|---:|
| Nemotron-Nano-9B-v2 | part H (2026-10-08) | n16k64-fast | 80,280 | `e65d233feb6e10b1` | — | — | 478 |
| Nemotron-Nano-9B-v2 | repeat 1 | n16k64-fast | 80,408 | `659470f20ca3b705` | **no** | 1 | 448 |
| Nemotron-Nano-9B-v2 | repeat 2 | n16k64-fast | 80,199 | `efca46f841107c59` | **no** | 1 | 446 |
| Qwen3.8-27B | part H (2026-10-08) | n16k64-fast | 101,471 | `b00f95b3d226a998` | — | — | 1,349 |
| Qwen3.8-27B | repeat 1 | n16k64-fast | 101,471 | `b00f95b3d226a998` | yes | none | 1,355 |

- **Nemotron-Nano-9B-v2:** not bit-reproducible run to run: the repeats of one command differ on this board and code (the repeats' train_kl first differs at epoch 1).
- **Qwen3.8-27B:** reproduces part H's map bit for bit.

## Checks

| check | passed |
|---|---|
| nemotron-nano-9b-v2: GPTQ quantized Hessian and propagation, 1,024 windows, one data hash | True |
| nemotron-nano-9b-v2: gptq_nvfp4 GSM8K codes = PPL codes | True |
| nemotron-nano-9b-v2: gptq_fo6 GSM8K codes = PPL codes | True |
| nemotron-nano-9b-v2: gptq_fq-16x64 GSM8K codes = PPL codes | True |
| nemotron-nano-9b-v2: bf16 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| nemotron-nano-9b-v2: rtn_nvfp4 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| nemotron-nano-9b-v2: rtn_fo6 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| nemotron-nano-9b-v2: rtn_fq-16x64 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| nemotron-nano-9b-v2: hadamard_nvfp4 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| nemotron-nano-9b-v2: hadamard_fo6 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| qwen3.8-27b: GPTQ quantized Hessian and propagation, 1,024 windows, one data hash | True |
| qwen3.8-27b: gptq_nvfp4 GSM8K codes = PPL codes | True |
| qwen3.8-27b: gptq_fo6 GSM8K codes = PPL codes | True |
| qwen3.8-27b: gptq_fq-16x64 GSM8K codes = PPL codes | True |
| qwen3.8-27b: bf16 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| qwen3.8-27b: rtn_nvfp4 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| qwen3.8-27b: rtn_fo6 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| qwen3.8-27b: rtn_fq-16x64 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| qwen3.8-27b: hadamard_nvfp4 PPL against the paper's record (reported): exact / equivalent / differs | exact |
| qwen3.8-27b: hadamard_fo6 PPL against the paper's record (reported): exact / equivalent / differs | exact |
