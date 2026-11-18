# Item 1: the MMLU column (Table 3, Tables E.1-E.3) at the 1k calibration

MMLU 5-shot (lm-eval 0.4.11, no chat template, acc x 100, 14,042 questions; `evaluation.downstream --tasks mmlu`, its default batch) of the 1k rows, native on this repository's SM120 build, with part AA's evaluation flags. ± 2 SE: binomial. In parentheses: the paper's 256-window value. Avg.: Table 3's mean of seven tasks, this MMLU and part AA's six zero-shot tasks of the same artifact. The other rows keep the paper's values. Run from flipquant alone (`python -m experiments.paper_1k.run item1`).

## Step 0

Llama-3.2-1B-Instruct FlipQuant 16x64 with the 256-window release map: MMLU 41.40 against the published 41.40 — **exact** (+1 questions; 2 SE = 0.83 pp).

## Deviations (registered 2026-10-26)

Nemotron-Nano-9B-v2's first row ran out of GPU memory at the tool's batch 16 on this 96 GB board (lm-eval's float32 log-softmax over a padded batch of 5-shot prompts and a 131k vocabulary; 28.5 GiB of the process's memory reserved but unallocated). The published 256 MMLU column was run on another RTX PRO 6000 pod (also 96 GB) with the colleague's spec; how it fitted there is not recorded here.

- **¤ (environment):** the job ran with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (PAPER1K_ALLOC_CONF); the command and batch are unchanged.
- **‡b (command):** the job ran at a reduced batch (`--batch-size`), the fallback for a model that still ran out of memory under ¤.

Consistency rows (FlipQuant 16x64 with the 256-window release map under the deviation, against the published 256 cell; gate `item1-consistency`, the step-0 two-tier rule):

| model | batch | environment | MMLU (%) | published | tier | Δ questions |
|---|---:|---|---:|---:|---|---:|
| Nemotron-Nano-9B-v2 | 16 | ¤ | 73.34 | 73.34 | exact | +1 |

## MMLU and Avg. at 1k

| model | row | MMLU (%) | Avg. (%) |
|---|---|---:|---:|
| Llama-3.2-1B-Instruct | FlipQuant 8x64 | 41.91 ± 0.83 (40.74) | 55.55 (55.47) |
| Llama-3.2-1B-Instruct | FlipQuant 16x64 | 41.92 ± 0.83 (41.40) | 55.65 (54.99) |
| Llama-3.2-1B-Instruct | FlipQuant 256x64 | 41.68 ± 0.83 (42.02) | 55.77 (55.48) |
| Llama-3.2-1B-Instruct | FOCUS | 42.54 ± 0.83 (42.40) | 55.70 (55.26) |
| Llama-3.2-1B-Instruct | GPTQ‡ | 42.37 ± 0.83 (41.84) | 54.52 (54.44) |
| Llama-3.1-8B-Instruct | FlipQuant 8x64 | 65.31 ± 0.80 (66.52) | 71.38 (71.72) |
| Llama-3.1-8B-Instruct | FlipQuant 16x64 | 65.72 ± 0.80 (66.01) | 71.34 (71.64) |
| Llama-3.1-8B-Instruct | FlipQuant 256x64 | 65.94 ± 0.80 (65.60) | 70.83 (71.33) |
| Llama-3.1-8B-Instruct | FOCUS | 65.95 ± 0.80 (65.97) | 71.90 (71.43) |
| Llama-3.1-8B-Instruct | GPTQ‡ | 65.44 ± 0.80 (65.57) | 70.76 (70.95) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 8x64 | 60.01 ± 0.83 (60.23) | 73.31 (73.33) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 | 60.13 ± 0.83 (59.80) | 73.47 (73.49) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 256x64 | 60.10 ± 0.83 (59.59) | 73.04 (73.14) |
| Mistral-7B-Instruct-v0.3 | FOCUS | 60.19 ± 0.83 (60.20) | 73.20 (73.02) |
| Mistral-7B-Instruct-v0.3 | GPTQ‡ | 60.25 ± 0.83 (59.35) | 73.42 (73.48) |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | 73.29 ± 0.75¤ (73.54) | 71.58¤ (71.41) |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | 73.54 ± 0.74¤ (73.34) | 71.44¤ (71.11) |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | 73.49 ± 0.74¤ (73.58) | 71.56¤ (71.15) |
| Nemotron-Nano-9B-v2 | FOCUS | 73.54 ± 0.74¤ (73.14) | 71.93¤ (71.83) |
| Nemotron-Nano-9B-v2 | GPTQ‡ | 72.77 ± 0.75¤ (73.05) | 71.39¤ (71.18) |
| Phi-4 | FlipQuant 8x64 | 79.20 ± 0.69¤ (79.23) | 73.40¤ (73.54) |
| Phi-4 | FlipQuant 16x64 | 79.18 ± 0.69¤ (79.08) | 73.69¤ (73.51) |
| Phi-4 | FlipQuant 256x64 | 79.31 ± 0.68¤ (79.01) | 73.57¤ (73.47) |
| Phi-4 | FOCUS | 79.08 ± 0.69¤ (79.17) | 73.91¤ (74.00) |
| Phi-4 | GPTQ‡ | 79.27 ± 0.68¤ (79.46) | 73.76¤ (73.59) |
| Qwen3.8-27B | FlipQuant 8x64 | 82.55 ± 0.64¤ (82.62) | 75.18¤ (75.28) |
| Qwen3.8-27B | FlipQuant 16x64 | 82.63 ± 0.64¤ (82.35) | 75.59¤ (75.11) |
| Qwen3.8-27B | FlipQuant 256x64 | 82.60 ± 0.64¤ (81.68) | 75.21¤ (75.04) |
| Qwen3.8-27B | FOCUS | 83.04 ± 0.63¤ (82.82) | 75.91¤ (75.52) |
| Qwen3.8-27B | GPTQ‡ | 81.41 ± 0.66¤ (81.24) | 74.57¤ (75.05) |

## Paired per question, at 1k (A − B, pp ± 2 SE; A-only / B-only; McNemar's exact p)

| model | A vs B | Δ |
|---|---|---:|
| Llama-3.2-1B-Instruct | FlipQuant 16x64 vs FOCUS | -0.63 ± 0.83 (1645 / 1733, p 0.134) |
| Llama-3.2-1B-Instruct | FlipQuant 16x64 vs GPTQ‡ | -0.46 ± 0.81 (1586 / 1650, p 0.268) |
| Llama-3.2-1B-Instruct | FlipQuant 8x64 vs FlipQuant 16x64 | -0.01 ± 0.72 (1291 / 1292, p 1) |
| Llama-3.2-1B-Instruct | FlipQuant 256x64 vs FlipQuant 16x64 | -0.24 ± 0.77 (1427 / 1460, p 0.551) |
| Llama-3.1-8B-Instruct | FlipQuant 16x64 vs FOCUS | -0.22 ± 0.60 (865 / 896, p 0.475) |
| Llama-3.1-8B-Instruct | FlipQuant 16x64 vs GPTQ‡ | +0.28 ± 0.63 (989 / 949, p 0.376) |
| Llama-3.1-8B-Instruct | FlipQuant 8x64 vs FlipQuant 16x64 | -0.41 ± 0.55 (711 / 769, p 0.138) |
| Llama-3.1-8B-Instruct | FlipQuant 256x64 vs FlipQuant 16x64 | +0.21 ± 0.56 (800 / 770, p 0.464) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 vs FOCUS | -0.06 ± 0.54 (726 / 735, p 0.834) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 16x64 vs GPTQ‡ | -0.13 ± 0.56 (762 / 780, p 0.665) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 8x64 vs FlipQuant 16x64 | -0.12 ± 0.50 (598 / 615, p 0.646) |
| Mistral-7B-Instruct-v0.3 | FlipQuant 256x64 vs FlipQuant 16x64 | -0.03 ± 0.52 (666 / 670, p 0.935) |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 vs FOCUS | +0.00 ± 0.46 (524 / 524, p 1) |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 vs GPTQ‡ | +0.76 ± 0.48* (632 / 525, p 0.00182) |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 vs FlipQuant 16x64 | -0.24 ± 0.40 (384 / 418, p 0.244) |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 vs FlipQuant 16x64 | -0.05 ± 0.42 (429 / 436, p 0.838) |
| Phi-4 | FlipQuant 16x64 vs FOCUS | +0.11 ± 0.43 (467 / 452, p 0.644) |
| Phi-4 | FlipQuant 16x64 vs GPTQ‡ | -0.09 ± 0.42 (431 / 443, p 0.71) |
| Phi-4 | FlipQuant 8x64 vs FlipQuant 16x64 | +0.01 ± 0.37 (338 / 336, p 0.969) |
| Phi-4 | FlipQuant 256x64 vs FlipQuant 16x64 | +0.13 ± 0.39 (377 / 359, p 0.531) |
| Qwen3.8-27B | FlipQuant 16x64 vs FOCUS | -0.41 ± 0.44 (439 / 497, p 0.0624) |
| Qwen3.8-27B | FlipQuant 16x64 vs GPTQ‡ | +1.22 ± 0.48* (643 / 472, p 3.38e-07) |
| Qwen3.8-27B | FlipQuant 8x64 vs FlipQuant 16x64 | -0.08 ± 0.40 (392 / 403, p 0.723) |
| Qwen3.8-27B | FlipQuant 256x64 vs FlipQuant 16x64 | -0.03 ± 0.42 (428 / 432, p 0.919) |
