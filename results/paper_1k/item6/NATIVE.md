# Item 6: Table A.2 (`tab:app-native`) at the 1k calibration

Native minus simulated perplexity of the 1k maps, both on this GPU and in one env per model (n16k64; n16k64-fast for Nemotron-Nano-9B-v2), on the same windows: native = this repository's SM120 kernels (--kernel-set auto); simulated = `--mode fake`, the same weights, maps and per-token FourOverSix activations. Paired ΔNLL native − simulated, x 1e-3 nats/token, ± 2 SE; § = |ΔNLL| beyond 2 SE. In parentheses: the paper's 256-window cell. Run from flipquant alone (`python -m experiments.paper_1k.run item6`).

## Step 0

Mistral-7B 16x64 with the 256-window release map: **PASS-exact** — simulated vs part C2's record: exact (max per-window |ΔNLL| 0); native vs the release evaluation's: exact (max 0); the cell wiki: -0.16 ± 0.94 (published -0.16 ± 0.94); c4: +0.66 ± 1.24 (published +0.66 ± 1.24).

## Native − simulated

| model | tile | WikiText-2 | C4 |
|---|---|---:|---:|
| Llama-3.2-1B-Instruct | 8x64 | +0.06 ± 1.41 (+0.30 ± 1.70) | -0.94 ± 1.22 (+0.71 ± 1.35) |
| Llama-3.2-1B-Instruct | 16x64 | -0.71 ± 1.85 (+1.88 ± 1.80) | -0.29 ± 1.21 (-1.02 ± 1.24) |
| Llama-3.2-1B-Instruct | 256x64 | +0.15 ± 1.76 (-1.41 ± 1.70) | +0.17 ± 1.19 (-0.42 ± 1.27) |
| Llama-3.1-8B-Instruct | 8x64 | -0.81 ± 1.29 (-0.54 ± 1.33) | -0.87 ± 1.24 (+0.81 ± 1.36) |
| Llama-3.1-8B-Instruct | 16x64 | +0.17 ± 1.28 (+0.21 ± 1.25) | -0.75 ± 1.08 (-1.22 ± 2.23) |
| Llama-3.1-8B-Instruct | 256x64 | +0.44 ± 1.40 (-1.30 ± 1.49) | +0.21 ± 1.33 (+0.37 ± 1.16) |
| Mistral-7B-Instruct-v0.3 | 8x64 | -0.20 ± 0.95 (-0.24 ± 0.99) | -0.03 ± 0.96 (+0.12 ± 0.66) |
| Mistral-7B-Instruct-v0.3 | 16x64 | -0.16 ± 0.98 (-0.16 ± 0.94) | -0.17 ± 0.71 (+0.66 ± 1.24) |
| Mistral-7B-Instruct-v0.3 | 256x64 | +0.39 ± 0.92 (+0.26 ± 0.95) | -0.28 ± 0.70 (-0.01 ± 0.71) |
| Nemotron-Nano-9B-v2 | 8x64 | -1.04 ± 1.03 § (+0.62 ± 1.12) | +0.36 ± 0.67 (-0.40 ± 0.71) |
| Nemotron-Nano-9B-v2 | 16x64 | +0.29 ± 0.90 (+0.04 ± 0.98) | +0.36 ± 0.66 (+0.26 ± 0.71) |
| Nemotron-Nano-9B-v2 | 256x64 | -0.45 ± 1.08 (+0.95 ± 1.05) | +0.63 ± 0.66 (-0.70 ± 0.70) |
| Phi-4 | 8x64 | +0.10 ± 1.16 (+0.08 ± 1.27) | +0.12 ± 0.65 (+0.28 ± 0.73) |
| Phi-4 | 16x64 | +0.67 ± 1.13 (-0.50 ± 1.29) | +0.11 ± 0.64 (+0.25 ± 0.72) |
| Phi-4 | 256x64 | -0.37 ± 1.13 (-0.45 ± 1.29) | +0.08 ± 0.75 (-0.57 ± 0.71) |

**29 of 30 cells agree within 2 SE** (the paper at 256: 28 of 30).

## The native runs against part AA's records

Reported, not gated. exact: equal per window bit for bit; equivalent: the paired ΔNLL within 2 SE on both corpora; differs: otherwise.

| model | tile | tier | max abs ΔNLL | ΔNLL x 1e-3 (WikiText-2; C4) |
|---|---|---|---:|---|
| Llama-3.2-1B-Instruct | 8x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Llama-3.2-1B-Instruct | 16x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Llama-3.2-1B-Instruct | 256x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Llama-3.1-8B-Instruct | 8x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Llama-3.1-8B-Instruct | 16x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Llama-3.1-8B-Instruct | 256x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Mistral-7B-Instruct-v0.3 | 8x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Mistral-7B-Instruct-v0.3 | 16x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Mistral-7B-Instruct-v0.3 | 256x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Nemotron-Nano-9B-v2 | 8x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Nemotron-Nano-9B-v2 | 16x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Nemotron-Nano-9B-v2 | 256x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Phi-4 | 8x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Phi-4 | 16x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
| Phi-4 | 256x64 | exact | 0 | +0.000 ± 0.000; +0.000 ± 0.000 |
