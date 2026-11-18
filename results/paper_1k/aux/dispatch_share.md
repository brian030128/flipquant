# The all-E2M1 dispatch-pattern share (Appendix D, optimization (i)), 256 and 1k maps

Share of (warp, K-tile) dispatch patterns that are all E2M1, FLOP-weighted over each model's quantized linears (out x in), by kernel-opt's `dispatch_ceiling.py` definition (`experiments/paper_1k/dispatch_share.py`, CPU only, on the pinned Hugging Face maps). 16x64 and 256x64: n16k64_wA (a warp's two 16-row granules, rows w and w + 4 of a 128-row panel; for 256x64 maps this equals the deployed 32-row-granule 4-arm kernel's share); 8x64: n8k64_wB (128-column panels; the 64-column builds give the same to 0.01 point). In brackets: the maps' E0M3 tile share.

## The 256-window release maps

| model | 16x64 | 256x64 | 8x64 |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 89.5 % (2.8 %) | 78.9 % (11.4 %) | 92.3 % (2.0 %) |
| Llama-3.1-8B-Instruct | 97.0 % (0.8 %) | 95.7 % (2.2 %) | 97.6 % (0.6 %) |
| Mistral-7B-Instruct-v0.3 | 95.8 % (1.1 %) | 94.8 % (2.7 %) | 96.7 % (0.9 %) |
| Nemotron-Nano-9B-v2 | 95.9 % (1.1 %) | 94.8 % (2.7 %) | 96.6 % (0.9 %) |
| Phi-4 | 97.6 % (0.6 %) | 97.4 % (1.3 %) | 98.1 % (0.5 %) |
| Qwen3.8-27B | 98.2 % (0.4 %) | 98.2 % (0.9 %) | 98.6 % (0.4 %) |

Range over the six models and three units: **78.9–98.6 %** (16x64 89.5–98.2 %; 256x64 78.9–98.2 %; 8x64 92.3–98.6 %); without Llama-3.2-1B-Instruct 94.8–98.6 %.

## The 1k maps

| model | 16x64 | 256x64 | 8x64 |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 53.1 % (14.8 %) | 36.5 % (39.7 %) | 62.3 % (11.3 %) |
| Llama-3.1-8B-Instruct | 79.8 % (5.5 %) | 72.3 % (15.0 %) | 84.0 % (4.3 %) |
| Mistral-7B-Instruct-v0.3 | 80.7 % (5.3 %) | 72.8 % (14.8 %) | 84.8 % (4.1 %) |
| Nemotron-Nano-9B-v2 | 79.7 % (5.6 %) | 70.9 % (16.0 %) | 84.1 % (4.3 %) |
| Phi-4 | 86.1 % (3.7 %) | 80.5 % (10.3 %) | 89.1 % (2.9 %) |
| Qwen3.8-27B | 90.4 % (2.5 %) | 86.5 % (7.0 %) | 91.3 % (2.3 %) |

Range over the six models and three units: **36.5–91.3 %** (16x64 53.1–90.4 %; 256x64 36.5–86.5 %; 8x64 62.3–91.3 %); without Llama-3.2-1B-Instruct 70.9–91.3 %.

## Check against the paper

The release maps give 78.9–98.6 %, the paper's "79–99 %" (its lower end is Llama-3.2-1B-Instruct at 256x64). The source note of that sentence (2026-10-12) lists narrower ranges per unit (16x64 95.4–98.3, 256x64 92.5–98.2, 8x64 96.5–98.6 %) for six release maps it does not name; they do not reach Llama-3.2-1B-Instruct's 78.9 %, so that set was not these six models. Without Llama-3.2-1B-Instruct the maps here give 16x64 95.8–98.2, 256x64 94.8–98.2, 8x64 96.6–98.6 %, close to the note's ranges but not equal to them.

