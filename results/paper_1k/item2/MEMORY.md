# Item 2: memory on the RTX PRO 6000 at 1k (the paper's memory paragraph)

Part B's probe (`experiments/paper_1k/mem_probe.py`: flipquant's loader and install, this repository's SM120 build) with the 1k maps (the Calib1k repositories), fo6-wB added to part B's six policies.

- **Weights:** every parameter and buffer on the GPU after the load and install.
- **After load:** torch.cuda.memory_allocated.
- **Peak:** torch.cuda.max_memory_allocated during one 1x2048 eager prefill.

## Step 0: the anchor (Mistral, FlipQuant 16x64 with the 256-window release map, against part B's record)

- Installed quantized linears equal: True; whole model equal: True.
- After load Δ +0.0000 GiB; 1x2048 peak Δ +0.0000 GiB (tolerance ±0.02 GiB). **PASS.**

## The panel

| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | bf16 | 2.30 GiB | 1.0000 | 1.000 | 2.30 GiB | 2.87 GiB |
| Llama-3.2-1B-Instruct | nvfp4 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fo6 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fo6-wB | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fq-8x64 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fq-16x64 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.2-1B-Instruct | fq-256x64 | 1.00 GiB | 0.2812 | 0.434 | 1.00 GiB | 1.61 GiB |
| Llama-3.1-8B-Instruct | bf16 | 14.96 GiB | 1.0000 | 1.000 | 14.96 GiB | 15.72 GiB |
| Llama-3.1-8B-Instruct | nvfp4 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.49 GiB |
| Llama-3.1-8B-Instruct | fo6 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.49 GiB |
| Llama-3.1-8B-Instruct | fo6-wB | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.49 GiB |
| Llama-3.1-8B-Instruct | fq-8x64 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.48 GiB |
| Llama-3.1-8B-Instruct | fq-16x64 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.48 GiB |
| Llama-3.1-8B-Instruct | fq-256x64 | 5.61 GiB | 0.2812 | 0.375 | 5.65 GiB | 6.48 GiB |
| Mistral-7B-Instruct-v0.3 | bf16 | 13.50 GiB | 1.0000 | 1.000 | 13.50 GiB | 13.99 GiB |
| Mistral-7B-Instruct-v0.3 | nvfp4 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fo6 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fo6-wB | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fq-8x64 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fq-16x64 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Mistral-7B-Instruct-v0.3 | fq-256x64 | 4.16 GiB | 0.2812 | 0.308 | 4.19 GiB | 4.68 GiB |
| Nemotron-Nano-9B-v2 | bf16 | 16.56 GiB | 1.0000 | 1.000 | 16.56 GiB | 18.25 GiB |
| Nemotron-Nano-9B-v2 | nvfp4 | 6.23 GiB | 0.2813 | 0.376 | 6.25 GiB | 8.02 GiB |
| Nemotron-Nano-9B-v2 | fo6 | 6.23 GiB | 0.2813 | 0.376 | 6.25 GiB | 8.02 GiB |
| Nemotron-Nano-9B-v2 | fo6-wB | 6.23 GiB | 0.2813 | 0.376 | 6.25 GiB | 8.02 GiB |
| Nemotron-Nano-9B-v2 | fq-8x64 | 6.23 GiB | 0.2813 | 0.376 | 6.25 GiB | 8.02 GiB |
| Nemotron-Nano-9B-v2 | fq-16x64 | 6.23 GiB | 0.2813 | 0.376 | 6.25 GiB | 8.02 GiB |
| Nemotron-Nano-9B-v2 | fq-256x64 | 6.23 GiB | 0.2813 | 0.376 | 6.25 GiB | 8.02 GiB |
| Phi-4 | bf16 | 27.31 GiB | 1.0000 | 1.000 | 27.31 GiB | 28.11 GiB |
| Phi-4 | nvfp4 | 9.06 GiB | 0.2812 | 0.332 | 9.08 GiB | 9.97 GiB |
| Phi-4 | fo6 | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fo6-wB | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fq-8x64 | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fq-16x64 | 9.06 GiB | 0.2812 | 0.332 | 9.07 GiB | 9.96 GiB |
| Phi-4 | fq-256x64 | 9.06 GiB | 0.2812 | 0.332 | 9.08 GiB | 9.97 GiB |
| Qwen3.8-27B | bf16 | 50.96 GiB | 1.0000 | 1.000 | 50.96 GiB | 52.23 GiB |
| Qwen3.8-27B | nvfp4 | 18.36 GiB | 0.2813 | 0.360 | 18.40 GiB | 19.75 GiB |
| Qwen3.8-27B | fo6 | 18.36 GiB | 0.2813 | 0.360 | 18.41 GiB | 19.75 GiB |
| Qwen3.8-27B | fo6-wB | 18.36 GiB | 0.2813 | 0.360 | 18.41 GiB | 19.75 GiB |
| Qwen3.8-27B | fq-8x64 | 18.36 GiB | 0.2813 | 0.360 | 18.39 GiB | 19.75 GiB |
| Qwen3.8-27B | fq-16x64 | 18.36 GiB | 0.2813 | 0.360 | 18.39 GiB | 19.75 GiB |
| Qwen3.8-27B | fq-256x64 | 18.36 GiB | 0.2813 | 0.360 | 18.39 GiB | 19.75 GiB |

## The paper's memory claims, per model

- **Llama-3.2-1B-Instruct:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and FourOverSix (wB): **yes**; after load and at the 1x2048 peak at most 0.0000 / 0.0010 GiB apart (within 0.02 GiB: **yes**); BF16 / FlipQuant whole model 2.30x.
- **Llama-3.1-8B-Instruct:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and FourOverSix (wB): **yes**; after load and at the 1x2048 peak at most 0.0073 / 0.0079 GiB apart (within 0.02 GiB: **yes**); BF16 / FlipQuant whole model 2.66x.
- **Mistral-7B-Instruct-v0.3:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and FourOverSix (wB): **yes**; after load and at the 1x2048 peak at most 0.0059 / 0.0051 GiB apart (within 0.02 GiB: **yes**); BF16 / FlipQuant whole model 3.25x.
- **Nemotron-Nano-9B-v2:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and FourOverSix (wB): **yes**; after load and at the 1x2048 peak at most 0.0028 / 0.0023 GiB apart (within 0.02 GiB: **yes**); BF16 / FlipQuant whole model 2.66x.
- **Phi-4:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and FourOverSix (wB): **yes**; after load and at the 1x2048 peak at most 0.0084 / 0.0071 GiB apart (within 0.02 GiB: **yes**); BF16 / FlipQuant whole model 3.02x.
- **Qwen3.8-27B:** FlipQuant 8x64 / 16x64 / 256x64 byte-identical to NVFP4, FourOverSix and FourOverSix (wB): **yes**; after load and at the 1x2048 peak at most 0.0128 / 0.0030 GiB apart (within 0.02 GiB: **yes**); BF16 / FlipQuant whole model 2.78x.
