# Item 2: Table 4 (`tab:latency`) SM120 rows; the memory paragraph

Status: **latency done (33bad82), memory done**.

## Summary (memory, 2026-10-25)

- **Run:** the memory probe from a fresh flipquant clone on an RTX PRO 6000 (500 W), 43 processes (the anchor and
  6 models × 7 policies with the 1k maps), one session; every job rc 0. The latency rows (`LATENCY.md`) are unchanged
  (33bad82).
- **Step 0 (gate `item2-anchor`): pass.** Mistral-7B FlipQuant 16x64 with the 256-window release map against part B's
  record: installed quantized linears and the whole model byte-identical; after load and at the 1x2048 peak
  +0.0000 GiB (tolerance 0.02).
- **The paper's claims, all six models: yes.** FlipQuant 8x64 / 16x64 / 256x64 install byte-identical quantized
  linears and whole models to NVFP4 / FourOverSix / FourOverSix (wB) (quantized linears 0.281 of BF16), and after load
  and at the 1x2048 peak they are at most 0.0128 GiB apart (Qwen3.8-27B, after load). BF16 / FlipQuant whole model:
  2.30x (Llama-3.2-1B-Instruct) to 3.25x (Mistral-7B-Instruct-v0.3).

Files: `LATENCY.md`, `MEMORY.md`, `item2.json`, `memory.json`.
