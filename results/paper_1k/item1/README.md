# Item 1: MMLU column (Table 3, Tables E.1–E.3)

Status: **done**.

## Summary (2026-10-27)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000: step 0 and 31 MMLU jobs, all rc 0. Llama-3.2-1B-Instruct,
  Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 ran on 2026-10-26 with the registered command. Phi-4, Qwen3.8-27B
  and Nemotron-Nano-9B-v2 ran on 2026-10-27, after the GPU hang of 2026-10-26 (`INCIDENT_2026-10-26.md`) and the pod
  restart, in the user's order (Phi-4, Qwen3.8-27B, Nemotron last), one invocation per model.
- **Step 0 (gate `item1-step0`): pass, exact.** Llama-3.2-1B-Instruct FlipQuant 16x64 with the release map: 41.40
  (published 41.40).
- **Deviation (A), ¤ (environment):** the Phi-4, Qwen3.8-27B and Nemotron rows ran with
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, after Nemotron's first row ran out of GPU memory at batch 16. The
  command and batch are unchanged (batch 16; Qwen3.8-27B 8). No job ran out of memory under (A); the batch fallback
  (B) was not used. How the published 256 column fitted on its 96 GB pod is not recorded.
- **Consistency row (gate `item1-consistency`): pass, exact.** Nemotron FlipQuant 16x64 with the release map under
  (A): 73.34 (published 73.34).
- **MMLU at 1k (FlipQuant 8x64 / 16x64 / 256x64 / FOCUS / GPTQ‡, %):** Llama-3.2-1B 41.91 / 41.92 / 41.68 / 42.54 /
  42.37; Llama-3.1-8B 65.31 / 65.72 / 65.94 / 65.95 / 65.44; Mistral 60.01 / 60.13 / 60.10 / 60.19 / 60.25; Nemotron
  73.29 / 73.54 / 73.49 / 73.54 / 72.77; Phi-4 79.20 / 79.18 / 79.31 / 79.08 / 79.27; Qwen3.8-27B 82.55 / 82.63 /
  82.60 / 83.04 / 81.41.
- **Paired per question (FlipQuant 16x64 against):** FOCUS: no model beyond 2 SE (largest Qwen3.8-27B −0.41 ± 0.44).
  GPTQ‡: FlipQuant better on Nemotron (+0.76 ± 0.48) and Qwen3.8-27B (+1.22 ± 0.48), no difference beyond 2 SE on the
  other four. The three tiles: no difference beyond 2 SE on any model.
- **Table 3's Avg. at 1k** (this MMLU with part AA's six zero-shot tasks) is in `MMLU.md` / `table_mmlu.tex`.

Files: `INCIDENT_2026-10-26.md`, `MMLU.md`, `item1.json`, `table_mmlu.tex`.
