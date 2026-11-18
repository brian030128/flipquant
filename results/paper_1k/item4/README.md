# Item 4: Table F.3 (`tab:ptq`)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000 (500 W): the 43 registered jobs (17 h; all rc 0) and three
  auxiliary jobs (not gates).
- **Step 0 (gate `item4-step0`): pass.** Round 3's Nemotron GPTQ NVFP4 job at 256 windows: codes_sha256 equal to
  round 3's, data hash equal, per-window NLL exact.
- **Reused rows:** the PPL of BF16, RTN NVFP4 / FourOverSix / FlipQuant 1k and Hadamard NVFP4 / FourOverSix equals the
  paper's records per window bit for bit (exact). Their GSM8K results are the paper's, so GSM8K pairs with them compare
  two boards.
- **Headline (ΔNLL vs RTN FourOverSix, x 1e-3, WikiText-2 / C4):**
  - Nemotron: GPTQ NVFP4 +0.8 / +0.0, GPTQ FourOverSix −5.2 / −3.4, GPTQ FlipQuant −6.2 / −3.8, GPTQ candidates
    + map −12.3 / −6.6 (the best PPL row), RTN FlipQuant −11.9 / −5.6, Hadamard FlipQuant −10.8 / −3.6.
  - Qwen3.8-27B: GPTQ NVFP4 +18.8 / −2.5, GPTQ FourOverSix +1.5 / −5.9, GPTQ FlipQuant −11.8 / −5.9, GPTQ candidates
    + map −26.2 / −8.7, RTN FlipQuant −31.2 / −6.7 (still the best on WikiText-2), Hadamard FlipQuant −13.4 / −3.7.
  - Retraining on the GPTQ candidates beats the re-check (the 1k RTN map on them) on WikiText-2 for both models
    (−3.8 and −7.2).
  - GSM8K (paired): on Nemotron, GPTQ FlipQuant and GPTQ candidates + map are 3.6 pp below RTN FlipQuant (strict,
    beyond 2 SE). On Qwen3.8-27B nothing differs beyond 2 SE except RTN FlipQuant vs RTN FourOverSix (+1.3 pp).
- **Checks:** per model the three GPTQ rows share one data hash, with 1,024 windows and the quantized Hessian; every
  GSM8K row ran its PPL row's codes; both trainings' fit extensions are the paper's 1k set.
- **Environments (a paper-text point for F.3):** the GPTQ-candidate trainings run in n16k64 (round 3's route: the
  recurrent layers on their fallback, as Appendix C states), the Hadamard trainings in n16k64-fast (part H's route,
  confirmed from its job logs: mamba-ssm / causal-conv1d / fla). F.3's Hadamard maps are calibrated on the fast
  kernels.
- **Auxiliary fast-env repeat (not a gate):** part H's exact 256 Hadamard training once more (Nemotron twice).
  - Qwen3.8-27B reproduces part H's map bit for bit (101,471 E0M3 tiles).
  - Nemotron is **not bit-reproducible run to run**: 80,280 (part H), 80,408 and 80,199 E0M3 tiles with three
    different digests. Every run's epoch-0 train_kl is equal; they diverge from epoch 1.
  - One forward/backward epoch at a fixed map does reproduce in this env: phase A's rebuild of part G's Nemotron
    one-shot gradient (n16k64-fast) was bit-equal.
  - So the 1k Nemotron Hadamard map is one draw of a non-deterministic training.

Files: `REPORT.md`, `item4.json`, `table_ptq.tex`.
