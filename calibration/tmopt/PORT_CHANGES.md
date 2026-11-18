# calibration/tmopt: NVFP4-RaZeR's TM-OPT+TC trainer, vendored

**Source:** NVFP4-RaZeR, branch `topk-cal` at a53ce55. `VENDORED.json` lists every file and its sha256;
`tests/test_stage3.py` checks them.
- **First vendored** from `flipquant-maps` at c89ee2c. Its calibration code is identical to `tm-opt`'s and
  `kernel-opt`'s. That commit trained the committed maps of Qwen3-1.7B, Qwen3-8B, Mistral-7B-Instruct-v0.3 and
  Nemotron-Nano-9B-v2. Before that, it reproduced the paper's Mistral-7B-v0.3 8x64 map bit for bit.
- **Re-vendored on 2026-10-05** from `topk-cal`, which was cut from c89ee2c. It differs only by flipquant's top-K
  teacher (`run_train_map.py --teacher-topk K`; NVFP4-RaZeR `results/topk_cal`):
  - `run_train_map.py` and `chunked_loss.py` (`train_kl_gradient_topk`);
  - the new `topk_teacher.py` and `repro_local/realquant/test_topk_teacher.py`.

  Every other file is unchanged. With `--teacher-topk 0` (the default) the trainer is the paper path, bit for bit.
- **Re-vendored again on 2026-10-05** from `topk-cal` a53ce55: `run_train_map.py --fit-windows N` and the new
  `fit_extension.py` (the paper's 128 fit windows, then N - 128 by flipquant's extension rule). With
  `--fit-windows 128` (the default) the trainer is unchanged. Added with it, as records at their NVFP4-RaZeR paths, the
  inputs of the extension's skip set for llama8b: `results/kse_paper/job_336566/llama8b/report.json` (the published C4
  evaluation documents) and the three development manifests under `results/task_reorder/transfer_20260920/`.
  `calibration.tmopt_common.prepare_development` places the manifests where the trainer's `data_paths` reads them.
- **The test** `repro_local/realquant/test_topk_teacher.py` compares the top-K path with flipquant's own functions. It
  reads them from `$FLIPQUANT_ROOT`'s co-authors' trainer at `$FLIPQUANT_REF`; set them to this checkout and `HEAD`
  (see `validation/topk/`). **The one change made here to a vendored file:** the trainer moved from
  `calibration/train_map.py` to `calibration/train_map_legacy.py`, so the test reads `$FLIPQUANT_TRAINER` if set, else
  `calibration/train_map_legacy.py`, else (at refs before the move) `calibration/train_map.py`, and records which. Its
  hash is listed in `tests/test_stage3.py` (`PORT_MODIFIED`), not in `VENDORED.json`.

- **Calibration robustness runs** (branch `calib-robustness`, paper Appendix C.4): `run_train_map.py` gains
  `--fit-windows N` below 128 (the first N/2 math and N/2 code windows of the paper's fit set) and `--fit-source S`
  (N windows of FineWeb-Edu, UltraChat or the WikiText-2 training split, drawn by the new `fit_sources.py`). Their
  defaults (`--fit-windows 128`, `--fit-source paper`) leave the paper's fit set and every map unchanged.
  Non-deterministic runs use the trainer's existing `--no-deterministic`, which `calibration.train_map
  --no-deterministic` now passes.

**Unchanged** (except that test and the robustness options above). Every file here is NVFP4-RaZeR's, byte for byte, at its NVFP4-RaZeR path. This
repository's code is `calibration/train_map.py` and `calibration/tmopt_common.py`. They run the trainer in its own
process, from this directory, with the environment of NVFP4-RaZeR's paper runs. This file and `validation/` are the
port's own.

**What it contains:**
- **The trainer:** `run_train_map.py` (TM-OPT+TC) and its import closure.
  - `run_multiround.py`, `run_math_code_calibration.py` and the other `run_*.py` it imports;
  - `quantize/`: the reference quantizers, fused FourOverSix, packed candidates;
  - `repro_local/realquant/`: the candidate store and its Triton kernels;
  - `chunked_loss.py`, `cost_monitor.py`, `profile_regions.py`;
  - `topk_teacher.py`: flipquant's top-K teacher (`--teacher-topk`), and its test in `repro_local/realquant/`.
- **The fit sets:** `prepare_model_data.py`, and `research/n16k64/software/primary/campaign/data.py`
  (`builder_seed0`, the rule behind Llama-3.1-8B's fit set).
- **The records the trainer and the wrapper are checked against:**
  - the archived fit sets: `results/math_code_adaptive/calibration_333779_{llama8b,qwen4b}`,
    `calibration_333787_qwen27b`, and `research/n16k64/campaigns/primary/calibration_manifests/mistral7b_seed0.json`;
  - the paper's map sha256 values (`experiments/paper/maps.sha256.json`);
  - for `--fit-windows`, the development records and published C4 evaluation documents the extension rule skips:
    Llama-3.1-8B's three development manifests (`results/task_reorder/transfer_20260920/`) and
    `results/kse_paper/job_336566/llama8b/report.json`; Qwen3.8-27B's development fresh.pt sha256 values
    (`results/mixfp4_potential/qwen27b/multiround_kl_8x64_final.json`; its sets are redrawn and accepted only by
    them) and `results/kse_paper/job_336969/qwen27b/report.json`;
  - the paper's calibration flags (`experiments/paper/paper_common.py`: `TM_OPT_TC`, `TRAIN_BATCH`).

**Scope.** Item 9 of the port request, NVFP4-RaZeR's QAT and scale-only (SCALE) trainers of its unified OURS / QAT /
SCALE comparison, was dropped by the user's decision on 2026-10-02. This repository does not contain them:
- **`run_cost_distill.py`**, the QAT and SCALE trainers, is not vendored.
- **`quantize/learned_scale.py`** is not vendored either. Only `run_train_map.py --base-scales` reads it (TM-OPT+TC on
  top of a SCALE run's learned scales), so that option stops with `ModuleNotFoundError` here.
  `calibration.train_map` never passes it.
- **`quantize/fused_fourover6.py`** is in TM-OPT+TC's import closure and is vendored unchanged. It still defines
  NVFP4-RaZeR's `LearnedScaleE2M1`, the SCALE arm's training function, but nothing here uses it.

**Requirements:**
- **Hardware and software:** a CUDA GPU with Triton. Bitwise reproduction of the paper's maps is established on an RTX
  PRO 6000 (sm_120), with torch 2.9.0+cu128, transformers 5.16.1 and triton 3.5.0.
- **Data:** the models at NVFP4-RaZeR's pinned revisions, and the calibration datasets at their pinned revisions
  (open-web-math, codeparrot-clean). `prepare` reads both through the Hugging Face cache and downloads what is
  missing; Llama-3.1-8B is gated.
- **Tokenizers:** a model without an archived fit set has its fit set drawn by the builder rule. That rule must first
  reproduce the Llama-3.1-8B, Qwen3-4B and Mistral-7B-v0.3 records, so their tokenizers are needed too.

**Validation:** `validation/`, and the README's stage 3 section.
