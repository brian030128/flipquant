# RUNBOOK: the paper's remaining reruns at the 1,024-window calibration, from this repository alone

This is for an agent or a person with **no other context**. It assumes:

- a fresh clone of flipquant (`main`);
- one **NVIDIA RTX PRO 6000 Blackwell** (SM120) GPU;
- the public Hugging Face artifacts and datasets.

Nothing else is needed: no other repository, no private storage, no paths from the machine where the earlier items ran.

- [PLAN.md](PLAN.md): what each item is and why it is rerun.
- [README.md](README.md): which items are **ready** in this checkout.
- [PORTING.md](PORTING.md): where each piece came from.

**Order:** 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 12 → 13 → 1. (Item 11 was cancelled by the user (2026-10-26); filled from part AA records: not run.)

- Run only items the README index marks **ready**.
- When an item is not ready yet, skip it and run the next ready one. Then `git pull` and come back.

**The general rules:**
- **Stop rules.** Every item starts with a step 0 that reproduces, or drift-checks, an earlier record. **A failed step
  0, a failed gate or a failed job stops the item.** Report it and do not work around it.
- **Resuming.** A finished record is never run again, so after a fix the same command resumes the item where it
  stopped.
- **Smoke runs.** `PAPER1K_SMOKE_JOBS=N python -m experiments.paper_1k.run <item> …` stops the item (`SMOKE` in
  `queue.log`, exit 3) before it would start its (N+1)-th job. The gates between the jobs that ran are evaluated as
  usual. Running the item again without it continues from there. With N = the number of step-0 jobs (the first jobs
  of the dry run), this runs step 0 alone.
- **Cluster rules.** Read `AGENTS.md` first. On a Slurm login node, run nothing heavy: submit jobs instead.
- **One GPU job at a time.** The runner waits until no other compute process is on the GPU. Timing items need an idle
  GPU with nothing else running. Each job starts at least 5 s after the previous GPU job exited (a mitigation, after
  a GPU hang about 1 s after a job's exit on the porting machine; it changes no command).
- **Power limit.** The paper's SM120 numbers were measured at a **500 W** power limit, with no clock locking and no
  profiler. The runner records the board's power limit, clocks, driver and temperatures (`gpu_state.jsonl`, and each
  record's own snapshots). Change nothing on the board (power limit, clocks, persistence) without your user's OK.
- **Comparisons across boards.** Your board, driver or build need not reproduce the paper's floating-point numbers bit
  for bit.
  - **Two tiers for GPU numerics.** A gate that compares your run's perplexity numbers with the paper's records
    (`experiments/paper_1k/compare.py`) passes as **PASS-exact** when every per-window NLL is equal bit for bit. It
    passes as **PASS-equivalent** when they are not equal but, per corpus, the paired ΔNLL (your run − the record) is
    within 2 SE. Otherwise it is a **FAIL**. The record says which tier applied and gives the largest per-window
    difference.
  - **Exact gates.** Board-independent quantities stay exact: map tile digests, GPTQ code digests, readback rows and
    installed bytes.
  - **Timing** compares only within your own session and against your own step 0. Where a step 0 drift-checks the
    paper's timing, the criterion is relative (item 5) or reported only.
- **When a gate fails.** Every check against the paper's records is a named gate (the table below). A failed gate stops
  the item with `STOP … gate <name> failed: …` in `queue.log`, before the long part of the item.
  - **This is a question for your user, not something to work around.** Report the gate, its record and the numbers.
  - Continue only with your user's explicit OK, by running the item again with `PAPER1K_ACCEPT=<name>` (several names:
    comma-separated). The failure is then logged as ACCEPTED, and the item's report marks it.

  | gate | item | what must hold | kind |
  |---|---|---|---|
  | `item2-anchor` | 2 | the memory anchor: installed and whole-model bytes equal part B's; after load and peak within 0.02 GiB | exact bytes + tolerance |
  | `item3-rebuild` | 3 | part G's 256 one-shot map rebuilt: its tiles digest equals G's | exact (map digest) |
  | `item4-step0` | 4 | round 3's 256 GPTQ NVFP4 codes: `codes_sha256` equals round 3's | exact (codes digest) |
  | `item5-step0` | 5 | Phi-4's T ≥ 128 overheads within 2 points of the published run's | relative timing |
  | `item6-step0` | 6 | the 256 cell, simulated and native, against C2's and the release evaluation's records | two-tier |
  | `item7-step0` | 7 | part C1's readback rows of the 256 release map (digest) | exact |
  | `item8-step0` | 8 | the release maps' real tags within 1 point of random placement, every typical cell | relative timing |
  | `item9-sass` | 9 | the seven extra builds' SASS equals the references | exact |
  | `item9-census` | 9 | the D.2 kernels' static SASS census equals amendment 20's | exact |
  | `item9-check` | 9 | per-MMA outputs equal per-K-tile outputs bit for bit | exact |
  | `item11-maps` | 11 | every calibration's map tiles digest equals the 1k map's (item 11 cancelled: unused) | exact (map digest) |
  | `item12-step0` | 12 | a seed-0 256 recalibration of Mistral-7B 16x64 reproduces the release map (tiles digest) | exact (map digest) |
  | `item12-audit` | 12 | each fit source's 1,024 windows share no 13-gram with an evaluation text; the control finds every splice | exact |
  | `item13-step0` | 13 | the re-implemented F.1 at 256 reproduces the published results (criteria 1–4 of its section) | statistical |
  | `item1-step0` | 1 | Llama-3.2-1B-Instruct FlipQuant 16x64 (256) MMLU against the published 41.40 | two-tier (accuracy) |
  | `item1-consistency` | 1 | under a deviation (below): FlipQuant 16x64 (256) MMLU against the published 256 cell | two-tier (accuracy) |

  The other stops (a failed job, a missing prerequisite, a failed negative control, a structural check) are not
  gates: fix the cause, or report it.
- **Publishing.** Push to `paper-1k-reruns` only (never `main`), after each item, with the publish step of section 3.

## 1. Setup (once)

### 1.1 Environments

Three conda environments, pinned in `experiments/paper_1k/env/`:

| file | for | key pins |
|---|---|---|
| `n16k64.yml` | evaluation and calibration (all models but the two hybrids) | Python 3.11.11, torch 2.9.0+cu128, transformers 5.16.1, lm-eval 0.4.11, accelerate 1.13.0, datasets 4.8.5, triton 3.5.0, huggingface_hub 1.31.0 |
| `n16k64-fast.yml` | Nemotron-Nano-9B-v2 and Qwen3.8-27B (their recurrent layers' fast kernels) | the same + mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, flash-linear-attention / fla-core 0.5.2 (CUDA 12.8 build tools) |
| `mixfp4-cuda131.yml` | building the SM120 kernels (CPU only) | the CUDA 13.1.115 toolkit (nvcc, ptxas, cuobjdump, nvdisasm), gcc 15.3, cmake 3.31.8 |

```bash
conda env create -f experiments/paper_1k/env/n16k64.yml        # mamba-ssm / causal-conv1d build against CUDA 12.8:
conda env create -f experiments/paper_1k/env/n16k64-fast.yml   #   if pip's wheel fails, install them with --no-build-isolation
conda env create -f experiments/paper_1k/env/mixfp4-cuda131.yml
export PAPER1K_PY=$(conda run -n n16k64 which python)            # the driver's interpreters
export PAPER1K_PY_FAST=$(conda run -n n16k64-fast which python)
python -m experiments.paper_1k.setup env                          # PASS: every pinned version in both interpreters
```

Run every command in this file from the repository root, with `$PAPER1K_PY` (or the n16k64 env active) as `python`.
`PYTHONPATH` must not point at any other checkout. The runner drops `PYTHONPATH`, `SM120_BUILD_DIR` and
`PYTORCH_CUDA_ALLOC_CONF` from every job.

### 1.2 The SM120 kernels (CPU build, SASS gate)

```bash
git submodule update --init kernels/3rdparty/cutlass     # CUTLASS at the pinned e64a9136 (or CUTLASS_DIR=<checkout>)
export CUDA_HOME=$(conda run -n mixfp4-cuda131 sh -c 'echo $CONDA_PREFIX')
python -m experiments.paper_1k.setup kernels      # kernels/sm120/build_deploy.sh, then the SASS gate
```

- **Pass:** every build's patched and unpatched SASS equals `kernels/sm120/SASS_REFERENCE.json`. Its deployment builds
  are the paper's `build_V`.
- **If the SASS differs**, the toolkit is not CUDA 13.1.115 (or the sources changed). Stop: the native items would not
  run the paper's kernels.
- **Never build on a machine whose GPU driver is hung** (nvidia-smi hangs or reports [N/A]). Even a CPU-only build with
  no visible GPU is unsafe there: its `nvlink` step opens the NVIDIA UVM device, then blocks in uninterruptible sleep
  and cannot be killed until the GPU is reset.
- The build goes to `kernels/sm120/build_deploy` (`PAPER1K_BUILD` overrides). The native jobs pass it as
  `--kernel-build-dir`. Their records' library hashes are checked against this build's manifests.

### 1.3 Hugging Face

```bash
hf auth login                                     # a token whose account accepted the Llama 3.1 / 3.2 licenses
python -m experiments.paper_1k.setup hf           # PASS: the gated Llama models, every pinned artifact repository
python -m experiments.paper_1k.setup fetch        # every pinned artifact (about 69 GB: maps 0.4, GPTQ‡ codes 68),
                                                  # each checked by sha256
```

- `experiments/paper_1k/artifacts.json` pins 24 public repositories (`edgeai-lab/<Base>-{FlipQuant,
  GPTQ-BF16Calib-NVFP4}[-Calib1k]`) at their revisions of 2026-10-25, with every artifact's sha256. A file whose sha256
  differs is refused.
- The model weights come from the models' own repositories, at the revisions pinned in `models/registry.py`.
- `fetch` can be limited: `python -m experiments.paper_1k.artifacts fetch --kind flipquant_1k,flipquant_256`.
- **Disk.** The artifacts go to the Hugging Face cache (`HF_HOME`). To put them on another disk while the model weights
  stay in `HF_HOME`, set `PAPER1K_HF_CACHE=<dir>` (a hub cache for the artifacts only) for setup and for every item.
  Only item 1 reads the GPTQ‡ codes, and only their Calib1k versions; the 256 versions are pinned for completeness
  and used by no item.

### 1.4 Calibration records and windows (CPU; needed from item 3 on)

```bash
python -m experiments.paper_1k.setup records --data-root paper1k_data
python -m experiments.paper_1k.setup windows --data-root paper1k_data --out paper1k_runs
```

- `records`: `calibration.tmopt_ext record` per model (the release fit set, from the pinned public datasets). **Pass:**
  each record's revision, fit-set token hashes and matrix digest equal `experiments/paper_1k/reference/records.json`.
- `windows`: `calibration.fit_windows` with 256, 1,024 and 2,048 windows per model. **Pass:** every window's token
  hash equals `experiments/paper_1k/windows/<model>_<N>.json`: the earlier tables' 256 set (step 0s reuse it), the
  paper's 1k set and the 2k set. The three nest: 256 ⊂ 1k ⊂ 2k.
- The outputs (`paper1k_runs/windows/<model>_<N>.json`) are the `--calib-extension` files later items read.

## 2. The items

### Item 2: Table 4 (`tab:latency`) SM120 rows and the memory paragraph — the memory probe only

- **Do not re-run the latency panel.** Its 210 jobs are done and published (`item2/LATENCY.md`, `item2/item2.json`;
  flipquant 33bad82). The driver has no latency job.
- **What remains:** part B's memory probe (`experiments/paper_1k/mem_probe.py`). It records the installed bytes, the
  allocation after the load and the peak during one 1x2048 prefill.
  - The 1k maps; 6 models × 7 policies: bf16, nvfp4, fo6, fo6-wB, fq-8x64, fq-16x64, fq-256x64.
  - Each job is its own process, with the paper-eval runs' exact flags.
  - The earlier board finished the anchor and Llama-3.2-1B-Instruct BF16 before its GPU failed
    (`reference/item2.json`, `measured_here_20261024`). Run the whole panel on your board anyway, so that the memory
    paragraph comes from one session.
- **Prerequisites:** sections 1.1–1.3 (maps: `--kind flipquant_1k,flipquant_256`).
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item2 --dry-run   # paper1k_runs/dry_run/item2.txt: the 43 commands
  python -m experiments.paper_1k.run item2             # step 0 (the anchor), then the panel
  ```
- **Step 0, the anchor.** Mistral-7B FlipQuant 16x64 with the 256-window release map, against part B's record
  (`reference/item2.json` 'anchor').
  - **Pass** (gate `item2-anchor`): installed and whole-model bytes equal; after load and at the 1x2048 peak within
    0.02 GiB.
  - On a failure the driver stops before the panel (`STOP` in `paper1k_runs/queue.log`; the comparison is in
    `paper1k_runs/item2/memory_anchor_check.json`). See "When a gate fails".
- **Gates:** every job rc 0 and a complete record.
  - The report checks the paper's claims: FlipQuant 8x64 / 16x64 / 256x64 install byte-identical quantized linears and
    whole models to NVFP4 / FourOverSix / FourOverSix (wB), and after load and at the peak differ by at most 0.02 GiB.
  - These are results to report, not gates.
- **Time:** about 20 minutes (43 processes of 10–40 s each).
- **Outputs:** `paper1k_runs/item2/memory/<model>/<policy>.json`, `memory_anchor/`, `memory_anchor_check.json`, the
  job logs in `paper1k_runs/logs/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item2 --report            # results/paper_1k/item2/MEMORY.md and memory.json
  python -m experiments.paper_1k.publish item2 --status "latency done (f60666e), memory done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  Write `SUMMARY.md` yourself, a few lines with the anchor result and the claims, and keep it out of the repository. The
  latency files in `results/paper_1k/item2/` stay as published.

### Item 3: Table 5 (`tab:ablation`) at the 1k calibration

- **What:** how the map is chosen, on Nemotron-Nano-9B-v2 and Qwen3.8-27B at 8x64 / 16x64 / 256x64 (part G's protocol
  with the 1k maps; registered in NVFP4-RaZeR paper-eval PROTOCOL.md, part AC item 3).
  - Every baseline flips exactly the 1k map's E0M3 tiles per layer (`--match` the 1k map):
    - random, seeds 0 / 1 / 2 (`calibration.baseline_maps --method random`);
    - activation-weighted (`--method act`, the BF16 inputs over the 1k windows: the record's 128, then the extension's
      896);
    - one-shot gradient (`calibration.tmopt_ext oneshot --fit-windows 1024`: the trainer's tile gradient at the
      all-E2M1 start; Qwen3.8-27B keeps the top-1000 teacher on the host, `--teacher-host`).
  - Each map's native WikiText-2 / C4 perplexity, against FourOverSix.
  - FourOverSix and the 1k FlipQuant maps are re-measured in the item.
- **Prerequisites:**
  - sections 1.1–1.4, with the records and windows of both models;
  - the maps: `python -m experiments.paper_1k.artifacts fetch --kind flipquant_1k,flipquant_256 --models
    nemotron-nano-9b-v2,qwen3.8-27b`.
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item3 --data-root paper1k_data --dry-run   # paper1k_runs/dry_run/item3.txt: 75 commands
  python -m experiments.paper_1k.run item3 --data-root paper1k_data
  ```
  `--data-root` is the root of `setup records`. The 1k windows must be in `paper1k_runs/windows/` (`setup windows`):
  the driver checks their token hashes and stops if they are missing or differ.
- **Step 0, in this order; any failure stops the item:**
  - **(a) the 1k smoke, per model.** On a random-init 3-layer model: `tmopt_ext record --tiny-layers 3` into
    `paper1k_runs/item3/smoke/data_root` (the development records are linked read-only from `--data-root`); the 1k
    16x64 map remapped to it (`models.tiny remap`); `tmopt_ext oneshot --tiny-layers 3 --fit-windows 1024
    --oneshot-steps 2` (Qwen3.8-27B with `--teacher-host`).
    - **Pass** (`item3/step0_smoke_<model>.json`): status complete; 2 steps; 1,024 fit windows, in the report and in the
      trainer's arguments; the fit extension has 896 windows whose token hashes equal the paper's 1k set
      (`windows/<model>_1024.json`); the teacher on the host exactly for Qwen3.8-27B.
  - **(b) one 256-window map rebuilt.** Part G's Nemotron 16x64 one-shot map (`tmopt_ext oneshot --match` the 256
    release map, the default 256 windows).
    - **Pass** (gate `item3-rebuild`, `item3/step0_rebuild.json`): its tiles' digest equals `reference/item3.json`'s
      (120 modules, 80,098 E0M3 tiles), bit for bit. A map digest is an exact gate. If it differs, the one-shot
      gradient sums of your board order differently: ask your user before the panel ("When a gate fails").
    - The summed-gradient file's sha256 is compared too, but only reported: it depends on the GPU's floating-point
      order.
- **Gates** (any failure stops the item):
  - every job rc 0 and a finished record;
  - per (model, unit), before its PPL jobs: every baseline has the 1k map's modules, shapes and per-layer E0M3 counts;
    the act map read the 1,024 windows; the one-shot trainer's fit extension is the paper's 1k set.
- **Reported, not gated:**
  - the re-measured FourOverSix and FlipQuant rows against the paper's per-window NLLs (`reference/item3.json`), with
    their tier (exact / equivalent / differs);
  - each cell against the published 256-window table.
- **Time:** about 4 h. The one-shot maps take about 4× part G's 256-window times (175 s on Nemotron, 470 s on
  Qwen3.8-27B), so 12–35 min each. The 38 PPL jobs take about 1 min (Nemotron) and 2.3 min (Qwen3.8-27B) each.
- **Outputs:** `paper1k_runs/item3/`:
  - `<model>/<unit>/{random0,random1,random2,act,oneshot}.pt` with their `.json` reports, and `oneshot.run/`;
  - `<model>/<unit>/ppl_<map>.json`, `ppl_flipquant.json`, `<model>/ppl_fo6.json`;
  - `step0_*.json`, `smoke/`, `step0/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item3 --report   # results/paper_1k/item3/: ABLATION.md, item3.json, table_ablation_{wiki,c4}.tex
  python -m experiments.paper_1k.publish item3 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0's results, the table's headline (FlipQuant vs each baseline, per unit) and any row that moved
  against the 256 table.

### Item 4: Table F.3 (`tab:ptq`) at the 1k calibration

- **What:** GPTQ, GPTQ candidates and the Hadamard basis on Nemotron-Nano-9B-v2 and Qwen3.8-27B at 16x64. It is
  round 3's protocol (NVFP4-RaZeR paper-eval part O) and part H's Hadamard rows, with the 1k windows. Per model:
  - **GPTQ** with activation quantization on during calibration (`--ptq gptq --gptq-propagate quantized
    --gptq-hessian-input quantized`, the fixed RTN grid, block 128, damp 0.01) on the 1k windows: NVFP4, FourOverSix
    and FlipQuant 16x64 with the 1k map fixed. Then the E0M3 candidate codes (`calibration.gptq_codes` with an
    all-E0M3 map made from the 1k map).
  - **Two maps trained at 1k:** on the GPTQ candidates (`tmopt_ext train --candidates-e2m1 / --candidates-e0m3`, in
    the n16k64 env as round 3) and in the block-16 Hadamard basis (`--rotate hadamard --rotate-block 16`, as part H).
    Qwen3.8-27B uses `--teacher-host`.
  - **Native PPL:** the three GPTQ rows, the GPTQ-candidate map, the re-check (the 1k RTN map on the 1k GPTQ
    candidates, no retraining) and the Hadamard map.
  - **GSM8K** (part N's protocol: lm-eval gsm8k_llama, chat template, greedy, thinking off, batch 64): the three GPTQ
    rows, the GPTQ-candidate map and the Hadamard map.
  - **The reused rows:** BF16, RTN NVFP4 / FourOverSix and Hadamard NVFP4 / FourOverSix do not depend on the
    calibration windows; RTN FlipQuant 1k is part AA's. Their PPL is re-measured here with their records' exact
    commands, and the ΔNLL column uses this session's RTN FourOverSix. Their GSM8K results are the paper's, per problem
    (`reference/item4.json`). So the GSM8K pairs with them compare this board with the paper's: the report says so.
- **Prerequisites:**
  - sections 1.1–1.4, with the records and windows of both models (Nemotron's 256 set is step 0's);
  - the maps: `python -m experiments.paper_1k.artifacts fetch --kind flipquant_1k --models
    nemotron-nano-9b-v2,qwen3.8-27b`.
- **Disk:** about 77 GB of GPTQ code files (Nemotron 4.5 GB, Qwen3.8-27B 14–15 GB each) under
  `paper1k_runs/item4/codes`. `PAPER1K_CODES=<dir>` puts them on another disk. Each GPTQ job first checks for room
  for its file plus 10 GB, and stops the item if there is none.
- **GPU memory:** Qwen3.8-27B's 1k trainings peaked at about 90.5 GiB of the 96 GB in part AA. Nothing else may hold
  GPU memory (the runner waits for an idle GPU).
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item4 --data-root paper1k_data --dry-run   # paper1k_runs/dry_run/item4.txt: 46 commands
  python -m experiments.paper_1k.run item4 --data-root paper1k_data
  ```
- **Step 0:** round 3's Nemotron GPTQ NVFP4 job at 256 windows: its exact command, with the 256 set from `setup
  windows` and a new code file.
  - **Pass** (gate `item4-step0`, `item4/step0/step0.json`): `codes_sha256` equals round 3's (`6a9c9d29…`,
    `reference/item4.json`). A codes digest is an exact gate.
  - Reported: the calibration data hash (equal when the windows are) and the per-window NLLs against round 3's, with
    their tier.
- **Gates** (any failure stops the item):
  - every job rc 0 and a finished record;
  - both trainings' fit extensions are the paper's 1k set (896 token hashes).
- **Reported checks:** per model, the three GPTQ rows share one data hash, with 1,024 windows and the quantized
  Hessian; each GSM8K row ran the same codes as its PPL row; each reused row's re-measured per-window NLLs against the
  paper's, with their tier.
- **The trainings' environments** (in the report): the GPTQ-candidate trainings run in n16k64, round 3's env (the
  hybrid models' recurrent layers on their fallback); the Hadamard trainings in n16k64-fast, part H's env (the fast
  mamba-ssm / causal-conv1d / fla kernels). Both are the published routes (their job logs' first lines).
- **Auxiliary check (not a gate), the item's last three jobs:** part H's exact 256-window Hadamard training command
  once more per model, Nemotron twice (n16k64-fast; about 10 min per Nemotron run, 25 min for Qwen3.8-27B), into
  `item4/aux/`. Per run: the map's E0M3 tiles and tiles digest against part H's map (`reference/item4.json`
  'hadamard_256_part_h'), the first epoch whose train_kl differs from part H's; Nemotron's two repeats against each
  other separate run-to-run nondeterminism from the code history. In `item4/fastenv_repeat.json` with the
  conclusion. Reported either way; nothing stops on it.
- **Time:** about 24–30 h.
  - GPTQ: at 256 windows round 3's jobs took 11–13 min (Nemotron) and 34–40 min (Qwen3.8-27B). The Hessian pass over
    1,024 windows takes longer: assume up to about 4×.
  - The four trainings: about 2.6–2.8 h each (part AA's 1k trainings).
  - GSM8K: about 6 min (Nemotron) and 35 min (Qwen3.8-27B) per row.
  - The 18 other PPL jobs: 1–2.5 min each.
- **Outputs:** `paper1k_runs/item4/`:
  - `step0/`;
  - `<model>/reused/*.json` (the reused rows' PPL);
  - `<model>/gptq_{nvfp4,fo6,fq-16x64,e0m3}.json`;
  - `<model>/fq-16x64_{gptqcand,hadamard}.pt` with `.json` and `.run/`;
  - `<model>/{gptqcand_fq-16x64,recheck_rtnmap_gptqcand,hadamard_fq-16x64}.json`;
  - `gsm8k/<model>/*.json` with the per-sample `.gsm8k_llama.jsonl.gz`;
  - `codes/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item4 --report   # results/paper_1k/item4/: REPORT.md, item4.json, table_ptq.tex
  python -m experiments.paper_1k.publish item4 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  Never commit the code files or the per-sample GSM8K files: they stay in the run directory. `SUMMARY.md`: step 0, the
  table's headline (GPTQ and the GPTQ-candidate map against RTN, per model) and the checks.

### Item 5: Table D.5 (`tab:app-gemm`) at the 1k calibration

- **What:** the per-forward GEMM overhead of the deployed kernels over the stock kernel, with each projection's tags
  taken from the 1k maps.
  - One `evaluation.gemm_time` process per model (the paper's six) with its three 1k maps, at the tool's defaults,
    which are the paper's method: CUPTI GEMM time on the installed NativeLinears; cold weights (a 512 MiB L2 flush,
    rotated weight copies); 3 rotated rounds of 30 calls; T = 1 … 8192.
  - Per projection: the tags of its median-share module (typical) and of its max-share module (densest). The deployed
    tile table is used with its fallbacks.
  - `evaluation.gemm_time_report` gives the six-model medians in Table D.5's layout.
- **This is a timing item.** Nothing else may run on the GPU (gemm_time checks NVML and stops otherwise) or load the
  host. Do not lock clocks, change the power limit or run a profiler (see the general rules).
- **Prerequisites:** sections 1.1–1.3 (maps: `--kind flipquant_1k,flipquant_256`). No data root.
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item5 --dry-run   # paper1k_runs/dry_run/item5.txt: 7 commands
  python -m experiments.paper_1k.run item5
  ```
- **Step 0:** Phi-4 with the 256-window release maps, the paper's validation model.
  - **Pass** (gate `item5-step0`, `item5/step0/step0.json`): per tile, the median over its T ≥ 128 cells (typical tags) is within 2 points
    of the published run's 6.72 / 0.75 / 0.91 % (8x64 / 16x64 / 256x64). The criterion is the user's relative one for
    this harness (2026-10-13).
  - The published run was on another RTX PRO 6000 board. The step-0 record gives the difference per tile; a
    difference within the criterion is the board offset, not a failure.
- **Gates:** every job rc 0. gemm_time stops on its own checks: each isolated launch equals the module's fused forward
  bit for bit; each install resolves to the expected kernel set; every profiled block holds exactly 30 GEMM launches;
  no other compute process.
- **Time:** about 1–3 h (seven processes).
- **Outputs:** `paper1k_runs/item5/<model>.json`, `step0/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item5 --report   # results/paper_1k/item5/: GEMM.md, item5.json
  python -m experiments.paper_1k.publish item5 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0's three differences, the T ≥ 128 row and the largest cell below T = 128.

### Item 6: Table A.2 (`tab:app-native`) at the 1k calibration

- **What:** native minus simulated perplexity of the 1k maps (part C2 with amendment 9, at 1k).
  - Models: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2 and Phi-4,
    at 8x64 / 16x64 / 256x64. Qwen3.8-27B is left out: its simulation does not fit one 96 GB GPU.
  - Each cell runs both ways on this GPU, in one env: n16k64, and n16k64-fast for Nemotron-Nano-9B-v2 on both sides.
    - native: part AA's command, `evaluation.ppl --mode native --kernel-set auto --weight mixfp4 --act fourover6
      --map <1k map> --paper-convention`;
    - simulated: part C2's command, the same with `--mode fake`.
  - Per corpus: the paired ΔNLL native − simulated ± 2 SE. A cell agrees when |ΔNLL| ≤ 2 SE. The report gives the
    count over the 30 cells (the paper at 256: 28 of 30).
- **Prerequisites:** sections 1.1–1.3 (maps: `--kind flipquant_1k,flipquant_256`). No data root.
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item6 --dry-run   # paper1k_runs/dry_run/item6.txt: 32 commands
  python -m experiments.paper_1k.run item6
  ```
- **Step 0:** Mistral-7B 16x64 with the 256-window release map, both ways.
  - **Two-tier** (gate `item6-step0`, `item6/step0/step0.json`): the simulated run against part C2's record, the
    native run against the release evaluation's (`reference/item6.json`).
    - **PASS-exact:** both equal per window bit for bit, so the cell equals the published −0.16 ± 0.94 / +0.66 ± 1.24.
    - **PASS-equivalent:** not equal, but each side's paired ΔNLL against its record is within 2 SE on both corpora.
    - **FAIL** otherwise: the item stops; the tiers and the largest per-window differences are in `step0.json`.
  - The report states the tier.
- **Gates:** every job rc 0 and a finished record.
- **Reported:** each native run against part AA's per-window NLLs (`reference/item6.json`), with its tier.
- **Time:** about 1 h (32 PPL jobs of 0.5–4 min).
- **Outputs:** `paper1k_runs/item6/<model>/<tile>_{native,fake}.json`, `step0/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item6 --report   # results/paper_1k/item6/: NATIVE.md, item6.json, table_app_native.tex
  python -m experiments.paper_1k.publish item6 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0, the agreement count and the largest |ΔNLL|.

### Item 7: Appendix D.6 (reading the weights back) at the 1k calibration

- **What:** part C1's readback of the 18 1k maps (six models × three tiles), and a negative control.
  - The readback: `evaluation.ppl --mode native --weight mixfp4 --act fourover6 --map <1k map> --paper-convention
    --ownership-check --limit 1`. With the map installed, every quantized linear runs through the native kernel on an
    identity activation. Every decoded weight element must equal its stored value under the map's format; the
    install raises on any mismatch.
  - The negative control (`experiments/paper_1k/ownership_control.py`), per model on its 16x64 1k map. It takes the
    first quantized linear with an E0M3 tile and packs it twice, with the map and with an all-E2M1 map. The same check
    must pass on the map's packing and must detect both wrong pairs: the all-E2M1 packing against the map, and the
    map's packing against an all-E2M1 map. Its logic is also tested on the CPU (`tests/test_paper_1k.py`).
- **Prerequisites:** sections 1.1–1.3 (maps: `--kind flipquant_1k,flipquant_256`). No data root.
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item7 --dry-run   # paper1k_runs/dry_run/item7.txt: 25 commands
  python -m experiments.paper_1k.run item7
  ```
- **Step 0:** part C1's readback of Mistral-7B's 16x64 release map, its command unchanged.
  - **Pass** (gate `item7-step0`, `item7/step0/step0.json`): the per-module rows (exact, format mismatches, informative elements, E0M3
    elements observed, E0M3 tiles) equal C1's (`reference/item7.json`, a digest).
- **Gates:** every readback job rc 0 (a mismatch makes the install raise); every control passes.
- **Time:** about 30 min (25 processes; each loads a model).
- **Outputs:** `paper1k_runs/item7/readback/<model>_<tile>.json`, `control/<model>_16x64.json`, `step0/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item7 --report   # results/paper_1k/item7/: READBACK.md, item7.json
  python -m experiments.paper_1k.publish item7 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0, the totals (the paper at 256: 181.9 billion elements, exact, 0 mismatches) and the six controls.

### Item 8: Appendix D.4 (the real tags against random placement) at the 1k calibration

- **What:** the D.4 sentence ("the release maps of Mistral-7B-Instruct-v0.3 lie within 1 point of the random placement
  at their share"), at 1k.
  - `evaluation.gemm_time` runs on Mistral-7B with its three maps and, in the same process, one map per unit with the
    same E0M3 count per module at random tiles (`experiments/paper_1k/random_maps.py`, seeded).
  - The random maps use gemm_time's opt-in `<unit>_random` policies; its defaults (item 5) are unchanged.
  - Projections o_proj, gate_proj and down_proj; T = the D.5 set (gemm_time's default, 1 … 8192, twelve values). The
    paper's source ran six T values that are not recorded; the D.5 set contains the usual six (1, 16, 128, 512,
    2048, 8192).
  - Per cell: the real tags' overhead over stock_ko minus the random tags', in points. The typical module is the
    result; the densest module is reported too.
  - This is the method behind the paper's current sentence: its source note of 2026-10-13, "real tags vs random at the
    same share, 3 units x o/gate/down x 6 T; bitwise 216/216". It replaced the earlier C3v comparison on
    Llama-3.1-8B's pre-release maps, which needed kernel-opt's research builds. This is a registered deviation from
    the plan's "C3v / amendment 19 method" (NVFP4-RaZeR paper-eval PROTOCOL.md, part AC item 8, 442866f).
  - The report counts gemm_time's per-cell bitwise checks (each isolated launch equals the module's fused forward).
- **This is a timing item:** the rules of item 5 apply.
- **Prerequisites:** sections 1.1–1.3 (maps: `--kind flipquant_1k,flipquant_256`).
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item8 --dry-run   # paper1k_runs/dry_run/item8.txt: 8 commands
  python -m experiments.paper_1k.run item8
  ```
- **Step 0** (gate `item8-step0`): the release maps. Pass: every typical cell within 1 point of random placement (the
  paper's statement).
  - The published ranges (another RTX PRO 6000 board, other random draws): 8x64 −0.53..+0.43, 16x64 −0.37..+0.94,
    256x64 −0.28..+0.43 points.
- **Gates:** every job rc 0; gemm_time's own checks (each launch bitwise equal to the module's fused forward, the
  expected kernel sets, the launch counts, no other compute process).
- **Time:** about 1 h (two gemm_time processes, seven policies each, twelve T values).
- **Outputs:** `paper1k_runs/item8/{256,1k}/` (`random_<unit>.pt` with `.json`, `gemm_time.json`), `step0.json`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item8 --report   # results/paper_1k/item8/: REAL_VS_RANDOM.md, item8.json
  python -m experiments.paper_1k.publish item8 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0's worst cell, and the 1k ranges per unit with the 1k shares.

### Item 9: Tables D.2 (`tab:app-ktile`) and D.3 (`tab:app-c2`), RTX PRO 6000 rows, at the 1k calibration

- **What** (at 4096³, `experiments/paper_1k/ktile.py`):
  - **Table D.2:** per-MMA branch against per-K-tile dispatch, kernel-opt amendment 20's ablation. Kernels: the
    deployed 16x64 and 8x64 kernels and their per-MMA-branch variants, with the stock kernel and cuBLAS FP8 as
    references.
  - **Table D.3:** the overhead over the deployed stock kernel, split into parts, before and after the optimizations.
    Each part is the ratio of two configurations that differ in one respect:
    - tiles: a no-dispatch build without tags against stock;
    - tag copies (first kernels only): the no-dispatch build with the site-0 tags against without;
    - dispatch: all tiles E2M1 against the no-dispatch build;
    - E0M3 tiles: the real tags against all E2M1;
    - total: the real tags against stock.
    "Before" uses the paper's first kernels (n16k64_wA, n8k64_wB). The first 256x64 kernel is the 16-arm 16x64 kernel
    with 256x64 tags. These definitions are re-derived from the paper's text and kernel-opt's C2 measurements, because
    the paper's unified run's code is not available.
  - The session: one process, 10 repeats, each the median of 3 rotated rounds, all configurations interleaved.
    - iso_cold: a 512 MiB flush and rotated cold weights per call, 3 + 30 calls.
    - b2b_cold: 20 back-to-back calls over the rotating copies.
    - CUPTI kernel times; mean ± SD over the repeats.
    - Tags: layer 0 o_proj of Mistral-7B's release maps (step 0) and 1k maps, in the same session.
  - The RTX 5090 rows of Table D.2 are unchanged and not run.
- **The kernels.** `kernels/sm120` is re-vendored at kernel-opt 9580a2a, which adds amendment 20's per-MMA-branch
  option; every other configuration's generated blob is byte-identical (`kernels/sm120/PORT_CHANGES.md`).
  - Item 9 needs seven builds beyond `build_deploy.sh`: the two per-MMA builds, the no-dispatch ceilings, and the
    first kernels' no-dispatch builds with and without tags. Its first job builds them (CPU) into
    `kernels/sm120/build_item9` (`PAPER1K_BUILD_ITEM9` overrides).
  - The build needs `CUDA_HOME` (the CUDA 13.1.115 toolkit of section 1.2) and, if `kernels/3rdparty/cutlass` is not
    checked out, `CUTLASS_DIR` (a CUTLASS checkout at e64a9136).
- **This is a timing item:** the rules of item 5 apply. Change nothing on the board (500 W limit).
- **Prerequisites:** sections 1.1–1.3 (the deployment build; maps `--kind flipquant_1k,flipquant_256`).
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item9 --dry-run   # paper1k_runs/dry_run/item9.txt: 4 commands
  python -m experiments.paper_1k.run item9
  ```
- **Gates, in order** (exact; each a named gate, "When a gate fails"):
  - `item9-sass`: each of the seven builds has its reference SASS, patched and unpatched (`reference/item9.json`,
    from kernel-opt's manifests). A different toolkit or different sources fail here. **This gate is the first check
    of these builds from this repository's sources:** the porting machine's GPU driver was hung, so they were not
    built there. What was checked there, on the CPU: the re-vendored blob generator emits byte-identical headers for
    all 47 existing configurations; the two per-MMA blobs equal kernel-opt's build_KT headers; and the ported SASS
    census reproduces amendment 20's census on kernel-opt's own libraries.
  - `item9-census`: the static SASS census of the five D.2 kernels equals amendment 20's: OMMAs, predicated OMMAs,
    BRX, OMMAs per steady k-iteration (32 per-K-tile, 64 per-MMA), and the tensor-pipe estimate.
  - `item9-check`: each per-MMA build's output equals its per-K-tile build's bit for bit, for the release and 1k tags,
    all E2M1, all E0M3 and 30 % random tags (and again on the timed operands).
- **Step 0:** the session's release-tag half against the paper's values, which were measured on another RTX PRO 6000
  board. The report gives every offset: D.2's µs, overheads and speedups, and D.3's parts. Offsets are reported, not
  gated.
- **Time:** about 30–60 min. Seven CPU builds of about 5–10 min (in parallel); the census (seconds); the check
  (minutes); the session (about 1–2 min per repeat).
- **Outputs:** `kernels/sm120/build_item9/` (with `item9_sass_gate.json`); `paper1k_runs/item9/{census,check,time}.json`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item9 --report   # results/paper_1k/item9/: KTILE.md, item9.json
  python -m experiments.paper_1k.publish item9 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: the gates, step 0's largest offsets, and both tables with the 1k tags. Their sections appear here, and the index marks them ready, as each one lands. `git pull` to get them.

### Item 11: Tables C.2 / C.3 (the calibration cost) at the 1k calibration — cancelled by the user (2026-10-26); filled from part AA records

- **Not run.** The user cancelled the GPU re-measurement (about 19.5 h) on 2026-10-26. `run item11` and its dry run
  refuse; Tables C.2 / C.3 are filled from part AA's records of the 18 1k calibrations (six models × three tiles,
  `calibration.train_map --epochs 5 --teacher-topk 1000 --fit-windows 1024`), committed as
  `reference/item11.json` (the job records) and `reference/item11_cost.json` (the trainers' phases and epochs, and
  the release calibrations' at 256 windows for the comparison).
- **Report and publish** (CPU only, no data root):
  ```bash
  python -m experiments.paper_1k.run item11 --report   # results/paper_1k/item11/: COST.md, item11.json, table_calib_cost.tex
  python -m experiments.paper_1k.publish item11 --status "cancelled by the user (2026-10-26); filled from part AA records" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
- The report gives Table C.2 at 16x64 (setup, training, trainer, Total, the training share, GPU peaks, the teacher),
  the setup phases and the step time, Table C.3 (the three tiles, the max. differences, the GPU peak 8x64 − 256x64),
  the 256 → 1k ratios, the caveats (Qwen3.8-27B's teacher on the host; the 16x64 runs from part AA's panel, the 8x64 /
  256x64 runs from its amendment 6) and phase A's reproduction of step 0 (Llama-3.2-1B-Instruct 16x64: 223 s against
  part AA's 225 s, the map bit for bit equal).
- The re-measurement's driver (`item11.run`, gate `item11-maps`) is kept unchanged in the code, unused.

### Item 12: Table C.5 (`tab:app-robust`) at the 1k calibration

- **What:** calibration choices on Mistral-7B at 16x64, at 1,024 sequences (the paper's Appendix C.4; flipquant's
  calib-robustness options). Each run is part AA's 1k calibration command with one change:
  - seed 1 and seed 2 (`--seed`: the per-epoch order of the sequences);
  - non-deterministic kernels (`--no-deterministic`);
  - other calibration data, 1,024 sequences of 512 tokens (`--fit-source`): generic web text (FineWeb-Edu),
    instruction data (UltraChat), the WikiText-2 training split.
  - Each map, the 1k paper map and FourOverSix are evaluated natively. Reported per run: the paired ΔNLL against the
    1k paper map ± 2 SE, the E0M3 share, and the Jaccard index of the E0M3 tiles with the paper map's.
- **The 13-gram audit** (`experiments/paper_1k/audit.py`, CPU), before any training on a source.
  - It draws that source's 1,024 windows exactly as the trainer will, and checks them against every evaluation text
    for shared word 13-grams: WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, GSM8K test and
    its few-shot exemplars.
  - Its positive control splices five evaluation items into thirty windows; each must be found.
  - Gate `item12-audit`: no hit, and the control finds every splice. After training, the trained windows must be the
    audited ones.
- **Prerequisites:** sections 1.1–1.4 (Mistral's records); maps `--kind flipquant_1k`. The sources download from the
  Hub at pinned revisions (FineWeb-Edu one parquet shard, UltraChat one shard, WikiText-2).
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item12 --data-root paper1k_data --dry-run   # paper1k_runs/dry_run/item12.txt: 18 commands
  python -m experiments.paper_1k.run item12 --data-root paper1k_data
  ```
- **Step 0** (gate `item12-step0`): a seed-0 recalibration at 256 windows (part AA's step-0 command) must reproduce
  the release map bit for bit, as the paper states for the environment of these runs. It takes about 5 min.
  - A map digest is an exact gate. If it differs, ask your user before the runs ("When a gate fails").
- **Stop rules:** as item 11. The driver never deletes a run directory: an interrupted calibration stops the item; move
  its directory aside, then run again.
- **Time:** about 2.5 h: six 1k calibrations of about 15 min, three audits of 5–20 min (CPU; FineWeb-Edu's shard is
  the largest download), nine PPL jobs of about 1.5 min.
- **Outputs:** `paper1k_runs/item12/<row>/` (map, `reproduction.json`, `run/`), `audit/<source>.json`, `ppl/`,
  `overlap.json`, `step0/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item12 --report   # results/paper_1k/item12/: ROBUST.md, item12.json
  python -m experiments.paper_1k.publish item12 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0, the six rows against their 256 values, FourOverSix's distance and the audits.

### Item 13: Appendix F.1 (the tile-gradient check) at the 1k calibration

- **Status of the code — read this first.** The paper's F.1 scripts were never published.
  - `calibration/gradcheck.py` re-implements the check from the paper's text and its source notes. It is
    CPU-tested (its statistics, the driver's commands and its criteria) but **has not run on a GPU before**.
  - Its step 0 is its validation against the published 256-sequence results. If step 0 fails, do not tune the code
    to pass: report the gap ("When a gate fails"); a failed step 0 means the re-implementation is not the paper's
    check yet.
- **What it measures** (Mistral-7B, 16x64).
  - g_u: the trainer's tile gradient at the all-E2M1 map, from `calibration.tmopt_ext oneshot`, one pass over the fit
    windows.
  - The tile sets: thirty groups of 1,000 tiles from ten bins of g_u's rank (0–0.1, 0.1–0.3, 0.3–1, 1–3, 3–10,
    10–50, 50–90, 90–97, 97–99.7 and 99.7–100 %); the k most negative tiles, k = 10 … 3 × 10^5; and the paper's map.
  - Each set is flipped from the all-E2M1 map. The measurement is the paired change of the trainer's bucketed KL
    (top-1,000 teacher plus a tail bucket) on in-sample (the fit windows) and held-out sequences.
  - Spearman correlations of g_u, the activation-weighted score and the weight error with the measured decrease,
    with 95 % bootstrap intervals over the groups.
- **Prerequisites:** sections 1.1–1.4 (Mistral's records); maps `--kind flipquant_1k,flipquant_256`.
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item13 --data-root paper1k_data --dry-run   # paper1k_runs/dry_run/item13.txt: 5 commands
  python -m experiments.paper_1k.run item13 --data-root paper1k_data
  ```
- **Step 0 (a), smoke:** `--smoke 16` at 256 windows (16 sequences per set, two groups, top-10 and the map). It must
  complete with its checks passing (the fit windows are the one-shot run's, the held-out windows disjoint, the steps as
  expected). It takes a few minutes.
- **Step 0 (b), the 256 run** (gate `item13-step0`; the published values are in `reference/item13.json`).
  - Inputs: the release map, the release one-shot gradient, and held-out `512:64:128` (the extension rule's math and
    code windows 65–192, as published).
  - **Pass** requires all of:
    1. the run's own checks;
    2. the in-sample all-E2M1 KL within 10^-6 (relative) of the trainer's epoch-0 KL (published 2.7 × 10^-9);
    3. the release map's measured change within the published 2 SE of the published values: held-out −17.8 ± 1.7,
       in-sample −26.5 ± 2.0 (× 10^-3);
    4. g_u's Spearman correlation inside the published 95 % intervals: held-out [+0.37, +0.92], in-sample
       [+0.73, +0.98].
  - Reported against the published values: the one-layer scores' correlations; the groups beyond 2 SE and their sign
    agreement (published: 6 and 12, all with g_u's sign); the pooled middle bins; the top-k ratios (in-sample
    0.88–0.97 for k = 10–100, 0.57 at 1,000); the map's tile ranks (21 % in the top 75,906, median rank 5.1 %, 942
    with g_u > 0).
  - The published run was on an RTX 6000 Ada with the reference activation quantizer. On SM120 the fused quantizer
    is bitwise equal to it (the run checks its first 64 calls). g_u, and with it the groups drawn from its ranks, can
    differ slightly from the published run's; the criteria above allow for that.
- **The 1k run:** the 1k map, and the one-shot gradient over the 1,024 windows (in-sample: all of them).
  - The held-out set is **new**: the paper's lies inside the 1k calibration set. It is `2048:448:128`, the
    extension rule's math and code windows 449–576 of the 2,048-window draw, 128 of each. It is disjoint from the
    1,024 by the nesting, and the run checks this by token hash.
- **Time:** about 4–5 h. Two one-shot runs (5 and 15 min); the smoke; the 256 run (about 1 h); the 1k run (about
  2.5 h, since in-sample is 1,024 sequences).
- **Outputs:** `paper1k_runs/item13/{256,1k}/` (`oneshot.pt` with its `.run/`, `gradcheck.json`, `gradcheck.sets.pt`,
  `smoke.json`), `step0.json`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item13 --report   # results/paper_1k/item13/: GRADCHECK.md, item13.json
  python -m experiments.paper_1k.publish item13 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0 criterion by criterion, then the 1k run's Table F.1 and the map's measured changes.

### Item 1 (last): the MMLU column of Table 3 and Tables E.1–E.3 at the 1k calibration

- **What:** MMLU 5-shot of the 1k rows on the six models: lm-eval 0.4.11, no chat template, acc × 100, 14,042
  questions (the colleague's spec).
  - Rows: FlipQuant 8x64 / 16x64 / 256x64 (the 1k maps) and GPTQ‡-1k (its public Calib1k artifact).
  - Each row runs with part AA's evaluation flags and `evaluation.downstream --tasks mmlu`, at the tool's defaults:
    5-shot, batch 16 (Qwen3.8-27B 8).
  - GPTQ‡ reads its codes file (`--gptq-cache`). It must match this run's calibration windows, or flipquant refuses
    it; it never overwrites the file.
  - The other rows (BF16, NVFP4, FourOverSix, IF4, MixFP4) do not depend on the calibration windows and keep the
    paper's values.
  - Table 3's Avg. of a 1k row is the mean of this MMLU and part AA's six zero-shot tasks of the same artifact
    (`reference/item1.json`).
- **Prerequisites:**
  - sections 1.1–1.4 (records and the 1k windows of all six models);
  - the artifacts: `python -m experiments.paper_1k.artifacts fetch --kind flipquant_1k,flipquant_256,gptq_1k`
    (the Calib1k GPTQ‡ codes are most of it).
- **Command:**
  ```bash
  python -m experiments.paper_1k.run item1 --data-root paper1k_data --dry-run   # paper1k_runs/dry_run/item1.txt: 25 commands
  python -m experiments.paper_1k.run item1 --data-root paper1k_data
  ```
- **Step 0:** Llama-3.2-1B-Instruct FlipQuant 16x64 with the 256-window release map, against the published 41.40.
  - Gate `item1-step0`, **two-tier**. Exact when the accuracy rounds to 41.40. Equivalent when it is within 2 SE of
    41.40 (0.83 pp: the binomial SE over 14,042 questions). The published run's per-question records are not
    available, so this test is unpaired and looser than a paired one. The record gives the difference in questions.
- **Gates:** every job rc 0 and a complete record.
- **Time:** about 7–9 h (25 processes).
- **Registered deviations (2026-10-26)**, after Nemotron-Nano-9B-v2's first row ran out of GPU memory at batch 16 on a
  96 GB RTX PRO 6000 (lm-eval's float32 log-softmax over a padded batch; the published 256 column ran on another
  96 GB RTX PRO 6000 pod, and how it fitted there is not recorded):
  - **(A) environment, ¤:** `PAPER1K_ALLOC_CONF=expandable_segments:True python -m experiments.paper_1k.run item1 …`
    gives the remaining jobs that `PYTORCH_CUDA_ALLOC_CONF` (recorded per job in `logs/<job>.env.json`). The command
    and batch are unchanged.
  - **Consistency rows:** under (A) Nemotron, and every model with a reduced batch, first runs FlipQuant 16x64 with the
    256-window release map under the same deviation, against its published 256 cell (gate `item1-consistency`,
    step 0's two-tier rule). A failure stops the item.
  - **(B) the fallback, command, ‡b:** only if a model still runs out of memory under (A):
    `PAPER1K_ITEM1_BATCH="<model>=8"` (Qwen3.8-27B: `=4`) passes `--batch-size` for that model's jobs, with its own
    consistency row at that batch. If the reduced batch also fails, stop.
  - **Model order (2026-10-27, the user's):** `PAPER1K_ITEM1_MODELS=<model>` runs one model's rows (step 0 is skipped
    once finished; under (A) Nemotron's consistency row comes right before its rows). The remaining models were run
    one invocation each, Phi-4, then Qwen3.8-27B, then Nemotron-Nano-9B-v2, so that an out-of-memory failure stops only
    its own model. The commands are unchanged.
- **Outputs:** `paper1k_runs/item1/<model>/<row>.json`, `step0/`.
- **Report and publish:**
  ```bash
  python -m experiments.paper_1k.run item1 --report   # results/paper_1k/item1/: MMLU.md, item1.json, table_mmlu.tex
  python -m experiments.paper_1k.publish item1 --status "done" --summary SUMMARY.md
  git push origin paper-1k-reruns
  ```
  `SUMMARY.md`: step 0's tier, the four 1k rows per model against their 256 values, and the paired FlipQuant 16x64
  vs GPTQ‡ comparisons.

## 3. Publishing an item

- `python -m experiments.paper_1k.run <item> --report` writes the item's report into `results/paper_1k/<item>/`.
- `python -m experiments.paper_1k.publish <item> --status "…" [--summary FILE]` does the rest:
  - scans every file: no tokens, no local or home paths, no host names, IP addresses or e-mail addresses;
  - writes the item's README;
  - updates `results/paper_1k/index.json` and the README index;
  - commits with `git add -f` (`.gitignore` holds `results/*`), then records the report commit in the index.
- Then `git push origin paper-1k-reruns`.
- A scan hit stops before any commit. Remove the offending text from the report; never commit a token or a path.
