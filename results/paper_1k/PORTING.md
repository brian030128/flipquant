# Running the remaining items from flipquant alone: the audit and the porting plan

Goal (the user, 2026-10-25): another author finishes every not-yet-run item of [PLAN.md](PLAN.md) using only a fresh
clone of flipquant (branch `paper-1k-reruns`), the public Hugging Face artifacts and public datasets. That means
nothing from NVFP4-RaZeR, no local paths and no private storage. The items are item 2's memory remainder, 3–9, 11–13
and 1.

This file lists, per item, every dependency that lives outside flipquant today, and how it is brought in. The status
column is updated as the port proceeds.

## Cross-cutting dependencies (every item)

| dependency | where it is today | port into flipquant | status |
|---|---|---|---|
| The per-item driver: registered settings, step 0, gates, the job queue (one job at a time, idle-GPU check, skip finished records, START/END log, GPU-state rows) | NVFP4-RaZeR `experiments/paper_eval/run_paper1k.py` + `run_gpu.py` (+ `run_calib_size.py`, `run_sm120.py`, `run_ptq_round3.py`, `run_gsm8k*.py`, `run_zeroshot6.py`) | `experiments/paper_1k/` with `python -m experiments.paper_1k.run <item> [--out DIR]`. One module per item, plus a small job runner; every command built from flipquant's own CLIs | runner done (`experiments/paper_1k/jobs.py`, `run.py`); items: the README's 'ready' column |
| Report and publish generators | NVFP4-RaZeR `report_paper1k.py`, `publish_paper1k.py` (+ `report_ablation.py`, `report_ptq_round3.py`, `report_gsm8k*.py`, `tables_cpu.py`) | `experiments/paper_1k/report.py`, `publish.py`, writing `results/paper_1k/<item>/` directly | publish done (`experiments/paper_1k/publish.py`); each item module has its `report()` |
| Kernels: the runs used `build_V`, a local build directory, and checked every loaded library's sha256 against build_V's manifests | a local build directory | flipquant's `kernels/sm120/build_deploy.sh` (CPU only). Its deployment builds must reproduce build_V's SASS: `SASS_REFERENCE.json` 'builds_defined', `tests/test_sm120_native.py`. Library sha256 depends on the compiler, so the gate becomes the SASS check plus "every loaded library is one of the runner's own build manifests" | done: `python -m experiments.paper_1k.setup kernels` / `setup sass` |
| Environments: n16k64 (eval / calibration) and n16k64-fast (the hybrid models' mamba-ssm / causal-conv1d / fla kernels); CUDA 13.1 for the kernel build | local conda envs | `experiments/paper_1k/env/`: pinned package lists exported from the two envs (torch 2.9.0+cu128, transformers 5.16.1, lm-eval 0.4.11, fla 0.5.2, mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0, …) and the build steps, in `docs/BUILD_AND_USE.md` | done: `experiments/paper_1k/env/*.yml`, `setup env` |
| Artifacts: the 1k maps, FOCUS-1k states and GPTQ‡-1k codes; the 256 release maps, FOCUS / GPTQ‡ for step 0 | local run directories and private storage | the public HF repos `edgeai-lab/<Base>-{FlipQuant,FOCUS-NVFP4,GPTQ-BF16Calib-NVFP4}-Calib1k` and the 256 release repos, at pinned revisions with every file's sha256 (`experiments/paper_1k/artifacts.json`, from the upload records). The tiles of each HF map equal the local map's bit for bit (checked at upload) | done: `experiments/paper_1k/artifacts.json` (36 repositories, read anonymously 2026-10-25), `setup fetch` |
| Calibration windows: 1k / 2k sets and their extension records (`--calib-extension`, `--gptq-calib-extension`) | local trainer reports | rebuilt on the CPU by `calibration.fit_windows`; checked against committed per-window token sha256 lists (`experiments/paper_1k/windows/<model>_<N>.json`: hashes only, no data) | done: `experiments/paper_1k/windows/`, `setup windows` |
| Calibration records (the release fit set) and development records | a local data root prepared by `tmopt_ext record` | `python -m calibration.tmopt_ext record --model M` (flipquant; public datasets at pinned revisions); its record sha256 checked against a committed value | done: `reference/records.json` (revision, fit-set token hashes, matrix digest; the record file holds a local path, so its own sha256 is not portable), `setup records` |
| Step-0 expected values and the reused rows (per-window NLLs, GSM8K per-problem results, published cells) | paper-eval result records | small committed JSON under `experiments/paper_1k/reference/`, copied from the paper-eval records with their source commit | item 2 and the records done; the rest with each item |
| Model snapshots | the HF cache | flipquant's registry pins every revision; downloaded on first use | in place |

## Per item

| item | what it needs beyond the cross-cutting parts | where it is today | port | flipquant-only? |
|---|---|---|---|---|
| 2, memory remainder | `mem_probe.py` (flipquant's loader and install, then the byte / allocation / 1x2048-peak measurements); the part B anchor record | NVFP4-RaZeR `experiments/paper_eval/mem_probe.py` | copy to `experiments/paper_1k/mem_probe.py`; anchor expected values to `reference/` | yes — **ready** (`experiments/paper_1k/item2.py`) |
| 3, `tab:ablation` | `baseline_maps`, `tmopt_ext oneshot --fit-windows 1024` (already in flipquant); G's 256 one-shot map for step 0; the FO6 / 1k-map per-window NLLs; G's published cells | paper-eval G (7c06fb3) records | step 0 compares the rebuilt map's tile digest with a committed digest; references to JSON; FourOverSix and the 1k maps re-measured in the item (part AA's commands) | yes — **ready** (`experiments/paper_1k/item3.py`) |
| 4, `tab:ptq` | round 3's GPTQ (activation-quantized), the E0M3 candidate, the GPTQ-candidate and Hadamard trainings, GSM8K (`evaluation.downstream`); the round-3 codes digest for step 0; the reused rows | NVFP4-RaZeR `run_ptq_round3.py`, `run_sm120.py` (H), `run_gsm8k_lmeval.py` (N) and their records | the commands into the driver; references to JSON (per-window NLLs, per-problem GSM8K); the 256 windows for step 0 (`windows/<model>_256.json`); the reused rows' PPL re-measured | yes — **ready** (`experiments/paper_1k/item4.py`) |
| 5, `tab:app-gemm` | `evaluation.gemm_time` / `gemm_time_report` | flipquant branch `gemm-bench` (505bcf8; two new files on top of main 3db0072) | cherry-pick e4a131f, 1744c90 and 505bcf8 (they only add the two files). This branch's registry already has `llama3.1-8b-instruct`, which gemm-bench's lacks. Step 0: the published six-model medians (the per-model records are not available) | yes — **ready** (`experiments/paper_1k/item5.py`; cherry-picked e0b8d51, 83cfd4e, f927391; step 0: Phi-4's per-model row from the paper's source note) |
| 6, `tab:app-native` | `evaluation.ppl --mode fake` (in flipquant), the native 1k per-window NLLs | the fakeppl route (NVFP4-RaZeR `run_gpu.fakeppl`) and the Llamas' D7-L route (run on another machine; the same command) | both routes are one flipquant command. Run native and simulated on the runner's GPU, so the pairs come from one machine | yes — **ready** (`experiments/paper_1k/item6.py`) |
| 7, D.6 read-back | `evaluation.ppl --ownership-check` (in flipquant); a negative control | — | a small negative-control script on `kernels/sm120/eval/ownership_check.check_module` with an all-E2M1 mask on a module that has E0M3 tiles (as the SM100 check did) | yes — **ready** (`experiments/paper_1k/item7.py`, `ownership_control.py`) |
| 8, D.4 (one sentence) | the C3v method: the isolated cold GEMM timing (M1) of real tags against random placement at the same share | NVFP4-RaZeR `experiments/kernel_opt/c3v_fraction.py` + `sm120/bench/bench_gemm_isolated.py` + `experiments/paper/common.py` | the paper's current D.4 source is `evaluation.gemm_time` with random-placement maps (another board; it replaced C3v's pre-release Llama-3.1-8B comparison): gemm_time's opt-in `<unit>_random` policies and `random_maps.py` | yes — **ready** (`experiments/paper_1k/item8.py`) |
| 9, D.2 / D.3 | amendment 20's cold harness (`ktile_ablation.py`); the per-MMA builds; D.3's builds and parts | NVFP4-RaZeR `experiments/kernel_opt/ktile_ablation.py`, the C2 scripts (`c2_t0.py`, `c2_w8.py`, `c2_U.py`, `c2_V.py`), `experiments/paper_extra/C2_sass.py`; kernel-opt 9580a2a's `sm120/` | `kernels/sm120` re-vendored at 9580a2a (amendment 20's per-MMA option; every other blob byte-identical); `experiments/paper_1k/ktile.py` (build + SASS gate, census, check, the one-session timing with D.2 and D.3), `sass_census.py` (C2_sass, ported); D.3's part definitions re-derived from the paper's text and the C2 scripts. **Unverified on the porting machine:** the seven extra builds' SASS equality with their references (its GPU driver was hung, and a CPU build blocks at nvlink there); the runner's `item9-sass` gate is the first check. Verified on the CPU there: 47 of 47 existing blobs byte-identical under the re-vendored generator, the per-MMA blobs equal kernel-opt's build_KT, the ported census equal to amendment 20's | yes — **ready** (`experiments/paper_1k/item9.py`) |
| 11, C.2 / C.3 | `train_map --fit-windows 1024` (in flipquant); the device-peak sampler; the 1k map hashes for the bitwise gate | NVFP4-RaZeR `run_llama_instruct.measured` / `run_calib_size` | the sampler into the job runner (`jobs.DevicePeak`); the 1k maps' tiles digests (`reference/item11.json`) | yes — **ready** (`experiments/paper_1k/item11.py`) |
| 12, C.5 | `train_map --fit-source …`, `--no-deterministic`, `--fit-windows` below 128 (`calibration/tmopt/fit_sources.py`); the 13-gram audit | flipquant branch `calib-robustness` (932e01b, based on main 120173a); NVFP4-RaZeR `calib_size_audit.py` | merge calib-robustness. Conflicts are expected in `calibration/train_map.py` and `tests/test_stage3.py`; it also edits the vendored `run_train_map.py`, so `VENDORED.json` / `PORT_CHANGES.md` must follow. Port the audit | yes — **ready** (`experiments/paper_1k/item12.py`, `audit.py`; merged a0d004f, 304a43e, 61e4b51, `train_map.py` conflicts resolved with both option sets) |
| 13, F.1 | `calibration/gradcheck.py` (the re-implementation, untested); `tmopt_ext oneshot` for g_u | flipquant local branch `paper-1k-gradcheck` (11a09a8) | merge into this branch once it passes its 256 step 0 | yes — **ready** (`experiments/paper_1k/item13.py`; the re-implementation 9c65ca3, CPU-tested, never run on a GPU: validated by its 256 step 0 on the runner's GPU) |
| 1, MMLU | `evaluation.downstream` (MMLU 5-shot, lm-eval 0.4.11; in flipquant since paper-sm120-runs 4bbe0b1); FOCUS-1k / GPTQ‡-1k from HF; the published 256 cells | paper-eval records | step-0 values and the paper's MMLU column to `reference/item1.json`, with part AA's 1k zero-shot accuracies for Table 3's Avg. | yes — **ready** (`experiments/paper_1k/item1.py`) |
| (latency-rounds) | the multi-round prefill driver | flipquant branch `latency-rounds` (80cd3f0) | not needed for the remaining items (item 2's latency is done); its driver is the reference if a latency rerun is ever asked | — |

## Risks and open points

- **D.2 / D.3 (item 9)** was the likeliest to stay partial; it is ported (see its row). What remains open from below:
  the unified run's exact configuration list is not available, so D.3's parts follow the paper's text and kernel-opt's
  C2 definitions (RUNBOOK item 9).
  - The per-MMA kernel options must be re-vendored, which touches `kernels/sm120`'s generator, patcher, mainloop and
    `build.py`. The deployed builds' SASS must stay equal to the reference.
  - The paper's D.3 numbers come from a run whose code is not available, so the four parts are re-derived from the
    text: tiles = no-dispatch build vs stock; tag copies = tagged vs TAG0=0; dispatch = all-E2M1 vs no-dispatch;
    E0M3 = real tags vs all-E2M1. The pre-optimization kernels are the paper's sets.
- **Kernel gate on another machine:** a different compiler gives different library hashes. The port gates on SASS
  equality with `SASS_REFERENCE.json` instead. If the SASS differs, the runner's build is not the paper's, and the
  timing items say so.
- **Timing items (2's memory part, 5, 8, 9)** compare only within the runner's own session and its step 0. The paper's
  values come from other boards and sessions.
- **The hybrid models** need the fast env (mamba-ssm, causal-conv1d, fla) and CUDA 12.8 wheels; its exact pins are
  committed.

## Phase A on the porting machine's GPU (2026-10-25)

After the GPU was reset, the flipquant-only path was validated on the porting machine's RTX PRO 6000 from a fresh
clone of this branch (NVFP4-RaZeR off the path), before running the items there. Changes and fixes found on the way:

- **Mitigation, the runner's 5 s pause** (`jobs.GPU_GAP_S`): every job starts at least 5 s after the previous GPU
  job's exit. The GPU hung on 2026-10-24 about 1 s after a job's exit. The pause is outside the commands: no command
  changes (`tests/test_paper_1k.py::test_runner_pauses_after_a_gpu_job`).
- **`PAPER1K_HF_CACHE`** (`artifacts.cache_dir`): a separate hub cache for the pinned artifacts (160 GB), so that
  they can live on another disk while the model weights stay in `HF_HOME` (the porting machine's home disk had 30 GB
  free). The sha256 checks are unchanged (`test_artifact_cache_dir`).
- **RUNBOOK 1.2** now says to initialize the CUTLASS submodule before the build (a fresh clone does not have it).
- **Bug (tests): the dry-run tests read the runner's own settings.** With `PAPER1K_CODES` set (item 4's codes on
  another disk), `test_item4_commands_are_the_runs` compared a code path outside the run directory with the
  reference's `{OUT}/codes/…`. The tests' `dry()` now clears `PAPER1K_CODES`, `PAPER1K_HF_CACHE`, `PAPER1K_BUILD_ITEM9`,
  `PAPER1K_ACCEPT` and `PAPER1K_SMOKE_JOBS` first.
- **Item 9, the timing session's first GPU run** (one repeat, into a scratch directory, discarded): it ran, and
  the report rendered from its record. Cosmetic fix: Table D.2's µs columns printed a sign ("+118.48"); they are
  absolute values now.
- **Bug: `setup windows` kept an unfinished windows file.** Llama-3.1-8B-Instruct's 2,048-window draw hit a cut-off
  dataset download (open-web-math), retried, and its child process aborted at exit (rc -6) after writing the file;
  the wrapper's last write (`model`, `flipquant`) never happened. Setup skipped any existing file, so a rerun would
  have kept it. A file now counts as drawn only with the wrapper's `flipquant` field; otherwise it is drawn again
  (`test_setup_windows_redraws_an_unfinished_file`). The same abort hit 3 of the first 13 draws (with or without a
  visible download retry; an artifact fetch was sharing the network), so setup now tries a draw up to 3 times
  (`DRAW_ATTEMPTS`); the token hashes are checked whatever the attempt. fit_windows refuses an existing `--out`, so
  an unfinished file is first moved aside as `<name>.unfinished<k>.json` (never deleted). In the first pass, 6 of the
  18 draws aborted at exit this way; their files were complete.
- **Not in this port (reported, unchanged):** in the full suite on the GPU, two tests outside `experiments/paper_1k`
  fail in the n16k64 env as on flipquant main: `test_ifeval_scoring` (`langdetect` is not in `env/n16k64.yml`; no item
  uses IFEval) and `test_teacher_topk.py::test_topk_kl_bounds_full_kl` (also on the CPU: the BF16-stored top-k values
  exceed the test's 1e-3 bound by 2e-3 at k = 299 of 300; the trainings' maps are gated bit for bit in items 11/12).
  With the GPU hidden (the CPU suite: 289 passed, 107 skipped), two more fail, both independent of the port:
  `test_sm120_native.py::test_auto_refuses_unoptimized_builds` (it loads the libraries, so it needs the GPU, as on
  main) and `test_fo6_tensor_qmax.py::test_every_block_can_take_four_with_256` (part R's test, d92c032: it compares
  two of its own expressions at the 2/3 boundary, which the CPU rounds differently; it passes on the GPU). With the
  GPU visible: 340 passed, 55 skipped, and only the two above (`langdetect`, the top-k bound) fail;
  `test_sm120_native.py` alone: 22 passed, 9 skipped (they need a NVFP4-RaZeR reference checkout).
- **`PAPER1K_SMOKE_JOBS`** (the runner): stop an item before its (N+1)-th started job, so that a step 0 runs alone
  through the real driver and its gates (`test_runner_smoke_limit`). Phase A's step-0 smokes use it.
- **Bug: `setup kernels` / `setup sass` hid the GPU** (`CUDA_VISIBLE_DEVICES=""`), but one of the gate's two tests,
  `test_auto_refuses_unoptimized_builds`, loads the libraries, and the load initializes CUDA. On the GPU it failed with
  "No CUDA GPUs are available" (the SASS test itself passed). Fixed: the gate keeps the caller's GPU visible
  (`test_sass_gate_keeps_the_gpu_visible`).

### Phase B additions

- **Item 4, the trainings' environments and an auxiliary fast-env repeat** (coordinator, 2026-10-26). Part H's job
  logs show its Hadamard trainings ran in n16k64-fast, round 3's GPTQ-candidate trainings in n16k64; the 1k driver
  keeps both routes. The report now says which env each training used, with its time. After the item, two auxiliary
  jobs (not gates) re-run part H's exact 256 Hadamard command (Nemotron, Qwen3.8-27B) and compare the maps' tiles with
  part H's (digests committed in `reference/item4.json`): whether the fast kernels' trainings reproduce bit for bit.
  The dry-run equivalence test checks both commands against part H's, token for token.

- **Item 11 cancelled by the user (2026-10-26); filled from part AA records.** `run.CANCELLED` /
  `RUN_ORDER` (run and dry run refuse it; the index says so); `item11.report` builds Tables C.2 / C.3 from
  `reference/item11.json` (part AA's job records) and the new `reference/item11_cost.json` (the trainers' phases and
  epochs at 1k, the release trainings' at 256, and phase A's step 0), so the report regenerates from flipquant alone.
  The re-measurement driver is kept, unused.

- **Bug: item 7's report copied the control records' local paths** (the map file in the artifact cache, the build
  directory in the kernel routing note), and publish's scan refused `item7.json`. `item7.portable` now writes the map
  as its Hugging Face repository and file and drops the build directory (`test_item7_report_has_no_local_paths`).

- **Item 1's deviations (coordinator, 2026-10-26)** after Nemotron's first MMLU row ran out of GPU memory at batch
  16: the runner's opt-in `PAPER1K_ALLOC_CONF` (an environment deviation, recorded per job, marked ¤), consistency
  rows against the published 256 cells (gate `item1-consistency`), and the fallback `PAPER1K_ITEM1_BATCH` (a command
  deviation, marked ‡b). Tests: `test_runner_alloc_conf`, `test_item1_consistency_gate`,
  `test_item1_deviations_marked`.

- **INCIDENT 2026-10-26 21:49 UTC: the GPU hung again** during item 1's Nemotron consistency row, about 4.5 min into
  the job, after the out-of-memory stop at 21:39 and the restart under (A) at 21:45
  (`results/paper_1k/item1/INCIDENT_2026-10-26.md`). Item 1 stopped there; 16 of its 31 jobs remain. Every other item
  is published. Recovery is the user's.

### Phase A results (2026-10-25, the porting machine's RTX PRO 6000, 500 W, driver 595.71.05)

From a fresh clone of this branch, with the fixes above pulled in as they were pushed. Every step 0 below ran through
the driver (`PAPER1K_SMOKE_JOBS` = its step-0 job count) into the run directory that phase B continues.

| check | what ran | result | time |
|---|---|---|---|
| CUDA | a 4096² BF16 matmul | pass | 1 s |
| `setup env`, `kernels`, `sass`, `hf` | both interpreters; build_deploy.sh + SASS gate; 2 gated models + 36 repositories | pass (kernels after the SASS-gate fix) | 2 min build |
| `setup fetch` | 60 files, 160 GB | pass (every sha256) | 3 h 16 min |
| `setup records` | 6 models | pass | 34 min |
| `setup windows` | 18 files | pass (after the two windows fixes) | about 1 h |
| tests | the full suite (GPU visible / hidden), `test_sm120_native.py` | 340 / 289 passed; failures as listed above, none in the port | 9 min |
| item 9 gates | the seven extra builds, the census, the per-MMA check | `item9-sass`, `item9-census`, `item9-check` pass | 1.5 min |
| item 9 session | one repeat (scratch, discarded) | ran; report rendered (µs sign fix) | 13 s |
| item 2 step 0 | the memory anchor | pass: bytes equal, 0.00 GiB | 21 s |
| item 3 step 0 | the two 1k smokes, the 256 rebuild | pass: tiles digest equal (the gradient file's sha256 too) | 17 min |
| item 4 step 0 | round 3's Nemotron GPTQ NVFP4 at 256 | pass: codes digest equal, data equal, per-window NLL exact | 12 min |
| item 5 step 0 | Phi-4, the release maps | pass: T ≥ 128 +6.24 / +0.97 / +1.28 % (published 6.72 / 0.75 / 0.91) | 3.5 min |
| item 6 step 0 | Mistral 16x64 256, native and simulated | PASS-exact (both bit for bit) | 5 min |
| item 7 step 0 | C1's readback, Mistral 16x64 256 | pass: rows equal | 22 s |
| item 8 step 0 | random maps + gemm_time, the release maps | pass: worst 0.86 points; ranges 8x64 −0.50..+0.86, 16x64 −0.74..+0.46, 256x64 −0.76..+0.69 | 3.4 min |
| item 11 step 0 | Llama-3.2-1B-Instruct 16x64 at 1k | pass: map equal to the 1k map; 223 s (part AA 225 s) | 3.7 min |
| item 12 step 0 | the seed-0 256 recalibration, the three audits | pass: map equal; no 13-gram hit, every splice found | 13 min |
| item 13 step 0 | one-shot, smoke, the 256 run | pass, all four criteria: baseline KL bit-equal; map −17.51 ± 1.68 / −26.06 ± 1.95 (published −17.8 / −26.5); ρ(g_u) 0.73 / 0.91 | 24 min |
| item 1 step 0 | Llama-3.2-1B-Instruct FlipQuant 16x64 256, MMLU | pass (exact): 41.40 | 7.4 min |

Timing step 0s (items 5, 8, 11) were first run as smokes into a scratch run directory while the 160 GB fetch shared
the host, then for real on the idle host. In the scratch run item 8's gate failed (worst 1.84 points: two single
cells, 16x64 o_proj at T = 2048 and 8x64 gate_proj at T = 16; every other cell within 1 point); on the idle host it
passed. Items 5 and 11 passed both times (item 11's calibration took 315 s under the fetch).

