# FlipQuant: Gradient-Guided Hardware-Native MixFP4

Code for **"Gradient-Guided Hardware-Native MixFP4"** ([arXiv:2611.18769](https://arxiv.org/abs/2611.18769)).

Jian-Jia Chen, Brian J. Chan, Wen-Hung Lee, Yi-Ting Li, Po-Shen Chen, Kuan-Te Kuo, Hsien-Cheng Huang, Cody Hao Yu,
Chi-Chih Chang, Mohamed S. Abdelfattah, Kai-Chiang Wu.

## The paper in brief

NVFP4 stores every weight as a 4-bit E2M1 code with one E4M3 scale per 16 elements. Formats that pick a different
4-bit grid per 16-element block, such as IF4 and MixFP4, close more of NVFP4's gap to BF16. But Blackwell's tensor
cores fix the operand format per MMA instruction, so those per-block formats need new hardware.

**FlipQuant** picks the format per **MMA tile** instead: per `r x 64` weight tile (`r` = 8, 16 or 256), either
**E2M1** with FourOverSix block scaling or **E0M3**, a uniform grid {0, ..., 7}. That runs on today's tensor cores.

- **The map.** One bit per tile, learned by gradient descent. Both candidates are quantized once. A latent logit
  per tile goes through a hard threshold in the forward pass and a straight-through estimator in the backward pass.
- **The loss.** KL divergence to the frozen BF16 model over its top-1,000 tokens plus a tail bucket, on 1,024
  math and code sequences of 512 tokens (5 epochs, 640 Adam steps).
- **Storage.** The tag lives in the unused sign bit of the tile's E4M3 scales on SM120. On SM100 it costs one bit per
  256x64 tile.

**Results.** FlipQuant closes 42.5% of NVFP4's perplexity gap to BF16 at 16x64 tiles, against 13.3% for FourOverSix,
averaged over WikiText-2, C4 and six models. Per-block IF4 and MixFP4, run at hardware-tile granularity, reach 22.3%.
The GEMM overhead is 1.9% on SM120 (RTX PRO 6000, 16x64) and 1.7% on SM100 (GB200, 256x64).

**Models:** Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Nemotron-Nano-9B-v2 (Mamba-2
hybrid), Phi-4 and Qwen3.8-27B (Gated DeltaNet hybrid). Their pinned revisions are in `models/registry.py`.

## Repository layout

```
flipquant/        formats (NVFP4, FourOverSix, E0M3; bitwise reference quantizers), tile maps, FlipLinear,
                  GPTQ on the fixed FP4 grid, block-16 Hadamard rotation, the IF4 and MixFP4 baselines, eval data
models/           model registry, loading and in-place quantization; the SM120 (sm120.py) and SM100 (sm100.py)
                  native paths; tiny random models for tests
calibration/      train_map.py, the FlipQuant trainer: the paper's TM-OPT+TC trainer, vendored unchanged in tmopt/
                  tmopt_ext.py / baseline_maps.py: the map-selection ablation (one-shot gradient, random, activation)
                  gptq_codes.py, fit_windows.py, gradcheck.py: GPTQ codes, calibration windows, the F.1 check
evaluation/       ppl.py (WikiText-2, C4), significance.py / paired_nll.py (paired ΔNLL ± 2 SE),
                  downstream.py (lm-eval: multiple choice, gsm8k_llama), latency.py, gemm_time.py, decode_check.py
kernels/          sm120/: the SM120 deployment kernels (per-K-tile format dispatch) and their build
                  src/, capi/: the SM100 CUTLASS per-tile kernel and the C API; scripts/: SM100 jobs (GB200)
experiments/      paper_1k/: one driver per paper result at the paper's 1,024-window calibration, with gates
                  against the published records (results/paper_1k/RUNBOOK.md)
scripts/          table2.py, table2_zou.py, map_weight_error.py: Appendix B (per-block rules coarsened to tiles)
maps/             small committed maps for tests and quick runs (the paper's maps are on the Hugging Face Hub)
tests/            unit and integration tests (most run on the CPU)
```

## Setup

The pinned environments are in `experiments/paper_1k/env/`:

| file | use |
|---|---|
| `n16k64.yml` | evaluation and calibration (Python 3.11, torch 2.9.0+cu128, transformers 5.16.1, lm-eval 0.4.11, triton 3.5.0) |
| `n16k64-fast.yml` | the same plus mamba-ssm and flash-linear-attention, for Nemotron-Nano-9B-v2 and Qwen3.8-27B |
| `mixfp4-cuda131.yml` | the CUDA 13.1 toolkit, to build the SM120 kernels (CPU only) |

```bash
git clone --recursive https://github.com/brian030128/flipquant && cd flipquant
conda env create -f experiments/paper_1k/env/n16k64.yml && conda activate n16k64
python -m experiments.paper_1k.setup env       # checks every pinned version
hf auth login                                  # the Llama models are gated
python -m experiments.paper_1k.setup fetch     # the paper's maps and GPTQ codes, checked by sha256
```

Build the SM120 kernels once (CPU only; the SASS is checked against `kernels/sm120/SASS_REFERENCE.json`):

```bash
export CUDA_HOME=$(conda run -n mixfp4-cuda131 sh -c 'echo $CONDA_PREFIX')
python -m experiments.paper_1k.setup kernels   # runs kernels/sm120/build_deploy.sh, then the SASS gate
```

**Execution modes** (`--mode`, every command):

- `bf16`: the unquantized model.
- `fake`: simulated quantization, bit-faithful weights, on any CUDA GPU.
- `native`: FP4 tensor-core kernels. An RTX PRO 6000 or RTX 5090 (SM120) runs `models/sm120.py`. A GB200 or B200
  (SM100) runs `--native-backend sm100-tuned` (`models/sm100.py`).

Activations are always FourOverSix, per token. `--paper-convention` sets this for perplexity.

## Using FlipQuant

**Calibrate a map** with the paper's settings:

```bash
python -m calibration.train_map --model mistral-7b --unit 16x64 \
    --fit-windows 1024 --epochs 5 --teacher-topk 1000 --out runs/mistral-7b_16x64
# Qwen3.8-27B on one 96 GB GPU: add --teacher-host (the top-K teacher in host memory; the same map)
```

- **Output.** `runs/.../map.pt`, with the settings, model revision and cost in its `meta`.
- **Hardware and cost.** The paper's maps ran on one RTX PRO 6000: 3.7 minutes (Llama-1B) to 170 minutes
  (Nemotron-9B), and at most 91 GiB.
- **Defaults.** The trainer's own defaults (128 windows, 20 epochs, full-vocabulary teacher) are an earlier
  protocol. They are kept so that `--require-reproduction` can check the committed earlier maps bit for bit.

**Evaluate:**

```bash
M="--weight mixfp4 --map runs/mistral-7b_16x64/map.pt"
python -m evaluation.ppl --model mistral-7b --mode native $M --paper-convention --out results/ppl_fq16.json
python -m evaluation.ppl --model mistral-7b --mode native --weight fourover6 --paper-convention --out results/ppl_fo6.json
python -m evaluation.significance --ref fo6=results/ppl_fo6.json --cmp fq16=results/ppl_fq16.json

python -m evaluation.downstream --model mistral-7b --mode native $M --tasks appendix --out results/mc_fq16.json
python -m evaluation.downstream --model mistral-7b --mode native $M --tasks gsm8k_llama --apply-chat-template \
    --out results/gsm8k_fq16.json
python -m evaluation.latency prefill --model mistral-7b --mode native $M --out results/prefill_fq16.json
```

**Baselines** use the same commands with another weight format:

- `--weight nvfp4 --act nvfp4`: NVFP4 RTN.
- `--weight fourover6`: FourOverSix.
- `--weight if4 --unit <tile>`: IF4. Simulated only, so use `--mode fake`.
- `--weight zou_mixfp4 --unit <tile>`: MixFP4 of Zou et al. Simulated only, so use `--mode fake`.
- `--weight nvfp4 --ptq gptq ...`: GPTQ‡. See `experiments/paper_1k/item1.py` for the exact flags.
- `--rotate hadamard`: Hadamard rotation.

## Reproducing the paper

Every result has a driver in `experiments/paper_1k`. A driver writes all of its commands, runs them one after
another on one GPU, and resumes where it stopped. It first reproduces an earlier record (step 0), and stops on any
failed gate. Read **`results/paper_1k/RUNBOOK.md`** before running one. It covers setup, prerequisites, gates and
expected run times for each item. `results/paper_1k/<item>/` holds the published records.

```bash
python -m experiments.paper_1k.run <item> --dry-run   # the item's commands
python -m experiments.paper_1k.run <item>             # run (resumable)
python -m experiments.paper_1k.run <item> --report    # its tables, into results/paper_1k/<item>/
```

| paper | what | how |
|---|---|---|
| Table 2, Figure 1 | WikiText-2 / C4 perplexity, W4A4 | `evaluation.ppl --paper-convention` per model × method; `evaluation.significance` |
| Table 3, Tables E.1–E.3 | seven multiple-choice tasks, GSM8K | `evaluation.downstream --tasks appendix` / `--tasks gsm8k_llama`; MMLU at 1k: `item1` |
| Table 4 (SM120 rows) | prefill latency, 1x2048 | `item2` (`evaluation.latency prefill`) |
| Table 4 (GB200 rows) | prefill latency on SM100 | `kernels/scripts/prefill_setup_gb200.sbatch`, `prefill_rounds_gb200.sbatch` |
| Table 5 | how the map is chosen: random, activation-weighted, one-shot gradient, FlipQuant | `item3` (`calibration.baseline_maps`, `calibration.tmopt_ext`) |
| Appendix A.4 | native vs simulated perplexity | `item6`; SM100: `kernels/scripts/ppl_native_vs_sim_gb200.sbatch` |
| Appendix B | per-block IF4 / MixFP4 rules coarsened to tiles | `scripts/table2.py`, `scripts/table2_zou.py`, `scripts/map_weight_error.py` |
| Appendix C | calibration cost and robustness (seeds, data, nondeterminism) | `item11` (records), `item12` |
| Appendix D.3–D.4 | SM120 per-MMA vs per-K-tile dispatch, overhead split, real vs random tags | `item9`, `item8` |
| Appendix D.5 | forward-pass GEMM time per model (SM120) | `item5` (`evaluation.gemm_time`) |
| Appendix D.6 | the SM100 per-tile kernel: GEMM overhead, MMA width, N/K tails | `kernels/scripts/forward_gemm_gb200.sbatch`, `pad_sweep_gb200.sbatch` |
| Appendix D.7 | correctness: reading every weight back through the kernel | `item7` (`--ownership-check`); SM100: `models/sm100.py` |
| Appendix F.1 | the tile-gradient ranking check | `item13` (`calibration.gradcheck`) |
| Appendix F.2 | FlipQuant with GPTQ and with Hadamard rotation | `item4` |

- **Numbering.** The item READMEs and `results/paper_1k/PLAN.md` use the numbering of a pre-print draft. The table
  above uses the arXiv numbering.
- **SM100 jobs.** `kernels/docs/results/sm100_paper/README.md` lists each SM100 job with its outputs.

## Tests

```bash
pytest tests        # GPU-, kernel- and model-dependent tests skip themselves when their prerequisites are missing
```

## Citation

```bibtex
@article{chen2026flipquant,
  title   = {Gradient-Guided Hardware-Native {MixFP4}},
  author  = {Chen, Jian-Jia and Chan, Brian J. and Lee, Wen-Hung and Li, Yi-Ting and Chen, Po-Shen and Kuo, Kuan-Te
             and Huang, Hsien-Cheng and Yu, Cody Hao and Chang, Chi-Chih and Abdelfattah, Mohamed S. and Wu, Kai-Chiang},
  journal = {arXiv preprint arXiv:2611.18769},
  year    = {2026}
}
```
