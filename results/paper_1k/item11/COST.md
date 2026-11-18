# Item 11: Tables C.2 / C.3 (the calibration cost) at the 1k calibration — from part AA's records

**The GPU re-measurement was cancelled by the user (2026-10-26); filled from part AA records.** The 18 1k FlipQuant calibrations (six models x 8x64 / 16x64 / 256x64) are part AA's runs of `calibration.train_map --epochs 5 --teacher-topk 1000 --fit-windows 1024` (deterministic kernels, no development set, the n16k64 env, one RTX PRO 6000 at 500 W): their job records (`reference/item11.json`) and the trainer's own reports (`reference/item11_cost.json`), committed so that this report regenerates from flipquant alone (`python -m experiments.paper_1k.run item11 --report`).

- **Setup / Training:** the trainer's phases (setup = model load, data load, teacher precompute, candidate packing). **Trainer** = Setup + Training. **Total:** the job's wall clock, from launch to the written map.
- **GPU:** the trainer's peak allocated / reserved memory; **device peak:** nvidia-smi's memory.used, sampled every 2 s. **Teacher:** the top-1,000 teacher's storage (values BF16, ids int32, tail log-mass FP32).
- **Epoch / step:** the mean of the five epochs; 128 optimizer steps per epoch at 1,024 windows (32 at 256).

## Reproduction evidence

Phase A (2026-10-25) ran item 11's step 0 from a fresh flipquant clone on the porting machine's idle RTX PRO 6000: Llama-3.2-1B-Instruct 16x64, part AA's command. Launch to map 223 s against part AA's 225 s, and the map equal to the 1k map bit for bit (gate `item11-maps`).

## Table C.2: the cost at 16x64

| model | Setup (min) | Training (min) | Trainer (min) | Total (min) | training share | GPU alloc. / reserved (GiB) | device peak (GiB) | teacher |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Llama-3.2-1B-Instruct | 0.9 | 2.6 | 3.6 | 3.7 | 73 % | 16.6 / 17.0 | 17.7 | 2.93 GiB on the GPU |
| Llama-3.1-8B-Instruct | 1.8 | 13.3 | 15.2 | 15.4 | 88 % | 43.9 / 44.8 | 45.5 | 2.93 GiB on the GPU |
| Mistral-7B-Instruct-v0.3 | 1.7 | 12.7 | 14.4 | 14.6 | 88 % | 38.8 / 39.7 | 40.4 | 2.93 GiB on the GPU |
| Nemotron-Nano-9B-v2 | 11.2 | 159.0 | 170.2 | 170.4 | 93 % | 55.9 / 57.4 | 58.1 | 2.93 GiB on the GPU |
| Phi-4 | 3.0 | 25.0 | 28.0 | 28.1 | 89 % | 62.1 / 65.3 | 66.0 | 2.93 GiB on the GPU |
| Qwen3.8-27B | 13.7 | 144.0 | 157.8 | 158.0 | 91 % | 90.5 / 92.0 | 92.8 | 2.93 GiB on the host |

### The setup phases and the step time at 16x64 (s)

| model | model load | data load | teacher precompute | candidate packing | epoch (mean) | per step |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | 0.7 | 27.4 | 23.8 | 5.1 | 31.3 | 0.245 |
| Llama-3.1-8B-Instruct | 1.6 | 27.4 | 45.0 | 35.8 | 160.1 | 1.251 |
| Mistral-7B-Instruct-v0.3 | 2.3 | 24.8 | 40.9 | 36.0 | 152.1 | 1.188 |
| Nemotron-Nano-9B-v2 | 1.9 | 27.7 | 603.3 | 41.0 | 1907.5 | 14.902 |
| Phi-4 | 4.3 | 28.1 | 71.9 | 73.2 | 300.0 | 2.344 |
| Qwen3.8-27B | 5.3 | 30.2 | 657.8 | 129.2 | 1728.5 | 13.504 |

## Table C.3: the three tiles

| model | trainer 8x64 / 16x64 / 256x64 (s) | Total 8x64 / 16x64 / 256x64 (s) | max. diff. trainer (%) | max. diff. Total (%) | GPU 8x64 − 256x64 (GiB) |
|---|---|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 211 / 213 / 209 | 223 / 225 / 220 | 1.9 | 2.0 | +0.04 |
| Llama-3.1-8B-Instruct | 901 / 910 / 915 | 913 / 922 / 926 | 1.5 | 1.5 | +0.25 |
| Mistral-7B-Instruct-v0.3 | 858 / 864 / 865 | 869 / 876 / 876 | 0.8 | 0.8 | +0.24 |
| Nemotron-Nano-9B-v2 | 10,163 / 10,211 / 10,217 | 10,175 / 10,223 / 10,228 | 0.5 | 0.5 | +0.23 |
| Phi-4 | 1,661 / 1,677 / 1,655 | 1,673 / 1,688 / 1,666 | 1.4 | 1.3 | +0.47 |
| Qwen3.8-27B | 9,476 / 9,465 / 9,416 | 9,489 / 9,478 / 9,428 | 0.6 | 0.6 | +0.87 |

## 256 against 1,024 windows (the release calibrations' trainer reports)

Ratios 1k / 256 per model at 16x64; in brackets the range over the three tiles.

| model | setup | teacher precompute | candidate packing | model load | epoch | per step | teacher memory (GiB, 256 → 1k) |
|---|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2-1B-Instruct | 1.76 [1.65–1.79] | 3.81 [3.49–3.81] | 1.02 [0.98–1.02] | 1.98 [1.67–1.98] | 4.05 [3.98–4.05] | 1.01 [1.00–1.01] | 0.73 (GPU) → 2.93 (GPU) |
| Llama-3.1-8B-Instruct | 1.57 [1.45–1.78] | 3.80 [3.60–3.83] | 0.98 [0.98–0.99] | 1.01 [0.38–1.38] | 3.98 [3.92–3.98] | 0.99 [0.98–0.99] | 0.73 (GPU) → 2.93 (GPU) |
| Mistral-7B-Instruct-v0.3 | 1.55 [1.53–1.66] | 3.85 [3.83–3.85] | 1.00 [0.98–1.00] | 1.57 [1.35–1.57] | 3.98 [3.94–3.98] | 0.99 [0.98–0.99] | 0.73 (GPU) → 2.93 (GPU) |
| Nemotron-Nano-9B-v2 | 3.13 [3.13–3.16] | 3.98 [3.96–4.01] | 1.00 [1.00–1.02] | 1.10 [0.98–1.28] | 3.98 [3.96–3.98] | 1.00 [0.99–1.00] | 0.73 (GPU) → 2.93 (GPU) |
| Phi-4 | 1.54 [1.54–1.56] | 3.93 [3.88–4.05] | 1.01 [0.97–1.01] | 1.20 [1.20–1.36] | 3.99 [3.93–3.99] | 1.00 [0.98–1.00] | 0.73 (GPU) → 2.93 (GPU) |
| Qwen3.8-27B | 2.57 [2.54–2.57] | 4.03 [3.94–4.03] | 0.99 [0.99–1.01] | 0.80 [0.80–0.89] | 4.01 [3.99–4.01] | 1.00 [1.00–1.00] | 0.73 (GPU) → 2.93 (host) |

Over all 18: setup x1.45–3.16, of which the teacher precompute x3.49–4.05 (four times the windows), while the candidate packing (x0.97–1.02) and the model load (a few seconds) do not grow; one epoch x3.92–4.05; the time per optimizer step x0.981–1.012, i.e. unchanged.

## Caveats

- **Qwen3.8-27B keeps its top-1,000 teacher on the host (`--teacher-host`)**: at 1,024 windows the teacher (2.93 GiB) does not fit on the GPU next to the model and the training state. Its per-step time is unchanged against 256 (x1.003 at 16x64).
- **Two campaigns.** The 16x64 runs are part AA's panel (2026-10-19/20; Llama-3.2-1B-Instruct and Phi-4 in its pilot); the 8x64 and 256x64 runs are amendment 6 (2026-10-23/24), which ran while the Hugging Face staging used the CPU and the network. The per-step times of the three tiles agree within 1.6 %.
- **Paper-text points.** The teacher's memory at 1k is about 2.9 GiB (2.93 GiB), not the 0.73 GiB of 256 windows. Qwen3.8-27B's teacher is on the host at 1k, which contradicts Table C.1's "kept on the GPU" and Section 3's "under 1 GiB" for the 1k maps.

## Every calibration

| model | tile | date (UTC) | campaign | E0M3 tiles | Setup (s) | Training (s) | Total (s) | epoch (s) | per step (s) | GPU alloc. / reserved (GiB) | device peak (GiB) | teacher |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| Llama-3.2-1B-Instruct | 8x64 | 2026-10-23 | amendment 6 (394e683) | 214,432 | 57 | 154 | 223 | 30.8 | 0.241 | 16.6 / 17.2 | 17.9 | GPU |
| Llama-3.2-1B-Instruct | 16x64 | 2026-10-19 | part AA panel (f912a30) | 141,062 | 57 | 156 | 225 | 31.3 | 0.245 | 16.6 / 17.0 | 17.7 | GPU |
| Llama-3.2-1B-Instruct | 256x64 | 2026-10-23 | amendment 6 (394e683) | 23,572 | 55 | 154 | 220 | 30.8 | 0.241 | 16.6 / 17.2 | 17.9 | GPU |
| Llama-3.1-8B-Instruct | 8x64 | 2026-10-23 | amendment 6 (394e683) | 586,062 | 110 | 791 | 913 | 158.2 | 1.236 | 44.0 / 45.1 | 45.8 | GPU |
| Llama-3.1-8B-Instruct | 16x64 | 2026-10-19 | part AA panel (898d00e) | 377,393 | 110 | 800 | 922 | 160.1 | 1.251 | 43.9 / 44.8 | 45.5 | GPU |
| Llama-3.1-8B-Instruct | 256x64 | 2026-10-23 | amendment 6 (394e683) | 64,091 | 125 | 790 | 926 | 157.9 | 1.234 | 43.8 / 44.8 | 45.5 | GPU |
| Mistral-7B-Instruct-v0.3 | 8x64 | 2026-10-23 | amendment 6 (394e683) | 557,943 | 106 | 752 | 869 | 150.4 | 1.175 | 39.0 / 39.9 | 40.6 | GPU |
| Mistral-7B-Instruct-v0.3 | 16x64 | 2026-10-19 | part AA panel (898d00e) | 359,840 | 104 | 760 | 876 | 152.1 | 1.188 | 38.8 / 39.7 | 40.4 | GPU |
| Mistral-7B-Instruct-v0.3 | 256x64 | 2026-10-23 | amendment 6 (394e683) | 63,099 | 113 | 752 | 876 | 150.5 | 1.175 | 38.7 / 39.6 | 40.3 | GPU |
| Nemotron-Nano-9B-v2 | 8x64 | 2026-10-23 | amendment 6 (394e683) | 645,487 | 671 | 9,492 | 10,175 | 1898.5 | 14.832 | 56.1 / 57.5 | 58.2 | GPU |
| Nemotron-Nano-9B-v2 | 16x64 | 2026-10-19 | part AA panel (898d00e) | 420,667 | 674 | 9,537 | 10,223 | 1907.5 | 14.902 | 55.9 / 57.4 | 58.1 | GPU |
| Nemotron-Nano-9B-v2 | 256x64 | 2026-10-23 | amendment 6 (394e683) | 76,421 | 671 | 9,546 | 10,228 | 1909.2 | 14.916 | 55.9 / 57.3 | 58.0 | GPU |
| Phi-4 | 8x64 | 2026-10-23 | amendment 6 (394e683) | 758,874 | 178 | 1,483 | 1,673 | 296.6 | 2.317 | 62.4 / 65.4 | 66.1 | GPU |
| Phi-4 | 16x64 | 2026-10-19 | part AA panel (f912a30) | 491,665 | 177 | 1,500 | 1,688 | 300.0 | 2.344 | 62.1 / 65.3 | 66.0 | GPU |
| Phi-4 | 256x64 | 2026-10-23 | amendment 6 (394e683) | 85,876 | 175 | 1,480 | 1,666 | 296.0 | 2.312 | 61.9 / 65.1 | 65.9 | GPU |
| Qwen3.8-27B | 8x64 | 2026-10-23 | amendment 6 (394e683) | 1,074,352 | 826 | 8,650 | 9,489 | 1730.1 | 13.516 | 91.0 / 92.5 | 93.3 | host |
| Qwen3.8-27B | 16x64 | 2026-10-20 | part AA panel (898d00e) | 602,694 | 823 | 8,643 | 9,478 | 1728.5 | 13.504 | 90.5 / 92.0 | 92.8 | host |
| Qwen3.8-27B | 256x64 | 2026-10-24 | amendment 6 (394e683) | 105,154 | 813 | 8,603 | 9,428 | 1720.6 | 13.442 | 90.1 / 91.6 | 92.3 | host |
