# Item 3: Table 5 (`tab:ablation`) at the 1k calibration (Nemotron-Nano-9B-v2, Qwen3.8-27B)

Part G's protocol at the 1k calibration (NVFP4-RaZeR paper-eval PROTOCOL.md, part AC item 3), run from flipquant alone (`python -m experiments.paper_1k.run item3`). Native: this repository's SM120 build, --kernel-set auto, paper convention. Every baseline flips exactly the 1k map's k_l tiles per layer. The activation-weighted and one-shot statistics use the 1k windows (the record's 128 + the extension's 896); Qwen3.8-27B's one-shot keeps the top-1000 teacher on the host. Cells: paired ΔNLL vs FourOverSix, x 1e-3 nats/token, ± 2 SE (* = beyond 2 SE); Random = the per-window mean of seeds 0 / 1 / 2. FourOverSix and the 1k FlipQuant maps are re-measured in this session (their agreement with the paper's records is below).

## Step 0

- **Smoke at 1k** (random-init 3-layer models, 2 optimizer steps, `--fit-windows 1024`): nemotron-nano-9b-v2: pass (extension 896 windows, token hashes equal the paper's: True; teacher on cuda:0); qwen3.8-27b: pass (extension 896 windows, token hashes equal the paper's: True; teacher on cpu).
- **The 256 rebuild** (G's Nemotron 16x64 one-shot map on this code): tiles bit for bit equal (120 modules, 80098 E0M3 tiles); the summed-gradient file equal.

## WikiText-2

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | Random | -0.70 ± 0.86 | -0.32 ± 0.88 | +0.71 ± 0.94 |
| Nemotron-Nano-9B-v2 | Activation-weighted | -1.26 ± 1.19* | -0.81 ± 1.17 | +2.78 ± 1.17* |
| Nemotron-Nano-9B-v2 | One-shot gradient | +89.90 ± 4.64* | +56.81 ± 3.33* | -2.06 ± 1.74* |
| Nemotron-Nano-9B-v2 | FlipQuant | -12.94 ± 1.18* | -11.86 ± 1.32* | -9.63 ± 1.35* |
| Qwen3.8-27B | Random | +2.12 ± 2.79 | -2.83 ± 2.88 | +4.19 ± 2.47* |
| Qwen3.8-27B | Activation-weighted | -8.29 ± 3.93* | -8.27 ± 3.93* | -3.92 ± 4.03 |
| Qwen3.8-27B | One-shot gradient | +3.92 ± 8.61 | -14.52 ± 7.88* | -29.86 ± 6.69* |
| Qwen3.8-27B | FlipQuant | -35.42 ± 6.00* | -31.15 ± 5.18* | -17.71 ± 4.92* |

## C4

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | Random | +0.36 ± 0.59 | +0.79 ± 0.59* | +1.34 ± 0.66* |
| Nemotron-Nano-9B-v2 | Activation-weighted | +0.51 ± 0.81 | -0.07 ± 0.78 | +2.13 ± 0.92* |
| Nemotron-Nano-9B-v2 | One-shot gradient | +72.54 ± 4.08* | +47.48 ± 2.87* | +1.48 ± 1.13* |
| Nemotron-Nano-9B-v2 | FlipQuant | -5.66 ± 0.82* | -5.56 ± 0.84* | -4.11 ± 0.91* |
| Qwen3.8-27B | Random | -0.25 ± 0.71 | -0.02 ± 0.68 | +0.09 ± 0.77 |
| Qwen3.8-27B | Activation-weighted | -1.38 ± 0.81* | -0.29 ± 0.91 | -0.87 ± 0.87 |
| Qwen3.8-27B | One-shot gradient | +44.69 ± 3.20* | +25.25 ± 2.47* | -1.96 ± 1.45* |
| Qwen3.8-27B | FlipQuant | -7.76 ± 1.16* | -6.68 ± 1.07* | -4.43 ± 0.96* |

## Against the published 256-window table (part G)

Each cell's ΔNLL vs FourOverSix (x 1e-3) at 1k, then (in parentheses) at 256, WikiText-2 / C4. The 256 values are the paper's (their FourOverSix reference is part F's record).

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | Random | -0.7 (-0.2) / +0.4 (+0.2) | -0.3 (-0.6) / +0.8 (+0.3) | +0.7 (-0.6) / +1.3 (+0.4) |
| Nemotron-Nano-9B-v2 | Activation-weighted | -1.3 (-2.0) / +0.5 (+0.5) | -0.8 (-0.2) / -0.1 (+0.3) | +2.8 (+2.4) / +2.1 (+1.6) |
| Nemotron-Nano-9B-v2 | One-shot gradient | +89.9 (+22.0) / +72.5 (+21.2) | +56.8 (+12.6) / +47.5 (+13.4) | -2.1 (-5.9) / +1.5 (-1.8) |
| Nemotron-Nano-9B-v2 | FlipQuant | -12.9 (-8.6) / -5.7 (-5.2) | -11.9 (-8.7) / -5.6 (-5.1) | -9.6 (-5.5) / -4.1 (-3.1) |
| Qwen3.8-27B | Random | +2.1 (+0.3) / -0.3 (-0.7) | -2.8 (+0.4) / -0.0 (-0.1) | +4.2 (+3.3) / +0.1 (-0.1) |
| Qwen3.8-27B | Activation-weighted | -8.3 (-3.6) / -1.4 (-1.1) | -8.3 (+0.6) / -0.3 (-0.3) | -3.9 (-2.2) / -0.9 (-0.4) |
| Qwen3.8-27B | One-shot gradient | +3.9 (-10.6) / +44.7 (+10.2) | -14.5 (-8.1) / +25.3 (+4.8) | -29.9 (-13.8) / -2.0 (-1.2) |
| Qwen3.8-27B | FlipQuant | -35.4 (-22.5) / -7.8 (-4.5) | -31.1 (-16.1) / -6.7 (-4.8) | -17.7 (-11.4) / -4.4 (-2.8) |

## PPL per cell (WikiText-2 / C4)

| model | map | 8x64 | 16x64 | 256x64 |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | FourOverSix | 8.4260 / 11.5034 | 8.4260 / 11.5034 | 8.4260 / 11.5034 |
| Nemotron-Nano-9B-v2 | Random | 8.4200 / 11.5076 | 8.4233 / 11.5125 | 8.4320 / 11.5188 |
| Nemotron-Nano-9B-v2 | Activation-weighted | 8.4154 / 11.5092 | 8.4192 / 11.5026 | 8.4494 / 11.5279 |
| Nemotron-Nano-9B-v2 | One-shot gradient | 9.2186 / 12.3689 | 8.9185 / 12.0628 | 8.4086 / 11.5205 |
| Nemotron-Nano-9B-v2 | FlipQuant | 8.3176 / 11.4385 | 8.3267 / 11.4397 | 8.3453 / 11.4562 |
| Qwen3.8-27B | FourOverSix | 7.3023 / 10.1903 | 7.3023 / 10.1903 | 7.3023 / 10.1903 |
| Qwen3.8-27B | Random | 7.3178 / 10.1877 | 7.2817 / 10.1901 | 7.3330 / 10.1912 |
| Qwen3.8-27B | Activation-weighted | 7.2420 / 10.1763 | 7.2422 / 10.1874 | 7.2738 / 10.1815 |
| Qwen3.8-27B | One-shot gradient | 7.3310 / 10.6560 | 7.1971 / 10.4509 | 7.0875 / 10.1704 |
| Qwen3.8-27B | FlipQuant | 7.0482 / 10.1115 | 7.0784 / 10.1224 | 7.1741 / 10.1453 |

## FlipQuant vs each baseline (paired ΔNLL FlipQuant − baseline, x 1e-3, ± 2 SE; negative = FlipQuant better; WikiText-2; C4)

| model | unit | vs Random | vs Activation-weighted | vs One-shot |
|---|---|---:|---:|---:|
| Nemotron-Nano-9B-v2 | 8x64 | -12.24 ± 1.12*; -6.03 ± 0.76* | -11.69 ± 1.23*; -6.17 ± 0.88* | -102.85 ± 4.52*; -78.21 ± 4.03* |
| Nemotron-Nano-9B-v2 | 16x64 | -11.54 ± 1.15*; -6.34 ± 0.81* | -11.05 ± 1.48*; -5.49 ± 0.96* | -68.66 ± 3.18*; -53.04 ± 2.79* |
| Nemotron-Nano-9B-v2 | 256x64 | -10.34 ± 1.13*; -5.45 ± 0.81* | -12.40 ± 1.51*; -6.23 ± 1.00* | -7.56 ± 1.48*; -5.59 ± 1.01* |
| Qwen3.8-27B | 8x64 | -37.54 ± 5.20*; -7.51 ± 1.09* | -27.13 ± 4.47*; -6.39 ± 1.15* | -39.34 ± 4.05*; -52.46 ± 3.24* |
| Qwen3.8-27B | 16x64 | -28.32 ± 3.81*; -6.66 ± 0.94* | -22.88 ± 4.23*; -6.39 ± 1.15* | -16.63 ± 4.05*; -31.93 ± 2.47* |
| Qwen3.8-27B | 256x64 | -21.90 ± 4.61*; -4.52 ± 0.87* | -13.79 ± 3.84*; -3.56 ± 0.98* | +12.15 ± 4.08*; -2.47 ± 1.24* |

## The maps' records

| model | unit | act windows | act layers short of positives | one-shot steps | one-shot fit windows | one-shot layers short of negatives |
|---|---|---:|---:|---:|---:|---:|
| Nemotron-Nano-9B-v2 | 8x64 | 1024 | 0 | 128 | 1024 | 0 |
| Nemotron-Nano-9B-v2 | 16x64 | 1024 | 0 | 128 | 1024 | 0 |
| Nemotron-Nano-9B-v2 | 256x64 | 1024 | 27 | 128 | 1024 | 0 |
| Qwen3.8-27B | 8x64 | 1024 | 0 | 128 | 1024 | 0 |
| Qwen3.8-27B | 16x64 | 1024 | 1 | 128 | 1024 | 0 |
| Qwen3.8-27B | 256x64 | 1024 | 43 | 128 | 1024 | 0 |

## The re-measured rows against the paper's records

FourOverSix: part F's record (= part AA's re-measurement, per window); FlipQuant: part AA's 1k maps' records. Reported, not gated. exact: equal per window bit for bit; equivalent: the paired ΔNLL within 2 SE on both corpora; differs: otherwise.

| model | row | tier | max abs ΔNLL | PPL now (paper) WikiText-2 / C4 |
|---|---|---|---:|---|
| Nemotron-Nano-9B-v2 | FourOverSix | exact | 0 | 8.4260 (8.4260) / 11.5034 (11.5034) |
| Nemotron-Nano-9B-v2 | FlipQuant 8x64 | exact | 0 | 8.3176 (8.3176) / 11.4385 (11.4385) |
| Nemotron-Nano-9B-v2 | FlipQuant 16x64 | exact | 0 | 8.3267 (8.3267) / 11.4397 (11.4397) |
| Nemotron-Nano-9B-v2 | FlipQuant 256x64 | exact | 0 | 8.3453 (8.3453) / 11.4562 (11.4562) |
| Qwen3.8-27B | FourOverSix | exact | 0 | 7.3023 (7.3023) / 10.1903 (10.1903) |
| Qwen3.8-27B | FlipQuant 8x64 | exact | 0 | 7.0482 (7.0482) / 10.1115 (10.1115) |
| Qwen3.8-27B | FlipQuant 16x64 | exact | 0 | 7.0784 (7.0784) / 10.1224 (10.1224) |
| Qwen3.8-27B | FlipQuant 256x64 | exact | 0 | 7.1741 (7.1741) / 10.1453 (10.1453) |
