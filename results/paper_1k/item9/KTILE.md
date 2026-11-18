# Item 9: Tables D.2 (`tab:app-ktile`) and D.3 (`tab:app-c2`), RTX PRO 6000 rows, at the 1k calibration

4096³, one session of 10 repeats (each the median of 3 rotated rounds), all configurations interleaved; iso_cold (512 MiB flush and rotated cold weights per call) and b2b_cold (20 back-to-back calls over the rotating copies); CUPTI kernel times; mean ± SD over the repeats. Tags: layer 0 o_proj of Mistral-7B's release maps (step 0) and 1k maps. Run from flipquant alone (`python -m experiments.paper_1k.run item9`).

Gates: builds' SASS True; census True; per-MMA == per-K-tile True. The seven extra builds' SASS gate (item9-sass) is the first check of those builds from flipquant's sources: they could not be built on the machine where item 9 was ported (its GPU driver was hung). There, on the CPU, the re-vendored generator gave byte-identical blobs for all 47 existing configurations, the per-MMA blobs equalled kernel-opt's build_KT, and the ported census reproduced amendment 20's.

## Table D.2 with the release-map tags (step 0)

| tile | branch | µs | TFLOP/s | Iso. | B2B | MMAs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| stock NVFP4 | | 118.48 ± 0.42 | 1160.0 | -- | -- | 32 | |
| 8x64 | per MMA | 184.50 ± 0.53 | 745.0 | +55.7 ± 0.5 % | +69.9 ± 0.4 % | 64 |  |
| 8x64 | per K-tile | 125.20 ± 0.62 | 1097.7 | +5.7 ± 0.5 % | +7.0 ± 0.1 % | 32 | +1.474 ± 0.003x |
| 16x64 | per MMA | 177.59 ± 0.33 | 773.9 | +49.9 ± 0.4 % | +64.4 ± 0.4 % | 64 |  |
| 16x64 | per K-tile | 119.59 ± 0.39 | 1149.2 | +0.9 ± 0.2 % | +3.5 ± 0.2 % | 32 | +1.485 ± 0.002x |

With all tiles E2M1 (iso_cold): 8x64 per MMA +55.8 ± 0.3 %, per K-tile +5.4 ± 0.2 %; 16x64 per MMA +49.9 ± 0.2 %, per K-tile +0.5 ± 0.1 %.

## Table D.3 with the release-map tags (step 0) (iso_cold, %)

| part | 8x64 | 16x64 | 256x64 |
|---|---:|---:|---:|
| *Before optimization* | | | |
| Tiles | +3.6 ± 0.2 | +1.1 ± 0.2 | +1.1 ± 0.2 |
| Tag copies | +0.9 ± 0.1 | +0.6 ± 0.2 | +0.6 ± 0.2 |
| Dispatch | +3.9 ± 0.3 | +1.8 ± 0.3 | +1.8 ± 0.3 |
| E0M3 tiles | +0.2 ± 0.5 | +0.3 ± 0.1 | +0.3 ± 0.4 |
| Total | +8.7 ± 0.5 | +3.9 ± 0.2 | +3.9 ± 0.3 |
| *After optimization (deployed)* | | | |
| Tiles | +3.6 ± 0.2 | +0.1 ± 0.4 | -0.0 ± 0.4 |
| Tag copies | -- | -- | -- |
| Dispatch | +1.7 ± 0.3 | +0.5 ± 0.4 | +0.9 ± 0.4 |
| E0M3 tiles | +0.3 ± 0.5 | +0.4 ± 0.3 | -0.0 ± 0.4 |
| Total | +5.7 ± 0.5 | +0.9 ± 0.2 | +0.9 ± 0.4 |

All tiles E0M3 against all E2M1 (iso_cold, points): before 8x64 +1.4 ± 0.6, 16x64 +4.2 ± 0.4, 256x64 +4.2 ± 0.4; after 8x64 +6.6 ± 0.6, 16x64 +4.9 ± 0.4, 256x64 +0.2 ± 0.4.

## Table D.2 with the 1k-map tags

| tile | branch | µs | TFLOP/s | Iso. | B2B | MMAs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| stock NVFP4 | | 118.48 ± 0.42 | 1160.0 | -- | -- | 32 | |
| 8x64 | per MMA | 184.58 ± 0.52 | 744.6 | +55.8 ± 0.6 % | +69.9 ± 0.4 % | 64 |  |
| 8x64 | per K-tile | 125.99 ± 0.56 | 1090.9 | +6.3 ± 0.4 % | +7.8 ± 0.3 % | 32 | +1.465 ± 0.003x |
| 16x64 | per MMA | 177.48 ± 0.45 | 774.4 | +49.8 ± 0.4 % | +64.3 ± 0.4 % | 64 |  |
| 16x64 | per K-tile | 119.96 ± 0.55 | 1145.7 | +1.2 ± 0.3 % | +3.8 ± 0.1 % | 32 | +1.480 ± 0.003x |

With all tiles E2M1 (iso_cold): 8x64 per MMA +55.8 ± 0.3 %, per K-tile +5.4 ± 0.2 %; 16x64 per MMA +49.9 ± 0.2 %, per K-tile +0.5 ± 0.1 %.

## Table D.3 with the 1k-map tags (iso_cold, %)

| part | 8x64 | 16x64 | 256x64 |
|---|---:|---:|---:|
| *Before optimization* | | | |
| Tiles | +3.6 ± 0.2 | +1.1 ± 0.2 | +1.1 ± 0.2 |
| Tag copies | +0.9 ± 0.1 | +0.6 ± 0.2 | +0.6 ± 0.2 |
| Dispatch | +3.9 ± 0.3 | +1.8 ± 0.3 | +1.8 ± 0.3 |
| E0M3 tiles | +0.6 ± 0.6 | +0.3 ± 0.4 | +0.9 ± 0.4 |
| Total | +9.2 ± 0.5 | +3.9 ± 0.4 | +4.5 ± 0.3 |
| *After optimization (deployed)* | | | |
| Tiles | +3.6 ± 0.2 | +0.1 ± 0.4 | -0.0 ± 0.4 |
| Tag copies | -- | -- | -- |
| Dispatch | +1.7 ± 0.3 | +0.5 ± 0.4 | +0.9 ± 0.4 |
| E0M3 tiles | +0.9 ± 0.5 | +0.7 ± 0.3 | +0.1 ± 0.5 |
| Total | +6.3 ± 0.4 | +1.2 ± 0.3 | +1.0 ± 0.5 |

All tiles E0M3 against all E2M1 (iso_cold, points): before 8x64 +1.4 ± 0.6, 16x64 +4.2 ± 0.4, 256x64 +4.2 ± 0.4; after 8x64 +6.6 ± 0.6, 16x64 +4.9 ± 0.4, 256x64 +0.2 ± 0.4.

## Step 0: the release-tag half against the paper's values (another RTX PRO 6000 board)

Stock: 118.48 µs here, 106.58 µs in the paper (+11.90 µs).

| quantity | offset (points; speedup: x) |
|---|---:|
| D.2 8x64 permma iso | -14.786 |
| D.2 8x64 ktile iso | -1.428 |
| D.2 8x64 speedup | -0.118 |
| D.2 16x64 permma iso | -14.512 |
| D.2 16x64 ktile iso | -0.465 |
| D.2 16x64 speedup | -0.137 |
| D.3 before 8x64 tiles | -1.14 |
| D.3 before 8x64 tag_copies | +0.00 |
| D.3 before 8x64 dispatch | -0.00 |
| D.3 before 8x64 e0m3 | -0.15 |
| D.3 before 8x64 total | -1.37 |
| D.3 before 16x64 tiles | -0.07 |
| D.3 before 16x64 tag_copies | -0.35 |
| D.3 before 16x64 dispatch | -0.32 |
| D.3 before 16x64 e0m3 | -0.02 |
| D.3 before 16x64 total | -0.82 |
| D.3 before 256x64 tiles | -0.07 |
| D.3 before 256x64 tag_copies | -0.35 |
| D.3 before 256x64 dispatch | -0.32 |
| D.3 before 256x64 e0m3 | -0.04 |
| D.3 before 256x64 total | -0.84 |
| D.3 after 8x64 tiles | -1.24 |
| D.3 after 8x64 dispatch | +0.04 |
| D.3 after 8x64 e0m3 | -0.10 |
| D.3 after 8x64 total | -1.43 |
| D.3 after 16x64 tiles | -0.24 |
| D.3 after 16x64 dispatch | -0.33 |
| D.3 after 16x64 e0m3 | +0.10 |
| D.3 after 16x64 total | -0.47 |
| D.3 after 256x64 tiles | -0.32 |
| D.3 after 256x64 dispatch | +0.04 |
| D.3 after 256x64 e0m3 | -0.10 |
| D.3 after 256x64 total | -0.39 |

