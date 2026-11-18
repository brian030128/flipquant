# Item 12: Table C.5 (`tab:app-robust`) at the 1k calibration (Mistral-7B, 16x64)

Part AA's 1k calibration with one change per row (flipquant's calib-robustness options), evaluated natively with part AA's PPL command. Paired ΔNLL against the 1k paper map, x 1e-3 nats/token ± 2 SE (* beyond 2 SE; negative is better); E0M3: tiles flipped (%); Jaccard: E0M3 tiles shared with the paper map. In parentheses: the paper's 256-sequence row. Run from flipquant alone (`python -m experiments.paper_1k.run item12`).

## Step 0

A seed-0 recalibration at 256 windows: tiles bit for bit equal to the release map's (75906 E0M3 tiles).

## Table C.5 at 1k

| calibration | WikiText-2 | C4 | E0M3 (%) | Jaccard |
|---|---:|---:|---:|---:|
| Paper's map (1k) | 0 | 0 | 5.28 (1.11) | 1 |
| Seed 1 | +0.44 ± 0.95 (+0.45 ± 0.96) | -0.02 ± 0.77 (-0.37 ± 1.44) | 5.36 (1.08) | 0.344 (0.398) |
| Seed 2 | +0.15 ± 0.98 (-0.94 ± 0.98) | +0.38 ± 0.76 (-0.40 ± 0.93) | 5.28 (1.15) | 0.346 (0.397) |
| Non-deterministic | -0.30 ± 0.93 (+0.60 ± 0.95) | -0.26 ± 0.71 (-0.38 ± 1.29) | 5.30 (1.10) | 0.382 (0.503) |
| Web text (FineWeb-Edu) | +0.28 ± 1.02 (+2.58 ± 1.01*) | -0.39 ± 0.73 (+1.87 ± 1.77*) | 5.47 (1.05) | 0.053 (0.057) |
| Instruction data (UltraChat) | +2.30 ± 1.02* (+3.47 ± 0.96*) | +0.80 ± 0.89 (+2.40 ± 1.07*) | 5.66 (1.09) | 0.050 (0.053) |
| WikiText-2 train | -6.08 ± 1.09* (-3.18 ± 1.10*) | +0.48 ± 1.08 (+0.46 ± 1.35) | 5.87 (1.21) | 0.050 (0.061) |

FourOverSix against the 1k paper map: WikiText-2 +12.81 ± 1.18*; C4 +11.39 ± 1.12* (at 256: +9.7 / +9.0).


## The 13-gram audits of the three fit sources

| source | windows | 13-grams scanned | windows with hits | control (spliced items found) | passed |
|---|---:|---:|---|---|---|
| fineweb-edu | 1024 | 351,689 | none | {'gsm8k_test': 30, 'hellaswag': 30, 'lambada_openai': 30, 'arc_challenge': 30, 'wikitext2_test': 30} of 30 | True |
| ultrachat | 1024 | 362,995 | none | {'gsm8k_test': 30, 'hellaswag': 30, 'lambada_openai': 30, 'arc_challenge': 30, 'wikitext2_test': 30} of 30 | True |
| wikitext2-train | 1024 | 314,982 | none | {'gsm8k_test': 30, 'hellaswag': 30, 'lambada_openai': 30, 'arc_challenge': 30, 'wikitext2_test': 30} of 30 | True |
