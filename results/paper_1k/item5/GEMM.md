# Item 5: Table D.5 (`tab:app-gemm`) at the 1k calibration

Per-forward GEMM overhead of the deployed kernels over the stock kernel (`evaluation.gemm_time`, its defaults: CUPTI time, cold weights, 3 rotated rounds of 30 calls), with each projection's tags from the 1k maps; `evaluation.gemm_time_report`'s tables below. Run from flipquant alone (`python -m experiments.paper_1k.run item5`).

## Step 0

Phi-4 with the 256-window release maps, against the published run (another RTX PRO 6000 board): the median over the T >= 128 cells, typical tags, % — 8x64: +6.24 (published +6.72, -0.48 points); 16x64: +0.97 (published +0.75, +0.22 points); 256x64: +1.28 (published +0.91, +0.37 points). Criterion: within 2.0 points per tile: pass.

## The six models with the 1k maps

Models: llama3.2-1b-instruct, llama3.1-8b-instruct, mistral-7b, nemotron-nano-9b-v2, phi4-14b, qwen3.8-27b. Each cell: 100 x (map on its set / FourOverSix on stock_ko - 1), %, median over the models; typical module tags, densest in parentheses. Last row: the median over all (model, T) cells with T >= 128.

GPU per record: llama3.2-1b-instruct: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677.0 MHz (min 2595), sw power-cap samples 0; llama3.1-8b-instruct: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677 MHz (min 2467), sw power-cap samples 30; mistral-7b: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677 MHz (min 2452), sw power-cap samples 30; nemotron-nano-9b-v2: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677 MHz (min 2445), sw power-cap samples 55; phi4-14b: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2752 MHz (min 2302), sw power-cap samples 50; qwen3.8-27b: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2685.0 MHz (min 2460), sw power-cap samples 60.

| T | 8x64 | 16x64 | 256x64 |
|---:|---:|---:|---:|
| 1 | +0.65 (+0.80) | +0.58 (+0.94) | +0.12 (+0.25) |
| 4 | +0.12 (+0.39) | +0.44 (+0.81) | +0.04 (+0.04) |
| 16 | +0.96 (+1.48) | +0.94 (+1.12) | +0.20 (+0.30) |
| 32 | +2.45 (+3.33) | +1.15 (+1.43) | +0.39 (+0.38) |
| 64 | +1.29 (+1.55) | +1.50 (+2.17) | +0.32 (+0.39) |
| 128 | +2.18 (+2.92) | +1.67 (+2.25) | +0.53 (+0.54) |
| 256 | +6.12 (+6.99) | +2.83 (+4.13) | +1.28 (+1.27) |
| 512 | +9.90 (+11.28) | +2.84 (+4.02) | +1.59 (+1.91) |
| 1024 | +6.33 (+7.46) | +1.29 (+2.17) | +1.05 (+1.20) |
| 2048 | +7.34 (+8.21) | +1.57 (+2.46) | +1.56 (+1.70) |
| 4096 | +7.61 (+8.58) | +1.80 (+2.81) | +1.45 (+1.70) |
| 8192 | +8.16 (+9.13) | +1.88 (+2.73) | +1.79 (+1.88) |
| median, T >= 128 | +7.27 (+8.09) | +1.93 (+2.82) | +1.39 (+1.50) |

## Per model

### llama3.2-1b-instruct

7 projections, 112 NativeLinears timed per forward (q_proj 2048x2048 x16, k_proj 512x2048 x16, v_proj 512x2048 x16, o_proj 2048x2048 x16, gate_proj 8192x2048 x16, up_proj 8192x2048 x16, down_proj 2048x8192 x16). Maps: 8x64 c45684a0abb3, 16x64 8ba6e70703d9, 256x64 309f115a041c. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677.0 MHz. Linear modules not on the NVFP4 path (excluded): lm_head.

| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |
|---:|---:|---:|---:|---:|---:|
| 1 | 808.4 | +0.98 (+1.11) | +1.52 (+2.73) | +0.10 (+0.23) | +73.50 |
| 4 | 799.7 | -0.35 (+0.22) | +2.98 (+3.49) | -0.00 (+0.19) | +74.10 |
| 16 | 787.4 | +1.40 (+2.21) | +3.58 (+4.62) | +0.00 (-0.06) | +74.12 |
| 32 | 809.0 | +4.68 (+5.57) | +8.16 (+9.49) | +0.70 (+0.76) | +66.33 |
| 64 | 921.6 | -2.05 (-1.00) | +9.70 (+11.58) | +1.61 (+1.67) | +45.03 |
| 128 | 1,353.7 | +14.09 (+14.75) | +3.71 (+5.11) | +0.72 (+0.64) | -0.79 |
| 256 | 1,374.4 | +13.32 (+13.99) | +4.53 (+5.68) | +0.50 (+0.47) | -1.21 |
| 512 | 1,626.1 | +12.63 (+13.29) | +4.09 (+5.64) | +0.66 (+0.71) | -0.60 |
| 1024 | 1,923.6 | +13.35 (+14.11) | +3.23 (+4.55) | +0.52 (+0.67) | -0.74 |
| 2048 | 3,980.3 | +14.18 (+15.10) | +4.66 (+5.51) | +2.31 (+2.56) | -0.24 |
| 4096 | 6,726.4 | +9.53 (+10.11) | +2.82 (+3.57) | +1.21 (+1.16) | -0.05 |
| 8192 | 13,322.2 | +10.22 (+10.77) | +3.38 (+4.35) | +1.96 (+1.91) | -0.12 |
| median, T >= 128 | | +13.32 (+13.99) | +3.71 (+5.11) | +0.72 (+0.71) | |

### llama3.1-8b-instruct

7 projections, 224 NativeLinears timed per forward (q_proj 4096x4096 x32, k_proj 1024x4096 x32, v_proj 1024x4096 x32, o_proj 4096x4096 x32, gate_proj 14336x4096 x32, up_proj 14336x4096 x32, down_proj 4096x14336 x32). Maps: 8x64 989ef5ec3d99, 16x64 c973cb98c23b, 256x64 8c2fb731fdf5. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677 MHz. Linear modules not on the NVFP4 path (excluded): lm_head.

| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,295.7 | +2.30 (+3.08) | +0.68 (+1.10) | +0.36 (+0.33) | +64.70 |
| 4 | 3,289.5 | +1.74 (+2.43) | +0.33 (+1.07) | +0.19 (+0.02) | +70.86 |
| 16 | 3,261.9 | +2.20 (+3.03) | +0.97 (+1.52) | +0.42 (+0.30) | +66.82 |
| 32 | 3,260.3 | +3.94 (+4.90) | +1.19 (+1.85) | +0.52 (+0.50) | +61.61 |
| 64 | 3,314.6 | +3.31 (+3.74) | +1.48 (+2.29) | +0.50 (+0.40) | +49.29 |
| 128 | 3,502.5 | +2.10 (+2.75) | +2.63 (+3.14) | +0.79 (+0.83) | +41.63 |
| 256 | 4,239.8 | +5.32 (+6.35) | +3.71 (+5.31) | +1.40 (+1.68) | +34.61 |
| 512 | 6,933.4 | +12.07 (+13.94) | +3.66 (+4.53) | +2.47 (+2.55) | +5.42 |
| 1024 | 13,325.0 | +5.94 (+6.99) | +0.84 (+2.09) | +0.88 (+1.12) | +2.12 |
| 2048 | 23,306.7 | +7.75 (+9.04) | +1.93 (+2.59) | +1.39 (+1.73) | -0.08 |
| 4096 | 45,340.6 | +8.85 (+10.09) | +2.16 (+3.33) | +2.19 (+2.28) | +0.24 |
| 8192 | 87,203.8 | +8.47 (+9.77) | +1.95 (+3.00) | +1.93 (+2.06) | -0.07 |
| median, T >= 128 | | +7.75 (+9.04) | +2.16 (+3.14) | +1.40 (+1.73) | |

### mistral-7b

7 projections, 224 NativeLinears timed per forward (q_proj 4096x4096 x32, k_proj 1024x4096 x32, v_proj 1024x4096 x32, o_proj 4096x4096 x32, gate_proj 14336x4096 x32, up_proj 14336x4096 x32, down_proj 4096x14336 x32). Maps: 8x64 e3fa4365e2f3, 16x64 8bb3d25d704d, 256x64 dbec75c2fc97. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677 MHz. Linear modules not on the NVFP4 path (excluded): lm_head.

| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,287.5 | +2.20 (+3.49) | +0.48 (+0.78) | +0.13 (+0.27) | +70.49 |
| 4 | 3,277.2 | +1.58 (+2.70) | +0.50 (+0.77) | -0.20 (-0.03) | +71.58 |
| 16 | 3,262.8 | +2.06 (+3.28) | +1.04 (+1.35) | +0.27 (+0.36) | +66.94 |
| 32 | 3,261.3 | +3.61 (+5.10) | +1.27 (+1.62) | +0.24 (+0.38) | +62.10 |
| 64 | 3,317.2 | +2.98 (+3.75) | +1.51 (+2.05) | +0.31 (+0.39) | +49.27 |
| 128 | 3,504.5 | +2.25 (+3.08) | +2.63 (+3.10) | +0.63 (+0.73) | +41.49 |
| 256 | 4,234.7 | +4.72 (+5.68) | +3.93 (+5.36) | +1.39 (+1.34) | +34.65 |
| 512 | 6,921.6 | +11.73 (+13.72) | +3.50 (+4.89) | +2.42 (+2.45) | +5.58 |
| 1024 | 13,376.7 | +5.76 (+6.82) | +0.85 (+2.25) | +1.01 (+0.86) | +1.88 |
| 2048 | 23,405.1 | +7.89 (+8.81) | +2.16 (+2.87) | +1.59 (+1.32) | -0.16 |
| 4096 | 45,569.7 | +7.88 (+8.88) | +2.07 (+3.22) | +2.13 (+2.47) | +0.07 |
| 8192 | 87,390.5 | +8.43 (+9.41) | +1.91 (+2.86) | +1.79 (+1.84) | -0.14 |
| median, T >= 128 | | +7.88 (+8.81) | +2.16 (+3.10) | +1.59 (+1.34) | |

### nemotron-nano-9b-v2

8 projections, 120 NativeLinears timed per forward (in_proj 22656x4480 x27, out_proj 4480x10240 x27, up_proj 15680x4480 x25, down_proj 4480x15680 x25, q_proj 5120x4480 x4, k_proj 1024x4480 x4, v_proj 1024x4480 x4, o_proj 4480x5120 x4). Maps: 8x64 bbfba4371323, 16x64 7469f895ec4d, 256x64 8738a1c478ee. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2677 MHz. Linear modules not on the NVFP4 path (excluded): lm_head.

| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,133.8 | +0.33 (+0.41) | +0.94 (+1.14) | +0.11 (+0.42) | +61.20 |
| 4 | 3,130.1 | +0.03 (+0.34) | +0.88 (+0.85) | +0.08 (+0.26) | +61.12 |
| 16 | 3,128.4 | +0.14 (+0.42) | +0.90 (+0.88) | -0.05 (+0.30) | +52.46 |
| 32 | 3,150.2 | +0.56 (+0.84) | +1.11 (+1.25) | +0.19 (+0.33) | +44.77 |
| 64 | 3,250.0 | -1.31 (-0.94) | +2.00 (+2.68) | +0.33 (+0.51) | +30.32 |
| 128 | 4,241.5 | +4.68 (+5.30) | +0.29 (+1.33) | +0.42 (+0.44) | +0.00 |
| 256 | 4,636.3 | +7.25 (+8.55) | +0.65 (+2.18) | +0.96 (+0.91) | -0.22 |
| 512 | 7,137.9 | +8.06 (+9.27) | +2.17 (+3.52) | +1.79 (+2.06) | -0.05 |
| 1024 | 14,541.3 | +6.15 (+6.88) | +0.91 (+1.80) | +1.09 (+1.28) | -0.09 |
| 2048 | 25,406.7 | +6.73 (+7.62) | +1.20 (+2.34) | +1.52 (+1.74) | -0.04 |
| 4096 | 48,753.3 | +7.35 (+8.28) | +1.52 (+2.40) | +1.45 (+1.67) | -0.00 |
| 8192 | 95,627.8 | +7.89 (+8.85) | +1.86 (+2.60) | +1.79 (+1.92) | +0.11 |
| median, T >= 128 | | +7.25 (+8.28) | +1.20 (+2.34) | +1.45 (+1.67) | |

### phi4-14b

4 projections, 160 NativeLinears timed per forward (o_proj 5120x5120 x40, qkv_proj 7680x5120 x40, gate_up_proj 35840x5120 x40, down_proj 5120x17920 x40). Maps: 8x64 657e80d319da, 16x64 265c1cb113b6, 256x64 d292e9e7f660. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2752 MHz. Linear modules not on the NVFP4 path (excluded): lm_head.

| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,330.5 | +0.20 (+0.49) | -0.10 (-0.02) | -0.14 (+0.12) | +46.99 |
| 4 | 5,324.7 | +0.17 (+0.39) | -0.11 (+0.22) | -0.01 (+0.05) | +51.80 |
| 16 | 5,300.4 | +0.53 (+0.57) | +0.23 (+0.52) | +0.46 (+0.44) | +39.56 |
| 32 | 5,303.6 | +1.15 (+1.29) | +0.59 (+0.69) | +0.37 (+0.37) | +36.72 |
| 64 | 5,370.2 | +1.01 (+1.07) | +0.49 (+0.66) | +0.30 (+0.13) | +23.10 |
| 128 | 5,560.3 | +0.57 (+0.56) | +0.51 (+1.19) | +0.18 (+0.03) | +19.63 |
| 256 | 6,952.9 | +5.78 (+6.40) | +1.96 (+2.21) | +1.47 (+1.50) | +9.38 |
| 512 | 12,045.9 | +7.28 (+7.98) | +1.33 (+1.76) | +0.85 (+1.76) | -0.08 |
| 1024 | 22,952.1 | +7.38 (+7.94) | +1.92 (+2.00) | +1.22 (+1.32) | +0.18 |
| 2048 | 45,344.7 | +6.93 (+7.56) | +1.11 (+1.56) | +1.60 (+1.67) | +0.09 |
| 4096 | 85,806.3 | +7.04 (+7.44) | +1.04 (+1.31) | +0.94 (+1.08) | -0.31 |
| 8192 | 167,481.5 | +7.62 (+8.08) | +1.49 (+1.82) | +1.42 (+1.49) | +0.03 |
| median, T >= 128 | | +7.04 (+7.56) | +1.33 (+1.76) | +1.22 (+1.49) | |

### qwen3.8-27b

12 projections, 496 NativeLinears timed per forward (out_proj 5120x6144 x48, in_proj_qkv 10240x5120 x48, in_proj_z 6144x5120 x48, in_proj_b 48x5120 x48, in_proj_a 48x5120 x48, gate_proj 17408x5120 x64, up_proj 17408x5120 x64, down_proj 5120x17408 x64, q_proj 12288x5120 x16, k_proj 1024x5120 x16, v_proj 1024x5120 x16, o_proj 5120x6144 x16). Maps: 8x64 1a07a2f3ec96, 16x64 63117f22990a, 256x64 73b703adf5a1. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 595.71.05, power limit 500.0 W, busy SM clock median 2685.0 MHz. Linear modules not on the NVFP4 path (excluded): linear_fc1, linear_fc2, lm_head, proj, qkv.

| T | stock_ko µs | 8x64 | 16x64 | 256x64 | stock_wB_ko |
|---:|---:|---:|---:|---:|---:|
| 1 | 10,645.4 | +0.13 (+0.48) | +0.40 (+0.66) | +0.12 (+0.05) | +58.00 |
| 4 | 10,671.8 | +0.06 (+0.39) | +0.38 (+0.64) | +0.13 (+0.03) | +55.31 |
| 16 | 10,618.6 | +0.31 (+0.74) | +0.45 (+0.79) | +0.13 (+0.08) | +51.35 |
| 32 | 10,578.4 | +1.28 (+1.77) | +0.73 (+1.07) | +0.41 (+0.38) | +45.25 |
| 64 | 10,654.1 | +1.58 (+2.03) | +0.93 (+1.25) | +0.24 (+0.13) | +33.79 |
| 128 | 11,040.0 | +1.64 (+2.03) | +0.71 (+1.40) | +0.35 (+0.31) | +29.32 |
| 256 | 13,796.0 | +6.46 (+7.59) | +1.69 (+2.94) | +1.17 (+1.19) | +17.72 |
| 512 | 23,252.7 | +6.25 (+8.00) | +1.93 (+3.03) | +1.39 (+1.71) | +5.39 |
| 1024 | 43,641.2 | +6.52 (+8.11) | +1.68 (+2.77) | +1.56 (+1.66) | +2.21 |
| 2048 | 82,778.9 | +5.57 (+6.99) | +0.96 (+1.90) | +1.18 (+1.38) | +0.54 |
| 4096 | 155,484.1 | +6.30 (+7.61) | +1.30 (+2.21) | +1.46 (+1.73) | +0.26 |
| 8192 | 301,484.1 | +6.85 (+8.13) | +1.19 (+1.97) | +1.31 (+1.55) | +0.12 |
| median, T >= 128 | | +6.30 (+7.61) | +1.30 (+2.21) | +1.31 (+1.55) | |

## Tile-table fallbacks

Calls whose width (``fallback_width``) or scheduler row ((0, 1)) did not come from the deployed tile table (scheduler rows only for the sets whose table has them).

| model | projection | shape | set | fallback | T |
|---|---|---|---|---|---|
| llama3.2-1b-instruct | q_proj | 2048x2048 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | q_proj | 2048x2048 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | q_proj | 2048x2048 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | q_proj | 2048x2048 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | q_proj | 2048x2048 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | k_proj | 512x2048 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | k_proj | 512x2048 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | k_proj | 512x2048 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | k_proj | 512x2048 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | k_proj | 512x2048 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | v_proj | 512x2048 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | v_proj | 512x2048 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | v_proj | 512x2048 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | v_proj | 512x2048 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | v_proj | 512x2048 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | o_proj | 2048x2048 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | o_proj | 2048x2048 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | o_proj | 2048x2048 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | o_proj | 2048x2048 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | o_proj | 2048x2048 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | gate_proj | 8192x2048 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | gate_proj | 8192x2048 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | gate_proj | 8192x2048 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | gate_proj | 8192x2048 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | gate_proj | 8192x2048 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | up_proj | 8192x2048 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | up_proj | 8192x2048 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | up_proj | 8192x2048 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | up_proj | 8192x2048 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | up_proj | 8192x2048 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | down_proj | 2048x8192 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | down_proj | 2048x8192 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | down_proj | 2048x8192 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | down_proj | 2048x8192 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.2-1b-instruct | down_proj | 2048x8192 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | q_proj | 4096x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | k_proj | 1024x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | v_proj | 1024x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | o_proj | 4096x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | gate_proj | 14336x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | up_proj | 14336x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| llama3.1-8b-instruct | down_proj | 4096x14336 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | q_proj | 4096x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | k_proj | 1024x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | v_proj | 1024x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | o_proj | 4096x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | gate_proj | 14336x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | up_proj | 14336x4096 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| mistral-7b | down_proj | 4096x14336 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | in_proj | 22656x4480 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | in_proj | 22656x4480 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | in_proj | 22656x4480 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | in_proj | 22656x4480 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | in_proj | 22656x4480 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | out_proj | 4480x10240 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | out_proj | 4480x10240 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | out_proj | 4480x10240 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | out_proj | 4480x10240 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | out_proj | 4480x10240 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | up_proj | 15680x4480 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | up_proj | 15680x4480 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | up_proj | 15680x4480 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | up_proj | 15680x4480 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | up_proj | 15680x4480 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | down_proj | 4480x15680 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | down_proj | 4480x15680 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | down_proj | 4480x15680 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | down_proj | 4480x15680 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | down_proj | 4480x15680 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | q_proj | 5120x4480 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | q_proj | 5120x4480 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | q_proj | 5120x4480 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | q_proj | 5120x4480 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | q_proj | 5120x4480 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | k_proj | 1024x4480 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | k_proj | 1024x4480 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | k_proj | 1024x4480 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | k_proj | 1024x4480 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | k_proj | 1024x4480 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | v_proj | 1024x4480 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | v_proj | 1024x4480 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | v_proj | 1024x4480 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | v_proj | 1024x4480 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | v_proj | 1024x4480 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | o_proj | 4480x5120 | stock_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | o_proj | 4480x5120 | stock_wB_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | o_proj | 4480x5120 | mixed_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | o_proj | 4480x5120 | mixed_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| nemotron-nano-9b-v2 | o_proj | 4480x5120 | mixed256_ko | width,schedule | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| phi4-14b | o_proj | 5120x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| phi4-14b | qkv_proj | 7680x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| phi4-14b | gate_up_proj | 35840x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| phi4-14b | down_proj | 5120x17920 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | out_proj | 5120x6144 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | in_proj_qkv | 10240x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | in_proj_z | 6144x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | in_proj_b | 48x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | in_proj_a | 48x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | gate_proj | 17408x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | up_proj | 17408x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | down_proj | 5120x17408 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | q_proj | 12288x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | k_proj | 1024x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | v_proj | 1024x5120 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| qwen3.8-27b | o_proj | 5120x6144 | stock_wB_ko | width | 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |

## Against the published table (the 256-window release maps, another board)

Each cell: the six-model median at 1k, then the published value; typical tags.

| T | 8x64 | 16x64 | 256x64 |
|---:|---:|---:|---:|
| 1 | +0.65 (+0.11) | +0.58 (-0.09) | +0.12 (+0.20) |
| 4 | +0.12 (+0.07) | +0.44 (+0.23) | +0.04 (-0.03) |
| 16 | +0.96 (+0.19) | +0.94 (+0.49) | +0.20 (+0.33) |
| 32 | +2.45 (+1.27) | +1.15 (+0.50) | +0.39 (+0.43) |
| 64 | +1.29 (+1.09) | +1.50 (+0.71) | +0.32 (+0.45) |
| 128 | +2.18 (+1.73) | +1.67 (+0.50) | +0.53 (+0.42) |
| 256 | +6.12 (+6.01) | +2.83 (+0.61) | +1.28 (+0.62) |
| 512 | +9.90 (+9.33) | +2.84 (+1.23) | +1.59 (+1.40) |
| 1024 | +6.33 (+7.45) | +1.29 (+1.39) | +1.05 (+1.27) |
| 2048 | +7.34 (+7.30) | +1.57 (+0.78) | +1.56 (+1.26) |
| 4096 | +7.61 (+7.48) | +1.80 (+1.15) | +1.45 (+1.33) |
| 8192 | +8.16 (+7.52) | +1.88 (+1.24) | +1.79 (+1.45) |
| T>=128 | +7.27 (+7.24) | +1.93 (+1.20) | +1.39 (+1.20) |
