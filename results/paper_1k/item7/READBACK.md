# Item 7: Appendix D.6 (reading the weights back) at the 1k calibration

`evaluation.ppl --ownership-check` of the 18 1k maps on this repository's SM120 build (part C1's command): every quantized linear runs through the native kernel on an identity activation, and every decoded weight element must equal its stored value under the map's format. Weight elements: the quantized linears' out x in (the checkpoints' safetensors headers). Informative elements: those whose two formats decode differently (every nibble but ±0). Run from flipquant alone (`python -m experiments.paper_1k.run item7`).

## Step 0

Part C1's readback of Mistral-7B's 16x64 release map: per-module rows equal C1's: True (224 modules, exact True, 0 format mismatches).

## The readback

| model | tile | modules | weight elements checked | informative elements | E0M3 tiles | E0M3 elements observed | exact | format mismatches |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama-3.2-1B-Instruct | 8x64 | 112 of 112 | 973,078,528 | 887,127,842 | 214,432 | 96,467,751 | True | 0 |
| Llama-3.2-1B-Instruct | 16x64 | 112 of 112 | 973,078,528 | 885,843,521 | 141,062 | 126,919,789 | True | 0 |
| Llama-3.2-1B-Instruct | 256x64 | 112 of 112 | 973,078,528 | 876,908,613 | 377,152 | 339,429,748 | True | 0 |
| Llama-3.1-8B-Instruct | 8x64 | 224 of 224 | 6,979,321,856 | 6,383,640,924 | 586,062 | 263,908,693 | True | 0 |
| Llama-3.1-8B-Instruct | 16x64 | 224 of 224 | 6,979,321,856 | 6,380,496,379 | 377,393 | 339,896,643 | True | 0 |
| Llama-3.1-8B-Instruct | 256x64 | 224 of 224 | 6,979,321,856 | 6,356,334,692 | 1,025,456 | 923,631,831 | True | 0 |
| Mistral-7B-Instruct-v0.3 | 8x64 | 224 of 224 | 6,979,321,856 | 6,375,296,063 | 557,943 | 250,659,358 | True | 0 |
| Mistral-7B-Instruct-v0.3 | 16x64 | 224 of 224 | 6,979,321,856 | 6,372,266,980 | 359,840 | 323,462,086 | True | 0 |
| Mistral-7B-Instruct-v0.3 | 256x64 | 224 of 224 | 6,979,321,856 | 6,348,026,914 | 1,009,584 | 907,435,089 | True | 0 |
| Nemotron-Nano-9B-v2 | 8x64 | 120 of 120 | 7,711,621,120 | 7,057,461,648 | 645,487 | 290,361,943 | True | 0 |
| Nemotron-Nano-9B-v2 | 16x64 | 120 of 120 | 7,711,621,120 | 7,053,681,112 | 420,667 | 378,444,194 | True | 0 |
| Nemotron-Nano-9B-v2 | 256x64 | 120 of 120 | 7,711,621,120 | 7,023,466,330 | 1,205,072 | 1,084,278,867 | True | 0 |
| Phi-4 | 8x64 | 160 of 160 | 13,631,488,000 | 12,508,116,012 | 758,874 | 343,094,455 | True | 0 |
| Phi-4 | 16x64 | 160 of 160 | 13,631,488,000 | 12,504,039,104 | 491,665 | 444,569,646 | True | 0 |
| Phi-4 | 256x64 | 160 of 160 | 13,631,488,000 | 12,472,022,403 | 1,374,016 | 1,242,500,901 | True | 0 |
| Qwen3.8-27B | 8x64 | 496 of 496 | 24,350,556,160 | 22,312,125,280 | 1,074,352 | 484,175,835 | True | 0 |
| Qwen3.8-27B | 16x64 | 496 of 496 | 24,350,556,160 | 22,309,601,794 | 602,694 | 543,043,409 | True | 0 |
| Qwen3.8-27B | 256x64 | 496 of 496 | 24,350,556,160 | 22,269,539,075 | 1,676,068 | 1,510,746,831 | True | 0 |

**18 maps, 181,876,162,560 weight elements checked: exact True, 0 format mismatches** (the paper at 256: 181.9 billion elements, exact, 0).

## The negative control

Per model, the first quantized linear with an E0M3 tile in its 16x64 1k map, packed with the map and with an all-E2M1 map, through the same check (`ownership_control.py`). The check must fail on both wrong pairs.

| model | module | map vs map (exact, mismatches) | all-E2M1 deployed vs map: mismatches | map deployed vs all-E2M1: mismatches | detected |
|---|---|---|---:|---:|---|
| Llama-3.2-1B-Instruct | `model.layers.0.self_attn.q_proj` | True, 0 | 374,829 | 356,723 | True |
| Llama-3.1-8B-Instruct | `model.layers.0.self_attn.q_proj` | True, 0 | 523,640 | 477,243 | True |
| Mistral-7B-Instruct-v0.3 | `model.layers.0.self_attn.q_proj` | True, 0 | 532,654 | 495,067 | True |
| Nemotron-Nano-9B-v2 | `model.layers.0.mixer.in_proj` | True, 0 | 2,580,457 | 2,479,547 | True |
| Phi-4 | `model.layers.0.self_attn.o_proj` | True, 0 | 503,209 | 483,630 | True |
| Qwen3.8-27B | `model.language_model.layers.0.linear_attn.out_proj` | True, 0 | 392,967 | 376,455 | True |
