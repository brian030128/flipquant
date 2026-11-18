# FlipQuant maps

One bit per weight tile (True = E0M3, False = E2M1 / FourOverSix) at the paper's three tile sizes: `8x64`, `16x64` and
`256x64`. The files use the `flipquant-map/1` format: `flipquant.maps.load(path)` returns `(tiles, unit, meta)`, and
every command takes `--map maps/<model>/flipquant_<unit>.pt`. `flipquant_sha256.json` lists each file's sha256, and
`tests/test_stage3.py` checks them.

**These are earlier, smaller calibrations, not the paper's final maps.** The paper calibrates on 1,024 windows for
5 epochs with a top-1,000 teacher (Section 5.1). Its maps are the public `edgeai-lab/<Base>-FlipQuant-Calib1k`
repositories, pinned with their sha256 in `experiments/paper_1k/artifacts.json` and fetched by
`python -m experiments.paper_1k.setup fetch`. The 256-window release maps (`edgeai-lab/<Base>-FlipQuant`) that the
reruns' step 0s check against are pinned there too. The maps here are small, self-contained inputs for the tests and
for quick runs. Each file's `meta` records its trainer, settings, model revision, E0M3 share and calibration time.

| model | files | trained with |
|---|---|---|
| `mistral-7b` (Mistral-7B-Instruct-v0.3) | 8x64, 16x64, 256x64 | `calibration.train_map`, 128 windows, 20 epochs |
| `phi4-14b` (Phi-4) | 8x64, 16x64, 256x64 | `calibration.train_map`, 128 windows, 20 epochs |
| `nemotron-nano-9b-v2` | 8x64, 16x64, 256x64 | `calibration.train_map --fit-windows 256 --epochs 5 --teacher-topk 1000` |
| `qwen3.8-27b` | 8x64, 16x64, 256x64 | `calibration.train_map --fit-windows 256 --epochs 5 --teacher-topk 1000` |

The Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct maps are only on the Hugging Face Hub (the Llama licence).
