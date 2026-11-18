# Item 11: Tables C.2 / C.3 (calibration cost)

Status: **cancelled by the user (2026-10-26); filled from part AA records**.

## Summary (2026-10-26)

- **The GPU re-measurement (about 19.5 h) was cancelled by the user.** Tables C.2 / C.3 at 1k are filled from part AA's
  records of the 18 calibrations (six models × 8x64 / 16x64 / 256x64, `train_map --epochs 5 --teacher-topk 1000
  --fit-windows 1024`, one RTX PRO 6000), committed as JSON so that the report regenerates from flipquant alone.
- **Reproduction evidence:** phase A (2026-10-25) re-ran the first calibration (Llama-3.2-1B-Instruct 16x64) from a
  fresh clone: 223 s against part AA's 225 s, the map bit for bit equal to the 1k map.
- **Table C.2 at 16x64, Total (launch to map):** 3.7 min (Llama-3.2-1B-Instruct) to 170.4 min (Nemotron-Nano-9B-v2);
  training 73–93 % of the trainer's time; GPU peak allocated 16.6–90.5 GiB.
- **Table C.3:** the trainer's time differs by at most 0.5–1.9 % across the three tiles; the GPU peak of 8x64 exceeds
  256x64's by 0.04–0.87 GiB.
- **256 → 1k:** setup ×1.45–3.16, of which the teacher precompute ×3.5–4.05 while the candidate packing and the model
  load do not grow; one epoch ×3.92–4.05; the time per optimizer step unchanged (×0.98–1.01).
- **Caveats:** Qwen3.8-27B keeps its teacher on the host (`--teacher-host`; its per-step time is unchanged); the 16x64
  runs are part AA's panel (10/19–20), the 8x64 / 256x64 runs its amendment 6 (10/23–24, concurrent with the HF
  staging; per-step times within 1.6 % of the 16x64 runs). Paper-text points: the teacher takes about 2.9 GiB at 1k,
  not 0.73 GiB; Qwen3.8-27B's teacher on the host contradicts Table C.1's "kept on the GPU" and Section 3's "under
  1 GiB".

Files: `COST.md`, `item11.json`, `table_calib_cost.tex`.
