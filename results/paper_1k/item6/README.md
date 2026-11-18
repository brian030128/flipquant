# Item 6: Table A.2 (`tab:app-native`)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000 (500 W): 32 jobs (30 cells native and simulated, plus step 0),
  all rc 0, 68 min; native and simulated on this GPU, in one env per model.
- **Step 0 (gate `item6-step0`): PASS-exact.** Mistral-7B 16x64 with the release map: the simulated run equals part C2's
  record and the native run the release evaluation's, per window bit for bit, so the cell equals the published
  −0.16 ± 0.94 / +0.66 ± 1.24.
- **Agreement:** 29 of 30 cells have |native − simulated| within 2 SE on both corpora (the paper at 256: 28 of 30). The
  one beyond: Nemotron-Nano-9B-v2 8x64 on WikiText-2, −1.04 ± 1.03 x 1e-3 (its C4: +0.36 ± 0.67). The largest |ΔNLL|
  is that cell's 1.04 x 1e-3.
- **The native runs against part AA's 1k records:** exact (per window bit for bit) in all 15 cells.

Files: `NATIVE.md`, `item6.json`, `table_app_native.tex`.
