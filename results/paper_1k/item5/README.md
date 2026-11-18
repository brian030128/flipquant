# Item 5: Table D.5 (`tab:app-gemm`)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an idle RTX PRO 6000 (500 W, driver 595.71.05): Phi-4 step 0 (phase A, on
  the idle host) and six `evaluation.gemm_time` processes with the 1k maps, all rc 0 (27 min). gemm_time's own checks
  held (bitwise launches, expected kernel sets, 30 launches per block, no other compute process).
- **Step 0 (gate `item5-step0`): pass.** Phi-4 with the release maps, median over T >= 128 (typical tags):
  8x64 +6.24 % (published +6.72, −0.48 points), 16x64 +0.97 % (+0.75, +0.22), 256x64 +1.28 % (+0.91, +0.37); the
  criterion is within 2 points (another board).
- **T >= 128 (six-model median, typical tags; densest in brackets):** 8x64 +7.27 % (+8.09), 16x64 +1.93 % (+2.82),
  256x64 +1.39 % (+1.50). The published 256-window values: +7.24, +1.20, +1.20 (another board). The 16x64 overhead
  grows at 1k, where its maps flip more tiles.
- **Below T = 128:** the largest median cell is 8x64 at T = 32, +2.45 % (densest +3.33); the largest 16x64 cell below
  T = 128 is +1.50 % (T = 64), the largest 256x64 cell +0.39 % (T = 32).

Files: `GEMM.md`, `item5.json`.
