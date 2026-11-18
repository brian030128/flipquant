# Item 8: Appendix D.4 (real tags vs random placement)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an idle RTX PRO 6000 (500 W): random maps and `evaluation.gemm_time` for the
  release (256) and 1k maps of Mistral-7B-Instruct-v0.3, all rc 0; gemm_time's bitwise checks 468 of 468 equal in each.
- **Step 0 (gate `item8-step0`): pass** (on the idle host; a scratch smoke that shared the host with an artifact fetch
  had failed with two single cells at 1.84 / 1.28 points). Worst typical cell 0.86 points; ranges 8x64 −0.50..+0.86,
  16x64 −0.74..+0.46, 256x64 −0.76..+0.69 (published −0.53..+0.43, −0.37..+0.94, −0.28..+0.43 on another board).
- **The 1k maps (typical module):** ranges 8x64 −1.09..+0.77, 16x64 −1.06..+0.95, 256x64 −0.67..+0.67 points. 106 of
  108 cells lie within 1 point of random placement at the same per-module share. The two beyond are o_proj at
  T = 2048 for 8x64 (−1.09) and 16x64 (−1.06), both with the real tags faster than random. Densest module: worst
  1.75 points.
- **The 1k maps flip more tiles** (typical module, E0M3 granules): 8x64 1,625 / 3,577 / 4,331, 16x64 1,089 / 2,432 /
  2,835, 256x64 3,136 / 7,136 / 8,336 (o / gate / down), against 358 / 588 / 749, 233 / 387 / 495, 688 / 960 /
  1,088 for the release maps.

Files: `REAL_VS_RANDOM.md`, `item8.json`.
