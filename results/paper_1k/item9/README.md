# Item 9: Tables D.2 / D.3 (`tab:app-ktile`, `tab:app-c2`), RTX PRO 6000 rows

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an idle RTX PRO 6000 (500 W, driver 595.71.05): the seven extra builds, the
  census, the per-MMA check and one session of 10 repeats (74 s), all rc 0.
- **Gates:** `item9-sass` pass: the seven extra builds' patched and unpatched SASS equal their references, the first
  check of these builds from flipquant's sources. `item9-census` pass. `item9-check` pass: 10 of 10 per-MMA outputs
  equal their per-K-tile builds' bit for bit, plus the timed operands.
- **Table D.2 (iso_cold, % over stock; speedup per-MMA / per-K-tile):**
  - Release tags (step 0): 8x64 per MMA +55.7 %, per K-tile +5.7 %, 1.474x; 16x64 +49.9 % / +0.9 %, 1.485x.
  - 1k tags: 8x64 +55.8 % / +6.3 %, 1.465x; 16x64 +49.8 % / +1.2 %, 1.480x.
- **Table D.3 total (iso_cold, %), after optimization:**
  - Release tags: 8x64 +5.7, 16x64 +0.9, 256x64 +0.9.
  - 1k tags: +6.3, +1.2, +1.0.
  - Before optimization: release +8.7 / +3.9 / +3.9, 1k +9.2 / +3.9 / +4.5. The 1k tags' extra cost is in the
    "E0M3 tiles" part.
- **Step 0 against the paper's values (another RTX PRO 6000 board; reported, not gated):** the stock kernel takes
  118.48 µs here against 106.58 µs, and the per-K-tile kernels 125.20 / 119.59 µs against 114.13 / 108.08 µs, while
  the per-MMA kernels nearly match (184.50 / 177.59 against 181.74 / 175.25 µs). So the speedups are lower here
  (1.474 / 1.485x against 1.592 / 1.622x) and the per-MMA overheads 14.8 / 14.5 points lower. On this board they agree
  with kernel-opt amendment 20's measurement (1.47 / 1.45x). The offset falls on the tensor-saturated kernels
  (stock and per-K-tile, about 1,100–1,160 TFLOP/s); the session does not record clocks, so its cause (e.g. the
  500 W limit's clocks under full tensor load) is not established. Within this session, 1k against release tags
  is the item's result.

Files: `KTILE.md`, `item9.json`.
