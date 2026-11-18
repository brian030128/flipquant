# Item 13: Appendix F.1 (the tile-gradient check)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000: the 256 one-shot, the smoke and the 256 run (phase A), the
  1k one-shot and the 1k run (47 min), all rc 0. `calibration/gradcheck.py` (the re-implementation) ran on a GPU for
  the first time here.
- **Step 0 (gate `item13-step0`): pass, every criterion.**
  1. The run's own checks hold (the fit windows are the one-shot run's, the held-out windows disjoint, the steps as
     expected).
  2. The in-sample all-E2M1 KL equals the trainer's epoch-0 KL bit for bit (published: 2.7e-9 relative).
  3. The release map's measured change: held-out −17.51 ± 1.68, in-sample −26.06 ± 1.95 x 1e-3 (published
     −17.8 ± 1.7, −26.5 ± 2.0).
  4. g_u's Spearman correlation: held-out +0.73 [+0.41, +0.90], in-sample +0.91 [+0.75, +0.97] (published +0.71
     [+0.37, +0.92], +0.90 [+0.73, +0.98]).
  - Reported against the published run: groups beyond 2 SE 6 / 10 (published 6 / 12), all with g_u's sign; the map's
    tiles under g_u: 19,129 in the top 75,906, median rank 3.2 %, 130 with g_u > 0 (published 21 %, 5.1 %, 942).
- **The 1k run** (the 1k map, 1,024 in-sample windows, a NEW held-out set `2048:448:128`, disjoint):
  - g_u's Spearman: held-out +0.72 [+0.40, +0.90], in-sample +0.88 [+0.69, +0.97]; the activation-weighted score
    −0.43 / −0.46 and the weight error −0.27 / −0.19 (intervals spanning or below 0).
  - Groups beyond 2 SE: held-out 10, in-sample 11, all with g_u's sign.
  - Top-k flips: the measured / predicted ratio falls from 0.79 / 0.73 at k = 10 to 0.47 / 0.51 at k = 1,000; from about
    30,000 tiles on, flipping by g_u alone stops helping (+2.9 ± 1.9 held-out), and from 100,000 it makes the KL
    clearly worse (+24.5 ± 2.5).
  - The 1k map (359,840 tiles): predicted −53.65; measured held-out −22.46 ± 1.66 (ratio 0.42), in-sample
    −33.23 ± 1.09 (ratio 0.62). Its tiles under g_u: 91,729 in the top 359,840, median rank 13.5 %, 25,979 with
    g_u > 0.

Files: `GRADCHECK.md`, `item13.json`.
