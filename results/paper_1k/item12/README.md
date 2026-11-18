# Item 12: Table C.5 (`tab:app-robust`)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000: the step 0 recalibration, three audits, six 1k calibrations
  and nine native PPL jobs, all rc 0 (step 0 and the audits in phase A; the rest 1 h 32 min).
- **Step 0 (gate `item12-step0`): pass.** A seed-0 recalibration at 256 windows reproduces the release map bit for bit
  (75,906 E0M3 tiles).
- **Audits (gate `item12-audit`): pass for all three sources.** FineWeb-Edu, UltraChat and the WikiText-2 training
  split: 1,024 windows each, no 13-gram shared with any evaluation text, and the positive control finds every splice.
  The trained windows are the audited ones.
- **Table C.5 at 1k (paired ΔNLL vs the 1k paper map, x 1e-3, WikiText-2 / C4; 256 values in brackets):**
  - Seed 1: +0.44 / −0.02 (+0.45 / −0.37). Seed 2: +0.15 / +0.38 (−0.94 / −0.40). Non-deterministic kernels:
    −0.30 / −0.26 (+0.60 / −0.38). All within 2 SE, as at 256.
  - FineWeb-Edu: +0.28 / −0.39, within 2 SE (at 256 worse: +2.58* / +1.87*).
  - UltraChat: +2.30* / +0.80 (+3.47* / +2.40*).
  - WikiText-2 train: −6.08* / +0.48 (−3.18* / +0.46): better on its own domain only.
- **E0M3 share:** 5.28 % for the paper map (1.11 % at 256), 5.28–5.87 % for the six rows. **Jaccard with the paper
  map:** seeds 0.344 / 0.346, non-deterministic 0.382 (0.40–0.50 at 256); the other sources 0.050–0.053.
- **FourOverSix vs the 1k paper map:** +12.81* / +11.39* (at 256: +9.7 / +9.0). Every row stays far closer to the
  paper map than FourOverSix.

Files: `ROBUST.md`, `item12.json`.
