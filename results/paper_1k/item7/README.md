# Item 7: Appendix D.6 (reading the weights back)

Status: **done**.

## Summary (2026-10-26)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000: 25 jobs (step 0, 18 readbacks, 6 controls), all rc 0, 11 min.
- **Step 0 (gate `item7-step0`): pass.** Part C1's readback of Mistral-7B's 16x64 release map: the per-module rows
  equal C1's (224 modules, exact, 0 format mismatches).
- **The readback of the 18 1k maps:** 181,876,162,560 weight elements checked, exact, 0 format mismatches (the paper at
  256: 181.9 billion, exact, 0); every module of every map (112 to 496 per model).
- **The negative control (all six models, 16x64):** the check passes on the map's packing and detects both wrong
  pairs, with 356,723 to 2,580,457 mismatching elements per module.

Files: `READBACK.md`, `item7.json`.
