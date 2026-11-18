# Item 3: Table 5 (`tab:ablation`)

Status: **done**.

## Summary (2026-10-25)

- **Run:** from a fresh flipquant clone on an RTX PRO 6000 (500 W): 79 jobs, all rc 0, 2 h 53 min.
- **Step 0: pass.** The 1k smokes of both models pass (2 steps, 1,024 fit windows, the 896 extension windows equal
  the paper's 1k set, Qwen3.8-27B's teacher on the host). Gate `item3-rebuild`: part G's Nemotron 16x64 one-shot map
  rebuilt bit for bit (120 modules, 80,098 E0M3 tiles); the summed-gradient file's sha256 is equal too.
- **Re-measured rows:** FourOverSix and the three 1k FlipQuant maps of both models equal the paper's records per window
  bit for bit (exact).
- **Headline:** FlipQuant beats random, activation-weighted and one-shot maps in every (model, unit) cell on both
  corpora beyond 2 SE, except Qwen3.8-27B 256x64 on WikiText-2, where the one-shot map is better (+12.15 ± 4.08 x 1e-3;
  at 256 the one-shot map was better there too, −13.8 vs −11.4). FlipQuant vs FourOverSix at 1k (WikiText-2 / C4,
  x 1e-3): Nemotron −12.9 / −5.7, −11.9 / −5.6, −9.6 / −4.1; Qwen3.8-27B −35.4 / −7.8, −31.1 / −6.7, −17.7 / −4.4
  (8x64 / 16x64 / 256x64).
- **Rows that moved against the 256 table:** the one-shot maps at 8x64 and 16x64 are much worse at 1k (Nemotron
  WikiText-2 +89.9 vs +22.0 and +56.8 vs +12.6; Qwen3.8-27B C4 +44.7 vs +10.2 and +25.3 vs +4.8), while Qwen3.8-27B's
  256x64 one-shot map improves (WikiText-2 −29.9 vs −13.8). Random and activation-weighted rows stay within a few
  1e-3 of their 256 values; FlipQuant improves in every cell.

Files: `ABLATION.md`, `item3.json`, `table_ablation_c4.tex`, `table_ablation_wiki.tex`.
