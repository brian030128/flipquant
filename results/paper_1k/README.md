# The paper's reruns at the 1,024-window calibration (NVFP4-RaZeR paper-eval part AC)

The plan, with every item's scale, estimate and status: [PLAN.md](PLAN.md). How to run the remaining items from this repository alone: [RUNBOOK.md](RUNBOOK.md) (setup and per-item steps) and [PORTING.md](PORTING.md) (what was ported, and from where).

**Ready** means `python -m experiments.paper_1k.run <item>` works in this checkout. Run only the items marked ready, in the order of this table. When an item is not ready yet, skip it, run the next ready one, and come back after a `git pull`.

| item | paper item | ready | status | report (paper-eval) | report (flipquant) |
|---|---|---|---|---|---|
| [2](item2/README.md) | Table 4 (`tab:latency`) SM120 rows; the memory paragraph | ready | latency done (33bad82), memory done | `f60666e` | `bc903d2` |
| [3](item3/README.md) | Table 5 (`tab:ablation`) | ready | done |  | `7b30554` |
| [4](item4/README.md) | Table F.3 (`tab:ptq`) | ready | done |  | `140cd5b` |
| [5](item5/README.md) | Table D.5 (`tab:app-gemm`) | ready | done |  | `33a5653` |
| [6](item6/README.md) | Table A.2 (`tab:app-native`) | ready | done |  | `9460576` |
| [7](item7/README.md) | Appendix D.6 (reading the weights back) | ready | done |  | `a76a282` |
| [8](item8/README.md) | Appendix D.4 (real tags vs random placement) | ready | done |  | `756cea2` |
| [9](item9/README.md) | Tables D.2 / D.3 (`tab:app-ktile`, `tab:app-c2`), RTX PRO 6000 rows | ready | done |  | `7027a8a` |
| [11](item11/README.md) | Tables C.2 / C.3 (calibration cost) | cancelled by the user (2026-10-26); filled from part AA records | cancelled by the user (2026-10-26); filled from part AA records |  | `9c65e7c` |
| [12](item12/README.md) | Table C.5 (`tab:app-robust`) | ready | done |  | `eaded15` |
| [13](item13/README.md) | Appendix F.1 (the tile-gradient check) | ready | done |  | `2ffefa6` |
| [1](item1/README.md) | MMLU column (Table 3, Tables E.1–E.3) | ready | done |  | `472409b` |
