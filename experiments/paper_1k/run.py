"""The paper's reruns at the 1,024-window calibration, from flipquant alone (results/paper_1k/RUNBOOK.md).

    python -m experiments.paper_1k.run <item> [--out DIR] [--data-root DIR]  # run the item (one GPU job at a time)
    python -m experiments.paper_1k.run <item> --dry-run [--out DIR] [--data-root DIR]  # its commands, DIR/dry_run/<item>.txt
    python -m experiments.paper_1k.run <item> --report [--out DIR]    # its report into results/paper_1k/<item>/
    python -m experiments.paper_1k.run --list                          # the items and whether each is ready

Items, in the order to run them: item2 (memory only), item3, item4, item5, item6, item7, item8, item9, item12,
item13, item1. item11 was cancelled by the user (2026-10-26): `--report` fills it from part AA's records. The run directory (--out, default ./paper1k_runs) holds the records, logs/, queue.log and
gpu_state.jsonl; a finished record is never run again, so a stopped item resumes where it stopped.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import config as C
from .jobs import Runner

ORDER = ["item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item11", "item12", "item13", "item1"]
#: items the user cancelled: not run (run / --dry-run refuse); their report is filled from records (--report)
CANCELLED = {"item11": "cancelled by the user (2026-10-26); filled from part AA records"}
#: the order to run the items in
RUN_ORDER = [i for i in ORDER if i not in CANCELLED]
#: the items this checkout can run; the rest are being ported (results/paper_1k/README.md's "ready" column)
READY = set(ORDER)              # every item is ready


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item", nargs="?", choices=ORDER)
    ap.add_argument("--out", type=Path, default=Path("paper1k_runs"))
    ap.add_argument("--data-root", type=Path, default=None,
                    help="the calibration data root of `setup records` (items 3, 4, 11, 12, 13)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.item:
        for it in ORDER:
            print(f"{it:7s} {CANCELLED[it] if it in CANCELLED else 'ready' if it in READY else 'not ready yet'}")
        return
    if a.item not in READY:
        sys.exit(f"{a.item} is not ready yet in this checkout (git pull; results/paper_1k/README.md)")
    if a.item in CANCELLED and not a.report:
        sys.exit(f"{a.item}: {CANCELLED[a.item]}; `--report` writes its report from the records")
    mod = importlib.import_module(f"experiments.paper_1k.{a.item}")
    out = a.out.resolve()
    if a.report:
        dest = C.REPO / "results" / "paper_1k" / a.item
        print(mod.report(out, dest))
        return
    R = Runner(out, a.item, dry_run=a.dry_run)
    if not a.dry_run:
        R.log(f"---- {a.item} (flipquant {C.REPO.name}) ----")
    mod.run(R, out, dry_run=a.dry_run, data_root=a.data_root.resolve() if a.data_root else None)
    if a.dry_run:
        print(f"dry run: {R.dry}")
    else:
        R.log(f"DONE {a.item}")


if __name__ == "__main__":
    main()
