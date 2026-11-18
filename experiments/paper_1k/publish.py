"""Publish an item's report: results/paper_1k/<item>/ (written by `run <item> --report`) and the index row.

    python -m experiments.paper_1k.publish <item> --status "TEXT" [--summary FILE] [--no-commit]
    python -m experiments.paper_1k.publish --index      # only regenerate (and scan) the README index, e.g. after a
                                                        # port stage changes which items are ready; nothing committed

1. Scans every file of results/paper_1k/<item>/ and the index: no tokens (Hugging Face / GitHub patterns, and this
   machine's own tokens), no local absolute or home paths, no host names, IP addresses or e-mail addresses. Any hit
   stops before anything is committed; the file and the rule are printed, never the text.
2. Writes results/paper_1k/<item>/README.md (the summary file's text, the status, the file list), updates
   results/paper_1k/index.json and regenerates results/paper_1k/README.md (the PLAN.md / PORTING.md / RUNBOOK.md links,
   and per item its paper item, readiness, status and report commit).
3. Commits with `git add -f` (.gitignore holds results/*), then records that commit in the index in a second commit.
   Pushing is the runner's step: `git push origin paper-1k-reruns` (never main).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

from . import config as C
from .run import CANCELLED, ORDER, READY

DST = C.REPO / "results" / "paper_1k"
#: per item: the paper item and its status before it is published
PAPER_ITEM = {
    "item2": "Table 4 (`tab:latency`) SM120 rows; the memory paragraph",
    "item3": "Table 5 (`tab:ablation`)",
    "item4": "Table F.3 (`tab:ptq`)",
    "item5": "Table D.5 (`tab:app-gemm`)",
    "item6": "Table A.2 (`tab:app-native`)",
    "item7": "Appendix D.6 (reading the weights back)",
    "item8": "Appendix D.4 (real tags vs random placement)",
    "item9": "Tables D.2 / D.3 (`tab:app-ktile`, `tab:app-c2`), RTX PRO 6000 rows",
    "item11": "Tables C.2 / C.3 (calibration cost)",
    "item12": "Table C.5 (`tab:app-robust`)",
    "item13": "Appendix F.1 (the tile-gradient check)",
    "item1": "MMLU column (Table 3, Tables E.1–E.3)",
}
RULES = [("absolute path", rb"(?<![A-Za-z0-9_.~-])/(?:home|vault|tmp|root|mnt|scratch|data|work|opt|share1|Users)[/-]"),
         ("home path", rb"(?<![A-Za-z0-9_])~/(?!\.cache/huggingface)"),
         ("host name", re.escape(socket.gethostname()).encode()),
         ("ip address", rb"(?<![=\d.])\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),
         ("token pattern", rb"(?:hf_|ghp_|gho_|github_pat_|sk-ant-)[A-Za-z0-9_]{20,}"),
         ("e-mail", rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")]


def own_tokens() -> list[bytes]:
    """This machine's Hugging Face tokens (in-process only, never printed)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    out = set()
    for p in (home / "token", home / "stored_tokens"):
        if p.is_file():
            out |= {m.group(0) for m in re.finditer(rb"hf_[A-Za-z0-9]{20,}", p.read_bytes())}
    return sorted(out)


def scan(paths) -> list[tuple[str, str]]:
    toks = own_tokens()
    hits = []
    for p in paths:
        data = Path(p).read_bytes()
        hits += [(str(p), n) for n, rx in RULES if re.search(rx, data)]
        if any(t in data or t[:15] in data for t in toks):
            hits.append((str(p), "this machine's token"))
    return hits


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True, check=True).stdout.strip()


def index_md(idx: dict, current: str | None) -> str:
    rows = ["# The paper's reruns at the 1,024-window calibration (NVFP4-RaZeR paper-eval part AC)", "",
            "The plan, with every item's scale, estimate and status: [PLAN.md](PLAN.md). How to run the remaining items "
            "from this repository alone: [RUNBOOK.md](RUNBOOK.md) (setup and per-item steps) and "
            "[PORTING.md](PORTING.md) (what was ported, and from where).", "",
            "**Ready** means `python -m experiments.paper_1k.run <item>` works in this checkout. Run only the items marked "
            "ready, in the order of this table. When an item is not ready yet, skip it, run the next ready one, and come "
            "back after a `git pull`.", "",
            "| item | paper item | ready | status | report (paper-eval) | report (flipquant) |", "|---|---|---|---|---|---|"]
    for k in ORDER:
        v = idx.get(k, {})
        ready = CANCELLED[k] if k in CANCELLED else "ready" if k in READY else "not ready yet"
        fq = v.get("flipquant") or ("this commit" if k == current else "")
        link = f"[{k[4:]}]({k}/README.md)" if (DST / k / "README.md").exists() or k == current else k[4:]
        rows.append(f"| {link} | {PAPER_ITEM[k]} | {ready} | {v.get('status', 'pending')} | "
                    f"{('`' + v['paper_eval'] + '`') if v.get('paper_eval') else ''} | {('`' + fq + '`') if fq and fq != 'this commit' else fq} |")
    return "\n".join(rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item", nargs="?", choices=ORDER)
    ap.add_argument("--status", default=None)
    ap.add_argument("--summary", type=Path, default=None, help="a Markdown summary for the item's README")
    ap.add_argument("--no-commit", action="store_true", help="write and scan only")
    ap.add_argument("--index", action="store_true", help="only regenerate the README index (no item, no commit)")
    a = ap.parse_args(argv)
    idx_path = DST / "index.json"
    if a.index:
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        (DST / "README.md").write_text(index_md(idx, None))
        hits = scan([DST / "README.md"])
        if hits:
            sys.exit("SCAN FAILED: " + "; ".join(f"{f}: {r}" for f, r in hits))
        print(f"index written: {DST / 'README.md'} (ready: {', '.join(k for k in ORDER if k in READY)})")
        return
    if not a.item or not a.status:
        ap.error("an item and --status are required (or --index)")
    d = DST / a.item
    files = sorted(p for p in d.glob("*") if p.is_file() and p.name != "README.md")
    if not files:
        sys.exit(f"nothing in {d}: run `python -m experiments.paper_1k.run {a.item} --report` first")
    readme = [f"# Item {a.item[4:]}: {PAPER_ITEM[a.item]}", "", f"Status: **{a.status}**.", ""]
    if a.summary:
        readme += [a.summary.read_text().rstrip("\n"), ""]
    readme += ["Files: " + ", ".join(f"`{p.name}`" for p in files) + "."]
    (d / "README.md").write_text("\n".join(readme) + "\n")
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx.setdefault(a.item, {}).update(status=a.status, flipquant=None)
    idx_path.write_text(json.dumps(idx, indent=1) + "\n")
    (DST / "README.md").write_text(index_md(idx, a.item))
    written = files + [d / "README.md", idx_path, DST / "README.md"]
    hits = scan(written)
    if hits:
        sys.exit("SCAN FAILED, nothing committed: " + "; ".join(f"{f}: {r}" for f, r in hits))
    if a.no_commit:
        print(f"written and scanned: {len(written)} files")
        return
    trailer = os.environ.get("PAPER1K_COMMIT_TRAILER", "")
    git("add", "-f", *[str(p.relative_to(C.REPO)) for p in written])
    git("commit", "-q", "-m", f"results/paper_1k/{a.item}: {PAPER_ITEM[a.item]} ({a.status})" + (f"\n\n{trailer}" if trailer else ""))
    h = git("rev-parse", "--short=7", "HEAD")
    idx[a.item]["flipquant"] = h
    idx_path.write_text(json.dumps(idx, indent=1) + "\n")
    (DST / "README.md").write_text(index_md(idx, None))
    git("add", "-f", str(idx_path.relative_to(C.REPO)), str((DST / "README.md").relative_to(C.REPO)))
    git("commit", "-q", "-m", f"results/paper_1k: the index names {a.item}'s report commit ({h})" + (f"\n\n{trailer}" if trailer else ""))
    print(f"committed {h} and the index; push with: git push origin paper-1k-reruns")


if __name__ == "__main__":
    main()
