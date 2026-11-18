"""Paired significance of perplexity differences, and "loss recovered", from per-window NLLs.

Reads the JSON reports of ``evaluation.ppl`` (``results[corpus]["nll"]``) or ``calibration.train_map_legacy``
(``evaluation[corpus]["nll"]``), the per-window token-mean NLLs of the same evaluation windows.

* **Paired ΔNLL** (the FlipQuant paper's convention, the paper's ``07_tables.py``): for each
  corpus, the mean over windows of ``nll(cmp) - nll(ref)``, in nats per token, which is exactly
  ``log PPL(cmp) - log PPL(ref)``. ``±`` is 2 SE of the paired differences (sample standard deviation / sqrt(n));
  ``*`` marks ``|Δ| > 2 SE``.
* **Loss recovered** (needs ``--bf16`` and ``--nvfp4``): the quantization loss of a method is its summed log-PPL
  loss over the corpora, ``L(M) = Σ_c [log PPL_c(M) - log PPL_c(BF16)]``, and ``recovered(M) = 1 - L(M) / L(NVFP4)``,
  so NVFP4 is 0 % and BF16 100 %. Its ``±`` is 2 SE of the paired numerator ``L(NVFP4) - L(M)``
  (``sqrt(Σ_c SE_c²)``, corpora independent) divided by ``L(NVFP4)``, which is treated as fixed.

Windows must match: every report must have the same number of windows per corpus and, when recorded, the same
window metadata (``results[corpus]["data"]``). A mismatch is an error, never silently paired.

  python -m evaluation.significance --ref fo6=results/fo6.json --cmp ours=results/mixfp4.json \\
      --bf16 results/bf16.json --nvfp4 results/nvfp4.json --out results/significance.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_nll(path) -> dict:
    """``{corpus: {"nll": [...], "data": meta or None}}`` from an evaluation.ppl or train_map_legacy report."""
    rec = json.loads(Path(path).read_text())
    results = rec.get("results") or rec.get("evaluation") or {}
    out = {c: dict(nll=[float(v) for v in r["nll"]], data=r.get("data")) for c, r in results.items() if r.get("nll")}
    if not out:
        raise SystemExit(f"{path}: no per-window NLLs (results[corpus].nll or evaluation[corpus].nll)")
    return out


def check_same_windows(a: dict, b: dict, corpus: str, names=("a", "b")) -> None:
    if len(a["nll"]) != len(b["nll"]):
        raise SystemExit(f"{corpus}: {names[0]} has {len(a['nll'])} windows, {names[1]} {len(b['nll'])}")
    if a.get("data") is not None and b.get("data") is not None and a["data"] != b["data"]:
        raise SystemExit(f"{corpus}: {names[0]} and {names[1]} were evaluated on different windows")


def paired(x: list[float], y: list[float]) -> dict:
    """Mean and SE of the paired differences ``x - y``."""
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    mean = sum(d) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1) / n) if n > 1 else float("nan")
    return dict(delta=mean, se=se, n=n, significant=abs(mean) > 2 * se)


def log_ppl(nll: list[float]) -> float:
    return sum(nll) / len(nll)


def loss_recovered(method: dict, bf16: dict, nvfp4: dict, corpora) -> dict:
    """``1 - L(M) / L(NVFP4)`` with ``L`` the summed log-PPL loss over ``corpora``, and 2 SE of the paired
    numerator ``L(NVFP4) - L(M)`` over ``L(NVFP4)``."""
    loss_m = sum(log_ppl(method[c]["nll"]) - log_ppl(bf16[c]["nll"]) for c in corpora)
    loss_n = sum(log_ppl(nvfp4[c]["nll"]) - log_ppl(bf16[c]["nll"]) for c in corpora)
    num_se = math.sqrt(sum(paired(nvfp4[c]["nll"], method[c]["nll"])["se"] ** 2 for c in corpora))
    if loss_n <= 0:
        return dict(loss=loss_m, nvfp4_loss=loss_n, recovered=None, two_se=None)
    return dict(loss=loss_m, nvfp4_loss=loss_n, recovered=1 - loss_m / loss_n, two_se=2 * num_se / loss_n)


def labeled(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition("=")
    return (label, Path(path)) if sep else (Path(spec).stem, Path(spec))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="[label=]report.json, the reference of the paired differences")
    ap.add_argument("--cmp", action="append", required=True, help="[label=]report.json (repeatable)")
    ap.add_argument("--bf16", type=Path, default=None, help="BF16 report (for loss recovered)")
    ap.add_argument("--nvfp4", type=Path, default=None, help="NVFP4 report (for loss recovered)")
    ap.add_argument("--corpora", default=None, help="comma list (default: every corpus all reports share)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    ref_label, ref_path = labeled(args.ref)
    runs = {ref_label: load_nll(ref_path)}
    for spec in args.cmp:
        label, path = labeled(spec)
        if label in runs:
            raise SystemExit(f"duplicate label {label!r}")
        runs[label] = load_nll(path)
    anchors = {}
    if (args.bf16 is None) != (args.nvfp4 is None):
        raise SystemExit("loss recovered needs both --bf16 and --nvfp4")
    if args.bf16 is not None:
        anchors = dict(bf16=load_nll(args.bf16), nvfp4=load_nll(args.nvfp4))
    everything = {**runs, **anchors}
    common = set.intersection(*(set(r) for r in everything.values()))
    corpora = args.corpora.split(",") if args.corpora else sorted(common)
    missing = [c for c in corpora if c not in common]
    if missing:
        raise SystemExit(f"corpora {missing} are not in every report")
    for c in corpora:
        for label, r in everything.items():
            check_same_windows(runs[ref_label][c], r[c], c, (ref_label, label))
    out = dict(reference=ref_label, corpora=corpora, ppl={}, paired={}, loss_recovered={})
    for label, r in everything.items():
        out["ppl"][label] = {c: math.exp(log_ppl(r[c]["nll"])) for c in corpora}
    lines = ["| method | " + " | ".join(f"PPL {c}" for c in corpora) + " | "
             + " | ".join(f"ΔNLL {c} vs {ref_label} (± 2 SE)" for c in corpora)
             + (" | loss recovered (± 2 SE) |" if anchors else " |"),
             "|---|" + "---:|" * len(corpora) * 2 + ("---:|" if anchors else "")]
    for label, r in everything.items():
        cells = [f"{out['ppl'][label][c]:.4f}" for c in corpora]
        if label != ref_label:
            out["paired"][label] = {c: paired(r[c]["nll"], runs[ref_label][c]["nll"]) for c in corpora}
            cells += [f"{p['delta']:+.4f} ± {2 * p['se']:.4f}{' *' if p['significant'] else ''}"
                      for p in out["paired"][label].values()]
        else:
            cells += ["—"] * len(corpora)
        if anchors:
            rec = loss_recovered(r, anchors["bf16"], anchors["nvfp4"], corpora)
            out["loss_recovered"][label] = rec
            cells.append("—" if rec["recovered"] is None else f"{100 * rec['recovered']:.1f} % ± {100 * rec['two_se']:.1f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(out, table=text), indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
