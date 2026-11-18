"""Comparisons of a run's numbers with the paper's records, for runs on another board (results/paper_1k/RUNBOOK.md,
"Comparisons across boards").

A run on another GPU board, driver or software build need not reproduce the paper's floating-point numbers bit for bit.
Gates that compare a run's GPU numerics with a record therefore have two passing tiers:

* ``exact``: every value equal, bit for bit;
* ``equivalent``: not equal, but per corpus the paired ΔNLL (run − record, per window) is within 2 SE;
* ``differs``: otherwise (a failed gate).

Board-independent quantities stay exact gates (digests of map tiles, GPTQ codes, readback rows, installed bytes).
"""
from __future__ import annotations

import math

CORPORA = ("wiki", "c4")


def paired(x, ref) -> dict:
    """Paired ΔNLL x − ref per window: the mean, 2 SE, n (nats/token)."""
    if len(x) != len(ref):
        raise ValueError(f"{len(x)} windows against {len(ref)}")
    d = [a - b for a, b in zip(x, ref)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) if n > 1 else float("nan")
    return dict(delta=m, two_se=2 * se, n=n)


def tier(now: dict, ref: dict, corpora=CORPORA) -> dict:
    """``now`` / ``ref``: {corpus: [per-window NLL]}. The tier, the largest per-window difference, the paired ΔNLL."""
    cells = {c: paired(now[c], ref[c]) for c in corpora}
    max_abs = max(abs(a - b) for c in corpora for a, b in zip(now[c], ref[c]))
    if all(now[c] == ref[c] for c in corpora):
        t = "exact"
    elif all(abs(cells[c]["delta"]) <= cells[c]["two_se"] for c in corpora):
        t = "equivalent"
    else:
        t = "differs"
    return dict(tier=t, max_abs=max_abs, paired=cells)


def passed(*tiers: dict) -> bool:
    return all(t["tier"] in ("exact", "equivalent") for t in tiers)


def overall(*tiers: dict) -> str:
    """PASS-exact, PASS-equivalent or FAIL over several comparisons."""
    if not passed(*tiers):
        return "FAIL"
    return "PASS-exact" if all(t["tier"] == "exact" for t in tiers) else "PASS-equivalent"


def accuracy_tier(acc: float, published_pct: float, n: int) -> dict:
    """An accuracy against a published value whose per-question records are not available (e.g. MMLU at 256): exact
    when it rounds to the published two decimals; equivalent when within 2 SE of it (the binomial SE of the published
    accuracy over n questions, the paper's convention for accuracies; unpaired, so looser than a paired test);
    differs otherwise."""
    p = published_pct / 100
    two_se = 2 * math.sqrt(p * (1 - p) / n)
    if round(100 * acc, 2) == published_pct:
        t = "exact"
    elif abs(acc - p) <= two_se:
        t = "equivalent"
    else:
        t = "differs"
    return dict(tier=t, accuracy_pct=100 * acc, published_pct=published_pct, delta_pct=100 * (acc - p),
                two_se_pct=100 * two_se, delta_questions=round((acc - p) * n, 1), n=n)
