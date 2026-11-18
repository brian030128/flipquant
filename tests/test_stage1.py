"""The port, stage 1: train_map_legacy --no-dev / --deterministic, ppl --paper-convention, and
evaluation.significance (CPU).

``test_fit_only_windows`` streams the calibration datasets and needs a tokenizer; it runs only with
``FLIPQUANT_DATA_TESTS=1`` (network or a warm HF cache).
"""
import argparse
import json
import math
import os

import pytest
import torch

from calibration import train_map_legacy as T
from evaluation import ppl as PPL
from evaluation import significance as S


# --- train_map_legacy --no-dev / --deterministic --------------------------------------------------

def _args(**kw):
    base = dict(no_dev=False, ppl_datasets=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_ppl_datasets_default_unchanged():
    assert T.resolve_ppl_datasets(_args()) == "wiki,c4"
    assert T.resolve_ppl_datasets(_args(ppl_datasets="")) == ""
    assert T.resolve_ppl_datasets(_args(ppl_datasets="wiki")) == "wiki"


def test_no_dev_turns_final_ppl_off():
    assert T.resolve_ppl_datasets(_args(no_dev=True)) == ""
    assert T.resolve_ppl_datasets(_args(no_dev=True, ppl_datasets="")) == ""
    with pytest.raises(SystemExit):
        T.resolve_ppl_datasets(_args(no_dev=True, ppl_datasets="wiki,c4"))


def test_deterministic_needs_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert T.deterministic_setup(False) == dict(deterministic=False, cublas_workspace_config=None)
    with pytest.raises(SystemExit):
        T.deterministic_setup(True)


def test_deterministic_turns_on_algorithms(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = torch.are_deterministic_algorithms_enabled()
    try:
        assert T.deterministic_setup(True) == dict(deterministic=True, cublas_workspace_config=":4096:8")
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)


@pytest.mark.skipif(os.environ.get("FLIPQUANT_DATA_TESTS") != "1", reason="streams the calibration datasets")
def test_fit_only_windows():
    """--no-dev's fit windows are exactly the fit windows of the standard set."""
    from transformers import AutoTokenizer
    from flipquant import data as D
    tok = AutoTokenizer.from_pretrained(os.environ.get("FLIPQUANT_TEST_TOKENIZER", "Qwen/Qwen3-1.7B"))
    full, fit_only = D.calibration_sets(tok), D.calibration_fit(tok)
    assert len(fit_only["fit"]) == len(full["fit"]) == 128
    assert all(torch.equal(a, b) for a, b in zip(fit_only["fit"], full["fit"]))
    assert fit_only["meta"]["fit"] == full["meta"]["fit"]


# --- ppl --paper-convention ----------------------------------------------------------------------

def _ppl_args(**kw):
    base = dict(paper_convention=False, act_scope="document", act="fourover6", nvidia_nvfp4=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_paper_convention_off_keeps_default():
    a = _ppl_args()
    PPL.apply_paper_convention(a, [])
    assert a.act_scope == "document" and a.convention is None


def test_paper_convention_sets_per_token():
    a = _ppl_args(paper_convention=True)
    PPL.apply_paper_convention(a, ["--paper-convention"])
    assert a.act_scope == "row" and a.convention == PPL.PAPER_CONVENTION
    a = _ppl_args(paper_convention=True, act_scope="row")
    PPL.apply_paper_convention(a, ["--paper-convention", "--act-scope", "row"])
    assert a.act_scope == "row"


def test_paper_convention_rejects_conflicts():
    with pytest.raises(SystemExit):
        PPL.apply_paper_convention(_ppl_args(paper_convention=True), ["--paper-convention", "--act-scope", "document"])
    with pytest.raises(SystemExit):
        PPL.apply_paper_convention(_ppl_args(paper_convention=True, act="nvfp4"), ["--paper-convention"])
    with pytest.raises(SystemExit):
        PPL.apply_paper_convention(_ppl_args(paper_convention=True, nvidia_nvfp4=True), ["--paper-convention"])


# --- evaluation.significance ---------------------------------------------------------------------

def _report(tmp_path, name, nll, data=None):
    results = {c: dict(ppl=math.exp(sum(v) / len(v)), nll=v, **({"data": data} if data else {})) for c, v in nll.items()}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(results=results)))
    return path


def _series(seed, n, shift):
    g = torch.Generator().manual_seed(seed)
    return (2.0 + shift + 0.3 * torch.randn(n, generator=g, dtype=torch.float64)).tolist()


def test_paired_matches_numpy_definition():
    x, y = _series(1, 50, 0.02), _series(2, 50, 0.0)
    p = S.paired(x, y)
    d = torch.tensor(x, dtype=torch.float64) - torch.tensor(y, dtype=torch.float64)
    assert math.isclose(p["delta"], float(d.mean()), rel_tol=1e-12)
    assert math.isclose(p["se"], float(d.std(unbiased=True) / math.sqrt(len(d))), rel_tol=1e-12)


def test_loss_recovered_anchors(tmp_path):
    n = 40
    base = {c: _series(i, n, 0.0) for i, c in enumerate(("wiki", "c4"))}
    nv = {c: [v + 0.10 for v in base[c]] for c in base}
    ours = {c: [v + 0.04 for v in base[c]] for c in base}
    paths = dict(bf16=_report(tmp_path, "bf16", base), nvfp4=_report(tmp_path, "nvfp4", nv),
                 ours=_report(tmp_path, "ours", ours))
    out = S.main(["--ref", f"nvfp4={paths['nvfp4']}", "--cmp", f"ours={paths['ours']}",
                  "--bf16", str(paths["bf16"]), "--nvfp4", str(paths["nvfp4"])])
    rec = out["loss_recovered"]
    assert math.isclose(rec["bf16"]["recovered"], 1.0, abs_tol=1e-12)
    assert math.isclose(rec["nvfp4"]["recovered"], 0.0, abs_tol=1e-12)
    assert math.isclose(rec["ours"]["recovered"], 0.6, rel_tol=1e-9)       # loss 0.08 of 0.20
    for c in ("wiki", "c4"):
        assert math.isclose(out["paired"]["ours"][c]["delta"], -0.06, rel_tol=1e-9)
        assert out["paired"]["ours"][c]["significant"]


def test_window_mismatch_is_an_error(tmp_path):
    a = _report(tmp_path, "a", {"wiki": _series(1, 10, 0.0)})
    b = _report(tmp_path, "b", {"wiki": _series(2, 11, 0.0)})
    with pytest.raises(SystemExit):
        S.main(["--ref", str(a), "--cmp", str(b)])
    c = _report(tmp_path, "c", {"wiki": _series(1, 10, 0.0)}, data={"documents": [1]})
    d = _report(tmp_path, "d", {"wiki": _series(2, 10, 0.0)}, data={"documents": [2]})
    with pytest.raises(SystemExit):
        S.main(["--ref", str(c), "--cmp", str(d)])
