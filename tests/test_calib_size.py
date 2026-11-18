"""The calibration-size study's options (NVFP4-RaZeR paper-eval part AA), on the CPU.

* ``calibration/tmopt_launch.py --teacher-host``: the top-K teacher's rows are copied to the host and are bit for bit the
  GPU rows; the vendored trainer reads the rows only through ``topk_teacher.batch_rows``, which moves them to the
  logits' device, so the loss sees the same values. ``calibration.train_map --teacher-host`` routes the trainer
  through the launcher (with ``--register-model`` for a runtime model); refused without ``--teacher-topk``.
* ``--probe-steps``: ``calibration.train_map`` passes it to the trainer.
* The extension rule (``calibration/tmopt/fit_extension.py``) is nested: the N-window extension starts, per source,
  with every smaller one's windows.
* ``calibration.fit_windows``'s output is a ``--calib-extension`` for ``flipquant.data.calibration_release``.
The GPU check, that ``--teacher-host`` gives the GPU-teacher map bit for bit, is a calibration run (part AA's step 0).
"""
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from calibration import tmopt_common as RC
from calibration import train_map as TM

ENGINE = RC.ENGINE


def _vendored(name):
    """A vendored module, imported from calibration/tmopt as the trainer imports it."""
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    return __import__(name)


def _logprobs(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 4).log_softmax(-1)


def test_teacher_host_rows_are_the_teacher_rows():
    from calibration import tmopt_launch as L
    T = _vendored("topk_teacher")
    fake = types.SimpleNamespace(topk_rows=T.topk_rows)
    L.install_teacher_host(fake)
    lp = _logprobs(1, 9, 300)
    (v0, i0, t0), m0 = T.topk_rows(lp, 50)
    (v1, i1, t1), m1 = fake.topk_rows(lp, 50)
    assert m0 == m1 and all(x.device.type == "cpu" for x in (v1, i1, t1))
    assert (v1.dtype, i1.dtype, t1.dtype) == (torch.bfloat16, torch.int32, torch.float32)
    assert torch.equal(v0.view(torch.int16), v1.view(torch.int16)) and torch.equal(i0, i1) and torch.equal(t0, t1)
    rows = [fake.topk_rows(_logprobs(1, 9, 300, seed=s), 50)[0] for s in range(3)]
    ref = [T.topk_rows(_logprobs(1, 9, 300, seed=s), 50)[0] for s in range(3)]
    for a, b in zip(T.batch_rows(rows, "cpu"), T.batch_rows(ref, "cpu")):
        assert torch.equal(a, b)
    assert T.topk_rows is not fake.topk_rows                       # the module itself is not touched here


def test_trainer_reads_teacher_rows_only_through_batch_rows():
    """The hook's premise: run_train_map.py makes the top-K rows with topk_teacher.topk_rows and hands them to the
    losses, which read them only through topk_teacher.batch_rows (chunked_loss.train_kl_gradient_topk too)."""
    trainer = (ENGINE / "run_train_map.py").read_text()
    assert trainer.count("topk_teacher.topk_rows(") == 1 and "import topk_teacher" in trainer
    uses = [line.strip() for line in trainer.splitlines() if re.search(r"\bteacher\[", line)]
    allowed = ("chunked_loss.train_kl_gradient_topk(", "chunked_loss.train_kl_gradient(", "topk_teacher.batch_rows(",
               "torch.cat([teacher[i] for i in idx])")       # the last: the full-vocabulary teacher (--teacher-topk 0)
    assert uses and all(any(a in u for a in allowed) for u in uses), uses
    chunked = (ENGINE / "chunked_loss.py").read_text()
    body = chunked[chunked.index("def train_kl_gradient_topk"):]
    body = body[:body.index("\ndef ") if "\ndef " in body else len(body)]
    assert [l.strip() for l in body.splitlines() if re.search(r"\bteacher\[", l)] == [
        "values, idx, tail = topk_teacher.batch_rows(teacher[d0:d1], device)"]


def test_command_routes_teacher_host_and_probe():
    base = TM.command("phi4", "16x64", Path("/root"), [], Path("/out"), 5, teacher_topk=1000, fit_windows=1024)
    assert base[1] == "run_train_map.py" and "--probe-steps" not in base
    cmd = TM.command("phi4", "16x64", Path("/root"), [], Path("/out"), 5, teacher_topk=1000, fit_windows=1024,
                     teacher_host=True, probe_steps=3)
    sep = cmd.index("--")
    assert str(cmd[1]).endswith("calibration/tmopt_launch.py") and cmd[2:sep] == ["--teacher-host"]
    assert cmd[sep + 1:] == base[1:-2] + ["--probe-steps", "3", "--out", Path("/out")]
    one = TM.command("llama1b_ins", "16x64", Path("/root"), [], Path("/out"), 5, teacher_topk=1000, fit_windows=1024,
                     teacher_host=True)
    assert one[2:one.index("--")] == ["--register-model", "llama1b_ins", "--teacher-host"]
    plain = TM.command("llama1b_ins", "16x64", Path("/root"), [], Path("/out"), 5, teacher_topk=1000)
    assert plain[2:plain.index("--")] == ["--register-model", "llama1b_ins"]


def test_teacher_host_needs_topk(tmp_path):
    from calibration import tmopt_launch as L
    with pytest.raises(SystemExit, match="--teacher-topk"):
        TM.main(["--model", "phi4-14b", "--unit", "16x64", "--out", str(tmp_path / "x"), "--teacher-host"])
    with pytest.raises(SystemExit, match="--teacher-topk"):
        L._parse(["--teacher-host", "--", "run_train_map.py", "--model", "phi4", "--unit", "16x64"])
    ext, trainer = L._parse(["--teacher-host", "--", "run_train_map.py", "--model", "phi4", "--teacher-topk", "1000"])
    assert ext.teacher_host and trainer == ["--model", "phi4", "--teacher-topk", "1000"]
    with pytest.raises(SystemExit, match="probe"):
        TM.main(["--model", "phi4-14b", "--unit", "16x64", "--out", str(tmp_path / "y"), "--probe-steps", "0"])
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


class _Tok:
    """Token ids = one id per character (so a document's token count is its length)."""

    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 251 for c in text]]))


def _stream(source, n=400, seed=5):
    g = torch.Generator().manual_seed(seed)
    field = {"math": "text", "code": "content"}[source]
    rows = []
    for i in range(n):
        length = int(torch.randint(300, 1400, (1,), generator=g))           # some below 512 tokens: skipped
        rows.append({field: f"{source}{i:04d}" + "x" * length})
    return rows


def test_extension_rule_is_nested(monkeypatch):
    FE = _vendored("fit_extension")
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, **kw: _stream(repo))
    prior = {"fit": {s: {"repo": s, "revision": "r", "path": "p"} for s in ("math", "code")}}
    skip = {"fit": set(), "development": {__import__("hashlib").sha256(_stream("math")[3]["text"].encode()).hexdigest()},
            "c4_evaluation": set()}
    tok = _Tok()
    sets = {n: FE.extend(tok, prior, n, skip) for n in (256, 400, 600)}
    for small, big in ((256, 400), (400, 600), (256, 600)):
        (ws, es), (wb, eb) = sets[small], sets[big]
        for source in ("math", "code"):
            rs = [r for r in es["records"] if r["source"] == source]
            rb = [r for r in eb["records"] if r["source"] == source]
            assert len(rs) == small // 2 - 64 and rb[:len(rs)] == rs
        assert {r["token_sha256"] for r in es["records"]} <= {r["token_sha256"] for r in eb["records"]}
    assert all(r["document_sha256"] not in skip["development"] for r in sets[600][1]["records"])


def test_fit_windows_output_is_a_calib_extension(monkeypatch, tmp_path):
    """calibration.fit_windows writes the trainer report's fit_extension; flipquant.data.calibration_release reads it
    as the release set's extension (the 128 of the record, then the extension, every token hash checked)."""
    from flipquant import data as D
    monkeypatch.setenv("FLIPQUANT_CACHE", str(tmp_path / "cache"))
    win = {f"h{i}": torch.full((1, 512), i) for i in range(300)}
    sha = {id(w): h for h, w in win.items()}
    monkeypatch.setattr(D, "sha", lambda w: sha[id(w)])
    base = [f"h{i}" for i in range(128)]
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"fit": {"math": {"token_sha256": base[:64]}, "code": {"token_sha256": base[64:]}}}))
    monkeypatch.setattr(D, "calibration_from_record", lambda tok, rec: {"fit": [win[h] for h in base]})
    rows = [dict(source="math" if i < 236 else "code", document_sha256=f"d{i}", offset=0, token_sha256=f"h{i}")
            for i in range(128, 300)]
    monkeypatch.setattr(D, "_rebuild_windows", lambda tok, rs: [win[r["token_sha256"]] for r in rs])
    out = tmp_path / "windows_300.json"
    out.write_text(json.dumps(dict(fit_windows=300, fit_extension=dict(windows=len(rows), rule="rule", records=rows),
                                   token_sha256=base + [r["token_sha256"] for r in rows])))
    tok = SimpleNamespace(name_or_path="tok", get_vocab=lambda: {"a": 0})
    rel = D.calibration_release(tok, record, out)
    assert len(rel["fit"]) == 300 and rel["meta"]["token_sha256"] == json.loads(out.read_text())["token_sha256"]
    assert rel["meta"]["extension_windows"] == 172 and rel["meta"]["base_windows"] == 128


def test_save_theta_records_adam_parameters(tmp_path, monkeypatch):
    """tmopt_launch --save-theta: the Adam subclass remembers the trainer's parameters and leaves the step alone; the
    file holds them by module name after the last step."""
    from calibration import tmopt_launch as L
    monkeypatch.setattr(torch.optim, "Adam", torch.optim.Adam)       # restored after the test
    state = L.install_theta_capture()
    a, b = torch.nn.Parameter(torch.tensor([-1.0, 0.5])), torch.nn.Parameter(torch.tensor([[0.25]]))
    ref_a, ref_b = a.detach().clone(), b.detach().clone()
    opt = torch.optim.Adam([a, b], lr=0.02, eps=1e-12)
    ref = torch.optim.Adam.__mro__[1]([torch.nn.Parameter(ref_a.clone()), torch.nn.Parameter(ref_b.clone())], lr=0.02,
                                      eps=1e-12)
    for o, ps in ((opt, (a, b)), (ref, ref.param_groups[0]["params"])):
        ps[0].grad, ps[1].grad = torch.tensor([0.3, -0.2]), torch.tensor([[0.1]])
        o.step()
    assert state["params"][0] is a and state["params"][1] is b
    assert torch.equal(a.detach(), ref.param_groups[0]["params"][0].detach())      # the base class's step
    L.save_theta(state, ["m.a", "m.b"], tmp_path / "theta.pt")
    saved = torch.load(tmp_path / "theta.pt")
    assert list(saved) == ["m.a", "m.b"] and torch.equal(saved["m.a"], a.detach()) and torch.equal(saved["m.b"], b.detach())
    with pytest.raises(SystemExit):
        L.save_theta(state, ["m.a"], tmp_path / "bad.pt")
