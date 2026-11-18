"""calibration.tmopt_ext's opt-in ``--fit-windows N`` / ``--teacher-host`` (NVFP4-RaZeR paper-eval part AC: the paper's
reruns at the 1,024-window calibration), on the CPU. ``tmopt_common.run`` is replaced, so nothing is trained; the tests
check the trainer command that ``run_trainer`` builds.

* With the defaults the command is the release one: the trainer's ``--fit-windows 256``, no ``--teacher-host``.
* ``fit_windows=1024`` changes only the trainer's ``--fit-windows`` value.
* ``teacher_host=True`` adds the launcher's own ``--teacher-host`` hook, before ``--``.
* ``main`` refuses an odd count or one below 128.
"""
import json
import sys
from pathlib import Path

import pytest

from calibration import tmopt_common as RC
from calibration import tmopt_ext as X
from calibration.train_map import command


def _capture(monkeypatch, run):
    seen = []

    def fake_run(cmd, log):
        seen.append([str(c) for c in cmd])
        (run / "run").mkdir(parents=True, exist_ok=True)
        (run / "run" / "report.json").write_text(json.dumps(dict(status="complete")))
        return 0

    monkeypatch.setattr(RC, "run", fake_run)
    return seen


def _record(tmp_path):
    rec = tmp_path / "report.json"
    rec.write_text(json.dumps(dict(revision="r")))      # no transformers_version: no --transformers-deviation
    return rec


def _want(key, run, hooks, windows, epochs=1, unit="16x64", root=Path("/root")):
    trainer = command(key, unit, root, [], run / "run", epochs, teacher_topk=1000, fit_windows=windows)
    return [str(c) for c in [sys.executable, X.EXT, *hooks, "--", *trainer[1:]]]


@pytest.mark.parametrize("model", ["nemotron-nano-9b-v2", "qwen3.8-27b", "mistral-7b"])
def test_default_is_the_release_command(monkeypatch, tmp_path, model):
    run = tmp_path / "run"
    seen = _capture(monkeypatch, run)
    spec, key = RC.resolve(model)
    X.run_trainer(spec, key, "16x64", run, Path("/root"), _record(tmp_path), 1, ["--oneshot", "g.pt"])
    assert seen == [_want(key, run, ["--oneshot", "g.pt"], 256)]
    cmd = seen[0]
    assert cmd[cmd.index("--fit-windows") + 1] == "256" and "--teacher-host" not in cmd
    assert X.RELEASE["fit_windows"] == 256


def test_fit_windows_changes_only_the_trainer_count(monkeypatch, tmp_path):
    run = tmp_path / "run"
    seen = _capture(monkeypatch, run)
    spec, key = RC.resolve("nemotron-nano-9b-v2")
    rec = _record(tmp_path)
    X.run_trainer(spec, key, "8x64", run, Path("/root"), rec, 5, [])
    X.run_trainer(spec, key, "8x64", run, Path("/root"), rec, 5, [], fit_windows=1024)
    base, big = seen
    i = base.index("--fit-windows")
    assert base[i + 1] == "256" and big[i + 1] == "1024"
    assert base[:i + 1] + base[i + 2:] == big[:i + 1] + big[i + 2:]
    assert big == _want(key, run, [], 1024, epochs=5, unit="8x64")


def test_teacher_host_is_the_launchers_hook(monkeypatch, tmp_path):
    run = tmp_path / "run"
    seen = _capture(monkeypatch, run)
    spec, key = RC.resolve("qwen3.8-27b")
    X.run_trainer(spec, key, "16x64", run, Path("/root"), _record(tmp_path), 1, ["--oneshot", "g.pt"],
                  fit_windows=1024, teacher_host=True)
    cmd = seen[0]
    sep = cmd.index("--")
    assert cmd[2:sep] == ["--oneshot", "g.pt", "--teacher-host"]
    assert cmd == _want(key, run, ["--oneshot", "g.pt", "--teacher-host"], 1024)
    # the launcher accepts it with the trainer's top-K teacher
    from calibration import tmopt_launch as L
    ext, trainer = L._parse(cmd[2:])
    assert ext.teacher_host and ext.oneshot == Path("g.pt") and trainer[trainer.index("--fit-windows") + 1] == "1024"


@pytest.mark.parametrize("n", ["127", "1023", "64"])
def test_main_refuses_bad_counts(tmp_path, n):
    with pytest.raises(SystemExit, match="even and >= 128"):
        X.main(["oneshot", "--model", "phi4-14b", "--unit", "16x64", "--match", str(tmp_path / "m.pt"),
                "--out", str(tmp_path / "x.pt"), "--fit-windows", n])
    assert not (tmp_path / "x.run").exists()
