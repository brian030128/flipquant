"""The paper's TM-OPT+TC trainer, vendored unchanged in ``calibration/tmopt``, behind ``calibration.train_map`` (CPU).

* The vendored files are the source's, byte for byte (``VENDORED.json``); the one port change is listed in
  ``PORT_MODIFIED`` (``calibration/tmopt/PORT_CHANGES.md``).
* The trainer runs the paper's flags (the vendored ``experiments/paper/paper_common.py``) and its per-model batch rule.
  Every committed map's recorded training settings are what the trainer would run for that model.
* ``--epochs`` / ``--lr`` / ``--init-logit`` / ``--teacher-topk`` / ``--fit-windows`` / ``--seed`` / ``--batch`` /
  ``--accum``: other values than the paper's are passed to the vendored trainer, and ``--require-reproduction`` refuses
  them.
* The options of ``calibration.train_map_legacy``: accepted at the method's value, an error naming the legacy trainer
  otherwise.
* Every registry key maps to the trainer's model at the same repository and revision (``prepare_model_data.MODELS``).
  A model neither knows (``RUNTIME_MODELS``) is registered at run time, in the preparation's and the trainer's own
  processes, as the models without an archived record; nothing changes for the others.
* The run's environment is the paper runs' (cuBLAS workspace, expandable segments).
* The map conversion checks, and the committed paper maps carry the paper's map sha256.

The bitwise reproduction itself trains on a GPU for 10-20 minutes per map. ``FLIPQUANT_REPRODUCE=1`` runs it for
Llama-3.1-8B 8x64 (needs an sm_120 GPU, the model and the calibration datasets); the recorded runs are in
``calibration/tmopt/validation``.
"""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calibration import tmopt_common as RC
from calibration import train_map as TM
from flipquant import maps as M

VENDORED = json.loads((RC.ENGINE / "VENDORED.json").read_text())["files_at_vendoring"]
PORT_OWNED = {"VENDORED.json", "PORT_CHANGES.md", "fit_sources.py"}
# vendored files changed by the port (PORT_CHANGES.md), with their hash here
PORT_MODIFIED = {"repro_local/realquant/test_topk_teacher.py":
                 "e8ecf044a6e1431b49ba2496057d59bd7b99cd9cc5427ae8cff3cac75c147606",
                 # calib-robustness: --fit-windows below 128 and --fit-source (PORT_CHANGES.md)
                 "run_train_map.py": "b537453948b710c312356dc984bf72ff795511261d73406b6e6b5d02c32308b8"}


def _paper_common():
    spec = importlib.util.spec_from_file_location("tmopt_paper_common", RC.ENGINE / "experiments" / "paper" / "paper_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trainer_models():
    """The vendored prepare_model_data.MODELS, read from its source (dict(repo=..., revision=..., record=...) per key)."""
    tree = ast.parse((RC.ENGINE / "prepare_model_data.py").read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "MODELS")
    return {ast.literal_eval(k): {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
            for k, v in zip(node.keys, node.values)}


def _flags(cmd):
    """``--flag value`` / ``--switch`` pairs of a run_train_map.py command."""
    args = [str(c) for c in cmd[cmd.index("run_train_map.py") + 1:]]
    out, i = {}, 0
    while i < len(args):
        key = args[i][2:].replace("-", "_")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key], i = args[i + 1], i + 2
        else:
            out[key], i = True, i + 1
    return out


def test_vendored_files_unchanged():
    digest = {f: hashlib.sha256((RC.ENGINE / f).read_bytes()).hexdigest() for f in VENDORED}
    bad = [f for f, d in VENDORED.items() if digest[f] != PORT_MODIFIED.get(f, d)]
    assert not bad, bad
    assert set(PORT_MODIFIED) <= set(VENDORED)
    on_disk = {str(p.relative_to(RC.ENGINE)) for p in RC.ENGINE.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    unlisted = sorted(f for f in on_disk - set(VENDORED) - PORT_OWNED if not f.startswith("validation/"))
    assert not unlisted, f"files in calibration/tmopt that are not vendored records: {unlisted}"


def test_flags_are_the_papers():
    P = _paper_common()
    assert TM.TM_OPT_TC == P.TM_OPT_TC
    assert set(P.TRAIN_BATCH) == {"qwen27b"} and TM.BATCH["qwen27b"] == P.TRAIN_BATCH["qwen27b"]
    # Nemotron-Nano-9B-v2 is not a paper model: micro-batch 2 x accumulation 4, as its committed maps were trained
    assert TM.BATCH == {"qwen27b": P.TRAIN_BATCH["qwen27b"], "nemotron9b": ["--batch", "2", "--accum", "4"]}
    # 01_calibrate.py's command: run_train_map.py --model --data-root [deviation] --unit <TM_OPT_TC> <batch> --out
    cmd = TM.command("llama8b", "16x64", Path("/data"), ["--transformers-deviation"], Path("/out"))
    want = ["run_train_map.py", "--model", "llama8b", "--data-root", "/data", *P.DEVIATION["llama8b"], "--unit", "16x64",
            *P.TM_OPT_TC, "--out", "/out"]
    assert [str(c) for c in cmd[1:]] == want
    assert _flags(TM.command("qwen27b", "8x64", Path("/d"), [], Path("/o"), epochs=1))["epochs"] == "1"
    assert TM.paper_batch("llama8b") == dict(batch=8, accum=1)
    assert TM.paper_batch("qwen27b") == TM.paper_batch("nemotron9b") == dict(batch=2, accum=4)


def test_options_are_passed_and_non_defaults_refused(tmp_path):
    """The paper's values give the paper's command; other values are passed to run_train_map.py, and
    --require-reproduction refuses them before anything runs."""
    assert TM.PAPER == dict(epochs=20, lr=0.02, init_logit=-1.0, teacher_topk=0, fit_windows=128, seed=0,
                            fit_source="paper", deterministic=True)
    f = _flags(TM.command("llama8b", "16x64", Path("/d"), [], Path("/o"), epochs=5, lr=0.05, init_logit=-0.5,
                          teacher_topk=256, fit_windows=256, seed=3, batch=4, accum=2))
    assert (f["epochs"], f["lr"], f["init_logit"], f["teacher_topk"], f["fit_windows"], f["seed"], f["batch"],
            f["accum"]) == ("5", "0.05", "-0.5", "256", "256", "3", "4", "2")
    f = _flags(TM.command("qwen27b", "8x64", Path("/d"), [], Path("/o"), batch=4))   # replaces the per-model value
    assert (f["batch"], f["accum"], f["gpus"]) == ("4", "4", "1")
    plain = _flags(TM.command("llama8b", "16x64", Path("/d"), [], Path("/o")))
    assert not {"teacher_topk", "fit_windows", "seed", "batch", "accum", "fit_source", "no_deterministic"} & set(plain)
    # the robustness options (paper Appendix C.4)
    f = _flags(TM.command("llama8b", "16x64", Path("/d"), [], Path("/o"), fit_windows=64, fit_source="ultrachat",
                          deterministic=False))
    assert (f["fit_windows"], f["fit_source"]) == ("64", "ultrachat") and "no_deterministic" in f
    for extra in (["--teacher-topk", "256"], ["--lr", "0.05"], ["--init-logit", "-0.5"], ["--epochs", "5"],
                  ["--fit-windows", "256"], ["--seed", "1"], ["--batch", "4"], ["--accum", "2"],
                  ["--fit-windows", "64"], ["--fit-source", "fineweb-edu"], ["--no-deterministic"]):
        with pytest.raises(SystemExit, match="paper's settings"):
            TM.main(["--model", "llama3.1-8b", "--unit", "16x64", "--require-reproduction", "--out",
                     str(tmp_path / "out"), *extra])
    # the paper's own values are not deviations: the run gets as far as the output check
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        TM.main(["--model", "qwen3.8-27b", "--unit", "8x64", "--require-reproduction", "--batch", "2", "--accum", "4",
                 "--seed", "0", "--fit-windows", "128", "--out", str(tmp_path / "exists")])
    assert not (tmp_path / "out").exists()


def test_legacy_options():
    """train_map_legacy's options: no-ops at the method's value, an error naming train_map_legacy otherwise."""
    P = TM.parser()

    def legacy(*argv):
        return TM.legacy_options(P.parse_args(["--model", "llama3.1-8b", "--unit", "16x64", "--out", "/x", *argv]),
                                 "llama8b")
    accepted, refused = legacy("--deterministic", "--no-dev", "--mode", "fake", "--param", "ste", "--optimizer", "adam",
                               "--schedule", "constant", "--eps", "1e-12", "--betas", "0.9", "0.999",
                               "--revision", RC.SNAPSHOTS["llama8b"][1])
    assert len(accepted) == 9 and not refused
    for argv in (["--mode", "native"], ["--param", "sigmoid"], ["--optimizer", "sgd"], ["--schedule", "cosine"],
                 ["--eps", "1e-8"], ["--betas", "0.8", "0.99"], ["--revision", "main"], ["--native-backend", "auto"],
                 ["--device-map", "balanced"], ["--tau-start", "2"], ["--eval-every", "2"], ["--eval-batch", "8"],
                 ["--teacher-dir", "/t"], ["--calib-record", "/r.json"], ["--nvidia-nvfp4"], ["--ppl-datasets", "wiki"]):
        accepted, refused = legacy(*argv)
        assert not accepted and len(refused) == 1 and refused[0].split()[0] == argv[0], argv
    _, refused = TM.legacy_options(P.parse_args(["--model", "x", "--unit", "1x16", "--out", "/x"]), "llama8b")
    assert refused == ["--unit 1x16"]
    with pytest.raises(SystemExit, match="calibration.train_map_legacy"):
        TM.main(["--model", "llama3.1-8b", "--unit", "16x64", "--out", "/nonexistent/x", "--mode", "native"])


def test_committed_maps_were_trained_with_these_settings():
    """Each committed map's meta records its training settings: the trainer's command for that model sets the same."""
    checked = 0
    for path in sorted((RC.ROOT / "maps").glob("*/flipquant_*.pt")):
        _, unit, meta = M.load(path)
        spec_key = path.parent.name
        if spec_key not in RC.MODEL_KEYS or "run_train_map.py" not in meta.get("trainer", ""):
            continue
        s = meta["settings"]
        # a map trained off the paper's settings records them (the nemotron and qwen3.8-27b maps: 256 windows,
        # 5 epochs, top-1000 teacher); the command with those settings must match the rest
        non_default = meta.get("non_default_settings", {})
        f = _flags(TM.command(RC.MODEL_KEYS[spec_key], unit, Path("/d"), [], Path("/o"), **non_default))
        got = dict(param=f["param"], lr=float(f["lr"]), init_logit=float(f["init_logit"]), epochs=int(f["epochs"]),
                   batch=int(f.get("batch", 8)), accum=int(f.get("accum", 1)), tm_opt=f["tm_opt"],
                   tile_grad_tc=f["tile_grad_tc"], unit=f["unit"])
        if non_default:
            got.update(teacher_topk=int(f.get("teacher_topk", 0)), fit_windows=int(f.get("fit_windows", 128)))
        if meta["source"]["kind"] == "trained":
            got.update(no_dev=f["no_dev"], no_eval=f["no_eval"])
        # else a reused paper map: its run predates --no-dev and monitored a development set, which never touches
        # training (the reference's results/nodev_cost: the no-dev reruns are bitwise equal)
        assert got == {k: s[k] for k in got}, path
        assert (s["optimizer"], s["schedule"], s["seed"], s["deterministic"]) == ("adam", "constant", 0, True), path
        checked += 1
    assert checked == 12, checked


def _multiround_models():
    """The vendored run_multiround's CALIBRATIONS and LOCAL_DEVELOPMENT keys (its MODELS: what run_train_map accepts)."""
    tree = ast.parse((RC.ENGINE / "run_multiround.py").read_text())
    keys = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("CALIBRATIONS", "DEVELOPMENT", "PUBLISHED"):
            keys |= {ast.literal_eval(k) for k in n.value.keys}
    return keys


def test_registry_keys_map_to_the_trainers_models():
    models = _trainer_models()
    for registry_key, key in RC.MODEL_KEYS.items():
        spec, got = RC.resolve(registry_key)
        assert got == key
        if key in RC.INSTRUCT_BASE:
            # not in prepare_model_data (calibration.tmopt_common.INSTRUCT_PREPARE builds its record); the trainer
            # knows it, and its base model's record is archived
            assert key in _multiround_models() and RC.INSTRUCT_BASE[key] in RC.RECORDED
            assert spec.hf_id == RC.SNAPSHOTS[key][0] and spec.revision == RC.SNAPSHOTS[key][1]
            continue
        if key in RC.RUNTIME_MODELS:
            # known to neither prepare_model_data nor the trainer: registered at run time (tmopt_common.register,
            # tmopt_launch.py --register-model)
            assert key not in models and key not in (RC.ENGINE / "run_multiround.py").read_text()
            assert spec.hf_id == RC.SNAPSHOTS[key][0] and spec.revision == RC.SNAPSHOTS[key][1]
            continue
        assert RC.SNAPSHOTS[key] == (models[key]["repo"], models[key]["revision"])
        assert spec.hf_id == models[key]["repo"] and spec.revision in (None, models[key]["revision"])
    for key in ("qwen4b", "mistral7b"):       # the builder rule's checks
        assert RC.SNAPSHOTS[key] == (models[key]["repo"], models[key]["revision"])
    assert {k for k in RC.MODEL_KEYS.values() if k in models and models[k]["record"]} == RC.RECORDED
    assert all((RC.ENGINE / models[k]["record"]).exists() for k in models if models[k]["record"])
    # the registry's aliases resolve; this registry's mistral-7b is the Instruct model, as in the committed maps
    assert RC.resolve("llama8b")[1] == "llama8b" and RC.resolve("mistral7b")[1] == "mistral7b_ins"
    # mistral-7b-base is the paper's Mistral-7B-v0.3, at its revision
    assert RC.resolve("mistral-7b-base")[1] == "mistral7b"
    with pytest.raises(SystemExit):
        RC.resolve("llama3.2-3b-instruct")


def test_runtime_models_are_registered_at_launch(tmp_path):
    """A RUNTIME_MODELS key (Llama-3.2-1B-Instruct) is prepared with a registration in prepare_model_data.MODELS (no
    record) and trained through tmopt_launch.py --register-model, which adds it to the trainer's tables as its models
    without a record (three development draws, no published record). Every other model's trainer command and
    preparation code are unchanged, and the launcher refuses to register a model the trainer knows."""
    root, run = tmp_path / "data", tmp_path / "run"
    launcher = str(RC.ROOT / "calibration" / "tmopt_launch.py")
    for key in set(RC.MODEL_KEYS.values()):
        cmd = [str(c) for c in TM.command(key, "16x64", root, [], run)]
        if key in RC.RUNTIME_MODELS:
            assert cmd[1:6] == [launcher, "--register-model", key, "--", "run_train_map.py"]
            assert "assert key not in P.MODELS" in RC.register(key) and "record=None" in RC.register(key)
        else:
            assert cmd[1] == "run_train_map.py" and "--register-model" not in cmd and RC.register(key) == ""
    assert RC.RUNTIME_MODELS == {"llama1b_ins"}
    code = RC.PREPARE.format(builder="B", key="llama1b_ins", out="O", fit_only=True, register=RC.register("llama1b_ins"))
    assert ("P.MODELS[key] = dict(repo='meta-llama/Llama-3.2-1B-Instruct', "
            "revision='9213176726f574b556790deb65791e0c5aa438b6', record=None)") in code
    probe = f"""
import sys
sys.path[0:1] = [{str(RC.ENGINE)!r}]
sys.path.append({str(RC.ROOT)!r})
import run_train_map as R, run_multiround as RM
from calibration import tmopt_launch as L
known, published, local = RM.MODELS, dict(RM.PUBLISHED), dict(RM.LOCAL_DEVELOPMENT)
for args in ((['--model', 'llama8b'], 'llama8b'), (['--model', 'llama8b'], 'llama1b_ins')):
    try:
        L.register_model(R, *args)
        print('REGISTERED', args)
    except SystemExit as e:
        print('REFUSED', e)
assert RM.MODELS == known and R.MODELS == known
L.register_model(R, ['--model', 'llama1b_ins'], 'llama1b_ins')
assert R.MODELS == RM.MODELS == known + ('llama1b_ins',) and RM.PUBLISHED == published
assert {{k: v for k, v in RM.LOCAL_DEVELOPMENT.items() if k != 'llama1b_ins'}} == local
cal, dev = RM.data_paths('llama1b_ins', {str(root)!r})
print('PATHS', cal, ' '.join(str(d) for d in dev))
"""
    out = subprocess.run([sys.executable, "-c", probe], cwd=RC.ENGINE, env=RC.env({"CUDA_VISIBLE_DEVICES": ""}),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert out.stdout.count("REFUSED") == 2 and "REGISTERED" not in out.stdout, out.stdout
    want = " ".join(str(root / "llama1b_ins" / n) for n in RC.DRAWN_DEVELOPMENT)
    assert f"PATHS {root / 'llama1b_ins' / 'calibration'} {want}" in out.stdout, out.stdout


def test_fit_windows_development_records(tmp_path):
    """--fit-windows: Llama-3.1-8B's development manifests come from the vendored records, where the trainer's
    data_paths reads them. Qwen3.8-27B's are reproduced by the full preparation, which accepts exactly the sets whose
    fresh.pt sha256 the vendored record holds, under the names the trainer reads; its published C4 documents (in the
    skip set) are vendored too."""
    got = RC.prepare_development("llama8b", tmp_path)
    assert [Path(p).parent.name for p in got] == list(RC.LLAMA_DEVELOPMENT)
    for p, source in zip(got, RC.LLAMA_DEVELOPMENT.values()):
        assert Path(p).read_bytes() == (RC.ENGINE / source).read_bytes()
        assert source in VENDORED
    assert RC.prepare_development("llama8b", tmp_path) == got          # idempotent
    record = json.loads((RC.ENGINE / RC.QWEN27B_RECORD).read_text())
    assert [d["name"].rsplit("/", 1)[1] for d in record["development"]] == list(RC.QWEN27B_DEVELOPMENT)
    published = "results/kse_paper/job_336969/qwen27b/report.json"
    assert RC.QWEN27B_RECORD in VENDORED and published in VENDORED
    assert len(json.loads((RC.ENGINE / published).read_text())["data"]["c4_paper"]["documents"]) == 256


def test_environment_is_the_papers(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/extra")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    e = RC.env()
    assert e["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert e["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert e["PYTHONPATH"].split(":") == [str(RC.ENGINE), str(RC.ROOT / "kernels" / "sm120"), "/extra"]


def test_check_tiles():
    shapes = {"a": (24, 128), "b": (16, 64)}
    tiles = {n: torch.zeros(M.tile_shape(*s, "8x64"), dtype=torch.bool) for n, s in shapes.items()}
    assert TM.check_tiles(tiles, "8x64", shapes) == []
    assert TM.check_tiles(tiles, "16x64", shapes)                         # wrong unit -> wrong tile shapes
    assert TM.check_tiles({**tiles, "c": tiles["b"]}, "8x64", shapes)     # extra module
    assert TM.check_tiles({"a": tiles["a"].float(), "b": tiles["b"]}, "8x64", shapes)


def test_committed_paper_maps_are_the_papers():
    """phi4-14b ships the paper's maps: the reproduction check accepts exactly them. (qwen3.8-27b shipped them too
    until 2026-10-09; its maps are now the 256-window, 5-epoch Hugging Face release.)"""
    paper = json.loads(TM.PAPER_MAPS.read_text())["maps"]
    for registry_key, key in (("phi4-14b", "phi4"),):
        spec = RC.resolve(registry_key)[0]
        for unit in ("8x64", "16x64", "256x64"):
            tiles, _, meta = M.load(RC.ROOT / "maps" / registry_key / f"flipquant_{unit}.pt")
            want = paper[f"{key}_{unit}"]["map_sha256"]
            assert meta["source_map_sha256"] == want
            rep = TM.reproduction(spec, key, unit, want, tiles)
            assert rep["paper_map"]["equal"] and rep["committed_map"]["equal"]
            other = TM.reproduction(spec, key, unit, "0" * 64, tiles)
            assert not other["paper_map"]["equal"] and not other["committed_map"]["equal"]


def test_checkout_is_recorded():
    """train_map records the commit it runs from (and whether the tracked files were unmodified) in the map's meta."""
    code = TM.checkout()
    assert len(code["commit"]) == 40 and set(code["commit"]) <= set("0123456789abcdef")
    assert code["clean"] in (True, False)


def test_committed_maps_match_their_manifest():
    """maps/flipquant_sha256.json lists every committed FlipQuant map with its sha256 and the provenance wording of its
    meta (made neutral on 2026-10-05; the sha256 from before is recorded too). An entry with its own meta (the
    nemotron and qwen3.8-27b maps from Hugging Face, not the paper's settings) is checked against that instead."""
    manifest = json.loads((RC.ROOT / "maps" / "flipquant_sha256.json").read_text())
    files = sorted(str(p.relative_to(RC.ROOT / "maps")) for p in (RC.ROOT / "maps").glob("*/flipquant_*.pt"))
    assert sorted(manifest["files"]) == files and len(files) == 12
    for name, entry in manifest["files"].items():
        assert RC.sha256(RC.ROOT / "maps" / name) == entry["sha256"] != entry["previous_sha256"], name
        meta = M.load(RC.ROOT / "maps" / name)[2]
        want = entry.get("meta", manifest["meta"])
        assert {k: meta[k] for k in want} == want, name


@pytest.mark.skipif(os.environ.get("FLIPQUANT_REPRODUCE") != "1",
                    reason="trains a map (sm_120 GPU, ~10 min); set FLIPQUANT_REPRODUCE=1")
def test_reproduces_the_paper_map(tmp_path):
    TM.main(["--model", "llama3.1-8b", "--unit", "8x64", "--out", str(tmp_path / "run"), "--require-reproduction"])
    rep = json.loads((tmp_path / "run" / "reproduction.json").read_text())
    assert rep["references"]["paper_map"]["equal"]
    assert M.load(tmp_path / "run" / "map.pt")[1] == "8x64"
