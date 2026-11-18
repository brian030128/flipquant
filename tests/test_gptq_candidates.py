"""Opt-in GPTQ candidates (NVFP4-RaZeR paper-eval part O), on CPU.

``flipquant.gptq.compose_candidates`` (``--gptq-candidates E2M1 E0M3``): an E0M3 tile takes the E0M3 file's codes and
scales, an E2M1 tile the E2M1 file's; RTN files compose to rtn_grid's mixfp4 codes bit for bit; uniform maps give one
file's codes; files made with other GPTQ settings, another weight format, or E2M1 blocks in the E0M3 file are refused.

``calibration/tmopt_launch.py --candidates-e2m1 / --candidates-e0m3``: the hook driven through run_train_map's own
lean-store call sequence (tests/_candidates_hook_driver.py, the launcher's sys.path, its own process): with ``rtn`` the
packed store is the unhooked one byte for byte; with code files it decodes to the files' codes; codes on another grid
are refused; the options are validated.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flipquant import gptq, maps
from flipquant.linear import QuantSpec

REPO = Path(__file__).resolve().parents[1]
UNIT = "16x64"
SHAPES = {"layers.0.a": (64, 256), "layers.0.b": (48, 128), "layers.1.c": (40, 192)}


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, (o, k) in SHAPES.items():
        w = torch.randn(o, k, generator=g) * 0.05
        w[0, :16] = 0.0                         # an all-zero scale block
        w[1, 3] = -0.0                          # a negative zero
        w[2, 5] = 0.9                           # an outlier
        out[n] = w.bfloat16()
    return out


def _hessian(k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * k, k, generator=g) @ (torch.randn(k, k, generator=g) / k ** 0.5 + torch.eye(k))
    return x.T @ x / x.shape[0]


def _codes(w, weight, mask=None, run_gptq=True):
    grid, rtn = gptq.rtn_grid(w, weight, mask, UNIT)
    return gptq.gptq(w, _hessian(w.shape[1], 1), grid) if run_gptq else rtn


def _key(weight, tiles=None, **over):
    spec = QuantSpec(weight=weight, act="fourover6", act_scope="row", unit=UNIT, mode="fake")
    key = gptq.cache_key("toy", spec, tiles, "none", 0.01, 128, 256, False, "release", "quantized", "d" * 64)
    key.update(over)
    return key


def _ones():
    return {n: torch.ones(maps.tile_shape(*s, UNIT), dtype=torch.bool) for n, s in SHAPES.items()}


def _mask(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.rand(maps.tile_shape(*s, UNIT), generator=g) < 0.4 for n, s in SHAPES.items()}


def _write(path, codes, key):
    gptq._save(path, {n: gptq._packed(c) for n, c in codes.items()}, key)
    return path


def _files(tmp_path, run_gptq=True, e2m1=("fourover6", None, {}), e0m3=("mixfp4", "ones", {})):
    ws = _weights()
    tiles = {"ones": _ones(), "mask": _mask(7), None: None}
    b = {n: _codes(w, e2m1[0], None if tiles[e2m1[1]] is None else tiles[e2m1[1]][n], run_gptq) for n, w in ws.items()}
    a = {n: _codes(w, e0m3[0], None if tiles[e0m3[1]] is None else tiles[e0m3[1]][n], run_gptq) for n, w in ws.items()}
    pb = _write(tmp_path / "e2m1.pt", b, _key(e2m1[0], tiles[e2m1[1]], **e2m1[2]))
    pa = _write(tmp_path / "e0m3.pt", a, _key(e0m3[0], tiles[e0m3[1]], **e0m3[2]))
    return ws, b, a, pb, pa


def test_compose_takes_each_tile_from_its_file(tmp_path):
    ws, b, a, pb, pa = _files(tmp_path)
    m = _mask()
    out, info = gptq.compose_candidates((pb, pa), _key("mixfp4", m), m, UNIT)
    for n, (o, k) in SHAPES.items():
        e0 = maps.expand(m[n], 16, 4, o)
        wide = e0.repeat_interleave(16, 1)
        assert bool(e0.any()) and not bool(e0.all())
        assert torch.equal(out[n].codes, torch.where(wide, a[n].codes, b[n].codes))
        assert torch.equal(out[n].scale, torch.where(e0, a[n].scale, b[n].scale))
        assert torch.equal(out[n].fmt, e0) and torch.equal(out[n].gscale.float(), b[n].gscale.float())
        want = torch.where(wide, a[n].dequantize(), b[n].dequantize())
        assert torch.equal(out[n].dequantize().view(torch.int16), want.view(torch.int16))
        assert not torch.equal(b[n].codes, gptq.rtn_grid(ws[n], "fourover6")[1].codes)    # GPTQ moved codes
    assert info["e2m1"]["codes_sha256"] == gptq.codes_sha256(b)
    assert info["e0m3"]["codes_sha256"] == gptq.codes_sha256(a)


def test_compose_of_rtn_files_is_rtn_mixfp4(tmp_path):
    ws, b, a, pb, pa = _files(tmp_path, run_gptq=False)
    m = _mask(1)
    out, _ = gptq.compose_candidates((pb, pa), _key("mixfp4", m), m, UNIT)
    for n, w in ws.items():
        _, ref = gptq.rtn_grid(w, "mixfp4", m[n], UNIT)
        assert torch.equal(out[n].codes, ref.codes) and torch.equal(out[n].scale, ref.scale)
        assert torch.equal(out[n].fmt, ref.fmt) and torch.equal(out[n].gscale.float(), ref.gscale.float())


@pytest.mark.parametrize("e0m3", [False, True])
def test_uniform_maps_give_one_file(tmp_path, e0m3):
    _, b, a, pb, pa = _files(tmp_path)
    m = {n: torch.full(maps.tile_shape(*s, UNIT), e0m3) for n, s in SHAPES.items()}
    out, _ = gptq.compose_candidates((pb, pa), _key("mixfp4", m), m, UNIT)
    src = a if e0m3 else b
    for n in SHAPES:
        assert torch.equal(out[n].codes, src[n].codes) and torch.equal(out[n].scale, src[n].scale)


@pytest.mark.parametrize("case", ["damp", "propagate", "data", "e2m1_nvfp4", "e0m3_fourover6", "e0m3_has_e2m1_blocks",
                                  "swapped"])
def test_compose_refuses(tmp_path, case):
    kw = dict(damp=dict(e2m1=("fourover6", None, dict(damp=0.02))),
              propagate=dict(e0m3=("mixfp4", "ones", dict(propagate="bf16"))),
              data=dict(e0m3=("mixfp4", "ones", dict(data_sha256="e" * 64))),
              e2m1_nvfp4=dict(e2m1=("nvfp4", None, {})),
              e0m3_fourover6=dict(e0m3=("fourover6", None, {})),
              e0m3_has_e2m1_blocks=dict(e0m3=("mixfp4", "mask", {})),
              swapped={})[case]
    _, _, _, pb, pa = _files(tmp_path, **kw)
    m = _mask()
    with pytest.raises(SystemExit):
        gptq.compose_candidates((pa, pb) if case == "swapped" else (pb, pa), _key("mixfp4", m), m, UNIT)


# ------------------------------------------------------------------------------------------------- the trainer hook
def _drive(tmp_path, src_b, src_a, weights=None):
    wf = tmp_path / "weights.pt"
    torch.save(weights or _weights(), wf)
    out = tmp_path / "hook"
    out.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(REPO / "tests" / "_candidates_hook_driver.py"), str(wf), str(out),
                        str(src_b), str(src_a)], cwd=REPO / "calibration" / "tmopt", capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    return r, out


def test_hook_rtn_is_the_unhooked_store(tmp_path):
    r, out = _drive(tmp_path, "rtn", "rtn")
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "result.json").read_text())
    assert res["hook"]["lean"] and res["hook"]["modules"] == len(SHAPES)
    assert all(m["store_equals_unhooked"] and m["packed_path"] for m in res["modules"].values()), res
    rec = json.loads((out / "candidates_hook.json").read_text())
    assert not rec["missing_or_repeated"] and rec["mean_codes_changed_vs_rtn"] == {"e2m1": 0.0, "e0m3": 0.0}


def test_hook_files_give_the_files_codes(tmp_path):
    _, _, _, pb, pa = _files(tmp_path)
    r, out = _drive(tmp_path, pb, pa)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "result.json").read_text())
    for m in res["modules"].values():
        assert m["e2m1_decodes_to_file"] and m["e0m3_decodes_to_file"] and not m["store_equals_unhooked"], res
        assert m["packed_path"], res                # the trainer's packed path, not its dense fallback
    rec = json.loads((out / "candidates_hook.json").read_text())
    assert rec["mean_codes_changed_vs_rtn"]["e2m1"] > 0 and rec["mean_codes_changed_vs_rtn"]["e0m3"] > 0


def test_hook_rtn_files_are_the_unhooked_store(tmp_path):
    _, _, _, pb, pa = _files(tmp_path, run_gptq=False)
    r, out = _drive(tmp_path, pb, pa)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "result.json").read_text())
    assert all(m["store_equals_unhooked"] and m["e2m1_decodes_to_file"] and m["e0m3_decodes_to_file"]
               and m["packed_path"] for m in res["modules"].values()), res


@pytest.mark.parametrize("case", ["nvfp4_grid", "swapped"])
def test_hook_refuses_another_grid(tmp_path, case):
    kw = dict(nvfp4_grid=dict(e2m1=("nvfp4", None, {})), swapped={})[case]
    _, _, _, pb, pa = _files(tmp_path, **kw)
    r, _ = _drive(tmp_path, *((pa, pb) if case == "swapped" else (pb, pa)))
    assert r.returncode != 0 and "--candidates-e" in r.stderr, r.stderr[-3000:]


def test_launcher_options():
    spec = importlib.util.spec_from_file_location("tmopt_launch", REPO / "calibration" / "tmopt_launch.py")
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    ext, trainer = L._parse(["--candidates-e2m1", "rtn", "--candidates-e0m3", "x.pt", "--", "run_train_map.py", "--a"])
    assert (ext.candidates_e2m1, ext.candidates_e0m3, trainer) == ("rtn", "x.pt", ["--a"])
    assert L._parse(["--", "--a"])[0].candidates_e2m1 is None
    for bad in (["--candidates-e2m1", "rtn", "--"], ["--candidates-e2m1", "rtn", "--candidates-e0m3", "rtn",
                                                    "--rotate", "hadamard", "--"]):
        with pytest.raises(SystemExit):
            L._parse(bad)
