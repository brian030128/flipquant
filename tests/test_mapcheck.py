"""models/mapcheck.py: every reason a map must not be used is caught, and a good map passes. A main-table map names
its model by ``registry_key`` (calibration.train_map) and must come from TM-OPT+TC."""
import pytest
import torch

from flipquant import maps
from models.mapcheck import TMOPT_TC, validate

SHAPES = {"a": (40, 192), "b": (64, 128)}
MODEL = "qwen3-1.7b"


def _save(path, unit="16x64", shapes=SHAPES, **meta):
    tiles = {n: torch.zeros(maps.tile_shape(o, k, unit), dtype=torch.bool) for n, (o, k) in shapes.items()}
    base = dict(registry_key=MODEL, trainer=f"the reference {TMOPT_TC} (vendored: calibration/tmopt)",
                method="FlipQuant = TM-OPT+TC (the paper's calibration)")
    maps.save(path, tiles, unit, {k: v for k, v in dict(base, **meta).items() if v is not None})
    return path


def test_good_map_passes(tmp_path):
    assert "2 modules" in validate(_save(tmp_path / "m.pt"), MODEL, "16x64", shapes=SHAPES)
    p = _save(tmp_path / "r.pt", rotate="hadamard", rotate_block=16)
    assert "rotate=hadamard" in validate(p, MODEL, "16x64", rotate="hadamard", rotate_block=16, shapes=SHAPES)
    # a map written by calibration.train_map (``model``, no trainer) passes only when TM-OPT+TC is not required
    p = _save(tmp_path / "old.pt", registry_key=None, trainer=None, method=None, model=MODEL)
    assert "2 modules" in validate(p, MODEL, "16x64", shapes=SHAPES, require_tmopt=False)
    with pytest.raises(ValueError, match="not TM-OPT"):
        validate(p, MODEL, "16x64", shapes=SHAPES)
    # a shortened TM-OPT+TC run passes only where the runner asked for it (smoke --train-epochs)
    p = _save(tmp_path / "short.pt", method="FlipQuant = TM-OPT+TC -- 1 epochs, NOT the method's 20")
    assert "2 modules" in validate(p, MODEL, "16x64", shapes=SHAPES, allow_short=True)


@pytest.mark.parametrize("case", ["missing", "model", "nvidia_name", "nvidia_meta", "unit", "rotate_extra",
                                  "rotate_missing", "rotate_block", "subset", "shape", "trainer", "short",
                                  "non_default"])
def test_bad_maps_are_rejected(tmp_path, case):
    kw, name, rot = {}, "m.pt", dict(rotate="none", rotate_block=None)
    if case == "model":
        kw["registry_key"] = "qwen3-8b"
    if case == "trainer":
        kw["trainer"] = "calibration/train_map.py"
    if case == "short":
        kw["method"] = "FlipQuant = TM-OPT+TC -- 2 epochs, NOT the method's 20"
    if case == "non_default":
        kw["method"] = "FlipQuant = TM-OPT+TC -- NOT the paper's settings (teacher_topk=256)"
    if case == "nvidia_name":
        name = "mixfp4_nvidia_16x64.pt"
    if case == "nvidia_meta":
        kw["source"] = dict(hf_id="nvidia/Qwen3.8-27B-NVFP4")
    if case == "rotate_extra":
        kw.update(rotate="hadamard", rotate_block=16)
    if case == "rotate_missing":
        rot = dict(rotate="hadamard", rotate_block=16)
    if case == "rotate_block":
        kw.update(rotate="hadamard", rotate_block=128)
        rot = dict(rotate="hadamard", rotate_block=16)
    unit = "8x64" if case == "unit" else "16x64"
    shapes = {"a": SHAPES["a"]} if case == "subset" else ({"a": (64, 192), "b": SHAPES["b"]} if case == "shape" else SHAPES)
    path = tmp_path / name
    if case != "missing":
        if case == "shape":
            _save(path, shapes=shapes, **kw)
        else:
            _save(path, unit=unit, **kw)
    err = FileNotFoundError if case == "missing" else ValueError
    with pytest.raises(err):
        validate(path, MODEL, "16x64", shapes=SHAPES if case != "subset" else {**SHAPES, "c": (16, 64)}, **rot)
