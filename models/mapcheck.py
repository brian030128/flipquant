"""Validate a MixFP4 map file before an experiment uses it (scripts/run_*.py, ``--maps-dir``).

A usable map for model M at unit U:
  * loads (flipquant.maps format), at unit U;
  * its metadata names model M (``registry_key``, as written by calibration.train_map, or ``model``);
  * was trained by TM-OPT+TC, the paper's trainer (the reference ``run_train_map.py --tm-opt --tile-grad-tc``,
    vendored in calibration/tmopt and run by calibration.train_map or calibration.tmopt): its ``trainer``
    metadata names it, and its ``method`` does not flag a shortened run (``NOT the method``);
  * was trained on the BF16 model, not on a vendor NVFP4 release: no ``nvidia`` in the file name, no
    NVFP4-checkpoint source in the metadata, and it covers EVERY quantizable linear (a map trained
    with ``--nvidia-nvfp4`` covers only the release's NVFP4 subset);
  * has the expected rotation (``rotate`` / ``rotate_block`` metadata; unrotated maps have none);
  * covers exactly the BF16 model's quantizable linears, each with the tile shape of its weight at U
    (derived from the model config on the meta device, so no weights are read).

``validate`` raises ``FileNotFoundError`` / ``ValueError`` with the reason, or returns a summary line.
"""

from __future__ import annotations

from pathlib import Path

from flipquant import maps as maps_

TMOPT_TC = "run_train_map.py --tm-opt --tile-grad-tc"


def model_key(meta: dict) -> str | None:
    """The registry key a map was made for: ``registry_key`` (calibration.train_map, calibration.tmopt) or
    ``model`` (calibration.train_map, calibration.baseline_maps)."""
    return meta.get("registry_key") or meta.get("model")


def trainer_problem(meta: dict, allow_short: bool = False) -> str | None:
    """Why the map was not made by TM-OPT+TC (None if it was); ``allow_short``: a run with fewer epochs passes
    (smoke runs that set the epochs themselves)."""
    trainer = str(meta.get("trainer") or "")
    if TMOPT_TC not in trainer:
        return f"trainer {trainer or None!r} is not TM-OPT+TC ({TMOPT_TC!r})"
    # calibration.train_map / tmopt flag a shortened run ("NOT the method's 20") and, since its --teacher-topk /
    # --lr / --init-logit, any non-default setting ("NOT the paper's settings") -- except the FlipQuant release
    # maps' settings (5 epochs, --teacher-topk 1000, --fit-windows 256), the paper's final ones (paper-sm120-runs)
    if is_release(meta):
        return None
    if "NOT the " in str(meta.get("method") or "") and not allow_short:
        return f"method {meta['method']!r}"
    return None


#: the FlipQuant release maps' TM-OPT+TC settings, as calibration.train_map records them (non_default_settings)
RELEASE_SETTINGS = dict(epochs=5, teacher_topk=1000, fit_windows=256)


def is_release(meta: dict) -> bool:
    """A map made with exactly the release settings: calibration.train_map's ``non_default_settings``, or
    calibration.tmopt_ext's ``release_settings`` (train: 5 epochs; oneshot: 1, which is not a trained map)."""
    if meta.get("non_default_settings") == RELEASE_SETTINGS:
        return True
    rel = meta.get("release_settings") or {}
    return (meta.get("source", {}) or {}).get("kind") == "trained" and \
        {k: rel.get(k) for k in RELEASE_SETTINGS} == RELEASE_SETTINGS


def _nvidia_source(meta: dict) -> str | None:
    for key in ("nvfp4_checkpoint", "nvidia_nvfp4"):
        if meta.get(key):
            return f"meta[{key!r}] = {meta[key]!r}"
    src = meta.get("source")
    hf = src.get("hf_id", "") if isinstance(src, dict) else str(src or "")
    if "nvfp4" in hf.lower():
        return f"meta source {hf!r}"
    return None


def validate(path, model: str, unit: str, *, rotate: str = "none", rotate_block: int | None = None,
             tiny_layers: int | None = None, revision: str | None = None, shapes: dict | None = None,
             require_tmopt: bool = True, allow_short: bool = False) -> str:
    from .registry import get
    from .tiny import meta_linear_shapes
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"map not found: {path}")
    if "nvidia" in path.name.lower():
        raise ValueError(f"{path.name}: a map for the vendor NVFP4 release ('nvidia' in the name); "
                         "these experiments use the BF16-model ('ours') maps")
    tiles, got_unit, meta = maps_.load(path)
    if maps_.unit_name(got_unit) != maps_.unit_name(unit):
        raise ValueError(f"{path.name}: unit {got_unit}, expected {unit}")
    spec = get(model)
    if model_key(meta) != spec.key:
        raise ValueError(f"{path.name}: metadata model {model_key(meta)!r}, expected {spec.key!r}")
    if require_tmopt:
        bad = trainer_problem(meta, allow_short)
        if bad:
            raise ValueError(f"{path.name}: {bad}")
    bad = _nvidia_source(meta)
    if bad:
        raise ValueError(f"{path.name}: trained on a vendor NVFP4 release ({bad})")
    got_rotate = meta.get("rotate", "none")
    if got_rotate != rotate:
        raise ValueError(f"{path.name}: rotate={got_rotate!r}, expected {rotate!r}")
    if rotate != "none" and meta.get("rotate_block") != rotate_block:
        raise ValueError(f"{path.name}: rotate_block={meta.get('rotate_block')!r}, expected {rotate_block!r}")
    shapes = shapes if shapes is not None else meta_linear_shapes(spec, revision, tiny_layers)
    missing, extra = sorted(set(shapes) - set(tiles)), sorted(set(tiles) - set(shapes))
    if missing or extra:
        raise ValueError(f"{path.name}: covers {len(tiles)} modules, the BF16 model quantizes {len(shapes)} "
                         f"(missing e.g. {missing[:2]}, extra e.g. {extra[:2]}) -- a map for the NVFP4 "
                         "release covers only its NVFP4 subset")
    for n, (o, k) in shapes.items():
        want = maps_.tile_shape(o, k, unit)
        if tuple(tiles[n].shape) != want:
            raise ValueError(f"{path.name}: {n} has tile shape {tuple(tiles[n].shape)}, expected {want} at {unit}")
    s = maps_.summary(tiles)
    return (f"{path.name}: {spec.key} {unit} rotate={got_rotate}, {s['modules']} modules, "
            f"{s['e0m3_tiles']}/{s['tiles']} E0M3 tiles ({s['e0m3_fraction']:.4%})")
