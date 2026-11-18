"""MixFP4 format maps: one bool per ``rows x cols`` weight tile, True where the tile is E0M3.

A map for one ``[O, K]`` linear weight has shape ``[ceil(O / rows), K / cols]``. The tile unit is
what calibration trains at; the native kernels need the unit to be a whole number of their own
format granule (sm_100 N8xK64: rows % 8 == 0 and cols == 64).

On disk (``save``/``load``)::

    {"format": "flipquant-map/1", "unit": "8x64", "tiles": {module_name: bool tensor}, "meta": {...}}

``load`` also accepts a trainer's ``map.pt`` (a bare ``{module_name: bool tensor}`` dict), in which
case the unit must be given.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

UNITS = {"256x64": (256, 64), "16x64": (16, 64), "8x64": (8, 64), "1x16": (1, 16)}
MAP_FORMAT = "flipquant-map/1"


def parse_unit(unit) -> tuple[int, int]:
    if isinstance(unit, tuple):
        return unit
    if unit in UNITS:
        return UNITS[unit]
    rows, cols = unit.lower().split("x")
    return int(rows), int(cols)


def unit_name(unit) -> str:
    rows, cols = parse_unit(unit)
    return f"{rows}x{cols}"


def tile_shape(out_features: int, in_features: int, unit) -> tuple[int, int]:
    rows, cols = parse_unit(unit)
    if in_features % cols:
        raise ValueError(f"in_features={in_features} is not a multiple of the tile width {cols}")
    return -(-out_features // rows), in_features // cols


def expand(mask: torch.Tensor, rows: int, cols: int, height: int | None = None) -> torch.Tensor:
    """Tile map -> element map (``[:height]`` trims the ragged last row block)."""
    return mask.repeat_interleave(rows, 0).repeat_interleave(cols, 1)[:height]


def reduce(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Element tensor -> per-tile sums (the adjoint of :func:`expand`)."""
    o, k = x.shape
    x = F.pad(x, (0, 0, 0, (-o) % rows))
    return x.reshape(-1, rows, k // cols, cols).sum((1, 3))


def to_granule(mask: torch.Tensor, unit, granule: tuple[int, int], out_features: int) -> torch.Tensor:
    """Re-express a tile map at a kernel's format granule, as ``uint8`` with 0 = E0M3, 1 = E2M1.

    The map must be at least as coarse as the granule and aligned to it; a finer map cannot be
    executed natively and raises rather than being silently coarsened.
    """
    rows, cols = parse_unit(unit)
    g_rows, g_cols = granule
    if rows % g_rows or cols % g_cols:
        raise ValueError(f"map unit {rows}x{cols} is finer than / misaligned with the kernel's "
                         f"{g_rows}x{g_cols} format granule; calibrate at a unit the kernel supports")
    if out_features % g_rows:
        raise ValueError(f"out_features={out_features} is not a multiple of the granule height {g_rows}")
    fine = mask.repeat_interleave(rows // g_rows, 0).repeat_interleave(cols // g_cols, 1)
    return (~fine[: out_features // g_rows]).to(torch.uint8).contiguous()


def save(path, tiles: dict[str, torch.Tensor], unit, meta: dict | None = None) -> None:
    torch.save({"format": MAP_FORMAT, "unit": unit_name(unit),
                "tiles": {n: m.detach().bool().cpu() for n, m in tiles.items()},
                "meta": dict(meta or {})}, Path(path))


def load(path, unit=None) -> tuple[dict[str, torch.Tensor], str, dict]:
    """Returns ``(tiles, unit, meta)``."""
    # Maps saved before the source record stringified torch.__version__ hold a TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and obj.get("format") == MAP_FORMAT:
        if unit is not None and unit_name(unit) != obj["unit"]:
            raise ValueError(f"{path} was saved at unit {obj['unit']}, not {unit_name(unit)}")
        return obj["tiles"], obj["unit"], obj.get("meta", {})
    if unit is None:
        raise ValueError(f"{path} is a bare map (a trainer's map.pt); pass its tile unit, e.g. 8x64")
    return {n: m.bool() for n, m in obj.items()}, unit_name(unit), {"source": str(path)}


def uniform(shapes: dict[str, tuple[int, int]], unit, e0m3: bool) -> dict[str, torch.Tensor]:
    """An all-E2M1 (FourOverSix) or all-E0M3 map for the given ``{name: (out, in)}`` shapes."""
    return {n: torch.full(tile_shape(o, k, unit), e0m3, dtype=torch.bool) for n, (o, k) in shapes.items()}


def summary(tiles: dict[str, torch.Tensor]) -> dict:
    total = sum(m.numel() for m in tiles.values())
    e0m3 = sum(int(m.sum()) for m in tiles.values())
    return {"modules": len(tiles), "tiles": total, "e0m3_tiles": e0m3,
            "e0m3_fraction": e0m3 / max(total, 1)}
