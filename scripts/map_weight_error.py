"""The weight error of FlipQuant maps relative to FourOverSix, for Table 2's "weight err / FO6" column.

For each map ``<maps>/<model>/flipquant_<unit>.pt``, every quantizable linear is installed as ``evaluation.ppl --mode fake
--weight mixfp4 --map`` installs it (``FlipLinear.from_linear``), and the squared error of its BF16 weight against the
model's is summed (float64), next to FourOverSix's (``formats.fake_quant(w, "fourover6")``). The map's modules and tile
shapes must equal ``models.loader.quantizable_linears`` of the loaded model, and its ``meta`` revision the model's.

  python scripts/map_weight_error.py --model qwen3-8b --revision <rev> --maps maps --out runs/qwen3-8b/flipquant_weight_error.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flipquant import formats as fmt  # noqa: E402
from flipquant import maps as maps_  # noqa: E402
from flipquant.linear import FlipLinear, QuantSpec  # noqa: E402
from models.loader import load, quantizable_linears  # noqa: E402

UNITS = ("8x64", "16x64", "256x64")


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--maps", type=Path, required=True, help="directory with <model>/flipquant_<unit>.pt")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    started = time.time()
    model, _, spec = load(args.model, revision=args.revision, device_map=args.device_map)
    linears = quantizable_linears(model, spec)
    revision = model.flipquant_source["revision"]
    maps = {}
    for unit in UNITS:
        path = args.maps / spec.key / f"flipquant_{unit}.pt"
        if not path.exists():
            continue
        tiles, u, meta = maps_.load(path)
        assert u == unit, (path, u)
        assert list(tiles) == list(linears), f"{path}: modules differ from quantizable_linears"
        for n, lin in linears.items():
            assert tuple(tiles[n].shape) == maps_.tile_shape(lin.out_features, lin.in_features, unit), (path, n)
        if meta.get("revision") and meta["revision"] != revision:
            raise SystemExit(f"{path}: map revision {meta['revision']} != model revision {revision}")
        maps[unit] = dict(tiles=tiles, path=path, meta=meta)
    out = dict(model=spec.key, revision=revision, units={})
    fo6_sq = 0.0
    sq = {u: 0.0 for u in maps}
    e0m3_elements = {u: 0 for u in maps}
    elements = 0
    for n, lin in linears.items():
        w = lin.weight.detach()
        fo6_sq += float(((fmt.fake_quant(w, "fourover6").double() - w.double()) ** 2).sum())
        elements += w.numel()
        for unit, m in maps.items():
            spec_q = QuantSpec(weight="mixfp4", act="fourover6", act_scope="row", unit=unit, mode="fake")
            layer = FlipLinear.from_linear(lin, spec_q, mask=m["tiles"][n], name=n)
            sq[unit] += float(((layer.weight.double() - w.double()) ** 2).sum())
            r, c = maps_.parse_unit(unit)
            e0m3_elements[unit] += int(maps_.expand(m["tiles"][n].to(w.device), r, c, lin.out_features).sum())
            del layer
    for unit, m in maps.items():
        s = maps_.summary(m["tiles"])
        out["units"][unit] = dict(map=str(m["path"]), map_sha256=hashlib.sha256(m["path"].read_bytes()).hexdigest(),
                                  map_revision=m["meta"].get("revision"), weight_sq_error=sq[unit],
                                  fourover6_weight_sq_error=fo6_sq, weight_error_vs_fourover6=sq[unit] / fo6_sq,
                                  e0m3_tile_fraction=s["e0m3_fraction"],
                                  e0m3_element_fraction=e0m3_elements[unit] / elements)
        print(f"{spec.key} flipquant {unit}: weight err / FO6 {sq[unit] / fo6_sq:.4f}, "
              f"E0M3 tiles {s['e0m3_fraction']:.4f}", flush=True)
    out["seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
