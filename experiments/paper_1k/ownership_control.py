"""Item 7's negative control: the readback check (`evaluation.ppl --ownership-check`) must fail when the tags the
kernel executes are not the map's.

    python -m experiments.paper_1k.ownership_control --model M --map MAP --out OUT.json [--kernel-build-dir DIR]

On the first quantized linear (in the model's order) that has an E0M3 tile in the map, its BF16 weight is packed as
`--ownership-check` packs it (models.sm120's packing of the map at the kernels' granularity) twice: with the map, and
with an all-E2M1 map. kernels/sm120/eval/ownership_check.check_module, the check `--ownership-check` runs on every
module, then compares what the kernel decodes with the stored weights and the formats a map assigns:
- the map's packing against the map: exact, 0 format mismatches (the positive case, as in the full check);
- the all-E2M1 packing against the map (a deployed all-E2M1 map, the SM100 control): format mismatches > 0;
- the map's packing against an all-E2M1 map: format mismatches > 0.
Pass: all three. The kernel is the 'auto' route's primary build for the map's unit (the readback's own kernel).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def control(probe, w, bias, tile, unit):
    """The three checks on one module (``tile``: its map tiles); ``probe``: the kernel. Returns ``(rows, passed)``."""
    import torch
    from models import sm120 as S
    R = S.package()
    mask, tb = S._packing(tile.to(w.device), unit, w.shape[0])
    none = torch.zeros_like(mask)
    packed = R.artifact.pack_module("control", w, bias, S.KIND["mixfp4"], mask, tb)
    all_e2m1 = R.artifact.pack_module("control", w, bias, S.KIND["mixfp4"], none, tb)
    check = S._ownership().check_module
    rows = dict(map_deployed_vs_map=check(probe, packed, mask),
                all_e2m1_deployed_vs_map=check(probe, all_e2m1, mask),
                map_deployed_vs_all_e2m1=check(probe, packed, none))
    passed = (rows["map_deployed_vs_map"]["exact"] and rows["map_deployed_vs_map"]["format_mismatches"] == 0
              and rows["all_e2m1_deployed_vs_map"]["format_mismatches"] > 0
              and rows["map_deployed_vs_all_e2m1"]["format_mismatches"] > 0)
    return rows, passed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--kernel-build-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from flipquant import maps as M
    from models import sm120 as S
    from models.loader import load, quantizable_linears
    t0 = time.time()
    tiles, unit, _ = M.load(a.map)
    model, _, spec = load(a.model)
    linears = quantizable_linears(model, spec)
    name = next(n for n in linears if bool(tiles[n].any()))
    lin = linears[name]
    R = S.package()
    kern, label, routing = S.resolve_kernel("auto", "mixfp4", unit, a.kernel_build_dir)
    probe = kern.primary if isinstance(kern, R.select.KernelSet) else kern
    w = lin.weight.data
    rows, passed = control(probe, w, lin.bias, tiles[name], unit)
    rec = dict(status="complete", passed=passed, model=a.model, map=str(a.map), unit=M.unit_name(unit), module=name,
               shape=list(w.shape), map_e0m3_tiles=int(tiles[name].sum()), kernel=label, routing=routing,
               probe=dict(name=probe.cfg.name, sha256=probe.sha256), rows=rows, seconds=round(time.time() - t0, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"CONTROL {a.model} {M.unit_name(unit)} {name}: {'PASS' if passed else 'FAIL'} "
          + "; ".join(f"{k}: exact {v['exact']}, mismatches {v['format_mismatches']}" for k, v in rows.items()), flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
