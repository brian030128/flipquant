"""A map with a real map's E0M3 tiles placed at random: per module the same number of E0M3 tiles, at positions drawn
uniformly without replacement (``torch.randperm`` with a generator seeded ``--seed`` + the module's index in name order).
Item 8 times it beside the real map (`evaluation.gemm_time`'s `<unit>_random` policies): the same share per module, so
each projection's typical and densest modules are the real map's. CPU.

    python -m experiments.paper_1k.random_maps --map REAL.pt --out RANDOM.pt [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def randomize(tiles: dict, seed: int = 0) -> dict:
    import torch
    out = {}
    for i, n in enumerate(sorted(tiles)):
        t = tiles[n]
        k = int(t.sum())
        g = torch.Generator().manual_seed(seed + i)
        m = torch.zeros(t.numel(), dtype=torch.bool)
        m[torch.randperm(t.numel(), generator=g)[:k]] = True
        out[n] = m.reshape(t.shape)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from flipquant import maps as M
    tiles, unit, meta = M.load(a.map)
    rand = randomize(tiles, a.seed)
    assert all(int(rand[n].sum()) == int(tiles[n].sum()) and rand[n].shape == tiles[n].shape for n in tiles)
    src = hashlib.sha256(a.map.read_bytes()).hexdigest()
    keep = {k: meta.get(k) for k in ("model", "registry_key", "revision", "rotate") if k in meta}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    M.save(a.out, rand, unit, dict(keep, method="random placement: per module the real map's E0M3 count at uniformly "
                                                "random tiles (experiments/paper_1k/random_maps.py)",
                                   seed=a.seed, real_map_sha256=src))
    rec = dict(status="complete", real_map_sha256=src, seed=a.seed, modules=len(rand),
               e0m3_tiles=sum(int(t.sum()) for t in rand.values()), tiles=sum(t.numel() for t in rand.values()))
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"RANDOM {a.out}: {rec['e0m3_tiles']} E0M3 tiles of {rec['tiles']} over {rec['modules']} modules", flush=True)


if __name__ == "__main__":
    main()
