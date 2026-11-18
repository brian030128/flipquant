"""The public Hugging Face artifacts of the 1k reruns, at the revisions pinned in artifacts.json, checked by sha256.

    python -m experiments.paper_1k.artifacts fetch [--kind flipquant_1k,flipquant_256] [--models m1,m2]
    python -m experiments.paper_1k.artifacts verify       # re-hash every fetched file

Files go to the Hugging Face cache (HF_HOME), or to PAPER1K_HF_CACHE when it is set: a separate hub cache for these
artifacts only (about 160 GB), e.g. on another disk, while the model weights stay in HF_HOME. A file is used only if
its sha256 equals the pinned one. The maps of the Calib1k repositories hold the same tiles as the maps the paper's 1k
runs used (checked bit for bit at upload).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from . import config as C

MANIFEST = json.loads((C.HERE / "artifacts.json").read_text())
KIND_SUFFIX = {"flipquant_1k": "FlipQuant-Calib1k", "gptq_1k": "GPTQ-BF16Calib-NVFP4-Calib1k",
               "flipquant_256": "FlipQuant", "gptq_256": "GPTQ-BF16Calib-NVFP4"}
_checked: dict = {}


def repo(model: str, kind: str) -> str:
    return f"{C.HF_ORG}/{C.HF_BASE[model]}-{KIND_SUFFIX[kind]}"


def file_name(model: str, kind: str, unit: str | None = None) -> str:
    """The file of an artifact kind in its repository."""
    if kind.startswith("flipquant"):
        return f"flipquant_{unit}.pt"
    return "codes_nvfp4_act-fo6.pt"


def cache_dir() -> str | None:
    """The hub cache of the artifacts: PAPER1K_HF_CACHE, or None (Hugging Face's own cache, under HF_HOME)."""
    return os.environ.get("PAPER1K_HF_CACHE") or None


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned(model: str, kind: str, unit: str | None = None) -> dict:
    rid = repo(model, kind)
    name = file_name(model, kind, unit)
    r = MANIFEST["repos"][rid]
    return dict(repo=rid, revision=r["revision"], file=name, **r["files"][name])


def fetch(model: str, kind: str, unit: str | None = None, check: bool = True) -> Path:
    """The artifact's local path (downloaded at the pinned revision); raises if its sha256 is not the pinned one."""
    from huggingface_hub import hf_hub_download
    p = pinned(model, kind, unit)
    path = Path(hf_hub_download(p["repo"], p["file"], revision=p["revision"], cache_dir=cache_dir()))
    if check and str(path) not in _checked:
        got = sha256(path)
        if got != p["sha256"]:
            raise SystemExit(f"{p['repo']}@{p['revision'][:12]}/{p['file']}: sha256 {got[:16]} != pinned {p['sha256'][:16]}")
        _checked[str(path)] = got
    return path


def local(model: str, kind: str, unit: str | None = None) -> Path:
    """The path fetch() returns, without downloading or hashing (dry runs)."""
    p = pinned(model, kind, unit)
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(p["repo"], p["file"], revision=p["revision"], cache_dir=cache_dir())
    return Path(hit) if isinstance(hit, str) else Path(f"<hf:{p['repo']}@{p['revision'][:12]}/{p['file']}>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("fetch", "verify", "list"))
    ap.add_argument("--kind", default=",".join(KIND_SUFFIX))
    ap.add_argument("--models", default=",".join(C.MODELS))
    a = ap.parse_args(argv)
    for m in a.models.split(","):
        for kind in a.kind.split(","):
            units = C.UNITS if kind.startswith("flipquant") else (None,)
            for u in units:
                p = pinned(m, kind, u)
                if a.command == "list":
                    print(f"{p['repo']}@{p['revision'][:12]} {p['file']} {p['size']} B sha256 {p['sha256'][:16]}")
                    continue
                path = fetch(m, kind, u)
                print(f"ok {p['repo']}/{p['file']} -> {path}")


if __name__ == "__main__":
    main()
