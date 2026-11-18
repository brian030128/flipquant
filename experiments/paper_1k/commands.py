"""Command normalization for the port's equivalence checks: a run's argv with its machine-specific paths replaced by
placeholders, so that the commands this driver builds can be compared, token for token, with the commands the
paper-eval runs executed (reference/commands_<item>.json, taken from those runs' job logs).

Placeholders: {PY} / {PY_FAST} (the eval / hybrid-model interpreter), {SCRIPT:<name>} (a script of this repository or
of the original driver), {BUILD} (the kernel build directory), {MAP:<kind>:<model>:<unit>} (a FlipQuant map, as --map
or --match; kind flipquant_1k or flipquant_256), {EXT:flipquant_256:<model>:<unit>} (a release map's trainer report as
--calib-extension or --gptq-calib-extension), {DATA}/<path> (below the calibration data root), {OUT}/<path> (below the
item's run directory), or {OUT} alone for every --out / --samples-out value when the comparison leaves the output
layout out (drop_out); further prefixes map to placeholders through `roots`.
"""
from __future__ import annotations

import re

from . import config as C

_KIND = {"FlipQuant-Calib1k": "flipquant_1k", "FlipQuant": "flipquant_256"}
_BASE_TO_MODEL = {v: k for k, v in C.HF_BASE.items()}


def _map(path: str) -> str:
    m = re.search(r"calib_size/([^/]+)/fq(8|16|256)_1024/map\.pt$", path)
    if m:
        return f"{{MAP:flipquant_1k:{m.group(1)}:{ {'8': '8x64', '16': '16x64', '256': '256x64'}[m.group(2)] }}}"
    m = re.search(r"(?:flipquant_release|llama_instruct)/([^/]+)/(?:maps/)?flipquant_(\w+)\.pt$", path)
    if m:
        return f"{{MAP:flipquant_256:{m.group(1)}:{m.group(2)}}}"
    m = re.search(r"edgeai-lab/(.+?)-(FlipQuant-Calib1k|FlipQuant)@[0-9a-f]+/flipquant_(\w+)\.pt>?$", path)
    if m:
        return f"{{MAP:{_KIND[m.group(2)]}:{_BASE_TO_MODEL[m.group(1)]}:{m.group(3)}}}"
    m = re.search(r"models--edgeai-lab--(.+?)-(FlipQuant-Calib1k|FlipQuant)/snapshots/[0-9a-f]+/flipquant_(\w+)\.pt$", path)
    if m:
        return f"{{MAP:{_KIND[m.group(2)]}:{_BASE_TO_MODEL[m.group(1)]}:{m.group(3)}}}"
    return path


_ART = {"FOCUS-NVFP4-Calib1k": "focus_1k", "FOCUS-NVFP4": "focus_256", "GPTQ-BF16Calib-NVFP4-Calib1k": "gptq_1k",
        "GPTQ-BF16Calib-NVFP4": "gptq_256"}


def _art(path: str) -> str:
    """A FOCUS state or a GPTQ‡ codes file: {ART:<kind>:<model>}."""
    m = re.search(r"calib_size/([^/]+)/(focus|gptq)_1024/(?:state/focus\.pt|codes_nvfp4_act-fo6\.pt)$", path)
    if m:
        return f"{{ART:{m.group(2)}_1k:{m.group(1)}}}"
    m = re.search(r"edgeai-lab[/-]{1,2}(.+?)-(FOCUS-NVFP4(?:-Calib1k)?|GPTQ-BF16Calib-NVFP4(?:-Calib1k)?)(?:@[0-9a-f]+|/snapshots/[0-9a-f]+)/"
                  r"(?:[^/]*focus\.pt|codes_nvfp4_act-fo6\.pt)>?$", path)
    if m:
        return f"{{ART:{_ART[m.group(2)]}:{_BASE_TO_MODEL[m.group(1)]}}}"
    return path


def _ext(path: str) -> str:
    m = re.search(r"flipquant_release/([^/]+)/records/(\w+)/trainer_report\.json$", path)
    return f"{{EXT:flipquant_256:{m.group(1)}:{m.group(2)}}}" if m else path


def _under(path: str, root: str | None) -> bool:
    return bool(root) and (path == root or path.startswith(root.rstrip("/") + "/"))


def normalize(argv: list[str], out_root: str, item_dirs: tuple = (), data_root: str | None = None,
              drop_out: bool = False, roots: dict | None = None) -> list[str]:
    """argv with placeholders; out_root is the run directory whose records the --out values are below; item_dirs are
    path components dropped right after it (the new driver's per-item directory, e.g. 'item2'); data_root is the
    calibration data root; drop_out writes every --out and --samples-out value as {OUT}; roots maps further path
    prefixes to placeholders (e.g. an earlier run's code-file directory to {OUT}/codes)."""
    a = [str(x) for x in argv]
    for prefix, ph in (roots or {}).items():
        a = [ph + t[len(prefix.rstrip("/")):] if _under(t, prefix) else t for t in a]

    def below_out(t):
        tail = t[len(out_root):].lstrip("/") if t.startswith(out_root) else t
        parts = tail.split("/")
        while parts and parts[0] in item_dirs:
            parts = parts[1:]
        return "{OUT}/" + "/".join(parts)

    out = ["{PY_FAST}" if "fast" in a[0] else "{PY}"]      # n16k64-fast (the hybrid models) or n16k64
    for i in range(1, len(a)):
        t, prev = a[i], a[i - 1]
        if prev == "--kernel-build-dir":
            out.append("{BUILD}")
        elif prev == "--out" or (prev == "--samples-out" and drop_out):
            out.append("{OUT}" if drop_out else below_out(t))
        elif prev in ("--maps256", "--maps1k"):         # experiments.paper_1k.ktile: PATH,PATH,PATH
            out.append(",".join(_map(q) for q in t.split(",")))
        elif (prev in ("--map", "--match") or (t.startswith("<hf:") and prev != "--maps")) and _map(t) != t:
            out.append(_map(t))
        elif prev in ("--focus", "--gptq-cache") and _art(t) != t:
            out.append(_art(t))
        elif prev == "--maps":                         # evaluation.gemm_time: UNIT=PATH,...
            def one(q):
                return _map(q) if _map(q) != q else below_out(q) if _under(q, out_root) else q
            out.append(",".join(f"{u}={one(q)}" for u, q in (x.split("=", 1) for x in t.split(","))))
        elif prev in ("--calib-extension", "--gptq-calib-extension") and _ext(t) != t:
            out.append(_ext(t))
        elif _under(t, data_root):
            out.append("{DATA}" + t[len(data_root.rstrip("/")):])
        elif _under(t, out_root):
            out.append(below_out(t))
        elif t.endswith(".py") and "/" in t:
            out.append("{SCRIPT:" + t.rsplit("/", 1)[1] + "}")
        else:
            out.append(t)
    return out
