"""Command-line plumbing shared by evaluation/ and calibration/: which model, which execution mode,
which format, which map."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from flipquant import if4 as if4_
from flipquant import zou_mixfp4 as zou_
from flipquant import maps as maps_
from flipquant import rotate as rotate_
from flipquant.linear import ACT_FORMATS, WEIGHT_FORMATS, QuantSpec

from .loader import _set_module, apply_nvfp4_checkpoint, load, quantizable_linears, quantize
from .registry import MODELS

#: IF4 (Cook et al.) weights, fake mode only (flipquant/if4.py): the format, and its E2M1 base (IF4's FP candidate
#: everywhere; with --e2m1 fourover6, FourOverSix everywhere)
IF4_WEIGHTS = ("if4", "if4_e2m1")
#: MixFP4 of Zou et al. (arXiv 2605.31035), fake mode only (flipquant/zou_mixfp4.py): the format, and its E2M1 base
#: (Zou's E2M1 candidate everywhere; with --e2m1 fourover6, FourOverSix everywhere). Not FlipQuant's own --weight
#: mixfp4.
ZOU_WEIGHTS = ("zou_mixfp4", "zou_mixfp4_e2m1")
#: the fake-only baselines with their own activation rule (--e2m1 / --act-method)
OWN_ACT_WEIGHTS = IF4_WEIGHTS + ZOU_WEIGHTS
assert if4_.E2M1_CANDIDATES == zou_.E2M1_CANDIDATES and if4_.ACT_METHODS == zou_.ACT_METHODS

#: ``--native-backend``: ``sm120`` and ``sm100-tuned`` are the optimized deployment paths (models/sm120.py,
#: models/sm100.py), the others FlipLinear backends
NATIVE_BACKENDS = ("auto", "sm100", "sm100-tuned", "sm120", "sm120-capi", "sm120-capi-b8", "emulate")


def native_backend(args) -> str:
    """The backend ``--mode native`` runs: ``auto`` resolved by the GPU's compute capability (sm_120 -> the optimized
    ``sm120`` path, sm_100 -> ``sm100``), anything else as given."""
    if args.native_backend != "auto":
        return args.native_backend
    from flipquant.native import detect_arch
    return detect_arch()


def add_model_args(ap: argparse.ArgumentParser, *, act_scope: str = "row", modes=("bf16", "fake", "native")):
    g = ap.add_argument_group("model and quantization")
    g.add_argument("--model", required=True, help=f"one of {sorted(MODELS)} (or an alias)")
    g.add_argument("--revision", default=None, help="override the registry's pinned revision")
    g.add_argument("--mode", choices=modes, default="fake",
                   help="bf16: unquantized reference; fake: simulated quant (any GPU, e.g. H100/H200); "
                        "native: FP4 tensor-core kernels (B200 = sm100, RTX Pro 6000 = sm120)")
    g.add_argument("--weight", choices=WEIGHT_FORMATS + IF4_WEIGHTS + ZOU_WEIGHTS, default="mixfp4",
                   help="mixfp4 needs --map; fourover6 / e0m3 are the all-E2M1 / all-E0M3 ends; if4 = IF4 (Cook et "
                        "al.) per 16-block or coarsened to --unit tiles, if4_e2m1 = its E2M1 base; zou_mixfp4 = MixFP4 "
                        "of Zou et al. (not FlipQuant's mixfp4), zou_mixfp4_e2m1 = its E2M1 base (fake mode only)")
    g.add_argument("--act", choices=ACT_FORMATS, default="fourover6")
    g.add_argument("--act-scope", choices=("row", "document", "tensor", "static"), default=act_scope,
                   help="activation global-scale scope (row = per token, document = per evaluation window, "
                        "tensor = per call, static = a fixed per-layer scale)")
    g.add_argument("--nvidia-nvfp4", action="store_true",
                   help="start from the vendor's ModelOpt NVFP4 release (registry nvfp4_checkpoint): its BF16 "
                        "tensors and BF16-kept linears, its exact weight codes (--weight modelopt, or the E2M1 "
                        "candidate of --weight mixfp4) and its static NVFP4 activation scales; a uniform "
                        "--weight (e.g. fourover6) instead quantizes the original BF16 weights of its quantized set")
    g.add_argument("--modelopt-checkpoint", default=None, metavar="REPO_OR_DIR[@REV]",
                   help="another ModelOpt NVFP4 checkpoint, e.g. our GPTQ export (calibration/gptq_modelopt.py): "
                        "its exact weight codes with --weight modelopt, but the activations stay --act / "
                        "--act-scope (FourOverSix per token by default), unlike --nvidia-nvfp4")
    g.add_argument("--map", type=Path, default=None, help="map file from calibration/train_map.py "
                                                         "(or a bare map.pt with --unit)")
    g.add_argument("--unit", default=None, help="tile unit of the map, e.g. 8x64 (read from the map file); with "
                                                "--weight if4 / zou_mixfp4, the selection granularity: 1x16 (default, "
                                                "the method's own per-block rule), 8x64, 16x64 or 256x64")
    g.add_argument("--e2m1", choices=if4_.E2M1_CANDIDATES, default="own",
                   help="--weight if4 / zou_mixfp4 (and their _e2m1): the FP candidate, the method's own E2M1 "
                        "(default) or FourOverSix "
                        "(the '+ FourOverSix' variant); weights only")
    g.add_argument("--act-method", choices=if4_.ACT_METHODS, default=None,
                   help="--weight if4 / zou_mixfp4 (and their _e2m1) only: own = the method's per-block "
                        "activation rule per token at 1x16 (Zou: Algorithm 1); "
                        "nvfp4 / fourover6 = --act nvfp4 / fourover6 --act-scope row. Default: --act / --act-scope")
    g.add_argument("--native-backend", choices=NATIVE_BACKENDS, default="auto",
                   help="auto picks by compute capability: an sm_120 GPU (RTX Pro 6000) runs the optimized sm120 path "
                        "(models/sm120.py), an sm_100 GPU (B200) the sm100 kernels; sm100-tuned = the tuned SM100 "
                        "kernels with the deployed 256x64 per-tile kernel (models/sm100.py); sm120-capi = the C-API kernels "
                        "(kernels/capi) on sm_120; emulate runs the packed buffers in torch")
    g.add_argument("--kernel-set", default="auto",
                   help="sm120 path: auto (the optimized sets; the default) | paper (the paper's kernels) | "
                        "a kernel set or build name")
    g.add_argument("--kernel-build-dir", type=Path, default=None,
                   help="sm120 path: build directory (default $SM120_BUILD_DIR, else kernels/sm120/build_deploy, the "
                        "output of kernels/sm120/build_deploy.sh)")
    g.add_argument("--ownership-check", action="store_true",
                   help="sm120 and sm100-tuned paths: check every map weight element through the kernel at install "
                        "time")
    g.add_argument("--sm100-quant", choices=("exact", "fast"), default="exact",
                   help="sm100-tuned path: the per-token activation quantizer; exact (default) = SM120's deployed "
                        "quantizer, bit-identical to the simulation; fast = the same rules with reciprocals and the "
                        "hardware E2M1 conversion (kernels/src/quant_act_fast.cuh), not bit-identical")
    g.add_argument("--any-act", action="store_true",
                   help="sm120 path: also NVFP4 weights with FourOverSix activations (Table 4's GPTQ setting)")
    g.add_argument("--fo6-tensor-qmax", type=float, default=448.0,
                   help="FourOverSix's tensor-scale constant: global scale = amax / (6 * Q). 448 (default) is this "
                        "repository's and every map's; 256 is the original FourOverSix (arXiv:2512.02010), under "
                        "which every block can map its max onto 4. Anything but 448: --weight fourover6 --act "
                        "fourover6 only (weights and per-token activations), fake or native")
    g.add_argument("--device-map", default="cuda", help="'cuda' or 'balanced' (multi-GPU)")
    g.add_argument("--fake-weight-cache", choices=("on", "off"), default="on",
                   help="fake mode: keep the dequantized BF16 weights resident (on, default) or decode them from "
                        "the packed candidates when needed (off: ~2 bytes per weight less GPU memory, same numbers)")
    g.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa", help="attention implementation")
    g.add_argument("--tiny-layers", type=int, default=None,
                   help="TESTS ONLY: the same architecture cut to N decoder layers (every layer type kept, "
                        "widths unchanged), randomly initialized; models/tiny.py")
    g.add_argument("--rotate", choices=rotate_.ROTATIONS, default="none",
                   help="hadamard: block-diagonal Hadamard on every quantized linear's input dimension "
                        "(W' = W H offline, x' = x H online before activation quantization; flipquant/rotate.py). "
                        "A --map must have been trained with the same --rotate")
    g.add_argument("--rotate-block", type=int, default=16,
                   help="largest Hadamard block (power of two; default 16 = one scale group; a layer uses the "
                        "largest power of two dividing its in_features, up to this)")
    g.add_argument("--ptq", choices=("rtn", "gptq"), default="rtn",
                   help="gptq: GPTQ on the fixed RTN grid of --weight (nvfp4 / fourover6 / mixfp4 with --map); "
                        "flipquant/gptq.py")
    g.add_argument("--gptq-damp", type=float, default=0.01, help="dampening, fraction of mean(diag H)")
    g.add_argument("--gptq-block", type=int, default=128, help="GPTQ lazy-update block (columns)")
    g.add_argument("--gptq-windows", type=int, default=None,
                   help="first N of the calibration windows (default: all of --gptq-calib's set; smoke tests use "
                        "fewer)")
    g.add_argument("--gptq-batch", type=int, default=8, help="calibration sequences per forward (1 for batch1_only)")
    g.add_argument("--gptq-act-order", action="store_true",
                   help="GPTQ in decreasing diag(H) order (static groups: the grid stays RTN's, per element); off by "
                        "default, as in NVIDIA ModelOpt's GPTQ")
    g.add_argument("--gptq-calib", choices=("record", "release", "mathcode", "wiki"), default="record",
                   help="calibration windows: record (default) = the model's reference calibration record, the "
                        "main-table maps' fit set; mathcode = flipquant.data.calibration_sets (128x512 math+code, "
                        "calibration/gptq_modelopt.py's set); wiki = 128x512 WikiText-2 TRAIN windows (diagnostics "
                        "only; checked not to overlap the test split)")
    g.add_argument("--gptq-calib-extension", type=Path, default=None,
                   help="--gptq-calib release: a FlipQuant release run's trainer_report.json; the windows are the "
                        "record's 128 then the 128 of its fit extension, the release maps' 256 x 512 fit set "
                        "(flipquant.data.calibration_release; every token sha256 checked)")
    g.add_argument("--gptq-calib-record", type=Path, default=None,
                   help="--gptq-calib record: the record's report.json (default <FLIPQUANT_CACHE>/tmopt_data/<key>/"
                        "calibration/report.json, written by python -m calibration.tmopt_ext record --model M)")
    g.add_argument("--gptq-hessian-input", choices=("quantized", "bf16"), default="quantized",
                   help="the inputs the Hessian is built from: quantized (default, as NVIDIA ModelOpt's GPTQ: the "
                        "activation quantizer's output, --act per token) or bf16 (the input before activation "
                        "quantization)")
    g.add_argument("--gptq-propagate", choices=("quantized", "bf16"), default="quantized",
                   help="the activations of the calibration forward that gives each next layer its inputs: quantized "
                        "(default, as NVIDIA ModelOpt / the co-author's runs: the layers as they will be evaluated) "
                        "or bf16 (paper-sm120-runs: every activation quantizer off, quantized weights only, as weight-"
                        "only GPTQ; needs --gptq-hessian-input bf16; the codes then do not depend on --act)")
    g.add_argument("--gptq-cache", type=Path, default=None,
                   help="GPTQ codes file: loaded if it exists (settings must match), else computed and saved")
    g.add_argument("--gptq-candidates", type=Path, nargs=2, default=None, metavar=("E2M1", "E0M3"),
                   help="opt-in, --weight mixfp4 --map M only: no GPTQ run; each tile takes its format's codes from two "
                        "GPTQ codes files made with this run's --gptq-* settings, FourOverSix codes (E2M1) and mixfp4 "
                        "codes with every tile E0M3 (flipquant.gptq.compose_candidates)")
    return g


def map_rotation_mismatch(meta: dict, rotate: str, rotate_block) -> str | None:
    """Why a map with metadata ``meta`` cannot be used under ``--rotate`` / ``--rotate-block`` (None
    if it can): a map lives in the basis it was trained in, rotation kind AND block size."""
    if meta.get("rotate", "none") != rotate:
        return f"trained with rotate={meta.get('rotate', 'none')!r}, this run uses --rotate {rotate}"
    if rotate != "none" and meta.get("rotate_block") != rotate_block:
        return (f"trained with rotate_block={meta.get('rotate_block')!r}, "
                f"this run uses --rotate-block {rotate_block}")
    return None
def resolve_act_method(args, argv=None) -> None:
    """``--act-method``: only with an IF4 or Zou MixFP4 weight; ``nvfp4`` / ``fourover6`` set ``--act`` and ``--act-scope row`` (an
    explicit conflicting ``--act`` / ``--act-scope`` is an error). Idempotent; a no-op without the flag."""
    import sys
    method = getattr(args, "act_method", None)
    weight = getattr(args, "weight", None)
    if weight not in OWN_ACT_WEIGHTS:
        if method is not None or getattr(args, "e2m1", "own") != "own":
            raise SystemExit("--act-method and --e2m1 go with --weight if4 / if4_e2m1 / zou_mixfp4 / zou_mixfp4_e2m1")
        return
    if method is None or getattr(args, "_act_method_resolved", False):
        return
    argv = sys.argv[1:] if argv is None else argv
    given = lambda flag: any(a == flag or a.startswith(flag + "=") for a in argv)  # noqa: E731
    if method != "own" and ((given("--act") and args.act != method) or (given("--act-scope") and args.act_scope != "row")):
        raise SystemExit(f"--act-method {method} is --act {method} --act-scope row; drop the conflicting flags")
    if method == "own" and (given("--act") or given("--act-scope")):
        raise SystemExit(f"--act-method own is {'IF4' if weight in IF4_WEIGHTS else 'Zou'}'s activations, per "
                         "token; drop --act / --act-scope")
    if method != "own":
        args.act = method
    args.act_scope = "row"
    args._act_method_resolved = True


def build(args, *, trainable: bool = False):
    """Load the model and install FlipLinear layers per ``args``. Returns
    ``(model, tok, spec, layers, info)``; ``layers`` is empty in bf16 mode."""
    model, tok, spec = load(args.model, revision=args.revision, device_map=args.device_map, attn=args.attn,
                            tiny_layers=getattr(args, "tiny_layers", None))
    layers, info = apply_quant(model, spec, args, trainable=trainable, tok=tok)
    return model, tok, spec, layers, info


def apply_quant(model, spec, args, *, trainable: bool = False, tok=None):
    """The quantization half of :func:`build`, on a model from :func:`models.loader.load`. Returns ``(layers, info)``.
    ``tok`` (the model's tokenizer) is needed by ``--ptq gptq`` only; it is loaded if not given."""
    info = dict(model=spec.key, source=model.flipquant_source, mode=args.mode)
    rotate = getattr(args, "rotate", "none")
    ptq = getattr(args, "ptq", "rtn")
    rotations = None
    if rotate != "none":
        if args.nvidia_nvfp4:
            raise SystemExit("--rotate needs our own quantization of the BF16 weights, not --nvidia-nvfp4")
        rotations = rotate_.rotate_model(model, spec, args.rotate_block)
        info["rotate"] = dict(kind=rotate, max_block=args.rotate_block, **rotate_.summary(rotations))
    resolve_act_method(args)
    tq = getattr(args, "fo6_tensor_qmax", 448.0)
    if tq != 448.0:
        if args.mode == "bf16" or (args.weight, args.act) != ("fourover6", "fourover6"):
            raise SystemExit(f"--fo6-tensor-qmax {tq:g}: the FourOverSix baseline only (--weight fourover6 --act "
                             f"fourover6, fake or native)")
        if args.map is not None or ptq != "rtn" or rotate != "none" or args.nvidia_nvfp4 or \
                getattr(args, "modelopt_checkpoint", None) or getattr(args, "act_override", None):
            raise SystemExit(f"--fo6-tensor-qmax {tq:g}: not with --map, --ptq, --rotate, ModelOpt checkpoints or "
                             f"--act-override")
    if args.mode == "bf16":
        if ptq != "rtn":
            raise SystemExit("--ptq gptq quantizes weights; use --mode fake or native")
        return {}, info
    if args.weight in IF4_WEIGHTS:
        return _apply_if4(model, spec, args, info, trainable)
    if args.weight in ZOU_WEIGHTS:
        return _apply_zou(model, spec, args, info, trainable)
    backend = native_backend(args) if args.mode == "native" else None
    if backend == "sm120" and (args.nvidia_nvfp4 or (getattr(args, "modelopt_checkpoint", None)
                                                     and args.weight != "modelopt")):
        raise SystemExit("the sm120 path does not run ModelOpt checkpoints (except their exact codes with --weight "
                         "modelopt, packed as they are); use --native-backend sm120-capi")
    given = None
    if args.nvidia_nvfp4 and getattr(args, "modelopt_checkpoint", None):
        raise SystemExit("--nvidia-nvfp4 and --modelopt-checkpoint are exclusive")
    if args.nvidia_nvfp4:
        args.act, args.act_scope = "nvfp4", "static"
        ckpt = apply_nvfp4_checkpoint(model, spec)
        given = ckpt["given"]
        info["nvfp4_checkpoint"] = ckpt["info"]
    elif getattr(args, "modelopt_checkpoint", None):
        repo, _, rev = args.modelopt_checkpoint.partition("@")
        ckpt = apply_nvfp4_checkpoint(model, spec, (repo, rev or None))
        given = ckpt["given"]
        info["nvfp4_checkpoint"] = ckpt["info"]
    elif args.weight == "modelopt":
        raise SystemExit("--weight modelopt needs --nvidia-nvfp4 or --modelopt-checkpoint")
    tiles, unit = None, args.unit or "8x64"
    if args.map is not None:
        tiles, unit, meta = maps_.load(args.map, args.unit)
        info["map"] = dict(path=str(args.map), unit=unit, meta=meta, **maps_.summary(tiles))
        problem = map_rotation_mismatch(meta, rotate, getattr(args, "rotate_block", None))
        if problem:
            raise SystemExit(f"{args.map}: {problem}")
    if backend == "sm120":
        from .sm120 import install as sm120_install
        from .sm120 import native_modules
        codes = linears = None
        if given is not None:                       # --modelopt-checkpoint --weight modelopt: its exact codes
            codes = {n: g["codes"] for n, g in given.items() if g.get("codes") is not None}
            from .loader import quantizable_linears
            linears = {n: lin for n, lin in quantizable_linears(model, spec).items() if n in codes}
        elif ptq == "gptq":
            # GPTQ's codes (computed with fake-quant layers, or read from --gptq-cache), then packed as they are
            codes = {}
            fake = QuantSpec(weight=args.weight, act=args.act, act_scope="row", unit=unit, mode="fake")
            linears, info["ptq"] = _gptq(model, spec, args, fake, tiles, rotate, tok, info, codes_out=codes)
        try:
            rep = sm120_install(model, spec, weight="nvfp4" if args.weight == "modelopt" else args.weight,
                                tiles=tiles, unit=unit if tiles is not None else None,
                                kernel=args.kernel_set, act=args.act, act_scope=args.act_scope,
                                build_root=args.kernel_build_dir, ownership=args.ownership_check,
                                activation_quantizer=getattr(args, "act_override", None), codes=codes,
                                linears=linears, rotations=rotations,
                                any_act=getattr(args, "any_act", False) or args.weight == "modelopt",
                                fo6_tensor_qmax=tq)
        except ValueError as e:
            raise SystemExit(f"--mode native (sm120): {e}")
        info["quant"] = dict(backend="sm120", weight=args.weight, act=args.act, act_scope=args.act_scope, unit=rep["unit"])
        if tq != 448.0:
            info["quant"]["fo6_tensor_qmax"] = tq
        info["native"] = rep
        info["quantized_modules"] = rep["modules"]
        info["native_backend"] = "sm120"
        return native_modules(model), info
    if backend == "sm100-tuned":
        from .sm100 import install as sm100_install
        from .sm100 import native_modules as sm100_modules
        if given is not None or ptq == "gptq" or rotations is not None or args.act_scope != "row":
            raise SystemExit("--native-backend sm100-tuned runs RTN weights with per-token activations only")
        from .sm100 import ACT as SM100_ACT
        quantizer = getattr(args, "act_override", None)
        if getattr(args, "sm100_quant", "exact") == "fast":
            quantizer = (quantizer or SM100_ACT.get(args.act, args.act)) + "_fast"
        try:
            rep = sm100_install(model, spec, weight=args.weight, tiles=tiles, unit=unit if tiles is not None else None,
                                act=args.act, activation_quantizer=quantizer, ownership=args.ownership_check)
        except ValueError as e:
            raise SystemExit(f"--mode native (sm100-tuned): {e}")
        info["quant"] = dict(backend="sm100-tuned", weight=args.weight, act=args.act, act_scope=args.act_scope,
                             unit=rep["unit"])
        info["native"] = rep
        info["quantized_modules"] = rep["modules"]
        info["native_backend"] = "sm100-tuned"
        return sm100_modules(model), info
    qspec = QuantSpec(weight=args.weight, act=args.act, act_scope=args.act_scope, unit=unit, mode=args.mode,
                      native_backend=backend or args.native_backend,
                      fake_weight_cache=getattr(args, "fake_weight_cache", "on") == "on", fo6_tensor_qmax=tq)
    if ptq == "gptq":
        if given is not None or trainable:
            raise SystemExit("--ptq gptq builds frozen layers from our own BF16 weights (no --nvidia-nvfp4)")
        layers, info["ptq"] = _gptq(model, spec, args, qspec, tiles, rotate, tok, info)
    else:
        layers = quantize(model, spec, qspec, maps=tiles, trainable=trainable, given=given)
        if rotations is not None:
            rotate_.attach(layers, rotations)
    info["quant"] = qspec.as_dict()
    info["quantized_modules"] = len(layers)
    if args.mode == "native":
        info["native_backend"] = next(iter(layers.values()))._backend.name if layers else None
    return layers, info


def _gptq(model, spec, args, qspec, tiles, rotate, tok, info, codes_out=None):
    """``flipquant.gptq.quantize_model`` with this CLI's --gptq-* settings. Returns ``(layers, ptq info)``."""
    from flipquant import gptq
    from flipquant.data import calibration_from_record, calibration_sets, wiki_train_windows
    if spec.batch1_only and args.gptq_batch != 1:
        raise SystemExit(f"{spec.key}: batched forward differs from batch 1; use --gptq-batch 1")
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=info["source"].get("revision") or spec.revision)
    calib = getattr(args, "gptq_calib", "record")
    if calib in ("record", "release"):
        record = getattr(args, "gptq_calib_record", None)
        if record is None:
            from calibration.tmopt_common import MODEL_KEYS, data_root
            if spec.key in MODEL_KEYS:
                default = data_root() / MODEL_KEYS[spec.key] / "calibration" / "report.json"
                record = default if default.exists() else None
        if record is None:
            raise SystemExit("--gptq-calib record needs --gptq-calib-record (the model's reference calibration "
                             "record, report.json: the main-table maps' fit set; python -m calibration.tmopt_ext record)")
        if calib == "release":
            from flipquant.data import calibration_release
            if getattr(args, "gptq_calib_extension", None) is None:
                raise SystemExit("--gptq-calib release needs --gptq-calib-extension (a release run's trainer_report.json)")
            source = calibration_release(tok, record, args.gptq_calib_extension)
            info["gptq_calibration"] = dict(trainer_report=str(args.gptq_calib_extension), record=str(record),
                                            windows=len(source["fit"]))
        else:
            source = calibration_from_record(tok, record)
    else:
        source = wiki_train_windows(tok) if calib == "wiki" else calibration_sets(tok)
    windows = source["fit"][:args.gptq_windows] if args.gptq_windows else source["fit"]
    candidates = getattr(args, "gptq_candidates", None)
    if candidates is not None and (args.weight != "mixfp4" or tiles is None or args.gptq_cache is not None):
        raise SystemExit("--gptq-candidates goes with --weight mixfp4 --map M and no --gptq-cache")
    return gptq.quantize_model(
        model, spec, qspec, windows, maps=tiles if args.weight == "mixfp4" else None, damp=args.gptq_damp,
        block=args.gptq_block, batch=args.gptq_batch, cache=args.gptq_cache, rotate=rotate,
        act_order=getattr(args, "gptq_act_order", False), calib=calib,
        hessian_input=getattr(args, "gptq_hessian_input", "quantized"), codes_out=codes_out,
        propagate=getattr(args, "gptq_propagate", "quantized"), candidates=candidates)
def _apply_if4(model, spec, args, info, trainable):
    """``--weight if4 / if4_e2m1``: :mod:`flipquant.if4` fake quant of every quantizable linear."""
    from .loader import _set_module, quantizable_linears
    if args.mode != "fake":
        raise SystemExit("--weight if4 runs in fake mode only (--mode fake)")
    if trainable:
        raise SystemExit("--weight if4 has no calibration")
    if args.map is not None or args.nvidia_nvfp4 or getattr(args, "modelopt_checkpoint", None):
        raise SystemExit("--weight if4 takes no --map, --nvidia-nvfp4 or --modelopt-checkpoint")
    unit = args.unit or "1x16"
    if args.weight == "if4_e2m1" and unit != "1x16":
        raise SystemExit("--weight if4_e2m1 has no tiles (it never selects INT4); drop --unit")
    act = args.act_method if args.act_method == "own" else args.act
    if act != "own" and args.act_scope not in ("row", "document", "tensor"):
        raise SystemExit(f"--weight if4: activation scope {args.act_scope!r} is not supported")
    layers, report = if4_.install(model, quantizable_linears(model, spec), unit=unit, e2m1=args.e2m1,
                                  base=args.weight == "if4_e2m1", act=act,
                                  act_scope="row" if act == "own" else args.act_scope, set_module=_set_module)
    info["quant"] = dict(weight=args.weight, unit=unit, e2m1=args.e2m1, act_method=args.act_method, act=act,
                         act_scope="row" if act == "own" else args.act_scope, mode="fake")
    info["quantized_modules"] = len(layers)
    info["adaptive"] = report
    return layers, info


def _apply_zou(model, spec, args, info, trainable):
    """``--weight zou_mixfp4 / zou_mixfp4_e2m1``: :mod:`flipquant.zou_mixfp4` fake quant of every quantizable linear."""
    from .loader import _set_module, quantizable_linears
    if args.mode != "fake":
        raise SystemExit("--weight zou_mixfp4 runs in fake mode only (--mode fake)")
    if trainable:
        raise SystemExit("--weight zou_mixfp4 has no calibration")
    if args.map is not None or args.nvidia_nvfp4 or getattr(args, "modelopt_checkpoint", None):
        raise SystemExit("--weight zou_mixfp4 takes no --map, --nvidia-nvfp4 or --modelopt-checkpoint")
    unit = args.unit or "1x16"
    if args.weight == "zou_mixfp4_e2m1" and unit != "1x16":
        raise SystemExit("--weight zou_mixfp4_e2m1 has no tiles (it never selects E1M2); drop --unit")
    act = args.act_method if args.act_method == "own" else args.act
    if act != "own" and args.act_scope not in ("row", "document", "tensor"):
        raise SystemExit(f"--weight zou_mixfp4: activation scope {args.act_scope!r} is not supported")
    layers, report = zou_.install(model, quantizable_linears(model, spec), unit=unit, e2m1=args.e2m1,
                                  base=args.weight == "zou_mixfp4_e2m1", act=act,
                                  act_scope="row" if act == "own" else args.act_scope, set_module=_set_module)
    info["quant"] = dict(weight=args.weight, unit=unit, e2m1=args.e2m1, act_method=args.act_method, act=act,
                         act_scope="row" if act == "own" else args.act_scope, mode="fake")
    info["quantized_modules"] = len(layers)
    info["adaptive"] = report
    return layers, info


def _json_default(o):
    """Live records (``flipquant.if4.Report``, ``flipquant.zou_mixfp4.Report``) serialize through ``as_dict``;
    everything else as before (``str``)."""
    return o.as_dict() if isinstance(o, (if4_.Report, zou_.Report)) else str(o)
def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    tmp.replace(path)


def resources(started: float) -> dict:
    n = torch.cuda.device_count()
    return dict(seconds=time.time() - started, job_id=os.environ.get("SLURM_JOB_ID"),
                gpus=[torch.cuda.get_device_name(i) for i in range(n)],
                gpu_peak_gib=[torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(n)])
