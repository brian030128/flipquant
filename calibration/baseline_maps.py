"""Baseline MixFP4 maps for the map-selection ablation: per layer, exactly as many E0M3 tiles as a
given FlipQuant map (``--match``), chosen without training.

Naming follows the code (flipquant/linear.py): candidate ``a`` = E0M3, candidate ``b`` = FourOverSix
E2M1, and a map bit is True where the tile is E0M3. For every quantized linear ``l`` the matched map
has ``k_l`` E0M3 tiles at the same unit; each method ranks the tiles of ``l`` and takes its top ``k_l``
(ties broken by tile index, row-major), so every baseline flips exactly ``k_l`` tiles in every layer.

  random   ``k_l`` tiles uniformly at random (``--seed``). Needs no model.
  act      activation-weighted layer-output error, from the inputs of the layer alone. With ``X`` the
           BF16 inputs of the linear on the calibration windows and ``H_cc = X_c^T X_c`` the diagonal
           ``cols x cols`` blocks of ``X^T X`` (c = the tile's column block),
               s_u = ||(P_u o (W - B)) X^T||^2 - ||(P_u o (W - A)) X^T||^2
                   = sum_{r in u} e_B[r, c] H_cc e_B[r, c]^T - e_A[r, c] H_cc e_A[r, c]^T,
           the drop in layer-output error when tile ``u`` goes from E2M1 to E0M3; top ``k_l`` by
           ``s_u``. ``A``/``B`` are FlipLinear's own candidates (``_decoded("a")`` / ``_decoded("b")``).
           If fewer than ``k_l`` tiles have ``s_u > 0`` the top ``k_l`` are still taken; the report
           records how many were positive.
  oneshot  the map gradient dKL/dm of calibration/train_map_legacy.py at its starting point (all-E2M1, i.e.
           ``--init-logit -1``), accumulated once over every fit window with the same KL loss, data,
           per-token FourOverSix activations and execution mode, with no optimizer step; top ``k_l``
           by ``-grad`` (the tiles whose flip to E0M3 decreases the KL fastest). The report records
           how many gradients were negative.

The calibration windows are train_map_legacy's fit set (``calibration_sets_extended``; 128 x 512 math+code by
default), or with ``--calib-record`` the exact windows of an the reference calibration record -- the fit set of
the main-table maps (calibration.train_map), which Table 5's ``act`` maps use (experiments/paper_1k item 3;
its one-shot maps come from calibration.tmopt_ext, the TM-OPT+TC trainer). Output: a map file (flipquant.maps
format, loadable by ``--map``) and ``<out>.json``.

  python -m calibration.baseline_maps --model nemotron-nano-9b-v2 --method random --seed 0 \\
      --match maps/nemotron-nano-9b-v2/mixfp4_8x64.pt --out runs/ablation/nemotron_random0_8x64.pt
  python -m calibration.baseline_maps --model nemotron-nano-9b-v2 --method act \\
      --match maps/nemotron-nano-9b-v2/mixfp4_8x64.pt --out runs/ablation/nemotron_act_8x64.pt
  python -m calibration.baseline_maps --model nemotron-nano-9b-v2 --method oneshot --mode native \\
      --match maps/nemotron-nano-9b-v2/mixfp4_8x64.pt --out runs/ablation/nemotron_oneshot_8x64.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path

import torch

from flipquant import maps as maps_
from flipquant.data import calibration_from_record, calibration_release, calibration_sets_extended
from flipquant.linear import FlipLinear, QuantSpec
from models import get, load, quantizable_linears, quantize
from models.cli import resources, save_json
from calibration.train_map_legacy import DiskTeacher, kl_backward, teacher_logprobs

METHODS = ("random", "act", "oneshot")


def linear_shapes(spec, revision=None, tiny_layers=None) -> dict[str, tuple[int, int]]:
    """``{name: (out, in)}`` of the BF16 model's quantizable linears, from its config alone (the model
    is instantiated on the meta device; no weights are read)."""
    from models.tiny import meta_linear_shapes
    return meta_linear_shapes(spec, revision, tiny_layers)


def check_match(match: dict, shapes: dict, unit) -> None:
    """The matched map must cover exactly the model's quantizable linears, at ``unit``. (A map trained
    on a vendor NVFP4 release covers only that release's subset and is rejected here.)"""
    if set(match) != set(shapes):
        raise SystemExit(f"--match covers {len(match)} modules, the BF16 model quantizes {len(shapes)}; "
                         f"e.g. {sorted(set(match) ^ set(shapes))[:3]}")
    for n, (o, k) in shapes.items():
        if tuple(match[n].shape) != maps_.tile_shape(o, k, unit):
            raise SystemExit(f"{n}: --match shape {tuple(match[n].shape)} does not fit {o}x{k} at {unit}")


# ------------------------------------------------------------------------------------------------
# selection


def top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Bool mask (shape of ``scores``) of the ``k`` largest scores; ties go to the lower flat index."""
    flat = scores.detach().double().cpu().flatten()
    order = torch.sort(flat, descending=True, stable=True).indices
    mask = torch.zeros(flat.numel(), dtype=torch.bool)
    mask[order[:k]] = True
    return mask.reshape(scores.shape)


def random_map(match: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """``k_l`` uniformly random tiles per layer. Each layer draws from its own generator, seeded by
    ``seed`` and the layer's position in the map, so a layer's tiles do not depend on the others."""
    out = {}
    for i, (name, m) in enumerate(match.items()):
        g = torch.Generator().manual_seed(seed * 1_000_003 + i)
        flat = torch.zeros(m.numel(), dtype=torch.bool)
        flat[torch.randperm(m.numel(), generator=g)[:int(m.sum())]] = True
        out[name] = flat.reshape(m.shape)
    return out


# ------------------------------------------------------------------------------------------------
# activation-weighted


@torch.no_grad()
def input_gram_blocks(model, linears: dict, windows, device, batch: int, width: int) -> dict[str, torch.Tensor]:
    """Per linear, the diagonal ``width x width`` blocks of ``X^T X`` over all calibration tokens:
    ``[K / width, width, width]`` fp32 (``X`` = the BF16 input of the linear)."""
    grams, hooks = {}, []

    def hook(name):
        def fn(mod, args):
            x = args[0].detach().reshape(-1, mod.in_features).float()
            x = x.reshape(x.shape[0], mod.in_features // width, width).transpose(0, 1)     # [C, T, w]
            g = grams.get(name)
            if g is None:
                grams[name] = torch.bmm(x.transpose(1, 2), x)
            else:
                g.baddbmm_(x.transpose(1, 2), x)
        return fn
    for name, lin in linears.items():
        if lin.in_features % width:
            raise ValueError(f"{name}: in_features {lin.in_features} is not a multiple of the tile width {width}")
        hooks.append(lin.register_forward_pre_hook(hook(name)))
    try:
        for start in range(0, len(windows), batch):
            ids = torch.cat(windows[start:start + batch]).to(device)
            model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    missing = set(linears) - set(grams)
    if missing:
        raise RuntimeError(f"no inputs seen for {len(missing)} linears, e.g. {sorted(missing)[:3]}")
    return grams


def tile_quadratic(e: torch.Tensor, gram: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """``sum_{r in u} e[r, c] G_c e[r, c]^T`` for every ``rows x cols`` tile ``u`` (= ``||(P_u o e) X^T||^2``
    when ``G_c`` are the diagonal blocks of ``X^T X``). ``e``: ``[O, K]``; ``gram``: ``[K/cols, cols, cols]``."""
    o, k = e.shape
    eb = e.float().reshape(o, k // cols, cols).transpose(0, 1)                # [C, O, w]
    per_row = (torch.bmm(eb, gram.float()) * eb).sum(-1).T                    # [O, C]
    return maps_.reduce(per_row, rows, 1)


@torch.no_grad()
def act_scores(lin: torch.nn.Linear, gram: torch.Tensor, unit) -> torch.Tensor:
    """``s_u`` (see the module docstring) with FlipLinear's own candidates A (E0M3) and B (E2M1)."""
    rows, cols = maps_.parse_unit(unit)
    layer = FlipLinear.from_linear(lin, QuantSpec(weight="mixfp4", act="fourover6", unit=maps_.unit_name(unit),
                                                  mode="fake"), trainable=True)
    w = lin.weight.float()
    e_a = w - layer._decoded("a").float()
    e_b = w - layer._decoded("b").float()
    del layer
    return tile_quadratic(e_b, gram, rows, cols) - tile_quadratic(e_a, gram, rows, cols)


# ------------------------------------------------------------------------------------------------
# one-shot gradient


def oneshot_gradients(model, layers: dict, windows, teacher, device, batch: int) -> tuple[dict, list]:
    """dKL/dm for every tile at the layers' current map, accumulated over all ``windows`` (the
    gradient of the mean per-sequence KL); no parameter is changed. Returns ``(grads, kl_values)``."""
    for l in layers.values():
        l.map_grad = torch.zeros(l.mask.shape, dtype=torch.float32, device=l.gscale.device)
    kls = []
    try:
        for start in range(0, len(windows), batch):
            idx = range(start, min(start + batch, len(windows)))
            ids = torch.cat([windows[i] for i in idx]).to(device)
            kls += kl_backward(model, ids, torch.cat([teacher[i] for i in idx]), len(windows))
        return {n: l.map_grad.detach().clone() for n, l in layers.items()}, kls
    finally:
        for l in layers.values():
            l.map_grad = None


# ------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--match", type=Path, required=True, help="FlipQuant map whose per-layer E0M3 counts k_l to match")
    ap.add_argument("--unit", default=None, help="tile unit; must equal the --match map's (default: read from it)")
    ap.add_argument("--seed", type=int, default=0, help="random: the draw (the paper averages seeds 0, 1, 2)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mode", choices=("fake", "native"), default="fake",
                    help="oneshot: execution mode of the student forward (as train_map_legacy --mode)")
    ap.add_argument("--native-backend", choices=("auto", "sm100", "sm120-capi", "sm120-capi-b8", "emulate"), default="auto")
    ap.add_argument("--fake-weight-cache", choices=("on", "off"), default="on",
                    help="fake mode: keep the dequantized BF16 weights resident (on, default) or decode them from "
                         "the packed candidates when needed (off: ~2 bytes per weight less GPU memory, same numbers)")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--tiny-layers", type=int, default=None,
                    help="TESTS ONLY: random-init model cut to N layers (models/tiny.py); --match must use its names")
    ap.add_argument("--batch", type=int, default=8, help="calibration sequences per forward (act, oneshot)")
    ap.add_argument("--fit-windows", type=int, default=128, help="train_map_legacy's fit set size (even, >= 128)")
    ap.add_argument("--calib-record", type=Path, default=None,
                    help="act / oneshot: the windows of this the reference calibration record (report.json; the "
                         "main-table maps' fit set, python -m calibration.tmopt_ext record) instead of train_map's")
    ap.add_argument("--calib-extension", type=Path, default=None,
                    help="act / oneshot, with --calib-record: the FlipQuant release maps' fit set -- the record's 128 "
                         "windows then the 128 of the release trainer's fit extension, rebuilt from this "
                         "trainer_report.json (flipquant.data.calibration_release; token sha256 checked)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="use only the first N fit windows (smoke tests; the paper uses all)")
    ap.add_argument("--teacher-dir", type=Path, default=None,
                    help="oneshot: teacher log-probs on disk here (default: in memory for <= 128 windows)")
    ap.add_argument("--out", type=Path, required=True, help="output map file (.pt); the report goes to <out>.json")
    args = ap.parse_args(argv)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False

    match, unit, match_meta = maps_.load(args.match, args.unit)
    rows, cols = maps_.parse_unit(unit)
    spec = get(args.model)
    from models.mapcheck import model_key
    if model_key(match_meta) not in (None, spec.key):
        raise SystemExit(f"--match was trained for {model_key(match_meta)}, not {spec.key}")
    if args.calib_record is not None and args.fit_windows != 128:
        raise SystemExit("--calib-record holds 128 windows; drop --fit-windows")
    check_match(match, linear_shapes(spec, args.revision, args.tiny_layers), unit)
    k = {n: int(m.sum()) for n, m in match.items()}
    report = dict(status="running", method=args.method, model=spec.key, unit=unit,
                  match=dict(path=str(args.match), meta=match_meta, **maps_.summary(match)),
                  args={key: str(v) for key, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.out.with_suffix(".json")

    per_layer = {n: dict(k=k[n], tiles=m.numel()) for n, m in match.items()}
    if args.method == "random":
        tiles = random_map(match, args.seed)
    else:
        model, tok, spec = load(spec, revision=args.revision, device_map=args.device_map, tiny_layers=args.tiny_layers)
        report["source"] = model.flipquant_source
        if spec.batch1_only and args.batch != 1:
            raise SystemExit(f"{spec.key}: batched forward is not identical to batch 1; use --batch 1")
        linears = quantizable_linears(model, spec)
        check_match(match, {n: (l.out_features, l.in_features) for n, l in linears.items()}, unit)
        if args.calib_extension is not None:
            if args.calib_record is None:
                raise SystemExit("--calib-extension extends --calib-record's windows; give both")
            release = calibration_release(tok, args.calib_record, args.calib_extension)
            fit = release["fit"][:args.max_windows]
            report["data_release"] = dict(trainer_report=str(args.calib_extension),
                                          token_sha256=release["meta"]["token_sha256"][:len(fit)])
        elif args.calib_record is not None:
            fit = calibration_from_record(tok, args.calib_record)["fit"][:args.max_windows]
        else:
            fit = calibration_sets_extended(tok, args.fit_windows)["fit"][:args.max_windows]
        report["data"] = dict(fit_windows=len(fit), tokens=sum(w.numel() for w in fit),
                              record=None if args.calib_record is None else str(args.calib_record),
                              windows_sha256=hashlib.sha256(b"".join(w.numpy().tobytes() for w in fit)).hexdigest())
        device = model.get_input_embeddings().weight.device
        tiles = {}
        if args.method == "act":
            grams = input_gram_blocks(model, linears, fit, device, args.batch, cols)
            for n, lin in linears.items():
                gram = grams.pop(n)
                s = act_scores(lin, gram, unit).cpu()
                tiles[n] = top_k(s, k[n])
                per_layer[n].update(input_energy=float(gram.diagonal(dim1=1, dim2=2).sum()),
                                    positive=int((s > 0).sum()),
                                    selected_min_score=float(s[tiles[n]].min()) if k[n] else None)
        else:
            del linears                                     # do not pin the BF16 weights (see below)
            teacher_dir = args.teacher_dir
            if teacher_dir is None and len(fit) > 128:
                teacher_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"teacher_{os.getpid()}"
            teacher = teacher_logprobs(model, fit, device, DiskTeacher(teacher_dir) if teacher_dir else None)
            qspec = QuantSpec(weight="mixfp4", act="fourover6", act_scope="row", unit=unit, mode=args.mode,
                              native_backend=args.native_backend, fake_weight_cache=args.fake_weight_cache == "on")
            # one module per call: each BF16 weight is released as soon as its layer is built
            # (a single quantize() call holds every original linear until it returns)
            layers = {}
            for n in match:
                layers.update(quantize(model, spec, qspec, trainable=True, names=[n]))
            for n, l in layers.items():                     # train_map's start: theta = -1 < 0 -> all E2M1
                l.set_map(torch.zeros(match[n].shape, dtype=torch.bool))
            grads, kls = oneshot_gradients(model, layers, fit, teacher, device, args.batch)
            if isinstance(teacher, DiskTeacher) and args.teacher_dir is None:
                shutil.rmtree(teacher.root, ignore_errors=True)
            report["quant"] = qspec.as_dict()
            report["fit_kl"] = sum(kls) / len(kls)
            for n, g in grads.items():
                tiles[n] = top_k(-g, k[n])
                per_layer[n].update(negative=int((g < 0).sum()), grad_abs_sum=float(g.abs().sum()))
            torch.save({n: g.cpu() for n, g in grads.items()}, args.out.with_suffix(".grad.pt"))

    for n, m in tiles.items():                               # the fairness rule, checked
        assert int(m.sum()) == k[n] and m.shape == match[n].shape, n
    meta = dict(model=spec.key, method=args.method, seed=args.seed if args.method == "random" else None,
                match=str(args.match), fit_windows=report.get("data", {}).get("fit_windows"),
                calib_record=None if args.calib_record is None else str(args.calib_record),
                calib_extension=None if args.calib_extension is None else str(args.calib_extension),
                tiny_layers=args.tiny_layers,
                mode=args.mode if args.method == "oneshot" else None)
    maps_.save(args.out, tiles, unit, meta)
    report.update(map=maps_.summary(tiles), per_layer=per_layer, resources=resources(started), status="complete")
    if args.method == "act":
        report["layers_short_of_positive"] = sum(v["positive"] < v["k"] for v in per_layer.values())
    if args.method == "oneshot":
        report["layers_short_of_negative"] = sum(v["negative"] < v["k"] for v in per_layer.values())
    save_json(report_path, report)
    print(f"MAP {args.method} {spec.key} {unit}: {report['map']['e0m3_tiles']} E0M3 tiles "
          f"({report['map']['e0m3_fraction']:.4%}) -> {args.out}", flush=True)
    return report


if __name__ == "__main__":
    main()
