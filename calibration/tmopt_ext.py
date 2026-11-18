"""Ablation maps with the paper's own trainer, TM-OPT+TC (the reference ``run_train_map.py --tm-opt --tile-grad-tc``,
vendored unchanged in ``calibration/tmopt``), through the hooks of ``calibration/tmopt_launch.py``.

Everything that is not the studied variable is the main-table maps' (calibration.train_map): the trainer and
its flags (``train_map.TM_OPT_TC``), the per-model batch (``train_map.BATCH``: micro-batch 2 x
accumulation 4 for Qwen3.8-27B and Nemotron-Nano-9B-v2, 8 x 1 otherwise), the fit set (the model's reference
calibration record), the environment (``tmopt_common.env``: ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, deterministic) and
the process layout (its own process, from ``calibration/tmopt``).

    python -m calibration.tmopt_ext record  --model M                     # the calibration record (once per model)
    python -m calibration.tmopt_ext oneshot --model M --unit U --match maps/M/flipquant_U.pt --out OUT.pt
    python -m calibration.tmopt_ext train   --model M --unit U --out OUT.pt [--rotate hadamard --rotate-block 16]

``oneshot`` (Table 8, "One-shot gradient"): the trainer from the all-E2M1 start (``--init-logit -1``), one epoch,
every optimizer step intercepted before its update (``tmopt_launch.py --oneshot``): the steps' tile gradients are summed
over the fit windows (256 here) and no logit ever moves. Per module, the ``k_l`` tiles with the most negative summed gradient
become E0M3, ``k_l`` = that module's E0M3 count in ``--match`` (the main-table map of the same model and unit); ties
go to the lower tile index (``baseline_maps.top_k``). The tile gradient is TM-OPT+TC's own (``--tile-grad-tc``):
``dKL/dm_u = sum over tile u of G * (A - B)`` with ``G = dy^T x`` on the BF16 tensor cores with FP32 accumulation, ``x``
the per-token FourOverSix-quantized input (straight-through) and ``A - B`` the E0M3 minus FourOverSix candidate.

``train``: the trainer through the launcher without hooks, i.e. a FlipQuant map exactly as calibration.train_map
makes it (tests/test_tmopt.py checks the two bit for bit).

``train --rotate hadamard --rotate-block 16`` (Table 9, the Hadamard row's FlipQuant 16x64 map): the trainer with
``tmopt_launch.py --rotate``: every quantized linear's weight becomes ``W H`` before its two candidates are quantized, its
input ``x H`` before the per-token activation quantizer, and the teacher stays the unrotated BF16 model. The map's
``meta`` records ``rotate`` and ``rotate_block`` (``--map`` accepts it only with the same ``--rotate`` /
``--rotate-block``). Without ``--rotate`` the run is calibration.train_map's, bit for bit (tests/test_tmopt.py);
``--rotate-block 1`` (H = [1]) reproduces it bit for bit too.

**paper-sm120-runs: the release settings.** Both commands run the trainer with the FlipQuant release maps' settings
(``--fit-windows 256 --teacher-topk 1000``, 5 epochs for ``train``, lr 0.02, init -1, seed 0, deterministic; the 256
fit windows are the paper's 128 plus 128 by the extension rule, ``tmopt_common.prepare_development``), so a one-shot
gradient is one pass over the same 256 windows with the same top-1000 teacher, and a trained map is made as the release
maps were. Give the release runs' ``--data-root`` to reuse their records (the same token hashes).

``--fit-windows N`` and ``--teacher-host`` (opt-in; NVFP4-RaZeR paper-eval part AC, the paper's reruns at the
1,024-window calibration of its part AA): ``train`` and ``oneshot`` with N fit windows instead of the release's 256
(the record's 128, then N - 128 by the same extension rule, as ``calibration.train_map --fit-windows N``), and the
top-K teacher's rows in host memory (``calibration/tmopt_launch.py --teacher-host``, as ``calibration.train_map
--teacher-host``). With the defaults (256, off) the trainer's command is the one above, unchanged.

Outputs: ``OUT.pt`` (flipquant.maps format; ``meta`` names the trainer, the hooks, the settings, the record and the
model revision), ``OUT.json`` (report; ``status: complete`` last), and ``OUT.run/`` (the trainer's own outputs and log;
re-created if a previous attempt left it incomplete).

``train --candidates-e2m1 SRC --candidates-e0m3 SRC`` (opt-in; NVFP4-RaZeR paper-eval part O): the trainer with
``tmopt_launch.py --candidates-*``: the two candidates of every module are the given ones (a flipquant-gptq/1 codes file:
GPTQ FourOverSix codes / GPTQ codes on the all-E0M3 grid; or ``rtn``, flipquant's own RTN codes) instead of the
trainer's RTN quantizers'; everything else is the release training. With ``rtn`` the run is calibration.train_map's
bit for bit (tests/test_acceptance_candidates.py (g)). The map's ``meta.candidates`` records the sources.

``--tiny-layers N`` (TESTS ONLY): the randomly initialized N-layer model of models/tiny.py, with a calibration record
for it (``record --tiny-layers N``: the real model's fit set, that model's weight hashes) under
``<data root>/tiny<N>``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

from calibration import tmopt_common as RC
from calibration.train_map import SETTINGS, check_tiles, command
from flipquant import maps as M

EXT = RC.ROOT / "calibration" / "tmopt_launch.py"
TRAINER = ("TM-OPT+TC: run_train_map.py --tm-opt --tile-grad-tc of the reference implementation (vendored as "
           "calibration/tmopt/run_train_map.py), run through calibration/tmopt_launch.py")
FIT_SET = ("128 windows x 512 tokens: 64 open-web-math @ fde8ef8 + 64 codeparrot-clean @ 35a59fb, then 128 by the "
           "extension rule (256 in all, the release maps' set)")
# the FlipQuant release maps' settings (A1 of the paper-sm120 runs): every trained or one-shot map here uses them
RELEASE = dict(epochs=5, fit_windows=256, teacher_topk=1000, lr=0.02, init_logit=-1.0, seed=0)
# the pinned dataset files the reference implementation's fit-set rule (campaign.data.builder_seed0) reads from the Hugging Face cache
DATASET_FILES = (("open-web-math/open-web-math", "fde8ef8de2300f5e778f56261843dab89f230815",
                  "data/train-00000-of-00114-5a023365406cb9c4.parquet"),
                 ("codeparrot/codeparrot-clean", "35a59fb025bc0a102f7d96eac09d145b896d487b", "file-000000000001.json.gz"))


def fit_set(windows: int) -> str:
    """The fit set's description for ``--fit-windows`` (FIT_SET for the release's 256)."""
    if windows == 256:
        return FIT_SET
    return (f"128 windows x 512 tokens: 64 open-web-math @ fde8ef8 + 64 codeparrot-clean @ 35a59fb, then {windows - 128} "
            f"by the extension rule ({windows} in all; --fit-windows {windows}, not the release maps' 256)")


def _sha(t: torch.Tensor) -> str:
    """the reference run_conditional_format.sha (the trainer's per-matrix weight check)."""
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def ensure_dataset_files() -> None:
    import os
    from huggingface_hub import hf_hub_download
    env = RC.env()
    hub = env.get("HF_HUB_CACHE") or str(Path(env["HF_HOME"]) / "hub")
    for repo, revision, path in DATASET_FILES:
        hf_hub_download(repo, path, revision=revision, repo_type="dataset", cache_dir=hub)
    os.environ.setdefault("HF_HOME", env["HF_HOME"])


def data_root(path, tiny_layers=None) -> Path:
    root = RC.data_root(path)
    return root / f"tiny{tiny_layers}" if tiny_layers else root


def record_path(key: str, root: Path) -> Path:
    return root / key / "calibration" / "report.json"


def tiny_record(spec, key: str, root: Path, layers: int) -> Path:
    """TESTS ONLY: a calibration record for models.tiny's N-layer model: the real model's fit set (its archived
    record, or the builder rule on its tokenizer) and the tiny model's per-matrix weight hashes."""
    from models import tiny
    target = record_path(key, root)
    if target.exists() and (root / key / "prepare_summary.json").exists():
        return target
    RC.ensure_snapshot(key, weights=False)
    ensure_dataset_files()
    sys.path[:0] = [str(RC.ENGINE), str(RC.ENGINE / "research" / "n16k64" / "software" / "primary")]
    import prepare_model_data as P
    from campaign.data import builder_seed0
    from transformers import AutoTokenizer
    source = str(P.snapshot(key))
    revision = P.MODELS[key]["revision"]
    if P.MODELS[key]["record"]:
        rec = json.loads((RC.ENGINE / P.MODELS[key]["record"]).read_text())
        fit = P.fit_spec(rec["fit"] if "fit" in rec else rec)
    else:
        _, built = builder_seed0(AutoTokenizer.from_pretrained(source))
        fit = P.fit_spec(built)
    model, _, kept = tiny.build(spec, None, layers, device="cuda", attn="sdpa")
    if spec.loader == "qwen3_5":
        mods = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                and "language_model" in n and "head" not in n}
    else:
        head = model.get_output_embeddings()
        mods = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}
    import transformers
    prior = dict(model=key, source=source, revision=revision, torch_version=torch.__version__,
                 transformers_version=transformers.__version__, fit=fit,
                 matrices={n: dict(shape=list(m.weight.shape), source_sha256=_sha(m.weight)) for n, m in mods.items()},
                 local_preparation=dict(script="calibration.tmopt_ext record --tiny-layers", tiny_layers=layers,
                                        tiny_kept_layers=kept, note="TESTS ONLY: random-init tiny model"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(prior, indent=2) + "\n")
    (root / key / "prepare_summary.json").write_text(json.dumps(dict(model=key, tiny_layers=layers, matrices=len(mods)),
                                                                indent=2) + "\n")
    return target


def record(model: str, root_arg=None, tiny_layers=None) -> Path:
    spec, key = RC.resolve(model)
    root = data_root(root_arg, tiny_layers)
    if tiny_layers:
        return tiny_record(spec, key, root, tiny_layers)
    ensure_dataset_files()
    rec = RC.prepare(key, root)
    RC.prepare_development(key, root)          # the 128 extension windows (--fit-windows 256)
    return rec


def shapes_of(spec, revision, tiny_layers):
    if tiny_layers:
        from models.tiny import meta_linear_shapes
        return meta_linear_shapes(spec, None, tiny_layers)
    return RC.expected_shapes(spec, revision)


def run_trainer(spec, key: str, unit: str, run: Path, root: Path, rec: Path, epochs: int, hooks: list,
                tiny_layers=None, partial: bool = False, fit_windows: int = RELEASE["fit_windows"],
                teacher_host: bool = False) -> dict:
    """The TM-OPT+TC command of calibration.train_map, through tmopt_launch.py with ``hooks`` (and ``--teacher-host``
    when asked: the launcher's own hook, as calibration.train_map adds it)."""
    hooks = list(hooks) + (["--tiny-layers", str(tiny_layers)] if tiny_layers else []) + \
        (["--teacher-host"] if teacher_host else [])
    # calibration.train_map's own command (the paper's flags and per-model batch), with the launcher in front
    trainer = command(key, unit, root, RC.transformers_deviation(rec), run / "run", epochs,
                      teacher_topk=RELEASE["teacher_topk"], fit_windows=fit_windows)
    assert trainer[:2] == [sys.executable, "run_train_map.py"], trainer[:2]
    cmd = [sys.executable, EXT, *hooks, "--", *trainer[1:]]
    rc = RC.run(cmd, run / "train.log")
    report = run / "run" / "report.json"
    rep = json.loads(report.read_text()) if report.exists() else {}
    if rc != 0 or (rep.get("status") != "complete" and not partial):     # partial: --oneshot-steps stops early
        raise SystemExit(f"run_train_map.py failed (rc {rc}, status {rep.get('status')}); see {run / 'train.log'}")
    return rep


def write_map(out: Path, tiles: dict, unit: str, spec, rec: Path, shapes: dict, meta: dict) -> dict:
    bad = check_tiles(tiles, unit, shapes)
    if bad:
        raise SystemExit(f"the map does not fit {spec.key}'s quantizable linears: {bad[:5]}")
    M.save(out, tiles, unit, meta)
    back, unit2, _ = M.load(out)
    if unit2 != unit or set(back) != set(tiles) or not all(torch.equal(back[n], tiles[n]) for n in tiles):
        raise SystemExit(f"{out} does not read back")
    return M.summary(tiles)


def base_meta(spec, key, unit, rep: dict, rec: Path, run: Path, args) -> dict:
    prior = json.loads(rec.read_text())
    a = rep["args"]
    vendored = json.loads((RC.ENGINE / "VENDORED.json").read_text())
    meta = dict(trainer=TRAINER, repo="the reference implementation", branch=vendored["branch"], commit=vendored["commit"],
                settings=dict({k: a.get(k) for k in SETTINGS}, **rep.get("settings", {}), fit_set=fit_set(args.fit_windows),
                              activations="per-token FourOverSix fake quant (STE)", loss="KL to the BF16 teacher"),
                model_id=spec.hf_id, revision=prior["revision"], registry_key=spec.key, unit=unit,
                calibration_time=dict(wall_seconds=rep.get("resources", {}).get("total_seconds"),
                                      setup_seconds=rep.get("setup_seconds"),
                                      training_seconds=rep.get("training_seconds")),
                transformers=rep.get("host", {}).get("transformers"), torch=rep.get("host", {}).get("torch"),
                gpu=(rep.get("host", {}).get("gpus") or [None])[0], deviations=rep.get("deviations"),
                calibration_record=dict(path=str(rec), sha256=RC.sha256(rec)),
                written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    if args.tiny_layers:
        meta.update(tiny_layers=args.tiny_layers, note="SMOKE ONLY: random-init tiny model")
    if args.fit_windows != RELEASE["fit_windows"] or args.teacher_host:
        meta.update(non_release=dict(fit_windows=args.fit_windows, teacher_host=args.teacher_host))
    return meta


def cmd_train(args, spec, key, rec, root, run):
    hooks = [] if args.rotate == "none" else ["--rotate", args.rotate, "--rotate-block", str(args.rotate_block)]
    if args.candidates_e2m1 is not None:
        hooks += ["--candidates-e2m1", str(args.candidates_e2m1), "--candidates-e0m3", str(args.candidates_e0m3)]
    rep = run_trainer(spec, key, args.unit, run, root, rec, args.epochs, hooks, args.tiny_layers,
                      fit_windows=args.fit_windows, teacher_host=args.teacher_host)
    tiles = {n: m.bool() for n, m in torch.load(run / "run" / "map.pt", map_location="cpu", weights_only=True).items()}
    meta = base_meta(spec, key, args.unit, rep, rec, run, args)
    meta.update(method="FlipQuant = TM-OPT+TC with the release settings" + ("" if args.rotate == "none" else
                                                   f", in the Hadamard basis (block {args.rotate_block})")
                + ("" if args.candidates_e2m1 is None else
                   f", candidates from --candidates-e2m1 {args.candidates_e2m1} / --candidates-e0m3 "
                   f"{args.candidates_e0m3}")
                + ("" if args.epochs == RELEASE["epochs"] else
                   f" -- {args.epochs} epochs, NOT the release's {RELEASE['epochs']}")
                + ("" if args.fit_windows == RELEASE["fit_windows"] else
                   f" -- {args.fit_windows} fit windows, NOT the release's {RELEASE['fit_windows']}"),
                release_settings=RELEASE,
                source_map_sha256=RC.sha256(run / "run" / "map.pt"),
                source=dict(kind="trained", run=str(run / "run"), hooks=[str(h) for h in hooks]))
    if args.rotate != "none":
        meta.update(rotate=args.rotate, rotate_block=args.rotate_block,
                    trainer=TRAINER + " + calibration/tmopt_launch.py --rotate (W H offline, x H before the activation "
                                      "quantizer, unrotated BF16 teacher)")
    extra = dict(trainer_report=str(run / "run" / "report.json"))
    if args.candidates_e2m1 is not None:
        hook = json.loads((run / "run" / "candidates_hook.json").read_text())
        meta.update(trainer=TRAINER + " + calibration/tmopt_launch.py --candidates-e2m1 / --candidates-e0m3 (the "
                                      "given candidates in place of the trainer's RTN quantizers)",
                    candidates=dict(e2m1=dict(source=str(args.candidates_e2m1)),
                                    e0m3=dict(source=str(args.candidates_e0m3)),
                                    mean_codes_changed_vs_rtn=hook["mean_codes_changed_vs_rtn"]))
        for k in ("e2m1", "e0m3"):
            src = hook["sources"][k]
            if src["source"] != "rtn":
                meta["candidates"][k].update(key=src["key"], bytes=src["bytes"])
        extra["candidates_hook"] = str(run / "run" / "candidates_hook.json")
    meta["settings"]["output"] = "last-epoch map (theta > 0 -> E0M3)"
    return tiles, meta, extra


def cmd_oneshot(args, spec, key, rec, root, run):
    from calibration.baseline_maps import top_k
    match, unit, match_meta = M.load(args.match, args.unit)
    if M.unit_name(unit) != M.unit_name(args.unit):
        raise SystemExit(f"--match is a {unit} map, not {args.unit}")
    from models.mapcheck import model_key
    if model_key(match_meta) not in (None, spec.key):
        raise SystemExit(f"--match was made for {model_key(match_meta)}, not {spec.key}")
    grads_file = run / "oneshot_grads.pt"
    hooks = ["--oneshot", grads_file] + (["--oneshot-steps", str(args.oneshot_steps)] if args.oneshot_steps else [])
    rep = run_trainer(spec, key, args.unit, run, root, rec, 1, hooks, args.tiny_layers,
                      partial=bool(args.oneshot_steps), fit_windows=args.fit_windows, teacher_host=args.teacher_host)
    g = torch.load(grads_file, map_location="cpu", weights_only=True)
    if set(g["sum"]) != set(match):
        raise SystemExit(f"--match covers {len(match)} modules, the trainer quantizes {len(g['sum'])}")
    tiles, per_layer = {}, {}
    for n, s in g["sum"].items():
        k = int(match[n].sum())
        tiles[n] = top_k(-s, k)
        assert int(tiles[n].sum()) == k and tiles[n].shape == match[n].shape, n
        per_layer[n] = dict(k=k, tiles=s.numel(), negative=int((s < 0).sum()), grad_abs_sum=float(s.abs().sum()))
    meta = base_meta(spec, key, args.unit, rep, rec, run, args)
    meta.update(method="One-shot gradient: TM-OPT+TC's tile gradient (--tile-grad-tc) at the all-E2M1 start "
                       f"(--init-logit -1), summed over one epoch's optimizer steps (all {args.fit_windows} fit "
                       "windows, top-1000 teacher), no update; "
                       "per module the k_l most negative tiles, k_l from --match",
                model=spec.key, match=dict(path=str(args.match), sha256=RC.sha256(args.match), meta_method=match_meta.get("method")),
                release_settings=dict(RELEASE, epochs=1),
                fit_windows=args.fit_windows,
                oneshot=dict(steps=g["steps"], grads_file=str(grads_file), grads_sha256=RC.sha256(grads_file),
                             train_kl=(rep.get("epochs") or [{}])[0].get("train_kl")),
                source=dict(kind="oneshot", run=str(run / "run"), hooks=["--oneshot"]))
    extra = dict(per_layer=per_layer, layers_short_of_negative=sum(v["negative"] < v["k"] for v in per_layer.values()))
    return tiles, meta, extra


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("record", "train", "oneshot"))
    ap.add_argument("--model", required=True, help=f"registry key or alias, one of {sorted(RC.MODEL_KEYS)}")
    ap.add_argument("--unit", default=None, choices=("8x64", "16x64", "256x64"))
    ap.add_argument("--out", type=Path, default=None, help="output map file (.pt); OUT.json and OUT.run/ next to it")
    ap.add_argument("--match", type=Path, default=None, help="oneshot: the main-table map whose k_l to match")
    ap.add_argument("--epochs", type=int, default=RELEASE["epochs"], help="train: 5, the release maps'")
    ap.add_argument("--rotate", choices=("none", "hadamard"), default="none", help="train: learn in the Hadamard basis")
    ap.add_argument("--rotate-block", type=int, default=16,
                    help="largest Hadamard block (power of two; 16 = one scale group, flipquant/rotate.py)")
    ap.add_argument("--candidates-e2m1", default=None,
                    help="train, opt-in: the E2M1 candidate, a flipquant-gptq/1 codes file (FourOverSix) or 'rtn' "
                         "(tmopt_launch.py --candidates-e2m1)")
    ap.add_argument("--candidates-e0m3", default=None,
                    help="train, opt-in: the E0M3 candidate, a flipquant-gptq/1 codes file (mixfp4, every tile E0M3) "
                         "or 'rtn'")
    ap.add_argument("--oneshot-steps", type=int, default=None, help="oneshot, TESTS ONLY: stop after N steps")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="the reference calibration records (default $FLIPQUANT_CACHE/tmopt_data)")
    ap.add_argument("--tiny-layers", type=int, default=None, help="TESTS ONLY: models/tiny.py's N-layer model")
    ap.add_argument("--fit-windows", type=int, default=RELEASE["fit_windows"], metavar="N",
                    help="train / oneshot, opt-in: N fit windows (even, >= 128; the record's 128 then N - 128 by the "
                         "extension rule) instead of the release's 256")
    ap.add_argument("--teacher-host", action="store_true",
                    help="train / oneshot, opt-in: the top-K teacher's rows in host memory (tmopt_launch.py "
                         "--teacher-host)")
    args = ap.parse_args(argv)
    if args.fit_windows < 128 or args.fit_windows % 2:
        raise SystemExit(f"--fit-windows {args.fit_windows}: even and >= 128")
    if (args.candidates_e2m1 is None) != (args.candidates_e0m3 is None):
        raise SystemExit("--candidates-e2m1 and --candidates-e0m3 go together")
    if args.candidates_e2m1 is not None and (args.command != "train" or args.rotate != "none"):
        raise SystemExit("--candidates-* go with train, without --rotate")
    started = time.time()
    if args.command == "record":
        print(f"RECORD {record(args.model, args.data_root, args.tiny_layers)}")
        return
    if args.unit is None or args.out is None or (args.command == "oneshot" and args.match is None):
        raise SystemExit(f"{args.command} needs --unit, --out" + (" and --match" if args.command == "oneshot" else ""))
    spec, key = RC.resolve(args.model)
    root = data_root(args.data_root, args.tiny_layers)
    rec = record(args.model, args.data_root, args.tiny_layers)
    out = args.out.resolve()
    run = out.with_suffix(".run")
    if run.exists():
        shutil.rmtree(run)                      # an incomplete earlier attempt: the trainer needs a fresh --out
    run.mkdir(parents=True)
    prior = json.loads(rec.read_text())
    tiles, meta, extra = (cmd_train if args.command == "train" else cmd_oneshot)(args, spec, key, rec, root, run)
    summ = write_map(out, tiles, args.unit, spec, rec, shapes_of(spec, prior["revision"], args.tiny_layers), meta)
    meta.update(e0m3_tiles=summ["e0m3_tiles"], tiles=summ["tiles"], e0m3_share=summ["e0m3_fraction"],
                modules=summ["modules"])
    M.save(out, tiles, args.unit, meta)
    report = dict(status="complete", command=args.command, model=spec.key, unit=args.unit, map=str(out),
                  map_file_sha256=RC.sha256(out), map_summary=summ, meta=meta, **extra,
                  resources=dict(wall_seconds=time.time() - started))
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"MAP {args.command} {spec.key} {args.unit}: {summ['e0m3_tiles']} E0M3 tiles "
          f"({summ['e0m3_fraction']:.4%}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
