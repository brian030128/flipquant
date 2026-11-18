"""FlipQuant maps with the paper's trainer: TM-OPT+TC, ``run_train_map.py``, vendored unchanged in ``calibration/tmopt``
(``calibration/tmopt/VENDORED.json``, ``PORT_CHANGES.md``).

    python -m calibration.train_map --model llama3.1-8b --unit 8x64 --out runs/llama3.1-8b_8x64

This is the trainer behind every map in ``maps/<model>/flipquant_<unit>.pt``, and it reproduces the paper's maps bit for
bit (on an RTX Pro 6000 / sm_120 GPU, with torch 2.9.0 and transformers 5.16.1, as recorded). The co-authors' own
implementation of the idea is ``calibration.train_map_legacy``, with its numerics and its options (fake / native
training, development evaluation, ModelOpt checkpoints, record replay).

**Method (the paper's settings, the defaults):**
- `--tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2 --no-dev --no-eval`
  (the vendored experiments/paper/paper_common.TM_OPT_TC): one logit per tile, a straight-through hard map, KL to the
  BF16 teacher, and per-token FourOverSix activations.
- Adam with betas (0.9, 0.999), eps 1e-12, a constant schedule, seed 0. The output is the last-epoch map.
- Deterministic, always: `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Optimizer batch 8 sequences. Qwen3.8-27B and Nemotron-Nano-9B-v2 run as micro-batch 2 x accumulation 4 (their
  memory), Qwen3.8-27B on one GPU. That is the paper's only per-model difference.
- Fit set: 128 math/code windows of 512 tokens, the model's calibration record (``calibration.tmopt_common.prepare``, in
  ``--data-root``). No development set is drawn, and no development evaluation runs.

**Options.** The defaults are the paper's settings. A different value of any of these is recorded in the map's
``meta`` (``non_default_settings``) and in ``reproduction.json``, and the method string says "NOT the paper's
settings". ``--require-reproduction`` refuses them.
- ``--epochs`` (20).
- ``--lr`` (0.02): Adam's learning rate for the tile logits.
- ``--init-logit`` (-1): the initial logit of every tile; negative starts at FourOverSix E2M1.
- ``--teacher-topk K`` (0 = the full-vocabulary teacher): the top-K teacher of ``calibration.train_map_legacy
  --teacher-topk`` (vendored as ``calibration/tmopt/topk_teacher.py``).
  - **The teacher:** per window, the top-K of the FP32 teacher log-probabilities (BF16 values, int32 ids) plus the
    exact log-mass of the rest. It stays on the GPU instead of the host.
  - **The loss:** KL over the K tokens plus one tail bucket.
  - **Measured cost** (Llama-3.1-8B 16x64): K = 256 with 5 epochs takes 0.30 of the paper calibration's time. Host
    memory after the model load is 0.14 of the paper's, and GPU memory is 1.5 % higher.
- ``--fit-windows N`` (128): the paper's 128 fit windows, then N - 128 more by the extension rule of
  ``calibration.train_map_legacy --fit-windows`` (``calibration/tmopt/fit_extension.py``: its stream rule with seed
  20260930, on the paper record's math and code streams, half each), skipping the paper's fit, development and
  published C4 evaluation documents. Every extra window is recorded by document hash, offset and token hash. The
  development records come from ``calibration.tmopt_common.prepare_development`` (vendored for Llama-3.1-8B; for
  Qwen3.8-27B its three sets, redrawn and accepted only by their recorded hashes; drawn once for the other models).
  A model the vendored trainer does not know (``calibration.tmopt_common.RUNTIME_MODELS``: Llama-3.2-1B-Instruct) runs
  through ``calibration/tmopt_launch.py --register-model``, which adds it to the trainer's tables as those models.
- ``--seed``, ``--batch``, ``--accum``: passed to the trainer (the paper: seed 0; batch 8 x accumulation 1, or 2 x 4
  for Qwen3.8-27B and Nemotron-Nano-9B-v2).

**Run options** (they change where things are kept or when the run stops, not the map; not training settings):
- ``--teacher-host``: the top-K teacher's rows in host memory instead of on the GPU (``calibration/tmopt_launch.py
  --teacher-host``). The losses move each micro-batch's rows to the GPU, so the map is the GPU-teacher map. The teacher
  takes about 3 MB per fit window (K = 1000, 512 tokens), so 1,024 / 2,048 windows add 2.2 / 5.1 GiB to a 256-window
  run; Qwen3.8-27B's 256-window run already peaks at 91.3 GiB.
- ``--probe-steps N``: a memory probe. The trainer stops after N optimizer steps (``run_train_map.py --probe-steps``)
  and records each phase's peak memory; ``<out>/probe.json`` has them, and no map is written.

**Options of ``calibration.train_map_legacy``.** ``--deterministic`` and ``--no-dev`` are accepted and change nothing
(this trainer is always deterministic and never draws a development set). ``--mode fake``, ``--param ste``,
``--optimizer adam``, ``--schedule constant``, ``--eps 1e-12``, ``--betas 0.9 0.999`` and ``--revision`` at the pinned
revision are accepted the same way: they are the method. Any other legacy option or value (``--mode native``,
``--native-backend``, ``--param sigmoid``, development evaluation, ``--teacher-dir``, ``--calib-record``,
``--nvidia-nvfp4``, ...) is an error naming ``calibration.train_map_legacy``; nothing falls back to it.

**Steps:**
1. The calibration record (prepared once per data root), and the pinned model snapshot.
2. ``run_train_map.py`` in its own process, from ``calibration/tmopt``. Its outputs go to ``<out>/run``: the bare map
   ``map.pt``, ``report.json`` and the per-epoch log.
3. ``<out>/map.pt``: the map in ``flipquant-map/1`` (``flipquant.maps.save``), with the settings, model revision,
   trainer provenance, this checkout's commit (``flipquant``: the commit, and whether its tracked files were unmodified)
   and cost in ``meta``; evaluate it with ``--map <out>/map.pt``. It is verified on a meta-device
   model: ``maps.load`` reads it, and its modules and tile shapes equal ``models.loader.quantizable_linears``.
4. ``<out>/reproduction.json`` compares the result with two references, where they exist:
   - the paper's map: the run's ``map.pt`` sha256 against ``calibration/tmopt/experiments/paper/maps.sha256.json``;
   - this repository's committed map ``maps/<model>/flipquant_<unit>.pt``: its tiles, and the sha256 of the run's
     ``map.pt`` recorded in its meta.
   ``--require-reproduction`` turns a difference into an error.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from calibration import tmopt_common as RC
from flipquant import maps as M

ROOT = RC.ROOT
UNITS = ("8x64", "16x64", "256x64")
TM_OPT_TC = ["--tm-opt", "--tile-grad-tc", "--param", "ste", "--lr", "0.02", "--init-logit", "-1", "--epochs", "20",
             "--eval-every", "2", "--no-dev", "--no-eval"]
BATCH = {"qwen27b": ["--gpus", "1", "--batch", "2", "--accum", "4"], "nemotron9b": ["--batch", "2", "--accum", "4"]}
SETTINGS = ("param", "optimizer", "lr", "eps", "betas", "schedule", "init_logit", "epochs", "batch", "accum", "seed",
            "tm_opt", "tile_grad_tc", "no_dev", "no_eval", "unit", "teacher_topk", "fit_windows", "fit_source",
            "deterministic")
# The paper's values of the options this trainer exposes (TM_OPT_TC; --teacher-topk 0 is the full-vocabulary teacher;
# batch and accumulation are per model, see paper_batch)
PAPER = dict(epochs=20, lr=0.02, init_logit=-1.0, teacher_topk=0, fit_windows=128, seed=0, fit_source="paper",
             deterministic=True)
PAPER_MAPS = RC.ENGINE / "experiments" / "paper" / "maps.sha256.json"
# calibration.train_map_legacy's options: accepted at the method's value, an error otherwise
LEGACY_AT = dict(mode="fake", param="ste", optimizer="adam", schedule="constant", eps=1e-12, betas=(0.9, 0.999))
LEGACY_ONLY = ("native_backend", "device_map", "tau_start", "tau_end", "eval_every", "eval_batch", "teacher_dir",
               "calib_record", "nvidia_nvfp4", "ppl_datasets")


def paper_batch(key: str) -> dict:
    """The paper's micro-batch and accumulation of a model."""
    flags = BATCH.get(key, [])

    def get(flag, default):
        return int(flags[flags.index(flag) + 1]) if flag in flags else default
    return dict(batch=get("--batch", 8), accum=get("--accum", 1))


def checkout() -> dict:
    """This repository's commit, and whether its tracked files are unmodified (``git``; ``None`` outside a checkout)."""
    def git(*args):
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=no")
    return dict(commit=git("rev-parse", "HEAD"), clean=None if status is None else status == "")


def check_tiles(tiles: dict, unit: str, shapes: dict) -> list[str]:
    """The problems of a tile dict against the quantizable linears (empty = ok)."""
    bad = []
    if set(tiles) != set(shapes):
        extra, missing = sorted(set(tiles) - set(shapes)), sorted(set(shapes) - set(tiles))
        bad.append(f"module set differs: {len(extra)} extra (e.g. {extra[:3]}), {len(missing)} missing (e.g. {missing[:3]})")
    for n in sorted(set(tiles) & set(shapes)):
        want = M.tile_shape(*shapes[n], unit)
        if tiles[n].dtype != torch.bool or tuple(tiles[n].shape) != want:
            bad.append(f"{n}: {tiles[n].dtype} {tuple(tiles[n].shape)}, expected bool {want}")
    return bad


def reproduction(spec, key: str, unit: str, map_sha: str, tiles: dict) -> dict:
    out = {}
    paper = json.loads(PAPER_MAPS.read_text())["maps"].get(f"{key}_{unit}")
    if paper is not None:
        out["paper_map"] = dict(sha256=paper["map_sha256"], record=paper["record"], equal=map_sha == paper["map_sha256"])
    committed = ROOT / "maps" / spec.key / f"flipquant_{unit}.pt"
    if committed.exists():
        ref, ref_unit, meta = M.load(committed)
        same = ref_unit == unit and set(ref) == set(tiles) and all(torch.equal(ref[n], tiles[n]) for n in tiles)
        recorded = meta.get("source_map_sha256")
        out["committed_map"] = dict(file=str(committed.relative_to(ROOT)), tiles_equal=same, source_map_sha256=recorded,
                                    equal=same and (recorded is None or recorded == map_sha))
    return out


def command(key: str, unit: str, root: Path, deviation: list, run_dir: Path, epochs: int = 20, lr: float | None = None,
            init_logit: float | None = None, teacher_topk: int = 0, fit_windows: int = 128, seed: int | None = None,
            batch: int | None = None, accum: int | None = None, teacher_host: bool = False,
            probe_steps: int | None = None, fit_source: str = "paper", deterministic: bool = True) -> list:
    """The run_train_map.py command (run from calibration/tmopt). ``lr`` / ``init_logit`` / ``seed`` / ``batch`` /
    ``accum`` replace the paper's values when given; ``teacher_topk`` > 0 adds ``--teacher-topk``, ``fit_windows`` other
    than 128 adds ``--fit-windows``. With the defaults it is the paper's command. ``teacher_host`` runs it through
    calibration/tmopt_launch.py ``--teacher-host``; ``probe_steps`` adds the trainer's ``--probe-steps``."""
    flags = list(TM_OPT_TC)
    flags[flags.index("--epochs") + 1] = str(epochs)
    if lr is not None:
        flags[flags.index("--lr") + 1] = repr(float(lr))
    if init_logit is not None:
        flags[flags.index("--init-logit") + 1] = repr(float(init_logit))
    if teacher_topk:
        flags += ["--teacher-topk", str(teacher_topk)]
    if fit_windows != 128:
        flags += ["--fit-windows", str(fit_windows)]
    if fit_source != "paper":
        flags += ["--fit-source", fit_source]
    if not deterministic:
        flags += ["--no-deterministic"]
    if seed is not None:
        flags += ["--seed", str(seed)]
    per_model = list(BATCH.get(key, []))
    for flag, value in (("--batch", batch), ("--accum", accum)):
        if value is not None:
            if flag in per_model:
                per_model[per_model.index(flag) + 1] = str(value)
            else:
                per_model += [flag, str(value)]
    if probe_steps is not None:
        flags += ["--probe-steps", str(probe_steps)]
    trainer = ["run_train_map.py", "--model", key, "--data-root", root, *deviation, "--unit", unit, *flags, *per_model,
               "--out", run_dir]
    hooks = []
    if key in RC.RUNTIME_MODELS:          # unknown to the vendored trainer: registered at launch, nothing else changes
        hooks += ["--register-model", key]
    if teacher_host:
        hooks += ["--teacher-host"]
    if hooks:
        return [sys.executable, RC.ROOT / "calibration" / "tmopt_launch.py", *hooks, "--", *trainer]
    return [sys.executable, *trainer]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"registry key or alias, one of {sorted(RC.MODEL_KEYS)}")
    ap.add_argument("--unit", required=True, help=f"tile unit: one of {', '.join(UNITS)}")
    ap.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="calibration records, prepared once per data root (default $FLIPQUANT_CACHE/tmopt_data)")
    ap.add_argument("--epochs", type=int, default=PAPER["epochs"], help="20 is the method; anything else is not a FlipQuant map")
    ap.add_argument("--lr", type=float, default=PAPER["lr"], help="Adam's learning rate for the tile logits (the paper: 0.02)")
    ap.add_argument("--init-logit", type=float, default=PAPER["init_logit"],
                    help="the initial logit of every tile (the paper: -1, i.e. start at FourOverSix E2M1)")
    ap.add_argument("--teacher-topk", type=int, default=PAPER["teacher_topk"], metavar="K",
                    help="the top-K + tail-bucket teacher kept on the GPU (the paper: 0 = the full vocabulary)")
    ap.add_argument("--fit-windows", type=int, default=PAPER["fit_windows"], metavar="N",
                    help="the paper's 128 fit windows, then N - 128 by the extension rule (the paper: 128); below "
                         "128, the first N/2 math and N/2 code windows of the paper's fit set (robustness runs)")
    ap.add_argument("--fit-source", default=PAPER["fit_source"],
                    choices=("paper", "fineweb-edu", "ultrachat", "wikitext2-train"),
                    help="N fit windows from another corpus (calibration/tmopt/fit_sources.py; robustness runs)")
    ap.add_argument("--no-deterministic", action="store_true",
                    help="run the trainer without deterministic algorithms (robustness runs; the paper: deterministic)")
    ap.add_argument("--seed", type=int, default=None, help="the trainer's seed (the paper: 0)")
    ap.add_argument("--batch", type=int, default=None, help="calibration sequences per forward/backward (the paper: "
                                                            "8, or 2 for Qwen3.8-27B and Nemotron-Nano-9B-v2)")
    ap.add_argument("--accum", type=int, default=None,
                    help="forward/backward passes per optimizer step (the paper: 1, or 4 for those two models)")
    ap.add_argument("--teacher-host", action="store_true",
                    help="keep the top-K teacher's rows in host memory instead of on the GPU (the same map)")
    ap.add_argument("--probe-steps", type=int, default=None, metavar="N",
                    help="memory probe: stop after N optimizer steps; <out>/probe.json, no map")
    ap.add_argument("--require-reproduction", action="store_true",
                    help="fail unless the map equals every reference that exists (paper map, committed map); "
                         "the paper's settings only")
    g = ap.add_argument_group("options of calibration.train_map_legacy",
                              "--deterministic and --no-dev are accepted and change nothing (always deterministic, never "
                              "a development set); --mode fake, --param ste, --optimizer adam, --schedule constant, "
                              "--eps 1e-12, --betas 0.9 0.999 and --revision at the pinned revision are accepted as the "
                              "method; other values and the legacy-only options are errors")
    g.add_argument("--deterministic", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--no-dev", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--revision", default=None, help=argparse.SUPPRESS)
    g.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    g.add_argument("--param", default=None, help=argparse.SUPPRESS)
    g.add_argument("--optimizer", default=None, help=argparse.SUPPRESS)
    g.add_argument("--schedule", default=None, help=argparse.SUPPRESS)
    g.add_argument("--eps", type=float, default=None, help=argparse.SUPPRESS)
    g.add_argument("--betas", type=float, nargs=2, default=None, help=argparse.SUPPRESS)
    for name in LEGACY_ONLY:
        if name == "nvidia_nvfp4":
            g.add_argument("--nvidia-nvfp4", action="store_true", help=argparse.SUPPRESS)
        else:
            g.add_argument("--" + name.replace("_", "-"), default=None, help=argparse.SUPPRESS)
    return ap


def legacy_options(args, key: str) -> tuple[list[str], list[str]]:
    """``(accepted, refused)``: the legacy options given, as written."""
    accepted, refused = [], []
    for flag in ("deterministic", "no_dev"):
        if getattr(args, flag):
            accepted.append("--" + flag.replace("_", "-"))
    for name, value in LEGACY_AT.items():
        given = getattr(args, name)
        if given is None:
            continue
        flag = f"--{name} {' '.join(map(str, given)) if name == 'betas' else given}"
        same = tuple(given) == value if name == "betas" else given == value
        (accepted if same else refused).append(flag)
    if args.revision is not None:
        (accepted if args.revision == RC.SNAPSHOTS[key][1] else refused).append(f"--revision {args.revision}")
    for name in LEGACY_ONLY:
        if getattr(args, name) not in (None, False):
            refused.append("--" + name.replace("_", "-"))
    if args.unit not in UNITS:
        refused.append(f"--unit {args.unit}")
    return accepted, refused


def write_probe(args, spec, key: str, code: dict, chosen: dict, non_default: dict, report: dict, rc: int,
                wall: float) -> None:
    """``--probe-steps``: ``<out>/probe.json`` from the trainer's probe report (its per-phase peaks), or an error."""
    status = report.get("status")
    probe = dict(model=spec.key, model_key=key, unit=args.unit, flipquant=code, settings=chosen,
                 non_default_settings=non_default, teacher_host=args.teacher_host, rc=rc, status=status,
                 probe=report.get("probe"), resources=report.get("resources"), out_of_memory=report.get("out_of_memory"),
                 teacher_storage=report.get("teacher_storage"), setup_seconds=report.get("setup_seconds"),
                 fit_extension=None if report.get("fit_extension") is None else
                 {k: v for k, v in report["fit_extension"].items() if k != "records"},
                 wall_seconds=wall)
    (args.out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
    print(json.dumps({k: v for k, v in probe.items() if k != "resources"}, indent=1))
    if rc != 0 or status != "probe_complete":
        raise SystemExit(f"the probe did not complete (rc {rc}, status {status}); see {args.out / 'train.log'}")


def main(argv=None):
    args = parser().parse_args(argv)
    spec, key = RC.resolve(args.model)
    accepted, refused = legacy_options(args, key)
    if refused:
        raise SystemExit(f"{', '.join(refused)}: not an option of the paper's trainer; use python -m "
                         f"calibration.train_map_legacy for it")
    defaults = dict(PAPER, **paper_batch(key))
    chosen = dict(epochs=args.epochs, lr=args.lr, init_logit=args.init_logit, teacher_topk=args.teacher_topk,
                  fit_windows=args.fit_windows, seed=PAPER["seed"] if args.seed is None else args.seed,
                  fit_source=args.fit_source, deterministic=not args.no_deterministic,
                  **{k: defaults[k] if getattr(args, k) is None else getattr(args, k) for k in ("batch", "accum")})
    non_default = {k: v for k, v in chosen.items() if v != defaults[k]}
    if args.require_reproduction and non_default:
        raise SystemExit(f"--require-reproduction compares with the paper's maps, which need the paper's settings; "
                         f"not the paper's: {non_default}")
    if args.teacher_topk < 0:
        raise SystemExit("--teacher-topk must be >= 0")
    if args.teacher_host and not args.teacher_topk:
        raise SystemExit("--teacher-host keeps the top-K teacher on the host; it needs --teacher-topk K")
    if args.probe_steps is not None and (args.probe_steps < 1 or args.require_reproduction):
        raise SystemExit("--probe-steps N: N >= 1, and a probe writes no map to reproduce")
    if args.fit_windows < 2 or args.fit_windows % 2:
        raise SystemExit("--fit-windows: an even number (the paper: 128)")
    args.out = args.out.resolve()          # the trainer runs from calibration/tmopt
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    code = checkout()                      # the code this run starts from
    started = time.time()
    root = RC.data_root(args.data_root)
    record = RC.prepare(key, root)
    development = (RC.prepare_development(key, root) if args.fit_windows > 128 or args.fit_source != "paper"
                   else None)
    run_dir = args.out / "run"
    cmd = command(key, args.unit, root, RC.transformers_deviation(record), run_dir, args.epochs,
                  lr=args.lr if "lr" in non_default else None,
                  init_logit=args.init_logit if "init_logit" in non_default else None, teacher_topk=args.teacher_topk,
                  fit_windows=args.fit_windows, seed=args.seed if "seed" in non_default else None,
                  batch=chosen["batch"] if "batch" in non_default else None,
                  accum=chosen["accum"] if "accum" in non_default else None, teacher_host=args.teacher_host,
                  probe_steps=args.probe_steps, fit_source=args.fit_source, deterministic=not args.no_deterministic)
    args.out.mkdir(parents=True)
    rc = RC.run(cmd, args.out / "train.log")
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if args.probe_steps is not None:
        write_probe(args, spec, key, code, chosen, non_default, report, rc, time.time() - started)
        return
    if rc != 0 or report.get("status") != "complete":
        raise SystemExit(f"run_train_map.py failed (rc {rc}, status {report.get('status')}); see {args.out / 'train.log'}")
    map_sha = RC.sha256(run_dir / "map.pt")
    if map_sha != report["map_sha256"]:
        raise SystemExit("map.pt does not match its report")
    prior = json.loads(record.read_text())
    tiles = {n: m.bool() for n, m in torch.load(run_dir / "map.pt", map_location="cpu", weights_only=True).items()}
    shapes = RC.expected_shapes(spec, prior["revision"])
    bad = check_tiles(tiles, args.unit, shapes)
    if bad:
        raise SystemExit(f"the map does not fit {spec.key}'s quantizable linears: {bad[:5]}")
    a, summ = report["args"], M.summary(tiles)
    vendored = json.loads((RC.ENGINE / "VENDORED.json").read_text())
    extension, source, subset = report.get("fit_extension"), report.get("fit_source"), report.get("fit_subset")
    meta = dict(
        method="FlipQuant = TM-OPT+TC (the paper's calibration)" + ("" if not non_default else
                                                                     " -- NOT the paper's settings: " + ", ".join(
                                                                         f"{k.replace('_', '-')} {v} (paper {defaults[k]})"
                                                                         for k, v in non_default.items())),
        paper_settings=defaults, non_default_settings=non_default,
        teacher=report.get("teacher_storage"), teacher_host=args.teacher_host,
        trainer="TM-OPT+TC: run_train_map.py --tm-opt --tile-grad-tc (calibration/tmopt/run_train_map.py, vendored "
                "from the reference implementation)" + (
                    f"; {key} registered at launch (calibration/tmopt_launch.py --register-model)"
                    if key in RC.RUNTIME_MODELS else ""),
        # the vendored source, named as in the README and the committed maps' meta (VENDORED.json records it)
        repo="the reference implementation", branch=vendored["branch"], commit=vendored["commit"], flipquant=code,
        settings=dict({k: a.get(k) for k in SETTINGS}, **report.get("settings", {}),
                      fit_set=(f"{source['windows']} windows x 512 tokens of {source['source']} "
                               f"({source['dataset']['repo']} @ {source['dataset']['revision'][:7]})" if source else
                               f"{subset['windows']} windows x 512 tokens: the first {subset['windows'] // 2} "
                               "open-web-math @ fde8ef8 + the first "
                               f"{subset['windows'] // 2} codeparrot-clean @ 35a59fb of the paper's fit set" if subset
                               else "128 windows x 512 tokens: 64 open-web-math @ fde8ef8 + 64 codeparrot-clean @ "
                               "35a59fb" + ("" if not extension else
                                            f", then {extension['windows']} by the extension rule")),
                      activations="per-token FourOverSix fake quant (STE)", loss="KL to the BF16 teacher",
                      output="last-epoch map (theta > 0 -> E0M3)"),
        model_id=spec.hf_id, revision=prior["revision"], registry_key=spec.key, unit=args.unit,
        e0m3_tiles=summ["e0m3_tiles"], tiles=summ["tiles"], e0m3_share=summ["e0m3_fraction"], modules=summ["modules"],
        calibration_time=dict(wall_seconds=report.get("resources", {}).get("total_seconds"),
                              setup_seconds=report.get("setup_seconds"), training_seconds=report.get("training_seconds")),
        transformers=report.get("host", {}).get("transformers"), torch=report.get("host", {}).get("torch"),
        gpu=(report.get("host", {}).get("gpus") or [None])[0], deviations=report.get("deviations"),
        source_map_sha256=map_sha,
        source=dict(kind="trained", run=str(run_dir), report_sha256=RC.sha256(report_path),
                    calibration_record=str(record), calibration_record_sha256=RC.sha256(record)),
        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    if extension:
        meta["fit_extension"] = {k: v for k, v in extension.items() if k != "records"}
    if source:
        meta["fit_source"] = {k: v for k, v in source.items() if k != "records"}
    target = args.out / "map.pt"
    M.save(target, tiles, args.unit, meta)
    tiles2, unit2, _ = M.load(target)
    bad = check_tiles(tiles2, unit2, shapes)
    if bad or unit2 != args.unit or not all(torch.equal(tiles2[n], tiles[n]) for n in tiles):
        raise SystemExit(f"{target} does not read back: {bad[:5]}")
    rep = dict(model=spec.key, model_key=key, unit=args.unit, flipquant=code, epochs=args.epochs, settings=chosen,
               paper_settings=defaults, non_default_settings=non_default, legacy_options_accepted=accepted,
               teacher_host=args.teacher_host, teacher_storage=report.get("teacher_storage"),
               map=str(target), map_file_sha256=RC.sha256(target), run_map_sha256=map_sha,
               e0m3_share=summ["e0m3_fraction"], wall_seconds=time.time() - started,
               development_records=development, references=reproduction(spec, key, args.unit, map_sha, tiles))
    equal = [v["equal"] for v in rep["references"].values()]
    rep["reproduces_all_references"] = bool(equal) and all(equal)
    (args.out / "reproduction.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    if args.require_reproduction and not rep["reproduces_all_references"]:
        raise SystemExit("the map does not reproduce " + ("its references" if equal else "anything: no reference exists"))


if __name__ == "__main__":
    main()
