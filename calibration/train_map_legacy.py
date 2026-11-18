"""Train a MixFP4 format map: one logit per weight tile, KL to the BF16 teacher (no backtracking).

The co-authors' trainer, kept as ``calibration.train_map_legacy``; ``calibration.train_map`` is the paper's TM-OPT+TC
trainer, which reproduces the paper's maps.

Port of the reference ``run_train_map.py`` (the MixFP4 map only -- the NVFP4 scale-only and joint
variants are deliberately not included). Every tile ``u`` of every quantized linear gets a latent
logit ``theta_u``; the weight is ``W = where(expand(theta > 0), A, B)`` with A = E0M3 (block max on
7) and B = FourOverSix E2M1.

  --param ste      hard map in the forward, straight-through backward dm/dtheta = 1
  --param sigmoid  W = B + sigmoid(theta / tau) (A - B), tau annealed; fake mode only

The loss is the token-mean KL(teacher || student) over 128 math/code calibration windows of 512
tokens (``--fit-windows`` grows the set; the development set stays the same), with per-token FourOverSix activations (straight-through). dKL/dm_u = <dy^T x_q, P_u(A - B)>
is accumulated by FlipLinear's backward. The 192 development windows are evaluated with the hard map
and per-document activations every ``--eval-every`` epochs, for monitoring only. WikiText-2 / C4
PPL are evaluated once on the final map. ``--teacher-topk K`` trains on the KL over the teacher's top-K
tokens plus one bucket for the rest of the vocabulary (the exact KL of that coarse-grained distribution),
with the teacher kept on the GPU instead of on node-local disk.

``--nvidia-nvfp4`` starts from the vendor's ModelOpt NVFP4 release (models/registry.py
``nvfp4_checkpoint``): the teacher is still the original BF16 model, but the student takes the
release's BF16 tensors, quantizes only its quantized linears, uses its exact NVFP4 codes as candidate
B (A = E0M3 of the original BF16 weight under the release's global scale) and its static NVFP4
activation scales, in training and evaluation. The all-B map is then exactly the vendor's model.

``--rotate hadamard`` trains the map in the Hadamard-rotated basis (flipquant/rotate.py): the student's
linears get ``W' = W H`` and ``x' = x H`` before activation quantization, so both candidates are
quantized from ``W'``; the teacher stays the unrotated BF16 model (identical up to rounding). The map
records ``rotate`` in its metadata and is only accepted by evaluations with the same ``--rotate``.

``--mode fake`` runs on any GPU (H100/H200); ``--mode native`` runs the student forward on the
Blackwell FP4 kernels (B200: sm100; RTX Pro 6000: ``--native-backend sm120-capi``), so the map is trained
against the numerics it will be deployed with. The unit must then be a multiple of the kernel's granule (8x64 on sm100).

``--no-dev`` is the real calibration scenario of the paper's runs: no development windows are drawn,
loaded or evaluated (the 128 fit windows are the same ones), and the final WikiText-2 / C4 PPL is off; evaluate
the map separately (evaluation.ppl). ``--deterministic`` turns on ``torch.use_deterministic_algorithms`` and
requires ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (or ``:16:8``) in the environment, as the paper's runs did.

  python -m calibration.train_map_legacy --model qwen3-8b --mode fake --unit 8x64 --out runs/qwen3-8b_8x64
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m calibration.train_map_legacy --model qwen3-8b --unit 8x64 --no-dev \
      --deterministic --out runs/qwen3-8b_8x64_nodev
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flipquant import maps as maps_
from flipquant import rotate as rotate_
from flipquant.data import calibration_fit, calibration_from_record, calibration_sets_extended, eval_windows
from flipquant.linear import QuantSpec
from models import apply_nvfp4_checkpoint, load, quantize
from models.cli import resources, save_json
from evaluation.ppl import PROTOCOL_USE_CACHE, perplexity


def kl_per_sequence(lp, teacher):
    return (teacher.exp() * (teacher - lp)).sum(-1).mean(-1)


def kl_backward(model, ids, teacher, n_seq):
    """Forward/backward of one batch: ``(sum of per-sequence KL) / n_seq``. FlipLinear layers whose
    ``map_grad`` is set accumulate dLoss/dm into it. Returns the per-sequence KL values."""
    with torch.enable_grad():
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
        lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
        t = teacher.to(lp.device).float()
        kl = kl_per_sequence(lp, t)
        (kl.sum() / n_seq).backward()
        return kl.tolist()


def tail_logprob(lp, idx):
    """log of the probability mass outside ``idx`` (exact: logsumexp over the masked remainder). The mask is a
    large finite negative, not -inf, so an empty remainder gives ~-1e30 with a finite gradient."""
    return lp.scatter(-1, idx, -1e30).logsumexp(-1)


def kl_topk_per_sequence(lp, values, idx, tail):
    """KL(teacher || student) over the teacher's top-k tokens plus one bucket for the rest of the vocabulary:
    the exact KL of the coarse-grained distributions, a lower bound on the full KL. ``values`` [B,T,k] and
    ``tail`` [B,T] are teacher log-probs, ``idx`` [B,T,k] their token ids."""
    q_tail = tail_logprob(lp, idx)
    top = (values.exp() * (values - lp.gather(-1, idx))).sum(-1)
    rest = torch.where(torch.isfinite(tail), tail.exp() * (tail - q_tail), torch.zeros_like(tail))
    return (top + rest).mean(-1)


class TopKTeacher:
    """Teacher log-probabilities as the top-k tokens per position (BF16 values, int32 ids) plus the exact
    log-mass of the remaining vocabulary (FP32), kept on the GPU: ~3 MB per 512-token window at k=1000
    instead of 131 MB (Llama vocabulary), so even 20000 windows stay in device memory."""

    def __init__(self, k: int, device):
        self.k, self.device, self.rows, self.tail_mass = k, device, [], []

    def append_logprobs(self, lp):
        values, idx = lp.topk(self.k, dim=-1)
        tail = tail_logprob(lp, idx)
        self.tail_mass.append(float(tail.exp().mean()))
        self.rows.append((values.bfloat16().to(self.device), idx.int().to(self.device), tail.to(self.device)))

    def __len__(self):
        return len(self.rows)

    def batch(self, ids, device):
        values, idx, tail = (torch.cat([self.rows[i][j] for i in ids]).to(device) for j in range(3))
        return values.float(), idx.long(), tail


class DiskTeacher:
    """Teacher log-probabilities kept on (node-local) disk, one file per window, for fit sets whose
    ``[windows, 511, vocab]`` BF16 log-probabilities do not fit in host memory."""

    def __init__(self, root: Path):
        self.root, self.n = root, 0
        root.mkdir(parents=True, exist_ok=True)

    def append(self, t):
        torch.save(t, self.root / f"{self.n:06d}.pt")
        self.n += 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.load(self.root / f"{i:06d}.pt")


@torch.no_grad()
def teacher_logprobs(model, windows, device, out=None):
    out = [] if out is None else out
    for ids in windows:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        lp = logits[:, :-1].float().log_softmax(-1)
        if isinstance(out, TopKTeacher):
            out.append_logprobs(lp)
        else:
            out.append(lp.bfloat16().cpu())
    return out


def resolve_ppl_datasets(args) -> str:
    """The final-PPL datasets: ``--ppl-datasets``, else ``wiki,c4``; ``--no-dev`` turns the final PPL off."""
    if args.no_dev:
        if args.ppl_datasets:
            raise SystemExit("--no-dev turns the final PPL off; drop --ppl-datasets (evaluate with evaluation.ppl)")
        return ""
    return "wiki,c4" if args.ppl_datasets is None else args.ppl_datasets


def deterministic_setup(enabled: bool) -> dict:
    """``--deterministic``: ``torch.use_deterministic_algorithms(True)``. cuBLAS is deterministic only with
    ``CUBLAS_WORKSPACE_CONFIG`` set before its first use, so it must come from the environment."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if enabled:
        if workspace not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) in the environment")
        torch.use_deterministic_algorithms(True)
    return dict(deterministic=enabled, cublas_workspace_config=workspace)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mode", choices=("fake", "native"), default="fake")
    ap.add_argument("--native-backend", choices=("auto", "sm100", "sm120-capi", "sm120-capi-b8", "emulate"),
                    default="auto",
                    help="--mode native: auto = sm100 on sm_100; on sm_120 name sm120-capi (the C-API kernels) or "
                         "sm120-capi-b8 (its weights-on-B 8x64 library)")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--tiny-layers", type=int, default=None,
                    help="TESTS ONLY: random-init model cut to N layers (models/tiny.py)")
    ap.add_argument("--unit", default="8x64", help="tile unit, e.g. 8x64 (sm100 granule), 16x64, 256x64, 1x16")
    ap.add_argument("--param", choices=("ste", "sigmoid"), default="ste")
    ap.add_argument("--fake-weight-cache", choices=("on", "off"), default="on",
                    help="fake mode: keep the dequantized BF16 weights resident (on, default) or decode them from "
                         "the packed candidates when needed (off: ~2 bytes per weight less GPU memory, same numbers)")
    ap.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Adam epsilon; tile gradients are ~1e-6, so the torch default 1e-8 would dominate")
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    ap.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--init-logit", type=float, default=-1.0, help="negative = start at FourOverSix E2M1")
    ap.add_argument("--tau-start", type=float, default=1.0)
    ap.add_argument("--tau-end", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8, help="calibration sequences per forward/backward")
    ap.add_argument("--accum", type=int, default=1, help="forward/backward passes per optimizer step")
    ap.add_argument("--eval-every", type=int, default=2, help="epochs between development evaluations")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--fit-windows", type=int, default=128,
                    help="calibration windows (even, >= 128): the standard 128 plus further documents")
    ap.add_argument("--teacher-dir", type=Path, default=None,
                    help="keep fit-set teacher log-probs on disk here (default: in memory for 128 windows, "
                         "else under $TMPDIR, which run.sbatch puts on node-local /tmp)")
    ap.add_argument("--teacher-topk", type=int, default=0,
                    help="keep only the teacher's top-k log-probs per token (+ one tail bucket) on the GPU and "
                         "train on that KL; 0 = full vocabulary. The development KL stays full-vocabulary")
    ap.add_argument("--calib-record", type=Path, default=None,
                    help="a reference calibration record (report.json) to reproduce its exact calibration windows")
    ap.add_argument("--nvidia-nvfp4", action="store_true",
                    help="candidate B and activation scales from the vendor's NVFP4 release (see above)")
    ap.add_argument("--rotate", choices=rotate_.ROTATIONS, default="none",
                    help="hadamard: learn the map in the rotated basis (see above)")
    ap.add_argument("--rotate-block", type=int, default=16,
                    help="largest Hadamard block (power of two; 16 = one scale group, see flipquant/rotate.py)")
    ap.add_argument("--ppl-datasets", default=None, help="final PPL (default wiki,c4; '' to skip; off with --no-dev)")
    ap.add_argument("--no-dev", action="store_true",
                    help="no development windows (none drawn, loaded or evaluated) and no final PPL")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms; needs CUBLAS_WORKSPACE_CONFIG=:4096:8")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="SMOKE TESTS ONLY: use the first N fit and the first N development windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.param == "sigmoid" and args.fake_weight_cache == "off":
        raise SystemExit("--param sigmoid interpolates the weights, which the packed candidates cannot hold; "
                         "it needs --fake-weight-cache on")
    if args.param == "sigmoid" and args.mode != "fake":
        raise SystemExit("--param sigmoid interpolates weights and exists only in fake mode")
    if args.rotate != "none" and args.nvidia_nvfp4:
        raise SystemExit("--rotate needs our own quantization of the BF16 weights, not --nvidia-nvfp4")
    map_meta = {} if args.rotate == "none" else dict(rotate=args.rotate, rotate_block=args.rotate_block)
    if args.no_dev and args.fit_windows != 128:
        raise SystemExit("--no-dev draws the standard 128 fit windows (an extended set skips development "
                         "documents); drop --fit-windows")
    ppl_datasets = resolve_ppl_datasets(args)
    determinism = deterministic_setup(args.deterministic)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, cols = maps_.parse_unit(args.unit)

    model, tok, spec = load(args.model, revision=args.revision, device_map=args.device_map, tiny_layers=args.tiny_layers)
    if spec.batch1_only and (args.batch != 1 or args.eval_batch != 1):
        raise SystemExit(f"{spec.key}: batched forward is not identical to batch 1; use --batch 1 --eval-batch 1 "
                         "(and --accum 8 to keep 8 sequences per step)")
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", model=spec.key, source=model.flipquant_source,
                  args={k: str(v) for k, v in vars(args).items()}, epochs=[],
                  settings=dict(determinism, no_dev=args.no_dev, ppl_datasets=ppl_datasets))
    save_json(args.out / "report.json", report)

    if args.calib_record and args.fit_windows != 128:
        raise SystemExit("--calib-record reproduces a 128-window record; drop --fit-windows")
    if args.no_dev:          # no development windows at all; the fit windows are the same
        data, dev = (None if args.calib_record else calibration_fit(tok)), []
    else:
        data = calibration_sets_extended(tok, args.fit_windows)
        dev = data["dev"]
    fit = calibration_from_record(tok, args.calib_record)["fit"] if args.calib_record else data["fit"]
    if args.limit_windows:
        fit, dev = fit[:args.limit_windows], dev[:args.limit_windows]
    report["data"] = dict(fit=len(fit), dev=len(dev), record=str(args.calib_record) if args.calib_record else None,
                          meta=data["meta"] if data is not None else None)
    device = model.get_input_embeddings().weight.device
    if args.teacher_topk and args.teacher_dir:
        raise SystemExit("--teacher-topk keeps the teacher on the GPU; drop --teacher-dir")
    teacher_dir = args.teacher_dir
    if teacher_dir is None and len(fit) > 128 and not args.teacher_topk:
        teacher_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"teacher_{os.getpid()}"
    store = (TopKTeacher(args.teacher_topk, device) if args.teacher_topk
             else DiskTeacher(teacher_dir) if teacher_dir else None)
    teacher = teacher_logprobs(model, fit, device, store)
    if args.teacher_topk:
        report["teacher_topk"] = dict(k=args.teacher_topk, mean_tail_mass=sum(teacher.tail_mass) / len(teacher),
                                      max_window_tail_mass=max(teacher.tail_mass))
        print(f"TEACHER top-{args.teacher_topk} mean tail mass {report['teacher_topk']['mean_tail_mass']:.3e}",
              flush=True)
    dev_teacher = teacher_logprobs(model, dev, device)

    given = None
    act, train_scope, eval_scope = "fourover6", "row", "document"
    if args.nvidia_nvfp4:
        ckpt = apply_nvfp4_checkpoint(model, spec)
        given = ckpt["given"]
        report["nvfp4_checkpoint"] = ckpt["info"]
        act, train_scope, eval_scope = "nvfp4", "static", "static"
    rotations = None
    if args.rotate != "none":
        rotations = rotate_.rotate_model(model, spec, args.rotate_block)
        report["rotate"] = dict(kind=args.rotate, max_block=args.rotate_block, **rotate_.summary(rotations))
    qspec = QuantSpec(weight="mixfp4", act=act, act_scope=train_scope, unit=args.unit, mode=args.mode,
                      native_backend=args.native_backend, fake_weight_cache=args.fake_weight_cache == "on")
    layers = quantize(model, spec, qspec, trainable=True, given=given)
    if rotations is not None:
        rotate_.attach(layers, rotations)
    theta = {n: torch.full(maps_.tile_shape(l.out_features, l.in_features, args.unit), args.init_logit,
                           dtype=torch.float32, device=l.gscale.device) for n, l in layers.items()}
    for n, l in layers.items():
        l.set_map(theta[n] > 0)
    report["tiles"] = sum(t.numel() for t in theta.values())
    report["quant"] = qspec.as_dict()
    if args.mode == "native":                     # the backend 'auto' resolved to
        report["native_backend"] = next(iter(layers.values()))._backend.name
    print(f"CANDIDATES modules={len(layers)} tiles={report['tiles']} mode={args.mode}", flush=True)

    tau = [args.tau_start]

    def hard_map():
        return {n: t > 0 for n, t in theta.items()}

    def apply_soft():
        for n, l in layers.items():
            l.set_soft(torch.sigmoid(theta[n] / tau[0]))

    @torch.no_grad()
    def dev_eval():
        """Development CE/KL of the HARD map with per-document activations (monitor only)."""
        if args.param == "sigmoid":
            for n, l in layers.items():
                l.set_map(theta[n] > 0)
        qspec.act_scope = eval_scope
        ce, kl = [], []
        for start in range(0, len(dev), args.eval_batch):
            ids = torch.cat(dev[start:start + args.eval_batch]).to(device)
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = torch.cat(dev_teacher[start:start + args.eval_batch]).to(lp.device).float()
            ce += F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction="none").mean(-1).tolist()
            kl += kl_per_sequence(lp, t).tolist()
            del lp, t
        qspec.act_scope = train_scope
        if args.param == "sigmoid":
            apply_soft()
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), kl_values=kl)

    params = list(theta.values())
    opt = (torch.optim.Adam(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps)
           if args.optimizer == "adam" else torch.optim.SGD(params, lr=args.lr))
    steps_per_epoch = math.ceil(math.ceil(len(fit) / args.batch) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    rng = random.Random(args.seed)

    report["initial_dev"] = dev_eval() if dev else None
    report["setup_seconds"] = time.time() - started
    save_json(args.out / "report.json", report)
    if dev:
        print(f"START dev CE {report['initial_dev']['ce']:.6f} KL {report['initial_dev']['kl']:.6f} "
              f"steps/epoch={steps_per_epoch} total={total_steps}", flush=True)
    else:
        print(f"START (no development set) steps/epoch={steps_per_epoch} total={total_steps}", flush=True)
    if args.param == "sigmoid":
        apply_soft()

    step, previous = 0, hard_map()
    for epoch in range(args.epochs):
        t0 = time.time()
        order = list(range(len(fit)))
        rng.shuffle(order)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        losses, flips = [], 0
        for g0 in range(0, len(batches), args.accum):
            group = batches[g0:g0 + args.accum]
            n_seq = sum(len(b) for b in group)
            for n, l in layers.items():
                l.map_grad = torch.zeros_like(theta[n])
            with torch.enable_grad():
                for idx in group:
                    ids = torch.cat([fit[i] for i in idx]).to(device)
                    embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
                    lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                    if args.teacher_topk:
                        t = teacher.batch(idx, lp.device)
                        kl = kl_topk_per_sequence(lp, *t)
                    else:
                        t = torch.cat([teacher[i] for i in idx]).to(lp.device).float()
                        kl = kl_per_sequence(lp, t)
                    (kl.sum() / n_seq).backward()
                    losses += kl.tolist()
                    del embeds, lp, t, kl
            if args.schedule == "cosine":
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for n, p in theta.items():
                g = layers[n].map_grad
                if args.param == "sigmoid":
                    s = torch.sigmoid(p / tau[0])
                    g = g * s * (1 - s) / tau[0]
                p.grad = g.clone()
                layers[n].map_grad = None
            opt.step()
            step += 1
            if args.param == "sigmoid":
                tau[0] = args.tau_start * (args.tau_end / args.tau_start) ** (step / total_steps)
                apply_soft()
            for n, l in layers.items():
                now = theta[n] > 0
                changed = int((now != previous[n]).sum())
                if changed:
                    flips += changed
                    previous[n] = now
                    if args.param == "ste":
                        l.set_map(now)
        e0m3 = sum(int((t > 0).sum()) for t in theta.values())
        entry = dict(epoch=epoch, step=step, train_kl=sum(losses) / len(losses), e0m3_units=e0m3,
                     hard_flips=flips, tau=tau[0], lr=opt.param_groups[0]["lr"], epoch_seconds=time.time() - t0)
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            if dev:
                d = dev_eval()
                entry.update(dev_ce=d["ce"], dev_kl=d["kl"], dev_kl_values=d["kl_values"])
            maps_.save(args.out / f"map_epoch{epoch + 1:03d}.pt", hard_map(), args.unit,
                       dict(model=spec.key, epoch=epoch + 1, **map_meta))
        report["epochs"].append(entry)
        save_json(args.out / "report.json", report)
        print("EPOCH " + str({k: v for k, v in entry.items() if k != "dev_kl_values"}), flush=True)

    final = hard_map()
    if isinstance(teacher, DiskTeacher) and args.teacher_dir is None:
        shutil.rmtree(teacher.root, ignore_errors=True)       # 100s of GB on the node's /tmp
    maps_.save(args.out / "map.pt", final, args.unit,
               dict(model=spec.key, source=model.flipquant_source, mode=args.mode, param=args.param,
                    epochs=args.epochs, **map_meta))
    torch.save({n: t.cpu() for n, t in theta.items()}, args.out / "theta.pt")
    for n, l in layers.items():
        l.set_map(final[n])
    report["final_dev"] = dev_eval() if dev else None
    report["map"] = maps_.summary(final)
    report["optimization_seconds"] = time.time() - started - report["setup_seconds"]
    save_json(args.out / "report.json", report)
    if dev:
        print(f"FINAL dev CE {report['final_dev']['ce']:.6f} KL {report['final_dev']['kl']:.6f} "
              f"e0m3={report['map']['e0m3_fraction']:.4f}", flush=True)
    else:
        print(f"FINAL e0m3={report['map']['e0m3_fraction']:.4f}", flush=True)

    report["evaluation"] = {}
    qspec.act_scope = eval_scope
    for name in filter(None, ppl_datasets.split(",")):
        windows, _ = eval_windows(tok, name, 2048)
        r = perplexity(model, windows, use_cache=PROTOCOL_USE_CACHE[name])
        report["evaluation"][name] = dict(ppl=r["ppl"], nll=r["nll"])
        print(f"PPL train_{args.param}_{args.unit} {name} {r['ppl']:.6f}", flush=True)
        save_json(args.out / "report.json", report)
    report["resources"] = resources(started)
    report["status"] = "complete"
    save_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
