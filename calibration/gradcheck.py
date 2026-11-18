"""Appendix F.1's check of the tile gradient ("Does the tile gradient rank flips correctly?"), re-implemented from the
paper's text and its source comments (NVFP4-RaZeR paper-eval part AC item 13; the original scripts, flipquant branch
gradcheck aab768f / 3f7f4e3, were never pushed).

    python -m calibration.gradcheck --model mistral-7b --unit 16x64 --oneshot ONESHOT.pt --map MAP.pt \\
        --fit-windows 256 --heldout 512:64:128 [--data-root ROOT] --out OUT.json

**g_u.** The trainer's own tile gradient at the all-E2M1 map: ``calibration.tmopt_ext oneshot`` over the fit windows
(``ONESHOT.run/oneshot_grads.pt``), i.e. the sum over one epoch's optimizer steps of each step's gradient of its mean
bucketed KL, divided here by the number of steps (every step has the same number of sequences), so g_u is the
derivative of the mean bucketed KL over the fit windows and its loss is the trainer's.

**The measurement.** The trainer's training forward at a hard map (TM-OPT+TC; ``calibration/tmopt/run_train_map.py``):
every quantized linear's weight is B where the map says E2M1 and A where it says E0M3 (B = ``quant_nvfp4_4over6``,
the FourOverSix E2M1 candidate; A = ``quant_mix_4_6(..., elect='always')``, the E0M3 candidate), its input is quantized
per token by FourOverSix (``fourover6_rows``, straight-through, the trainer's pre-hook), and the loss of a sequence is
the token mean of the bucketed KL from the BF16 teacher (its top-1,000 tokens plus one tail bucket;
``topk_teacher.kl_topk_tokens``, two documents at a time as ``chunked_loss``). The forward runs under autograd with the
embeddings requiring grad, as the trainer's does; nothing is updated. A set of tiles is flipped from the all-E2M1 map;
its change is the per-sequence KL minus the all-E2M1 one, paired over the sequences (mean ± 2 SE). Two sets of 512-token
sequences: **in-sample**, the fit windows (evaluated in the trainer's epoch-0 batches, ``random.Random(0)``), and
**held-out**, ``--heldout N:START:COUNT``: per source (math, code) the extension rule's windows START .. START+COUNT-1
of an N-window fit set (``fit_extension.extend``), disjoint from the fit windows (checked by token hash).

**Sets of tiles.**
- (a) Ten bins by the rank of g_u (most negative first): 0-0.1, 0.1-0.3, 0.3-1, 1-3, 3-10, 10-50, 50-90, 90-97,
  97-99.7 and 99.7-100 % of the tiles; from each, three disjoint groups of 1,000 tiles drawn uniformly
  (``torch.Generator`` seeded ``--seed`` + bin). For each group: the predicted change (sum of g_u), the
  activation-weighted score (``baseline_maps.act_scores`` on the fit windows' BF16 inputs: the drop in the layer-output
  error from E2M1 to E0M3, summed over the group) and the weight-error score (sum over the group's tiles of
  ``||W - B||^2 - ||W - A||^2``). Spearman correlation of each score with the measured decrease of the KL, over the 30
  groups, with a 95 % bootstrap interval over groups.
- (b) The k tiles with the most negative g_u over the whole model, for the k of ``--topk`` (and ``--extra-k``).
- The map of ``--map``, and the rank distribution of its tiles under g_u.

**Checks.** The fit windows' token hashes are those of the one-shot run's trainer report; the held-out windows are
disjoint from them; the in-sample all-E2M1 mean KL against the trainer's epoch-0 KL of the one-shot run (whose map never
moves); the number of optimizer steps.

Outputs: ``OUT.json`` (report; ``status: complete`` last), ``OUT.sets.pt`` (each set's tile indices and per-sequence KLs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from calibration import tmopt_common as RC  # noqa: E402

BINS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.10, 0.50, 0.90, 0.97, 0.997, 1.0)
BIN_NAMES = ("0-0.1", "0.1-0.3", "0.3-1", "1-3", "3-10", "10-50", "50-90", "90-97", "97-99.7", "99.7-100")
TOPK = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000)
GROUP, GROUPS_PER_BIN = 1000, 3
TEACHER_K = 1000
BATCH = 8
BOOT = 10000


def engine():
    """The vendored trainer's modules, imported as run_train_map.py imports them."""
    for p in (str(RC.ENGINE), str(RC.ROOT)):
        if p not in sys.path:
            sys.path.insert(0 if p == str(RC.ENGINE) else len(sys.path), p)
    import fit_extension
    import topk_teacher
    from fit_extension import token_sha
    from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
    from quantize.fused_fourover6 import fourover6_rows
    from quantize.causal_four_over_six import quantize_rows
    from run_math_code_calibration import load_model, math_code_data
    from run_multiround import PUBLISHED, UNITS, data_paths, expand, reduce
    return dict(fit_extension=fit_extension, topk_teacher=topk_teacher, quant_mix_4_6=quant_mix_4_6,
                quant_nvfp4_4over6=quant_nvfp4_4over6, fourover6_rows=fourover6_rows, quantize_rows=quantize_rows,
                load_model=load_model, math_code_data=math_code_data, PUBLISHED=PUBLISHED, UNITS=UNITS,
                data_paths=data_paths, expand=expand, reduce=reduce, token_sha=token_sha)


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).statistic
    return float(r)


def bootstrap(x, y, seed=0, n=BOOT):
    import numpy as np
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        vals.append(spearman(x[i], y[i]))
    vals = np.sort(np.asarray(vals))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))], len(vals)


def paired(new, base):
    d = (new.double() - base.double())
    n = d.numel()
    m = float(d.mean())
    se = float(d.std(unbiased=True) / math.sqrt(n))
    return dict(mean=m, two_se=2 * se, n=n, beyond_2se=abs(m) > 2 * se)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--unit", default="16x64")
    ap.add_argument("--oneshot", type=Path, required=True, help="calibration.tmopt_ext oneshot's output map (.pt)")
    ap.add_argument("--map", type=Path, required=True, help="the calibrated map to measure (its tiles' g_u ranks too)")
    ap.add_argument("--fit-windows", type=int, required=True, help="the one-shot run's fit windows (in-sample)")
    ap.add_argument("--heldout", required=True, help="N:START:COUNT of the extension rule, per source")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--topk", default=",".join(map(str, TOPK)))
    ap.add_argument("--extra-k", default="", help="more k (comma-separated), e.g. a map's size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=None, metavar="W",
                    help="TESTS ONLY: W windows per set, the baseline, two groups, top-10 and the map")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    E = engine()
    spec, key = RC.resolve(args.model)
    rows, cols = E["UNITS"][args.unit]
    root = RC.data_root(args.data_root)
    calibration, development = E["data_paths"](key, root)
    prior = json.loads((calibration / "report.json").read_text())
    report = dict(status="running", model=spec.key, trainer_key=key, unit=args.unit, fit_windows=args.fit_windows,
                  heldout=args.heldout, seed=args.seed, oneshot=str(args.oneshot), map=str(args.map), checks={})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    save()

    # ---------------------------------------------------------------- g_u: the one-shot run's summed gradient / steps
    one = json.loads(args.oneshot.with_suffix(".json").read_text())
    trainer_rep = json.loads((args.oneshot.with_suffix(".run") / "run" / "report.json").read_text())
    grads = torch.load(args.oneshot.with_suffix(".run") / "oneshot_grads.pt", map_location="cpu", weights_only=True)
    steps = int(grads["steps"])
    if trainer_rep["args"].get("fit_windows") != args.fit_windows:
        raise SystemExit(f"the one-shot run used {trainer_rep['args'].get('fit_windows')} fit windows, not {args.fit_windows}")
    batch, accum = trainer_rep["args"]["batch"], trainer_rep["args"].get("accum", 1)
    report["checks"]["steps"] = dict(steps=steps, expected=math.ceil(math.ceil(args.fit_windows / batch) / accum),
                                     batch=batch, accum=accum)
    if steps != report["checks"]["steps"]["expected"] or args.fit_windows % (batch * accum):
        raise SystemExit(f"one-shot steps {steps}: the per-step means are not equal-weight")

    # ---------------------------------------------------------------- model, windows, teacher
    model, modules = E["load_model"](prior, False)
    model.set_attn_implementation("sdpa")
    device = model.get_input_embeddings().weight.device
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(prior["source"], revision=prior["revision"])
    fit, _ = E["math_code_data"](tok, prior["fit"])
    fit = [b for source in ("math", "code") for b in fit[source]]
    skip = E["fit_extension"].skip_documents(prior, development, E["PUBLISHED"].get(key))
    if args.fit_windows != 128:
        extra, _ = E["fit_extension"].extend(tok, prior, args.fit_windows, skip)
        fit += extra
    want = ([h for s in ("math", "code") for h in prior["fit"][s]["token_sha256"]]
            + [r["token_sha256"] for r in (trainer_rep.get("fit_extension") or {}).get("records", [])])
    got = [E["token_sha"](w) for w in fit]
    report["checks"]["fit_windows_equal_trainer"] = got == want
    if got != want:
        raise SystemExit("the fit windows are not the one-shot run's")
    n_h, start, count = (int(x) for x in args.heldout.split(":"))
    per = n_h // 2 - 64
    ext_h, _ = E["fit_extension"].extend(tok, prior, n_h, skip)
    heldout = ext_h[start:start + count] + ext_h[per + start:per + start + count]
    h_hash = [E["token_sha"](w) for w in heldout]
    report["checks"]["heldout"] = dict(windows=len(heldout), disjoint=not set(h_hash) & set(got),
                                       distinct=len(set(h_hash)) == len(h_hash))
    if set(h_hash) & set(got) or len(heldout) != 2 * count:
        raise SystemExit("the held-out windows overlap the fit windows")
    report["windows"] = dict(fit_sha256=hashlib.sha256("".join(got).encode()).hexdigest(),
                             heldout_sha256=hashlib.sha256("".join(h_hash).encode()).hexdigest())
    save()
    t0 = time.time()
    if args.smoke:
        fit, heldout = fit[:args.smoke], heldout[:args.smoke]
        report["smoke"] = args.smoke
    teacher = {}
    with torch.no_grad():
        for name, ws in (("fit", fit), ("heldout", heldout)):
            rows_ = []
            for ids in ws:
                logits = model(input_ids=ids.to(device), use_cache=False).logits
                r, _ = E["topk_teacher"].topk_rows(logits[:, :-1].float().log_softmax(-1), TEACHER_K)
                rows_.append(r)
            del logits
            teacher[name] = rows_
    report["seconds"] = dict(teacher=time.time() - t0)

    # ---------------------------------------------------------------- scores: activation-weighted, weight error
    from calibration import baseline_maps as BM
    t0 = time.time()
    grams = BM.input_gram_blocks(model, modules, fit, device, BATCH, cols)
    act, werr, cand = {}, {}, {}
    names = list(modules)
    for n in names:
        lin = modules[n]
        act[n] = BM.act_scores(lin, grams.pop(n), args.unit).float().cpu()
        w = lin.weight.detach()
        b = E["quant_nvfp4_4over6"](w, 4, 16)
        a = E["quant_mix_4_6"](w, 4, 16, type_block=(8, 64), clip="a1", elect="always")
        werr[n] = E["reduce"]((w.float() - b.float()) ** 2 - (w.float() - a.float()) ** 2, rows, cols).float().cpu()
        cand[n] = (b, a)
    del grams
    torch.cuda.empty_cache()
    report["seconds"]["scores"] = time.time() - t0
    g = {n: (grads["sum"][n].double() / steps).float() for n in names}
    for n in names:
        if g[n].shape != act[n].shape or g[n].shape != werr[n].shape:
            raise SystemExit(f"{n}: tile shapes differ ({tuple(g[n].shape)}, {tuple(act[n].shape)}, {tuple(werr[n].shape)})")
    shapes = {n: tuple(g[n].shape) for n in names}
    sizes = [g[n].numel() for n in names]
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    flat_g = torch.cat([g[n].reshape(-1) for n in names]).double()
    flat_act = torch.cat([act[n].reshape(-1) for n in names]).double()
    flat_w = torch.cat([werr[n].reshape(-1) for n in names]).double()
    total = flat_g.numel()
    order = torch.sort(flat_g, stable=True).indices          # most negative first; ties by tile index
    rank = torch.empty_like(order)
    rank[order] = torch.arange(total)
    report["tiles"] = dict(total=total, negative=int((flat_g < 0).sum()), zero=int((flat_g == 0).sum()),
                           g_sum=float(flat_g.sum()), g_min=float(flat_g.min()), g_max=float(flat_g.max()))

    from flipquant import maps as M
    mtiles, munit, _ = M.load(args.map, args.unit)
    if set(mtiles) != set(names) or any(tuple(mtiles[n].shape) != shapes[n] for n in names):
        raise SystemExit("--map does not cover the model's quantized modules")
    flat_map = torch.cat([mtiles[n].reshape(-1) for n in names])
    map_idx = torch.nonzero(flat_map).flatten()

    def masks(idx):
        """global tile indices -> {module: bool tile mask}"""
        sel = torch.zeros(total, dtype=torch.bool)
        sel[idx] = True
        return {n: sel[offsets[i]:offsets[i + 1]].reshape(shapes[n]) for i, n in enumerate(names)}

    # ---------------------------------------------------------------- the trainer's forward and per-sequence KL
    row_checks = [0]

    def act_hook(module, inputs):
        x = inputs[0]
        q = E["fourover6_rows"](x.detach())
        if row_checks[0] < 64:
            assert torch.equal(q.view(torch.int16), E["quantize_rows"](x.detach()).view(torch.int16)), \
                "fused per-token activation quantizer differs from quantize_rows"
            row_checks[0] += 1
        return (q + (x - x.detach()), *inputs[1:])

    handles = [m.register_forward_pre_hook(act_hook) for m in modules.values()]
    rng = random.Random(0)                                    # the trainer's epoch-0 batches (seed 0)
    perm = list(range(len(fit)))
    rng.shuffle(perm)
    fit_batches = [perm[i:i + batch] for i in range(0, len(perm), batch)]
    if args.smoke:
        fit_batches = [list(range(i, min(i + batch, len(fit)))) for i in range(0, len(fit), batch)]
    held_batches = [list(range(i, min(i + BATCH, len(heldout)))) for i in range(0, len(heldout), BATCH)]

    def install(sel):
        with torch.no_grad():
            for n, m in modules.items():
                b, a = cand[n]
                s = sel.get(n)
                if s is None or not bool(s.any()):
                    m.weight.copy_(b)
                else:
                    m.weight.copy_(torch.where(E["expand"](s.to(b.device), rows, cols, b.shape[0]), a, b))

    def kls(windows, batches, rows_t):
        out = torch.empty(len(windows), dtype=torch.float64)
        for idx in batches:
            ids = torch.cat([windows[i] for i in idx]).to(device)
            embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
            with torch.enable_grad():
                logits = model(inputs_embeds=embeds, use_cache=False).logits
            source = logits.detach()
            del logits, embeds
            with torch.no_grad():
                kl_tok = torch.empty(source.shape[0], source.shape[1] - 1, dtype=torch.float32, device=device)
                for d0 in range(0, source.shape[0], 2):
                    d1 = min(d0 + 2, source.shape[0])
                    lp = source[d0:d1, :-1].float().log_softmax(-1)
                    values, tidx, tail = E["topk_teacher"].batch_rows([rows_t[i] for i in idx[d0:d1]], device)
                    kl_tok[d0:d1] = E["topk_teacher"].kl_topk_tokens(lp, values, tidx, tail)
                    del lp, values, tidx, tail
                seq = kl_tok.mean(-1)
            for j, i in enumerate(idx):
                out[i] = float(seq[j])
            del source, kl_tok, seq
        return out

    def measure(sel):
        install(sel)
        return dict(fit=kls(fit, fit_batches, teacher["fit"]), heldout=kls(heldout, held_batches, teacher["heldout"]))

    t0 = time.time()
    base = measure({})
    report["seconds"]["one_evaluation"] = time.time() - t0
    trainer_kl = (trainer_rep.get("epochs") or [{}])[0].get("train_kl")
    ordered = [float(base["fit"][i]) for b_ in fit_batches for i in b_]
    mean_in_order = sum(ordered) / len(ordered)
    report["checks"]["baseline_kl"] = dict(evaluator_mean=float(base["fit"].mean()), evaluator_mean_trainer_order=mean_in_order,
                                           trainer_epoch0_train_kl=trainer_kl,
                                           rel_diff=None if trainer_kl is None else abs(mean_in_order - trainer_kl) / trainer_kl,
                                           bitwise=trainer_kl is not None and mean_in_order == trainer_kl)
    report["baseline"] = dict(fit_mean_kl=float(base["fit"].mean()), heldout_mean_kl=float(base["heldout"].mean()))
    save()
    sets, record = {}, {}

    def run_set(name, idx, **info):
        r = measure(masks(idx))
        res = dict(info, tiles=int(idx.numel()), predicted=float(flat_g[idx].sum()),
                   act_score=float(flat_act[idx].sum()), weight_score=float(flat_w[idx].sum()),
                   heldout=paired(r["heldout"], base["heldout"]), insample=paired(r["fit"], base["fit"]))
        for k in ("heldout", "insample"):
            res[k]["ratio"] = res[k]["mean"] / res["predicted"] if res["predicted"] else None
        record[name] = dict(idx=idx.clone(), fit=r["fit"], heldout=r["heldout"])
        sets[name] = res
        print(f"SET {name}: {res['tiles']} tiles, predicted {res['predicted']:+.3e}, held-out "
              f"{res['heldout']['mean']:+.3e} ± {res['heldout']['two_se']:.1e}, in-sample {res['insample']['mean']:+.3e} ± "
              f"{res['insample']['two_se']:.1e}", flush=True)
        return res

    # (a) the 30 groups
    gen_groups = []
    bins = []
    for b_, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        r0, r1 = int(math.floor(lo * total)), int(math.floor(hi * total))
        members = order[r0:r1]
        gen = torch.Generator().manual_seed(args.seed + b_)
        pick = members[torch.randperm(members.numel(), generator=gen)[:GROUP * GROUPS_PER_BIN]]
        bins.append(dict(name=BIN_NAMES[b_], rank_range=[r0, r1], size=int(members.numel()),
                         g_range=[float(flat_g[members].min()), float(flat_g[members].max())]))
        for j in range(GROUPS_PER_BIN):
            gen_groups.append((b_, j, pick[j * GROUP:(j + 1) * GROUP]))
    report["bins"] = bins
    report["sets"] = sets
    if args.smoke:
        gen_groups = [gen_groups[0], gen_groups[-1]]
    for b_, j, idx in gen_groups:
        run_set(f"group_{BIN_NAMES[b_]}_{j}", idx, kind="group", bin=BIN_NAMES[b_], draw=j)
        save()
    # (b) top-k, and the map
    ks = sorted({int(k) for k in args.topk.split(",") if k} | {int(k) for k in args.extra_k.split(",") if k}
                | {int(map_idx.numel())})
    if args.smoke:
        ks = [10]
    for k in ks:
        run_set(f"topk_{k}", order[:k], kind="topk", k=k)
        save()
    run_set("map", map_idx, kind="map")
    report["sets"] = sets
    # ---------------------------------------------------------------- the analysis
    groups = [sets[f"group_{BIN_NAMES[b_]}_{j}"] for b_, j, _ in gen_groups]
    ana = dict(spearman={}, beyond_2se={}, middle_bins_pooled={})
    for where in (() if args.smoke else ("heldout", "insample")):
        dec = [-x[where]["mean"] for x in groups]
        for score, vals in (("g_u", [-x["predicted"] for x in groups]), ("act", [x["act_score"] for x in groups]),
                            ("weight_error", [x["weight_score"] for x in groups])):
            ci, used = bootstrap(vals, dec, seed=args.seed)
            ana["spearman"].setdefault(score, {})[where] = dict(rho=spearman(vals, dec), ci95=ci, resamples=used)
        beyond = [x for x in groups if x[where]["beyond_2se"]]
        ana["beyond_2se"][where] = dict(groups=len(beyond), agree_with_sign_of_g=sum(
            (x[where]["mean"] < 0) == (x["predicted"] < 0) for x in beyond))
        mids = [f"group_{BIN_NAMES[b_]}_{j}" for b_, j, _ in gen_groups if 1 <= b_ <= 8]
        key_ = "fit" if where == "insample" else "heldout"
        pooled = torch.stack([record[m][key_] - base[key_] for m in mids]).mean(0)
        ana["middle_bins_pooled"][where] = dict(groups=len(mids), **paired(pooled, torch.zeros_like(pooled)))
    m_rank = rank[map_idx].double() / total
    edges = sorted({0.0, 0.001, map_idx.numel() / total, 0.03, 0.10, 0.50, 1.0})
    ana["map_ranks"] = dict(tiles=int(map_idx.numel()), in_top_same_size=int((rank[map_idx] < map_idx.numel()).sum()),
                            positive_g=int((flat_g[map_idx] > 0).sum()), median_rank=float(m_rank.median()),
                            bins={f"{100 * lo:.3g}-{100 * hi:.3g} %": float(((m_rank >= lo) & (m_rank < hi)).double().mean())
                                  for lo, hi in zip(edges[:-1], edges[1:])})
    report["analysis"] = ana
    report["checks"]["act_quant_rows_checked"] = row_checks[0]
    for h in handles:
        h.remove()
    torch.save(record, args.out.with_suffix(".sets.pt"))
    report["sets_file"] = str(args.out.with_suffix(".sets.pt"))
    report["resources"] = dict(wall_seconds=time.time() - t_start,
                               peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    report["status"] = "complete"
    save()
    sp = ana["spearman"].get("g_u")
    print(("GRADCHECK " + (f"g_u Spearman held-out {sp['heldout']['rho']:+.2f}, in-sample {sp['insample']['rho']:+.2f}; "
                           if sp else "(smoke) ")
           + f"map {sets['map']['heldout']['mean']:+.3e} / {sets['map']['insample']['mean']:+.3e}"), flush=True)


if __name__ == "__main__":
    main()
