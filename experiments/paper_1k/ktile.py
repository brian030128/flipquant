"""Item 9's harness: Tables D.2 (`tab:app-ktile`, per-MMA branch vs per-K-tile dispatch) and D.3 (`tab:app-c2`, the
overhead split before and after the optimizations) at 4096^3 on the RTX PRO 6000, in one session.

    python -m experiments.paper_1k.ktile build  [--build-dir DIR]                 # CPU: the seven extra builds + SASS gate
    python -m experiments.paper_1k.ktile census --out census.json                # CPU: static SASS census (cuobjdump)
    python -m experiments.paper_1k.ktile check  --maps256 a,b,c --maps1k a,b,c --out check.json   # GPU, bitwise
    python -m experiments.paper_1k.ktile time   --maps256 a,b,c --maps1k a,b,c --out time.json    # GPU, the session

Ported from NVFP4-RaZeR kernel-opt amendment 20 (experiments/kernel_opt/ktile_ablation.py) and its C2 measurements
(c2_t0.py, c2_w8.py, c2_U.py, c2_V.py: the builds and the parts), onto this repository's kernels/sm120.

- **Operands** (as amendment 20 / C2U): weights N(0, 0.02) seed 4096; 4096 tokens N(0, 1) quantized FourOverSix per
  token; n = k = t = 4096. Tags: layer 0 o_proj of Mistral-7B's maps of each unit (`--maps256` the release maps: step
  0; `--maps1k` the 1k maps), all-E2M1, and for the check all-E0M3 and a seeded 30 % random set. A 256x64 map's tags
  are stored as its 16x64 granules (as the deployment's install does).
- **Kernels.** From the deployment build (`build_deploy.sh`, `--deploy-dir`): stock_ko, mixed_ko, mixed_wB_ko and
  mixed256_ko, each the set's pick and scheduler row at this shape; and the paper's first kernels n16k64_wA and
  n8k64_wB. From the extra build (`build`): the per-MMA builds n16k64_wA_e64_t0_permma / n8k64_wB_t0_permma, the
  no-dispatch ceilings n16k64_wA_nodisp_e64_t0 / n8k64_wB_nodisp_t0, and the first kernels' no-dispatch builds with
  and without the site-0 tags (n16k64_wA_nodisp / _nodisp_t0, n8k64_wB_nodisp). cuBLAS FP8 (`torch._scaled_mm`) as a
  reference. The ceilings take FourOverSix weights (all E2M1), stock NVFP4 weights, as C2.
- **Timing** (the paper's cold modes): iso_cold, per call a 512 MiB read-flush and the next weight copy of a rotation
  over more than 4x the L2, 3 warm-up and 30 timed calls; b2b_cold, 20 back-to-back calls over the rotating copies.
  CUPTI kernel times. Per repeat: 3 rounds in a rotated order, every configuration in each round; the repeat's value
  is the median of the round medians. `--repeats` (10) repeats in one process; all configurations interleaved.
- **The parts of D.3** (each the ratio of two configurations that differ in one respect, in %):
  tiles = no-dispatch build (without tags) / stock_ko; tag copies (first kernels only) = the no-dispatch build with
  the site-0 tags / without; dispatch = the kernel with all tiles E2M1 / its no-dispatch build; E0M3 tiles = the real
  tags / all E2M1; total = the real tags / stock_ko. The first 256x64 kernel is the 16-arm 16x64 kernel with the
  256x64 tags (its tiles and tag copies are the 16x64 builds'). Each kernel runs with its set's scheduler row at this
  shape (what `auto` gives), each no-dispatch ceiling with its kernel's; the first kernels and their no-dispatch
  builds with the default (0, 1), as C2 ran them.
- **Checks:** before timing, each per-MMA output equals its per-K-tile build's bit for bit (every tag set); no other
  compute process on the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SM120 = ROOT / "kernels" / "sm120"
REF = json.loads((Path(__file__).resolve().parent / "reference" / "item9.json").read_text())
N_ = K_ = T_ = 4096
FLOP = 2 * N_ * K_ * T_
LAYER = REF["layer"]
UNITS = ("8x64", "16x64", "256x64")


def extra_build_dir() -> Path:
    return Path(os.environ.get("PAPER1K_BUILD_ITEM9") or SM120 / "build_item9")


def deploy_dir() -> Path:
    return Path(os.environ.get("PAPER1K_BUILD") or SM120 / "build_deploy")


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


# ------------------------------------------------------------------------------------------------------------- build
def build(args):
    """The seven extra builds (CPU), with the reference defines; then each one's SASS against the reference."""
    d = Path(args.build_dir or extra_build_dir())
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, SM120_BUILD_DIR=str(d), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if os.environ.get("CUTLASS_DIR"):
        env["RAZER_CUTLASS"] = os.environ["CUTLASS_DIR"]
    procs = []
    for b in REF["builds"]:
        if (d / b["config"] / "manifest.json").exists():
            continue
        cmd = [sys.executable, str(SM120 / "build.py")]
        for k, v in sorted(b["extra_defines"].items()):
            cmd += ["--define", f"{k}={v}"]
        cmd += ["--config", b["config"]]
        procs.append((b["config"], subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(d / "logs" / f"{b['config']}.log", "w"),
                                                   stderr=subprocess.STDOUT)))
    bad = [c for c, p in procs if p.wait() != 0]
    if bad:
        raise SystemExit(f"builds failed: {bad} (logs in {d / 'logs'})")
    res = dict(build_dir=str(d), builds={})
    for b in REF["builds"]:
        m = json.loads((d / b["config"] / "manifest.json").read_text())
        res["builds"][b["config"]] = dict(sass=m["sass_sha256"] == b["sass_sha256"],
                                          unpatched=m.get("unpatched_sass_sha256") == b["unpatched_sass_sha256"],
                                          defines=m.get("extra_defines") or {}, want_defines=b["extra_defines"])
    res["passed"] = all(v["sass"] and v["unpatched"] and v["defines"] == v["want_defines"] for v in res["builds"].values())
    res["status"] = "complete"
    write(d / "item9_sass_gate.json", res)
    print(json.dumps(res, indent=1))
    if not res["passed"]:          # the driver's gate item9-sass decides; standalone, say it loudly
        print("FAIL: an item-9 build's SASS differs from its reference (a different toolkit or sources)", flush=True)


# ------------------------------------------------------------------------------------------------------------ census
CENSUS = {"stock_ko": ("deploy", "stock_wA_e64"), "ktile16": ("deploy", "n16k64_wA_e64_t0"),
          "permma16": ("extra", "n16k64_wA_e64_t0_permma"), "ktile8": ("deploy", "n8k64_wB_t0"),
          "permma8": ("extra", "n8k64_wB_t0_permma")}


def census(args):
    from .sass_census import census_lib
    cuobjdump = str(Path(os.environ["CUDA_HOME"]) / "bin" / "cuobjdump") if os.environ.get("CUDA_HOME") else "cuobjdump"
    res = dict(caveat="static SASS census, not measured counters", kernels={}, gates={})
    for key, (where, cfg) in CENSUS.items():
        root = deploy_dir() if where == "deploy" else extra_build_dir()
        man = json.loads((root / cfg / "manifest.json").read_text())
        c = census_lib(root / cfg / man["library"], cuobjdump)
        c.pop("loops")
        res["kernels"][key] = dict(config=cfg, sass_sha256=man["sass_sha256"], **c)
        want = REF["census"][key]
        res["gates"][key] = all(c[k] == want[k] for k in ("omma", "omma_predicated", "brx", "omma_per_steady_iteration",
                                                          "tensor_pipe_estimate_4096")) and man["sass_sha256"] == want["sass_sha256"]
        print(key, cfg, {k: c[k] for k in ("omma", "omma_predicated", "brx", "omma_per_steady_iteration")}, res["gates"][key],
              flush=True)
    res["passed"] = all(res["gates"].values())
    res["status"] = "complete"
    write(args.out, res)
    if not res["passed"]:          # the driver's gate item9-census decides
        print("FAIL: the census differs from amendment 20's", flush=True)


# ---------------------------------------------------------------------------------------------- operands and kernels
def _pkg():
    for p in (str(SM120), str(SM120 / "bench")):
        if p not in sys.path:
            sys.path.insert(0, p)


def operands(maps256: dict, maps1k: dict):
    """(w, x, xq, weights(kind, mask, tb), {(unit, tagset): (mask, tb, share)})"""
    import torch
    _pkg()
    from kernel import place
    from mixfp4_sm120 import numerics as N
    sys.path.insert(0, str(ROOT))
    from flipquant import maps as M
    g = torch.Generator("cpu").manual_seed(4096)
    w = (torch.randn(N_, K_, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(T_, K_, generator=g).cuda().bfloat16()
    xn, xsb, gsx = N.quantize_act(x, "four_over_six_rows")
    xq = (N.pack_nibbles(xn), place(xsb, K_), gsx)

    def weights(kind, mask=None, tb=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, tb)
        return N.pack_nibbles(wn), place(wsb, K_), float(gsw)

    tags = {}
    rows = {"8x64": 8, "16x64": 16, "256x64": 256}
    for name, maps in (("real256", maps256), ("real1k", maps1k)):
        for u in UNITS:
            t = M.load(maps[u])[0][LAYER].bool()
            r = rows[u]
            mask, tb = (t, (r, 64)) if r in (8, 16) else (t.repeat_interleave(r // 16, 0)[:N_ // 16], (16, 64))
            assert tuple(mask.shape) == (N_ // tb[0], K_ // 64), (name, u, tuple(mask.shape))
            tags[(u, name)] = (mask, tb, float(t.float().mean()))
    for u in UNITS:
        tb = (8, 64) if u == "8x64" else (16, 64)
        grid = (N_ // tb[0], K_ // 64)
        tags[(u, "e2m1")] = (torch.zeros(grid, dtype=torch.bool), tb, 0.0)
        tags[(u, "e0m3")] = (torch.ones(grid, dtype=torch.bool), tb, 1.0)
        tags[(u, "rand30")] = (torch.rand(grid, generator=torch.Generator("cpu").manual_seed(20261009)) < 0.3, tb, 0.3)
    return w, x, xq, weights, tags


def kernels():
    """{name: (Kernel, schedule, weights on B)} for every timed kernel."""
    _pkg()
    from mixfp4_sm120 import select as S
    from mixfp4_sm120.lib import Kernel
    from mixfp4_sm120.select import KernelSet
    table = S.TABLE_DIR / f"{S.gpu_slug()}.ko.json"
    dep, ext = deploy_dir(), extra_build_dir()
    out = {}
    for name, fam, want in (("stock_ko", "stock_ko", "stock_wA_e64"), ("ktile16", "mixed_ko", "n16k64_wA_e64_t0"),
                            ("ktile8", "mixed_wB_ko", "n8k64_wB_t0"), ("k256", "mixed256_ko", "n16k64_wA_g32_e64_t0")):
        ks = KernelSet(fam, build_root=dep, table=table)
        k = ks.pick(N_, K_, T_)
        if k.cfg.name != want:
            raise SystemExit(f"{fam} picks {k.cfg.name} at 4096^3, not {want}")
        out[name] = (k, tuple(ks.schedule(N_, K_, T_)), k.weight_operand == 1)
    sched = {u: out[f"ktile{u}"][1] for u in ("16", "8")}
    sched["256"] = out["k256"][1]
    for name, cfg, root, s in (("permma16", "n16k64_wA_e64_t0_permma", ext, sched["16"]),
                               ("permma8", "n8k64_wB_t0_permma", ext, sched["8"]),
                               ("ceil16", "n16k64_wA_nodisp_e64_t0", ext, sched["16"]),
                               ("ceil256", "n16k64_wA_nodisp_e64_t0", ext, sched["256"]),
                               ("ceil8", "n8k64_wB_nodisp_t0", ext, sched["8"]),
                               ("paper16", "n16k64_wA", dep, (0, 1)), ("paper8", "n8k64_wB", dep, (0, 1)),
                               ("nodisp16", "n16k64_wA_nodisp", ext, (0, 1)), ("nodisp16_t0", "n16k64_wA_nodisp_t0", ext, (0, 1)),
                               ("nodisp8", "n8k64_wB_nodisp", ext, (0, 1))):
        k = Kernel.load(cfg, build_root=root)
        out[name] = (k, tuple(s), k.weight_operand == 1)
    return out, str(table)


def call(kern, wts, xq, on_b, sched):
    wp, wsf, gsw = wts
    xp, xsf, gsx = xq
    if not on_b:
        return lambda wp=wp, wsf=wsf: kern.gemm(wp, wsf, xp, xsf, N_, T_, K_, scale_m_default=gsw, scale_n=gsx, check=False,
                                                schedule=sched)
    return lambda wp=wp, wsf=wsf: kern.gemm(xp, xsf, wp, wsf, T_, N_, K_, scale_m=gsx, scale_n_default=gsw, check=False,
                                            schedule=sched)


#: the timed configurations: name -> (kernel, weights, tag set's unit, tag set). Real tags in both sets of maps.
def configurations():
    cfg = {"stock_ko": ("stock_ko", "nvfp4", None, None), "fp8": ("fp8", None, None, None)}
    for t in ("e2m1", "real256", "real1k"):
        for u, n in (("16x64", "16"), ("8x64", "8")):
            cfg[f"permma{n}_{t}"] = (f"permma{n}", "map", u, t)
            cfg[f"ktile{n}_{t}"] = (f"ktile{n}", "map", u, t)
            cfg[f"paper{n}_{t}"] = (f"paper{n}", "map", u, t)
        cfg[f"k256_{t}"] = ("k256", "map", "256x64", t)
        if t != "e2m1":
            cfg[f"paper16x256_{t}"] = ("paper16", "map", "256x64", t)
    # every tile E0M3 (the text's all-E0M3 against all-E2M1, before and after; the 256x64 path's first kernel is paper16)
    for name, kk, u in (("ktile16_e0m3", "ktile16", "16x64"), ("ktile8_e0m3", "ktile8", "8x64"),
                        ("k256_e0m3", "k256", "256x64"), ("paper16_e0m3", "paper16", "16x64"),
                        ("paper8_e0m3", "paper8", "8x64")):
        cfg[name] = (kk, "map", u, "e0m3")
    for name in ("ceil16", "ceil256", "ceil8", "nodisp16", "nodisp16_t0", "nodisp8"):
        cfg[name] = (name, "four_over_six", None, None)
    return cfg


def _setup(args):
    import torch
    m256 = dict(zip(UNITS, args.maps256.split(",")))
    m1k = dict(zip(UNITS, args.maps1k.split(",")))
    w, x, xq, weights, tags = operands(m256, m1k)
    K, table = kernels()
    built = {}
    for name, (kk, kind, u, t) in configurations().items():
        if kk == "fp8":
            continue
        if kind == "map":
            mask, tb, share = tags[(u, t)]
            built[name] = (weights("map", mask, tb), share)
        else:
            built[name] = (weights(kind), 0.0)
    return torch, w, x, xq, weights, tags, K, table, built


def fp8_operands(w, x):
    import torch
    sx = (x.float().abs().amax() / 448.0).reshape(())
    sw = (w.float().abs().amax() / 448.0).reshape(())
    return (x.float() / sx).to(torch.float8_e4m3fn), (w.float() / sw).to(torch.float8_e4m3fn), sx.cuda().float(), sw.cuda().float()


# ------------------------------------------------------------------------------------------------------------- check
def check(args):
    torch, w, x, xq, weights, tags, K, table, built = _setup(args)
    res = dict(checks=[], tags={f"{u}/{t}": v[2] for (u, t), v in tags.items()})
    ok_all = True
    for n, u in (("16", "16x64"), ("8", "8x64")):
        for t in ("real256", "real1k", "e2m1", "e0m3", "rand30"):
            mask, tb, share = tags[(u, t)]
            wts = weights("map", mask, tb)
            ka, sa, ba = K[f"ktile{n}"]
            kb, sb, bb = K[f"permma{n}"]
            ya, yb = call(ka, wts, xq, ba, sa)(), call(kb, wts, xq, bb, sb)()
            torch.cuda.synchronize()
            ok = bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
            res["checks"].append(dict(unit=u, tags=t, e0m3_share=share, equal=ok, finite=bool(torch.isfinite(ya.float()).all())))
            print(f"{u} {t:8s} share {share:.4f}: per-MMA == per-K-tile {ok}", flush=True)
            ok_all &= ok
    res["passed"] = ok_all
    res["status"] = "complete"
    write(args.out, res)
    if not ok_all:                 # the driver's gate item9-check decides
        print("FAIL: a per-MMA build's output differs from its per-K-tile build's", flush=True)


# -------------------------------------------------------------------------------------------------------------- time
def timing(args):
    torch, w, x, xq, weights, tags, K, table, built = _setup(args)
    from torch.profiler import ProfilerActivity, profile
    sys.path.insert(0, str(ROOT))
    from evaluation.latency import Telemetry, bench, classify
    B, _ = bench()
    B.require_idle()
    tel = Telemetry()
    if tel.others():
        raise SystemExit(f"another compute process is on the GPU: {tel.others()}")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    if args.flush_mib * 2 ** 20 < 4 * l2:
        raise SystemExit("the flush is below 4x the L2")
    flush = torch.ones(args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device="cuda")
    cfgs = configurations()
    xf, wf, sx, sw = fp8_operands(w, x)
    fns, copies = {}, {}
    for name, (kk, kind, u, t) in cfgs.items():
        if kk == "fp8":
            n = math.ceil(4 * l2 / (wf.numel() * wf.element_size())) + 1
            copies[name] = [wf.clone() for _ in range(n)]
            fns[name] = (lambda c, xf=xf, sx=sx, sw=sw: torch._scaled_mm(xf, c.t(), scale_a=sx, scale_b=sw,
                                                                           out_dtype=torch.bfloat16))
            continue
        (wp, wsf, gsw), _ = built[name]
        n = math.ceil(4 * l2 / (wp.numel() * wp.element_size() + wsf.numel() * wsf.element_size())) + 1
        copies[name] = [(wp.clone(), wsf.clone()) for _ in range(n)]
        kern, sched, on_b = K[kk]
        f = call(kern, (wp, wsf, gsw), xq, on_b, sched)
        fns[name] = (lambda c, f=f: f(c[0], c[1]))
    res = dict(status="running", tool="experiments.paper_1k.ktile time", gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(),
               l2_bytes=l2, flush_mib=args.flush_mib, repeats=args.repeats, rounds=args.rounds, reps=args.reps,
               warmup=args.warmup, b2b=args.b2b, table=table,
               kernels={k: dict(config=v[0].cfg.name, sha256=v[0].sha256, defines=v[0].manifest.get("extra_defines"),
                                schedule=list(v[1]), weights_on_b=v[2]) for k, v in K.items()},
               configurations={k: dict(kernel=v[0], weights=v[1], unit=v[2], tags=v[3],
                                       e0m3_share=built[k][1] if k in built else None) for k, v in cfgs.items()},
               rotation_copies={k: len(v) for k, v in copies.items()}, repeats_data=[], checks={})
    # the registered check on the timed operands: per-MMA == per-K-tile, bit for bit
    for n in ("16", "8"):
        for t in ("e2m1", "real256", "real1k"):
            a = fns[f"ktile{n}_{t}"](copies[f"ktile{n}_{t}"][0])
            b = fns[f"permma{n}_{t}"](copies[f"permma{n}_{t}"][0])
            torch.cuda.synchronize()
            ok = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            res["checks"][f"permma{n}_{t} == ktile{n}_{t}"] = ok
            if not ok:
                write(args.out, res)
                raise SystemExit(f"permma{n}_{t} differs from ktile{n}_{t}")
    names = list(cfgs)
    counter = {nm: 0 for nm in names}

    def nxt(nm):
        c = copies[nm][counter[nm] % len(copies[nm])]
        counter[nm] += 1
        return c

    for rep in range(args.repeats):
        rec = dict(repeat=rep, modes={})
        for mode in ("iso_cold", "b2b_cold"):
            per = {nm: [] for nm in names}
            for r in range(args.rounds):
                shift = (r * len(names) // args.rounds) % len(names)
                for nm in names[shift:] + names[:shift]:
                    if tel.others():
                        res["checks"]["other_processes"] = tel.others()
                        write(args.out, res)
                        raise SystemExit(f"another compute process on the GPU: {tel.others()}")
                    fn = fns[nm]
                    if mode == "iso_cold":
                        for _ in range(args.warmup):
                            flush.sum()
                            torch.cuda.synchronize()
                            fn(nxt(nm))
                            torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            for _ in range(args.reps):
                                flush.sum()
                                torch.cuda.synchronize()
                                fn(nxt(nm))
                                torch.cuda.synchronize()
                        want = args.reps
                    else:
                        fn(nxt(nm))
                        torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            for _ in range(args.b2b):
                                fn(nxt(nm))
                            torch.cuda.synchronize()
                        want = args.b2b
                    us = [e.device_time_total for e in prof.events() if e.device_type.name == "CUDA"
                          and classify(e.name) == "gemm"]
                    if len(us) != want:
                        res["checks"]["counts"] = f"{mode} {nm} repeat {rep} round {r}: {len(us)} kernels, expected {want}"
                        write(args.out, res)
                        raise SystemExit(res["checks"]["counts"])
                    per[nm].append(statistics.median(us))
            rec["modes"][mode] = {nm: dict(us=statistics.median(v), rounds_us=v) for nm, v in per.items()}
        rec["gpu"] = tel.snap()
        res["repeats_data"].append(rec)
        write(args.out, res)
        print(f"repeat {rep}: stock_ko {rec['modes']['iso_cold']['stock_ko']['us']:.2f} us, ktile16 "
              f"{rec['modes']['iso_cold']['ktile16_real1k']['us']:.2f}, permma16 {rec['modes']['iso_cold']['permma16_real1k']['us']:.2f}",
              flush=True)
    res["gpu_end"] = B.gpu_info()
    res["status"] = "complete"
    write(args.out, res)


# ------------------------------------------------------------------------------------------------------------ analysis
def ratio(a, b):
    return 100 * (a / b - 1)


def parts(us: dict, tags: str) -> dict:
    """D.3's parts of one repeat (``us``: configuration -> µs, one mode), for the real tags ``tags``."""
    s = us["stock_ko"]
    after = {"8x64": ("ceil8", "ktile8_e2m1", f"ktile8_{tags}"), "16x64": ("ceil16", "ktile16_e2m1", f"ktile16_{tags}"),
             "256x64": ("ceil256", "k256_e2m1", f"k256_{tags}")}
    before = {"8x64": ("ceil8", "nodisp8", "paper8_e2m1", f"paper8_{tags}"),
              "16x64": ("nodisp16_t0", "nodisp16", "paper16_e2m1", f"paper16_{tags}"),
              "256x64": ("nodisp16_t0", "nodisp16", "paper16_e2m1", f"paper16x256_{tags}")}
    out = dict(after={}, before={})
    for u, (c, e, r) in after.items():
        out["after"][u] = dict(tiles=ratio(us[c], s), dispatch=ratio(us[e], us[c]), e0m3=ratio(us[r], us[e]),
                               total=ratio(us[r], s))
    for u, (t0, tg, e, r) in before.items():
        out["before"][u] = dict(tiles=ratio(us[t0], s), tag_copies=ratio(us[tg], us[t0]), dispatch=ratio(us[e], us[tg]),
                                e0m3=ratio(us[r], us[e]), total=ratio(us[r], s))
    return out


def d2(us: dict, tags: str) -> dict:
    """D.2's numbers of one repeat and mode: µs, TFLOP/s and the overhead over stock_ko."""
    s = us["stock_ko"]
    out = dict(stock=dict(us=s, tflops=FLOP / (s * 1e-6) / 1e12), fp8=dict(us=us["fp8"], vs_stock=ratio(us["fp8"], s)))
    for u, n in (("8x64", "8"), ("16x64", "16")):
        for kind in ("permma", "ktile"):
            for t in (tags, "e2m1"):
                v = us[f"{kind}{n}_{t}"]
                out.setdefault(u, {}).setdefault(kind, {})[t] = dict(us=v, tflops=FLOP / (v * 1e-6) / 1e12,
                                                                     vs_stock=ratio(v, s))
        out[u]["speedup"] = us[f"permma{n}_{tags}"] / us[f"ktile{n}_{tags}"]
    return out


def all_e0m3(us: dict) -> dict:
    """Every tile E0M3 against every tile E2M1, per unit, before and after (points)."""
    return dict(after={"8x64": ratio(us["ktile8_e0m3"], us["ktile8_e2m1"]), "16x64": ratio(us["ktile16_e0m3"], us["ktile16_e2m1"]),
                       "256x64": ratio(us["k256_e0m3"], us["k256_e2m1"])},
                before={"8x64": ratio(us["paper8_e0m3"], us["paper8_e2m1"]),
                        "16x64": ratio(us["paper16_e0m3"], us["paper16_e2m1"]),
                        "256x64": ratio(us["paper16_e0m3"], us["paper16_e2m1"])})


def mean_sd(vals):
    return [statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0]


def summarize(rec: dict, tags: str) -> dict:
    """Mean and SD over the repeats of D.2's and D.3's numbers, per mode."""
    out = {}
    for mode in ("iso_cold", "b2b_cold"):
        reps = [{k: v["us"] for k, v in r["modes"][mode].items()} for r in rec["repeats_data"]]
        ps = [parts(u, tags) for u in reps]
        ds = [d2(u, tags) for u in reps]
        es = [all_e0m3(u) for u in reps]

        def agg(get):
            return mean_sd([get(x) for x in ds])
        o = dict(parts={side: {u: {p: mean_sd([x[side][u][p] for x in ps]) for p in ps[0][side][u]} for u in ps[0][side]}
                        for side in ("before", "after")},
                 d2=dict(stock=dict(us=agg(lambda x: x["stock"]["us"]), tflops=agg(lambda x: x["stock"]["tflops"])),
                         fp8=dict(us=agg(lambda x: x["fp8"]["us"]), vs_stock=agg(lambda x: x["fp8"]["vs_stock"]))))
        for u in ("8x64", "16x64"):
            o["d2"][u] = {kind: {t: {k: agg(lambda x, u=u, kind=kind, t=t, k=k: x[u][kind][t][k]) for k in ("us", "tflops", "vs_stock")}
                                 for t in (tags, "e2m1")} for kind in ("permma", "ktile")}
            o["d2"][u]["speedup"] = agg(lambda x, u=u: x[u]["speedup"])
        o["all_e0m3"] = {side: {u: mean_sd([x[side][u] for x in es]) for u in UNITS} for side in ("before", "after")}
        out[mode] = o
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("build", "census", "check", "time"))
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--maps256", default=None, help="the release maps, 8x64,16x64,256x64")
    ap.add_argument("--maps1k", default=None, help="the 1k maps, 8x64,16x64,256x64")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--b2b", type=int, default=20)
    ap.add_argument("--flush-mib", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.what in ("check", "time") and not (a.maps256 and a.maps1k):
        ap.error("--maps256 and --maps1k are required")
    {"build": build, "census": census, "check": check, "time": timing}[a.what](a)


if __name__ == "__main__":
    main()
