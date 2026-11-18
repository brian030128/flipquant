"""Per-forward GEMM time of the sm120 deployment path (the FlipQuant paper's Table 15, Appendix D.5), timed on the
NativeLinears that flipquant's own install puts into the model.

    python -m evaluation.gemm_time --model phi4-14b \\
        --maps 8x64=maps/phi4-14b/flipquant_8x64.pt,16x64=maps/phi4-14b/flipquant_16x64.pt,256x64=maps/phi4-14b/flipquant_256x64.pt \\
        --out results/gemm_time/phi4-14b.json
    python -m evaluation.gemm_time_report results/gemm_time/*.json --out-dir results/gemm_time

**Modules.** Per policy the model is built by ``models.cli.build`` (``models.loader.load``, then ``models.sm120.install``)
with ``--mode native --native-backend sm120 --kernel-set K``; the timed modules are the NativeLinears the install put
into the model, i.e. every quantized Linear that runs the NVFP4 GEMM (``models.sm120.native_modules``). Policies:

    stock_ko      FourOverSix weights, --kernel-set auto            (weights on A; the reference)
    stock_wB_ko   FourOverSix weights, --kernel-set auto_stock_wB   (weights on B; recorded for reference)
    8x64          the 8x64 map,   --kernel-set auto -> mixed_wB_ko
    16x64         the 16x64 map,  --kernel-set auto -> mixed_ko
    256x64        the 256x64 map, --kernel-set auto -> mixed256_ko

Opt-in, for Appendix D.4 (the real tags against random placement): ``8x64_random``, ``16x64_random`` and
``256x64_random``, the same units and sets with a map whose E0M3 tiles are placed at random (``--maps UNIT_random=PATH``;
``experiments/paper_1k/random_maps.py`` keeps each module's E0M3 count, so each projection's typical and densest
modules are the real map's). The default policies are the five above.

Modules are grouped into projections by their last name component (and shape). Per projection a map is timed on two
modules: ``typical`` (sorted by (E0M3 granules, layer order), index (n - 1) // 2: the lower median) and ``densest``
(the first module with the most E0M3 granules); FourOverSix on the projection's first module. The per-forward GEMM
time is the sum over projections of (modules) x (the timed module's GEMM time).

**Timing** (the reference implementation's M1 / deviation 2, as its flipquant-parity worker ran it):
- per (projection, T) and configuration, one repetition is: a ``--flush-mib`` (512 MiB) read-flush, synchronize; a
  fresh BF16 activation from a seeded pool of two (seed n + k + T), synchronize; the activation quantizer of the build
  the set picks (``Kernel.quant_rows``), synchronize, not timed; then the GEMM with its fused epilogue
  (``Kernel.gemm_ptr`` with the set's width and scheduler row, as NativeLinear launches it) on the next of K copies of
  the module's packed weights and placed scales (K = ceil(4 x L2 / size) + 1), between CUDA events, synchronize;
- ``--rounds`` (3) rounds; in round r the configurations start at position r * len / rounds; per configuration
  ``--warmup`` (3) untimed repetitions, then ``--reps`` (30) under the CUDA profiler (CUPTI);
- the value is the median over all rounds of the GEMM's CUPTI device time; the quantizer's device time and the CUDA-event
  time are recorded too.

**Checks** (a failure stops the run): per (configuration, projection, T) the isolated launch equals the module's fused
forward (NativeLinear.forward, the real path) bit for bit; each install resolves to the expected set; every profiled
block holds exactly --reps GEMM and --reps quantizer launches; no other compute process on the GPU (NVML), and the GPU
idle at the start. Recorded: per (configuration, projection, T) the build, width and scheduler row, and whether the
width / scheduler row came from the tile table or its fallback (``fallback_width``, (0, 1)); NVML telemetry per block
and a 100 ms nvidia-smi sampler; map sha256s; the kernel sets (build sha256s, tables); the flipquant commit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

from evaluation.latency import Telemetry, bench, classify, stats
from models import sm120 as NS
from models.cli import add_model_args, build

ROOT = Path(__file__).resolve().parents[1]
TOKENS = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
ACT = "four_over_six_rows"
# policy -> (weights, map unit, --kernel-set, the set the route must resolve to)
POLICIES = OrderedDict([
    ("stock_ko", ("fourover6", None, "auto", "stock_ko")),
    ("stock_wB_ko", ("fourover6", None, "auto_stock_wB", "stock_wB_ko")),
    ("8x64", ("mixfp4", "8x64", "auto", "mixed_wB_ko")),
    ("16x64", ("mixfp4", "16x64", "auto", "mixed_ko")),
    ("256x64", ("mixfp4", "256x64", "auto", "mixed256_ko")),
])
DEFAULT_POLICIES = tuple(POLICIES)
# opt-in (Appendix D.4's real tags against random placement, paper_1k item 8): a map of the same unit whose E0M3 tiles
# are placed at random, given as --maps <unit>_random=PATH; timed beside the real map in the same process
POLICIES.update((f"{u}_random", (w, u, k, f)) for u, (w, _, k, f) in
                (("8x64", POLICIES["8x64"]), ("16x64", POLICIES["16x64"]), ("256x64", POLICIES["256x64"])))


def map_key(pol):
    """The --maps key of a map policy: its unit, or the policy's own name for a random-placement map."""
    return pol if pol.endswith("_random") else POLICIES[pol][1]
TAGS = ("typical", "densest")


def configs(policies):
    """Timed configurations in their rotation order: (config, policy, module choice)."""
    out = []
    for pol in policies:
        if POLICIES[pol][0] == "fourover6":
            out.append((pol, pol, "first"))
        else:
            out += [(f"{pol}_{v}", pol, v) for v in TAGS]
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tsha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def write(path, obj):
    """Like kernels/sm120/bench/common.write, with str() for anything JSON does not know."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def git_state():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return dict(head=head, clean=not dirty.strip(), status=dirty.strip().splitlines()[:20])


def group(layers):
    """{projection: [(name, NativeLinear), ...]} in layer order; a projection is a last name component, split by shape
    if one name holds several shapes."""
    by = OrderedDict()
    for name, lin in layers.items():
        by.setdefault(name.rsplit(".", 1)[-1], []).append((name, lin))
    out = OrderedDict()
    for leaf, ms in by.items():
        shapes = OrderedDict.fromkeys((lin.out_features, lin.in_features) for _, lin in ms)
        for s in shapes:
            key = leaf if len(shapes) == 1 else f"{leaf}[{s[0]}x{s[1]}]"
            out[key] = [(n, lin) for n, lin in ms if (lin.out_features, lin.in_features) == s]
    return out


def choose(ms):
    """(first, typical, densest) of one projection's modules: (name, NativeLinear, E0M3 granules)."""
    e = [int(lin.e0m3_tiles) for _, lin in ms]
    order = sorted(range(len(ms)), key=lambda i: (e[i], i))
    ti = order[(len(ms) - 1) // 2]
    di = max(range(len(ms)), key=lambda i: (e[i], -i))           # the first maximum
    return {v: (ms[i][0], ms[i][1], e[i]) for v, i in (("first", 0), ("typical", ti), ("densest", di))}


def table_source(kset, n, k, t):
    """Where the call's width and scheduler row come from: the tile table or the fallback rule."""
    S = NS.package().select
    row = kset.table.get((n, k))
    w = row.get(S.bucket(t)) if row else None
    srow = kset.schedules.get((n, k))
    sched = srow.get(S.bucket(t)) if srow else None
    return dict(width_from_table=w is not None and w in kset.kernels, shape_in_table=row is not None,
                schedule_from_table=sched is not None, shape_in_schedule=srow is not None,
                set_has_schedule_rows=bool(kset.schedules))


def build_policy(args, pol):
    weight, unit, kset, family = POLICIES[pol]
    argv = ["--model", args.model, "--mode", "native", "--native-backend", "sm120", "--kernel-set", kset,
            "--weight", weight, "--act", "fourover6", "--act-scope", "row"]
    if args.revision:
        argv += ["--revision", args.revision]
    if args.kernel_build_dir:
        argv += ["--kernel-build-dir", str(args.kernel_build_dir)]
    if weight == "mixfp4":
        argv += ["--map", str(args.maps[map_key(pol)]), "--unit", unit]
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    a = ap.parse_args(argv)
    t0 = time.time()
    model, tok, spec, layers, info = build(a)
    rep = info["native"]
    if rep["kernel"] != family:
        raise SystemExit(f"{pol}: --kernel-set {kset} resolved to {rep['kernel']!r}, expected {family} ({rep['routing']})")
    sets = {id(m.kernel) for m in layers.values()}
    if len(sets) != 1 or len(layers) != rep["modules"]:
        raise SystemExit(f"{pol}: {len(sets)} kernel objects over {len(layers)} NativeLinears ({rep['modules']} installed)")
    bad = [n for n, m in layers.items() if m.act_kind != ACT]
    if bad:
        raise SystemExit(f"{pol}: {len(bad)} modules without FourOverSix row activations, e.g. {bad[:3]}")
    other = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n not in layers]
    out = dict(argv=argv, seconds=round(time.time() - t0, 1), source=getattr(model, "flipquant_source", None),
               spec=dict(key=spec.key, hf_id=spec.hf_id, revision=spec.revision),
               install={k: rep.get(k) for k in ("kernel", "routing", "build_dir", "kernel_set", "modules", "e0m3_tiles",
                                                "unit", "activation_quantizer", "device_bytes")},
               map=info.get("map"), linear_not_native=other)
    return model, layers, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="flipquant registry key")
    ap.add_argument("--revision", default=None, help="override the registry's pinned revision")
    ap.add_argument("--maps", required=True, help="UNIT=PATH,... for 8x64, 16x64 and 256x64 (and UNIT_random=PATH for "
                                                  "the opt-in random-placement policies)")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--kernel-build-dir", type=Path, default=None)
    ap.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    ap.add_argument("--projections", default=None)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--flush-mib", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.maps = dict(s.split("=", 1) for s in args.maps.split(",")) if args.maps else {}
    args.policies = [p for p in args.policies.split(",") if p]
    for p in args.policies:
        if p not in POLICIES:
            raise SystemExit(f"unknown policy {p!r}")
        if POLICIES[p][0] == "mixfp4" and map_key(p) not in args.maps:
            raise SystemExit(f"policy {p} needs --maps {map_key(p)}=PATH")
    B, _ = bench()
    B.require_idle()
    csv_path = Path(str(args.out) + ".telemetry.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(csv_path)
    try:
        run(args, B, tel)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res["sampler"] = Telemetry.summarize(csv_path)
    write(args.out, res)


def run(args, B, tel):
    from torch.profiler import ProfilerActivity, profile
    torch.backends.cuda.matmul.allow_tf32 = False
    R = NS.package()
    cfgs = configs(args.policies)
    res = dict(status="running", tool="evaluation.gemm_time", argv=sys.argv, flipquant=git_state(), gpu=B.gpu_info(),
               power_limit_w=tel.power_limit_w(), l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
               python=sys.executable, torch=torch.__version__, package=str(Path(R.__file__).resolve().parent),
               model=args.model, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               maps={u: dict(path=str(p), sha256=sha256_file(p)) for u, p in args.maps.items()},
               protocol=dict(method="isolated launches, cold weights (M1 / deviation 2): K weight copies > 4x L2, a "
                                    "read-flush before each launch, the activation quantizer synchronized and untimed",
                             reps=args.reps, warmup=args.warmup, rounds=args.rounds, flush_mib=args.flush_mib,
                             order="rotated: round r starts at position r * len / rounds",
                             value="median over all rounds of the CUPTI device time of the isolated GEMM launch"),
               configs={c: dict(policy=p, module=v, weights=POLICIES[p][0], unit=POLICIES[p][1],
                                kernel_set_arg=POLICIES[p][2], set=POLICIES[p][3]) for c, p, v in cfgs},
               policies={}, projections=OrderedDict(), rows=[], blocks=[],
               checks=dict(bitwise=[], counts_ok=True, other_processes=[]))
    if args.flush_mib * 2 ** 20 < 4 * res["l2_bytes"]:
        raise SystemExit(f"--flush-mib {args.flush_mib} is below 4x the L2 ({res['l2_bytes']} bytes)")
    if not res["flipquant"]["clean"]:
        print(f"warning: the flipquant tree is not clean: {res['flipquant']['status']}", file=sys.stderr)
    # the timed modules: per policy, install into a freshly loaded model and keep only the chosen NativeLinears
    lins, lin_names, groups_ref = {}, {}, None
    for pol in args.policies:
        model, layers, info = build_policy(args, pol)
        groups = group(layers)
        sig = OrderedDict((p, (len(ms), [ms[0][1].out_features, ms[0][1].in_features])) for p, ms in groups.items())
        if groups_ref is None:
            groups_ref = sig
            res["native_modules"] = sorted(layers)
        elif sig != groups_ref or sorted(layers) != res["native_modules"]:
            raise SystemExit(f"{pol}: a different set of NativeLinears than {args.policies[0]}")
        res["policies"][pol] = info
        for proj, ms in groups.items():
            if args.projections and proj not in args.projections.split(","):
                continue
            ch = choose(ms)
            p = res["projections"].setdefault(proj, dict(shape=[ms[0][1].out_features, ms[0][1].in_features],
                                                         modules=len(ms), choices={}))
            p["choices"][pol] = {v: dict(module=ch[v][0], e0m3_granules=ch[v][2]) for v in ("first", "typical", "densest")}
            p["choices"][pol]["e0m3_granules_sorted"] = sorted(int(lin.e0m3_tiles) for _, lin in ms)
            for c, cp, v in cfgs:
                if cp == pol:
                    lin = ch[v][1]
                    lin.share_input = False
                    lins[(c, proj)] = lin
                    lin_names[(c, proj)] = ch[v][0]
        print(f"{args.model} {pol}: {info['install']['modules']} NativeLinears, set {info['install']['kernel']}, "
              f"{info['seconds']} s", flush=True)
        del model, layers, groups
        gc.collect()
        torch.cuda.empty_cache()
    res["policies_linear_not_native"] = {p: v.pop("linear_not_native") for p, v in res["policies"].items()}
    write(args.out, res)
    tokens = [int(t) for t in str(args.tokens).split(",")]
    flush = torch.ones(args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    names = [c for c, _, _ in cfgs]
    for proj, p in res["projections"].items():
        n, k = p["shape"]
        copies, counter = {}, {}
        for c in names:
            lin = lins[(c, proj)]
            size = lin.packed.numel() + lin.sf.numel()
            copies[c] = [(lin.packed.clone(), lin.sf.clone()) for _ in range(math.ceil(4 * res["l2_bytes"] / size) + 1)]
            counter[c] = 0
        p["rotation_copies"] = {c: len(copies[c]) for c in names}
        p["operands"] = {c: dict(module=lin_names[(c, proj)], packed_sha256=tsha(lins[(c, proj)].packed),
                                 sf_sha256=tsha(lins[(c, proj)].sf), global_scale=repr(lins[(c, proj)].global_scale),
                                 e0m3_granules=int(lins[(c, proj)].e0m3_tiles)) for c in names}
        for t in tokens:
            g = torch.Generator("cpu").manual_seed(n + k + t)
            pool = [torch.randn(t, k, generator=g).to("cuda", torch.bfloat16) for _ in range(2)]
            x = torch.empty_like(pool[0])
            fns, meta = {}, {}
            for c in names:
                lin = lins[(c, proj)]
                kset = lin.kernel_set
                kern = kset.pick(n, k, t) if kset is not None else lin.kernel
                sched = kset.schedule(n, k, t) if kset is not None else (0, 1)
                width = kset.width(n, k, t) if kset is not None else None
                y = torch.empty((t, n), dtype=torch.bfloat16, device="cuda")
                bptr = None if lin.bias_bf16 is None else lin.bias_bf16.data_ptr()

                def quant(kern=kern, act=lin.act_kind):
                    return kern.quant_rows(x, act)

                def gemm(q, kern=kern, lin=lin, y=y, bptr=bptr, c=c, sched=sched):
                    xp, xsf, gs = q
                    wp, wsf = copies[c][counter[c] % len(copies[c])]
                    counter[c] += 1
                    if lin.weights_on_a:
                        kern.gemm_ptr(wp.data_ptr(), wsf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k, None,
                                      lin.global_scale, gs.data_ptr(), 1.0, bptr, y, stream.cuda_stream, schedule=sched)
                    else:
                        kern.gemm_ptr(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), t, n, k,
                                      gs.data_ptr(), 1.0, None, lin.global_scale, bptr, y, stream.cuda_stream,
                                      schedule=sched)
                    return y
                fns[c] = (quant, gemm)
                # registered check: the isolated launch equals the fused forward (the real path) bit for bit
                x.copy_(pool[0])
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    ref = lin(x)
                    torch.cuda.synchronize()
                fwd = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA" and classify(e.name) == "gemm"})
                ref = ref.clone()
                got = gemm(quant()).clone()
                torch.cuda.synchronize()
                ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
                res["checks"]["bitwise"].append(dict(proj=proj, tokens=t, config=c, equal=ok))
                if not ok:
                    res["status"] = "failed"
                    write(args.out, res)
                    raise SystemExit(f"{proj} T={t} {c}: the isolated GEMM differs from NativeLinear.forward")
                m = kern.manifest
                meta[c] = dict(kernel=kern.cfg.name, kernel_sha256=kern.sha256, sass_sha256=m.get("sass_sha256"),
                               extra_defines=m.get("extra_defines") or {}, width=width, schedule=list(sched),
                               set=kset.family if kset is not None else None,
                               table=kset.table_source if kset is not None else None,
                               forward_gemm_names=fwd,
                               **(table_source(kset, n, k, t) if kset is not None else {}))

            def rep(c, i, timed):
                quant, gemm = fns[c]
                flush.sum()
                torch.cuda.synchronize()
                x.copy_(pool[i % 2])
                torch.cuda.synchronize()
                q = quant()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                gemm(q)
                b.record()
                torch.cuda.synchronize()
                return a.elapsed_time(b) * 1e3 if timed else None

            per = {c: [] for c in names}
            block_before = tel.snap()
            for r in range(args.rounds):
                shift = (r * len(names) // args.rounds) % len(names)
                for pos, c in enumerate(names[shift:] + names[:shift]):
                    others = tel.others()
                    if others:
                        res["checks"]["other_processes"].append(dict(proj=proj, tokens=t, config=c, round=r, pids=others))
                        res["status"] = "failed"
                        write(args.out, res)
                        raise SystemExit(f"another compute process on the GPU: {others}")
                    for i in range(args.warmup):
                        rep(c, i, False)
                    before = tel.snap()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        ev = [rep(c, i, True) for i in range(args.reps)]
                    after = tel.snap()
                    kt, kn = {"gemm": [], "quant": []}, {"gemm": set(), "quant": set()}
                    for e in prof.events():
                        if e.device_type.name == "CUDA":
                            cls = classify(e.name)
                            if cls in kt:
                                kt[cls].append(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
                                kn[cls].add(e.name)
                    if len(kt["gemm"]) != args.reps or len(kt["quant"]) != args.reps:
                        res["checks"]["counts_ok"] = False
                        res["status"] = "failed"
                        write(args.out, res)
                        raise SystemExit(f"{proj} T={t} {c} round {r}: {len(kt['gemm'])} GEMM / {len(kt['quant'])} "
                                         f"quantizer launches profiled, expected {args.reps}")
                    per[c].append(dict(round=r, position=pos, gemm_us=kt["gemm"], quant_us=kt["quant"], event_us=ev,
                                       gemm_names=sorted(kn["gemm"]), telemetry=Telemetry.block(before, after)))
            res["blocks"].append(dict(proj=proj, tokens=t, **Telemetry.block(block_before, tel.snap())))
            for c in names:
                g_all = [v for blk in per[c] for v in blk["gemm_us"]]
                q_all = [v for blk in per[c] for v in blk["quant_us"]]
                e_all = [v for blk in per[c] for v in blk["event_us"]]
                res["rows"].append(dict(
                    proj=proj, out=n, inp=k, tokens=t, config=c, policy=res["configs"][c]["policy"],
                    module=lin_names[(c, proj)], **meta[c],
                    gemm_names=sorted({nm for blk in per[c] for nm in blk["gemm_names"]}),
                    gemm_us=statistics.median(g_all), gemm=stats(g_all), quant_us=statistics.median(q_all),
                    quant=stats(q_all), event_us=statistics.median(e_all), event=stats(e_all),
                    rounds=[dict(round=b["round"], position=b["position"], gemm_us=statistics.median(b["gemm_us"]),
                                 quant_us=statistics.median(b["quant_us"]), event_us=statistics.median(b["event_us"]),
                                 telemetry=b["telemetry"]) for b in per[c]]))
            print(f"{args.model} {proj} T={t} " + " ".join(f"{r['config']}={r['gemm_us']:.1f}" for r in res["rows"]
                                                            if r["proj"] == proj and r["tokens"] == t), flush=True)
            del pool, x, fns
            write(args.out, res)
        del copies
        torch.cuda.empty_cache()
    res["per_forward_gemm_us"] = per_forward(res)
    res["status"] = "complete"
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["flipquant_after"] = git_state()
    write(args.out, res)


def per_forward(res):
    """{config@T: us}: the sum over projections of (modules) x (the timed module's GEMM time)."""
    by = {(r["config"], r["proj"], r["tokens"]): r["gemm_us"] for r in res["rows"]}
    out = {}
    for t in sorted({r["tokens"] for r in res["rows"]}):
        for c in res["configs"]:
            if all((c, p, t) in by for p in res["projections"]):
                out[f"{c}@{t}"] = sum(v["modules"] * by[(c, p, t)] for p, v in res["projections"].items())
    return out


if __name__ == "__main__":
    main()
