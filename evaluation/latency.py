"""The FlipQuant paper's latency benchmarks on the sm120 deployment path (``models/sm120.py``), on ``models.cli``. One
process measures one model under one policy. There are four subcommands.

    python -m evaluation.latency prefill --model llama3.1-8b --mode native \\
        --weight mixfp4 --map maps/.../flipquant_8x64.pt --shapes 1x128,1x2048 --out results/lat/prefill_ours_8x64.json
    python -m evaluation.latency decode  --model llama3.1-8b --mode native --weight fourover6 \\
        --settings 1x512,4x512 --gen 64 --out results/lat/decode_fo6.json
    python -m evaluation.latency gemm    --model llama3.1-8b --maps 8x64=...pt,16x64=...pt,256x64=...pt \\
        --tokens 128,2048 --out results/lat/gemm_isolated.json
    python -m evaluation.latency gemm-warm --model llama3.1-8b --maps 8x64=...pt --tokens 128,2048 --out gemm_warm.json

All of them refuse to start on a busy GPU and run with the default caching allocator, as the paper's did. A policy is
``--mode bf16`` or ``--mode native`` (the sm120 path, on an sm_120 GPU) with ``--weight`` / ``--map`` / ``--kernel-set``.
``--kernel-set auto`` (the default) runs the optimized kernel sets; the paper's kernels are ``--kernel-set paper``.
``gemm`` and ``gemm-warm`` take ``--kernel-set {auto,paper}`` and ``--kernel-build-dir`` too, routed as the install
routes a policy.

**prefill** (the paper's ``bench_prefill.py``, step 05): end-to-end prefill.
- **Eager:** one forward over ``[batch, prompt]`` random tokens (seed 0) that writes the KV cache and returns the
  logits (``kernels/sm120/bench/model.py`` prefill). After 2 warm-ups, ``--reps`` forwards are timed with CUDA events:
  median, min and max. The host's enqueue time is recorded too.
- **CUDA graph:** the same forward, captured once and replayed: 2 warm-ups, then ``--reps`` timed replays.
  - **Registered check:** the graph's logits equal the eager forward's bit for bit.
  - During capture only, ``transformers.masking_utils.is_tracing`` ignores stream capture, so the mask decision is the
    eager one: flash SDPA with ``is_causal=True``. A failed capture is recorded, and no later shape is captured.
- **Host-bound flag:** set when the graph is more than 5 % faster than eager.
- **Also recorded:** coverage (every quantized Linear ran natively), and the reuse of the quantized activation per
  projection. q/k/v and gate/up read one input; ``gemm --quant-reuse`` uses this.
- **``--act-override nvfp4_rows``** is latency only: it switches the activation quantizer of a FourOverSix-activation
  policy. The maps were calibrated with FourOverSix activations, so the result is not an evaluated configuration.

**decode** (the paper's ``bench_decode.py``, Experiment D): StaticCache decode in a CUDA graph.
- **Timing:** a ``batch x prompt`` random prompt fills a StaticCache, and the single-token forward is captured in a CUDA
  graph (``kernels/sm120/bench/model.py`` decode_graph). ``--gen`` replays are timed with the host clock, giving ms per
  token and tokens per second.
- **Registered check:** the graph's greedy tokens equal an eager StaticCache greedy decode of the same prompt, for the
  first 33 tokens.
- **Recorded, not a check:** the same comparison against an eager DynamicCache decode.
- **Widths:** the CTA tile width each quantized Linear gets at ``T = batch``.

**gemm** (the paper's ``bench_gemm_isolated.py`` and ``gemm_isolated_tables.per_forward``, its deviation 2): GEMM and
activation-quantizer kernel times per text-Linear shape, isolated launches, cold weights.
- **One repetition, on an idle GPU:**
  1. a ``--flush-mib`` read-flush (4x the L2), with every call using the next of K distinct weight copies
     (K x size >= 4x the L2);
  2. a fresh BF16 activation;
  3. TIMED: the activation quantizer;
  4. TIMED: the GEMM with its fused epilogue, with the kernel set's width and scheduler row for the call, as
     NativeLinear launches it.
- **Measurement:**
  - CUPTI device time per launch;
  - ``--rounds`` rounds in a rotated configuration order, ``--reps`` repetitions each, after ``--warmup``;
  - NVML telemetry per block, and a 100 ms nvidia-smi sampler over the run.
- **Configurations** (the paper's names): FourOverSix and NVFP4 weights on the baseline set (``stock_wA``,
  ``stock_wA_nvfp4``), FourOverSix with the weights on B (``stock_wB``), and the maps (``mixed_16x64_*``,
  ``mixed_256x64_*``, ``n8k64_wB_*``). Each runs on the set ``--kernel-set`` routes it to (``auto``: stock_ko,
  stock_wB_ko, mixed_ko, mixed256_ko, mixed_wB_ko; ``paper``: stock, stock_wB, mixed, mixed, n8k64_wB); every row
  records the set, the build, the width and the scheduler row. The maps are timed with two modules' tags per
  projection: ``typical`` (the lower-median E0M3 share) and ``worst`` (the densest module).
- **Registered checks:**
  - every configuration's isolated output equals NativeLinear's fused forward bit for bit;
  - every profiled block holds exactly ``--reps`` launches of each kernel;
  - no other compute process runs on the GPU.
- **Per-forward sums:** each projection's GEMM time times its module count. The quantizer's launches are counted net of
  the reuse that ``prefill`` measured (``--quant-reuse PREFILL.json``).

**gemm-warm** (the paper's ``bench_gemm.py``, step 06; the paper keeps it as the alternative method): one NativeLinear
forward per call, back to back on L2-resident weights.
- **Timing:** CUPTI per kernel, the median of ``--iters`` calls after 5 warm-ups, in ``--rounds`` rotated rounds.
- **Weights:** per projection, the densest module of each map (the most E0M3 granules); the first module for
  FourOverSix and NVFP4.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib.util
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

from flipquant import maps as M
from models import sm120 as NS
from models.cli import add_model_args, apply_quant, native_backend
from models.loader import load, quantizable_linears

BENCH = NS.KERNELS / "bench"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench():
    """The vendored bench/common.py and model.py (kernels/sm120/bench)."""
    NS.package()
    return _module("sm120_bench_common", BENCH / "common.py"), _module("sm120_bench_model", BENCH / "model.py")


def load_policy(args):
    """The model, quantized per ``args`` (bf16, or the sm120 path). Returns (model, spec, layers, info, seconds)."""
    if not (args.mode == "bf16" or (args.mode == "native" and native_backend(args) in ("sm120", "sm100-tuned"))):
        raise SystemExit("the latency benchmarks run --mode bf16 or --mode native on the sm120 path (an sm_120 GPU) or "
                         "--native-backend sm100-tuned (an sm_100 GPU)")
    t0 = time.time()
    model, _, spec = load(args.model, revision=args.revision, device_map=args.device_map, attn=args.attn)
    seconds = dict(load=time.time() - t0)
    t0 = time.time()
    layers, info = apply_quant(model, spec, args)
    seconds["install"] = time.time() - t0
    return model, spec, layers, info, seconds


def _backend_module(model):
    """models.sm100 when the model runs the tuned SM100 kernels, else models.sm120."""
    from models import sm100
    return sm100 if sm100.native_modules(model) else NS


def coverage(model, layers):
    cov = _backend_module(model).coverage(model)
    assert cov["native_called"] == cov["native"] == len(layers), cov        # every quantized Linear ran natively
    return cov


# --- prefill (bench_prefill.py) --------------------------------------------------------------------------------------

def tokens(batch, prompt):
    # the same tokens as sm120/bench/model.py prefill (seed 0), so the graph's logits compare with eager ones
    return torch.randint(100, 20000, (batch, prompt), device="cuda", generator=torch.Generator("cuda").manual_seed(0))


@torch.no_grad()
def enqueue_ms(model, batch, prompt, reps):
    ids = tokens(batch, prompt)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=True)
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    return statistics.median(ts)


@contextlib.contextmanager
def eager_mask_decision():
    """During capture, transformers.masking_utils.is_tracing without its CUDA-stream-capture condition."""
    import transformers.masking_utils as MU
    from transformers.utils import import_utils as IU
    original = MU.is_tracing

    def is_tracing(tensor=None):
        t = IU.is_torchdynamo_compiling() or IU.is_jit_tracing()
        if tensor is not None:
            t = t or IU.is_torch_fx_proxy(tensor) or IU.is_fake_tensor(tensor) or IU.is_jax_jitting(tensor)
        return t
    MU.is_tracing = is_tracing
    try:
        yield
    finally:
        MU.is_tracing = original


def _logit_diff(a, b):
    d = (a.float() - b.float()).abs()
    return dict(equal=bool(torch.equal(a, b)), max_abs=float(d.max()), mean_abs=float(d.mean()),
                argmax_agree=float((a.argmax(-1) == b.argmax(-1)).float().mean()), finite=bool(torch.isfinite(a).all()))


@torch.no_grad()
def prefill_graph(model, batch, prompt, reps, check="bitwise"):
    """``check``: bitwise = the graph's logits must equal eager's; report = the model's eager forward is not bitwise
    reproducible itself (e.g. flash-linear-attention's kernels), so a second eager
    forward is compared too and both differences are recorded instead (the run fails only on non-finite logits)."""
    ids = tokens(batch, prompt)
    ref = model(input_ids=ids, use_cache=True).logits.clone()
    reference = None
    if check == "report":
        reference = dict(eager_vs_eager=_logit_diff(model(input_ids=ids, use_cache=True).logits, ref))
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            model(input_ids=ids, use_cache=True)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with eager_mask_decision(), torch.cuda.graph(g):
        logits = model(input_ids=ids, use_cache=True).logits
    g.replay()
    torch.cuda.synchronize()
    equal = bool(torch.equal(logits, ref))
    diff = 0.0 if equal else float((logits.float() - ref.float()).abs().max())
    if reference is not None:
        reference["graph_vs_eager"] = _logit_diff(logits, ref)
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    del g, logits, ref
    NS.package().linear.clear_quant_cache()        # it may hold the capture's scratch
    from models import sm100
    sm100.clear_quant_cache()
    torch.cuda.empty_cache()
    out = dict(ms=ts[len(ts) // 2], min_ms=ts[0], max_ms=ts[-1], tokens_per_s=batch * prompt / (ts[len(ts) // 2] / 1e3),
               logits_equal_eager=equal, max_abs_diff_eager=diff)
    if reference is not None:
        out["nondeterministic_reference"] = reference
    return out


def run_prefill(args):
    B, BM = bench()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    res = dict(status="running", bench="prefill", gpu=B.gpu_info(), model=args.model, label=args.label, round=args.round,
               reps=args.reps, pid=os.getpid(), allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
               eager={}, graph={}, host_bound={},
               graph_capture_note="during capture, transformers.masking_utils.is_tracing ignores stream capture (the "
                                  "eager mask decision: flash SDPA with is_causal=True); logits compared bitwise with eager")
    model, spec, layers, info, seconds = load_policy(args)
    res.update(policy=info, seconds=seconds)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res["weights_gib"] = torch.cuda.memory_allocated() / 2 ** 30
    shapes = args.shapes.split(",")
    for s in shapes:
        b, p = (int(v) for v in s.split("x"))
        try:
            res["eager"][s] = BM.prefill(model, b, p, reps=args.reps)
            res["eager"][s]["enqueue_ms"] = enqueue_ms(model, b, p, min(3, args.reps))
        except torch.OutOfMemoryError as e:
            res["eager"][s] = dict(error=f"OOM: {str(e)[:160]}")
            torch.cuda.empty_cache()
        print(args.label, s, "eager", json.dumps(res["eager"][s]), flush=True)
    res["peak_eager_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    capture_error = None
    for s in shapes if not args.no_graph else ():
        if "error" in res["eager"][s]:
            continue
        if capture_error is not None:
            res["graph"][s] = dict(error="not attempted: an earlier capture in this process failed")
            continue
        b, p = (int(v) for v in s.split("x"))
        try:
            res["graph"][s] = prefill_graph(model, b, p, args.reps, check=args.graph_check)
        except Exception as e:  # noqa: BLE001  (recorded; the eager numbers stand)
            capture_error = f"{type(e).__name__}: {str(e)[:400]}"
            res["graph"][s] = dict(error=capture_error)
        print(args.label, s, "graph", json.dumps(res["graph"][s]), flush=True)
    for s in shapes:
        e, g = res["eager"][s], res["graph"].get(s, {})
        if "error" in e:
            continue
        if "ms" in g:
            res["host_bound"][s] = dict(flag=e["ms"] > 1.05 * g["ms"], rule="eager > 1.05 x graph",
                                        graph_saves=1 - g["ms"] / e["ms"])
        else:
            res["host_bound"][s] = dict(flag=e["enqueue_ms"] >= 0.9 * e["ms"], rule="enqueue >= 0.9 x eager (no graph)",
                                        enqueue_over_eager=e["enqueue_ms"] / e["ms"])
    res["capture_error"] = capture_error
    res["graph_equals_eager"] = all(g.get("logits_equal_eager") for g in res["graph"].values()) if res["graph"] else None
    if layers:
        res["coverage"] = coverage(model, layers)
        first = next(iter(layers.values()))
        if first.kernel_set is not None:
            res["kernel_set"] = first.kernel_set.describe()
        reuse = {}           # the quantized activation's reuse per projection (q/k/v and gate/up read one input)
        for n, m in layers.items():
            r = reuse.setdefault(n.rsplit(".", 1)[-1], dict(modules=0, calls=0, quant_reused=0))
            r["modules"] += 1
            r["calls"] += m.calls
            r["quant_reused"] += m.quant_reused
        res["quant_reuse"] = reuse
    res["peak_total_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    res["gpu_end"] = B.gpu_info()
    res["status"] = "complete"
    res["graph_check"] = args.graph_check
    B.write(args.out, res)
    if args.graph_check == "report":
        if not all(g["nondeterministic_reference"]["graph_vs_eager"]["finite"] for g in res["graph"].values() if "ms" in g):
            raise SystemExit("REGISTERED CHECK FAILED: a CUDA graph's logits are not finite")
    elif res["graph_equals_eager"] is False:
        raise SystemExit("REGISTERED CHECK FAILED: a CUDA graph's logits differ from eager")


# --- decode (bench_decode.py) ----------------------------------------------------------------------------------------

@torch.no_grad()
def dynamic_tokens(model, batch, prompt, n):
    """Greedy tokens [first, ...] of an eager DynamicCache decode of decode_graph's prompt (seed 1)."""
    from transformers import DynamicCache
    ids = torch.randint(100, 20000, (batch, prompt), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    toks = [out.logits[:, -1:].argmax(-1)]
    for _ in range(n - 1):
        out = model(input_ids=toks[-1], past_key_values=cache, use_cache=True)
        toks.append(out.logits[:, -1:].argmax(-1))
    return torch.cat(toks, 1)


def run_decode(args):
    B, BM = bench()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    from mixfp4_sm120.select import key_order
    res = dict(status="running", bench="decode", gpu=B.gpu_info(), model=args.model, label=args.label, round=args.round,
               gen=args.gen, pid=os.getpid(), allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"), decode={})
    model, spec, layers, info, seconds = load_policy(args)
    res.update(policy=info, seconds=seconds)
    torch.cuda.synchronize()
    res["weights_gib"] = torch.cuda.memory_allocated() / 2 ** 30
    for s in args.settings.split(","):
        b, p = (int(v) for v in s.split("x"))
        torch.cuda.reset_peak_memory_stats()
        entry = res["decode"][s] = {}
        try:
            timing, graph_tokens = BM.decode_graph(model, b, p, args.gen)
            entry.update(timing)
            _, ref = BM.decode_eager_tokens(model, b, p, graph_tokens.shape[1])
            n = min(ref.shape[1], graph_tokens.shape[1])
            same = ref[:, :n] == graph_tokens[:, :n]
            # the registered check: the graph's greedy tokens equal an eager StaticCache decode
            entry.update(tokens_compared=n, token_match=float(same.double().mean()), tokens_equal_static_eager=bool(same.all()))
            dyn = dynamic_tokens(model, b, p, n)            # recorded, not a check
            entry["token_match_dynamic_eager"] = float((dyn[:, :n] == graph_tokens[:, :n]).double().mean())
        except Exception as e:  # noqa: BLE001  (recorded; a failed setting fails the check below)
            entry["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        entry["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        if layers:
            ks = next(iter(layers.values())).kernel_set
            entry["decode_widths"] = (sorted({ks.width(m.out_features, m.in_features, b) for m in layers.values()},
                                             key=key_order) if ks is not None else next(iter(layers.values())).kernel.cfg.name)
        torch.cuda.empty_cache()
        print(args.label, s, json.dumps(entry), flush=True)
    if layers:
        res["coverage"] = coverage(model, layers)
    res["check_passed"] = all(e.get("tokens_equal_static_eager") for e in res["decode"].values())
    res["gpu_end"] = B.gpu_info()
    res["status"] = "complete"
    B.write(args.out, res)
    if not res["check_passed"]:
        raise SystemExit("REGISTERED CHECK FAILED: graph decode tokens differ from eager (or a setting failed)")


# --- gemm (bench_gemm_isolated.py, gemm_isolated_tables.per_forward) ---------------------------------------------------

# configuration -> (weights: a map unit or a baseline, tag variant, kernel, activation quantizer)
# configuration (the paper's names) -> (weights, tags, kernel key, activation quantizer); the kernel key's set comes
# from kernel_sets(), as models.sm120.install routes a policy
CONFIGS = OrderedDict(
    stock_wA=("fourover6", "first", "baseline", "four_over_six_rows"),
    stock_wA_nvfp4=("nvfp4", "first", "baseline", "nvfp4_rows"),
    stock_wB=("fourover6", "first", "baseline_wB", "four_over_six_rows"),
    mixed_16x64_typical=("16x64", "typical", "16x64", "four_over_six_rows"),
    mixed_16x64_worst=("16x64", "worst", "16x64", "four_over_six_rows"),
    mixed_256x64_typical=("256x64", "typical", "256x64", "four_over_six_rows"),
    mixed_256x64_worst=("256x64", "worst", "256x64", "four_over_six_rows"),
    n8k64_wB_typical=("8x64", "typical", "8x64", "four_over_six_rows"),
    n8k64_wB_worst=("8x64", "worst", "8x64", "four_over_six_rows"))
KIND = {"fourover6": "four_over_six", "nvfp4": "nvfp4"}


def kernel_sets(kernel_set: str, build_root=None) -> tuple[dict, dict]:
    """``({kernel key: kernel or set}, {kernel key: (label, routing note)})``, resolved by ``models.sm120.resolve_kernel``
    as the install resolves a policy: ``auto`` gives stock_ko, stock_wB_ko, mixed_ko, mixed256_ko and mixed_wB_ko,
    ``paper`` gives stock, stock_wB, mixed, mixed and n8k64_wB."""
    if kernel_set not in ("auto", "paper"):
        raise SystemExit(f"--kernel-set {kernel_set!r}: auto or paper")
    routes = dict(baseline=(kernel_set, "fourover6", None),
                  baseline_wB=("auto_stock_wB" if kernel_set == "auto" else "stock_wB", "fourover6", None),
                  **{u: (kernel_set, "mixfp4", u) for u in ("16x64", "256x64", "8x64")})
    out, notes = {}, {}
    for key, (route, weight, unit) in routes.items():
        try:
            kern, label, note = NS.resolve_kernel(route, weight, unit, build_root)
        except ValueError as e:
            raise SystemExit(f"--kernel-set {kernel_set}: {e}")
        out[key], notes[key] = kern, dict(set=label, routing=note)
    return out, notes


class CheckFailed(SystemExit):
    pass


def classify(name):
    n = name.lower()
    if "quant_rows_kernel" in n:
        return "quant"
    if "cutlass" in n or "device_kernel" in n or "gemm" in n:
        return "gemm"
    return "other"


class Telemetry:
    """NVML snapshots, and a background nvidia-smi sampler (bench_gemm_isolated.Telemetry)."""

    def __init__(self, csv_path=None):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.proc = None
        if csv_path is not None:
            q = ("timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,"
                 "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown")
            self.csv = open(csv_path, "w")
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,nounits", "-lms", "100"],
                                         stdout=self.csv, stderr=subprocess.DEVNULL)

    def snap(self):
        nv = self.nv
        reasons = (nv.nvmlDeviceGetCurrentClocksEventReasons(self.h) if hasattr(nv, "nvmlDeviceGetCurrentClocksEventReasons")
                   else nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.h))
        return dict(t=time.time(), sm_mhz=nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
                    mem_mhz=nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_MEM),
                    power_w=nv.nvmlDeviceGetPowerUsage(self.h) / 1e3,
                    temp_c=nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU),
                    energy_mj=nv.nvmlDeviceGetTotalEnergyConsumption(self.h),
                    power_cap_ns=nv.nvmlDeviceGetViolationStatus(self.h, nv.NVML_PERF_POLICY_POWER).violationTime,
                    reasons=hex(reasons))

    def others(self):
        return [p.pid for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h) if p.pid != os.getpid()]

    def power_limit_w(self):
        return self.nv.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1e3

    @staticmethod
    def block(a, b):
        dt = b["t"] - a["t"]
        de = b["energy_mj"] - a["energy_mj"]      # NVML's energy counter updates too slowly for blocks under ~0.5 s
        return dict(seconds=dt, sm_mhz=[a["sm_mhz"], b["sm_mhz"]], power_w=[a["power_w"], b["power_w"]],
                    mean_power_w=de / 1e3 / dt if de > 0 and dt >= 0.5 else None,
                    power_cap_ms=(b["power_cap_ns"] - a["power_cap_ns"]) / 1e6, temp_c=[a["temp_c"], b["temp_c"]],
                    reasons_after=b["reasons"])

    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.csv.close()

    @staticmethod
    def summarize(csv_path):
        def num(v, f):
            try:
                return f(v.strip())
            except ValueError:
                return None
        rows = [r for r in list(csv.reader(open(csv_path)))[1:] if len(r) >= 8]
        pairs = [(num(r[1], int), num(r[3], float)) for r in rows]
        sm = [a for a, _ in pairs if a is not None]
        pw = [b for _, b in pairs if b is not None]
        cap = sum(1 for r in rows if r[6].strip() == "Active")
        busy = [a for a, b in pairs if a is not None and b is not None and b > 100]
        return dict(samples=len(rows), sm_mhz_min=min(sm), sm_mhz_median=statistics.median(sm), sm_mhz_max=max(sm),
                    busy_samples=len(busy), busy_sm_mhz_median=statistics.median(busy) if busy else None,
                    busy_sm_mhz_min=min(busy) if busy else None, power_w_max=max(pw), power_w_median=statistics.median(pw),
                    sw_power_cap_samples=cap) if sm and pw else dict(samples=len(rows))


def stats(v):
    q = statistics.quantiles(v, n=4) if len(v) >= 4 else [min(v), statistics.median(v), max(v)]
    return dict(median=statistics.median(v), q1=q[0], q3=q[2], min=min(v), max=max(v), n=len(v))


@torch.no_grad()
def packed_modules(model, spec, maps):
    """{kind: {module: (PackedWeight, E0M3 granules)}} and {kind: type block}: every quantizable Linear packed as the
    sm120 install packs it, for the baselines and every given map."""
    R = NS.package()
    linears = quantizable_linears(model, spec)
    out, tbs = {}, {}
    for kind in ("fourover6", "nvfp4", *maps):
        out[kind] = {}
        tiles = unit = None
        if kind in maps:                  # a flipquant-map/1 file of that unit, or a bare map.pt
            tiles, unit, _ = M.load(maps[kind], kind)
        for name, lin in linears.items():
            mask = tb = None
            if tiles is not None:
                mask, tb = NS._packing(tiles[name].to(lin.weight.device), unit, lin.out_features)
            pw = R.artifact.pack_module(name, lin.weight.data, lin.bias, NS.KIND["mixfp4" if tiles is not None else kind],
                                        mask, tb)
            out[kind][name] = (dataclasses.replace(pw, packed=pw.packed.cpu(), scales=pw.scales.cpu(),
                                                   bias=None if pw.bias is None else pw.bias.cpu()),
                               int(mask.sum()) if mask is not None else 0)
        tbs[kind] = list(tb) if tiles is not None else None
    return out, tbs


def isolated_quant(kern, lin, x):
    """The activation quantizer alone, as sm120_linear launches it: (packed codes, scale bytes, global scales)."""
    return kern.quant_rows(x, lin.act_kind)


def isolated_gemm(kern, lin, q, wp, wsf, y, stream):
    """The GEMM with its fused epilogue alone, as sm120_linear launches it, on the weights (wp, wsf): a rotation copy.
    The launch has the kernel set's scheduler row for the call, as NativeLinear's has (the row changes no output bit,
    so the bitwise check cannot tell)."""
    xp, xsf, gs = q
    n, k, t = lin.out_features, lin.in_features, xp.shape[0]
    sched = lin.kernel_set.schedule(n, k, t) if lin.kernel_set is not None else (0, 1)
    bptr = None if lin.bias_bf16 is None else lin.bias_bf16.data_ptr()
    if lin.weights_on_a:
        kern.gemm_ptr(wp.data_ptr(), wsf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k,
                      None, lin.global_scale, gs.data_ptr(), 1.0, bptr, y, stream.cuda_stream, schedule=sched)
    else:
        kern.gemm_ptr(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), t, n, k,
                      gs.data_ptr(), 1.0, None, lin.global_scale, bptr, y, stream.cuda_stream, schedule=sched)
    return y


def tag_modules(mods, tb):
    """{projection: dict(shape, modules, tiles_per_module, first, typical, worst)}; each choice is (name, E0M3 granules)."""
    by = OrderedDict()
    for name, (pw, tiles) in mods.items():
        by.setdefault(name.rsplit(".", 1)[-1], []).append((name, pw, tiles))
    out = OrderedDict()
    for proj, ms in by.items():
        shape = tuple(ms[0][1].shape)
        assert all(tuple(pw.shape) == shape for _, pw, _ in ms), proj
        order = sorted(range(len(ms)), key=lambda i: (ms[i][2], i))
        typical = ms[order[(len(ms) - 1) // 2]]
        worst = max(ms, key=lambda m: m[2])                      # the first maximum
        out[proj] = dict(shape=list(shape), modules=len(ms),
                         tiles_per_module=None if not tb else -(-shape[0] // tb[0]) * (shape[1] // tb[1]),
                         first=(ms[0][0], ms[0][2]), typical=(typical[0], typical[2]), worst=(worst[0], worst[2]),
                         shares=sorted(m[2] for m in ms))
    return out


def per_forward(rec, reuse):
    """{config@T: us} GEMM sums and {act@T: us} quantizer sums (gemm_isolated_tables.per_forward): each projection's
    time times its module count; quantizer launches net of the reuse measured by ``prefill``."""
    projs = rec["projections"]
    by = {(r["config"], r["proj"], r["tokens"]): r for r in rec["rows"]}
    timed = [p for p in projs if any(r["proj"] == p for r in rec["rows"])]
    sums, qsums = {}, {}
    for t in sorted({r["tokens"] for r in rec["rows"]}):
        for c in sorted({r["config"] for r in rec["rows"]}):
            if all((c, p, t) in by for p in timed):
                sums[f"{c}@{t}"] = sum(projs[p]["modules"] * by[(c, p, t)]["gemm_us"] for p in timed)
        for c, act in (("stock_wA", "four_over_six_rows"), ("stock_wA_nvfp4", "nvfp4_rows")):
            if all((c, p, t) in by for p in timed):
                launches = {p: projs[p]["modules"] * (1 - (reuse[p]["quant_reused"] / reuse[p]["calls"]
                                                           if reuse and p in reuse and reuse[p]["calls"] else 0.0))
                            for p in timed}
                qsums[f"{act}@{t}"] = sum(launches[p] * by[(c, p, t)]["quant_us"] for p in timed)
    return dict(per_forward_gemm_us=sums, per_forward_quant_us=qsums, projections_timed=timed,
                complete=len(timed) == len(projs), quant_reuse=reuse)


def run_gemm(args):
    B, _ = bench()
    B.require_idle()
    csv_path = Path(str(args.out) + ".telemetry.csv")
    tel = Telemetry(csv_path)
    try:
        _run_gemm(args, B, tel)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res["sampler"] = Telemetry.summarize(csv_path)
    B.write(args.out, res)


def _run_gemm(args, B, tel):
    from torch.profiler import ProfilerActivity, profile
    R = NS.package()
    Kernel, KernelSet, NativeLinear = R.lib.Kernel, R.select.KernelSet, R.linear.NativeLinear
    torch.backends.cuda.matmul.allow_tf32 = False
    maps = dict(s.split("=", 1) for s in args.maps.split(",")) if args.maps else {}
    model, _, spec = load(args.model, revision=args.revision, device_map="cuda", attn="sdpa")
    packed, tbs = packed_modules(model, spec, maps)
    del model
    torch.cuda.empty_cache()
    kernels, routing = kernel_sets(args.kernel_set, args.kernel_build_dir)
    cfgs = [c for c, (kind, *_) in CONFIGS.items() if kind in packed and (not args.configs or c in args.configs.split(","))]
    res = dict(status="running", bench="gemm", gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(),
               l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size, model=spec.key, maps=maps,
               kernel_set=args.kernel_set, build_dir=str(NS.build_dir(args.kernel_build_dir)), routing=routing,
               protocol=dict(method="isolated launches, cold weights (the paper's deviation 2, PROTOCOL_GEMM_ISOLATED)",
                             reps=args.reps, warmup=args.warmup, rounds=args.rounds, flush_mib=args.flush_mib,
                             order="rotated: round r starts at position r * len / rounds",
                             value="median over all rounds of the CUPTI device time of isolated launches"),
               kernels={k: (v.sha256 if isinstance(v, Kernel) else v.describe()) for k, v in kernels.items()},
               configs={c: dict(weights=CONFIGS[c][0], tags=CONFIGS[c][1], kernel=CONFIGS[c][2],
                                activation_quantizer=CONFIGS[c][3]) for c in cfgs},
               projections={}, rows=[], blocks=[], checks=dict(bitwise=[], counts_ok=True, other_processes=[]))
    if args.flush_mib * 2 ** 20 < 4 * res["l2_bytes"]:
        raise SystemExit(f"--flush-mib {args.flush_mib} is below 4x the L2 ({res['l2_bytes']} bytes)")
    pw = {}
    for kind, mods in packed.items():
        tags = tag_modules(mods, tbs[kind])
        for proj, t in tags.items():
            p = res["projections"].setdefault(proj, dict(shape=t["shape"], modules=t["modules"], tags={}))
            assert p["shape"] == t["shape"] and p["modules"] == t["modules"], (kind, proj)
            p["tags"][kind] = dict(type_block=tbs[kind], tiles_per_module=t["tiles_per_module"],
                                   **{v: dict(module=t[v][0], e0m3_tiles=t[v][1],
                                              e0m3_share=None if not t["tiles_per_module"] else t[v][1] / t["tiles_per_module"])
                                      for v in ("first", "typical", "worst")})
            for v in ("first", "typical", "worst"):
                pw[(kind, v, proj)] = mods[t[v][0]][0]
    del packed
    projs = list(res["projections"])
    if args.projections:
        projs = [p for p in projs if p in args.projections.split(",")]
    tokens_list = [int(t) for t in args.tokens.split(",")]
    flush = torch.ones(args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()
    for proj in projs:
        n, k = res["projections"][proj]["shape"]
        lins, copies, counter = {}, {}, {}
        for c in cfgs:
            kind, variant, kname, act = CONFIGS[c]
            w = pw[(kind, variant, proj)]
            w = dataclasses.replace(w, packed=w.packed.cuda(), scales=w.scales.cuda(), bias=None if w.bias is None else w.bias.cuda())
            lins[c] = NativeLinear(w, kernels[kname], act, name=f"{proj}@{c}")
            lins[c].share_input = False
            # rotation: distinct copies of the weights, K x size >= 4x L2; each call uses the next one
            size = lins[c].packed.numel() + lins[c].sf.numel()
            copies[c] = [(lins[c].packed.clone(), lins[c].sf.clone()) for _ in range(math.ceil(4 * res["l2_bytes"] / size) + 1)]
            counter[c] = 0
        res["projections"][proj]["rotation_copies"] = {c: len(copies[c]) for c in cfgs}
        for t in tokens_list:
            g = torch.Generator("cpu").manual_seed(n + k + t)
            pool = [torch.randn(t, k, generator=g).to("cuda", torch.bfloat16) for _ in range(2)]
            x = torch.empty_like(pool[0])
            per = {c: [] for c in cfgs}
            fns = {}
            for c in cfgs:
                lin = lins[c]
                kern = lin.kernel_set.pick(n, k, t) if lin.kernel_set is not None else lin.kernel
                y = torch.empty((t, n), dtype=torch.bfloat16, device="cuda")

                def quant(kern=kern, lin=lin):
                    return isolated_quant(kern, lin, x)

                def gemm(q, kern=kern, lin=lin, y=y, c=c):
                    wp, wsf = copies[c][counter[c] % len(copies[c])]
                    counter[c] += 1
                    return isolated_gemm(kern, lin, q, wp, wsf, y, stream)
                fns[c] = (quant, gemm, kern)
                # registered check: the isolated path equals the deployment path (NativeLinear's fused forward) bitwise
                x.copy_(pool[0])
                ref = lin(x)
                got = gemm(quant()).clone()
                ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
                res["checks"]["bitwise"].append(dict(proj=proj, tokens=t, config=c, equal=ok))
                if not ok:
                    res["status"] = "failed"
                    B.write(args.out, res)
                    raise CheckFailed(f"{proj} T={t} {c}: the isolated GEMM differs from NativeLinear")

            def rep(c, i, timed):
                quant, gemm, _ = fns[c]
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

            block_before = tel.snap()
            for r in range(args.rounds):
                shift = (r * len(cfgs) // args.rounds) % len(cfgs)
                for pos, c in enumerate(cfgs[shift:] + cfgs[:shift]):
                    others = tel.others()
                    if others:
                        res["checks"]["other_processes"].append(dict(proj=proj, tokens=t, config=c, round=r, pids=others))
                        res["status"] = "failed"
                        B.write(args.out, res)
                        raise CheckFailed(f"another compute process on the GPU: {others}")
                    for i in range(args.warmup):
                        rep(c, i, False)
                    before = tel.snap()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        ev = [rep(c, i, True) for i in range(args.reps)]
                    after = tel.snap()
                    kt = {"gemm": [], "quant": []}
                    for e in prof.events():
                        if e.device_type.name == "CUDA":
                            cls = classify(e.name)
                            if cls in kt:
                                kt[cls].append(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
                    if len(kt["gemm"]) != args.reps or len(kt["quant"]) != args.reps:
                        res["checks"]["counts_ok"] = False
                        res["status"] = "failed"
                        B.write(args.out, res)
                        raise CheckFailed(f"{proj} T={t} {c} round {r}: {len(kt['gemm'])} GEMM / {len(kt['quant'])} "
                                          f"quantizer launches profiled, expected {args.reps}")
                    per[c].append(dict(round=r, position=pos, gemm_us=kt["gemm"], quant_us=kt["quant"], event_us=ev,
                                       telemetry=Telemetry.block(before, after)))
            res["blocks"].append(dict(proj=proj, tokens=t, **Telemetry.block(block_before, tel.snap())))
            for c in cfgs:
                kind, variant, kname, act = CONFIGS[c]
                kern = fns[c][2]
                g_all = [v for blk in per[c] for v in blk["gemm_us"]]
                q_all = [v for blk in per[c] for v in blk["quant_us"]]
                e_all = [v for blk in per[c] for v in blk["event_us"]]
                res["rows"].append(dict(
                    proj=proj, out=n, inp=k, tokens=t, config=c, weights=kind, tags=variant, set=routing[kname]["set"],
                    kernel=kern.cfg.name, act=act,
                    width=lins[c].kernel_set.width(n, k, t) if lins[c].kernel_set is not None else None,
                    schedule=list(lins[c].kernel_set.schedule(n, k, t)) if lins[c].kernel_set is not None else None,
                    gemm_us=statistics.median(g_all), gemm=stats(g_all), quant_us=statistics.median(q_all), quant=stats(q_all),
                    event_us=statistics.median(e_all), event=stats(e_all),
                    rounds=[dict(round=b["round"], position=b["position"], gemm_us=statistics.median(b["gemm_us"]),
                                 quant_us=statistics.median(b["quant_us"]), event_us=statistics.median(b["event_us"]),
                                 telemetry=b["telemetry"]) for b in per[c]]))
            print(f"{spec.key} {proj} T={t} " + " ".join(f"{r['config']}={r['gemm_us']:.1f}" for r in res["rows"]
                                                         if r["proj"] == proj and r["tokens"] == t), flush=True)
            del pool, x, fns
            B.write(args.out, res)
        del lins, copies
    reuse = json.loads(Path(args.quant_reuse).read_text()).get("quant_reuse") if args.quant_reuse else None
    res["per_forward"] = per_forward(res, reuse)
    res["kernel_sets_after"] = {k: v.describe() for k, v in kernels.items() if isinstance(v, KernelSet)}
    res["gpu_end"] = B.gpu_info()
    res["status"] = "complete"
    B.write(args.out, res)


# configuration -> (weights, kernel, activation quantizer)    (bench_gemm.py, step 06)
WARM_CONFIGS = OrderedDict(stock_wA=("fourover6", "baseline", "four_over_six_rows"),
                           stock_wA_nvfp4=("nvfp4", "baseline", "nvfp4_rows"),
                           stock_wB=("fourover6", "baseline_wB", "four_over_six_rows"),
                           mixed_16x64=("16x64", "16x64", "four_over_six_rows"),
                           mixed_256x64=("256x64", "256x64", "four_over_six_rows"),
                           n8k64_wB=("8x64", "8x64", "four_over_six_rows"))


def representatives(mods):
    """{projection: (module, E0M3 granules, module count, shape)}: the module with the most E0M3 granules (the first
    module when none has any), as bench_gemm.representatives."""
    out = OrderedDict()
    for name, (pw, tiles) in mods.items():
        proj = name.rsplit(".", 1)[-1]
        best, most, count, shape = out.get(proj, (None, -1, 0, None))
        if tiles > most:
            best, most = name, tiles
        out[proj] = (best, most, count + 1, tuple(pw.shape))
    return out


def run_gemm_warm(args):
    B, _ = bench()
    B.require_idle()
    R = NS.package()
    Kernel, KernelSet, NativeLinear = R.lib.Kernel, R.select.KernelSet, R.linear.NativeLinear
    torch.backends.cuda.matmul.allow_tf32 = False
    maps = dict(s.split("=", 1) for s in args.maps.split(",")) if args.maps else {}
    model, _, spec = load(args.model, revision=args.revision, device_map="cuda", attn="sdpa")
    packed, tbs = packed_modules(model, spec, maps)
    del model
    torch.cuda.empty_cache()
    kernels, routing = kernel_sets(args.kernel_set, args.kernel_build_dir)
    res = dict(status="running", bench="gemm-warm", gpu=B.gpu_info(), model=spec.key, maps=maps, iters=args.iters,
               kernel_set=args.kernel_set, build_dir=str(NS.build_dir(args.kernel_build_dir)), routing=routing,
               protocol=dict(rounds=args.rounds, order="rotated: round r starts at position r * len / rounds",
                             value="median over rounds of the per-round CUPTI median of --iters calls"),
               kernels={k: (v.sha256 if isinstance(v, Kernel) else v.describe()) for k, v in kernels.items()},
               configs={c: dict(weights=w, kernel=k, activation_quantizer=q) for c, (w, k, q) in WARM_CONFIGS.items()
                        if w in packed}, projections={}, rows=[])
    reps = {}
    for kind, mods in packed.items():
        sel = representatives(mods)
        reps[kind] = {proj: mods[name][0] for proj, (name, *_) in sel.items()}
        for proj, (name, tiles, count, shape) in sel.items():
            p = res["projections"].setdefault(proj, dict(shape=list(shape), modules=count, weights={}))
            assert p["shape"] == list(shape) and p["modules"] == count, (kind, proj)
            tb = tbs[kind]
            p["weights"][kind] = dict(module=name, e0m3_tiles=tiles, type_block=tb,
                                      tiles_in_module=None if not tb else -(-shape[0] // tb[0]) * (shape[1] // tb[1]))
    del packed
    projs = [p for p in res["projections"] if not args.projections or p in args.projections.split(",")]
    for proj in projs:
        n, k = res["projections"][proj]["shape"]
        for t in [int(v) for v in args.tokens.split(",")]:
            x = torch.randn(t, k, generator=torch.Generator("cpu").manual_seed(n + k + t)).to("cuda", torch.bfloat16)
            cfgs = [c for c, (kind, _, _) in WARM_CONFIGS.items() if kind in reps]
            per = {c: [] for c in cfgs}
            for r in range(args.rounds):
                shift = (r * len(cfgs) // args.rounds) % len(cfgs)
                for pos, cfg in enumerate(cfgs[shift:] + cfgs[:shift]):
                    kind, kname, act = WARM_CONFIGS[cfg]
                    pw = reps[kind][proj]
                    pw = dataclasses.replace(pw, packed=pw.packed.cuda(), scales=pw.scales.cuda(),
                                             bias=None if pw.bias is None else pw.bias.cuda())
                    lin = NativeLinear(pw, kernels[kname], act, name=f"{proj}@{cfg}")
                    lin.share_input = False          # quantize on every call, as the first projection of a group does
                    times = B.kernel_times(lambda: lin(x), iters=args.iters)
                    by = {}
                    for name, v in times.items():
                        c = classify(name)
                        by[c] = by.get(c, 0.0) + v["us"]
                    per[cfg].append(dict(round=r, position=pos, gemm_us=by.get("gemm"), quant_us=by.get("quant"),
                                         other_us=by.get("other", 0.0),
                                         kernels={name: round(v["us"], 3) for name, v in times.items()}))
                    del lin
            for cfg in cfgs:
                kind, kname, act = WARM_CONFIGS[cfg]
                kern = kernels[kname]

                def med(key, cfg=cfg):
                    vals = [v[key] for v in per[cfg]]
                    return statistics.median(vals) if all(v is not None for v in vals) else None
                res["rows"].append(dict(proj=proj, out=n, inp=k, tokens=t, config=cfg, set=routing[kname]["set"],
                                        kernel=(kern.kernels[kern.width(n, k, t)] if isinstance(kern, KernelSet)
                                                else kern).cfg.name,
                                        act=act, width=kern.width(n, k, t) if isinstance(kern, KernelSet) else None,
                                        schedule=list(kern.schedule(n, k, t)) if isinstance(kern, KernelSet) else None,
                                        gemm_us=med("gemm_us"), quant_us=med("quant_us"), other_us=med("other_us"),
                                        rounds=[{k2: v[k2] for k2 in ("round", "position", "gemm_us", "quant_us")}
                                                for v in per[cfg]],
                                        kernels=per[cfg][-1]["kernels"]))
            print(f"{spec.key} {proj} T={t} " + " ".join(f"{r['config']}={r['gemm_us']:.1f}+{r['quant_us']:.1f}us"
                                                         for r in res["rows"] if r["proj"] == proj and r["tokens"] == t),
                  flush=True)
            B.write(args.out, res)
    res["kernel_sets_after"] = {k: v.describe() for k, v in kernels.items() if isinstance(v, KernelSet)}
    res["gpu_end"] = B.gpu_info()
    res["status"] = "complete"
    B.write(args.out, res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="bench", required=True)
    p = sub.add_parser("prefill", help="end-to-end prefill, eager and CUDA graph")
    add_model_args(p, act_scope="row")
    p.add_argument("--shapes", default="1x128,1x256,1x512,1x1024,1x2048,1x4096,1x8192,4x2048",
                   help="comma-separated BATCHxPROMPT (default: the paper's)")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--graph-check", choices=("bitwise", "report"), default="bitwise",
                   help="bitwise (default): the graph's logits must equal eager's; report: for a model whose eager "
                        "forward is not bitwise reproducible (Qwen3.8 on flash-linear-attention), record graph vs eager "
                        "and eager vs eager instead")
    p.add_argument("--act-override", choices=("nvfp4_rows", "four_over_six_rows"), default=None,
                   help="latency only: switch the activation quantizer (not an evaluated configuration)")
    d = sub.add_parser("decode", help="StaticCache decode in a CUDA graph")
    add_model_args(d, act_scope="row")
    d.add_argument("--settings", default="1x512,4x512,16x512,1x2048,4x2048,16x2048",
                   help="comma-separated BATCHxPROMPT (default: Experiment D's)")
    d.add_argument("--gen", type=int, default=64)
    d.add_argument("--act-override", choices=("nvfp4_rows", "four_over_six_rows"), default=None,
                   help="latency only: switch the activation quantizer (not an evaluated configuration)")
    for q in (p, d):
        q.add_argument("--label", default=None)
        q.add_argument("--round", type=int, default=1)
        q.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gemm", help="isolated cold GEMM and quantizer kernels per projection (deviation 2)")
    g.add_argument("--model", required=True)
    g.add_argument("--revision", default=None)
    g.add_argument("--kernel-set", choices=("auto", "paper"), default="auto",
                   help="auto: the optimized sets (the default); paper: the paper's kernels")
    g.add_argument("--kernel-build-dir", type=Path, default=None,
                   help="build directory (default $SM120_BUILD_DIR, else kernels/sm120/build_deploy)")
    g.add_argument("--maps", default=None,
                   help="comma-separated UNIT=FILE (8x64, 16x64, 256x64): flipquant-map/1 files or bare map.pt")
    g.add_argument("--tokens", default="128,256,512,1024,2048,4096,8192")
    g.add_argument("--projections", default=None, help="comma-separated subset (default: every text projection)")
    g.add_argument("--configs", default=None, help="comma-separated subset of the configurations")
    g.add_argument("--reps", type=int, default=30, help="timed repetitions per configuration per round")
    g.add_argument("--warmup", type=int, default=3)
    g.add_argument("--rounds", type=int, default=3)
    g.add_argument("--flush-mib", type=int, default=512)
    g.add_argument("--quant-reuse", type=Path, default=None, help="a prefill report (its quant_reuse) for the sums")
    g.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("gemm-warm", help="back-to-back NativeLinear calls on L2-resident weights (step 06)")
    w.add_argument("--model", required=True)
    w.add_argument("--revision", default=None)
    w.add_argument("--kernel-set", choices=("auto", "paper"), default="auto",
                   help="auto: the optimized sets (the default); paper: the paper's kernels")
    w.add_argument("--kernel-build-dir", type=Path, default=None,
                   help="build directory (default $SM120_BUILD_DIR, else kernels/sm120/build_deploy)")
    w.add_argument("--maps", default=None,
                   help="comma-separated UNIT=FILE (8x64, 16x64, 256x64): flipquant-map/1 files or bare map.pt")
    w.add_argument("--tokens", default="128,256,512,1024,2048,4096,8192")
    w.add_argument("--projections", default=None, help="comma-separated subset (default: every text projection)")
    w.add_argument("--iters", type=int, default=20)
    w.add_argument("--rounds", type=int, default=3, help="rounds per (projection, T), in rotated order")
    w.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.bench == "gemm":
        return run_gemm(args)
    if args.bench == "gemm-warm":
        return run_gemm_warm(args)
    args.label = args.label or (args.mode if args.mode == "bf16" else f"{args.weight}"
                                + (f"-{Path(args.map).stem}" if args.map else ""))
    return run_prefill(args) if args.bench == "prefill" else run_decode(args)


if __name__ == "__main__":
    sys.exit(main())
