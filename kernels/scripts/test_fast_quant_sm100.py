#!/usr/bin/env python3
"""The fast SM100 activation quantizer (kernels/src/quant_act_fast.cuh, the ``*_rows_fast`` kinds of models/sm100.py)
against the exact one (SM120's deployed quantizer, bit-identical to the simulation), on an sm_100 GPU.

1. Agreement: per (T, K, mode), the global scales must be identical (same amax, same multiply); reported are the shares
   of identical scale bytes and E2M1 codes, and the activation quantization MSE (decoded vs the bf16 input) of the fast
   quantizer over the exact one's. The padding of the scale buffer must be zero. Inputs: Gaussian with outliers.
2. GEMM: Sm100Linear's output against the unquantized product x @ W_dec^T (FP64), with the exact and the fast quantizer.
3. CUDA graphs: a captured fast-quantizer forward replays bit-identically to eager.
4. Speed: CUDA-event time of the quantizer calls of one decoder layer at T tokens (one call per distinct input: q/k/v
   share one, gate/up share one), exact vs fast, for the paper's models.

    python kernels/scripts/test_fast_quant_sm100.py OUT.json
"""
import json
import statistics
import sys

import torch

sys.path.insert(0, ".")
from models import sm100, sm120  # noqa: E402

R = sm120.package()
N_ = R.numerics
KINDS = {"nvfp4_rows": "nvfp4_rows_fast", "four_over_six_rows": "four_over_six_rows_fast"}
#: K of the distinct quantizer inputs of one decoder layer (count per layer)
LAYER_K = {
    "llama3.2-1b-instruct": {2048: 2, 8192: 1},
    "llama3.1-8b-instruct": {4096: 3, 14336: 1},
    "mistral-7b-instruct": {4096: 3, 14336: 1},
    "phi-4": {5120: 3, 17920: 1},
    "qwen3.8-27b (either layer type)": {5120: 2, 6144: 1, 17408: 1},
    "nemotron-nano-9b-v2 (attention layer)": {4480: 1, 5120: 1},
    "nemotron-nano-9b-v2 (mamba layer)": {4480: 1, 10240: 1},
    "nemotron-nano-9b-v2 (mlp layer)": {4480: 1, 15680: 1},
}


def activations(t, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, k, device="cuda", generator=g)
    x[:, torch.randint(0, k, (max(1, k // 512),), device="cuda", generator=g)] *= 40     # outlier channels
    return x.to(torch.bfloat16)


def sf_index(t, k):
    """Byte offset of row r's scale of K16 block j in the SFA buffer (quant_act.cuh's sf_offset), [t, k/16]."""
    kb = k // 16
    k_atoms = (kb + 3) // 4
    r = torch.arange(t, device="cuda")[:, None]
    j = torch.arange(kb, device="cuda")[None, :]
    return ((r // 128) * k_atoms + j // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + j % 4


def decode(packed, sf, gs, t, k):
    nib = N_.unpack_nibbles(packed)
    sb = sf[sf_index(t, k)]
    return N_.decode_exact(nib, sb, gs.double()[:, None])


def agreement(t, k, act_kind, seed=0):
    x = activations(t, k, seed)
    exact = sm100.quantize_rows(x, act_kind)
    fast = sm100.quantize_rows(x, KINDS[act_kind])
    n_sf = R.lib.sf_buffer_size(t, k)
    idx = sf_index(t, k)
    pad = torch.ones(n_sf, dtype=torch.bool, device="cuda")
    pad[idx.flatten()] = False
    ref = x.double()
    err = {name: float(((decode(p, s, g, t, k) - ref) ** 2).mean()) for name, (p, s, g) in (("exact", exact), ("fast", fast))}
    nib_e, nib_f = N_.unpack_nibbles(exact[0]), N_.unpack_nibbles(fast[0])
    zero = lambda n: (n & 7) == 0  # noqa: E731  (+0 and -0 are the same value)
    same_code = (nib_e == nib_f) | (zero(nib_e) & zero(nib_f))
    return dict(t=t, k=k, act=act_kind, gs_identical=bool(torch.equal(exact[2], fast[2])),
                scale_bytes_identical=float((exact[1][idx] == fast[1][idx]).float().mean()),
                codes_identical=float(same_code.float().mean()),
                padding_zero=bool((fast[1][:n_sf][pad] == 0).all()),
                mse_fast_over_exact=err["fast"] / err["exact"])


def gemm_error(t, n, k, weight, act_kind, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    tiles = None
    if weight == "mixfp4":
        tiles = torch.rand((n + 255) // 256, k // 64, device="cuda", generator=g) < 0.1
        pw = R.artifact.pack_module("w", w, None, "map", tiles.repeat_interleave(256, 0)[:n], (1, 64))
    else:
        pw = R.artifact.pack_module("w", w, None, sm100.KIND[weight])
    wd = N_.decode_exact(N_.unpack_nibbles(pw.packed), pw.scales, pw.global_scale)
    x = activations(t, k, seed)
    ref = x.double() @ wd.T
    out = dict(t=t, n=n, k=k, weight=weight)
    for name, kind in (("exact", act_kind), ("fast", KINDS[act_kind])):
        sm100.clear_quant_cache()
        layer = sm100.Sm100Linear(pw, kind, tiles, name=name)
        out[f"rel_err_{name}"] = float((layer(x).double() - ref).norm() / ref.norm())
    return out


def graph_check():
    t, n, k = 512, 4096, 4096
    g = torch.Generator(device="cuda").manual_seed(2)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    pw = R.artifact.pack_module("w", w, None, sm100.KIND["fourover6"])
    layer = sm100.Sm100Linear(pw, "four_over_six_rows_fast", None, name="graph")
    x = activations(t, k, 3)
    sm100.clear_quant_cache()
    eager = layer(x).clone()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        sm100.clear_quant_cache()
        layer(x)
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    sm100.clear_quant_cache()
    with torch.cuda.graph(graph):
        out = layer(x)
    graph.replay()
    torch.cuda.synchronize()
    return dict(graph_equals_eager=bool(torch.equal(out, eager)))


def time_kind(x, kind, calls=20, reps=10):
    """GPU time per call (us): `calls` back-to-back calls captured in one CUDA graph, the median of `reps` replays
    (an eager call's time would be the host's launch overhead for the small shapes)."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            sm100.quantize_rows(x, kind)
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            sm100.quantize_rows(x, kind)
    for _ in range(3):
        graph.replay()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / calls)
    del graph
    return statistics.median(ts)


def speed(t=2048):
    rows = []
    cache = {}
    for k in sorted({k for ks in LAYER_K.values() for k in ks}):
        x = activations(t, k, 4)
        for act in KINDS:
            for kind in (act, KINDS[act]):
                cache[(k, kind)] = time_kind(x, kind)
        gb = (t * k * 2 + t * k // 2) / 1e9
        rows.append(dict(t=t, k=k, **{f"{kind}_us": round(cache[(k, kind)], 2) for act in KINDS for kind in (act, KINDS[act])},
                         fast_fourover6_GBps=round(gb / (cache[(k, "four_over_six_rows_fast")] * 1e-6), 0)))
    layers = {}
    for model, ks in LAYER_K.items():
        layers[model] = {kind: round(sum(c * cache[(k, kind)] for k, c in ks.items()), 1)
                         for act in KINDS for kind in (act, KINDS[act])}
    return dict(per_call=rows, per_layer_us=layers)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fast_quant.json"
    torch.manual_seed(0)
    report = dict(gpu=torch.cuda.get_device_name(), agreement=[], gemm=[])
    for t, k in ((1, 4096), (7, 4480), (130, 1344), (2048, 2048), (2048, 4096), (2048, 5120), (2048, 14336),
                 (1000, 17920), (64, 64)):
        for act in KINDS:
            report["agreement"].append(agreement(t, k, act))
            print(json.dumps(report["agreement"][-1]), flush=True)
    for t, n, k in ((1, 4096, 4096), (2048, 4096, 4096), (2048, 4480, 15680), (300, 48, 5120)):
        for weight, act in (("nvfp4", "nvfp4_rows"), ("fourover6", "four_over_six_rows"),
                            ("mixfp4", "four_over_six_rows")):
            report["gemm"].append(gemm_error(t, n, k, weight, act))
            print(json.dumps(report["gemm"][-1]), flush=True)
    report["graph"] = graph_check()
    print(json.dumps(report["graph"]), flush=True)
    report["speed"] = speed()
    for r in report["speed"]["per_call"]:
        print(json.dumps(r), flush=True)
    for m, r in report["speed"]["per_layer_us"].items():
        print(m, json.dumps(r), flush=True)
    a = report["agreement"]
    ok = (all(r["gs_identical"] and r["padding_zero"] and r["mse_fast_over_exact"] < 1.01 for r in a)
          and all(r["rel_err_fast"] < 1.01 * r["rel_err_exact"] + 1e-6 for r in report["gemm"])
          and report["graph"]["graph_equals_eager"])
    report["ok"] = ok
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("ok" if ok else "CHECK FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
