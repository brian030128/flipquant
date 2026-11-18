#!/usr/bin/env python3
"""Checks of the tuned SM100 native path (models/sm100.py + libflipquant_sm100_linear.so) on an sm_100 GPU.

1. Activation quantizer: the Triton quantizer's codes, global scales and placed scale factors equal the reference
   quantizer's (mixfp4_sm120.numerics) placed in CUTLASS's SFA layout.
2. GEMM + epilogue: Sm100Linear(x) against an FP64 reference built from the exact decoded operands,
   (A_dec @ W_dec^T) + bias, for the stock kernel (NVFP4 and FourOverSix weights) and the per-tile kernel (random
   256x64 maps), over shapes with N/K tails and T = 1..2048. Negative control: the per-tile kernel with an all-E2M1
   map must miss the mixed reference.
3. CUDA graphs: a captured forward replays bit-identically to eager.
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from models import sm100, sm120  # noqa: E402

R = sm120.package()
N_ = R.numerics


def reference(x2d, act_kind, pw):
    nib, sb, gs = N_.quantize_act(x2d, act_kind)
    a = N_.decode_exact(nib, sb, gs.double()[:, None])
    w = N_.decode_exact(N_.unpack_nibbles(pw.packed), pw.scales, pw.global_scale)
    out = a @ w.T
    if pw.bias is not None:
        out = out + pw.bias.double()
    return out


def rel_fro(got, ref):
    return float((got.double() - ref).norm() / ref.norm().clamp_min(1e-30))


def check_quantizer(t, k, act_kind):
    """Both quantizers (the library's CUDA one, used by Sm100Linear, and the Triton one) against the reference."""
    x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16) * 3
    nib, sb, gs_ref = N_.quantize_act(x, act_kind)
    placed = R.linear.place_scales(sb, k)
    n_sf = R.lib.sf_buffer_size(t, k)
    out = dict(t=t, k=k, act=act_kind)
    for name, (packed, sf, gs) in (("cuda", sm100.quantize_rows(x, act_kind)),
                                   ("triton", R.quant_act.quantize(x, act_kind))):
        out[name] = bool(torch.equal(packed, N_.pack_nibbles(nib)) and torch.equal(gs, gs_ref.float())
                         and torch.equal(sf[:n_sf], placed[:n_sf]))
    return out


def build(n, k, weight, bias, density, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(n, device="cuda", generator=g) * 0.1).to(torch.bfloat16) if bias else None
    tiles = None
    if weight == "mixfp4":
        tiles = torch.rand((n + 255) // 256, k // 64, device="cuda", generator=g) < density
        rows = tiles.repeat_interleave(256, 0)[:n]
        pw = R.artifact.pack_module("w", w, b, "map", rows, (1, 64))
    else:
        pw = R.artifact.pack_module("w", w, b, sm100.KIND[weight])
    return pw, tiles


def main():
    report = dict(gpu=torch.cuda.get_device_name(0), library=str(sm100.library_path()), quantizer=[], gemm=[],
                  graph=None, negative_control=None)
    sm100.library()                       # also checks the SF placement against CUTLASS's layout
    ok = True
    for t, k in ((1, 64), (7, 4480), (130, 1344), (2048, 4096)):
        for act in ("nvfp4_rows", "four_over_six_rows"):
            r = check_quantizer(t, k, act)
            report["quantizer"].append(r)
            ok &= r["cuda"] and r["triton"]
            print(json.dumps(r), flush=True)
    shapes = [(1, 4096, 4096), (7, 48, 5120), (130, 720, 4480), (2048, 4480, 15680), (2048, 22656, 4480),
              (2048, 1024, 4096), (333, 6144, 2048)]
    for t, n, k in shapes:
        for weight, act in (("nvfp4", "nvfp4_rows"), ("fourover6", "four_over_six_rows"),
                            ("mixfp4", "four_over_six_rows")):
            pw, tiles = build(n, k, weight, bias=(n % 2 == 0), density=0.3, seed=t + n + k)
            layer = sm100.Sm100Linear(pw, act, tiles, name=f"{weight}-{t}x{n}x{k}")
            x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
            got = layer(x)
            torch.cuda.synchronize()
            err = rel_fro(got, reference(x, act, pw))
            r = dict(t=t, n=n, k=k, weight=weight, variant=layer.variant, rel_fro=err,
                     e0m3_tiles=0 if tiles is None else int(tiles.sum()), passed=err < 5e-3)
            ok &= r["passed"]
            report["gemm"].append(r)
            print(json.dumps(r), flush=True)
    # Negative control: the per-tile kernel told "all E2M1" must miss a reference with E0M3 tiles.
    pw, tiles = build(4480, 4480, "mixfp4", bias=True, density=0.5, seed=1)
    wrong = sm100.Sm100Linear(pw, "four_over_six_rows", torch.zeros_like(tiles), name="all-e2m1-map")
    x = torch.randn(256, 4480, device="cuda", dtype=torch.bfloat16)
    err = rel_fro(wrong(x), reference(x, "four_over_six_rows", pw))
    report["negative_control"] = dict(rel_fro=err, failed_as_required=err > 5e-2)
    ok &= err > 5e-2
    print(json.dumps(report["negative_control"]), flush=True)
    # CUDA graph: capture one forward, replay, compare bitwise with eager.
    layer = sm100.Sm100Linear(pw, "four_over_six_rows", tiles, name="graph")
    x = torch.randn(2048, 4480, device="cuda", dtype=torch.bfloat16)
    eager = layer(x).clone()
    sm100.clear_quant_cache()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer(x)
        sm100.clear_quant_cache()         # the graph must contain the quantizer, not reuse the warm-up's output
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    report["graph"] = dict(equal_to_eager=bool(torch.equal(out, eager)))
    ok &= report["graph"]["equal_to_eager"]
    print(json.dumps(report["graph"]), flush=True)
    report["passed"] = bool(ok)
    print(json.dumps(dict(passed=report["passed"])))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
