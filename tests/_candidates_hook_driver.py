"""Helper of tests/test_gptq_candidates.py (not a test module): drives calibration/tmopt_launch.py's ``Candidates`` hook
through run_train_map's own call sequence for the lean store -- ``sha``, the two candidate quantizers, ``pack`` and
``native_dev.pack_candidates`` (CandidateStore.add) -- with the launcher's sys.path, in its own process.

    python tests/_candidates_hook_driver.py WEIGHTS.pt OUT_DIR rtn rtn
    python tests/_candidates_hook_driver.py WEIGHTS.pt OUT_DIR E2M1.pt E0M3.pt

Writes OUT_DIR/result.json: per module, whether the packed store entry equals the unhooked one byte for byte, and
whether it decodes to the given codes (E2M1 bit for bit, E0M3 by value: the store keeps E0M3 zeros as +0).
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "calibration" / "tmopt"
sys.path[0:0] = [str(ENGINE), str(ENGINE / "repro_local" / "realquant")]     # as run_train_map's lean mode
sys.path.append(str(REPO))

import torch  # noqa: E402

import candidate_store  # noqa: E402,F401  (the trainer has imported it when the lean store exists)
import native_dev  # noqa: E402
import rq  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402


def main():
    weights_file, out_dir, src_b, src_a = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location("tmopt_launch", REPO / "calibration" / "tmopt_launch.py")
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    weights = torch.load(weights_file, weights_only=True)
    mods = {}
    for n, w in weights.items():
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.bfloat16)
        lin.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        mods[n] = lin
    R = types.SimpleNamespace(sha=lambda t: "sha", quant_nvfp4_4over6=quant_nvfp4_4over6, quant_mix_4_6=quant_mix_4_6,
                              pack=pack)
    captured = {"weights": {id(m.weight): (n, m) for n, m in mods.items()}}
    hook = L.Candidates(R, captured, dict(e2m1=src_b, e0m3=src_a), list(mods), Path(out_dir))
    packed, packed_path = {}, {}
    for n, m in mods.items():                       # run_train_map's candidate loop (lean memory mode)
        assert R.sha(m.weight) == "sha"
        b = R.quant_nvfp4_4over6(m.weight, 4, 16)
        a = R.quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip="a1", elect="always")
        p = R.pack(m.weight, b, a)
        packed_path[n] = p is not None              # False: the trainer's dense fallback
        packed[n] = native_dev.pack_candidates(n, m.weight, *((b, a) if p is None else (decode_base(p), decode_alt(p))))
    rec = hook.finish()
    rq.weight_four_over_six, rq.weight_e0m3 = hook.orig_rq["weight_four_over_six"], hook.orig_rq["weight_e0m3"]
    from flipquant import formats as F
    from flipquant.gptq import load_codes_file
    files = {k: load_codes_file(Path(s))[1] for k, s in (("e2m1", src_b), ("e0m3", src_a)) if s != "rtn"}
    result = dict(hook=dict(lean=rec["lean_store"], modules=rec["modules"], other_calls=rec["other_calls"]), modules={})
    for n, m in mods.items():
        w = m.weight
        b4, b0, sb4, sb0, gs, o, k = packed[n]
        # the unhooked store entry: the trainer's own quantizers and rq's own encoders
        p = pack(w, quant_nvfp4_4over6(w, 4, 16), quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip="a1", elect="always"))
        ref = native_dev.pack_candidates(n, w, decode_base(p), decode_alt(p))
        r = dict(store_equals_unhooked=all(torch.equal(x, y) for x, y in zip(packed[n][:5], ref[:5])),
                 packed_path=packed_path[n])
        for kind, nib, sb in (("e2m1", b4, sb4), ("e0m3", b0, sb0)):
            got = native_dev.decode_packed(nib, sb, gs, o, k)
            if kind in files:
                e = files[kind][n]
                want = F.FP4Codes(F.unpack_nibbles(e["q"]), e["s"].view(torch.float8_e4m3fn).float(), e["g"],
                                  e["fmt"]).dequantize(torch.bfloat16)
                r[f"{kind}_decodes_to_file"] = bool(torch.equal(got.view(torch.int16), want.view(torch.int16))
                                                    if kind == "e2m1" else torch.equal(got, want))
        result["modules"][n] = r
    (Path(out_dir) / "result.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
