"""The sm120 deployment path: the vendored kernels (kernels/sm120) and their adapter (models/sm120.py).

* test_sass_matches_reference (CPU): every build in the build directory ($SM120_BUILD_DIR, else the deployment
  directory kernels/sm120/build_deploy) has exactly the patched and unpatched SASS of the reference's gated build of
  that configuration with the same defines (SASS_REFERENCE.json).
* test_auto_refuses_unoptimized_builds (CPU): 'auto' raises, naming build_deploy.sh, on a build directory without the
  optimized sets and on builds of them without their dispatch variants; it never falls back.
* GPU tests (an sm_120 GPU and the builds), on a small random Qwen3 model:
  - the adapter's in-memory install equals the vendored artifact path (export -> install) bit for bit, logits of the
    whole model, for 8x64 / 16x64 / 256x64 maps and the FourOverSix / NVFP4 baselines, on the 'auto' and 'paper' sets
    (and 'auto_stock_wB' for the baselines);
  - a CUDA-graph replay of the installed model equals its eager forward bit for bit, for every unit and baseline;
  - the ownership check passes for every module of 8x64 and 16x64 maps;
  - with $REFERENCE_ROOT (a checkout of the reference implementation) and $REFERENCE_BUILD (its paper builds;
    $REFERENCE_BUILD_DEPLOY its deployment directory for the optimized sets, default $REFERENCE_BUILD), the same model,
    map and inputs through the reference's own package and builds, in a separate process, give the same logits bit
    for bit.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flipquant import maps as M
from models import sm120 as NS
from models.registry import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
BUILD = NS.build_dir()
_REF = json.loads((ROOT / "kernels" / "sm120" / "SASS_REFERENCE.json").read_text())
REFERENCE = _REF["builds"]
REFERENCE_DEFINED = {(e["config"], json.dumps(e["extra_defines"], sort_keys=True)): e for e in _REF["builds_defined"]}
SPEC = ModelSpec("tiny-qwen3", "tiny-qwen3")


def _built():
    return sorted(p.parent.name for p in BUILD.glob("*/manifest.json")) if BUILD.is_dir() else []


def test_sass_matches_reference():
    built = _built()
    if not built:
        pytest.skip(f"no builds in {BUILD}")
    bad = []
    for name in built:
        man = json.loads((BUILD / name / "manifest.json").read_text())
        defines = man.get("extra_defines") or {}
        ref = REFERENCE_DEFINED.get((name, json.dumps(defines, sort_keys=True))) if defines else REFERENCE.get(name)
        if ref is None:
            bad.append(f"{name}: no reference for defines {defines}")
        elif (man["sass_sha256"], man["unpatched_sass_sha256"]) != (ref["sass_sha256"], ref["unpatched_sass_sha256"]):
            bad.append(f"{name}: SASS differs from the reference's {ref['source']}")
    assert not bad, bad


def test_auto_refuses_unoptimized_builds(tmp_path):
    """'auto' never falls back: an empty directory, a paper-only directory and the optimized sets built without their
    dispatch variants are all errors that name build_deploy.sh."""
    if not all((BUILD / n / "manifest.json").exists() for n in ("n16k64_wA_n16_t0", "n16k64_wA", "stock_wA")):
        pytest.skip(f"the deployment builds are not in {BUILD}")
    empty = tmp_path / "empty"
    empty.mkdir()
    paper = tmp_path / "paper"
    for name in ("n16k64_wA", "n16k64_wA_n16", "n16k64_wA_n32", "n16k64_wA_n64", "stock_wA", "stock_wA_n16",
                 "stock_wA_n32", "stock_wA_n64", "n8k64_wB", "stock_wB"):
        (paper / name).mkdir(parents=True)
        for f in (BUILD / name).iterdir():
            if f.is_file():
                (paper / name / f.name).symlink_to(f)
    plain = tmp_path / "plain"                          # mixed_ko's builds, their manifests without the defines
    for name in ("n16k64_wA_n16_t0", "n16k64_wA_n32_t0", "n16k64_wA_n64_t0", "n16k64_wA_e64_t0"):
        (plain / name).mkdir(parents=True)
        for f in (BUILD / name).iterdir():
            if f.name == "manifest.json":
                man = json.loads(f.read_text())
                man["extra_defines"] = {}
                (plain / name / f.name).write_text(json.dumps(man))
            elif f.is_file():
                (plain / name / f.name).symlink_to(f)
    for root, unit, why in ((empty, "16x64", "is not built"), (paper, "16x64", "is not built"),
                            (paper, None, "is not built"), (plain, "16x64", "optimized dispatch variants")):
        weight = "mixfp4" if unit else "fourover6"
        with pytest.raises(NS.KernelSetError, match=why) as e:
            NS.resolve_kernel("auto", weight, unit, root)
        assert "build_deploy.sh" in str(e.value)
    kern, label, note = NS.resolve_kernel("paper", "mixfp4", "16x64", paper)   # the paper's kernels still run there
    assert label == "mixed" and "paper" in note


# --- GPU ------------------------------------------------------------------------------------------

def _sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("needs an sm_120 GPU")
    for need in ("n16k64_wA", "n8k64_wB", "stock_wA"):
        if need not in _built():
            pytest.skip(f"{need} is not built in {BUILD}")


TINY = dict(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=64, max_position_embeddings=512, tie_word_embeddings=False)


def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.bfloat16).cuda().eval()
    model.set_attn_implementation("sdpa")
    return model.requires_grad_(False)


def tiny_map(model, unit, seed=1, p=0.3):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.rand(M.tile_shape(m.out_features, m.in_features, unit), generator=g) < p
            for n, m in NS.quantizable_linears(model, SPEC).items()}


def tiny_ids(seed=2):
    return torch.randint(0, TINY["vocab_size"], (2, 48), generator=torch.Generator().manual_seed(seed)).cuda()


CASES = [("mixfp4", "8x64"), ("mixfp4", "16x64"), ("mixfp4", "256x64"), ("fourover6", None), ("nvfp4", None)]
KERNEL_CASES = ([(w, u, k) for w, u in CASES for k in ("auto", "paper")]
                + [(w, u, "auto_stock_wB") for w, u in CASES if w != "mixfp4"])


@pytest.mark.parametrize("weight,unit,kernel", KERNEL_CASES)
def test_install_equals_artifact_path(tmp_path, weight, unit, kernel):
    _sm120()
    R = NS.package()
    from mixfp4_sm120 import artifact as A, mapio
    from mixfp4_sm120 import model as RM
    ids = tiny_ids()
    a, b = tiny_model(), tiny_model()
    tiles = tiny_map(a, unit) if weight == "mixfp4" else None
    act = "nvfp4" if weight == "nvfp4" else "fourover6"
    try:
        kern, _, _ = NS.resolve_kernel(kernel, weight, unit, None)
    except (R.lib.LibraryError, NS.KernelSetError) as e:
        pytest.skip(f"set not built: {e}")
    # the vendored artifact path: export (the packing check) -> install, on the same kernel set
    lin_b = NS.quantizable_linears(b, SPEC)
    kind = NS.KIND[weight]
    map_path = None
    if weight == "mixfp4":
        packed = {n: NS._packing(tiles[n].cuda(), unit, l.out_features) for n, l in lin_b.items()}
        tb = next(iter(packed.values()))[1]
        header, masks = mapio.legacy_masks_to_map({n: m for n, (m, _) in packed.items()}, type_block=tb,
                                                  weight_shapes={n: list(l.weight.shape) for n, l in lin_b.items()},
                                                  model=dict(model_id="tiny-qwen3"), policy="test")
        map_path = tmp_path / "test.mixfp4map"
        mapio.write_map(map_path, header, masks)
    A.export(tmp_path / "art", lin_b, kind=kind, map_path=map_path, verify_fake=False)
    RM.install(b, tmp_path / "art", kernel=kern, loader="causal_lm")
    rep = NS.install(a, SPEC, weight=weight, tiles=tiles, unit=unit, kernel=kernel, act=act)
    assert rep["modules"] == len(lin_b) == 14
    NS.reset_counters(a)
    with torch.no_grad():
        la, lb = a(input_ids=ids).logits, b(input_ids=ids).logits
    assert NS.coverage(a)["native_called"] == 14
    assert torch.equal(la.view(torch.int16), lb.view(torch.int16))


@pytest.mark.parametrize("weight,unit,kernel", [(w, u, "auto") for w, u in CASES] + [("fourover6", None, "auto_stock_wB")])
def test_cuda_graph_replay_equals_eager(weight, unit, kernel):
    _sm120()
    model = tiny_model()
    tiles = tiny_map(model, unit) if weight == "mixfp4" else None
    try:
        NS.install(model, SPEC, weight=weight, tiles=tiles, unit=unit, kernel=kernel,
                   act="nvfp4" if weight == "nvfp4" else "fourover6")
    except (NS.package().lib.LibraryError, NS.KernelSetError) as e:
        pytest.skip(f"set not built: {e}")
    ids = tiny_ids()
    with torch.no_grad():
        eager = model(input_ids=ids, use_cache=False).logits.clone()
        static = ids.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model(input_ids=static, use_cache=False)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = model(input_ids=static, use_cache=False).logits
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("unit", ["8x64", "16x64"])
def test_ownership(unit):
    _sm120()
    model = tiny_model()
    rep = NS.install(model, SPEC, weight="mixfp4", tiles=tiny_map(model, unit), unit=unit, ownership=True)
    assert len(rep["ownership"]) == 14
    assert all(o["exact"] and o["format_mismatches"] == 0 for o in rep["ownership"].values())


ORIGINAL = """
import json, sys, torch
sys.path[:0] = [{sm120!r}, {root!r}]
from transformers import Qwen3Config, Qwen3ForCausalLM
from mixfp4_sm120 import artifact as A, mapio, model as RM
from mixfp4_sm120.select import KernelSet
from mixfp4_sm120.lib import Kernel
torch.manual_seed(0)
model = Qwen3ForCausalLM(Qwen3Config(**{tiny!r})).to(torch.bfloat16).cuda().eval()
model.set_attn_implementation("sdpa")
model.requires_grad_(False)
lin = {{n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}}
kind, kernel, map_path, out = {kind!r}, {kernel!r}, {map_path!r}, {out!r}
A.export(out + "_art", lin, kind=kind, map_path=map_path or None, verify_fake=True)
kern = KernelSet(kernel) if kernel in __import__("mixfp4_sm120.select", fromlist=["FAMILIES"]).FAMILIES else Kernel.load(kernel)
RM.install(model, out + "_art", kernel=kern)
ids = torch.load({ids!r}).cuda()
with torch.no_grad():
    torch.save(model(input_ids=ids).logits.cpu(), out)
"""


# the reference's own set for each of the adapter's routes (its mixfp4_sm120.model.resolve_kernel at kernel-opt ae389a5)
ORIGINAL_SET = {"paper": {"8x64": "n8k64_wB", "16x64": "mixed", "256x64": "mixed", None: "stock"},
                "auto": {"8x64": "mixed_wB_ko", "16x64": "mixed_ko", "256x64": "mixed256_ko", None: "stock_ko"},
                "auto_stock_wB": {None: "stock_wB_ko"}}


@pytest.mark.skipif(not (os.environ.get("REFERENCE_ROOT") and os.environ.get("REFERENCE_BUILD")),
                    reason="needs $REFERENCE_ROOT (a reference checkout) and $REFERENCE_BUILD (its builds)")
@pytest.mark.parametrize("weight,unit,kernel", [(w, u, k) for w, u in CASES[:4] for k in ("paper", "auto")]
                         + [("fourover6", None, "auto_stock_wB")])
def test_logits_equal_reference(tmp_path, weight, unit, kernel):
    """The same model, map and inputs through the reference's own package and builds (another process)."""
    _sm120()
    NS.package()                                        # the vendored package on the path, whatever ran before
    from mixfp4_sm120 import mapio
    model = tiny_model()
    tiles = tiny_map(model, unit) if weight == "mixfp4" else None
    original = ORIGINAL_SET[kernel][unit]
    map_path = ""
    if tiles is not None:
        # the map as the reference's artifacts store it (a 256x64 map as its 16x64 granules)
        lin = NS.quantizable_linears(model, SPEC)
        packed = {n: NS._packing(tiles[n], unit, l.out_features) for n, l in lin.items()}
        header, masks = mapio.legacy_masks_to_map({n: m for n, (m, _) in packed.items()},
                                                  type_block=next(iter(packed.values()))[1],
                                                  weight_shapes={n: list(l.weight.shape) for n, l in lin.items()},
                                                  model=dict(model_id="tiny-qwen3"), policy="test")
        map_path = str(tmp_path / "test.mixfp4map")
        mapio.write_map(map_path, header, masks)
    ids = tiny_ids()
    torch.save(ids.cpu(), tmp_path / "ids.pt")
    out = str(tmp_path / "original_logits.pt")
    root = os.environ["REFERENCE_ROOT"]
    script = ORIGINAL.format(sm120=str(Path(root) / "sm120"), root=root, tiny=TINY, kind=NS.KIND[weight],
                             kernel=original, map_path=map_path, out=out, ids=str(tmp_path / "ids.pt"))
    build = (os.environ["REFERENCE_BUILD"] if kernel == "paper"
             else os.environ.get("REFERENCE_BUILD_DEPLOY", os.environ["REFERENCE_BUILD"]))
    env = dict(os.environ, SM120_BUILD_DIR=build, PYTHONPATH="")
    subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=root)
    NS.install(model, SPEC, weight=weight, tiles=tiles, unit=unit, kernel=kernel)
    with torch.no_grad():
        ours = model(input_ids=ids).logits.cpu()
    theirs = torch.load(out)
    assert torch.equal(ours.view(torch.int16), theirs.view(torch.int16))
