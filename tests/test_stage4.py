"""The port, stage 4: the paper's evaluation harnesses (``evaluation.downstream``, ``.decode_check``,
``.latency``).

CPU:
* the lm-eval record format: per-example correctness, per-choice log-likelihoods, sample digests and MMLU groups, on
  lm-eval-shaped results;
* the GEMM bench's module tagging (first / typical / worst) and its per-forward sums;
* the paper's defaults: tasks, few-shot, batch and the bench configurations.
GPU (an sm_120 GPU and the builds), on a small random Qwen3 model through the sm120 path:
* the prefill CUDA graph equals eager bit for bit, and the StaticCache decode graph's tokens equal eager decoding;
* the decode check calls every quantized Linear once per decode step;
* the isolated quantizer + GEMM launches equal NativeLinear's fused forward bit for bit, on every bench kernel;
* the latency-only activation-quantizer override is applied and recorded.
Checks against the reference's own harnesses are recorded in ``evaluation/port_validation/``: lm-eval per-example
log-likelihoods, the decode check, and latency spot checks.
"""
import hashlib
from collections import OrderedDict

import pytest
import torch

from evaluation import downstream as DS
from evaluation import latency as L


# --- CPU: the downstream record format -----------------------------------------------------------------------------

def _mc_sample(doc_id, acc, lls, prompt):
    return dict(doc_id=doc_id, acc=acc, acc_norm=acc, target=0, filtered_resps=[[ll, False] for ll in lls],
                doc_hash=f"d{doc_id}", prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(), target_hash="t")


def test_paper_defaults():
    assert DS.TASKS.split(",") == ["mmlu", "arc_challenge", "arc_easy", "hellaswag", "piqa"]
    assert DS.FEWSHOT == {"mmlu": 5} and DS.BATCH == {"qwen3.8-27b": 8}
    assert [DS.primary(t) for t in ("arc_easy", "arc_challenge", "hellaswag", "piqa", "mmlu", "mmlu_anatomy")] == \
        ["acc_norm"] * 4 + ["acc", "acc"]


def test_summarize_multiple_choice():
    res = dict(results={"arc_easy": {"acc,none": 0.5, "acc_norm,none": 0.5, "alias": "arc_easy"}})
    rows = [_mc_sample(0, 1.0, [-1.0, -2.0], "a"), _mc_sample(1, 0.0, [-3.0, -0.5], "b")]
    out = DS.summarize(res, {"arc_easy": rows})["arc_easy"]
    assert out["examples"] == {"0": 1.0, "1": 0.0}
    assert out["loglikelihoods"]["1"] == dict(target=0, ll=[-3.0, -0.5])
    assert "alias" not in out["metrics"] and out["metrics"]["acc_norm,none"] == 0.5
    # the digest depends on what was evaluated, not on the order the samples were logged in
    assert DS.summarize(res, {"arc_easy": rows[::-1]})["arc_easy"]["sample_digest"] == out["sample_digest"]
    other = DS.summarize(res, {"arc_easy": [rows[0], _mc_sample(1, 0.0, [-3.0, -0.5], "c")]})["arc_easy"]
    assert other["sample_digest"] != out["sample_digest"]


def test_summarize_group_mmlu():
    res = dict(results={"mmlu": {"acc,none": 0.5}, "mmlu_anatomy": {"acc,none": 1.0}, "mmlu_law": {"acc,none": 0.0}},
               samples={"mmlu_anatomy": [_mc_sample(0, 1.0, [-1, -2, -3, -4], "x")],
                        "mmlu_law": [_mc_sample(0, 0.0, [-4, -3, -2, -1], "y")]})
    out = DS.summarize_group(res, "mmlu")
    assert out["metrics"] == {"acc,none": 0.5}
    assert set(out["subtasks"]) == {"mmlu_anatomy", "mmlu_law"}
    assert out["examples"] == {"mmlu_anatomy:0": 1.0, "mmlu_law:0": 0.0}
    assert out["loglikelihoods"]["mmlu_law:0"]["ll"] == [-4, -3, -2, -1]


# --- CPU: the GEMM bench ---------------------------------------------------------------------------------------------

class _PW:
    def __init__(self, shape):
        self.shape = shape


def test_tag_modules_first_typical_worst():
    mods = OrderedDict((f"model.layers.{i}.mlp.down_proj", (_PW((64, 256)), t)) for i, t in enumerate([5, 1, 9, 3, 9]))
    t = L.tag_modules(mods, [16, 64])["down_proj"]
    assert t["modules"] == 5 and t["tiles_per_module"] == 4 * 4
    assert t["first"] == ("model.layers.0.mlp.down_proj", 5)
    assert t["typical"] == ("model.layers.0.mlp.down_proj", 5)       # the median of [1, 3, 5, 9, 9]
    assert t["worst"] == ("model.layers.2.mlp.down_proj", 9)         # the first maximum
    assert L.tag_modules(mods, None)["down_proj"]["tiles_per_module"] is None
    four = OrderedDict(list(mods.items())[:4])                       # [5, 1, 9, 3]: the lower median is 3
    assert L.tag_modules(four, [16, 64])["down_proj"]["typical"] == ("model.layers.3.mlp.down_proj", 3)


def test_warm_representatives_are_the_densest():
    mods = OrderedDict((f"model.layers.{i}.self_attn.o_proj", (_PW((64, 256)), n)) for i, n in enumerate([2, 7, 7, 0]))
    assert L.representatives(mods)["o_proj"] == ("model.layers.1.self_attn.o_proj", 7, 4, (64, 256))
    flat = OrderedDict((k, (pw, 0)) for k, (pw, _) in mods.items())     # E2M1 only: the first module
    assert L.representatives(flat)["o_proj"][0] == "model.layers.0.self_attn.o_proj"
    assert list(L.WARM_CONFIGS) == ["stock_wA", "stock_wA_nvfp4", "stock_wB", "mixed_16x64", "mixed_256x64", "n8k64_wB"]


def test_per_forward_sums():
    rows = [dict(config=c, proj=p, tokens=128, gemm_us=g, quant_us=q)
            for c, p, g, q in (("stock_wA", "q_proj", 10.0, 2.0), ("stock_wA", "o_proj", 20.0, 4.0),
                               ("n8k64_wB_worst", "q_proj", 11.0, 2.0), ("n8k64_wB_worst", "o_proj", 22.0, 4.0))]
    rec = dict(projections={"q_proj": dict(modules=2), "o_proj": dict(modules=2)}, rows=rows)
    reuse = {"q_proj": dict(calls=10, quant_reused=5), "o_proj": dict(calls=10, quant_reused=0)}
    out = L.per_forward(rec, reuse)
    assert out["per_forward_gemm_us"] == {"n8k64_wB_worst@128": 66.0, "stock_wA@128": 60.0}
    assert out["per_forward_quant_us"] == {"four_over_six_rows@128": 2 * 0.5 * 2.0 + 2 * 4.0}
    assert out["complete"]


def test_gemm_configurations_are_the_papers():
    assert list(L.CONFIGS) == ["stock_wA", "stock_wA_nvfp4", "stock_wB", "mixed_16x64_typical", "mixed_16x64_worst",
                               "mixed_256x64_typical", "mixed_256x64_worst", "n8k64_wB_typical", "n8k64_wB_worst"]
    assert L.classify("void cutlass::device_kernel<...>") == "gemm" and L.classify("quant_rows_kernel<1>") == "quant"


# --- GPU -------------------------------------------------------------------------------------------------------------

def _gpu():
    from tests.test_sm120_native import _sm120
    _sm120()


# the bench draws prompt ids in [100, 20000), so the small model gets a vocabulary that holds them
TINY_VOCAB = 20480


def _tiny(unit=None, weight="mixfp4", **kw):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from models import sm120 as NS
    from tests.test_sm120_native import SPEC, TINY, tiny_map
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**dict(TINY, vocab_size=TINY_VOCAB))).to(torch.bfloat16).cuda().eval()
    model.set_attn_implementation("sdpa")
    model.requires_grad_(False)
    tiles = tiny_map(model, unit) if weight == "mixfp4" else None
    rep = NS.install(model, SPEC, weight=weight, tiles=tiles, unit=unit if tiles is not None else None, **kw)
    return model, rep, NS.native_modules(model)


@pytest.mark.parametrize("unit,kernel", [("8x64", "paper"), ("16x64", "auto"), ("16x64", "paper")])
def test_prefill_graph_equals_eager(unit, kernel):
    _gpu()
    model, _, _ = _tiny(unit, kernel=kernel)
    out = L.prefill_graph(model, 2, 48, reps=3)
    assert out["logits_equal_eager"], out


def test_decode_graph_tokens_equal_eager():
    _gpu()
    _, BM = L.bench()
    model, _, _ = _tiny("8x64", kernel="paper")
    _, graph = BM.decode_graph(model, 2, 32, 16)
    _, eager = BM.decode_eager_tokens(model, 2, 32, graph.shape[1])
    assert torch.equal(graph, eager)


def test_decode_check_counts_every_linear():
    _gpu()
    from evaluation.decode_check import incremental_vs_full
    model, _, layers = _tiny("8x64", kernel="paper")
    prompts = torch.randint(100, TINY_VOCAB, (2, 32), generator=torch.Generator().manual_seed(3)).cuda()
    out = incremental_vs_full(model, prompts, 8, lambda m: sum(x.calls for x in layers.values()))
    assert out["native_calls_per_decode_step"] == [len(layers)] == [14]


@pytest.mark.parametrize("weight,unit,kernel", [("fourover6", None, "stock"), ("fourover6", None, "stock_wB"),
                                                ("nvfp4", None, "stock"), ("mixfp4", "16x64", "mixed"),
                                                ("mixfp4", "256x64", "mixed"), ("mixfp4", "8x64", "n8k64_wB")])
def test_isolated_launches_equal_native_linear(weight, unit, kernel):
    """The GEMM bench's registered check: the timed launches compute NativeLinear's output, bit for bit."""
    _gpu()
    _, _, layers = _tiny(unit, weight, kernel=kernel, act="nvfp4" if weight == "nvfp4" else "fourover6")
    stream = torch.cuda.current_stream()
    for name, lin in layers.items():
        lin.share_input = False
        for t in (1, 48, 200):
            x = torch.randn(t, lin.in_features, generator=torch.Generator().manual_seed(t)).to("cuda", torch.bfloat16)
            kern = lin.kernel_set.pick(lin.out_features, lin.in_features, t) if lin.kernel_set is not None else lin.kernel
            y = torch.empty((t, lin.out_features), dtype=torch.bfloat16, device="cuda")
            got = L.isolated_gemm(kern, lin, L.isolated_quant(kern, lin, x), lin.packed.clone(), lin.sf.clone(), y, stream)
            assert torch.equal(got.view(torch.int16), lin(x).view(torch.int16)), (name, t)


def test_act_override_is_applied_and_recorded():
    _gpu()
    _, rep, layers = _tiny("16x64", activation_quantizer="nvfp4_rows")
    assert rep["activation_quantizer"] == "nvfp4_rows" and rep["activation_quantizer_override"]
    assert {m.act_kind for m in layers.values()} == {"nvfp4_rows"}
    _, rep, layers = _tiny("16x64")
    assert rep["activation_quantizer"] == "four_over_six_rows" and not rep["activation_quantizer_override"]
    assert {m.act_kind for m in layers.values()} == {"four_over_six_rows"}
