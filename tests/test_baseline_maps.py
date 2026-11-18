"""Baseline maps of the map-selection ablation (calibration/baseline_maps.py) and the paired-NLL tool."""
import json
import math

import pytest
import torch
from torch import nn

from calibration.baseline_maps import (act_scores, check_match, input_gram_blocks, oneshot_gradients, random_map,
                                      top_k)
from calibration.train_map_legacy import kl_per_sequence, teacher_logprobs
from evaluation import paired_nll
from flipquant import maps
from flipquant.linear import FlipLinear, QuantSpec
from models import quantize
from models.registry import ModelSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _match(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = {"a": (3, 4), "b": (5, 2), "c": (2, 6), "d": (4, 4)}
    out = {n: torch.rand(s, generator=g) < 0.3 for n, s in shapes.items()}
    out["c"][:] = False                                   # k = 0
    out["d"][:] = True                                    # k = all tiles
    return out


def test_random_map_matches_counts_and_is_seeded():
    match = _match()
    r0, r0b, r1 = random_map(match, 0), random_map(match, 0), random_map(match, 1)
    for n, m in match.items():
        assert r0[n].shape == m.shape and r0[n].dtype == torch.bool
        assert int(r0[n].sum()) == int(m.sum()) == int(r1[n].sum())
        assert torch.equal(r0[n], r0b[n])
    assert any(not torch.equal(r0[n], r1[n]) for n in match)


def test_check_match_rejects_partial_or_misshapen_maps():
    shapes = {"x": (40, 192), "y": (64, 128)}
    good = {"x": torch.zeros(3, 3, dtype=torch.bool), "y": torch.zeros(4, 2, dtype=torch.bool)}
    check_match(good, shapes, "16x64")
    with pytest.raises(SystemExit):          # a subset (e.g. a map trained on a vendor NVFP4 release)
        check_match({"x": good["x"]}, shapes, "16x64")
    with pytest.raises(SystemExit):          # wrong unit
        check_match(good, shapes, "8x64")


def test_top_k_exact_count_and_ties():
    s = torch.tensor([[0.5, 2.0, 2.0], [-1.0, 2.0, 0.0]])
    m = top_k(s, 2)
    assert m.tolist() == [[False, True, True], [False, False, False]]    # tie at 2.0 -> lower index
    assert int(top_k(s, 0).sum()) == 0 and bool(top_k(s, 6).all())


class _Toy(nn.Module):
    """``input_ids -> embedding -> linear``: enough of a model for the input hooks."""

    def __init__(self, k=192, o=40):
        super().__init__()
        self.emb = nn.Embedding(50, k, device=DEV, dtype=torch.bfloat16)
        self.emb.weight.data = (torch.randn(50, k, device=DEV) * torch.linspace(0.1, 3, k, device=DEV)).bfloat16()
        self.lin = nn.Linear(k, o, bias=False, device=DEV, dtype=torch.bfloat16)
        self.lin.weight.data = (torch.randn(o, k, device=DEV) * 0.05).bfloat16()

    def forward(self, input_ids, use_cache=False):
        return self.lin(self.emb(input_ids))


@pytest.mark.parametrize("unit", ["16x64", "8x64", "256x64"])
def test_act_scores_equal_brute_force_output_error(unit):
    torch.manual_seed(0)
    model = _Toy()
    windows = [torch.randint(0, 50, (1, 33)) for _ in range(5)]
    rows, cols = maps.parse_unit(unit)
    grams = input_gram_blocks(model, {"lin": model.lin}, windows, DEV, 2, cols)
    s = act_scores(model.lin, grams["lin"], unit)

    x = model.emb(torch.cat(windows).to(DEV)).reshape(-1, 192).double()        # BF16 inputs, all tokens
    layer = FlipLinear.from_linear(model.lin, QuantSpec(unit=unit), trainable=True)
    w = model.lin.weight.double()
    e_a, e_b = w - layer._decoded("a").double(), w - layer._decoded("b").double()
    brute = torch.zeros(maps.tile_shape(40, 192, unit), dtype=torch.float64)
    for i in range(brute.shape[0]):
        for j in range(brute.shape[1]):
            p = torch.zeros_like(w)
            p[i * rows:(i + 1) * rows, j * cols:(j + 1) * cols] = 1
            brute[i, j] = ((p * e_b) @ x.T).square().sum() - ((p * e_a) @ x.T).square().sum()
    assert torch.allclose(s.double().cpu(), brute, rtol=1e-4, atol=1e-6 * brute.abs().max().item())


def _tiny_qwen3(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=96, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=64, max_position_embeddings=64,
                      initializer_range=0.2)
    return Qwen3ForCausalLM(cfg).to(DEV, torch.float32).eval().requires_grad_(False)


def test_oneshot_gradient_is_the_kl_derivative_and_selection_takes_most_negative():
    """With only the last linear quantized (nothing quantized downstream of it, so the KL is smooth
    in its weights), the accumulated map gradient equals the finite-difference derivative of the
    mean fit KL along ``W = B + p (A - B)`` at ``p = 0`` (the all-E2M1 start), tile by tile."""
    model = _tiny_qwen3()
    windows = [torch.randint(0, 96, (1, 24), generator=torch.Generator().manual_seed(i)) for i in range(6)]
    teacher = teacher_logprobs(model, windows, DEV)
    name = "model.layers.1.mlp.down_proj"
    qspec = QuantSpec(weight="mixfp4", act="fourover6", act_scope="row", unit="16x64", mode="fake")
    layers = quantize(model, ModelSpec("toy", "toy"), qspec, trainable=True, names=[name])
    layer = layers[name]
    grads, kls = oneshot_gradients(model, layers, windows, teacher, DEV, batch=4)
    assert layer.map_grad is None and not layer.mask.any()               # no step, map untouched

    @torch.no_grad()
    def mean_kl(p):
        layer.set_soft(p)
        ids = torch.cat(windows).to(DEV)
        lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
        return kl_per_sequence(lp, torch.cat(teacher).to(DEV).float()).mean().item()
    base = mean_kl(torch.zeros_like(grads[name]))
    assert math.isclose(base, sum(kls) / len(kls), rel_tol=1e-4)
    eps = 1e-2
    g = grads[name].cpu()
    for idx in [(0, 0), (1, 2), (3, 3), (7, 1)]:
        p = torch.zeros_like(grads[name])
        p[idx] = eps
        fd = (mean_kl(p) - base) / eps
        assert math.isclose(fd, g[idx].item(), rel_tol=3e-2, abs_tol=1e-2 * g.abs().max().item()), (idx, fd, g[idx])

    k = 5
    sel = top_k(-g, k)
    order = torch.argsort(g.flatten())[:k]
    assert int(sel.sum()) == k and set(sel.flatten().nonzero().flatten().tolist()) == set(order.tolist())


@pytest.mark.parametrize("mode", ["fake", "native"])
def test_saved_map_loads_and_drives_flip_linear(tmp_path, mode):
    torch.manual_seed(0)
    lin = nn.Linear(192, 48, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = (torch.randn(48, 192, device=DEV) * 0.05).bfloat16()
    match = {"lin": torch.rand(maps.tile_shape(48, 192, "16x64")) < 0.4}
    tiles = random_map(match, 3)
    path = tmp_path / "random.pt"
    maps.save(path, tiles, "16x64", dict(method="random", seed=3))
    loaded, unit, meta = maps.load(path)
    assert unit == "16x64" and meta["method"] == "random" and torch.equal(loaded["lin"], tiles["lin"])
    layer = FlipLinear.from_linear(lin, QuantSpec(unit=unit, mode=mode, native_backend="emulate"), mask=loaded["lin"])
    ref = FlipLinear.from_linear(lin, QuantSpec(unit=unit), trainable=True)
    expect = torch.where(maps.expand(loaded["lin"].to(DEV), 16, 64, 48), ref._decoded("a"), ref._decoded("b"))
    assert torch.equal(layer.effective_weight().view(torch.int16), expect.view(torch.int16))


def _report(nll, model="m", mode="native", data=None, limit="None"):
    data = data or {"revision": "r", "windows": 10}
    return dict(model=model, source=dict(hf_id="x", revision="y"), mode=mode,
                quant=dict(mode=mode, act="fourover6", act_scope="document"), native_backend="sm120",
                args=dict(seqlen="2048", limit=limit, use_cache="protocol"),
                results={"wiki": dict(nll=nll, ppl=math.exp(sum(nll) / len(nll)), windows=len(nll),
                                      use_cache=True, data=data)})


def test_paired_nll_mean_se_and_seed_average(tmp_path):
    base = [2.0, 2.1, 1.9, 2.05]
    runs = [[2.01, 2.12, 1.905, 2.06], [2.02, 2.11, 1.91, 2.04], [2.0, 2.13, 1.92, 2.07]]
    paths = []
    for i, nll in enumerate([base] + runs):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(_report(nll)))
        paths.append(str(p))
    rep = paired_nll.main(["--base", paths[0], "--runs", *paths[1:], "--label", "random"])
    r = rep["results"]["wiki"]
    dbar = [sum(run[i] - base[i] for run in runs) / 3 for i in range(4)]
    m = sum(dbar) / 4
    se = math.sqrt(sum((d - m) ** 2 for d in dbar) / 3 / 4)
    assert math.isclose(r["delta_nll_milli"], m * 1e3) and math.isclose(r["two_se_milli"], 2 * se * 1e3)
    assert len(r["per_run"]) == 3 and "seed_sd_milli" in r

    bad = tmp_path / "bad.json"
    for kwargs in (dict(limit="2"), dict(data={"revision": "other", "windows": 10}), dict(mode="fake"),
                   dict(model="other")):
        bad.write_text(json.dumps(_report(runs[0], **kwargs)))
        with pytest.raises(SystemExit):
            paired_nll.main(["--base", paths[0], "--runs", str(bad)])
    bad.write_text(json.dumps(_report(runs[0][:3])))
    with pytest.raises(SystemExit):
        paired_nll.main(["--base", paths[0], "--runs", str(bad)])
