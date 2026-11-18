"""GPTQ on the fixed RTN grid (flipquant/gptq.py) and FlipLinear.from_codes."""
import pytest
import torch
from torch import nn

from flipquant import formats as F
from flipquant import gptq, maps
from flipquant.linear import FlipLinear, QuantSpec
from flipquant.native import get_backend
from models.registry import ModelSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = ["nvfp4", "fourover6", "mixfp4"]


def _sm120(name="sm120-capi"):
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and get_backend(name)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _problem(o=64, k=256, t=512, seed=0):
    """A weight with an outlier and correlated inputs (so GPTQ has something to compensate)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(o, k, generator=g) * 0.05
    w[3, 7] = 0.8
    mix = torch.randn(k, k, generator=g) / k ** 0.5 + torch.eye(k)
    x = (torch.randn(t, k, generator=g) * torch.linspace(0.2, 2, k)) @ mix
    return w.bfloat16().to(DEV), x.bfloat16().to(DEV)


def _mask(o, k, unit="16x64", seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(maps.tile_shape(o, k, unit), generator=g) < 0.4


def _grid(w, weight, unit="16x64"):
    mask = _mask(*w.shape, unit) if weight == "mixfp4" else None
    grid, rtn = gptq.rtn_grid(w, weight, mask, unit)
    return grid, rtn, mask


@pytest.mark.parametrize("weight", WEIGHTS)
def test_rtn_grid_is_the_layers_rtn(weight):
    """The fixed grid's RTN codes are bitwise what FlipLinear.from_linear installs."""
    w, _ = _problem()
    grid, rtn, mask = _grid(w, weight)
    lin = nn.Linear(256, 64, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = w
    spec = QuantSpec(weight=weight, unit="16x64", mode="fake")
    ref = FlipLinear.from_linear(lin, spec, mask=mask)
    assert torch.equal(rtn.dequantize().view(torch.int16), ref.weight.view(torch.int16))
    # re-rounding the unmodified weight on the grid gives the RTN codes back, column by column
    codes = torch.cat([grid.quantize(w.float()[:, j:j + 1], j)[0] for j in range(w.shape[1])], 1)
    assert torch.equal(codes, rtn.codes)


@pytest.mark.parametrize("weight", WEIGHTS)
def test_gptq_keeps_the_grid_and_lowers_output_error(weight):
    w, x = _problem()
    grid, rtn, mask = _grid(w, weight)
    h = x.float().T @ x.float() / x.shape[0]
    q = gptq.gptq(w, h, grid)
    # fixed grid: scales, global scale and per-block formats are RTN's, bit for bit
    assert torch.equal(q.scale, rtn.scale) and torch.equal(q.gscale, rtn.gscale)
    if weight == "mixfp4":
        assert torch.equal(q.fmt, rtn.fmt)
    else:
        assert q.fmt == rtn.fmt == F.FMT_E2M1
    assert (q.codes != rtn.codes).float().mean() > 0.01            # it did re-round
    e_rtn, e_gptq = gptq.output_error(w, rtn, x), gptq.output_error(w, q, x)
    assert e_gptq < 0.9 * e_rtn, (e_gptq, e_rtn)


@pytest.mark.parametrize("weight", WEIGHTS)
def test_gptq_with_uncorrelated_inputs_is_rtn(weight):
    """Diagonal Hessian -> no error propagation -> exactly the RTN codes."""
    w, _ = _problem()
    grid, rtn, _ = _grid(w, weight)
    h = torch.diag(torch.rand(w.shape[1], device=DEV) + 0.5)
    assert torch.equal(gptq.gptq(w, h, grid).codes, rtn.codes)


def test_dead_inputs_are_kept_and_rounded_rtn():
    w, x = _problem()
    x[:, 10:20] = 0
    grid, rtn, _ = _grid(w, "fourover6")
    q = gptq.gptq(w, x.float().T @ x.float() / x.shape[0], grid)
    assert torch.isfinite(q.dequantize()).all()


def test_dead_weight_columns_are_modelopts():
    """ModelOpt's rule: an all-zero weight column gets a zero row / column in H (and H_ii = 1), so no error is
    propagated into it: it stays exactly zero (GPTQ would otherwise move it off zero)."""
    w, x = _problem(seed=5)
    w[:, 32:48] = 0
    grid, rtn, _ = _grid(w, "fourover6")
    q = gptq.gptq(w, x.float().T @ x.float() / x.shape[0], grid)
    assert torch.equal(q.dequantize(torch.float32)[:, 32:48], torch.zeros_like(w[:, 32:48], dtype=torch.float32))


def test_hessian_of_the_quantized_input():
    """hessian_input='quantized' (the default, as ModelOpt's input_quantizer in its Hessian hook): X is the
    activation quantizer's per-token output; None: the input as it is."""
    from flipquant import formats as F
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False, device=DEV, dtype=torch.bfloat16)
    layer = torch.nn.Sequential(lin)
    x = torch.randn(2, 8, 64, device=DEV, dtype=torch.bfloat16)
    h_q = gptq._hessians(layer, {"0": lin}, [x], [((), {})], "fourover6")["0"]
    h_b = gptq._hessians(layer, {"0": lin}, [x], [((), {})], None)["0"]
    xq = F.fake_quant(x, "fourover6", "row").reshape(-1, 64).float()
    xb = x.reshape(-1, 64).float()
    assert torch.allclose(h_q, xq.T @ xq / 16) and torch.allclose(h_b, xb.T @ xb / 16) and not torch.equal(h_q, h_b)


@pytest.mark.parametrize("weight", WEIGHTS)
def test_from_codes_fake_and_native_agree(weight):
    """GPTQ codes go into FlipLinear unchanged: fake and native (emulate, and the sm120 kernel when
    present) read the same effective weight, and their outputs agree."""
    w, x = _problem()
    grid, _, mask = _grid(w, weight)
    q = gptq.gptq(w, x.float().T @ x.float() / x.shape[0], grid)
    lin = nn.Linear(256, 64, bias=True, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = w
    backends = ["emulate"] + [b for b in ("sm120-capi", "sm120-capi-b8") if _sm120(b)]
    fake = FlipLinear.from_codes(lin, QuantSpec(weight=weight, unit="16x64", mode="fake"), q, mask=mask)
    assert torch.equal(fake.weight.view(torch.int16), q.dequantize().view(torch.int16))
    xin = x[:40].reshape(2, 20, 256)
    yf = fake(xin)
    for backend in backends:
        nat = FlipLinear.from_codes(lin, QuantSpec(weight=weight, unit="16x64", mode="native", native_backend=backend),
                                    q, mask=mask)
        assert torch.equal(nat.effective_weight().view(torch.int16), fake.weight.view(torch.int16)), backend
        yn = nat(xin)
        rel = ((yf.float() - yn.float()).norm() / yf.float().norm()).item()
        assert rel < 1e-2, (backend, rel)


def test_from_codes_rejects_inconsistent_formats():
    w, _ = _problem()
    lin = nn.Linear(256, 64, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.weight.data = w
    grid, rtn, mask = _grid(w, "mixfp4")
    with pytest.raises(ValueError):          # wrong map for these codes
        FlipLinear.from_codes(lin, QuantSpec(weight="mixfp4", unit="16x64"), rtn, mask=~mask)
    with pytest.raises(ValueError):          # mixed codes as a uniform format
        FlipLinear.from_codes(lin, QuantSpec(weight="fourover6"), rtn)
    with pytest.raises(ValueError):          # uniform codes, wrong uniform format
        FlipLinear.from_codes(lin, QuantSpec(weight="e0m3"), F.fourover6(w))


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    def forward(self, hidden_states, scale=None):
        return hidden_states + self.down(torch.relu(self.up(hidden_states))) * scale


class _Toy(nn.Module):
    """Decoder-layer list whose layers take an extra keyword (as hybrid layers take per-type masks)."""

    def __init__(self, d=128, n=3, vocab=50):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(_Block(d) for _ in range(n))
        self.head = nn.Linear(d, vocab, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, use_cache=False):
        h = self.emb(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, scale=torch.tensor(0.5 + i, device=h.device))
        return self.head(h)


@pytest.mark.parametrize("weight", ["fourover6", "mixfp4"])
def test_quantize_model_sequential(tmp_path, weight):
    torch.manual_seed(0)
    model = _Toy().to(DEV, torch.bfloat16).eval().requires_grad_(False)
    windows = [torch.randint(0, 50, (1, 32)) for _ in range(6)]
    ids = torch.cat(windows).to(DEV)
    ref = model(ids)
    names = [f"layers.{i}.{p}" for i in range(3) for p in ("up", "down")]
    tiles = {n: _mask(*model.get_submodule(n).weight.shape, seed=j) for j, n in enumerate(names)} \
        if weight == "mixfp4" else None
    qspec = QuantSpec(weight=weight, unit="16x64", act_scope="document", mode="fake")
    spec = ModelSpec("toy", "toy")
    cache = tmp_path / "codes.pt"
    layers, info = gptq.quantize_model(model, spec, qspec, windows, maps=tiles, batch=4, cache=cache, log=lambda *a, **k: None)
    assert sorted(layers) == sorted(names) and qspec.act_scope == "document"
    assert all(isinstance(model.get_submodule(n), FlipLinear) for n in names)
    assert info["mean_codes_changed"] > 0 and cache.exists()
    out = model(ids)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 0.2, rel
    # the cache reproduces the same layers without re-running GPTQ
    torch.manual_seed(0)                                                   # same BF16 weights as model
    model2 = _Toy().to(DEV, torch.bfloat16).eval().requires_grad_(False)
    layers2, info2 = gptq.quantize_model(model2, spec, qspec, windows, maps=tiles, cache=cache, log=lambda *a, **k: None)
    assert info2["loaded"] and torch.equal(model2(ids), out)
    with pytest.raises(SystemExit):     # a different setting does not silently reuse the cache
        gptq.quantize_model(_Toy().to(DEV, torch.bfloat16), spec, qspec, windows, maps=tiles, damp=0.1, cache=cache)


@pytest.mark.parametrize("weight", WEIGHTS)
def test_act_order_keeps_the_grid(weight):
    """act-order (static groups): scales, global scale, 4/6 choices and the map stay RTN's bit for bit;
    every column is rounded on its own group's grid; the error is still below RTN; and with a
    diagonal Hessian it is exactly RTN (no permutation bookkeeping error)."""
    w, x = _problem(seed=3)
    grid, rtn, _ = _grid(w, weight)
    h = x.float().T @ x.float() / x.shape[0]
    q = gptq.gptq(w, h, grid, act_order=True)
    assert torch.equal(q.scale, rtn.scale) and torch.equal(q.gscale, rtn.gscale)
    if weight == "mixfp4":
        assert torch.equal(q.fmt, rtn.fmt)
    plain = gptq.gptq(w, h, grid)
    assert not torch.equal(q.codes, plain.codes)                       # the order did change something
    assert gptq.output_error(w, q, x) < 0.9 * gptq.output_error(w, rtn, x)
    diag = torch.diag(torch.rand(w.shape[1], device=DEV) + 0.5)
    assert torch.equal(gptq.gptq(w, diag, grid, act_order=True).codes, rtn.codes)


def test_cache_key_is_unchanged_by_default_options():
    from flipquant.linear import QuantSpec as Q
    base = gptq.cache_key("m", Q(weight="fourover6"), None, "none", 0.01, 128, 128)
    assert gptq.cache_key("m", Q(weight="fourover6"), None, "none", 0.01, 128, 128, False, "mathcode") == base
    assert gptq.cache_key("m", Q(weight="fourover6"), None, "none", 0.01, 128, 128, True, "wiki") != base
    # the ModelOpt-aligned rules and the data are part of the key: no cache of the earlier rules is reused
    new = gptq.cache_key("m", Q(weight="fourover6"), None, "none", 0.01, 128, 128, False, "record", "quantized", "ab")
    assert new["rules"] == "modelopt-0.47" and new["data_sha256"] == "ab" and new != base
    assert new != gptq.cache_key("m", Q(weight="fourover6"), None, "none", 0.01, 128, 128, False, "record", "bf16", "ab")


@pytest.mark.parametrize("weight", ["nvfp4", "mixfp4"])
def test_propagate_bf16_codes_do_not_depend_on_the_activations(weight):
    """paper-sm120-runs: with propagate='bf16' (every activation quantizer off during calibration) the codes are the
    same whatever --act the model is evaluated with; with the default propagation they are not (the layers already
    done feed the next ones their quantized activations). The layers end with the evaluation's spec."""
    windows = [torch.randint(0, 50, (1, 32), generator=torch.Generator().manual_seed(i)) for i in range(6)]
    names = [f"layers.{i}.{p}" for i in range(3) for p in ("up", "down")]
    spec = ModelSpec("toy", "toy")
    digests = {}
    for propagate in ("bf16", "quantized"):
        for act in ("nvfp4", "fourover6"):
            torch.manual_seed(0)
            model = _Toy().to(DEV, torch.bfloat16).eval().requires_grad_(False)
            tiles = {n: _mask(*model.get_submodule(n).weight.shape, seed=j) for j, n in enumerate(names)} \
                if weight == "mixfp4" else None
            if weight == "nvfp4" and act == "fourover6":
                pass                                     # NVFP4 weights with FourOverSix activations (Table 4's GPTQ)
            qspec = QuantSpec(weight=weight, act=act, unit="16x64", act_scope="row", mode="fake")
            layers, info = gptq.quantize_model(model, spec, qspec, windows, maps=tiles, batch=3,
                                               hessian_input="bf16", propagate=propagate, log=lambda *a, **k: None)
            assert all(layers[n].spec is qspec for n in names)
            assert (info.get("propagate") == "bf16") == (propagate == "bf16")
            digests[propagate, act] = info["codes_sha256"]
    assert digests["bf16", "nvfp4"] == digests["bf16", "fourover6"]
    assert digests["quantized", "nvfp4"] != digests["quantized", "fourover6"]
    with pytest.raises(ValueError):                      # bf16 propagation needs the unquantized Hessian input
        gptq.quantize_model(_Toy().to(DEV, torch.bfloat16), spec, QuantSpec(weight="nvfp4", act="nvfp4", mode="fake"),
                            windows, hessian_input="quantized", propagate="bf16")
