"""--fake-weight-cache off on a real model (Qwen3-1.7B): logits, one batch's KL and every layer's map
gradient are bitwise those of the cached fake mode. The backward is made deterministic (deterministic
algorithms, eager attention); without that, two cached runs already differ by ~2% in map_grad."""

import os

# cuBLAS reads its workspace setting once, at its first use in the process: these bitwise tests turn on
# torch.use_deterministic_algorithms, so it is set here, when pytest imports the module, before any test runs
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from calibration.train_map_legacy import kl_backward, teacher_logprobs
from flipquant.linear import QuantSpec
from models import get, load, quantize


def _cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        s = get("qwen3-1.7b")
        return isinstance(try_to_load_from_cache(s.hf_id, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not (torch.cuda.is_available() and _cached()), reason="needs a GPU and Qwen3-1.7B in the HF cache")
def test_qwen3_cache_off_is_bitwise_cache_on():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        ids = torch.randint(0, 50000, (2, 128), generator=torch.Generator().manual_seed(0)).cuda()
        model, tok, spec = load("qwen3-1.7b", attn="eager")
        teacher = teacher_logprobs(model, [ids[:1], ids[1:]], "cuda")
        del model
        out = {}
        for cache in (True, False):
            model, tok, spec = load("qwen3-1.7b", attn="eager")
            q = QuantSpec(weight="mixfp4", act="fourover6", act_scope="row", unit="16x64", mode="fake",
                          fake_weight_cache=cache)
            layers = quantize(model, spec, q, trainable=True)
            g = torch.Generator().manual_seed(1)
            for l in layers.values():
                l.set_map(torch.rand(l.mask.shape, generator=g) < 0.1)
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits.float()
            for l in layers.values():
                l.map_grad = torch.zeros(l.mask.shape, device="cuda")
            kl = kl_backward(model, ids, torch.cat(teacher), 2)
            out[cache] = (logits.cpu(), kl, {n: l.map_grad.cpu() for n, l in layers.items()})
            del model, layers
            torch.cuda.empty_cache()
        assert torch.equal(out[True][0], out[False][0])
        assert out[True][1] == out[False][1]
        assert all(torch.equal(out[True][2][n], out[False][2][n]) for n in out[True][2])
    finally:
        torch.use_deterministic_algorithms(False)
