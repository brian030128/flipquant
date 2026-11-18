"""The decode-kernel verification of the FlipQuant paper's native path (the vendored ``kernels/sm120/eval/
decode_check.py``, ported onto ``models.cli``): prefill and token-by-token decode with the KV cache, against one full
forward.

    python -m evaluation.decode_check --model llama3.1-8b --mode native \\
        --weight mixfp4 --map maps/llama3.1-8b/flipquant_8x64.pt --out results/llama3.1-8b/decode_check_8x64.json

**What it does.** For a batch of real prompts (the first ``--batch`` WikiText-2 windows, cut to ``--prompt`` tokens),
it greedy-generates ``--gen`` tokens with a DynamicCache: the prompt's prefill, then one token per forward.
- **The comparison.** At every generated position, the incremental decode's logits are compared with those of ONE full
  forward over prompt + generated tokens, without a cache. Recorded: the max and mean absolute logit difference, top-1
  agreement, and the mean and max KL.
- **Why a gap would come from attention.** Both runs execute the same native Linears on the same token rows, and the
  native path is batch-invariant (a token's output does not depend on the other tokens in the GEMM;
  kernels/sm120/tests/test_select.py). So any gap comes from attention over the cache.
- **The reference size.** The same comparison on the BF16 model, before it is quantized, gives the size of that gap.

**Also recorded:**
- the native GEMM calls per decode step: every quantized Linear, once per step;
- that ``generate()`` runs end to end (16 tokens, greedy).

**Passes when:**
- every decode step calls every quantized Linear once;
- ``generate()`` produces its 16 tokens.
The logit gap is reported against the BF16 gap and is not thresholded, as in the paper's check.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from flipquant.data import eval_windows
from models.cli import add_model_args, apply_quant, save_json
from models.loader import load


@torch.no_grad()
def incremental_vs_full(model, prompts, gen, calls_of=None):
    """The paper's decode_check.incremental_vs_full. ``calls_of(model)``: the native GEMM calls so far (sm120 path)."""
    from transformers import DynamicCache
    cache = DynamicCache()
    out = model(input_ids=prompts, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    inc_logits, toks = [out.logits[:, -1].float()], [nxt]
    calls = []
    for _ in range(gen - 1):
        before = calls_of(model) if calls_of else None
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        if calls_of:
            calls.append(calls_of(model) - before)
        nxt = out.logits[:, -1:].argmax(-1)
        inc_logits.append(out.logits[:, -1].float())
        toks.append(nxt)
    seq = torch.cat([prompts] + toks[:-1], dim=1)
    full = model(input_ids=seq, use_cache=False).logits[:, prompts.shape[1] - 1:].float()
    inc = torch.stack(inc_logits, dim=1)
    diff = (inc - full).abs()
    agree = (inc.argmax(-1) == full.argmax(-1)).double().mean()
    kl = (full.log_softmax(-1).exp() * (full.log_softmax(-1) - inc.log_softmax(-1))).sum(-1)
    return dict(max_abs_logit_diff=float(diff.max()), mean_abs_logit_diff=float(diff.mean()),
                top1_agreement=float(agree), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                native_calls_per_decode_step=sorted(set(calls)) if calls else None,
                generated_tokens=int(gen * prompts.shape[0]),
                generated_sha256=hashlib.sha256(torch.cat(toks, 1).cpu().numpy().tobytes()).hexdigest())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, act_scope="row")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.mode == "bf16":
        raise SystemExit("the check compares a quantized model with BF16: use --mode fake or native")
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    model, tok, spec = load(args.model, revision=args.revision, device_map=args.device_map, attn=args.attn)
    wins, _ = eval_windows(tok, "wiki")
    prompts = torch.cat([w[:, :args.prompt] for w in wins[:args.batch]]).to(model.get_input_embeddings().weight.device)
    res = dict(model=spec.key, batch=args.batch, prompt=args.prompt, gen=args.gen,
               prompt_sha256=hashlib.sha256(prompts.cpu().numpy().tobytes()).hexdigest())
    res["bf16"] = incremental_vs_full(model, prompts, args.gen)
    print("bf16", res["bf16"], flush=True)
    layers, info = apply_quant(model, spec, args)
    res["policy"] = info
    native = info.get("native_backend") == "sm120"
    calls_of = (lambda m: sum(x.calls for x in layers.values())) if native else None
    res["quantized"] = incremental_vs_full(model, prompts, args.gen, calls_of)
    print("quantized", res["quantized"], flush=True)
    out = model.generate(prompts[:1], max_new_tokens=16, do_sample=False)
    res["generate_ok"] = bool(out.shape[1] == args.prompt + 16)
    res["generated_text"] = tok.decode(out[0, args.prompt:])
    steps = res["quantized"]["native_calls_per_decode_step"]
    res["every_linear_every_step"] = steps == [len(layers)] if native else None
    res["passed"] = res["generate_ok"] and (res["every_linear_every_step"] is not False)
    res["seconds"] = time.time() - t0
    save_json(args.out, res)
    print("PASSED" if res["passed"] else "FAILED", flush=True)
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
