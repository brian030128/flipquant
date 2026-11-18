# Stage 4 validation: the port against NVFP4-RaZeR's own harnesses

**Setup.** One RTX PRO 6000 session, 2026-10-02 05:11–05:27 UTC: the port at razer-port 8a279c7 plus this stage's
changes, and NVFP4-RaZeR at `kernel-opt` a84ab6f.
- **Model:** Llama-3.1-8B.
- **Kernels:** the paper's (8x64: n8k64_wB; FourOverSix: the paper stock set).
- **Maps:**
  - 8x64: stage 3's reproduction of the paper map, bitwise equal to it;
  - 16x64: the paper's `map.pt`, read directly (`--maps 16x64=...`).
- **NVFP4-RaZeR's side** used its exported paper artifacts of the same maps.
- **Commands:** `queue.sh`, then the warm-GEMM pair below; `commands.log` has the timeline.

These are small checks, as the user scoped stage 4. They are not full evaluations. `comparison.json` holds every
comparison.

| check | NVFP4-RaZeR harness | result |
|---|---|---|
| lm-eval, `--limit 24`, FourOverSix and FlipQuant 8x64 | `run_lmeval_deploy.py` | Equal per-choice log-likelihoods, per-example correctness, sample digests and metrics, bit for bit, on every task. Every quantized Linear ran natively in every forward. |
| decode check, 8x64 | `sm120/eval/decode_check.py` | BF16 and native metrics identical, the same generated text, 224 native calls per decode step |
| prefill 1x128 / 1x2048, 8x64 and FourOverSix | `experiments/paper/bench_prefill.py` | CUDA-graph logits equal eager bit for bit on both sides; graph times within 0.7 %; quantizer-reuse counts identical |
| decode 1x512 / 4x512, 8x64 | `experiments/paper_extra/bench_decode.py` | graph tokens equal eager StaticCache decoding (33 of 33) on both sides; ms per token within 0.2 % |
| isolated cold GEMM, `o_proj`, T = 128 / 2048, 7 configurations | `experiments/paper/bench_gemm_isolated.py` | Isolated launches equal NativeLinear bit for bit. The first, typical and worst modules and their E0M3 counts are identical for every projection and weight kind. GEMM times 0.98–1.01×. |
| warm GEMM, the same | `experiments/paper/bench_gemm.py` | the representative modules are identical (28); times 0.98–1.04× (a single warm round, the order-sensitive method) |

**lm-eval subset.**
- MMLU 5-shot: 24 questions per subject, 1,368 in all.
- ARC-Challenge, ARC-Easy, HellaSwag and PIQA: 24 each, 0-shot.
- Batch 16.
- 5,808 per-choice log-likelihoods per policy.

**The decode check's native gap.** It is much larger than BF16's: top-1 agreement 0.895 vs 1.0, mean KL 0.061 vs
0.0005. NVFP4-RaZeR's harness shows exactly the same gap. Its `sm120/RESULTS.md` §4.3 attributes it to W4A4, since
the fake-quant model shows the same order of gap; its archived Llama run measured top-1 0.875 and mean KL 0.054.

Warm-GEMM pair (after the queue):

```bash
(cd NVFP4-RaZeR && PYTHONPATH=sm120:. python experiments/paper/bench_gemm.py --model llama8b --artifact fo6=ART/llama8b_fo6 \
   --artifact nvfp4=ART/llama8b_nvfp4 --artifact tc_8x64=ART/llama8b_tc_8x64 --artifact tc_16x64=ART/llama8b_tc_16x64 \
   --projections o_proj --tokens 128,2048 --rounds 1 --out gemm_warm_reference.json)
python -m evaluation.latency gemm-warm --model llama3.1-8b --maps 8x64=MAP8,16x64=MAP16 --projections o_proj \
   --tokens 128,2048 --rounds 1 --out gemm_warm_port.json
```

The lm-eval reports are gzipped (`*.json.gz`); NVFP4-RaZeR's holds both policies.
