#!/bin/bash
# Stage 4 validation (small checks, the user's reduced scope): NVFP4-RaZeR's own harnesses vs the port, Llama-3.1-8B,
# the paper kernels (8x64: n8k64_wB; FourOverSix: the paper stock set), back to back on an idle GPU.
set -u
V=/home/dev/n16k64_campaign/fqport/validation/stage4
WT=/home/dev/n16k64_campaign/fqport/wt
RZ=/home/dev/NVFP4-RaZeR
PY=/home/dev/.conda/envs/n16k64/bin/python
ART=/home/dev/n16k64_campaign/paper/artifacts
MAP8=/home/dev/n16k64_campaign/fqport/validation/tmopt_llama3.1-8b_8x64/flipquant_8x64.pt    # = the paper map, bitwise
MAP16=/home/dev/n16k64_campaign/paper/maps/llama8b_16x64/map.pt
DATA=/home/dev/n16k64_campaign/cost_comparison/data
log() { echo "$(date -u +%FT%TZ) $*" >> $V/commands.log; }
theirs() {   # name, accuracy|latency, command...
  name=$1; kind=$2; shift 2
  log "START $name (NVFP4-RaZeR)"
  if [ "$kind" = accuracy ]; then alloc="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"; else alloc="PYTORCH_CUDA_ALLOC_CONF="; fi
  (cd $RZ && env PYTHONPATH=$RZ/sm120:$RZ CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=0 $alloc "$@") > $V/$name.out 2>&1
  log "END $name rc=$?"
}
ours() {
  name=$1; kind=$2; shift 2
  log "START $name (port)"
  if [ "$kind" = accuracy ]; then alloc="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"; else alloc="PYTORCH_CUDA_ALLOC_CONF="; fi
  (cd $WT && env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=0 $alloc "$@") > $V/$name.out 2>&1
  log "END $name rc=$?"
}
OURS8="--model llama3.1-8b --mode native --native-backend razer --razer-kernel paper --weight mixfp4 --map $MAP8"
OURSFO6="--model llama3.1-8b --mode native --native-backend razer --razer-kernel paper --weight fourover6"
log "BEGIN stage 4 validation at port $(git -C $WT rev-parse --short HEAD)+wip, NVFP4-RaZeR $(git -C $RZ rev-parse --short HEAD)"

# 1. lm-eval, --limit 24 per task (MMLU: per subject), the paper's tasks and settings
theirs lmeval_razer accuracy $PY run_lmeval_deploy.py --model llama8b --data-root $DATA --transformers-deviation \
  --evaluate fo6=native:$ART/llama8b_fo6:paper_stock --evaluate ours-8x64=native:$ART/llama8b_tc_8x64:n8k64_wB \
  --tasks mmlu,arc_challenge,arc_easy,hellaswag,piqa --num-fewshot mmlu=5 --limit 24 --batch-size 16 --out $V/lmeval_razer
ours lmeval_port_fo6 accuracy $PY -m evaluation.downstream $OURSFO6 --limit 24 --out $V/lmeval_port_fo6.json
ours lmeval_port_8x64 accuracy $PY -m evaluation.downstream $OURS8 --limit 24 --out $V/lmeval_port_8x64.json

# 2. the decode-kernel verification
theirs decode_check_razer accuracy $PY sm120/eval/decode_check.py --model llama8b --artifact $ART/llama8b_tc_8x64 \
  --kernel n8k64_wB --out $V/decode_check_razer.json
ours decode_check_port accuracy $PY -m evaluation.decode_check $OURS8 --out $V/decode_check_port.json

# 3. latency smoke: prefill (eager + graph), decode (StaticCache graph), isolated GEMM (one projection)
theirs prefill_razer_8x64 latency $PY experiments/paper/bench_prefill.py --model llama8b --label ours-8x64 \
  --artifact $ART/llama8b_tc_8x64 --kernel n8k64_wB --round 1 --shapes 1x128,1x2048 --out $V/prefill_razer_8x64.json
ours prefill_port_8x64 latency $PY -m evaluation.latency prefill $OURS8 --shapes 1x128,1x2048 --out $V/prefill_port_8x64.json
theirs prefill_razer_fo6 latency $PY experiments/paper/bench_prefill.py --model llama8b --label fo6 \
  --artifact $ART/llama8b_fo6 --kernel paper_stock --round 1 --shapes 1x128,1x2048 --out $V/prefill_razer_fo6.json
ours prefill_port_fo6 latency $PY -m evaluation.latency prefill $OURSFO6 --shapes 1x128,1x2048 --out $V/prefill_port_fo6.json
theirs decode_razer_8x64 latency $PY experiments/paper_extra/bench_decode.py --model llama8b --label ours-8x64 \
  --artifact $ART/llama8b_tc_8x64 --kernel n8k64_wB --round 1 --settings 1x512,4x512 --gen 64 --out $V/decode_razer_8x64.json
ours decode_port_8x64 latency $PY -m evaluation.latency decode $OURS8 --settings 1x512,4x512 --gen 64 --out $V/decode_port_8x64.json
theirs gemm_razer latency $PY experiments/paper/bench_gemm_isolated.py --model llama8b --artifact fo6=$ART/llama8b_fo6 \
  --artifact nvfp4=$ART/llama8b_nvfp4 --artifact tc_8x64=$ART/llama8b_tc_8x64 --artifact tc_16x64=$ART/llama8b_tc_16x64 \
  --projections o_proj --tokens 128,2048 --rounds 1 --out $V/gemm_razer.json
ours gemm_port latency $PY -m evaluation.latency gemm --model llama3.1-8b --maps 8x64=$MAP8,16x64=$MAP16 \
  --projections o_proj --tokens 128,2048 --rounds 1 --quant-reuse $V/prefill_port_8x64.json --out $V/gemm_port.json
log "QUEUE DONE"
