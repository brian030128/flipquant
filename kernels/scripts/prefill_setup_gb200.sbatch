#!/bin/bash
# HackMD #2, step 1 on one GB200 GPU: build libflipquant_sm100_linear.so, check the tuned SM100 native path
# (kernels/scripts/test_sm100_linear.py), then smoke-test evaluation.latency prefill on Qwen3-1.7B for the four methods.
#   sbatch kernels/scripts/prefill_setup_gb200.sbatch     # from the repository root
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -J fq-prefill-setup
#SBATCH -o kernels/docs/results/sm100_paper/02_prefill_latency/setup_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT=kernels/docs/results/sm100_paper/02_prefill_latency
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
SRC="$PWD/kernels" JOBS=12 TARGETS=flipquant_sm100_linear bash kernels/scripts/build_sm100.sh 2>&1 \
  | grep -E 'error|Error|built into' | head -40
export FLIPQUANT_SM100_LINEAR_LIB="$BUILD/libflipquant_sm100_linear.so"
ls -l "$FLIPQUANT_SM100_LINEAR_LIB"
source kernels/scripts/sm100_pyenv.sh
export PYTHONPATH="$PWD"
"$PYTHON" kernels/scripts/test_sm100_linear.py "$OUT/test_sm100_linear_$SLURM_JOB_ID.json"
MAP=$("$PYTHON" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('edgeai-lab/Qwen3-1.7B-FlipQuant', 'flipquant_256x64.pt'))")
for method in bf16 nvfp4 fourover6 flipquant; do
  case $method in
    bf16)      args=(--mode bf16) ;;
    nvfp4)     args=(--mode native --native-backend sm100-tuned --weight nvfp4 --act nvfp4) ;;
    fourover6) args=(--mode native --native-backend sm100-tuned --weight fourover6 --act fourover6) ;;
    flipquant) args=(--mode native --native-backend sm100-tuned --weight mixfp4 --act fourover6 --map "$MAP") ;;
  esac
  "$PYTHON" -m evaluation.latency prefill --model qwen3-1.7b "${args[@]}" --shapes 1x2048 --reps 3 \
    --label "smoke-$method" --out "$OUT/smoke_qwen3-1.7b_${method}_$SLURM_JOB_ID.json" 2>&1 | tail -4
done
rm -rf "$BUILD"
echo "prefill setup done"
