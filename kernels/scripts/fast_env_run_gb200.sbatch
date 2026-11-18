#!/bin/bash
# Run one command on a GB200 GPU in the fast Python stack (kernels/scripts/sm100_fast_env.sh; $PYTHON, the SM100 linear
# library built). Log: $LOG_DIR (default kernels/docs/results/sm100_paper/02_prefill_latency/graph_diag).
#   sbatch kernels/scripts/fast_env_run_gb200.sbatch kernels/scripts/test_fast_quant_sm100.py OUT.json
#SBATCH -A gov113008
#SBATCH -p gb200-dev
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -J fq-fast-run
#SBATCH -o kernels/docs/results/sm100_paper/02_prefill_latency/graph_diag/run_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted: $*"
export BUILD="/tmp/fq-build-$(uname -m)-$SLURM_JOB_ID"
SRC="$PWD/kernels" JOBS=12 TARGETS=flipquant_sm100_linear bash kernels/scripts/build_sm100.sh 2>&1 | grep -E 'error|built into' | head
export FLIPQUANT_SM100_LINEAR_LIB="$BUILD/libflipquant_sm100_linear.so"
source kernels/scripts/sm100_fast_env.sh
export PYTHONPATH="$PWD"
"$PYTHON" "$@" 2>&1 | grep -v Loading
rm -rf "$BUILD"
echo "run done"
