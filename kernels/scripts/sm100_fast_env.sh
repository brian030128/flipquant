#!/bin/bash
# Source inside a GB200 job from the repository root: kernels/scripts/sm100_pyenv.sh plus the fast paths of the paper's
# Appendix C.1 -- mamba-ssm 2.3.2.post1 and causal-conv1d 1.7.0 (aarch64 cu13 wheels), flash-linear-attention 0.5.2
# (pure Python / Triton) -- then checks with kernels/scripts/fla_probe.py that Qwen3.8-27B's Gated DeltaNet layers run
# FLA's kernels (the job stops otherwise). Writes $FLA_PROBE_OUT (default: no file kept).
#   FLA_VERSION   flash-linear-attention version (default 0.5.2)
W=https://github.com/
export FLIPQUANT_EXTRA_PIP="${W}state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu13torch2.9cxx11abiTRUE-cp311-cp311-linux_aarch64.whl ${W}Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu13torch2.9cxx11abiTRUE-cp311-cp311-linux_aarch64.whl flash-linear-attention==${FLA_VERSION:-0.5.2} fla-core==${FLA_VERSION:-0.5.2}"
source kernels/scripts/sm100_pyenv.sh
"$PYTHON" -c "import mamba_ssm, causal_conv1d, fla; print('mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__, 'fla', fla.__version__)"
PYTHONPATH="$PWD" "$PYTHON" kernels/scripts/fla_probe.py --out "${FLA_PROBE_OUT:-$PYENV_ROOT/fla_probe.json}"
