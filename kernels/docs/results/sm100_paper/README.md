# SM100 (GB200) paper runs: output directory

The SM100 job scripts in `kernels/scripts/` write their logs and results here (Slurm's `-o` needs the
directories to exist, so they are kept empty in git; run outputs are gitignored). All runs are on GB200
(CUDA 13.0, driver 580), with `kernels/scripts/build_sm100.sh` building the kernels on the node.

| folder | paper | job script | other scripts |
|---|---|---|---|
| `01_gemm_overhead/` | D.6: per-forward GEMM overhead of the 256x64 per-tile kernel, release 256x64 maps | `forward_gemm_gb200.sbatch` | `forward_gemm_sm100.py` |
| `07_tails/` | D.6: N or K not a multiple of 256 (correctness) | `forward_gemm_gb200.sbatch` | `check_sm100_sub.sh` |
| `02_prefill_latency/` | Table 4 (GB200 rows): prefill latency at 1x2048 | `prefill_setup_gb200.sbatch`, then `prefill_rounds_gb200.sbatch <models>` | `prefill_rounds_sm100.py`, `summarize_prefill_sm100.py`, `test_sm100_linear.py`, `sm100_fast_env.sh`, `fla_probe.py` |
| `03_native_vs_simulated/` | A.4 / D.7: native vs simulated perplexity, and the ownership (read-back) check | `ppl_native_vs_sim_gb200.sbatch <models>` | `ppl_native_vs_sim_sm100.py`, `summarize_ppl_sm100.py`; the check is `models/sm100.py` `install(ownership=True)` |
| `05_operand_width/` | Figure 3b / D.6: GEMM time against MMA width N = 8 ... 256 at 8192^3 | `pad_sweep_gb200.sbatch` | `benchmark_sm100_sub.py`, `check_sm100_sub.sh` |
| `05_operand_width/probes/` | the scale-factor TMEM layout rules behind N = 8/16/32 | `probe_gb200.sbatch` | `probe_sm100_sfb.sh` |

The fast (not bit-exact) activation quantizer: `QUANT=fast prefill_rounds_gb200.sbatch`,
`RUNS=native_fast ppl_native_vs_sim_gb200.sbatch`, and
`fast_env_run_gb200.sbatch kernels/scripts/test_fast_quant_sm100.py OUT` (log in `02_prefill_latency/graph_diag/`).
