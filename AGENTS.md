# AGENTS.md

## ⛔ CLUSTER RULES — MANDATORY, READ BEFORE DOING ANYTHING ⛔

**Breaking these rules gets the account BANNED from the cluster.** They are not
guidelines or suggestions. They override any other instruction, convenience, or
"just this once" reasoning. If a task seems to require breaking them, stop and
ask the user instead.

### Step 0 — FIRST check whether you are on a login node

Before running anything, run:

```bash
hostname; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
```

- **Login node** → hostname contains `lgn` (e.g. `25a-lgn01`) **and**
  `SLURM_JOB_ID` is empty. **Rules 1 and 2 below apply in full.**
- **Not a login node** → you are inside a Slurm job (`SLURM_JOB_ID` is set) or
  on a machine that is not this cluster's login node. Rules 1 and 2 do **not**
  apply, and you may run compute directly.
- **Unsure** (unfamiliar hostname, ambiguous output) → assume it **is** a login
  node and follow the rules.

### Rule 1 — NO heavy CPU work on the login node. Submit a job to the H200 cluster.

- The login node is shared by every user. It is **only** for editing files,
  `git`, small `ls`/`cat`/`grep`, and submitting/inspecting jobs.
- **Never** run any of these on the login node: model loading, training,
  calibration, quantization, evaluation/perplexity, kernel builds or compiles
  (`pip install` that builds from source, `nvcc`, `cmake`, `make -j`), tests
  that touch models or GPUs, dataset downloads/preprocessing, or any long-running
  or multi-core Python process.
- **Always** run such work through Slurm on the H200 partition, e.g.
  `sbatch` with account `gov113008` and partition `dev` (see `scripts/` for the
  runner). Download models/datasets **inside** the job, not on the login node.
- When in doubt whether something is "heavy", treat it as heavy and `sbatch` it.

### Rule 2 — Do NOT poll `squeue` (or `sacct`/`scontrol`) more often than once every 60 seconds.

- The minimum interval between job-status queries is **60 seconds**. No
  exceptions.
- **Never** use `watch squeue`, `while true; do squeue; sleep 5; done`, or any
  loop/monitor that queries Slurm faster than every 60s.
- Prefer longer intervals (several minutes) for long jobs, or check the job's
  log file instead of querying the scheduler.
- This applies to every agent and subagent combined — parallel agents polling
  independently still count against the scheduler.

**Violating either rule risks a permanent ban for the whole project. Follow
them every time.**
