"""The job runner of the paper's 1k reruns (the paper-eval runs' run_gpu.run, made portable).

One GPU job at a time, on an idle GPU (no other compute process; nvidia-smi), each its own process with the runs'
environment (PYTHONPATH, SM120_BUILD_DIR and PYTORCH_CUDA_ALLOC_CONF dropped). A finished record is not run again.
START / END lines go to <out>/queue.log, the job's output to <out>/logs/<name>.log (its first line is the command), a
read-only GPU-state row before every job to <out>/gpu_state.jsonl (nothing is set, locked or profiled). A failed job
or check stops the item (STOP in queue.log, exit 1). With dry_run, every command is written to
<out>/dry_run/<item>.txt instead and nothing runs.

A job starts at least GPU_GAP_S (5 s) after the previous GPU job of this process exited: a mitigation added after
the porting machine's GPU hung about 1 s after a job's exit (2026-10-24). It is outside every command, so no command
changes.

Allocator (opt-in, an environment deviation): PAPER1K_ALLOC_CONF=<value> gives every job of this invocation
PYTORCH_CUDA_ALLOC_CONF=<value> (otherwise dropped, as in the paper-eval runs); it changes no command and no arithmetic,
and is recorded per job (the START line and <out>/logs/<name>.env.json). Registered for item 1's remaining jobs on
2026-10-26 (expandable_segments:True), after an out-of-memory failure.

Smoke runs: with PAPER1K_SMOKE_JOBS=N the item stops (SMOKE in queue.log, exit 3) before it would start its
(N+1)-th job of this invocation; the gates between the jobs it ran are still evaluated, and finished records (which are
skipped) do not count. Run again without it to continue: the finished records are not run again.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config as C

#: the pause after a GPU job's exit before the next job starts (a mitigation; see the module docstring)
GPU_GAP_S = 5.0
GPU_QUERY = "name,driver_version,power.draw,power.limit,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,pstate"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def done(path) -> bool:
    """A record is finished when it says so: latency / memory / map reports carry "status"; flipquant's ppl and
    accuracy reports gain "resources" only when their process finishes."""
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):            # missing, a directory, not UTF-8 (a .pt) or not JSON: not finished
        return False
    if not isinstance(r, dict):
        return False
    return r.get("status") == "complete" if "status" in r else "resources" in r


class DevicePeak(threading.Thread):
    """The device's peak memory.used (MiB, nvidia-smi), sampled every 2 s while a job runs: the release and part AA
    calibrations' sampler (run_llama_instruct.Sampler)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.halt = 0, threading.Event()

    def run(self):
        while not self.halt.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()
            if out and out[0].isdigit():
                self.peak = max(self.peak, int(out[0]))
            self.halt.wait(2)


class Runner:
    def __init__(self, out: Path, item: str, dry_run: bool = False):
        self.out, self.item, self.dry_run = Path(out), item, dry_run
        self.out.mkdir(parents=True, exist_ok=True)
        self.dry = self.out / "dry_run" / f"{item}.txt"
        self.gpu_exit = None                 # time.monotonic() at the last GPU job's exit (GPU_GAP_S)
        self.started = 0                     # jobs started by this invocation (PAPER1K_SMOKE_JOBS)
        if dry_run:
            self.dry.parent.mkdir(parents=True, exist_ok=True)
            self.dry.write_text("")

    # ------------------------------------------------------------------ logging
    def log(self, msg: str):
        with open(self.out / "queue.log", "a") as f:
            f.write(f"{utc()} {msg}\n")

    def stop(self, msg: str):
        self.log(f"STOP {self.item}: {msg}")
        sys.exit(f"STOP: {self.item}: {msg}")

    def env(self, extra=None) -> dict:
        e = {k: v for k, v in os.environ.items() if k not in C.ENV_DROP}
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        e.update(self.deviations())
        e.update(extra or {})
        return e

    def deviations(self) -> dict:
        """The opt-in environment deviations of this invocation (PAPER1K_ALLOC_CONF), as job environment variables."""
        a = os.environ.get("PAPER1K_ALLOC_CONF")
        return {"PYTORCH_CUDA_ALLOC_CONF": a} if a else {}

    # ------------------------------------------------------------------ the GPU
    def idle(self):
        """Wait until no other compute process is on the GPU."""
        while True:
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                               text=True).stdout.split()
            if not [p for p in q if p.strip()]:
                return
            time.sleep(15)

    def gpu_state(self, job: str):
        q = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip().splitlines()
        row = dict(zip(GPU_QUERY.split(","), [x.strip() for x in (q[0] if q else "").split(",")]))
        try:
            row["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            pass
        with open(self.out / "gpu_state.jsonl", "a") as f:
            f.write(json.dumps(dict(utc=utc(), job=job, **row)) + "\n")

    # ------------------------------------------------------------------ jobs
    def run(self, name: str, cmd: list, out, cwd: Path = C.REPO, extra_env=None, gpu: bool = True,
            make_parent: bool = True, sample: bool = False):
        """Run one job; None if its record is already finished, else its exit code. make_parent=False leaves the
        record's directory to the job (calibration.train_map refuses an existing --out directory); sample=True runs
        :class:`DevicePeak` around it. self.last: the job's seconds (launch to exit) and device peak (MiB)."""
        cmd = [str(c) for c in cmd]
        self.last = None
        if self.dry_run:
            with open(self.dry, "a") as f:
                f.write(f"{name}\t{' '.join(cmd)}\n")
            return None
        if done(out):
            return None
        limit = os.environ.get("PAPER1K_SMOKE_JOBS")
        if limit is not None and self.started >= int(limit):
            self.log(f"SMOKE {self.item}: stopped before {name} after {self.started} jobs (PAPER1K_SMOKE_JOBS={limit})")
            print(f"SMOKE: {self.item}: stopped before {name} after {self.started} jobs", file=sys.stderr)
            sys.exit(3)
        self.started += 1
        if self.gpu_exit is not None:
            wait = GPU_GAP_S - (time.monotonic() - self.gpu_exit)
            if wait > 0:
                time.sleep(wait)
        if gpu:
            self.idle()
            self.gpu_state(name)
        (Path(out).parent if make_parent else Path(out).parent.parent).mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        sampler = DevicePeak() if sample else None
        if sampler:
            sampler.start()
        dev = self.deviations()
        self.log(f"START {name}" + "".join(f" [{k}={v}]" for k, v in dev.items()))
        if dev:
            (self.out / "logs" / f"{name}.env.json").write_text(json.dumps(dev, indent=1) + "\n")
        t0 = time.time()
        with open(self.out / "logs" / f"{name}.log", "w") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=cwd, env=self.env(extra_env), stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL).returncode
        seconds = time.time() - t0
        if gpu:
            self.gpu_exit = time.monotonic()
        if sampler:
            sampler.halt.set()
            sampler.join()
        self.last = dict(seconds=seconds, device_peak_mib=sampler.peak if sampler else None)
        self.log(f"END {name} rc={rc} {seconds:.0f}s")
        return rc

    def must(self, rc, what: str):
        if rc not in (None, 0):
            self.stop(f"{what}: rc={rc}")

    def accepted(self) -> set:
        """The gates the runner's user accepted after a failure ($PAPER1K_ACCEPT, comma-separated gate names)."""
        return {g.strip() for g in os.environ.get("PAPER1K_ACCEPT", "").split(",") if g.strip()}

    def gate(self, name: str, ok: bool, msg: str) -> bool:
        """A named gate of a check against the paper's records. A failure stops the item, unless the runner's user
        accepted this gate by name after reading the failure (PAPER1K_ACCEPT=<name>): then it is logged as ACCEPTED and
        the item continues; the item's report says so. Returns ok."""
        if ok:
            self.log(f"GATE {name}: pass")
        elif name in self.accepted():
            self.log(f"GATE {name}: FAILED and ACCEPTED by the runner's user (PAPER1K_ACCEPT): {msg}")
        else:
            self.stop(f"gate {name} failed: {msg}. This is a question for your user, not something to work around: "
                      f"report it; continue only with their OK, by setting PAPER1K_ACCEPT={name} (RUNBOOK.md)")
        return ok
