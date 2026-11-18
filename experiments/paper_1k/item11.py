"""Item 11 (Tables C.2 / C.3, the calibration cost) at the 1k calibration. **The GPU re-measurement was cancelled by
the user (2026-10-26): the report is filled from part AA's records** (reference/item11.json, item11_cost.json); `run
item11` refuses to run, `run item11 --report` writes the report. The re-measurement's driver below is kept unchanged:
a clean re-measurement of the 18 1k FlipQuant calibrations, six models x three tiles (NVFP4-RaZeR paper-eval part AA and its amendment 6; part AC item 11).

Each calibration is part AA's command, `calibration.train_map --unit U --epochs 5 --teacher-topk 1000 --fit-windows
1024` (deterministic kernels, no development set; the n16k64 env for every model; Qwen3.8-27B with `--teacher-host`),
one at a time on an idle GPU, timed from launch to the written map, with the device's peak memory sampled every 2 s
(nvidia-smi) and the trainer's phases and peaks from its report. Every map must equal its 1k map bit for bit: its tiles
digest equals part AA's (reference/item11.json), which the HF Calib1k maps share.

Step 0: the first calibration, Llama-3.2-1B-Instruct 16x64 (the smallest): its map equals part AA's, and its cost is set
beside part AA's record (a drift check, reported). The other 17 follow under the same gate.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from .item3 import tiles_digest

REF = json.loads((C.HERE / "reference" / "item11.json").read_text())
HOST = ("qwen3.8-27b",)                 # part AA amendment 1: its top-K teacher does not fit on the GPU at 1k
TAG = {"8x64": "8", "16x64": "16", "256x64": "256"}
STEP0 = ("llama3.2-1b-instruct", "16x64")


def train_cmd(model, unit, data_root, out):
    """run_calib_size.train_map_cmd."""
    cmd = [C.py_release(), "-m", "calibration.train_map", "--model", model, "--unit", unit, "--epochs", 5,
           "--teacher-topk", 1000, "--fit-windows", 1024, "--data-root", data_root]
    if model in HOST:
        cmd.append("--teacher-host")
    return cmd + ["--out", out]


def order():
    """Step 0 first, then the paper's model order, the three tiles each."""
    rest = [(m, u) for m in C.MODELS for u in ("16x64", "8x64", "256x64") if (m, u) != STEP0]
    return [STEP0] + rest


def job_record(model, unit, cmd, last, out: Path) -> dict:
    """run_calib_size.calibrate_fq's job record, with the map's tiles digest."""
    rep = json.loads((out / "reproduction.json").read_text())
    report = json.loads((out / "run" / "report.json").read_text())
    res = report["resources"]
    return dict(model=model, unit=unit, windows=1024, teacher_host=model in HOST, command=" ".join(map(str, cmd[1:])),
                job_seconds=last["seconds"], device_peak_mib=last["device_peak_mib"],
                run_map_sha256=rep["run_map_sha256"], map_file_sha256=rep["map_file_sha256"], e0m3_share=rep["e0m3_share"],
                train_map_wall_seconds=rep["wall_seconds"], settings=rep["settings"],
                trainer=dict(total_seconds=res.get("total_seconds"), setup_seconds=report.get("setup_seconds"),
                             training_seconds=report.get("training_seconds"),
                             gpu_peak_allocated_gib=res.get("gpu_peak_allocated_gib"),
                             gpu_peak_reserved_gib=res.get("gpu_peak_reserved_gib"),
                             cpu_peak_rss_gib=res.get("cpu_peak_rss_gib"),
                             phases={p["name"]: dict(seconds=p.get("seconds"), host_peak_rss=p.get("host_peak_rss"),
                                                     gpu_peak_allocated=p.get("gpu_peak_allocated"))
                                     for p in (res.get("phases") or {}).get("phases", [])}),
                teacher_storage=report.get("teacher_storage"), tiles=tiles_digest(out / "map.pt"))


def run(R, out: Path, dry_run=False, data_root: Path | None = None):
    if data_root is None:
        R.stop("--data-root: the calibration data root of `setup records`")
    d = out / "item11"
    for model, unit in order():
        o = d / model / f"fq{TAG[unit]}_1024"
        meta = d / model / f"fq{TAG[unit]}_1024.job.json"
        cmd = train_cmd(model, unit, data_root, o)
        name = f"item11_train_{model}_{unit}"
        if dry_run:
            R.run(name, cmd, o / "reproduction.json")
            continue
        if not meta.exists():
            if o.exists():
                R.stop(f"{o} exists without its job record (an interrupted calibration): move it aside (nothing is "
                       "deleted by this driver), then run the item again")
            R.must(R.run(name, cmd, o / "reproduction.json", make_parent=False, sample=True), f"train_map {model} {unit}")
            if R.last is None:          # the record was finished but its job record is missing: not timed here
                R.stop(f"{o}: a finished calibration without its job record; move it aside and run the item again")
            meta.write_text(json.dumps(job_record(model, unit, cmd, R.last, o), indent=1) + "\n")
        rec = json.loads(meta.read_text())
        want = REF["jobs"][model][unit]["tiles"]
        same = rec["tiles"]["digest"] == want["digest"]
        R.gate("item11-maps", same, f"{model} {unit}: the map's tiles differ from the 1k map's "
                                    f"({rec['tiles']['e0m3_tiles']} vs {want['e0m3_tiles']} E0M3 tiles)")
        R.log(f"item11 {model} {unit}: map equal to the 1k map {same}; {rec['job_seconds']:.0f} s "
              f"(part AA {REF['jobs'][model][unit]['job_seconds']:.0f} s)")
        if (model, unit) == STEP0:
            a = REF["jobs"][model][unit]
            (d / "step0.json").write_text(json.dumps(dict(
                model=model, unit=unit, map_equal=same, job_seconds=rec["job_seconds"], part_aa_job_seconds=a["job_seconds"],
                ratio=rec["job_seconds"] / a["job_seconds"], trainer_seconds=rec["trainer"]["total_seconds"],
                part_aa_trainer_seconds=a["trainer"]["total_seconds"], run_map_sha256_equal=rec["run_map_sha256"] ==
                a["run_map_sha256"]), indent=1) + "\n")
    R.log("item11 done")


# ---------------------------------------------------------------------------------------------- the report
#: the GPU re-measurement was cancelled by the user (2026-10-26): Tables C.2 / C.3 are filled from part AA's records
CANCELLED = "cancelled by the user (2026-10-26); filled from part AA records"
COST = json.loads((C.HERE / "reference" / "item11_cost.json").read_text())
PHASES = ("model_load", "data_load", "teacher_precompute", "candidate_packing")
UNITS = ("8x64", "16x64", "256x64")


def _gib(x):
    return max(x) if isinstance(x, list) and x else x


def _mean(xs):
    return sum(xs) / len(xs)


def row(model, unit) -> dict:
    """One 1k calibration from the records: reference/item11.json (the job record) + item11_cost.json (the trainer's
    epochs and phases)."""
    j, k = REF["jobs"][model][unit], COST["k1"][model][unit]
    t = j["trainer"]
    ep = _mean(k["epoch_seconds"])
    return dict(setup_s=t["setup_seconds"], training_s=t["training_seconds"],
                trainer_s=t["setup_seconds"] + t["training_seconds"], job_s=j["job_seconds"],
                training_share=t["training_seconds"] / (t["setup_seconds"] + t["training_seconds"]),
                gpu_alloc_gib=_gib(t["gpu_peak_allocated_gib"]), gpu_reserved_gib=_gib(t["gpu_peak_reserved_gib"]),
                device_peak_gib=j["device_peak_mib"] / 1024, phases={p: k["phases"][p] for p in PHASES},
                epoch_mean_s=ep, steps_per_epoch=k["steps_per_epoch"][0], s_per_step=ep / k["steps_per_epoch"][0],
                teacher_gib=k["teacher"]["bytes"] / 2 ** 30, teacher_devices=k["teacher"]["devices"],
                e0m3_tiles=j["tiles"]["e0m3_tiles"], date=k["date_utc"], checkout=j["checkout"][:7],
                source="part AA panel" if j["checkout"].startswith(("898d00e", "f912a30")) else "amendment 6")


def r256(model, unit) -> dict:
    r = COST["r256"][model][unit]
    ep = _mean(r["epoch_seconds"])
    return dict(setup_s=r["setup_seconds"], phases={p: r["phases"][p] for p in PHASES}, epoch_mean_s=ep,
                s_per_step=ep / r["steps_per_epoch"][0], teacher_gib=r["teacher"]["bytes"] / 2 ** 30,
                teacher_devices=r["teacher"]["devices"])


def report(out: Path, dest: Path):
    rec = dict(status=CANCELLED, rows={}, c3={}, vs_256={}, step0=COST["phase_a_step0"])
    for m in C.MODELS:
        for u in UNITS:
            rec["rows"].setdefault(m, {})[u] = row(m, u)
    md = ["# Item 11: Tables C.2 / C.3 (the calibration cost) at the 1k calibration — from part AA's records", "",
          f"**The GPU re-measurement was {CANCELLED}.** The 18 1k FlipQuant calibrations (six models x 8x64 / 16x64 / "
          "256x64) are part AA's runs of `calibration.train_map --epochs 5 --teacher-topk 1000 --fit-windows 1024` "
          "(deterministic kernels, no development set, the n16k64 env, one RTX PRO 6000 at 500 W): their job records "
          "(`reference/item11.json`) and the trainer's own reports (`reference/item11_cost.json`), committed so that this "
          "report regenerates from flipquant alone (`python -m experiments.paper_1k.run item11 --report`).", "",
          "- **Setup / Training:** the trainer's phases (setup = model load, data load, teacher precompute, candidate "
          "packing). **Trainer** = Setup + Training. **Total:** the job's wall clock, from launch to the written map.",
          "- **GPU:** the trainer's peak allocated / reserved memory; **device peak:** nvidia-smi's memory.used, sampled "
          "every 2 s. **Teacher:** the top-1,000 teacher's storage (values BF16, ids int32, tail log-mass FP32).",
          "- **Epoch / step:** the mean of the five epochs; 128 optimizer steps per epoch at 1,024 windows (32 at 256).", ""]
    s = rec["step0"]
    md += ["## Reproduction evidence", "",
           f"Phase A (2026-10-25) ran item 11's step 0 from a fresh flipquant clone on the porting machine's idle RTX PRO "
           f"6000: Llama-3.2-1B-Instruct 16x64, part AA's command. Launch to map {s['job_seconds']} s against part AA's "
           f"{s['part_aa_job_seconds']} s, and the map equal to the 1k map bit for bit (gate `item11-maps`).", ""]
    md += ["## Table C.2: the cost at 16x64", "",
           "| model | Setup (min) | Training (min) | Trainer (min) | Total (min) | training share | GPU alloc. / reserved "
           "(GiB) | device peak (GiB) | teacher |", "|---|---:|---:|---:|---:|---:|---|---:|---|"]
    tex2 = ["% tab:app-calib-models at 1k (16x64), from part AA's records: Setup / Training / Total (min), GPU peak "
            "allocated (GiB); one RTX PRO 6000"]
    for m in C.MODELS:
        r = rec["rows"][m]["16x64"]
        md.append(f"| {C.TITLE[m]} | {r['setup_s'] / 60:.1f} | {r['training_s'] / 60:.1f} | {r['trainer_s'] / 60:.1f} | "
                  f"{r['job_s'] / 60:.1f} | {100 * r['training_share']:.0f} % | {r['gpu_alloc_gib']:.1f} / "
                  f"{r['gpu_reserved_gib']:.1f} | {r['device_peak_gib']:.1f} | {r['teacher_gib']:.2f} GiB on "
                  f"{'the host' if r['teacher_devices'] == ['cpu'] else 'the GPU'} |")
        tex2.append(f"{C.TITLE[m]} & {r['setup_s'] / 60:.1f} & {r['training_s'] / 60:.1f} & {r['job_s'] / 60:.1f} & "
                    f"{r['gpu_alloc_gib']:.1f} \\\\")
    md += ["", "### The setup phases and the step time at 16x64 (s)", "",
           "| model | model load | data load | teacher precompute | candidate packing | epoch (mean) | per step |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m in C.MODELS:
        r = rec["rows"][m]["16x64"]
        md.append(f"| {C.TITLE[m]} | " + " | ".join(f"{r['phases'][p]:.1f}" for p in PHASES)
                  + f" | {r['epoch_mean_s']:.1f} | {r['s_per_step']:.3f} |")
    md += ["", "## Table C.3: the three tiles", "",
           "| model | trainer 8x64 / 16x64 / 256x64 (s) | Total 8x64 / 16x64 / 256x64 (s) | max. diff. trainer (%) | "
           "max. diff. Total (%) | GPU 8x64 − 256x64 (GiB) |", "|---|---|---|---:|---:|---:|"]
    tex3 = ["% tab:app-calib-granularity at 1k, from part AA's records: the trainer's time (setup + training, s) at "
            "8x64 / 16x64 / 256x64; max diff relative to the smallest"]
    for m in C.MODELS:
        rs = rec["rows"][m]
        tr, tot = {u: rs[u]["trainer_s"] for u in UNITS}, {u: rs[u]["job_s"] for u in UNITS}
        c3 = dict(trainer_s=tr, job_s=tot, max_diff_trainer_pct=100 * (max(tr.values()) - min(tr.values())) / min(tr.values()),
                  max_diff_job_pct=100 * (max(tot.values()) - min(tot.values())) / min(tot.values()),
                  gpu_8_minus_256_gib=rs["8x64"]["gpu_alloc_gib"] - rs["256x64"]["gpu_alloc_gib"],
                  s_per_step={u: rs[u]["s_per_step"] for u in UNITS})
        rec["c3"][m] = c3
        md.append(f"| {C.TITLE[m]} | " + " / ".join(f"{tr[u]:,.0f}" for u in UNITS) + " | "
                  + " / ".join(f"{tot[u]:,.0f}" for u in UNITS)
                  + f" | {c3['max_diff_trainer_pct']:.1f} | {c3['max_diff_job_pct']:.1f} | {c3['gpu_8_minus_256_gib']:+.2f} |")
        tex3.append(f"{C.TITLE[m]} & " + " & ".join(f"{tr[u]:,.0f}".replace(",", "{,}") for u in UNITS)
                    + f" & {c3['max_diff_trainer_pct']:.1f} \\\\")
    md += ["", "## 256 against 1,024 windows (the release calibrations' trainer reports)", "",
           "Ratios 1k / 256 per model at 16x64; in brackets the range over the three tiles.", "",
           "| model | setup | teacher precompute | candidate packing | model load | epoch | per step | teacher memory "
           "(GiB, 256 → 1k) |", "|---|---:|---:|---:|---:|---:|---:|---|"]

    def ratio(m, u, f):
        return f(rec["rows"][m][u]) / f(r256(m, u))

    def rng(m, f):
        v = [ratio(m, u, f) for u in UNITS]
        return f"{ratio(m, '16x64', f):.2f} [{min(v):.2f}–{max(v):.2f}]"

    for m in C.MODELS:
        fs = dict(setup=lambda r: r["setup_s"], teacher=lambda r: r["phases"]["teacher_precompute"],
                  packing=lambda r: r["phases"]["candidate_packing"], model_load=lambda r: r["phases"]["model_load"],
                  epoch=lambda r: r["epoch_mean_s"], step=lambda r: r["s_per_step"])
        rec["vs_256"][m] = {k: {u: ratio(m, u, f) for u in UNITS} for k, f in fs.items()}
        a, b = r256(m, "16x64"), rec["rows"][m]["16x64"]
        md.append(f"| {C.TITLE[m]} | " + " | ".join(rng(m, f) for f in fs.values())
                  + f" | {a['teacher_gib']:.2f} ({'GPU' if a['teacher_devices'] != ['cpu'] else 'host'}) → "
                  f"{b['teacher_gib']:.2f} ({'host' if b['teacher_devices'] == ['cpu'] else 'GPU'}) |")
    allv = lambda k: [v for m in C.MODELS for v in rec["vs_256"][m][k].values()]
    md += ["", f"Over all 18: setup x{min(allv('setup')):.2f}–{max(allv('setup')):.2f}, of which the teacher precompute "
           f"x{min(allv('teacher')):.2f}–{max(allv('teacher')):.2f} (four times the windows), while the candidate packing "
           f"(x{min(allv('packing')):.2f}–{max(allv('packing')):.2f}) and the model load (a few seconds) do not grow; one "
           f"epoch x{min(allv('epoch')):.2f}–{max(allv('epoch')):.2f}; the time per optimizer step "
           f"x{min(allv('step')):.3f}–{max(allv('step')):.3f}, i.e. unchanged.", ""]
    md += ["## Caveats", "",
           "- **Qwen3.8-27B keeps its top-1,000 teacher on the host (`--teacher-host`)**: at 1,024 windows the teacher "
           "(2.93 GiB) does not fit on the GPU next to the model and the training state. Its per-step time is unchanged "
           f"against 256 (x{ratio('qwen3.8-27b', '16x64', lambda r: r['s_per_step']):.3f} at 16x64).",
           "- **Two campaigns.** The 16x64 runs are part AA's panel (2026-10-19/20; Llama-3.2-1B-Instruct and Phi-4 in its "
           "pilot); the 8x64 and 256x64 runs are amendment 6 (2026-10-23/24), which ran while the Hugging Face staging "
           "used the CPU and the network. The per-step times of the three tiles agree within "
           f"{max(100 * (max(c['s_per_step'].values()) / min(c['s_per_step'].values()) - 1) for c in rec['c3'].values()):.1f} %.",
           "- **Paper-text points.** The teacher's memory at 1k is about 2.9 GiB (2.93 GiB), not the 0.73 GiB of 256 "
           "windows. Qwen3.8-27B's teacher is on the host at 1k, which contradicts Table C.1's \"kept on the GPU\" and "
           "Section 3's \"under 1 GiB\" for the 1k maps.", ""]
    md += ["## Every calibration", "",
           "| model | tile | date (UTC) | campaign | E0M3 tiles | Setup (s) | Training (s) | Total (s) | epoch (s) | "
           "per step (s) | GPU alloc. / reserved (GiB) | device peak (GiB) | teacher |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for m in C.MODELS:
        for u in UNITS:
            r = rec["rows"][m][u]
            md.append(f"| {C.TITLE[m]} | {u} | {r['date']} | {r['source']} ({r['checkout']}) | {r['e0m3_tiles']:,} | "
                      f"{r['setup_s']:,.0f} | {r['training_s']:,.0f} | {r['job_s']:,.0f} | {r['epoch_mean_s']:.1f} | "
                      f"{r['s_per_step']:.3f} | {r['gpu_alloc_gib']:.1f} / {r['gpu_reserved_gib']:.1f} | "
                      f"{r['device_peak_gib']:.1f} | {'host' if r['teacher_devices'] == ['cpu'] else 'GPU'} |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "COST.md").write_text("\n".join(md) + "\n")
    (dest / "item11.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (dest / "table_calib_cost.tex").write_text("\n".join(tex2 + [""] + tex3) + "\n")
    return f"{dest}/COST.md"
