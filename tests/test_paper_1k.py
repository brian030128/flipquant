"""The paper's 1k reruns driver (experiments/paper_1k), on the CPU: no GPU, no network, no other repository.

* Every item marked ready builds, in a dry run, the commands the paper-eval runs executed (reference/commands_*.json,
  from those runs' job logs), token for token after experiments.paper_1k.commands' path normalization.
* The pinned artifacts, windows and references are well formed (sha256 / revision shapes, counts).
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paper_1k import commands as K  # noqa: E402
from experiments.paper_1k import config as C  # noqa: E402
from experiments.paper_1k import run as RUN  # noqa: E402

REF = ROOT / "experiments" / "paper_1k" / "reference"


def dry(item, tmp_path, monkeypatch, data_root=None, drop_out=False):
    for var in ("PAPER1K_CODES", "PAPER1K_HF_CACHE", "PAPER1K_BUILD_ITEM9", "PAPER1K_ACCEPT", "PAPER1K_SMOKE_JOBS",
                "PAPER1K_ALLOC_CONF", "PAPER1K_ITEM1_BATCH", "PAPER1K_ITEM1_MODELS"):
        monkeypatch.delenv(var, raising=False)          # the runner's own settings must not leak into the references
    monkeypatch.setenv("PAPER1K_PY", "/env/eval/bin/python")
    monkeypatch.setenv("PAPER1K_PY_FAST", "/env/fast/bin/python")
    monkeypatch.setenv("PAPER1K_BUILD", "/build")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    RUN.main([item, "--dry-run", "--out", str(tmp_path), *(["--data-root", str(data_root)] if data_root else [])])
    rows = [l.split("\t", 1) for l in (tmp_path / "dry_run" / f"{item}.txt").read_text().splitlines()]
    assert len({name for name, _ in rows}) == len(rows), "job names must be unique"
    return {name: K.normalize(cmd.split(" "), str(tmp_path), (item,), str(data_root) if data_root else None, drop_out)
            for name, cmd in rows}


def test_item2_memory_commands_are_the_runs(tmp_path, monkeypatch):
    got = dry("item2", tmp_path, monkeypatch)
    ref = json.loads((REF / "commands_item2.json").read_text())["commands"]
    assert len(got) == 1 + 6 * 7
    for name in ("item2_memory_anchor_mistral-7b_fq-16x64", "item2_memory_llama3.2-1b-instruct_bf16",
                 "item2_memory_llama3.2-1b-instruct_nvfp4"):
        assert got[name] == ref[name], name
    # every panel job is part B's probe with run_gpu.POLICIES' flags and the 1k map of its unit
    for name, cmd in got.items():
        assert cmd[1] == "{SCRIPT:mem_probe.py}" and cmd[cmd.index("--label") + 1] in C.POLICIES
        if "--map" in cmd and "anchor" not in name:
            assert cmd[cmd.index("--map") + 1].startswith("{MAP:flipquant_1k:")
        assert cmd[0] == ("{PY_FAST}" if cmd[cmd.index("--model") + 1] in C.HYBRID else "{PY}")


def g_to_1k(cmd, model):
    """A part G command with item 3's registered changes (paper-eval PROTOCOL.md, part AC item 3): --match the 1k map;
    act's --calib-extension the 1k windows; oneshot --fit-windows 1024 (and --teacher-host for Qwen3.8-27B) before
    --out."""
    c = list(cmd)
    i = c.index("--match") + 1
    assert c[i].startswith("{MAP:flipquant_256:")
    c[i] = c[i].replace("{MAP:flipquant_256:", "{MAP:flipquant_1k:")
    if "--calib-extension" in c:
        i = c.index("--calib-extension") + 1
        assert c[i].startswith("{EXT:flipquant_256:")
        c[i] = f"{{OUT}}/windows/{model}_1024.json"
    if "oneshot" in c:
        j = c.index("--out")
        c[j:j] = ["--fit-windows", "1024"] + (["--teacher-host"] if model == "qwen3.8-27b" else [])
    return c


def test_item3_commands_are_the_runs(tmp_path, monkeypatch):
    data = tmp_path / "data_root"
    got = dry("item3", tmp_path, monkeypatch, data_root=data, drop_out=True)
    ref = json.loads((REF / "commands_item3.json").read_text())["commands"]
    models, maps = ("nemotron-nano-9b-v2", "qwen3.8-27b"), ("random0", "random1", "random2", "act", "oneshot")
    assert len(got) == 2 * 3 + 1 + 2 * (1 + 3 * (5 + 5 + 1))
    # step 0 (b): G's Nemotron 16x64 one-shot command, unchanged
    assert got["item3_step0_rebuild_oneshot_256"] == ref["abl_map_nemotron-nano-9b-v2_16x64_oneshot"]
    for model in models:
        # step 0 (a): the registered smoke (run_paper1k.item3_step0 at b0d4da7)
        sm = "{OUT}/smoke/data_root"
        host = ["--teacher-host"] if model == "qwen3.8-27b" else []
        assert got[f"item3_smoke_record_{model}"] == ["{PY_FAST}", "-m", "calibration.tmopt_ext", "record", "--model", model,
                                                       "--tiny-layers", "3", "--data-root", sm]
        assert got[f"item3_smoke_remap_{model}"] == ["{PY_FAST}", "-m", "models.tiny", "remap", "--model", model,
                                                      "--tiny-layers", "3", "--map", f"{{MAP:flipquant_1k:{model}:16x64}}",
                                                      "--out", "{OUT}"]
        assert got[f"item3_smoke_oneshot_{model}"] == ["{PY_FAST}", "-m", "calibration.tmopt_ext", "oneshot", "--model", model,
                                                        "--unit", "16x64", "--tiny-layers", "3", "--fit-windows", "1024",
                                                        "--oneshot-steps", "2", "--match", f"{{OUT}}/smoke/{model}_tiny3_16x64.pt",
                                                        "--data-root", sm, "--out", "{OUT}", *host]
        # the re-measured rows: part AA's FourOverSix and 1k FlipQuant PPL commands
        assert got[f"item3_ppl_{model}_fo6"] == ref[f"cs_ppl_{model}_fo6"]
        for u, n in zip(C.UNITS, ("8", "16", "256")):
            assert got[f"item3_ppl_{model}_{u}_flipquant"] == ref[f"cs_ppl_{model}_fq{n}_1024"]
            for m in maps:      # G's maps with the registered changes, and G's PPL commands unchanged
                assert got[f"item3_map_{model}_{u}_{m}"] == g_to_1k(ref[f"abl_map_{model}_{u}_{m}"], model), (model, u, m)
                assert got[f"item3_ppl_{model}_{u}_{m}"] == ref[f"abl_ppl_{model}_{u}_{m}"], (model, u, m)


def test_item3_needs_the_data_root(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="data-root"):
        dry("item3", tmp_path, monkeypatch)


def _ppl_record(path, nll):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(results={c: dict(nll=v) for c, v in nll.items()}, resources={})))


def test_item3_report_on_synthetic_records(tmp_path):
    """The report's arithmetic on records shaped like flipquant's: FourOverSix and FlipQuant as the paper's per-window
    NLLs (agreement: equal), baselines FourOverSix + a constant."""
    import experiments.paper_1k.item3 as I
    d = tmp_path / "item3"
    for model in I.MODELS:
        fo6 = I.REF["fo6_nll"][model]
        _ppl_record(d / model / "ppl_fo6.json", fo6)
        for u in C.UNITS:
            _ppl_record(d / model / u / "ppl_flipquant.json", I.REF["flipquant_1k_nll"][model][u])
            for k, m in enumerate(I.MAPS):
                _ppl_record(d / model / u / f"ppl_{m}.json", {c: [x + 1e-3 * (k + 1) for x in v] for c, v in fo6.items()})
            (d / model / u / "act.json").write_text(json.dumps(dict(status="complete", data=dict(fit_windows=1024))))
            (d / model / u / "oneshot.json").write_text(json.dumps(dict(status="complete", meta=dict(fit_windows=1024))))
    I.report(tmp_path, tmp_path / "dest")
    md = (tmp_path / "dest" / "ABLATION.md").read_text()
    rec = json.loads((tmp_path / "dest" / "item3.json").read_text())
    assert "TBD" not in md and (tmp_path / "dest" / "table_ablation_wiki.tex").exists()
    for model in I.MODELS:
        cells = rec["cells"][model]
        assert abs(cells["random"]["16x64"]["wiki"]["delta_milli"] - 2.0) < 1e-9       # the seeds' mean: (1 + 2 + 3) / 3
        assert abs(cells["oneshot"]["8x64"]["c4"]["delta_milli"] - 5.0) < 1e-9
        assert all(v["tier"] == "exact" for v in rec["agreement"][model].values())
        assert len(rec["agreement"][model]) == 4 and set(rec["vs_256"][model]) == {"random", "act", "oneshot", "flipquant"}


def r3_to_1k(cmd, model, train=False):
    """A round 3 / part H command with item 4's registered changes: the 1k windows as --gptq-calib-extension, the 1k
    map for the fixed-map rows, the GPTQ-candidate map's name without round 3's attempt suffix, and for the trainings
    --fit-windows 1024 (--teacher-host for Qwen3.8-27B) after --data-root."""
    c = list(cmd)
    if "--gptq-calib-extension" in c:
        i = c.index("--gptq-calib-extension") + 1
        assert c[i] == f"{{EXT:flipquant_256:{model}:16x64}}"
        c[i] = f"{{OUT}}/windows/{model}_1024.json"
    if "--map" in c:
        i = c.index("--map") + 1
        c[i] = c[i].replace("{MAP:flipquant_256:", "{MAP:flipquant_1k:").replace("_gptqcand_a2.pt", "_gptqcand.pt")
    if train:
        j = c.index("--data-root") + 2
        c[j:j] = ["--fit-windows", "1024"] + (["--teacher-host"] if model == "qwen3.8-27b" else [])
    return c


def test_item4_commands_are_the_runs(tmp_path, monkeypatch):
    got = dry("item4", tmp_path, monkeypatch, data_root=tmp_path / "data_root", drop_out=True)
    ref = json.loads((REF / "commands_item4.json").read_text())["commands"]
    assert len(got) == 1 + 2 * 21 + 3
    s0 = list(ref["round3_gptq_nemotron-nano-9b-v2_nvfp4"])
    s0[s0.index("--gptq-calib-extension") + 1] = "{OUT}/windows/nemotron-nano-9b-v2_256.json"
    s0[s0.index("--gptq-cache") + 1] = "{OUT}/codes/step0/nemotron-nano-9b-v2/codes_nvfp4_256.pt"
    assert got["item4_step0_gptq_nvfp4_256"] == s0
    for m in ("nemotron-nano-9b-v2", "qwen3.8-27b"):
        for new, old in (("bf16", "f_ppl_{m}_bf16"), ("rtn_nvfp4", "f_ppl_{m}_nvfp4"), ("rtn_fo6", "f_ppl_{m}_fo6"),
                         ("rtn_fq-16x64", "aa_ppl_{m}_fq16_1024"), ("hadamard_nvfp4", "h_ppl_{m}_hadamard_nvfp4"),
                         ("hadamard_fo6", "h_ppl_{m}_hadamard_fo6"), ("hadamard_fq-16x64", "h_ppl_{m}_hadamard_fq-16x64")):
            assert got[f"item4_ppl_{m}_{new}"] == ref[old.format(m=m)], (m, new)
        for row in ("nvfp4", "fo6", "fq-16x64"):
            assert got[f"item4_gptq_{m}_{row}"] == r3_to_1k(ref[f"round3_gptq_{m}_{row}"], m), (m, row)
        assert got[f"item4_all_e0m3_{m}"] == [("{PY_FAST}"), "-m", "experiments.paper_1k.item4", "all-e0m3",
                                              f"{{MAP:flipquant_1k:{m}:16x64}}", f"{{OUT}}/{m}/all_e0m3_16x64.pt",
                                              f"{{OUT}}/{m}/all_e0m3_16x64.json"]
        assert got[f"item4_e0m3_{m}"] == r3_to_1k(ref[f"round3_e0m3_{m}"], m)
        assert got[f"item4_train_{m}_gptqcand"] == r3_to_1k(ref[f"round3_train_{m}_gptq_release"], m, train=True)
        assert got[f"item4_train_{m}_gptqcand"][0] == "{PY}"          # round 3's trainer env, n16k64
        assert got[f"item4_train_{m}_hadamard"] == r3_to_1k(ref[f"h_train_{m}_hadamard16"], m, train=True)
        assert got[f"item4_ppl_{m}_gptqcand_fq-16x64"] == r3_to_1k(ref[f"round3_ppl3_{m}"], m)
        recheck = r3_to_1k(ref[f"round3_ppl3_{m}"], m)
        recheck[recheck.index("--map") + 1] = f"{{MAP:flipquant_1k:{m}:16x64}}"
        assert got[f"item4_ppl_{m}_recheck_rtnmap_gptqcand"] == recheck
        for name in ("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64", "gptqcand_fq-16x64"):
            assert got[f"item4_gsm8k_{m}_{name}"] == r3_to_1k(ref[f"round3_gsm8k_{m}_{name}"], m), (m, name)
        assert got[f"item4_gsm8k_{m}_hadamard_fq-16x64"] == ref[f"n_gsm8k_{m}_hadamard_fq-16x64"]
        # the auxiliary fast-env repeat: part H's 256 Hadamard training command, unchanged (its env n16k64-fast)
        assert got[f"item4_aux_h256_{m}"] == ref[f"h_train_{m}_hadamard16"] and got[f"item4_aux_h256_{m}"][0] == "{PY_FAST}"
    r2 = list(ref["h_train_nemotron-nano-9b-v2_hadamard16"])            # Nemotron's second repeat: the same command
    assert got["item4_aux_h256_nemotron-nano-9b-v2_r2"] == r2


def _gsm_record(path, bits, codes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    import experiments.paper_1k.item4 as I
    ex = {d: {f: dict(exact_match=int(b)) for f in I.FILTERS} for d, b in zip(I.REF["doc_ids"], bits)}
    path.write_text(json.dumps(dict(status="complete", tasks={I.TASK: dict(examples=ex)},
                                    policy=dict(ptq=dict(codes_sha256=codes)))))


def test_item4_report_on_synthetic_records(tmp_path):
    import experiments.paper_1k.item4 as I
    d = tmp_path / "item4"
    for m in I.MODELS:
        fo6 = I.REF["reused_ppl"][m]["rtn_fo6"]
        for k, v in I.REF["reused_ppl"][m].items():
            _ppl_record(d / m / "reused" / f"{k}.json", v)
        for j, k in enumerate(("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64", "gptqcand_fq-16x64", "recheck_rtnmap_gptqcand",
                               "hadamard_fq-16x64")):
            _ppl_record(d / m / f"{k}.json", {c: [x + 1e-3 * (j + 1) for x in v] for c, v in fo6.items()})
            if k != "recheck_rtnmap_gptqcand":
                _gsm_record(d / "gsm8k" / m / f"{k}.json", I.REF["reused_gsm8k"][m]["rtn_fo6"]["strict_match"])
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    for m in I.MODELS:
        (logs / f"item4_train_{m}_gptqcand.log").write_text("/env/eval/bin/python -m calibration.tmopt_ext train\n")
        (logs / f"item4_train_{m}_hadamard.log").write_text("/env/eval-fast/bin/python -m calibration.tmopt_ext train\n")
    def rp(k, digest, kl):
        return dict(repeat=k, tiles=dict(e0m3_tiles=10, digest=digest), equal_part_h=digest == "h" * 64, epoch_train_kl=kl,
                    first_epoch_differs_from_part_h=I.first_diff(kl, [1.0, 0.5]), env="n16k64-fast", training_seconds=480.0)
    ph = dict(e0m3_tiles=10, digest="h" * 64, training_seconds=478.1, epoch_train_kl=[1.0, 0.5])
    fe = {I.MODELS[0]: dict(part_h=ph, repeats=[rp(1, "a" * 64, [1.0, 0.4]), rp(2, "b" * 64, [1.0, 0.3])]),
          I.MODELS[1]: dict(part_h=ph, repeats=[rp(1, "h" * 64, [1.0, 0.5])])}
    for v in fe.values():
        v["first_epoch_repeats_differ"] = I.first_diff(v["repeats"][0]["epoch_train_kl"], v["repeats"][1]["epoch_train_kl"]) \
            if len(v["repeats"]) > 1 else None
        v["conclusion"] = I.fastenv_conclusion(v)
    assert fe[I.MODELS[0]]["conclusion"].startswith("not bit-reproducible") and fe[I.MODELS[0]]["first_epoch_repeats_differ"] == 1
    assert fe[I.MODELS[1]]["conclusion"].startswith("reproduces part H")
    (tmp_path / "item4" / "fastenv_repeat.json").write_text(json.dumps(dict(status="complete", models=fe)))
    I.report(tmp_path, tmp_path / "dest")
    rec = json.loads((tmp_path / "dest" / "item4.json").read_text())
    md = (tmp_path / "dest" / "REPORT.md").read_text()
    assert "TBD" not in md.split("## The re-check")[0]
    envs = rec["environments"]
    assert all(envs[m]["gptqcand"]["env"] == "n16k64" and envs[m]["hadamard"]["env"] == "n16k64-fast" for m in I.MODELS)
    assert "equal to part H" in md and "| yes |" in md and "| **no** |" in md and "not bit-reproducible run to run" in md
    for m in I.MODELS:
        rows = rec["rows"][m]
        assert abs(rows["gptq_fo6"]["vs_rtn_fo6"]["wiki"]["delta_milli"] - 2.0) < 1e-9
        assert rows["rtn_fo6"]["vs_rtn_fo6"] is None and rows["hadamard_fq-16x64"]["published_256"]
        # a new row with the RTN FourOverSix answers: 0 pp against it, no discordant problems
        p = rec["gsm8k_paired"][m]["rtn_fq-16x64 vs rtn_fo6"]["strict_match"]
        assert p["n"] == 1319 and rec["gsm8k_paired"][m]["hadamard_fq-16x64 vs rtn_fq-16x64"]["strict_match"]["n"] == 1319
        assert all(rec["reused_agreement"][m][k]["tier"] == "exact" for k in rec["reused_agreement"][m])
        assert abs(rec["recheck"][m]["retrained_minus_recheck"]["wiki"]["delta_milli"] - (4 - 5)) < 1e-9


def test_item6_commands_are_the_runs(tmp_path, monkeypatch):
    got = dry("item6", tmp_path, monkeypatch, drop_out=True)
    ref = json.loads((REF / "commands_item6.json").read_text())["commands"]
    assert len(got) == 2 + 5 * 3 * 2

    def with_map(cmd, mp):
        c = list(cmd)
        c[c.index("--map") + 1] = mp
        return c
    assert got["item6_step0_native_256"] == with_map(ref["aa_native_mistral-7b_16x64"], "{MAP:flipquant_256:mistral-7b:16x64}")
    assert got["item6_step0_fake_256"] == ref["c2_fake_mistral-7b_16x64"]
    for m in ("llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b", "nemotron-nano-9b-v2", "phi4-14b"):
        for u in C.UNITS:
            assert got[f"item6_native_{m}_{u}"] == ref[f"aa_native_{m}_{u}"]
            if m.startswith("llama"):       # C2's command with the model's name (their 256 runs were on another machine)
                want = list(ref[f"c2_fake_mistral-7b_{u}"])
                want[want.index("--model") + 1] = m
            else:
                want = ref[f"c2_fake_{m}_{u}"]
            assert got[f"item6_fake_{m}_{u}"] == with_map(want, f"{{MAP:flipquant_1k:{m}:{u}}}"), (m, u)
            assert got[f"item6_fake_{m}_{u}"][0] == ("{PY_FAST}" if m in C.HYBRID else "{PY}")


def test_item6_report_on_synthetic_records(tmp_path):
    import experiments.paper_1k.item6 as I
    d = tmp_path / "item6"
    for m in I.MODELS:
        for u in C.UNITS:
            nat = I.REF["native_1k_nll"][m][u]
            _ppl_record(d / m / f"{u}_native.json", nat)
            off = 1e-3 if (m, u) == ("phi4-14b", "8x64") else 0.0       # one cell with a constant offset: beyond 2 SE
            _ppl_record(d / m / f"{u}_fake.json", {c: [x - off for x in v] for c, v in nat.items()})
    I.report(tmp_path, tmp_path / "dest")
    rec = json.loads((tmp_path / "dest" / "item6.json").read_text())
    assert rec["cells_n"] == 30 and rec["agree"] == 28
    assert abs(rec["cells"]["phi4-14b"]["8x64"]["wiki"]["delta_milli"] - 1.0) < 1e-9
    assert all(v["tier"] == "exact" for x in rec["native_vs_part_aa"].values() for v in x.values())
    assert "TBD" not in (tmp_path / "dest" / "NATIVE.md").read_text()


def test_two_tier_comparison():
    """compare.tier: exact when equal bit for bit; equivalent when the paired difference is within 2 SE; differs
    otherwise (a constant offset of 1e-3 nats on noisy windows is beyond 2 SE)."""
    from experiments.paper_1k import compare as K
    ref = json.loads((REF / "item6.json").read_text())["step0"]["native_nll"]
    assert K.tier(ref, ref)["tier"] == "exact"
    noisy = {c: [x + (1e-5 if i % 2 else -1e-5) for i, x in enumerate(v)] for c, v in ref.items()}
    t = K.tier(noisy, ref)
    assert t["tier"] == "equivalent" and abs(t["max_abs"] - 1e-5) < 1e-9
    shifted = {c: [x + 1e-3 + (1e-5 if i % 2 else -1e-5) for i, x in enumerate(v)] for c, v in ref.items()}
    assert K.tier(shifted, ref)["tier"] == "differs"
    assert K.overall(K.tier(ref, ref), t) == "PASS-equivalent" and K.overall(K.tier(ref, ref)) == "PASS-exact"
    assert K.overall(t, K.tier(shifted, ref)) == "FAIL"


def test_item7_commands_are_the_runs(tmp_path, monkeypatch):
    got = dry("item7", tmp_path, monkeypatch, drop_out=True)
    ref = json.loads((REF / "commands_item7.json").read_text())["commands"]
    assert len(got) == 1 + 18 + 6
    assert got["item7_step0_readback_256"] == ref["c1_ownership_mistral-7b_16x64"]
    for m in C.MODELS:
        for u in C.UNITS:
            want = list(ref[f"c1_ownership_{m}_{u}"])
            want[want.index("--map") + 1] = f"{{MAP:flipquant_1k:{m}:{u}}}"
            assert got[f"item7_readback_{m}_{u}"] == want, (m, u)
        assert got[f"item7_control_{m}"] == [("{PY_FAST}" if m in C.HYBRID else "{PY}"), "-m",
                                             "experiments.paper_1k.ownership_control", "--model", m, "--map",
                                             f"{{MAP:flipquant_1k:{m}:16x64}}", "--kernel-build-dir", "{BUILD}", "--out", "{OUT}"]


def test_item7_negative_control_logic():
    """ownership_control.control on the CPU, with a stand-in kernel that decodes exactly what is stored (tags
    included): the map's packing passes the check, and both wrong pairs are detected."""
    torch = pytest.importorskip("torch")
    from experiments.paper_1k import ownership_control as O
    from models import sm120 as S
    R = S.package()
    N = R.numerics
    packs = []
    real_pack = R.artifact.pack_module

    def pack(*a, **k):
        pw = real_pack(*a, **k)
        packs.append(pw)
        return pw

    class Exact:
        weight_operand = 1

        def gemm(self, xp, xsf, wp, wsf, t, n, k):
            pw = next(p for p in packs if p.packed is wp)
            dec = N.decode_exact(N.unpack_nibbles(wp), pw.scales, 1.0)
            off = int(torch.nonzero(N.unpack_nibbles(xp)[0])[0])
            return dec[:, off:off + t].t().to(torch.bfloat16)

    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    for unit, tile in (("16x64", torch.zeros(4, 4, dtype=torch.bool)), ("8x64", torch.zeros(8, 4, dtype=torch.bool)),
                       ("256x64", torch.zeros(1, 4, dtype=torch.bool))):
        tile[0, 2] = True
        R.artifact.pack_module = pack
        try:
            rows, passed = O.control(Exact(), w, None, tile, unit)
        finally:
            R.artifact.pack_module = real_pack
        assert passed, (unit, rows)
        assert rows["map_deployed_vs_map"]["format_mismatches"] == 0 and rows["all_e2m1_deployed_vs_map"]["format_mismatches"] > 0


def test_item7_report_has_no_local_paths(tmp_path):
    """The control records hold the map's local path and the build directory; the published report must not (publish's
    scan refused item7.json on the porting machine)."""
    import re
    import experiments.paper_1k.item7 as I
    from experiments.paper_1k import publish as P
    c = tmp_path / "item7" / "control"
    c.mkdir(parents=True)
    rows = {k: dict(exact=k == "map_deployed_vs_map", format_mismatches=0 if k == "map_deployed_vs_map" else 5)
            for k in ("map_deployed_vs_map", "all_e2m1_deployed_vs_map", "map_deployed_vs_all_e2m1")}
    (c / "mistral-7b_16x64.json").write_text(json.dumps(dict(
        status="complete", passed=True, model="mistral-7b", module="model.layers.0.self_attn.q_proj", kernel="mixed_ko",
        map="/vault/x/hf/models--edgeai-lab--Mistral-7B-Instruct-v0.3-FlipQuant-Calib1k/snapshots/" + "a" * 40
            + "/flipquant_16x64.pt",
        routing="auto: 16x64 map -> mixed_ko with its optimized dispatch variants, from /home/u/fq/kernels/sm120/build_deploy",
        rows=rows)))
    I.report(tmp_path, tmp_path / "dest")
    rec = json.loads((tmp_path / "dest" / "item7.json").read_text())
    ctl = rec["controls"]["mistral-7b"]
    assert ctl["map"] == "edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant-Calib1k@aaaaaaaaaaaa/flipquant_16x64.pt"
    assert ctl["routing"] == "auto: 16x64 map -> mixed_ko with its optimized dispatch variants"
    assert not P.scan([tmp_path / "dest" / "item7.json", tmp_path / "dest" / "READBACK.md"])


def test_item7_reference_elements():
    """The per-model weight elements: 3 x their sum is the paper's 181.9 billion (18 release maps); the step-0 totals are
    C1's Mistral-7B 16x64 row."""
    r = json.loads((REF / "item7.json").read_text())
    total = 3 * sum(v["weight_elements"] for v in r["weight_elements"].values())
    assert total == 181_876_162_560 and round(total / 1e9, 1) == 181.9
    assert r["step0"]["informative"] == r["published_256"]["mistral-7b"]["16x64"]["informative"] == 6_382_853_652


def test_item11_cancelled_but_its_commands_kept(tmp_path, monkeypatch):
    """item 11's GPU re-measurement was cancelled by the user (2026-10-26): `run` and `--dry-run` refuse it; the driver's
    commands are still part AA's (checked through the module's own dry run)."""
    with pytest.raises(SystemExit, match="cancelled by the user"):
        dry("item11", tmp_path, monkeypatch, data_root=tmp_path / "data_root", drop_out=True)
    assert "item11" not in RUN.RUN_ORDER and RUN.RUN_ORDER[RUN.RUN_ORDER.index("item9") + 1] == "item12"
    import experiments.paper_1k.item11 as I
    from experiments.paper_1k.jobs import Runner
    monkeypatch.setenv("PAPER1K_PY", "/env/eval/bin/python")
    R = Runner(tmp_path, "item11", dry_run=True)
    I.run(R, tmp_path, dry_run=True, data_root=tmp_path / "data_root")
    rows = [l.split("\t", 1) for l in R.dry.read_text().splitlines()]
    got = {n: K.normalize(c.split(" "), str(tmp_path), ("item11",), str(tmp_path / "data_root"), True) for n, c in rows}
    ref = json.loads((REF / "commands_item11.json").read_text())["commands"]
    assert len(got) == 18 and next(iter(got)) == "item11_train_llama3.2-1b-instruct_16x64"
    for m in C.MODELS:
        for u in C.UNITS:
            assert got[f"item11_train_{m}_{u}"] == ref[f"aa_train_{m}_{u}"], (m, u)


def test_item11_report_from_part_aa_records(tmp_path):
    """The report from the committed records alone (no run directory): Table C.3's spread as the paper computes it,
    (max - min) / min of the trainer's time; the 256 -> 1k ratios (an epoch x4, the step time unchanged)."""
    import experiments.paper_1k.item11 as I
    I.report(tmp_path / "nothing_here", tmp_path / "dest")
    rec = json.loads((tmp_path / "dest" / "item11.json").read_text())
    md = (tmp_path / "dest" / "COST.md").read_text()
    assert "cancelled by the user" in md and "TBD" not in md and (tmp_path / "dest" / "table_calib_cost.tex").exists()
    for m in C.MODELS:
        tr = {u: I.REF["jobs"][m][u]["trainer"]["setup_seconds"] + I.REF["jobs"][m][u]["trainer"]["training_seconds"]
              for u in I.UNITS}
        assert rec["c3"][m]["max_diff_trainer_pct"] == pytest.approx(100 * (max(tr.values()) - min(tr.values())) / min(tr.values()))
        assert all(3.9 < v < 4.1 for v in rec["vs_256"][m]["epoch"].values())
        assert all(0.97 < v < 1.03 for v in rec["vs_256"][m]["step"].values())
        assert rec["rows"][m]["16x64"]["steps_per_epoch"] == 128
    assert rec["rows"]["qwen3.8-27b"]["16x64"]["teacher_devices"] == ["cpu"]
    assert rec["rows"]["mistral-7b"]["16x64"]["teacher_gib"] == pytest.approx(2.926, abs=1e-3)
    assert rec["step0"]["map_equal"] and rec["step0"]["job_seconds"] == 223

def test_item5_commands(tmp_path, monkeypatch):
    """evaluation.gemm_time's documented form, its defaults (the paper's method), the three maps of a model."""
    got = dry("item5", tmp_path, monkeypatch, drop_out=True)
    assert len(got) == 1 + 6

    def want(m, kind):
        return [("{PY_FAST}" if m in C.HYBRID else "{PY}"), "-m", "evaluation.gemm_time", "--model", m, "--maps",
                ",".join(f"{u}={{MAP:{kind}:{m}:{u}}}" for u in C.UNITS), "--kernel-build-dir", "{BUILD}", "--out", "{OUT}"]
    assert got["item5_step0_gemm_256"] == want("phi4-14b", "flipquant_256")
    for m in C.MODELS:
        assert got[f"item5_gemm_{m}"] == want(m, "flipquant_1k")


def _gemm_record(model, scale):
    """A gemm_time record: stock_ko 100 us per forward at every T, each map +scale % (densest +2 x scale)."""
    from evaluation.gemm_time import TOKENS
    s = {}
    for t in TOKENS:
        s[f"stock_ko@{t}"] = 100.0
        s[f"stock_wB_ko@{t}"] = 101.0
        for u in C.UNITS:
            s[f"{u}_typical@{t}"] = 100.0 + scale[u]
            s[f"{u}_densest@{t}"] = 100.0 + 2 * scale[u]
    return dict(status="complete", model=model, per_forward_gemm_us=s, gpu=dict(device="X", driver_version="1"),
                power_limit_w=500, sampler={}, projections={"q_proj": dict(shape=[4096, 4096], modules=32)},
                maps={u: dict(sha256="0" * 64) for u in C.UNITS}, rows=[], flipquant=dict(head="0", clean=True))


def test_item5_report_on_synthetic_records(tmp_path):
    pytest.importorskip("torch")                  # evaluation.gemm_time imports torch (no GPU is touched)
    import experiments.paper_1k.item5 as I
    d = tmp_path / "item5"
    d.mkdir()
    for m in C.MODELS:
        (d / f"{m}.json").write_text(json.dumps(_gemm_record(m, {"8x64": 7.0, "16x64": 1.0, "256x64": 1.5})))
    rec = _gemm_record("phi4-14b", {"8x64": 6.0, "16x64": 0.5, "256x64": 1.0})
    assert I.ge128(rec) == pytest.approx({"8x64_typical": 6.0, "8x64_densest": 12.0, "16x64_typical": 0.5,
                                          "16x64_densest": 1.0, "256x64_typical": 1.0, "256x64_densest": 2.0})
    I.report(tmp_path, tmp_path / "dest")
    r = json.loads((tmp_path / "dest" / "item5.json").read_text())
    assert "records" not in r["summary"] and r["summary"]["table"]["8x64_typical"]["T>=128"] == pytest.approx(7.0)
    assert "| T>=128 | +7.00 (+7.24) |" in (tmp_path / "dest" / "GEMM.md").read_text()


def test_gate_stops_unless_accepted(tmp_path, monkeypatch):
    """Runner.gate: a pass is logged; a failure stops the item, naming PAPER1K_ACCEPT=<gate>; an accepted failure is
    logged as ACCEPTED and the item continues."""
    from experiments.paper_1k.jobs import Runner
    R = Runner(tmp_path, "item6")
    monkeypatch.delenv("PAPER1K_ACCEPT", raising=False)
    assert R.gate("item6-step0", True, "x") is True
    with pytest.raises(SystemExit, match="PAPER1K_ACCEPT=item6-step0"):
        R.gate("item6-step0", False, "the cell differs")
    monkeypatch.setenv("PAPER1K_ACCEPT", "item3-rebuild, item6-step0")
    assert R.gate("item6-step0", False, "the cell differs") is False
    log = (tmp_path / "queue.log").read_text()
    assert "GATE item6-step0: pass" in log and "STOP item6" in log and "ACCEPTED by the runner's user" in log


def test_item1_commands_are_the_runs(tmp_path, monkeypatch):
    """Part AA's zero-shot commands of the 1k rows, with --tasks mmlu (item 1's registered change)."""
    got = dry("item1", tmp_path, monkeypatch, data_root=tmp_path / "data_root", drop_out=True)
    ref = json.loads((REF / "commands_item1.json").read_text())["commands"]
    assert len(got) == 1 + 6 * 4

    def mmlu(cmd):
        c = list(cmd)
        c[c.index("--tasks") + 1] = "mmlu"
        return c
    s0 = mmlu(ref["aa_zs_llama3.2-1b-instruct_fq-16x64"])
    s0[s0.index("--map") + 1] = "{MAP:flipquant_256:llama3.2-1b-instruct:16x64}"
    assert got["item1_step0_mmlu_256"] == s0
    for m in C.MODELS:
        for row in ("fq-8x64", "fq-16x64", "fq-256x64", "gptq"):
            assert got[f"item1_mmlu_{m}_{row}"] == mmlu(ref[f"aa_zs_{m}_{row}"]), (m, row)


def test_accuracy_tier():
    from experiments.paper_1k import compare as K
    n = 14042
    assert K.accuracy_tier(5813 / n, 41.40, n)["tier"] == "exact"          # 41.397 rounds to 41.40
    t = K.accuracy_tier(5850 / n, 41.40, n)                                # +37 questions, inside 2 SE (0.83 pp)
    assert t["tier"] == "equivalent" and t["delta_questions"] == round(5850 - 0.414 * n, 1)
    assert K.accuracy_tier(5950 / n, 41.40, n)["tier"] == "differs"        # +0.97 pp


def test_item1_report_on_synthetic_records(tmp_path):
    import experiments.paper_1k.item1 as I
    for m in C.MODELS:
        for k, row in enumerate(I.ROWS):
            ex = {f"mmlu_x:{i}": float(i % (2 + k) == 0) for i in range(600)}
            f = tmp_path / "item1" / m / f"{row}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            acc = sum(ex.values()) / len(ex)
            f.write_text(json.dumps(dict(status="complete", tasks=dict(mmlu=dict(metrics={"acc,none": acc}, examples=ex)))))
    I.report(tmp_path, tmp_path / "dest")
    rec = json.loads((tmp_path / "dest" / "item1.json").read_text())
    r = rec["rows"]["phi4-14b"]["fq-8x64"]
    zs = I.REF["zeroshot_1k"]["phi4-14b"]["fq-8x64"]
    assert r["mmlu_pct"] == pytest.approx(50.0) and r["avg_pct"] == pytest.approx(100 * (sum(zs.values()) + 0.5) / 7)
    p = rec["paired"]["phi4-14b"]["fq-16x64 vs gptq"]
    assert p["n"] == 600 and p["a_only"] + p["b_only"] > 0
    assert "TBD" not in (tmp_path / "dest" / "MMLU.md").read_text()


def test_item1_deviations_marked(tmp_path, monkeypatch):
    """The 2026-10-26 deviations: ¤ for a job that ran with PAPER1K_ALLOC_CONF (logs/<job>.env.json), ‡b for a reduced
    batch; the consistency rows' table; the batch override's command form."""
    import experiments.paper_1k.item1 as I
    monkeypatch.setenv("PAPER1K_ITEM1_BATCH", "nemotron-nano-9b-v2=8, qwen3.8-27b=4")
    assert I.batch_overrides() == {"nemotron-nano-9b-v2": 8, "qwen3.8-27b": 4}
    cmd = [str(x) for x in I.mmlu_cmd("nemotron-nano-9b-v2", ["--mode", "native"], "o.json", 8)]
    assert cmd[cmd.index("--tasks"):] == ["--tasks", "mmlu", "--batch-size", "8", "--out", "o.json"]
    assert "--batch-size" not in [str(x) for x in I.mmlu_cmd("mistral-7b", [], "o.json")]
    monkeypatch.setenv("PAPER1K_ITEM1_BATCH", "nope=8")
    with pytest.raises(SystemExit):
        I.batch_overrides()
    logs = tmp_path / "logs"
    logs.mkdir()
    for m in C.MODELS:
        for k, row in enumerate(I.ROWS):
            ex = {f"mmlu_x:{i}": float(i % (2 + k) == 0) for i in range(600)}
            f = tmp_path / "item1" / m / f"{row}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            batch = 8 if (m, row) == ("nemotron-nano-9b-v2", "gptq") else I.DEFAULT_BATCH.get(m, 16)
            f.write_text(json.dumps(dict(status="complete", batch_size=batch, tasks=dict(mmlu=dict(
                metrics={"acc,none": sum(ex.values()) / len(ex)}, examples=ex)))))
            if m == "nemotron-nano-9b-v2":
                (logs / f"item1_mmlu_{m}_{row}.env.json").write_text(json.dumps(
                    {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
    c = tmp_path / "item1" / "consistency"
    c.mkdir()
    (c / "nemotron-nano-9b-v2_fq-16x64_256.check.json").write_text(json.dumps(dict(
        model="nemotron-nano-9b-v2", batch=16, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        accuracy_pct=73.30, published_pct=73.34, tier="equivalent", delta_questions=-6, passed=True)))
    I.report(tmp_path, tmp_path / "dest")
    rec = json.loads((tmp_path / "dest" / "item1.json").read_text())
    rows = rec["rows"]["nemotron-nano-9b-v2"]
    assert rows["fq-8x64"]["marks"] == "¤" and rows["gptq"]["marks"] == "¤‡b"
    assert rec["rows"]["mistral-7b"]["fq-8x64"]["marks"] == "" and rec["rows"]["qwen3.8-27b"]["fq-8x64"]["marks"] == ""
    md = (tmp_path / "dest" / "MMLU.md").read_text()
    assert "how it fitted there is not recorded here" in md and "| Nemotron-Nano-9B-v2 | 16 | ¤ | 73.30 | 73.34 |" in md


def test_item1_model_selection(tmp_path, monkeypatch):
    """PAPER1K_ITEM1_MODELS: only those models, in that order, step 0 first; under (A) Nemotron's consistency row comes
    before its rows; the commands are unchanged (the same as the default dry run's)."""
    import experiments.paper_1k.item1 as I
    monkeypatch.setenv("PAPER1K_ITEM1_MODELS", "phi4-14b, qwen3.8-27b,nemotron-nano-9b-v2")
    assert I.models_selected() == ["phi4-14b", "qwen3.8-27b", "nemotron-nano-9b-v2"]
    monkeypatch.setenv("PAPER1K_ITEM1_MODELS", "phi4-14b,nope")
    with pytest.raises(SystemExit, match="unknown models"):
        I.models_selected()
    monkeypatch.delenv("PAPER1K_ITEM1_MODELS")
    assert I.models_selected() == list(C.MODELS)
    data = tmp_path / "data_root"
    full = dry("item1", tmp_path / "a", monkeypatch, data_root=data, drop_out=True)
    monkeypatch.setattr("experiments.paper_1k.item1.models_selected",
                        lambda: ["phi4-14b", "qwen3.8-27b", "nemotron-nano-9b-v2"])
    sel = dry("item1", tmp_path / "b", monkeypatch, data_root=data, drop_out=True)
    names = list(sel)
    assert names[0] == "item1_step0_mmlu_256" and len(names) == 1 + 12
    assert [n.split("_")[2] for n in names[1:]] == ["phi4-14b"] * 4 + ["qwen3.8-27b"] * 4 + ["nemotron-nano-9b-v2"] * 4
    assert all(sel[n] == full[n] for n in names)


def test_item1_dry_run_lists_the_consistency_row(tmp_path, monkeypatch):
    """Under (A) the dry run lists Nemotron's consistency row before its rows (the 256 release map, batch unchanged)."""
    from experiments.paper_1k import jobs as J
    from experiments.paper_1k import run as RN
    import experiments.paper_1k.item1 as I
    monkeypatch.setenv("PAPER1K_ITEM1_MODELS", "nemotron-nano-9b-v2")
    monkeypatch.setenv("PAPER1K_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PAPER1K_PY", "/env/eval/bin/python")
    monkeypatch.setenv("PAPER1K_PY_FAST", "/env/fast/bin/python")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    R = J.Runner(tmp_path, "item1", dry_run=True)
    I.run(R, tmp_path, dry_run=True, data_root=tmp_path / "data")
    names = [l.split("\t", 1)[0] for l in R.dry.read_text().splitlines()]
    assert names[:3] == ["item1_step0_mmlu_256", "item1_consistency_nemotron-nano-9b-v2_fq-16x64_256",
                         "item1_mmlu_nemotron-nano-9b-v2_fq-8x64"] and len(names) == 6
    cons = R.dry.read_text().splitlines()[1].split("\t", 1)[1]
    assert "--batch-size" not in cons and "-FlipQuant@" in cons and "Calib1k" not in cons     # the 256 release map


def test_item1_consistency_gate(tmp_path, monkeypatch):
    """The consistency row's gate: within 2 SE of the published 256 cell passes; beyond stops the item."""
    import experiments.paper_1k.item1 as I
    from experiments.paper_1k import jobs as J
    pub = I.REF["published_256"]["nemotron-nano-9b-v2"]["fq-16x64"]["mmlu"]

    def fake_run(acc):
        def run(self, name, cmd, out, **kw):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(dict(status="complete", tasks=dict(mmlu=dict(metrics={"acc,none": acc})))))
            return 0
        return run

    monkeypatch.setenv("PAPER1K_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(J.Runner, "run", fake_run(pub / 100 + 0.002))
    R = J.Runner(tmp_path, "item1")
    I.consistency(R, tmp_path / "item1", "nemotron-nano-9b-v2", lambda *a: "release.pt", None, False)
    chk = json.loads((tmp_path / "item1" / "consistency" / "nemotron-nano-9b-v2_fq-16x64_256.check.json").read_text())
    assert chk["passed"] and chk["tier"] == "equivalent" and chk["batch"] == 16 and chk["env"]
    monkeypatch.setattr(J.Runner, "run", fake_run(pub / 100 - 0.05))
    with pytest.raises(SystemExit, match="item1-consistency"):
        I.consistency(R, tmp_path / "item1", "nemotron-nano-9b-v2", lambda *a: "release.pt", 8, False)


def test_runner_alloc_conf(tmp_path, monkeypatch):
    """PAPER1K_ALLOC_CONF: the job's PYTORCH_CUDA_ALLOC_CONF (otherwise dropped), recorded per job; unset: dropped."""
    from experiments.paper_1k import jobs as J
    monkeypatch.setattr(J.Runner, "idle", lambda self: None)
    monkeypatch.setattr(J.Runner, "gpu_state", lambda self, job: None)
    monkeypatch.setattr(J, "GPU_GAP_S", 0.0)
    probe = [sys.executable, "-c", "import os; print('ALLOC=' + os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'unset'))"]
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "from-the-shell")
    monkeypatch.delenv("PAPER1K_ALLOC_CONF", raising=False)
    R = J.Runner(tmp_path, "itemx")
    assert R.run("plain", probe, tmp_path / "plain.json") == 0
    assert "ALLOC=unset" in (tmp_path / "logs" / "plain.log").read_text()
    assert not (tmp_path / "logs" / "plain.env.json").exists()
    monkeypatch.setenv("PAPER1K_ALLOC_CONF", "expandable_segments:True")
    assert R.run("dev", probe, tmp_path / "dev.json") == 0
    assert "ALLOC=expandable_segments:True" in (tmp_path / "logs" / "dev.log").read_text()
    assert json.loads((tmp_path / "logs" / "dev.env.json").read_text()) == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert "START dev [PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True]" in (tmp_path / "queue.log").read_text()


def test_item12_commands(tmp_path, monkeypatch):
    """Part AA's calibration command with each row's change before --out (step 0: its 256-window form); part AA's
    native PPL commands; the audits."""
    got = dry("item12", tmp_path, monkeypatch, data_root=tmp_path / "data_root", drop_out=True)
    train = json.loads((REF / "commands_item11.json").read_text())["commands"]["aa_train_mistral-7b_16x64"]
    native = json.loads((REF / "commands_item6.json").read_text())["commands"]["aa_native_mistral-7b_16x64"]
    assert len(got) == 1 + 3 + 6 + 8

    def windows(cmd, n):
        c = list(cmd)
        c[c.index("--fit-windows") + 1] = str(n)
        return c

    def extra(cmd, opts):
        c = list(cmd)
        j = c.index("--out")
        return c[:j] + opts + c[j:]
    assert got["item12_step0_train_256"] == windows(train, 256)
    for row, opts in (("seed1", ["--seed", "1"]), ("seed2", ["--seed", "2"]), ("nondet", ["--no-deterministic"]),
                      ("fineweb-edu", ["--fit-source", "fineweb-edu"]), ("ultrachat", ["--fit-source", "ultrachat"]),
                      ("wikitext2-train", ["--fit-source", "wikitext2-train"])):
        assert got[f"item12_train_{row}"] == extra(train, opts), row
        m = list(native)
        m[m.index("--map") + 1] = f"{{OUT}}/{row}/map.pt"
        assert got[f"item12_ppl_{row}"] == m
    assert got["item12_ppl_paper_1k"] == native
    fo6 = list(native)
    j = fo6.index("--map")
    del fo6[j:j + 2]
    fo6[fo6.index("--weight") + 1] = "fourover6"
    assert got["item12_ppl_fo6"] == fo6
    for src in ("fineweb-edu", "ultrachat", "wikitext2-train"):
        assert got[f"item12_audit_{src}"] == ["{PY}", "-m", "experiments.paper_1k.audit", "--model", "mistral-7b",
                                               "--fit-source", src, "--fit-windows", "1024", "--data-root", "{DATA}",
                                               "--out", "{OUT}"]


def test_item12_audit_logic(monkeypatch):
    """audit.audit: a window sharing a 13-gram with an evaluation text is a hit; the positive control finds every
    spliced item (here: synthetic texts and items, no download)."""
    from experiments.paper_1k import audit as AU
    idx = AU.Index()
    evals = {("gsm8k_test", 3): " ".join(f"q{i}" for i in range(40)), ("wikitext2_test", "test"): " ".join(f"w{i}" for i in range(40))}
    for (name, item), text in evals.items():
        idx.add(name, item, text)
    clean = [" ".join(f"t{j}x{i}" for i in range(100)) for j in range(40)]
    monkeypatch.setattr(AU, "splices", lambda seed=7: [(n, i, t) for (n, i), t in evals.items()])
    r = AU.audit(clean, idx)
    assert r["passed"] and not r["hits"] and r["control"]["ok"] and r["control"]["picked"] == 30
    dirty = list(clean)
    dirty[5] = dirty[5] + " " + " ".join(f"q{i}" for i in range(10, 23))       # exactly one shared 13-gram
    r = AU.audit(dirty, idx)
    assert not r["passed"] and r["windows_with_hits"] == {"gsm8k_test": 1} and r["hits"][0]["window"] == 5


def test_item12_overlap():
    torch = pytest.importorskip("torch")
    import experiments.paper_1k.item12 as I
    a = {"m": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)}
    b = {"m": torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)}
    o = I.overlap(a, b)
    assert o["e0m3_pct"] == 50.0 and abs(o["jaccard"] - 1 / 3) < 1e-12


def test_item13_commands(tmp_path, monkeypatch):
    got = dry("item13", tmp_path, monkeypatch, data_root=tmp_path / "data_root", drop_out=True)
    assert list(got) == ["item13_oneshot_256", "item13_step0_smoke", "item13_gradcheck_256", "item13_oneshot_1k",
                         "item13_gradcheck_1k"]
    one = ["{PY}", "-m", "calibration.tmopt_ext", "oneshot", "--model", "mistral-7b", "--unit", "16x64", "--match"]
    assert got["item13_oneshot_256"] == one + ["{MAP:flipquant_256:mistral-7b:16x64}", "--data-root", "{DATA}", "--out", "{OUT}"]
    assert got["item13_oneshot_1k"] == one + ["{MAP:flipquant_1k:mistral-7b:16x64}", "--data-root", "{DATA}",
                                              "--fit-windows", "1024", "--out", "{OUT}"]
    gc = ["{PY}", "-m", "calibration.gradcheck", "--model", "mistral-7b", "--unit", "16x64", "--oneshot"]
    assert got["item13_gradcheck_256"] == gc + ["{OUT}/256/oneshot.pt", "--map", "{MAP:flipquant_256:mistral-7b:16x64}",
                                                "--fit-windows", "256", "--heldout", "512:64:128", "--data-root", "{DATA}",
                                                "--out", "{OUT}"]
    assert got["item13_step0_smoke"] == got["item13_gradcheck_256"][:-2] + ["--smoke", "16", "--out", "{OUT}"]
    assert got["item13_gradcheck_1k"] == gc + ["{OUT}/1k/oneshot.pt", "--map", "{MAP:flipquant_1k:mistral-7b:16x64}",
                                               "--fit-windows", "1024", "--heldout", "2048:448:128", "--data-root", "{DATA}",
                                               "--out", "{OUT}"]


def _gradcheck_report(g_rho=(0.70, 0.88), map_milli=(-17.5, -26.0), rel=1e-8):
    groups = {f"group_b{i}_{j}": dict(kind="group") for i in range(10) for j in range(3)}
    sets = dict(groups, **{f"topk_{k}": dict(kind="topk", k=k, predicted=-1e-4 * k,
                                             heldout=dict(mean=-6e-5 * k, two_se=5e-4, ratio=0.6),
                                             insample=dict(mean=-9e-5 * k, two_se=5e-4, ratio=0.9)) for k in (10, 100)})
    sets["map"] = dict(kind="map", tiles=75906, predicted=-0.037, heldout=dict(mean=map_milli[0] / 1e3, two_se=1.7e-3, ratio=0.48),
                       insample=dict(mean=map_milli[1] / 1e3, two_se=2e-3, ratio=0.71))
    sp = {sc: {w: dict(rho=r, ci95=[r - 0.3, r + 0.1]) for w, r in zip(("heldout", "insample"), rr)}
          for sc, rr in (("g_u", g_rho), ("act", (-0.2, -0.4)), ("weight_error", (0.05, -0.2)))}
    return dict(status="complete", sets=sets, baseline=dict(fit_mean_kl=0.05, heldout_mean_kl=0.06),
                tiles=dict(total=6_800_000),
                checks=dict(fit_windows_equal_trainer=True, heldout=dict(windows=256, disjoint=True, distinct=True),
                            steps=dict(steps=32, expected=32), baseline_kl=dict(rel_diff=rel)),
                analysis=dict(spearman=sp, beyond_2se={w: dict(groups=6, agree_with_sign_of_g=6) for w in ("heldout", "insample")},
                              middle_bins_pooled={w: dict(mean=-1e-4, two_se=4e-4) for w in ("heldout", "insample")},
                              map_ranks=dict(tiles=75906, in_top_same_size=15942, positive_g=942, median_rank=0.051, bins={})))


def test_item13_step0_criteria():
    import experiments.paper_1k.item13 as I
    assert I.step0_check(_gradcheck_report())["passed"]
    assert not I.step0_check(_gradcheck_report(g_rho=(0.30, 0.88)))["passed"]         # held-out rho below [0.37, 0.92]
    assert not I.step0_check(_gradcheck_report(map_milli=(-15.0, -26.0)))["passed"]   # 2.8 from -17.8 > 1.7
    assert not I.step0_check(_gradcheck_report(rel=1e-5))["passed"]


def test_item13_report(tmp_path):
    import experiments.paper_1k.item13 as I
    for name in I.RUNS:
        f = tmp_path / "item13" / name / "gradcheck.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(_gradcheck_report()))
    (tmp_path / "item13" / "step0.json").write_text(json.dumps(I.step0_check(_gradcheck_report())))
    I.report(tmp_path, tmp_path / "dest")
    md = (tmp_path / "dest" / "GRADCHECK.md").read_text()
    assert "TBD" not in md and "**pass**" in md and "| g_u | +0.70" in md


def test_gradcheck_statistics():
    pytest.importorskip("scipy")
    torch = pytest.importorskip("torch")
    from calibration import gradcheck as GC
    x = list(range(30))
    assert GC.spearman(x, [v * v for v in x]) == pytest.approx(1.0)
    ci, used = GC.bootstrap(x, [-v for v in x], n=200)
    assert ci == [pytest.approx(-1.0), pytest.approx(-1.0)] and used == 200
    p = GC.paired(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 1.5, 2.5]))
    assert p["mean"] == pytest.approx(0.5) and p["two_se"] == pytest.approx(0.0) and p["n"] == 3


def test_item8_commands(tmp_path, monkeypatch):
    got = dry("item8", tmp_path, monkeypatch, drop_out=True)
    assert len(got) == 2 * 4
    for name, kind in (("256", "flipquant_256"), ("1k", "flipquant_1k")):
        for u in C.UNITS:
            assert got[f"item8_random_{name}_{u}"] == ["{PY}", "-m", "experiments.paper_1k.random_maps", "--map",
                                                        f"{{MAP:{kind}:mistral-7b:{u}}}", "--seed", "0", "--out", "{OUT}"]
        maps = ",".join([f"{u}={{MAP:{kind}:mistral-7b:{u}}}" for u in C.UNITS]
                        + [f"{u}_random={{OUT}}/{name}/random_{u}.pt" for u in C.UNITS])
        assert got[f"item8_gemm_{name}"] == ["{PY}", "-m", "evaluation.gemm_time", "--model", "mistral-7b", "--maps", maps,
                                             "--policies", "stock_ko,8x64,16x64,256x64,8x64_random,16x64_random,256x64_random",
                                             "--projections", "o_proj,gate_proj,down_proj", "--kernel-build-dir", "{BUILD}",
                                             "--out", "{OUT}"]
    from evaluation.gemm_time import TOKENS
    import experiments.paper_1k.item8 as I
    assert I.TOKENS == TOKENS                                    # gemm_time's default: the D.5 set


def test_item8_cells_and_random_maps():
    torch = pytest.importorskip("torch")
    import experiments.paper_1k.item8 as I
    from experiments.paper_1k.random_maps import randomize
    rows = []
    for p in I.PROJ:
        for t in I.TOKENS:
            rows.append(dict(config="stock_ko", proj=p, tokens=t, gemm_us=100.0))
            for u in C.UNITS:
                for tag in ("typical", "densest"):
                    rows.append(dict(config=f"{u}_{tag}", proj=p, tokens=t, gemm_us=102.0))
                    rows.append(dict(config=f"{u}_random_{tag}", proj=p, tokens=t, gemm_us=101.5))
    c = I.cells(dict(rows=rows))
    assert all(abs(v - 0.5) < 1e-9 for tag in c.values() for u in tag.values() for p in u.values() for v in p.values())
    tiles = {"a": torch.tensor([[1, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool), "b": torch.zeros(3, 2, dtype=torch.bool)}
    r1, r2 = randomize(tiles, 0), randomize(tiles, 0)
    assert all(torch.equal(r1[n], r2[n]) and int(r1[n].sum()) == int(tiles[n].sum()) for n in tiles)


def test_gemm_time_defaults_unchanged():
    """The random-placement policies are opt-in: gemm_time's default policies (item 5's command) are the five."""
    pytest.importorskip("torch")
    from evaluation import gemm_time as GT
    assert GT.DEFAULT_POLICIES == ("stock_ko", "stock_wB_ko", "8x64", "16x64", "256x64")
    assert GT.map_key("8x64_random") == "8x64_random" and GT.map_key("8x64") == "8x64"


def test_item9_commands(tmp_path, monkeypatch):
    got = dry("item9", tmp_path, monkeypatch, drop_out=True)
    k = ["{PY}", "-m", "experiments.paper_1k.ktile"]
    m256 = ",".join(f"{{MAP:flipquant_256:mistral-7b:{u}}}" for u in C.UNITS)
    m1k = ",".join(f"{{MAP:flipquant_1k:mistral-7b:{u}}}" for u in C.UNITS)
    assert got == {"item9_build": k + ["build"], "item9_census": k + ["census", "--out", "{OUT}"],
                   "item9_check": k + ["check", "--maps256", m256, "--maps1k", m1k, "--out", "{OUT}"],
                   "item9_time": k + ["time", "--maps256", m256, "--maps1k", m1k, "--repeats", "10", "--out", "{OUT}"]}


def _ktile_record(repeats=3):
    """A time.json shaped like ktile's: stock 100 us; every configuration a fixed multiple, plus a per-repeat jitter."""
    import experiments.paper_1k.ktile as KT
    base = {"stock_ko": 100.0, "fp8": 60.0, "ceil8": 104.8, "ceil16": 100.3, "ceil256": 100.3, "nodisp16_t0": 101.2,
            "nodisp16": 102.2, "nodisp8": 105.7}
    mult = {"ktile8": 1.017, "ktile16": 1.008, "k256": 1.009, "paper8": 1.039, "paper16": 1.021, "permma8": 1.6, "permma16": 1.62}
    us = {}
    for name, (kk, kind, u, t) in KT.configurations().items():
        if name in base:
            us[name] = base[name]
            continue
        ref = {"ktile8": "ceil8", "ktile16": "ceil16", "k256": "ceil256", "paper8": "nodisp8", "paper16": "nodisp16",
               "permma8": "ceil8", "permma16": "ceil16"}[kk]
        e = base[ref] * mult[kk]
        us[name] = e * {"e2m1": 1.0, "e0m3": 1.05, "real256": 1.003, "real1k": 1.004}[t]
    reps = []
    for r in range(repeats):
        j = 1 + 0.001 * r
        reps.append(dict(repeat=r, modes={m: {n: dict(us=v * j) for n, v in us.items()} for m in ("iso_cold", "b2b_cold")}))
    return dict(status="complete", repeats_data=reps)


def test_item9_parts_arithmetic():
    import experiments.paper_1k.ktile as KT
    s = KT.summarize(_ktile_record(), "real1k")["iso_cold"]
    a16 = s["parts"]["after"]["16x64"]
    assert a16["tiles"][0] == pytest.approx(0.3) and a16["dispatch"][0] == pytest.approx(0.8)
    assert a16["e0m3"][0] == pytest.approx(0.4) and a16["total"][0] == pytest.approx(100 * (1.003 * 1.008 * 1.004 - 1))
    b8 = s["parts"]["before"]["8x64"]
    assert b8["tiles"][0] == pytest.approx(4.8) and b8["tag_copies"][0] == pytest.approx(100 * (105.7 / 104.8 - 1))
    assert b8["dispatch"][0] == pytest.approx(3.9)
    assert s["parts"]["before"]["256x64"]["tiles"] == s["parts"]["before"]["16x64"]["tiles"]
    assert s["d2"]["16x64"]["speedup"][0] == pytest.approx(1.62 / 1.008) and s["d2"]["16x64"]["speedup"][1] == pytest.approx(0.0)
    assert s["all_e0m3"]["after"]["8x64"][0] == pytest.approx(5.0)
    # a jitter that scales every configuration alike leaves the ratios unchanged (SD 0)
    assert a16["dispatch"][1] == pytest.approx(0.0, abs=1e-9)


def test_item9_report(tmp_path, monkeypatch):
    import experiments.paper_1k.item9 as I
    monkeypatch.setenv("PAPER1K_BUILD_ITEM9", str(tmp_path / "b9"))
    (tmp_path / "b9").mkdir()
    (tmp_path / "b9" / "item9_sass_gate.json").write_text(json.dumps(dict(status="complete", passed=True)))
    d = tmp_path / "item9"
    d.mkdir()
    (d / "time.json").write_text(json.dumps(_ktile_record(10)))
    ref = json.loads((REF / "item9.json").read_text())["census"]
    (d / "census.json").write_text(json.dumps(dict(status="complete", passed=True, kernels=ref)))
    (d / "check.json").write_text(json.dumps(dict(status="complete", passed=True)))
    I.report(tmp_path, tmp_path / "dest")
    md = (tmp_path / "dest" / "KTILE.md").read_text()
    rec = json.loads((tmp_path / "dest" / "item9.json").read_text())
    assert "Table D.2 with the 1k-map tags" in md and "Step 0" in md and rec["gates"]["sass"] is True
    assert "| stock NVFP4 | | +" not in md and "| 16x64 | per K-tile | +" not in md     # µs: absolute, no sign
    assert set(rec["step0"]["offsets"]["d3"]) >= {"after 16x64 dispatch", "before 8x64 tag_copies"}
    assert rec["runs"]["real1k"]["iso_cold"]["parts"]["after"]["16x64"]["dispatch"][0] == pytest.approx(0.8)


def test_item9_sass_census_parser():
    from experiments.paper_1k import sass_census as SC
    sass = """
        Function : _ZN7cutlass13device_kernelIfoo
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   OMMA.SF.16864.F32.E2M1.E2M1 R4, R8, R12, R4 ;
        /*0020*/               @P0 OMMA.SF.16864.F32.E0M3.E2M1 R4, R8, R12, R4 ;
        /*0030*/              @!P0 OMMA.SF.16864.F32.E2M1.E2M1 R4, R8, R12, R4 ;
        /*0040*/               @P1 BRA 0x10 ;
        /*0050*/                   EXIT ;
    """
    c = SC.census_text(sass)
    assert c["omma"] == 3 and c["omma_predicated"] == 2 and c["brx"] == 0
    assert c["omma_per_steady_iteration"] == {"min": 3, "max": 3} and c["tensor_pipe_estimate_4096"] == 3 * 32 * 8 * 1024


def test_item9_reference_builds():
    """The seven extra builds' references: one SASS each; the per-MMA builds allow predicated OMMAs and carry their
    per-K-tile counterparts' defines; every configuration exists in this repository's kernels."""
    import importlib.util
    r = json.loads((REF / "item9.json").read_text())
    spec = importlib.util.spec_from_file_location("k_configs", ROOT / "kernels" / "sm120" / "mixfp4_sm120" / "configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = {b["config"] for b in r["builds"]}
    assert len(names) == 7 and names <= set(mod.CONFIGS)
    for b in r["builds"]:
        assert len(b["sass_sha256"]) == 64 and b["allow_predicated"] == ("permma" in b["config"])
        assert mod.CONFIGS[b["config"]].allow_predicated == ("permma" in b["config"])


def test_latency_is_not_offered():
    """item 2's latency panel is done and published: the driver has no latency job."""
    import experiments.paper_1k.item2 as I
    assert not hasattr(I, "latency") and "latency" not in I.run.__doc__.lower().split("panel")[0]


def test_artifacts_pinned():
    man = json.loads((ROOT / "experiments" / "paper_1k" / "artifacts.json").read_text())["repos"]
    assert len(man) == 6 * 4
    for rid, r in man.items():
        assert re.fullmatch(r"[0-9a-f]{40}", r["revision"]), rid
        pt = {n: f for n, f in r["files"].items() if n.endswith(".pt")}
        assert pt and all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0 for f in pt.values()), rid


def test_artifact_cache_dir(tmp_path, monkeypatch):
    """PAPER1K_HF_CACHE puts the artifacts in their own hub cache (fetch and the dry runs' lookup); unset: HF's own."""
    import huggingface_hub
    from experiments.paper_1k import artifacts as A
    seen = []
    blob = tmp_path / "flipquant_16x64.pt"
    blob.write_bytes(b"x")

    def download(repo, name, revision=None, cache_dir=None):
        seen.append(("download", cache_dir))
        return str(blob)

    def lookup(repo, name, revision=None, cache_dir=None):
        seen.append(("lookup", cache_dir))
        return None

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lookup)
    monkeypatch.delenv("PAPER1K_HF_CACHE", raising=False)
    A.fetch("mistral-7b", "flipquant_1k", "16x64", check=False)
    monkeypatch.setenv("PAPER1K_HF_CACHE", str(tmp_path / "artifacts"))
    A.fetch("mistral-7b", "flipquant_1k", "16x64", check=False)
    assert str(A.local("mistral-7b", "flipquant_1k", "16x64")).startswith("<hf:edgeai-lab/Mistral-7B-Instruct-v0.3-")
    assert seen == [("download", None), ("download", str(tmp_path / "artifacts")), ("lookup", str(tmp_path / "artifacts"))]
    with pytest.raises(SystemExit, match="sha256"):                 # a file whose sha256 is not the pinned one
        A.fetch("mistral-7b", "flipquant_1k", "16x64")


def test_runner_pauses_after_a_gpu_job(tmp_path, monkeypatch):
    """Every job starts at least GPU_GAP_S after the previous GPU job's exit (the mitigation); the command is
    unchanged, and nothing pauses before the first job or after a CPU job."""
    from experiments.paper_1k import jobs as J
    slept, clock = [], [100.0]
    monkeypatch.setattr(J.Runner, "idle", lambda self: None)
    monkeypatch.setattr(J.Runner, "gpu_state", lambda self, job: None)
    monkeypatch.setattr(J.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(J.time, "sleep", lambda s: slept.append(s))
    R = J.Runner(tmp_path, "itemx")
    py = [sys.executable, "-c", "pass"]
    assert R.run("a", py, tmp_path / "a.json", gpu=False) == 0 and slept == []
    assert R.run("b", py, tmp_path / "b.json") == 0 and slept == []          # the first GPU job: no pause
    clock[0] += 1.0                                                           # 1 s after its exit
    assert R.run("c", py, tmp_path / "c.json", gpu=False) == 0 and slept == [pytest.approx(J.GPU_GAP_S - 1.0)]
    clock[0] += 60.0
    assert R.run("d", py, tmp_path / "d.json") == 0 and len(slept) == 1     # long after: no pause
    assert (tmp_path / "logs" / "d.log").read_text().splitlines()[0] == " ".join(py)


def test_sass_gate_keeps_the_gpu_visible(tmp_path, monkeypatch):
    """setup sass runs test_auto_refuses_unoptimized_builds, which loads the libraries (CUDA init): the gate must not
    hide the GPU (phase A: it failed with 'No CUDA GPUs are available' when it did)."""
    from experiments.paper_1k import setup as S
    seen = {}

    def fake_run(cmd, cwd=None, env=None, **kw):
        seen.update(cmd=cmd, env=env)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert S.sass_gate(tmp_path / "build")
    assert "CUDA_VISIBLE_DEVICES" not in seen["env"] and seen["env"]["SM120_BUILD_DIR"] == str(tmp_path / "build")
    assert "sass_matches_reference or refuses_unoptimized" in seen["cmd"]


def test_runner_smoke_limit(tmp_path, monkeypatch):
    """PAPER1K_SMOKE_JOBS=N: the item stops (exit 3, SMOKE in queue.log) before its (N+1)-th started job; finished
    records are skipped and do not count; without it everything runs."""
    from experiments.paper_1k import jobs as J
    monkeypatch.setattr(J.Runner, "idle", lambda self: None)
    monkeypatch.setattr(J.Runner, "gpu_state", lambda self, job: None)
    monkeypatch.setattr(J, "GPU_GAP_S", 0.0)
    rec = tmp_path / "done.json"
    rec.write_text(json.dumps({"status": "complete"}))
    py = [sys.executable, "-c", "pass"]
    monkeypatch.setenv("PAPER1K_SMOKE_JOBS", "1")
    R = J.Runner(tmp_path, "itemx")
    assert R.run("finished", py, rec) is None                     # skipped: not counted
    assert R.run("first", py, tmp_path / "first.json") == 0
    with pytest.raises(SystemExit) as e:
        R.run("second", py, tmp_path / "second.json")
    assert e.value.code == 3 and "SMOKE itemx: stopped before second after 1 jobs" in (tmp_path / "queue.log").read_text()
    assert not (tmp_path / "logs" / "second.log").exists()
    monkeypatch.delenv("PAPER1K_SMOKE_JOBS")
    assert J.Runner(tmp_path, "itemx").run("second", py, tmp_path / "second.json") == 0


def test_setup_windows_redraws_an_unfinished_file(tmp_path, monkeypatch):
    """A windows file that fit_windows' wrapper did not finish (no 'flipquant': the child wrote it, then the draw
    failed) is drawn again; a finished one is only checked."""
    from experiments.paper_1k import setup as S
    drawn = []

    def ref(m, n):
        return json.loads((ROOT / "experiments" / "paper_1k" / "windows" / f"{m}_{n}.json").read_text())

    def fake_run(cmd, cwd=None, env=None, **kw):
        m, n, dst = cmd[cmd.index("--model") + 1], cmd[cmd.index("--fit-windows") + 1], Path(cmd[cmd.index("--out") + 1])
        if dst.exists():                                                  # fit_windows refuses an existing --out
            return type("R", (), {"returncode": 1})()
        drawn.append((m, int(n)))
        dst.write_text(json.dumps(dict(ref(m, int(n)), flipquant="x")))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    w = tmp_path / "windows"
    w.mkdir()
    for m in C.MODELS:
        for n in (256, 1024, 2048):
            (w / f"{m}_{n}.json").write_text(json.dumps(dict(ref(m, n), flipquant="x")))
    m = C.MODELS[1]
    (w / f"{m}_2048.json").write_text(json.dumps(ref(m, 2048)))           # the child's file, the wrapper's write missing
    assert S.windows(tmp_path / "data", tmp_path) and drawn == [(m, 2048)]
    assert "flipquant" not in json.loads((w / f"{m}_2048.unfinished1.json").read_text())   # moved aside, not deleted
    # a draw that fails twice (rc -6 at exit) and then succeeds: retried, then checked
    calls = []

    def flaky(cmd, cwd=None, env=None, **kw):
        calls.append(1)
        if len(calls) < 3:                    # the child writes its file, then aborts at exit
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(ref(m, 1024)))
            return type("R", (), {"returncode": -6})()
        return fake_run(cmd, cwd, env)

    monkeypatch.setattr(S.subprocess, "run", flaky)
    (w / f"{m}_1024.json").write_text(json.dumps(ref(m, 1024)))
    assert S.windows(tmp_path / "data", tmp_path) and len(calls) == 3
    assert sorted(x.name for x in w.glob(f"{m}_1024.unfinished*.json")) == [f"{m}_1024.unfinished{k}.json" for k in (1, 2, 3)]
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": -6})())
    (w / f"{m}_1024.json").write_text(json.dumps(ref(m, 1024)))
    assert not S.windows(tmp_path / "data", tmp_path)                     # every attempt fails: FAIL


def test_dispatch_pattern_share():
    """dispatch_share.pattern0: kernel-opt's per-(warp, K-tile) pattern counting (a warp's granules w and w + 4)."""
    import numpy as np
    from experiments.paper_1k import dispatch_share as D
    m = np.zeros((8, 4), bool)                       # one 128-row panel, two K-tiles
    assert D.pattern0(m, *D.WA128) == (8, 8)
    m[0, 1] = True                                   # granule 0 (warp 0), 64-K column 1 (K-tile 0)
    assert D.pattern0(m, *D.WA128) == (7, 8)
    m[4, 0] = True                                   # granule 4 is warp 0's too, same K-tile: still one pattern
    assert D.pattern0(m, *D.WA128) == (7, 8)
    m[1, 3] = True                                   # warp 1, K-tile 1
    assert D.pattern0(m, *D.WA128) == (6, 8)
    assert D.pattern0(np.zeros((5, 4), bool), *D.WA128) == (8, 8)       # rows padded to a whole panel
    assert D.pattern0(np.zeros((16, 2), bool), *D.WB128) == (8, 8)      # 8 warps, one K-tile


def test_windows_nested():
    for m in C.MODELS:
        w = {n: json.loads((ROOT / "experiments" / "paper_1k" / "windows" / f"{m}_{n}.json").read_text())
             for n in (256, 1024, 2048)}
        z, a, b = (w[n]["token_sha256"] for n in (256, 1024, 2048))
        assert len(z) == 256 and len(a) == 1024 and len(b) == 2048 and len(set(b)) == 2048
        # the record's 128 first, then each source's extension windows nest (256 c 1k c 2k)
        assert z[:128] == a[:128] == b[:128] and set(z) <= set(a) <= set(b)
        assert all(w[n]["fit_extension"]["windows"] == n - 128 == len(w[n]["fit_extension"]["records"]) for n in w)
        assert all(w[n]["token_sha256"][128:] == [r["token_sha256"] for r in w[n]["fit_extension"]["records"]] for n in w)


@pytest.mark.parametrize("item", RUN.ORDER)
def test_every_item_listed(item):
    assert item in RUN.ORDER and (item in RUN.READY) == (ROOT / "experiments" / "paper_1k" / f"{item}.py").exists()
