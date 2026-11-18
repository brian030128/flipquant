"""Plumbing of ``calibration.train_map``: the paper's TM-OPT+TC trainer is vendored unchanged in ``calibration/tmopt``
(``VENDORED.json``, ``PORT_CHANGES.md``). It runs in its own process, from that directory, with the environment of the
paper's runs.

* ``MODEL_KEYS``: this repository's registry keys -> the trainer's model keys. Only these models have calibration
  records.
* ``prepare``: the calibration record of a model (``<data root>/<key>/calibration/report.json``), written by the
  vendored ``prepare_model_data.prepare(..., fit_only=True)``:
  - Llama-3.1-8B, Qwen3.8-27B and Mistral-7B-v0.3 (``mistral-7b-base``): the fit set comes from their archived record
    (vendored);
  - every other model: drawn with ``campaign.data.builder_seed0``, the rule behind Llama's fit set. That rule must
    first reproduce the Llama, Qwen3-4B and Mistral-7B-v0.3 records exactly (document, offset, token hash).
  Every one of the 128 windows is then re-tokenized and its token hash checked. The model snapshot must be in the
  Hugging Face cache at the paper's pinned revision (``ensure_snapshot`` downloads it).
* ``prepare_development``: for ``--fit-windows``, the development records whose documents the extension rule skips.
* Llama-3.1-8B-Instruct (``llama3.1-8b-instruct``, the trainer's ``llama8b_ins``; NVFP4-RaZeR paper-eval part P): the
  vendored trainer knows the model (``run_multiround``: Llama-3.1-8B's development documents and published windows,
  "Instruct shares the base tokenizer"), but ``prepare_model_data`` does not. ``prepare`` builds its record here
  (``INSTRUCT_PREPARE``), as the reference implementation's ``make_instruct_prior.py`` did: the fit set is drawn with
  ``builder_seed0`` and this model's tokenizer, and it must equal Llama-3.1-8B's archived record (document, offset,
  token hash); the 128 windows are then re-tokenized and checked; the matrices are this model's weights.
  ``prepare_development`` copies Llama-3.1-8B's three development records, as for the base model.
* Llama-3.2-1B-Instruct (``llama3.2-1b-instruct``, the trainer's ``llama1b_ins``; paper-eval part P): neither the
  vendored trainer nor ``prepare_model_data`` knows it (``RUNTIME_MODELS``). It is prepared as the release's other
  models without an archived record (Phi-4, Qwen3-1.7B, ...): the preparation registers it in its own process in
  ``prepare_model_data.MODELS`` with no record, so ``prepare`` draws the fit set with ``builder_seed0`` and this model's
  tokenizer after the rule's three checks, and ``prepare_development`` draws its three development records.
  ``calibration/tmopt_launch.py --register-model`` registers it in the trainer's tables at launch, and
  ``calibration.train_map`` runs the trainer through the launcher for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "calibration" / "tmopt"
MODEL_KEYS = {"llama3.1-8b": "llama8b", "phi4-14b": "phi4", "qwen3.8-27b": "qwen27b", "qwen3-1.7b": "qwen3_1p7b",
              "qwen3-8b": "qwen3_8b", "mistral-7b": "mistral7b_ins", "mistral-7b-base": "mistral7b",
              "nemotron-nano-9b-v2": "nemotron9b", "llama3.1-8b-instruct": "llama8b_ins",
              "llama3.2-1b-instruct": "llama1b_ins"}
# the vendored prepare_model_data.MODELS (the pinned snapshot of each model)
SNAPSHOTS = {"llama8b": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
             "phi4": ("microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"),
             "qwen27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
             "qwen3_1p7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
             "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
             "mistral7b_ins": ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
             "nemotron9b": ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "6533e8de2c68e4536bf7c411d7a3ce5734111476"),
             # the reference implementation's Instruct record (make_instruct_prior.py) pins the same revision
             "llama8b_ins": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
             # the Hub's main as of 2026-10-10 (also this registry's earlier pin)
             "llama1b_ins": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
             # the builder rule's checks read these two models' tokenizers (and Llama-3.1-8B's); Mistral-7B-v0.3 is
             # also the paper's Mistral, mistral-7b-base
             "qwen4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
             "mistral7b": ("mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1")}
# fit sets from an archived record; the others are drawn with the builder rule
RECORDED = {"llama8b", "qwen27b", "mistral7b"}
# Instruct models whose fit set is their base model's archived record, re-drawn with the builder rule and this model's
# tokenizer (INSTRUCT_PREPARE): trainer key -> base trainer key
INSTRUCT_BASE = {"llama8b_ins": "llama8b"}
# models neither the vendored trainer nor prepare_model_data knows, registered at run time as the models without an
# archived record: the fit set by the builder rule, three development draws (DRAWN_DEVELOPMENT), no published record
RUNTIME_MODELS = {"llama1b_ins"}


def resolve(model: str):
    """``(registry spec, the trainer's model key)`` of a registry key or alias."""
    from models.registry import get
    spec = get(model)
    if spec.key not in MODEL_KEYS:
        raise SystemExit(f"{spec.key}: no calibration record for the paper's trainer; one of {sorted(MODEL_KEYS)} "
                         "(other models: calibration.train_map_legacy)")
    key = MODEL_KEYS[spec.key]
    repo, revision = SNAPSHOTS[key]
    if spec.hf_id != repo or (spec.revision and spec.revision != revision):
        raise SystemExit(f"{spec.key}: the registry's {spec.hf_id}@{spec.revision} is not the paper's {repo}@{revision}")
    return spec, key


def data_root(path=None) -> Path:
    if path is not None:
        return Path(path).resolve()
    from flipquant.data import cache_root
    return cache_root() / "tmopt_data"


def env(extra: dict | None = None) -> dict:
    """The environment of the paper's calibrations (the vendored experiments/paper/paper_common.env + ACCURACY_ENV)."""
    e = dict(os.environ)
    paths = [str(ENGINE), str(ROOT / "kernels" / "sm120")] + [p for p in e.get("PYTHONPATH", "").split(":") if p]
    e.update(PYTHONPATH=":".join(paths), CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
             PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if not e.get("HF_HOME"):
        # prepare_model_data.snapshot evaluates os.environ['HF_HOME'] even when HF_HUB_CACHE is set; this is
        # huggingface_hub's own default, so setting it changes nothing else
        e["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    e.update(extra or {})
    return e


def ensure_snapshot(key: str, weights: bool = True) -> None:
    """The pinned snapshot in the Hugging Face cache, downloaded if incomplete: configs and tokenizer, and the
    safetensors weights unless ``weights=False``."""
    from huggingface_hub import snapshot_download
    repo, revision = SNAPSHOTS[key]
    e = env()
    hub = Path(e.get("HF_HUB_CACHE") or Path(e["HF_HOME"]) / "hub")
    snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    have = (snap / "tokenizer_config.json").exists() and (snap / "config.json").exists()
    if weights:
        index = snap / "model.safetensors.index.json"
        if index.exists():
            # every shard: a job sharing this node's cache may still be downloading them (huggingface_hub's own
            # locks make the snapshot_download below wait for it instead)
            shards = set(json.loads(index.read_text())["weight_map"].values())
            have = have and all((snap / s).exists() for s in shards)
        else:
            have = have and (snap / "model.safetensors").exists()
    if not have:
        patterns = ["*.json", "*.model", "*.txt", "*.tiktoken", "*.py", "tokenizer*"] + (["*.safetensors"] if weights else [])
        snapshot_download(repo, revision=revision, cache_dir=hub, allow_patterns=patterns)


PREPARE = """
import json, sys
from pathlib import Path
sys.path.insert(0, {builder!r})
import prepare_model_data as P
from campaign.data import builder_seed0
key, out = {key!r}, Path({out!r})
{register}
checks = P.builder_rule_checks(builder_seed0) if P.MODELS[key]['record'] is None else None
summary = P.prepare(key, out, builder_seed0, fit_only={fit_only!r})
if checks is not None:
    summary['builder_rule_reproduces'] = checks
    (out / key / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
"""


INSTRUCT_PREPARE = """
import json, sys, torch, transformers
from pathlib import Path
sys.path.insert(0, {builder!r})
import prepare_model_data as P
from campaign.data import builder_seed0
from transformers import AutoTokenizer
key, base, out, source, revision = {key!r}, {base!r}, Path({out!r}), {source!r}, {revision!r}
base_path = P.REPO / P.MODELS[base]['record']
base_record = json.loads(base_path.read_text())
recorded = P.fit_spec(base_record['fit'] if 'fit' in base_record else base_record)
tok = AutoTokenizer.from_pretrained(source)
_, built = builder_seed0(tok)
fit = P.fit_spec(built)
assert P.same_fit(fit, recorded), 'the builder rule does not reproduce the base record with this tokenizer'
assert all(fit[s]['token_sha256'] == recorded[s]['token_sha256'] for s in ('math', 'code'))
_, fit_meta = P.math_code_data(tok, fit)          # the 128 windows as the trainer reads them; every token hash asserted
assert P.same_fit(P.fit_spec(fit_meta), fit)
matrices = P.matrices_of(key, source)
fit_from = dict(rule='campaign.data.builder_seed0 (run_pooled_scale.shared_data math/code rule) with this model tokenizer',
                equals_record=dict(path=str(base_path.relative_to(P.REPO)), sha256=P.digest_file(base_path)),
                note='as make_instruct_prior.py: the base model windows apply unchanged (shared tokenizer)')
prior = dict(model=key, source=source, revision=revision, torch_version=torch.__version__,
             transformers_version=transformers.__version__, fit=fit, matrices=matrices,
             local_preparation=dict(script='flipquant calibration/tmopt_common.py INSTRUCT_PREPARE', fit_from=fit_from,
                                    matrices_from='computed from the pinned snapshot (sha256 of each loaded bf16 weight)'))
calibration = out / key / 'calibration'
calibration.mkdir(parents=True, exist_ok=True)
(calibration / 'report.json').write_text(json.dumps(prior, indent=2) + '\\n')
summary = dict(model=key, source=source, revision=revision, fit_from=fit_from, fit_windows_reproduced=128,
               matrices=len(matrices), calibration_report_sha256=P.digest_file(calibration / 'report.json'),
               development='the base model development records (calibration.tmopt_common.prepare_development)')
(out / key / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
"""


def register(key: str) -> str:
    """PREPARE's registration of a ``RUNTIME_MODELS`` key in ``prepare_model_data.MODELS`` (in the preparation's own
    process): no archived record, so the builder rule draws the fit set. Empty for the models it knows."""
    if key not in RUNTIME_MODELS:
        return ""
    repo, revision = SNAPSHOTS[key]
    return f"assert key not in P.MODELS\nP.MODELS[key] = dict(repo={repo!r}, revision={revision!r}, record=None)"


def snapshot_path(key: str) -> Path:
    repo, revision = SNAPSHOTS[key]
    e = env()
    hub = Path(e.get("HF_HUB_CACHE") or Path(e["HF_HOME"]) / "hub")
    return hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def prepare(key: str, root: Path) -> Path:
    """The calibration record for ``key`` under ``root`` (prepared once, then reused)."""
    record = root / key / "calibration" / "report.json"
    summary = root / key / "prepare_summary.json"
    if record.exists() and summary.exists():
        # the record names the model by its snapshot path in the Hugging Face cache; on another node (node-local
        # HF_HOME) that snapshot is absent, and the trainer would read the path as a repo id
        ensure_snapshot(key)
        return record
    ensure_snapshot(key)
    if key in INSTRUCT_BASE:
        code = INSTRUCT_PREPARE.format(builder=str(ENGINE / "research" / "n16k64" / "software" / "primary"), key=key,
                                       base=INSTRUCT_BASE[key], out=str(root), source=str(snapshot_path(key)),
                                       revision=SNAPSHOTS[key][1])
        subprocess.run([sys.executable, "-c", code], cwd=ENGINE, env=env())
        if not (record.exists() and summary.exists()):
            raise SystemExit(f"preparing the calibration record of {key} failed (see the output above)")
        return record
    if key not in RECORDED:
        for other in ("llama8b", "qwen4b", "mistral7b"):
            ensure_snapshot(other, weights=False)
    code = PREPARE.format(builder=str(ENGINE / "research" / "n16k64" / "software" / "primary"), key=key, out=str(root),
                          fit_only=True, register=register(key))
    # prepare_model_data can crash at interpreter exit (PyGILState_Release while the dataset stream is finalized)
    # after writing its outputs, so the outputs, not the exit code, decide
    subprocess.run([sys.executable, "-c", code], cwd=ENGINE, env=env())
    if not (record.exists() and summary.exists()):
        raise SystemExit(f"preparing the calibration record of {key} failed (see the output above)")
    return record


# --fit-windows: Llama-3.1-8B's development records, vendored at their source paths (the trainer's data_paths reads
# them as <data root>/llama8b/<name>; its extension rule needs only each manifest's document hashes)
LLAMA_DEVELOPMENT = {
    "confirmation": "results/task_reorder/transfer_20260920/llama_confirmation/fresh_manifest.json",
    "gate_up_confirm": "results/task_reorder/transfer_20260920/renewed_llama/gate_up_confirm/fresh_manifest.json",
    "tile_refine_confirm": "results/task_reorder/transfer_20260920/renewed_llama/tile_refine_confirm/fresh_manifest.json"}
# Qwen3.8-27B's three development sets (the vendored run_multiround.DEVELOPMENT names): the full preparation redraws
# them and accepts each only by the fresh.pt sha256 that results/mixfp4_potential/qwen27b/multiround_kl_8x64_final.json
# records
QWEN27B_DEVELOPMENT = ("fisher_subset_validate_v2_confirm", "preserved_row_confirm", "ce_target_combinations_confirm")
QWEN27B_RECORD = "results/mixfp4_potential/qwen27b/multiround_kl_8x64_final.json"
DRAWN_DEVELOPMENT = ("fresh_dev1", "fresh_dev2", "fresh_dev3")      # the vendored run_multiround.LOCAL_DEVELOPMENT


def prepare_development(key: str, root: Path) -> list[str]:
    """For ``--fit-windows``: the development records whose documents the extension rule skips (with the paper's fit
    and published C4 documents), under ``root``; returns their manifests.
    - llama8b: its three development records, copied from the vendored manifests;
    - qwen27b: the paper's three development sets, reproduced by the full preparation (``prepare_model_data.prepare``
      without ``fit_only``): it redraws the pilot's six sets in their order (each excluding the published C4 documents
      and the sets before it) and accepts each of the three only by its recorded fresh.pt sha256;
    - every other model: the three development draws of the full preparation, drawn once."""
    if key == "llama8b" or INSTRUCT_BASE.get(key) == "llama8b":
        out = []
        for name, source in LLAMA_DEVELOPMENT.items():
            target = root / key / name / "fresh_manifest.json"
            data = (ENGINE / source).read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise SystemExit(f"{target} differs from the vendored {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            out.append(str(target))
        return out
    names = QWEN27B_DEVELOPMENT if key == "qwen27b" else DRAWN_DEVELOPMENT
    manifests = [root / key / name / "fresh_manifest.json" for name in names]
    if not all(m.exists() for m in manifests):
        # the full preparation re-runs the builder rule, which reads these tokenizers (and the model's own snapshot);
        # prepare() fetches them only when it builds the record, so a record made by an earlier job lacks them here
        ensure_snapshot(key)
        if key not in RECORDED:
            for other in ("llama8b", "qwen4b", "mistral7b"):
                ensure_snapshot(other, weights=False)
        code = PREPARE.format(builder=str(ENGINE / "research" / "n16k64" / "software" / "primary"), key=key,
                              out=str(root), fit_only=False, register=register(key))
        subprocess.run([sys.executable, "-c", code], cwd=ENGINE, env=env())
        if not all(m.exists() for m in manifests):
            raise SystemExit(f"drawing the development records of {key} failed (see the output above)")
    return [str(m) for m in manifests]


def transformers_deviation(record: Path) -> list[str]:
    """``--transformers-deviation`` when the record was made with another transformers version (the trainer then
    records the deviation in its report; Llama-3.1-8B's record was made with 4.57.3)."""
    import transformers
    version = json.loads(record.read_text()).get("transformers_version")
    return ["--transformers-deviation"] if version and version != transformers.__version__ else []


def run(cmd: list, log: Path) -> int:
    """One vendored command from calibration/tmopt; its output goes to the terminal and to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        f.write(" ".join(str(c) for c in cmd) + "\n")
        f.flush()
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ENGINE, env=env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def expected_shapes(spec, revision) -> dict:
    """``{module: (out_features, in_features)}`` of :func:`models.loader.quantizable_linears` on a meta-device model
    built from the registry's config at ``revision`` (CPU only, no weights)."""
    import torch
    import transformers
    from models.loader import quantizable_linears
    config = transformers.AutoConfig.from_pretrained(spec.hf_id, revision=revision)
    if spec.loader == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        cls = transformers.AutoModelForCausalLM
    with torch.device("meta"):
        model = cls.from_config(config) if hasattr(cls, "from_config") else cls._from_config(config)
    return {n: (m.out_features, m.in_features) for n, m in quantizable_linears(model, spec).items()}
