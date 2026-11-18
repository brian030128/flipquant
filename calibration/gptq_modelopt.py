"""GPTQ baseline with NVIDIA ModelOpt: standard NVFP4 weights, GPTQ-compensated on our 128 calibration windows.

This is ModelOpt's ``nvfp4_default-kv_none-gptq`` recipe (NVFP4 W4A4, static weight scales, dynamic NVFP4
inputs, GPTQ with layerwise calibration; ModelOpt >= 0.47), with two changes so that it is comparable
with our maps:

* the quantized set is exactly :func:`models.loader.quantizable_linears` (every text-model linear but
  the head), not ModelOpt's default exclusions (which keep e.g. Qwen3.8's ``linear_attn.in_proj_a/b``
  in BF16);
* the calibration data are the 128 math/code windows of 512 tokens that ``calibration/train_map_legacy.py``
  fits on (``flipquant.data.calibration_sets_extended(tok, 128)["fit"]``);
* by default (``--calib-act fourover6``) the activations during GPTQ go through our FourOverSix
  quantizer (per-token scale, a ModelOpt custom backend), so the Hessians see the activations that the
  evaluation uses; ``--calib-act nvfp4`` keeps ModelOpt's dynamic NVFP4.

The export is a ModelOpt NVFP4 checkpoint (``hf_quant_config.json`` + packed codes). Evaluate it with
our activation quantizer, e.g. FourOverSix (the Table 5 GPTQ baseline):

  python -m evaluation.ppl --model nemotron-nano-9b-v2 --weight modelopt \\
      --modelopt-checkpoint runs/nemotron_gptq128/checkpoint --act fourover6

``--ppl`` runs exactly that after the export (and checks that every quantizable linear came back as
NVFP4); ``--upload <repo>`` then pushes the checkpoint to the HuggingFace Hub (private unless
``--public``), with the token read from ``--token-file``.

  python -m calibration.gptq_modelopt --model nemotron-nano-9b-v2 --out runs/nemotron_gptq128 --ppl \\
      --upload edgeai-lab/nemotron-nano-9b-v2-gptq-128

ModelOpt pins ``transformers<5.15`` in its ``hf`` extra, but our models need 5.16, so install the core
package only (``pip install nvidia-modelopt==0.47.0``; its core requirements do not include
transformers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from flipquant import formats as fmt
from flipquant.data import calibration_sets_extended
from models.cli import resources, save_json
from models.loader import load, quantizable_linears

#: ModelOpt's ``configs/numerics/nvfp4_static`` and ``configs/numerics/nvfp4``. The YAML loader turns
#: ``e2m1`` / ``e4m3`` into (exponent, mantissa) tuples; a Python dict must give the tuples itself.
NVFP4_STATIC_WEIGHT = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}
NVFP4_DYNAMIC_INPUT = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
DEFAULT_TOKEN_FILE = Path.home() / ".cache" / "flipquant" / "hf_upload_token"
FOUROVER6_BACKEND = "flipquant_fourover6"


def _fourover6_backend(x: torch.Tensor, quantizer) -> torch.Tensor:
    """ModelOpt functional backend: our FourOverSix activation fake quant (flipquant.formats), the
    quantizer evaluation uses, so GPTQ's Hessians and the next layers' inputs see the deployed
    activations. ``backend_extra_args={"scope": ...}`` picks the global-scale scope."""
    scope = (quantizer.backend_extra_args or {}).get("scope", "row")
    return fmt.fake_quant(x, "fourover6", scope).to(x.dtype)


def quant_config(names: list[str], perc_damp: float, block_size: int, fused: bool,
                 calib_act: str = "fourover6", calib_act_scope: str = "row") -> dict:
    """``calib_act``: the activation quantizer during GPTQ. ``nvfp4`` is ModelOpt's own dynamic NVFP4
    (the recipe as shipped); ``fourover6`` keeps the same NVFP4 input-quantizer attributes (so the
    activation amax is still calibrated and exported as ``input_scale``) but fake-quantizes through
    :func:`_fourover6_backend`."""
    act = dict(NVFP4_DYNAMIC_INPUT)
    if calib_act == "fourover6":
        act.update(backend=FOUROVER6_BACKEND, backend_extra_args={"scope": calib_act_scope})
    elif calib_act != "nvfp4":
        raise ValueError(calib_act)
    quant_cfg = [{"quantizer_name": "*", "enable": False}]
    for n in names:
        quant_cfg.append({"quantizer_name": f"{n}.weight_quantizer", "cfg": NVFP4_STATIC_WEIGHT})
        quant_cfg.append({"quantizer_name": f"{n}.input_quantizer", "cfg": act})
    algorithm = {"method": "gptq", "perc_damp": perc_damp, "block_size": block_size, "fused": fused,
                 "layerwise": {"enable": True}}
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


def model_card(spec, args, report) -> str:
    return f"""---
base_model: {spec.hf_id}
tags: [nvfp4, gptq, modelopt, flipquant]
---
# {Path(args.upload).name if args.upload else spec.key + '-gptq-128'}

GPTQ-quantized NVFP4 weights of [{spec.hf_id}](https://huggingface.co/{spec.hf_id}) (revision
`{report['source']['revision']}`), the GPTQ baseline of the FlipQuant evaluation.

* NVIDIA ModelOpt {report['modelopt']} GPTQ (`nvfp4_default-kv_none-gptq` numerics: NVFP4 W4A4, static
  E4M3 weight block scales, group 16), perc_damp {args.perc_damp}, block size {args.block_size},
  layerwise calibration with quantized inputs from the previous layer. Activations during GPTQ:
  {'FourOverSix (per-' + args.calib_act_scope + ' global scale), the activation quantizer of the evaluation'
   if args.calib_act == 'fourover6' else 'ModelOpt dynamic NVFP4'}.
* Quantized: every text-model linear except the output head ({report['quantized_modules']} modules).
* Calibration: {report['data']['fit']} windows of 512 tokens, half OpenWebMath and half CodeParrot
  (no WikiText or C4), the calibration set of the FlipQuant maps.
* The weights are standard ModelOpt NVFP4 (`hf_quant_config.json`); FlipQuant evaluates them with
  FourOverSix activations.

The license of the base model applies.
"""


def upload(folder: Path, repo: str, token_file: Path, private: bool) -> str:
    from huggingface_hub import HfApi
    token = token_file.read_text().strip()
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model",
                             commit_message="GPTQ NVFP4 checkpoint (ModelOpt, 128 math/code windows)")
    return getattr(info, "oid", None) or str(info)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", type=Path, required=True, help="run directory; the export goes to <out>/checkpoint")
    ap.add_argument("--batch", type=int, default=1, help="calibration sequences per forward")
    ap.add_argument("--perc-damp", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--fused", action="store_true", help="ModelOpt's fused Triton GPTQ kernel")
    ap.add_argument("--calib-act", choices=("fourover6", "nvfp4"), default="fourover6",
                    help="activation quantizer during GPTQ: ours (FourOverSix, as evaluated) or ModelOpt's NVFP4")
    ap.add_argument("--calib-act-scope", choices=("row", "document", "tensor"), default="row",
                    help="FourOverSix global-scale scope during GPTQ (row = per token, as in train_map)")
    ap.add_argument("--limit-windows", type=int, default=None, help="smoke tests only: fewer calibration windows")
    ap.add_argument("--ppl", action="store_true", help="evaluate the export (FourOverSix activations) after it")
    ap.add_argument("--ppl-limit", type=int, default=None, help="--limit for the PPL evaluation (smoke tests)")
    ap.add_argument("--upload", default=None, metavar="REPO", help="HuggingFace repo to push the checkpoint to")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--token-file", type=Path, default=Path(os.environ.get("HF_UPLOAD_TOKEN_FILE", DEFAULT_TOKEN_FILE)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.upload and not args.token_file.is_file():
        raise SystemExit(f"--upload needs a token file, {args.token_file} does not exist")
    if args.upload and args.limit_windows:
        raise SystemExit("refusing to upload a smoke-test checkpoint (--limit-windows)")

    import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    started = time.time()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok, spec = load(args.model, revision=args.revision)
    names = list(quantizable_linears(model, spec))
    fit = calibration_sets_extended(tok, 128)["fit"][:args.limit_windows]
    device = model.get_input_embeddings().weight.device
    report = dict(model=spec.key, source=model.flipquant_source, modelopt=modelopt.__version__,
                  args={k: str(v) for k, v in vars(args).items() if k != "token_file"},
                  data=dict(fit=len(fit), fit_sha256=hashlib.sha256(b"".join(
                      t.cpu().numpy().tobytes() for t in fit)).hexdigest()),
                  quantized_modules=len(names))
    print(f"GPTQ {spec.key}: {len(names)} linears, {len(fit)} calibration windows, modelopt {modelopt.__version__}",
          flush=True)

    @torch.no_grad()
    def forward_loop(m):
        for i in range(0, len(fit), args.batch):
            m(input_ids=torch.cat(fit[i:i + args.batch]).to(device), use_cache=False)

    if args.calib_act == "fourover6":
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import register_quant_backend
        register_quant_backend(FOUROVER6_BACKEND, _fourover6_backend)
    t0 = time.time()
    mtq.quantize(model, quant_config(names, args.perc_damp, args.block_size, args.fused,
                                     args.calib_act, args.calib_act_scope), forward_loop)
    report["gptq_seconds"] = time.time() - t0
    named = dict(model.named_modules())
    missing = [n for n in names if not named[n].weight_quantizer.is_enabled]
    extra = [n for n, m in named.items() if n.endswith("weight_quantizer") and m.is_enabled
             and n[:-len(".weight_quantizer")] not in set(names)]
    if missing or extra:
        raise RuntimeError(f"quantized set differs from quantizable_linears: missing {missing[:4]}, extra {extra[:4]}")
    print(f"GPTQ done in {report['gptq_seconds']:.0f}s", flush=True)

    ckpt = args.out / "checkpoint"
    t0 = time.time()
    export_hf_checkpoint(model, export_dir=ckpt)
    tok.save_pretrained(ckpt)
    report["export_seconds"] = time.time() - t0
    print(f"exported to {ckpt}", flush=True)
    del model
    torch.cuda.empty_cache()

    if args.ppl:
        from evaluation.ppl import main as ppl_main
        ppl_argv = ["--model", spec.key, "--weight", "modelopt", "--modelopt-checkpoint", str(ckpt),
                    "--act", "fourover6", "--out", str(args.out / "ppl.json")]
        if args.revision:
            ppl_argv += ["--revision", args.revision]
        if args.ppl_limit:
            ppl_argv += ["--limit", str(args.ppl_limit)]
        ppl = ppl_main(ppl_argv)
        loaded = ppl["nvfp4_checkpoint"]
        if loaded["bf16_linears"] or loaded["quantized_modules"] != len(names):
            raise RuntimeError(f"export did not round-trip: {loaded['quantized_modules']} NVFP4 linears of "
                               f"{len(names)}, BF16 {loaded['bf16_linears'][:4]}")
        report["ppl"] = {k: v["ppl"] for k, v in ppl["results"].items()}
        report["checkpoint_load"] = {k: loaded[k] for k in ("quantized_modules", "dense_tensors_changed",
                                                             "skipped_tensors", "nvfp4_head")}

    report["resources"] = resources(started)
    save_json(args.out / "report.json", report)
    (ckpt / "README.md").write_text(model_card(spec, args, report))
    (ckpt / "flipquant_gptq.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.upload:
        report["upload"] = dict(repo=args.upload, private=not args.public,
                                commit=upload(ckpt, args.upload, args.token_file, private=not args.public))
        save_json(args.out / "report.json", report)
        print(f"uploaded to https://huggingface.co/{args.upload} ({'public' if args.public else 'private'})", flush=True)
    return report


if __name__ == "__main__":
    main()
