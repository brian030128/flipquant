"""The models FlipQuant is evaluated on, and how each one is loaded and quantized.

Every model is loaded in BF16 with SDPA attention. Quantized = every ``nn.Linear`` of the language
model except the output head (and, for Qwen3.8-27B, anything outside ``language_model``, i.e. the
vision tower). Revisions are pinned where a reference calibration record already pinned them; the
others resolve ``main`` at download time and the resolved commit is recorded in every report.

``thinking`` says how to switch reasoning on (``chat_template_kwargs`` for Qwen3's ``enable_thinking``, a system
prompt for Nemotron's ``/think``); the paper's GSM8K runs with thinking off. ``decoding`` records each model card's
recommended sampling settings; the paper decodes greedily.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str | None = None
    #: ``causal_lm`` (AutoModelForCausalLM) or ``qwen3_5`` (Qwen3_5ForConditionalGeneration).
    loader: str = "causal_lm"
    #: Only linears whose qualified name contains this substring are quantized.
    include: str = ""
    #: Batched forward is not bitwise equal to batch 1 (calibration then forces batch 1).
    batch1_only: bool = False
    min_transformers: str = "4.57"
    config_overrides: dict = field(default_factory=dict)
    #: Accuracy harness: how to request reasoning, and the default generation budget.
    thinking: dict = field(default_factory=dict)
    max_new_tokens: dict = field(default_factory=lambda: {"gsm8k": 2048, "aime": 32768})
    #: ``{"think": {...}, "no_think": {...}, "source": ...}``: temperature, top_p, top_k, min_p, presence_penalty.
    decoding: dict = field(default_factory=dict)
    #: ``(hf_id, revision)`` of the vendor's ModelOpt NVFP4 release of this model (``--nvidia-nvfp4``).
    nvfp4_checkpoint: tuple | None = None
    #: Checkpoint key prefix -> module-name prefix, as transformers renames them on load.
    checkpoint_renames: tuple = ()
    notes: str = ""


MODELS: dict[str, ModelSpec] = {m.key: m for m in (
    ModelSpec("phi4-14b", "microsoft/phi-4",
              decoding={"no_think": dict(temperature=0.0), "source": "model card (inference parameters: temperature 0)"},
              notes="Phi3ForCausalLM; fused qkv_proj and gate_up_proj."),
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B",
              thinking={"chat_template_kwargs": {"enable_thinking": True}},
              decoding={"think": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
                        "no_think": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
                        "source": "Qwen3 model card, best practices (generation_config.json: the thinking mode); "
                                  "the non-thinking presence penalty 1.5 of the Qwen3 technical report"}),
    ModelSpec("qwen3-8b", "Qwen/Qwen3-8B",
              thinking={"chat_template_kwargs": {"enable_thinking": True}},
              decoding={"think": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
                        "no_think": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
                        "source": "Qwen3 model card, best practices (generation_config.json: the thinking mode); "
                                  "the non-thinking presence penalty 1.5 of the Qwen3 technical report"}),
    ModelSpec("qwen3.8-27b", "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
              loader="qwen3_5", include="language_model", batch1_only=True, min_transformers="5.16",
              thinking={"chat_template_kwargs": {"enable_thinking": True}},
              decoding={"think": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0),
                        "no_think": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
                        "source": "Qwen3.8-27B model card (generation_config.json: the thinking mode)"},
              nvfp4_checkpoint=("nvidia/Qwen3.8-27B-NVFP4", "482ca0f3832238542f8f5295dde86b5f22711d80"),
              notes="Hybrid Gated-DeltaNet/attention; revision and module set from the reference's qwen27b "
                    "calibration record (496 matrices)."),
    ModelSpec("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3",
              decoding={"no_think": dict(temperature=0.0), "source": "model card (its examples decode with temperature 0)"},
              notes="Instruct variant, so that GSM8K/AIME can be run with a chat template."),
    ModelSpec("mistral-7b-base", "mistralai/Mistral-7B-v0.3", revision="caa1feb0e54d415e2df31207e5f4e273e33509b1",
              notes="The base model, the FlipQuant paper's Mistral, at the paper's revision (mistral-7b is the "
                    "Instruct model)."),
    ModelSpec("nemotron-nano-9b-v2", "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
              revision="6533e8de2c68e4536bf7c411d7a3ce5734111476", min_transformers="5.16",
              thinking={"system": "/think"},
              decoding={"think": dict(temperature=0.6, top_p=0.95), "no_think": dict(temperature=0.0),
                        "source": "model card: temperature 0.6, top_p 0.95 with reasoning, greedy without"},
              nvfp4_checkpoint=("nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4", "8556c9164ddb43fe1f4f4ad730593b3c5e3f7328"),
              checkpoint_renames=(("backbone.", "model."),),
              notes="Hybrid Mamba-2/attention (NemotronH). Mamba in_proj/out_proj are quantized. Keep the "
                    "model in eval(): the fused training path reads out_proj.weight directly."),
    # The model of the reference results (reproduction only).
    ModelSpec("llama3.1-8b", "meta-llama/Llama-3.1-8B"),
    # The paper's Llama-3.1-8B (instruction-tuned; its release map is edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant).
    ModelSpec("llama3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct",
              revision="0e9e39f249a16976918f6564b8830bc894c89659", notes="the paper's Llama-3.1-8B"),
    # Small model for smoke tests of the whole pipeline.
    ModelSpec("llama3.2-1b-instruct", "meta-llama/Llama-3.2-1B-Instruct",
              revision="9213176726f574b556790deb65791e0c5aa438b6",
              max_new_tokens={"gsm8k": 512, "aime": 4096},
              notes="smoke tests; FOCUS reproduction; tab:main-ppl (paper-eval part P)"),
    # FOCUS reproduction (arXiv:2608.01847, its two smallest models).
    ModelSpec("llama3.2-3b-instruct", "meta-llama/Llama-3.2-3B-Instruct",
              revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
              max_new_tokens={"gsm8k": 512, "aime": 4096}, notes="FOCUS reproduction"),
)}

ALIASES = {"phi4": "phi4-14b", "qwen1.7b": "qwen3-1.7b", "qwen8b": "qwen3-8b", "qwen27b": "qwen3.8-27b",
           "mistral7b": "mistral-7b", "nemotron9b": "nemotron-nano-9b-v2", "llama8b": "llama3.1-8b",
           "llama1b": "llama3.2-1b-instruct", "llama3b": "llama3.2-3b-instruct", "llama8b-instruct": "llama3.1-8b-instruct"}


def get(key: str) -> ModelSpec:
    key = ALIASES.get(key, key)
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODELS)} (aliases {sorted(ALIASES)})")
    return MODELS[key]
