"""Model registry and in-place kernel installation (see registry.py and loader.py)."""
from .registry import ALIASES, MODELS, ModelSpec, get
from .loader import apply_nvfp4_checkpoint, load, quantizable_linears, quantize, weight_sha256

__all__ = ["ALIASES", "MODELS", "ModelSpec", "apply_nvfp4_checkpoint", "get", "load", "quantizable_linears", "quantize", "weight_sha256"]
