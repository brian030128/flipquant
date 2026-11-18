"""Native W4A4 MixFP4 GEMM backends of :class:`flipquant.linear.FlipLinear`.

``sm100``       B200 / GB200 (tcgen05, per-N8xK64 weight formats)   libflipquant_sm100.so
``sm120-capi``  RTX Pro 6000 Blackwell / RTX 5090 (patched OMMA)     libflipquant_sm120.so
``sm120-capi-b8`` the same GPUs, weights on CUTLASS operand B (1x8 warp arrangement), 8x64 weight granule
                (this branch's addition)                             libflipquant_sm120_b8.so
``emulate``     a torch reference over the *same packed buffers* the kernels receive, for validating
                packing, layouts and scale handling on any GPU (e.g. H100/H200) or CPU.

On sm_120 the optimized deployment path is not a FlipLinear backend: ``--mode native`` replaces the Linears with
its NativeLinear modules (``models/sm120.py``). Here ``auto`` therefore resolves only on sm_100; on sm_120 the C-API
kernels must be asked for by name (``sm120-capi``, ``sm120-capi-b8``).

All backends compute ``D[M, N] = sum_k dec(a)[m, k] * sa[m, k/16] * dec(b)[n, k] * sb[n, k/16]`` in
FP32 and return BF16; the per-tensor / per-row global scales are applied by the caller. Buffer
layouts are documented in kernels/capi/README.md:

    a     uint8 [M, K/2]    activation nibbles, element 2j in the low nibble
    sa    uint8 [M, K/16]   E4M3 bytes
    b     uint8 [N, K/2]    weight nibbles
    sb    uint8 [N, K/16]   E4M3 bytes
    fmt_b uint8 [N/gr, K/gc] per weight granule, 0 = E0M3, 1 = E2M1
    fmt_a int              uniform activation format (1 = E2M1)

The shared libraries are looked up in ``$FLIPQUANT_KERNEL_LIB`` (a directory), then
``kernels/build*/`` of this checkout.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

from .formats import FMT_E0M3, FMT_E2M1, lut, unpack_nibbles

REPO = Path(__file__).resolve().parents[1]
_BACKENDS: dict[str, "Backend"] = {}


class Backend:
    name = "?"
    granule = (8, 64)

    def gemm(self, a, sa, b, sb, fmt_b, fmt_a: int = FMT_E2M1) -> torch.Tensor:
        raise NotImplementedError


class EmulatedBackend(Backend):
    """Decode the packed operands exactly as the tensor core does, then an FP32 matmul."""

    name = "emulate"

    def __init__(self, granule=(8, 64)):
        self.granule = tuple(granule)

    @staticmethod
    def _decode(packed, scale_bytes, fmt, granule):
        codes = unpack_nibbles(packed).long()
        rows, k = codes.shape
        if isinstance(fmt, int):
            values = lut(fmt, codes.device)[codes]
        else:
            gr, gc = granule
            e0m3 = (fmt == FMT_E0M3).repeat_interleave(gr, 0).repeat_interleave(gc, 1)[:rows]
            values = torch.where(e0m3, lut(FMT_E0M3, codes.device)[codes], lut(FMT_E2M1, codes.device)[codes])
        scale = scale_bytes.view(torch.float8_e4m3fn).float()
        return (values.reshape(rows, -1, 16) * scale.unsqueeze(-1)).reshape(rows, k)

    def gemm(self, a, sa, b, sb, fmt_b, fmt_a: int = FMT_E2M1):
        x = self._decode(a, sa, int(fmt_a), None)
        w = self._decode(b, sb, fmt_b, self.granule)
        return (x @ w.T).to(torch.bfloat16)


class CudaLibBackend(Backend):
    """ctypes binding of kernels/capi (``fq_<arch>_mixfp4_gemm``)."""

    def __init__(self, arch: str, path: Path):
        self.name = arch
        self.lib = ctypes.CDLL(str(path))
        self._gemm = getattr(self.lib, f"fq_{arch}_mixfp4_gemm")
        self._gemm.restype = ctypes.c_int
        self._gemm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_size_t,
                                                                          ctypes.c_void_p]
        self._ws_bytes = getattr(self.lib, f"fq_{arch}_workspace_bytes")
        self._ws_bytes.restype = ctypes.c_size_t
        self._ws_bytes.argtypes = [ctypes.c_int] * 3
        rows, cols = ctypes.c_int(), ctypes.c_int()
        getattr(self.lib, f"fq_{arch}_weight_granule")(ctypes.byref(rows), ctypes.byref(cols))
        self.granule = (rows.value, cols.value)
        self.lib.fq_last_error.restype = ctypes.c_char_p
        self._workspace: dict[torch.device, torch.Tensor] = {}

    def _ws(self, device, nbytes):
        buf = self._workspace.get(device)
        if buf is None or buf.numel() < nbytes:
            buf = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=device)
            self._workspace[device] = buf
        return buf

    def gemm(self, a, sa, b, sb, fmt_b, fmt_a: int = FMT_E2M1):
        m, n, k = a.shape[0], b.shape[0], a.shape[1] * 2
        for t in (a, sa, b, sb, fmt_b):
            assert t.is_cuda and t.dtype == torch.uint8 and t.is_contiguous()
        d = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
        nbytes = int(self._ws_bytes(m, n, k))
        ws = self._ws(a.device, nbytes)
        stream = torch.cuda.current_stream(a.device).cuda_stream
        status = self._gemm(a.data_ptr(), sa.data_ptr(), b.data_ptr(), sb.data_ptr(), fmt_b.data_ptr(),
                            d.data_ptr(), m, n, k, int(fmt_a), ws.data_ptr(), nbytes, stream)
        if status:
            raise RuntimeError(f"{self.name} MixFP4 GEMM failed ({status}): "
                               f"{self.lib.fq_last_error().decode(errors='replace')}")
        return d


def _find_lib(arch: str) -> Path:
    name = CAPI_LIB.get(arch, f"libflipquant_{arch}.so")
    roots = [Path(p) for p in os.environ.get("FLIPQUANT_KERNEL_LIB", "").split(os.pathsep) if p]
    roots += sorted((REPO / "kernels").glob("build*")) + sorted((REPO / "kernels").glob("build*/*"))
    for root in roots:
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"{name} not found; build it (kernels/capi/README.md) and/or set FLIPQUANT_KERNEL_LIB")


def detect_arch(device=None) -> str:
    major, _ = torch.cuda.get_device_capability(device)
    if major == 10:
        return "sm100"
    if major == 12:
        return "sm120"
    raise RuntimeError(f"no native MixFP4 kernel for compute capability {major}.x "
                       "(native needs B200/GB200 = sm_100 or RTX Pro 6000 = sm_120); use --mode fake "
                       "or --native-backend emulate")


#: FlipLinear backend name -> the kernels/capi library architecture
CAPI_ARCH = {"sm100": "sm100", "sm120-capi": "sm120", "sm120-capi-b8": "sm120b8"}
#: library file of an architecture whose name is not libflipquant_<arch>.so
CAPI_LIB = {"sm120b8": "libflipquant_sm120_b8.so"}


def get_backend(name: str = "auto") -> Backend:
    """``auto`` (sm_100 only) | ``sm100`` | ``sm120-capi`` | ``sm120-capi-b8`` | ``emulate``."""
    if name == "auto":
        name = detect_arch()
        if name == "sm120":
            raise RuntimeError("on sm_120, FlipLinear's native backend must be named: --native-backend sm120-capi (the "
                               "C-API kernels); the optimized kernels run through --mode native with models.cli "
                               "(models/sm120.py)")
    if name == "sm120":
        raise RuntimeError("'sm120' is the optimized deployment path (models/sm120.py), not a FlipLinear backend; "
                           "use 'sm120-capi' for the C-API kernels")
    if name not in _BACKENDS:
        if name == "emulate":
            _BACKENDS[name] = EmulatedBackend()
        elif name in CAPI_ARCH:
            backend = CudaLibBackend(CAPI_ARCH[name], _find_lib(CAPI_ARCH[name]))
            backend.name = name             # 'sm120-capi' / 'sm120-capi-b8', never the optimized path's 'sm120'
            _BACKENDS[name] = backend
        else:
            raise ValueError(f"unknown native backend {name!r}: auto, sm100, sm120-capi, sm120-capi-b8 or emulate")
    return _BACKENDS[name]
