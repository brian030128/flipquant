"""Hooks on the reference implementation's TM-OPT+TC trainer (``calibration/tmopt/run_train_map.py``, vendored unchanged), installed
from outside the vendored directory.

calibration.tmopt_ext runs this file in the trainer's own process, exactly as calibration.train_map runs
run_train_map.py (cwd ``calibration/tmopt``, ``tmopt_common.env()``)::

    python <repo>/calibration/tmopt_launch.py [hook options] -- <run_train_map.py arguments>

**Why a launcher.** ``run_train_map.main()`` is a single function without extension points, and the trainer runs in
its own process. The launcher imports ``run_train_map`` as a module, replaces module-level names that ``main()``
looks up while it runs, sets ``sys.argv`` to the trainer's arguments and calls ``main()``. Nothing in
``calibration/tmopt`` is edited (``VENDORED.json``'s sha256 values are checked by tests/test_stage3.py),
and without hook options the run is run_train_map.py itself (tests/test_tmopt.py, acceptance test (a)).

Hooks (each replaces one name; the trainer's own code does everything else):

``--record-grads DIR [--record-steps N]``
    Passive. ``torch.optim.Adam`` becomes a subclass whose ``step()`` first saves every logit's ``.grad`` -- the
    trainer's accumulated tile gradient of that optimizer step (``p.grad = grads[n].clone()``) -- to
    ``DIR/step0001.pt``, ... (the first N steps), then runs Adam's own step. The training is unchanged.
``--oneshot OUT [--oneshot-steps N]``
    One-shot gradient. The same subclass adds each step's gradient to an FP32 running sum and SKIPS the update, so
    the logits stay at ``--init-logit`` (all E2M1 for -1) and every accumulation group is evaluated at the start
    map. With ``--epochs 1`` that is one pass over the 128 fit windows, in the trainer's own shuffled order,
    micro-batches and accumulation groups. ``OUT`` holds ``{module: summed gradient}`` (the sum of the steps'
    gradients; each step's is the gradient of that group's mean KL), the first step's gradient alone and the step
    count. ``--oneshot-steps N`` stops after N steps (tests).
``--rotate hadamard --rotate-block B``
    The map is learned in the Hadamard basis of flipquant/rotate.py (block size: the largest power of two dividing
    K, at most B; B = 1 is the identity, acceptance test (c)). ``run_train_map.sha`` -- which the trainer calls
    exactly once per quantized module, on its BF16 weight, right after the teacher's log-probabilities are computed
    and right before the module's candidates are built -- becomes a wrapper that returns the hash of the original
    weight (so the trainer's per-matrix check of the checkpoint is unchanged), then rotates that weight in place,
    ``W <- W H`` (``RotatedLinear.from_linear``'s arithmetic: FP32 matmul, cast back to BF16), and gives the module a
    forward pre-hook ``x -> x H`` (``flipquant.rotate.Hadamard``). So both candidates (FourOverSix and E0M3) are
    quantized from ``W H``; the input is rotated before the trainer's own pre-hook quantizes it per token (hooks run
    in registration order; the trainer registers its own per epoch, later); the tile gradient's ``G = dy^T x`` sees
    the rotated, quantized input; and the teacher stays the unrotated BF16 model. The launcher checks that every
    module was rotated.
``--candidates-e2m1 SRC --candidates-e0m3 SRC``
    External candidates (paper-sm120-runs; NVFP4-RaZeR paper-eval part O). ``SRC`` is a flipquant-gptq/1 codes file
    (flipquant/gptq.py's ``--gptq-cache`` format: per module the packed nibbles, E4M3 scale bytes, global scale and
    block formats; FourOverSix codes for E2M1, mixfp4 codes with every block E0M3 for E0M3) or ``rtn``
    (flipquant.formats' own RTN codes of the module's BF16 weight). ``run_train_map.sha`` -- called once per quantized
    module right before its candidates are built (the --rotate hook uses the same place) -- becomes a wrapper that
    records the module and prepares its two candidates. The trainer's candidate quantizers ``quant_nvfp4_4over6`` /
    ``quant_mix_4_6`` (module-level names of run_train_map) then return, for that module's weight, the given E2M1 /
    E0M3 candidate decoded exactly as the lean store decodes it (``native_dev.decode_packed``); and the lean store's
    weight encoders ``rq.weight_four_over_six`` / ``rq.weight_e0m3`` -- whose codes and scales
    ``native_dev.pack_candidates`` packs -- return the given codes (replaced at the first module, after the trainer has
    imported them). With code files, run_train_map's RTN pre-packer ``pack`` is replaced for that module too: it
    returns the given codes in ``packed_candidates.encode``'s layout, checked with the trainer's own decoders, so the
    module takes the trainer's packed path instead of its dense fallback (which keeps both BF16 candidates of every
    module on the GPU until the loop ends). The trainer's own check, that each packed candidate decodes bit for bit to
    the candidate it holds, runs unchanged. Guards: every module's given E4M3 scales and global scale equal the RTN grid of its BF16 weight
    (GPTQ on flipquant's fixed grid keeps it, so codes on any other grid fail); with ``rtn`` both candidates equal the
    trainer's own quantizers' outputs (E2M1 bit for bit; E0M3 up to the sign of zero, which the store does not keep)
    and the given codes equal rq's; every module gets both candidates, and both encodes in the lean store, exactly once.
    ``<--out>/candidates_hook.json`` records the sources and the per-module checks. Without these options nothing is
    replaced.
``--register-model KEY``
    A model the vendored trainer does not know (``calibration.tmopt_common.RUNTIME_MODELS``; NVFP4-RaZeR paper-eval
    part P: Llama-3.2-1B-Instruct, ``llama1b_ins``), registered as the trainer's models without an archived record
    (``run_multiround.LOCAL_DEVELOPMENT``: Phi-4, Qwen3-1.7B, ...): its three development draws
    (``tmopt_common.DRAWN_DEVELOPMENT``, written by ``prepare_development``; the extension rule reads their document
    hashes) go into ``run_multiround.LOCAL_DEVELOPMENT``, and the key is added to ``MODELS`` (``run_multiround.MODELS``
    and run_train_map's imported name, which its ``--model`` choices read). ``PUBLISHED`` is left alone, so the trainer
    records the evaluation windows by token hash. Refused for a model the trainer knows, or for a key other than the
    trainer's ``--model``. ``calibration.train_map`` passes it for those models; without it nothing is registered.
``--save-theta PATH``
    Passive. ``torch.optim.Adam`` becomes a subclass that only remembers its parameters (the trainer's tile logits
    theta, in module order); after the trainer returns, ``PATH`` holds ``{module: theta}`` after the last optimizer
    step, the values whose sign is the final map (theta > 0 -> E0M3). The trainer saves theta itself only when it draws
    a development set; this records it for a ``--no-dev`` run without changing anything (NVFP4-RaZeR paper-eval part AA
    amendment 4: how much of a map sits near theta = 0). Nothing else is replaced.
``--teacher-host``
    The top-K teacher (``run_train_map.py --teacher-topk K``) in host memory instead of on the GPU (NVFP4-RaZeR
    paper-eval part AA: 1,024 and 2,048 fit windows, about 3 MB per window, do not fit next to Qwen3.8-27B on one
    96 GB GPU). ``topk_teacher.topk_rows`` -- which the trainer calls once per fit window and whose rows it keeps --
    becomes a wrapper that copies its three tensors (BF16 values, int32 ids, FP32 tail log-mass) to the host. The
    losses read the rows only through ``topk_teacher.batch_rows``, which concatenates a micro-batch's rows and moves
    them to the logits' device, so the loss sees the same values and the map is the GPU-teacher map. The trainer's
    ``teacher_storage`` records the rows' device. Refused without ``--teacher-topk``.
``--tiny-layers N``
    TESTS ONLY. The model loaders the trainer calls (``run_train_map.load_model``;
    ``transformers.Qwen3_5ForConditionalGeneration.from_pretrained`` for Qwen3.8-27B) return
    ``models.tiny.build(...)``'s randomly initialized N-layer model instead of the checkpoint; its calibration
    record (written by ``calibration.tmopt_ext record --tiny-layers N``) holds that model's weight hashes, so the
    trainer's own per-matrix weight check still runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "calibration" / "tmopt"


class _Stop(Exception):
    """Raised from the optimizer to end a run after ``--oneshot-steps``."""


def _parse(argv):
    if "--" not in argv:
        raise SystemExit("usage: tmopt_launch.py [hook options] -- <run_train_map.py arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="tmopt_launch.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-grads", type=Path, default=None)
    ap.add_argument("--record-steps", type=int, default=None)
    ap.add_argument("--oneshot", type=Path, default=None)
    ap.add_argument("--oneshot-steps", type=int, default=None)
    ap.add_argument("--rotate", choices=("none", "hadamard"), default="none")
    ap.add_argument("--rotate-block", type=int, default=16)
    ap.add_argument("--candidates-e2m1", default=None, help="a flipquant-gptq/1 codes file (FourOverSix) or 'rtn'")
    ap.add_argument("--candidates-e0m3", default=None, help="a flipquant-gptq/1 codes file (all-E0M3 mixfp4) or 'rtn'")
    ap.add_argument("--register-model", default=None, help="a model the vendored trainer does not know")
    ap.add_argument("--teacher-host", action="store_true", help="the top-K teacher's rows in host memory")
    ap.add_argument("--save-theta", type=Path, default=None, help="save the tile logits after the last step")
    ap.add_argument("--tiny-layers", type=int, default=None)
    ext = ap.parse_args(argv[:i])
    if (ext.candidates_e2m1 is None) != (ext.candidates_e0m3 is None):
        raise SystemExit("--candidates-e2m1 and --candidates-e0m3 go together")
    if ext.candidates_e2m1 is not None and ext.rotate != "none":
        raise SystemExit("--candidates-* and --rotate are separate hooks; use one")
    trainer = argv[i + 1:]
    if trainer and trainer[0].endswith("run_train_map.py"):
        trainer = trainer[1:]
    if ext.teacher_host and int(_trainer_arg(trainer, "--teacher-topk", 0)) <= 0:
        raise SystemExit("--teacher-host: the full-vocabulary teacher is already on the host; pass --teacher-topk K")
    return ext, trainer


def _trainer_arg(trainer, flag, default=None):
    return trainer[trainer.index(flag) + 1] if flag in trainer else default


def register_model(R, trainer, key: str) -> None:
    """``--register-model KEY`` (module docstring): KEY in the trainer's tables, as its models without a record."""
    import run_multiround as RM
    from calibration.tmopt_common import DRAWN_DEVELOPMENT, RUNTIME_MODELS      # this repository: on the path (main)
    if key not in RUNTIME_MODELS:
        raise SystemExit(f"--register-model {key}: not one of calibration.tmopt_common.RUNTIME_MODELS")
    if key in RM.MODELS or key in RM.LOCAL_DEVELOPMENT or key in RM.PUBLISHED:
        raise SystemExit(f"--register-model {key}: the vendored trainer knows it")
    if _trainer_arg(trainer, "--model") != key:
        raise SystemExit(f"--register-model {key}: the trainer's --model is {_trainer_arg(trainer, '--model')}")
    RM.LOCAL_DEVELOPMENT[key] = DRAWN_DEVELOPMENT
    RM.MODELS = R.MODELS = RM.MODELS + (key,)
    print(f"REGISTER {key}: run_multiround.LOCAL_DEVELOPMENT {list(DRAWN_DEVELOPMENT)}, no published record", flush=True)


def install_theta_capture() -> dict:
    """``--save-theta`` (module docstring): ``torch.optim.Adam`` -> a subclass that remembers its parameters. It wraps
    whatever Adam is at the time (the other optimizer hooks' subclass too), and its step is the base class's."""
    import torch
    state = dict(params=None)
    base = torch.optim.Adam

    class Adam(base):
        def __init__(self, params, *args, **kwargs):
            super().__init__(params, *args, **kwargs)
            state["params"] = [p for g in self.param_groups for p in g["params"]]

    torch.optim.Adam = Adam
    return state


def save_theta(state: dict, names: list, path: Path) -> None:
    import torch
    params = state["params"]
    if params is None or len(params) != len(names):
        raise SystemExit(f"--save-theta: {0 if params is None else len(params)} optimizer parameters, {len(names)} modules")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu().clone() for n, p in zip(names, params)}, path)
    print(f"SAVE-THETA {len(names)} modules -> {path}", flush=True)


def install_teacher_host(T) -> None:
    """``--teacher-host`` (module docstring): ``T.topk_rows`` (T = the trainer's ``topk_teacher``) keeps its rows on
    the host."""
    original = T.topk_rows

    def topk_rows(lp, k):
        rows, mass = original(lp, k)
        return tuple(x.cpu() for x in rows), mass

    T.topk_rows = topk_rows
    print("TEACHER-HOST topk_teacher.topk_rows: rows copied to the host", flush=True)


def module_names(R, trainer) -> list[str]:
    """The trainer's quantized modules in its own order: ``run_train_map`` asserts ``list(modules) ==
    list(prior['matrices'])``, and its logits (``theta``) and optimizer parameters follow ``modules``."""
    root = _trainer_arg(trainer, "--data-root")
    calibration, _ = R.data_paths(_trainer_arg(trainer, "--model", "llama8b"), root and Path(root))
    return list(json.loads((calibration / "report.json").read_text())["matrices"])


def install_optimizer_hooks(ext, names):
    """``torch.optim.Adam`` -> a subclass that records / accumulates (and with --oneshot skips) each step."""
    import torch
    state = dict(steps=0, sum=None, first=None)

    class Adam(torch.optim.Adam):
        def step(self, closure=None):
            params = [p for g in self.param_groups for p in g["params"]]
            if len(params) != len(names):
                raise RuntimeError(f"{len(params)} optimizer parameters, {len(names)} modules")
            state["steps"] += 1
            grads = [p.grad for p in params]
            if ext.record_grads is not None and (ext.record_steps is None or state["steps"] <= ext.record_steps):
                ext.record_grads.mkdir(parents=True, exist_ok=True)
                torch.save({n: g.detach().cpu() for n, g in zip(names, grads)},
                           ext.record_grads / f"step{state['steps']:04d}.pt")
            if ext.oneshot is None:
                return super().step(closure)
            if state["sum"] is None:
                state["first"] = [g.detach().clone() for g in grads]
                state["sum"] = [g.detach().float().clone() for g in grads]
            else:
                for s, g in zip(state["sum"], grads):
                    s += g.detach().float()
            if ext.oneshot_steps is not None and state["steps"] >= ext.oneshot_steps:
                raise _Stop
            return None                         # no update: the logits stay at the start map

    torch.optim.Adam = Adam
    return state


def install_rotation(R, captured: dict, max_block: int) -> list:
    """``run_train_map.sha`` -> hash, then ``W <- W H`` and the input pre-hook ``x -> x H`` (module docstring)."""
    import torch
    from flipquant.rotate import Hadamard, block_size
    original_sha = R.sha
    rotated = []

    def sha(t):
        digest = original_sha(t)                    # the checkpoint's weight, as the trainer checks it
        hit = captured.get("weights", {}).get(id(t))
        if hit is not None:
            name, module = hit
            rot = Hadamard(module.in_features, block_size(module.in_features, max_block))
            with torch.no_grad():
                t.copy_(rot(t.float()).to(t.dtype))
            module.register_forward_pre_hook(lambda mod, args, rot=rot: (rot(args[0]), *args[1:]))
            rotated.append(name)
        return digest

    R.sha = sha
    return rotated


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class Candidates:
    """``--candidates-e2m1`` / ``--candidates-e0m3`` (module docstring). Installs itself on construction; ``finish()``
    checks that every module got both candidates once and writes the record."""

    KINDS = ("e2m1", "e0m3")

    def __init__(self, R, captured: dict, sources: dict, names: list, out: Path | None):
        sys.path.append(str(ROOT))                 # flipquant (the launcher's own repository), after the trainer's paths
        from flipquant.gptq import load_codes_file
        self.R, self.captured, self.sources, self.names, self.out = R, captured, dict(sources), names, out
        self.files = {k: load_codes_file(Path(src)) for k, src in self.sources.items() if src != "rtn"}
        self.orig = dict(sha=R.sha, quant_nvfp4_4over6=R.quant_nvfp4_4over6, quant_mix_4_6=R.quant_mix_4_6, pack=R.pack)
        self.orig_rq, self.lean = {}, None
        self.current = None                        # (name, weight, {kind: (values, scales, global scale)})
        self.modules, self.other_calls = {}, dict.fromkeys(("quant_nvfp4_4over6", "quant_mix_4_6", "pack",
                                                             "weight_four_over_six", "weight_e0m3"), 0)
        R.sha, R.quant_nvfp4_4over6, R.quant_mix_4_6 = self.sha, self.quant_nvfp4_4over6, self.quant_mix_4_6
        R.pack = self.pack

    # -- the store's modules: imported by the trainer itself (lean memory mode), else here
    def _realquant(self):
        if "rq" not in sys.modules or "native_dev" not in sys.modules:
            sys.path.append(str(ENGINE / "repro_local" / "realquant"))
            import native_dev  # noqa: F401
        return sys.modules["native_dev"], sys.modules["rq"]

    def _patch_rq(self):
        if self.lean is not None:
            return
        self.lean = "rq" in sys.modules and "candidate_store" in sys.modules     # the trainer built a lean store
        if self.lean:
            rq = sys.modules["rq"]
            self.orig_rq = dict(weight_four_over_six=rq.weight_four_over_six, weight_e0m3=rq.weight_e0m3)
            rq.weight_four_over_six, rq.weight_e0m3 = self.weight_four_over_six, self.weight_e0m3

    def sha(self, t):
        digest = self.orig["sha"](t)                # the checkpoint's weight, as the trainer checks it
        hit = self.captured.get("weights", {}).get(id(t))
        if hit is not None:
            self._patch_rq()
            if self.current is not None:
                raise SystemExit(f"--candidates: {self.current[0]} did not get all its candidates")
            self.current = (hit[0], t, self._prepare(hit[0], t))
        return digest

    def _prepare(self, name, w):
        import torch
        from flipquant import formats as F
        if name in self.modules:
            raise SystemExit(f"--candidates: {name} seen twice")
        rtn = dict(e2m1=F.fourover6(w), e0m3=F.e0m3(w))
        rec = self.modules[name] = dict(calls={})
        out = {}
        for k in self.KINDS:
            r = rtn[k]
            r_scale, r_gs = r.scale.float(), r.gscale.float().reshape(())
            if k not in self.files:
                codes, scale, gs = r.codes, r_scale, r_gs
            else:
                e = self.files[k][1].get(name)
                if e is None:
                    raise SystemExit(f"--candidates-{k}: {name} is not in {self.sources[k]}")
                f = e["fmt"]
                uniform = not torch.is_tensor(f)
                if k == "e2m1" and not (uniform and int(f) == F.FMT_E2M1):
                    raise SystemExit(f"--candidates-e2m1: {name}: not uniform E2M1 codes")
                if k == "e0m3" and not (bool(f.all()) if not uniform else int(f) == F.FMT_E0M3):
                    raise SystemExit(f"--candidates-e0m3: {name}: not every block is E0M3")
                codes = F.unpack_nibbles(e["q"]).to(w.device)
                scale = e["s"].view(torch.float8_e4m3fn).float().to(w.device)
                gs = e["g"].float().reshape(()).to(w.device)
                if codes.shape != w.shape:
                    raise SystemExit(f"--candidates-{k}: {name}: codes {tuple(codes.shape)}, weight {tuple(w.shape)}")
                # the fixed grid: GPTQ re-rounds on the RTN grid of the original weight and never changes it
                if not (torch.equal(scale, r_scale) and torch.equal(gs, r_gs)):
                    raise SystemExit(f"--candidates-{k}: {name}: the scales are not the RTN grid of this weight")
            mag, neg = (codes & 7).long(), (codes & 8).bool()
            lv = (torch.tensor(E2M1_LEVELS, dtype=torch.float32, device=w.device)[mag] if k == "e2m1"
                  else mag.float())
            out[k] = (torch.where(neg, -lv, lv), scale, gs)        # rq's representation: signed code values
            rec[k] = dict(codes_changed_vs_rtn=float((codes != r.codes).float().mean()))
        return out

    def _mine(self, w):
        return self.current is not None and w is self.current[1]

    def _want(self):
        """The hooked calls one module goes through: the two quantizers, the pre-packer (code files only), and the
        lean store's two encoders."""
        want = {"quant_e2m1", "quant_e0m3"} | ({"pack"} if self.files else set())
        return want | ({"rq_e2m1", "rq_e0m3"} if self.lean else set())

    def _done(self, what):
        name = self.current[0]
        calls = self.modules[name]["calls"]
        calls[what] = calls.get(what, 0) + 1
        if set(calls) == self._want():
            self.current = None                    # releases the weight and the prepared candidates

    def _decode(self, k):
        """The candidate as the lean store decodes it (native_dev.pack_candidates' own check)."""
        nd, rq = self._realquant()
        v, s, g = self.current[2][k]
        nib = nd.e2m1_nibbles(v) if k == "e2m1" else rq.e0m3_nibbles(v)
        sb = rq.scale_bytes(s) if k == "e2m1" else rq.scale_bytes(s) | 128
        return nd.decode_packed(rq.pack_nibbles(nib), sb, g, *v.shape)

    def quant_nvfp4_4over6(self, w, *args, **kwargs):
        import torch
        if not self._mine(w):
            self.other_calls["quant_nvfp4_4over6"] += 1
            return self.orig["quant_nvfp4_4over6"](w, *args, **kwargs)
        if args != (4, 16) or kwargs:
            raise SystemExit(f"--candidates: unexpected candidate call quant_nvfp4_4over6{args}{kwargs}")
        b = self._decode("e2m1")
        if "e2m1" not in self.files:               # rtn: the trainer's own quantizer, bit for bit
            ref = self.orig["quant_nvfp4_4over6"](w, *args)
            if not torch.equal(b.view(torch.int16), ref.view(torch.int16)):
                raise SystemExit(f"--candidates-e2m1 rtn: {self.current[0]}: differs from quant_nvfp4_4over6")
        self._done("quant_e2m1")
        return b

    def quant_mix_4_6(self, w, *args, **kwargs):
        import torch
        if not self._mine(w):
            self.other_calls["quant_mix_4_6"] += 1
            return self.orig["quant_mix_4_6"](w, *args, **kwargs)
        if args != (4, 16) or kwargs != dict(type_block=(8, 64), clip="a1", elect="always"):
            raise SystemExit(f"--candidates: unexpected candidate call quant_mix_4_6{args}{kwargs}")
        a = self._decode("e0m3")
        if "e0m3" not in self.files:               # rtn: equal values (the store keeps E0M3 zeros as +0)
            ref = self.orig["quant_mix_4_6"](w, *args, **kwargs)
            if not torch.equal(a, ref):
                raise SystemExit(f"--candidates-e0m3 rtn: {self.current[0]}: differs from quant_mix_4_6")
        self._done("quant_e0m3")
        return a

    def pack(self, w, base_ref, alt_ref):
        """run_train_map's RTN pre-packer (``packed_candidates.pack``). With code files it would return None for the
        given candidates, and the trainer's dense fallback would keep both BF16 candidates of every module on the GPU
        until the candidate loop ends (about 4 bytes per weight: more than the GPU holds for Qwen3.8-27B). So the
        given codes are put into ``packed_candidates.encode``'s layout and checked with the trainer's own decoders
        (E2M1 bit for bit, E0M3 by value); the module then takes the trainer's packed path, as RTN candidates do.
        With ``rtn`` the trainer's own ``pack`` runs and must succeed."""
        import torch
        if not self._mine(w):
            self.other_calls["pack"] += 1
            return self.orig["pack"](w, base_ref, alt_ref)
        if not self.files:
            p = self.orig["pack"](w, base_ref, alt_ref)
            if p is None:
                raise SystemExit(f"--candidates rtn: {self.current[0]}: the trainer's pack refused the RTN candidates")
            return p
        pc = sys.modules["quantize.packed_candidates"]
        nd, _ = self._realquant()
        (v4, s4, g4), (v0, s0, g0) = self.current[2]["e2m1"], self.current[2]["e0m3"]
        if not torch.equal(g4, g0):
            raise SystemExit(f"--candidates: {self.current[0]}: the two candidates' global scales differ")
        p = dict(shape=tuple(w.shape), gs=g4.clone(), e2m1=pc._pack_nibbles(nd.e2m1_nibbles(v4)),
                 s2=s4.reshape(-1).to(torch.float8_e4m3fn), e0m3=pc._pack_nibbles((v0 + 8).to(torch.int64)),
                 s0=s0.reshape(-1).to(torch.float8_e4m3fn))
        if not (torch.equal(pc.decode_base(p).view(torch.int16), base_ref.view(torch.int16))
                and torch.equal(pc.decode_alt(p), alt_ref)):
            raise SystemExit(f"--candidates: {self.current[0]}: the packed candidates do not decode to the given ones")
        self._done("pack")
        return p

    def _encode(self, k, fn, w):
        import torch
        if not self._mine(w):
            self.other_calls[fn] += 1
            return self.orig_rq[fn](w)
        v, s, g = self.current[2][k]
        if k not in self.files:                    # rtn: rq's own codes (E2M1 with the sign of zero), scales, scale
            rc, rs, rg = self.orig_rq[fn](w)
            same = (torch.equal(v.view(torch.int32), rc.float().view(torch.int32)) if k == "e2m1"
                    else torch.equal(v, rc.float()))
            if not (same and torch.equal(s, rs.float()) and torch.equal(g, rg.float().reshape(()))):
                raise SystemExit(f"--candidates-{k} rtn: {self.current[0]}: codes differ from rq.{fn}")
        self._done(f"rq_{k}")
        return v.clone(), s.clone(), g.clone()

    def weight_four_over_six(self, w):
        return self._encode("e2m1", "weight_four_over_six", w)

    def weight_e0m3(self, w):
        return self._encode("e0m3", "weight_e0m3", w)

    def finish(self):
        want = self._want()
        missing = [n for n in self.names if set(self.modules.get(n, {}).get("calls", {})) != want
                   or any(c != 1 for c in self.modules[n]["calls"].values())]
        extra = sorted(set(self.modules) - set(self.names))
        rec = dict(sources={k: dict(source=s, **({} if s == "rtn" else dict(
                       key=self.files[k][0], bytes=Path(s).stat().st_size))) for k, s in self.sources.items()},
                   lean_store=self.lean, modules=len(self.modules), expected_modules=len(self.names),
                   missing_or_repeated=missing, extra=extra, other_calls=self.other_calls,
                   mean_codes_changed_vs_rtn={k: sum(m[k]["codes_changed_vs_rtn"] for m in self.modules.values())
                                              / max(len(self.modules), 1) for k in self.KINDS},
                   per_module={n: {k: m[k] for k in self.KINDS} for n, m in self.modules.items()})
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "candidates_hook.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        if missing or extra:
            raise SystemExit(f"--candidates: {len(missing)} modules missing or repeated, {len(extra)} extra")
        print(f"CANDIDATES-HOOK e2m1={self.sources['e2m1']} e0m3={self.sources['e0m3']}: {len(self.modules)} modules, "
              f"lean store {self.lean}, mean codes changed vs RTN {rec['mean_codes_changed_vs_rtn']}", flush=True)
        return rec


def capture_model(R, captured: dict):
    """Record the model the trainer loads (``{id(weight): (name, module)}`` of its linears) by wrapping the loaders
    it calls; the loaders themselves are unchanged."""
    import torch
    import transformers

    def remember(model):
        captured["weights"] = {id(m.weight): (n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        return model

    load_model = R.load_model

    def load_model_captured(prior, target):
        model, modules = load_model(prior, target)
        remember(model)
        return model, modules

    R.load_model = load_model_captured
    qwen = transformers.Qwen3_5ForConditionalGeneration
    load_qwen = qwen.from_pretrained                # bound: the loader in effect (the real one, or --tiny-layers')

    def from_pretrained(cls, *args, **kwargs):
        out = load_qwen(*args, **kwargs)
        remember(out[0] if isinstance(out, tuple) else out)
        return out

    qwen.from_pretrained = classmethod(from_pretrained)


def install_tiny(R, trainer, layers: int):
    """TESTS ONLY: the trainer loads ``models.tiny.build(...)`` instead of the checkpoint."""
    import torch
    import transformers
    from calibration.tmopt_common import MODEL_KEYS
    from models.registry import get
    from models import tiny
    key = _trainer_arg(trainer, "--model")
    spec = get({v: k for k, v in MODEL_KEYS.items()}[key])
    model, _, _ = tiny.build(spec, None, layers, device="cuda", attn="sdpa")
    model.eval().requires_grad_(False)

    def load_model(prior, target):
        head = model.get_output_embeddings()
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}
        assert list(modules) == list(prior["matrices"])
        return model, modules

    def from_pretrained(cls, *args, **kwargs):  # what run_train_map's Qwen branch calls
        return (model, dict(missing_keys=[], mismatched_keys=[], error_msgs=[])) \
            if kwargs.get("output_loading_info") else model

    R.load_model = load_model
    # on the class itself: run_train_map imports the class inside main(), whatever reference it gets
    transformers.Qwen3_5ForConditionalGeneration.from_pretrained = classmethod(from_pretrained)
    return model


def main(argv=None):
    ext, trainer = _parse(sys.argv[1:] if argv is None else argv)
    # as `python run_train_map.py`: the vendored directory first on the path; this repository's packages last
    sys.path[0:1] = [str(ENGINE)]
    sys.path.append(str(ROOT))
    import run_train_map as R
    if ext.register_model is not None:
        register_model(R, trainer, ext.register_model)
    if ext.teacher_host:
        install_teacher_host(R.topk_teacher)
    names = module_names(R, trainer)
    state = None
    if ext.record_grads is not None or ext.oneshot is not None:
        state = install_optimizer_hooks(ext, names)
    theta_state = install_theta_capture() if ext.save_theta is not None else None
    if ext.tiny_layers:
        install_tiny(R, trainer, ext.tiny_layers)
    rotated = None
    if ext.rotate == "hadamard":
        captured = {}
        capture_model(R, captured)
        rotated = install_rotation(R, captured, ext.rotate_block)
    candidates = None
    if ext.candidates_e2m1 is not None:
        captured = {}
        capture_model(R, captured)
        out = _trainer_arg(trainer, "--out")
        candidates = Candidates(R, captured, dict(e2m1=ext.candidates_e2m1, e0m3=ext.candidates_e0m3), names,
                                Path(out) if out else None)
    sys.argv = ["run_train_map.py", *trainer]
    try:
        R.main()
    except _Stop:
        print(f"ONESHOT stopped after {state['steps']} optimizer steps (--oneshot-steps)", flush=True)
    if ext.oneshot is not None:
        import torch
        if state["sum"] is None:
            raise SystemExit("--oneshot: the trainer made no optimizer step")
        ext.oneshot.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(steps=state["steps"], names=names,
                        sum={n: s.cpu() for n, s in zip(names, state["sum"])},
                        first={n: g.cpu() for n, g in zip(names, state["first"])}), ext.oneshot)
        print(f"ONESHOT {state['steps']} optimizer steps -> {ext.oneshot}", flush=True)
    if theta_state is not None:
        save_theta(theta_state, names, ext.save_theta)
    if rotated is not None:
        if sorted(rotated) != sorted(names):
            raise SystemExit(f"--rotate: {len(rotated)} of {len(names)} modules were rotated")
        print(f"ROTATE hadamard (max block {ext.rotate_block}): {len(rotated)} modules", flush=True)
    if candidates is not None:
        candidates.finish()


if __name__ == "__main__":
    main()
