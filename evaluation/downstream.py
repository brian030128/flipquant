"""Downstream accuracy through lm-eval, as in the FlipQuant paper: the paper's ``run_lmeval_deploy.py`` (the harness)
and ``04_downstream.py`` (its settings), ported onto this repository's models, maps and backends (``models.cli``). One
process evaluates one model under one policy.

    python -m evaluation.downstream --model phi4-14b --mode native \\
        --weight mixfp4 --map maps/phi4-14b/flipquant_8x64.pt --out results/phi4-14b/downstream/ours_8x64.json
    python -m evaluation.downstream --model phi4-14b --mode native \\
        --weight fourover6 --out results/phi4-14b/downstream/fo6.json
    python -m evaluation.downstream --model phi4-14b --mode bf16 --out results/phi4-14b/downstream/bf16.json

**The paper's settings (the defaults):**
- **Tasks:** MMLU 5-shot; ARC-Challenge, ARC-Easy, HellaSwag and PIQA with lm-eval's defaults (0-shot).
  ``--tasks appendix`` is the paper's full multiple-choice suite (Table 3, Tables E.1-E.3): these five plus WinoGrande
  and LAMBADA (``lambada_openai``), 0-shot. ``--tasks gsm8k_llama`` is the paper's GSM8K (8-shot chain of thought in
  the chat template, greedy, strict match).
- **Harness:** lm-eval 0.4.11's HFLM with its default BOS handling. The datasets are pinned to fixed revisions by
  ``evaluation/lm_eval_datasets.py``, the paper's module, vendored unchanged.
- **Batch:** 16 (Qwen3.8-27B: 8). Activations are per token (``--act-scope row``, the paper's convention).
  lm-eval's seeds are the paper runs'.
- **MMLU's score** is lm-eval's group aggregate (weight_by_size: the micro-average over all 14,042 questions). Its 57
  subjects and 4 categories are recorded too.
- **The paper's policies:**
  - ``--mode bf16``;
  - ``--mode native`` (on an RTX Pro 6000: the sm120 path, optimized kernels) with ``--weight nvfp4 --act nvfp4`` or
    ``--weight fourover6`` (the baseline kernels);
  - FlipQuant maps with ``--weight mixfp4 --map``. The paper ran the paper's kernels (``--kernel-set paper``; for 8x64
    n8k64_wB). Every kernel set computes the same outputs bit for bit.

**Recorded** (``--out``, one JSON report), per task:
- lm-eval's metrics and their stderr;
- per example, the correctness under the primary metric (acc_norm for ARC, HellaSwag and PIQA; acc for MMLU) and the
  per-choice log-likelihoods, for paired comparisons;
- ``sample_digest``, a hash of lm-eval's doc, prompt and target hashes of every example (paired comparisons need equal
  digests);
- the time, in total and inside the model calls, and the peak GPU memory;
- for native runs on the sm120 path, the coverage: every quantized Linear ran natively in every forward, and no
  Linear outside the quantization scope ran (e.g. Qwen3.8-27B's vision tower). Also the native GEMM calls per forward
  token-count bucket and per CTA tile width.

``--resume`` keeps the finished tasks of an earlier run with the same ``--out`` and settings. ``--limit N`` evaluates
the first N examples of each task (of each MMLU subject): a check, not a result. The decode-kernel verification is
``evaluation.decode_check``.

**Chat-template generation tasks** (paper-sm120-runs, the paper's GSM8K column; every option off by default, which
leaves the run above unchanged): ``--apply-chat-template`` (lm-eval's ``apply_chat_template=True`` with
``fewshot_as_multiturn=True``), ``--system-instruction`` (lm-eval's system message, e.g. Nemotron's ``/no_think``),
``--enable-thinking false`` (HFLM's ``enable_thinking``, passed to the chat template). ``gsm8k_llama`` (lm-eval's
llama3/instruct task, as shipped) is summarized per example and filter like ``gsm8k``. ``--gen-lengths`` records each
request's generated token count -- up to and including the first end-of-sequence token, or the whole budget -- by
watching HFLM's batched ``tok_batch_encode`` / ``_model_generate`` calls without changing them. ``--samples-out``
writes lm-eval's per-sample records of every task (JSONL, gzipped).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import torch
import transformers

from models.cli import add_model_args, build, save_json

TASKS = "mmlu,arc_challenge,arc_easy,hellaswag,piqa"
TASK_SETS = {"appendix": "mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande,lambada_openai"}
FEWSHOT = {"mmlu": 5}
#: dataset pins for the appendix tasks, added to the vendored lm_eval_datasets.PINS (that module stays unchanged)
EXTRA_PINS = {'EleutherAI/lambada_openai': '900124bf3b8235c6daf21033af9948b3f07346c4'}   # 2026-10-02
BATCH = {"qwen3.8-27b": 8}
# the paper's run_lmeval_deploy.py
PRIMARY = dict(arc_easy="acc_norm", arc_challenge="acc_norm", hellaswag="acc_norm", openbookqa="acc_norm", piqa="acc_norm",
               boolq="acc", winogrande="acc", gsm8k="exact_match", mmlu="acc", lambada_openai="acc",
               gsm8k_llama="exact_match")
CHANCE = dict(arc_easy=0.25, arc_challenge=0.25, hellaswag=0.25, openbookqa=0.25, piqa=0.5, boolq=0.5, winogrande=0.5,
              gsm8k=0.0, mmlu=0.25, lambada_openai=0.0, gsm8k_llama=0.0)
GSM8K_FILTERS = ("strict-match", "flexible-extract")     # gsm8k.yaml's filter_list, in order
#: generative tasks summarized per (document, filter), with their filter_list in order
GEN_FILTERS = {"gsm8k": GSM8K_FILTERS, "gsm8k_llama": ("strict_match", "flexible_extract")}


def primary(task):
    return "acc" if task.startswith("mmlu") else PRIMARY[task]


def mc_record(s, key):
    """(correctness, {target, ll}) of one multiple-choice sample: lm-eval's per-choice (log-likelihood, is_greedy)."""
    lls = [float(r[0]) for r in s["filtered_resps"]] if s.get("filtered_resps") else None
    return float(s[key]), dict(target=s.get("target"), ll=lls)


def sample_hashes(entry, keyed_rows):
    """``sample_digest`` hashes lm-eval's per-sample doc / prompt / target hashes, sorted by example key: two policies
    saw the same samples iff their digests are equal. ``prompt_hash`` keeps a 12-character prefix per example."""
    lines, prefix = [], {}
    for k, s in keyed_rows:
        if k in prefix or "prompt_hash" not in s:
            continue
        prefix[k] = s["prompt_hash"][:12]
        lines.append(f"{k} {s.get('doc_hash')} {s['prompt_hash']} {s.get('target_hash')}")
    if lines:
        entry["sample_digest"] = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()
        entry["prompt_hash"] = prefix


def request_context(s):
    """The context string of a generate_until sample (lm-eval's logged 'arguments', in either of its layouts)."""
    a = s.get("arguments")
    if isinstance(a, dict):
        first = next(iter(a.values()))
        return first.get("arg_0") if isinstance(first, dict) else first[0]
    if isinstance(a, (list, tuple)) and a:
        return a[0][0] if isinstance(a[0], (list, tuple)) else a[0]
    return None


def summarize(results, samples, gen_log=None):
    out = {}
    for task, metrics in results["results"].items():
        entry = dict(metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != "alias"})
        rows = samples.get(task, [])
        if task in GEN_FILTERS:
            # one sample per (filter, document), in the task's filter order
            examples, seen = {}, {}
            for s in rows:
                doc = str(s["doc_id"])
                seen[doc] = seen.get(doc, -1) + 1
                e = examples.setdefault(doc, dict(raw=s["resps"][0][0] if s.get("resps") else None))
                if gen_log is not None and "generated" not in e:
                    e["generated"] = gen_log.get(request_context(s))
                e[s.get("filter") or GEN_FILTERS[task][seen[doc]]] = dict(
                    exact_match=float(s.get("exact_match", float("nan"))),
                    response=s["filtered_resps"][0] if s.get("filtered_resps") else None)
            entry["examples"] = examples
        else:
            key = primary(task)
            entry["examples"], entry["loglikelihoods"] = {}, {}
            for s in rows:
                if key in s:
                    entry["examples"][str(s["doc_id"])], entry["loglikelihoods"][str(s["doc_id"])] = mc_record(s, key)
        sample_hashes(entry, [(str(s["doc_id"]), s) for s in rows])
        out[task] = entry
    return out


def summarize_group(res, task):
    """A group task (mmlu): lm-eval's group aggregate, the metrics of every subtask and subgroup, and the examples of
    every subtask keyed '<subtask>:<doc_id>'."""
    results = res["results"]
    top = results.get(task) or res.get("groups", {}).get(task)
    entry = dict(metrics={k: v for k, v in top.items() if isinstance(v, (int, float)) and k != "alias"},
                 subtasks={t: {k: v for k, v in m.items() if isinstance(v, (int, float)) and k != "alias"}
                           for t, m in results.items() if t != task},
                 examples={}, loglikelihoods={})
    for sub, rows in res.get("samples", {}).items():
        for s in rows:
            if "acc" in s:
                k = f"{sub}:{s['doc_id']}"
                entry["examples"][k], entry["loglikelihoods"][k] = mc_record(s, "acc")
    sample_hashes(entry, [(f"{sub}:{s['doc_id']}", s) for sub, rows in res.get("samples", {}).items() for s in rows])
    return entry


class NativeCoverage:
    """Native GEMM evidence of an sm120 policy (the paper's run_lmeval_deploy.py): calls and tokens, forwards per
    token-count bucket, CTA widths, and the Linears outside the scope, which must never run."""

    def __init__(self, model):
        from models import sm120 as NS
        from mixfp4_sm120.select import bucket
        self.NS, self.model = NS, model
        self.nat = NS.native_modules(model)
        self.ksets = list({id(m.kernel_set): m.kernel_set for m in self.nat.values() if m.kernel_set is not None}.values())
        self.forwards, self.unscoped = {}, {}

        def count_forward(mod, inputs):        # every quantized Linear of a forward sees the same token count
            b = bucket(inputs[0].numel() // inputs[0].shape[-1])
            self.forwards[b] = self.forwards.get(b, 0) + 1
        head = model.get_output_embeddings()
        self.hooks = [next(iter(self.nat.values())).register_forward_pre_hook(count_forward)]
        self.hooks += [m.register_forward_hook(lambda mod, i, o, _n=n: self.unscoped.__setitem__(_n, self.unscoped.get(_n, 0) + 1))
                       for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head]
        NS.reset_counters(model)

    def counts(self):
        widths = {}
        for ks in self.ksets:
            for w, c in ks.stats.items():
                widths[str(w)] = widths.get(str(w), 0) + c
        return dict(calls=sum(m.calls for m in self.nat.values()), tokens=sum(m.tokens for m in self.nat.values()),
                    calls_by_width=widths, forwards_by_token_bucket={str(k): v for k, v in self.forwards.items()})

    def delta(self, before):
        after = self.counts()
        diff = lambda a, b: {k: a[k] - b.get(k, 0) for k in a if a[k] != b.get(k, 0)}  # noqa: E731
        g = dict(calls=after["calls"] - before["calls"], tokens=after["tokens"] - before["tokens"],
                 forwards_by_token_bucket=diff(after["forwards_by_token_bucket"], before["forwards_by_token_bucket"]),
                 calls_by_width=diff(after["calls_by_width"], before["calls_by_width"]) if self.ksets else None)
        g["forwards"] = sum(g["forwards_by_token_bucket"].values())
        g["every_linear_every_forward"] = g["calls"] == len(self.nat) * g["forwards"]
        return g

    def close(self):
        for h in self.hooks:
            h.remove()
        cov = self.NS.coverage(self.model)
        head = self.model.get_output_embeddings()
        cov["remaining_bf16_linears"] = [n for n, m in self.model.named_modules()
                                         if isinstance(m, torch.nn.Linear) and m is not head]
        cov["remaining_bf16_linears_called"] = dict(self.unscoped)
        if self.ksets:
            cov["kernel_set"] = self.ksets[0].describe() if len(self.ksets) == 1 else [k.describe() for k in self.ksets]
        return cov


def gen_lengths(lm, model):
    """{context string: dict(tokens, eos, budget)} of every generate_until request, filled while HFLM runs. HFLM encodes
    each batch with tok_batch_encode right before its _model_generate call; the rows of the output are those contexts in
    order. A row's generated tokens end at its first end-of-sequence token (counted); without one it used the whole
    budget. Both methods are wrapped, not changed."""
    log, last = {}, {}
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])
    if lm.tokenizer.eos_token_id is not None:
        eos_ids.add(lm.tokenizer.eos_token_id)
    pad = lm.tokenizer.pad_token_id
    encode, generate = lm.tok_batch_encode, lm._model_generate

    def tok_batch_encode(strings, *a, **k):
        enc = encode(strings, *a, **k)
        last.update(strings=list(strings), width=enc[0].shape[1])
        return enc

    def model_generate(context, max_length, stop, **kw):
        out = generate(context, max_length, stop, **kw)
        strings = last.pop("strings", None)
        if strings is None or len(strings) != out.shape[0] or last.get("width") != context.shape[1]:
            raise RuntimeError("--gen-lengths: a _model_generate call without its tok_batch_encode batch")
        budget = max_length - context.shape[1]
        for text, row in zip(strings, out[:, context.shape[1]:].tolist()):
            n = next((i + 1 for i, t in enumerate(row) if t in eos_ids), None)
            if n is None:               # no end-of-sequence token: the row ran to the end (pad only after a stop)
                n = len(row)
                while n and pad is not None and row[n - 1] == pad and pad not in eos_ids:
                    n -= 1
            log[text] = dict(tokens=n, eos=any(t in eos_ids for t in row[:n]), budget=budget)
        return out

    lm.tok_batch_encode, lm._model_generate = tok_batch_encode, model_generate
    return log


def write_samples(prefix, samples):
    import gzip
    for task, rows in samples.items():
        path = Path(f"{prefix}.{task}.jsonl.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            for s in rows:
                f.write(json.dumps(s, default=str) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, act_scope="row")
    ap.add_argument("--tasks", default=TASKS, help=f"comma list of lm-eval tasks, or {sorted(TASK_SETS)}")
    ap.add_argument("--num-fewshot", action="append", default=[], metavar="TASK=N",
                    help="few-shot count of one task (default: mmlu=5, the others lm-eval's)")
    ap.add_argument("--limit", type=int, default=None, help="examples per task / MMLU subject (checks); default all")
    ap.add_argument("--batch-size", type=int, default=None, help="default 16 (Qwen3.8-27B: 8)")
    ap.add_argument("--resume", action="store_true", help="keep the finished tasks of an earlier run of --out")
    ap.add_argument("--apply-chat-template", action="store_true",
                    help="lm-eval's apply_chat_template=True with fewshot_as_multiturn=True (default: off)")
    ap.add_argument("--system-instruction", default=None, help="lm-eval's system instruction (default: none)")
    ap.add_argument("--enable-thinking", choices=("true", "false"), default=None,
                    help="HFLM enable_thinking, passed to the chat template (default: not passed)")
    ap.add_argument("--gen-lengths", action="store_true",
                    help="record each generate_until request's generated token count (passive; default: off)")
    ap.add_argument("--samples-out", type=Path, default=None,
                    help="write lm-eval's per-sample records of each task to <samples-out>.<task>.jsonl.gz")
    ap.add_argument("--doc-ids", type=Path, default=None,
                    help="JSON {task: [doc ids]}: evaluate only these documents (lm-eval's samples=; a check, not a "
                         "result; default: all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from evaluation import lm_eval_datasets      # pins the datasets before lm-eval builds its tasks
    lm_eval_datasets.PINS.update(EXTRA_PINS)
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fewshot = dict(FEWSHOT, **{t: int(n) for t, n in (s.split("=") for s in args.num_fewshot)})
    tasks = TASK_SETS.get(args.tasks, args.tasks).split(",")
    previous = {}
    if args.resume and args.out.exists():
        old = json.loads(args.out.read_text())
        same = (old["argv"]["model"], old["num_fewshot"], old["limit"]) == (args.model, fewshot, args.limit)
        assert same, "--resume: the earlier run used other settings"
        previous = old.get("tasks", {})
    t0 = time.time()
    model, _, spec, layers, info = build(args)
    # the tokenizer as lm-eval would load it (models.loader sets a pad token; HFLM chooses its own)
    tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=info["source"]["revision"])
    batch = args.batch_size or BATCH.get(spec.key, 16)
    report = dict(status="running", argv={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  model=spec.key, policy=info, tasks_requested=tasks, num_fewshot=fewshot, limit=args.limit,
                  batch_size=batch, load_seconds=time.time() - t0,
                  versions=dict(lm_eval=md.version("lm_eval"), transformers=transformers.__version__,
                                torch=torch.__version__, datasets=md.version("datasets")),
                  dataset_pins=lm_eval_datasets.PINS, host=dict(hostname=platform.node(), gpu=torch.cuda.get_device_name(0)),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), tasks={})
    native = NativeCoverage(model) if info.get("native_backend") == "sm120" else None
    doc_ids = json.loads(args.doc_ids.read_text()) if args.doc_ids else None
    hflm_kw = {} if args.enable_thinking is None else dict(enable_thinking=args.enable_thinking == "true")
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch, backend="causal", **hflm_kw)
    gen_log = gen_lengths(lm, model) if args.gen_lengths else None
    chat = dict(apply_chat_template=True, fewshot_as_multiturn=True) if args.apply_chat_template else {}
    if args.system_instruction is not None:
        chat["system_instruction"] = args.system_instruction
    report.update(chat_options=dict(chat, enable_thinking=args.enable_thinking), hflm_max_length=lm.max_length,
                  chat_template_args=getattr(lm, "chat_template_args", None))
    ids = lm.tok_encode("Question: 1 + 1 =")
    report["tokenization"] = dict(add_bos_token=lm.add_bos_token, bos_token=tok.bos_token, bos_token_id=tok.bos_token_id,
                                  context_starts_with_bos=tok.bos_token_id is not None and ids[0] == tok.bos_token_id,
                                  probe_ids=ids[:4])
    timing = {}

    def timed(fn, kind):          # the model's share of the time: lm-eval dispatches through getattr(lm, type)
        def call(requests, *a, **k):
            t = time.time()
            result = fn(requests, *a, **k)
            torch.cuda.synchronize()
            timing[kind + "_seconds"] = timing.get(kind + "_seconds", 0.0) + time.time() - t
            timing[kind + "_requests"] = timing.get(kind + "_requests", 0) + len(requests)
            return result
        return call
    lm.loglikelihood = timed(lm.loglikelihood, "loglikelihood")
    lm.generate_until = timed(lm.generate_until, "generate_until")
    task_manager = TaskManager()
    for task in tasks:
        done = previous.get(task)
        if done is not None and done.get("examples"):
            report["tasks"][task] = dict(done, resumed=True)
            continue
        timing.clear()
        before = native.counts() if native else None
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        # lm-eval's defaults. Every metric here aggregates by mean, whose stderr is the closed form (no bootstrap draws)
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=args.limit, log_samples=True, bootstrap_iters=100000,
                                      random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                                      fewshot_random_seed=1234, num_fewshot=fewshot.get(task),
                                      task_manager=task_manager, **chat,
                                      **({"samples": {task: doc_ids[task]}} if doc_ids and task in doc_ids else {}))
        if args.samples_out is not None:
            write_samples(args.samples_out, res.get("samples", {}))
        if task in res.get("samples", {}):
            summary = summarize(res, res.get("samples", {}), gen_log)[task]
        else:                                       # a group of subtasks (mmlu)
            summary = summarize_group(res, task)
            summary["aggregate"] = "lm-eval group aggregate, weight_by_size (micro-average over all questions)"
        n_shot = res.get("n-shot", {}).get(task)
        if n_shot is None and "subtasks" in summary:
            n_shot = sorted({v for t, v in res.get("n-shot", {}).items() if t in summary["subtasks"]})
        summary.update(seconds=time.time() - t1, examples_evaluated=len(summary["examples"]), batch_size=batch,
                       peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, n_shot=n_shot,
                       chance=CHANCE.get(task, 0.25), model_timing=dict(timing))
        if native:
            summary["native_gemm"] = native.delta(before)
        report["tasks"][task] = summary
        value = summary["metrics"].get(primary(task) + ",none", summary["metrics"].get(primary(task) + ",strict-match",
                                       summary["metrics"].get(primary(task) + ",strict_match")))
        print(f"DOWNSTREAM {spec.key} {task} {primary(task)} {value} ({summary['examples_evaluated']} examples, "
              f"{summary['seconds']:.0f}s)", flush=True)
        save_json(args.out, report)
    if native:
        cov = report["coverage"] = native.close()
        # every quantized Linear ran natively, and the Linears outside the scope never ran
        assert cov["native_called"] == cov["native"] == len(layers), cov
        assert not cov["remaining_bf16_linears_called"], cov["remaining_bf16_linears_called"]
    report["dataset_loads"] = lm_eval_datasets.LOADS
    report["status"] = "complete"
    report["seconds"] = time.time() - t0
    save_json(args.out, report)


if __name__ == "__main__":
    main()
