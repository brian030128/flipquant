"""Evaluation and calibration data, pinned so results are comparable across runs and models.

Evaluation (the reference's ``run_baseline_protocol_audit.data``, unchanged protocol):
  * WikiText-2 raw test @ ``WIKI_REVISION``, rows joined with ``"\\n\\n"``, tokenized once, cut into
    non-overlapping ``seqlen`` windows over the largest multiple of 2048 tokens.
  * C4 validation shard 0 @ ``C4_REVISION``: 256 documents drawn with ``random.Random(0)`` (only
    documents longer than 2049 tokens), one random 2048-token crop each, split into ``seqlen``
    windows.

Calibration (generalized from the reference's ``make_nemotron_prior.py`` so it works for any
tokenizer): pinned OpenWebMath and CodeParrot shards; per source, the first 64 documents of >= 512
tokens with one random 512-token window each (seed ``SEED``) for fitting, and 96 further documents
(seed ``SEED + 1``) for development/monitoring. 128 calibration + 192 development windows.

Everything is cached under ``$FLIPQUANT_CACHE`` (default ``/work/$USER/flipquant_cache`` if
``/work/$USER`` exists, else ``~/.cache/flipquant``), keyed by the tokenizer, with SHA-256 digests
of every window so two runs can be checked to have used identical tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
from pathlib import Path

import torch

WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"
STREAMS = {
    "math": dict(repo="open-web-math/open-web-math", revision="fde8ef8de2300f5e778f56261843dab89f230815",
                 path="data/train-00000-of-00114-5a023365406cb9c4.parquet", field="text"),
    "code": dict(repo="codeparrot/codeparrot-clean", revision="35a59fb025bc0a102f7d96eac09d145b896d487b",
                 path="file-000000000001.json.gz", field="content"),
}
SEED = 20260928
CALIB_PER_SOURCE, DEV_PER_SOURCE, CALIB_TOKENS = 64, 96, 512


def cache_root() -> Path:
    if os.environ.get("FLIPQUANT_CACHE"):
        return Path(os.environ["FLIPQUANT_CACHE"])
    work = Path("/work") / os.environ.get("USER", "")
    return (work / "flipquant_cache") if work.is_dir() else Path.home() / ".cache" / "flipquant"


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _tok_key(tok) -> str:
    name = getattr(tok, "name_or_path", "tok").replace("/", "--")
    vocab = hashlib.sha256(json.dumps(sorted(tok.get_vocab().items())).encode()).hexdigest()[:12]
    return f"{name}-{vocab}"


def _cached(tok, name: str, build):
    path = cache_root() / "data" / _tok_key(tok) / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    # a per-process temporary name: jobs sharing the cache may build the same entry at once (the contents are equal;
    # the atomic rename lets the last writer win instead of one rename failing)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{socket.gethostname()}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    return obj


# ------------------------------------------------------------------------------------------------
# evaluation windows


def eval_windows(tok, dataset: str, seqlen: int = 2048) -> tuple[list[torch.Tensor], dict]:
    """``(windows, meta)`` for ``wiki`` or ``c4``; each window is ``[1, seqlen]`` int64."""
    assert 2048 % seqlen == 0, "seqlen must divide 2048 (the protocol's parent window)"

    def build():
        from datasets import load_dataset
        if dataset == "wiki":
            wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKI_REVISION, split="test")
            ids = tok("\n\n".join(wiki["text"]), return_tensors="pt").input_ids
            usable = ids.shape[1] // 2048 * 2048
            wins = [ids[:, i:i + seqlen].clone() for i in range(0, usable, seqlen)]
            meta = dict(revision=WIKI_REVISION, total_tokens=ids.shape[1], used_tokens=usable)
        elif dataset == "c4":
            ds = load_dataset("allenai/c4", revision=C4_REVISION, data_files={"validation": C4_FILE},
                              split="validation")
            rng, wins, docs = random.Random(0), [], []
            for _ in range(256):
                while True:
                    index = rng.randint(0, len(ds) - 1)
                    text = ds[index]["text"]
                    tokens = tok(text, return_tensors="pt").input_ids
                    if tokens.shape[1] > 2049:
                        break
                offset = rng.randint(0, tokens.shape[1] - 2048 - 1)
                crop = tokens[:, offset:offset + 2048]
                wins.extend(crop[:, i:i + seqlen].clone() for i in range(0, 2048, seqlen))
                docs.append(dict(index=index, offset=offset,
                                 document_sha256=hashlib.sha256(text.encode()).hexdigest()))
            meta = dict(revision=C4_REVISION, path=C4_FILE, seed=0, documents=docs)
        else:
            raise ValueError(f"unknown dataset {dataset!r} (wiki|c4)")
        meta.update(windows=len(wins), window_tokens=seqlen, token_sha256=[sha(w) for w in wins])
        return wins, meta

    return _cached(tok, f"eval_{dataset}_{seqlen}", build)


# ------------------------------------------------------------------------------------------------
# calibration / development windows


def _stream_windows(tok, source: str, count: int, skip: set, seed: int, length: int = CALIB_TOKENS) -> list[dict]:
    from datasets import load_dataset
    s = STREAMS[source]
    stream = load_dataset(s["repo"], revision=s["revision"], data_files={"train": s["path"]},
                          split="train", streaming=True)
    rng, seen, out = random.Random(seed), set(skip), []
    for row in stream:
        text = row[s["field"]]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] < length:
            continue
        offset = rng.randrange(ids.shape[1] - length + 1)
        out.append(dict(source=source, document_sha256=digest, offset=offset,
                        ids=ids[:, offset:offset + length].clone()))
        if len(out) == count:
            return out
    raise RuntimeError(f"{source}: only {len(out)} usable documents")


def calibration_sets(tok) -> dict:
    """``{"fit": [128 x [1,512]], "dev": [192 x [1,512]], "meta": {...}}``."""

    def build():
        fit, dev, meta = [], [], {"streams": STREAMS, "seed": SEED}
        for source in ("math", "code"):
            cal = _stream_windows(tok, source, CALIB_PER_SOURCE, set(), SEED)
            held = _stream_windows(tok, source, DEV_PER_SOURCE, {r["document_sha256"] for r in cal}, SEED + 1)
            fit += cal
            dev += held
        for name, rows in (("fit", fit), ("dev", dev)):
            meta[name] = [dict(source=r["source"], document_sha256=r["document_sha256"], offset=r["offset"],
                               token_sha256=sha(r["ids"])) for r in rows]
        assert len({r["document_sha256"] for r in fit + dev}) == len(fit) + len(dev)
        return {"fit": [r["ids"] for r in fit], "dev": [r["ids"] for r in dev], "meta": meta}

    return _cached(tok, "calibration_v1", build)


def calibration_fit(tok) -> dict:
    """The 128 fit windows of :func:`calibration_sets` without drawing the development set
    (``train_map_legacy --no-dev``): ``{"fit": [128 x [1,512]], "meta": {...}}``. The windows are identical,
    because the fit draw of each stream does not depend on the development draw that follows it."""

    def build():
        fit, meta = [], {"streams": STREAMS, "seed": SEED}
        for source in ("math", "code"):
            fit += _stream_windows(tok, source, CALIB_PER_SOURCE, set(), SEED)
        meta["fit"] = [dict(source=r["source"], document_sha256=r["document_sha256"], offset=r["offset"],
                            token_sha256=sha(r["ids"])) for r in fit]
        return {"fit": [r["ids"] for r in fit], "meta": meta}

    return _cached(tok, "calibration_v1_fit_only", build)


def calibration_sets_extended(tok, fit_windows: int) -> dict:
    """:func:`calibration_sets` with the fit set grown to ``fit_windows`` (half math, half code): the
    standard 128 windows first, then further documents from each stream (seed ``SEED + 2``) that
    skip every fit and development document. The development set is unchanged, so development KL
    stays comparable across fit-set sizes."""
    base = calibration_sets(tok)
    per_source = fit_windows // 2
    if fit_windows % 2 or per_source < CALIB_PER_SOURCE:
        raise ValueError(f"fit_windows must be even and >= {2 * CALIB_PER_SOURCE}, got {fit_windows}")
    if per_source == CALIB_PER_SOURCE:
        return base

    def build():
        used = {r["document_sha256"] for r in base["meta"]["fit"] + base["meta"]["dev"]}
        extra = []
        for source in ("math", "code"):
            extra += _stream_windows(tok, source, per_source - CALIB_PER_SOURCE, used, SEED + 2)
        meta = dict(base["meta"], extra_seed=SEED + 2,
                    fit=base["meta"]["fit"] + [dict(source=r["source"], document_sha256=r["document_sha256"],
                                                    offset=r["offset"], token_sha256=sha(r["ids"]))
                                               for r in extra])
        assert len({r["document_sha256"] for r in meta["fit"] + meta["dev"]}) == len(meta["fit"]) + len(meta["dev"])
        return {"fit": base["fit"] + [r["ids"] for r in extra], "dev": base["dev"], "meta": meta}

    return _cached(tok, f"calibration_v1_fit{fit_windows}", build)


def calibration_from_record(tok, record: Path) -> dict:
    """Rebuild the exact calibration windows of a reference calibration record
    (``report.json`` with ``fit: {math|code: {repo, revision, path, documents, token_sha256}}``)."""
    from datasets import load_dataset
    prior = json.loads(Path(record).read_text())
    fit = []
    for source in ("math", "code"):
        meta = prior["fit"][source]
        wanted = {d["document_sha256"]: d["offset"] for d in meta["documents"]}
        stream = load_dataset(meta["repo"], revision=meta["revision"], data_files={"train": meta["path"]},
                              split="train", streaming=True)
        found = {}
        for row in stream:
            text = row["text" if source == "math" else "content"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in wanted and digest not in found:
                ids = tok(text, return_tensors="pt").input_ids
                found[digest] = ids[:, wanted[digest]:wanted[digest] + CALIB_TOKENS].clone()
                if len(found) == len(wanted):
                    break
        rows = [found[d["document_sha256"]] for d in meta["documents"]]
        assert [sha(r) for r in rows] == meta["token_sha256"], f"{source}: token digests differ from the record"
        fit += rows
    return {"fit": fit, "meta": {"record": str(record)}}


def _rebuild_windows(tok, rows: list[dict], length: int = CALIB_TOKENS) -> list[torch.Tensor]:
    """The windows ``rows`` name (``source``, ``document_sha256``, ``offset``, ``token_sha256``), cut from the pinned
    :data:`STREAMS` exactly as :func:`_stream_windows` cut them; every window's token sha256 is checked."""
    from datasets import load_dataset
    found = {}
    for source in ("math", "code"):
        wanted = {r["document_sha256"]: r["offset"] for r in rows if r["source"] == source}
        if not wanted:
            continue
        s = STREAMS[source]
        stream = load_dataset(s["repo"], revision=s["revision"], data_files={"train": s["path"]},
                              split="train", streaming=True)
        hit = 0
        for row in stream:
            text = row[s["field"]]
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in wanted and (source, digest) not in found:
                ids = tok(text, return_tensors="pt").input_ids
                found[(source, digest)] = ids[:, wanted[digest]:wanted[digest] + length].clone()
                hit += 1
                if hit == len(wanted):
                    break
        if hit != len(wanted):
            raise RuntimeError(f"{source}: {len(wanted) - hit} of the recorded documents are not in the stream")
    out = [found[(r["source"], r["document_sha256"])] for r in rows]
    bad = [i for i, (r, w) in enumerate(zip(rows, out)) if sha(w) != r["token_sha256"]]
    if bad:
        raise RuntimeError(f"{len(bad)} rebuilt windows differ from their recorded token sha256 (first: {bad[:3]})")
    return out


def calibration_release(tok, record: Path, trainer_report: Path) -> dict:
    """The FlipQuant release maps' fit set (``calibration.train_map --fit-windows 256``): the reference calibration
    record's 128 windows (:func:`calibration_from_record`), then the 128 windows of the trainer's fit extension,
    rebuilt from the records the release run wrote (``trainer_report.json``'s ``fit_extension.records``: source,
    document sha256, offset and token sha256 per window). Every window's token sha256 equals the release run's, so
    GPTQ, FOCUS and the activation statistics see exactly the windows the maps were trained on.
    ``{"fit": [256 x [1, 512]], "meta": {...}}``."""
    rep = json.loads(Path(trainer_report).read_text())
    ext = rep["fit_extension"]
    rows = ext["records"]
    if len(rows) != ext["windows"]:
        raise ValueError(f"{trainer_report}: {len(rows)} extension records, {ext['windows']} windows")
    prior = json.loads(Path(record).read_text())
    key = hashlib.sha256((json.dumps(prior["fit"], sort_keys=True) + json.dumps(rows, sort_keys=True)).encode()).hexdigest()[:16]

    def build():
        base = calibration_from_record(tok, record)["fit"]
        extra = _rebuild_windows(tok, rows)
        fit = base + extra
        base_sha = [h for source in ("math", "code") for h in prior["fit"][source]["token_sha256"]]
        meta = dict(record=str(record), trainer_report=str(trainer_report), windows=len(fit),
                    base_windows=len(base), extension_windows=len(extra), extension_rule=ext.get("rule"),
                    token_sha256=[sha(w) for w in fit])
        assert meta["token_sha256"] == base_sha + [r["token_sha256"] for r in rows]
        assert len(set(meta["token_sha256"])) == len(fit)
        return {"fit": fit, "meta": meta}

    return _cached(tok, f"calibration_release_{key}", build)


def wiki_train_windows(tok, count: int = 128, length: int = CALIB_TOKENS, ngram: int = 64) -> dict:
    """DIAGNOSTICS ONLY (``--gptq-calib wiki``): ``count`` random ``length``-token windows of the
    WikiText-2 *train* split (same revision and row joining as the test split), seed ``SEED``,
    non-overlapping. Checked to share no ``ngram``-token sequence with the test split's token stream,
    i.e. with any PPL evaluation window. ``{"fit": [...], "meta": {...}}``."""

    def build():
        from datasets import load_dataset
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKI_REVISION, split="train")
        ids = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids[0]
        rng = random.Random(SEED)
        starts = sorted(rng.sample(range(ids.numel() // length), count))
        wins = [ids[s * length:(s + 1) * length].unsqueeze(0).clone() for s in starts]
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKI_REVISION, split="test")
        tids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0].tolist()
        grams = {tuple(tids[i:i + ngram]) for i in range(len(tids) - ngram + 1)}
        shared = sum(tuple(w[0, i:i + ngram].tolist()) in grams for w in wins for i in range(length - ngram + 1))
        if shared:
            raise RuntimeError(f"{shared} {ngram}-token sequences of the train windows occur in the test split")
        meta = dict(revision=WIKI_REVISION, split="train", seed=SEED, starts=[s * length for s in starts],
                    train_tokens=ids.numel(), test_ngram_overlap=0, ngram=ngram,
                    token_sha256=[sha(w) for w in wins])
        return {"fit": wins, "meta": meta}

    return _cached(tok, f"wiki_train_{count}x{length}", build)
# ------------------------------------------------------------------------------------------------
# FOCUS calibration set (arXiv:2608.01847)


def wikitext_train_blocks(tok, seqlen: int = 2048) -> tuple[list[torch.Tensor], dict]:
    """The FOCUS paper's calibration set: WikiText-2 raw *train* @ ``WIKI_REVISION``, built as the official
    AngelSlim ``QATDataset._build_from_raw``: every row tokenized on its own (with the tokenizer's special
    tokens, e.g. Llama's BOS), all rows concatenated, cut into non-overlapping ``seqlen`` blocks, tail
    dropped. The paper states 1,248 samples of 2,048 tokens; the Llama-3 tokenizer gives 1,206."""

    def build():
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKI_REVISION, split="train")
        ids: list[int] = []
        for text in ds["text"]:
            ids.extend(tok(text)["input_ids"])
        n = len(ids) // seqlen
        wins = [torch.tensor(ids[i * seqlen:(i + 1) * seqlen], dtype=torch.long).unsqueeze(0) for i in range(n)]
        meta = dict(revision=WIKI_REVISION, split="train", rows=len(ds), total_tokens=len(ids), blocks=n,
                    window_tokens=seqlen, token_sha256=[sha(w) for w in wins])
        return wins, meta

    return _cached(tok, f"wikitext2_train_{seqlen}", build)


def calibration_windows(tok, count: int, seqlen: int = CALIB_TOKENS) -> tuple[list[torch.Tensor], dict]:
    """``count`` math/code calibration windows of ``seqlen`` tokens (half from each stream), and their record.

    ``seqlen`` 512 is :func:`calibration_sets_extended`'s fit set (the standard 128 windows first, so the sets are
    nested). Any other length draws one window per document of at least ``seqlen`` tokens from the same pinned
    streams (seed ``SEED + 3``), skipping the development documents, so development KL stays held out."""
    if seqlen == CALIB_TOKENS:
        data = calibration_sets_extended(tok, count)
        return data["fit"], dict(seqlen=seqlen, count=count, fit=data["meta"]["fit"], streams=STREAMS)
    if count % 2:
        raise ValueError(f"count must be even (half math, half code), got {count}")

    def build():
        dev = {r["document_sha256"] for r in calibration_sets(tok)["meta"]["dev"]}
        rows = []
        for source in ("math", "code"):
            rows += _stream_windows(tok, source, count // 2, dev, SEED + 3, length=seqlen)
        meta = dict(seqlen=seqlen, count=count, streams=STREAMS, seed=SEED + 3,
                    fit=[dict(source=r["source"], document_sha256=r["document_sha256"], offset=r["offset"],
                              token_sha256=sha(r["ids"])) for r in rows])
        return [r["ids"] for r in rows], meta

    return _cached(tok, f"calibration_v1_windows{count}x{seqlen}", build)
