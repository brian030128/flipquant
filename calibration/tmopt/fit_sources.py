"""run_train_map.py --fit-source: fit windows from another corpus than the paper's math and code, for the calibration
robustness runs (paper Appendix C.4). Not part of the method: the paper's maps use ``--fit-source paper``.

Each source gives ``count`` windows of 512 tokens, drawn in the manner of ``fit_extension.stream_windows``:
- ``fineweb-edu``: generic web text, HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet; one window per document,
  at ``random.Random(seed).randrange(len - 512 + 1)``, from the first ``count`` documents of at least 512 tokens.
- ``ultrachat``: instruction data, HuggingFaceH4/ultrachat_200k train_sft shard 0; each conversation rendered with the
  model's chat template, then windows as for fineweb-edu.
- ``wikitext2-train``: the WikiText-2 (raw) training split, joined with ``"\\n\\n"`` as the evaluation joins its test
  split, cut into consecutive 512-token chunks; ``count`` chunks drawn without replacement with ``random.Random(seed)``.
  The training split shares no text with the test split used for evaluation.
The document sources skip the documents in ``skip`` (the paper's fit, development and published C4 evaluation
documents, by sha256), as the extension rule does. Every window is recorded by document sha256 (or chunk index), offset
and token sha256.
"""
import hashlib
import random

from fit_extension import TOKENS, token_sha

SEED = 20260931          # fit_extension.SEED + 3: a draw of its own, not the extension's
SOURCES = {
    'fineweb-edu': dict(repo='HuggingFaceFW/fineweb-edu', revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
                        path='sample/10BT/000_00000.parquet', field='text'),
    'ultrachat': dict(repo='HuggingFaceH4/ultrachat_200k', revision='8049631c405ae6576f93f445c6b8166f76f5505a',
                      path='data/train_sft-00000-of-00003-a3ecf92756993583.parquet', field='messages'),
    'wikitext2-train': dict(repo='Salesforce/wikitext', revision='b08601e04326c79dfdd32d625aee71d232d685c3',
                            path='wikitext-2-raw-v1/train-00000-of-00001.parquet', field='text'),
}


def _rows(meta):
    from datasets import load_dataset
    return load_dataset(meta['repo'], revision=meta['revision'], data_files={'train': meta['path']}, split='train',
                        streaming=True)


def _document_windows(tok, meta, name, count, skip, seed):
    rng, seen, out = random.Random(seed), set(skip), []
    for row in _rows(meta):
        if name == 'ultrachat':
            text = tok.apply_chat_template(row['messages'], tokenize=False)
        else:
            text = row[meta['field']]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tok(text, return_tensors='pt', add_special_tokens=name != 'ultrachat').input_ids
        if ids.shape[1] < TOKENS:
            continue
        offset = rng.randrange(ids.shape[1] - TOKENS + 1)
        out.append(dict(document_sha256=digest, offset=offset, ids=ids[:, offset:offset + TOKENS].clone()))
        if len(out) == count:
            return out
    raise RuntimeError(f'{name}: only {len(out)} usable documents')


def _wikitext_windows(tok, meta, count, seed):
    text = '\n\n'.join(row[meta['field']] for row in _rows(meta))
    ids = tok(text, return_tensors='pt').input_ids
    chunks = ids.shape[1] // TOKENS
    picked = random.Random(seed).sample(range(chunks), count)
    return [dict(chunk=c, offset=c * TOKENS, ids=ids[:, c * TOKENS:(c + 1) * TOKENS].clone()) for c in picked], dict(
        tokens=ids.shape[1], chunks=chunks, text_sha256=hashlib.sha256(text.encode()).hexdigest())


def windows(tok, name, count, skip):
    """``count`` fit windows from source ``name`` and their record."""
    meta = SOURCES[name]
    used = set().union(*skip.values())
    extra = {}
    if name == 'wikitext2-train':
        out, extra = _wikitext_windows(tok, meta, count, SEED)
    else:
        out = _document_windows(tok, meta, name, count, used, SEED)
        docs = [r['document_sha256'] for r in out]
        assert len(set(docs)) == len(docs) and not set(docs) & used
    records = [dict({k: v for k, v in r.items() if k != 'ids'}, token_sha256=token_sha(r['ids'])) for r in out]
    return [r['ids'] for r in out], dict(
        source=name, dataset=dict(meta), seed=SEED, windows=len(records), tokens_per_window=TOKENS,
        rule=__doc__.split('\n\n')[1], skipped_documents={k: len(v) for k, v in skip.items()}, records=records, **extra)
