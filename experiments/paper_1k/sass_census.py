"""Static SASS census of a kernel build's GEMM function (NVFP4-RaZeR experiments/paper_extra/C2_sass.py, ported for item 9).
CPU (cuobjdump of the built library).

The GEMM function (cutlass::device_kernel) is split into basic blocks (leaders: branch targets and the instructions
after a branch or EXIT). Loops are the back edges (a branch to a lower address). For each loop whose body holds OMMAs,
the minimum and the maximum OMMA count over every path from the loop head to the back edge are computed on the body's
acyclic graph (inner back edges dropped, and counted). Static counts, not measured counters.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter

INSN = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T\d]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);')


def gemm_function(sass: str) -> str:
    parts = re.split(r'\n\s*Function : ', sass)
    for p in parts:
        if p.startswith('_ZN7cutlass13device_kernel'):
            return p
    raise SystemExit('no cutlass::device_kernel in the disassembly')


def parse(text: str) -> list[dict]:
    out = []
    for m in INSN.finditer(text):
        addr, pred, op, args = int(m.group(1), 16), (m.group(2) or '').strip(), m.group(3), m.group(4)
        target = None
        if op.split('.')[0] in ('BRA', 'BRX', 'JMP', 'CALL'):
            t = re.search(r'0x([0-9a-f]+)', args)
            target = int(t.group(1), 16) if t else None
        out.append(dict(addr=addr, pred=pred, op=op, base=op.split('.')[0], args=args.strip(), target=target))
    return out


def blocks(insns: list[dict]) -> dict:
    addrs = [i['addr'] for i in insns]
    index = {a: k for k, a in enumerate(addrs)}
    leaders = {addrs[0]}
    for k, i in enumerate(insns):
        if i['base'] in ('BRA', 'BRX', 'JMP', 'EXIT', 'RET'):
            if i['target'] is not None and i['target'] in index:
                leaders.add(i['target'])
            if k + 1 < len(insns):
                leaders.add(addrs[k + 1])
    starts = sorted(leaders)
    bl = {}
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else addrs[-1] + 16
        body = [i for i in insns[index[s]:] if i['addr'] < e]
        last = body[-1]
        succ = []
        uncond = not last['pred'] or last['pred'] in ('@PT', '@UPT')
        conditional_uniform = '.U' in last['op'] and ('!UP' in last['args'] or 'UP' in last['args'])
        if last['base'] in ('BRA', 'JMP'):
            if last['target'] is not None:
                succ.append(last['target'])
            if not uncond or conditional_uniform:
                succ.append(e)
        elif last['base'] in ('EXIT', 'RET'):
            if not uncond:
                succ.append(e)
        elif last['base'] == 'BRX':
            succ.append(('BRX', last['addr']))
        else:
            succ.append(e)
        bl[s] = dict(start=s, end=e, omma=sum(1 for i in body if i['base'] == 'OMMA'),
                     succ=[x for x in succ if x in index or isinstance(x, tuple)])
    return bl


def loop_paths(bl: dict) -> list[dict]:
    """For each back edge (u -> h, h <= u) whose body holds OMMAs: min / max OMMAs over head->u paths in the body."""
    res = []
    order = sorted(bl)
    for u in order:
        for h in bl[u]['succ']:
            if isinstance(h, tuple) or h > u:
                continue
            body = [b for b in order if h <= b <= u]
            omma = sum(bl[b]['omma'] for b in body)
            if not omma:
                continue
            lo, hi = {h: bl[h]['omma']}, {h: bl[h]['omma']}
            inner_back = 0
            for b in body:
                if b not in lo:
                    continue
                for s in bl[b]['succ']:
                    if isinstance(s, tuple):
                        continue
                    if s <= b:
                        if not (b == u and s == h):
                            inner_back += 1
                        continue
                    if s > u:
                        continue
                    lo[s] = min(lo.get(s, 1 << 30), lo[b] + bl[s]['omma'])
                    hi[s] = max(hi.get(s, -1), hi[b] + bl[s]['omma'])
            res.append(dict(head=hex(h), latch=hex(u), blocks=len(body), omma_in_body=omma,
                            omma_per_iteration_min=lo.get(u), omma_per_iteration_max=hi.get(u),
                            inner_back_edges=inner_back))
    return res


def census_text(sass: str) -> dict:
    """The census of a disassembly (amendment 20's census: OMMAs, predicated OMMAs, BRX / BRA / WARPSYNC, the OMMAs per
    path through the steady k-loop iteration and the tensor-pipe estimate at 4096^3)."""
    insns = parse(gemm_function(sass))
    ops = Counter(i['base'] for i in insns)
    omma = [i for i in insns if i['base'] == 'OMMA']
    loops = loop_paths(blocks(insns))
    inner = [lp for lp in loops if lp['inner_back_edges'] == 0 and lp['omma_per_iteration_max'] is not None]
    steady = min(inner, key=lambda lp: lp['blocks']) if inner else None
    per_iter = (steady['omma_per_iteration_min'], steady['omma_per_iteration_max']) if steady else (None, None)
    est = per_iter[1] * 32 * 8 * 1024 if per_iter[1] is not None else None
    return dict(instructions=len(insns), omma=len(omma), omma_predicated=sum(1 for i in omma if i['pred']),
                brx=ops.get('BRX', 0), bra=ops.get('BRA', 0), warpsync=ops.get('WARPSYNC', 0), bssy=ops.get('BSSY', 0),
                prmt=ops.get('PRMT', 0), omma_per_steady_iteration=dict(min=per_iter[0], max=per_iter[1]),
                tensor_pipe_estimate_4096=est, loops=loops)


def census_lib(lib, cuobjdump: str = 'cuobjdump') -> dict:
    sass = subprocess.run([cuobjdump, '--dump-sass', str(lib)], capture_output=True, text=True, check=True).stdout
    return census_text(sass)
