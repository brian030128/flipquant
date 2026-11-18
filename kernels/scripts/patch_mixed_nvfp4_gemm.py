#!/usr/bin/env python3
"""Offset-aware SASS patcher for the mixed E0M3/E2M1 nvfp4 GEMM.

PTX has no e0m3 token, so every mma.sync in src/collective/mma_sm120_mixed.hpp compiles as plain
e2m1 x e2m1 and the real format is installed here, by setting bits 14:15 of the second encoding
word of each OMMA to the format its dispatch site stands for:

    site 0 -> e2m1 x e2m1 (0b00, native -- left unpatched)
    site 1 -> e0m3 x e2m1 (0b01)
    site 2 -> e2m1 x e0m3 (0b10)
    site 3 -> e0m3 x e0m3 (0b11)

HOW SITES ARE IDENTIFIED
------------------------
Each site's atom emits a leading identity `prmt.b32 %sfa_t, %sfa, %sfa, SEL` whose selector is
unique to that site (see mma_sm120_mixed.hpp: the prmt exists to keep the four arms textually
distinct so ptxas cannot tail-merge them, and doubles as this tag). So an OMMA's site is read off
the PRMT that defines its SFA register operand.

This replaces an earlier scheme that sorted all OMMA occurrences by address and grouped them into
consecutive runs of 4. That worked when the format branch lived inside the atom and the four
mma.sync calls were emitted back-to-back at each call site. It is wrong for the current kernel:
the dispatch was hoisted to whole-k_tile granularity to stop ptxas from if-converting it, so the
four arms are now large blocks that ptxas interleaves in the instruction stream. Their address
ranges overlap (verified: site 0 spans 0x53a0-0x6c00 while site 1 spans 0x4e10-0x6870), so no
address-ordering rule recovers the grouping. The PRMT tag does, exactly.

SASS address -> file offset
---------------------------
Instruction addresses from cuobjdump are section-relative. Rather than assume a section layout,
the base is solved for: every offset in the file where the first OMMA's 16 bytes occur is a
candidate base, and the unique candidate is the one at which *every* parsed OMMA's encoding also
matches. Instructions are 16 bytes, so file_offset = base + sass_address.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
from collections import Counter

FIELD_SHIFT = 14

# How far back to look for the instruction that writes an OMMA's SFA operand. The scan stops at
# the *first* writer regardless, so a generous window only costs time on a pathological miss --
# whereas too small a window turns a legal build into a hard parse failure. The generated PTX
# paths put whole jump tables between one OMMA and the next, and ptxas CSEs the identity PRMTs
# toward the top of a straight-line block, so distances scale with the table size.
SCAN_WINDOW = 200000
# prmt selector -> (site index, format name, format bits)
SITE_BY_SELECTOR = {
    0x3210: (0, "e2m1_e2m1", 0b00),
    0x3214: (1, "e0m3_e2m1", 0b01),
    0x3254: (2, "e2m1_e0m3", 0b10),
    0x3654: (3, "e0m3_e0m3", 0b11),
}
BITS_BY_SITE = {site: bits for (site, _, bits) in SITE_BY_SELECTOR.values()}

# An OMMA line: address, optional predicate, opcode, then its register operands. The 5th register
# operand is the SFA the instruction reads.
OMMA_RE = re.compile(
    r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T\d]+\s+)?"
    r"(OMMA\.SF\.\S+)\s+"
    r"R\d+, R\d+(?:\.reuse)?, R\d+(?:\.reuse)?, R\d+(?:\.reuse)?, (R\d+)(?:\.reuse)?, "
    r"[^;]*;\s*"
    r"/\* 0x([0-9a-fA-F]{16}) \*/"
)
SECOND_WORD_RE = re.compile(r"/\* 0x([0-9a-fA-F]{16}) \*/")

# Any instruction whose first operand is a plain register writes that register. Store-style
# instructions put a bracketed address first (`STS [R0+0x10], R76`) so they do not match, which is
# what we want -- their register operand is a source.
#
# Operands can carry a `.reuse` suffix and the opcode can be predicated, both of which an earlier
# version of this script failed to allow for. That was not a cosmetic bug: a PRMT it declined to
# match did not stop the backward scan, so the scan ran past it to an *older* PRMT that happened to
# write the same register number, and silently attributed those OMMAs to the wrong format site
# (observed as a 128/132/128/124 census where all four must be 128). Hence the structure below:
# find the nearest writer of the register, whatever it is, and insist it be a tagged PRMT.
DEF_RE = re.compile(
    r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T\d]+\s+)?([A-Z][A-Z0-9._]*)\s+(R\d+)(?:\.reuse)?\s*,"
)
PRMT_SEL_RE = re.compile(r"PRMT\S*\s+R\d+(?:\.reuse)?\s*,\s*[^,]+,\s*(0x[0-9a-fA-F]+)\s*,")


def parse_ommas(sass: str):
    """Return [(sass_address, word0, word1, site, selector)] for every OMMA in the dump."""
    lines = sass.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = OMMA_RE.search(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        sfa_reg = m.group(3)
        word0 = int(m.group(4), 16)

        # The second encoding word sits on the following continuation line.
        word1 = None
        for j in range(i + 1, min(i + 4, len(lines))):
            m2 = SECOND_WORD_RE.search(lines[j])
            if m2:
                word1 = int(m2.group(1), 16)
                break
        if word1 is None:
            raise RuntimeError(f"no second encoding word after OMMA at 0x{addr:x}")

        # Walk back to the nearest instruction that writes this OMMA's SFA operand, and require
        # it to be one of our tagged identity PRMTs. Anything else is a parse failure, not a
        # reason to keep looking -- see the DEF_RE comment.
        definer = None
        for j in range(i - 1, max(0, i - SCAN_WINDOW), -1):
            md = DEF_RE.search(lines[j])
            if md and md.group(2) == sfa_reg:
                definer = lines[j]
                break
        if definer is None:
            raise RuntimeError(
                f"OMMA at 0x{addr:x} reads {sfa_reg} with no visible definition above it"
            )
        mp = PRMT_SEL_RE.search(definer)
        if not mp:
            raise RuntimeError(
                f"OMMA at 0x{addr:x}: {sfa_reg} is defined by a non-PRMT instruction, so its "
                f"format site cannot be identified:\n  {definer.strip()}\n"
                "Did ptxas fold the per-site identity prmt away, or was the atom changed?"
            )
        selector = int(mp.group(1), 16)
        if selector not in SITE_BY_SELECTOR:
            raise RuntimeError(
                f"OMMA at 0x{addr:x} tagged with unknown prmt selector 0x{selector:x}; "
                f"expected one of {sorted(hex(s) for s in SITE_BY_SELECTOR)}"
            )
        site = SITE_BY_SELECTOR[selector][0]
        out.append((addr, word0, word1, site, selector))
    return out


def solve_base(data: bytes, ommas) -> int:
    """Find the unique file offset B with data[B + addr : B + addr + 16] == encoding, for all."""
    addr0, w0, w1, _, _ = ommas[0]
    needle = struct.pack("<QQ", w0, w1)
    candidates = []
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx == -1:
            break
        candidates.append(idx - addr0)
        start = idx + 1

    good = []
    for base in candidates:
        if base < 0:
            continue
        if all(
            data[base + a : base + a + 16] == struct.pack("<QQ", x, y)
            for a, x, y, _, _ in ommas
        ):
            good.append(base)
    if len(good) != 1:
        raise RuntimeError(
            f"could not uniquely locate the kernel text in the binary "
            f"({len(good)} candidate base offsets matched all {len(ommas)} OMMAs)"
        )
    return good[0]


def verify_undisassemblable(before: str, check: str, ommas, counts: Counter) -> bool:
    """Byte-exact verification for a disassembler that cannot print the patched opcodes.

    Such a disassembler (observed: CUDA 13.0 cuobjdump) replaces the whole instruction line --
    address comment included -- with just the two encoding words, but keeps the listing
    line-for-line aligned. So diff the pre- and post-patch listings line by line: the ONLY lines
    that may differ are the patched OMMAs' two lines, and each must now carry exactly the
    encoding that was written.
    """
    got_native = Counter(re.findall(r"OMMA\.SF\.\d+\.F32\.(E\dM\d)\.(E\dM\d)\.", check))
    if +got_native != +Counter({("E2M1", "E2M1"): counts[0]}):
        return False
    a, b = before.split("\n"), check.split("\n")
    if len(a) != len(b):
        print("VERIFY FAILED: pre/post-patch listings are not line-aligned", file=sys.stderr)
        return False
    # Line index of each baseline OMMA -> the encoding it must have after patching. Addresses
    # repeat across kernels, so index by line, not by address.
    want = {}
    it = iter(ommas)
    for i, line in enumerate(a):
        if OMMA_RE.search(line):
            addr, w0, w1, site, _ = next(it)
            bits = BITS_BY_SITE[site]
            if bits:
                want[i] = (w0, w1 | (bits << FIELD_SHIFT))
    for i, (la, lb) in enumerate(zip(a, b)):
        if la == lb or (i - 1) in want:   # the continuation line of a patched OMMA may differ
            continue
        if i not in want:
            print(f"VERIFY FAILED: listing line {i} changed but is not a patched OMMA:\n"
                  f"  before: {la.strip()}\n  after:  {lb.strip()}", file=sys.stderr)
            return False
    for i, (w0, w1) in want.items():
        m0 = SECOND_WORD_RE.search(b[i])
        m1 = SECOND_WORD_RE.search(b[i + 1]) if i + 1 < len(b) else None
        if not (m0 and m1 and int(m0.group(1), 16) == w0 and int(m1.group(1), 16) == w1):
            print(f"VERIFY FAILED: listing line {i} does not carry the patched encoding "
                  f"0x{w0:016x} 0x{w1:016x}:\n  {b[i].strip()}", file=sys.stderr)
            return False
    want_n = len(want)
    print(f"WARNING: this cuobjdump cannot disassemble E0M3 OMMA encodings (CUDA 13.0 prints an "
          f"empty mnemonic). Verified at the byte level instead: {counts[0]} native OMMAs still "
          f"disassemble as E2M1.E2M1 and all {want_n} patched OMMAs carry exactly the intended "
          f"encoding. Re-run with a CUDA >= 13.1 cuobjdump for the opcode-text census.")
    return True


def default_cuobjdump() -> pathlib.Path:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and (pathlib.Path(cuda_home) / "bin" / "cuobjdump").exists():
        return pathlib.Path(cuda_home) / "bin" / "cuobjdump"
    found = shutil.which("cuobjdump")
    if found:
        return pathlib.Path(found)
    return pathlib.Path("/usr/local/cuda/bin/cuobjdump")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--cuobjdump", type=pathlib.Path, default=default_cuobjdump(),
        help="cuobjdump to use (default: $CUDA_HOME/bin/cuobjdump, else the one on PATH, else "
             "/usr/local/cuda/bin/cuobjdump)")
    parser.add_argument(
        "--expect-sites", type=lambda s: sorted({int(x) for x in s.split(",") if x != ""}),
        default=None, metavar="LIST",
        help="exact set of format sites the build must contain, e.g. '0,1' for a build with the "
             "B operand pinned to E2M1 (weights-as-A, MIXFP4_B_ALL_E2M1). Stricter than "
             "--allow-missing-sites: any other census is a hard failure.")
    parser.add_argument(
        "--mark-patched", metavar="NAME", default=None,
        help="after a verified patch, rewrite the unique ASCII marker 'NAME:UNPATCHED' in the "
             "binary to 'NAME:PATCHED__' (same length). Used by libflipquant_sm120.so, which "
             "refuses to run while its marker still says UNPATCHED.")
    parser.add_argument(
        "--allow-missing-sites", action="store_true",
        help="accept a build that emits only some of the four format sites. Legitimate when one "
             "operand is pinned to E2M1 (-DMIXFP4_A_ALL_E2M1=1), which makes the two E0M3-on-A "
             "sites unreachable, so the kernel contains only sites 0 and 2. Equal counts among "
             "the sites that ARE present is still required.")
    args = parser.parse_args()

    data = bytearray(args.baseline.read_bytes())
    sass = subprocess.run(
        [str(args.cuobjdump), "--dump-sass", str(args.baseline)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout

    ommas = parse_ommas(sass)
    if not ommas:
        raise RuntimeError("cuobjdump found no OMMA instructions")

    counts = Counter(site for _, _, _, site, _ in ommas)
    print(f"found {len(ommas)} OMMAs; per-site counts: " +
          ", ".join(f"site {s}={counts[s]}" for s in sorted(counts)))
    if args.expect_sites is not None:
        if sorted(counts) != args.expect_sites:
            raise RuntimeError(
                f"site census {sorted(counts)} does not match --expect-sites {args.expect_sites}; "
                "the build does not contain the dispatch arms it was configured for")
        print(f"note: site census matches --expect-sites {args.expect_sites}")
    elif sorted(counts) != [0, 1, 2, 3]:
        if not args.allow_missing_sites:
            raise RuntimeError(
                f"expected all four dispatch sites to be present, got {sorted(counts)}. If one "
                f"operand is pinned to E2M1 this is expected -- pass --allow-missing-sites.")
        print(f"note: only sites {sorted(counts)} are present (--allow-missing-sites)")
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"the four sites should emit equal numbers of OMMAs, got {dict(counts)} -- ptxas may "
            "have tail-merged or duplicated an arm; do not proceed without re-verifying"
        )

    base = solve_base(bytes(data), ommas)
    print(f"kernel text base offset = 0x{base:x}")

    patched = 0
    for addr, word0, word1, site, _ in ommas:
        if ((word1 >> FIELD_SHIFT) & 0b11) != 0:
            raise RuntimeError(
                f"OMMA at 0x{addr:x}: expected baseline E2M1 selector bits to be zero, "
                f"got word1=0x{word1:016x}"
            )
        bits = BITS_BY_SITE[site]
        if bits == 0:
            continue  # site 0 is native e2m1 x e2m1; nothing to install
        off = base + addr
        data[off : off + 16] = struct.pack("<QQ", word0, word1 | (bits << FIELD_SHIFT))
        patched += 1

    if args.mark_patched is not None:
        old = f"{args.mark_patched}:UNPATCHED".encode()
        new = f"{args.mark_patched}:PATCHED__".encode()
        assert len(old) == len(new)
        hits = []
        start = 0
        while True:
            idx = data.find(old, start)
            if idx == -1:
                break
            hits.append(idx)
            start = idx + 1
        if len(hits) != 1:
            raise RuntimeError(
                f"--mark-patched: expected exactly one '{old.decode()}' marker in "
                f"{args.baseline}, found {len(hits)}")
        data[hits[0] : hits[0] + len(new)] = new
        print(f"patch marker at file offset 0x{hits[0]:x} set to '{new.decode()}'")

    # Write to a temporary file and only move it into place once the re-disassembly verifies, so
    # a failed patch can never leave a plausible-looking output behind for a build system to
    # consider up to date.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".patching")
    tmp.write_bytes(bytes(data))
    os.chmod(tmp, args.baseline.stat().st_mode)
    print(f"patched {patched} OMMAs (site 0 left native), output={args.output}")

    try:
        # Verify by re-disassembling: each site's format must now be visible in the opcode text.
        check = subprocess.run(
            [str(args.cuobjdump), "--dump-sass", str(tmp)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        got = Counter(re.findall(r"OMMA\.SF\.\d+\.F32\.(E\dM\d)\.(E\dM\d)\.", check))
        # Unary + drops zero counts, so a build with some sites absent compares equal on every
        # Python version (Counter equality only ignores zero entries from 3.10 on).
        expect = +Counter({
            ("E2M1", "E2M1"): counts[0],
            ("E0M3", "E2M1"): counts[1],
            ("E2M1", "E0M3"): counts[2],
            ("E0M3", "E0M3"): counts[3],
        })
        print("post-patch opcode census:", dict(got))
        if +got != expect:
            # Older disassemblers (observed: CUDA 13.0 cuobjdump) cannot render the E0M3 OMMA
            # encodings and print the instruction with an EMPTY mnemonic -- only the two encoding
            # words appear on the line. Fall back to an exact byte-level check in that case: the
            # native OMMAs must still disassemble as E2M1.E2M1, and every patched address must be
            # an empty-mnemonic line whose encoding is exactly what was written. Anything else
            # still fails.
            if not verify_undisassemblable(sass, check, ommas, counts):
                print(f"VERIFY FAILED: expected {dict(expect)}", file=sys.stderr)
                tmp.unlink(missing_ok=True)
                return 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, args.output)
    print("verify OK: every site disassembles to its intended format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
