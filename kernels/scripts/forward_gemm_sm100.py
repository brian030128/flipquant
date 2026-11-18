#!/usr/bin/env python3
"""Per-forward GEMM overhead of the SM100 per-tile kernel with the release 256x64 maps.

The SM100 counterpart of the paper's Appendix D.5 (`evaluation.latency gemm` on SM120) and the
authors' requirement #1: for every model, time the GEMM of each quantized projection at T tokens,
multiply by its module count and sum over a forward pass, for the stock NVFP4 kernel and for the
per-tile kernel with two tag sets per projection -- those of its module with the median E0M3 share
and of its module with the most E0M3 tiles. Overhead = per-tile / stock - 1.

Timing (in the driver's --forward mode): weights on B, a 512 MiB read before every call, the
activations read back in, weight copies rotating past 4x L2, CUPTI kernel time, 3 warmup + 30
calls; 3 rounds with the kernel order rotated; the median of the 90 samples per cell.

Standard library only (GB200 nodes have no torch): the release maps are read from their .pt zip
archives directly. Map unit 256x64, True = E0M3, grid [ceil(N/256), K/64].
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import urllib.request
import zipfile

# (out_features N, in_features K) per projection name: the HF configs of each release model.
SHAPES = {
    "llama3.2-1b-instruct": {"q_proj": (2048, 2048), "k_proj": (512, 2048), "v_proj": (512, 2048),
                             "o_proj": (2048, 2048), "gate_proj": (8192, 2048), "up_proj": (8192, 2048),
                             "down_proj": (2048, 8192)},
    "llama3.1-8b-instruct": {"q_proj": (4096, 4096), "k_proj": (1024, 4096), "v_proj": (1024, 4096),
                             "o_proj": (4096, 4096), "gate_proj": (14336, 4096), "up_proj": (14336, 4096),
                             "down_proj": (4096, 14336)},
    "mistral-7b-instruct": {"q_proj": (4096, 4096), "k_proj": (1024, 4096), "v_proj": (1024, 4096),
                            "o_proj": (4096, 4096), "gate_proj": (14336, 4096), "up_proj": (14336, 4096),
                            "down_proj": (4096, 14336)},
    "nemotron-nano-9b-v2": {"in_proj": (22656, 4480), "out_proj": (4480, 10240), "up_proj": (15680, 4480),
                            "down_proj": (4480, 15680), "q_proj": (5120, 4480), "k_proj": (1024, 4480),
                            "v_proj": (1024, 4480), "o_proj": (4480, 5120)},
    "phi-4": {"qkv_proj": (7680, 5120), "o_proj": (5120, 5120), "gate_up_proj": (35840, 5120),
              "down_proj": (5120, 17920)},
    "qwen3.8-27b": {"q_proj": (12288, 5120), "k_proj": (1024, 5120), "v_proj": (1024, 5120),
                    "o_proj": (5120, 6144), "in_proj_qkv": (10240, 5120), "in_proj_z": (6144, 5120),
                    "in_proj_a": (48, 5120), "in_proj_b": (48, 5120), "out_proj": (5120, 6144),
                    "gate_proj": (17408, 5120), "up_proj": (17408, 5120), "down_proj": (5120, 17408)},
    "qwen3-8b": {"q_proj": (4096, 4096), "k_proj": (1024, 4096), "v_proj": (1024, 4096),
                 "o_proj": (4096, 4096), "gate_proj": (12288, 4096), "up_proj": (12288, 4096),
                 "down_proj": (4096, 12288)},
    "qwen3-1.7b": {"q_proj": (2048, 2048), "k_proj": (1024, 2048), "v_proj": (1024, 2048),
                   "o_proj": (2048, 2048), "gate_proj": (6144, 2048), "up_proj": (6144, 2048),
                   "down_proj": (2048, 6144)},
}
# The release maps (edgeai-lab Hugging Face repos).
REPOS = {
    "llama3.2-1b-instruct": "edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant",
    "llama3.1-8b-instruct": "edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant",
    "mistral-7b-instruct": "edgeai-lab/Mistral-7B-Instruct-v0.3-FlipQuant",
    "nemotron-nano-9b-v2": "edgeai-lab/NVIDIA-Nemotron-Nano-9B-v2-FlipQuant",
    "phi-4": "edgeai-lab/phi-4-FlipQuant",
    "qwen3.8-27b": "edgeai-lab/Qwen3.8-27B-FlipQuant",
    "qwen3-8b": "edgeai-lab/Qwen3-8B-FlipQuant",
    "qwen3-1.7b": "edgeai-lab/Qwen3-1.7B-FlipQuant",
}
#: release revisions pinned by the authors (the others: the repo's head at run time, recorded in the output)
REVISIONS = {"llama3.2-1b-instruct": "1896c62f9e45dc734fc0213468362d61ec23e0a8"}
PAPER_MODELS = ["llama3.2-1b-instruct", "llama3.1-8b-instruct", "mistral-7b-instruct", "nemotron-nano-9b-v2", "phi-4", "qwen3.8-27b"]
TOKENS = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CONFIGS = ["stock", "typical", "worst"]


# ---------------------------------------------------------------- reading .pt maps without torch
class _Tensor:
    def __init__(self, data, offset, size, stride):
        self.data, self.offset, self.size, self.stride = data, offset, tuple(size), tuple(stride)

    def at(self, i, j):
        return self.data[self.offset + i * self.stride[0] + j * self.stride[1]] != 0


def _rebuild_tensor(storage, offset, size, stride, *_):
    kind, data = storage
    if kind != "BoolStorage":
        raise ValueError(f"expected a bool tile map, got {kind}")
    return _Tensor(data, offset, size, stride)


class _MapUnpickler(pickle.Unpickler):
    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unexpected class {module}.{name} in a map file")

    def persistent_load(self, pid):
        _, kind, key, _location, _numel = pid
        return kind, self.archive.read(f"{self.prefix}/data/{key}")


def load_map(path):
    with zipfile.ZipFile(path) as archive:
        pkl = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with archive.open(pkl) as file:
            obj = _MapUnpickler(file, archive, prefix).load()
    if not (isinstance(obj, dict) and obj.get("format") == "flipquant-map/1" and obj.get("unit") == "256x64"):
        raise ValueError(f"{path}: not a flipquant-map/1 256x64 map")
    return obj["tiles"], obj.get("meta", {})


# ---------------------------------------------------------------- maps, tags, spec files
def fetch(repo, filename, dest, token, revision=None):
    if revision is None:
        api = urllib.request.Request(f"https://huggingface.co/api/models/{repo}",
                                     headers={"Authorization": f"Bearer {token}"})
        revision = json.load(urllib.request.urlopen(api, timeout=60))["sha"]
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    dest.write_bytes(urllib.request.urlopen(request, timeout=300).read())
    return revision, hashlib.sha256(dest.read_bytes()).hexdigest()


def write_kernel_map(tensor, n, k, path):
    """The deployed storage: one bit per 256x64 tile, 1 = E0M3; a row per 256-wide N tile of
    ceil(K/2048) little-endian uint32 words, bit j of word w = K64 block 32w + j."""
    words = (k // 64 + 31) // 32
    values = bytearray()
    for r in range((n + 255) // 256):
        row = [0] * words
        for c in range(k // 64):
            if tensor.at(r, c):
                row[c // 32] |= 1 << (c % 32)
        for word in row:
            values += word.to_bytes(4, "little")
    path.write_bytes(bytes(values))
    return len(values)


def prepare(model, work, token):
    repo = REPOS[model]
    model_dir = work / model
    model_dir.mkdir(parents=True, exist_ok=True)
    map_path = model_dir / "flipquant_256x64.pt"
    revision, sha256 = fetch(repo, "flipquant_256x64.pt", map_path, token, REVISIONS.get(model))
    tiles, meta = load_map(map_path)
    groups = collections.defaultdict(list)
    for name, tensor in tiles.items():
        groups[name.rsplit(".", 1)[-1]].append((name, tensor))
    shapes = SHAPES[model]
    unknown = sorted(set(groups) - set(shapes))
    if unknown:
        raise ValueError(f"{model}: projections without a known shape: {unknown}")
    projections = {}
    for projection, modules in sorted(groups.items()):
        n, k = shapes[projection]
        grid = ((n + 255) // 256, k // 64)
        counts = []
        for name, tensor in modules:
            if tensor.size != grid:
                raise ValueError(f"{model} {name}: map grid {tensor.size}, expected {grid} for {n}x{k}")
            counts.append(sum(tensor.at(r, c) for r in range(grid[0]) for c in range(grid[1])))
        order = sorted(range(len(modules)), key=lambda i: counts[i])
        typical = order[(len(order) - 1) // 2]
        worst = counts.index(max(counts))
        entry = {"shape": [n, k], "modules": len(modules), "tiles": grid[0] * grid[1], "tags": {}}
        for tagset, index in (("typical", typical), ("worst", worst)):
            name, tensor = modules[index]
            path = model_dir / f"{projection}.{tagset}.u32"
            map_bytes = write_kernel_map(tensor, n, k, path)
            entry["tags"][tagset] = {"module": name, "e0m3_tiles": counts[index], "file": str(path),
                                     "map_bytes_per_module": map_bytes}
        entry["e0m3_tiles_total"] = sum(counts)
        projections[projection] = entry
    for config in CONFIGS:
        with open(model_dir / f"{config}.spec", "w") as spec:
            for projection, entry in projections.items():
                n, k = entry["shape"]
                mapfile = "-" if config == "stock" else entry["tags"][config]["file"]
                spec.write(f"{projection} {config} {n} {k} {entry['modules']} {mapfile}\n")
    return {"repo": repo, "revision": revision, "map_sha256": sha256, "map_meta": meta,
            "projections": projections}


# ---------------------------------------------------------------- measurement and aggregation
def run_rounds(model, build, work, tokens, rounds, reps):
    binary = {"stock": build / "mixed_nvfp4_gemm_sm100", "typical": build / "mixed_sub_n256_sm100",
              "worst": build / "mixed_sub_n256_sm100"}
    env = dict(os.environ, MIXFP4_PDL="0")  # isolated calls: no programmatic dependent launch
    raw = work / model / "raw"
    raw.mkdir(exist_ok=True)
    for r in range(rounds):
        shift = r * len(CONFIGS) // rounds
        for config in CONFIGS[shift:] + CONFIGS[:shift]:
            out = raw / f"round{r}_{config}.jsonl"
            out.unlink(missing_ok=True)
            subprocess.run([str(binary[config]), f"--forward={work / model / f'{config}.spec'}", f"--out={out}",
                            f"--tokens={','.join(map(str, tokens))}", f"--reps={reps}", "--warmup=3",
                            "--flush-mib=512"], check=True, env=env, stdout=subprocess.DEVNULL)


def aggregate(model, work, info, tokens, rounds):
    samples = collections.defaultdict(list)
    timers = set()
    for r in range(rounds):
        for config in CONFIGS:
            for text in (work / model / "raw" / f"round{r}_{config}.jsonl").read_text().splitlines():
                row = json.loads(text)
                samples[(row["projection"], config, row["tokens"])] += row["samples_us"]
                timers.add(row["timer"])
    gemm_us = {f"{p}|{c}|{t}": statistics.median(v) for (p, c, t), v in samples.items()}
    per_forward = {c: {} for c in CONFIGS}
    for config in CONFIGS:
        for t in tokens:
            per_forward[config][t] = sum(entry["modules"] * gemm_us[f"{p}|{config}|{t}"]
                                         for p, entry in info["projections"].items())
    overhead = {tag: {t: 100 * (per_forward[tag][t] / per_forward["stock"][t] - 1) for t in tokens}
                for tag in ("typical", "worst")}
    return {"timer": sorted(timers), "gemm_us": gemm_us, "per_forward_us": per_forward, "overhead_pct": overhead}


def summary(results, tokens, models, paper):
    lines = ["# SM100 per-forward GEMM overhead, per-tile kernel with release 256x64 maps", "",
             "Overhead of the GEMMs of a forward pass over the stock NVFP4 kernel (%): tags of each projection",
             "from its module with the median E0M3 share, and (in parentheses) from its module with the most",
             "E0M3 tiles. Weights on B; tuned 256x256x256 2-SM tile for both kernels.",
             f"The paper median is over the {len(paper)} paper models; the all-models median adds the optional ones.", "",
             "| T | " + " | ".join(models) + f" | median, {len(paper)} paper models | median, all |",
             "|---" * (len(models) + 3) + "|"]
    cells = {(scope, tag): [] for scope in ("paper", "all") for tag in ("typical", "worst")}
    small = {"typical": [], "worst": []}
    for t in tokens:
        row, paper_row = [], []
        for tag in ("typical", "worst"):
            values = [results[m]["overhead_pct"][tag][t] for m in models]
            paper_values = [results[m]["overhead_pct"][tag][t] for m in paper]
            if t >= 128:
                cells[("all", tag)] += values
                cells[("paper", tag)] += paper_values
            else:
                small[tag] += [(v, m, t) for v, m in zip(paper_values, paper)]
            row.append(values)
            paper_row.append(paper_values)
        text = [f"{a:+.2f} ({b:+.2f})" for a, b in zip(*row)]
        for r in (paper_row, row):
            text.append(f"{statistics.median(r[0]):+.2f} ({statistics.median(r[1]):+.2f})")
        lines.append(f"| {t} | " + " | ".join(text) + " |")
    blank = " | ".join("" for _ in models)
    med = {key: statistics.median(v) for key, v in cells.items()}
    lines.append(f"| ≥128, median over (model, T) | {blank} | {med[('paper', 'typical')]:+.2f} "
                 f"({med[('paper', 'worst')]:+.2f}) | {med[('all', 'typical')]:+.2f} ({med[('all', 'worst')]:+.2f}) |")
    lines += ["", f"T < 128 over the {len(paper)} paper models (single cells, % over stock):", "",
              "| tags | max | min | median |", "|---|---|---|---|"]
    for tag in ("typical", "worst"):
        hi, lo = max(small[tag]), min(small[tag])
        lines.append(f"| {tag} | {hi[0]:+.2f} ({hi[1]}, T={hi[2]}) | {lo[0]:+.2f} ({lo[1]}, T={lo[2]}) | "
                     f"{statistics.median(x[0] for x in small[tag]):+.2f} |")
    lines += ["", "| model | map repo @ revision | map sha256 | E0M3 tiles | job @ commit |", "|---|---|---|---|---|"]
    for m in models:
        info = results[m]["map"]
        total = sum(e["e0m3_tiles_total"] for e in info["projections"].values())
        tiles = sum(e["tiles"] * e["modules"] for e in info["projections"].values())
        run = results[m]["run"]
        lines.append(f"| {m} | {info['repo']} @ {info['revision'][:10]} | {info['map_sha256'][:16]}… | "
                     f"{100 * total / tiles:.2f}% | {run['job']} @ {(run['commit'] or '')[:7]} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="maps, spec files and raw samples")
    parser.add_argument("--output", type=Path, required=True, help="results JSON; a .md summary is written next to it")
    parser.add_argument("--models", nargs="+", default=PAPER_MODELS + ["qwen3-8b", "qwen3-1.7b"])
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--merge", action="store_true",
                        help="keep the other models of an existing --output (every model records its own job and commit)")
    args = parser.parse_args()
    token = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"],
                         text=True, capture_output=True).stdout.strip()
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          text=True, capture_output=True).stdout.strip()
    if busy:
        raise SystemExit(f"GPU not idle: {busy}")
    commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], text=True,
                            capture_output=True).stdout.strip()
    results = {}
    if args.merge and args.output.exists():
        old = json.loads(args.output.read_text())
        if old["tokens"] != args.tokens:
            raise SystemExit(f"--merge: {args.output} has tokens {old['tokens']}, this run {args.tokens}")
        for model, entry in old["models"].items():
            entry.setdefault("run", {"job": old["job"], "commit": old["commit"], "gpu": old["gpu"]})
            for field in ("per_forward_us", "overhead_pct"):       # JSON keys are strings: T back to int
                entry[field] = {c: {int(t): v for t, v in by_t.items()} for c, by_t in entry[field].items()}
            results[model] = entry
    for model in args.models:
        info = prepare(model, args.work, token)
        print(f"{model}: {info['repo']} @ {info['revision'][:10]}, {len(info['projections'])} projections", flush=True)
        run_rounds(model, args.build, args.work, args.tokens, args.rounds, args.reps)
        results[model] = {"map": info, **aggregate(model, args.work, info, args.tokens, args.rounds),
                          "run": {"job": os.environ.get("SLURM_JOB_ID"), "commit": commit, "gpu": gpu}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"gpu": gpu, "commit": commit, "job": os.environ.get("SLURM_JOB_ID"),
                                           "account": os.environ.get("SLURM_JOB_ACCOUNT"),
                                           "protocol": "D.5 on SM100: 512 MiB flush, 4xL2 weight copies, CUPTI, "
                                                       "3 rounds x 30 calls, rotated order, PDL off",
                                           "tokens": args.tokens, "models": results}, indent=1) + "\n")
        print(f"{model}: overhead typical " + " ".join(f"T{t}:{results[model]['overhead_pct']['typical'][t]:+.2f}%"
                                                      for t in args.tokens), flush=True)
    models = [m for m in REPOS if m in results]
    args.output.with_suffix(".md").write_text(summary(results, args.tokens, models,
                                                      [m for m in PAPER_MODELS if m in results]))


if __name__ == "__main__":
    main()
