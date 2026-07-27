#!/usr/bin/env python3
"""Batch CFG-based pattern analysis for generated CUDA kernels.

Scans `<run_dir>/exp_*/kernel_*.cu`, builds a control-flow tree for every
`__global__` kernel (via `analyze_pattern.build_cfg`), and emits:

  * kernels.jsonl / kernels.csv  -- per-kernel features incl. CFG fields
  * ngrams_{1,2,3}.csv           -- call-sequence n-gram frequency
  * dedup_cfg_exact.json         -- groups by CFG hash (CFG structural equivalence)
  * clusters_cfg.json            -- clusters by normalized tree edit distance
  * turn_evolution.csv           -- per-exp turn-to-turn TED diffs
  * turn_evolution_summary.json
  * report.md                    -- human-readable summary incl. methodology

CFG clustering algorithm
------------------------

The goal is to group kernels whose control-flow shapes are "close" without
paying O(N^2) Zhang-Shasha tree edit distance (TED) across every pair. We run
two stages — an exact hash bucket, then pairwise TED on one representative
per bucket — and union-find the results.

1. Per-kernel CFG. Each `__global__` body is parsed into an ordered tree
   (`build_cfg`) whose leaves are `CALL:<name>`, `JMP:<kind>`, and asm
   markers, and whose interior nodes are `SEQ`/`IF`/`FOR`/`WHILE`/`DO`/
   `SWITCH`/`CASE`/`DEFAULT`. Files with multiple `__global__` kernels are
   wrapped in one top-level `SEQ`.

2. Exact bucket via `cfg_hash`. `cfg_hash` is a 128-bit Weisfeiler-Lehman
   hash over the ordered tree with `kind:name` labels. Two kernels collide
   iff their CFGs are identical in both structure *and* call-name order.
   We bucket by this hash first — typically collapsing large duplicate
   groups in O(N) — and run the rest of the algorithm on one representative
   per bucket.

3. Pairwise normalized TED on representatives. For every pair of
   representatives we compute
        cfg_ted_norm(a, b) = simple_distance(a, b) / max(|a|, |b|, 1)
   in [0, 1], where `simple_distance` is Zhang-Shasha with unit
   insert/delete/relabel cost (via `zss`). Because labels are `kind:name`,
   one edit covers both "different callee" and "different control
   structure" in the same currency. 0 means identical trees, 1 means no
   shared structure.

4. Single-linkage merge via union-find. Representatives are unioned
   whenever `cfg_ted_norm <= --ted-threshold` (default 0.1). Each connected
   component becomes one cluster; members of that cluster are the union of
   the hash buckets of every representative in the component.
   [TODO] Other options like:
    - Complete-linkage: max distance across the pair — all members must be close to all other members. Tight, compact clusters.
    - Average-linkage: mean pairwise distance. A middle ground.

5. Medoid selection. For each cluster with >1 representative, the medoid
   is the rep minimising total TED to the other reps (ties broken by
   `kernel_id`); `intra_ted_mean` is the mean pairwise TED-norm among
   reps. Singleton clusters use their sole member with `intra_ted_mean=0`.

6. Fallback. If the representative count exceeds `--max-pairs-ted-reps`
   (default 500 → ~125k pairs), the pairwise step is skipped and each
   hash bucket becomes its own cluster — a warning is logged and
   `clusters_cfg.json.used_pairwise_ted` is set to `false`.

Cost: hashing is O(N); pairwise TED is O(R^2) in the number of
representatives R (not the raw kernel count N). The hash prefilter keeps R
small enough to make the quadratic step tractable on real runs.

Usage:
    python analyze_patterns_batch.py <run_dir> [--output-dir DIR]
        [--reference-fns PATH] [--jobs N] [--ted-threshold 0.1]
        [--max-pairs-ted-reps 500]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from zss import simple_distance

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_REFERENCE_FNS = SCRIPT_DIR / "prompt_configs" / "hopper-no-hint.fns.yaml"
TED_CACHE_FILENAME = "ted_matrix.npz"
TED_HEATMAP_FILENAME = "ted_heatmap.png"

from analyze_pattern import (  # noqa: E402
    C_KEYWORDS_NOT_A_FN,
    FN_MODIFIERS,
    CFGNode,
    build_cfg,
    cfg_hash,
    cfg_sexpr,
    cfg_stats,
    cfg_to_zss,
    filter_block_text,
    parse_top_level,
    strip_comments,
)


def load_reference_fn_names(yaml_path: Path) -> set[str]:
    if not yaml_path.is_file():
        raise SystemExit(
            f"reference-fns YAML not found: {yaml_path}\n"
            f"Run `python extract_reference_fns.py` to generate it."
        )
    data = yaml.safe_load(yaml_path.read_text())
    fns = data.get("functions") if isinstance(data, dict) else data
    if not isinstance(fns, list):
        raise SystemExit(
            f"{yaml_path}: expected a list under 'functions' "
            f"(run extract_reference_fns.py to regenerate)"
        )
    return set(fns)


def get_model_name(run_root: Path) -> str:
    """Extract short model name from the first trajectory under
    `run_root/trajectories/`. Falls back to `run_root.name` when the
    trajectories dir is missing, empty, or malformed."""
    traj_dir = run_root / "trajectories"
    trajs = sorted(traj_dir.glob("*.json")) if traj_dir.is_dir() else []
    if not trajs:
        return run_root.name
    try:
        with trajs[0].open() as f:
            t = json.load(f)
    except (OSError, json.JSONDecodeError):
        return run_root.name
    m = t.get("info", {}).get("config", {}).get("model", {}).get("model_name", "")
    return m.split("/")[-1] if m else run_root.name

EXP_RE = re.compile(r"(exp_\d+)")
KERNEL_IDX_RE = re.compile(r"kernel_([a-zA-Z])(\d+)\.cu$")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}]*?>)?\s*\(")
STRUCT_KEYWORDS = ("for", "while", "do", "if", "switch", "asm", "__asm__", "__asm")


# ---------------------------------------------------------------------------
# Per-kernel extraction.

@dataclass
class KernelRecord:
    kernel_id: str
    path: str
    exp: str
    family: str        # first letter after "kernel_" (e.g. "t" or "v")
    index: int
    call_sequence: list[str] = field(default_factory=list)
    structural_counts: dict[str, int] = field(default_factory=dict)
    global_kernel_names: list[str] = field(default_factory=list)
    helpers: list[str] = field(default_factory=list)
    num_global: int = 0
    # CFG fields.
    cfg_hash: str = ""
    cfg_sexpr: str = ""
    cfg_node_count: int = 0
    cfg_depth: int = 0
    cfg_branches: int = 0
    cfg_loops: int = 0


def _parse_exp_and_index(path: Path) -> tuple[str, str, int]:
    m_exp = EXP_RE.search(str(path))
    exp = m_exp.group(1) if m_exp else "unknown"
    m_idx = KERNEL_IDX_RE.search(path.name)
    if m_idx:
        return exp, m_idx.group(1), int(m_idx.group(2))
    return exp, "?", -1


def _extract_calls_from_filtered_body(body: str) -> list[str]:
    """Return callees in order. `body` is the *filtered* kernel body produced
    by `filter_block_text`, so control-flow statements have already been
    emitted and other statements reduced to call-only lines."""
    calls: list[str] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == '"':
            j = i + 1
            while j < n:
                if body[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if body[j] == '"':
                    j += 1
                    break
                j += 1
            i = j
            continue
        if c == "'":
            j = i + 1
            while j < n:
                if body[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if body[j] == "'":
                    j += 1
                    break
                j += 1
            i = j
            continue
        m = CALL_RE.match(body, i)
        if m:
            name = m.group(1)
            if name not in C_KEYWORDS_NOT_A_FN and name not in FN_MODIFIERS:
                calls.append(name)
            i = m.end()
            continue
        i += 1
    return calls


def _count_structural(body: str) -> dict[str, int]:
    counts = {k: 0 for k in STRUCT_KEYWORDS}
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", body):
        tok = m.group(1)
        if tok in counts:
            counts[tok] += 1
    return counts


def analyze_one(args: tuple[Path, set[str]]) -> dict:
    path, reference_names = args
    src = path.read_text()

    exp, family, index = _parse_exp_and_index(path)
    kernel_id = f"{exp}/{path.name}"

    stripped = strip_comments(src)
    try:
        items = parse_top_level(stripped)
    except ValueError as e:
        # Truncated kernels (LLM output cut mid-function) leave unbalanced
        # braces. Skip the file rather than aborting the whole run — the
        # degraded record has num_global=0 so it's filtered out of
        # clustering and the heatmap downstream.
        print(f"[analyze_patterns_batch] WARNING: parse failed for "
              f"{kernel_id}: {e}; skipping", file=sys.stderr)
        items = []

    call_sequence: list[str] = []
    global_kernel_names: list[str] = []
    structural_totals: Counter[str] = Counter()
    helpers: list[str] = []
    num_global = 0
    cfg_roots: list[CFGNode] = []

    for item in items:
        if item.kind != "func":
            continue
        if "__global__" in item.qualifiers:
            num_global += 1
            global_kernel_names.append(item.name)
            filtered = filter_block_text(item.body, indent_spaces=0)
            call_sequence.extend(_extract_calls_from_filtered_body(filtered))
            for k, v in _count_structural(filtered).items():
                structural_totals[k] += v
            cfg_roots.append(build_cfg(item.body))
        elif item.name not in reference_names:
            helpers.append(item.name)

    structural_totals["call"] = len(call_sequence)

    # One tree per file. Multiple __global__ kernels are wrapped in a SEQ.
    if len(cfg_roots) == 1:
        root = cfg_roots[0]
    elif cfg_roots:
        root = CFGNode("SEQ", "", tuple(cfg_roots))
    else:
        root = CFGNode("SEQ", "", ())
    stats = cfg_stats(root)

    rec = KernelRecord(
        kernel_id=kernel_id,
        path=str(path),
        exp=exp,
        family=family,
        index=index,
        call_sequence=call_sequence,
        structural_counts=dict(structural_totals),
        global_kernel_names=global_kernel_names,
        helpers=helpers,
        num_global=num_global,
        cfg_hash=cfg_hash(root),
        cfg_sexpr=cfg_sexpr(root),
        cfg_node_count=stats["node_count"],
        cfg_depth=stats["depth"],
        cfg_branches=stats["branches"],
        cfg_loops=stats["loops"],
    )
    return asdict(rec)


# ---------------------------------------------------------------------------
# Aggregates.

def ngram_table(records: list[dict], n: int) -> list[dict]:
    corpus_count: Counter = Counter()
    doc_freq: Counter = Counter()
    exp_cov: defaultdict[tuple, set[str]] = defaultdict(set)
    for r in records:
        seq = r["call_sequence"]
        seen_here: set[tuple] = set()
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i:i + n])
            corpus_count[gram] += 1
            seen_here.add(gram)
        for gram in seen_here:
            doc_freq[gram] += 1
            exp_cov[gram].add(r["exp"])
    rows = []
    for gram, count in corpus_count.most_common():
        rows.append({
            "ngram": " ".join(gram),
            "count": count,
            "doc_freq": doc_freq[gram],
            "exp_cov": len(exp_cov[gram]),
        })
    rows.sort(key=lambda r: (-r["doc_freq"], -r["count"]))
    return rows


def exact_cfg_dedup(records: list[dict]) -> dict[str, list[str]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["num_global"] > 0:
            groups[r["cfg_hash"]].append(r["kernel_id"])
    return dict(groups)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Tree edit distance clustering.

def _rehydrate_zss(record: dict):
    """Rebuild the zss.Node tree from the stored sexpr-reachable fields.

    We don't pickle the CFGNode across the process pool, so analyze_one returns
    only the sexpr/hash/stats; here (in the driver process) we reparse the raw
    source to rebuild the zss tree for the representatives that need TED.
    Trees are cached per kernel_id to avoid doing this twice.
    """
    src = Path(record["path"]).read_text()
    stripped = strip_comments(src)
    items = parse_top_level(stripped)
    roots: list[CFGNode] = []
    for item in items:
        if item.kind == "func" and "__global__" in item.qualifiers:
            roots.append(build_cfg(item.body))
    if len(roots) == 1:
        root = roots[0]
    elif roots:
        root = CFGNode("SEQ", "", tuple(roots))
    else:
        root = CFGNode("SEQ", "", ())
    return cfg_to_zss(root)


class ZssCache:
    def __init__(self, records: list[dict]):
        self._by_id = {r["kernel_id"]: r for r in records}
        self._cache: dict[str, object] = {}

    def get(self, kernel_id: str):
        z = self._cache.get(kernel_id)
        if z is None:
            z = _rehydrate_zss(self._by_id[kernel_id])
            self._cache[kernel_id] = z
        return z


def ted_norm(a_rec: dict, b_rec: dict, cache: ZssCache) -> tuple[int, float]:
    za = cache.get(a_rec["kernel_id"])
    zb = cache.get(b_rec["kernel_id"])
    raw: float = simple_distance(za, zb)  # type: ignore[assignment]
    d = int(raw)
    denom = max(a_rec["cfg_node_count"], b_rec["cfg_node_count"], 1)
    return d, d / denom


def _pick_representatives(
    records: list[dict],
) -> tuple[list[dict], dict[str, list[str]]]:
    """Bucket eligible kernels by cfg_hash and pick one rep per bucket.

    Returns (rep_records, members_by_rep) — rep order is cfg_hash insertion
    order, which mirrors the input record order. members_by_rep maps rep
    kernel_id → sorted-by-input list of member kernel_ids.
    """
    eligible = [r for r in records if r["num_global"] > 0]
    hash_buckets: defaultdict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        hash_buckets[r["cfg_hash"]].append(r)
    rep_records = [bucket[0] for bucket in hash_buckets.values()]
    members_by_rep = {
        rep["kernel_id"]: [r["kernel_id"] for r in hash_buckets[rep["cfg_hash"]]]
        for rep in rep_records
    }
    return rep_records, members_by_rep


def _compute_pairwise_distances(
    rep_records: list[dict],
    cache: ZssCache,
) -> dict[tuple[int, int], float]:
    n = len(rep_records)
    distances: dict[tuple[int, int], float] = {}
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            _, dn = ted_norm(rep_records[i], rep_records[j], cache)
            distances[(i, j)] = dn
    dt = time.time() - t0
    print(
        f"[analyze_patterns_batch] pairwise TED: {n} reps, "
        f"{n * (n - 1) // 2} pairs in {dt:.1f}s"
    )
    return distances


def _distances_to_dense(
    rep_records: list[dict],
    distances: dict[tuple[int, int], float],
) -> np.ndarray:
    n = len(rep_records)
    M = np.zeros((n, n), dtype=np.float32)
    for (i, j), d in distances.items():
        M[i, j] = d
        M[j, i] = d
    return M


def save_ted_matrix(
    rep_records: list[dict],
    distances: dict[tuple[int, int], float],
    out_dir: Path,
) -> Path:
    """Persist the rep-level TED matrix to <out_dir>/ted_matrix.npz.

    Stores `matrix` (float32, NxN, symmetric, diagonal=0) and `labels`
    (object array of rep kernel_ids, in rep order).
    """
    M = _distances_to_dense(rep_records, distances)
    labels = np.array([r["kernel_id"] for r in rep_records], dtype=object)
    path = out_dir / TED_CACHE_FILENAME
    np.savez_compressed(path, matrix=M, labels=labels)
    print(f"[analyze_patterns_batch] saved TED matrix (n={len(rep_records)}) "
          f"to {path}")
    return path


def load_ted_matrix(
    out_dir: Path,
    rep_records: list[dict],
) -> dict[tuple[int, int], float] | None:
    """Reload cached distances if the cache matches the current reps.

    Returns a {(i, j): dist} dict keyed by indices into `rep_records`, or
    None if the cache is missing or its label set doesn't match the
    current rep kernel_ids. Re-indexes the loaded matrix if the cached
    order differs from the current rep order.
    """
    path = out_dir / TED_CACHE_FILENAME
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    cached_labels = [str(x) for x in data["labels"]]
    M = data["matrix"]
    current_ids = [r["kernel_id"] for r in rep_records]
    if set(cached_labels) != set(current_ids):
        return None
    pos_in_cache = {lbl: i for i, lbl in enumerate(cached_labels)}
    n = len(rep_records)
    distances: dict[tuple[int, int], float] = {}
    for i in range(n):
        ci = pos_in_cache[current_ids[i]]
        for j in range(i + 1, n):
            cj = pos_in_cache[current_ids[j]]
            distances[(i, j)] = float(M[ci, cj])
    print(f"[analyze_patterns_batch] loaded cached TED matrix (n={n}) "
          f"from {path}")
    return distances


def _expand_distances_by_exp(
    rep_records: list[dict],
    members_by_rep: dict[str, list[str]],
    distances: dict[tuple[int, int], float],
) -> tuple[list[str], np.ndarray, list[int], list[str]]:
    """Expand rep-level distances into a per-kernel matrix ordered by exp.

    Rows/columns are sorted by (exp, family, index) so every exp_* occupies
    a contiguous block. Within a block, pairs that share a rep have
    distance 0; cross-rep pairs use the precomputed rep-to-rep TED.

    Returns (labels, matrix, exp_offsets, exp_names). `exp_offsets` has
    length len(exp_names)+1 and marks the row index where each exp block
    starts (last entry is N).
    """
    kid_to_rep: dict[str, int] = {}
    for rep_idx, rep in enumerate(rep_records):
        for kid in members_by_rep[rep["kernel_id"]]:
            kid_to_rep[kid] = rep_idx

    def _sort_key(kid: str) -> tuple[str, str, int]:
        exp, name = kid.split("/", 1)
        m = KERNEL_IDX_RE.search(name)
        family = m.group(1) if m else ""
        idx = int(m.group(2)) if m else -1
        return (exp, family, idx)

    labels = sorted(kid_to_rep.keys(), key=_sort_key)
    N = len(labels)
    M = np.zeros((N, N), dtype=np.float32)
    rep_of = [kid_to_rep[kid] for kid in labels]
    for i in range(N):
        ri = rep_of[i]
        for j in range(i + 1, N):
            rj = rep_of[j]
            if ri == rj:
                continue
            key = (ri, rj) if ri < rj else (rj, ri)
            d = distances.get(key, 0.0)
            M[i, j] = d
            M[j, i] = d

    exp_names: list[str] = []
    exp_offsets: list[int] = [0]
    prev: str | None = None
    for i, kid in enumerate(labels):
        exp = kid.split("/", 1)[0]
        if exp != prev:
            if prev is not None:
                exp_offsets.append(i)
            exp_names.append(exp)
            prev = exp
    exp_offsets.append(N)
    return labels, M, exp_offsets, exp_names


def plot_ted_heatmap(
    rep_records: list[dict],
    members_by_rep: dict[str, list[str]],
    distances: dict[tuple[int, int], float],
    out_path: Path,
    run_dir: Path,
) -> None:
    """Upper-triangular heatmap of pairwise CFG TED over all eligible kernels.

    Rows/columns are ordered by (exp, family, index) so each exp_* occupies
    a contiguous block; black lines mark the block boundaries and y-axis
    ticks label each block with its exp name. Lower triangle AND diagonal
    are masked — self-distance is always 0, and intra-rep off-diagonal
    cells are 0 (identical CFG).
    """
    labels, M, exp_offsets, exp_names = _expand_distances_by_exp(
        rep_records, members_by_rep, distances,
    )
    N = len(labels)
    if N == 0:
        return
    mask = np.tril(np.ones_like(M, dtype=bool), k=-1)
    M_masked = np.ma.array(M, mask=mask)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    size = max(6.0, min(20.0, N * 0.2))
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(
        M_masked, cmap=cmap, vmin=0.0, vmax=1.0,
        interpolation="nearest", aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25)
    cbar.set_label("cfg_ted_norm", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Black lines at exp_* block boundaries.
    for off in exp_offsets[1:-1]:
        ax.axhline(off - 0.5, color="black", linewidth=0.6, alpha=0.7)
        ax.axvline(off - 0.5, color="black", linewidth=0.6, alpha=0.7)

    # One ytick per exp, placed at the centre of its block.
    centres = [
        (exp_offsets[i] + exp_offsets[i + 1] - 1) / 2
        for i in range(len(exp_names))
    ]
    ax.set_yticks(centres)
    ax.set_yticklabels(exp_names, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])

    run_root = run_dir.parent
    model_name = get_model_name(run_root)
    title = (
        f"{model_name} ({run_root.name}) — "
        f"Pairwise CFG TED (N={N} kernels, {len(rep_records)} reps)"
    )
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyze_patterns_batch] wrote heatmap to {out_path}")


def ensure_rep_distances(
    records: list[dict],
    cache: ZssCache,
    out_dir: Path,
    max_reps: int,
) -> tuple[list[dict], dict[str, list[str]], dict[tuple[int, int], float] | None, bool]:
    """Pick CFG-hash representatives and produce their pairwise TED matrix.

    The expensive O(R^2) TED computation is done here and persisted to
    `ted_matrix.npz` before any clustering, so re-running with a different
    `--ted-threshold` reuses the cached matrix and downstream crashes don't
    lose the work. Returns (rep_records, members_by_rep, distances,
    used_pairwise). `distances` is None and `used_pairwise` is False only on
    the hash-bucket fallback path (too many reps for pairwise).
    """
    rep_records, members_by_rep = _pick_representatives(records)
    if not rep_records:
        return rep_records, members_by_rep, {}, True

    cached = load_ted_matrix(out_dir, rep_records)
    if cached is not None:
        return rep_records, members_by_rep, cached, True

    if len(rep_records) > max_reps:
        print(
            f"[analyze_patterns_batch] WARNING: {len(rep_records)} CFG-hash "
            f"representatives exceeds --max-pairs-ted-reps ({max_reps}); "
            f"skipping pairwise TED. Clusters will equal hash buckets."
        )
        return rep_records, members_by_rep, None, False

    distances = _compute_pairwise_distances(rep_records, cache)
    save_ted_matrix(rep_records, distances, out_dir)
    return rep_records, members_by_rep, distances, True


def cluster_from_distances(
    rep_records: list[dict],
    members_by_rep: dict[str, list[str]],
    distances: dict[tuple[int, int], float] | None,
    threshold: float,
    used_pairwise: bool,
) -> list[dict]:
    """Pure clustering step on a precomputed distance matrix.

    Each cluster is `{"size": N, "medoid": kernel_id, "members": [...],
    "intra_ted_mean": float}`. When `used_pairwise` is False (hash-bucket
    fallback), each hash bucket becomes its own cluster.
    """
    if not used_pairwise or distances is None:
        clusters = [
            {
                "size": len(members_by_rep[rep["kernel_id"]]),
                "medoid": rep["kernel_id"],
                "members": sorted(members_by_rep[rep["kernel_id"]]),
                "intra_ted_mean": 0.0,
            }
            for rep in rep_records
        ]
        clusters.sort(key=lambda c: -c["size"])
        return clusters

    n = len(rep_records)
    uf = UnionFind(n)
    for (i, j), dn in distances.items():
        if dn <= threshold:
            uf.union(i, j)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    clusters: list[dict] = []
    for rep_indices in groups.values():
        members: list[str] = []
        for idx in rep_indices:
            members.extend(members_by_rep[rep_records[idx]["kernel_id"]])
        if len(rep_indices) == 1:
            medoid_idx = rep_indices[0]
            intra_mean = 0.0
        else:
            medoid_idx = min(
                rep_indices,
                key=lambda i: (
                    sum(
                        distances.get((min(i, j), max(i, j)), 0.0)
                        for j in rep_indices if j != i
                    ),
                    rep_records[i]["kernel_id"],
                ),
            )
            pair_count = len(rep_indices) * (len(rep_indices) - 1) // 2
            intra_sum = 0.0
            for a in range(len(rep_indices)):
                for b in range(a + 1, len(rep_indices)):
                    ia, ib = rep_indices[a], rep_indices[b]
                    intra_sum += distances.get((min(ia, ib), max(ia, ib)), 0.0)
            intra_mean = intra_sum / pair_count if pair_count else 0.0
        clusters.append({
            "size": len(members),
            "medoid": rep_records[medoid_idx]["kernel_id"],
            "members": sorted(members),
            "intra_ted_mean": round(intra_mean, 4),
        })
    clusters.sort(key=lambda c: (-c["size"], c["medoid"]))
    return clusters


# ---------------------------------------------------------------------------
# Turn evolution.

def turn_evolution(
    records: list[dict],
    cache: ZssCache,
) -> tuple[list[dict], dict]:
    by_exp: defaultdict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["index"] >= 0 and r["num_global"] > 0:
            by_exp[r["exp"]].append(r)

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for exp, kernels in sorted(by_exp.items()):
        kernels.sort(key=lambda r: (r["family"], r["index"]))
        ted_norms: list[float] = []
        for a, b in zip(kernels, kernels[1:]):
            if a["family"] != b["family"]:
                continue
            d, dn = ted_norm(a, b, cache)
            deltas = {
                k: b["structural_counts"].get(k, 0) - a["structural_counts"].get(k, 0)
                for k in set(a["structural_counts"]) | set(b["structural_counts"])
            }
            rows.append({
                "exp": exp,
                "family": a["family"],
                "from": a["index"],
                "to": b["index"],
                "cfg_ted": d,
                "cfg_ted_norm": round(dn, 4),
                "cfg_hash_changed": int(a["cfg_hash"] != b["cfg_hash"]),
                "delta_call": deltas.get("call", 0),
                "delta_for": deltas.get("for", 0),
                "delta_if": deltas.get("if", 0),
                "delta_while": deltas.get("while", 0),
            })
            ted_norms.append(dn)
        if kernels:
            first, last = kernels[0], kernels[-1]
            if first["kernel_id"] != last["kernel_id"]:
                _, fl_dn = ted_norm(first, last, cache)
            else:
                fl_dn = 0.0
            summary[exp] = {
                "n_turns": len(kernels),
                "mean_cfg_ted_norm": round(sum(ted_norms) / len(ted_norms), 4) if ted_norms else 0.0,
                "median_cfg_ted_norm": round(sorted(ted_norms)[len(ted_norms) // 2], 4) if ted_norms else 0.0,
                "first_vs_last_cfg_hash_equal": first["cfg_hash"] == last["cfg_hash"],
                "first_vs_last_ted_norm": round(fl_dn, 4),
            }
    return rows, summary


# ---------------------------------------------------------------------------
# Output writers.

def write_kernels_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_kernels_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "kernel_id", "exp", "family", "index", "path", "num_global", "num_calls",
        "num_helpers", "for", "while", "do", "if", "switch", "asm",
        "cfg_hash", "cfg_node_count", "cfg_depth", "cfg_branches", "cfg_loops",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            sc = r["structural_counts"]
            w.writerow({
                "kernel_id": r["kernel_id"],
                "exp": r["exp"],
                "family": r["family"],
                "index": r["index"],
                "path": r["path"],
                "num_global": r["num_global"],
                "num_calls": len(r["call_sequence"]),
                "num_helpers": len(r["helpers"]),
                "for": sc.get("for", 0),
                "while": sc.get("while", 0),
                "do": sc.get("do", 0),
                "if": sc.get("if", 0),
                "switch": sc.get("switch", 0),
                "asm": sc.get("asm", 0) + sc.get("__asm__", 0) + sc.get("__asm", 0),
                "cfg_hash": r["cfg_hash"][:16],
                "cfg_node_count": r["cfg_node_count"],
                "cfg_depth": r["cfg_depth"],
                "cfg_branches": r["cfg_branches"],
                "cfg_loops": r["cfg_loops"],
            })


def write_ngrams_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ngram", "count", "doc_freq", "exp_cov"])
        w.writeheader()
        w.writerows(rows)


METHODOLOGY = """\
## Methodology

Each `__global__` kernel is parsed into a control-flow tree whose leaves are
function calls labelled by callee name (`CALL:st_matrix`), jump statements
(`JMP:return`), and inline asm markers. Interior nodes are `SEQ`,
`IF`, `FOR`, `WHILE`, `DO`, `SWITCH`, `CASE`, `DEFAULT` in source order.
Condition expressions are dropped except for any embedded call statements,
which are prepended to the body's `SEQ` so work moved between condition and
body is still visible to the metrics. Files with multiple `__global__` kernels
are wrapped in a single top-level `SEQ`.

### Metrics

- **`cfg_hash`** — 128-bit Weisfeiler–Lehman hash over the ordered tree with
  node labels `kind:name`. Two kernels share a hash iff their CFGs are
  identical (same structure AND same call names in the same order). Used for
  exact dedup (`dedup_cfg_exact.json`).
- **`cfg_node_count`, `cfg_depth`, `cfg_branches`, `cfg_loops`** — raw size
  features. Useful at a glance; not used for clustering.
- **`cfg_ted`** — Zhang–Shasha tree edit distance (unit-cost insert / delete /
  relabel) between two CFGs. Because node labels are `kind:name`, one edit
  covers "different callee" and "different control structure" in the same
  currency. Computed via the `zss` library.
- **`cfg_ted_norm`** — `cfg_ted / max(cfg_node_count(a), cfg_node_count(b))`,
  in [0, 1]. 0 means identical trees, 1 means no shared structure. This is
  the primary similarity metric.

### Clustering (`clusters_cfg.json`)

1. Group kernels by `cfg_hash` (exact duplicates).
2. One representative per hash bucket enters pairwise `cfg_ted_norm`
   comparison; pairs with distance ≤ **{threshold}** merge under
   single-linkage.
3. Each cluster's medoid is the representative with minimum total TED to the
   others; ties broken by `kernel_id`. Single-member clusters use that
   member.
4. If the number of representatives exceeds **`--max-pairs-ted-reps`**
   ({max_reps}), the pairwise step is skipped and each hash bucket becomes
   its own cluster (a warning is logged).

### Turn evolution

For each experiment, adjacent turns (`kernel_*N.cu` → `kernel_*(N+1).cu`
within the same family letter) are compared via TED. `cfg_ted_norm` near 0
means the model only tweaked variable names or constants; values near 1
mean the kernel was restructured. `cfg_hash_changed` is a quick boolean for
"did the structure change at all". The `delta_*` columns track raw counts
of control-flow keywords for orthogonal context.

### Legacy fields

- `call_sequence` (and its n-gram tables `ngrams_{1,2,3}.csv`) is an
  extracted feature kept for orthogonal corpus-level analysis.

"""


def write_report(
    records: list[dict],
    ngrams1: list[dict],
    ngrams2: list[dict],
    ngrams3: list[dict],
    exact_cfg: dict[str, list[str]],
    clusters: list[dict],
    used_pairwise: bool,
    ev_summary: dict,
    ted_threshold: float,
    max_reps: int,
    path: Path,
) -> None:
    total = len(records)
    with_global = sum(1 for r in records if r["num_global"] > 0)
    largest_cfg_group = max((len(v) for v in exact_cfg.values()), default=0)
    multi_member_clusters = [c for c in clusters if c["size"] > 1]

    with path.open("w") as f:
        f.write("# Pattern analysis report (CFG + TED)\n\n")
        f.write(
            METHODOLOGY
            .replace("{threshold}", str(ted_threshold))
            .replace("{max_reps}", str(max_reps))
        )

        f.write("## Summary\n\n")
        f.write(f"- Total kernels analyzed: **{total}**\n")
        f.write(f"- Kernels with at least one `__global__`: **{with_global}**\n")
        f.write(f"- Distinct CFG hashes: **{len(exact_cfg)}** "
                f"(largest group: {largest_cfg_group})\n")
        f.write(f"- CFG clusters at TED norm ≤ {ted_threshold}: "
                f"**{len(clusters)}** (multi-member: {len(multi_member_clusters)})"
                + ("" if used_pairwise else " — hash-bucket fallback, pairwise TED skipped")
                + "\n\n")

        f.write("## Top 20 calls (by document frequency)\n\n")
        f.write("| call | count | doc_freq | exp_cov |\n|---|---|---|---|\n")
        for row in ngrams1[:20]:
            f.write(f"| `{row['ngram']}` | {row['count']} | {row['doc_freq']} | {row['exp_cov']} |\n")
        f.write("\n")

        f.write("## Top 10 bigrams\n\n")
        f.write("| bigram | count | doc_freq | exp_cov |\n|---|---|---|---|\n")
        for row in ngrams2[:10]:
            f.write(f"| `{row['ngram']}` | {row['count']} | {row['doc_freq']} | {row['exp_cov']} |\n")
        f.write("\n")

        f.write("## Top 10 trigrams\n\n")
        f.write("| trigram | count | doc_freq | exp_cov |\n|---|---|---|---|\n")
        for row in ngrams3[:10]:
            f.write(f"| `{row['ngram']}` | {row['count']} | {row['doc_freq']} | {row['exp_cov']} |\n")
        f.write("\n")

        f.write("## Largest CFG-hash duplicate groups\n\n")
        sorted_cfg_groups = sorted(
            ((h, members) for h, members in exact_cfg.items() if len(members) > 1),
            key=lambda kv: -len(kv[1]),
        )
        if not sorted_cfg_groups:
            f.write("_No CFG-hash duplicates found._\n\n")
        else:
            for i, (h, members) in enumerate(sorted_cfg_groups[:10], 1):
                f.write(f"**Group {i}** — `{h[:16]}` — {len(members)} members\n")
                for kid in members[:8]:
                    f.write(f"  - `{kid}`\n")
                if len(members) > 8:
                    f.write(f"  - ... and {len(members) - 8} more\n")
                f.write("\n")

        f.write("## Largest near-duplicate CFG clusters (TED norm ≤ "
                f"{ted_threshold})\n\n")
        for i, cluster in enumerate(multi_member_clusters[:10], 1):
            f.write(f"**Cluster {i}** — {cluster['size']} members "
                    f"(medoid: `{cluster['medoid']}`, "
                    f"mean intra-TED norm: {cluster['intra_ted_mean']})\n")
            for kid in cluster["members"][:8]:
                f.write(f"  - `{kid}`\n")
            if len(cluster["members"]) > 8:
                f.write(f"  - ... and {len(cluster['members']) - 8} more\n")
            f.write("\n")

        f.write("## Turn evolution (per experiment)\n\n")
        f.write("| exp | n_turns | mean_ted_norm | median_ted_norm | first_vs_last_hash_equal | first_vs_last_ted_norm |\n"
                "|---|---|---|---|---|---|\n")
        for exp, s in sorted(ev_summary.items()):
            f.write(f"| {exp} | {s['n_turns']} | {s['mean_cfg_ted_norm']} | "
                    f"{s['median_cfg_ted_norm']} | "
                    f"{str(s['first_vs_last_cfg_hash_equal']).lower()} | "
                    f"{s['first_vs_last_ted_norm']} |\n")


# ---------------------------------------------------------------------------
# Driver.

def discover_kernels(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("kernel_*.cu"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "")
    ap.add_argument("run_dir", type=Path,
                    help="Directory containing exp_*/kernel_*.cu")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--reference-fns", type=Path, default=DEFAULT_REFERENCE_FNS,
                    help="YAML file listing reference function names "
                         "(produced by extract_reference_fns.py)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--ted-threshold", type=float, default=0.1,
                    help="Clustering cutoff on cfg_ted_norm (default 0.1)")
    ap.add_argument("--max-pairs-ted-reps", type=int, default=500,
                    help="If more than this many CFG-hash representatives, "
                         "skip O(N^2) pairwise TED and fall back to "
                         "hash-bucket clustering (default 500)")
    ap.add_argument("--correct-only", action="store_true",
                    help="Analyze only kernels that passed evaluation "
                         "(walks <run_root>/success/ instead of kernels/). "
                         "Default output dir becomes pattern_analysis_correct/.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"run_dir not found: {run_dir}")

    if run_dir.name == "kernels":
        run_root = run_dir.parent
    elif (run_dir / "kernels").is_dir():
        run_root = run_dir
    else:
        run_root = run_dir.parent

    if args.correct_only:
        source_dir = run_root / "success"
        if not source_dir.is_dir():
            sys.exit(
                f"--correct-only: cannot locate sibling success/ at {source_dir}"
            )
        default_out_name = "pattern_analysis_correct"
    else:
        source_dir = (run_dir / "kernels") if (run_dir / "kernels").is_dir() else run_dir
        default_out_name = "pattern_analysis"

    out_dir: Path = args.output_dir or (run_root / default_out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_names = load_reference_fn_names(args.reference_fns)
    kernels = discover_kernels(source_dir)
    if not kernels:
        sys.exit(f"no kernel_*.cu under {source_dir}")
    mode_tag = " (correct-only)" if args.correct_only else ""
    print(f"[analyze_patterns_batch]{mode_tag} {len(kernels)} kernels → {out_dir}")

    tasks = [(p, reference_names) for p in kernels]
    if args.jobs > 1 and len(tasks) > 1:
        with Pool(args.jobs) as pool:
            records = pool.map(analyze_one, tasks)
    else:
        records = [analyze_one(t) for t in tasks]

    records.sort(key=lambda r: (r["exp"], r["family"], r["index"]))

    write_kernels_jsonl(records, out_dir / "kernels.jsonl")
    write_kernels_csv(records, out_dir / "kernels.csv")

    ngrams1 = ngram_table(records, 1)
    ngrams2 = ngram_table(records, 2)
    ngrams3 = ngram_table(records, 3)
    write_ngrams_csv(ngrams1, out_dir / "ngrams_1.csv")
    write_ngrams_csv(ngrams2, out_dir / "ngrams_2.csv")
    write_ngrams_csv(ngrams3, out_dir / "ngrams_3.csv")

    exact_cfg = exact_cfg_dedup(records)
    (out_dir / "dedup_cfg_exact.json").write_text(json.dumps({
        "distinct": len(exact_cfg),
        "total": sum(1 for r in records if r["num_global"] > 0),
        "groups": exact_cfg,
    }, indent=2))

    cache = ZssCache(records)
    rep_records, members_by_rep, distances, used_pairwise = ensure_rep_distances(
        records, cache, out_dir, args.max_pairs_ted_reps,
    )
    clusters = cluster_from_distances(
        rep_records, members_by_rep, distances,
        args.ted_threshold, used_pairwise,
    )
    (out_dir / "clusters_cfg.json").write_text(json.dumps({
        "threshold": args.ted_threshold,
        "metric": "cfg_ted_norm",
        "used_pairwise_ted": used_pairwise,
        "n_clusters": len(clusters),
        "clusters": clusters,
    }, indent=2))

    if used_pairwise and distances is not None:
        plot_ted_heatmap(
            rep_records, members_by_rep, distances,
            out_dir / TED_HEATMAP_FILENAME,
            run_dir,
        )

    ev_rows, ev_summary = turn_evolution(records, cache)
    with (out_dir / "turn_evolution.csv").open("w", newline="") as f:
        if ev_rows:
            w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
            w.writeheader()
            w.writerows(ev_rows)
    (out_dir / "turn_evolution_summary.json").write_text(json.dumps(ev_summary, indent=2))

    write_report(
        records, ngrams1, ngrams2, ngrams3,
        exact_cfg, clusters, used_pairwise, ev_summary,
        args.ted_threshold, args.max_pairs_ted_reps,
        out_dir / "report.md",
    )

    print(f"[analyze_patterns_batch] done. "
          f"{len(records)} kernels, {len(exact_cfg)} distinct CFG hashes, "
          f"{len(clusters)} CFG clusters, {len(ev_rows)} evolution rows.")


if __name__ == "__main__":
    main()
