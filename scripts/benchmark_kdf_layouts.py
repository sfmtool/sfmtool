# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the two `.kdf` descriptor layouts on a real SIFT corpus.

Runs the staged sweep from `specs/core/features/lazy-kdforest-query.md`
("Benchmark plan and provisional defaults") against a workspace's `.sift`
files. The question it exists to answer is whether one shared descriptor corpus
beats T tree-local copies, and at what chunk and block sizes.

The stages are deliberately not a Cartesian product: stage 1 screens chunk sizes
with the layouts held against each other, stage 2 sweeps shared block sizes at
the chunk size stage 1 liked, and stage 3 stresses the survivors across cache
budgets and worker counts. Each stage narrows what the next one has to try.

Every cell holds the corpus, the forest topology, the query set, k and the check
budget fixed, so the only thing varying is storage. The forest is built once per
run and exported repeatedly; two cells that differed in their forest would not
be comparing layouts at all.

Run:
    pixi run python scripts/benchmark_kdf_layouts.py --workspace WS --out results.json

A reopened file is not evidence of cold physical storage. The OS page cache is
uncontrolled here, so "cold" below means a fresh `LazyKdForest` with an empty
application cache, and nothing stronger.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.spatial import (
    KdForest,
    LazyKdForest,
    kdf_file_summary,
    write_kdf,
)
from sfmtool.sift.file import SiftReader

KIB = 1 << 10
MIB = 1 << 20

# Stage 1 screens these with four trees and 16-feature leaves, per the plan.
CHUNK_SIZES = [256 * KIB, 1 * MIB, 4 * MIB, 8 * MIB, 16 * MIB]
BLOCK_SIZES = [2 * KIB, 4 * KIB, 8 * KIB, 16 * KIB, 64 * KIB, 256 * KIB]
CACHE_BUDGETS = [64 * MIB, 256 * MIB, 1024 * MIB]
WORKER_COUNTS = [1, 4, 8]


# ── Corpus ────────────────────────────────────────────────────────────────


def find_features(workspace: Path, explicit: str | None) -> Path:
    """Locate the `sift-*` directory holding a corpus's `.sift` files.

    A workspace keeps them under `images/features/`, but a dataset laid out any
    other way (`frames/features/`, say) is just as valid a corpus, so an
    explicit directory wins and the workspace layout is only the default.
    """
    if explicit:
        directory = Path(explicit)
        if not directory.is_dir():
            raise SystemExit(f"{directory} is not a directory")
        return directory
    for parent in ("images/features", "frames/features", "features"):
        roots = sorted((workspace / parent).glob("sift-*"))
        if roots:
            return roots[0]
    raise SystemExit(
        f"no sift-* directory under {workspace}; pass --features explicitly"
    )


def load_corpus(features: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Pool every `.sift` descriptor in a feature directory, in image order.

    Returns the descriptors alongside the origin columns that map each corpus
    row back to `(image_index, image_feature_index)` — the same provenance a
    real export would carry, so its cost lands in the measurement instead of
    being assumed away.

    A `.sift` file has no descriptor count in its header, so the total is only
    known after every file has been read and the blocks cannot be streamed into
    a preallocated array in one pass. Each block is instead released as it is
    copied, so the peak declines through the copy rather than holding both the
    blocks and the finished array to the end. Each reader is closed as soon as
    its descriptors are in hand, which frees that file's positions, affine
    shapes and thumbnail immediately — only the descriptors are wanted here.
    """
    files = sorted(features.glob("*.sift"))
    if not files:
        raise SystemExit(f"no .sift files in {features}")

    blocks = []
    for path in files:
        reader = SiftReader(path)
        blocks.append(reader.read_descriptors())
        reader.close()

    total = sum(len(b) for b in blocks)
    out = np.empty((total, blocks[0].shape[1]), dtype=np.uint8)
    image_indexes = np.empty(total, dtype=np.uint32)
    feature_indexes = np.empty(total, dtype=np.uint32)
    at = 0
    for image_index, block in enumerate(blocks):
        n = len(block)
        out[at : at + n] = block
        image_indexes[at : at + n] = image_index
        feature_indexes[at : at + n] = np.arange(n, dtype=np.uint32)
        at += n
        blocks[image_index] = None

    return (
        out,
        [p.name.removesuffix(".sift") for p in files],
        image_indexes,
        feature_indexes,
    )


def split_queries(descriptors: np.ndarray, count: int, seed: int):
    """Hold out `count` rows as queries and index the rest.

    Held-out queries rather than a self-query: a descriptor that is in the index
    is its own nearest neighbor at distance zero, which every layout finds and
    which therefore measures nothing. Holding them out also makes recall a
    question about the index rather than about an exclusion rule.
    """
    rng = np.random.default_rng(seed)
    held = rng.choice(
        len(descriptors), size=min(count, len(descriptors) // 4), replace=False
    )
    mask = np.ones(len(descriptors), dtype=bool)
    mask[held] = False
    return descriptors[mask], descriptors[held], mask


def exact_nearest(
    index: np.ndarray,
    queries: np.ndarray,
    q_block: int = 512,
    db_block: int = 262_144,
) -> np.ndarray:
    """Exact top-1 index for each query, by squared-L2 scan blocked both ways.

    Computed once per run and reused by every cell: it depends on the corpus and
    the query set, neither of which a storage layout can change.

    Blocking over the database as well as the queries is what makes this usable
    at corpus scale. A 9.7M-row index is 5 GB as float32 and the full distance
    matrix for even 512 queries against it is 20 GB, so the scan carries a
    running minimum over database blocks and never materializes either.

    float32 is enough: SIFT bytes are 0-255 over 128 dimensions, so a squared
    distance is at most 8.3e6 and every value in the accumulation is an exact
    integer well inside float32's 24-bit mantissa.
    """
    n_queries = len(queries)
    best_dist = np.full(n_queries, np.inf, dtype=np.float32)
    best_idx = np.zeros(n_queries, dtype=np.uint32)
    for j in range(0, len(index), db_block):
        db = index[j : j + db_block].astype(np.float32)
        db_sq = np.einsum("ij,ij->i", db, db)
        for i in range(0, n_queries, q_block):
            q = queries[i : i + q_block].astype(np.float32)
            d = db_sq[None, :] - 2.0 * (q @ db.T)
            local = np.argmin(d, axis=1)
            local_dist = d[np.arange(len(q)), local]
            better = local_dist < best_dist[i : i + len(q)]
            best_dist[i : i + len(q)][better] = local_dist[better]
            best_idx[i : i + len(q)][better] = (local[better] + j).astype(np.uint32)
    return best_idx


# ── One measured cell ─────────────────────────────────────────────────────


def export(
    forest, path: Path, layout: str, chunk_bytes: int, block_bytes: int | None, sources
):
    extra = {} if layout == "tree_local" else {"descriptor_block_bytes": block_bytes}
    start = time.perf_counter()
    write_kdf(
        forest,
        str(path),
        layout=layout,
        chunk_bytes=chunk_bytes,
        sources=sources,
        **extra,
    )
    return time.perf_counter() - start


def measure(
    path: Path,
    queries: np.ndarray,
    truth: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray],
    *,
    k: int,
    budget: int,
    cache_bytes: int,
    workers: int,
    latency_samples: int,
) -> dict:
    """Open a `.kdf` and measure one configuration of it."""
    # Seed probe: a fresh file, one query, nothing cached. Within a single query
    # the check set is deduplicated, so `checks * dim` is exactly the unique
    # evaluated vector bytes the spec divides decoded bytes by. Across a batch
    # it is not — later queries re-evaluate descriptors already decoded — so the
    # batch ratio further down answers a different, also useful, question.
    #
    # This is the number that separates chunk sizes: seeding a four-tree search
    # costs four independent subtree misses whatever the query, so a larger
    # chunk decodes proportionally more to answer the same first query.
    probe = LazyKdForest(
        str(path),
        cache_bytes=cache_bytes,
        max_in_flight_bytes=cache_bytes,
        max_chunk_bytes=max(cache_bytes, 64 * MIB),
        max_compressed_bytes=max(cache_bytes, 64 * MIB),
        query_workers=1,
    )
    probe.reset_io_stats()
    probe_start = time.perf_counter()
    _, _, probe_stats = probe.query_with_stats(queries[:1], k=k, max_leaf_checks=budget)
    seed_seconds = time.perf_counter() - probe_start
    probe_io = probe.io_stats()
    seed_checks = probe_stats["checks"]
    seed = {
        "seed_seconds": seed_seconds,
        "seed_read_calls": probe_io["read_calls"],
        "seed_decoded_bytes": probe_io["decoded_bytes"],
        "seed_checks": seed_checks,
        "seed_amplification": (
            probe_io["decoded_bytes"] / (seed_checks * probe.dim)
            if seed_checks
            else None
        ),
    }
    del probe

    start = time.perf_counter()
    lazy = LazyKdForest(
        str(path),
        cache_bytes=cache_bytes,
        max_in_flight_bytes=cache_bytes,
        max_chunk_bytes=max(cache_bytes, 64 * MIB),
        max_compressed_bytes=max(cache_bytes, 64 * MIB),
        query_workers=workers,
    )
    open_seconds = time.perf_counter() - start
    at_open = lazy.io_stats()

    # Cold batch: the application cache is empty, so this pays for every chunk
    # the traversal reaches. Its counters are the ones read amplification uses.
    lazy.reset_io_stats()
    start = time.perf_counter()
    cold_idx, cold_dist, cold_stats = lazy.query_with_stats(
        queries, k=k, max_leaf_checks=budget
    )
    cold_seconds = time.perf_counter() - start
    cold_io = lazy.io_stats()

    # Warm batch: same queries against a populated cache. The gap between the
    # two is what the cache is worth at this budget.
    lazy.reset_io_stats()
    start = time.perf_counter()
    lazy.query(queries, k=k, max_leaf_checks=budget)
    warm_seconds = time.perf_counter() - start
    warm_io = lazy.io_stats()

    # Per-query latency, measured one at a time so the percentiles describe a
    # query rather than a share of a batch. Taken warm, so it is not dominated
    # by first-touch decodes.
    sample = queries[:latency_samples]
    timings = []
    for row in sample:
        one = row.reshape(1, -1)
        start = time.perf_counter()
        lazy.query(one, k=k, max_leaf_checks=budget)
        timings.append((time.perf_counter() - start) * 1e3)
    timings.sort()

    def pct(p):
        return timings[min(len(timings) - 1, int(p * len(timings)))]

    ref_idx, ref_dist = reference
    np.testing.assert_array_equal(cold_idx, ref_idx)
    np.testing.assert_array_equal(cold_dist, ref_dist)
    dim = lazy.dim
    checks = cold_stats["checks"]
    summary = kdf_file_summary(str(path))
    sections = {s["section"]: s for s in summary["sections"]}

    return {
        "file_bytes": summary["file_bytes"],
        "payload_compressed_bytes": summary["payload_compressed_bytes"],
        "payload_decoded_bytes": summary["payload_decoded_bytes"],
        "entries": sum(s["entries"] for s in summary["sections"]),
        "chunks_per_tree": summary["chunks_per_tree"],
        "sections": {
            name: {"compressed": s["compressed_bytes"], "decoded": s["decoded_bytes"]}
            for name, s in sections.items()
        },
        "open_seconds": open_seconds,
        "open_read_calls": at_open["read_calls"],
        "address_map_bytes": at_open["address_map_bytes"],
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "cold_qps": len(queries) / cold_seconds,
        "warm_qps": len(queries) / warm_seconds,
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "latency_p99_ms": pct(0.99),
        "checks": checks,
        "read_calls": cold_io["read_calls"],
        "compressed_bytes": cold_io["compressed_bytes"],
        "decoded_bytes": cold_io["decoded_bytes"],
        "cache_misses": cold_io["cache_misses"],
        "evictions": cold_io["evictions"],
        "duplicate_load_waits": cold_io["duplicate_load_waits"],
        "peak_resident_bytes": cold_io["peak_resident_bytes"],
        "warm_read_calls": warm_io["read_calls"],
        "warm_cache_hits": warm_io["cache_hits"],
        # Batch-level ratio of decoded bytes to evaluated vector bytes. Below 1
        # means the batch re-evaluated cached descriptors more than it decoded
        # fresh ones, i.e. the queries shared chunks. This is *not* the spec's
        # read amplification, whose denominator is unique evaluated bytes; that
        # one is `seed_amplification`, measured on a single cold query above.
        "batch_decode_ratio": (cold_io["decoded_bytes"] / (checks * dim))
        if checks
        else None,
        "recall_at_1": float(np.mean(cold_idx[:, 0] == truth)),
        "matches_in_memory": bool(
            np.array_equal(cold_idx, ref_idx) and np.allclose(cold_dist, ref_dist)
        ),
        **seed,
    }


# ── Stages ────────────────────────────────────────────────────────────────


def run(args) -> dict:
    workspace = Path(args.workspace)
    features = find_features(workspace, args.features)
    print(f"features: {features}")
    descriptors, image_names, image_indexes, feature_indexes = load_corpus(features)
    index, queries, mask = split_queries(descriptors, args.queries, args.seed)
    print(
        f"corpus {len(descriptors):,} descriptors x {descriptors.shape[1]}D"
        f" -> index {len(index):,}, queries {len(queries):,}"
    )

    sources = {
        "workspace": {
            "absolute_path": str(workspace.resolve()),
            "relative_path": ".",
            "contents": {
                "feature_tool": "sfmtool",
                "feature_type": "sift",
                "feature_options": json.dumps({}),
                "feature_prefix_dir": str(features.parent.name),
            },
        },
        "image_names": image_names,
        # Placeholder digests: this run measures what the table costs to store,
        # and its size does not depend on which bytes the hashes hold.
        "feature_tool_hashes": [bytes(16)] * len(image_names),
        "sift_content_hashes": [bytes(16)] * len(image_names),
        "image_indexes": image_indexes[mask].tolist(),
        "image_feature_indexes": feature_indexes[mask].tolist(),
    }

    print(f"building forest: {args.trees} trees, leaf {args.leaf_size} ...", flush=True)
    start = time.perf_counter()
    forest = KdForest(
        index, num_trees=args.trees, leaf_size=args.leaf_size, seed=args.seed
    )
    build_seconds = time.perf_counter() - start
    print(f"  built in {build_seconds:.1f}s", flush=True)

    print("exact nearest neighbors for recall ...", flush=True)
    start = time.perf_counter()
    truth = exact_nearest(index, queries)
    print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)

    reference = forest.query(queries, k=args.k, max_leaf_checks=args.budget)

    out = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def cell(label, layout, chunk_bytes, block_bytes, cache_bytes, workers, stage):
        path = out / f"{stage}-{label}.kdf"
        if path.exists():
            path.unlink()
        export_seconds = export(forest, path, layout, chunk_bytes, block_bytes, sources)
        row = {
            "stage": stage,
            "label": label,
            "layout": layout,
            "chunk_bytes": chunk_bytes,
            "block_bytes": block_bytes,
            "cache_bytes": cache_bytes,
            "workers": workers,
            "export_seconds": export_seconds,
            "repeats": args.repeats,
        }
        # Repeat the whole measurement against the one exported file. Byte
        # counts are deterministic and repeat identically; wall-clock does not,
        # and on a corpus this small the spread is comparable to the effect
        # being measured, so reporting a single timing would invite reading
        # noise as signal.
        runs = [
            measure(
                path,
                queries,
                truth,
                reference,
                k=args.k,
                budget=args.budget,
                cache_bytes=cache_bytes,
                workers=workers,
                latency_samples=args.latency_samples,
            )
            for _ in range(args.repeats)
        ]
        row.update(runs[0])
        for field in (
            "open_seconds",
            "cold_seconds",
            "warm_seconds",
            "seed_seconds",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "cold_qps",
            "warm_qps",
        ):
            values = [r[field] for r in runs]
            row[field] = statistics.median(values)
            row[f"{field}_min"] = min(values)
            row[f"{field}_max"] = max(values)
        # `checks` is a property of the traversal, so it is deterministic
        # regardless of worker count: results are identical across layouts and
        # runs, which is what the parity column asserts. Any variation there
        # would be a bug.
        for field in ("checks", "seed_decoded_bytes", "seed_read_calls"):
            distinct = {r[field] for r in runs}
            if len(distinct) != 1:
                raise SystemExit(f"{label}: {field} varied across repeats: {distinct}")
        # Decoded bytes and read calls are *not* deterministic above one worker.
        # Concurrent queries interleave their misses, so which chunk is resident
        # when the next one needs room differs run to run, and so does how much
        # gets re-decoded after eviction. Record the spread rather than assert it
        # away: at one worker it collapses to zero, and a nonzero spread there
        # would be the bug this used to look for.
        for field in ("decoded_bytes", "read_calls", "compressed_bytes", "evictions"):
            values = [r[field] for r in runs]
            row[field] = statistics.median(values)
            row[f"{field}_min"] = min(values)
            row[f"{field}_max"] = max(values)
            if workers == 1 and len(set(values)) != 1:
                raise SystemExit(
                    f"{label}: {field} varied across single-worker repeats: {set(values)}"
                )
        checks = row["checks"]
        row["batch_decode_ratio"] = (
            row["decoded_bytes"] / (checks * descriptors.shape[1]) if checks else None
        )
        results.append(row)
        print(
            f"  {stage} {label:<28} {row['file_bytes'] / MIB:8.2f} MiB"
            f"  cold {row['cold_seconds']:6.2f}s"
            f"  warm {row['warm_seconds']:6.2f}s"
            f"  seed-amp {row['seed_amplification'] or float('nan'):7.1f}"
            f"  recall {row['recall_at_1']:.3f}"
            f"  parity {'ok' if row['matches_in_memory'] else 'DIFFERS'}",
            flush=True,
        )
        if not args.keep:
            path.unlink()
        return row

    if args.stage in ("1", "all"):
        print("\nstage 1: chunk-size screen, both layouts")
        for chunk in CHUNK_SIZES:
            cell(
                f"tree_local-chunk{chunk // KIB}k",
                "tree_local",
                chunk,
                None,
                args.cache_bytes,
                args.workers,
                "1",
            )
            cell(
                f"shared-chunk{chunk // KIB}k",
                "shared",
                chunk,
                64 * KIB,
                args.cache_bytes,
                args.workers,
                "1",
            )

    if args.stage in ("2", "all"):
        print("\nstage 2: shared descriptor block sizes")
        for block in BLOCK_SIZES:
            cell(
                f"shared-block{block // KIB}k",
                "shared",
                args.chunk_bytes,
                block,
                args.cache_bytes,
                args.workers,
                "2",
            )

    if args.stage in ("3", "all"):
        print("\nstage 3: cache budgets and worker counts")
        for layout, block in (("tree_local", None), ("shared", args.block_bytes)):
            for cache in CACHE_BUDGETS:
                cell(
                    f"{layout}-cache{cache // MIB}m",
                    layout,
                    args.chunk_bytes,
                    block,
                    cache,
                    args.workers,
                    "3",
                )
            for workers in WORKER_COUNTS:
                cell(
                    f"{layout}-workers{workers}",
                    layout,
                    args.chunk_bytes,
                    block,
                    args.cache_bytes,
                    workers,
                    "3",
                )

    return {
        "dataset": workspace.name,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "build_profile": build_profile(),
        },
        "corpus": {
            "descriptors": int(len(descriptors)),
            "indexed": int(len(index)),
            "queries": int(len(queries)),
            "dimension": int(descriptors.shape[1]),
            "images": len(image_names),
            "raw_bytes": int(index.nbytes),
        },
        "forest": {
            "trees": args.trees,
            "leaf_size": args.leaf_size,
            "seed": args.seed,
            "build_seconds": build_seconds,
        },
        "query": {"k": args.k, "max_leaf_checks": args.budget},
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True, help="workspace or dataset root")
    p.add_argument(
        "--features",
        help="sift-* directory holding the .sift files (overrides the default search)",
    )
    p.add_argument(
        "--scratch", required=True, help="directory for the exported .kdf files"
    )
    p.add_argument("--out", help="write the full results as JSON here")
    p.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--budget", type=int, default=128, help="max_leaf_checks per query")
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--latency-samples", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3, help="timed repeats per cell")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-bytes", type=int, default=256 * MIB)
    p.add_argument("--chunk-bytes", type=int, default=1 * MIB, help="stages 2 and 3")
    p.add_argument(
        "--block-bytes", type=int, default=64 * KIB, help="stage 3 shared blocks"
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--keep", action="store_true", help="keep the exported .kdf files")
    args = p.parse_args()

    if build_profile() != "release":
        raise SystemExit("refusing to benchmark a debug build of _sfmtool")

    report = run(args)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
