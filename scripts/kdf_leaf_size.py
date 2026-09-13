# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""What leaf size costs and buys in a persistent `.kdf` forest.

Leaf size is a build parameter, not a storage one: it changes the index, so a
larger leaf both improves descriptor locality — a leaf's members are contiguous
in the stored corpus, so more of them share a block — and changes how many
distance computations one leaf visit costs. Comparing leaf sizes at a fixed check
budget would therefore compare different indexes and prove nothing.

So this compares at **equal recall**. For each leaf size it first finds the
smallest check budget reaching a recall target against exhaustive search, using
the in-memory forest, then measures the file-backed cost at exactly that budget.
Cheaper at the same recall is the only claim worth making.

Run:
    pixi run python scripts/kdf_leaf_size.py --workspace WS
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.spatial import KdForest, LazyKdForest, kdf_file_summary, write_kdf
from sfmtool.sift.file import SiftReader

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30
LEAF_SIZES = [8, 16, 32, 64, 128]
BUDGETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def load_corpus(features: Path) -> np.ndarray:
    files = sorted(features.glob("*.sift"))
    if not files:
        raise SystemExit(f"no .sift files in {features}")
    blocks = []
    for path in files:
        reader = SiftReader(path)
        blocks.append(reader.read_descriptors())
        reader.close()
    return np.vstack(blocks)


def exact_nearest(index: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact top-1 index per query, without a query-by-database temporary."""
    exhaustive = KdForest(index, num_trees=1, leaf_size=len(index), seed=0)
    indices, _ = exhaustive.query(
        queries,
        k=1,
        max_leaf_checks=len(index),
    )
    return indices[:, 0]


def budget_for_recall(
    forest, queries, truth, target: float
) -> tuple[int | None, float]:
    """Smallest integer budget whose recall@1 reaches `target`."""
    lower = 0
    for budget in BUDGETS:
        idx, _ = forest.query(queries, k=2, max_leaf_checks=budget)
        recall = float(np.mean(idx[:, 0] == truth))
        if recall >= target:
            upper = budget
            while lower < upper:
                candidate = (lower + upper) // 2
                idx, _ = forest.query(queries, k=2, max_leaf_checks=candidate)
                candidate_recall = float(np.mean(idx[:, 0] == truth))
                if candidate_recall >= target:
                    upper = candidate
                else:
                    lower = candidate + 1
            idx, _ = forest.query(queries, k=2, max_leaf_checks=lower)
            return lower, float(np.mean(idx[:, 0] == truth))
        lower = budget + 1
    return None, recall


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--features")
    p.add_argument("--scratch", required=True)
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--recall", type=float, default=0.60, help="iso-recall target")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-mib", type=int, default=4096)
    p.add_argument("--block-bytes", type=int, default=2 * KIB)
    p.add_argument("--chunk-bytes", type=int, default=1 * MIB)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()

    if build_profile() != "release":
        raise SystemExit("refusing to benchmark a debug build of _sfmtool")

    workspace = Path(args.workspace)
    features = Path(args.features) if args.features else None
    if features is None:
        roots = sorted((workspace / "images/features").glob("sift-*")) or sorted(
            (workspace / "frames/features").glob("sift-*")
        )
        if not roots:
            raise SystemExit(f"no sift-* directory under {workspace}")
        features = roots[0]

    descriptors = load_corpus(features)
    rng = np.random.default_rng(args.seed)
    held = rng.choice(
        len(descriptors), size=min(args.queries, len(descriptors) // 4), replace=False
    )
    mask = np.ones(len(descriptors), dtype=bool)
    mask[held] = False
    index, queries = descriptors[mask], descriptors[held]
    print(
        f"corpus {len(descriptors):,} -> index {len(index):,}, queries {len(queries):,}"
    )
    print("exact nearest neighbors ...", flush=True)
    truth = exact_nearest(index, queries)

    out = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    opts = dict(
        cache_bytes=args.cache_mib * MIB,
        max_in_flight_bytes=4 * GIB,
        max_chunk_bytes=4 * GIB,
        max_compressed_bytes=4 * GIB,
        query_workers=args.workers,
    )
    rows = []

    print(
        f"\niso-recall comparison at recall@1 >= {args.recall:.2f},"
        f" {args.trees} trees, {args.block_bytes // KIB} KiB blocks,"
        f" {args.cache_mib} MiB cache, {args.workers} workers\n"
    )
    print(
        f"{'leaf':>5} {'budget':>7} {'recall':>7} {'checks':>7} {'blocks':>7} {'reuse':>7}"
        f" {'file MB':>8} {'chunk MB':>9} {'cold s':>7} {'batch reads':>12}"
    )
    for leaf in LEAF_SIZES:
        forest = KdForest(index, num_trees=args.trees, leaf_size=leaf, seed=args.seed)
        budget, recall = budget_for_recall(forest, queries, truth, args.recall)
        if budget is None:
            print(f"{leaf:>5} {'-':>7} {recall:>7.3f}  (never reached the target)")
            continue

        # Tree-chunk reads for one query, isolated by a single-block corpus.
        solo = out / f"solo{leaf}.kdf"
        if solo.exists():
            solo.unlink()
        write_kdf(
            forest,
            str(solo),
            chunk_bytes=args.chunk_bytes,
            descriptor_block_bytes=len(index) * index.shape[1],
        )
        probe_opts = opts | {"cache_bytes": 4 * GIB}
        probe = LazyKdForest(str(solo), **probe_opts)
        probe.query(queries[:1], k=2, max_leaf_checks=budget)
        chunk_reads = probe.io_stats()["read_calls"] - 1
        del probe
        solo.unlink()

        path = out / f"leaf{leaf}.kdf"
        if path.exists():
            path.unlink()
        write_kdf(
            forest,
            str(path),
            chunk_bytes=args.chunk_bytes,
            descriptor_block_bytes=args.block_bytes,
        )
        summary = kdf_file_summary(str(path))
        chunk_mb = (
            next(
                s["compressed_bytes"]
                for s in summary["sections"]
                if s["section"] == "tree_chunks"
            )
            / 1e6
        )

        lazy = LazyKdForest(str(path), **opts)
        _, _, stats = lazy.query_with_stats(queries[:1], k=2, max_leaf_checks=budget)
        blocks = max(lazy.io_stats()["read_calls"] - chunk_reads, 0)
        checks = stats["checks"]
        del lazy

        reference = forest.query(queries, k=2, max_leaf_checks=budget)
        runs, batch = [], None
        for _ in range(3):
            lazy = LazyKdForest(str(path), **opts)
            start = time.perf_counter()
            got = lazy.query(queries, k=2, max_leaf_checks=budget)
            runs.append(time.perf_counter() - start)
            for actual, expected in zip(got, reference, strict=True):
                np.testing.assert_array_equal(actual, expected)
            batch = lazy.io_stats()
            del lazy

        row = {
            "leaf_size": leaf,
            "budget": budget,
            "recall_at_1": recall,
            "checks": checks,
            "blocks": blocks,
            "reuse": (checks - blocks) / checks if checks else float("nan"),
            "file_bytes": summary["file_bytes"],
            "tree_chunks_bytes": int(chunk_mb * 1e6),
            "chunks_per_tree": summary["chunks_per_tree"][0],
            "cold_seconds": statistics.median(runs),
            "cold_seconds_runs": runs,
            "batch_reads": batch["read_calls"],
            "batch_decoded": batch["decoded_bytes"],
        }
        rows.append(row)
        print(
            f"{leaf:>5} {budget:>7} {recall:>7.3f} {checks:>7,} {blocks:>7,}"
            f" {row['reuse'] * 100:>6.1f}% {summary['file_bytes'] / 1e6:>8.1f} {chunk_mb:>9.1f}"
            f" {row['cold_seconds']:>7.2f} {batch['read_calls']:>12,}"
        )
        path.unlink()

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "corpus": {
                        "indexed": int(len(index)),
                        "queries": int(len(queries)),
                    },
                    "trees": args.trees,
                    "workers": args.workers,
                    "cache_bytes": args.cache_mib * MIB,
                    "chunk_bytes": args.chunk_bytes,
                    "seed": args.seed,
                    "recall_target": args.recall,
                    "block_bytes": args.block_bytes,
                    "results": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
