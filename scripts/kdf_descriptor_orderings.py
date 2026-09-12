# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare shared-corpus descriptor orderings for a `.kdf` forest.

The shared descriptor layout stores one copy of the corpus in a stored order of
the writer's choosing, and a query reads whole blocks of it. How well that order
matches the leaves a query actually visits decides how much of each block is
useful. The default — tree 0's leaf order — makes tree 0's leaves contiguous and
leaves every other randomized tree scattered: measured block reuse falls from 89%
at one tree to 31% at four.

This measures alternatives against that default. Nothing here changes the file
format: the storage-row map is an explicit permutation, so an ordering policy is a
writer-side choice a reader follows without knowing it happened.

Policies:

* `tree0` — the current default, tree 0's leaf order.
* `projection` — a locality order from descriptor space itself rather than from
  any one tree's partition. Every tree's leaf is a spatially compact set, so an
  order that keeps nearby descriptors nearby should serve all T trees somewhat
  instead of one perfectly. Implemented as a Morton (Z-order) curve over the top
  few principal components.
* `greedy` — greedy co-occurrence packing over the leaf hypergraph. Walks leaves
  and emits their still-unplaced members together, so descriptors that a query
  evaluates together land in the same block when they can.

Run:
    pixi run python scripts/kdf_descriptor_orderings.py --workspace WS
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
GIB = 1 << 30


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


# ── Ordering policies ─────────────────────────────────────────────────────


def order_tree0(forest: KdForest, descriptors: np.ndarray) -> np.ndarray:
    """The current default: tree 0's leaf order."""
    ids, _ = forest.leaf_layout(0)
    return ids.astype(np.uint32)


def order_projection(
    forest: KdForest, descriptors: np.ndarray, dims: int = 3, bits: int = 10
) -> np.ndarray:
    """Morton (Z-order) curve over the top `dims` principal components.

    Descriptor space locality rather than one tree's partition. PCA first because
    a space-filling curve is only meaningful in few dimensions, and SIFT's 128 are
    strongly correlated; the top three carry enough of the variance to separate
    neighbourhoods. Coordinates are rank-transformed to percentiles before
    interleaving so each axis contributes equally regardless of its scale.
    """
    sample = descriptors
    if len(sample) > 200_000:
        sample = sample[:: len(sample) // 200_000 + 1]
    centered = sample.astype(np.float32) - sample.mean(axis=0, dtype=np.float32)
    # Top `dims` right singular vectors of the sample, i.e. its principal axes.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:dims].T.astype(np.float32)

    projected = (
        descriptors.astype(np.float32) - descriptors.mean(axis=0, dtype=np.float32)
    ) @ axes
    # Rank per axis, scaled into `bits` bits: equal-population buckets, so no axis
    # dominates the interleave because it happens to have a wider range.
    scale = (1 << bits) - 1
    grid = np.empty(projected.shape, dtype=np.uint64)
    for d in range(projected.shape[1]):
        ranks = np.argsort(np.argsort(projected[:, d]))
        grid[:, d] = (ranks * scale // max(len(ranks) - 1, 1)).astype(np.uint64)

    # Interleave the bits, most significant first.
    keys = np.zeros(len(grid), dtype=np.uint64)
    for bit in range(bits - 1, -1, -1):
        for d in range(grid.shape[1]):
            keys = (keys << np.uint64(1)) | (
                (grid[:, d] >> np.uint64(bit)) & np.uint64(1)
            )
    return np.argsort(keys, kind="stable").astype(np.uint32)


def order_greedy(forest: KdForest, descriptors: np.ndarray) -> np.ndarray:
    """Greedy co-occurrence packing over the leaves of every tree.

    Visits leaves in round-robin across trees, emitting each leaf's not-yet-placed
    members consecutively. A descriptor is placed once, by whichever leaf reaches
    it first, so early leaves get their members fully contiguous and later leaves
    keep whatever their remaining members can still be given. Round-robin rather
    than tree-by-tree because taking one tree's leaves first would reproduce the
    default: that tree perfectly served and the rest scattered.
    """
    n = len(descriptors)
    trees = forest.num_trees
    leaves: list[list[np.ndarray]] = []
    for t in range(trees):
        ids, starts = forest.leaf_layout(t)
        bounds = list(starts.astype(np.int64)) + [len(ids)]
        leaves.append([ids[bounds[i] : bounds[i + 1]] for i in range(len(starts))])

    placed = np.zeros(n, dtype=bool)
    out = np.empty(n, dtype=np.uint32)
    at = 0
    for i in range(max(len(v) for v in leaves)):
        for t in range(trees):
            if i >= len(leaves[t]):
                continue
            members = leaves[t][i]
            fresh = members[~placed[members]]
            if len(fresh) == 0:
                continue
            out[at : at + len(fresh)] = fresh
            placed[members] = True
            at += len(fresh)
    if at != n:
        # Anything no leaf claimed (possible only if a point is in no leaf) tails.
        out[at:] = np.flatnonzero(~placed).astype(np.uint32)
    return out


POLICIES = {
    "tree0": order_tree0,
    "projection": order_projection,
    "greedy": order_greedy,
}


# ── Measurement ───────────────────────────────────────────────────────────


def chunk_reads(path: Path, query: np.ndarray, budget: int, n: int, dim: int) -> int:
    """Reads attributable to tree chunks, found by giving the corpus one block."""
    lazy = LazyKdForest(
        str(path),
        cache_bytes=4 * GIB,
        max_in_flight_bytes=4 * GIB,
        max_chunk_bytes=4 * GIB,
        max_compressed_bytes=4 * GIB,
    )
    lazy.query(query, k=2, max_leaf_checks=budget)
    return lazy.io_stats()["read_calls"] - 1


def measure(
    path: Path, queries: np.ndarray, budget: int, baseline_chunks: int, reference
) -> dict:
    opts = dict(
        cache_bytes=4 * GIB,
        max_in_flight_bytes=4 * GIB,
        max_chunk_bytes=4 * GIB,
        max_compressed_bytes=4 * GIB,
    )
    # Single cold query: blocks read, and the reuse that implies.
    lazy = LazyKdForest(str(path), **opts)
    _, _, stats = lazy.query_with_stats(queries[:1], k=2, max_leaf_checks=budget)
    reads = lazy.io_stats()["read_calls"]
    blocks = max(reads - baseline_chunks, 0)
    checks = stats["checks"]
    del lazy

    runs = []
    for _ in range(3):
        lazy = LazyKdForest(str(path), **opts)
        start = time.perf_counter()
        got = lazy.query(queries, k=2, max_leaf_checks=budget)
        runs.append(time.perf_counter() - start)
        for actual, expected in zip(got, reference, strict=True):
            np.testing.assert_array_equal(actual, expected)
        batch = lazy.io_stats()
        del lazy

    summary = kdf_file_summary(str(path))
    return {
        "file_bytes": summary["file_bytes"],
        "checks": checks,
        "blocks": blocks,
        "checks_per_block": checks / blocks if blocks else float("nan"),
        "reuse": (checks - blocks) / checks if checks else float("nan"),
        "cold_seconds": statistics.median(runs),
        "batch_reads": batch["read_calls"],
        "batch_decoded": batch["decoded_bytes"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--features")
    p.add_argument("--scratch", required=True)
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--block-bytes", type=int, default=4 * KIB)
    p.add_argument("--chunk-bytes", type=int, default=1 << 20)
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

    forest = KdForest(
        index, num_trees=args.trees, leaf_size=args.leaf_size, seed=args.seed
    )
    reference = forest.query(queries, k=2, max_leaf_checks=args.budget)
    out = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)

    # Tree-chunk reads for one query, isolated by giving the corpus one block.
    solo = out / "solo.kdf"
    if solo.exists():
        solo.unlink()
    write_kdf(
        forest,
        str(solo),
        layout="shared",
        chunk_bytes=args.chunk_bytes,
        descriptor_block_bytes=len(index) * index.shape[1],
    )
    baseline = chunk_reads(solo, queries[:1], args.budget, len(index), index.shape[1])
    solo.unlink()
    print(f"tree-chunk reads for one query: {baseline}\n")

    rows = []
    print(
        f"{'policy':<12} {'build s':>8} {'file MB':>9} {'checks':>7} {'blocks':>7}"
        f" {'per block':>10} {'reuse':>7} {'cold s':>7} {'batch reads':>12} {'decoded MiB':>12}"
    )
    for name, policy in POLICIES.items():
        start = time.perf_counter()
        order = policy(forest, index)
        build = time.perf_counter() - start
        assert len(order) == len(index) and len(np.unique(order)) == len(index), name

        path = out / f"{name}.kdf"
        if path.exists():
            path.unlink()
        write_kdf(
            forest,
            str(path),
            layout="shared",
            chunk_bytes=args.chunk_bytes,
            descriptor_block_bytes=args.block_bytes,
            descriptor_order=order.tolist(),
        )
        m = measure(path, queries, args.budget, baseline, reference)
        m.update(policy=name, order_seconds=build)
        rows.append(m)
        print(
            f"{name:<12} {build:>8.1f} {m['file_bytes'] / 1e6:>9.1f} {m['checks']:>7,}"
            f" {m['blocks']:>7,} {m['checks_per_block']:>10.2f} {m['reuse'] * 100:>6.1f}%"
            f" {m['cold_seconds']:>7.2f} {m['batch_reads']:>12,}"
            f" {m['batch_decoded'] / (1 << 20):>12.1f}"
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
                    "forest": {"trees": args.trees, "leaf_size": args.leaf_size},
                    "block_bytes": args.block_bytes,
                    "baseline_chunk_reads": baseline,
                    "results": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
