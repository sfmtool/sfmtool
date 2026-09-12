# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Find which other images contain a patch, by querying a `.kdf`.

Given a rectangle in one image, take the constellation of SIFT features inside it,
look each one up in an existing capture's index, and ask which other images hold
the same constellation under a roughly-affine transform. A match is a candidate
image with enough mutually consistent correspondences to survive RANSAC on an
affine model — consistent geometry, not just a pile of descriptor hits.

This is the access pattern the file-backed index is actually suited to. A whole
image is not: 8,192 descriptors at a 128-leaf budget make about a million checks,
which reach most of a nine-million-descriptor corpus, so the working set is the
whole file. A patch is a few dozen to a few hundred features, so it touches a small
fraction and a small cache is enough.

`k` is deliberately larger here than for whole-corpus matching. Most of a
constellation feature's nearest neighbours will belong to images that do not
contain the patch; RANSAC needs enough candidates per feature that the right image
is among them, so the default is 32 rather than the matcher's 11.

Feature IDs come back as corpus rows, and the `.kdf` carries the SIFT provenance
that maps each one to its source image and feature index, so the candidate grouping
needs no side table.

Run:
    pixi run python scripts/kdf_patch_localize.py --workspace WS --scratch DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.spatial import KdForest, LazyKdForest, write_kdf
from sfmtool.sift.file import SiftReader

KIB = 1 << 10
MIB = 1 << 20


def load_images(features: Path, limit: int | None):
    """Descriptors, keypoint positions and names, per image, in filename order."""
    descriptors, positions, names, total = [], [], [], 0
    for path in sorted(features.glob("*.sift")):
        reader = SiftReader(path)
        descriptors.append(reader.read_descriptors())
        positions.append(reader.read_positions())
        reader.close()
        names.append(path.name.removesuffix(".sift"))
        total += len(descriptors[-1])
        if limit is not None and total >= limit:
            break
    return descriptors, positions, names


def affine_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray | None]:
    """Inlier count and model for the best affine fit from `src` to `dst`.

    Three correspondences determine an affine transform, so each trial draws three
    and scores the rest by reprojection distance. Affine rather than homography
    because a small patch seen from a nearby viewpoint is well approximated by one,
    and a 3-point model needs far fewer trials to hit a clean sample than a
    4-point one.
    """
    n = len(src)
    if n < 3:
        return 0, None
    best_count, best_model = 0, None
    ones = np.ones((n, 1), dtype=np.float64)
    src_h = np.hstack([src.astype(np.float64), ones])
    for _ in range(iterations):
        pick = rng.choice(n, size=3, replace=False)
        a = src_h[pick]
        if abs(np.linalg.det(a)) < 1e-9:
            continue
        try:
            model = np.linalg.solve(a, dst[pick].astype(np.float64))
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(src_h @ model - dst.astype(np.float64), axis=1)
        count = int(np.count_nonzero(residual <= threshold))
        if count > best_count:
            best_count, best_model = count, model
    return best_count, best_model


def localize(
    neighbours: np.ndarray,
    distances: np.ndarray,
    patch_xy: np.ndarray,
    image_of: np.ndarray,
    feature_of: np.ndarray,
    positions: list[np.ndarray],
    args,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Candidate images ranked by affine-consistent inliers.

    `neighbours` is (M, k) corpus rows per constellation feature. Grouping by
    source image first keeps RANSAC off images that cannot possibly win: a model
    needs three correspondences, so an image with fewer is skipped outright.

    RANSAC is seeded per candidate image rather than from a generator threaded
    through the run. Sharing one generator would make each path's draws depend on
    how many calls preceded it, so two paths given identical neighbours would still
    disagree — and the disagreement would look like an index bug.
    """
    by_image: dict[int, list[tuple[int, int]]] = {}
    for feature, row in enumerate(neighbours):
        for corpus_row, distance in zip(row, distances[feature]):
            if corpus_row == np.iinfo(np.uint32).max or not np.isfinite(distance):
                continue
            image = int(image_of[corpus_row])
            by_image.setdefault(image, []).append(
                (feature, int(feature_of[corpus_row]))
            )

    results = []
    for image, pairs in by_image.items():
        if len(pairs) < args.min_correspondences:
            continue
        src = patch_xy[[f for f, _ in pairs]]
        dst = positions[image][[g for _, g in pairs]]
        inliers, _ = affine_ransac(
            src,
            dst,
            args.threshold,
            args.iterations,
            np.random.default_rng(seed + image),
        )
        if inliers >= args.min_inliers:
            results.append((image, inliers, len(pairs)))
    results.sort(key=lambda r: -r[1])
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--features")
    p.add_argument("--scratch", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--patches", type=int, default=8, help="patches to localize")
    p.add_argument(
        "--patch-size", type=float, default=400.0, help="rectangle edge, pixels"
    )
    p.add_argument(
        "--k", type=int, default=32, help="neighbours per constellation feature"
    )
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--block-bytes", type=int, default=4 * KIB)
    p.add_argument("--chunk-bytes", type=int, default=1 * MIB)
    p.add_argument("--caches", default="16,64,256", help="cache budgets in MiB to try")
    p.add_argument("--threshold", type=float, default=8.0, help="RANSAC inlier px")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--min-correspondences", type=int, default=3)
    p.add_argument("--min-inliers", type=int, default=6)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", action="store_true")
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

    descriptors, positions, names = load_images(features, args.limit)
    counts = [len(d) for d in descriptors]
    corpus = np.vstack(descriptors)
    # Corpus row -> (image, feature index). The `.kdf` can answer this from its own
    # provenance; kept here too so the in-memory path can be compared against it.
    image_of = np.concatenate(
        [np.full(c, i, dtype=np.uint32) for i, c in enumerate(counts)]
    )
    feature_of = np.concatenate([np.arange(c, dtype=np.uint32) for c in counts])
    print(f"capture {len(corpus):,} descriptors, {len(descriptors)} images")

    t = time.perf_counter()
    forest = KdForest(
        corpus,
        num_trees=args.trees,
        leaf_size=args.leaf_size,
        max_leaf_checks=args.budget,
        seed=args.seed,
    )
    build_seconds = time.perf_counter() - t
    out = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "capture.kdf"
    if path.exists():
        path.unlink()
    write_kdf(
        forest,
        str(path),
        layout="shared",
        chunk_bytes=args.chunk_bytes,
        descriptor_block_bytes=args.block_bytes,
    )
    print(
        f"index: build {build_seconds:.1f}s, file {path.stat().st_size / 1e6:.0f} MB\n"
    )

    # Pick patches from images spread through the capture, centred on a dense
    # cluster of features so the constellation is not mostly empty sky.
    rng = np.random.default_rng(args.seed)
    step = max(1, len(descriptors) // args.patches)
    patches = []
    for image in list(range(0, len(descriptors), step))[: args.patches]:
        xy = positions[image]
        centre = xy[rng.integers(len(xy))]
        half = args.patch_size / 2
        inside = np.flatnonzero(
            (np.abs(xy[:, 0] - centre[0]) <= half)
            & (np.abs(xy[:, 1] - centre[1]) <= half)
        )
        if len(inside) >= args.min_inliers:
            patches.append((image, inside))
    if not patches:
        raise SystemExit("no patch held enough features; raise --patch-size")
    sizes = [len(i) for _, i in patches]
    print(
        f"{len(patches)} patches of {args.patch_size:.0f}px:"
        f" {min(sizes)}-{max(sizes)} features each, {statistics.median(sizes):.0f} median\n"
    )

    rows = []
    print(
        f"{'path':>18} {'open ms':>8} {'query ms':>9} {'ransac ms':>10}"
        f" {'reads':>8} {'decoded MB':>11} {'found':>6} {'agree':>6}"
    )

    reference = []
    query_ms, ransac_ms = [], []
    for image, inside in patches:
        patch = descriptors[image][inside]
        patch_xy = positions[image][inside]
        t = time.perf_counter()
        idx, dist = forest.query(patch, k=args.k, max_leaf_checks=args.budget)
        query_ms.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        found = localize(
            idx, dist, patch_xy, image_of, feature_of, positions, args, args.seed
        )
        ransac_ms.append((time.perf_counter() - t) * 1e3)
        reference.append((idx, dist, {i for i, _, _ in found}))
    print(
        f"{'in memory':>18} {'-':>8} {statistics.median(query_ms):>9.1f}"
        f" {statistics.median(ransac_ms):>10.1f} {'-':>8} {'-':>11}"
        f" {statistics.median(len(r[2]) for r in reference):>6.0f} {'ref':>6}"
    )
    rows.append({"path": "memory", "query_ms": query_ms, "ransac_ms": ransac_ms})

    for mib in (int(v) for v in args.caches.split(",")):
        budget = mib * MIB
        t = time.perf_counter()
        lazy = LazyKdForest(
            str(path),
            cache_bytes=budget,
            max_in_flight_bytes=budget,
            max_chunk_bytes=min(budget, 4 * MIB),
            max_compressed_bytes=64 * MIB,
            query_workers=args.workers,
        )
        open_ms = (time.perf_counter() - t) * 1e3

        q_ms, r_ms, agree, reads, decoded, found_counts = [], [], 0, 0, 0, []
        for (image, inside), want in zip(patches, reference):
            patch = descriptors[image][inside]
            patch_xy = positions[image][inside]
            lazy.reset_io_stats()
            t = time.perf_counter()
            idx, dist = lazy.query(patch, k=args.k, max_leaf_checks=args.budget)
            q_ms.append((time.perf_counter() - t) * 1e3)
            np.testing.assert_array_equal(idx, want[0])
            np.testing.assert_array_equal(dist, want[1])
            io = lazy.io_stats()
            reads += io["read_calls"]
            decoded += io["decoded_bytes"]
            t = time.perf_counter()
            found = localize(
                idx, dist, patch_xy, image_of, feature_of, positions, args, args.seed
            )
            r_ms.append((time.perf_counter() - t) * 1e3)
            found_counts.append(len(found))
            assert {i for i, _, _ in found} == want[2]
            agree += 1
        print(
            f"{f'kdf {mib} MiB':>18} {open_ms:>8.0f} {statistics.median(q_ms):>9.1f}"
            f" {statistics.median(r_ms):>10.1f} {reads // len(patches):>8,}"
            f" {decoded / len(patches) / 1e6:>11.1f}"
            f" {statistics.median(found_counts):>6.0f} {f'{agree}/{len(patches)}':>6}"
        )
        rows.append(
            {
                "path": f"kdf-{mib}MiB",
                "open_seconds": open_ms / 1e3,
                "query_ms": q_ms,
                "ransac_ms": r_ms,
                "reads_per_patch": reads // len(patches),
                "decoded_bytes_per_patch": decoded // len(patches),
                "agreed": agree,
                "patches": len(patches),
            }
        )
        del lazy

    if not args.keep:
        path.unlink()
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "corpus": int(len(corpus)),
                    "images": len(descriptors),
                    "k": args.k,
                    "patch_size": args.patch_size,
                    "patch_features": sizes,
                    "results": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
