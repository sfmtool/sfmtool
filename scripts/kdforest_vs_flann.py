# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sfmtool's randomized kd-tree forest against OpenCV's FLANN.

Both index the *same* SIFT descriptors with the *same* algorithm (multiple
randomized kd-trees, Muja & Lowe 2009) and expose the same knobs: number of
trees (T) and the per-query check budget (checks / max_leaf_checks). This makes
the comparison apples-to-apples; the only difference is the implementation
(sfmtool: pure-Rust, integer u8 squared-L2; FLANN: C++, float32 squared-L2 over
the identical values).

We report build time, mean query time, recall@1 (vs exact brute force), and the
speedup over a brute-force scan, across a budget sweep at a fixed tree count.

Run:  pixi run python scripts/kdforest_vs_flann.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from sfmtool._sfmtool import KdForest

TREES = 4
BUDGETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_QUERIES = 1500
SEED = 0
DATASET = Path(__file__).resolve().parent.parent / "test-data/images/dino_dog_toy"
FLANN_KDTREE = 1  # cv2 FLANN_INDEX_KDTREE


def extract_descriptors(max_images: int = 8, per_image_cap: int = 4000) -> np.ndarray:
    """Pool SIFT descriptors from the first few dataset images."""
    sift = cv2.SIFT_create()
    images = sorted(DATASET.glob("*.jpg"))[:max_images]
    chunks = []
    for path in images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        _, desc = sift.detectAndCompute(gray, None)
        if desc is not None:
            chunks.append(desc[:per_image_cap])
    desc = np.vstack(chunks)
    # Quantize to u8 once so both indices see byte-identical numeric values.
    return np.clip(np.rint(desc), 0, 255).astype(np.uint8)


def exact_nn_distsq(db: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact squared-L2 distance to the true nearest neighbor for each query."""
    db64 = db.astype(np.float64)
    db_sq = (db64 * db64).sum(axis=1)
    out = np.empty(len(queries), dtype=np.float64)
    for i in range(0, len(queries), 256):
        q = queries[i : i + 256].astype(np.float64)
        d = q @ db64.T
        dist = (q * q).sum(axis=1, keepdims=True) + db_sq[None, :] - 2.0 * d
        out[i : i + 256] = dist.min(axis=1)
    return out


def recall_at_1(found_distsq: np.ndarray, exact_distsq: np.ndarray) -> float:
    """Fraction of queries whose returned neighbor is at the exact min distance.

    Distance-based (not index-based) so descriptor ties count as hits for both.
    """
    tol = np.maximum(1.0, exact_distsq * 1e-4)
    return float(np.mean(found_distsq <= exact_distsq + tol))


def main() -> None:
    rng = np.random.default_rng(SEED)
    desc = extract_descriptors()
    perm = rng.permutation(len(desc))
    q_idx, db_idx = perm[:N_QUERIES], perm[N_QUERIES:]
    db_u8, q_u8 = desc[db_idx], desc[q_idx]
    db_f32, q_f32 = db_u8.astype(np.float32), q_u8.astype(np.float32)
    threads = os.environ.get("RAYON_NUM_THREADS", "all")
    print(
        f"database: {len(db_u8)} descriptors x {db_u8.shape[1]}d   queries: {len(q_u8)}"
    )
    print(
        f"trees T = {TREES}   sfmtool rayon threads = {threads}   (FLANN knnSearch is single-threaded)\n"
    )

    exact = exact_nn_distsq(db_u8, q_u8)

    # Brute-force scan time (the speedup baseline).
    t0 = time.perf_counter()
    _ = exact_nn_distsq(db_u8, q_u8)
    brute_ms = (time.perf_counter() - t0) * 1e3

    # --- Build ---
    t0 = time.perf_counter()
    forest = KdForest(db_u8, num_trees=TREES, max_leaf_checks=BUDGETS[0])
    sfm_build_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    flann = cv2.flann.Index(db_f32, {"algorithm": FLANN_KDTREE, "trees": TREES})
    flann_build_ms = (time.perf_counter() - t0) * 1e3

    print(f"build:  sfmtool {sfm_build_ms:7.1f} ms    FLANN {flann_build_ms:7.1f} ms")
    print(f"brute-force scan of {len(q_u8)} queries: {brute_ms:7.1f} ms\n")

    hdr = (
        f"{'budget':>6} | {'sfm ms':>8} {'sfm rec':>8} {'sfm x':>7} | "
        f"{'flann ms':>9} {'fln rec':>8} {'fln x':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for budget in BUDGETS:
        # sfmtool
        t0 = time.perf_counter()
        s_idx, s_dist = forest.query(q_u8, k=1, max_leaf_checks=budget)
        sfm_ms = (time.perf_counter() - t0) * 1e3
        sfm_distsq = (s_dist[:, 0].astype(np.float64)) ** 2
        sfm_rec = recall_at_1(sfm_distsq, exact)

        # FLANN
        t0 = time.perf_counter()
        f_idx, f_dist = flann.knnSearch(q_f32, 1, params={"checks": budget})
        fln_ms = (time.perf_counter() - t0) * 1e3
        fln_rec = recall_at_1(f_dist[:, 0].astype(np.float64), exact)

        print(
            f"{budget:>6} | {sfm_ms:8.2f} {sfm_rec:8.3f} {brute_ms / sfm_ms:6.0f}x | "
            f"{fln_ms:9.2f} {fln_rec:8.3f} {brute_ms / fln_ms:6.0f}x"
        )


if __name__ == "__main__":
    main()
