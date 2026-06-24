# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KdForest randomized kd-tree forest Rust binding."""

import numpy as np
import pytest

from sfmtool._sfmtool.spatial import KdForest


def _random_descriptors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, dim), dtype=np.uint8)


def _brute_force_nn(points, queries):
    """Exact top-1 nearest-neighbor index for each query (squared L2)."""
    pts = points.astype(np.int64)
    out = np.empty(len(queries), dtype=np.uint32)
    for i, q in enumerate(queries.astype(np.int64)):
        d = ((pts - q) ** 2).sum(axis=1)
        out[i] = int(np.argmin(d))
    return out


class TestKdForestBuild:
    def test_shape_and_attrs(self):
        pts = _random_descriptors(100, 128, 1)
        forest = KdForest(pts)
        assert forest.len == 100
        assert forest.dim == 128
        assert forest.max_leaf_checks > 0

    def test_dim_inferred_from_width(self):
        pts = _random_descriptors(20, 64, 2)
        forest = KdForest(pts)
        assert forest.dim == 64

    def test_dtype_getter(self):
        forest = KdForest(_random_descriptors(10, 16, 30))
        assert forest.dtype == "uint8"

    def test_len_and_is_empty(self):
        forest = KdForest(_random_descriptors(5, 16, 33))
        assert forest.len == 5
        assert forest.is_empty is False
        empty = KdForest(np.empty((0, 16), dtype=np.uint8))
        assert empty.len == 0
        assert empty.is_empty is True

    def test_unknown_preset_raises(self):
        pts = _random_descriptors(10, 16, 3)
        with pytest.raises(ValueError, match="unknown preset"):
            KdForest(pts, preset="turbo")

    def test_wrong_dtype_raises(self):
        pts = _random_descriptors(10, 16, 31).astype(np.float32)
        with pytest.raises(TypeError, match="uint8"):
            KdForest(pts)

    def test_presets(self):
        pts = _random_descriptors(50, 32, 4)
        for preset in ("balanced", "fast", "accurate"):
            forest = KdForest(pts, preset=preset)
            assert forest.len == 50


class TestKdForestQuery:
    def test_query_shapes_and_dtypes(self):
        pts = _random_descriptors(200, 128, 5)
        forest = KdForest(pts)
        queries = _random_descriptors(30, 128, 6)
        idx, dist = forest.query(queries, k=2)
        assert idx.shape == (30, 2)
        assert dist.shape == (30, 2)
        assert idx.dtype == np.uint32
        assert dist.dtype == np.float32

    def test_self_query_is_exact(self):
        # A descriptor queried against itself must return itself at distance 0.
        # Single leaf (leaf_size >= n) makes the search exact by construction.
        pts = _random_descriptors(150, 128, 7)
        forest = KdForest(pts, num_trees=1, leaf_size=len(pts))
        idx, dist = forest.query(pts, k=1, max_leaf_checks=len(pts))
        assert np.array_equal(idx[:, 0], np.arange(150, dtype=np.uint32))
        assert np.allclose(dist[:, 0], 0.0)

    def test_single_leaf_matches_brute_force(self):
        # A single-leaf forest scans every point, so it is exact.
        pts = _random_descriptors(300, 128, 8)
        queries = _random_descriptors(60, 128, 9)
        forest = KdForest(pts, num_trees=1, leaf_size=len(pts))
        idx, _ = forest.query(queries, k=1, max_leaf_checks=len(pts))
        expected = _brute_force_nn(pts, queries)
        assert np.array_equal(idx[:, 0], expected)

    def test_deep_tree_high_recall(self):
        # A real (deep) tree at full budget is approximate but recovers the
        # great majority of true neighbors (distance-based to tolerate ties).
        pts = _random_descriptors(300, 128, 8)
        queries = _random_descriptors(60, 128, 9)
        forest = KdForest(pts, num_trees=4, leaf_size=8)
        idx, _ = forest.query(queries, k=1, max_leaf_checks=len(pts))
        got = ((pts[idx[:, 0]].astype(np.int64) - queries.astype(np.int64)) ** 2).sum(
            axis=1
        )
        exact_idx = _brute_force_nn(pts, queries)
        exact = ((pts[exact_idx].astype(np.int64) - queries.astype(np.int64)) ** 2).sum(
            axis=1
        )
        recall = float(np.mean(got <= exact))
        assert recall >= 0.9

    def test_noncontiguous_arrays(self):
        # Non-C-contiguous build/query inputs (a sliced view) must still index
        # the correct descriptors via the copy fallback.
        base = _random_descriptors(80, 128, 20)
        wide = np.repeat(base, 2, axis=1)  # (80, 256)
        view = wide[:, ::2]  # non-contiguous (80, 128) equal to base
        assert not view.flags["C_CONTIGUOUS"]
        forest = KdForest(view, num_trees=1, leaf_size=len(view))
        idx, dist = forest.query(view[:5], k=1, max_leaf_checks=len(view))
        assert np.array_equal(idx[:, 0], np.arange(5, dtype=np.uint32))
        assert np.allclose(dist[:, 0], 0.0)

    def test_distances_are_euclidean(self):
        pts = _random_descriptors(120, 128, 10)
        queries = _random_descriptors(20, 128, 11)
        forest = KdForest(pts, preset="accurate")
        idx, dist = forest.query(queries, k=1, max_leaf_checks=len(pts))
        for i in range(len(queries)):
            j = int(idx[i, 0])
            ref = np.linalg.norm(
                pts[j].astype(np.float64) - queries[i].astype(np.float64)
            )
            assert dist[i, 0] == pytest.approx(ref, rel=1e-4)

    def test_determinism(self):
        pts = _random_descriptors(200, 128, 12)
        queries = _random_descriptors(40, 128, 13)
        f1 = KdForest(pts, seed=42)
        f2 = KdForest(pts, seed=42)
        i1, _ = f1.query(queries, k=3, max_leaf_checks=80)
        i2, _ = f2.query(queries, k=3, max_leaf_checks=80)
        assert np.array_equal(i1, i2)

    def test_fewer_than_k_padding(self):
        pts = _random_descriptors(3, 16, 14)
        forest = KdForest(pts)
        idx, dist = forest.query(pts[:1], k=5, max_leaf_checks=100)
        assert idx.shape == (1, 5)
        # Only 3 points exist; remaining slots padded.
        assert idx[0, 3] == np.iinfo(np.uint32).max
        assert np.isinf(dist[0, 4])

    def test_max_dist_cutoff(self):
        near = np.ones((10, 8), dtype=np.uint8)
        far = np.full((10, 8), 200, dtype=np.uint8)
        pts = np.vstack([near, far])
        forest = KdForest(pts, preset="accurate")
        query = np.zeros((1, 8), dtype=np.uint8)
        idx, _ = forest.query(query, k=20, max_leaf_checks=len(pts), max_dist=10.0)
        filled = idx[0][idx[0] != np.iinfo(np.uint32).max]
        assert len(filled) > 0
        assert np.all(filled < 10), "cutoff admitted a far-cluster point"

    def test_dimension_mismatch_raises(self):
        pts = _random_descriptors(20, 32, 15)
        forest = KdForest(pts)
        with pytest.raises(ValueError, match="does not match forest dim"):
            forest.query(_random_descriptors(5, 16, 16))
