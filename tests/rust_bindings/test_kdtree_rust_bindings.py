# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for KdTree2d and KdTree3d Rust bindings."""

import numpy as np
import pytest

from sfmtool._sfmtool import KdTree2d, KdTree3d


class TestKdTree2dF64:
    def test_nearest(self):
        positions = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
        tree = KdTree2d(positions)
        assert tree.len == 3
        assert tree.dtype == "float64"
        queries = np.array([[1.0, 1.0], [9.0, 0.0]])
        result = tree.nearest(queries)
        assert result.dtype == np.uint32
        assert result.shape == (2,)
        assert result[0] == 0
        assert result[1] == 2

    def test_nearest_k(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
        tree = KdTree2d(positions)
        queries = np.array([[0.5, 0.0]])
        result = tree.nearest_k(queries, k=2)
        assert result.dtype == np.uint32
        assert result.shape == (1, 2)
        assert set(result[0].tolist()) == {0, 1}

    def test_within_radius(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
        tree = KdTree2d(positions)
        queries = np.array([[0.5, 0.0]])
        offsets, indices = tree.within_radius(queries, radius=2.0)
        assert offsets.dtype == np.uint32
        assert indices.dtype == np.uint32
        assert offsets[0] == 0
        assert offsets[-1] == len(indices)
        assert sorted(indices.tolist()) == [0, 1, 2]

    def test_within_radius_batch(self):
        positions = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        tree = KdTree2d(positions)
        queries = np.array([[0.0, 0.0], [10.0, 0.0]])
        offsets, indices = tree.within_radius(queries, radius=1.0)
        assert offsets.shape == (3,)
        q0_results = sorted(indices[offsets[0] : offsets[1]].tolist())
        assert q0_results == [0]
        q1_results = sorted(indices[offsets[1] : offsets[2]].tolist())
        assert q1_results == [2]

    def test_self_nearest_k_default(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        tree = KdTree2d(positions)
        result = tree.self_nearest_k()
        assert result.dtype == np.uint32
        assert result.shape == (3, 1)
        assert result[0, 0] == 1
        assert result[1, 0] == 0
        assert result[2, 0] == 1

    def test_self_nearest_k_multiple(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
        tree = KdTree2d(positions)
        result = tree.self_nearest_k(k=2)
        assert result.shape == (4, 2)
        assert set(result[0].tolist()) == {1, 2}

    def test_wrong_dimension_raises(self):
        positions = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="expected \\(N, 2\\)"):
            KdTree2d(positions)

    def test_wrong_query_dimension_raises(self):
        positions = np.array([[0.0, 0.0]])
        tree = KdTree2d(positions)
        with pytest.raises((ValueError, TypeError)):
            tree.nearest(np.array([[0.0, 0.0, 0.0]]))


class TestKdTree2dF32:
    def test_nearest(self):
        positions = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]], dtype=np.float32)
        tree = KdTree2d(positions)
        assert tree.len == 3
        assert tree.dtype == "float32"
        queries = np.array([[1.0, 1.0], [9.0, 0.0]], dtype=np.float32)
        result = tree.nearest(queries)
        assert result.dtype == np.uint32
        assert result[0] == 0
        assert result[1] == 2

    def test_nearest_k(self):
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]], dtype=np.float32
        )
        tree = KdTree2d(positions)
        queries = np.array([[0.5, 0.0]], dtype=np.float32)
        result = tree.nearest_k(queries, k=2)
        assert result.shape == (1, 2)
        assert set(result[0].tolist()) == {0, 1}

    def test_within_radius(self):
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]], dtype=np.float32
        )
        tree = KdTree2d(positions)
        queries = np.array([[0.5, 0.0]], dtype=np.float32)
        offsets, indices = tree.within_radius(queries, radius=2.0)
        assert sorted(indices.tolist()) == [0, 1, 2]

    def test_self_nearest_k(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        tree = KdTree2d(positions)
        result = tree.self_nearest_k()
        assert result.shape == (3, 1)
        assert result[0, 0] == 1
        assert result[1, 0] == 0
        assert result[2, 0] == 1

    def test_wrong_dtype_raises(self):
        positions = np.array([[0.0, 0.0]], dtype=np.int32)
        with pytest.raises(TypeError):
            KdTree2d(positions)


class TestKdTree3dF64:
    def test_nearest(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        tree = KdTree3d(positions)
        assert tree.len == 3
        assert tree.dtype == "float64"
        queries = np.array([[0.5, 0.0, 0.0], [0.9, 0.0, 0.0]])
        result = tree.nearest(queries)
        assert result[0] == 0
        assert result[1] == 1

    def test_nearest_k(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
        )
        tree = KdTree3d(positions)
        queries = np.array([[0.5, 0.0, 0.0]])
        result = tree.nearest_k(queries, k=3)
        assert result.shape == (1, 3)
        assert set(result[0].tolist()) == {0, 1, 2}

    def test_within_radius(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        tree = KdTree3d(positions)
        queries = np.array([[0.5, 0.0, 0.0]])
        offsets, indices = tree.within_radius(queries, radius=2.0)
        assert sorted(indices.tolist()) == [0, 1]

    def test_self_nearest_k(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        tree = KdTree3d(positions)
        result = tree.self_nearest_k(k=1)
        assert result.shape == (3, 1)
        assert result[0, 0] == 1
        assert result[1, 0] == 0
        assert result[2, 0] == 1

    def test_wrong_dimension_raises(self):
        positions = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="expected \\(N, 3\\)"):
            KdTree3d(positions)


class TestKdTree3dF32:
    def test_nearest(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32
        )
        tree = KdTree3d(positions)
        assert tree.len == 3
        assert tree.dtype == "float32"
        queries = np.array([[0.5, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32)
        result = tree.nearest(queries)
        assert result[0] == 0
        assert result[1] == 1

    def test_nearest_k(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        tree = KdTree3d(positions)
        queries = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        result = tree.nearest_k(queries, k=3)
        assert result.shape == (1, 3)
        assert set(result[0].tolist()) == {0, 1, 2}

    def test_within_radius(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        tree = KdTree3d(positions)
        queries = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        offsets, indices = tree.within_radius(queries, radius=2.0)
        assert sorted(indices.tolist()) == [0, 1]

    def test_self_nearest_k(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32
        )
        tree = KdTree3d(positions)
        result = tree.self_nearest_k(k=1)
        assert result.shape == (3, 1)
        assert result[0, 0] == 1
        assert result[1, 0] == 0
        assert result[2, 0] == 1
