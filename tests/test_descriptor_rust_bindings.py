# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Rust descriptor matching bindings."""

import numpy as np
import pytest

from sfmtool._sfmtool import (
    descriptor_distance as rust_descriptor_distance,
    find_best_descriptor_match as rust_find_best_match,
)
from sfmtool._sift_file import SiftReader


class TestDescriptorDistance:
    def test_identical_descriptors(self):
        desc = np.full(128, 42, dtype=np.uint8)
        assert rust_descriptor_distance(desc, desc) == 0.0

    def test_known_distance(self):
        a = np.zeros(128, dtype=np.uint8)
        b = np.zeros(128, dtype=np.uint8)
        b[0] = 3
        b[1] = 4
        assert abs(rust_descriptor_distance(a, b) - 5.0) < 1e-10

    def test_max_distance_elements(self):
        a = np.zeros(128, dtype=np.uint8)
        b = np.full(128, 255, dtype=np.uint8)
        expected = np.sqrt(128 * 255.0**2)
        assert abs(rust_descriptor_distance(a, b) - expected) < 1e-6

    def test_random_descriptors(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.integers(0, 256, 128, dtype=np.uint8)
            b = rng.integers(0, 256, 128, dtype=np.uint8)
            rust_d = rust_descriptor_distance(a, b)
            expected = np.linalg.norm(a.astype(float) - b.astype(float))
            assert abs(rust_d - expected) < 1e-10

    def test_symmetry(self):
        rng = np.random.default_rng(123)
        a = rng.integers(0, 256, 128, dtype=np.uint8)
        b = rng.integers(0, 256, 128, dtype=np.uint8)
        assert rust_descriptor_distance(a, b) == rust_descriptor_distance(b, a)


class TestFindBestMatch:
    def test_empty_candidates(self):
        query = np.zeros(128, dtype=np.uint8)
        candidates = np.empty((0, 128), dtype=np.uint8)
        r_idx, r_dist = rust_find_best_match(query, candidates, None)
        assert r_idx is None
        assert np.isinf(r_dist)

    def test_single_candidate(self):
        query = np.zeros(128, dtype=np.uint8)
        candidates = np.ones((1, 128), dtype=np.uint8)
        r_idx, r_dist = rust_find_best_match(query, candidates, None)
        assert r_idx == 0
        expected = np.sqrt(128.0)
        assert abs(r_dist - expected) < 1e-10

    def test_selects_closest(self):
        query = np.zeros(128, dtype=np.uint8)
        candidates = np.array(
            [
                np.full(128, 100, dtype=np.uint8),
                np.full(128, 1, dtype=np.uint8),
                np.full(128, 50, dtype=np.uint8),
            ]
        )
        r_idx, r_dist = rust_find_best_match(query, candidates, None)
        assert r_idx == 1
        expected = np.sqrt(128.0)
        assert abs(r_dist - expected) < 1e-10

    def test_threshold_rejects(self):
        query = np.zeros(128, dtype=np.uint8)
        candidates = np.full((1, 128), 100, dtype=np.uint8)
        r_idx, r_dist = rust_find_best_match(query, candidates, 10.0)
        assert r_idx is None
        assert np.isinf(r_dist)

    def test_threshold_accepts(self):
        query = np.zeros(128, dtype=np.uint8)
        candidates = np.full((1, 128), 1, dtype=np.uint8)
        r_idx, r_dist = rust_find_best_match(query, candidates, 20.0)
        assert r_idx == 0
        expected = np.sqrt(128.0)
        assert abs(r_dist - expected) < 1e-10

    def test_random_candidates(self):
        rng = np.random.default_rng(99)
        query = rng.integers(0, 256, 128, dtype=np.uint8)
        candidates = rng.integers(0, 256, (50, 128), dtype=np.uint8)
        r_idx, r_dist = rust_find_best_match(query, candidates, None)
        dists = [
            np.linalg.norm(query.astype(float) - c.astype(float)) for c in candidates
        ]
        expected_idx = int(np.argmin(dists))
        assert r_idx == expected_idx
        assert abs(r_dist - dists[expected_idx]) < 1e-10

    def test_random_with_threshold(self):
        rng = np.random.default_rng(77)
        query = rng.integers(0, 256, 128, dtype=np.uint8)
        candidates = rng.integers(0, 256, (30, 128), dtype=np.uint8)
        for threshold in [50.0, 100.0, 200.0, 500.0, 1000.0]:
            r_idx, r_dist = rust_find_best_match(query, candidates, threshold)
            dists = [
                np.linalg.norm(query.astype(float) - c.astype(float))
                for c in candidates
            ]
            expected_idx = int(np.argmin(dists))
            expected_dist = dists[expected_idx]
            if expected_dist > threshold:
                assert r_idx is None, f"threshold={threshold}: should reject"
            else:
                assert r_idx == expected_idx, f"threshold={threshold}"
                assert abs(r_dist - expected_dist) < 1e-10


class TestDescriptorMatchingOnRealData:
    """Test Rust matching on real SIFT features from Seoul Bull dataset."""

    @pytest.fixture(scope="class")
    def seoul_bull_features(self, sfmrfile_reconstruction_with_17_images_once):
        workspace = sfmrfile_reconstruction_with_17_images_once.parent
        img_dir = workspace / "test_17_image"
        features_base = img_dir / "features"
        sift_dirs = list(features_base.glob("sift-*"))
        assert len(sift_dirs) >= 1
        sift_dir = sift_dirs[0]
        sift_files = sorted(sift_dir.glob("*.sift"))
        assert len(sift_files) >= 2

        features = []
        for sf in sift_files[:2]:
            with SiftReader(sf) as r:
                features.append(
                    {
                        "positions": r.read_positions(),
                        "descriptors": r.read_descriptors(),
                    }
                )
        return features

    def test_pairwise_distances(self, seoul_bull_features):
        descs1 = seoul_bull_features[0]["descriptors"]
        descs2 = seoul_bull_features[1]["descriptors"]
        n = min(50, len(descs1), len(descs2))
        for i in range(n):
            rust_d = rust_descriptor_distance(descs1[i], descs2[i])
            expected = np.linalg.norm(descs1[i].astype(float) - descs2[i].astype(float))
            assert abs(rust_d - expected) < 1e-10

    def test_best_match_search(self, seoul_bull_features):
        descs1 = seoul_bull_features[0]["descriptors"]
        descs2 = seoul_bull_features[1]["descriptors"]
        n_queries = min(30, len(descs1))
        candidates = np.ascontiguousarray(descs2, dtype=np.uint8)
        for i in range(n_queries):
            query = np.ascontiguousarray(descs1[i], dtype=np.uint8)
            r_idx, r_dist = rust_find_best_match(query, candidates, None)
            dists = [
                np.linalg.norm(query.astype(float) - c.astype(float))
                for c in candidates
            ]
            expected_idx = int(np.argmin(dists))
            assert r_idx == expected_idx
            assert abs(r_dist - dists[expected_idx]) < 1e-10

    def test_best_match_with_window_slice(self, seoul_bull_features):
        descs1 = seoul_bull_features[0]["descriptors"]
        descs2 = seoul_bull_features[1]["descriptors"]
        n_queries = min(20, len(descs1))
        window_size = 30
        for i in range(n_queries):
            start = max(0, i * 3)
            end = min(len(descs2), start + window_size)
            window = np.ascontiguousarray(descs2[start:end], dtype=np.uint8)
            query = np.ascontiguousarray(descs1[i], dtype=np.uint8)
            r_idx, r_dist = rust_find_best_match(query, window, None)
            if r_idx is not None:
                expected = np.linalg.norm(
                    query.astype(float) - window[r_idx].astype(float)
                )
                assert abs(r_dist - expected) < 1e-10
