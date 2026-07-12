# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster-covisibility Rust bindings
(``sfmtool._sfmtool.matching.ClusterCovisibility``; see
``specs/core/cluster-covisibility.md``)."""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.io import write_matches
from sfmtool._sfmtool.matching import ClusterCovisibility


# ── Fixture data ──────────────────────────────────────────────────────────

N_IMAGES = 4

# 4 images, 5 clusters. Cluster 0 spans {0, 1, 2}; clusters 1-2 span {0, 2};
# cluster 3 spans {1, 3}; cluster 4 has span 1 and contributes nothing.
CLUSTER_STARTS = np.array([0, 3, 5, 7, 9, 10], dtype=np.uint32)
MEMBER_IMAGES = np.array([0, 1, 2, 0, 2, 0, 2, 1, 3, 3], dtype=np.uint32)
MEMBER_FEATURES = np.array([0, 0, 0, 1, 1, 2, 2, 1, 0, 1], dtype=np.uint32)

# Statuses for the cluster_patches variant: cluster 1's image-2 member and
# cluster 3's image-3 member are rejected, removing one (0, 2) vote and all
# of cluster 3 (masked span 1).
MEMBER_STATUS = np.array([0, 1, 1, 0, 2, 0, 1, 0, 3, 5], dtype=np.uint8)

# File-fixture variants without the span-1 cluster: the `.matches` writer
# requires every stored cluster to have >= 2 members. Covisibility is
# unchanged (span-1 clusters contribute nothing).
FILE_STARTS = CLUSTER_STARTS[:-1]
FILE_IMAGES = MEMBER_IMAGES[:9]
FILE_FEATURES = MEMBER_FEATURES[:9]
FILE_STATUS = MEMBER_STATUS[:9]


def _ref_covisibility(starts, images, num_images, accepted=None):
    """Numpy reference: dense symmetric shared-cluster counts."""
    counts = np.zeros((num_images, num_images), dtype=np.uint32)
    for c in range(len(starts) - 1):
        members = images[starts[c] : starts[c + 1]]
        if accepted is not None:
            members = members[accepted[starts[c] : starts[c + 1]]]
        span = np.unique(members)
        for a in range(len(span)):
            for b in range(a + 1, len(span)):
                counts[span[a], span[b]] += 1
                counts[span[b], span[a]] += 1
    return counts


def _ref_seed_groups(counts, group_size=5, min_shared=8):
    """Numpy reference implementation of the spec's seed-group algorithm."""
    n = len(counts)
    excluded = np.zeros(n, dtype=bool)
    groups = []
    while True:
        best = None
        for i in range(n):
            if excluded[i]:
                continue
            for j in range(i + 1, n):
                if excluded[j]:
                    continue
                if best is None or counts[i, j] > best[0]:
                    best = (int(counts[i, j]), i, j)
        if best is None or best[0] < min_shared:
            return groups
        group = [best[1], best[2]]
        while len(group) < group_size:
            best_k = None
            for k in range(n):
                if excluded[k] or k in group:
                    continue
                min_w = min(int(counts[k, g]) for g in group)
                if best_k is None or min_w > best_k[0]:
                    best_k = (min_w, k)
            if best_k is None or best_k[0] < min_shared:
                break
            group.append(best_k[1])
        group.sort()
        groups.append(group)
        excluded[group] = True


def _write_cluster_matches(path, with_patches):
    """Write a minimal cluster-backbone .matches file (format version 4)."""
    metadata = {
        "version": 4,
        "matching_method": "cluster",
        "matching_tool": "sfmtool",
        "matching_tool_version": "0.2",
        "matching_options": {"d": 10, "alpha": 0.8},
        "workspace": {
            "absolute_path": "/tmp/workspace",
            "relative_path": "..",
            "contents": {
                "feature_tool": "sfmtool",
                "feature_type": "sift",
                "feature_options": {},
                "feature_prefix_dir": "features/sift-sfmtool-abc123",
            },
        },
        "timestamp": "2026-07-12T10:00:00Z",
        "image_count": N_IMAGES,
        "cluster_count": len(FILE_STARTS) - 1,
        "cluster_member_count": len(FILE_IMAGES),
        "has_two_view_geometries": False,
        "has_clusters": True,
        "has_cluster_patches": with_patches,
    }
    data = {
        "metadata": metadata,
        "image_names": [f"frames/frame_{i:03d}.jpg" for i in range(N_IMAGES)],
        "feature_tool_hashes": [b"\x00" * 16] * N_IMAGES,
        "sift_content_hashes": [b"\x01" * 16] * N_IMAGES,
        "feature_counts": np.full(N_IMAGES, 10, dtype=np.uint32),
        "image_dims": np.full((N_IMAGES, 2), [640, 480], dtype=np.uint32),
        "has_clusters": True,
        "cluster_starts": FILE_STARTS,
        "member_images": FILE_IMAGES,
        "member_features": FILE_FEATURES,
        "matcher_options": {"d": 10, "alpha": 0.8, "min_size": 2},
        "has_cluster_patches": with_patches,
        "has_two_view_geometries": False,
    }
    if with_patches:
        m = len(FILE_IMAGES)
        affines = np.zeros((m, 2, 3), dtype=np.float64)
        reference_members = []
        for c in range(len(FILE_STARTS) - 1):
            lo, hi = int(FILE_STARTS[c]), int(FILE_STARTS[c + 1])
            ref = next(
                (k for k in range(lo, hi) if FILE_STATUS[k] == 0),
                np.iinfo(np.uint32).max,
            )
            reference_members.append(ref)
            for k in range(lo, hi):
                if FILE_STATUS[k] in (0, 1, 2, 3):
                    affines[k] = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]
                if FILE_STATUS[k] != 0:
                    affines[k, 0, 0] = 1.01  # non-reference rows need not be identity
        data.update(
            {
                "reference_members": np.array(reference_members, dtype=np.uint32),
                "member_status": FILE_STATUS,
                "member_affines": affines,
                "member_zncc": np.ones(m, dtype=np.float32),
                "member_shift_px": np.zeros(m, dtype=np.float32),
                "member_consistency_residual": np.full(m, np.nan, dtype=np.float32),
                "refine_options": {"radius": 4.0, "resolution": 15},
            }
        )
    write_matches(path, data)


def _write_pairwise_matches(path):
    """Write a minimal pairwise-backbone .matches file."""
    data = {
        "metadata": {
            "version": 4,
            "matching_method": "sequential",
            "matching_tool": "sfmtool",
            "matching_tool_version": "0.2",
            "matching_options": {},
            "workspace": {
                "absolute_path": "/tmp/workspace",
                "relative_path": "..",
                "contents": {
                    "feature_tool": "sfmtool",
                    "feature_type": "sift",
                    "feature_options": {},
                    "feature_prefix_dir": "features/sift-sfmtool-abc123",
                },
            },
            "timestamp": "2026-07-12T10:00:00Z",
            "image_count": 2,
            "image_pair_count": 1,
            "match_count": 1,
            "has_two_view_geometries": False,
        },
        "image_names": ["a.jpg", "b.jpg"],
        "feature_tool_hashes": [b"\x00" * 16] * 2,
        "sift_content_hashes": [b"\x01" * 16] * 2,
        "feature_counts": np.array([1, 1], dtype=np.uint32),
        "image_dims": np.array([[640, 480], [640, 480]], dtype=np.uint32),
        "image_index_pairs": np.array([[0, 1]], dtype=np.uint32),
        "match_counts": np.array([1], dtype=np.uint32),
        "match_feature_indexes": np.array([[0, 0]], dtype=np.uint32),
        "match_descriptor_distances": np.array([100.0], dtype=np.float32),
        "has_two_view_geometries": False,
    }
    write_matches(path, data)


def _edges_to_arrays(num_images, edges):
    """Synthesize cluster arrays whose covisibility equals the weighted
    edges: edge (i, j, w) becomes w two-member clusters."""
    starts = [0]
    images = []
    for i, j, w in edges:
        for _ in range(w):
            images.extend([min(i, j), max(i, j)])
            starts.append(len(images))
    return (
        np.array(starts, dtype=np.uint32),
        np.array(images, dtype=np.uint32),
    )


# ── Constructors ──────────────────────────────────────────────────────────


class TestConstructors:
    def test_from_arrays_counts_match_reference(self):
        cov = ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        assert cov.num_images == N_IMAGES
        expected = _ref_covisibility(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        npt.assert_array_equal(cov.counts, expected)
        # Spot values: (0, 2) shares clusters 0-2, (0, 1) and (1, 2) share
        # cluster 0, (1, 3) shares cluster 3.
        assert cov.counts[0, 2] == 3
        assert cov.counts[0, 1] == cov.counts[1, 2] == 1
        assert cov.counts[1, 3] == 1

    def test_from_arrays_with_mask(self):
        accepted = np.isin(MEMBER_STATUS, (0, 1))
        cov = ClusterCovisibility.from_arrays(
            CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES, member_accepted=accepted
        )
        expected = _ref_covisibility(
            CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES, accepted=accepted
        )
        npt.assert_array_equal(cov.counts, expected)
        assert cov.counts[0, 2] == 2  # cluster 1's (0, 2) vote is masked out
        assert cov.counts[1, 3] == 0  # cluster 3's masked span drops to 1

    def test_counts_properties(self):
        cov = ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        counts = cov.counts
        assert counts.dtype == np.uint32
        assert counts.shape == (N_IMAGES, N_IMAGES)
        npt.assert_array_equal(counts, counts.T)
        npt.assert_array_equal(np.diag(counts), 0)

    def test_from_matches_file_all_members(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        cov = ClusterCovisibility.from_matches_file(path)
        expected = _ref_covisibility(FILE_STARTS, FILE_IMAGES, N_IMAGES)
        npt.assert_array_equal(cov.counts, expected)

    def test_from_matches_file_str_path(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        cov = ClusterCovisibility.from_matches_file(str(path))
        assert cov.num_images == N_IMAGES

    def test_from_matches_file_default_mask_is_reference_or_kept(self, tmp_path):
        path = tmp_path / "cluster-patches.matches"
        _write_cluster_matches(path, with_patches=True)
        cov = ClusterCovisibility.from_matches_file(path)
        accepted = np.isin(FILE_STATUS, (0, 1))
        expected = _ref_covisibility(
            FILE_STARTS, FILE_IMAGES, N_IMAGES, accepted=accepted
        )
        npt.assert_array_equal(cov.counts, expected)
        # The rejected members' votes are actually gone: one (0, 2) vote and
        # cluster 3's whole (1, 3) edge.
        assert cov.counts[0, 2] == 2
        assert cov.counts[1, 3] == 0

    def test_from_matches_file_rejects_pairwise_backbone(self, tmp_path):
        path = tmp_path / "pairwise.matches"
        _write_pairwise_matches(path)
        with pytest.raises(ValueError, match="clusters"):
            ClusterCovisibility.from_matches_file(path)


class TestValidation:
    def test_wrong_starts_dtype_raises(self):
        with pytest.raises(TypeError, match="uint32"):
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS.astype(np.int64), MEMBER_IMAGES, N_IMAGES
            )

    def test_wrong_mask_dtype_raises(self):
        with pytest.raises(TypeError, match="bool"):
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS,
                MEMBER_IMAGES,
                N_IMAGES,
                member_accepted=np.ones(len(MEMBER_IMAGES), dtype=np.uint8),
            )

    def test_bad_csr_raises(self):
        bad = CLUSTER_STARTS.copy()
        bad[-1] += 1
        with pytest.raises(ValueError, match="cluster_starts"):
            ClusterCovisibility.from_arrays(bad, MEMBER_IMAGES, N_IMAGES)

    def test_mask_not_parallel_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS,
                MEMBER_IMAGES,
                N_IMAGES,
                member_accepted=np.ones(3, dtype=bool),
            )

    def test_image_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, 2)

    def test_dense_bound_raises(self):
        starts = np.array([0], dtype=np.uint32)
        images = np.array([], dtype=np.uint32)
        with pytest.raises(ValueError, match="dense covisibility bound"):
            ClusterCovisibility.from_arrays(starts, images, 5000)


# ── Queries ───────────────────────────────────────────────────────────────


class TestRankByCovisibility:
    def test_orders_and_drops_zeros(self):
        cov = ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        candidates = np.array([3, 2, 1], dtype=np.uint32)
        ranked = cov.rank_by_covisibility(0, candidates)
        assert ranked.dtype == np.uint32
        # W[0] = [0, 1, 3, 0]: image 2 first, then 1; 3 has zero covisibility.
        npt.assert_array_equal(ranked, [2, 1])

    def test_out_of_range_raises(self):
        cov = ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        with pytest.raises(ValueError, match="out of range"):
            cov.rank_by_covisibility(N_IMAGES, np.array([0], dtype=np.uint32))
        with pytest.raises(ValueError, match="out of range"):
            cov.rank_by_covisibility(0, np.array([N_IMAGES], dtype=np.uint32))


class TestSeedGroups:
    def test_parity_with_numpy_reference(self):
        # A deterministic pseudo-random weighted graph on 10 images.
        rng = np.random.default_rng(7)
        edges = [
            (i, j, int(w))
            for (i, j), w in zip(
                [(i, j) for i in range(10) for j in range(i + 1, 10)],
                rng.integers(0, 16, size=45),
            )
            if w > 0
        ]
        starts, images = _edges_to_arrays(10, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 10)
        for group_size, min_shared in [(5, 8), (3, 4), (2, 1), (4, 12)]:
            got = list(cov.seed_groups(group_size=group_size, min_shared=min_shared))
            want = _ref_seed_groups(
                cov.counts, group_size=group_size, min_shared=min_shared
            )
            assert got == want, (group_size, min_shared)

    def test_defaults_and_iterator_protocol(self):
        # Two disjoint cliques above the default min_shared of 8.
        edges = [(0, 1, 10), (0, 2, 10), (1, 2, 10), (3, 4, 9)]
        starts, images = _edges_to_arrays(5, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 5)
        it = cov.seed_groups()
        assert iter(it) is it
        assert next(it) == [0, 1, 2]
        assert next(it) == [3, 4]
        with pytest.raises(StopIteration):
            next(it)

    def test_lazy_prefix_matches_eager_list(self):
        edges = [(0, 1, 10), (0, 2, 10), (1, 2, 10), (3, 4, 9), (5, 6, 8)]
        starts, images = _edges_to_arrays(7, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 7)
        eager = list(cov.seed_groups())
        assert len(eager) == 3
        lazy = cov.seed_groups()
        assert [next(lazy), next(lazy)] == eager[:2]

    def test_iterators_are_independent(self):
        edges = [(0, 1, 10), (2, 3, 9)]
        starts, images = _edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4)
        a = cov.seed_groups()
        b = cov.seed_groups()
        assert next(a) == [0, 1]
        assert next(b) == [0, 1]  # b holds its own exclusion state

    def test_star_topology_does_not_form_a_group(self):
        edges = [(0, 1, 10), (0, 2, 10), (0, 3, 10)]
        starts, images = _edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4)
        assert list(cov.seed_groups()) == [[0, 1]]
