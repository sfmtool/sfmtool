# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster-covisibility Rust bindings
(``sfmtool._sfmtool.matching.ClusterCovisibility``; see
``specs/core/features/cluster-covisibility.md`` and, for the selection queries —
pair displacement, banded thinning, reach —
``specs/core/features/covisibility-selection.md``)."""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.io import MatchesFile, write_matches
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

# Distinct per-member keypoint positions, at the backbone's own float32
# width: the counts ignore them, the displacement queries a from_matches
# matrix answers do not.
FILE_POSITIONS = np.arange(len(FILE_IMAGES) * 2, dtype=np.float32).reshape(-1, 2) * 2.0


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


def _ref_seed_image_groups(counts, group_size=5, min_shared=8, pair_mean=None):
    """Numpy reference implementation of the spec's seed-group algorithm.

    Yields the same dicts the binding does: ``images``, ``seed_pair``,
    ``seed_shared``, ``pair_shared`` and ``pair_displacement``. The last two
    are the group's internal pairs in condensed upper-triangle order, the
    order ``scipy.spatial.distance.pdist`` uses. ``pair_mean`` is a dense
    (N, N) matrix of the displacement neighborhood's per-pair means (0 for an
    unrealized pair); None reproduces a matrix built without positions.
    """
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
        pairs = [(a, b) for a in range(len(group)) for b in range(a + 1, len(group))]
        groups.append(
            {
                "images": group,
                "seed_pair": (best[1], best[2]),
                "seed_shared": best[0],
                "pair_shared": [int(counts[group[a], group[b]]) for a, b in pairs],
                "pair_displacement": (
                    None
                    if pair_mean is None
                    else [float(pair_mean[group[a], group[b]]) for a, b in pairs]
                ),
            }
        )
        excluded[group] = True


def _condensed_index(images, x, y):
    """Index of the pair ``(x, y)`` in a group's condensed upper triangle,
    written from the documented formula rather than by re-deriving the
    enumeration."""
    a, b = sorted((images.index(x), images.index(y)))
    n = len(images)
    return a * (2 * n - a - 1) // 2 + (b - a - 1)


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
        # The backbone's member geometry (format version 6); the counts ignore
        # the positions, the displacement queries ride on them.
        "member_positions": FILE_POSITIONS.copy(),
        "member_affine_shapes": np.tile(
            np.eye(2, dtype=np.float32), (len(FILE_IMAGES), 1, 1)
        ).copy(),
        "matcher_options": {"d": 10, "alpha": 0.8, "min_size": 2},
        "has_cluster_patches": with_patches,
        "has_two_view_geometries": False,
    }
    if with_patches:
        m = len(FILE_IMAGES)
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
                    # Measured: the refinement's own values.
                    data["member_positions"][k] = [5.0, 5.0]
                if FILE_STATUS[k] != 0:
                    # A shape distinct from the reference's.
                    data["member_affine_shapes"][k, 0, 0] = 1.01
        data.update(
            {
                "reference_members": np.array(reference_members, dtype=np.uint32),
                "member_status": FILE_STATUS,
                "member_zncc": np.ones(m, dtype=np.float32),
                "member_shift_px": np.zeros(m, dtype=np.float32),
                "member_consistency_residual": np.full(m, np.nan, dtype=np.float32),
                "refine_options": {"patch_size": 8.0, "resolution": 15},
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

    def test_from_matches_all_members(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        cov = ClusterCovisibility.from_matches(MatchesFile(path))
        expected = _ref_covisibility(FILE_STARTS, FILE_IMAGES, N_IMAGES)
        npt.assert_array_equal(cov.counts, expected)

    def test_from_matches_takes_a_selection(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        sel = MatchesFile(path).select_clusters(min_span=2)
        cov = ClusterCovisibility.from_matches(sel)
        # Every fixture cluster spans 2 images, so the selection is the file.
        assert cov.num_images == N_IMAGES
        npt.assert_array_equal(
            cov.counts, _ref_covisibility(FILE_STARTS, FILE_IMAGES, N_IMAGES)
        )

    def test_from_matches_default_mask_is_reference_or_kept(self, tmp_path):
        path = tmp_path / "cluster-patches.matches"
        _write_cluster_matches(path, with_patches=True)
        cov = ClusterCovisibility.from_matches(MatchesFile(path))
        accepted = np.isin(FILE_STATUS, (0, 1))
        expected = _ref_covisibility(
            FILE_STARTS, FILE_IMAGES, N_IMAGES, accepted=accepted
        )
        npt.assert_array_equal(cov.counts, expected)
        # The rejected members' votes are actually gone: one (0, 2) vote and
        # cluster 3's whole (1, 3) edge.
        assert cov.counts[0, 2] == 2
        assert cov.counts[1, 3] == 0

    @pytest.mark.parametrize("with_patches", [False, True])
    def test_from_matches_counts_equal_from_arrays(self, tmp_path, with_patches):
        # The file constructor is the array one plus the format's own mask:
        # same arrays in, same counts out, under both mask policies.
        path = tmp_path / f"parity-{with_patches}.matches"
        _write_cluster_matches(path, with_patches=with_patches)
        cov = ClusterCovisibility.from_matches(MatchesFile(path))
        accepted = np.isin(FILE_STATUS, (0, 1)) if with_patches else None
        arrays = ClusterCovisibility.from_arrays(
            FILE_STARTS, FILE_IMAGES, N_IMAGES, member_accepted=accepted
        )
        npt.assert_array_equal(cov.counts, arrays.counts)

    def test_from_matches_positions_enable_displacement_queries(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        cov = ClusterCovisibility.from_matches(MatchesFile(path))
        # The backbone's member positions come along, so the neighborhood
        # exists and the pair answers instead of raising.
        assert cov.pair_stats(0, 1) is not None
        arrays = ClusterCovisibility.from_arrays(
            FILE_STARTS, FILE_IMAGES, N_IMAGES, positions_xy=FILE_POSITIONS
        )
        npt.assert_array_equal(cov.pair_displacement(), arrays.pair_displacement())
        npt.assert_array_equal(
            cov.pair_displacement_counts(), arrays.pair_displacement_counts()
        )
        npt.assert_array_equal(cov.nearest(0, 4), arrays.nearest(0, 4))

    def test_from_matches_rejects_pairwise_backbone(self, tmp_path):
        path = tmp_path / "pairwise.matches"
        _write_pairwise_matches(path)
        with pytest.raises(ValueError, match="no clusters/ section"):
            ClusterCovisibility.from_matches(MatchesFile(path))


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


class TestSeedImageGroups:
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
            got = list(cov.seed_image_groups(group_size, min_shared=min_shared))
            want = _ref_seed_image_groups(
                cov.counts, group_size=group_size, min_shared=min_shared
            )
            assert got == want, (group_size, min_shared)

    def test_defaults_and_iterator_protocol(self):
        # Two disjoint cliques above the default min_shared of 8.
        edges = [(0, 1, 10), (0, 2, 10), (1, 2, 10), (3, 4, 9)]
        starts, images = _edges_to_arrays(5, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 5)
        it = cov.seed_image_groups()
        assert iter(it) is it
        assert next(it) == {
            "images": [0, 1, 2],
            "seed_pair": (0, 1),
            "seed_shared": 10,
            "pair_shared": [10, 10, 10],
            "pair_displacement": None,
        }
        assert next(it) == {
            "images": [3, 4],
            "seed_pair": (3, 4),
            "seed_shared": 9,
            "pair_shared": [9],
            "pair_displacement": None,
        }
        with pytest.raises(StopIteration):
            next(it)

    def test_min_shared_is_keyword_only(self):
        edges = [(0, 1, 10), (2, 3, 9)]
        starts, images = _edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4)
        with pytest.raises(TypeError):
            cov.seed_image_groups(5, 8)

    def test_lazy_prefix_matches_eager_list(self):
        edges = [(0, 1, 10), (0, 2, 10), (1, 2, 10), (3, 4, 9), (5, 6, 8)]
        starts, images = _edges_to_arrays(7, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 7)
        eager = list(cov.seed_image_groups())
        assert len(eager) == 3
        lazy = cov.seed_image_groups()
        assert [next(lazy), next(lazy)] == eager[:2]

    def test_iterators_are_independent(self):
        edges = [(0, 1, 10), (2, 3, 9)]
        starts, images = _edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4)
        a = cov.seed_image_groups()
        b = cov.seed_image_groups()
        assert next(a)["images"] == [0, 1]
        # b holds its own exclusion state
        assert next(b)["images"] == [0, 1]

    def test_star_topology_does_not_form_a_group(self):
        edges = [(0, 1, 10), (0, 2, 10), (0, 3, 10)]
        starts, images = _edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4)
        assert [g["images"] for g in cov.seed_image_groups()] == [[0, 1]]

    def test_seed_pair_is_the_groups_maximum_shared_pair(self):
        # A small weight alphabet forces exact ties, so the invariant is not
        # simply "the first pair happens to be the largest".
        rng = np.random.default_rng(11)
        edges = [
            (i, j, int(w))
            for (i, j), w in zip(
                [(i, j) for i in range(12) for j in range(i + 1, 12)],
                rng.integers(4, 9, size=66),
            )
        ]
        starts, images = _edges_to_arrays(12, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 12)
        counts = cov.counts
        groups = list(cov.seed_image_groups(5, min_shared=5))
        assert len(groups) >= 2
        for g in groups:
            i, j = g["seed_pair"]
            assert i < j
            assert {i, j} <= set(g["images"])
            assert counts[i, j] == g["seed_shared"] >= 5
            within = counts[np.ix_(g["images"], g["images"])]
            assert within.max() == g["seed_shared"]
            # The same invariant through the enriched yield.
            at = _condensed_index(g["images"], i, j)
            assert g["pair_shared"][at] == g["seed_shared"]
            assert max(g["pair_shared"]) == g["seed_shared"]

    def test_pair_shared_is_the_condensed_upper_triangle(self):
        # Six distinct edge weights on four images: any ordering mistake in
        # the condensed triangle shows up as a mismatched entry.
        edges = [(0, 1, 20), (0, 2, 19), (0, 3, 18), (1, 2, 17), (1, 3, 16), (2, 3, 15)]
        starts, images = _edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4)
        groups = list(cov.seed_image_groups(4, min_shared=8))
        assert len(groups) == 1
        g = groups[0]
        assert g["images"] == [0, 1, 2, 3]
        # The documented order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        assert g["pair_shared"] == [20, 19, 18, 17, 16, 15]
        assert all(isinstance(v, int) for v in g["pair_shared"])
        counts = cov.counts
        for a, x in enumerate(g["images"]):
            for y in g["images"][a + 1 :]:
                assert (
                    g["pair_shared"][_condensed_index(g["images"], x, y)]
                    == (counts[x, y])
                ), (x, y)
        # Without positions there is no displacement to report.
        assert g["pair_displacement"] is None

    def test_pair_displacement_parity_on_a_positioned_build(self):
        # Distinct per-edge displacements: each two-member cluster forces its
        # pair, so the neighborhood's exhaustive mean for an edge is exactly
        # that edge's distance.
        edges = [
            (0, 1, 20, 5.0),
            (0, 2, 19, 7.0),
            (0, 3, 18, 9.0),
            (1, 2, 17, 11.0),
            (1, 3, 16, 13.0),
            (2, 3, 15, 3.0),
            (3, 4, 12, 21.0),
        ]
        starts, images, positions = _positioned_edges_to_arrays(5, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 5, positions_xy=positions)
        # Dense per-pair means, straight from the neighborhood serialization.
        arrs = cov.neighborhood_arrays()
        pair_mean = np.zeros((5, 5))
        pair_mean[arrs["i"], arrs["j"]] = arrs["mean_disp"]
        pair_mean[arrs["j"], arrs["i"]] = arrs["mean_disp"]
        for group_size, min_shared in [(5, 8), (4, 8), (3, 4), (2, 1)]:
            got = list(cov.seed_image_groups(group_size, min_shared=min_shared))
            want = _ref_seed_image_groups(
                cov.counts,
                group_size=group_size,
                min_shared=min_shared,
                pair_mean=pair_mean,
            )
            assert got == want, (group_size, min_shared)
        # And each entry is the neighborhood's own pair answer.
        for g in list(cov.seed_image_groups(5, min_shared=8)):
            assert all(isinstance(v, float) for v in g["pair_displacement"])
            for a, x in enumerate(g["images"]):
                for y in g["images"][a + 1 :]:
                    at = _condensed_index(g["images"], x, y)
                    shared, mean = cov.pair_stats(x, y)
                    assert g["pair_displacement"][at] == mean, (x, y)
                    assert g["pair_shared"][at] == shared

    def test_pair_displacement_zero_for_an_unrealized_pair(self):
        # min_shared=0 lets the extension take an image sharing nothing with
        # the group, so the group holds a pair the neighborhood never saw.
        starts, images, positions = _positioned_edges_to_arrays(3, [(0, 1, 5, 4.0)])
        cov = ClusterCovisibility.from_arrays(starts, images, 3, positions_xy=positions)
        (g,) = list(cov.seed_image_groups(3, min_shared=0))
        assert g["images"] == [0, 1, 2]
        assert g["pair_shared"] == [5, 0, 0]
        assert cov.pair_stats(0, 2) is None
        assert g["pair_displacement"] == [4.0, 0.0, 0.0]

    def test_pair_displacement_is_none_without_positions(self):
        edges = [(0, 1, 10), (0, 2, 10), (1, 2, 10)]
        starts, images = _edges_to_arrays(3, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 3)
        for g in cov.seed_image_groups(3, min_shared=8):
            assert g["pair_displacement"] is None
            n = len(g["images"])
            assert len(g["pair_shared"]) == n * (n - 1) // 2

    def test_seed_shared_counts_clusters_in_both_seed_images(self, tmp_path):
        # The identity a consumer relies on: seed_shared is the number of
        # clusters whose accepted (reference/kept) members land in both
        # seed-pair images, on the very data the matrix was built from.
        path = tmp_path / "cluster-patches.matches"
        _write_cluster_matches(path, with_patches=True)
        cov = ClusterCovisibility.from_matches(MatchesFile(path))
        accepted = np.isin(FILE_STATUS, (0, 1))
        groups = list(cov.seed_image_groups(5, min_shared=1))
        assert groups, "fixture must yield a group"
        for g in groups:
            i, j = g["seed_pair"]
            shared = 0
            for c in range(len(FILE_STARTS) - 1):
                lo, hi = int(FILE_STARTS[c]), int(FILE_STARTS[c + 1])
                span = set(FILE_IMAGES[lo:hi][accepted[lo:hi]])
                shared += {i, j} <= span
            assert shared == g["seed_shared"]
        # The fixture's own numbers: (0, 2) is the strongest edge at 2
        # clusters, and image 1 joins on its single shared cluster each way.
        keys = ("images", "seed_pair", "seed_shared")
        assert {k: groups[0][k] for k in keys} == {
            "images": [0, 1, 2],
            "seed_pair": (0, 2),
            "seed_shared": 2,
        }
        # The internal pairs, condensed: (0, 1), (0, 2), (1, 2).
        assert groups[0]["pair_shared"] == [1, 2, 1]


# ── Selection queries (specs/core/features/covisibility-selection.md) ──────────────


def _positioned_edges_to_arrays(num_images, edges):
    """Synthesize positioned cluster arrays: edge (i, j, w, d) becomes w
    two-member clusters with one member in image i at the origin and one in
    image j at distance d. Two-member clusters force the sampled pair, so
    the displacement tables are exact for any seed: mean[i, j] == d and
    counts[i, j] == w."""
    starts = [0]
    images = []
    positions = []
    for i, j, w, d in edges:
        for _ in range(w):
            images.extend([i, j])
            positions.extend([[0.0, 0.0], [d, 0.0]])
            starts.append(len(images))
    return (
        np.array(starts, dtype=np.uint32),
        np.array(images, dtype=np.uint32),
        np.array(positions, dtype=np.float64),
    )


def _chain8():
    """An 8-image chain with geometrically decaying covisibility:
    W[i, j] = 128 >> |i - j|."""
    edges = [(i, j, 128 >> (j - i)) for i in range(8) for j in range(i + 1, 8)]
    starts, images = _edges_to_arrays(8, edges)
    return ClusterCovisibility.from_arrays(starts, images, 8)


class TestPairDisplacement:
    def test_exact_on_forced_two_member_samples(self):
        # Clusters: (0, 1) at distances 5 and 10; (0, 2) at distance 17.
        starts = np.array([0, 2, 4, 6], dtype=np.uint32)
        images = np.array([0, 1, 0, 1, 0, 2], dtype=np.uint32)
        positions = np.array(
            [[0, 0], [3, 4], [0, 0], [6, 8], [0, 0], [8, 15]], dtype=np.float64
        )
        cov = ClusterCovisibility.from_arrays(
            starts, images, 3, positions_xy=positions, seed=42
        )
        mean = cov.pair_displacement()
        count = cov.pair_displacement_counts()
        assert mean.dtype == np.float64 and mean.shape == (3, 3)
        assert count.dtype == np.uint32 and count.shape == (3, 3)
        npt.assert_array_equal(mean, mean.T)
        npt.assert_array_equal(count, count.T)
        npt.assert_array_equal(np.diag(mean), 0.0)
        assert mean[0, 1] == 7.5 and count[0, 1] == 2
        assert mean[0, 2] == 17.0 and count[0, 2] == 1
        assert mean[1, 2] == 0.0 and count[1, 2] == 0  # unsampled pair
        # Positions leave the shared-cluster counts unchanged.
        bare = ClusterCovisibility.from_arrays(starts, images, 3)
        npt.assert_array_equal(cov.counts, bare.counts)

    def test_seeded_determinism(self):
        # A four-member cluster exercises the sampling RNG.
        starts = np.array([0, 4], dtype=np.uint32)
        images = np.array([0, 1, 2, 3], dtype=np.uint32)
        positions = np.arange(8, dtype=np.float64).reshape(4, 2)
        a, b = (
            ClusterCovisibility.from_arrays(
                starts, images, 4, positions_xy=positions, seed=7
            )
            for _ in range(2)
        )
        npt.assert_array_equal(a.pair_displacement(), b.pair_displacement())
        npt.assert_array_equal(
            a.pair_displacement_counts(), b.pair_displacement_counts()
        )
        # All members sit in distinct images, so the sample always lands.
        assert a.pair_displacement_counts().sum() == 2

    def test_raises_without_positions(self):
        cov = ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        with pytest.raises(ValueError, match="without positions_xy"):
            cov.pair_displacement()
        with pytest.raises(ValueError, match="without positions_xy"):
            cov.pair_displacement_counts()

    def test_float32_and_float64_positions_agree(self):
        # float32 is the backbone's own width; a float64 array is cast, which
        # is exact for values that came out of such a file.
        positions = np.arange(len(MEMBER_IMAGES) * 2, dtype=np.float32).reshape(-1, 2)
        built = [
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES, positions_xy=p
            )
            for p in (positions, positions.astype(np.float64))
        ]
        npt.assert_array_equal(*[c.pair_displacement() for c in built])

    def test_wrong_positions_dtype_raises(self):
        positions = np.zeros((len(MEMBER_IMAGES), 2), dtype=np.uint8)
        with pytest.raises(TypeError, match="float32 or float64"):
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES, positions_xy=positions
            )

    def test_wrong_positions_width_raises(self):
        positions = np.zeros((len(MEMBER_IMAGES), 3), dtype=np.float64)
        with pytest.raises(ValueError, match=r"\(M, 2\)"):
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES, positions_xy=positions
            )

    def test_positions_not_parallel_raises(self):
        positions = np.zeros((3, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="parallel"):
            ClusterCovisibility.from_arrays(
                CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES, positions_xy=positions
            )


class TestDisplacementNeighborhood:
    """Sparse displacement-neighborhood substrate queries and serialization
    (see ``specs/core/geometry/pose-verification.md``)."""

    # 4 images. Same-image pairs give exact means: (0, 1) twice at distances
    # 5 and 15 (mean 10); (0, 2) once at 5; (1, 2) once at 25.
    EDGES = [(0, 1, 5.0), (0, 1, 15.0), (0, 2, 5.0), (1, 2, 25.0)]

    def _cov(self):
        starts = [0]
        images, positions = [], []
        for i, j, d in self.EDGES:
            images.extend([i, j])
            positions.extend([[0.0, 0.0], [d, 0.0]])
            starts.append(len(images))
        return ClusterCovisibility.from_arrays(
            np.array(starts, np.uint32),
            np.array(images, np.uint32),
            4,
            positions_xy=np.array(positions, np.float64),
        )

    def test_pair_stats_exact(self):
        cov = self._cov()
        assert cov.pair_stats(0, 1) == (2, 10.0)
        assert cov.pair_stats(1, 0) == (2, 10.0)
        assert cov.pair_stats(0, 2) == (1, 5.0)
        assert cov.pair_stats(1, 2) == (1, 25.0)
        assert cov.pair_stats(0, 3) is None  # unrealized
        assert cov.pair_stats(0, 0) is None  # diagonal

    def test_nearest_and_farthest(self):
        cov = self._cov()
        npt.assert_array_equal(cov.nearest(0, 4), [2, 1])  # 5 < 10
        npt.assert_array_equal(cov.farthest(0, 4), [1, 2])
        npt.assert_array_equal(cov.nearest(1, 1), [0])
        # The shared-count floor drops single-cluster pairs.
        npt.assert_array_equal(cov.nearest(0, 4, min_shared=2), [1])
        npt.assert_array_equal(cov.nearest(3, 4), [])
        out = cov.nearest(0, 4)
        assert out.dtype == np.uint32

    def test_neighborhood_arrays_layout(self):
        arrs = self._cov().neighborhood_arrays()
        assert set(arrs) == {"i", "j", "count", "mean_disp"}
        npt.assert_array_equal(arrs["i"], [0, 0, 1])
        npt.assert_array_equal(arrs["j"], [1, 2, 2])
        npt.assert_array_equal(arrs["count"], [2, 1, 1])
        npt.assert_array_equal(arrs["mean_disp"], [10.0, 5.0, 25.0])
        assert arrs["i"].dtype == np.uint32
        assert arrs["count"].dtype == np.uint32
        assert arrs["mean_disp"].dtype == np.float64

    def test_raises_without_positions(self):
        cov = ClusterCovisibility.from_arrays(CLUSTER_STARTS, MEMBER_IMAGES, N_IMAGES)
        with pytest.raises(ValueError, match="without positions_xy"):
            cov.nearest(0, 4)
        with pytest.raises(ValueError, match="without positions_xy"):
            cov.farthest(0, 4)
        with pytest.raises(ValueError, match="without positions_xy"):
            cov.pair_stats(0, 1)
        with pytest.raises(ValueError, match="without positions_xy"):
            cov.neighborhood_arrays()

    def test_out_of_range_raises(self):
        cov = self._cov()
        with pytest.raises(ValueError, match="out of range"):
            cov.nearest(4, 1)
        with pytest.raises(ValueError, match="out of range"):
            cov.farthest(9, 1)
        with pytest.raises(ValueError, match="out of range"):
            cov.pair_stats(0, 4)


class TestThin:
    def test_band_selection_on_chain(self):
        cov = _chain8()
        # Band [8, 64): adjacent images (64) duplicate, distance-2 (32)
        # stays linked — a stride-2 skeleton.
        kept = cov.thin(64.0)
        assert kept.dtype == np.uint32
        npt.assert_array_equal(kept, [0, 2, 4, 6])
        # Band [16, 128): every image keeps its adjacent link.
        npt.assert_array_equal(cov.thin(128.0), np.arange(8))
        # A tau below every count keeps only the first swept image.
        npt.assert_array_equal(cov.thin(0.5), [0])

    def test_isolation_order_with_positions(self):
        # W: (0,1)=10, (0,2)=10, (1,2)=3; isolations 0:1, 1:1, 2:9 → the
        # sweep starts at the most isolated image 2, keeping {1, 2} where
        # the construction-order fallback keeps only {0}.
        edges = [(0, 1, 10, 1.0), (0, 2, 10, 2.0), (1, 2, 3, 9.0)]
        starts, images, positions = _positioned_edges_to_arrays(3, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 3, positions_xy=positions)
        npt.assert_array_equal(cov.thin(8.0), [1, 2])
        bare = ClusterCovisibility.from_arrays(starts, images, 3)
        npt.assert_array_equal(bare.thin(8.0), [0])

    def test_permutation_invariance(self):
        # Isolations [1, 1, 2, 4]: the global-minimum edge ties its
        # endpoints {0, 1} (ties break by index), so the permutation
        # preserves the index order inside that tie class; the kept set must
        # then relabel exactly at every tau.
        edges = [
            (0, 1, 10, 1.0),
            (0, 2, 10, 2.0),
            (1, 2, 3, 9.0),
            (2, 3, 20, 4.0),
            (1, 3, 6, 7.0),
        ]
        perm = np.array([1, 3, 0, 2], dtype=np.uint32)  # perm[0] < perm[1]
        permuted = [(perm[i], perm[j], w, d) for i, j, w, d in edges]
        starts, images, positions = _positioned_edges_to_arrays(4, edges)
        cov = ClusterCovisibility.from_arrays(starts, images, 4, positions_xy=positions)
        pstarts, pimages, ppositions = _positioned_edges_to_arrays(4, permuted)
        cov_p = ClusterCovisibility.from_arrays(
            pstarts, pimages, 4, positions_xy=ppositions
        )
        for tau in [2.0, 8.0, 16.0, 32.0, 64.0]:
            npt.assert_array_equal(
                cov_p.thin(tau), np.sort(perm[cov.thin(tau)]), err_msg=f"tau={tau}"
            )

    def test_non_finite_tau_raises(self):
        cov = _chain8()
        with pytest.raises(ValueError, match="finite"):
            cov.thin(float("nan"))


class TestThinTo:
    def test_hits_requested_sizes(self):
        cov = _chain8()
        # Reachable sizes over tau in (1, median row peak]: 2, 3, 4 (size 1
        # would need tau <= 1, below the search range).
        for target in range(2, 5):
            kept = cov.thin_to(target)
            assert kept.dtype == np.uint32
            assert len(kept) == target, target
        assert len(cov.thin_to(1)) == 2  # closest reachable size
        # Larger targets saturate at the stride-2 skeleton.
        npt.assert_array_equal(cov.thin_to(8), [0, 2, 4, 6])


class TestReach:
    def _cov(self):
        # 5 images; image 4 shares nothing.
        starts, images = _edges_to_arrays(5, [(0, 1, 8), (1, 2, 7), (0, 3, 9)])
        return ClusterCovisibility.from_arrays(starts, images, 5)

    def test_exact_fractions(self):
        cov = self._cov()
        subset = np.array([0], dtype=np.uint32)
        assert cov.reach(subset) == 3 / 5  # 0 (member), 1, 3 at min_shared=8
        assert cov.reach(np.array([1], dtype=np.uint32)) == 2 / 5
        assert cov.reach(np.array([], dtype=np.uint32)) == 0.0
        # A member counts as reached even with zero covisibility.
        assert cov.reach(np.array([4], dtype=np.uint32)) == 1 / 5
        assert cov.reach(np.arange(5, dtype=np.uint32)) == 1.0

    def test_min_shared_boundary(self):
        cov = self._cov()
        # Exactly at the bar counts; one below does not.
        assert cov.reach(np.array([1], dtype=np.uint32), min_shared=7) == 3 / 5
        assert cov.reach(np.array([0], dtype=np.uint32), min_shared=9) == 2 / 5

    def test_out_of_range_raises(self):
        cov = self._cov()
        with pytest.raises(ValueError, match="out of range"):
            cov.reach(np.array([5], dtype=np.uint32))
