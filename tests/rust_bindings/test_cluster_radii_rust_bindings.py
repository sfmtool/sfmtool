# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster-radius Rust bindings
(``sfmtool._sfmtool.analysis.cluster_radii`` / ``coarsest_clusters``; see
``specs/core/analysis/source-clusters.md``)."""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.analysis import cluster_radii, coarsest_clusters
from sfmtool._sfmtool.io import MatchesFile, write_matches

# ── Fixture data ──────────────────────────────────────────────────────────

N_IMAGES = 2
PATCH_SIZE = 8.0
REFINE_RADIUS = PATCH_SIZE / 2.0

# 4 clusters of 2 members each. Member shapes are scaled identities, so a
# member's radius is REFINE_RADIUS * scale and a cluster's is its widest
# member's. Clusters 0 and 2 tie at 1.0, which is what the id tiebreak is read
# off; every scale is a power of two, so widening f32 -> f64 is exact and the
# expected radii are exact too.
CLUSTER_STARTS = np.array([0, 2, 4, 6, 8], dtype=np.uint32)
MEMBER_SCALES = np.array([0.5, 1.0, 2.0, 0.25, 1.0, 0.75, 0.5, 0.25], dtype=np.float32)
EXPECTED_RADII = np.array([4.0, 8.0, 4.0, 2.0])

MEMBER_IMAGES = np.array([0, 1] * 4, dtype=np.uint32)
MEMBER_FEATURES = np.arange(len(MEMBER_IMAGES), dtype=np.uint32)
MEMBER_POSITIONS = (
    np.arange(len(MEMBER_IMAGES) * 2, dtype=np.float32).reshape(-1, 2) * 2.0
)
MEMBER_SHAPES = np.einsum(
    "m,ij->mij", MEMBER_SCALES, np.eye(2, dtype=np.float32)
).astype(np.float32)


def _write_cluster_matches(path, with_patches):
    """Write a minimal cluster-backbone .matches file."""
    m = len(MEMBER_IMAGES)
    n_cl = len(CLUSTER_STARTS) - 1
    data = {
        "metadata": {
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
            "timestamp": "2026-09-05T10:00:00Z",
            "image_count": N_IMAGES,
            "cluster_count": n_cl,
            "cluster_member_count": m,
            "has_two_view_geometries": False,
            "has_clusters": True,
            "has_cluster_patches": with_patches,
        },
        "image_names": [f"frames/frame_{i:03d}.jpg" for i in range(N_IMAGES)],
        "feature_tool_hashes": [b"\x00" * 16] * N_IMAGES,
        "sift_content_hashes": [b"\x01" * 16] * N_IMAGES,
        "feature_counts": np.full(N_IMAGES, m, dtype=np.uint32),
        "image_dims": np.full((N_IMAGES, 2), [640, 480], dtype=np.uint32),
        "has_clusters": True,
        "cluster_starts": CLUSTER_STARTS,
        "member_images": MEMBER_IMAGES,
        "member_features": MEMBER_FEATURES,
        "member_positions": MEMBER_POSITIONS.copy(),
        "member_affine_shapes": MEMBER_SHAPES.copy(),
        "matcher_options": {"d": 10, "alpha": 0.8, "min_size": 2},
        "has_cluster_patches": with_patches,
        "has_two_view_geometries": False,
    }
    if with_patches:
        data.update(
            {
                "reference_members": CLUSTER_STARTS[:-1].astype(np.uint32),
                "member_status": np.zeros(m, dtype=np.uint8),
                "member_zncc": np.ones(m, dtype=np.float32),
                "member_shift_px": np.zeros(m, dtype=np.float32),
                "member_consistency_residual": np.full(m, np.nan, dtype=np.float32),
                "refine_options": {"patch_size": PATCH_SIZE, "resolution": 15},
            }
        )
    write_matches(path, data)


@pytest.fixture
def cluster_file(tmp_path):
    path = tmp_path / "cluster-patches.matches"
    _write_cluster_matches(path, with_patches=True)
    return MatchesFile(path)


# ── cluster_radii ─────────────────────────────────────────────────────────


class TestClusterRadii:
    def test_array_form_reads_the_widest_member(self):
        radii = cluster_radii(CLUSTER_STARTS, MEMBER_SHAPES, REFINE_RADIUS)
        assert radii.dtype == np.float64
        npt.assert_array_equal(radii, EXPECTED_RADII)

    def test_object_form_equals_array_form(self, cluster_file):
        npt.assert_array_equal(
            cluster_radii(cluster_file),
            cluster_radii(CLUSTER_STARTS, MEMBER_SHAPES, REFINE_RADIUS),
        )

    def test_a_selection_is_an_object_form_input(self, cluster_file):
        sel = cluster_file.select_clusters(min_span=2)
        # Every fixture cluster spans both images, so the selection is the file.
        npt.assert_array_equal(cluster_radii(sel), EXPECTED_RADII)

    def test_float64_shapes_are_accepted_and_exact(self):
        npt.assert_array_equal(
            cluster_radii(
                CLUSTER_STARTS, MEMBER_SHAPES.astype(np.float64), REFINE_RADIUS
            ),
            EXPECTED_RADII,
        )

    def test_the_two_forms_do_not_mix(self, cluster_file):
        with pytest.raises(ValueError, match="takes no shape arrays"):
            cluster_radii(cluster_file, MEMBER_SHAPES, REFINE_RADIUS)
        with pytest.raises(ValueError, match="array form takes"):
            cluster_radii(CLUSTER_STARTS)

    def test_a_file_without_cluster_patches_is_refused(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        with pytest.raises(ValueError, match="no cluster_patches/ section"):
            cluster_radii(MatchesFile(path))

    def test_a_short_cluster_starts_is_refused(self):
        with pytest.raises(ValueError, match="close at the member count"):
            cluster_radii(
                np.array([0, 2], dtype=np.uint32), MEMBER_SHAPES, REFINE_RADIUS
            )


# ── coarsest_clusters ─────────────────────────────────────────────────────


class TestCoarsestClusters:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, []),
            (1, [1]),
            # Clusters 0 and 2 tie at radius 4.0; the smaller id is taken.
            (2, [0, 1]),
            (3, [0, 1, 2]),
            (99, [0, 1, 2, 3]),
        ],
    )
    def test_radius_descending_with_ties_by_id_returned_ascending(self, n, expected):
        got = coarsest_clusters(CLUSTER_STARTS, n, MEMBER_SHAPES, REFINE_RADIUS)
        assert got.dtype == np.uint32
        npt.assert_array_equal(got, np.array(expected, dtype=np.uint32))

    def test_object_form_equals_array_form(self, cluster_file):
        for n in (1, 2, 3, 4):
            npt.assert_array_equal(
                coarsest_clusters(cluster_file, n),
                coarsest_clusters(CLUSTER_STARTS, n, MEMBER_SHAPES, REFINE_RADIUS),
            )

    def test_the_cut_matches_the_ordering_of_the_radii(self, cluster_file):
        radii = cluster_radii(cluster_file)
        # The hand ordering the binding replaces: descending radius, stable in
        # id, then the ids sorted ascending.
        expected = np.sort(np.argsort(-radii, kind="stable")[:3])
        npt.assert_array_equal(coarsest_clusters(cluster_file, 3), expected)

    def test_a_file_without_cluster_patches_is_refused(self, tmp_path):
        path = tmp_path / "clusters.matches"
        _write_cluster_matches(path, with_patches=False)
        with pytest.raises(ValueError, match="no cluster_patches/ section"):
            coarsest_clusters(MatchesFile(path), 2)
