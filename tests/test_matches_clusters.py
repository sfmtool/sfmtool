# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for cluster-bearing `.matches` files through the
`sfmtool._sfmtool.io` bindings (format version 3: `clusters/` +
`cluster_patches/` sections, pairs-or-clusters backbone)."""

import numpy as np
import numpy.testing as npt

from sfmtool._sfmtool.io import read_matches, verify_matches, write_matches


def _base_metadata() -> dict:
    return {
        "version": 3,
        "matching_method": "cluster",
        "matching_tool": "sfmtool",
        "matching_tool_version": "0.2",
        "matching_options": {"d": 8, "alpha": 1.2},
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
        "timestamp": "2026-07-09T10:00:00Z",
        "image_count": 3,
        "has_two_view_geometries": False,
        "has_clusters": False,
        "has_cluster_patches": False,
    }


def _images_section() -> dict:
    return {
        "image_names": [
            "frames/frame_000.jpg",
            "frames/frame_001.jpg",
            "frames/frame_002.jpg",
        ],
        "feature_tool_hashes": [b"\x00" * 16] * 3,
        "sift_content_hashes": [b"\x01" * 16] * 3,
        "feature_counts": np.array([100, 150, 200], dtype=np.uint32),
    }


def _cluster_data() -> dict:
    """3 images, 2 clusters, 5 members. Cluster 0 = members 0..3 in images
    (0, 1, 2); cluster 1 = members 3..5 in images (0, 2)."""
    metadata = _base_metadata()
    metadata["has_clusters"] = True
    metadata["cluster_count"] = 2
    metadata["cluster_member_count"] = 5
    return {
        "metadata": metadata,
        **_images_section(),
        "has_clusters": True,
        "cluster_starts": np.array([0, 3, 5], dtype=np.uint32),
        "member_images": np.array([0, 1, 2, 0, 2], dtype=np.uint32),
        "member_features": np.array([0, 1, 2, 5, 10], dtype=np.uint32),
        "matcher_options": {"d": 8, "alpha": 1.2, "min_size": 2, "preset": "default"},
        "has_cluster_patches": False,
        "has_two_view_geometries": False,
    }


def _cluster_patch_data() -> dict:
    """Cluster data enriched with cluster_patches: cluster 0 refined
    (reference + kept + rejected), cluster 1 unrefinable."""
    data = _cluster_data()
    data["metadata"]["has_cluster_patches"] = True
    data["has_cluster_patches"] = True
    affines = np.zeros((5, 2, 3), dtype=np.float64)
    affines[0] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # reference: identity | 0
    affines[1] = [[1.1, -0.05, 42.5], [0.03, 0.95, -17.25]]  # kept
    affines[2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # rejected, affine retained
    data.update(
        {
            "reference_members": np.array(
                [0, np.iinfo(np.uint32).max], dtype=np.uint32
            ),
            "member_status": np.array([0, 1, 2, 5, 5], dtype=np.uint8),
            "member_affines": affines,
            "member_zncc": np.array(
                [1.0, 0.93, 0.41, np.nan, np.nan], dtype=np.float32
            ),
            "member_shift_px": np.array(
                [0.0, 1.25, 0.8, np.nan, np.nan], dtype=np.float32
            ),
            "member_consistency_residual": np.array(
                [0.02, 0.05, np.nan, np.nan, np.nan], dtype=np.float32
            ),
            "refine_options": {
                "radius": 4.0,
                "resolution": 15,
                "min_zncc": 0.85,
                "max_shift_px": 3.0,
            },
        }
    )
    return data


def _pairwise_data() -> dict:
    """3 images, 2 pairs, 5 matches (the pre-version-3 pairwise shape)."""
    metadata = _base_metadata()
    metadata["matching_method"] = "sequential"
    metadata["image_pair_count"] = 2
    metadata["match_count"] = 5
    return {
        "metadata": metadata,
        **_images_section(),
        "image_index_pairs": np.array([[0, 1], [0, 2]], dtype=np.uint32),
        "match_counts": np.array([3, 2], dtype=np.uint32),
        "match_feature_indexes": np.array(
            [[0, 0], [1, 1], [2, 3], [5, 10], [10, 50]], dtype=np.uint32
        ),
        "match_descriptor_distances": np.array(
            [100.0, 120.0, 90.0, 200.0, 180.0], dtype=np.float32
        ),
        "has_two_view_geometries": False,
    }


def test_clusters_round_trip(tmp_path):
    path = tmp_path / "clusters.matches"
    data = _cluster_data()
    write_matches(path, data)

    valid, errors = verify_matches(path)
    assert valid, f"verification failed: {errors}"

    loaded = read_matches(path)
    assert loaded["metadata"]["version"] == 3
    assert loaded["metadata"]["has_clusters"] is True
    assert loaded["metadata"]["has_cluster_patches"] is False
    assert loaded["metadata"]["cluster_count"] == 2
    assert loaded["metadata"]["cluster_member_count"] == 5
    assert "image_pair_count" not in loaded["metadata"]
    assert "match_count" not in loaded["metadata"]

    assert loaded["has_clusters"] is True
    assert loaded["has_cluster_patches"] is False
    assert loaded["has_two_view_geometries"] is False
    npt.assert_array_equal(loaded["cluster_starts"], data["cluster_starts"])
    npt.assert_array_equal(loaded["member_images"], data["member_images"])
    npt.assert_array_equal(loaded["member_features"], data["member_features"])
    assert loaded["matcher_options"] == data["matcher_options"]

    # Cluster files carry no pairwise keys.
    for key in (
        "image_index_pairs",
        "match_counts",
        "match_feature_indexes",
        "match_descriptor_distances",
    ):
        assert key not in loaded

    # Content hash: clusters digest present, pairs digest absent.
    assert "clusters_xxh128" in loaded["content_hash"]
    assert "image_pairs_xxh128" not in loaded["content_hash"]


def test_cluster_patches_round_trip(tmp_path):
    path = tmp_path / "cluster-patches.matches"
    data = _cluster_patch_data()
    write_matches(path, data)

    valid, errors = verify_matches(path)
    assert valid, f"verification failed: {errors}"

    loaded = read_matches(path)
    assert loaded["metadata"]["has_cluster_patches"] is True
    assert loaded["has_cluster_patches"] is True

    npt.assert_array_equal(loaded["reference_members"], data["reference_members"])
    npt.assert_array_equal(loaded["member_status"], data["member_status"])
    npt.assert_array_equal(loaded["member_affines"], data["member_affines"])
    # assert_array_equal treats NaN positions as equal.
    npt.assert_array_equal(loaded["member_zncc"], data["member_zncc"])
    npt.assert_array_equal(loaded["member_shift_px"], data["member_shift_px"])
    npt.assert_array_equal(
        loaded["member_consistency_residual"], data["member_consistency_residual"]
    )
    assert loaded["member_consistency_residual"].dtype == np.float32
    assert loaded["refine_options"] == data["refine_options"]

    assert loaded["member_affines"].dtype == np.float64
    assert loaded["member_zncc"].dtype == np.float32
    assert loaded["member_status"].dtype == np.uint8
    assert loaded["reference_members"].dtype == np.uint32
    assert np.isnan(loaded["member_zncc"][3:]).all()

    assert "cluster_patches_xxh128" in loaded["content_hash"]

    # Lossless second-generation round trip: write the loaded dict back out.
    path2 = tmp_path / "rewritten.matches"
    write_matches(path2, loaded)
    reloaded = read_matches(path2)
    npt.assert_array_equal(reloaded["member_zncc"], loaded["member_zncc"])
    npt.assert_array_equal(reloaded["member_affines"], loaded["member_affines"])
    npt.assert_array_equal(reloaded["cluster_starts"], loaded["cluster_starts"])


def test_pairwise_round_trip_regression(tmp_path):
    """A pairwise dict (no cluster keys at all) still writes and reads with
    the pre-version-3 key set, plus the new has_* flags."""
    path = tmp_path / "pairwise.matches"
    data = _pairwise_data()
    write_matches(path, data)

    valid, errors = verify_matches(path)
    assert valid, f"verification failed: {errors}"

    loaded = read_matches(path)
    assert loaded["metadata"]["image_pair_count"] == 2
    assert loaded["metadata"]["match_count"] == 5
    assert "cluster_count" not in loaded["metadata"]
    assert loaded["has_clusters"] is False
    assert loaded["has_cluster_patches"] is False
    assert loaded["has_two_view_geometries"] is False

    npt.assert_array_equal(loaded["image_index_pairs"], data["image_index_pairs"])
    npt.assert_array_equal(loaded["match_counts"], data["match_counts"])
    npt.assert_array_equal(
        loaded["match_feature_indexes"], data["match_feature_indexes"]
    )
    npt.assert_array_equal(
        loaded["match_descriptor_distances"], data["match_descriptor_distances"]
    )

    for key in ("cluster_starts", "member_images", "member_features"):
        assert key not in loaded

    assert "image_pairs_xxh128" in loaded["content_hash"]
    assert "clusters_xxh128" not in loaded["content_hash"]


def test_write_rejects_cluster_file_with_stale_pair_counts(tmp_path):
    import pytest

    data = _cluster_data()
    data["metadata"]["image_pair_count"] = 2
    data["metadata"]["match_count"] = 5
    with pytest.raises(OSError, match="must not set metadata.image_pair_count"):
        write_matches(tmp_path / "bad.matches", data)
