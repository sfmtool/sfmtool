# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `write_clusters_matches` binding: a clusters file written from
its arrays reads back with the counts and flags the writer derives, and arrays
that disagree are refused."""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool.fileio import read_matches, verify_matches, write_clusters_matches

WORKSPACE = {
    "absolute_path": "/tmp/workspace",
    "relative_path": "..",
    "contents": {
        "feature_tool": "sfmtool",
        "feature_type": "sift",
        "feature_options": {},
        "feature_prefix_dir": "features/sift-sfmtool-abc123",
    },
}


def _clusters_kwargs() -> dict:
    """3 images, 2 clusters, 5 members: cluster 0 on images (0, 1, 2), cluster
    1 on images (0, 2)."""
    return {
        "matching_method": "cluster",
        "matching_tool": "sfmtool",
        "matching_tool_version": "0.3",
        "matching_options": {"mode": "background-floor", "d": 10},
        "workspace": WORKSPACE,
        "timestamp": "2026-10-31T10:00:00Z",
        "image_names": [f"frames/frame_{j:03d}.jpg" for j in range(3)],
        "feature_tool_hashes": [b"\x00" * 16] * 3,
        "sift_content_hashes": [b"\x01" * 16] * 3,
        "feature_counts": np.array([100, 150, 200], dtype=np.uint32),
        "image_dims": np.array([[640, 480], [640, 480], [1024, 768]], dtype=np.uint32),
        "cluster_starts": np.array([0, 3, 5], dtype=np.uint32),
        "member_images": np.array([0, 1, 2, 0, 2], dtype=np.uint32),
        "member_features": np.array([0, 1, 2, 5, 10], dtype=np.uint32),
        "member_positions": np.array(
            [[10.5 + k, 20.25 + k] for k in range(5)], dtype=np.float32
        ),
        "member_affine_shapes": np.array(
            [[[2.0 + k, 0.5], [0.25, 1.5 + k]] for k in range(5)], dtype=np.float32
        ),
        "matcher_options": {"d": 10, "alpha": 0.8, "min_size": 2},
    }


def test_round_trip(tmp_path):
    kwargs = _clusters_kwargs()
    path = tmp_path / "clusters.matches"
    write_clusters_matches(path, **kwargs)

    data = read_matches(path)
    meta = data["metadata"]
    assert meta["image_count"] == 3
    assert meta["cluster_count"] == 2
    assert meta["cluster_member_count"] == 5
    assert meta["has_clusters"] and not meta["has_cluster_patches"]
    assert not meta["has_two_view_geometries"]
    assert "image_pair_count" not in meta
    assert meta["matching_method"] == "cluster"
    assert meta["matching_options"] == kwargs["matching_options"]
    assert meta["workspace"] == WORKSPACE
    assert not data["has_cluster_patches"]
    assert not data["has_two_view_geometries"]

    assert data["image_names"] == kwargs["image_names"]
    npt.assert_array_equal(data["feature_counts"], kwargs["feature_counts"])
    npt.assert_array_equal(data["image_dims"], kwargs["image_dims"])
    for key in ("cluster_starts", "member_images", "member_features"):
        npt.assert_array_equal(data[key], kwargs[key])
    for key in ("member_positions", "member_affine_shapes"):
        npt.assert_array_equal(data[key].view(np.uint32), kwargs[key].view(np.uint32))
    assert data["matcher_options"] == kwargs["matcher_options"]

    valid, errors = verify_matches(path)
    assert valid, errors


def test_member_feature_past_feature_count_is_refused(tmp_path):
    kwargs = _clusters_kwargs()
    kwargs["member_features"][1] = 150  # feature_counts[1] = 150
    path = tmp_path / "clusters.matches"
    with pytest.raises(IOError, match=r"member_features\[1\] = 150 >= feature_counts"):
        write_clusters_matches(path, **kwargs)
    assert not path.exists()


def test_member_image_past_image_list_is_refused(tmp_path):
    kwargs = _clusters_kwargs()
    kwargs["member_images"][0] = 3
    with pytest.raises(IOError, match=r"member_images\[0\] = 3 >= image_count 3"):
        write_clusters_matches(tmp_path / "clusters.matches", **kwargs)


def test_arguments_are_keyword_only(tmp_path):
    with pytest.raises(TypeError):
        write_clusters_matches(tmp_path / "clusters.matches", "cluster")
