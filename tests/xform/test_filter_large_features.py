# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoveLargeFeaturesFilter."""

import numpy as np
import pytest

from sfmtool._sfmtool import SfmrReconstruction, read_sift_partial
from sfmtool._sift_file import feature_size
from sfmtool.xform import RemoveLargeFeaturesFilter

from .conftest import apply_transforms_to_file, load_reconstruction_data


def _compute_max_feature_sizes(sfmr_path):
    """Compute the max feature size per point from the reconstruction's SIFT files."""
    recon = SfmrReconstruction.load(sfmr_path)
    track_image_indexes = np.asarray(recon.track_image_indexes)
    track_feature_indexes = np.asarray(recon.track_feature_indexes)
    track_point_ids = np.asarray(recon.track_point_ids)

    metadata = recon.metadata()
    feature_prefix_dir = metadata["workspace"]["contents"]["feature_prefix_dir"]
    from pathlib import Path

    workspace_dir = Path(recon.workspace_dir)
    image_names = recon.image_names

    # Load feature sizes per image
    image_feature_sizes = {}
    for img_idx in np.unique(track_image_indexes):
        img_idx = int(img_idx)
        image_rel = Path(image_names[img_idx])
        sift_path = (
            workspace_dir
            / image_rel.parent
            / feature_prefix_dir
            / f"{image_rel.name}.sift"
        )
        if sift_path.exists():
            mask = track_image_indexes == img_idx
            max_feat_idx = int(track_feature_indexes[mask].max()) + 1
            sift_data = read_sift_partial(sift_path, max_feat_idx)
            image_feature_sizes[img_idx] = feature_size(sift_data["affine_shapes"])

    # Build per-observation sizes, then scatter-max into per-point array
    obs_sizes = np.zeros(len(track_image_indexes), dtype=np.float32)
    for img_idx, sizes in image_feature_sizes.items():
        mask = track_image_indexes == img_idx
        feat_idxs = track_feature_indexes[mask]
        valid = feat_idxs < len(sizes)
        obs_positions = np.flatnonzero(mask)
        obs_sizes[obs_positions[valid]] = sizes[feat_idxs[valid]]

    max_sizes = np.zeros(recon.point_count, dtype=np.float32)
    np.maximum.at(max_sizes, track_point_ids, obs_sizes)
    return max_sizes


def test_remove_large_features_basic(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that large features filter removes some points."""
    # Use a threshold that should remove some but not all points
    max_sizes = _compute_max_feature_sizes(sfmrfile_reconstruction_with_17_images)
    median_size = float(np.median(max_sizes[max_sizes > 0]))

    output_path = tmp_path / "filtered.sfmr"
    transforms = [RemoveLargeFeaturesFilter(median_size)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert filtered["observation_count"] < original["observation_count"]


def test_remove_large_features_verifies_sizes(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that all remaining points have max feature size <= threshold."""
    threshold = 10.0
    output_path = tmp_path / "filtered_verify.sfmr"
    transforms = [RemoveLargeFeaturesFilter(threshold)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    max_sizes = _compute_max_feature_sizes(output_path)
    assert np.all(max_sizes <= threshold)


def test_remove_large_features_lenient(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that a very large threshold keeps all points."""
    output_path = tmp_path / "filtered_lenient.sfmr"
    transforms = [RemoveLargeFeaturesFilter(1e6)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] == original["point_count"]


def test_remove_large_features_tiny_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that a tiny threshold either removes all points or leaves valid ones."""
    output_path = tmp_path / "filtered_tiny.sfmr"
    transforms = [RemoveLargeFeaturesFilter(0.1)]

    try:
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images, output_path, transforms
        )
        load_reconstruction_data(output_path)
        max_sizes = _compute_max_feature_sizes(output_path)
        assert np.all(max_sizes <= 0.1)
    except ValueError as e:
        assert "No points remain" in str(e)


def test_remove_large_features_preserves_images(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that filtering by feature size doesn't remove images."""
    max_sizes = _compute_max_feature_sizes(sfmrfile_reconstruction_with_17_images)
    median_size = float(np.median(max_sizes[max_sizes > 0]))

    output_path = tmp_path / "filtered_images.sfmr"
    transforms = [RemoveLargeFeaturesFilter(median_size)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == original["image_count"]


def test_remove_large_features_preserves_tracks(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that remaining points have valid observation counts."""
    output_path = tmp_path / "filtered_tracks.sfmr"
    transforms = [RemoveLargeFeaturesFilter(20.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)
    assert np.all(filtered["observation_counts"] > 0)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


def test_remove_large_features_invalid_threshold():
    """Test that invalid thresholds raise errors."""
    with pytest.raises(ValueError, match="must be positive"):
        RemoveLargeFeaturesFilter(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        RemoveLargeFeaturesFilter(-1.0)


def test_remove_large_features_description():
    """Test the description method."""
    f = RemoveLargeFeaturesFilter(42.5)
    desc = f.description()
    assert "42.5" in desc
    assert "feature size" in desc.lower()


def test_remove_large_features_combined_with_short_tracks(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test chaining with RemoveShortTracksFilter."""
    from sfmtool.xform import RemoveShortTracksFilter

    output_path = tmp_path / "filtered_combined.sfmr"
    transforms = [
        RemoveShortTracksFilter(2),
        RemoveLargeFeaturesFilter(20.0),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] > 2)
