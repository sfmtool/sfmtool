# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BundleAdjustTransform."""

import numpy as np

from sfmtool._sfmtool import RotQuaternion
from sfmtool.xform import (
    BundleAdjustTransform,
    RemoveShortTracksFilter,
)

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_bundle_adjust_transform(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that bundle adjustment works."""
    output_path = tmp_path / "bundle_adjusted.sfmr"

    transforms = [BundleAdjustTransform()]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    assert adjusted["point_count"] == original["point_count"]
    assert len(adjusted["positions"]) > 0
    assert len(adjusted["quaternions_wxyz"]) > 0


def test_bundle_adjust_with_filter(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test bundle adjustment combined with filtering."""
    output_path = tmp_path / "filtered_and_adjusted.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        BundleAdjustTransform(),
    ]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result_data = load_reconstruction_data(output_path)

    assert result_data["point_count"] < original["point_count"]
    assert np.all(result_data["observation_counts"] > 2)


def test_bundle_adjust_preserves_image_count(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves the number of images."""
    output_path = tmp_path / "ba_images.sfmr"

    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    assert adjusted["image_count"] == original["image_count"]


def test_bundle_adjust_preserves_observation_count(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves the observation count."""
    output_path = tmp_path / "ba_observations.sfmr"

    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    assert adjusted["observation_count"] == original["observation_count"]


def test_bundle_adjust_quaternion_consistency(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves quaternion ordering (xyzw->wxyz conversion).

    Regression test: pycolmap returns quaternions in xyzw order, but our
    storage format uses wxyz. If the conversion is wrong, camera centers
    computed from the BA result will be wildly different from the original.
    """
    output_path = tmp_path / "ba_quat_check.sfmr"
    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    # Compute camera centers from quaternions and translations
    orig_centers = []
    for i in range(original["image_count"]):
        q = RotQuaternion.from_wxyz_array(original["quaternions_wxyz"][i])
        r = q.to_rotation_matrix()
        t = original["translations"][i]
        orig_centers.append(-r.T @ t)

    adj_centers = []
    for i in range(adjusted["image_count"]):
        q = RotQuaternion.from_wxyz_array(adjusted["quaternions_wxyz"][i])
        r = q.to_rotation_matrix()
        t = adjusted["translations"][i]
        adj_centers.append(-r.T @ t)

    orig_centers = np.array(orig_centers)
    adj_centers = np.array(adj_centers)

    scene_extent = np.ptp(orig_centers, axis=0).max()
    center_diffs = np.linalg.norm(adj_centers - orig_centers, axis=1)
    max_drift = center_diffs.max()

    assert max_drift < 0.1 * scene_extent, (
        f"BA moved cameras too far: max drift {max_drift:.4f} vs "
        f"scene extent {scene_extent:.4f} (ratio {max_drift / scene_extent:.2f}). "
        f"This likely indicates a quaternion ordering bug (xyzw vs wxyz)."
    )


def test_bundle_adjust_description():
    """Test the description method."""
    ba = BundleAdjustTransform()
    desc = ba.description()

    assert "bundle" in desc.lower() or "adjust" in desc.lower() or "BA" in desc
