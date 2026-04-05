# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterByReprojectionErrorTransform."""

import numpy as np
import pytest

from sfmtool.xform import FilterByReprojectionErrorTransform

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_filter_by_reprojection_error_basic(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "filtered_reproj.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=2.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] <= original["point_count"]
    assert np.all(filtered["errors"] <= 2.0)


def test_filter_by_reprojection_error_strict(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "filtered_reproj_strict.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=0.5)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] <= original["point_count"]
    assert np.all(filtered["errors"] <= 0.5)


def test_filter_by_reprojection_error_lenient(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "filtered_reproj_lenient.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=100.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] == original["point_count"]


def test_filter_by_reprojection_error_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold must be positive"):
        FilterByReprojectionErrorTransform(threshold=0.0)
    with pytest.raises(ValueError, match="Threshold must be positive"):
        FilterByReprojectionErrorTransform(threshold=-1.0)


def test_filter_by_reprojection_error_preserves_images(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "filtered_reproj_images.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=2.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == original["image_count"]


def test_filter_by_reprojection_error_preserves_tracks(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "filtered_reproj_tracks.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=2.0)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)

    assert np.all(filtered["observation_counts"] > 0)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


def test_filter_by_reprojection_error_combined(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    from sfmtool.xform import RemoveShortTracksFilter

    output_path = tmp_path / "filtered_combined.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        FilterByReprojectionErrorTransform(threshold=2.0),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] > 2)
    assert np.all(filtered["errors"] <= 2.0)


def test_filter_by_reprojection_error_tiny_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "filtered_reproj_tiny.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=0.1)]

    try:
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images, output_path, transforms
        )
        filtered = load_reconstruction_data(output_path)
        assert np.all(filtered["errors"] <= 0.1)
    except ValueError as e:
        assert "No points remain" in str(e)
