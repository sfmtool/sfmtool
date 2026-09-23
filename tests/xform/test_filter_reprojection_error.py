# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterByReprojectionErrorTransform."""

import numpy as np
import pytest

from sfmtool.xform import FilterByReprojectionErrorTransform

from .conftest import (
    apply_transforms_to_file,
    inject_reprojection_outliers,
    load_reconstruction_data,
)

# Points moved off their tracks by ``inject_reprojection_outliers``. Every
# other point of the fixture reprojects within 2 px, so a 2 px threshold
# separates the two populations exactly.
OUTLIER_COUNT = 25


def _with_outliers(seoul_bull_workspace, tmp_path):
    """The fixture with injected outliers, its data, and the outlier indexes.

    Asserts the premise the 2 px tests rest on: the injected points are exactly
    the ones above 2 px.
    """
    input_path = tmp_path / "with_outliers.sfmr"
    outliers = inject_reprojection_outliers(
        seoul_bull_workspace, input_path, OUTLIER_COUNT
    )
    original = load_reconstruction_data(input_path)
    np.testing.assert_array_equal(np.flatnonzero(original["errors"] > 2.0), outliers)
    return input_path, original, outliers


def test_filter_by_reprojection_error_basic(seoul_bull_workspace, tmp_path):
    """The filter removes exactly the points above the threshold, keeping the rest in order."""
    input_path, original, outliers = _with_outliers(seoul_bull_workspace, tmp_path)
    output_path = tmp_path / "filtered_reproj.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=2.0)]

    apply_transforms_to_file(input_path, output_path, transforms)

    filtered = load_reconstruction_data(output_path)

    keep = np.ones(original["point_count"], dtype=bool)
    keep[outliers] = False
    assert filtered["point_count"] == original["point_count"] - OUTLIER_COUNT
    np.testing.assert_array_equal(filtered["positions"], original["positions"][keep])
    assert np.all(filtered["errors"] <= 2.0)


def test_filter_by_reprojection_error_strict(seoul_bull_sfmr_only, tmp_path):
    output_path = tmp_path / "filtered_reproj_strict.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=0.5)]

    apply_transforms_to_file(seoul_bull_sfmr_only, output_path, transforms)

    original = load_reconstruction_data(seoul_bull_sfmr_only)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] <= original["point_count"]
    assert np.all(filtered["errors"] <= 0.5)


def test_filter_by_reprojection_error_lenient(seoul_bull_sfmr_only, tmp_path):
    output_path = tmp_path / "filtered_reproj_lenient.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=100.0)]

    apply_transforms_to_file(seoul_bull_sfmr_only, output_path, transforms)

    original = load_reconstruction_data(seoul_bull_sfmr_only)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] == original["point_count"]


def test_filter_by_reprojection_error_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold must be positive"):
        FilterByReprojectionErrorTransform(threshold=0.0)
    with pytest.raises(ValueError, match="Threshold must be positive"):
        FilterByReprojectionErrorTransform(threshold=-1.0)


def test_filter_by_reprojection_error_preserves_images(seoul_bull_workspace, tmp_path):
    input_path, original, _ = _with_outliers(seoul_bull_workspace, tmp_path)
    output_path = tmp_path / "filtered_reproj_images.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=2.0)]

    apply_transforms_to_file(input_path, output_path, transforms)

    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] == original["point_count"] - OUTLIER_COUNT
    assert filtered["image_count"] == original["image_count"]


def test_filter_by_reprojection_error_preserves_tracks(seoul_bull_workspace, tmp_path):
    input_path, original, outliers = _with_outliers(seoul_bull_workspace, tmp_path)
    output_path = tmp_path / "filtered_reproj_tracks.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=2.0)]

    apply_transforms_to_file(input_path, output_path, transforms)

    filtered = load_reconstruction_data(output_path)

    # The removed points take exactly their own observations with them.
    removed_observations = int(np.sum(original["observation_counts"][outliers]))
    assert (
        filtered["observation_count"]
        == original["observation_count"] - removed_observations
    )
    assert np.all(filtered["observation_counts"] > 0)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


def test_filter_by_reprojection_error_combined(seoul_bull_workspace, tmp_path):
    from sfmtool.xform import RemoveShortTracksFilter

    input_path, original, outliers = _with_outliers(seoul_bull_workspace, tmp_path)
    output_path = tmp_path / "filtered_combined.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        FilterByReprojectionErrorTransform(threshold=2.0),
    ]

    apply_transforms_to_file(input_path, output_path, transforms)

    filtered = load_reconstruction_data(output_path)

    # Survivors are the long tracks that are not outliers; both filters bite.
    expected_keep = original["observation_counts"] > 2
    assert np.any(expected_keep[outliers])
    expected_keep[outliers] = False
    assert filtered["point_count"] == int(np.sum(expected_keep))
    np.testing.assert_array_equal(
        filtered["positions"], original["positions"][expected_keep]
    )
    assert np.all(filtered["observation_counts"] > 2)
    assert np.all(filtered["errors"] <= 2.0)


def test_filter_by_reprojection_error_tiny_threshold(seoul_bull_sfmr_only, tmp_path):
    output_path = tmp_path / "filtered_reproj_tiny.sfmr"
    transforms = [FilterByReprojectionErrorTransform(threshold=0.1)]

    try:
        apply_transforms_to_file(seoul_bull_sfmr_only, output_path, transforms)
        filtered = load_reconstruction_data(output_path)
        assert np.all(filtered["errors"] <= 0.1)
    except ValueError as e:
        assert "No points remain" in str(e)
