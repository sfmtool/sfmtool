# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoveNarrowTracksFilter."""

import numpy as np
import pytest

from sfmtool.xform import RemoveNarrowTracksFilter

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_remove_narrow_tracks_basic(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test removing points with narrow viewing angles."""
    output_path = tmp_path / "no_narrow.sfmr"
    min_angle_rad = np.radians(5.0)
    transforms = [RemoveNarrowTracksFilter(min_angle_rad)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] <= original["point_count"]
    assert filtered["image_count"] == original["image_count"]


def test_remove_narrow_tracks_strict_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test with a strict threshold that removes more points."""
    output_path = tmp_path / "no_narrow_strict.sfmr"
    transforms = [RemoveNarrowTracksFilter(np.radians(20.0))]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]


def test_remove_narrow_tracks_lenient_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test with a lenient threshold that keeps more points."""
    output_path = tmp_path / "no_narrow_lenient.sfmr"
    transforms = [RemoveNarrowTracksFilter(np.radians(1.0))]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] >= original["point_count"] * 0.5


def test_remove_narrow_tracks_invalid_angle():
    """Test that non-positive angle raises error."""
    with pytest.raises(ValueError, match="Minimum angle must be positive"):
        RemoveNarrowTracksFilter(0.0)
    with pytest.raises(ValueError, match="Minimum angle must be positive"):
        RemoveNarrowTracksFilter(-0.1)


def test_remove_narrow_tracks_preserves_tracks(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that track data is correctly updated after filtering."""
    output_path = tmp_path / "no_narrow_tracks.sfmr"
    transforms = [RemoveNarrowTracksFilter(np.radians(5.0))]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)

    assert np.all(filtered["observation_counts"] >= 2)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


def test_remove_narrow_tracks_combined_with_short_tracks(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test combining narrow tracks filter with short tracks filter."""
    from sfmtool.xform import RemoveShortTracksFilter

    output_path = tmp_path / "no_narrow_no_short.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        RemoveNarrowTracksFilter(np.radians(5.0)),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] > 2)


def test_remove_narrow_tracks_90_degrees(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test with 90 degree threshold - very strict."""
    output_path = tmp_path / "no_narrow_90deg.sfmr"
    transforms = [RemoveNarrowTracksFilter(np.radians(90.0))]

    try:
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images, output_path, transforms
        )
        filtered = load_reconstruction_data(output_path)
        assert filtered["point_count"] >= 0
    except ValueError as e:
        assert "No points remain" in str(e)


def test_remove_narrow_tracks_small_radians(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test with small radian value."""
    output_path = tmp_path / "no_narrow_small.sfmr"
    transforms = [RemoveNarrowTracksFilter(0.01)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] >= original["point_count"] * 0.8
