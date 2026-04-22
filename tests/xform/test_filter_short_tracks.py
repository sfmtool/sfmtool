# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoveShortTracksFilter."""

import numpy as np
import pytest

from sfmtool.xform import RemoveShortTracksFilter

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_remove_short_tracks_filter(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that short tracks filter works."""
    output_path = tmp_path / "filtered.sfmr"
    transforms = [RemoveShortTracksFilter(2)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert filtered["observation_count"] < original["observation_count"]
    assert np.all(filtered["observation_counts"] > 2)


def test_invalid_track_size_raises_error():
    """Test that invalid track size raises error."""
    with pytest.raises(ValueError, match="Track size must be >= 1"):
        RemoveShortTracksFilter(0)
    with pytest.raises(ValueError, match="Track size must be >= 1"):
        RemoveShortTracksFilter(-1)


def test_remove_short_tracks_length_one(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """`--remove-short-tracks 1` removes length-1 tracks. Valid
    reconstructions don't normally contain any, but the call must not
    reject the parameter — filters like --include-range can leave tracks
    stranded with a single surviving observation."""
    output_path = tmp_path / "filtered_one.sfmr"
    transforms = [RemoveShortTracksFilter(1)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)
    assert np.all(filtered["observation_counts"] > 1)


def test_remove_short_tracks_min_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test with minimum valid threshold of 2."""
    output_path = tmp_path / "filtered_min.sfmr"
    transforms = [RemoveShortTracksFilter(2)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)
    assert np.all(filtered["observation_counts"] > 2)


def test_remove_short_tracks_high_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test with a high threshold that removes most points."""
    output_path = tmp_path / "filtered_high.sfmr"
    transforms = [RemoveShortTracksFilter(5)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] > 5)


def test_remove_short_tracks_preserves_images(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that filtering tracks doesn't remove images."""
    output_path = tmp_path / "filtered_images.sfmr"
    transforms = [RemoveShortTracksFilter(2)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == original["image_count"]


def test_remove_short_tracks_description():
    """Test the description method of RemoveShortTracksFilter."""
    filter_transform = RemoveShortTracksFilter(3)
    desc = filter_transform.description()
    assert "track" in desc.lower() or "3" in desc
