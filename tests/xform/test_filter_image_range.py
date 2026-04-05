# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for IncludeRangeFilter and ExcludeRangeFilter."""

import numpy as np
import pytest
from openjd.model import IntRangeExpr

from sfmtool._filenames import number_from_filename
from sfmtool.xform import ExcludeRangeFilter, IncludeRangeFilter
from sfmtool.xform._filter_by_image_range import _filter_rig_frame_data

from .conftest import apply_transforms_to_file, load_reconstruction_data


# =============================================================================
# IncludeRangeFilter Tests
# =============================================================================


def test_include_range_filter(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that include range filter keeps only images in the specified range."""
    output_path = tmp_path / "include_range.sfmr"

    # Keep only images 1-5 (out of 1-17)
    range_expr = IntRangeExpr.from_str("1-5")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 5

    for name in filtered["image_names"]:
        num = number_from_filename(name)
        assert num is not None
        assert 1 <= num <= 5

    assert filtered["point_count"] <= original["point_count"]


def test_include_range_filter_with_comma_separated(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test include range filter with comma-separated values."""
    output_path = tmp_path / "include_range_comma.sfmr"

    range_expr = IntRangeExpr.from_str("1,3,5,7")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 4

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert kept_numbers == {1, 3, 5, 7}


def test_include_range_filter_mixed_format(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test include range filter with mixed ranges and individual values."""
    output_path = tmp_path / "include_range_mixed.sfmr"

    range_expr = IntRangeExpr.from_str("1-3,10,15-17")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 7

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert kept_numbers == {1, 2, 3, 10, 15, 16, 17}


def test_include_range_removes_orphaned_points(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that filtering images also removes 3D points with no remaining observations."""
    output_path = tmp_path / "include_range_orphaned.sfmr"

    range_expr = IntRangeExpr.from_str("1-3")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] >= 1)


def test_include_range_no_matching_images_raises_error(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that include range filter raises error when no images match."""
    output_path = tmp_path / "no_match.sfmr"

    range_expr = IntRangeExpr.from_str("100-200")
    transforms = [IncludeRangeFilter(range_expr)]

    with pytest.raises(ValueError, match="No images remain"):
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images, output_path, transforms
        )


# =============================================================================
# ExcludeRangeFilter Tests
# =============================================================================


def test_exclude_range_filter(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that exclude range filter removes images in the specified range."""
    output_path = tmp_path / "exclude_range.sfmr"

    range_expr = IntRangeExpr.from_str("1-5")
    transforms = [ExcludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 12

    for name in filtered["image_names"]:
        num = number_from_filename(name)
        assert num is not None
        assert num > 5

    assert filtered["point_count"] <= original["point_count"]


def test_exclude_range_filter_single_image(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test exclude range filter with a single image."""
    output_path = tmp_path / "exclude_single.sfmr"

    range_expr = IntRangeExpr.from_str("10")
    transforms = [ExcludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == original["image_count"] - 1

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert 10 not in kept_numbers


def test_exclude_range_all_images_raises_error(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that exclude range filter raises error when all images are excluded."""
    output_path = tmp_path / "exclude_all.sfmr"

    range_expr = IntRangeExpr.from_str("1-17")
    transforms = [ExcludeRangeFilter(range_expr)]

    with pytest.raises(ValueError, match="No images remain"):
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images, output_path, transforms
        )


# =============================================================================
# Combined Include/Exclude Tests
# =============================================================================


def test_include_and_exclude_range_combined(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test using both include and exclude range in sequence."""
    output_path = tmp_path / "include_exclude.sfmr"

    include_expr = IntRangeExpr.from_str("1-10")
    exclude_expr = IntRangeExpr.from_str("5")
    transforms = [
        IncludeRangeFilter(include_expr),
        ExcludeRangeFilter(exclude_expr),
    ]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 9

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert kept_numbers == {1, 2, 3, 4, 6, 7, 8, 9, 10}


def test_range_filter_preserves_observation_counts(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that observation counts are correctly updated after filtering."""
    output_path = tmp_path / "range_obs_counts.sfmr"

    range_expr = IntRangeExpr.from_str("5-15")
    transforms = [IncludeRangeFilter(range_expr)]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)

    assert np.all(filtered["observation_counts"] > 0)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


# =============================================================================
# Rig-Aware Filtering Tests (_filter_rig_frame_data standalone)
# =============================================================================


def test_filter_rig_frame_data_standalone():
    """Test _filter_rig_frame_data directly with specific scenarios."""
    # Create rig data for 3 sensors, 4 frames (12 images)
    rig_data = {
        "rigs_metadata": {"rig_count": 1, "sensor_count": 3},
        "sensor_camera_indexes": np.array([0, 0, 0], dtype=np.uint32),
        "sensor_quaternions_wxyz": np.tile([1.0, 0, 0, 0], (3, 1)),
        "sensor_translations_xyz": np.zeros((3, 3)),
        "frames_metadata": {"frame_count": 4},
        "rig_indexes": np.zeros(4, dtype=np.uint32),
        # sensor 0: images 0-3 (frames 0-3)
        # sensor 1: images 4-7 (frames 0-3)
        # sensor 2: images 8-11 (frames 0-3)
        "image_sensor_indexes": np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint32
        ),
        "image_frame_indexes": np.array(
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint32
        ),
    }

    # Keep images 0, 1, 4, 5, 8, 9 (frames 0 and 1 from all sensors)
    images_to_keep = np.array([0, 1, 4, 5, 8, 9], dtype=np.uint32)
    result = _filter_rig_frame_data(rig_data, images_to_keep)

    assert result is not None
    assert result["frames_metadata"]["frame_count"] == 2
    assert len(result["rig_indexes"]) == 2
    np.testing.assert_array_equal(result["image_sensor_indexes"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(result["image_frame_indexes"], [0, 1, 0, 1, 0, 1])

    # Sensor data unchanged
    assert result["sensor_quaternions_wxyz"].shape == (3, 4)


def test_filter_rig_frame_data_none():
    """Test _filter_rig_frame_data returns None when input is None."""
    result = _filter_rig_frame_data(None, np.array([0, 1], dtype=np.uint32))
    assert result is None


def test_filter_rig_all_frames_remain():
    """Test filtering that removes images but keeps all frames populated."""
    # 2 sensors, 2 frames = 4 images
    # sensor 0: images 0, 1 (frames 0, 1)
    # sensor 1: images 2, 3 (frames 0, 1)
    rig_data = {
        "rigs_metadata": {"rig_count": 1, "sensor_count": 2},
        "sensor_camera_indexes": np.array([0, 0], dtype=np.uint32),
        "sensor_quaternions_wxyz": np.tile([1.0, 0, 0, 0], (2, 1)),
        "sensor_translations_xyz": np.zeros((2, 3)),
        "frames_metadata": {"frame_count": 2},
        "rig_indexes": np.zeros(2, dtype=np.uint32),
        "image_sensor_indexes": np.array([0, 0, 1, 1], dtype=np.uint32),
        "image_frame_indexes": np.array([0, 1, 0, 1], dtype=np.uint32),
    }

    # Remove only sensor 1's images -- both frames still have sensor 0
    images_to_keep = np.array([0, 1], dtype=np.uint32)
    result = _filter_rig_frame_data(rig_data, images_to_keep)

    assert result is not None
    # Both frames should still be present
    assert result["frames_metadata"]["frame_count"] == 2
    assert len(result["rig_indexes"]) == 2
    # Only sensor 0 images remain
    np.testing.assert_array_equal(result["image_sensor_indexes"], [0, 0])
    np.testing.assert_array_equal(result["image_frame_indexes"], [0, 1])
