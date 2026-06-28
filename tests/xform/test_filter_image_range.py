# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for IncludeRangeFilter and ExcludeRangeFilter."""

import numpy as np
import pytest

from sfmtool import RangeExpr
from sfmtool._filenames import number_from_filename
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.xform import ExcludeRangeFilter, IncludeRangeFilter
from sfmtool.xform._filter_by_image_range import _filter_images

from .conftest import apply_transforms_to_file, load_reconstruction_data


# =============================================================================
# IncludeRangeFilter Tests
# =============================================================================


def test_include_range_filter(seoul_bull_workspace, tmp_path):
    """Test that include range filter keeps only images in the specified range."""
    output_path = tmp_path / "include_range.sfmr"

    # Keep only images 1-5 (out of 1-17)
    range_expr = RangeExpr("1-5")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(seoul_bull_workspace)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 5

    for name in filtered["image_names"]:
        num = number_from_filename(name)
        assert num is not None
        assert 1 <= num <= 5

    assert filtered["point_count"] <= original["point_count"]


def test_include_range_filter_with_comma_separated(seoul_bull_workspace, tmp_path):
    """Test include range filter with comma-separated values."""
    output_path = tmp_path / "include_range_comma.sfmr"

    range_expr = RangeExpr("1,3,5,7")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 4

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert kept_numbers == {1, 3, 5, 7}


def test_include_range_filter_mixed_format(seoul_bull_workspace, tmp_path):
    """Test include range filter with mixed ranges and individual values."""
    output_path = tmp_path / "include_range_mixed.sfmr"

    range_expr = RangeExpr("1-3,10,15-17")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 7

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert kept_numbers == {1, 2, 3, 10, 15, 16, 17}


def test_include_range_removes_orphaned_points(seoul_bull_workspace, tmp_path):
    """Test that filtering images also removes 3D points with no remaining observations."""
    output_path = tmp_path / "include_range_orphaned.sfmr"

    range_expr = RangeExpr("1-3")
    transforms = [IncludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(seoul_bull_workspace)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] >= 1)


def test_include_range_no_matching_images_raises_error(seoul_bull_workspace, tmp_path):
    """Test that include range filter raises error when no images match."""
    output_path = tmp_path / "no_match.sfmr"

    range_expr = RangeExpr("100-200")
    transforms = [IncludeRangeFilter(range_expr)]

    with pytest.raises(ValueError, match="No images remain"):
        apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)


# =============================================================================
# ExcludeRangeFilter Tests
# =============================================================================


def test_exclude_range_filter(seoul_bull_workspace, tmp_path):
    """Test that exclude range filter removes images in the specified range."""
    output_path = tmp_path / "exclude_range.sfmr"

    range_expr = RangeExpr("1-5")
    transforms = [ExcludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(seoul_bull_workspace)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 12

    for name in filtered["image_names"]:
        num = number_from_filename(name)
        assert num is not None
        assert num > 5

    assert filtered["point_count"] <= original["point_count"]


def test_exclude_range_filter_single_image(seoul_bull_workspace, tmp_path):
    """Test exclude range filter with a single image."""
    output_path = tmp_path / "exclude_single.sfmr"

    range_expr = RangeExpr("10")
    transforms = [ExcludeRangeFilter(range_expr)]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(seoul_bull_workspace)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == original["image_count"] - 1

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert 10 not in kept_numbers


def test_exclude_range_all_images_raises_error(seoul_bull_workspace, tmp_path):
    """Test that exclude range filter raises error when all images are excluded."""
    output_path = tmp_path / "exclude_all.sfmr"

    range_expr = RangeExpr("1-17")
    transforms = [ExcludeRangeFilter(range_expr)]

    with pytest.raises(ValueError, match="No images remain"):
        apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)


# =============================================================================
# Combined Include/Exclude Tests
# =============================================================================


def test_include_and_exclude_range_combined(seoul_bull_workspace, tmp_path):
    """Test using both include and exclude range in sequence."""
    output_path = tmp_path / "include_exclude.sfmr"

    include_expr = RangeExpr("1-10")
    exclude_expr = RangeExpr("5")
    transforms = [
        IncludeRangeFilter(include_expr),
        ExcludeRangeFilter(exclude_expr),
    ]

    result = apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    assert result == output_path
    assert output_path.exists()

    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 9

    kept_numbers = {number_from_filename(name) for name in filtered["image_names"]}
    assert kept_numbers == {1, 2, 3, 4, 6, 7, 8, 9, 10}


def test_range_filter_preserves_observation_counts(seoul_bull_workspace, tmp_path):
    """Test that observation counts are correctly updated after filtering."""
    output_path = tmp_path / "range_obs_counts.sfmr"

    range_expr = RangeExpr("5-15")
    transforms = [IncludeRangeFilter(range_expr)]

    apply_transforms_to_file(seoul_bull_workspace, output_path, transforms)

    filtered = load_reconstruction_data(output_path)

    assert np.all(filtered["observation_counts"] > 0)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


# =============================================================================
# Points at Infinity
# =============================================================================


def test_range_filter_preserves_points_at_infinity(seoul_bull_workspace):
    """Image filtering must preserve w=0 points.

    Regression: ``_filter_images`` used to rebuild from the ``(N, 3)`` Euclidean
    positions, which ``clone_with_changes`` interprets as ``w = 1`` — silently
    materialising every point at infinity. Delegating to the Rust
    ``subset_by_image_indices`` primitive keeps ``w = 0`` intact.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace)

    # Mark the first few points as at infinity (w = 0 directions).
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    xyzw[:5, 3] = 0.0
    injected = recon.clone_with_changes(positions=xyzw)
    n_inf = int(np.asarray(injected.point_is_at_infinity).sum())
    assert n_inf >= 5

    # Keep every image: no point is orphaned, so all infinity points survive.
    kept = np.arange(injected.image_count, dtype=np.uint32)
    result = _filter_images(injected, kept)

    assert int(np.asarray(result.point_is_at_infinity).sum()) == n_inf
    assert result.infinity_point_count == n_inf


# =============================================================================
# Rig-Aware Filtering (end-to-end through subset_by_image_indices)
# =============================================================================


def test_range_filter_keeps_rig_frame_data_consistent(kerry_park_workspace, tmp_path):
    """Filtering a rig reconstruction by image range keeps rig/frame data
    internally consistent — the Rust subset primitive owns the frame remap; this
    is the integration check across the CLI transform path."""
    recon = SfmrReconstruction.load(kerry_park_workspace)
    assert recon.rig_frame_data is not None
    before_frames = recon.rig_frame_data["frames_metadata"]["frame_count"]

    # Drop one frame number (files are numbered 1-8), which removes that frame
    # from every sensor.
    output_path = tmp_path / "rig_filtered.sfmr"
    apply_transforms_to_file(
        kerry_park_workspace, output_path, [ExcludeRangeFilter(RangeExpr("1"))]
    )

    filtered = SfmrReconstruction.load(output_path)
    rfd = filtered.rig_frame_data
    assert rfd is not None
    # One whole frame removed, and the frame index column was remapped to stay
    # contiguous and aligned with the surviving images.
    assert rfd["frames_metadata"]["frame_count"] == before_frames - 1
    assert len(rfd["image_sensor_indexes"]) == filtered.image_count
    assert len(rfd["image_frame_indexes"]) == filtered.image_count
    assert len(rfd["rig_indexes"]) == rfd["frames_metadata"]["frame_count"]
    assert (
        int(rfd["image_frame_indexes"].max())
        == rfd["frames_metadata"]["frame_count"] - 1
    )
