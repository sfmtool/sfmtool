# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for chaining multiple transforms together."""

import numpy as np
from openjd.model import IntRangeExpr

from sfmtool.xform import (
    BundleAdjustTransform,
    FilterByReprojectionErrorTransform,
    IncludeRangeFilter,
    RemoveShortTracksFilter,
    RotateTransform,
    ScaleTransform,
    TranslateTransform,
)

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_transform_chain(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that multiple transformations can be chained."""
    output_path = tmp_path / "chained.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        RotateTransform(np.array([0, 1, 0]), np.radians(45)),
        TranslateTransform(np.array([1, 2, 3])),
        ScaleTransform(2.0),
    ]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    transformed = load_reconstruction_data(output_path)

    assert transformed["point_count"] < original["point_count"]
    assert np.all(transformed["observation_counts"] > 2)


def test_filter_then_transform_chain(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test chaining multiple filters followed by transforms."""
    output_path = tmp_path / "multi_filter_chain.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        FilterByReprojectionErrorTransform(threshold=5.0),
        ScaleTransform(0.5),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert result["point_count"] < original["point_count"]
    assert np.all(result["observation_counts"] > 2)
    assert np.all(result["errors"] <= 5.0)


def test_range_filter_chain_with_other_transforms(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that range filter can be chained with other transforms."""
    output_path = tmp_path / "range_chain.sfmr"

    range_expr = IntRangeExpr.from_str("1-10")
    transforms = [
        IncludeRangeFilter(range_expr),
        RemoveShortTracksFilter(2),
        ScaleTransform(2.0),
    ]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["image_count"] == 10
    assert filtered["point_count"] < original["point_count"]
    assert np.all(filtered["observation_counts"] > 2)


def test_complex_pipeline(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test a complex multi-step pipeline."""
    output_path = tmp_path / "complex_pipeline.sfmr"

    transforms = [
        IncludeRangeFilter(IntRangeExpr.from_str("3-15")),
        RemoveShortTracksFilter(2),
        RotateTransform(np.array([1, 0, 0]), np.radians(30)),
        TranslateTransform(np.array([5, 0, 0])),
        ScaleTransform(1.5),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    result = load_reconstruction_data(output_path)

    assert result["image_count"] == 13  # 3-15 inclusive
    assert np.all(result["observation_counts"] > 2)


def test_empty_transform_list(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that empty transform list preserves reconstruction."""
    output_path = tmp_path / "no_transforms.sfmr"

    transforms = []

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert result["point_count"] == original["point_count"]
    assert result["image_count"] == original["image_count"]
    assert np.allclose(result["positions"], original["positions"])


def test_transforms_after_bundle_adjust(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that transforms work correctly after bundle adjustment."""
    output_path = tmp_path / "ba_then_transform.sfmr"

    transforms = [
        BundleAdjustTransform(),
        ScaleTransform(2.0),
        TranslateTransform(np.array([1, 1, 1])),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert result["point_count"] == original["point_count"]
    assert result["image_count"] == original["image_count"]


def test_multiple_scales(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test applying multiple scale transforms in sequence."""
    output_path = tmp_path / "multi_scale.sfmr"

    # Scale by 2, then by 0.5 = total scale of 1.0
    transforms = [
        ScaleTransform(2.0),
        ScaleTransform(0.5),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert np.allclose(result["positions"], original["positions"], atol=1e-6)


def test_multiple_translations(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test applying multiple translations in sequence."""
    output_path = tmp_path / "multi_translate.sfmr"

    # Translate by (1,2,3) then by (-1,-2,-3) = no net translation
    transforms = [
        TranslateTransform(np.array([1, 2, 3])),
        TranslateTransform(np.array([-1, -2, -3])),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert np.allclose(result["positions"], original["positions"], atol=1e-6)


def test_all_filters_combined(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test combining all filter types."""
    from sfmtool.xform import RemoveIsolatedPointsFilter, RemoveNarrowTracksFilter

    output_path = tmp_path / "all_filters.sfmr"

    transforms = [
        IncludeRangeFilter(IntRangeExpr.from_str("2-16")),
        RemoveShortTracksFilter(2),
        FilterByReprojectionErrorTransform(threshold=5.0),
        RemoveNarrowTracksFilter(np.radians(2.0)),
        RemoveIsolatedPointsFilter(factor=10.0, value_spec="median"),
    ]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result = load_reconstruction_data(output_path)

    assert result["image_count"] == 15  # 2-16 inclusive
    assert result["point_count"] <= original["point_count"]
    assert np.all(result["observation_counts"] > 2)
    assert np.all(result["errors"] <= 5.0)
