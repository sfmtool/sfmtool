# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoveIsolatedPointsFilter."""

import numpy as np
import pytest

from sfmtool.xform import RemoveIsolatedPointsFilter

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_remove_isolated_points_median(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_median.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=5.0, value_spec="median")]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] <= original["point_count"]
    assert filtered["image_count"] == original["image_count"]


def test_remove_isolated_points_percentile(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_percentile.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=3.0, value_spec="95percentile")]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] <= original["point_count"]


def test_remove_isolated_points_percent_shorthand(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_percent.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=2.0, value_spec="90percent")]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)
    assert filtered["point_count"] > 0


def test_remove_isolated_points_strict_threshold(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_strict.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=1.0, value_spec="median")]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] < original["point_count"]
    assert filtered["point_count"] > 0


def test_remove_isolated_points_invalid_factor():
    with pytest.raises(ValueError, match="Factor must be positive"):
        RemoveIsolatedPointsFilter(factor=0.0, value_spec="median")
    with pytest.raises(ValueError, match="Factor must be positive"):
        RemoveIsolatedPointsFilter(factor=-1.0, value_spec="median")


def test_remove_isolated_points_invalid_percentile_value(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_invalid_pct.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=2.0, value_spec="150percentile")]

    with pytest.raises(ValueError, match="Percentile must be in"):
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images, output_path, transforms
        )


def test_remove_isolated_points_preserves_tracks(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_tracks.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=3.0, value_spec="median")]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    filtered = load_reconstruction_data(output_path)

    assert np.all(filtered["observation_counts"] > 0)
    assert filtered["observation_count"] == np.sum(filtered["observation_counts"])


def test_remove_isolated_points_description():
    filter_transform = RemoveIsolatedPointsFilter(factor=3.0, value_spec="median")
    desc = filter_transform.description()

    assert "isolated" in desc.lower() or "NN" in desc
    assert "3" in desc
    assert "median" in desc


def test_remove_isolated_points_100th_percentile(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "no_isolated_100pct.sfmr"
    transforms = [RemoveIsolatedPointsFilter(factor=1.0, value_spec="100percentile")]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    filtered = load_reconstruction_data(output_path)

    assert filtered["point_count"] == original["point_count"]
