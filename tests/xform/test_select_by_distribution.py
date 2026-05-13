# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SelectByDistributionFilter (sfm xform --include-by-distribution)."""

import math

import numpy as np
import pytest

from sfmtool._inspect_images import _compute_camera_centers
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.xform import (
    BundleAdjustTransform,
    RemoveShortTracksFilter,
    SelectByDistributionFilter,
)

from .conftest import apply_transforms_to_file, load_reconstruction_data


def _max_triangulation_angles_deg(path) -> np.ndarray:
    """Per-point max pairwise viewing-ray angle (deg) over the kept images."""
    recon = SfmrReconstruction.load(path)
    centers = _compute_camera_centers(
        np.asarray(recon.quaternions_wxyz), np.asarray(recon.translations)
    )
    positions = np.asarray(recon.positions)
    track_img = np.asarray(recon.track_image_indexes).astype(np.int64)
    track_pt = np.asarray(recon.track_point_ids).astype(np.int64)
    vec = positions[track_pt] - centers[track_img]
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    rays_by_point: dict[int, list[np.ndarray]] = {}
    for i, p in enumerate(track_pt):
        rays_by_point.setdefault(int(p), []).append(vec[i])
    out = []
    for rays in rays_by_point.values():
        if len(rays) < 2:
            out.append(0.0)
            continue
        r = np.array(rays)
        out.append(math.degrees(math.acos(np.clip(float((r @ r.T).min()), -1.0, 1.0))))
    return np.array(out)


def test_count_below_two_rejected():
    with pytest.raises(ValueError):
        SelectByDistributionFilter(1)
    with pytest.raises(ValueError):
        SelectByDistributionFilter(0)


def test_description():
    assert (
        SelectByDistributionFilter(8).description()
        == "Select 8 cameras by distribution"
    )


def test_keeps_requested_count(sfmrfile_reconstruction_with_17_images, tmp_path):
    output_path = tmp_path / "selected.sfmr"
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images,
        output_path,
        [SelectByDistributionFilter(8)],
    )
    result = load_reconstruction_data(output_path)
    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    assert result["image_count"] == 8
    assert set(result["image_names"]).issubset(set(original["image_names"]))
    assert 0 < result["point_count"] <= original["point_count"]


def test_count_at_least_unit_count_is_noop(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    for count in (17, 100):
        out = tmp_path / f"noop_{count}.sfmr"
        apply_transforms_to_file(
            sfmrfile_reconstruction_with_17_images,
            out,
            [SelectByDistributionFilter(count)],
        )
        result = load_reconstruction_data(out)
        assert result["image_count"] == original["image_count"]
        assert result["point_count"] == original["point_count"]


def test_deterministic(sfmrfile_reconstruction_with_17_images, tmp_path):
    out_a = tmp_path / "a.sfmr"
    out_b = tmp_path / "b.sfmr"
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, out_a, [SelectByDistributionFilter(7)]
    )
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, out_b, [SelectByDistributionFilter(7)]
    )
    assert (
        load_reconstruction_data(out_a)["image_names"]
        == load_reconstruction_data(out_b)["image_names"]
    )


def test_selection_is_well_triangulated(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """The kept subset should leave a healthy fraction of its points triangulated
    across a real baseline (the whole point of the filter)."""
    out = tmp_path / "dist.sfmr"
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, out, [SelectByDistributionFilter(8)]
    )
    angles = _max_triangulation_angles_deg(out)
    assert angles.size > 0
    # A meaningful share of points sees >= 10 deg of parallax.
    assert np.mean(angles >= 10.0) >= 0.25


def test_chaining_with_filter_and_bundle_adjust(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    output_path = tmp_path / "chained.sfmr"
    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images,
        output_path,
        [
            SelectByDistributionFilter(10),
            RemoveShortTracksFilter(1),
            BundleAdjustTransform(),
        ],
    )
    result = load_reconstruction_data(output_path)
    assert result["image_count"] == 10
    assert np.all(result["observation_counts"] >= 2)
