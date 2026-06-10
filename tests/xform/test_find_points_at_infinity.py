# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the find/classify points-at-infinity xform operations.

These exercise the additive ``--find-points-at-infinity`` transform, which
appends new points and tracks, and the ``--classify-points-at-infinity``
reclassifier. Both read the workspace ``.sift`` files, so they use the
``sfmrfile_reconstruction_with_17_images`` fixture (a real solve with sift
artifacts on disk). See specs/cli/xform-find-points-at-infinity.md.
"""

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import (
    BundleAdjustTransform,
    ClassifyPointsAtInfinityTransform,
    FindPointsAtInfinityTransform,
)


def test_constructor_validation():
    """eps_deg <= 0 and min_views < 2 are rejected."""
    with pytest.raises(ValueError):
        FindPointsAtInfinityTransform(0.0, 300.0, 2)
    with pytest.raises(ValueError):
        FindPointsAtInfinityTransform(-1.0, 300.0, 2)
    with pytest.raises(ValueError):
        FindPointsAtInfinityTransform(0.1, 300.0, 1)


def test_find_is_additive_and_consistent(sfmrfile_reconstruction_with_17_images):
    """Find appends points/tracks, keeps integrity, and yields w=0 points."""
    original = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)

    result = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )

    # Additive: the point count grew.
    assert result.point_count > original.point_count

    # Observation bookkeeping is internally consistent.
    assert result.observation_count == int(np.asarray(result.observation_counts).sum())

    # Finite point positions stay finite.
    assert np.isfinite(np.asarray(result.positions)).all()

    # A tight eps yields some genuine w=0 points at infinity.
    assert int(np.asarray(result.point_is_at_infinity).sum()) > 0

    # The cached infinity_point_count matches the actual w=0 count after find.
    assert result.infinity_point_count == int(
        np.asarray(result.point_is_at_infinity).sum()
    )


def test_min_views_three_yields_fewer(sfmrfile_reconstruction_with_17_images):
    """Requiring 3 views finds no more new points than requiring 2."""
    original = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)

    two = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )
    three = FindPointsAtInfinityTransform(0.1, 300.0, 3, max_features=1500).apply(
        original
    )

    new_two = two.point_count - original.point_count
    new_three = three.point_count - original.point_count
    assert new_two >= new_three


def test_classify_preserves_point_count(sfmrfile_reconstruction_with_17_images):
    """Classify only relabels existing points, so the count is unchanged."""
    original = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)

    result = ClassifyPointsAtInfinityTransform(1.0).apply(original)

    assert result.point_count == original.point_count
    assert result.infinity_point_count == int(
        np.asarray(result.point_is_at_infinity).sum()
    )


def test_find_no_duplicate_observations(sfmrfile_reconstruction_with_17_images):
    """A 2D feature observes at most one 3D point.

    Discovery must skip keypoints already assigned to an existing point;
    reusing one would make a feature belong to two points, which the .sfmr
    list tolerates but COLMAP export (and bundle adjustment) rejects.
    """
    original = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    result = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )

    pairs = np.stack(
        [
            np.asarray(result.track_image_indexes),
            np.asarray(result.track_feature_indexes),
        ],
        axis=1,
    )
    unique = np.unique(pairs, axis=0)
    assert len(unique) == len(pairs), "a feature is observed by more than one point"


def test_found_reconstruction_survives_bundle_adjust(
    sfmrfile_reconstruction_with_17_images,
):
    """Discovered tracks export to COLMAP and bundle-adjust cleanly.

    Regression for the one-feature-two-points collision that crashed
    ``read_binary`` during the materialize -> BA -> reclassify round trip.
    """
    original = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    found = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )
    assert found.point_count > original.point_count

    adjusted = BundleAdjustTransform().apply(found)

    # The round trip keeps the bulk of the discovered points and stays finite.
    assert adjusted.point_count > original.point_count
    assert np.isfinite(np.asarray(adjusted.positions)).all()
    assert adjusted.infinity_point_count == int(
        np.asarray(adjusted.point_is_at_infinity).sum()
    )


def test_cli_find_points_at_infinity(sfmrfile_reconstruction_with_17_images):
    """End-to-end CLI run adds points; the sys.argv reparse needs patching."""
    # The fixture is already per-test isolated, and its .sfmr sits beside its
    # workspace, so the relative .sift paths resolve. Write the output there.
    input_sfmr = sfmrfile_reconstruction_with_17_images
    output_sfmr = input_sfmr.with_name("with_infinity.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--find-points-at-infinity",
        "0.1,300,3",
        "--max-features",
        "800",
    ]
    # The xform command re-parses sys.argv to preserve transform order, so the
    # CliRunner invocation must have argv contain "xform" + the same args.
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()

    original = SfmrReconstruction.load(input_sfmr)
    transformed = SfmrReconstruction.load(output_sfmr)
    assert transformed.point_count > original.point_count
    # Saved then reloaded: the cached count survives the write/read round trip.
    assert transformed.infinity_point_count == int(
        np.asarray(transformed.point_is_at_infinity).sum()
    )
