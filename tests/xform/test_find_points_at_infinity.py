# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the find/classify points-at-infinity xform operations.

These exercise the additive ``--find-points-at-infinity`` transform, which
appends new points and tracks, and the ``--classify-points-at-infinity``
reclassifier. Both read the workspace ``.sift`` files, so they use the
``seoul_bull_workspace`` fixture (a reconstruction with its ``.sift``
files on disk). See specs/cli/reconstruction/xform/find-points-at-infinity.md.
"""

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
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
    for sigma in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            FindPointsAtInfinityTransform(0.1, 300.0, 2, sigma_px=sigma)


def test_find_is_additive_and_consistent(seoul_bull_workspace):
    """Find appends points/tracks, keeps integrity, and yields only w=0 points."""
    original = SfmrReconstruction.load(seoul_bull_workspace)

    result = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )

    # Additive: the point count grew.
    assert result.point_count > original.point_count

    # Observation bookkeeping is internally consistent.
    assert result.observation_count == int(np.asarray(result.observation_counts).sum())

    # Finite point positions stay finite.
    assert np.isfinite(np.asarray(result.positions)).all()

    # Every appended point is a point at infinity: a candidate whose rays ask
    # for a depth is a finite point, which discovery does not add.
    n0 = original.point_count
    assert np.asarray(result.point_is_at_infinity)[n0:].all()

    # The cached infinity_point_count matches the actual w=0 count after find.
    assert result.infinity_point_count == int(
        np.asarray(result.point_is_at_infinity).sum()
    )


def test_find_assigns_finite_reprojection_errors(seoul_bull_workspace):
    """Discovered points carry a real, inline-computed reprojection error.

    A point at infinity still projects its bearing (rotation + intrinsics), so
    its error is well-defined — discovery measures it against the features the
    track was built from rather than leaving a 0.0 placeholder. This is what
    lets the reprojection-error filter score discovered infinity points.
    """
    original = SfmrReconstruction.load(seoul_bull_workspace)
    n0 = original.point_count

    result = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )
    assert result.point_count > n0

    # The appended (discovered) points: every error is finite.
    new_errors = np.asarray(result.errors)[n0:]
    assert np.all(np.isfinite(new_errors))

    # The discovered points at infinity are scored, not left at the 0.0
    # placeholder: errors are finite, some are nonzero, and the bearing
    # reprojects close to its member keypoints (a few pixels).
    at_inf = np.asarray(result.point_is_at_infinity)[n0:]
    inf_errors = new_errors[at_inf]
    assert inf_errors.size > 0
    assert np.any(inf_errors > 0.0)
    assert float(np.median(inf_errors)) < 10.0


def test_find_decides_on_the_point_or_bearing_test(seoul_bull_workspace):
    """Discovery scores its candidates at the measured noise, appends the
    bearings, and the reclassification pass that follows agrees with it."""
    original = SfmrReconstruction.load(seoul_bull_workspace)
    n0 = original.point_count
    result, summary = original.find_points_at_infinity(
        0.5, 300.0, 0.8, 2, max_features=1500
    )

    assert summary["sigma_px"] == pytest.approx(original.reprojection_noise_px())
    assert summary["noise"]["observation_count"] > 0
    assert summary["bearings"] == result.point_count - n0 > 0
    assert summary["short_baseline"] <= summary["bearings"]
    # The metadata states the count of points at infinity too, and follows
    # the appended bearings.
    assert result.metadata()["infinity_point_count"] == result.infinity_point_count
    assert result.infinity_point_count >= summary["bearings"]
    dropped = summary["finite"] + summary["bearing_behind_camera"] + summary["unscored"]
    assert summary["bearings"] + dropped == summary["candidates"]

    # Bearings do not enter the noise measure, so the result scores at the
    # same noise level and every appended point gets a bearing verdict.
    scores = result.point_or_bearing_scores(
        point_indexes=list(range(n0, result.point_count))
    )
    assert scores["sigma_px"] == pytest.approx(summary["sigma_px"])
    assert np.asarray(scores["scored"]).all()
    assert not np.asarray(scores["is_finite"]).any()

    # So reclassification, at the measured noise, changes nothing discovery
    # appended.
    _, classified = result.classify_points_at_infinity()
    assert classified["promoted"] == 0
    assert classified["demoted"] == 0


def test_find_takes_a_noise_level(seoul_bull_workspace):
    """A given sigma_px replaces the measured one; a smaller one gives each
    candidate's rays more weight, so no more of them are bearings."""
    original = SfmrReconstruction.load(seoul_bull_workspace)
    _, measured = original.find_points_at_infinity(0.5, 300.0, 0.8, 2, 1500)
    _, strict = original.find_points_at_infinity(
        0.5, 300.0, 0.8, 2, 1500, sigma_px=0.01
    )
    assert strict["sigma_px"] == 0.01
    assert strict["noise"] is None
    assert strict["candidates"] == measured["candidates"]
    assert strict["bearings"] <= measured["bearings"]
    assert strict["finite"] >= measured["finite"]
    with pytest.raises(ValueError):
        original.find_points_at_infinity(0.5, sigma_px=-1.0)


def test_find_refuses_an_embedded_patches_reconstruction(
    seoul_bull_ground_truth_sfmr,
):
    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    with pytest.raises(OSError, match="embedded_patches"):
        recon.find_points_at_infinity(0.1)


def test_min_views_three_yields_fewer(seoul_bull_workspace):
    """Requiring 3 views finds no more new points than requiring 2."""
    original = SfmrReconstruction.load(seoul_bull_workspace)

    two = FindPointsAtInfinityTransform(0.1, 300.0, 2, max_features=1500).apply(
        original
    )
    three = FindPointsAtInfinityTransform(0.1, 300.0, 3, max_features=1500).apply(
        original
    )

    new_two = two.point_count - original.point_count
    new_three = three.point_count - original.point_count
    assert new_two >= new_three


def test_classify_preserves_point_count(seoul_bull_workspace):
    """Classify only relabels existing points, so the count is unchanged."""
    original = SfmrReconstruction.load(seoul_bull_workspace)

    result = ClassifyPointsAtInfinityTransform(1.0).apply(original)

    assert result.point_count == original.point_count
    assert result.infinity_point_count == int(
        np.asarray(result.point_is_at_infinity).sum()
    )


def test_find_no_duplicate_observations(seoul_bull_workspace):
    """A 2D feature observes at most one 3D point.

    Discovery must skip keypoints already assigned to an existing point;
    reusing one would make a feature belong to two points, which the .sfmr
    list tolerates but COLMAP export (and bundle adjustment) rejects.
    """
    original = SfmrReconstruction.load(seoul_bull_workspace)
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
    seoul_bull_workspace,
):
    """Discovered tracks export to COLMAP and bundle-adjust cleanly.

    Regression for the one-feature-two-points collision that crashed
    ``read_binary`` during the materialize -> BA -> reclassify round trip.
    """
    original = SfmrReconstruction.load(seoul_bull_workspace)
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


def test_cli_find_points_at_infinity(seoul_bull_workspace):
    """End-to-end CLI run adds points."""
    # The fixture is already per-test isolated, and its .sfmr sits beside its
    # workspace, so the relative .sift paths resolve. Write the output there.
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("with_infinity.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--find-points-at-infinity",
        "0.1,300,2",
        "--max-features",
        "1500",
    ]
    result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()
    assert "Noise level:" in result.output
    assert "Candidate tracks:" in result.output

    original = SfmrReconstruction.load(input_sfmr)
    transformed = SfmrReconstruction.load(output_sfmr)
    assert transformed.point_count > original.point_count
    # Saved then reloaded: the cached count survives the write/read round trip.
    assert transformed.infinity_point_count == int(
        np.asarray(transformed.point_is_at_infinity).sum()
    )


def test_classify_cli_uses_the_measured_noise(seoul_bull_ground_truth_sfmr):
    """Bare ``--classify-points-at-infinity`` measures the noise and changes
    nothing on the ground truth, which agrees with the test at that level."""
    out_path = seoul_bull_ground_truth_sfmr.parent / "classified.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_ground_truth_sfmr),
            str(out_path),
            "--classify-points-at-infinity",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "Noise level: 0.4677 px, measured over 1,229 observations, "
        "4 excluded as outliers" in result.output
    )
    assert "Promoted to finite: 0; demoted to infinity: 0; kept: 280" in result.output
    original = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    classified = SfmrReconstruction.load(out_path)
    np.testing.assert_array_equal(
        classified.point_is_at_infinity, original.point_is_at_infinity
    )


def test_classify_cli_promotes_at_a_given_noise_level(seoul_bull_ground_truth_sfmr):
    """At a stated 0.216 px, bearing 188 scores 32.5 and is promoted to the
    fitted point, about 415 m from its observing cameras."""
    out_path = seoul_bull_ground_truth_sfmr.parent / "promoted.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_ground_truth_sfmr),
            str(out_path),
            "--classify-points-at-infinity",
            "0.216",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Noise level: 0.216 px (given)" in result.output
    assert "Promoted to finite: 1; demoted to infinity: 0" in result.output
    original = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    promoted = SfmrReconstruction.load(out_path)
    assert original.point_is_at_infinity[188]
    assert not promoted.point_is_at_infinity[188]
    assert promoted.infinity_point_count == original.infinity_point_count - 1
    # Its error is recomputed at the placed point, far under the bearing's.
    assert promoted.errors[188] < original.errors[188]


def test_classify_cli_rejects_a_bad_noise_level(seoul_bull_ground_truth_sfmr):
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_ground_truth_sfmr),
            str(seoul_bull_ground_truth_sfmr.parent / "out.sfmr"),
            "--classify-points-at-infinity=abc",
        ],
    )
    assert result.exit_code != 0
    assert "--classify-points-at-infinity" in result.output


def test_classify_without_usable_pixels_says_why(seoul_bull_ground_truth_sfmr, capsys):
    """Finite points whose pixels give no residual leave no noise level to
    measure, and the transform says that rather than that there are no finite
    points; the reconstruction is left as it is."""
    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    keypoints = np.full_like(np.asarray(recon.keypoints_xy), np.nan)
    blind = recon.clone_with_changes(keypoints_xy=keypoints)

    result = ClassifyPointsAtInfinityTransform().apply(blind)
    out = capsys.readouterr().out
    assert "No observation of a finite point gives a usable residual" in out
    np.testing.assert_array_equal(result.positions_xyzw, blind.positions_xyzw)
