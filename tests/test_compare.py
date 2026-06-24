# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for sfm compare command and compare_reconstructions."""

import io
import sys
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool import RangeExpr
from sfmtool._commands.compare import _parse_labels
from sfmtool._compare import compare_reconstructions
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._sfmtool.geometry import Se3Transform
from sfmtool.cli import main
from sfmtool.sift.file import get_sift_path_from_recon
from sfmtool.xform import IncludeRangeFilter, SimilarityTransform, apply_transforms


def _rot_quat_from_euler_angles(angles_rad):
    """Create a RotQuaternion from Euler angles (roll, pitch, yaw) in radians."""
    roll, pitch, yaw = angles_rad
    tx = Se3Transform.from_axis_angle(np.array([1, 0, 0]), roll)
    ty = Se3Transform.from_axis_angle(np.array([0, 1, 0]), pitch)
    tz = Se3Transform.from_axis_angle(np.array([0, 0, 1]), yaw)
    composed = tz @ ty @ tx
    return composed.rotation


def _apply_transforms_to_file(input_path, output_path, transforms):
    """Helper that wraps apply_transforms with file I/O for tests."""
    recon = SfmrReconstruction.load(input_path)
    recon = apply_transforms(recon, transforms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recon.save(output_path, operation="xform_test")
    return output_path


def _capture_compare(recon1, recon2, name1="recon1", name2="recon2"):
    """Run compare_reconstructions and capture stdout."""
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        compare_reconstructions(recon1, recon2, recon1_name=name1, recon2_name=name2)
        return captured.getvalue()
    finally:
        sys.stdout = old_stdout


class TestCompareIdentical:
    """Test comparing a reconstruction with itself."""

    def test_all_images_match(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        output = _capture_compare(recon, recon)
        assert "Matching images: 17" in output
        assert "Only in reference: 0" in output
        assert "Only in target: 0" in output

    def test_camera_parameters_match(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        output = _capture_compare(recon, recon)
        assert "All parameters match" in output

    def test_same_sift_files(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        output = _capture_compare(recon, recon)
        assert "Same SIFT file: 17" in output

    def test_identity_alignment(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        output = _capture_compare(recon, recon)
        assert "RMS error:" in output
        for line in output.split("\n"):
            if "RMS error:" in line:
                rms = float(line.split(":")[-1].strip())
                assert rms < 0.2

    def test_all_points_correspond(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        n_points = recon.point_count
        output = _capture_compare(recon, recon)
        assert f"Corresponding point pairs: {n_points}" in output
        # Identical recon: every pair is at zero distance, so all fall within the
        # tightest relative threshold (distances are reported as % of scene scale).
        assert f"< 0.1%: {n_points} (100.0%)" in output

    def test_conclusion_identical(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        output = _capture_compare(recon, recon)
        assert "IDENTICAL" in output


class TestCompareTransformed:
    """Test comparing original with a similarity-transformed version."""

    def test_cameras_still_match(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        rotation = _rot_quat_from_euler_angles(np.radians([45, 30, 15]))
        transform = Se3Transform(rotation=rotation, translation=[10, 5, -3], scale=2.5)
        transformed_path = _apply_transforms_to_file(
            original_path,
            workspace / "transformed.sfmr",
            [SimilarityTransform(transform)],
        )

        original = SfmrReconstruction.load(original_path)
        transformed = SfmrReconstruction.load(transformed_path)
        output = _capture_compare(original, transformed)

        assert "All parameters match" in output
        assert "Matching images: 17" in output
        assert "Same SIFT file: 17" in output

    def test_alignment_recovers_inverse_scale(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent
        scale = 2.5

        rotation = _rot_quat_from_euler_angles(np.radians([45, 30, 15]))
        transform = Se3Transform(
            rotation=rotation, translation=[10, 5, -3], scale=scale
        )
        transformed_path = _apply_transforms_to_file(
            original_path,
            workspace / "transformed_scale.sfmr",
            [SimilarityTransform(transform)],
        )

        original = SfmrReconstruction.load(original_path)
        transformed = SfmrReconstruction.load(transformed_path)
        output = _capture_compare(original, transformed)

        for line in output.split("\n"):
            if "    Scale:" in line:
                recovered_scale = float(line.split(":")[-1].strip())
                expected_scale = 1.0 / scale
                assert abs(recovered_scale - expected_scale) < 0.01

    def test_distance_metric_is_scale_independent(self, seoul_bull_workspace):
        # A reconstruction compared against a 100x-scaled copy of itself must
        # report the same scale-independent stats as comparing it to a 1x copy:
        # the similarity alignment removes the gauge, and residuals are reported
        # as a percentage of scene scale. Only the absolute "scene scale" line
        # should differ (by 100x).
        original_path = seoul_bull_workspace
        workspace = original_path.parent
        original = SfmrReconstruction.load(original_path)
        n_points = original.point_count

        identity_rot = _rot_quat_from_euler_angles(np.radians([0, 0, 0]))

        def _compare_to_scaled(scale):
            transform = Se3Transform(
                rotation=identity_rot, translation=[0, 0, 0], scale=scale
            )
            scaled_path = _apply_transforms_to_file(
                original_path,
                workspace / f"scaled_{scale}.sfmr",
                [SimilarityTransform(transform)],
            )
            scaled = SfmrReconstruction.load(scaled_path)
            return _capture_compare(original, scaled)

        out_1x = _compare_to_scaled(1.0)
        out_100x = _compare_to_scaled(100.0)

        # Identical-up-to-scale => every pair within the tightest relative bucket,
        # regardless of the 100x gauge change.
        assert f"< 0.1%: {n_points} (100.0%)" in out_1x
        assert f"< 0.1%: {n_points} (100.0%)" in out_100x
        assert "IDENTICAL" in out_1x
        assert "IDENTICAL" in out_100x

        # The reported absolute scene scale tracks the gauge (~100x apart).
        def _scene_scale(output):
            for line in output.split("\n"):
                if "Reference scene scale:" in line:
                    return float(line.split(":")[1].split("(")[0].strip())
            raise AssertionError("scene scale not reported")

        # Reference is the unscaled recon in both runs, so the scale line is the
        # same — the point is that the *percentage* stats above are identical.
        assert _scene_scale(out_1x) == _scene_scale(out_100x)

    def test_features_identical_after_transform(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        transform = Se3Transform(translation=[1, 2, 3], scale=1.5)
        transformed_path = _apply_transforms_to_file(
            original_path,
            workspace / "transformed_feat.sfmr",
            [SimilarityTransform(transform)],
        )

        original = SfmrReconstruction.load(original_path)
        transformed = SfmrReconstruction.load(transformed_path)
        output = _capture_compare(original, transformed)

        assert "Only in reference (mean): 0.0" in output
        assert "Only in target (mean): 0.0" in output


class TestCompareFiltered:
    """Test comparing original with a filtered (subset) version."""

    def test_subset_matching(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        filtered_path = _apply_transforms_to_file(
            original_path,
            workspace / "filtered.sfmr",
            [IncludeRangeFilter(RangeExpr("5-12"))],
        )

        original = SfmrReconstruction.load(original_path)
        filtered = SfmrReconstruction.load(filtered_path)
        output = _capture_compare(original, filtered)

        assert "All parameters match" in output

        for line in output.split("\n"):
            if "Matching images:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 8
            if "Only in reference:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 9

        assert "Only in target: 0" in output

    def test_same_sift_for_matching(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        filtered_path = _apply_transforms_to_file(
            original_path,
            workspace / "filtered_sift.sfmr",
            [IncludeRangeFilter(RangeExpr("5-12"))],
        )

        original = SfmrReconstruction.load(original_path)
        filtered = SfmrReconstruction.load(filtered_path)
        output = _capture_compare(original, filtered)

        for line in output.split("\n"):
            if "Same SIFT file:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 8


class TestCompareTransformAndFilter:
    """Test combining transform and filter."""

    def test_transform_then_filter(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        rotation = _rot_quat_from_euler_angles(np.radians([20, 0, 0]))
        transform = Se3Transform(rotation=rotation, translation=[2, -1, 0.5], scale=1.5)
        modified_path = _apply_transforms_to_file(
            original_path,
            workspace / "xform_and_filter.sfmr",
            [
                SimilarityTransform(transform),
                IncludeRangeFilter(RangeExpr("1-10")),
            ],
        )

        original = SfmrReconstruction.load(original_path)
        modified = SfmrReconstruction.load(modified_path)
        output = _capture_compare(original, modified)

        assert "All parameters match" in output
        assert "Matching images: 10" in output

        for line in output.split("\n"):
            if "Only in reference:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 7

        for line in output.split("\n"):
            if "    Same SIFT file:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 10


class TestCompareCLI:
    """Test the CLI command."""

    def test_compare_same_file(self, seoul_bull_workspace):
        sfmr = str(seoul_bull_workspace)
        result = CliRunner().invoke(main, ["compare", sfmr, sfmr])
        assert result.exit_code == 0, result.output
        assert "Comparing reconstructions:" in result.output
        assert "Matching images: 17" in result.output
        assert "All parameters match" in result.output
        assert "Alignment succeeded!" in result.output
        assert "Comparing feature usage" in result.output
        assert "Comparing 3D points" in result.output
        assert "Comparison complete!" in result.output
        assert "Completed in" in result.output

    def test_non_sfmr_first_arg(self, tmp_path):
        p = tmp_path / "input.txt"
        p.touch()
        sfmr = tmp_path / "other.sfmr"
        sfmr.touch()
        result = CliRunner().invoke(main, ["compare", str(p), str(sfmr)])
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_non_sfmr_second_arg(self, tmp_path):
        sfmr = tmp_path / "input.sfmr"
        sfmr.touch()
        p = tmp_path / "other.txt"
        p.touch()
        result = CliRunner().invoke(main, ["compare", str(sfmr), str(p)])
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_compare_with_transformed(self, seoul_bull_workspace):
        original_path = seoul_bull_workspace
        workspace = original_path.parent

        transform = Se3Transform(translation=[5, 0, 0], scale=2.0)
        transformed_path = _apply_transforms_to_file(
            original_path,
            workspace / "cli_transformed.sfmr",
            [SimilarityTransform(transform)],
        )

        result = CliRunner().invoke(
            main, ["compare", str(original_path), str(transformed_path)]
        )
        assert result.exit_code == 0, result.output
        assert "Matching images: 17" in result.output
        assert "Scale:" in result.output


class TestCompareCoordinateAndStrips:
    """Cross-backend coordinate matching, the --strips montage, and helpers."""

    def test_get_sift_path_from_recon(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        name = recon.image_names[0]
        path = get_sift_path_from_recon(recon, name)
        assert path.name == Path(name).name + ".sift"
        assert path.exists()

    def test_parse_labels(self):
        assert _parse_labels("ref,tgt") == ("ref", "tgt")
        assert _parse_labels(" OLD , NEW ") == ("OLD", "NEW")
        for bad in ("only", "a,b,c", "a,", ",b"):
            with pytest.raises(click.UsageError):
                _parse_labels(bad)

    def test_cli_compare_auto_uses_feature_index(self, seoul_bull_workspace):
        # Identical inputs share .sift hashes, so auto-resolution keys on
        # feature index rather than 2D keypoint coordinate.
        p = str(seoul_bull_workspace)
        result = CliRunner().invoke(main, ["compare", p, p])
        assert result.exit_code == 0, result.output
        assert "by feature index" in result.output

    def test_cli_compare_by_coordinate(self, seoul_bull_workspace):
        p = str(seoul_bull_workspace)
        result = CliRunner().invoke(main, ["compare", p, p, "--by-coordinate"])
        assert result.exit_code == 0, result.output
        assert "by keypoint coordinate" in result.output

    def test_cli_strips_writes_montage(self, seoul_bull_workspace, tmp_path):
        # Full vs a filtered subset: the full solve keeps points unique to it,
        # exercising the overview "unique to <label>" rows under custom labels.
        original_path = seoul_bull_workspace
        workspace = original_path.parent
        filtered_path = _apply_transforms_to_file(
            original_path,
            workspace / "strips_filtered.sfmr",
            [IncludeRangeFilter(RangeExpr("5-12"))],
        )
        out = tmp_path / "strips.png"
        result = CliRunner().invoke(
            main,
            [
                "compare",
                str(original_path),
                str(filtered_path),
                "--strips",
                str(out),
                "--strips-num",
                "8",
                "--strips-labels",
                "OLD,NEW",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "--strips: wrote" in result.output
        assert out.exists() and out.stat().st_size > 0

    def test_cli_strips_single_axis(self, seoul_bull_workspace, tmp_path):
        # A geometry-only single axis with an explicit end (no normal refine).
        original_path = seoul_bull_workspace
        workspace = original_path.parent
        transformed_path = _apply_transforms_to_file(
            original_path,
            workspace / "strips_axis.sfmr",
            [SimilarityTransform(Se3Transform(translation=[1, 0, 0], scale=1.0))],
        )
        out = tmp_path / "axis.png"
        result = CliRunner().invoke(
            main,
            [
                "compare",
                str(original_path),
                str(transformed_path),
                "--strips",
                str(out),
                "--strips-num",
                "6",
                "--strips-rank",
                "view-angle",
                "--strips-end",
                "high",
                "--strips-no-refine",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 0
