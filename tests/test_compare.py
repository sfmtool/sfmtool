# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for sfm compare command and compare_reconstructions."""

import io
import sys
import numpy as np
from click.testing import CliRunner
from openjd.model import IntRangeExpr

from sfmtool._compare import compare_reconstructions
from sfmtool._sfmtool import Se3Transform, SfmrReconstruction
from sfmtool.cli import main
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

    def test_all_images_match(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        output = _capture_compare(recon, recon)
        assert "Matching images: 17" in output
        assert "Only in reference: 0" in output
        assert "Only in target: 0" in output

    def test_camera_parameters_match(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        output = _capture_compare(recon, recon)
        assert "Camera parameters match" in output

    def test_same_sift_files(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        output = _capture_compare(recon, recon)
        assert "Same SIFT file: 17" in output

    def test_identity_alignment(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        output = _capture_compare(recon, recon)
        assert "RMS error:" in output
        for line in output.split("\n"):
            if "RMS error:" in line:
                rms = float(line.split(":")[-1].strip())
                assert rms < 0.2

    def test_all_points_correspond(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        n_points = recon.point_count
        output = _capture_compare(recon, recon)
        assert f"Corresponding point pairs: {n_points}" in output
        assert f"< 0.01: {n_points} (100.0%)" in output

    def test_conclusion_identical(self, sfmrfile_reconstruction_with_17_images):
        recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
        output = _capture_compare(recon, recon)
        assert "IDENTICAL" in output


class TestCompareTransformed:
    """Test comparing original with a similarity-transformed version."""

    def test_cameras_still_match(self, sfmrfile_reconstruction_with_17_images):
        original_path = sfmrfile_reconstruction_with_17_images
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

        assert "Camera parameters match" in output
        assert "Matching images: 17" in output
        assert "Same SIFT file: 17" in output

    def test_alignment_recovers_inverse_scale(
        self, sfmrfile_reconstruction_with_17_images
    ):
        original_path = sfmrfile_reconstruction_with_17_images
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

    def test_features_identical_after_transform(
        self, sfmrfile_reconstruction_with_17_images
    ):
        original_path = sfmrfile_reconstruction_with_17_images
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

    def test_subset_matching(self, sfmrfile_reconstruction_with_17_images):
        original_path = sfmrfile_reconstruction_with_17_images
        workspace = original_path.parent

        filtered_path = _apply_transforms_to_file(
            original_path,
            workspace / "filtered.sfmr",
            [IncludeRangeFilter(IntRangeExpr.from_str("5-12"))],
        )

        original = SfmrReconstruction.load(original_path)
        filtered = SfmrReconstruction.load(filtered_path)
        output = _capture_compare(original, filtered)

        assert "Camera parameters match" in output

        for line in output.split("\n"):
            if "Matching images:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 8
            if "Only in reference:" in line:
                n = int(line.split(":")[-1].strip())
                assert n == 9

        assert "Only in target: 0" in output

    def test_same_sift_for_matching(self, sfmrfile_reconstruction_with_17_images):
        original_path = sfmrfile_reconstruction_with_17_images
        workspace = original_path.parent

        filtered_path = _apply_transforms_to_file(
            original_path,
            workspace / "filtered_sift.sfmr",
            [IncludeRangeFilter(IntRangeExpr.from_str("5-12"))],
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

    def test_transform_then_filter(self, sfmrfile_reconstruction_with_17_images):
        original_path = sfmrfile_reconstruction_with_17_images
        workspace = original_path.parent

        rotation = _rot_quat_from_euler_angles(np.radians([20, 0, 0]))
        transform = Se3Transform(rotation=rotation, translation=[2, -1, 0.5], scale=1.5)
        modified_path = _apply_transforms_to_file(
            original_path,
            workspace / "xform_and_filter.sfmr",
            [
                SimilarityTransform(transform),
                IncludeRangeFilter(IntRangeExpr.from_str("1-10")),
            ],
        )

        original = SfmrReconstruction.load(original_path)
        modified = SfmrReconstruction.load(modified_path)
        output = _capture_compare(original, modified)

        assert "Camera parameters match" in output
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

    def test_compare_same_file(self, sfmrfile_reconstruction_with_17_images):
        sfmr = str(sfmrfile_reconstruction_with_17_images)
        result = CliRunner().invoke(main, ["compare", sfmr, sfmr])
        assert result.exit_code == 0, result.output
        assert "Comparing reconstructions:" in result.output
        assert "Matching images: 17" in result.output
        assert "Camera parameters match" in result.output
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

    def test_compare_with_transformed(self, sfmrfile_reconstruction_with_17_images):
        original_path = sfmrfile_reconstruction_with_17_images
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
