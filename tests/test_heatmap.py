# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the heatmap command and visualization utilities."""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._commands.heatmap import _insert_metric_before_number
from sfmtool.cli import main
from sfmtool.visualization._colormap import COLORMAPS, apply_colormap, value_to_color
from sfmtool.visualization._heatmap_renderer import (
    compute_triangulation_angles,
    render_heatmap_overlay,
)


# =============================================================================
# Colormap tests
# =============================================================================


class TestValueToColor:
    def test_returns_rgb_tuple(self):
        r, g, b = value_to_color(0.5, 0.0, 1.0, "viridis")
        assert isinstance(r, int)
        assert isinstance(g, int)
        assert isinstance(b, int)

    def test_values_in_range(self):
        for cmap_name in COLORMAPS:
            r, g, b = value_to_color(0.5, 0.0, 1.0, cmap_name)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_min_maps_to_first_color(self):
        for cmap_name in COLORMAPS:
            r, g, b = value_to_color(0.0, 0.0, 1.0, cmap_name)
            expected = COLORMAPS[cmap_name][0]
            assert (r, g, b) == (expected[1], expected[2], expected[3])

    def test_max_maps_to_last_color(self):
        for cmap_name in COLORMAPS:
            r, g, b = value_to_color(1.0, 0.0, 1.0, cmap_name)
            expected = COLORMAPS[cmap_name][-1]
            assert (r, g, b) == (expected[1], expected[2], expected[3])

    def test_clamps_below_min(self):
        c1 = value_to_color(-10.0, 0.0, 1.0, "viridis")
        c2 = value_to_color(0.0, 0.0, 1.0, "viridis")
        assert c1 == c2

    def test_clamps_above_max(self):
        c1 = value_to_color(100.0, 0.0, 1.0, "viridis")
        c2 = value_to_color(1.0, 0.0, 1.0, "viridis")
        assert c1 == c2

    def test_equal_vmin_vmax(self):
        # Should return middle color without error
        r, g, b = value_to_color(5.0, 5.0, 5.0, "viridis")
        assert 0 <= r <= 255

    def test_unknown_colormap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            value_to_color(0.5, 0.0, 1.0, "nonexistent")

    def test_error_colormap_green_to_red(self):
        # Low values should be green-ish
        r_low, g_low, b_low = value_to_color(0.0, 0.0, 1.0, "error")
        assert g_low > r_low  # Green dominant at low end
        # High values should be red-ish
        r_high, g_high, b_high = value_to_color(1.0, 0.0, 1.0, "error")
        assert r_high > g_high  # Red dominant at high end


class TestApplyColormap:
    def test_shape(self):
        values = np.array([0.0, 0.5, 1.0])
        result = apply_colormap(values, colormap="viridis")
        assert result.shape == (3, 3)
        assert result.dtype == np.uint8

    def test_2d_input(self):
        values = np.array([[0.0, 0.5], [0.75, 1.0]])
        result = apply_colormap(values, colormap="plasma")
        assert result.shape == (2, 2, 3)

    def test_nan_values_are_gray(self):
        values = np.array([np.nan])
        result = apply_colormap(values, vmin=0.0, vmax=1.0, colormap="viridis")
        assert tuple(result[0]) == (128, 128, 128)


# =============================================================================
# Insert metric before number tests
# =============================================================================


class TestInsertMetricBeforeNumber:
    def test_underscore_separator(self):
        assert (
            _insert_metric_before_number("seoul_bull_sculpture_07", "reproj")
            == "seoul_bull_sculpture_reproj_07"
        )

    def test_three_digit_number(self):
        assert _insert_metric_before_number("image_001", "angle") == "image_angle_001"

    def test_no_separator(self):
        assert _insert_metric_before_number("frame12", "tracks") == "frame_tracks12"

    def test_no_trailing_number(self):
        assert _insert_metric_before_number("noNumber", "reproj") == "noNumber_reproj"

    def test_dash_separator(self):
        assert _insert_metric_before_number("img-03", "tracks") == "img_tracks-03"


# =============================================================================
# Triangulation angle tests
# =============================================================================


class TestComputeTriangulationAngles:
    def test_two_cameras_known_angle(self):
        """Two cameras separated along X axis, point on Z axis."""
        from sfmtool._sfmtool import RotQuaternion

        # Identity rotation for both cameras
        q = RotQuaternion.identity()
        qwxyz = q.to_wxyz_array()
        quaternions = np.array([qwxyz, qwxyz])

        # Camera 0 at origin, Camera 1 at (2, 0, 0)
        # For identity rotation, camera center = -R^T @ t = -t
        translations = np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])

        # Point at (1, 0, 1) — equidistant from both cameras
        positions = np.array([[1.0, 0.0, 1.0]])

        track_image_indexes = np.array([0, 1], dtype=np.uint32)
        track_point_ids = np.array([0, 0], dtype=np.uint32)

        angles = compute_triangulation_angles(
            positions,
            quaternions,
            translations,
            track_image_indexes,
            track_point_ids,
        )

        # Expected angle: arctan(1/1) * 2 = 90 degrees
        assert angles.shape == (1,)
        assert abs(angles[0] - 90.0) < 1.0

    def test_single_observer_zero_angle(self):
        """Point observed by only one camera should have zero angle."""
        from sfmtool._sfmtool import RotQuaternion

        q = RotQuaternion.identity()
        qwxyz = q.to_wxyz_array()
        quaternions = np.array([qwxyz])
        translations = np.array([[0.0, 0.0, 0.0]])
        positions = np.array([[1.0, 0.0, 5.0]])

        track_image_indexes = np.array([0], dtype=np.uint32)
        track_point_ids = np.array([0], dtype=np.uint32)

        angles = compute_triangulation_angles(
            positions,
            quaternions,
            translations,
            track_image_indexes,
            track_point_ids,
        )

        assert angles[0] == 0.0

    def test_multiple_points(self):
        """Multiple points with different observer counts."""
        from sfmtool._sfmtool import RotQuaternion

        q = RotQuaternion.identity()
        qwxyz = q.to_wxyz_array()
        quaternions = np.array([qwxyz, qwxyz, qwxyz])
        translations = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-1.0, -2.0, 0.0],
            ]
        )
        positions = np.array(
            [
                [1.0, 0.0, 5.0],  # Point 0: observed by 2 cameras
                [0.0, 0.0, 3.0],  # Point 1: observed by 3 cameras
            ]
        )

        track_image_indexes = np.array([0, 1, 0, 1, 2], dtype=np.uint32)
        track_point_ids = np.array([0, 0, 1, 1, 1], dtype=np.uint32)

        angles = compute_triangulation_angles(
            positions,
            quaternions,
            translations,
            track_image_indexes,
            track_point_ids,
        )

        assert angles.shape == (2,)
        assert angles[0] > 0
        assert angles[1] > 0
        # Point with 3 observers should have >= angle of 2-observer point
        # (more baselines means at least as large a max angle)


# =============================================================================
# Heatmap renderer tests
# =============================================================================


class TestRenderHeatmapOverlay:
    def test_basic_render(self, tmp_path):
        """Basic rendering produces an output image."""
        # Create a test image
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        positions = np.array([[50.0, 50.0], [150.0, 50.0]], dtype=np.float32)
        values = np.array([0.0, 1.0], dtype=np.float64)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(
            img_path,
            positions,
            values,
            output_path,
            colormap="error",
            radius=5,
            alpha=0.7,
        )

        assert output_path.exists()
        result = cv2.imread(str(output_path))
        assert result is not None
        # Wider than input due to colorbar
        assert result.shape[1] > 200

    def test_no_colorbar(self, tmp_path):
        """Rendering without colorbar should match input width."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        values = np.array([0.5], dtype=np.float64)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(
            img_path,
            positions,
            values,
            output_path,
            show_colorbar=False,
        )

        result = cv2.imread(str(output_path))
        assert result.shape[1] == 200

    def test_nan_values_handled(self, tmp_path):
        """NaN values should not crash rendering."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        positions = np.array([[50.0, 50.0], [100.0, 50.0]], dtype=np.float32)
        values = np.array([np.nan, 1.0], dtype=np.float64)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(img_path, positions, values, output_path)
        assert output_path.exists()

    def test_all_nan_saves_original(self, tmp_path):
        """All NaN values should save the original image."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        values = np.array([np.nan], dtype=np.float64)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(img_path, positions, values, output_path)
        assert output_path.exists()

    def test_missing_image_raises(self, tmp_path):
        output_path = tmp_path / "output.png"
        with pytest.raises(FileNotFoundError):
            render_heatmap_overlay(
                tmp_path / "nonexistent.jpg",
                np.array([[0.0, 0.0]]),
                np.array([0.0]),
                output_path,
            )


# =============================================================================
# CLI tests
# =============================================================================


class TestHeatmapCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["heatmap", "--help"])
        assert result.exit_code == 0
        assert "Visualize reconstruction quality" in result.output

    def test_non_sfmr_rejected(self, tmp_path):
        recon = tmp_path / "recon.txt"
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "heatmap",
                str(recon),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_reconstruction_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "heatmap",
                str(tmp_path / "nonexistent.sfmr"),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_output_required(self):
        runner = CliRunner()
        result = runner.invoke(main, ["heatmap", "dummy.sfmr"])
        assert result.exit_code != 0


class TestHeatmapE2E:
    """End-to-end test using the Seoul Bull dataset."""

    def test_heatmap_reproj(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Generate reprojection error heatmaps for a real reconstruction."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        output_dir = tmp_path / "heatmaps"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "heatmap",
                str(sfmr_path),
                "-o",
                str(output_dir),
                "--metric",
                "reproj",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert output_dir.exists()
        # Should have generated at least one heatmap PNG
        pngs = list(output_dir.glob("*.png"))
        assert len(pngs) > 0, f"No PNGs generated. Output: {result.output}"

    def test_heatmap_all_metrics(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Generate all metric heatmaps."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        output_dir = tmp_path / "heatmaps_all"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "heatmap",
                str(sfmr_path),
                "-o",
                str(output_dir),
                "--metric",
                "all",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        pngs = list(output_dir.glob("*.png"))
        # 3 metrics x N images = at least 3 PNGs
        assert len(pngs) >= 3, f"Expected at least 3 PNGs, got {len(pngs)}"
