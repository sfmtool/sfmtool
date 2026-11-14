# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the heatmap command and visualization utilities."""

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._commands.heatmap import (
    _insert_metric_before_number,
    _observations_by_image,
    _output_stem,
    _max_ray_angles_deg,
)
from sfmtool.cli import main
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.fileio import read_image_rgb, write_image_rgb
from sfmtool.visualization import (
    COLORMAPS,
    apply_colormap,
    render_heatmap_overlay,
    value_to_color,
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


def _observations_by_image_one_at_a_time(images, features, points):
    """The grouping read one observation at a time, as the reference."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for image, feature, point in zip(images, features, points):
        grouped.setdefault(int(image), []).append((int(feature), int(point)))
    return {
        image: (
            np.array([f for f, _ in grouped[image]], dtype=np.int32),
            np.array([p for _, p in grouped[image]], dtype=np.int32),
        )
        for image in sorted(grouped)
    }


def _assert_same_grouping(actual, expected):
    assert list(actual) == list(expected)
    for image, (features, points) in expected.items():
        assert actual[image][0].dtype == np.int32
        assert actual[image][1].dtype == np.int32
        np.testing.assert_array_equal(actual[image][0], features)
        np.testing.assert_array_equal(actual[image][1], points)


class TestObservationsByImage:
    def test_groups_in_track_order(self):
        # Point-major track arrays visit images out of order; each image keeps
        # its observations in track order, and image 1, with none, is absent.
        images = np.array([2, 0, 3, 0, 2, 3, 0], dtype=np.uint32)
        features = np.array([5, 9, 1, 4, 7, 2, 8], dtype=np.uint32)
        points = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.uint32)
        grouped = _observations_by_image(images, features, points)
        assert list(grouped) == [0, 2, 3]
        np.testing.assert_array_equal(grouped[0][0], [9, 4, 8])
        np.testing.assert_array_equal(grouped[0][1], [0, 1, 2])
        _assert_same_grouping(
            grouped, _observations_by_image_one_at_a_time(images, features, points)
        )

    def test_empty(self):
        empty = np.zeros(0, dtype=np.uint32)
        assert _observations_by_image(empty, empty, empty) == {}

    def test_matches_one_at_a_time_on_a_reconstruction(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        arrays = (
            recon.track_image_indexes,
            recon.track_feature_indexes,
            recon.track_point_indexes,
        )
        _assert_same_grouping(
            _observations_by_image(*arrays),
            _observations_by_image_one_at_a_time(*arrays),
        )


class TestOutputStem:
    def test_flat_image_name(self):
        assert (
            _output_stem("images/seoul_bull_07.jpg", "reproj")
            == "images__seoul_bull_reproj_07"
        )

    def test_rig_sensors_do_not_collide(self):
        left = _output_stem("fisheye_left/frame_05.jpg", "reproj")
        right = _output_stem("fisheye_right/frame_05.jpg", "reproj")
        assert left == "fisheye_left__frame_reproj_05"
        assert right == "fisheye_right__frame_reproj_05"
        assert left != right


# =============================================================================
# Triangulation angle tests
# =============================================================================


def _expected_max_ray_angles_deg(recon) -> dict[int, float]:
    """Each finite point's widest angle between two of its viewing rays, in
    degrees, computed here from the camera centres and stored positions."""
    from sfmtool.geometry import RotQuaternion

    quats = recon.quaternions_wxyz
    translations = recon.translations
    centres = np.array(
        [
            RotQuaternion.from_wxyz_array(q).camera_center(t)
            for q, t in zip(quats, translations)
        ]
    )
    xyzw = recon.positions_xyzw
    rays: dict[int, list[np.ndarray]] = {}
    for img, pt in zip(recon.track_image_indexes, recon.track_point_indexes):
        if xyzw[pt, 3] == 0.0:
            continue
        ray = xyzw[pt, :3] / xyzw[pt, 3] - centres[img]
        rays.setdefault(int(pt), []).append(ray / np.linalg.norm(ray))
    out = {}
    for pt, rs in rays.items():
        r = np.array(rs)
        dots = np.clip(r @ r.T, -1.0, 1.0)
        out[pt] = float(np.degrees(np.arccos(dots.min())))
    return out


class TestMaxRayAngles:
    def test_finite_points_match_their_viewing_rays(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        angles = _max_ray_angles_deg(recon)
        expected = _expected_max_ray_angles_deg(recon)
        assert angles.shape == (recon.point_count,)
        assert len(expected) > 0
        idx = np.array(list(expected))
        np.testing.assert_allclose(angles[idx], [expected[i] for i in idx], atol=1e-3)
        assert (angles[idx] > 0.0).any()

    def test_point_at_infinity_has_zero_angle(self, seoul_bull_sfmr_only):
        """A direction is seen along the same ray from every camera, so its
        angle is 0, not the spread of rays to a point one unit from the
        origin."""
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        counts = np.asarray(recon.observation_counts)
        point = int(np.argmax(counts))
        assert counts[point] >= 2
        xyzw = recon.positions_xyzw.copy()
        xyzw[point] = [0.0, 0.0, 1.0, 0.0]
        recon = recon.clone_with_changes(positions=xyzw)
        assert recon.point_is_at_infinity[point]

        angles = _max_ray_angles_deg(recon)
        assert angles[point] == 0.0


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
        write_image_rgb(img_path, img)

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
        result = read_image_rgb(output_path)
        assert result is not None
        # Wider than input due to colorbar
        assert result.shape[1] > 200

    def test_no_colorbar(self, tmp_path):
        """Rendering without colorbar should match input width."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        write_image_rgb(img_path, img)

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

        result = read_image_rgb(output_path)
        assert result.shape[1] == 200

    def test_nan_values_handled(self, tmp_path):
        """NaN values should not crash rendering."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        write_image_rgb(img_path, img)

        positions = np.array([[50.0, 50.0], [100.0, 50.0]], dtype=np.float32)
        values = np.array([np.nan, 1.0], dtype=np.float64)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(img_path, positions, values, output_path)
        assert output_path.exists()

    def test_all_nan_saves_original(self, tmp_path):
        """All NaN values should save the original image."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        img_path = tmp_path / "test.jpg"
        write_image_rgb(img_path, img)

        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        values = np.array([np.nan], dtype=np.float64)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(img_path, positions, values, output_path)
        assert output_path.exists()

    def test_overlay_keeps_the_colormap_colour(self, tmp_path):
        """A feature at the top of the "error" map is drawn red, not blue."""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "grey.png"
        write_image_rgb(img_path, img)

        output_path = tmp_path / "output.png"
        render_heatmap_overlay(
            img_path,
            np.array([[150.0, 50.0]], dtype=np.float32),
            np.array([1.0]),
            output_path,
            colormap="error",
            vmin=0.0,
            vmax=1.0,
            radius=6,
            alpha=1.0,
            show_colorbar=False,
        )

        result = read_image_rgb(output_path)
        assert tuple(result[50, 150]) == (255, 0, 0)
        assert tuple(result[10, 10]) == (128, 128, 128)

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

    def test_writes_one_image_per_observing_image(self, seoul_bull_workspace, tmp_path):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            main,
            [
                "heatmap",
                str(seoul_bull_workspace),
                "-o",
                str(out),
                "--metric",
                "tracks",
            ],
        )
        assert result.exit_code == 0, result.output
        observed = np.unique(recon.track_image_indexes)
        names = recon.image_names
        expected = {f"{_output_stem(names[i], 'tracks')}.png" for i in observed}
        assert {p.name for p in out.iterdir()} == expected

    def test_embedded_patches_rejected(self, seoul_bull_workspace, tmp_path):
        from sfmtool.xform import ToEmbeddedPatchesTransform

        embedded = ToEmbeddedPatchesTransform().apply(
            SfmrReconstruction.load(seoul_bull_workspace)
        )
        path = seoul_bull_workspace.parent / "embedded.sfmr"
        embedded.save(path, operation="test")
        result = CliRunner().invoke(
            main, ["heatmap", str(path), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert "no feature indexes" in result.output


class TestHeatmapE2E:
    """End-to-end test using the Seoul Bull dataset."""

    def test_heatmap_reproj(self, seoul_bull_workspace, tmp_path):
        """Generate reprojection error heatmaps for a real reconstruction."""
        sfmr_path = seoul_bull_workspace
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

    def test_heatmap_all_metrics(self, seoul_bull_workspace, tmp_path):
        """Generate all metric heatmaps."""
        sfmr_path = seoul_bull_workspace
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
