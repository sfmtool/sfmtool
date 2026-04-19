# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for epipolar geometry visualization."""

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.visualization._epipolar_display import (
    _compute_fundamental_matrix,
    _draw_epipolar_line,
    _draw_polar_epipolar_line,
    _get_color_palette,
)
from sfmtool._sfmtool import CameraIntrinsics


# ===== Color Palette Tests =====


class TestGetColorPalette:
    def test_correct_count(self):
        colors = _get_color_palette(10)
        assert len(colors) == 10

    def test_valid_bgr_tuples(self):
        colors = _get_color_palette(10)
        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_distinct_colors(self):
        colors = _get_color_palette(10)
        colors_array = np.array(colors)
        std_dev = np.std(colors_array, axis=0)
        assert np.all(std_dev > 10), "Colors should have some variation"

    def test_single_color(self):
        colors = _get_color_palette(1)
        assert len(colors) == 1

    def test_deterministic(self):
        colors1 = _get_color_palette(5)
        colors2 = _get_color_palette(5)
        assert colors1 == colors2


# ===== Fundamental Matrix Tests =====


class TestComputeFundamentalMatrix:
    def test_rank_2(self):
        K1 = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
        K2 = K1.copy()
        R1 = np.eye(3)
        t1 = np.array([0.0, 0.0, 0.0])
        R2 = np.eye(3)
        t2 = np.array([1.0, 0.0, 0.0])

        F = _compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)

        assert F.shape == (3, 3)
        rank = np.linalg.matrix_rank(F)
        assert rank == 2

    def test_not_zero(self):
        K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
        R1 = np.eye(3)
        t1 = np.zeros(3)
        R2 = np.eye(3)
        t2 = np.array([1.0, 0.0, 0.0])

        F = _compute_fundamental_matrix(K, R1, t1, K, R2, t2)
        assert np.abs(F).max() > 1e-6

    def test_epipolar_constraint(self):
        """A 3D point projected into both cameras should satisfy x2^T F x1 = 0."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
        R1 = np.eye(3)
        t1 = np.zeros(3)
        R2 = np.eye(3)
        t2 = np.array([0.5, 0.0, 0.0])

        F = _compute_fundamental_matrix(K, R1, t1, K, R2, t2)

        # A point at [0, 0, 5] in world coords
        P = np.array([0.0, 0.0, 5.0])
        p1 = K @ (R1 @ P + t1)
        p1 = p1 / p1[2]
        p2 = K @ (R2 @ P + t2)
        p2 = p2 / p2[2]

        residual = p2 @ F @ p1
        assert abs(residual) < 1e-8


# ===== Epipolar Line Drawing Tests =====


class TestDrawEpipolarLine:
    def test_horizontal_line(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        line = np.array([0, 1, -50])  # y = 50
        _draw_epipolar_line(img, line, (255, 255, 255), 1)
        assert img[50, 100, 0] > 0  # Line should pass through (100, 50)

    def test_vertical_line(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        line = np.array([1, 0, -100])  # x = 100
        _draw_epipolar_line(img, line, (255, 255, 255), 1)
        assert img[50, 100, 0] > 0

    def test_line_outside_image(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        line = np.array([0, 1, -500])  # y = 500, outside image
        _draw_epipolar_line(img, line, (255, 255, 255), 1)
        assert img.sum() == 0  # No pixels drawn


class TestDrawPolarEpipolarLine:
    def test_draws_line(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        line = np.array([0, 1, -100])  # y = 100
        epipole = np.array([100, 100])
        feature_point = np.array([150, 100])
        _draw_polar_epipolar_line(img, line, epipole, feature_point, (255, 255, 255), 1)
        assert img.sum() > 0


# ===== Intrinsic Matrix Tests =====


class TestIntrinsicMatrix:
    def test_pinhole(self):
        camera = CameraIntrinsics(
            "PINHOLE",
            1920,
            1080,
            {
                "focal_length_x": 1000.0,
                "focal_length_y": 1000.0,
                "principal_point_x": 960.0,
                "principal_point_y": 540.0,
            },
        )
        K = camera.intrinsic_matrix()
        expected = np.array(
            [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(K, expected)

    def test_simple_radial(self):
        camera = CameraIntrinsics(
            "SIMPLE_RADIAL",
            1920,
            1080,
            {
                "focal_length": 1000.0,
                "principal_point_x": 960.0,
                "principal_point_y": 540.0,
                "radial_distortion_k1": 0.1,
            },
        )
        K = camera.intrinsic_matrix()
        expected = np.array(
            [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(K, expected)


# ===== Rectification Tests =====


class TestStereoRectification:
    def test_compute_rectification(self):
        import pycolmap
        from sfmtool._rectification import compute_stereo_rectification

        cam1 = pycolmap.Camera(
            model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240]
        )
        cam2 = pycolmap.Camera(
            model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240]
        )

        R_rel = np.eye(3)
        t_rel = np.array([1.0, 0.0, 0.0])

        rect = compute_stereo_rectification(cam1, cam2, cam1, cam2, R_rel, t_rel)

        assert rect.K1.shape == (3, 3)
        assert rect.K2.shape == (3, 3)
        assert rect.R1_rect.shape == (3, 3)
        assert rect.R2_rect.shape == (3, 3)
        assert rect.P1.shape == (3, 4)
        assert rect.P2.shape == (3, 4)
        assert rect.Q.shape == (4, 4)

    def test_rectify_points(self):
        import pycolmap
        from sfmtool._rectification import compute_stereo_rectification

        cam = pycolmap.Camera(
            model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240]
        )

        R_rel = np.eye(3)
        t_rel = np.array([1.0, 0.0, 0.0])

        rect = compute_stereo_rectification(cam, cam, cam, cam, R_rel, t_rel)

        # Principal point should map close to itself
        pts = np.array([[320.0, 240.0]])
        rect_pts = rect.rectify_points_1(pts)
        assert rect_pts.shape == (1, 2)


# ===== Resolve Image Name Tests =====


class TestResolveImageName:
    def test_filename_passthrough(self):
        from sfmtool._commands.epipolar import resolve_image_name

        class FakeRecon:
            image_names = ["img_001.jpg", "img_002.jpg"]

        assert resolve_image_name("img_001.jpg", FakeRecon()) == "img_001.jpg"

    def test_number_resolution(self):
        from sfmtool._commands.epipolar import resolve_image_name

        class FakeRecon:
            image_names = ["test/img_001.jpg", "test/img_002.jpg"]

        result = resolve_image_name("1", FakeRecon())
        assert result == "test/img_001.jpg"

    def test_number_not_found(self):
        from sfmtool._commands.epipolar import resolve_image_name

        class FakeRecon:
            image_names = ["test/img_001.jpg", "test/img_002.jpg"]

        with pytest.raises(Exception, match="not found"):
            resolve_image_name("99", FakeRecon())

    def test_ambiguous_number(self):
        from sfmtool._commands.epipolar import resolve_image_name

        class FakeRecon:
            image_names = ["a/img_001.jpg", "b/img_001.jpg"]

        with pytest.raises(Exception, match="ambiguous"):
            resolve_image_name("1", FakeRecon())


# ===== get_image_hint_message Tests =====


class TestGetImageHintMessage:
    def test_basic_message(self):
        from sfmtool._sfm_reconstruction import get_image_hint_message

        class FakeRecon:
            image_names = ["test/img_001.jpg", "test/img_002.jpg"]

        msg = get_image_hint_message(FakeRecon())
        assert "Image in reconstruction" in msg

    def test_missing_image_message(self):
        from sfmtool._sfm_reconstruction import get_image_hint_message

        class FakeRecon:
            image_names = ["test/img_001.jpg", "test/img_002.jpg"]

        msg = get_image_hint_message(FakeRecon(), "img_003.jpg")
        assert "not found" in msg

    def test_similar_names_suggestion(self):
        from sfmtool._sfm_reconstruction import get_image_hint_message

        class FakeRecon:
            image_names = ["test/img_001.jpg", "test/img_002.jpg"]

        msg = get_image_hint_message(FakeRecon(), "img_001.jpg")
        assert "Did you mean" in msg


# ===== CLI Tests =====


class TestEpipolarCLI:
    def test_help(self):
        from sfmtool.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["epipolar", "--help"])
        assert result.exit_code == 0
        assert "epipolar" in result.output.lower()

    def test_non_sfmr_rejected(self, tmp_path):
        from sfmtool.cli import main

        runner = CliRunner()
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello")
        result = runner.invoke(
            main,
            ["epipolar", str(input_file), "img1.jpg", "img2.jpg", "--draw", "out.png"],
        )
        assert result.exit_code != 0
        assert (
            "sfmr" in result.output.lower() or "sfmr" in str(result.exception).lower()
        )

    def test_rectify_undistort_mutually_exclusive(self, tmp_path):
        from sfmtool.cli import main

        runner = CliRunner()
        sfmr_file = tmp_path / "test.sfmr"
        sfmr_file.write_text("dummy")
        result = runner.invoke(
            main,
            [
                "epipolar",
                str(sfmr_file),
                "img1.jpg",
                "img2.jpg",
                "--draw",
                str(tmp_path / "out.png"),
                "--rectify",
                "--undistort",
            ],
        )
        assert result.exit_code != 0
        assert (
            "mutually exclusive" in result.output.lower()
            or "mutually exclusive" in str(result.exception).lower()
        )

    def test_pairs_dir_exclusive_with_images(self, tmp_path):
        from sfmtool.cli import main

        runner = CliRunner()
        sfmr_file = tmp_path / "test.sfmr"
        sfmr_file.write_text("dummy")
        result = runner.invoke(
            main,
            [
                "epipolar",
                str(sfmr_file),
                "img1.jpg",
                "img2.jpg",
                "--pairs-dir",
                str(tmp_path / "pairs"),
            ],
        )
        assert result.exit_code != 0

    def test_sweep_window_without_features_rejected(self, tmp_path):
        from sfmtool.cli import main

        runner = CliRunner()
        sfmr_file = tmp_path / "test.sfmr"
        sfmr_file.write_text("dummy")
        result = runner.invoke(
            main,
            [
                "epipolar",
                str(sfmr_file),
                "img1.jpg",
                "img2.jpg",
                "--draw",
                str(tmp_path / "out.png"),
                "--sweep-window-size",
                "50",
            ],
        )
        assert result.exit_code != 0


# ===== E2E Tests =====


class TestEpipolarE2E:
    def test_draw_epipolar_visualization(self, sfmrfile_reconstruction_with_17_images):
        """Test drawing epipolar visualization with a real reconstruction."""
        import cv2

        from sfmtool.visualization._epipolar_display import draw_epipolar_visualization
        from sfmtool._sfmtool import SfmrReconstruction
        from sfmtool._workspace import load_workspace_config

        sfmr_path = sfmrfile_reconstruction_with_17_images
        workspace_dir = sfmr_path.parent
        recon = SfmrReconstruction.load(sfmr_path)
        workspace_config = load_workspace_config(workspace_dir)

        image_names = recon.image_names
        assert len(image_names) >= 2

        output_path = workspace_dir / "epipolar_test.png"

        draw_epipolar_visualization(
            recon=recon,
            image1_name=image_names[0],
            image2_name=image_names[1],
            output_path=str(output_path),
            max_features=10,
            feature_tool=workspace_config["feature_tool"],
            feature_options=workspace_config["feature_options"],
        )

        assert output_path.exists()
        img = cv2.imread(str(output_path))
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_draw_side_by_side(self, sfmrfile_reconstruction_with_17_images):
        """Test side-by-side epipolar visualization."""
        import cv2

        from sfmtool.visualization._epipolar_display import draw_epipolar_visualization
        from sfmtool._sfmtool import SfmrReconstruction
        from sfmtool._workspace import load_workspace_config

        sfmr_path = sfmrfile_reconstruction_with_17_images
        workspace_dir = sfmr_path.parent
        recon = SfmrReconstruction.load(sfmr_path)
        workspace_config = load_workspace_config(workspace_dir)

        image_names = recon.image_names
        output_path = workspace_dir / "epipolar_sbs.png"

        draw_epipolar_visualization(
            recon=recon,
            image1_name=image_names[0],
            image2_name=image_names[1],
            output_path=str(output_path),
            max_features=5,
            side_by_side=True,
            feature_tool=workspace_config["feature_tool"],
            feature_options=workspace_config["feature_options"],
        )

        assert output_path.exists()
        img = cv2.imread(str(output_path))
        assert img is not None
        # Side-by-side should be wider than a single image
        single_img = cv2.imread(str(workspace_dir / image_names[0]))
        if single_img is not None:
            assert img.shape[1] > single_img.shape[1]
