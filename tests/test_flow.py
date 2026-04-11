# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flow command and flow visualization utilities."""


import cv2
import numpy as np
from click.testing import CliRunner

from sfmtool._sfmtool import compute_optical_flow as _rust_compute_optical_flow
from sfmtool.feature_match._flow_matching import _flow_match_pair, flow_match_sequential
from sfmtool._flow_viz import (
    _find_nearest_within_tolerance,
    _flow_to_color,
    _get_color_palette,
    _save_output,
)
from sfmtool.cli import main


# =============================================================================
# Helpers
# =============================================================================


def _create_textured_image(width: int, height: int, seed: int = 42) -> np.ndarray:
    """Create a textured grayscale image with features suitable for optical flow."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width), dtype=np.uint8)
    img[:] = (rng.rand(height, width) * 60 + 30).astype(np.uint8)

    for _ in range(50):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        r = rng.randint(5, 20)
        color = int(rng.randint(100, 255))
        cv2.circle(img, (cx, cy), r, color, -1)

    for _ in range(30):
        x1 = rng.randint(0, width - 40)
        y1 = rng.randint(0, height - 40)
        x2 = x1 + rng.randint(10, 40)
        y2 = y1 + rng.randint(10, 40)
        color = int(rng.randint(50, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return img


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift an image by (dx, dy) pixels using affine transform."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# =============================================================================
# Unit tests: flow visualization utilities
# =============================================================================


class TestGetColorPalette:
    def test_count(self):
        colors = _get_color_palette(10)
        assert len(colors) == 10

    def test_valid_bgr(self):
        colors = _get_color_palette(5)
        for b, g, r in colors:
            assert 0 <= b <= 255
            assert 0 <= g <= 255
            assert 0 <= r <= 255

    def test_single(self):
        colors = _get_color_palette(1)
        assert len(colors) == 1

    def test_deterministic(self):
        c1 = _get_color_palette(8)
        c2 = _get_color_palette(8)
        assert c1 == c2


class TestFlowToColor:
    def test_shape_and_dtype(self):
        flow_u = np.zeros((100, 200), dtype=np.float32)
        flow_v = np.zeros((100, 200), dtype=np.float32)
        result = _flow_to_color(flow_u, flow_v)
        assert result.shape == (100, 200, 3)
        assert result.dtype == np.uint8

    def test_zero_flow_is_white(self):
        flow_u = np.zeros((50, 50), dtype=np.float32)
        flow_v = np.zeros((50, 50), dtype=np.float32)
        result = _flow_to_color(flow_u, flow_v)
        # Zero magnitude → saturation=0, value=255 → white in BGR
        center = result[25, 25]
        assert center[0] == 255 and center[1] == 255 and center[2] == 255

    def test_nonzero_flow_has_color(self):
        flow_u = np.full((50, 50), 10.0, dtype=np.float32)
        flow_v = np.zeros((50, 50), dtype=np.float32)
        result = _flow_to_color(flow_u, flow_v)
        center = result[25, 25]
        # Should not be white (saturation > 0)
        assert not (center[0] == 255 and center[1] == 255 and center[2] == 255)


class TestFindNearestWithinTolerance:
    def test_exact_match(self):
        query = np.array([[10.0, 20.0]], dtype=np.float32)
        target = np.array([[10.0, 20.0], [100.0, 200.0]], dtype=np.float32)
        result = _find_nearest_within_tolerance(query, target, tolerance=1.0)
        assert 0 in result
        assert result[0][0] == 0  # matched to first target
        assert result[0][1] < 0.01  # near-zero distance

    def test_no_match_beyond_tolerance(self):
        query = np.array([[10.0, 20.0]], dtype=np.float32)
        target = np.array([[100.0, 200.0]], dtype=np.float32)
        result = _find_nearest_within_tolerance(query, target, tolerance=1.0)
        assert len(result) == 0

    def test_empty_arrays(self):
        empty = np.zeros((0, 2), dtype=np.float32)
        non_empty = np.array([[1.0, 2.0]], dtype=np.float32)
        assert _find_nearest_within_tolerance(empty, non_empty, 1.0) == {}
        assert _find_nearest_within_tolerance(non_empty, empty, 1.0) == {}

    def test_multiple_matches(self):
        query = np.array([[10.0, 10.0], [20.0, 20.0], [100.0, 100.0]], dtype=np.float32)
        target = np.array([[10.5, 10.5], [20.5, 20.5]], dtype=np.float32)
        result = _find_nearest_within_tolerance(query, target, tolerance=2.0)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result
        assert 2 not in result


class TestSaveOutput:
    def test_side_by_side(self, tmp_path):
        vis1 = np.zeros((100, 200, 3), dtype=np.uint8)
        vis2 = np.ones((100, 200, 3), dtype=np.uint8) * 128
        output = tmp_path / "test.png"
        _save_output(vis1, vis2, output, side_by_side=True)
        assert output.exists()
        img = cv2.imread(str(output))
        assert img.shape[1] == 400  # side-by-side width

    def test_separate(self, tmp_path):
        vis1 = np.zeros((100, 200, 3), dtype=np.uint8)
        vis2 = np.ones((100, 200, 3), dtype=np.uint8) * 128
        output = tmp_path / "test.png"
        _save_output(vis1, vis2, output, side_by_side=False)
        assert (tmp_path / "test_A.png").exists()
        assert (tmp_path / "test_B.png").exists()

    def test_different_heights_side_by_side(self, tmp_path):
        vis1 = np.zeros((100, 200, 3), dtype=np.uint8)
        vis2 = np.ones((150, 200, 3), dtype=np.uint8) * 128
        output = tmp_path / "test.png"
        _save_output(vis1, vis2, output, side_by_side=True)
        assert output.exists()
        img = cv2.imread(str(output))
        assert img.shape[0] == 150  # max height


# =============================================================================
# Unit tests: flow match pair
# =============================================================================


class TestFlowMatchPair:
    """Test the core _flow_match_pair function."""

    def test_known_shift_finds_matches(self):
        """With a known horizontal shift, flow matching should find correct matches."""
        w, h = 200, 150
        dx, dy = 10.0, 0.0

        img1 = _create_textured_image(w, h, seed=42)
        img2 = _shift_image(img1, dx, dy)

        n_pts = 50
        rng = np.random.RandomState(99)
        positions1 = np.column_stack(
            [rng.uniform(30, w - 30, n_pts), rng.uniform(30, h - 30, n_pts)]
        ).astype(np.float32)

        positions2 = positions1.copy()
        positions2[:, 0] += dx
        positions2[:, 1] += dy

        descriptors1 = rng.randint(0, 256, (n_pts, 128), dtype=np.uint8)
        descriptors2 = descriptors1.copy()

        flow_u, flow_v = _rust_compute_optical_flow(img1, img2, preset="default")

        matches = _flow_match_pair(
            positions1,
            positions2,
            descriptors1,
            descriptors2,
            flow_u,
            flow_v,
            spatial_tolerance=3.0,
            descriptor_threshold=100.0,
        )

        assert len(matches) > 0, "Should find at least some matches"
        assert matches.dtype == np.uint32
        assert matches.shape[1] == 2

        for src_idx, dst_idx in matches:
            assert src_idx == dst_idx, (
                f"Source {src_idx} matched to {dst_idx}, expected self-match"
            )

    def test_no_matches_with_unrelated_images(self):
        """Images with no relation should produce few or no matches."""
        w, h = 200, 150
        img1 = _create_textured_image(w, h, seed=42)
        img2 = _create_textured_image(w, h, seed=999)

        n_pts = 30
        rng = np.random.RandomState(99)
        positions1 = np.column_stack(
            [rng.uniform(20, w - 20, n_pts), rng.uniform(20, h - 20, n_pts)]
        ).astype(np.float32)
        positions2 = np.column_stack(
            [rng.uniform(20, w - 20, n_pts), rng.uniform(20, h - 20, n_pts)]
        ).astype(np.float32)

        descriptors1 = rng.randint(0, 256, (n_pts, 128), dtype=np.uint8)
        descriptors2 = rng.randint(0, 256, (n_pts, 128), dtype=np.uint8)

        flow_u, flow_v = _rust_compute_optical_flow(img1, img2, preset="fast")

        matches = _flow_match_pair(
            positions1,
            positions2,
            descriptors1,
            descriptors2,
            flow_u,
            flow_v,
            spatial_tolerance=3.0,
            descriptor_threshold=50.0,
        )

        assert len(matches) <= 5, f"Expected few matches, got {len(matches)}"

    def test_empty_positions(self):
        """Empty position arrays should return empty matches."""
        flow_u = np.zeros((100, 100), dtype=np.float32)
        flow_v = np.zeros((100, 100), dtype=np.float32)

        matches = _flow_match_pair(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((10, 2), dtype=np.float32),
            np.zeros((0, 128), dtype=np.uint8),
            np.zeros((10, 128), dtype=np.uint8),
            flow_u,
            flow_v,
            spatial_tolerance=3.0,
            descriptor_threshold=100.0,
        )

        assert len(matches) == 0
        assert matches.dtype == np.uint32

    def test_output_format(self):
        """Verify the output format: (M, 2) uint32 array."""
        w, h = 100, 100
        img1 = _create_textured_image(w, h, seed=42)
        img2 = _shift_image(img1, 5.0, 0.0)

        positions = np.array([[50.0, 50.0]], dtype=np.float32)
        positions2 = np.array([[55.0, 50.0]], dtype=np.float32)
        desc = np.zeros((1, 128), dtype=np.uint8)

        flow_u, flow_v = _rust_compute_optical_flow(img1, img2, preset="fast")

        matches = _flow_match_pair(
            positions,
            positions2,
            desc,
            desc,
            flow_u,
            flow_v,
            spatial_tolerance=5.0,
            descriptor_threshold=200.0,
        )

        assert matches.ndim == 2
        assert matches.shape[1] == 2
        assert matches.dtype == np.uint32


# =============================================================================
# CLI tests
# =============================================================================


class TestFlowCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["flow", "--help"])
        assert result.exit_code == 0
        assert "Visualize optical flow" in result.output

    def test_non_sfmr_reconstruction_rejected(self, tmp_path):
        """Passing a non-.sfmr file as --reconstruction should error."""
        img = tmp_path / "img.jpg"
        recon = tmp_path / "recon.txt"
        img.write_bytes(b"fake")
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "flow",
                str(img),
                str(img),
                "-r",
                str(recon),
            ],
        )
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_image_errors(self, tmp_path):
        """Non-existent image paths should fail."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "flow",
                str(tmp_path / "nonexistent1.jpg"),
                str(tmp_path / "nonexistent2.jpg"),
            ],
        )
        assert result.exit_code != 0


class TestFlowE2E:
    """End-to-end tests using the Seoul Bull dataset."""

    def test_flow_match_adjacent(self, isolated_seoul_bull_17_images, tmp_path):
        """Flow matching on adjacent Seoul Bull frames should produce matches."""
        image_paths = isolated_seoul_bull_17_images[:5]
        workspace_dir = image_paths[0].parent

        from sfmtool import init_workspace
        from sfmtool._sift_file import image_files_to_sift_files

        init_workspace(workspace_dir, domain_size_pooling=True)
        sift_paths = image_files_to_sift_files(image_paths, feature_tool="colmap")

        all_matches = flow_match_sequential(
            image_paths=image_paths,
            sift_paths=sift_paths,
            preset="default",
            window_size=2,
        )

        assert len(all_matches) > 0, "Should find matches on real images"

        for (i, j), matches in all_matches.items():
            if j - i == 1:
                assert len(matches) >= 5, (
                    f"Pair ({i}, {j}) has only {len(matches)} matches"
                )
