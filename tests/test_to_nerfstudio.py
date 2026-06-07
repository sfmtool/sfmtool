# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `sfm to-nerfstudio`: core helpers and end-to-end CLI."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._to_nerfstudio import (
    _APPLIED_TRANSFORM_3x4,
    apply_transform_to_points,
    frame_transform_matrix,
    write_sparse_ply,
)
from sfmtool.cli import main


class TestFrameTransformMatrix:
    def test_identity_pose_yields_applied_transform(self):
        """Identity cam-from-world inverts to identity world-from-cam.

        After flipping Y/Z and applying the nerfstudio applied_transform, the
        result should match _APPLIED_TRANSFORM_3x4 with [-1, -1] flips on the
        OpenGL columns.
        """
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity wxyz
        t = np.zeros(3)
        m = frame_transform_matrix(q, t)
        assert m.shape == (4, 4)
        np.testing.assert_array_almost_equal(m[3], [0.0, 0.0, 0.0, 1.0])

        # With identity input, world_from_cam is identity; after flipping Y/Z
        # columns we get diag(1, -1, -1, 1). Then applied_transform left-mults.
        flipped = np.diag([1.0, -1.0, -1.0, 1.0])
        expected = np.vstack([_APPLIED_TRANSFORM_3x4, [0, 0, 0, 1]]) @ flipped
        np.testing.assert_array_almost_equal(m, expected)

    def test_translation_only_pose(self):
        """A translation in cam-from-world flips sign in world-from-cam, then
        gets remapped by the applied_transform. Verify the final translation
        column matches the manual computation."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        t = np.array([1.0, 2.0, 3.0])  # cam_from_world translation
        m = frame_transform_matrix(q, t)

        # world_from_cam translation = -R^T @ t = -t (since R=I)
        # Then [:, 1] *= -1 leaves translation column unchanged (only flips
        # rotation columns 1,2). applied_transform permutes Y<->Z, negates Y.
        # So expected translation = applied_transform[:3, :3] @ (-t).
        wfc_t = -t
        expected_t = _APPLIED_TRANSFORM_3x4[:, :3] @ wfc_t
        np.testing.assert_array_almost_equal(m[:3, 3], expected_t)


class TestApplyTransformToPoints:
    def test_axes_permuted(self):
        """applied_transform is [[1,0,0,0],[0,0,1,0],[0,-1,0,0]]:
        x -> x, y -> -z (output y), z -> y (output z) ... actually:
        out_x = x; out_y = z; out_z = -y.
        """
        pts = np.array([[1.0, 2.0, 3.0]])
        out = apply_transform_to_points(pts)
        np.testing.assert_array_almost_equal(out, [[1.0, 3.0, -2.0]])

    def test_empty_input(self):
        out = apply_transform_to_points(np.zeros((0, 3)))
        assert out.shape == (0, 3)


class TestWriteSparsePly:
    def test_header_and_count(self, tmp_path):
        pts = np.array([[1.5, 2.5, 3.5], [-1.0, 0.0, 1.0]])
        cols = np.array([[10, 20, 30], [255, 0, 128]], dtype=np.uint8)
        out = tmp_path / "x.ply"
        write_sparse_ply(out, pts, cols)
        text = out.read_text(encoding="ascii")
        lines = text.splitlines()
        assert lines[0] == "ply"
        assert lines[1] == "format ascii 1.0"
        assert lines[2] == "element vertex 2"
        assert "end_header" in lines
        # 10 header lines (ply, format, element, 6 properties, end_header)
        # + 2 data lines = 12 total
        assert len(lines) == 12
        assert lines[10] == "1.500000 2.500000 3.500000 10 20 30"
        assert lines[11] == "-1.000000 0.000000 1.000000 255 0 128"

    def test_shape_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError):
            write_sparse_ply(
                tmp_path / "bad.ply",
                np.zeros((3, 3)),
                np.zeros((2, 3), dtype=np.uint8),
            )


def _undistort_to_pinhole(sfmr_path: Path) -> Path:
    """Run `sfm undistort` and return the path to the pinhole .sfmr it writes."""
    runner = CliRunner()
    result = runner.invoke(main, ["undistort", str(sfmr_path)])
    assert result.exit_code == 0, f"undistort failed: {result.output}"
    pinhole = (
        sfmr_path.parent / f"{sfmr_path.stem}_undistorted" / "sfmr" / "undistorted.sfmr"
    )
    assert pinhole.exists(), f"undistort did not produce {pinhole}"
    return pinhole


@pytest.fixture(scope="session")
def pinhole_sfmr_17_images_once(
    sfmrfile_reconstruction_with_17_images_once,
) -> Path:
    """Session-scoped pinhole .sfmr (undistort runs once for the whole suite)."""
    return _undistort_to_pinhole(sfmrfile_reconstruction_with_17_images_once)


@pytest.fixture
def pinhole_sfmr_17_images(pinhole_sfmr_17_images_once, tmp_path_factory) -> Path:
    """Per-test isolated copy of the pinhole .sfmr workspace."""
    source_workspace = pinhole_sfmr_17_images_once.parent.parent
    dest = tmp_path_factory.mktemp("pinhole_workspace")
    shutil.copytree(source_workspace, dest, dirs_exist_ok=True)
    return dest / "sfmr" / pinhole_sfmr_17_images_once.name


class TestToNerfstudioCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["to-nerfstudio", "--help"])
        assert result.exit_code == 0
        assert "Nerfstudio" in result.output
        assert "--num-downscales" in result.output

    def test_non_sfmr_rejected(self, tmp_path):
        bad = tmp_path / "not_a_recon.txt"
        bad.write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(main, ["to-nerfstudio", str(bad)])
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_input_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["to-nerfstudio", str(tmp_path / "missing.sfmr")])
        assert result.exit_code != 0

    def test_distorted_input_rejected(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(sfmrfile_reconstruction_with_17_images),
                "-o",
                str(tmp_path / "ns_out"),
            ],
        )
        assert result.exit_code != 0
        assert (
            "distortion" in result.output.lower()
            or "undistort" in result.output.lower()
        )


class TestToNerfstudioE2E:
    def test_basic_output_layout(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--num-downscales",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output

        assert (out / "transforms.json").exists()
        assert (out / "sparse_pc.ply").exists()
        assert (out / "images").is_dir()
        assert (out / "images_2").is_dir()
        assert (out / "images_4").is_dir()
        assert not (out / "images_8").exists()
        assert not (out / "sparse").exists()  # --include-colmap not set

    def test_transforms_json_schema(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--num-downscales",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        with open(out / "transforms.json") as f:
            data = json.load(f)

        # Single-camera reconstruction hoists intrinsics.
        for key in (
            "w",
            "h",
            "fl_x",
            "fl_y",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
            "camera_model",
        ):
            assert key in data, f"missing top-level key: {key}"
        assert data["camera_model"] == "OPENCV"
        assert data["k1"] == 0.0 and data["k2"] == 0.0
        assert data["p1"] == 0.0 and data["p2"] == 0.0

        # applied_transform always present
        applied = data["applied_transform"]
        assert len(applied) == 3 and len(applied[0]) == 4
        assert data["ply_file_path"] == "sparse_pc.ply"

        # frames have expected fields and 1-based contiguous colmap_im_id
        frames = data["frames"]
        assert len(frames) == 17
        for i, frame in enumerate(frames):
            assert frame["colmap_im_id"] == i + 1
            assert frame["file_path"].startswith("images/")
            mat = frame["transform_matrix"]
            assert len(mat) == 4 and len(mat[0]) == 4
            assert mat[3] == [0, 0, 0, 1]

    def test_image_files_present(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--num-downscales",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output

        with open(out / "transforms.json") as f:
            data = json.load(f)

        for frame in data["frames"]:
            full = out / frame["file_path"]
            assert full.exists(), f"missing original image: {full}"
            base = Path(frame["file_path"]).name
            assert (out / "images_2" / base).exists()

    def test_ply_has_header_and_vertices(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--num-downscales",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        text = (out / "sparse_pc.ply").read_text(encoding="ascii")
        lines = text.splitlines()
        assert lines[0] == "ply"
        assert lines[1] == "format ascii 1.0"
        # element vertex N
        vertex_line = next(line for line in lines if line.startswith("element vertex "))
        n = int(vertex_line.split()[-1])
        assert n > 0
        # header is 10 lines; total = 10 + n
        assert len(lines) == 10 + n

    def test_range_filters_frames(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--num-downscales",
                "0",
                "-r",
                "1-5",
            ],
        )
        assert result.exit_code == 0, result.output

        with open(out / "transforms.json") as f:
            data = json.load(f)

        # Only the five matching images should be exported.
        frames = data["frames"]
        assert len(frames) == 5

        images_dir = out / "images"
        assert sum(1 for _ in images_dir.iterdir()) == 5
        for frame in frames:
            assert (out / frame["file_path"]).exists()

    def test_range_with_no_matches_errors(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "-r",
                "100-200",
            ],
        )
        assert result.exit_code != 0
        assert "100-200" in result.output
        assert "Available file numbers" in result.output

    def test_filter_points_without_range_errors(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--filter-points",
            ],
        )
        assert result.exit_code != 0
        assert "--filter-points" in result.output
        assert "--range" in result.output

    def test_include_colmap_writes_sparse(self, pinhole_sfmr_17_images, tmp_path):
        out = tmp_path / "ns_dataset"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "to-nerfstudio",
                str(pinhole_sfmr_17_images),
                "-o",
                str(out),
                "--num-downscales",
                "0",
                "--include-colmap",
            ],
        )
        assert result.exit_code == 0, result.output

        sparse = out / "sparse"
        assert sparse.is_dir()
        assert (sparse / "cameras.bin").exists()
        assert (sparse / "images.bin").exists()
        assert (sparse / "points3D.bin").exists()
