# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI tests for `sfm to-nerfstudio`."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main


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
    import shutil

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
