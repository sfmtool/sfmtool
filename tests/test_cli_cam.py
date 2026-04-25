# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `sfm cam cp`."""

import json
from pathlib import Path

from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main


def test_cam_cp_single_camera(sfmrfile_reconstruction_with_17_images: Path, tmp_path):
    """Single-camera .sfmr → camera_config.json roundtrip."""
    out = tmp_path / "camera_config.json"
    result = CliRunner().invoke(
        main, ["cam", "cp", str(sfmrfile_reconstruction_with_17_images), str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    config = json.loads(out.read_text())
    assert config["version"] == 1
    block = config["camera_intrinsics"]
    assert "model" in block
    assert "width" in block
    assert "height" in block
    assert "parameters" in block

    # The values written should match the reconstruction's camera 0
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    cam_dict = recon.cameras[0].to_dict()
    assert block["model"] == cam_dict["model"]
    assert block["width"] == cam_dict["width"]
    assert block["height"] == cam_dict["height"]
    for name, value in block["parameters"].items():
        assert value == cam_dict["parameters"][name]


def test_cam_cp_creates_parent_dirs(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path
):
    out = tmp_path / "nested" / "dir" / "camera_config.json"
    result = CliRunner().invoke(
        main, ["cam", "cp", str(sfmrfile_reconstruction_with_17_images), str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cam_cp_rejects_non_sfmr(tmp_path):
    fake = tmp_path / "input.txt"
    fake.write_text("hi")
    out = tmp_path / "camera_config.json"
    result = CliRunner().invoke(main, ["cam", "cp", str(fake), str(out)])
    assert result.exit_code != 0
    assert "must be a .sfmr" in result.output.lower()


def test_cam_cp_rejects_missing_input(tmp_path):
    out = tmp_path / "camera_config.json"
    result = CliRunner().invoke(
        main, ["cam", "cp", str(tmp_path / "missing.sfmr"), str(out)]
    )
    # Click's exists=True on the argument fires before our handler.
    assert result.exit_code != 0


def test_cam_cp_index_out_of_range(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path
):
    out = tmp_path / "camera_config.json"
    result = CliRunner().invoke(
        main,
        [
            "cam",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
            str(out),
            "--index",
            "99",
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output


def test_cam_cp_explicit_index_zero(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path
):
    out = tmp_path / "camera_config.json"
    result = CliRunner().invoke(
        main,
        [
            "cam",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
            str(out),
            "--index",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cam_cp_output_is_valid_camera_config(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path
):
    """The written file must satisfy `load_camera_config`."""
    from sfmtool._camera_config import load_camera_config

    out = tmp_path / "camera_config.json"
    result = CliRunner().invoke(
        main, ["cam", "cp", str(sfmrfile_reconstruction_with_17_images), str(out)]
    )
    assert result.exit_code == 0, result.output

    intrinsics = load_camera_config(out)
    assert intrinsics is not None
    assert "model" in intrinsics
    assert "parameters" in intrinsics


def test_cam_cp_roundtrip_into_solve(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path
):
    """End-to-end: harvest intrinsics, drop into a workspace, re-solve uses them."""
    import shutil

    src_workspace = sfmrfile_reconstruction_with_17_images.parent

    # Make a fresh workspace with the same images, no camera_config yet
    new_ws = tmp_path / "ws"
    new_ws.mkdir()
    images_src = src_workspace / "test_17_image"
    images_dst = new_ws / "test_17_image"
    shutil.copytree(images_src, images_dst)

    runner = CliRunner()
    result = runner.invoke(main, ["init", str(new_ws)])
    assert result.exit_code == 0, result.output

    # Harvest intrinsics from the existing reconstruction into the new workspace
    cam_config = new_ws / "camera_config.json"
    result = runner.invoke(
        main,
        ["cam", "cp", str(sfmrfile_reconstruction_with_17_images), str(cam_config)],
    )
    assert result.exit_code == 0, result.output
    assert cam_config.exists()

    # The harvested file must satisfy our own loader
    from sfmtool._camera_config import load_camera_config

    intrinsics = load_camera_config(cam_config)
    assert intrinsics is not None

    # And be picked up by the resolver
    from sfmtool._camera_config import CameraConfigResolver

    resolver = CameraConfigResolver(new_ws)
    resolved = resolver.resolve_for_directory(images_dst)
    assert resolved is not None
    assert resolved[0] == cam_config


def test_cam_help_lists_cp():
    result = CliRunner().invoke(main, ["cam", "--help"])
    assert result.exit_code == 0
    assert "cp" in result.output
