# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `.camrig` conversion bindings and the `sfm camrig` command."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.camrig.resolver import (
    resolve_camrig_for_solve,
)
from sfmtool._sfmtool.io import (
    read_camrig,
    read_camrig_metadata,
    verify_camrig,
    write_camrig,
)
from sfmtool._sfmtool.spherical import SphericalTileRig
from sfmtool.cli import main

from ._camrig_helpers import _IMAGE_DATA, _copy_images, _pinhole_camera


def test_spherical_tile_rig_camrig_round_trip(tmp_path: Path):
    """Writing a rig to .camrig and reading it back preserves its geometry."""
    rig = SphericalTileRig(n=256, arc_per_pixel=0.02, centre=[1.0, -2.0, 0.5], seed=3)
    path = tmp_path / "rig.camrig"
    rig.write_camrig(str(path))
    assert path.exists()

    back = SphericalTileRig.read_camrig(str(path))
    assert len(back) == len(rig) == 256
    assert back.patch_size == rig.patch_size
    assert back.atlas_cols == rig.atlas_cols
    assert abs(back.half_fov_rad - rig.half_fov_rad) < 1e-12
    for k in range(3):
        assert abs(back.centre[k] - rig.centre[k]) < 1e-12
    for i in range(0, 256, 31):
        d0, d1 = rig.direction(i), back.direction(i)
        assert max(abs(a - b) for a, b in zip(d0, d1)) < 1e-9


def test_camrig_metadata_and_verify(tmp_path: Path):
    rig = SphericalTileRig(n=120, arc_per_pixel=0.03, seed=7)
    path = tmp_path / "tiles.camrig"
    rig.write_camrig(str(path), name="my_rig")

    valid, errors = verify_camrig(str(path))
    assert valid, errors

    meta = read_camrig_metadata(str(path))["metadata"]
    assert meta["name"] == "my_rig"
    assert meta["rig_type"] == "spherical_tiles"
    assert meta["sensor_count"] == 120
    assert meta["camera_count"] == 1
    assert meta["rig_attributes"]["patch_size"] == rig.patch_size


def test_camrig_spherical_tiles_cli(tmp_path: Path):
    out = tmp_path / "cli.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "spherical-tiles",
            str(out),
            "--n",
            "200",
            "--equirect-width",
            "512",
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "tiles:      200" in result.output

    rig = SphericalTileRig.read_camrig(str(out))
    assert len(rig) == 200


def test_camrig_spherical_tiles_arc_per_pixel_cli(tmp_path: Path):
    out = tmp_path / "arc.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "spherical-tiles", str(out), "--n", "80", "--arc-per-pixel", "0.04"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_camrig_spherical_tiles_requires_one_resolution_option(tmp_path: Path):
    out = tmp_path / "bad.camrig"

    # Neither resolution option.
    result = CliRunner().invoke(
        main, ["camrig", "spherical-tiles", str(out), "--n", "80"]
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output.lower()

    # Both resolution options.
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "spherical-tiles",
            str(out),
            "--n",
            "80",
            "--equirect-width",
            "256",
            "--arc-per-pixel",
            "0.02",
        ],
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output.lower()


def test_camrig_help_lists_subcommands():
    result = CliRunner().invoke(main, ["camrig", "--help"])
    assert result.exit_code == 0
    assert "create" in result.output
    assert "cp" in result.output
    assert "spherical-tiles" in result.output


# ── write_camrig binding ────────────────────────────────────────────────────


def test_write_camrig_binding_round_trip(tmp_path: Path):
    out = tmp_path / "manual.camrig"
    write_camrig(
        path=str(out),
        name="manual",
        rig_type="generic",
        cameras=[_pinhole_camera()],
        sensor_image_patterns=["images/*.jpg"],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.array([[0.0, 0.0, 0.0]]),
    )
    valid, errors = verify_camrig(str(out))
    assert valid, errors
    meta = read_camrig_metadata(str(out))["metadata"]
    assert meta["name"] == "manual"
    assert meta["sensor_count"] == 1
    assert meta["camera_count"] == 1


def test_write_camrig_binding_rejects_multi_sensor_without_frame_field(
    tmp_path: Path,
):
    # A multi-sensor rig needs a frame field in every pattern; the binding
    # surfaces that validation failure as an exception.
    out = tmp_path / "bad.camrig"
    with pytest.raises(Exception, match="frame field"):
        write_camrig(
            path=str(out),
            name="bad",
            rig_type="generic",
            cameras=[_pinhole_camera()],
            sensor_image_patterns=["left/*.jpg", "right/*.jpg"],
            camera_indexes=[0, 0],
            quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            translations_xyz=np.zeros((2, 3)),
        )


def test_write_camrig_binding_rejects_two_frame_fields(tmp_path: Path):
    # An image pattern may carry at most one frame field; a second `%d` makes
    # the captured frame index ambiguous, so the binding rejects it.
    out = tmp_path / "bad.camrig"
    with pytest.raises(Exception, match="at most one"):
        write_camrig(
            path=str(out),
            name="bad",
            rig_type="generic",
            cameras=[_pinhole_camera()],
            sensor_image_patterns=["cam_%d_%04d.jpg"],
            camera_indexes=[0],
            quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
            translations_xyz=np.zeros((1, 3)),
        )


# ── sfm camrig create ───────────────────────────────────────────────────────


def test_camrig_create_from_directory(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 5)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(main, ["camrig", "create", "imgs/*.jpg", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "pattern:  imgs/*.jpg" in result.output
    assert "images:   5" in result.output

    meta = read_camrig_metadata(str(out))["metadata"]
    assert meta["sensor_count"] == 1
    assert meta["camera_count"] == 1
    assert meta["rig_type"] == "generic"
    valid, errors = verify_camrig(str(out))
    assert valid, errors


def test_camrig_create_frame_field_pattern(tmp_path: Path):
    # `camrig create` interprets a `%d` frame field the same way `sfm solve`
    # does: it matches digit-named frames only, so a non-digit sibling image
    # is excluded rather than swept in by a literal-`%` glob.
    imgs = tmp_path / "imgs"
    _copy_images(imgs, "seoul_bull_sculpture", 4)
    shutil.copy(next(imgs.glob("*.jpg")), imgs / "seoul_bull_sculpture_extra.jpg")
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "create", "imgs/seoul_bull_sculpture_%d.jpg", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "images:   4" in result.output
    assert "pattern:  imgs/seoul_bull_sculpture_%d.jpg" in result.output

    # The stored pattern round-trips into the solve resolver unchanged.
    data = read_camrig(str(out))
    assert data["sensor_image_patterns"] == ["imgs/seoul_bull_sculpture_%d.jpg"]
    frames = sorted(p for p in imgs.glob("*.jpg") if "extra" not in p.name)
    camera = resolve_camrig_for_solve(frames, tmp_path, None)
    assert camera is not None


def test_camrig_create_explicit_params(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 3)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "create",
            "imgs/*.jpg",
            str(out),
            "--camera-model",
            "PINHOLE",
            "--params",
            "300,300,135,240",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PINHOLE 270x480" in result.output
    assert "300.0 px" in result.output
    valid, errors = verify_camrig(str(out))
    assert valid, errors


def test_camrig_create_camera_model_override(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 3)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "create", "imgs/*.jpg", str(out), "--camera-model", "OPENCV"],
    )
    assert result.exit_code == 0, result.output
    assert "OPENCV 270x480" in result.output


def test_camrig_create_focal_length_override(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "create",
            "imgs/*.jpg",
            str(out),
            "--camera-model",
            "PINHOLE",
            "--focal-length",
            "500",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "500.0 px" in result.output


def test_camrig_create_rejects_mixed_resolution(tmp_path: Path):
    imgs = tmp_path / "imgs"
    _copy_images(imgs, "seoul_bull_sculpture", 2)
    shutil.copy(
        _IMAGE_DATA / "dino_dog_toy" / "dino_dog_toy_01.jpg",
        imgs / "dino_dog_toy_01.jpg",
    )
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(main, ["camrig", "create", "imgs/*.jpg", str(out)])
    assert result.exit_code != 0
    assert "inconsistent resolutions" in result.output
    assert not out.exists()


def test_camrig_create_rejects_non_image(tmp_path: Path):
    imgs = tmp_path / "imgs"
    _copy_images(imgs, "seoul_bull_sculpture", 2)
    (imgs / "notes.txt").write_text("not an image")
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(main, ["camrig", "create", "imgs/*", str(out)])
    assert result.exit_code != 0
    assert "non-image" in result.output


def test_camrig_create_no_match(tmp_path: Path):
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(main, ["camrig", "create", "missing/*.jpg", str(out)])
    assert result.exit_code != 0
    assert "no files match" in result.output


def test_camrig_create_rejects_two_frame_fields(tmp_path: Path):
    # A pattern may carry at most one frame field; `create` rejects a second
    # one up front, before touching the filesystem.
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main, ["camrig", "create", "imgs/cam_%d_%04d.jpg", str(out)]
    )
    assert result.exit_code != 0
    assert "at most one" in result.output
    assert not out.exists()


def test_camrig_create_params_requires_camera_model(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "create", "imgs/*.jpg", str(out), "--params", "1,2,3,4"],
    )
    assert result.exit_code != 0
    assert "--params requires --camera-model" in result.output


def test_camrig_create_params_wrong_count(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "create",
            "imgs/*.jpg",
            str(out),
            "--camera-model",
            "PINHOLE",
            "--params",
            "300,300,135",
        ],
    )
    assert result.exit_code != 0
    assert "needs 4 values" in result.output


def test_camrig_create_params_conflicts_with_focal(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "create",
            "imgs/*.jpg",
            str(out),
            "--camera-model",
            "PINHOLE",
            "--params",
            "300,300,135,240",
            "--focal-length",
            "500",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


# ── read_camrig binding ─────────────────────────────────────────────────────


def test_read_camrig_binding_round_trip(tmp_path: Path):
    out = tmp_path / "r.camrig"
    write_camrig(
        path=str(out),
        name="rr",
        rig_type="generic",
        cameras=[_pinhole_camera()],
        sensor_image_patterns=["images/*.jpg"],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )
    data = read_camrig(str(out))
    assert data["metadata"]["name"] == "rr"
    assert data["sensor_image_patterns"] == ["images/*.jpg"]
    assert list(data["camera_indexes"]) == [0]
    assert data["cameras"][0]["model"] == "PINHOLE"
    assert data["quaternions_wxyz"].shape == (1, 4)
    assert data["translations_xyz"].shape == (1, 3)


# ── sfm solve with a .camrig ────────────────────────────────────────────────


def test_solve_uses_camrig(isolated_seoul_bull_17_images: list[Path]):
    """An auto-discovered .camrig supplies the camera for `sfm solve`."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent
    runner = CliRunner()

    assert runner.invoke(main, ["ws", "init", str(workspace_dir)]).exit_code == 0
    assert runner.invoke(main, ["sift", "--extract", str(workspace_dir)]).exit_code == 0

    camrig_path = workspace_dir / "rig.camrig"
    create = runner.invoke(
        main,
        ["camrig", "create", "*.jpg", str(camrig_path), "--camera-model", "PINHOLE"],
    )
    assert create.exit_code == 0, create.output

    output_path = workspace_dir / "camrig_solve.sfmr"
    result = runner.invoke(
        main, ["solve", "-i", "--output", str(output_path), str(workspace_dir)]
    )
    assert result.exit_code == 0, result.output
    # The resolver announces the .camrig it used, and notes it overrides the
    # camera_config.json the fixture also ships.
    assert "rig.camrig" in result.output
    assert "takes precedence over camera_config.json" in result.output
    assert output_path.exists()

    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    recon = SfmrReconstruction.load(output_path)
    assert recon.image_count > 0
    assert recon.camera_count > 0
