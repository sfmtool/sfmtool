# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm camrig cp` command (harvesting `.camrig` files)."""

from pathlib import Path

from click.testing import CliRunner

from sfmtool.camrig.resolver import resolve_camrig_for_solve
from sfmtool._sfmtool.io import (
    read_camrig,
    read_camrig_metadata,
    verify_camrig,
)
from sfmtool.cli import main

_IMAGE_DATA = Path(__file__).parent.parent / "test-data" / "images"
_KERRY_PARK_CAMRIG = _IMAGE_DATA / "kerry_park" / "kerry_park.camrig"


def test_cp_sfmr_single_camera_default(seoul_bull_workspace: Path, tmp_path: Path):
    """A rig-less .sfmr with one camera defaults to copying that camera."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(seoul_bull_workspace), str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    meta = read_camrig_metadata(str(out))["metadata"]
    assert meta["sensor_count"] == 1
    assert meta["camera_count"] == 1
    assert meta["rig_type"] == "generic"
    valid, errors = verify_camrig(str(out))
    assert valid, errors

    # The sensor pattern is inferred from the reconstruction's image names.
    data = read_camrig(str(out))
    assert len(data["sensor_image_patterns"]) == 1


def test_cp_sfmr_camera_explicit_pattern(seoul_bull_workspace: Path, tmp_path: Path):
    """`--pattern` overrides the inferred single-sensor image pattern."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(seoul_bull_workspace),
            str(out),
            "--camera",
            "0",
            "--pattern",
            "photos/*.jpg",
        ],
    )
    assert result.exit_code == 0, result.output
    data = read_camrig(str(out))
    assert data["sensor_image_patterns"] == ["photos/*.jpg"]


def test_cp_sfmr_camera_roundtrips_into_resolver(
    seoul_bull_workspace: Path,
):
    """A .camrig harvested by `cp` is discoverable by the solve resolver."""
    workspace = seoul_bull_workspace.parent
    out = workspace / "harvested.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(seoul_bull_workspace), str(out)],
    )
    assert result.exit_code == 0, result.output

    images = sorted((workspace / "test_17_image").glob("*.jpg"))
    assert images
    resolved = resolve_camrig_for_solve(images, workspace, None)
    assert resolved is not None
    assert resolved.camera is not None


def test_cp_sfmr_rig(kerry_park_workspace: Path, tmp_path: Path):
    """`--rig` copies a whole rig — its sensors, cameras, and poses."""
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(kerry_park_workspace),
            str(out),
            "--rig",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    meta = read_camrig_metadata(str(out))["metadata"]
    assert meta["sensor_count"] == 2
    valid, errors = verify_camrig(str(out))
    assert valid, errors

    # A multi-sensor rig needs a frame field in every inferred pattern.
    data = read_camrig(str(out))
    assert len(data["sensor_image_patterns"]) == 2
    assert all("%" in p for p in data["sensor_image_patterns"])


def test_cp_sfmr_default_selects_rig(kerry_park_workspace: Path, tmp_path: Path):
    """With one rig present and no selector, `cp` copies that rig."""
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main, ["camrig", "cp", str(kerry_park_workspace), str(out)]
    )
    assert result.exit_code == 0, result.output
    assert read_camrig_metadata(str(out))["metadata"]["sensor_count"] == 2


def test_cp_camrig_whole_copy(tmp_path: Path):
    """Copying a whole .camrig preserves its rig type and sensor count."""
    out = tmp_path / "copy.camrig"
    result = CliRunner().invoke(
        main, ["camrig", "cp", str(_KERRY_PARK_CAMRIG), str(out)]
    )
    assert result.exit_code == 0, result.output

    src = read_camrig_metadata(str(_KERRY_PARK_CAMRIG))["metadata"]
    dst = read_camrig_metadata(str(out))["metadata"]
    assert dst["sensor_count"] == src["sensor_count"]
    assert dst["rig_type"] == src["rig_type"]
    valid, errors = verify_camrig(str(out))
    assert valid, errors


def test_cp_camrig_sensor_subset(tmp_path: Path):
    """A sensor subset of a typed rig becomes a `generic` rig.

    The camera pool is reduced to the cameras the kept sensors use, and each
    kept sensor keeps its source image pattern verbatim.
    """
    out = tmp_path / "sub.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(_KERRY_PARK_CAMRIG), str(out), "--sensors", "0"],
    )
    assert result.exit_code == 0, result.output
    meta = read_camrig_metadata(str(out))["metadata"]
    assert meta["sensor_count"] == 1
    assert meta["camera_count"] == 1
    assert meta["rig_type"] == "generic"

    src = read_camrig(str(_KERRY_PARK_CAMRIG))
    dst = read_camrig(str(out))
    assert dst["sensor_image_patterns"] == [src["sensor_image_patterns"][0]]


def test_cp_camrig_sensors_range_selects_all(tmp_path: Path):
    """A range expression covering every sensor is a whole-rig copy.

    Exercises `--sensors` range parsing (`0-1`) and confirms that selecting
    all sensors preserves the source's typed `rig_type`.
    """
    out = tmp_path / "all.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(_KERRY_PARK_CAMRIG), str(out), "--sensors", "0-1"],
    )
    assert result.exit_code == 0, result.output
    src = read_camrig_metadata(str(_KERRY_PARK_CAMRIG))["metadata"]
    dst = read_camrig_metadata(str(out))["metadata"]
    assert dst["sensor_count"] == src["sensor_count"]
    assert dst["rig_type"] == src["rig_type"]


def test_cp_rejects_sensors_out_of_range(tmp_path: Path):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(_KERRY_PARK_CAMRIG), str(out), "--sensors", "0,9"],
    )
    assert result.exit_code != 0
    assert "outside the valid range" in result.output


def test_cp_name_override(seoul_bull_workspace: Path, tmp_path: Path):
    """`--name` sets the rig name stored in the output `.camrig`."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(seoul_bull_workspace),
            str(out),
            "--name",
            "harvested-rig",
        ],
    )
    assert result.exit_code == 0, result.output
    assert read_camrig_metadata(str(out))["metadata"]["name"] == "harvested-rig"


def test_cp_camrig_single_camera(tmp_path: Path):
    """`--camera` on a .camrig copies one pool camera as a single sensor."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(_KERRY_PARK_CAMRIG), str(out), "--camera", "0"],
    )
    assert result.exit_code == 0, result.output
    meta = read_camrig_metadata(str(out))["metadata"]
    assert meta["sensor_count"] == 1
    assert meta["camera_count"] == 1


def test_cp_rejects_rig_on_camrig(tmp_path: Path):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main, ["camrig", "cp", str(_KERRY_PARK_CAMRIG), str(out), "--rig", "0"]
    )
    assert result.exit_code != 0
    assert "--rig applies to a .sfmr" in result.output


def test_cp_rejects_sensors_on_sfmr(seoul_bull_workspace: Path, tmp_path: Path):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(seoul_bull_workspace),
            str(out),
            "--sensors",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "--sensors applies to a .camrig" in result.output


def test_cp_rejects_pattern_without_camera(seoul_bull_workspace: Path, tmp_path: Path):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(seoul_bull_workspace),
            str(out),
            "--pattern",
            "*.jpg",
        ],
    )
    assert result.exit_code != 0
    assert "--pattern applies only with --camera" in result.output


def test_cp_rejects_rig_and_camera(seoul_bull_workspace: Path, tmp_path: Path):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(seoul_bull_workspace),
            str(out),
            "--rig",
            "0",
            "--camera",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cp_rejects_non_recon_source(tmp_path: Path):
    fake = tmp_path / "input.txt"
    fake.write_text("hi")
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(main, ["camrig", "cp", str(fake), str(out)])
    assert result.exit_code != 0
    assert "must be a .sfmr or .camrig" in result.output


def test_cp_camera_out_of_range(seoul_bull_workspace: Path, tmp_path: Path):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(seoul_bull_workspace),
            str(out),
            "--camera",
            "99",
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output
