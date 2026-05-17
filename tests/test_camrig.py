# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `.camrig` conversion bindings and the `sfm camrig` command."""

from pathlib import Path

from click.testing import CliRunner

from sfmtool._sfmtool import SphericalTileRig, read_camrig_metadata, verify_camrig
from sfmtool.cli import main


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


def test_camrig_inspect_cli(tmp_path: Path):
    out = tmp_path / "inspect.camrig"
    build = CliRunner().invoke(
        main,
        [
            "camrig",
            "spherical-tiles",
            str(out),
            "--n",
            "150",
            "--equirect-width",
            "256",
        ],
    )
    assert build.exit_code == 0, build.output

    result = CliRunner().invoke(main, ["camrig", "inspect", str(out)])
    assert result.exit_code == 0, result.output
    assert "Rig type:     spherical_tiles" in result.output
    assert "Sensors:      150" in result.output
    assert "Integrity:    OK" in result.output


def test_camrig_inspect_rejects_non_camrig(tmp_path: Path):
    fake = tmp_path / "junk.camrig"
    fake.write_bytes(b"not a zip archive")
    result = CliRunner().invoke(main, ["camrig", "inspect", str(fake)])
    assert result.exit_code != 0


def test_camrig_help_lists_subcommands():
    result = CliRunner().invoke(main, ["camrig", "--help"])
    assert result.exit_code == 0
    assert "spherical-tiles" in result.output
    assert "inspect" in result.output
