# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `.camrig` conversion bindings and the `sfm camrig` command."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._camrig_resolver import (
    CamrigSolveError,
    resolve_camrig_for_solve,
)
from sfmtool._sfmtool import (
    SphericalTileRig,
    camrig_pattern_matches,
    read_camrig,
    read_camrig_metadata,
    verify_camrig,
    write_camrig,
)
from sfmtool.cli import main

_IMAGE_DATA = Path(__file__).parent.parent / "test-data" / "images"


def _copy_images(dest: Path, dataset: str, count: int) -> None:
    """Copy the first `count` images of a checked-in dataset into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted((_IMAGE_DATA / dataset).glob(f"{dataset}_*.jpg"))[:count]
    assert len(sources) == count
    for source in sources:
        shutil.copy(source, dest / source.name)


def _camera(width: int = 640, height: int = 480) -> dict:
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "parameters": {
            "focal_length_x": 500.0,
            "focal_length_y": 500.0,
            "principal_point_x": width / 2,
            "principal_point_y": height / 2,
        },
    }


def _pinhole_camera() -> dict:
    return _camera()


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
    result = CliRunner().invoke(main, ["camrig", "create", str(out), "imgs/*.jpg"])
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
        ["camrig", "create", str(out), "imgs/seoul_bull_sculpture_%d.jpg"],
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
            str(out),
            "imgs/*.jpg",
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
        ["camrig", "create", str(out), "imgs/*.jpg", "--camera-model", "OPENCV"],
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
            str(out),
            "imgs/*.jpg",
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
    result = CliRunner().invoke(main, ["camrig", "create", str(out), "imgs/*.jpg"])
    assert result.exit_code != 0
    assert "inconsistent resolutions" in result.output
    assert not out.exists()


def test_camrig_create_rejects_non_image(tmp_path: Path):
    imgs = tmp_path / "imgs"
    _copy_images(imgs, "seoul_bull_sculpture", 2)
    (imgs / "notes.txt").write_text("not an image")
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(main, ["camrig", "create", str(out), "imgs/*"])
    assert result.exit_code != 0
    assert "non-image" in result.output


def test_camrig_create_no_match(tmp_path: Path):
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(main, ["camrig", "create", str(out), "missing/*.jpg"])
    assert result.exit_code != 0
    assert "no files match" in result.output


def test_camrig_create_rejects_two_frame_fields(tmp_path: Path):
    # A pattern may carry at most one frame field; `create` rejects a second
    # one up front, before touching the filesystem.
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main, ["camrig", "create", str(out), "imgs/cam_%d_%04d.jpg"]
    )
    assert result.exit_code != 0
    assert "at most one" in result.output
    assert not out.exists()


def test_camrig_create_params_requires_camera_model(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "create", str(out), "imgs/*.jpg", "--params", "1,2,3,4"],
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
            str(out),
            "imgs/*.jpg",
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
            str(out),
            "imgs/*.jpg",
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


# ── resolve_camrig_for_solve ────────────────────────────────────────────────


def _touch_images(directory: Path, names: list[str]) -> list[Path]:
    """Create empty placeholder image files; return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = directory / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _make_camrig(
    path: Path,
    patterns: list[str],
    sensor_count: int = 1,
    camera: dict | None = None,
) -> None:
    """Write a minimal .camrig with the given per-sensor image patterns."""
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[camera or _pinhole_camera()],
        sensor_image_patterns=patterns,
        camera_indexes=[0] * sensor_count,
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * sensor_count),
        translations_xyz=np.zeros((sensor_count, 3)),
    )


def test_resolve_camrig_returns_camera(tmp_path: Path):
    # The resolver reads each covered image, so use real files; seoul images
    # are 270x480, so give the .camrig a camera of matching aspect ratio.
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 3)
    imgs = sorted((tmp_path / "imgs").glob("*.jpg"))
    _make_camrig(tmp_path / "rig.camrig", ["imgs/*.jpg"], camera=_camera(270, 480))
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert result is not None
    assert not result.is_multi_sensor
    assert result.camera["model"] == "PINHOLE"


def test_resolve_camrig_none_when_absent(tmp_path: Path):
    imgs = _touch_images(tmp_path / "imgs", ["a.jpg"])
    assert resolve_camrig_for_solve(imgs, tmp_path, None) is None


def test_resolve_camrig_none_when_pattern_misses(tmp_path: Path):
    imgs = _touch_images(tmp_path / "imgs", ["a.jpg"])
    _make_camrig(tmp_path / "rig.camrig", ["other/*.jpg"])
    assert resolve_camrig_for_solve(imgs, tmp_path, None) is None


def test_resolve_camrig_ignores_geometry_only(tmp_path: Path):
    imgs = _touch_images(tmp_path / "imgs", ["a.jpg"])
    _make_camrig(tmp_path / "tiles.camrig", [])
    assert resolve_camrig_for_solve(imgs, tmp_path, None) is None


def test_resolve_camrig_rejects_multiple(tmp_path: Path):
    a = _touch_images(tmp_path / "a", ["x.jpg"])
    b = _touch_images(tmp_path / "b", ["y.jpg"])
    _make_camrig(tmp_path / "a" / "ra.camrig", ["*.jpg"])
    _make_camrig(tmp_path / "b" / "rb.camrig", ["*.jpg"])
    with pytest.raises(CamrigSolveError, match="multiple .camrig"):
        resolve_camrig_for_solve(a + b, tmp_path, None)


def test_resolve_camrig_rejects_partial_coverage(tmp_path: Path):
    covered = _touch_images(tmp_path / "imgs", ["a.jpg", "b.jpg"])
    stray = _touch_images(tmp_path / "other", ["c.jpg"])
    _make_camrig(tmp_path / "rig.camrig", ["imgs/*.jpg"])
    with pytest.raises(CamrigSolveError, match="only"):
        resolve_camrig_for_solve(covered + stray, tmp_path, None)


def test_resolve_camrig_rejects_camera_model(tmp_path: Path):
    imgs = _touch_images(tmp_path / "imgs", ["a.jpg"])
    _make_camrig(tmp_path / "rig.camrig", ["imgs/*.jpg"])
    with pytest.raises(CamrigSolveError, match="camera-model"):
        resolve_camrig_for_solve(imgs, tmp_path, "PINHOLE")


def test_resolve_camrig_resolves_multi_sensor(tmp_path: Path):
    # A multi-sensor .camrig resolves to a rig with a per-image sensor/frame
    # assignment; the two sensors are paired by the captured frame index.
    _copy_images(tmp_path / "left", "seoul_bull_sculpture", 2)
    _copy_images(tmp_path / "right", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "left").glob("*.jpg")) + sorted(
        (tmp_path / "right").glob("*.jpg")
    )
    _make_camrig(
        tmp_path / "rig.camrig",
        [
            "left/seoul_bull_sculpture_%d.jpg",
            "right/seoul_bull_sculpture_%d.jpg",
        ],
        sensor_count=2,
        camera=_camera(270, 480),
    )
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert result is not None
    assert result.is_multi_sensor
    rig = result.rig
    assert {a[0] for a in rig.assignments.values()} == {0, 1}
    assert {a[1] for a in rig.assignments.values()} == {1, 2}


def test_resolve_camrig_multi_sensor_rejects_camera_model(tmp_path: Path):
    _copy_images(tmp_path / "left", "seoul_bull_sculpture", 2)
    _copy_images(tmp_path / "right", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "left").glob("*.jpg")) + sorted(
        (tmp_path / "right").glob("*.jpg")
    )
    _make_camrig(
        tmp_path / "rig.camrig",
        [
            "left/seoul_bull_sculpture_%d.jpg",
            "right/seoul_bull_sculpture_%d.jpg",
        ],
        sensor_count=2,
        camera=_camera(270, 480),
    )
    with pytest.raises(CamrigSolveError, match="camera-model"):
        resolve_camrig_for_solve(imgs, tmp_path, "PINHOLE")


def test_resolve_camrig_rejects_same_sensor_frame_collision(tmp_path: Path):
    # A variable-width `%d` field can capture the same frame index from two
    # files of one sensor (`frame_1.jpg` and `frame_001.jpg` both -> 1); the
    # resolver must reject this rather than build a rig frame carrying two
    # images for one sensor.
    src = sorted((_IMAGE_DATA / "seoul_bull_sculpture").glob("*.jpg"))[0]
    left = tmp_path / "left"
    left.mkdir()
    shutil.copy(src, left / "frame_1.jpg")
    shutil.copy(src, left / "frame_001.jpg")
    _copy_images(tmp_path / "right", "seoul_bull_sculpture", 1)
    imgs = sorted(left.glob("*.jpg")) + sorted((tmp_path / "right").glob("*.jpg"))
    _make_camrig(
        tmp_path / "rig.camrig",
        ["left/frame_%d.jpg", "right/seoul_bull_sculpture_%d.jpg"],
        sensor_count=2,
        camera=_camera(270, 480),
    )
    with pytest.raises(CamrigSolveError, match="same frame index"):
        resolve_camrig_for_solve(imgs, tmp_path, None)


def test_resolve_camrig_rejects_mixed_resolution(tmp_path: Path):
    imgs_dir = tmp_path / "imgs"
    _copy_images(imgs_dir, "seoul_bull_sculpture", 2)
    shutil.copy(
        _IMAGE_DATA / "dino_dog_toy" / "dino_dog_toy_01.jpg",
        imgs_dir / "dino_dog_toy_01.jpg",
    )
    imgs = sorted(imgs_dir.glob("*.jpg"))
    _make_camrig(tmp_path / "rig.camrig", ["imgs/*.jpg"], camera=_camera(270, 480))
    with pytest.raises(CamrigSolveError, match="mixed resolution"):
        resolve_camrig_for_solve(imgs, tmp_path, None)


def test_resolve_camrig_rejects_aspect_mismatch(tmp_path: Path):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "imgs").glob("*.jpg"))
    # seoul images are 270x480; a 640x480 camera has a different aspect ratio.
    _make_camrig(tmp_path / "rig.camrig", ["imgs/*.jpg"], camera=_camera(640, 480))
    with pytest.raises(CamrigSolveError, match="aspect ratio"):
        resolve_camrig_for_solve(imgs, tmp_path, None)


def test_resolve_camrig_frame_field_excludes_non_digit(tmp_path: Path):
    # A frame-field pattern must not cover a sibling whose name is not a frame
    # number, even though the loose glob (`cam_%04d.jpg` -> `cam_*.jpg`) hits
    # it. Here the stray file makes coverage partial rather than complete.
    imgs = _touch_images(tmp_path / "imgs", ["cam_0001.jpg", "cam_extra.jpg"])
    _make_camrig(tmp_path / "rig.camrig", ["imgs/cam_%04d.jpg"])
    with pytest.raises(CamrigSolveError, match="only"):
        resolve_camrig_for_solve(imgs, tmp_path, None)


def test_resolve_camrig_globstar_pattern(tmp_path: Path):
    # A `**` pattern still covers images nested below the rig directory.
    _copy_images(tmp_path / "a" / "b", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "a" / "b").glob("*.jpg"))
    _make_camrig(tmp_path / "rig.camrig", ["**/*.jpg"], camera=_camera(270, 480))
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert result is not None
    assert result.camera is not None
    assert result.camera["model"] == "PINHOLE"


# ── pattern matching (camrig-format grammar via the PyO3 binding) ───────────


def test_pattern_matches_frame_field_is_digits_only():
    # A frame field matches digits only — the whole point of the strict
    # confirm is that the loose glob (`cam_%04d.jpg` -> `cam_*.jpg`) does not.
    def m(path: str) -> bool:
        return camrig_pattern_matches("cam_%04d.jpg", path, False)

    assert m("cam_0007.jpg")
    assert m("cam_10000.jpg")  # frame index wider than the pad
    assert not m("cam_x.jpg")
    assert not m("cam_.jpg")
    assert not m("cam_007a.jpg")


def test_pattern_matches_star_stays_within_segment():
    assert camrig_pattern_matches("imgs/*.jpg", "imgs/a.jpg", False)
    assert not camrig_pattern_matches("imgs/*.jpg", "imgs/sub/a.jpg", False)


def test_pattern_matches_globstar_spans_segments():
    def m(path: str) -> bool:
        return camrig_pattern_matches("imgs/**/*.jpg", path, False)

    assert m("imgs/a.jpg")  # `**` matches zero segments
    assert m("imgs/x/a.jpg")
    assert m("imgs/x/y/a.jpg")
    assert not m("other/a.jpg")


def test_pattern_matches_escaped_percent_is_literal():
    assert camrig_pattern_matches("f%%.jpg", "f%.jpg", False)
    assert not camrig_pattern_matches("f%%.jpg", "f%%.jpg", False)


def test_pattern_matches_case_insensitive_is_opt_in():
    # The case-insensitive flag exists so the strict confirm never rejects a
    # hit the (case-insensitive) Windows glob accepted.
    assert not camrig_pattern_matches("cam_%d.JPG", "cam_7.jpg", False)
    assert camrig_pattern_matches("cam_%d.JPG", "cam_7.jpg", True)


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
        ["camrig", "create", str(camrig_path), "*.jpg", "--camera-model", "PINHOLE"],
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

    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(output_path)
    assert recon.image_count > 0
    assert recon.camera_count > 0


# ── sfm camrig cp ───────────────────────────────────────────────────────────

_KERRY_PARK_CAMRIG = _IMAGE_DATA / "kerry_park" / "kerry_park.camrig"


def test_cp_sfmr_single_camera_default(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path
):
    """A rig-less .sfmr with one camera defaults to copying that camera."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(sfmrfile_reconstruction_with_17_images), str(out)],
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


def test_cp_sfmr_camera_explicit_pattern(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path
):
    """`--pattern` overrides the inferred single-sensor image pattern."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
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
    sfmrfile_reconstruction_with_17_images: Path,
):
    """A .camrig harvested by `cp` is discoverable by the solve resolver."""
    workspace = sfmrfile_reconstruction_with_17_images.parent
    out = workspace / "harvested.camrig"
    result = CliRunner().invoke(
        main,
        ["camrig", "cp", str(sfmrfile_reconstruction_with_17_images), str(out)],
    )
    assert result.exit_code == 0, result.output

    images = sorted((workspace / "test_17_image").glob("*.jpg"))
    assert images
    resolved = resolve_camrig_for_solve(images, workspace, None)
    assert resolved is not None
    assert resolved.camera is not None


def test_cp_sfmr_rig(sfmrfile_reconstruction_kerry_park: Path, tmp_path: Path):
    """`--rig` copies a whole rig — its sensors, cameras, and poses."""
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_kerry_park),
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


def test_cp_sfmr_default_selects_rig(
    sfmrfile_reconstruction_kerry_park: Path, tmp_path: Path
):
    """With one rig present and no selector, `cp` copies that rig."""
    out = tmp_path / "rig.camrig"
    result = CliRunner().invoke(
        main, ["camrig", "cp", str(sfmrfile_reconstruction_kerry_park), str(out)]
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


def test_cp_name_override(sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path):
    """`--name` sets the rig name stored in the output `.camrig`."""
    out = tmp_path / "cam.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
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


def test_cp_rejects_sensors_on_sfmr(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path
):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
            str(out),
            "--sensors",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "--sensors applies to a .camrig" in result.output


def test_cp_rejects_pattern_without_camera(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path
):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
            str(out),
            "--pattern",
            "*.jpg",
        ],
    )
    assert result.exit_code != 0
    assert "--pattern applies only with --camera" in result.output


def test_cp_rejects_rig_and_camera(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path
):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
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


def test_cp_camera_out_of_range(
    sfmrfile_reconstruction_with_17_images: Path, tmp_path: Path
):
    out = tmp_path / "x.camrig"
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "cp",
            str(sfmrfile_reconstruction_with_17_images),
            str(out),
            "--camera",
            "99",
        ],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output
