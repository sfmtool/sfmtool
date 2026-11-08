# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `resolve_camrig_for_solve` and the `.camrig` pattern-matching
grammar (`camrig_pattern_matches`)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from sfmtool.camrig.resolver import (
    CamrigSolveError,
    resolve_camrig_for_solve,
)
from sfmtool.fileio import (
    camrig_pattern_matches,
    write_camrig,
)

from .conftest import _IMAGE_DATA, _camera, _copy_images, _pinhole_camera


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


def _simple_pinhole_camera(focal: float = 300.0) -> dict:
    """A SIMPLE_PINHOLE camera at seoul_bull_sculpture's 270x480, as
    `sfm estimate-intrinsics --write-camrig` writes a pinhole verdict."""
    return {
        "model": "SIMPLE_PINHOLE",
        "width": 270,
        "height": 480,
        "parameters": {
            "focal_length": focal,
            "principal_point_x": 135.0,
            "principal_point_y": 240.0,
        },
    }


def test_resolve_camrig_refits_camera_to_camera_model(tmp_path: Path, capsys):
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "imgs").glob("*.jpg"))
    camrig_file = tmp_path / "rig.camrig"
    _make_camrig(camrig_file, ["imgs/*.jpg"], camera=_simple_pinhole_camera())
    before = camrig_file.read_bytes()

    result = resolve_camrig_for_solve(imgs, tmp_path, "simple_radial")

    assert result is not None and not result.is_multi_sensor
    assert result.camera == {
        "model": "SIMPLE_RADIAL",
        "width": 270,
        "height": 480,
        "parameters": {
            "focal_length": 300.0,
            "principal_point_x": 135.0,
            "principal_point_y": 240.0,
            "radial_distortion_k1": 0.0,
        },
    }
    # The refit camera is the solve's start; the file on disk is unchanged.
    assert camrig_file.read_bytes() == before
    out = capsys.readouterr().out
    assert (
        f"Camera 0 of {camrig_file.resolve()}: SIMPLE_PINHOLE -> SIMPLE_RADIAL, "
        "fit over θ ≤ 42.5° (image_corner): rms 0.000 px, max 0.000 px"
    ) in out


def test_resolve_camrig_refit_reports_a_lossy_fit(tmp_path: Path, capsys):
    # A pinhole switched to a fisheye model cannot be matched exactly; the
    # report line carries the fit's rms.
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 1)
    imgs = sorted((tmp_path / "imgs").glob("*.jpg"))
    _make_camrig(
        tmp_path / "rig.camrig", ["imgs/*.jpg"], camera=_simple_pinhole_camera()
    )
    result = resolve_camrig_for_solve(imgs, tmp_path, "SIMPLE_RADIAL_FISHEYE")
    assert result.camera["model"] == "SIMPLE_RADIAL_FISHEYE"
    out = capsys.readouterr().out
    assert "SIMPLE_PINHOLE -> SIMPLE_RADIAL_FISHEYE" in out
    assert "rms 0.532 px" in out


def test_resolve_camrig_refused_refit_names_the_camera(tmp_path: Path):
    # An equidistant fisheye whose image corner is past 90 degrees has no
    # perspective fit.
    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 1)
    imgs = sorted((tmp_path / "imgs").glob("*.jpg"))
    camrig_file = tmp_path / "rig.camrig"
    fisheye = _simple_pinhole_camera(focal=150.0)
    fisheye["model"] = "EQUIDISTANT_FISHEYE"
    _make_camrig(camrig_file, ["imgs/*.jpg"], camera=fisheye)
    with pytest.raises(CamrigSolveError) as excinfo:
        resolve_camrig_for_solve(imgs, tmp_path, "PINHOLE")
    message = str(excinfo.value)
    assert (
        f"cannot switch camera 0 of {camrig_file.resolve()} from "
        "EQUIDISTANT_FISHEYE to PINHOLE"
    ) in message
    assert "90" in message


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


def test_resolve_camrig_multi_sensor_refits_each_camera(tmp_path: Path, capsys):
    # Two sensors with their own pool cameras: each is refit, and the rig the
    # solve gets carries the refit cameras in place of the file's.
    _copy_images(tmp_path / "left", "seoul_bull_sculpture", 2)
    _copy_images(tmp_path / "right", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "left").glob("*.jpg")) + sorted(
        (tmp_path / "right").glob("*.jpg")
    )
    camrig_file = tmp_path / "rig.camrig"
    write_camrig(
        path=str(camrig_file),
        name="rig",
        rig_type="generic",
        cameras=[_simple_pinhole_camera(300.0), _simple_pinhole_camera(310.0)],
        sensor_image_patterns=[
            "left/seoul_bull_sculpture_%d.jpg",
            "right/seoul_bull_sculpture_%d.jpg",
        ],
        camera_indexes=[0, 1],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
        translations_xyz=np.zeros((2, 3)),
    )
    before = camrig_file.read_bytes()

    result = resolve_camrig_for_solve(imgs, tmp_path, "RADIAL")

    assert result.is_multi_sensor
    cameras = result.rig.data["cameras"]
    assert [c["model"] for c in cameras] == ["RADIAL", "RADIAL"]
    assert [c["parameters"]["focal_length"] for c in cameras] == [300.0, 310.0]
    for camera in cameras:
        assert camera["parameters"]["radial_distortion_k1"] == 0.0
        assert camera["parameters"]["radial_distortion_k2"] == 0.0
    assert camrig_file.read_bytes() == before
    out = capsys.readouterr().out
    assert f"Camera 0 of {camrig_file.resolve()}: SIMPLE_PINHOLE -> RADIAL" in out
    assert f"Camera 1 of {camrig_file.resolve()}: SIMPLE_PINHOLE -> RADIAL" in out


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


# ── pattern matching (sfmtool-camrig-format grammar via the PyO3 binding) ───────────


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
