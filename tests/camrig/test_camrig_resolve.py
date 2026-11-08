# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `resolve_camrig_for_solve` and the `.camrig` pattern-matching
grammar (`camrig_pattern_matches`)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sfmtool.camera.config import CameraConfigResolver
from sfmtool.camrig.resolver import (
    CamrigSolveError,
    find_covering_camrigs,
    resolve_camrig_for_solve,
)
from sfmtool.fileio import (
    camrig_pattern_matches,
    camrig_pattern_matches_any_padding,
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
    assert result.cameras[0].camera["model"] == "PINHOLE"


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


def _split_seoul_images(directory: Path) -> tuple[list[Path], list[Path]]:
    """Three seoul_bull images in `directory`, renamed as two videos' frames:
    `shopA_00001.jpg`, `shopA_00002.jpg` and `shopB_00001.jpg`."""
    _copy_images(directory, "seoul_bull_sculpture", 3)
    sources = sorted(directory.glob("*.jpg"))
    names = ["shopA_00001.jpg", "shopA_00002.jpg", "shopB_00001.jpg"]
    paths = [source.rename(directory / name) for source, name in zip(sources, names)]
    return paths[:2], paths[2:]


def test_resolve_camrig_one_camera_per_single_sensor_file(tmp_path: Path):
    # Two single-sensor .camrig files side by side, each covering one video's
    # frames: each file is one camera, holding the images it covers.
    a, b = _split_seoul_images(tmp_path / "images")
    _make_camrig(
        tmp_path / "camera-shopA.camrig",
        ["images/shopA_%05d.jpg"],
        camera=_simple_pinhole_camera(747.0),
    )
    _make_camrig(
        tmp_path / "camera-shopB.camrig",
        ["images/shopB_%05d.jpg"],
        camera=_simple_pinhole_camera(1141.0),
    )
    result = resolve_camrig_for_solve(a + b, tmp_path, None)
    assert result is not None and not result.is_multi_sensor
    assert [c.camrig_file.name for c in result.cameras] == [
        "camera-shopA.camrig",
        "camera-shopB.camrig",
    ]
    assert [c.camera["parameters"]["focal_length"] for c in result.cameras] == [
        747.0,
        1141.0,
    ]
    assert result.cameras[0].images == {p.resolve() for p in a}
    assert result.cameras[1].images == {p.resolve() for p in b}


def test_resolve_camrig_rejects_an_image_matched_twice(tmp_path: Path):
    a, b = _split_seoul_images(tmp_path / "images")
    _make_camrig(tmp_path / "camera-shopA.camrig", ["images/shopA_%05d.jpg"])
    _make_camrig(tmp_path / "camera-all.camrig", ["images/*.jpg"])
    with pytest.raises(CamrigSolveError) as excinfo:
        resolve_camrig_for_solve(a + b, tmp_path, None)
    message = str(excinfo.value)
    assert "2 of the 3 images being solved are matched by more than one" in message
    assert f"e.g. {a[0].resolve()} is matched by" in message
    assert "camera-all.camrig and" in message


def test_resolve_camrig_leaves_uncovered_images_to_camera_config(
    tmp_path: Path, capsys
):
    # Images no .camrig covers keep the camera_config.json they resolve, and a
    # line says how many there are and where their cameras come from.
    a, b = _split_seoul_images(tmp_path / "images")
    other = _touch_images(tmp_path / "other", ["c.jpg", "d.jpg"])
    (tmp_path / "other" / "camera_config.json").write_text(
        json.dumps({"version": 1, "camera_intrinsics": _simple_pinhole_camera()})
    )
    for video in ("shopA", "shopB"):
        _make_camrig(
            tmp_path / f"camera-{video}.camrig",
            [f"images/{video}_%05d.jpg"],
            camera=_camera(270, 480),
        )
    result = resolve_camrig_for_solve(
        a + b + other, tmp_path, None, CameraConfigResolver(tmp_path)
    )
    assert [c.images for c in result.cameras] == [
        {p.resolve() for p in a},
        {p.resolve() for p in b},
    ]
    out = capsys.readouterr().out
    assert (
        "Camera intrinsics: 2 of 5 images being solved are covered by no "
        f".camrig, e.g. {other[0].resolve()}; they use camera_config.json.\n"
    ) in out


def _make_seoul_rig(tmp_path: Path, folders: tuple[str, str], name: str) -> None:
    """A two-sensor `.camrig` `name` in `tmp_path` over two seoul_bull images
    in each of `folders`, paired by file number."""
    for folder in folders:
        _copy_images(tmp_path / folder, "seoul_bull_sculpture", 2)
    _make_camrig(
        tmp_path / name,
        [f"{folder}/seoul_bull_sculpture_%02d.jpg" for folder in folders],
        sensor_count=2,
        camera=_camera(270, 480),
    )


def test_resolve_camrig_a_rig_beside_another_camrig(tmp_path: Path, capsys):
    # A rig and a single-sensor .camrig each take the images they cover, and
    # the images neither covers fall back, here to EXIF inference.
    _make_seoul_rig(tmp_path, ("left", "right"), "rig.camrig")
    _copy_images(tmp_path / "solo", "seoul_bull_sculpture", 1)
    _copy_images(tmp_path / "loose", "seoul_bull_sculpture", 1)
    _make_camrig(tmp_path / "solo.camrig", ["solo/*.jpg"], camera=_camera(270, 480))
    imgs = sorted(tmp_path.glob("*/*.jpg"))
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert result.is_multi_sensor
    (rig,) = result.rigs
    assert rig.camrig_file.name == "rig.camrig"
    assert {p.parent.name for p in rig.assignments} == {"left", "right"}
    assert sorted(rig.assignments.values()) == [(0, 1), (0, 2), (1, 1), (1, 2)]
    (camera,) = result.cameras
    assert camera.images == {(tmp_path / "solo" / imgs[0].name).resolve()}
    loose = (tmp_path / "loose" / imgs[0].name).resolve()
    assert (
        f"Camera intrinsics: 1 of 6 images being solved are covered by no "
        f".camrig, e.g. {loose}; they use EXIF inference.\n"
    ) in capsys.readouterr().out


def test_resolve_camrig_a_rig_covering_some_images(tmp_path: Path, capsys):
    # A rig covering only some of the images is the rig of those images; the
    # others fall back as they do beside a single-sensor .camrig, and the
    # message names the caller's images.
    _make_seoul_rig(tmp_path, ("left", "right"), "rig.camrig")
    _copy_images(tmp_path / "solo", "seoul_bull_sculpture", 1)
    imgs = sorted(tmp_path.glob("*/*.jpg"))
    result = resolve_camrig_for_solve(
        imgs, tmp_path, None, images="images of the .matches file"
    )
    assert result.cameras == []
    (rig,) = result.rigs
    assert len(rig.assignments) == 4
    stray = (tmp_path / "solo" / imgs[0].name).resolve()
    assert stray not in rig.assignments
    assert (
        f"Camera intrinsics: 1 of 5 images of the .matches file are covered by no "
        f".camrig, e.g. {stray}; they use EXIF inference.\n"
    ) in capsys.readouterr().out


def test_resolve_camrig_two_rigs(tmp_path: Path):
    # Each multi-sensor .camrig is its own rig of the images it covers.
    _make_seoul_rig(tmp_path, ("a_left", "a_right"), "a.camrig")
    _make_seoul_rig(tmp_path, ("b_left", "b_right"), "b.camrig")
    imgs = sorted(tmp_path.glob("*/*.jpg"))
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert [r.camrig_file.name for r in result.rigs] == ["a.camrig", "b.camrig"]
    assert [{p.parent.name for p in r.assignments} for r in result.rigs] == [
        {"a_left", "a_right"},
        {"b_left", "b_right"},
    ]
    assert result.cameras == []


def test_resolve_camrig_rejects_an_image_in_a_rig_and_another_camrig(
    tmp_path: Path,
):
    # An image a rig's sensor matches and a single-sensor .camrig matches too
    # is refused, as two single-sensor files matching one image are.
    _make_seoul_rig(tmp_path, ("left", "right"), "rig.camrig")
    _make_camrig(tmp_path / "left.camrig", ["left/*.jpg"], camera=_camera(270, 480))
    imgs = sorted(tmp_path.glob("*/*.jpg"))
    with pytest.raises(CamrigSolveError) as excinfo:
        resolve_camrig_for_solve(imgs, tmp_path, None)
    message = str(excinfo.value)
    assert "2 of the 4 images being solved are matched by more than one" in message
    assert "left.camrig and" in message and "rig.camrig" in message


def test_resolve_camrig_refits_each_file_camera(tmp_path: Path, capsys):
    a, b = _split_seoul_images(tmp_path / "images")
    shop_a = tmp_path / "camera-shopA.camrig"
    shop_b = tmp_path / "camera-shopB.camrig"
    _make_camrig(
        shop_a, ["images/shopA_%05d.jpg"], camera=_simple_pinhole_camera(747.0)
    )
    _make_camrig(
        shop_b, ["images/shopB_%05d.jpg"], camera=_simple_pinhole_camera(1141.0)
    )
    result = resolve_camrig_for_solve(a + b, tmp_path, "simple_radial")
    assert [c.camera["model"] for c in result.cameras] == ["SIMPLE_RADIAL"] * 2
    assert [c.camera["parameters"]["focal_length"] for c in result.cameras] == [
        747.0,
        1141.0,
    ]
    out = capsys.readouterr().out
    assert f"Camera 0 of {shop_a.resolve()}: SIMPLE_PINHOLE -> SIMPLE_RADIAL" in out
    assert f"Camera 0 of {shop_b.resolve()}: SIMPLE_PINHOLE -> SIMPLE_RADIAL" in out


def test_resolve_camrig_one_file_covering_some_images(tmp_path: Path, capsys):
    # One single-sensor .camrig covering only some of the images follows the
    # same rule as several: its images take its camera and the others fall
    # back, here to EXIF inference.
    a, b = _split_seoul_images(tmp_path / "images")
    _make_camrig(
        tmp_path / "camera-shopA.camrig",
        ["images/shopA_%05d.jpg"],
        camera=_simple_pinhole_camera(),
    )
    result = resolve_camrig_for_solve(a + b, tmp_path, None)
    assert [c.camrig_file.name for c in result.cameras] == ["camera-shopA.camrig"]
    assert result.cameras[0].images == {p.resolve() for p in a}
    out = capsys.readouterr().out
    assert (
        "Camera intrinsics: 1 of 3 images being solved are covered by no "
        f".camrig, e.g. {b[0].resolve()}; they use EXIF inference.\n"
    ) in out


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
    assert result.cameras[0].camera == {
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
    assert result.cameras[0].camera["model"] == "SIMPLE_RADIAL_FISHEYE"
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


def test_resolve_camrig_report_print_failure_is_not_a_refused_refit(
    tmp_path: Path, monkeypatch
):
    # The refit succeeds, but printing its report fails, as it does on a stdout
    # that cannot encode θ. `UnicodeEncodeError` is a `ValueError`; it must
    # come through as itself, not as a "cannot switch" refusal.
    import sfmtool.camera.setup as camera_setup

    def failing_print(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "θ", 0, 1, "character maps to <undefined>")

    _copy_images(tmp_path / "imgs", "seoul_bull_sculpture", 1)
    imgs = sorted((tmp_path / "imgs").glob("*.jpg"))
    _make_camrig(
        tmp_path / "rig.camrig", ["imgs/*.jpg"], camera=_simple_pinhole_camera()
    )
    monkeypatch.setattr(camera_setup, "print", failing_print, raising=False)
    with pytest.raises(UnicodeEncodeError):
        resolve_camrig_for_solve(imgs, tmp_path, "SIMPLE_RADIAL_FISHEYE")


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
            "left/seoul_bull_sculpture_%02d.jpg",
            "right/seoul_bull_sculpture_%02d.jpg",
        ],
        sensor_count=2,
        camera=_camera(270, 480),
    )
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert result is not None
    assert result.is_multi_sensor
    (rig,) = result.rigs
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
            "left/seoul_bull_sculpture_%02d.jpg",
            "right/seoul_bull_sculpture_%02d.jpg",
        ],
        camera_indexes=[0, 1],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
        translations_xyz=np.zeros((2, 3)),
    )
    before = camrig_file.read_bytes()

    result = resolve_camrig_for_solve(imgs, tmp_path, "RADIAL")

    assert result.is_multi_sensor
    (rig,) = result.rigs
    cameras = rig.data["cameras"]
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
    # A wildcard beside the frame field can capture the same frame index from
    # two files of one sensor (`a_1.jpg` and `b_1.jpg` both -> 1 under
    # `*_%d.jpg`); the resolver must reject this rather than build a rig frame
    # carrying two images for one sensor.
    src = sorted((_IMAGE_DATA / "seoul_bull_sculpture").glob("*.jpg"))[0]
    left = tmp_path / "left"
    left.mkdir()
    shutil.copy(src, left / "a_1.jpg")
    shutil.copy(src, left / "b_1.jpg")
    _copy_images(tmp_path / "right", "seoul_bull_sculpture", 1)
    imgs = sorted(left.glob("*.jpg")) + sorted((tmp_path / "right").glob("*.jpg"))
    _make_camrig(
        tmp_path / "rig.camrig",
        ["left/*_%d.jpg", "right/seoul_bull_sculpture_%02d.jpg"],
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
    # it.
    imgs = _touch_images(tmp_path / "imgs", ["cam_0001.jpg", "cam_extra.jpg"])
    _make_camrig(tmp_path / "rig.camrig", ["imgs/cam_%04d.jpg"])
    ((camrig_file, _data, covered),) = find_covering_camrigs(imgs, tmp_path)
    assert camrig_file.name == "rig.camrig"
    assert covered == {imgs[0].resolve()}


def test_resolve_camrig_globstar_pattern(tmp_path: Path):
    # A `**` pattern still covers images nested below the rig directory.
    _copy_images(tmp_path / "a" / "b", "seoul_bull_sculpture", 2)
    imgs = sorted((tmp_path / "a" / "b").glob("*.jpg"))
    _make_camrig(tmp_path / "rig.camrig", ["**/*.jpg"], camera=_camera(270, 480))
    result = resolve_camrig_for_solve(imgs, tmp_path, None)
    assert result is not None
    assert result.cameras is not None and len(result.cameras) == 1
    assert result.cameras[0].camera["model"] == "PINHOLE"


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
    # Only the number as printf writes it with %04d.
    assert not m("cam_007.jpg")
    assert not m("cam_00007.jpg")
    assert camrig_pattern_matches_any_padding("cam_%04d.jpg", "cam_007.jpg", False)
    assert not camrig_pattern_matches_any_padding("cam_%04d.jpg", "cam_x.jpg", False)


def test_an_image_a_camrig_misses_only_by_padding_is_refused(tmp_path: Path):
    # A pattern written when a frame field matched any run of digits: `%d`
    # over a sequence that crosses from 0999 to 1000 now matches only 1000 on.
    imgs = _touch_images(
        tmp_path / "imgs", ["x_0998.jpg", "x_0999.jpg", "x_1000.jpg", "x_1001.jpg"]
    )
    _make_camrig(tmp_path / "old.camrig", ["imgs/x_%d.jpg"])
    with pytest.raises(CamrigSolveError) as e:
        find_covering_camrigs(imgs, tmp_path)
    message = str(e.value)
    assert "2 of the 4 images being solved are matched by no .camrig" in message
    assert str(imgs[0].resolve()) in message
    assert "'imgs/x_%d.jpg'" in message and str(tmp_path / "old.camrig") in message
    # Also when the .camrig covers none of the images at all.
    with pytest.raises(CamrigSolveError, match="1 of the 1 images"):
        find_covering_camrigs(imgs[:1], tmp_path)


def test_an_image_another_camrig_covers_is_not_a_padding_miss(tmp_path: Path):
    # Two cameras told apart by the padding of their numbers: each image is
    # covered by one .camrig, so the other's near miss is no loss.
    imgs = _touch_images(tmp_path / "imgs", ["x_7.jpg", "x_8.jpg", "x_007.jpg"])
    _make_camrig(tmp_path / "a.camrig", ["imgs/x_%d.jpg"])
    _make_camrig(tmp_path / "b.camrig", ["imgs/x_%03d.jpg"])
    hits = find_covering_camrigs(imgs, tmp_path)
    assert [(f.name, sorted(p.name for p in c)) for f, _d, c in hits] == [
        ("a.camrig", ["x_7.jpg", "x_8.jpg"]),
        ("b.camrig", ["x_007.jpg"]),
    ]


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
