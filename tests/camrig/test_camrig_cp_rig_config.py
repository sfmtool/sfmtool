# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `sfm camrig cp` with a `rig_config.json` source: the `.camrig`
it writes groups the same images into the same sensors and frames, with the
same cameras and poses, as `rig_config.json` does in the commands."""

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_camrig

from .conftest import _IMAGE_DATA

_KERRY_PARK_CAMRIG = _IMAGE_DATA / "kerry_park" / "kerry_park.camrig"
_SMALL_IMAGE = _IMAGE_DATA / "seoul_bull_sculpture" / "seoul_bull_sculpture_01.jpg"
_SQUARE_IMAGE = _IMAGE_DATA / "kerry_park" / "fisheye_left" / "frame_01.jpg"


def _images(workspace: Path, *names: str, source: Path = _SMALL_IMAGE) -> None:
    for name in names:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, path)


def _rig_config(workspace: Path, rigs: list) -> Path:
    path = workspace / "rig_config.json"
    path.write_text(json.dumps(rigs))
    return path


def _cp(*args: str):
    return CliRunner().invoke(main, ["camrig", "cp", *args])


class _NoSift:
    """Stands in for `SiftFile` so a database is written without `.sift`
    files: one keypoint and descriptor per image."""

    def __init__(self, path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read_positions(self, count=None):
        return np.zeros((1, 2), np.float32)

    def read_descriptors(self, count=None):
        return np.zeros((1, 128), np.uint8)


def _database_rig_tables(workspace: Path, db_path: Path) -> dict:
    """The cameras, rigs, frames and images `sfm solve` writes to its COLMAP
    database for every image of `workspace`, from whatever rig description the
    workspace holds."""
    from sfmtool.colmap.db_setup import _resolve_camrigs, _write_image_features
    from sfmtool.rig.config import _load_rig_config

    images = sorted(p.resolve() for p in workspace.rglob("*.jpg"))
    rig_config = _load_rig_config(workspace)
    camrigs = _resolve_camrigs(images, workspace, rig_config, None, None, "images")
    sift_paths = [Path("unused.sift")] * len(images)
    _write_image_features(
        images, sift_paths, workspace, db_path, None, camrigs, rig_config, None, None
    )
    with pycolmap.Database.open(db_path) as db:
        return {
            "cameras": [
                (c.camera_id, c.model.name, c.width, c.height, list(c.params))
                + (c.has_prior_focal_length,)
                for c in db.read_all_cameras()
            ],
            "rigs": [
                (
                    r.rig_id,
                    r.ref_sensor_id.id,
                    sorted(
                        (s.id, list(p.rotation.quat), list(p.translation))
                        for s, p in r.non_ref_sensors.items()
                    ),
                )
                for r in db.read_all_rigs()
            ],
            "frames": [
                (f.frame_id, f.rig_id, sorted(d.id for d in f.data_ids))
                for f in db.read_all_frames()
            ],
            "images": [
                (i.image_id, i.name, i.camera_id, i.frame_id)
                for i in db.read_all_images()
            ],
        }


def test_kerry_park_rig_config_becomes_its_camrig(isolated_kerry_park_rig: Path):
    out = isolated_kerry_park_rig / "kerry_park.camrig"
    result = _cp(str(isolated_kerry_park_rig / "rig_config.json"), str(out))
    assert result.exit_code == 0, result.output
    assert "2 sensors, 24 frames, 1 camera(s)" in result.output
    assert (
        "sensor 1: fisheye_right/ -> fisheye_right/frame_%02d.jpg  "
        "(24 images, OPENCV_FISHEYE 480x480)"
    ) in result.output
    assert "note:" not in result.output

    converted = read_camrig(str(out))
    expected = read_camrig(str(_KERRY_PARK_CAMRIG))
    assert converted["sensor_image_patterns"] == expected["sensor_image_patterns"]
    assert converted["cameras"] == expected["cameras"]
    assert list(converted["camera_indexes"]) == list(expected["camera_indexes"])
    # q and -q are the same rotation.
    for q, q_expected in zip(
        converted["quaternions_wxyz"], expected["quaternions_wxyz"]
    ):
        assert abs(abs(np.dot(q, q_expected)) - 1.0) < 1e-12
    np.testing.assert_allclose(
        converted["translations_xyz"], expected["translations_xyz"], atol=1e-12
    )


def test_kerry_park_database_is_the_same_from_either_file(
    isolated_kerry_park_rig: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("sfmtool.colmap.db_builders.SiftFile", _NoSift)
    with_camrig = tmp_path / "with_camrig"
    shutil.copytree(isolated_kerry_park_rig, with_camrig)
    result = _cp(
        str(with_camrig / "rig_config.json"), str(with_camrig / "converted.camrig")
    )
    assert result.exit_code == 0, result.output
    (with_camrig / "rig_config.json").unlink()

    from_rig_config = _database_rig_tables(isolated_kerry_park_rig, tmp_path / "a.db")
    from_camrig = _database_rig_tables(with_camrig, tmp_path / "b.db")
    assert len(from_rig_config["frames"]) == 24
    assert len(from_rig_config["images"]) == 48
    assert from_camrig == from_rig_config


def test_each_rig_of_the_file_becomes_a_camrig(tmp_path: Path):
    frames = range(1, 4)
    _images(tmp_path, *(f"rig/cam_{s}/pan_{n:03d}.jpg" for s in "ab" for n in frames))
    # A frame of the second rig lacks the image of its second sensor.
    _images(tmp_path, "other/x/f_1.jpg", "other/x/f_2.jpg", "other/y/f_1.jpg")
    source = _rig_config(
        tmp_path,
        [
            {
                "cameras": [
                    # The reference sensor is listed second.
                    {
                        "image_prefix": "rig/cam_b/",
                        "cam_from_rig_rotation": [0.7071, 0, 0.7071, 0],
                        "cam_from_rig_translation": [0.1, 0.2, 0.3],
                    },
                    {
                        "image_prefix": "rig/cam_a/",
                        "ref_sensor": True,
                        "camera_model_name": "PINHOLE",
                        "camera_params": [500, 500, 135, 240],
                    },
                ]
            },
            {
                "cameras": [
                    {"image_prefix": "other/x/", "ref_sensor": True},
                    {
                        "image_prefix": "other/y/",
                        "cam_from_rig_rotation": [1, 0, 0, 0],
                        "camera_model_name": "OPENCV_FISHEYE",
                    },
                ]
            },
        ],
    )

    result = _cp(str(source), str(tmp_path / "r.camrig"))
    assert result.exit_code == 1
    assert "has 2 rigs; pass --rig N (0..1)" in result.output

    result = _cp(str(source), str(tmp_path / "rig" / "first.camrig"), "--rig", "0")
    assert result.exit_code == 0, result.output
    first = read_camrig(str(tmp_path / "rig" / "first.camrig"))
    # Relative to the folder of the .camrig, the reference sensor first.
    assert first["sensor_image_patterns"] == [
        "cam_a/pan_%03d.jpg",
        "cam_b/pan_%03d.jpg",
    ]
    assert first["cameras"][0]["parameters"]["focal_length_x"] == 500.0
    assert first["cameras"][0]["width"] == 270
    q = np.asarray(first["quaternions_wxyz"])
    np.testing.assert_allclose(q[0], [1, 0, 0, 0])
    # COLMAP's cam_from_rig conjugated by S = diag(1, -1, -1).
    np.testing.assert_allclose(
        q[1] * np.sign(q[1][0]), np.array([1, 0, -1, 0]) / np.sqrt(2), atol=1e-12
    )
    np.testing.assert_allclose(first["translations_xyz"][1], [0.1, -0.2, -0.3])
    assert "note:" in result.output and "sensor 1 has no camera_params" in (
        result.output
    )

    result = _cp(str(source), str(tmp_path / "second.camrig"), "--rig", "1")
    assert result.exit_code == 0, result.output
    assert "2 sensors, 2 frames" in result.output
    assert "other/y/ -> other/y/f_%d.jpg  (1 images, OPENCV_FISHEYE" in result.output
    second = read_camrig(str(tmp_path / "second.camrig"))
    fisheye = second["cameras"][second["camera_indexes"][1]]
    # Inferred as `sfm solve` infers a fisheye sensor: f = min(w, h) / pi.
    assert fisheye["parameters"]["focal_length_x"] == pytest.approx(270 / 3.14159)
    assert "sensor 0 has no camera_params; its camera was inferred from " in (
        result.output
    )
    assert "other/x/f_1.jpg" in result.output


def _two_sensor_rig(a: str = "a/", b: str = "b/") -> list:
    return [
        {
            "cameras": [
                {"image_prefix": a, "ref_sensor": True},
                {"image_prefix": b, "cam_from_rig_rotation": [1, 0, 0, 0]},
            ]
        }
    ]


@pytest.mark.parametrize(
    ("names", "rigs", "rig", "message"),
    [
        (
            ["a/left.jpg", "b/left.jpg"],
            _two_sensor_rig(),
            "0",
            "have no file number, so -r cannot select them and they have no place "
            "in a sequence: a/left.jpg",
        ),
        (
            ["a/f_7.jpg", "a/f_8.jpg", "b/f_07.jpg", "b/f_08.jpg"],
            _two_sensor_rig(),
            "0",
            "2 frame number(s) of the patterns join images that rig_config.json "
            "puts in different frames: frame 7 ('f_07.jpg', 'f_7.jpg'), frame 8",
        ),
        (
            ["a/f_01.jpg", "a/g_01.jpg", "b/f_01.jpg"],
            _two_sensor_rig(),
            "0",
            "the images under 'a/' form 2 sequences, and a .camrig sensor has one "
            "image pattern: a/f_%02d.jpg (1 images); a/g_%02d.jpg (1 images)",
        ),
        (
            ["cam/f_01.jpg", "cam/f_02.jpg", "cam/f_03.jpg", "cam2/f_01.jpg"],
            [
                {"cameras": [{"image_prefix": "cam/f_02", "ref_sensor": True}]},
                *_two_sensor_rig("cam/", "cam2/"),
            ],
            "1",
            "the pattern 'cam/f_%02d.jpg' also matches 1 file(s) that "
            "rig_config.json does not give this sensor: cam/f_02.jpg",
        ),
        (
            ["cam/f_01.jpg", "cam/f_02.jpg", "cam/f_10.jpg"],
            _two_sensor_rig("cam/f_0", "cam/"),
            "0",
            "is matched by sensor 0 and sensor 1",
        ),
        (
            ["a/f_01.jpg"],
            _two_sensor_rig(),
            "0",
            "no image files under",
        ),
    ],
    ids=[
        "no-number",
        "keys-apart",
        "two-sequences",
        "other-sensor",
        "overlap",
        "empty",
    ],
)
def test_a_rig_the_patterns_cannot_reproduce_is_refused(
    tmp_path: Path, names, rigs, rig, message
):
    _images(tmp_path, *names)
    source = _rig_config(tmp_path, rigs)
    out = tmp_path / "r.camrig"
    result = _cp(str(source), str(out), "--rig", rig)
    assert result.exit_code == 1, result.output
    assert message in " ".join(result.output.split()), result.output
    assert not out.exists()


def test_mixed_resolutions_of_a_sensor_are_refused(tmp_path: Path):
    _images(tmp_path, "a/f_1.jpg", "b/f_1.jpg")
    _images(tmp_path, "a/f_2.jpg", source=_SQUARE_IMAGE)
    source = _rig_config(tmp_path, _two_sensor_rig())
    result = _cp(str(source), str(tmp_path / "r.camrig"))
    assert result.exit_code == 1
    assert "(sensor 'a/') have mixed resolutions" in result.output
    assert "270x480" in result.output and "480x480" in result.output


def test_the_camrig_must_be_beside_or_under_the_rig_config(tmp_path: Path):
    _images(tmp_path / "ws", "rig/a/f_1.jpg", "rig/b/f_1.jpg")
    source = _rig_config(tmp_path / "ws", _two_sensor_rig("rig/a/", "rig/b/"))

    result = _cp(str(source), str(tmp_path / "r.camrig"))
    assert result.exit_code == 1
    assert "is outside" in result.output

    result = _cp(str(source), str(tmp_path / "ws" / "rig" / "a" / "r.camrig"))
    assert result.exit_code == 1
    assert "1 image(s) of rig 0 are not under" in result.output
    assert "rig/b/f_1.jpg" in result.output

    result = _cp(str(source), str(tmp_path / "ws" / "r.camrig"), "--camera", "0")
    assert result.exit_code == 2
    assert "takes only --rig" in result.output


def test_a_camrig_already_covering_the_images_is_named(tmp_path: Path):
    _images(tmp_path, "a/f_1.jpg", "b/f_1.jpg")
    source = _rig_config(tmp_path, _two_sensor_rig())
    assert _cp(str(source), str(tmp_path / "first.camrig")).exit_code == 0
    result = _cp(str(source), str(tmp_path / "second.camrig"))
    assert result.exit_code == 0, result.output
    assert "first.camrig already covers some of these images" in result.output
    # Writing over the same file is not a second file covering them.
    result = _cp(str(source), str(tmp_path / "second.camrig"))
    assert "second.camrig already covers" not in result.output


def test_a_camrig_missing_the_images_only_by_padding_is_not_named(tmp_path: Path):
    """A `.camrig` whose patterns miss the images only by the zero-padding of
    their numbers covers none of them, so `sfm solve` does not refuse them as
    covered twice, and `cp` does not name it."""
    _images(tmp_path, "a/f_1.jpg", "b/f_1.jpg")
    source = _rig_config(tmp_path, _two_sensor_rig())
    assert _cp(str(source), str(tmp_path / "first.camrig")).exit_code == 0
    assert read_camrig(tmp_path / "first.camrig")["sensor_image_patterns"] == [
        "a/f_%d.jpg",
        "b/f_%d.jpg",
    ]
    for sensor in "ab":
        (tmp_path / sensor / "f_1.jpg").rename(tmp_path / sensor / "f_01.jpg")

    result = _cp(str(source), str(tmp_path / "second.camrig"))
    assert result.exit_code == 0, result.output
    assert read_camrig(tmp_path / "second.camrig")["sensor_image_patterns"] == [
        "a/f_%02d.jpg",
        "b/f_%02d.jpg",
    ]
    assert "already covers" not in result.output


def test_frame_check_refuses_what_the_patterns_miss_or_split(tmp_path: Path):
    """The frame check against the files on disk, for differences that
    patterns derived from the images themselves do not produce."""
    from sfmtool.camrig.cp import CamrigCpError
    from sfmtool.camrig.from_rig_config import _check_frames

    _images(tmp_path, "a/f_1.jpg", "a/f_2.jpg")
    out = tmp_path / "r.camrig"
    one, two = (tmp_path / "a/f_1.jpg").resolve(), (tmp_path / "a/f_2.jpg").resolve()
    absent = (tmp_path / "a/g_1.jpg").resolve()

    with pytest.raises(CamrigCpError, match=r"does not match 1 image\(s\) .*a/g_1"):
        _check_frames(out, ["a/f_%d.jpg"], [[one, two, absent]], {})
    with pytest.raises(
        CamrigCpError, match=r"1 frame key\(s\) .* 'k' \(frames \[1, 2\]\)"
    ):
        _check_frames(out, ["a/f_%d.jpg"], [[one, two]], {one: "k", two: "k"})
    assignments = _check_frames(out, ["a/f_%d.jpg"], [[one, two]], {one: "1", two: "2"})
    assert assignments == {one: (0, 1), two: (0, 2)}


def test_an_unusable_rig_config_is_refused(tmp_path: Path):
    _images(tmp_path, "a/f_1.jpg", "b/f_1.jpg")
    source = tmp_path / "rig_config.json"
    out = str(tmp_path / "r.camrig")

    source.write_text("[{")
    result = _cp(str(source), out)
    assert result.exit_code == 1 and "Invalid JSON in" in result.output

    _rig_config(tmp_path, [{"cameras": [{"image_prefix": "a/"}]}])
    result = _cp(str(source), out)
    assert result.exit_code == 1
    assert "non-reference sensors need 'cam_from_rig_rotation'" in result.output

    # A .json that is not a rig_config.json, such as a camera_config.json.
    other = tmp_path / "camera_config.json"
    other.write_text(json.dumps({"camera_model": "PINHOLE"}))
    result = _cp(str(other), out)
    assert result.exit_code == 1
    assert "read as a rig_config.json, and this one is not one" in " ".join(
        result.output.split()
    )
    assert "top level must be a JSON array of rig objects" in result.output

    _rig_config(tmp_path, _two_sensor_rig())
    result = _cp(str(source), out, "--rig", "1")
    assert result.exit_code == 1
    assert "--rig 1 is out of range" in result.output and "has 1 rig(s)" in (
        result.output
    )
