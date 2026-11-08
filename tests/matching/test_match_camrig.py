# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm match` on an image set takes its cameras and rigs from the `.camrig`
files covering the images, as `sfm solve` does
(specs/cli/image-feature/match-command.md § Camera Intrinsics)."""

import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.feature_match._db_populate import _populate_db_features
from sfmtool.fileio import read_camrig, write_camrig

from ..conftest import KERRY_PARK_DIR, SEOUL_BULL_DIR

KERRY_PARK_CAMRIG = KERRY_PARK_DIR / "kerry_park.camrig"


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


def _database(workspace: Path, images: list[Path], db_path: Path, camera_model=None):
    """The cameras, rigs, frames and images of the database `sfm match`
    writes for `images`, with `.sift` reads stubbed."""
    _populate_db_features(
        db_path,
        images,
        [Path("unused.sift")] * len(images),
        [p.relative_to(workspace).as_posix() for p in images],
        workspace,
        None,
        camera_model,
    )
    with pycolmap.Database.open(db_path) as db:
        return {
            "cameras": {c.camera_id: c for c in db.read_all_cameras()},
            "rigs": db.read_all_rigs(),
            "frames": db.read_all_frames(),
            "images": {i.image_id: i for i in db.read_all_images()},
        }


@pytest.fixture
def no_sift(monkeypatch):
    monkeypatch.setattr("sfmtool.colmap.db_builders.SiftFile", _NoSift)


def test_kerry_park_match_database_holds_the_camrig_rig(tmp_path: Path, no_sift):
    # The checked-in images are only read, and the database goes to tmp_path.
    images = sorted(KERRY_PARK_DIR.glob("fisheye_*/frame_*.jpg"))
    tables = _database(KERRY_PARK_DIR, images, tmp_path / "db.db")

    camrig = read_camrig(str(KERRY_PARK_CAMRIG))
    p = camrig["cameras"][0]["parameters"]
    expected = [
        p["focal_length_x"],
        p["focal_length_y"],
        p["principal_point_x"],
        p["principal_point_y"],
        p["radial_distortion_k1"],
        p["radial_distortion_k2"],
        p["radial_distortion_k3"],
        p["radial_distortion_k4"],
    ]
    # One camera per sensor, each the .camrig's calibrated fisheye camera.
    assert len(tables["cameras"]) == 2
    for camera in tables["cameras"].values():
        assert camera.model.name == "OPENCV_FISHEYE"
        assert (camera.width, camera.height) == (480, 480)
        np.testing.assert_allclose(camera.params, expected, rtol=1e-12)
        assert camera.has_prior_focal_length

    # One rig whose right sensor sits 180 degrees about Y, 3.07 cm behind.
    (rig,) = tables["rigs"]
    ((_sensor, pose),) = rig.non_ref_sensors.items()
    np.testing.assert_allclose(np.abs(pose.rotation.quat), [0, 1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(pose.translation, [0, 0, -0.0307], atol=1e-12)

    # 24 frames of the rig, each the left and right image of one number.
    frames = tables["frames"]
    assert len(frames) == 24
    camera_of_sensor = {}
    for frame in frames:
        assert frame.rig_id == rig.rig_id
        names = sorted(tables["images"][d.id].name for d in frame.data_ids)
        assert len(names) == 2
        left, right = names
        assert left.startswith("fisheye_left/") and right.startswith("fisheye_right/")
        assert left.split("/")[1] == right.split("/")[1]
        for d in frame.data_ids:
            image = tables["images"][d.id]
            camera_of_sensor.setdefault(image.name.split("/")[0], set()).add(
                image.camera_id
            )
    assert all(len(ids) == 1 for ids in camera_of_sensor.values())
    assert len({next(iter(ids)) for ids in camera_of_sensor.values()}) == 2


def test_camera_model_refits_the_camrig_cameras(tmp_path: Path, no_sift, capsys):
    images = sorted(KERRY_PARK_DIR.glob("fisheye_*/frame_0[12].jpg"))
    tables = _database(
        KERRY_PARK_DIR, images, tmp_path / "db.db", camera_model="THIN_PRISM_FISHEYE"
    )
    for camera in tables["cameras"].values():
        assert camera.model.name == "THIN_PRISM_FISHEYE"
        # The fit carries the calibration over: fx of the .camrig's camera.
        assert float(camera.params[0]) == pytest.approx(129.1499937015594, rel=1e-6)
    assert "OPENCV_FISHEYE -> THIN_PRISM_FISHEYE" in capsys.readouterr().out


def test_a_single_sensor_camrig_gives_its_camera(tmp_path: Path, no_sift, capsys):
    workspace = tmp_path / "ws"
    (workspace / "a").mkdir(parents=True)
    (workspace / "b").mkdir()
    sources = sorted(SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg"))
    for folder, source in (("a", sources[0]), ("a", sources[1]), ("b", sources[2])):
        shutil.copy(source, workspace / folder / source.name)
    write_camrig(
        path=str(workspace / "a" / "a.camrig"),
        name="a",
        rig_type="generic",
        cameras=[
            {
                "model": "PINHOLE",
                "width": 270,
                "height": 480,
                "parameters": {
                    "focal_length_x": 321.0,
                    "focal_length_y": 322.0,
                    "principal_point_x": 135.0,
                    "principal_point_y": 240.0,
                },
            }
        ],
        sensor_image_patterns=["seoul_bull_sculpture_*.jpg"],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )
    images = sorted(workspace.rglob("*.jpg"))
    tables = _database(workspace, images, tmp_path / "db.db")

    camera_of = {
        i.name: tables["cameras"][i.camera_id] for i in tables["images"].values()
    }
    (a0, a1) = (camera_of[f"a/{s.name}"] for s in sources[:2])
    assert a0.camera_id == a1.camera_id
    assert a0.model.name == "PINHOLE"
    np.testing.assert_allclose(a0.params, [321.0, 322.0, 135.0, 240.0])
    # The image the .camrig does not cover takes an EXIF camera of its own.
    b = camera_of[f"b/{sources[2].name}"]
    assert b.camera_id != a0.camera_id
    assert tables["rigs"] and all(not r.non_ref_sensors for r in tables["rigs"])
    out = capsys.readouterr().out
    assert "1 of 3 images being matched are covered by no .camrig" in out


def _kerry_workspace(tmp_path: Path, frames: int) -> Path:
    workspace = tmp_path / "kp"
    for sensor in ("fisheye_left", "fisheye_right"):
        (workspace / sensor).mkdir(parents=True)
        for n in range(1, frames + 1):
            name = f"frame_{n:02d}.jpg"
            shutil.copy(KERRY_PARK_DIR / sensor / name, workspace / sensor / name)
    shutil.copy(KERRY_PARK_CAMRIG, workspace / KERRY_PARK_CAMRIG.name)
    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(workspace)]
    )
    assert result.exit_code == 0, result.output
    return workspace


def test_match_exhaustive_uses_the_camrig_and_cluster_reads_none(tmp_path: Path):
    workspace = _kerry_workspace(tmp_path, frames=2)
    runner = CliRunner()
    paths = [str(workspace / "fisheye_left"), str(workspace / "fisheye_right")]

    result = runner.invoke(
        main, ["match", "--exhaustive", *paths, "-o", str(workspace / "e.matches")]
    )
    assert result.exit_code == 0, result.output
    assert f"Camera rig: using {workspace / 'kerry_park.camrig'} (2 sensors" in (
        result.output
    )

    # --cluster builds no database and needs no camera.
    result = runner.invoke(
        main, ["match", "--cluster", *paths, "-o", str(workspace / "c.matches")]
    )
    assert result.exit_code == 0, result.output
    assert "Camera rig:" not in result.output
    assert "camrig" not in result.output
