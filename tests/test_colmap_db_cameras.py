# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The cameras `_setup_db_single_sensor` writes to a COLMAP database: one per
single-sensor `.camrig`, or one per resolved `camera_config.json` plus one for
the images that resolve none."""

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from sfmtool.camera.config import CameraConfigResolver
from sfmtool.camrig.resolver import resolve_camrig_for_solve
from sfmtool.colmap.db_builders import _setup_db_single_sensor
from sfmtool.fileio import write_camrig, write_sift

SEOUL_DIR = (
    Path(__file__).parent.parent / "test-data" / "images" / "seoul_bull_sculpture"
)


def _copy_seoul(dest: Path, names: list[str], first: int) -> list[Path]:
    """Copy consecutive seoul_bull images (from number `first`) to `dest`
    under `names`."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for offset, name in enumerate(names):
        source = SEOUL_DIR / f"seoul_bull_sculpture_{first + offset:02d}.jpg"
        shutil.copy(source, dest / name)
        paths.append(dest / name)
    return paths


def _write_sifts(image_paths: list[Path], sample_sift_data) -> list[Path]:
    """A `.sift` of the sample features beside each image; the database
    builder reads only their keypoints and descriptors."""
    feature_tool_metadata, metadata, positions, affine, descriptors, thumbnail = (
        sample_sift_data
    )
    sift_paths = []
    for image_path in image_paths:
        sift_path = image_path.with_name(image_path.name + ".sift")
        write_sift(
            sift_path,
            {
                "feature_tool_metadata": feature_tool_metadata,
                "metadata": dict(metadata, image_name=image_path.name),
                "positions_xy": positions,
                "affine_shapes": affine,
                "descriptors": descriptors,
                "thumbnail_y_x_rgb": thumbnail,
            },
        )
        sift_paths.append(sift_path)
    return sift_paths


def _simple_pinhole(focal: float) -> dict:
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


def _write_single_sensor_camrig(path: Path, pattern: str, camera: dict) -> None:
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[camera],
        sensor_image_patterns=[pattern],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        translations_xyz=np.zeros((1, 3)),
    )


def _cameras_by_image(db_path: Path) -> tuple[dict[int, pycolmap.Camera], dict]:
    """The database's cameras by id, and each image name's camera id and the
    rig its frame belongs to."""
    with pycolmap.Database.open(db_path) as db:
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        frames = {f.frame_id: f for f in db.read_all_frames()}
        rigs = {r.rig_id: r for r in db.read_all_rigs()}
        by_image = {}
        for image in db.read_all_images():
            rig = rigs[frames[image.frame_id].rig_id]
            by_image[image.name] = (image.camera_id, rig.ref_sensor_id.id)
    return cameras, by_image


def test_two_camrigs_give_two_cameras(tmp_path: Path, sample_sift_data):
    # The motivating case: two videos shot with different lenses, frames
    # side by side in one folder, one single-sensor .camrig per video.
    images = _copy_seoul(
        tmp_path / "images",
        ["shopA_00001.jpg", "shopB_00001.jpg", "shopA_00002.jpg"],
        first=1,
    )
    sift_paths = _write_sifts(images, sample_sift_data)
    _write_single_sensor_camrig(
        tmp_path / "camera-shopA.camrig", "images/shopA_%05d.jpg", _simple_pinhole(747)
    )
    _write_single_sensor_camrig(
        tmp_path / "camera-shopB.camrig",
        "images/shopB_%05d.jpg",
        _simple_pinhole(1141),
    )
    camrig = resolve_camrig_for_solve(images, tmp_path, None)
    db_path = tmp_path / "database.db"
    _setup_db_single_sensor(
        images,
        sift_paths,
        tmp_path,
        db_path,
        None,
        camrig_cameras=camrig.cameras,
    )

    cameras, by_image = _cameras_by_image(db_path)
    assert len(cameras) == 2
    a_id = by_image["images/shopA_00001.jpg"][0]
    b_id = by_image["images/shopB_00001.jpg"][0]
    assert by_image["images/shopA_00002.jpg"][0] == a_id
    assert a_id != b_id
    assert cameras[a_id].model.name == "SIMPLE_PINHOLE"
    assert list(cameras[a_id].params) == [747.0, 135.0, 240.0]
    assert list(cameras[b_id].params) == [1141.0, 135.0, 240.0]
    assert cameras[a_id].has_prior_focal_length
    assert cameras[b_id].has_prior_focal_length
    # Each image is a frame of the trivial rig of its own camera.
    for camera_id, rig_sensor_id in by_image.values():
        assert rig_sensor_id == camera_id


def test_one_camera_per_camera_config_and_one_for_exif(
    tmp_path: Path, sample_sift_data, capsys
):
    # Two folders with their own camera_config.json and a third with none:
    # each file is one camera, and the third folder's images share one
    # camera inferred from EXIF.
    gopro = _copy_seoul(tmp_path / "gopro", ["g1.jpg", "g2.jpg"], first=1)
    drone = _copy_seoul(tmp_path / "drone", ["d1.jpg"], first=3)
    phone = _copy_seoul(tmp_path / "phone", ["p1.jpg", "p2.jpg"], first=4)
    for folder, focal in (("gopro", 344.0), ("drone", 500.0)):
        (tmp_path / folder / "camera_config.json").write_text(
            json.dumps(
                {"version": 1, "camera_intrinsics": _simple_pinhole(focal)},
            )
        )
    images = [gopro[0], phone[0], drone[0], gopro[1], phone[1]]
    sift_paths = _write_sifts(images, sample_sift_data)
    db_path = tmp_path / "database.db"
    _setup_db_single_sensor(
        images,
        sift_paths,
        tmp_path,
        db_path,
        None,
        camera_model="simple_radial",
        camera_config_resolver=CameraConfigResolver(tmp_path),
    )

    cameras, by_image = _cameras_by_image(db_path)
    assert len(cameras) == 3
    gopro_id = by_image["gopro/g1.jpg"][0]
    phone_id = by_image["phone/p1.jpg"][0]
    drone_id = by_image["drone/d1.jpg"][0]
    assert by_image["gopro/g2.jpg"][0] == gopro_id
    assert by_image["phone/p2.jpg"][0] == phone_id
    assert len({gopro_id, phone_id, drone_id}) == 3
    # The files' cameras, refit to --camera-model, keep their focal lengths.
    assert cameras[gopro_id].model.name == "SIMPLE_RADIAL"
    assert cameras[gopro_id].params[0] == pytest.approx(344.0)
    assert cameras[drone_id].params[0] == pytest.approx(500.0)
    assert cameras[gopro_id].has_prior_focal_length
    assert cameras[drone_id].has_prior_focal_length
    # The EXIF camera takes the model --camera-model names and no prior.
    assert cameras[phone_id].model.name == "SIMPLE_RADIAL"
    assert not cameras[phone_id].has_prior_focal_length
    # One report line per refit file.
    out = capsys.readouterr().out
    for folder in ("gopro", "drone"):
        line = f"The camera of {(tmp_path / folder).resolve() / 'camera_config.json'}: "
        assert out.count(line + "SIMPLE_PINHOLE -> SIMPLE_RADIAL") == 1
