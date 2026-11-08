# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The cameras, rigs and frames the database builders write: one rig per
multi-sensor `.camrig`, with its cameras and rig frames; one camera per
single-sensor `.camrig`; and, for the images no `.camrig` covers, one camera
per resolved `camera_config.json` and one for the images that resolve none,
each such image a frame of its own."""

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from sfmtool.camera.config import CameraConfigResolver
from sfmtool.camrig.resolver import resolve_camrig_for_solve
from sfmtool.colmap.db_builders import _setup_db_single_sensor, _setup_db_with_camrigs
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
    _setup_db_with_camrigs(images, sift_paths, tmp_path, db_path, None, camrig)

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


def test_images_no_camrig_covers_fall_back(tmp_path: Path, sample_sift_data, capsys):
    # Two videos with a .camrig each beside a folder with a camera_config.json
    # and one with neither, as when `sfm estimate-intrinsics` finds no
    # consensus for some sequences: one camera per .camrig, one for the
    # camera_config.json and one from EXIF, each image on its own source's.
    videos = _copy_seoul(
        tmp_path / "images",
        ["shopA_00001.jpg", "shopB_00001.jpg", "shopA_00002.jpg"],
        first=1,
    )
    gopro = _copy_seoul(tmp_path / "gopro", ["g1.jpg", "g2.jpg"], first=4)
    phone = _copy_seoul(tmp_path / "phone", ["p1.jpg"], first=6)
    (tmp_path / "gopro" / "camera_config.json").write_text(
        json.dumps({"version": 1, "camera_intrinsics": _simple_pinhole(344.0)})
    )
    _write_single_sensor_camrig(
        tmp_path / "camera-shopA.camrig", "images/shopA_%05d.jpg", _simple_pinhole(747)
    )
    _write_single_sensor_camrig(
        tmp_path / "camera-shopB.camrig",
        "images/shopB_%05d.jpg",
        _simple_pinhole(1141),
    )
    images = [gopro[0], *videos, phone[0], gopro[1]]
    sift_paths = _write_sifts(images, sample_sift_data)
    resolver = CameraConfigResolver(tmp_path)
    camrig = resolve_camrig_for_solve(images, tmp_path, None, resolver)
    out = capsys.readouterr().out
    assert (
        "Camera intrinsics: 3 of 6 images being solved are covered by no .camrig, "
        f"e.g. {gopro[0].resolve()}; they use camera_config.json (2) and EXIF "
        "inference (1).\n"
    ) in out
    db_path = tmp_path / "database.db"
    _setup_db_with_camrigs(
        images,
        sift_paths,
        tmp_path,
        db_path,
        None,
        camrig,
        camera_config_resolver=resolver,
    )

    cameras, by_image = _cameras_by_image(db_path)
    assert len(cameras) == 4
    ids = {name: camera_id for name, (camera_id, _rig) in by_image.items()}
    assert ids["images/shopA_00002.jpg"] == ids["images/shopA_00001.jpg"]
    assert ids["gopro/g2.jpg"] == ids["gopro/g1.jpg"]
    focal = {name: cameras[camera_id].params[0] for name, camera_id in ids.items()}
    assert focal["images/shopA_00001.jpg"] == 747.0
    assert focal["images/shopB_00001.jpg"] == 1141.0
    assert focal["gopro/g1.jpg"] == 344.0
    # The EXIF camera is the fourth, distinct from the files' cameras.
    exif_id = ids["phone/p1.jpg"]
    assert exif_id not in {
        ids["images/shopA_00001.jpg"],
        ids["images/shopB_00001.jpg"],
        ids["gopro/g1.jpg"],
    }
    assert not cameras[exif_id].has_prior_focal_length
    for camera_id, rig_sensor_id in by_image.values():
        assert rig_sensor_id == camera_id


def _write_two_sensor_camrig(path: Path, focals: tuple[float, float]) -> None:
    """A rig `.camrig` of `left/f_%d.jpg` (sensor 0, the reference) and
    `right/f_%d.jpg` (sensor 1, 0.1 to its side), each with its own camera."""
    write_camrig(
        path=str(path),
        name=path.stem,
        rig_type="generic",
        cameras=[_simple_pinhole(focals[0]), _simple_pinhole(focals[1])],
        sensor_image_patterns=["left/f_%d.jpg", "right/f_%d.jpg"],
        camera_indexes=[0, 1],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
        translations_xyz=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )


def _copy_rig_frames(folder: Path, first: int) -> list[Path]:
    """Two rig frames in `folder`: `left/f_1.jpg`, `right/f_1.jpg`,
    `left/f_2.jpg` and `right/f_2.jpg`."""
    left = _copy_seoul(folder / "left", ["f_1.jpg", "f_2.jpg"], first=first)
    right = _copy_seoul(folder / "right", ["f_1.jpg", "f_2.jpg"], first=first + 2)
    return [left[0], right[0], left[1], right[1]]


def _db_layout(db_path: Path):
    """The database's cameras by id; each image name's `(camera_id,
    frame_id)`; each frame's `(rig_id, image names)`; and each rig's
    reference camera id and its other sensors' camera ids, each with the
    length of its `cam_from_rig` translation."""
    with pycolmap.Database.open(db_path) as db:
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        db_images = db.read_all_images()
        images = {i.name: (i.camera_id, i.frame_id) for i in db_images}
        id_to_name = {i.image_id: i.name for i in db_images}
        frames = {
            f.frame_id: (f.rig_id, sorted(id_to_name[d.id] for d in f.data_ids))
            for f in db.read_all_frames()
        }
        rigs = {
            r.rig_id: (
                r.ref_sensor_id.id,
                {
                    s.id: float(np.linalg.norm(pose.translation))
                    for s, pose in r.non_ref_sensors.items()
                },
            )
            for r in db.read_all_rigs()
        }
    return cameras, images, frames, rigs


def test_a_rig_beside_single_images(tmp_path: Path, sample_sift_data, capsys):
    # A rig .camrig covering some of the images is their rig; a single-sensor
    # .camrig, a camera_config.json and EXIF give the other images' cameras,
    # each of those images a frame of its own.
    rig_images = _copy_rig_frames(tmp_path / "rig", first=1)
    _write_two_sensor_camrig(tmp_path / "rig" / "rig.camrig", (300.0, 310.0))
    shop = _copy_seoul(
        tmp_path / "images", ["shopA_00001.jpg", "shopA_00002.jpg"], first=5
    )
    _write_single_sensor_camrig(
        tmp_path / "camera-shopA.camrig", "images/shopA_%05d.jpg", _simple_pinhole(747)
    )
    gopro = _copy_seoul(tmp_path / "gopro", ["g1.jpg"], first=7)
    (tmp_path / "gopro" / "camera_config.json").write_text(
        json.dumps({"version": 1, "camera_intrinsics": _simple_pinhole(344.0)})
    )
    phone = _copy_seoul(tmp_path / "phone", ["p1.jpg"], first=8)
    # The rig's images interleaved with the others.
    images = [shop[0], rig_images[0], gopro[0], rig_images[3], phone[0]]
    images += [rig_images[1], shop[1], rig_images[2]]
    sift_paths = _write_sifts(images, sample_sift_data)
    resolver = CameraConfigResolver(tmp_path)
    camrig = resolve_camrig_for_solve(images, tmp_path, None, resolver)
    assert (
        "Camera intrinsics: 2 of 8 images being solved are covered by no .camrig"
        in capsys.readouterr().out
    )
    db_path = tmp_path / "database.db"
    _setup_db_with_camrigs(
        images,
        sift_paths,
        tmp_path,
        db_path,
        None,
        camrig,
        camera_config_resolver=resolver,
    )

    cameras, by_image, frames, rigs = _db_layout(db_path)
    assert len(cameras) == 5  # two rig sensors, the .camrig, the file, EXIF
    camera_of = {name: camera_id for name, (camera_id, _f) in by_image.items()}
    focal = {name: cameras[c].params[0] for name, c in camera_of.items()}
    exif_focal = focal.pop("phone/p1.jpg")
    assert focal == {
        "rig/left/f_1.jpg": 300.0,
        "rig/left/f_2.jpg": 300.0,
        "rig/right/f_1.jpg": 310.0,
        "rig/right/f_2.jpg": 310.0,
        "images/shopA_00001.jpg": 747.0,
        "images/shopA_00002.jpg": 747.0,
        "gopro/g1.jpg": 344.0,
    }
    assert exif_focal not in focal.values()
    assert camera_of["images/shopA_00001.jpg"] == camera_of["images/shopA_00002.jpg"]
    assert len(set(camera_of.values())) == 5
    prior = {name: cameras[c].has_prior_focal_length for name, c in camera_of.items()}
    assert prior.pop("phone/p1.jpg") is False
    assert all(prior.values())

    # The rig frames: one per frame index, holding both sensors' images, in a
    # rig whose reference is the left camera and whose other sensor is the
    # right camera, 0.1 away.
    left, right = camera_of["rig/left/f_1.jpg"], camera_of["rig/right/f_1.jpg"]
    (rig_id,) = [r for r, (ref, _others) in rigs.items() if ref == left]
    assert rigs[rig_id][1] == {right: pytest.approx(0.1)}
    rig_frames = sorted(names for r, names in frames.values() if r == rig_id)
    assert rig_frames == [
        ["rig/left/f_1.jpg", "rig/right/f_1.jpg"],
        ["rig/left/f_2.jpg", "rig/right/f_2.jpg"],
    ]
    # Every other image is the one image of a frame of its camera's trivial
    # rig; the two shopA images share that rig.
    assert len(frames) == 6 and len(rigs) == 4
    for name, (camera_id, frame_id) in by_image.items():
        if name.startswith("rig/"):
            continue
        frame_rig, names = frames[frame_id]
        assert names == [name]
        assert rigs[frame_rig] == (camera_id, {})


def test_a_rig_covering_some_images_alone(tmp_path: Path, sample_sift_data):
    # With no other .camrig, the images the rig does not cover take the EXIF
    # camera, each a frame of its own beside the rig frames.
    rig_images = _copy_rig_frames(tmp_path / "rig", first=1)
    _write_two_sensor_camrig(tmp_path / "rig" / "rig.camrig", (300.0, 310.0))
    phone = _copy_seoul(tmp_path / "phone", ["p1.jpg", "p2.jpg"], first=5)
    images = [phone[0], *rig_images, phone[1]]
    sift_paths = _write_sifts(images, sample_sift_data)
    camrig = resolve_camrig_for_solve(images, tmp_path, None)
    db_path = tmp_path / "database.db"
    _setup_db_with_camrigs(images, sift_paths, tmp_path, db_path, None, camrig)

    cameras, by_image, frames, rigs = _db_layout(db_path)
    assert len(cameras) == 3 and len(rigs) == 2 and len(frames) == 4
    exif = by_image["phone/p1.jpg"][0]
    assert by_image["phone/p2.jpg"][0] == exif
    assert not cameras[exif].has_prior_focal_length
    assert sorted(names for _r, names in frames.values()) == [
        ["phone/p1.jpg"],
        ["phone/p2.jpg"],
        ["rig/left/f_1.jpg", "rig/right/f_1.jpg"],
        ["rig/left/f_2.jpg", "rig/right/f_2.jpg"],
    ]
    assert sorted(len(others) for _ref, others in rigs.values()) == [0, 1]


def test_two_rigs_are_two_rigs(tmp_path: Path, sample_sift_data):
    # Each rig .camrig is its own rig, with its own cameras and frames, though
    # their frame indexes coincide.
    a = _copy_rig_frames(tmp_path / "a", first=1)
    b = _copy_rig_frames(tmp_path / "b", first=5)
    _write_two_sensor_camrig(tmp_path / "a" / "a.camrig", (300.0, 310.0))
    _write_two_sensor_camrig(tmp_path / "b" / "b.camrig", (400.0, 410.0))
    images = a + b
    sift_paths = _write_sifts(images, sample_sift_data)
    camrig = resolve_camrig_for_solve(images, tmp_path, None)
    db_path = tmp_path / "database.db"
    _setup_db_with_camrigs(images, sift_paths, tmp_path, db_path, None, camrig)

    cameras, by_image, frames, rigs = _db_layout(db_path)
    assert len(cameras) == 4 and len(rigs) == 2 and len(frames) == 4
    for folder, focals in (("a", (300.0, 310.0)), ("b", (400.0, 410.0))):
        left = by_image[f"{folder}/left/f_1.jpg"][0]
        right = by_image[f"{folder}/right/f_1.jpg"][0]
        assert (cameras[left].params[0], cameras[right].params[0]) == focals
        rig_id = frames[by_image[f"{folder}/left/f_1.jpg"][1]][0]
        assert rigs[rig_id] == (left, {right: pytest.approx(0.1)})
        assert sorted(names for r, names in frames.values() if r == rig_id) == [
            [f"{folder}/left/f_1.jpg", f"{folder}/right/f_1.jpg"],
            [f"{folder}/left/f_2.jpg", f"{folder}/right/f_2.jpg"],
        ]
