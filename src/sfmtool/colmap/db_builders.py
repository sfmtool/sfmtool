# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP database feature population: the three per-camera-config-source
builders (single-sensor frames, `rig_config.json` rigs, `.camrig` files) and
their sensor and frame helpers. Driven by the orchestrators in
:mod:`db_setup`."""

import os
from pathlib import Path

import pycolmap

from ..camera.config import CameraConfigResolver
from ..camera.setup import (
    _infer_camera,
    _wrap_descriptors,
    build_intrinsics_from_camera_config,
    intrinsics_for_image,
    refit_intrinsics_to_model,
)
from ..camera.cameras import (
    colmap_camera_from_intrinsics,
    pycolmap_camera_to_intrinsics,
)
from ..camrig.resolver import CamrigCamera, CamrigRig, CamrigSolveResult
from ..rig.config import (
    _infer_frame_key,
    _match_image_to_sensor,
    _sensor_from_rig_pose,
)
from ..fileio import SiftFile


def _single_sensor_cameras(
    image_paths: list[str | Path],
    camera_model: str | None,
    camera_config_resolver: CameraConfigResolver | None,
    camrig_cameras: list[CamrigCamera] | None,
) -> list[tuple[pycolmap.Camera, list[int]]]:
    """The COLMAP cameras for images that each form a frame of their own.

    Returns `(camera, positions)` pairs, one per camera, in the order of each
    camera's first image; `positions` index `image_paths`. Each file of
    `camrig_cameras` (the single-sensor `.camrig` files covering some or all
    of the images) is one camera, and an image takes the camera of the file
    that covers it. Of the images no such file covers, those that resolve the
    same `camera_config.json` share one camera built from that file, and
    those that resolve none share one camera inferred from EXIF. A camera is
    built from its first image, so its focal length and principal point are
    scaled to that image's resolution. With `camera_model`, the camera of
    each `camera_config.json` is refit to it (one report line per camera) and
    the EXIF camera takes that model; `.camrig` cameras arrive already refit
    by the resolver.
    """
    camrig_index_of = {
        path: index
        for index, camrig_camera in enumerate(camrig_cameras or [])
        for path in camrig_camera.images
    }
    # A group key is the index of a `.camrig` camera, the path of a
    # `camera_config.json`, or `None` for the images whose camera EXIF gives.
    groups: dict[int | Path | None, list[int]] = {}
    for position, image_path in enumerate(image_paths):
        key = camrig_index_of.get(Path(image_path).resolve())
        if key is None and camera_config_resolver is not None:
            resolved = camera_config_resolver.resolve_for_image(Path(image_path))
            key = resolved[0] if resolved is not None else None
        groups.setdefault(key, []).append(position)

    cameras: list[tuple[pycolmap.Camera, list[int]]] = []
    for key, positions in groups.items():
        first_image = Path(image_paths[positions[0]])
        if isinstance(key, int):
            intrinsics, prior = build_intrinsics_from_camera_config(
                camrig_cameras[key].camera, first_image, None
            )
        else:
            intrinsics, prior = intrinsics_for_image(
                first_image, camera_config_resolver, camera_model
            )
        cam = colmap_camera_from_intrinsics(intrinsics)
        if prior:
            cam.has_prior_focal_length = True
        cameras.append((cam, positions))
    return cameras


def _write_single_sensor_images(
    db: pycolmap.Database,
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    max_feature_count: int | None,
    camera_model: str | None,
    camera_config_resolver: CameraConfigResolver | None,
    camrig_cameras: list[CamrigCamera] | None,
    include_descriptors: bool,
) -> None:
    """Write images that each form a frame of a single-sensor rig.

    Writes one camera per camera source (`_single_sensor_cameras`), each with
    a trivial rig whose only sensor it is, then each image, in the order
    given, as a frame of its camera's rig, with its keypoints and (unless
    `include_descriptors` is false) descriptors.
    """
    camera_ids: list[int] = [0] * len(image_paths)
    rig_ids: list[int] = [0] * len(image_paths)
    for cam, positions in _single_sensor_cameras(
        image_paths, camera_model, camera_config_resolver, camrig_cameras
    ):
        camera_id = db.write_camera(cam)
        rig = pycolmap.Rig()
        rig.add_ref_sensor(
            pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera_id)
        )
        rig_id = db.write_rig(rig)
        for position in positions:
            camera_ids[position] = camera_id
            rig_ids[position] = rig_id

    for image_path, sift_path, camera_id, rig_id in zip(
        image_paths, sift_paths, camera_ids, rig_ids
    ):
        with SiftFile(sift_path) as reader:
            image = pycolmap.Image(
                name=os.path.relpath(image_path, image_dir).replace("\\", "/"),
                camera_id=camera_id,
            )
            image_id = db.write_image(image)

            frame = pycolmap.Frame()
            frame.rig_id = rig_id
            frame.add_data_id(
                pycolmap.data_t(
                    sensor_id=pycolmap.sensor_t(
                        type=pycolmap.SensorType.CAMERA, id=camera_id
                    ),
                    id=image_id,
                )
            )
            frame_id = db.write_frame(frame)

            image.frame_id = frame_id
            image.image_id = image_id
            db.update_image(image)

            keypoints = reader.read_positions(count=max_feature_count)
            db.write_keypoints(image_id, keypoints)
            if include_descriptors:
                descriptors = reader.read_descriptors(count=max_feature_count)
                db.write_descriptors(image_id, _wrap_descriptors(descriptors))


def _setup_db_single_sensor(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    camera_model: str | None = None,
    camera_config_resolver: CameraConfigResolver | None = None,
    include_descriptors: bool = True,
) -> None:
    """Set up a COLMAP database whose images are single-sensor frames.

    Each image is a frame of a trivial rig whose only sensor is its camera.
    The cameras come from `camera_config.json` / EXIF inference, one per
    resolved file and one for the images that resolve none
    (`_single_sensor_cameras`).
    `include_descriptors=False` skips reading and writing the descriptor
    blocks — matchers that only use the database for pycolmap geometric
    verification (cluster, flow) never read them back, and they are by far
    the largest rows.
    """
    with pycolmap.Database.open(db_path) as db:
        _write_single_sensor_images(
            db,
            image_paths,
            sift_paths,
            image_dir,
            max_feature_count,
            camera_model,
            camera_config_resolver,
            None,
            include_descriptors,
        )


def _write_rig_frame(
    db: pycolmap.Database,
    rig_id: int,
    members: list[tuple[str, Path, int]],
    max_feature_count: int | None,
    include_descriptors: bool,
) -> None:
    """Write one frame of a rig and its images.

    Each member is `(name, sift_path, camera_id)`: the image is written on
    its sensor's camera with its keypoints and (unless `include_descriptors`
    is false) descriptors, then the frame that holds all of them, and each
    image is updated to point at that frame.
    """
    frame = pycolmap.Frame()
    frame.rig_id = rig_id

    image_entries = []
    for name, sift_path, camera_id in members:
        with SiftFile(sift_path) as reader:
            image = pycolmap.Image(name=name, camera_id=camera_id)
            image_id = db.write_image(image)
            image_entries.append((image_id, image))

            keypoints = reader.read_positions(count=max_feature_count)
            db.write_keypoints(image_id, keypoints)
            if include_descriptors:
                descriptors = reader.read_descriptors(count=max_feature_count)
                db.write_descriptors(image_id, _wrap_descriptors(descriptors))

        frame.add_data_id(
            pycolmap.data_t(
                sensor_id=pycolmap.sensor_t(
                    type=pycolmap.SensorType.CAMERA, id=camera_id
                ),
                id=image_id,
            )
        )

    frame_id = db.write_frame(frame)

    for image_id, image in image_entries:
        image.frame_id = frame_id
        image.image_id = image_id
        db.update_image(image)


def _camera_from_sensor_entry(
    cam_config: dict,
    image_path: str | Path,
    camera_model_override: str | None,
) -> pycolmap.Camera:
    """Create a pycolmap.Camera for one rig sensor.

    Uses the sensor entry's COLMAP-style ``camera_model_name`` /
    ``camera_params`` when present; otherwise falls back to ``_infer_camera``
    with the model name as a hint. Image dimensions always come from the
    image itself (the COLMAP rig config carries no width/height).

    ``camera_model_override`` (``--camera-model``) refits a sensor's
    ``camera_params`` calibration to that model
    (``refit_intrinsics_to_model``); for a sensor without one it replaces
    ``camera_model_name`` as the model inference uses.
    """
    model_name = cam_config.get("camera_model_name")
    camera_params = cam_config.get("camera_params")

    if camera_params is not None and not model_name:
        raise ValueError("rig_config.json: camera_params requires camera_model_name")

    if camera_params is None:
        return _infer_camera(str(image_path), camera_model_override or model_name)

    cam = _infer_camera(str(image_path), model_name)
    # The inferred camera already carries the correct parameter count for
    # this model; use it to validate the config's positional array.
    if len(camera_params) != len(cam.params):
        raise ValueError(
            f"rig_config.json: camera_params for {model_name} expects "
            f"{len(cam.params)} values, got {len(camera_params)}"
        )
    cam.params = [float(p) for p in camera_params]
    if camera_model_override is not None:
        intrinsics = refit_intrinsics_to_model(
            pycolmap_camera_to_intrinsics(cam),
            camera_model_override,
            f"the camera of rig_config.json sensor {cam_config.get('image_prefix')!r}",
        )
        cam = colmap_camera_from_intrinsics(intrinsics)
    cam.has_prior_focal_length = True
    return cam


def _setup_db_with_rigs(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    rig_configs: list[dict],
    camera_model: str | None = None,
    camera_config_resolver: CameraConfigResolver | None = None,
    include_descriptors: bool = True,
) -> None:
    """Set up COLMAP database with multi-sensor rigs from rig_config.json.

    `include_descriptors=False` skips the descriptor blocks (see
    `_setup_db_single_sensor`).
    """
    with pycolmap.Database.open(db_path) as db:
        # First pass: create cameras per sensor
        sensor_camera_ids: dict[tuple[int, int], int] = {}
        sensor_first_image_found: dict[tuple[int, int], bool] = {}

        for image_path in image_paths:
            rel_path = os.path.relpath(image_path, image_dir).replace("\\", "/")
            match = _match_image_to_sensor(rel_path, rig_configs)
            if match is None:
                continue
            rig_idx, sensor_idx = match
            key = (rig_idx, sensor_idx)
            if key not in sensor_first_image_found:
                sensor_first_image_found[key] = True
                cam_config = rig_configs[rig_idx]["cameras"][sensor_idx]
                cam = _camera_from_sensor_entry(cam_config, image_path, camera_model)
                camera_id = db.write_camera(cam)
                sensor_camera_ids[key] = camera_id

        # Create rigs in the database
        colmap_rig_ids: dict[int, int] = {}
        for rig_idx, rig_config in enumerate(rig_configs):
            rig = pycolmap.Rig()
            cameras = rig_config["cameras"]
            for sensor_idx, cam_config in enumerate(cameras):
                key = (rig_idx, sensor_idx)
                if key not in sensor_camera_ids:
                    continue

                camera_id = sensor_camera_ids[key]
                sensor = pycolmap.sensor_t(
                    type=pycolmap.SensorType.CAMERA, id=camera_id
                )

                if cam_config.get("ref_sensor", False):
                    rig.add_ref_sensor(sensor)
                else:
                    pose = _sensor_from_rig_pose(cam_config)
                    rig.add_sensor(sensor, pose)

            rig_id = db.write_rig(rig)
            colmap_rig_ids[rig_idx] = rig_id

        # Second pass: write images and group into frames
        frame_groups: dict[tuple[int, str], list[tuple[str, Path, int, int]]] = {}
        unmatched_images: list[tuple[str | Path, Path]] = []

        for image_path, sift_path in zip(image_paths, sift_paths):
            rel_path = os.path.relpath(image_path, image_dir).replace("\\", "/")
            match = _match_image_to_sensor(rel_path, rig_configs)
            if match is None:
                unmatched_images.append((image_path, sift_path))
                continue
            rig_idx, sensor_idx = match
            prefix = rig_configs[rig_idx]["cameras"][sensor_idx]["image_prefix"]
            frame_key = _infer_frame_key(rel_path, prefix)
            group_key = (rig_idx, frame_key)
            if group_key not in frame_groups:
                frame_groups[group_key] = []
            frame_groups[group_key].append((rel_path, sift_path, rig_idx, sensor_idx))

        # Write images and create frames
        for (rig_idx, _frame_key), group in sorted(frame_groups.items()):
            _write_rig_frame(
                db,
                colmap_rig_ids[rig_idx],
                [
                    (rel_path, sift_path, sensor_camera_ids[(r_idx, s_idx)])
                    for rel_path, sift_path, r_idx, s_idx in group
                ],
                max_feature_count,
                include_descriptors,
            )

        # Images under no sensor prefix are single-sensor frames, their
        # cameras from camera_config.json / EXIF inference.
        if unmatched_images:
            _write_single_sensor_images(
                db,
                [image_path for image_path, _sift in unmatched_images],
                [sift_path for _image, sift_path in unmatched_images],
                image_dir,
                max_feature_count,
                camera_model,
                camera_config_resolver,
                None,
                include_descriptors,
            )


def _rigid3d_sensor_from_rig(rig_data: dict, sensor_idx: int) -> pycolmap.Rigid3d:
    """Build the `sensor_from_rig` pose of a `.camrig` sensor as a Rigid3d.

    The `.camrig` stores canonical `sensor_from_rig` poses (WXYZ); this COLMAP
    database is a COLMAP-convention artifact, so S-conjugate the rig-relative
    pose to COLMAP here (`W` cancels for relative poses). pycolmap's
    `Rotation3d` takes XYZW.
    """
    import numpy as np

    from .convention import relative_pose_conjugate_s

    q = np.asarray(rig_data["quaternions_wxyz"][sensor_idx], dtype=np.float64)
    t = np.asarray(rig_data["translations_xyz"][sensor_idx], dtype=np.float64)
    q_colmap, t_colmap = relative_pose_conjugate_s(q, t)
    xyzw = np.array(
        [q_colmap[1], q_colmap[2], q_colmap[3], q_colmap[0]], dtype=np.float64
    )
    return pycolmap.Rigid3d(pycolmap.Rotation3d(xyzw), t_colmap)


def _write_camrig_rig(
    db: pycolmap.Database,
    rig: CamrigRig,
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    max_feature_count: int | None,
    include_descriptors: bool,
) -> None:
    """Write the rig of a multi-sensor `.camrig` and its images' frames.

    `image_paths` are the images being solved that `rig` covers. Writes one
    COLMAP camera per used sensor (intrinsics from the rig's camera pool,
    scaled to the actual image resolution), one rig whose reference sensor is
    the lowest-indexed sensor present — its `cam_from_rig` is the identity
    and the others are rebased relative to it — and one frame per captured
    frame index, so images sharing a frame index form one rig frame.
    """
    rig_data = rig.data
    assignments = rig.assignments

    # Resolve each image to (rel_path, sift_path, sensor, frame).
    resolved: list[tuple[str, Path, int, int]] = []
    for image_path, sift_path in zip(image_paths, sift_paths):
        sensor_idx, frame_idx = assignments[Path(image_path).resolve()]
        rel_path = os.path.relpath(image_path, image_dir).replace("\\", "/")
        resolved.append((rel_path, sift_path, sensor_idx, frame_idx))

    used_sensors = sorted({sensor_idx for _r, _s, sensor_idx, _f in resolved})
    ref_sensor = used_sensors[0]

    # A representative image per sensor, for resolution-aware intrinsics scaling.
    rep_image: dict[int, str] = {}
    for rel_path, _sift, sensor_idx, _frame in resolved:
        rep_image.setdefault(sensor_idx, rel_path)

    # One COLMAP camera per used sensor. The `.camrig` camera pool may share
    # intrinsics between sensors, but COLMAP identifies a rig sensor by its
    # camera id, so each sensor needs a distinct camera.
    sensor_camera_ids: dict[int, int] = {}
    for sensor_idx in used_sensors:
        cam_dict = rig_data["cameras"][rig_data["camera_indexes"][sensor_idx]]
        intrinsics, prior = build_intrinsics_from_camera_config(
            cam_dict, image_dir / rep_image[sensor_idx], None
        )
        cam = colmap_camera_from_intrinsics(intrinsics)
        if prior:
            cam.has_prior_focal_length = True
        sensor_camera_ids[sensor_idx] = db.write_camera(cam)

    # One rig: the reference sensor sits at the identity `cam_from_rig`; every
    # other sensor is rebased relative to it,
    # cam_from_rig[i] = sensor_from_rig[i] @ sensor_from_rig[ref]^-1.
    rig_obj = pycolmap.Rig()
    rig_from_ref = _rigid3d_sensor_from_rig(rig_data, ref_sensor).inverse()
    for sensor_idx in used_sensors:
        sensor_t = pycolmap.sensor_t(
            type=pycolmap.SensorType.CAMERA, id=sensor_camera_ids[sensor_idx]
        )
        if sensor_idx == ref_sensor:
            rig_obj.add_ref_sensor(sensor_t)
        else:
            cam_from_rig = _rigid3d_sensor_from_rig(rig_data, sensor_idx) * rig_from_ref
            rig_obj.add_sensor(sensor_t, cam_from_rig)
    rig_id = db.write_rig(rig_obj)

    # Group images into frames by captured frame index: images from different
    # sensors that share a frame index form one rig frame.
    frame_groups: dict[int, list[tuple[str, Path, int]]] = {}
    for rel_path, sift_path, sensor_idx, frame_idx in resolved:
        frame_groups.setdefault(frame_idx, []).append(
            (rel_path, sift_path, sensor_camera_ids[sensor_idx])
        )
    for _frame_idx, members in sorted(frame_groups.items()):
        _write_rig_frame(db, rig_id, members, max_feature_count, include_descriptors)


def _setup_db_with_camrigs(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    camrigs: CamrigSolveResult,
    camera_model: str | None = None,
    camera_config_resolver: CameraConfigResolver | None = None,
    include_descriptors: bool = True,
) -> None:
    """Set up a COLMAP database for images some `.camrig` files cover.

    Each multi-sensor `.camrig` of `camrigs.rigs` is written as its own rig,
    with its cameras and the frames of the images it covers
    (`_write_camrig_rig`), in path order. Every other image is a single-sensor
    frame (`_write_single_sensor_images`): an image a single-sensor `.camrig`
    of `camrigs.cameras` covers takes that file's camera, and the rest take
    cameras from `camera_config.json` / EXIF inference. The rig frames are
    written first, then the single-sensor frames in the order given.

    `include_descriptors=False` skips the descriptor blocks (see
    `_setup_db_single_sensor`).
    """
    rig_index_of = {
        path: index
        for index, rig in enumerate(camrigs.rigs)
        for path in rig.assignments
    }
    rig_members: list[list[int]] = [[] for _ in camrigs.rigs]
    single: list[int] = []
    for position, image_path in enumerate(image_paths):
        index = rig_index_of.get(Path(image_path).resolve())
        if index is None:
            single.append(position)
        else:
            rig_members[index].append(position)

    with pycolmap.Database.open(db_path) as db:
        for rig, positions in zip(camrigs.rigs, rig_members):
            _write_camrig_rig(
                db,
                rig,
                [image_paths[i] for i in positions],
                [sift_paths[i] for i in positions],
                image_dir,
                max_feature_count,
                include_descriptors,
            )
        if single:
            _write_single_sensor_images(
                db,
                [image_paths[i] for i in single],
                [sift_paths[i] for i in single],
                image_dir,
                max_feature_count,
                camera_model,
                camera_config_resolver,
                camrigs.cameras,
                include_descriptors,
            )
