# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP database feature population: the three per-camera-config-source
builders (single camera, `rig_config.json` rigs, `.camrig`) and their sensor
helpers. Driven by the orchestrators in :mod:`db_setup`."""

import os
from pathlib import Path

import pycolmap

from ..camera.config import CameraConfigResolver
from ..camera.setup import (
    _infer_camera,
    _wrap_descriptors,
    build_intrinsics_from_camera_config,
    intrinsics_for_image,
)
from ..camera.cameras import colmap_camera_from_intrinsics
from ..camrig.resolver import CamrigRig
from ..rig.config import (
    _infer_frame_key,
    _match_image_to_sensor,
    _sensor_from_rig_pose,
)
from ..sift.file import SiftReader


def _setup_db_single_camera(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    camera_model: str | None = None,
    camera_config_resolver: CameraConfigResolver | None = None,
    camrig_camera: dict | None = None,
    include_descriptors: bool = True,
) -> None:
    """Set up COLMAP database with a single-camera trivial rig.

    When `camrig_camera` (a `{model, width, height, parameters}` dict from a
    discovered `.camrig`) is given, it supplies the camera; otherwise the
    intrinsics come from `camera_config.json` / EXIF inference.
    `include_descriptors=False` skips reading and writing the descriptor
    blocks — matchers that only use the database for pycolmap geometric
    verification (cluster, flow) never read them back, and they are by far
    the largest rows.
    """
    with pycolmap.Database.open(db_path) as db:
        if camrig_camera is not None:
            intrinsics, prior = build_intrinsics_from_camera_config(
                camrig_camera, Path(image_paths[0]), None
            )
        else:
            intrinsics, prior = intrinsics_for_image(
                Path(image_paths[0]), camera_config_resolver, camera_model
            )
        cam = colmap_camera_from_intrinsics(intrinsics)
        if prior:
            cam.has_prior_focal_length = True
        camera_id = db.write_camera(cam)

        rig = pycolmap.Rig()
        rig.add_ref_sensor(
            pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera_id)
        )
        rig_id = db.write_rig(rig)

        for image_path, sift_path in zip(image_paths, sift_paths):
            with SiftReader(sift_path) as reader:
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
    """
    model_name = cam_config.get("camera_model_name")
    camera_params = cam_config.get("camera_params")

    if camera_params is not None and not model_name:
        raise ValueError("rig_config.json: camera_params requires camera_model_name")

    effective_model = camera_model_override or model_name
    cam = _infer_camera(str(image_path), effective_model)

    # An explicit --camera-model override takes precedence over the config's
    # calibrated parameters.
    if camera_model_override is None and camera_params is not None:
        # The inferred camera already carries the correct parameter count for
        # this model; use it to validate the config's positional array.
        if len(camera_params) != len(cam.params):
            raise ValueError(
                f"rig_config.json: camera_params for {model_name} expects "
                f"{len(cam.params)} values, got {len(camera_params)}"
            )
        cam.params = [float(p) for p in camera_params]
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
    `_setup_db_single_camera`).
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
        unmatched_images: list[tuple[str, Path]] = []

        for image_path, sift_path in zip(image_paths, sift_paths):
            rel_path = os.path.relpath(image_path, image_dir).replace("\\", "/")
            match = _match_image_to_sensor(rel_path, rig_configs)
            if match is None:
                unmatched_images.append((rel_path, sift_path))
                continue
            rig_idx, sensor_idx = match
            prefix = rig_configs[rig_idx]["cameras"][sensor_idx]["image_prefix"]
            frame_key = _infer_frame_key(rel_path, prefix)
            group_key = (rig_idx, frame_key)
            if group_key not in frame_groups:
                frame_groups[group_key] = []
            frame_groups[group_key].append((rel_path, sift_path, rig_idx, sensor_idx))

        # Write images and create frames
        for (rig_idx, frame_key), group in sorted(frame_groups.items()):
            colmap_rig_id = colmap_rig_ids[rig_idx]

            frame = pycolmap.Frame()
            frame.rig_id = colmap_rig_id

            image_entries = []
            for rel_path, sift_path, r_idx, s_idx in group:
                camera_id = sensor_camera_ids[(r_idx, s_idx)]
                with SiftReader(sift_path) as reader:
                    image = pycolmap.Image(name=rel_path, camera_id=camera_id)
                    image_id = db.write_image(image)
                    image_entries.append((image_id, camera_id, sift_path, image))

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

            for image_id, camera_id, sift_path, image in image_entries:
                image.frame_id = frame_id
                image.image_id = image_id
                db.update_image(image)

        # Handle unmatched images with trivial single-camera rigs
        if unmatched_images:
            first_unmatched = image_dir / unmatched_images[0][0]
            intrinsics, prior = intrinsics_for_image(
                first_unmatched, camera_config_resolver, camera_model
            )
            cam = colmap_camera_from_intrinsics(intrinsics)
            if prior:
                cam.has_prior_focal_length = True
            camera_id = db.write_camera(cam)
            fallback_rig = pycolmap.Rig()
            fallback_rig.add_ref_sensor(
                pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera_id)
            )
            fallback_rig_id = db.write_rig(fallback_rig)

            for rel_path, sift_path in unmatched_images:
                with SiftReader(sift_path) as reader:
                    image = pycolmap.Image(name=rel_path, camera_id=camera_id)
                    image_id = db.write_image(image)

                    frame = pycolmap.Frame()
                    frame.rig_id = fallback_rig_id
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


def _setup_db_with_camrig(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    rig: CamrigRig,
    include_descriptors: bool = True,
) -> None:
    """Set up a COLMAP database from a multi-sensor `.camrig`.

    `include_descriptors=False` skips the descriptor blocks (see
    `_setup_db_single_camera`).

    Builds one COLMAP camera per used sensor (intrinsics from the rig's camera
    pool, scaled to the actual image resolution), a single rig whose reference
    sensor is the lowest-indexed sensor present — its `cam_from_rig` is the
    identity and the others are rebased relative to it — and one frame per
    captured frame index, so images sharing a frame index form one rig frame.
    """
    rig_data = rig.data
    assignments = rig.assignments

    # Resolve each solve image to (rel_path, sift_path, sensor, frame).
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

    with pycolmap.Database.open(db_path) as db:
        # One COLMAP camera per used sensor. The `.camrig` camera pool may
        # share intrinsics between sensors, but COLMAP identifies a rig sensor
        # by its camera id, so each sensor needs a distinct camera.
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

        # One rig: the reference sensor sits at the identity `cam_from_rig`;
        # every other sensor is rebased relative to it,
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
                cam_from_rig = (
                    _rigid3d_sensor_from_rig(rig_data, sensor_idx) * rig_from_ref
                )
                rig_obj.add_sensor(sensor_t, cam_from_rig)
        rig_id = db.write_rig(rig_obj)

        # Group images into frames by captured frame index: images from
        # different sensors that share a frame index form one rig frame.
        frame_groups: dict[int, list[tuple[str, Path, int]]] = {}
        for rel_path, sift_path, sensor_idx, frame_idx in resolved:
            frame_groups.setdefault(frame_idx, []).append(
                (rel_path, sift_path, sensor_idx)
            )

        for _frame_idx, group in sorted(frame_groups.items()):
            frame = pycolmap.Frame()
            frame.rig_id = rig_id

            image_entries = []
            for rel_path, sift_path, sensor_idx in group:
                camera_id = sensor_camera_ids[sensor_idx]
                with SiftReader(sift_path) as reader:
                    image = pycolmap.Image(name=rel_path, camera_id=camera_id)
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
