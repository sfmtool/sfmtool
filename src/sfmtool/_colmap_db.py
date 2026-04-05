# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP database setup for SfM solvers."""

import os
from pathlib import Path

import pycolmap

from ._camera_setup import _infer_camera, _wrap_descriptors
from ._rig_config import (
    _infer_frame_key,
    _match_image_to_sensor,
    _sensor_from_rig_pose,
)
from ._rig_frames import _build_cross_frame_pairs, _build_same_frame_index_pairs
from ._sift_file import SiftReader, image_files_to_sift_files


def _setup_for_sfm(
    image_paths: list[str | Path],
    colmap_dir: str | Path,
    workspace_dir: str | Path,
    max_feature_count: int | None = None,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
    feature_prefix_dir: str | None = None,
    rig_config: list[dict] | None = None,
    camera_model: str | None = None,
    matching_mode: str = "exhaustive",
    flow_preset: str = "default",
    flow_wide_baseline_skip: int = 5,
) -> tuple[Path, Path]:
    """Prepare a COLMAP database for running the mapper.

    Returns:
        tuple: (db_path, image_dir)
    """
    sift_paths = image_files_to_sift_files(
        image_paths,
        feature_tool=feature_tool,
        feature_options=feature_options,
        feature_prefix_dir=feature_prefix_dir,
    )

    colmap_dir = Path(colmap_dir)
    colmap_dir.mkdir(exist_ok=True, parents=True)
    db_path = colmap_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    image_dir = Path(workspace_dir)

    if rig_config is not None:
        _setup_db_with_rigs(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count,
            rig_config,
            camera_model=camera_model,
        )
    else:
        _setup_db_single_camera(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            max_feature_count,
            camera_model=camera_model,
        )

    # Build same-frame exclusion data for multi-sensor rigs
    same_frame_index_pairs: set[tuple[int, int]] | None = None
    if rig_config is not None:
        same_frame_index_pairs = _build_same_frame_index_pairs(
            db_path, image_paths, image_dir
        )
        if not same_frame_index_pairs:
            same_frame_index_pairs = None

    # Run feature matching
    if matching_mode == "flow":
        from ._commands.match import _run_flow_matching

        _run_flow_matching(
            image_paths,
            sift_paths,
            image_dir,
            db_path,
            colmap_dir,
            max_feature_count=max_feature_count,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
        )
    elif rig_config is not None and same_frame_index_pairs:
        cross_frame_pairs = _build_cross_frame_pairs(db_path)
        pairs_path = colmap_dir / "match_pairs.txt"
        with open(pairs_path, "w") as f:
            for name_i, name_j in cross_frame_pairs:
                f.write(f"{name_i} {name_j}\n")
        pairing_opts = pycolmap.ImportedPairingOptions()
        pairing_opts.match_list_path = str(pairs_path)
        pycolmap.match_image_pairs(db_path, pairing_options=pairing_opts)
    else:
        pycolmap.match_exhaustive(db_path)

    return db_path, image_dir


def _setup_for_sfm_from_matches(
    matches_file: str | Path,
    colmap_dir: str | Path,
    camera_model: str | None = None,
) -> tuple[Path, Path, list[Path]]:
    """Prepare a COLMAP database from a .matches file for running the mapper.

    Returns:
        tuple: (db_path, image_dir, image_paths)
    """
    from ._sfmtool import read_matches
    from ._workspace import find_workspace_for_path

    matches_file = Path(matches_file)
    colmap_dir = Path(colmap_dir)

    print(f"Loading matches from: {matches_file}")
    matches_data = read_matches(str(matches_file))

    metadata = matches_data["metadata"]
    ws_meta = metadata["workspace"]
    image_names = matches_data["image_names"]
    image_count = metadata["image_count"]

    # Resolve workspace directory
    workspace_dir = None
    matches_dir = matches_file.parent.absolute()
    rel_path = ws_meta.get("relative_path", "")
    if rel_path:
        candidate = (matches_dir / rel_path).resolve()
        ws_marker = candidate / ".sfm-workspace.json"
        if ws_marker.exists():
            workspace_dir = candidate

    if workspace_dir is None:
        abs_path = ws_meta.get("absolute_path", "")
        if abs_path:
            candidate = Path(abs_path)
            if (candidate / ".sfm-workspace.json").exists():
                workspace_dir = candidate

    if workspace_dir is None:
        workspace_dir = find_workspace_for_path(matches_dir)

    if workspace_dir is None:
        raise RuntimeError(
            f"Cannot resolve workspace for {matches_file}. "
            "Ensure the workspace exists and contains .sfm-workspace.json."
        )

    print(f"Workspace: {workspace_dir}")
    print(
        f"Images: {image_count}, Pairs: {metadata['image_pair_count']}, "
        f"Matches: {metadata['match_count']}"
    )

    # Resolve image paths and .sift paths
    feature_prefix_dir = ws_meta.get("contents", {}).get("feature_prefix_dir", "")
    image_paths = []
    sift_paths = []
    for name in image_names:
        img_path = workspace_dir / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_paths.append(img_path)

        img_parent = Path(name).parent
        img_basename = Path(name).name
        sift_rel = img_parent / feature_prefix_dir / f"{img_basename}.sift"
        sift_path = workspace_dir / sift_rel
        if not sift_path.exists():
            raise FileNotFoundError(f"SIFT file not found: {sift_path}")
        sift_paths.append(sift_path)

    # Set up COLMAP directory and database
    colmap_dir.mkdir(exist_ok=True, parents=True)
    db_path = colmap_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    image_dir = workspace_dir

    # Populate DB with features
    _setup_db_single_camera(
        image_paths,
        sift_paths,
        image_dir,
        db_path,
        max_feature_count=None,
        camera_model=camera_model,
    )

    # Write matches and TVGs to the database
    _write_matches_to_db(db_path, matches_data, image_names, image_dir)

    return db_path, image_dir, image_paths


def _write_matches_to_db(
    db_path: Path,
    matches_data: dict,
    image_names: list[str],
    image_dir: Path,
) -> None:
    """Write matches and TVGs from a read_matches dict into a COLMAP database."""
    import numpy as np

    with pycolmap.Database.open(db_path) as db:
        db_images = db.read_all_images()
    name_to_db_id = {img.name: img.image_id for img in db_images}

    db_ids = []
    for name in image_names:
        if name not in name_to_db_id:
            raise RuntimeError(f"Image '{name}' not found in COLMAP database")
        db_ids.append(name_to_db_id[name])

    image_index_pairs = matches_data["image_index_pairs"]
    match_counts = matches_data["match_counts"]
    match_feature_indexes = matches_data["match_feature_indexes"]
    pair_count = len(image_index_pairs)

    with pycolmap.Database.open(db_path) as db:
        match_offset = 0
        for k in range(pair_count):
            idx_i = int(image_index_pairs[k, 0])
            idx_j = int(image_index_pairs[k, 1])
            count = int(match_counts[k])

            db_id_i = db_ids[idx_i]
            db_id_j = db_ids[idx_j]

            matches_slice = match_feature_indexes[match_offset : match_offset + count]
            db.write_matches(db_id_i, db_id_j, matches_slice)
            match_offset += count

        if matches_data.get("has_two_view_geometries", False):
            config_types = matches_data["config_types"]
            config_indexes = matches_data["config_indexes"]
            inlier_counts = matches_data["inlier_counts"]
            inlier_feature_indexes = matches_data["inlier_feature_indexes"]

            CONFIG_STR_TO_INT = {
                "undefined": 0,
                "degenerate": 1,
                "calibrated": 2,
                "uncalibrated": 3,
                "planar": 4,
                "planar_or_panoramic": 5,
                "panoramic": 6,
                "multiple": 7,
                "watermark_clean": 8,
                "watermark_bad": 9,
            }

            inlier_offset = 0
            for k in range(pair_count):
                idx_i = int(image_index_pairs[k, 0])
                idx_j = int(image_index_pairs[k, 1])
                db_id_i = db_ids[idx_i]
                db_id_j = db_ids[idx_j]

                ic = int(inlier_counts[k])
                config_str = config_types[int(config_indexes[k])]
                config_int = CONFIG_STR_TO_INT.get(config_str, 0)

                inlier_slice = inlier_feature_indexes[
                    inlier_offset : inlier_offset + ic
                ]

                tvg = pycolmap.TwoViewGeometry()
                tvg.config = config_int
                tvg.inlier_matches = inlier_slice

                f_mat = matches_data["f_matrices"][k]
                e_mat = matches_data["e_matrices"][k]
                h_mat = matches_data["h_matrices"][k]
                if np.any(f_mat != 0):
                    tvg.F = f_mat
                if np.any(e_mat != 0):
                    tvg.E = e_mat
                if np.any(h_mat != 0):
                    tvg.H = h_mat

                q = matches_data["quaternions_wxyz"][k]
                t = matches_data["translations_xyz"][k]
                is_identity_q = (
                    abs(q[0] - 1.0) < 1e-15
                    and abs(q[1]) < 1e-15
                    and abs(q[2]) < 1e-15
                    and abs(q[3]) < 1e-15
                )
                is_zero_t = all(abs(v) < 1e-15 for v in t)
                if not (is_identity_q and is_zero_t):
                    quat_xyzw = [q[1], q[2], q[3], q[0]]
                    pose = pycolmap.Rigid3d(
                        rotation=pycolmap.Rotation3d(quat_xyzw),
                        translation=t,
                    )
                    tvg.cam2_from_cam1 = pose

                db.write_two_view_geometry(db_id_i, db_id_j, tvg)
                inlier_offset += ic

    print(f"Wrote {pair_count} match pairs to database")
    if matches_data.get("has_two_view_geometries", False):
        total_inliers = int(matches_data["tvg_metadata"]["inlier_count"])
        print(f"Wrote {pair_count} two-view geometries ({total_inliers} total inliers)")


def _setup_db_single_camera(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    camera_model: str | None = None,
) -> None:
    """Set up COLMAP database with a single-camera trivial rig."""
    with pycolmap.Database.open(db_path) as db:
        cam = _infer_camera(image_paths[0], camera_model)
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
                descriptors = reader.read_descriptors(count=max_feature_count)
                db.write_keypoints(image_id, keypoints)
                db.write_descriptors(image_id, _wrap_descriptors(descriptors))


def _setup_db_with_rigs(
    image_paths: list[str | Path],
    sift_paths: list[Path],
    image_dir: Path,
    db_path: Path,
    max_feature_count: int | None,
    rig_configs: list[dict],
    camera_model: str | None = None,
) -> None:
    """Set up COLMAP database with multi-sensor rigs from rig_config.json."""
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
                effective_model = camera_model
                if effective_model is None:
                    intrinsics = rig_configs[rig_idx].get("camera_intrinsics")
                    if intrinsics is not None:
                        effective_model = intrinsics.get("model")
                cam = _infer_camera(str(image_path), effective_model)
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
                    descriptors = reader.read_descriptors(count=max_feature_count)
                    db.write_keypoints(image_id, keypoints)
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
            cam = pycolmap.infer_camera_from_image(
                str(image_dir / unmatched_images[0][0])
            )
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
                    descriptors = reader.read_descriptors(count=max_feature_count)
                    db.write_keypoints(image_id, keypoints)
                    db.write_descriptors(image_id, _wrap_descriptors(descriptors))
