# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP binary format import/export and conversion to SfmrReconstruction."""

from pathlib import Path

import numpy as np

from ._cameras import pycolmap_camera_to_intrinsics
from ._sift_file import SiftReader, get_sift_path_for_image
from ._workspace import find_workspace_for_path, load_workspace_config


def _get_feature_prefix_dir(workspace_config: dict) -> str:
    """Get feature_prefix_dir from workspace config."""
    prefix = workspace_config.get("feature_prefix_dir", "")
    if prefix:
        return prefix
    raise RuntimeError(
        "Workspace config missing 'feature_prefix_dir'. "
        "Re-initialize the workspace with 'sfm init'."
    )


def _resolve_workspace_and_sift(
    image_names: list[str],
    image_dir: Path,
) -> tuple[Path, dict, list[str], list[bytes], list[bytes], list[np.ndarray]]:
    """Resolve workspace and read .sift metadata for a list of image names.

    Returns:
        (workspace_dir, workspace_contents, resolved_names,
         feature_tool_hashes, sift_content_hashes, thumbnails)
    """
    workspace_dir = find_workspace_for_path(image_dir)
    if workspace_dir is None:
        raise RuntimeError(
            f"Could not find workspace for image directory: {image_dir}\n"
            "Initialize a workspace with 'sfm init <workspace_dir>'"
        )

    workspace_config = load_workspace_config(workspace_dir)
    feature_tool = workspace_config.get("feature_tool", "colmap")
    feature_options = workspace_config.get("feature_options", {})
    feature_prefix_dir = _get_feature_prefix_dir(workspace_config)

    workspace_contents = {
        "feature_tool": feature_tool,
        "feature_type": "sift",
        "feature_options": feature_options,
        "feature_prefix_dir": feature_prefix_dir,
    }

    resolved_names = []
    feature_tool_hashes = []
    sift_content_hashes = []
    thumbnails = []

    for name in image_names:
        if Path(name).is_absolute():
            img_abs_path = Path(name)
        elif (image_dir / name).exists():
            img_abs_path = image_dir / name
        else:
            img_abs_path = workspace_dir / name
        rel_path = img_abs_path.relative_to(workspace_dir)
        posix_path = rel_path.as_posix()

        if posix_path.startswith("/"):
            raise ValueError(
                f"Image path must be relative to workspace. "
                f"Got absolute POSIX path: {posix_path}\n"
                f"Image: {name}\nWorkspace: {workspace_dir}"
            )
        if posix_path.startswith("../") or "/../" in posix_path:
            raise ValueError(
                f"Image path cannot escape workspace directory. "
                f"Got path with '..': {posix_path}\n"
                f"Image: {name}\nWorkspace: {workspace_dir}"
            )

        resolved_names.append(posix_path)

        with SiftReader.for_image(
            img_abs_path,
            feature_tool=feature_tool,
            feature_options=feature_options,
        ) as reader:
            ft_hash = reader.content_hash["feature_tool_xxh128"]
            sc_hash = reader.content_hash["content_xxh128"]
            feature_tool_hashes.append(bytes.fromhex(ft_hash))
            sift_content_hashes.append(bytes.fromhex(sc_hash))
            thumbnails.append(reader.read_thumbnail())

    return (
        workspace_dir,
        workspace_contents,
        resolved_names,
        feature_tool_hashes,
        sift_content_hashes,
        thumbnails,
    )


def _build_sfmr_data_dict(
    *,
    cameras: list,
    image_names: list[str],
    camera_indexes: np.ndarray,
    quaternions_wxyz: np.ndarray,
    translations_xyz: np.ndarray,
    positions_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    reprojection_errors: np.ndarray,
    track_image_indexes: np.ndarray,
    track_feature_indexes: np.ndarray,
    track_point3d_indexes: np.ndarray,
    observation_counts: np.ndarray,
    feature_tool_hashes: list[bytes],
    sift_content_hashes: list[bytes],
    thumbnails: list[np.ndarray],
    metadata: dict,
    rig_frame_data: dict | None = None,
) -> dict:
    """Assemble a data dict for SfmrReconstruction.from_data."""
    data = {
        "metadata": metadata,
        "cameras": cameras,
        "image_names": image_names,
        "camera_indexes": np.asarray(camera_indexes, dtype=np.uint32),
        "quaternions_wxyz": np.asarray(quaternions_wxyz, dtype=np.float64),
        "translations_xyz": np.asarray(translations_xyz, dtype=np.float64),
        "positions_xyz": np.asarray(positions_xyz, dtype=np.float64),
        "colors_rgb": np.asarray(colors_rgb, dtype=np.uint8),
        "reprojection_errors": np.asarray(reprojection_errors, dtype=np.float32),
        "image_indexes": np.asarray(track_image_indexes, dtype=np.uint32),
        "feature_indexes": np.asarray(track_feature_indexes, dtype=np.uint32),
        "points3d_indexes": np.asarray(track_point3d_indexes, dtype=np.uint32),
        "observation_counts": np.asarray(observation_counts, dtype=np.uint32),
        "feature_tool_hashes": feature_tool_hashes,
        "sift_content_hashes": sift_content_hashes,
        "thumbnails_y_x_rgb": np.stack(thumbnails).astype(np.uint8),
    }
    if rig_frame_data is not None:
        data["rig_frame_data"] = rig_frame_data
    return data


def build_metadata(
    *,
    workspace_dir: Path,
    output_path: Path,
    workspace_config: dict,
    operation: str,
    tool_name: str,
    tool_options: dict | None = None,
    inputs: dict | None = None,
    image_count: int,
    points3d_count: int,
    observation_count: int,
    camera_count: int,
    world_space_unit: str | None = None,
) -> dict:
    """Build the metadata dict for a .sfmr file."""
    from datetime import datetime

    import pycolmap

    if tool_name == "colmap":
        tool_version = pycolmap.__version__
    elif tool_name == "glomap":
        tool_version = "unknown"
    elif tool_name == "sfmtool":
        try:
            from importlib.metadata import version as get_version

            tool_version = get_version("sfmtool")
        except Exception:
            tool_version = "unknown"
    else:
        tool_version = "unknown"

    workspace_relative_str = workspace_dir.relative_to(
        output_path.parent, walk_up=True
    ).as_posix()

    workspace_contents = {
        "feature_tool": workspace_config.get("feature_tool", "colmap"),
        "feature_type": "sift",
        "feature_options": workspace_config.get("feature_options", {}),
        "feature_prefix_dir": _get_feature_prefix_dir(workspace_config),
    }

    metadata = {
        "version": 1,
        "operation": operation,
        "tool": tool_name,
        "tool_version": tool_version,
        "workspace": {
            "absolute_path": str(workspace_dir),
            "relative_path": workspace_relative_str,
            "contents": workspace_contents,
        },
        "timestamp": datetime.now().astimezone().isoformat(),
        "image_count": image_count,
        "points3d_count": points3d_count,
        "observation_count": observation_count,
        "camera_count": camera_count,
        "tool_options": tool_options if tool_options is not None else {},
    }

    if inputs is not None:
        metadata["inputs"] = inputs
    if world_space_unit is not None:
        metadata["world_space_unit"] = world_space_unit

    return metadata


def colmap_binary_to_rust_sfmr(
    colmap_dir: str | Path,
    image_dir: str | Path,
    metadata: dict,
):
    """Load a COLMAP binary reconstruction as a Rust SfmrReconstruction."""
    from ._sfmtool import SfmrReconstruction, read_colmap_binary

    colmap_dir = Path(colmap_dir)
    image_dir = Path(image_dir).absolute()

    data = read_colmap_binary(str(colmap_dir))
    if len(data["positions_xyz"]) == 0:
        raise RuntimeError("No 3D points found in reconstruction.")

    (
        workspace_dir,
        _workspace_contents,
        resolved_names,
        feature_tool_hashes,
        sift_content_hashes,
        thumbnails,
    ) = _resolve_workspace_and_sift(data["image_names"], image_dir)

    rig_frame_data = data.get("rig_frame_data")

    sfmr_dict = _build_sfmr_data_dict(
        cameras=list(data["cameras"]),
        image_names=resolved_names,
        camera_indexes=data["camera_indexes"],
        quaternions_wxyz=data["quaternions_wxyz"],
        translations_xyz=data["translations_xyz"],
        positions_xyz=data["positions_xyz"],
        colors_rgb=data["colors_rgb"],
        reprojection_errors=data["reprojection_errors"],
        track_image_indexes=data["track_image_indexes"],
        track_feature_indexes=data["track_feature_indexes"],
        track_point3d_indexes=data["track_point3d_indexes"],
        observation_counts=data["observation_counts"],
        feature_tool_hashes=feature_tool_hashes,
        sift_content_hashes=sift_content_hashes,
        thumbnails=thumbnails,
        metadata=metadata,
        rig_frame_data=rig_frame_data,
    )

    return SfmrReconstruction.from_data(str(workspace_dir), sfmr_dict)


def pycolmap_to_rust_sfmr(
    reconstruction,
    image_dir: str | Path,
    metadata: dict,
):
    """Convert a pycolmap.Reconstruction to a Rust SfmrReconstruction."""
    from ._sfmtool import SfmrReconstruction

    image_dir = Path(image_dir).absolute()

    cameras = reconstruction.cameras
    images = reconstruction.images
    points3d = reconstruction.points3D

    if len(points3d) == 0:
        raise RuntimeError("No 3D points found in reconstruction.")

    # Prepare cameras
    sorted_camera_ids = sorted(cameras.keys())
    cameras_list = [
        pycolmap_camera_to_intrinsics(cameras[cam_id]) for cam_id in sorted_camera_ids
    ]
    camera_id_to_index = {cam_id: idx for idx, cam_id in enumerate(sorted_camera_ids)}

    # Prepare images (sorted by name)
    sorted_image_ids = sorted(images.keys(), key=lambda id: images[id].name)
    image_id_to_index = {img_id: idx for idx, img_id in enumerate(sorted_image_ids)}

    raw_image_names = []
    camera_indexes = []
    quaternions_wxyz = []
    translations_xyz = []

    for img_id in sorted_image_ids:
        image = images[img_id]
        raw_image_names.append(image.name)
        camera_indexes.append(camera_id_to_index[image.camera_id])

        if image.has_pose:
            rigid3d = image.cam_from_world()
            xyzw = rigid3d.rotation.quat
            wxyz = xyzw[[3, 0, 1, 2]]
            quaternions_wxyz.append(wxyz)
            translations_xyz.append(rigid3d.translation)
        else:
            raise RuntimeError(
                f"Image {image.name} has no pose in the COLMAP reconstruction."
            )

    (
        workspace_dir,
        _workspace_contents,
        resolved_names,
        feature_tool_hashes,
        sift_content_hashes,
        thumbnails,
    ) = _resolve_workspace_and_sift(raw_image_names, image_dir)

    # Prepare 3D points
    sorted_point_ids = sorted(points3d.keys())
    point_id_to_index = {pid: idx for idx, pid in enumerate(sorted_point_ids)}

    point_positions = []
    point_colors = []
    point_errors = []
    for pid in sorted_point_ids:
        pt = points3d[pid]
        point_positions.append(pt.xyz)
        point_colors.append(pt.color)
        point_errors.append(pt.error)

    # Prepare tracks
    image_indexes_list = []
    feature_indexes_list = []
    points3d_indexes_list = []
    observation_counts = [0] * len(sorted_point_ids)

    for pid in sorted_point_ids:
        pt = points3d[pid]
        pt_index = point_id_to_index[pid]
        observation_counts[pt_index] = len(pt.track.elements)

        for te in pt.track.elements:
            image_indexes_list.append(image_id_to_index[te.image_id])
            feature_indexes_list.append(te.point2D_idx)
            points3d_indexes_list.append(pt_index)

    # Extract rig/frame data
    rig_frame_data = _extract_rig_frame_data(
        reconstruction, camera_id_to_index, image_id_to_index
    )

    sfmr_dict = _build_sfmr_data_dict(
        cameras=cameras_list,
        image_names=resolved_names,
        camera_indexes=np.array(camera_indexes, dtype=np.uint32),
        quaternions_wxyz=np.array(quaternions_wxyz, dtype=np.float64),
        translations_xyz=np.array(translations_xyz, dtype=np.float64),
        positions_xyz=np.array(point_positions, dtype=np.float64),
        colors_rgb=np.array(point_colors, dtype=np.uint8),
        reprojection_errors=np.array(point_errors, dtype=np.float32),
        track_image_indexes=np.array(image_indexes_list, dtype=np.uint32),
        track_feature_indexes=np.array(feature_indexes_list, dtype=np.uint32),
        track_point3d_indexes=np.array(points3d_indexes_list, dtype=np.uint32),
        observation_counts=np.array(observation_counts, dtype=np.uint32),
        feature_tool_hashes=feature_tool_hashes,
        sift_content_hashes=sift_content_hashes,
        thumbnails=thumbnails,
        metadata=metadata,
        rig_frame_data=rig_frame_data,
    )

    return SfmrReconstruction.from_data(str(workspace_dir), sfmr_dict)


def _extract_rig_frame_data(
    reconstruction,
    camera_id_to_index: dict[int, int],
    image_id_to_index: dict[int, int],
) -> dict | None:
    """Extract rig and frame data from a pycolmap.Reconstruction.

    Returns rig_frame_data dict or None if only trivial single-camera rigs.
    """
    rigs = reconstruction.rigs
    frames = reconstruction.frames

    if len(rigs) == 0 or len(frames) == 0:
        return None

    all_trivial = all(len(rig.non_ref_sensors) == 0 for rig in rigs.values())
    if all_trivial:
        return None

    num_images = len(image_id_to_index)

    sorted_rig_ids = sorted(rigs.keys())
    rig_id_to_index = {rig_id: idx for idx, rig_id in enumerate(sorted_rig_ids)}

    rig_defs = []
    all_sensor_camera_indexes = []
    all_sensor_quaternions_wxyz = []
    all_sensor_translations_xyz = []
    global_sensor_offset = 0

    sensor_key_to_global_index: dict[tuple[int, tuple[int, int]], int] = {}

    for rig_id in sorted_rig_ids:
        rig = rigs[rig_id]
        ref_sensor = rig.ref_sensor_id
        non_ref_sensors = rig.non_ref_sensors

        sensors = [ref_sensor]
        sorted_non_ref = sorted(non_ref_sensors.keys(), key=lambda s: s.id)
        sensors.extend(sorted_non_ref)

        sensor_names = []
        for i, sensor in enumerate(sensors):
            sensor_key = (rig_id, (sensor.type.value, sensor.id))
            sensor_key_to_global_index[sensor_key] = global_sensor_offset + i

            if sensor.id not in camera_id_to_index:
                raise ValueError(
                    f"Rig {rig_id} sensor {sensor.id} references unknown camera_id"
                )
            cam_idx = camera_id_to_index[sensor.id]
            all_sensor_camera_indexes.append(cam_idx)

            if sensor == ref_sensor:
                all_sensor_quaternions_wxyz.append([1.0, 0.0, 0.0, 0.0])
                all_sensor_translations_xyz.append([0.0, 0.0, 0.0])
            else:
                pose = non_ref_sensors[sensor]
                xyzw = pose.rotation.quat
                wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
                all_sensor_quaternions_wxyz.append(wxyz)
                all_sensor_translations_xyz.append(list(pose.translation))

            sensor_names.append(f"sensor{i}")

        rig_defs.append(
            {
                "name": f"rig{rig_id_to_index[rig_id]}",
                "sensor_count": len(sensors),
                "sensor_offset": global_sensor_offset,
                "ref_sensor_name": "sensor0",
                "sensor_names": sensor_names,
            }
        )
        global_sensor_offset += len(sensors)

    total_sensors = global_sensor_offset

    sorted_frame_ids = sorted(frames.keys())
    num_frames = len(sorted_frame_ids)

    rig_indexes = np.zeros(num_frames, dtype=np.uint32)
    image_sensor_indexes = np.zeros(num_images, dtype=np.uint32)
    image_frame_indexes = np.zeros(num_images, dtype=np.uint32)

    for frame_idx, frame_id in enumerate(sorted_frame_ids):
        frame = frames[frame_id]
        rig_id = frame.rig_id
        rig_indexes[frame_idx] = rig_id_to_index[rig_id]

        for data_id in frame.image_ids:
            colmap_image_id = data_id.id
            if colmap_image_id not in image_id_to_index:
                continue

            img_idx = image_id_to_index[colmap_image_id]
            sensor = data_id.sensor_id
            sensor_key = (rig_id, (sensor.type.value, sensor.id))
            if sensor_key not in sensor_key_to_global_index:
                raise ValueError(
                    f"Frame {frame_id} references unknown sensor "
                    f"(rig_id={rig_id}, type={sensor.type.value}, id={sensor.id})"
                )
            global_sensor_idx = sensor_key_to_global_index[sensor_key]

            image_sensor_indexes[img_idx] = global_sensor_idx
            image_frame_indexes[img_idx] = frame_idx

    return {
        "rigs_metadata": {
            "rig_count": len(sorted_rig_ids),
            "sensor_count": total_sensors,
            "rigs": rig_defs,
        },
        "sensor_camera_indexes": np.array(all_sensor_camera_indexes, dtype=np.uint32),
        "sensor_quaternions_wxyz": np.array(
            all_sensor_quaternions_wxyz, dtype=np.float64
        ),
        "sensor_translations_xyz": np.array(
            all_sensor_translations_xyz, dtype=np.float64
        ),
        "frames_metadata": {"frame_count": num_frames},
        "rig_indexes": rig_indexes,
        "image_sensor_indexes": image_sensor_indexes,
        "image_frame_indexes": image_frame_indexes,
    }


def save_colmap_binary(recon, output_dir: Path, max_features: int | None = None):
    """Export a SfmrReconstruction to COLMAP binary format.

    Creates cameras.bin, images.bin, points3D.bin (and optionally rigs.bin,
    frames.bin) in the output directory.

    Args:
        recon: SfmrReconstruction object to export
        output_dir: Directory to write COLMAP binary files
        max_features: Maximum features to export per image (None = all features)
    """
    from ._sfmtool import write_colmap_binary

    output_dir = Path(output_dir)
    workspace_dir = recon.workspace_dir

    # Read keypoints from .sift files for each image
    keypoints_per_image = []
    for img_name in recon.image_names:
        image_path = Path(workspace_dir) / img_name
        sift_path = get_sift_path_for_image(image_path)
        with SiftReader(sift_path) as reader:
            positions = reader.read_positions(count=max_features)
        keypoints_per_image.append(np.asarray(positions, dtype=np.float64))

    # Get track arrays
    track_image_indexes = recon.track_image_indexes
    track_feature_indexes = recon.track_feature_indexes
    track_point3d_indexes = recon.track_point_ids
    positions_xyz = recon.positions
    colors_rgb = recon.colors
    reprojection_errors = recon.errors

    if max_features is not None:
        keypoint_counts = np.array(
            [len(kps) for kps in keypoints_per_image], dtype=np.uint32
        )
        per_obs_limit = keypoint_counts[track_image_indexes]
        obs_mask = track_feature_indexes < per_obs_limit
        track_image_indexes = track_image_indexes[obs_mask]
        track_feature_indexes = track_feature_indexes[obs_mask]
        track_point3d_indexes = track_point3d_indexes[obs_mask]

        point_obs_counts = np.bincount(
            track_point3d_indexes, minlength=len(positions_xyz)
        )
        surviving_point_ids = np.where(point_obs_counts >= 2)[0]
        obs_point_mask = point_obs_counts[track_point3d_indexes] >= 2
        track_image_indexes = track_image_indexes[obs_point_mask]
        track_feature_indexes = track_feature_indexes[obs_point_mask]
        track_point3d_indexes = track_point3d_indexes[obs_point_mask]
        if len(surviving_point_ids) < len(positions_xyz):
            old_to_new = np.full(len(positions_xyz), -1, dtype=np.int64)
            old_to_new[surviving_point_ids] = np.arange(len(surviving_point_ids))
            positions_xyz = positions_xyz[surviving_point_ids]
            colors_rgb = colors_rgb[surviving_point_ids]
            reprojection_errors = reprojection_errors[surviving_point_ids]
            track_point3d_indexes = old_to_new[track_point3d_indexes].astype(np.uint32)

    data = {
        "cameras": recon.cameras,
        "image_names": recon.image_names,
        "camera_indexes": recon.camera_indexes,
        "quaternions_wxyz": recon.quaternions_wxyz,
        "translations_xyz": recon.translations,
        "positions_xyz": positions_xyz,
        "colors_rgb": colors_rgb,
        "reprojection_errors": reprojection_errors,
        "track_image_indexes": track_image_indexes,
        "track_feature_indexes": track_feature_indexes,
        "track_point3d_indexes": track_point3d_indexes,
        "keypoints_per_image": keypoints_per_image,
    }

    if recon.rig_frame_data is not None:
        data["rig_frame_data"] = recon.rig_frame_data

    num_cameras = len(recon.cameras)
    num_images = len(recon.image_names)
    num_points = len(positions_xyz)

    print(
        f"Writing {num_cameras} cameras, {num_images} images, {num_points} 3D points..."
    )
    write_colmap_binary(str(output_dir), data)

    print("\nExport complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - cameras.bin: {num_cameras} cameras")
    print(f"  - images.bin: {num_images} images")
    print(f"  - points3D.bin: {num_points} points")
    if recon.rig_frame_data is not None:
        num_rigs = recon.rig_frame_data["rigs_metadata"]["rig_count"]
        num_frames = recon.rig_frame_data["frames_metadata"]["frame_count"]
        print(f"  - rigs.bin: {num_rigs} rigs")
        print(f"  - frames.bin: {num_frames} frames")
