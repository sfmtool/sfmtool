# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge multiple aligned SfM reconstructions into a single reconstruction."""

import time
from pathlib import Path

import click
import numpy as np

from ._merge_correspondences import find_point_correspondences, merge_points_and_tracks
from ._merge_pose_refinement import refine_camera_poses
from ._sfmtool import SfmrReconstruction


def merge_reconstructions(
    reconstructions: list[SfmrReconstruction],
    merge_percentile: float = 95.0,
) -> SfmrReconstruction:
    """Merge multiple aligned reconstructions into one.

    Args:
        reconstructions: List of SfmrReconstruction objects to merge (must be aligned)
        merge_percentile: Percentile of correspondence distances to use as threshold

    Returns:
        Merged SfmrReconstruction object
    """
    if len(reconstructions) < 2:
        raise ValueError("Need at least 2 reconstructions to merge")

    click.echo(f"\nMerging {len(reconstructions)} reconstructions...")
    click.echo("=" * 70)

    # [1/7] Build image-to-camera mapping
    click.echo("\n[1/7] Building image-to-camera mapping...")
    step_start = time.perf_counter()
    image_name_to_camera = {}
    for img_idx, img_name in enumerate(reconstructions[0].image_names):
        img_key = Path(img_name).as_posix()
        cam_idx = reconstructions[0].camera_indexes[img_idx]
        camera = reconstructions[0].cameras[cam_idx]
        image_name_to_camera[img_key] = camera
    unique_cameras = set(_camera_to_tuple(cam) for cam in image_name_to_camera.values())
    click.echo(f"  Reference cameras from first reconstruction: {len(unique_cameras)}")
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    # [2/7] Merge cameras
    click.echo("\n[2/7] Merging cameras...")
    step_start = time.perf_counter()
    merged_cameras, camera_mapping = _merge_cameras(
        reconstructions, image_name_to_camera
    )
    click.echo(f"  Merged cameras: {len(merged_cameras)}")
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    # [3/7] Merge images
    click.echo("\n[3/7] Merging images...")
    step_start = time.perf_counter()
    merged_images, image_mapping = _merge_images(reconstructions, camera_mapping)
    click.echo(f"  Total unique images: {len(merged_images['names'])}")
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    # [4/7] Find corresponding 3D points
    click.echo("\n[4/7] Finding corresponding 3D points...")
    step_start = time.perf_counter()
    point_correspondences = find_point_correspondences(
        reconstructions, image_mapping, merge_percentile
    )
    click.echo(f"  Found {len(point_correspondences)} point correspondence groups")
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    # [5/7] Merge 3D points and tracks
    click.echo("\n[5/7] Merging 3D points and tracks...")
    step_start = time.perf_counter()
    merged_points, merged_tracks = merge_points_and_tracks(
        reconstructions, point_correspondences, image_mapping
    )
    click.echo(f"  Merged 3D points: {len(merged_points['positions'])}")
    click.echo(f"  Total observations: {len(merged_tracks['image_indexes'])}")
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    # Get workspace info from first reconstruction
    workspace_dir = reconstructions[0].workspace_dir

    # [6/7] Refine camera poses
    click.echo("\n[6/7] Refining camera poses...")
    step_start = time.perf_counter()
    workspace_meta = reconstructions[0].source_metadata.get("workspace", {})
    workspace_contents = workspace_meta.get("contents", {})
    feature_tool = workspace_contents.get("feature_tool", "colmap")
    feature_options = workspace_contents.get("feature_options", {})

    merged_images = refine_camera_poses(
        merged_images=merged_images,
        merged_cameras=merged_cameras,
        merged_points=merged_points,
        merged_tracks=merged_tracks,
        workspace_dir=workspace_dir,
        feature_tool=feature_tool,
        feature_options=feature_options,
    )
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    # [7/7] Create merged reconstruction
    click.echo("\n[7/7] Creating merged reconstruction...")
    step_start = time.perf_counter()
    result = _create_merged_reconstruction(
        cameras=merged_cameras,
        images=merged_images,
        points=merged_points,
        tracks=merged_tracks,
        source_reconstructions=reconstructions,
    )
    click.echo(f"  Completed in {time.perf_counter() - step_start:.2f}s")

    click.echo("\n" + "=" * 70)
    click.echo("Merge complete!")

    return result


def _merge_cameras(
    reconstructions: list[SfmrReconstruction],
    image_name_to_camera: dict,
) -> tuple[list, list[dict[int, int]]]:
    """Merge cameras from all reconstructions using union-find."""
    union_find_data = {}

    def find(cam):
        if cam not in union_find_data:
            union_find_data[cam] = cam
        if union_find_data[cam] != cam:
            union_find_data[cam] = find(union_find_data[cam])
        return union_find_data[cam]

    def union(cam1, cam2):
        root1 = find(cam1)
        root2 = find(cam2)
        if root1 != root2:
            if root1[0] < root2[0]:
                union_find_data[root2] = root1
            else:
                union_find_data[root1] = root2

    # Build image -> [(recon_idx, cam_idx)] mapping
    image_to_cameras = {}
    for recon_idx, recon in enumerate(reconstructions):
        for img_idx, img_name in enumerate(recon.image_names):
            img_key = Path(img_name).as_posix()
            cam_idx = recon.camera_indexes[img_idx]

            if img_key not in image_to_cameras:
                image_to_cameras[img_key] = []
            image_to_cameras[img_key].append((recon_idx, cam_idx))

    # Union cameras that share images
    for img_name, camera_ids in image_to_cameras.items():
        if len(camera_ids) > 1:
            for i in range(1, len(camera_ids)):
                union(camera_ids[0], camera_ids[i])

    # Build merged camera list
    root_to_merged_idx = {}
    merged_cameras = []

    for recon_idx, recon in enumerate(reconstructions):
        for cam_idx in range(len(recon.cameras)):
            camera_id = (recon_idx, cam_idx)
            root = find(camera_id)

            if root not in root_to_merged_idx:
                root_recon_idx, root_cam_idx = root
                camera = reconstructions[root_recon_idx].cameras[root_cam_idx]
                root_to_merged_idx[root] = len(merged_cameras)
                merged_cameras.append(camera)

    # Build camera mapping for each reconstruction
    camera_mapping = []
    for recon_idx, recon in enumerate(reconstructions):
        mapping = {}
        for cam_idx in range(len(recon.cameras)):
            camera_id = (recon_idx, cam_idx)
            root = find(camera_id)
            mapping[cam_idx] = root_to_merged_idx[root]
        camera_mapping.append(mapping)

    return merged_cameras, camera_mapping


def _camera_to_tuple(cam) -> tuple:
    """Convert CameraIntrinsics to hashable tuple for exact deduplication."""
    params_tuple = tuple((k, float(v)) for k, v in sorted(cam.parameters.items()))
    return (cam.model, cam.width, cam.height, params_tuple)


def _merge_images(
    reconstructions: list[SfmrReconstruction],
    camera_mapping: list[dict[int, int]],
) -> tuple[dict, dict[str, list[tuple[int, int]]]]:
    """Merge images from all reconstructions.

    For overlapping images (same relative path), uses the first occurrence's pose.
    """
    merged_names = []
    merged_camera_indexes = []
    merged_quaternions = []
    merged_translations = []
    merged_feature_tool_hashes = []
    merged_sift_hashes = []
    merged_thumbnails = []

    # Rig frame data per-image arrays
    merged_image_sensor_indexes = []
    merged_image_frame_indexes = []
    # Frame deduplication: (recon_idx, old_frame_idx) -> merged_frame_idx
    frame_key_to_merged_idx = {}
    merged_rig_indexes = []

    has_rig_data = any(r.rig_frame_data is not None for r in reconstructions)

    image_mapping = {}
    name_to_merged_idx = {}

    for recon_idx, recon in enumerate(reconstructions):
        rfd = recon.rig_frame_data if has_rig_data else None
        for old_idx, img_name in enumerate(recon.image_names):
            img_key = Path(img_name).as_posix()

            if img_key in name_to_merged_idx:
                image_mapping[img_key].append((recon_idx, old_idx))
            else:
                merged_idx = len(merged_names)
                name_to_merged_idx[img_key] = merged_idx
                image_mapping[img_key] = [(recon_idx, old_idx)]

                old_cam_idx = recon.camera_indexes[old_idx]
                new_cam_idx = camera_mapping[recon_idx][old_cam_idx]

                merged_names.append(img_name)
                merged_camera_indexes.append(new_cam_idx)
                merged_quaternions.append(recon.quaternions_wxyz[old_idx])
                merged_translations.append(recon.translations[old_idx])
                merged_feature_tool_hashes.append(recon.feature_tool_hashes[old_idx])
                merged_sift_hashes.append(recon.sift_content_hashes[old_idx])
                merged_thumbnails.append(recon.thumbnails_y_x_rgb[old_idx])

                if rfd is not None:
                    merged_image_sensor_indexes.append(
                        rfd["image_sensor_indexes"][old_idx]
                    )
                    old_frame_idx = int(rfd["image_frame_indexes"][old_idx])
                    frame_key = (recon_idx, old_frame_idx)
                    if frame_key not in frame_key_to_merged_idx:
                        new_frame_idx = len(merged_rig_indexes)
                        frame_key_to_merged_idx[frame_key] = new_frame_idx
                        merged_rig_indexes.append(rfd["rig_indexes"][old_frame_idx])
                    merged_image_frame_indexes.append(
                        frame_key_to_merged_idx[frame_key]
                    )

    merged_images = {
        "names": merged_names,
        "camera_indexes": np.array(merged_camera_indexes, dtype=np.int32),
        "quaternions_wxyz": np.array(merged_quaternions),
        "translations": np.array(merged_translations),
        "feature_tool_hashes": merged_feature_tool_hashes,
        "sift_hashes": merged_sift_hashes,
        "thumbnails_y_x_rgb": np.array(merged_thumbnails, dtype=np.uint8),
    }

    if has_rig_data:
        merged_images["image_sensor_indexes"] = np.array(
            merged_image_sensor_indexes, dtype=np.uint32
        )
        merged_images["image_frame_indexes"] = np.array(
            merged_image_frame_indexes, dtype=np.uint32
        )
        merged_images["rig_indexes"] = np.array(merged_rig_indexes, dtype=np.uint32)

    return merged_images, image_mapping


def _create_merged_reconstruction(
    cameras,
    images,
    points,
    tracks,
    source_reconstructions,
) -> SfmrReconstruction:
    """Create a SfmrReconstruction from merged data."""
    first_recon = source_reconstructions[0]

    observation_counts = np.bincount(
        tracks["point_ids"], minlength=len(points["positions"])
    ).astype(np.uint32)

    # Build merged rig_frame_data if source reconstructions have it
    rig_frame_data = _merge_rig_frame_data(images, source_reconstructions)

    return first_recon.clone_with_changes(
        cameras=cameras,
        camera_indexes=np.ascontiguousarray(images["camera_indexes"], dtype=np.uint32),
        image_names=images["names"],
        quaternions_wxyz=np.ascontiguousarray(
            images["quaternions_wxyz"], dtype=np.float64
        ),
        translations=np.ascontiguousarray(images["translations"], dtype=np.float64),
        feature_tool_hashes=images["feature_tool_hashes"],
        sift_content_hashes=images["sift_hashes"],
        thumbnails_y_x_rgb=np.ascontiguousarray(
            images["thumbnails_y_x_rgb"], dtype=np.uint8
        ),
        positions=np.ascontiguousarray(points["positions"], dtype=np.float64),
        colors=np.ascontiguousarray(points["colors"], dtype=np.uint8),
        errors=np.ascontiguousarray(points["errors"], dtype=np.float32),
        track_image_indexes=np.ascontiguousarray(
            tracks["image_indexes"], dtype=np.uint32
        ),
        track_feature_indexes=np.ascontiguousarray(
            tracks["feature_indexes"], dtype=np.uint32
        ),
        track_point_ids=np.ascontiguousarray(tracks["point_ids"], dtype=np.uint32),
        observation_counts=observation_counts,
        rig_frame_data=rig_frame_data,
    )


def _merge_rig_frame_data(images, source_reconstructions):
    """Build merged rig_frame_data from merged image arrays, or None."""
    if "image_sensor_indexes" not in images:
        return None

    # Use rig/sensor definitions from the first reconstruction that has them
    ref_rfd = None
    for recon in source_reconstructions:
        if recon.rig_frame_data is not None:
            ref_rfd = recon.rig_frame_data
            break
    if ref_rfd is None:
        return None

    return {
        "rigs_metadata": ref_rfd["rigs_metadata"],
        "sensor_camera_indexes": ref_rfd["sensor_camera_indexes"],
        "sensor_quaternions_wxyz": ref_rfd["sensor_quaternions_wxyz"],
        "sensor_translations_xyz": ref_rfd["sensor_translations_xyz"],
        "frames_metadata": {"frame_count": len(images["rig_indexes"])},
        "rig_indexes": images["rig_indexes"],
        "image_sensor_indexes": images["image_sensor_indexes"],
        "image_frame_indexes": images["image_frame_indexes"],
    }
