# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Refine camera poses after merging reconstructions.

Uses pycolmap's absolute pose estimation (PnP + RANSAC) to re-estimate camera
poses against the merged 3D point cloud, running in parallel across all images.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np

from ._cameras import _CAMERA_PARAM_NAMES


def _camera_dict_to_pycolmap(cam_dict):
    """Convert a camera dict (model, width, height, parameters) to pycolmap.Camera."""
    import pycolmap

    model = cam_dict["model"]
    params = cam_dict["parameters"]
    param_names = _CAMERA_PARAM_NAMES.get(model)
    if param_names is None:
        raise ValueError(f"Unsupported camera model: {model}")

    param_list = [params[name] for name in param_names]

    return pycolmap.Camera(
        model=model,
        width=cam_dict["width"],
        height=cam_dict["height"],
        params=param_list,
    )


def _refine_single_camera_pose(
    img_idx,
    img_name,
    cam_dict,
    observations,
    merged_points_positions,
    workspace_dir,
    feature_tool,
    feature_options,
    original_quat,
    original_trans,
    max_num_trials=10000,
):
    """Refine a single camera pose using PnP + RANSAC.

    Args:
        cam_dict: Serializable camera dict with model/width/height/parameters keys.

    Returns:
        Tuple of (img_idx, refined_quat, refined_trans, status, io_time,
                  compute_time, num_correspondences, num_inliers)
    """
    import pycolmap

    from ._sift_file import SiftReader

    if len(observations) < 4:
        return img_idx, original_quat, original_trans, "no_observations", 0.0, 0.0, 0, 0

    try:
        io_start = time.perf_counter()
        img_path = Path(workspace_dir) / img_name
        sift_reader = SiftReader.for_image(
            img_path,
            feature_tool=feature_tool,
            feature_options=feature_options,
        )
        positions_2d = sift_reader.read_positions()
        sift_reader.close()
        io_time = time.perf_counter() - io_start

        points2d = []
        points3d = []
        for feat_idx, point_id in observations:
            if feat_idx < len(positions_2d):
                points2d.append(positions_2d[feat_idx])
                points3d.append(merged_points_positions[point_id])

        num_correspondences = len(points2d)

        if num_correspondences < 4:
            return (
                img_idx,
                original_quat,
                original_trans,
                "no_observations",
                io_time,
                0.0,
                num_correspondences,
                0,
            )

        points2d = np.array(points2d)
        points3d = np.array(points3d)

        camera = _camera_dict_to_pycolmap(cam_dict)

        compute_start = time.perf_counter()
        estimation_options = pycolmap.AbsolutePoseEstimationOptions()
        estimation_options.ransac.min_num_trials = 10
        estimation_options.ransac.max_num_trials = max_num_trials

        result = pycolmap.estimate_and_refine_absolute_pose(
            points2d, points3d, camera, estimation_options=estimation_options
        )
        compute_time = time.perf_counter() - compute_start

        if result is not None and "cam_from_world" in result:
            cam_from_world = result["cam_from_world"]
            num_inliers = result.get("num_inliers", 0)

            # pycolmap returns XYZW, we store WXYZ
            pycolmap_quat = cam_from_world.rotation.quat
            wxyz_quat = np.array(
                [
                    pycolmap_quat[3],
                    pycolmap_quat[0],
                    pycolmap_quat[1],
                    pycolmap_quat[2],
                ]
            )

            return (
                img_idx,
                wxyz_quat,
                cam_from_world.translation,
                "success",
                io_time,
                compute_time,
                num_correspondences,
                num_inliers,
            )
        else:
            return (
                img_idx,
                original_quat,
                original_trans,
                "failed",
                io_time,
                compute_time,
                num_correspondences,
                0,
            )

    except Exception:
        return img_idx, original_quat, original_trans, "failed", 0.0, 0.0, 0, 0


def refine_camera_poses(
    merged_images,
    merged_cameras,
    merged_points,
    merged_tracks,
    workspace_dir,
    feature_tool,
    feature_options,
    max_num_trials=5000,
):
    """Refine camera poses using merged 3D points and pycolmap's absolute pose estimation.

    Uses parallel processing to refine poses for all images concurrently.
    """
    click.echo("  Refining camera poses using merged 3D points...")

    # Pre-serialize CameraIntrinsics to dicts for pickling across processes
    camera_dicts = [cam.to_dict() for cam in merged_cameras]

    # Build image_idx -> [(feat_idx, point_id), ...]
    image_observations = {}
    for img_idx, feat_idx, point_id in zip(
        merged_tracks["image_indexes"],
        merged_tracks["feature_indexes"],
        merged_tracks["point_ids"],
    ):
        if img_idx not in image_observations:
            image_observations[img_idx] = []
        image_observations[img_idx].append((feat_idx, point_id))

    num_images = len(merged_images["names"])
    refined_quaternions = [None] * num_images
    refined_translations = [None] * num_images
    refinement_stats = {"success": 0, "failed": 0, "no_observations": 0}
    total_io_time = 0.0
    total_compute_time = 0.0
    total_correspondences = 0
    total_inliers = 0

    tasks = []
    for img_idx, img_name in enumerate(merged_images["names"]):
        cam_idx = merged_images["camera_indexes"][img_idx]
        cam_dict = camera_dicts[cam_idx]
        observations = image_observations.get(img_idx, [])

        tasks.append(
            (
                img_idx,
                img_name,
                cam_dict,
                observations,
                merged_points["positions"],
                workspace_dir,
                feature_tool,
                feature_options,
                merged_images["quaternions_wxyz"][img_idx],
                merged_images["translations"][img_idx],
                max_num_trials,
            )
        )

    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_refine_single_camera_pose, *task): task[0]
            for task in tasks
        }

        for future in as_completed(future_to_idx):
            (
                img_idx,
                refined_quat,
                refined_trans,
                status,
                io_time,
                compute_time,
                num_correspondences,
                num_inliers,
            ) = future.result()

            refined_quaternions[img_idx] = refined_quat
            refined_translations[img_idx] = refined_trans
            refinement_stats[status] += 1
            total_io_time += io_time
            total_compute_time += compute_time
            total_correspondences += num_correspondences
            total_inliers += num_inliers

    click.echo(
        f"    Refined: {refinement_stats['success']}, "
        f"Failed: {refinement_stats['failed']}, "
        f"No observations: {refinement_stats['no_observations']}"
    )

    avg_correspondences = (
        total_correspondences / refinement_stats["success"]
        if refinement_stats["success"] > 0
        else 0
    )
    avg_inliers = (
        total_inliers / refinement_stats["success"]
        if refinement_stats["success"] > 0
        else 0
    )
    inlier_ratio = (
        (total_inliers / total_correspondences * 100)
        if total_correspondences > 0
        else 0
    )

    click.echo("    Convergence statistics:")
    click.echo(f"      Avg correspondences per image: {avg_correspondences:.1f}")
    click.echo(f"      Avg inliers per image: {avg_inliers:.1f}")
    click.echo(f"      Overall inlier ratio: {inlier_ratio:.1f}%")
    click.echo("    Time breakdown:")
    click.echo(f"      I/O (reading SIFT files): {total_io_time:.2f}s")
    click.echo(f"      Compute (pose estimation): {total_compute_time:.2f}s")

    return {
        "names": merged_images["names"],
        "camera_indexes": merged_images["camera_indexes"],
        "quaternions_wxyz": np.array(refined_quaternions),
        "translations": np.array(refined_translations),
        "feature_tool_hashes": merged_images["feature_tool_hashes"],
        "sift_hashes": merged_images["sift_hashes"],
        "thumbnails_y_x_rgb": merged_images["thumbnails_y_x_rgb"],
    }
