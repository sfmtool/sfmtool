# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Image pair graph utilities for computing covisibility and frustum intersection pairs."""

import numpy as np

from ._sfmtool import (
    RotQuaternion,
    SfmrReconstruction,
    build_covisibility_pairs_py as _rust_build_covisibility_pairs,
    build_frustum_intersection_pairs_py as _rust_build_frustum_intersection_pairs,
)
from ._histogram_utils import estimate_z_from_histogram  # noqa: F401


def compute_camera_directions(quaternions: np.ndarray) -> np.ndarray:
    """Compute camera viewing directions from quaternions.

    Args:
        quaternions: (N, 4) array of quaternions in WXYZ format

    Returns:
        (N, 3) array of normalized camera viewing directions in world space
    """
    num_images = len(quaternions)
    directions = np.zeros((num_images, 3), dtype=np.float64)

    for img_idx in range(num_images):
        quat = RotQuaternion.from_wxyz_array(quaternions[img_idx])
        R_cam_from_world = quat.to_rotation_matrix()
        R_world_from_cam = R_cam_from_world.T
        direction = -R_world_from_cam[:, 2]
        directions[img_idx] = direction / np.linalg.norm(direction)

    return directions


def build_covisibility_pairs(
    recon: SfmrReconstruction, angle_threshold_deg: float = 90.0
) -> list[tuple[int, int, int]]:
    """Build covisibility pairs from a reconstruction."""
    result = _rust_build_covisibility_pairs(
        recon.quaternions_wxyz,
        recon.track_point_ids.astype(np.uint32),
        recon.track_image_indexes.astype(np.uint32),
        angle_threshold_deg,
    )
    return [(int(i), int(j), int(c)) for i, j, c in result]


def build_frustum_intersection_pairs(
    recon: SfmrReconstruction,
    near_percentile: float = 5.0,
    far_percentile: float = 95.0,
    num_samples: int = 100,
    angle_threshold_deg: float = 90.0,
) -> list[tuple[int, int, float]]:
    """Build frustum intersection pairs from a reconstruction."""
    depth_stats = recon.depth_statistics
    if not _has_valid_depth_statistics(depth_stats):
        raise ValueError(
            "No depth statistics found in reconstruction. "
            "Depth statistics are required for frustum intersection analysis."
        )

    num_images = recon.image_count
    cameras_meta = recon.cameras
    camera_indexes = recon.camera_indexes
    hist_counts = recon.depth_histogram_counts
    images_data = depth_stats["images"]

    fx_arr = np.zeros(num_images, dtype=np.float64)
    fy_arr = np.zeros(num_images, dtype=np.float64)
    cx_arr = np.zeros(num_images, dtype=np.float64)
    cy_arr = np.zeros(num_images, dtype=np.float64)
    widths_arr = np.zeros(num_images, dtype=np.uint32)
    heights_arr = np.zeros(num_images, dtype=np.uint32)
    hist_min_z = np.full(num_images, np.nan, dtype=np.float64)
    hist_max_z = np.full(num_images, np.nan, dtype=np.float64)

    for img_idx in range(num_images):
        cam_idx = camera_indexes[img_idx]
        camera = cameras_meta[cam_idx]
        K = camera.intrinsic_matrix()
        fx_arr[img_idx] = K[0, 0]
        fy_arr[img_idx] = K[1, 1]
        cx_arr[img_idx] = K[0, 2]
        cy_arr[img_idx] = K[1, 2]
        widths_arr[img_idx] = camera.width
        heights_arr[img_idx] = camera.height

        img_data = images_data[img_idx]
        if img_data["histogram_min_z"] is not None:
            hist_min_z[img_idx] = img_data["histogram_min_z"]
            hist_max_z[img_idx] = img_data["histogram_max_z"]

    hist_counts_u32 = np.asarray(hist_counts, dtype=np.uint32)

    result = _rust_build_frustum_intersection_pairs(
        recon.quaternions_wxyz,
        recon.translations,
        fx_arr,
        fy_arr,
        cx_arr,
        cy_arr,
        widths_arr,
        heights_arr,
        hist_counts_u32,
        hist_min_z,
        hist_max_z,
        near_percentile,
        far_percentile,
        num_samples,
        angle_threshold_deg,
        42,  # seed for reproducibility
    )
    return [(int(i), int(j), float(v)) for i, j, v in result]


def _has_valid_depth_statistics(depth_stats: dict) -> bool:
    """Check if depth statistics contain valid per-image data."""
    images = depth_stats.get("images", [])
    return any(img.get("histogram_min_z") is not None for img in images)
