# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""High-level feature matching interface for registered images.

This module provides the main API for matching features between image pairs
with known camera poses. It automatically selects between rectified and polar
sweep matching based on the camera geometry (epipole location).
"""

import numpy as np
import pycolmap

from ._geometric_filter import GeometricFilterConfig
from .._cameras import get_intrinsic_matrix
from .._sfmtool import (
    match_image_pair_py as _rust_match_image_pair,
    match_image_pairs_batch_py as _rust_match_image_pairs_batch,
)


def match_image_pair(
    img_i_cam_from_world: pycolmap.Rigid3d,
    img_j_cam_from_world: pycolmap.Rigid3d,
    cam_i: pycolmap.Camera,
    cam_j: pycolmap.Camera,
    positions_i: np.ndarray,
    descriptors_i: np.ndarray,
    positions_j: np.ndarray,
    descriptors_j: np.ndarray,
    window_size: int = 30,
    distance_threshold: float | None = None,
    rectification_margin: int = 50,
    affine_shapes_i: np.ndarray | None = None,
    affine_shapes_j: np.ndarray | None = None,
    geometric_config: GeometricFilterConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Match features between a single pair of registered images.

    Automatically selects between rectified sweep matching (for lateral camera
    motion) and polar sweep matching (for forward/backward motion) based on
    epipole location.

    Args:
        img_i_cam_from_world: Camera pose for image i (camera from world transform)
        img_j_cam_from_world: Camera pose for image j (camera from world transform)
        cam_i: Camera model for image i
        cam_j: Camera model for image j
        positions_i: Feature positions in image i (Nx2 array)
        descriptors_i: Feature descriptors for image i (Nx128 array)
        positions_j: Feature positions in image j (Mx2 array)
        descriptors_j: Feature descriptors for image j (Mx128 array)
        window_size: Window size for sweep matching (default: 30)
        distance_threshold: Maximum descriptor distance (None = no threshold)
        rectification_margin: Safety margin for epipole check in pixels (default: 50)
        affine_shapes_i: Affine shapes for image i (Nx2x2 array, optional)
        affine_shapes_j: Affine shapes for image j (Mx2x2 array, optional)
        geometric_config: Geometric filter configuration (None = disabled)

    Returns:
        List of (feat_i_idx, feat_j_idx, distance) tuples for mutual matches
    """
    K_i = get_intrinsic_matrix(cam_i)
    K_j = get_intrinsic_matrix(cam_j)
    R_i = np.asarray(img_i_cam_from_world.rotation.matrix(), dtype=np.float64)
    t_i = np.asarray(img_i_cam_from_world.translation, dtype=np.float64)
    R_j = np.asarray(img_j_cam_from_world.rotation.matrix(), dtype=np.float64)
    t_j = np.asarray(img_j_cam_from_world.translation, dtype=np.float64)

    # Prepare optional geometric filter params
    aff_i = None
    aff_j = None
    max_angle_diff = None
    min_tri_angle = None
    size_ratio_min = None
    size_ratio_max = None

    if (
        geometric_config is not None
        and geometric_config.enable_geometric_filtering
        and affine_shapes_i is not None
        and affine_shapes_j is not None
    ):
        aff_i = np.asarray(affine_shapes_i.reshape(-1, 4), dtype=np.float64)
        aff_j = np.asarray(affine_shapes_j.reshape(-1, 4), dtype=np.float64)
        max_angle_diff = geometric_config.max_angle_difference
        min_tri_angle = geometric_config.min_triangulation_angle
        size_ratio_min = geometric_config.geometric_size_ratio_min
        size_ratio_max = geometric_config.geometric_size_ratio_max

    return _rust_match_image_pair(
        np.asarray(K_i, dtype=np.float64),
        np.asarray(K_j, dtype=np.float64),
        R_i,
        R_j,
        t_i,
        t_j,
        cam_i.width,
        cam_i.height,
        cam_j.width,
        cam_j.height,
        np.asarray(positions_i, dtype=np.float64),
        np.asarray(descriptors_i, dtype=np.uint8),
        np.asarray(positions_j, dtype=np.float64),
        np.asarray(descriptors_j, dtype=np.uint8),
        window_size,
        distance_threshold,
        rectification_margin,
        aff_i,
        aff_j,
        max_angle_diff,
        min_tri_angle,
        size_ratio_min,
        size_ratio_max,
    )


def match_registered_images(
    image_names: list[str],
    pairs: list[tuple[int, int]],
    quaternions: np.ndarray,
    translations: np.ndarray,
    camera_indexes: np.ndarray,
    cameras: list[pycolmap.Camera],
    sift_positions: dict[int, np.ndarray],
    sift_descriptors: dict[int, np.ndarray],
    window_size: int = 30,
    distance_threshold: float | None = None,
    rectification_margin: int = 50,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Match features across multiple image pairs with known camera poses.

    This is a convenience wrapper around the Rust batch matcher.

    Args:
        image_names: List of image filenames
        pairs: List of (img_i, img_j) pairs to match
        quaternions: (N, 4) camera quaternions in WXYZ format
        translations: (N, 3) camera translations
        camera_indexes: (N,) camera index for each image
        cameras: List of pycolmap.Camera objects
        sift_positions: Dict mapping image index to Nx2 positions array
        sift_descriptors: Dict mapping image index to Nx128 descriptors array
        window_size: Window size for sweep matching
        distance_threshold: Maximum descriptor distance (None = no threshold)
        rectification_margin: Safety margin for epipole check in pixels

    Returns:
        Dict mapping (img_i, img_j) -> list of (feat_i, feat_j) match tuples
    """
    intrinsics_list = [
        np.asarray(get_intrinsic_matrix(cam), dtype=np.float64) for cam in cameras
    ]

    rotations_list = []
    translations_list = []
    for img_idx in range(len(image_names)):
        quat_xyzw = np.roll(quaternions[img_idx], -1)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(quat_xyzw), translations[img_idx])
        rotations_list.append(np.asarray(pose.rotation.matrix(), dtype=np.float64))
        translations_list.append(np.asarray(pose.translation, dtype=np.float64))

    num_images = len(image_names)
    positions_list = []
    descriptors_list = []
    for img_idx in range(num_images):
        if img_idx in sift_positions:
            positions_list.append(np.asarray(sift_positions[img_idx], dtype=np.float64))
            descriptors_list.append(
                np.asarray(sift_descriptors[img_idx], dtype=np.uint8)
            )
        else:
            positions_list.append(np.zeros((0, 2), dtype=np.float64))
            descriptors_list.append(np.zeros((0, 128), dtype=np.uint8))

    cam_widths = [cam.width for cam in cameras]
    cam_heights = [cam.height for cam in cameras]

    batch_results = _rust_match_image_pairs_batch(
        pairs,
        intrinsics_list,
        rotations_list,
        translations_list,
        np.asarray(camera_indexes, dtype=np.int64),
        positions_list,
        descriptors_list,
        cam_widths,
        cam_heights,
        window_size,
        distance_threshold,
        rectification_margin,
    )

    all_matches = {}
    for (img_i, img_j), matches in zip(pairs, batch_results):
        match_tuples = [(int(m[0]), int(m[1])) for m in matches]
        all_matches[(img_i, img_j)] = match_tuples

    return all_matches
