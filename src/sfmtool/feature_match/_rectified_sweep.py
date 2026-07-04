# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Sort and sweep matching algorithm for feature correspondences.

This algorithm finds matches between two sets of features by operating in a
stereo rectified image space. In this space, corresponding points are expected
to lie on the same horizontal scanline. The matching process involves sorting
features by their Y-coordinate and searching within a local window (the "sweep")
to find the best descriptor match, followed by a mutual consistency check.

All matching paths use Rust implementations.
"""

import numpy as np
import pycolmap

from ._geometric_filter import GeometricFilterConfig
from .._sfmtool.matching import (
    match_one_way_sweep_py as _rust_match_one_way_sweep,
    match_one_way_sweep_geometric_py as _rust_match_one_way_sweep_geometric,
    mutual_best_match_sweep_py as _rust_mutual_best_match_sweep,
    mutual_best_match_sweep_geometric_py as _rust_mutual_best_match_sweep_geometric,
)


def _match_one_way_sweep(
    sorted_kpts1: np.ndarray,
    sorted_descs1: np.ndarray,
    sorted_kpts2: np.ndarray,
    sorted_descs2: np.ndarray,
    window_size: int,
    threshold: float | None = None,
    sorted_affines1: np.ndarray | None = None,
    sorted_affines2: np.ndarray | None = None,
    K1: np.ndarray | None = None,
    K2: np.ndarray | None = None,
    pose1: pycolmap.Rigid3d | None = None,
    pose2: pycolmap.Rigid3d | None = None,
    R_2d: np.ndarray | None = None,
    geometric_config: GeometricFilterConfig | None = None,
) -> dict[int, tuple[int, float]]:
    """Perform a one-way sweep match from sorted image 1 to sorted image 2.

    Uses Rust for both descriptor-only and geometric filtering paths.
    """
    use_geometric_filter = (
        geometric_config is not None
        and geometric_config.enable_geometric_filtering
        and sorted_affines1 is not None
        and sorted_affines2 is not None
        and K1 is not None
        and K2 is not None
        and pose1 is not None
        and pose2 is not None
        and R_2d is not None
    )

    if not use_geometric_filter:
        result_tuples = _rust_match_one_way_sweep(
            np.asarray(sorted_kpts1, dtype=np.float64),
            np.asarray(sorted_descs1, dtype=np.uint8),
            np.asarray(sorted_kpts2, dtype=np.float64),
            np.asarray(sorted_descs2, dtype=np.uint8),
            window_size,
            threshold,
        )
        return {idx1: (idx2, dist) for idx1, idx2, dist in result_tuples}

    # The Rust geometric matcher expects COLMAP/OpenCV-frame poses; the .sfmr
    # poses passed in are canonical, so S-flip the camera frames (S-only, D3).
    # These are consumed as matrices, so flip the matrices directly rather than
    # round-tripping through a throwaway pycolmap.Rigid3d.
    from ..colmap.convention import flip_camera_pose_matrix_s

    R1, t1_vec = flip_camera_pose_matrix_s(pose1.rotation.matrix(), pose1.translation)
    R2, t2_vec = flip_camera_pose_matrix_s(pose2.rotation.matrix(), pose2.translation)
    K1_arr = np.asarray(K1, dtype=np.float64)
    K2_arr = np.asarray(K2, dtype=np.float64)
    aff1_flat = np.asarray(sorted_affines1.reshape(-1, 4), dtype=np.float64)
    aff2_flat = np.asarray(sorted_affines2.reshape(-1, 4), dtype=np.float64)
    result_tuples = _rust_match_one_way_sweep_geometric(
        np.asarray(sorted_kpts1, dtype=np.float64),
        np.asarray(sorted_descs1, dtype=np.uint8),
        np.asarray(sorted_kpts2, dtype=np.float64),
        np.asarray(sorted_descs2, dtype=np.uint8),
        aff1_flat,
        aff2_flat,
        K1_arr,
        K2_arr,
        R1,
        R2,
        t1_vec,
        t2_vec,
        window_size,
        threshold,
        geometric_config.max_angle_difference,
        geometric_config.min_triangulation_angle,
        geometric_config.geometric_size_ratio_min,
        geometric_config.geometric_size_ratio_max,
    )
    return {idx1: (idx2, dist) for idx1, idx2, dist in result_tuples}


def mutual_best_match_sweep(
    keypoints1: np.ndarray,
    descriptors1: np.ndarray,
    keypoints2: np.ndarray,
    descriptors2: np.ndarray,
    window_size: int,
    threshold: float | None = None,
    affine_shapes1: np.ndarray | None = None,
    affine_shapes2: np.ndarray | None = None,
    K1: np.ndarray | None = None,
    K2: np.ndarray | None = None,
    pose1: pycolmap.Rigid3d | None = None,
    pose2: pycolmap.Rigid3d | None = None,
    R_2d: np.ndarray | None = None,
    geometric_config: GeometricFilterConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Perform bidirectional sort-and-sweep matching with mutual best match.

    Uses Rust for both descriptor-only and geometric filtering paths.
    """
    use_geometric_filter = (
        geometric_config is not None
        and geometric_config.enable_geometric_filtering
        and affine_shapes1 is not None
        and affine_shapes2 is not None
        and pose1 is not None
        and pose2 is not None
    )

    if not use_geometric_filter:
        return _rust_mutual_best_match_sweep(
            np.asarray(keypoints1, dtype=np.float64),
            np.asarray(descriptors1, dtype=np.uint8),
            np.asarray(keypoints2, dtype=np.float64),
            np.asarray(descriptors2, dtype=np.uint8),
            window_size,
            threshold,
        )

    # The Rust geometric matcher expects COLMAP/OpenCV-frame poses; the .sfmr
    # poses passed in are canonical, so S-flip the camera frames (S-only, D3).
    # These are consumed as matrices, so flip the matrices directly rather than
    # round-tripping through a throwaway pycolmap.Rigid3d.
    from ..colmap.convention import flip_camera_pose_matrix_s

    R1, t1_vec = flip_camera_pose_matrix_s(pose1.rotation.matrix(), pose1.translation)
    R2, t2_vec = flip_camera_pose_matrix_s(pose2.rotation.matrix(), pose2.translation)
    K1_arr = np.asarray(K1, dtype=np.float64)
    K2_arr = np.asarray(K2, dtype=np.float64)
    aff1_flat = np.asarray(affine_shapes1.reshape(-1, 4), dtype=np.float64)
    aff2_flat = np.asarray(affine_shapes2.reshape(-1, 4), dtype=np.float64)
    return _rust_mutual_best_match_sweep_geometric(
        np.asarray(keypoints1, dtype=np.float64),
        np.asarray(descriptors1, dtype=np.uint8),
        np.asarray(keypoints2, dtype=np.float64),
        np.asarray(descriptors2, dtype=np.uint8),
        aff1_flat,
        aff2_flat,
        K1_arr,
        K2_arr,
        R1,
        R2,
        t1_vec,
        t2_vec,
        window_size,
        threshold,
        geometric_config.max_angle_difference,
        geometric_config.min_triangulation_angle,
        geometric_config.geometric_size_ratio_min,
        geometric_config.geometric_size_ratio_max,
    )
