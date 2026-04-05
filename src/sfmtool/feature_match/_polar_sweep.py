# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Polar sweep matching algorithm for in-frame epipole cases.

This module provides feature matching when standard stereo rectification fails
due to the epipole being inside the image (forward/backward camera motion).

Instead of rectifying to horizontal epipolar lines, this algorithm:
1. Transforms features to polar coordinates centered at the epipole
2. Sorts by angle theta (which corresponds to epipolar lines)
3. Uses sort-and-sweep matching in polar space

The matching algorithms are implemented in Rust. This module provides the
Python interface and helper utilities.
"""

from dataclasses import dataclass

import numpy as np
import pycolmap

from ._geometric_filter import GeometricFilterConfig
from .._sfmtool import (
    polar_mutual_best_match_py as _rust_polar_mutual_best_match,
    polar_mutual_best_match_geometric_py as _rust_polar_mutual_best_match_geometric,
)


@dataclass
class PolarCoordinates:
    """Polar coordinates for features relative to an epipole."""

    theta: np.ndarray  # (N,) angles in radians [-pi, pi]
    radius: np.ndarray  # (N,) distances from epipole
    original_indices: np.ndarray  # (N,) mapping to original feature indices
    epipole: np.ndarray  # (2,) epipole position


def _cartesian_to_polar(
    points: np.ndarray,
    epipole: np.ndarray,
    min_radius: float = 10.0,
) -> PolarCoordinates:
    """Transform 2D points to polar coordinates centered at the epipole."""
    dx = points[:, 0] - epipole[0]
    dy = points[:, 1] - epipole[1]

    radius = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    valid_mask = radius >= min_radius
    valid_indices = np.where(valid_mask)[0]

    return PolarCoordinates(
        theta=theta[valid_mask],
        radius=radius[valid_mask],
        original_indices=valid_indices,
        epipole=epipole.copy(),
    )


def _compute_epipole_pair_from_F(
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Compute both epipoles from the fundamental matrix.

    Returns:
        (e1, e2, e1_at_inf, e2_at_inf)
    """
    # e2 is null space of F^T
    _, _, Vt = np.linalg.svd(F.T)
    e2_h = Vt[-1, :]
    e2_at_inf = abs(e2_h[2]) < 1e-10
    e2 = e2_h[:2] / e2_h[2] if not e2_at_inf else e2_h

    # e1 is null space of F
    _, _, Vt = np.linalg.svd(F)
    e1_h = Vt[-1, :]
    e1_at_inf = abs(e1_h[2]) < 1e-10
    e1 = e1_h[:2] / e1_h[2] if not e1_at_inf else e1_h

    return e1, e2, e1_at_inf, e2_at_inf


def polar_mutual_best_match(
    positions1: np.ndarray,
    descriptors1: np.ndarray,
    positions2: np.ndarray,
    descriptors2: np.ndarray,
    F: np.ndarray,
    window_size: int = 15,
    threshold: float | None = None,
    min_radius: float = 10.0,
    affine_shapes1: np.ndarray | None = None,
    affine_shapes2: np.ndarray | None = None,
    K1: np.ndarray | None = None,
    K2: np.ndarray | None = None,
    pose1: pycolmap.Rigid3d | None = None,
    pose2: pycolmap.Rigid3d | None = None,
    R_2d: np.ndarray | None = None,
    geometric_config: GeometricFilterConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Perform bidirectional polar sweep matching for in-frame epipole cases.

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
        result = _rust_polar_mutual_best_match(
            np.asarray(positions1, dtype=np.float64),
            np.asarray(descriptors1, dtype=np.uint8),
            np.asarray(positions2, dtype=np.float64),
            np.asarray(descriptors2, dtype=np.uint8),
            np.asarray(F, dtype=np.float64),
            window_size,
            threshold,
            min_radius,
        )
        if result is None:
            raise ValueError(
                "Epipole is at infinity - use standard rectification instead"
            )
        return result

    R1 = np.asarray(pose1.rotation.matrix(), dtype=np.float64)
    t1_vec = np.asarray(pose1.translation, dtype=np.float64)
    R2 = np.asarray(pose2.rotation.matrix(), dtype=np.float64)
    t2_vec = np.asarray(pose2.translation, dtype=np.float64)
    K1_arr = np.asarray(K1, dtype=np.float64)
    K2_arr = np.asarray(K2, dtype=np.float64)
    aff1_flat = np.asarray(affine_shapes1.reshape(-1, 4), dtype=np.float64)
    aff2_flat = np.asarray(affine_shapes2.reshape(-1, 4), dtype=np.float64)
    f_arr = np.asarray(F, dtype=np.float64)
    result = _rust_polar_mutual_best_match_geometric(
        np.asarray(positions1, dtype=np.float64),
        np.asarray(descriptors1, dtype=np.uint8),
        np.asarray(positions2, dtype=np.float64),
        np.asarray(descriptors2, dtype=np.uint8),
        aff1_flat,
        aff2_flat,
        f_arr,
        K1_arr,
        K2_arr,
        R1,
        R2,
        t1_vec,
        t2_vec,
        window_size,
        threshold,
        min_radius,
        geometric_config.max_angle_difference,
        geometric_config.min_triangulation_angle,
        geometric_config.geometric_size_ratio_min,
        geometric_config.geometric_size_ratio_max,
    )
    if result is None:
        raise ValueError("Epipole is at infinity - use standard rectification instead")
    return result
