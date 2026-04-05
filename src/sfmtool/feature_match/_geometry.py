# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Geometric utilities for feature matching.

This module provides functions for:
- Computing essential and fundamental matrices
- Computing epipoles
- Checking if rectification is safe (epipole location)

These utilities determine which matching algorithm (rectified vs polar) is appropriate
based on the camera geometry.
"""

import numpy as np


def get_essential_matrix(
    R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray
) -> np.ndarray:
    """Compute the essential matrix E from two camera poses.

    Args:
        R1: 3x3 rotation matrix for camera 1 (world to camera)
        t1: 3x1 translation vector for camera 1 (world to camera)
        R2: 3x3 rotation matrix for camera 2 (world to camera)
        t2: 3x1 translation vector for camera 2 (world to camera)

    Returns:
        3x3 essential matrix E
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    t_skew = np.array(
        [
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0],
        ]
    )

    return t_skew @ R_rel


def get_fundamental_matrix(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """Compute the fundamental matrix F from two camera poses and intrinsics.

    Args:
        K1: 3x3 intrinsic matrix for camera 1
        R1: 3x3 rotation matrix for camera 1 (world to camera)
        t1: 3x1 translation vector for camera 1 (world to camera)
        K2: 3x3 intrinsic matrix for camera 2
        R2: 3x3 rotation matrix for camera 2 (world to camera)
        t2: 3x1 translation vector for camera 2 (world to camera)

    Returns:
        3x3 fundamental matrix F
    """
    E = get_essential_matrix(R1, t1, R2, t2)
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def compute_epipole(F: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute the epipole from a fundamental matrix.

    Args:
        F: 3x3 fundamental matrix

    Returns:
        Tuple of (epipole, is_at_infinity)
    """
    U, S, Vt = np.linalg.svd(F.T)
    epipole = Vt[-1, :]
    is_at_infinity = abs(epipole[2]) < 1e-10
    return epipole, is_at_infinity


def check_rectification_safe(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    width: int,
    height: int,
    margin: int = 50,
) -> bool:
    """Check if stereo rectification is safe for the given camera pair.

    Rectification becomes unstable when the epipole is inside or near the image.

    Args:
        K1: 3x3 intrinsic matrix for camera 1
        R1: 3x3 rotation matrix for camera 1 (world to camera)
        t1: 3x1 translation vector for camera 1 (world to camera)
        K2: 3x3 intrinsic matrix for camera 2
        R2: 3x3 rotation matrix for camera 2 (world to camera)
        t2: 3x1 translation vector for camera 2 (world to camera)
        width: Image width in pixels
        height: Image height in pixels
        margin: Safety margin in pixels (default: 50)

    Returns:
        True if rectification is safe, False otherwise
    """
    try:
        F = get_fundamental_matrix(K1, R1, t1, K2, R2, t2)
        epipole, is_at_infinity = compute_epipole(F)

        if is_at_infinity:
            return True

        x = epipole[0] / epipole[2]
        y = epipole[1] / epipole[2]

        outside = (
            x < -margin or x > width + margin or y < -margin or y > height + margin
        )

        return outside

    except np.linalg.LinAlgError:
        return False
