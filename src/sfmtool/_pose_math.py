# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Small pose derivations shared across the package.

Camera centers and relative rotation angles are read off a reconstruction's
`(quaternion, translation)` pose arrays all over the package — in `analyze/`,
`motion/`, `rig/`, `xform/` and the patch embedder. They live here so those
subpackages can share them without reaching into each other's private modules.
"""

import numpy as np
from numpy.typing import NDArray

from ._sfmtool.geometry import RotQuaternion


def camera_centers(quaternions, translations) -> NDArray[np.float64]:
    """World-space camera centers ``C = -R(q)ᵀ t`` for each image.

    ``quaternions`` are wxyz rows and ``translations`` the matching ``t`` of
    the ``x_cam = R x_world + t`` pose.
    """
    n = len(quaternions)
    centers = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        r_cam_from_world = RotQuaternion.from_wxyz_array(
            quaternions[i]
        ).to_rotation_matrix()
        centers[i] = -r_cam_from_world.T @ translations[i]
    return centers


def recon_camera_centers(recon) -> NDArray[np.float64]:
    """World-space camera centers ``(n_images, 3)`` for a whole reconstruction."""
    return camera_centers(recon.quaternions_wxyz, recon.translations)


def rotation_angle_deg(quat_a: RotQuaternion, quat_b: RotQuaternion) -> float:
    """Rotation angle in degrees between two quaternions."""
    return float(np.degrees((quat_b * quat_a.conjugate()).angle()))
