# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Rotation representations, in numpy alone.

Provenance: the study's `shape/shapelib.py` (`_rots_from_wxyz` 95-107,
`_quat_from_rot` 168-205, `_rot_angle_deg` 205-211) and `relaxlib.rot_angle`
(189-196).  The study reached for `scipy.spatial.transform.Rotation` in three
other places; those calls are restated here so the package carries no scipy
dependency.

The quaternion convention is WXYZ throughout, which is the adjustment
kernel's own.
"""

from __future__ import annotations

import numpy as np


def rots_from_wxyz(quats):
    """``(N, 3, 3)`` world-to-camera rotations from ``(N, 4)`` wxyz rows."""
    q = np.asarray(quats, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1
            ),
        ],
        1,
    )


def quat_from_rot(r):
    """The wxyz quaternion of one rotation matrix.

    Branch-free: the dominant eigenvector of the symmetric Bar-Itzhack matrix,
    which is defined for every rotation and needs no case analysis on the
    trace.  The sign is fixed by ``w >= 0`` so the map is a function."""
    r = np.asarray(r, float)
    k = (
        np.array(
            [
                [
                    r[0, 0] + r[1, 1] + r[2, 2],
                    r[2, 1] - r[1, 2],
                    r[0, 2] - r[2, 0],
                    r[1, 0] - r[0, 1],
                ],
                [
                    r[2, 1] - r[1, 2],
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                ],
                [
                    r[0, 2] - r[2, 0],
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                ],
                [
                    r[1, 0] - r[0, 1],
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                ],
            ],
            float,
        )
        / 3.0
    )
    _ev, evec = np.linalg.eigh(k)
    q = evec[:, -1]
    return q if q[0] >= 0 else -q


def quats_from_rots(rots):
    """``(N, 4)`` wxyz quaternions of ``(N, 3, 3)`` rotation matrices."""
    r = np.asarray(rots, float)
    if r.ndim == 2:
        return quat_from_rot(r)
    return np.stack([quat_from_rot(x) for x in r])


def rot_from_rotvec(v):
    """One rotation matrix from a rotation vector, by Rodrigues."""
    v = np.asarray(v, float)
    theta = float(np.linalg.norm(v))
    if not (theta > 0):
        return np.eye(3)
    axis = v / theta
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def rots_from_rotvecs(vecs):
    """``(N, 3, 3)`` rotation matrices from ``(N, 3)`` rotation vectors."""
    return np.stack([rot_from_rotvec(v) for v in np.asarray(vecs, float)])


def rotvec_from_rot(r):
    """The rotation vector of one rotation matrix.

    Through the quaternion, so the axis stays defined as the angle approaches
    zero and the branch at pi is the eigenvector's, not a trace formula's."""
    q = quat_from_rot(r)
    v = q[1:]
    n = float(np.linalg.norm(v))
    if not (n > 0):
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(n, float(q[0])))
    return (angle / n) * v


def rotvecs_from_rots(rots):
    """``(N, 3)`` rotation vectors of ``(N, 3, 3)`` rotation matrices."""
    return np.stack([rotvec_from_rot(r) for r in np.asarray(rots, float)])


def rot_angle_deg(r):
    """The rotation angle of ``R``, in degrees."""
    c = (float(np.trace(np.asarray(r, float))) - 1.0) / 2.0
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
