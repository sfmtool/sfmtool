# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate-convention conversion helpers (COLMAP/OpenCV <-> canonical).

Thin wrappers over the Rust primitives in ``sfmtool_core::geometry::convention``
(exposed on ``sfmtool._sfmtool.geometry``), which are the single source of
truth for the ``S``/``W`` math. See
``specs/formats/sfmr-file-format.md`` section "Coordinate System Conventions"
and ``specs/drafts/zup-camera-convention-migration.md`` sections 1-2 (D2).

Conventions:

- canonical: right-handed Z-up world; cameras look down -Z, image plane
  +X right / +Y up (OpenGL-style).
- COLMAP/OpenCV: cameras look down +Z with Y down; worlds typically -Y-up.

All pose functions take world-to-camera poses as WXYZ quaternions plus
translations and accept either a single pose (shapes ``(4,)``/``(3,)``) or a
batch (``(N, 4)``/``(N, 3)``), returning arrays of the same shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._sfmtool.geometry import (
    poses_canonical_to_colmap,
    poses_colmap_to_canonical,
    relative_poses_conjugate_s,
    world_rotate_w as _world_rotate_w_batch,
    world_rotate_w_inverse as _world_rotate_w_inverse_batch,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pycolmap

__all__ = [
    "pose_colmap_to_canonical",
    "pose_canonical_to_colmap",
    "relative_pose_conjugate_s",
    "world_rotate_w",
    "world_rotate_w_inverse",
    "points_xyzw_rotate_w",
    "points_xyzw_rotate_w_inverse",
    "rigid3d_colmap_to_canonical",
    "canonical_pose_to_rigid3d",
]


def _apply_pose_fn(fn, quats_wxyz, translations_xyz):
    """Call a batch pose converter, accepting single or batched poses."""
    q = np.asarray(quats_wxyz, dtype=np.float64)
    t = np.asarray(translations_xyz, dtype=np.float64)
    single = q.ndim == 1
    if single != (t.ndim == 1):
        raise ValueError(
            "quaternions and translations must both be single (1D) or both batched (2D)"
        )
    q_out, t_out = fn(np.atleast_2d(q), np.atleast_2d(t))
    if single:
        return q_out[0], t_out[0]
    return q_out, t_out


def _apply_vector_fn(fn, vectors_xyz):
    """Call a batch vector rotation, accepting a single ``(3,)`` or ``(N, 3)``."""
    v = np.asarray(vectors_xyz, dtype=np.float64)
    single = v.ndim == 1
    out = fn(np.atleast_2d(v))
    if single:
        return out[0]
    return out


def pose_colmap_to_canonical(quats_wxyz, translations_xyz):
    """COLMAP -> canonical world-to-camera pose: ``R' = S.R.W^T``, ``t' = S.t``."""
    return _apply_pose_fn(poses_colmap_to_canonical, quats_wxyz, translations_xyz)


def pose_canonical_to_colmap(quats_wxyz, translations_xyz):
    """Canonical -> COLMAP world-to-camera pose: ``R = S.R'.W``, ``t = S.t'``."""
    return _apply_pose_fn(poses_canonical_to_colmap, quats_wxyz, translations_xyz)


def relative_pose_conjugate_s(quats_wxyz, translations_xyz):
    """Conjugate relative poses (``cam2_from_cam1`` / ``sensor_from_rig``) by ``S``.

    ``R' = S.R.S``, ``t' = S.t``. Involutive: the same call converts
    COLMAP -> canonical and back.
    """
    return _apply_pose_fn(relative_poses_conjugate_s, quats_wxyz, translations_xyz)


def world_rotate_w(vectors_xyz):
    """Rotate world vectors by ``W``: ``(x, y, z) -> (x, z, -y)``.

    For finite point xyz, infinity directions, normals, and patch ``u``/``v``
    half-vectors on COLMAP -> canonical import.
    """
    return _apply_vector_fn(_world_rotate_w_batch, vectors_xyz)


def world_rotate_w_inverse(vectors_xyz):
    """Rotate world vectors by ``W^-1 = W^T``: ``(x, y, z) -> (x, -z, y)``."""
    return _apply_vector_fn(_world_rotate_w_inverse_batch, vectors_xyz)


def points_xyzw_rotate_w(points_xyzw):
    """Rotate homogeneous ``(N, 4)`` xyzw points by ``W``, carrying ``w`` unchanged.

    Finite points (``w = 1``) and infinity directions (``w = 0``) both rotate
    on their xyz part only.
    """
    pts = np.asarray(points_xyzw, dtype=np.float64)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = np.empty_like(pts)
    out[:, :3] = _world_rotate_w_batch(np.ascontiguousarray(pts[:, :3]))
    out[:, 3] = pts[:, 3]
    if single:
        return out[0]
    return out


def points_xyzw_rotate_w_inverse(points_xyzw):
    """Rotate homogeneous ``(N, 4)`` xyzw points by ``W^-1``, carrying ``w``."""
    pts = np.asarray(points_xyzw, dtype=np.float64)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = np.empty_like(pts)
    out[:, :3] = _world_rotate_w_inverse_batch(np.ascontiguousarray(pts[:, :3]))
    out[:, 3] = pts[:, 3]
    if single:
        return out[0]
    return out


def rigid3d_colmap_to_canonical(
    rigid3d: "pycolmap.Rigid3d",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a COLMAP-convention ``pycolmap.Rigid3d`` world-to-camera pose.

    Returns:
        Tuple of canonical ``(4,)`` WXYZ quaternion and ``(3,)`` translation.
    """
    xyzw = np.asarray(rigid3d.rotation.quat, dtype=np.float64)
    wxyz = xyzw[[3, 0, 1, 2]]
    translation = np.asarray(rigid3d.translation, dtype=np.float64)
    return pose_colmap_to_canonical(wxyz, translation)


def canonical_pose_to_rigid3d(quat_wxyz, translation_xyz) -> "pycolmap.Rigid3d":
    """Convert a canonical world-to-camera pose to a COLMAP ``pycolmap.Rigid3d``."""
    import pycolmap

    q, t = pose_canonical_to_colmap(quat_wxyz, translation_xyz)
    quat_xyzw = [q[1], q[2], q[3], q[0]]
    return pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(quat_xyzw),
        translation=t,
    )
