# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reprojection / pose-refinement Rust bindings
(``sfmtool._sfmtool.geometry.reprojection_residuals``, ``inlier_fraction``,
``refine_absolute_pose``).

Canonical camera convention throughout: the camera looks along −Z, so an
in-front point has ``z < 0`` in camera frame. ``CameraIntrinsics.ray_to_pixel_batch``
is the ground-truth forward model the residual must reproduce.  Rotations are
built with numpy (the test env has no scipy).
"""

import numpy as np
import numpy.testing as npt

from sfmtool._sfmtool.geometry import (
    CameraIntrinsics,
    inlier_fraction,
    refine_absolute_pose,
    reprojection_residuals,
)


def _cam(f=500.0, w=640, h=480):
    return CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": w,
            "height": h,
            "parameters": {
                "focal_length": f,
                "principal_point_x": w / 2.0,
                "principal_point_y": h / 2.0,
            },
        }
    )


def _rotvec_to_matrix(v):
    """Rodrigues: rotation matrix from an axis-angle vector."""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def _rotvec_to_quat_wxyz(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / theta
    return np.array([np.cos(theta / 2), *(np.sin(theta / 2) * axis)])


def _quat_wxyz_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rot_angle(r_a, r_b):
    c = (np.trace(r_a @ r_b.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def test_reprojection_residuals_zero_for_true_poses():
    cam = _cam()
    pts = np.array([[0.0, 0.0, -5.0], [1.0, 0.5, -4.0], [-2.0, 1.5, -6.0]])
    uv = cam.ray_to_pixel_batch(np.ascontiguousarray(pts))
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    t = np.array([[0.0, 0.0, 0.0]])
    res = reprojection_residuals(
        cam, q, t, pts, uv, np.zeros(3, np.uint32), np.arange(3, dtype=np.uint32)
    )
    assert res.shape == (3, 2)
    npt.assert_allclose(res, 0.0, atol=1e-9)
    assert inlier_fraction(res, 1.0) == 1.0


def test_reprojection_residuals_behind_camera_invalid():
    cam = _cam()
    pts = np.array([[0.0, 0.0, 5.0]])  # z > 0 → behind the canonical camera
    uv = np.array([[320.0, 240.0]])
    res = reprojection_residuals(
        cam,
        np.array([[1.0, 0, 0, 0]]),
        np.array([[0.0, 0, 0]]),
        pts,
        uv,
        np.zeros(1, np.uint32),
        np.arange(1, dtype=np.uint32),
        1e6,
    )
    assert res[0, 0] == 1e6
    assert inlier_fraction(res, 3.0) == 0.0


def test_reprojection_residuals_known_offset():
    cam = _cam()
    pts = np.array([[0.5, -0.3, -4.0]])
    uv_true = cam.ray_to_pixel_batch(np.ascontiguousarray(pts))
    uv = uv_true + np.array([[-2.0, 1.5]])
    res = reprojection_residuals(
        cam,
        np.array([[1.0, 0, 0, 0]]),
        np.array([[0.0, 0, 0]]),
        pts,
        uv,
        np.zeros(1, np.uint32),
        np.arange(1, dtype=np.uint32),
    )
    npt.assert_allclose(res[0], [2.0, -1.5], atol=1e-9)


def test_multi_image_reprojection_indexing():
    cam = _cam()
    world = np.array([[0.2, 0.1, -6.0], [-0.4, 0.3, -5.0]])
    q = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    t = np.array([[0.0, 0, 0], [0.5, 0, 1.0]])
    uv = []
    for tt in t:
        uv.append(cam.ray_to_pixel_batch(np.ascontiguousarray(world + tt)))
    uv = np.vstack(uv)
    obs_img = np.array([0, 0, 1, 1], np.uint32)
    obs_pt = np.array([0, 1, 0, 1], np.uint32)
    res = reprojection_residuals(cam, q, t, world, uv, obs_img, obs_pt)
    npt.assert_allclose(res, 0.0, atol=1e-9)


def test_refine_absolute_pose_recovers_pose():
    cam = _cam()
    rng = np.random.default_rng(3)
    rv_true = np.array([0.12, -0.08, 0.05])
    r_true = _rotvec_to_matrix(rv_true)
    t_true = np.array([0.2, -0.15, 0.3])
    world = rng.uniform(-2, 2, size=(50, 3))
    world[:, 2] = rng.uniform(-6, -3, size=50)
    cam_pts = world @ r_true.T + t_true
    front = cam_pts[:, 2] < -0.5
    world, cam_pts = world[front], cam_pts[front]
    uv = cam.ray_to_pixel_batch(np.ascontiguousarray(cam_pts))

    q0 = _rotvec_to_quat_wxyz(rv_true + np.array([0.06, -0.05, 0.03]))
    t0 = t_true + np.array([0.08, -0.06, 0.1])
    out = refine_absolute_pose(cam, uv, world, q0, t0, 5, 0.6, 3.0)
    assert out["inlier_fraction"] > 0.95
    r_est = _quat_wxyz_to_matrix(np.asarray(out["quaternion_wxyz"]))
    assert _rot_angle(r_est, r_true) < 1e-3
    npt.assert_allclose(np.asarray(out["translation"]), t_true, atol=1e-3)
