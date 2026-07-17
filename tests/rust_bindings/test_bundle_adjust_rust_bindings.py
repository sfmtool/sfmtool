# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the staged robust bundle adjustment Rust binding
(``sfmtool._sfmtool.geometry.bundle_adjust``).

Canonical camera convention throughout: the camera looks along −Z, so an
in-front point has ``z < 0`` in camera frame. Rotations are built with numpy
(the test env has no scipy). The synthetic scene mirrors the Rust unit
tests: cameras on a shallow arc looking at the origin, points in a cloud
around it, observations by exact projection.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics, bundle_adjust


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


def _matrix_to_quat_wxyz(m):
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-9:
        x = (m[2, 1] - m[1, 2]) / (4 * w)
        y = (m[0, 2] - m[2, 0]) / (4 * w)
        z = (m[1, 0] - m[0, 1]) / (4 * w)
    else:  # pragma: no cover - not hit by these scenes
        raise ValueError("quaternion near w=0 not needed here")
    return np.array([w, x, y, z])


def _look_at_origin(center):
    """World-to-camera rotation with the canonical −Z axis toward the origin."""
    z = center / np.linalg.norm(center)  # camera +Z points away from origin
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows = camera axes → world-to-camera


def _scene(n_img=6, n_pt=50, f=500.0, seed=7):
    rng = np.random.default_rng(seed)
    cam = _cam(f)
    rots, trans, centers = [], [], []
    for i in range(n_img):
        ang = 0.15 * (i - (n_img - 1) / 2)
        c = np.array([8.0 * np.sin(ang), 0.3 * rng.uniform(-1, 1), 8.0 * np.cos(ang)])
        r = _look_at_origin(c)
        rots.append(r)
        trans.append(-r @ c)
        centers.append(c)
    pts = rng.uniform(-2.0, 2.0, size=(n_pt, 3)) * np.array([1.0, 1.0, 0.75])
    uv, oi, op = [], [], []
    for p in range(n_pt):
        for i in range(n_img):
            c = rots[i] @ pts[p] + trans[i]
            if c[2] >= -0.5:
                continue
            u = f * (-c[0] / c[2]) + 320.0
            v = f * (c[1] / c[2]) + 240.0  # ray_to_pixel flips y internally
            uv.append([u, v])
            oi.append(i)
            op.append(p)
    # Recompute the observed pixels through the binding's own forward model so
    # the test does not depend on the projection's sign conventions.
    rays = np.array(
        [rots[i] @ pts[p] + trans[i] for i, p in zip(oi, op)], dtype=np.float64
    )
    uv = cam.ray_to_pixel_batch(np.ascontiguousarray(rays))
    quats = np.array([_matrix_to_quat_wxyz(r) for r in rots])
    return {
        "cam": cam,
        "quats": quats,
        "trans": np.array(trans),
        "points": pts,
        "uv": np.asarray(uv, dtype=np.float64),
        "obs_image": np.array(oi, dtype=np.uint32),
        "obs_point": np.array(op, dtype=np.uint32),
    }


def _run(s, **kw):
    return bundle_adjust(
        s["cam"],
        s["quats"],
        s["trans"],
        s["points"],
        s["uv"],
        s["obs_image"],
        s["obs_point"],
        **kw,
    )


def test_perfect_data_stays_put():
    s = _scene()
    out = _run(s)
    npt.assert_allclose(out["quaternions_wxyz"], s["quats"], atol=1e-6)
    npt.assert_allclose(out["translations"], s["trans"], atol=1e-5)
    assert out["focal"] == 500.0
    assert np.max(out["residual_norms"]) < 1e-5


def test_recovers_perturbed_poses_and_points():
    s = _scene()
    rng = np.random.default_rng(3)
    q_true = s["quats"].copy()
    # Perturb all but the first camera (gauge pin) and every point.
    pert = s["quats"].copy()
    for i in range(1, len(pert)):
        d = _rotvec_to_matrix(rng.uniform(-0.02, 0.02, 3))
        pert[i] = _matrix_to_quat_wxyz(d @ _quat_to_matrix(pert[i]))
    s["quats"] = pert
    s["trans"] = s["trans"] + np.concatenate(
        [np.zeros((1, 3)), rng.uniform(-0.05, 0.05, (len(pert) - 1, 3))]
    )
    s["points"] = s["points"] + rng.uniform(-0.05, 0.05, s["points"].shape)
    out = _run(s)
    res = out["residual_norms"]
    assert np.median(res) < 0.05, f"median {np.median(res)}"
    # Cameras land back near truth up to the free global gauge: compare the
    # relative rotations R_i · R_0ᵀ (gauge-invariant) against ground truth.
    r_est = [_quat_to_matrix(q) for q in out["quaternions_wxyz"]]
    r_true = [_quat_to_matrix(q) for q in q_true]
    for i in range(1, len(q_true)):
        rel = (r_est[i] @ r_est[0].T) @ (r_true[i] @ r_true[0].T).T
        ang = np.arccos(min(1.0, (np.trace(rel) - 1) / 2))
        assert ang < 5e-3, f"camera {i} relative rotation err {ang}"


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_opt_f_recovers_focal():
    s = _scene(n_img=8, n_pt=60)
    s["cam"] = _cam(600.0)  # observations generated at f=500
    out = _run(s, opt_f=True)
    assert abs(out["focal"] - 500.0) < 5.0, out["focal"]


def test_nan_points_readmitted_by_retriangulation():
    s = _scene()
    s["points"][::2] = np.nan
    out = _run(s, schedule=[(50.0, 2.0), (4.0, 1.0)])
    assert np.all(np.isfinite(out["points"][::2]))
    assert np.max(out["residual_norms"]) < 0.1


def test_degenerate_exit_passes_state_through():
    s = _scene(n_img=3, n_pt=6)
    out = _run(s, schedule=[(0.0, 1.0)])
    assert np.all(np.isinf(out["residual_norms"]))
    npt.assert_allclose(out["quaternions_wxyz"], s["quats"], atol=1e-12)
    npt.assert_allclose(out["translations"], s["trans"], atol=1e-12)


def test_opt_f_rejected_for_non_simple_pinhole():
    s = _scene()
    s["cam"] = CameraIntrinsics.from_dict(
        {
            "model": "PINHOLE",
            "width": 640,
            "height": 480,
            "parameters": {
                "focal_length_x": 500.0,
                "focal_length_y": 500.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
            },
        }
    )
    with pytest.raises(ValueError, match="SIMPLE_PINHOLE"):
        _run(s, opt_f=True)


def test_fortran_order_inputs_match_c_order():
    # A Fortran-ordered 2-D array satisfies as_slice() with column-major
    # memory; the binding's to_contiguous! must not read it as row-major
    # (a silent transpose regression gave focal 309 instead of 500 here).
    s = _scene(n_img=8, n_pt=60)
    s["cam"] = _cam(600.0)
    ref = _run(s, opt_f=True)
    f = _scene(n_img=8, n_pt=60)
    f["cam"] = _cam(600.0)
    for key in ("quats", "trans", "points", "uv"):
        f[key] = np.asfortranarray(f[key])
    out = _run(f, opt_f=True)
    npt.assert_allclose(out["focal"], ref["focal"], rtol=0, atol=1e-9)
    npt.assert_allclose(out["translations"], ref["translations"], atol=1e-9)
    npt.assert_allclose(out["residual_norms"], ref["residual_norms"], atol=1e-9)


def test_shape_validation():
    s = _scene()
    with pytest.raises(ValueError, match="uv"):
        bundle_adjust(
            s["cam"],
            s["quats"],
            s["trans"],
            s["points"],
            s["uv"][:, :1],
            s["obs_image"],
            s["obs_point"],
        )
    with pytest.raises(ValueError, match="out of range"):
        bundle_adjust(
            s["cam"],
            s["quats"],
            s["trans"],
            s["points"],
            s["uv"],
            s["obs_image"],
            np.full_like(s["obs_point"], 10_000),
        )
