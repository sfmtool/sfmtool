# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the absolute-pose Rust bindings
(``sfmtool._sfmtool.geometry.p3p_solve`` and ``estimate_absolute_pose``;
see ``specs/core/absolute-pose.md``).

Synthetic scenes are generated in the canonical camera convention: a camera
looks along −Z, so a point in front has ``z < 0`` in camera frame and its
bearing is ``normalize(R·X + t)`` with negative z. The pixels path builds a
COLMAP-model ``CameraIntrinsics``, projects the camera-frame points to pixels,
and checks the estimator recovers the generating pose after converting pixels
back to bearings internally.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import (
    CameraIntrinsics,
    estimate_absolute_pose,
    p3p_solve,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _quat_to_matrix(wxyz):
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = wxyz / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_angle(r_a, r_b):
    """Angular difference (radians) between two rotation matrices."""
    c = (np.trace(r_a @ r_b.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _random_rotation(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return _quat_to_matrix(q)


def _scene(n, seed, noise_rad=0.0):
    """Generate a world-to-camera pose and `n` correspondences.

    Returns (R, t, world_points (N,3), cam_points (N,3), bearings (N,3)).
    Camera-frame points have z < 0 (in front of the canonical camera).
    """
    rng = np.random.default_rng(seed)
    r = _random_rotation(rng)
    t = rng.uniform(-2.0, 2.0, size=3)
    cam = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, size=n),
            rng.uniform(-3.0, 3.0, size=n),
            rng.uniform(-8.0, -1.5, size=n),
        ]
    )
    bearings = cam / np.linalg.norm(cam, axis=1, keepdims=True)
    if noise_rad > 0.0:
        bearings = bearings + noise_rad * rng.standard_normal(bearings.shape)
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
    # X = R^{-1} (cam - t)
    world = (cam - t) @ r  # r^T applied on the right (r orthonormal)
    return r, t, world, cam, bearings


def _contaminate_bearings(bearings, inlier_frac, seed):
    """Replace a fraction of bearings with random forward-pointing rays."""
    rng = np.random.default_rng(seed)
    n = len(bearings)
    n_in = int(round(n * inlier_frac))
    out = bearings.copy()
    truth = np.zeros(n, dtype=bool)
    truth[:n_in] = True
    for i in range(n_in, n):
        d = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-2, -0.2)])
        out[i] = d / np.linalg.norm(d)
    return out, truth


# ── p3p_solve ──────────────────────────────────────────────────────────────


def test_p3p_solve_recovers_generating_pose():
    for seed in range(30):
        r, t, world, _cam, bearings = _scene(3, seed)
        solutions = p3p_solve(bearings, world)
        assert len(solutions) >= 1
        best = min(
            _rotation_angle(_quat_to_matrix(q), r) + np.linalg.norm(tr - t)
            for q, tr in solutions
        )
        assert best < 1e-6, f"seed {seed}: best error {best}"


def test_p3p_solve_degenerate_returns_empty():
    # Collinear world points.
    bearings = np.array([[0.1, 0.0, -1.0], [0.0, 0.1, -1.0], [-0.1, 0.05, -1.0]])
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
    collinear = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 6.0], [2.0, 2.0, 7.0]])
    assert p3p_solve(bearings, collinear) == []


# ── estimate_absolute_pose: bearings path ──────────────────────────────────


@pytest.mark.parametrize("inlier_frac", [0.6, 0.4, 0.2])
def test_estimate_bearings_path(inlier_frac):
    r, t, world, _cam, bearings = _scene(200, seed=7)
    contaminated, _truth = _contaminate_bearings(bearings, inlier_frac, seed=11)
    result = estimate_absolute_pose(
        contaminated,
        world,
        max_angular_error=0.01,
        min_inliers=6,
        max_iterations=100_000,
        seed=3,
    )
    assert result is not None
    r_est = _quat_to_matrix(result["quaternion_wxyz"])
    assert _rotation_angle(r_est, r) < 1e-3
    npt.assert_allclose(result["translation"], t, atol=1e-3)
    assert result["inliers"].dtype == bool
    assert result["inliers"].shape == (200,)
    # Recovered inliers should cover the true inlier fraction.
    assert result["inliers"].sum() >= int(round(200 * inlier_frac)) - 2


def test_estimate_bearings_requires_threshold_source():
    r, t, world, _cam, bearings = _scene(20, seed=1)
    with pytest.raises(ValueError):
        estimate_absolute_pose(bearings, world)  # no camera, no angular error


# ── estimate_absolute_pose: pixels + camera path ───────────────────────────


def _project(camera, cam_points):
    """Project camera-frame points (z < 0) to pixels with the canonical model."""
    pixels = np.empty((len(cam_points), 2))
    for i, p in enumerate(cam_points):
        u, v = camera.project(p[0] / -p[2], p[1] / -p[2])
        pixels[i] = (u, v)
    return pixels


@pytest.mark.parametrize(
    "model,params",
    [
        (
            "SIMPLE_PINHOLE",
            {
                "focal_length": 520.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
            },
        ),
        (
            "SIMPLE_RADIAL",
            {
                "focal_length": 500.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
                "radial_distortion_k1": -0.05,
            },
        ),
    ],
)
def test_estimate_pixels_path(model, params):
    camera = CameraIntrinsics(model, 640, 480, params)
    r, t, world, cam, _bearings = _scene(180, seed=5)
    pixels = _project(camera, cam)
    # Contaminate a third of the observations with random pixels.
    rng = np.random.default_rng(21)
    n_out = 60
    pixels[-n_out:] = rng.uniform([0, 0], [640, 480], size=(n_out, 2))
    result = estimate_absolute_pose(
        pixels,
        world,
        camera=camera,
        max_error_px=3.0,
        min_inliers=8,
        max_iterations=100_000,
        seed=1,
    )
    assert result is not None
    r_est = _quat_to_matrix(result["quaternion_wxyz"])
    assert _rotation_angle(r_est, r) < 1e-2
    npt.assert_allclose(result["translation"], t, atol=1e-2)
    assert result["inliers"].sum() >= 100


def test_estimate_pixels_requires_camera():
    _r, _t, world, _cam, _bearings = _scene(20, seed=1)
    pixels = np.zeros((20, 2))
    with pytest.raises(ValueError):
        estimate_absolute_pose(pixels, world)


# ── Determinism and robustness ─────────────────────────────────────────────


def test_determinism_same_seed():
    r, t, world, _cam, bearings = _scene(150, seed=2, noise_rad=0.001)
    contaminated, _truth = _contaminate_bearings(bearings, 0.3, seed=9)
    kwargs = dict(max_angular_error=0.01, min_inliers=6, max_iterations=50_000, seed=42)
    a = estimate_absolute_pose(contaminated, world, **kwargs)
    b = estimate_absolute_pose(contaminated, world, **kwargs)
    assert a is not None and b is not None
    npt.assert_array_equal(a["quaternion_wxyz"], b["quaternion_wxyz"])
    npt.assert_array_equal(a["translation"], b["translation"])
    npt.assert_array_equal(a["inliers"], b["inliers"])
    assert a["iterations"] == b["iterations"]


def test_none_on_garbage():
    rng = np.random.default_rng(0)
    # Random, unrelated bearings and points: no consensus.
    bearings = rng.standard_normal((80, 3))
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
    points = rng.uniform(-5, 5, size=(80, 3))
    result = estimate_absolute_pose(
        bearings,
        points,
        max_angular_error=0.005,
        min_inliers=10,
        max_iterations=20_000,
        seed=0,
    )
    assert result is None


# ── Differential agreement with pycolmap (optional) ────────────────────────


def test_differential_against_pycolmap():
    pycolmap = pytest.importorskip("pycolmap")
    camera_model = "SIMPLE_PINHOLE"
    params = {
        "focal_length": 520.0,
        "principal_point_x": 320.0,
        "principal_point_y": 240.0,
    }
    camera = CameraIntrinsics(camera_model, 640, 480, params)
    r, t, world, cam, _bearings = _scene(200, seed=13)
    pixels = _project(camera, cam)
    rng = np.random.default_rng(31)
    n_out = 70
    pixels[-n_out:] = rng.uniform([0, 0], [640, 480], size=(n_out, 2))

    ours = estimate_absolute_pose(
        pixels,
        world,
        camera=camera,
        max_error_px=3.0,
        min_inliers=8,
        max_iterations=100_000,
        seed=1,
    )
    assert ours is not None

    # pycolmap's estimator (API varies across versions; skip if unavailable).
    try:
        pyc_camera = pycolmap.Camera(
            model=camera_model, width=640, height=480, params=[520.0, 320.0, 240.0]
        )
        res = pycolmap.estimate_and_refine_absolute_pose(pixels, world, pyc_camera)
    except (AttributeError, TypeError) as exc:  # pragma: no cover - version drift
        pytest.skip(f"pycolmap absolute-pose API unavailable: {exc}")
    if res is None:
        pytest.skip("pycolmap failed to estimate a pose on this set")

    n_ours = int(ours["inliers"].sum())
    n_pyc = int(res["num_inliers"])
    assert abs(n_ours - n_pyc) <= 0.2 * max(n_ours, n_pyc), (
        f"consensus sizes differ: ours {n_ours}, pycolmap {n_pyc}"
    )

    # COLMAP uses +Z-forward; ours uses −Z-forward. The two poses relate by
    # F = diag(1, -1, -1). Compare camera centers (convention-free) and the
    # flipped rotation.
    r_ours = _quat_to_matrix(ours["quaternion_wxyz"])
    t_ours = np.asarray(ours["translation"])
    cam_from_world = res["cam_from_world"]
    r_pyc = np.asarray(cam_from_world.rotation.matrix())
    t_pyc = np.asarray(cam_from_world.translation)

    center_ours = -r_ours.T @ t_ours
    center_pyc = -r_pyc.T @ t_pyc
    npt.assert_allclose(center_ours, center_pyc, atol=0.05)

    flip = np.diag([1.0, -1.0, -1.0])
    angle = _rotation_angle(flip @ r_ours, r_pyc)
    assert np.degrees(angle) < 1.0, f"rotation disagreement {np.degrees(angle)} deg"
