# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ray-space two-view Rust bindings
(``sfmtool._sfmtool.geometry.estimate_essential_rays`` / ``fit_ray_rotation``).

The estimators consume unit rays, so a field of view past 180 degrees is
ordinary input: every fixture here plants a substantial share of its
correspondences at ``theta >= 90 deg`` (ray ``z <= 0``), the population a
``z = 1`` normalization would silently drop.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import estimate_essential_rays, fit_ray_rotation


def rodrigues(rotvec):
    """Rotation matrix of an axis-angle vector (no scipy in the test env)."""
    theta = np.linalg.norm(rotvec)
    k = rotvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def planted_rays(n, theta_max):
    """Deterministic unit rays spread over the sphere out to ``theta_max``."""
    i = np.arange(n)
    theta = theta_max * (i + 0.5) / n
    phi = 2.39996323 * i
    return np.ascontiguousarray(
        np.c_[
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


ROT = rodrigues(np.array([0.11, -0.31, 0.14]))
T = np.array([0.9, -0.25, 0.4])


def two_view(n=200, theta_max=2.1):
    """Rays of a planted two-view geometry, plus its (R, t)."""
    r1 = planted_rays(n, theta_max)
    depth = 3.0 + 2.0 * (np.sin(0.7 * np.arange(n)) + 1.0)
    r2 = unit((r1 * depth[:, None]) @ ROT.T + T)
    return r1, np.ascontiguousarray(r2)


def test_scene_reaches_past_the_hemisphere():
    r1, _ = two_view()
    assert (r1[:, 2] <= 0).sum() > 40


def test_essential_recovers_the_planted_geometry():
    r1, r2 = two_view()
    out = estimate_essential_rays(r1, r2, max_angle_rad=1e-6, min_inliers=20)
    assert out is not None
    assert np.asarray(out["inliers"]).all()
    assert out["essentialness"] < 1e-9
    tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    truth = tx @ ROT
    truth /= np.linalg.norm(truth)
    got = np.asarray(out["e_matrix"])
    assert min(np.linalg.norm(got - truth), np.linalg.norm(got + truth)) < 1e-9


def test_essential_is_deterministic_and_seed_dependent_only():
    r1, r2 = two_view()
    a = estimate_essential_rays(r1, r2, max_angle_rad=1e-4, min_inliers=20, seed=3)
    b = estimate_essential_rays(r1, r2, max_angle_rad=1e-4, min_inliers=20, seed=3)
    npt.assert_array_equal(np.asarray(a["e_matrix"]), np.asarray(b["e_matrix"]))
    npt.assert_array_equal(np.asarray(a["inliers"]), np.asarray(b["inliers"]))


def test_essential_abstains_without_a_consensus():
    r1, r2 = two_view(n=60)
    scrambled = np.ascontiguousarray(np.roll(r2, 17, axis=0))
    assert (
        estimate_essential_rays(r1, scrambled, max_angle_rad=1e-6, min_inliers=40)
        is None
    )


def test_essential_sides_are_selectable():
    r1, r2 = two_view()
    both = estimate_essential_rays(r1, r2, max_angle_rad=3e-3, min_inliers=20)
    one = estimate_essential_rays(
        r1, r2, max_angle_rad=3e-3, min_inliers=20, side="one"
    )
    assert both is not None and one is not None
    assert np.all(
        np.asarray(both["residuals_rad"]) >= np.asarray(one["residuals_rad"]) - 1e-12
    )
    with pytest.raises(ValueError, match="side must be one of"):
        estimate_essential_rays(r1, r2, max_angle_rad=1e-3, side="left")


def test_rotation_recovers_a_pure_ray_rotation():
    r1 = planted_rays(180, 2.2)
    r2 = np.ascontiguousarray(r1 @ ROT.T)
    out = fit_ray_rotation(r1, r2, max_angle_rad=1e-6)
    assert out is not None
    assert np.asarray(out["inliers"]).all()
    assert out["rms_rad"] < 1e-7
    npt.assert_allclose(np.asarray(out["rotation"]), ROT, atol=1e-10)


def test_rotation_abstains_on_a_parallax_rich_pair():
    r1, r2 = two_view()
    assert fit_ray_rotation(r1, r2, max_angle_rad=1e-4, min_inliers=60) is None


def test_shape_and_length_validation():
    r1, r2 = two_view(n=40)
    with pytest.raises(ValueError, match=r"rays1 must have shape"):
        estimate_essential_rays(np.ascontiguousarray(r1[:, :2]), r2, max_angle_rad=1e-3)
    with pytest.raises(ValueError, match="correspondence count mismatch"):
        estimate_essential_rays(r1, np.ascontiguousarray(r2[:20]), max_angle_rad=1e-3)
    zero = r1.copy()
    zero[3] = 0.0
    with pytest.raises(ValueError, match="zero or non-finite ray"):
        fit_ray_rotation(np.ascontiguousarray(zero), r2, max_angle_rad=1e-3)


def test_rays_need_not_arrive_normalized():
    r1, r2 = two_view()
    scaled = np.ascontiguousarray(r1 * np.linspace(0.5, 4.0, len(r1))[:, None])
    a = estimate_essential_rays(r1, r2, max_angle_rad=1e-5, min_inliers=20)
    b = estimate_essential_rays(scaled, r2, max_angle_rad=1e-5, min_inliers=20)
    npt.assert_allclose(
        np.abs(np.asarray(a["e_matrix"])), np.abs(np.asarray(b["e_matrix"])), atol=1e-9
    )
