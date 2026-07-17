# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the epipolar-estimation Rust bindings
(``sfmtool._sfmtool.geometry.estimate_fundamental`` and
``focal_from_fundamental``; see ``specs/core/epipolar-estimation.md``).

Synthetic camera pairs are generated in the OpenCV/optical pixel convention
(a point in front has camera-frame ``z > 0``), so the ground-truth fundamental
matrix ``F`` satisfies ``x2^T F x1 = 0`` for the pixel correspondences —
exactly what the solver estimates and what OpenCV's ``findFundamentalMat``
expects. Estimation runs against the release-built extension, so the
low-inlier-fraction sweep down to 0.2 is fast here.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import (
    estimate_fundamental,
    focal_from_fundamental,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _k(f, cx, cy):
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def _random_rotation(rng, max_angle=0.6):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.05, max_angle)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _fundamental(k1, r1, t1, k2, r2, t2):
    r_rel = r2 @ r1.T
    t_rel = t2 - r_rel @ t1
    tx = np.array(
        [[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]]
    )
    e = tx @ r_rel
    return np.linalg.inv(k2).T @ e @ np.linalg.inv(k1)


def _make_pair(seed, n, f1=650.0, f2=650.0, noise_px=0.0, outlier_frac=0.0):
    """Ground-truth F, principal points, and `n` pixel correspondences."""
    rng = np.random.default_rng(seed)
    pp1 = np.array([320.0, 240.0])
    pp2 = np.array([300.0, 260.0])
    k1 = _k(f1, *pp1)
    k2 = _k(f2, *pp2)
    r1 = _random_rotation(rng)
    t1 = rng.uniform(-0.3, 0.3, size=3)
    r2 = _random_rotation(rng)
    t2 = np.array(
        [rng.uniform(0.5, 1.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)]
    )
    f_true = _fundamental(k1, r1, t1, k2, r2, t2)

    n_out = int(round(n * outlier_frac))
    n_in = n - n_out
    x1, x2, inlier = [], [], []
    while len(x1) < n_in:
        cam1 = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(2.0, 8.0)])
        xw = r1.T @ (cam1 - t1)
        cam2 = r2 @ xw + t2
        if cam2[2] <= 0.2:
            continue
        p1 = k1 @ cam1
        p2 = k2 @ cam2
        u1 = p1[:2] / p1[2]
        u2 = p2[:2] / p2[2]
        if noise_px > 0.0:
            u1 = u1 + noise_px * rng.standard_normal(2)
            u2 = u2 + noise_px * rng.standard_normal(2)
        x1.append(u1)
        x2.append(u2)
        inlier.append(True)
    for _ in range(n_out):
        x1.append(rng.uniform([0, 0], [640, 480]))
        x2.append(rng.uniform([0, 0], [640, 480]))
        inlier.append(False)
    return (
        f_true,
        pp1,
        pp2,
        np.asarray(x1),
        np.asarray(x2),
        np.asarray(inlier, dtype=bool),
    )


def _sampson(f, x1, x2):
    """Per-correspondence squared Sampson distance."""
    p1 = np.hstack([x1, np.ones((len(x1), 1))])
    p2 = np.hstack([x2, np.ones((len(x2), 1))])
    fx1 = p1 @ f.T
    ftx2 = p2 @ f
    num = np.einsum("ij,ij->i", p2, fx1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-300)


def _scale_diff(a, b):
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)
    return min(np.linalg.norm(an - bn), np.linalg.norm(an + bn))


# ── Dict layout, dtype/shape validation, contiguity ─────────────────────────


def test_estimate_dict_layout():
    f_true, _pp1, _pp2, x1, x2, _inl = _make_pair(1, 120, outlier_frac=0.2)
    result = estimate_fundamental(x1, x2, seed=3)
    assert result is not None
    assert set(result) == {"f_matrix", "inliers", "iterations"}
    f = result["f_matrix"]
    assert f.shape == (3, 3)
    assert f.dtype == np.float64
    npt.assert_allclose(np.linalg.norm(f), 1.0, atol=1e-9)  # unit Frobenius
    # Rank 2: smallest singular value ~ 0.
    s = np.linalg.svd(f, compute_uv=False)
    assert s[2] / s[0] < 1e-8
    assert result["inliers"].dtype == bool
    assert result["inliers"].shape == (120,)
    assert isinstance(result["iterations"], int)
    assert _scale_diff(f, f_true) < 1e-2


def test_estimate_shape_validation():
    x = np.zeros((20, 3))  # wrong second dim
    with pytest.raises(ValueError):
        estimate_fundamental(x, x)
    with pytest.raises(ValueError):
        estimate_fundamental(np.zeros((20, 2)), np.zeros((19, 2)))  # length mismatch


def test_estimate_handles_noncontiguous_input():
    _f, _pp1, _pp2, x1, x2, _inl = _make_pair(2, 120, noise_px=0.3, outlier_frac=0.2)
    # A non-contiguous view (every column doubled then sliced).
    x1_nc = np.repeat(x1, 2, axis=1)[:, ::2]
    x2_nc = np.repeat(x2, 2, axis=1)[:, ::2]
    assert not x1_nc.flags["C_CONTIGUOUS"]
    a = estimate_fundamental(x1_nc, x2_nc, seed=1)
    b = estimate_fundamental(np.ascontiguousarray(x1), np.ascontiguousarray(x2), seed=1)
    assert a is not None and b is not None
    npt.assert_array_equal(a["f_matrix"], b["f_matrix"])


# ── Contamination sweep (release build → 0.2 floor is fast) ─────────────────


@pytest.mark.parametrize("inlier_frac", [0.9, 0.7, 0.5, 0.3, 0.2])
def test_contamination_sweep(inlier_frac):
    f_true, _pp1, _pp2, x1, x2, inl = _make_pair(7, 200, outlier_frac=1.0 - inlier_frac)
    result = estimate_fundamental(
        x1,
        x2,
        max_error_px=1.5,
        min_inliers=12,
        max_iterations=2_000_000,
        seed=5,
    )
    assert result is not None, f"frac {inlier_frac}: no estimate"
    assert _scale_diff(result["f_matrix"], f_true) < 5e-2
    # The true inliers have small Sampson residual under the estimated F.
    resid = _sampson(result["f_matrix"], x1[inl], x2[inl])
    assert np.median(resid) < 1.5**2


def test_below_min_inliers_returns_none():
    _f, _pp1, _pp2, x1, x2, _inl = _make_pair(3, 150, outlier_frac=0.9)
    result = estimate_fundamental(x1, x2, min_inliers=40, max_iterations=3000, seed=1)
    assert result is None


def test_none_on_garbage():
    rng = np.random.default_rng(0)
    x1 = rng.uniform([0, 0], [640, 480], size=(100, 2))
    x2 = rng.uniform([0, 0], [640, 480], size=(100, 2))
    result = estimate_fundamental(
        x1, x2, max_error_px=1.0, min_inliers=30, max_iterations=3000, seed=0
    )
    assert result is None


def test_determinism_same_seed():
    _f, _pp1, _pp2, x1, x2, _inl = _make_pair(9, 150, noise_px=0.4, outlier_frac=0.4)
    kwargs = dict(max_error_px=1.5, min_inliers=12, max_iterations=40_000, seed=42)
    a = estimate_fundamental(x1, x2, **kwargs)
    b = estimate_fundamental(x1, x2, **kwargs)
    assert a is not None and b is not None
    npt.assert_array_equal(a["f_matrix"], b["f_matrix"])
    npt.assert_array_equal(a["inliers"], b["inliers"])
    assert a["iterations"] == b["iterations"]


# ── Focal length (Bougnoux) ─────────────────────────────────────────────────


def test_focal_exact_recovery():
    recovered = 0
    for seed in range(50):
        f1 = 400.0 + seed * 4.0
        f2 = 900.0 - seed * 3.0
        f_true, pp1, pp2, *_ = _make_pair(seed + 100, 20, f1=f1, f2=f2)
        f = focal_from_fundamental(f_true, pp1, pp2)
        if f is not None:
            assert abs(f - f1) / f1 < 1e-6, f"seed {seed}: got {f}, want {f1}"
            recovered += 1
    assert recovered > 45


def test_focal_shape_validation():
    with pytest.raises(ValueError):
        focal_from_fundamental(np.zeros((2, 2)), [0.0, 0.0], [0.0, 0.0])


def test_focal_noisy_median_within_tolerance():
    f1 = 620.0
    focals = []
    for seed in range(40):
        f_true, pp1, pp2, x1, x2, _inl = _make_pair(
            seed + 400, 150, f1=f1, f2=f1, noise_px=0.5, outlier_frac=0.2
        )
        result = estimate_fundamental(
            x1, x2, max_error_px=2.0, min_inliers=20, max_iterations=50_000, seed=1
        )
        if result is None:
            continue
        f = focal_from_fundamental(result["f_matrix"], pp1, pp2)
        if f is not None:
            focals.append(f)
    assert len(focals) > 20
    median = float(np.median(focals))
    assert abs(median - f1) / f1 < 0.05, f"median {median}, true {f1}"


def test_focal_degenerate_returns_none():
    # Rotation-only motion (shared camera center) → F ≈ 0 → None.
    k1 = _k(600.0, 320.0, 240.0)
    k2 = _k(650.0, 300.0, 260.0)
    rng = np.random.default_rng(1)
    center = np.array([0.2, -0.1, 0.3])
    r1 = _random_rotation(rng)
    r2 = _random_rotation(rng)
    f_rot = _fundamental(k1, r1, -r1 @ center, k2, r2, -r2 @ center)
    assert focal_from_fundamental(f_rot, [320.0, 240.0], [300.0, 260.0]) is None

    # Fixating cameras: optical axes meet at a common target.
    def look_at(eye, target):
        z = target - eye
        z = z / np.linalg.norm(z)
        a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = a - z * (a @ z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.vstack([x, y, z])

    target = np.array([0.0, 0.0, 5.0])
    e1 = np.array([-1.0, 0.0, 0.0])
    e2 = np.array([1.0, 0.2, 0.0])
    rf1 = look_at(e1, target)
    rf2 = look_at(e2, target)
    f_fix = _fundamental(k1, rf1, -rf1 @ e1, k2, rf2, -rf2 @ e2)
    assert focal_from_fundamental(f_fix, [320.0, 240.0], [300.0, 260.0]) is None


# ── Differential agreement with OpenCV ──────────────────────────────────────


def test_differential_against_opencv():
    cv2 = pytest.importorskip("cv2")
    f_true, _pp1, _pp2, x1, x2, inl = _make_pair(
        13, 200, noise_px=0.3, outlier_frac=0.3
    )
    ours = estimate_fundamental(
        x1, x2, max_error_px=2.0, min_inliers=15, max_iterations=200_000, seed=1
    )
    assert ours is not None

    f_cv, mask_cv = cv2.findFundamentalMat(
        x1.astype(np.float64),
        x2.astype(np.float64),
        cv2.FM_RANSAC,
        2.0,
        0.999,
    )
    if f_cv is None or mask_cv is None:
        pytest.skip("OpenCV failed to estimate a fundamental matrix")

    n_ours = int(ours["inliers"].sum())
    n_cv = int(mask_cv.ravel().astype(bool).sum())
    # cv2 is nondeterministic; ours is not — consensus sizes within variation.
    assert abs(n_ours - n_cv) <= 0.25 * max(n_ours, n_cv), (
        f"consensus sizes differ: ours {n_ours}, cv2 {n_cv}"
    )

    # Inlier Sampson residuals are comparable on the true inliers.
    resid_ours = np.median(_sampson(ours["f_matrix"], x1[inl], x2[inl]))
    resid_cv = np.median(_sampson(f_cv, x1[inl], x2[inl]))
    assert resid_ours < 4.0
    assert abs(resid_ours - resid_cv) < 4.0
