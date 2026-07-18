# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the structure-free focal-vote Rust bindings
(``sfmtool._sfmtool.geometry.focal_vote`` and ``estimate_homography``; see
``specs/core/focal-vote.md``).

Synthetic scenes are built in the OpenCV/optical pixel convention (a point in
front has camera-frame ``z > 0``). A pure-rotation rig (all camera centres at
the world origin) produces parallax-free pairs that vote through rotation
self-calibration; a baseline camera track over finite structure produces
parallax-rich pairs that vote through the Bougnoux focal of a fundamental
matrix. The kernel arbitrates each scene to the correct family.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import estimate_homography, focal_vote

W, H = 1000, 1000
F_TRUE = 800.0
CX = CY = 500.0


# ── Helpers ────────────────────────────────────────────────────────────────


def _ry(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _k():
    return np.array([[F_TRUE, 0, CX], [0, F_TRUE, CY], [0, 0, 1.0]])


def _project(r, t, x):
    xc = r @ x + t
    if xc[2] <= 1e-3:
        return None
    p = _k() @ xc
    u, v = p[0] / p[2], p[1] / p[2]
    if not (0 <= u < W and 0 <= v < H):
        return None
    return np.array([u, v])


class _Obs:
    def __init__(self):
        self.cluster, self.image, self.pos, self.n = [], [], [], 0

    def push(self, ia, pa, ib, pb):
        c = self.n
        self.n += 1
        self.cluster += [c, c]
        self.image += [ia, ib]
        self.pos += [pa, pb]

    def arrays(self):
        return (
            np.asarray(self.cluster, dtype=np.uint32),
            np.asarray(self.image, dtype=np.uint32),
            np.asarray(self.pos, dtype=np.float64),
        )


def _rotation_cams(n_img, span, rng):
    cams = []
    for i in range(n_img):
        pan = -span + 2 * span * i / (n_img - 1)
        tilt = rng.uniform(-0.02, 0.02)
        cams.append((_rx(tilt) @ _ry(pan), np.zeros(3)))
    return cams


def _baseline_cams(n_img, baseline, rng):
    cams = []
    for i in range(n_img):
        r = _rx(rng.uniform(-0.03, 0.03)) @ _ry(rng.uniform(-0.03, 0.03))
        center = np.array([i * baseline, 0.0, 0.0])
        cams.append((r, -r @ center))
    return cams


def _emit_rotation_pair(obs, cams, ia, ib, m, rng):
    done, guard = 0, 0
    while done < m and guard < m * 200:
        guard += 1
        yaw, pitch = rng.uniform(-0.9, 0.9), rng.uniform(-0.6, 0.6)
        d = np.array([np.sin(yaw), np.sin(pitch), 1.0])
        d = d / np.linalg.norm(d) * 30.0
        pa = _project(*cams[ia], d)
        pb = _project(*cams[ib], d)
        if pa is not None and pb is not None:
            pa = pa + 0.3 * rng.standard_normal(2)
            pb = pb + 0.3 * rng.standard_normal(2)
            obs.push(ia, pa, ib, pb)
            done += 1


def _emit_parallax_pair(obs, cams, ia, ib, m, rng):
    done, guard = 0, 0
    while done < m and guard < m * 200:
        guard += 1
        x = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(4, 9)])
        pa = _project(*cams[ia], x)
        pb = _project(*cams[ib], x)
        if pa is not None and pb is not None:
            pa = pa + 0.3 * rng.standard_normal(2)
            pb = pb + 0.3 * rng.standard_normal(2)
            obs.push(ia, pa, ib, pb)
            done += 1


def _rotation_scene(seed):
    rng = np.random.default_rng(seed)
    n = 8
    cams = _rotation_cams(n, 0.24, rng)
    obs = _Obs()
    for i in range(n - 1):
        _emit_rotation_pair(obs, cams, i, i + 1, 45, rng)
    for i in range(n - 3):
        _emit_rotation_pair(obs, cams, i, i + 3, 45, rng)
    return obs.arrays()


def _parallax_scene(seed):
    rng = np.random.default_rng(seed)
    n = 8
    cams = _baseline_cams(n, 0.35, rng)
    obs = _Obs()
    for i in range(n - 1):
        _emit_parallax_pair(obs, cams, i, i + 1, 45, rng)
    for i in range(n - 2):
        _emit_parallax_pair(obs, cams, i, i + 2, 45, rng)
    return obs.arrays()


# ── estimate_homography ─────────────────────────────────────────────────────


def _random_h(rng):
    return np.array(
        [
            [1 + rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-40, 40)],
            [rng.uniform(-0.2, 0.2), 1 + rng.uniform(-0.2, 0.2), rng.uniform(-40, 40)],
            [rng.uniform(-3e-4, 3e-4), rng.uniform(-3e-4, 3e-4), 1.0],
        ]
    )


def _apply_h(h, p):
    v = h @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]


def test_homography_dict_layout():
    rng = np.random.default_rng(1)
    h_true = _random_h(rng)
    x1 = rng.uniform([0, 0], [640, 480], size=(120, 2))
    x2 = np.array([_apply_h(h_true, p) for p in x1])
    # 20% outliers.
    x2[:24] = rng.uniform([0, 0], [640, 480], size=(24, 2))
    result = estimate_homography(x1, x2, max_error_px=2.0, min_inliers=20, seed=3)
    assert result is not None
    assert set(result) == {"h_matrix", "inliers", "iterations"}
    h = result["h_matrix"]
    assert h.shape == (3, 3)
    assert h.dtype == np.float64
    npt.assert_allclose(np.linalg.norm(h), 1.0, atol=1e-9)  # unit Frobenius
    assert result["inliers"].dtype == bool
    assert result["inliers"].shape == (120,)
    assert isinstance(result["iterations"], int)
    # The planted inliers are recovered (allow a few misses).
    assert int(result["inliers"][24:].sum()) >= 90


def test_homography_shape_validation():
    x = np.zeros((20, 3))
    with pytest.raises(ValueError):
        estimate_homography(x, x)
    with pytest.raises(ValueError):
        estimate_homography(np.zeros((20, 2)), np.zeros((19, 2)))


def test_homography_handles_noncontiguous_input():
    rng = np.random.default_rng(2)
    h_true = _random_h(rng)
    x1 = rng.uniform([0, 0], [640, 480], size=(120, 2))
    x2 = np.array([_apply_h(h_true, p) for p in x1]) + 0.2 * rng.standard_normal(
        (120, 2)
    )
    x1_nc = np.repeat(x1, 2, axis=1)[:, ::2]
    x2_nc = np.repeat(x2, 2, axis=1)[:, ::2]
    assert not x1_nc.flags["C_CONTIGUOUS"]
    a = estimate_homography(x1_nc, x2_nc, seed=1)
    b = estimate_homography(np.ascontiguousarray(x1), np.ascontiguousarray(x2), seed=1)
    assert a is not None and b is not None
    npt.assert_array_equal(a["h_matrix"], b["h_matrix"])


def test_homography_determinism():
    rng = np.random.default_rng(9)
    h_true = _random_h(rng)
    x1 = rng.uniform([0, 0], [640, 480], size=(120, 2))
    x2 = np.array([_apply_h(h_true, p) for p in x1]) + 0.4 * rng.standard_normal(
        (120, 2)
    )
    a = estimate_homography(x1, x2, seed=55)
    b = estimate_homography(x1, x2, seed=55)
    assert a is not None and b is not None
    npt.assert_array_equal(a["h_matrix"], b["h_matrix"])
    npt.assert_array_equal(a["inliers"], b["inliers"])
    assert a["iterations"] == b["iterations"]


# ── focal_vote ──────────────────────────────────────────────────────────────


def test_focal_vote_dict_layout():
    cl, im, pos = _rotation_scene(2024)
    res = focal_vote(cl, im, pos, W, H, seed=0)
    assert set(res) == {
        "focal_px",
        "family",
        "epipolar_focal_px",
        "rotation_focal_px",
        "n_epipolar",
        "n_rotation",
        "parallax_poverty",
    }
    assert isinstance(res["n_epipolar"], int)
    assert isinstance(res["n_rotation"], int)
    assert isinstance(res["parallax_poverty"], float)
    assert res["family"] in ("Epipolar", "Rotation", None)


def test_focal_vote_shape_validation():
    with pytest.raises(ValueError):
        focal_vote(
            np.zeros(10, np.uint32), np.zeros(10, np.uint32), np.zeros((10, 3)), W, H
        )
    with pytest.raises(ValueError):
        focal_vote(
            np.zeros(10, np.uint32), np.zeros(9, np.uint32), np.zeros((10, 2)), W, H
        )
    # Non-monotone cluster ids are rejected.
    with pytest.raises(ValueError):
        focal_vote(
            np.array([0, 2, 1], np.uint32),
            np.array([0, 1, 2], np.uint32),
            np.zeros((3, 2)),
            W,
            H,
        )


def test_focal_vote_rotation_scene():
    cl, im, pos = _rotation_scene(2024)
    res = focal_vote(cl, im, pos, W, H, seed=0)
    assert res["family"] == "Rotation", res
    assert res["focal_px"] is not None
    assert abs(res["focal_px"] - F_TRUE) / F_TRUE < 0.1
    assert res["parallax_poverty"] >= 0.55


def test_focal_vote_parallax_scene():
    cl, im, pos = _parallax_scene(4048)
    res = focal_vote(cl, im, pos, W, H, seed=0)
    assert res["family"] == "Epipolar", res
    assert res["n_epipolar"] >= 8
    assert res["focal_px"] is not None
    assert abs(res["focal_px"] - F_TRUE) / F_TRUE < 0.15
    assert res["parallax_poverty"] < 0.55


def test_focal_vote_seed_reproducibility():
    cl, im, pos = _rotation_scene(2024)
    a = focal_vote(cl, im, pos, W, H, seed=42)
    b = focal_vote(cl, im, pos, W, H, seed=42)
    assert a["focal_px"] == b["focal_px"]
    assert a["family"] == b["family"]
    assert a["n_epipolar"] == b["n_epipolar"]
    assert a["n_rotation"] == b["n_rotation"]
    assert a["parallax_poverty"] == b["parallax_poverty"]


def test_focal_vote_noncontiguous_input():
    cl, im, pos = _rotation_scene(2024)
    pos_nc = np.repeat(pos, 2, axis=1)[:, ::2]
    assert not pos_nc.flags["C_CONTIGUOUS"]
    a = focal_vote(cl, im, pos_nc, W, H, seed=0)
    b = focal_vote(cl, im, np.ascontiguousarray(pos), W, H, seed=0)
    assert a["focal_px"] == b["focal_px"]
    assert a["family"] == b["family"]


def test_focal_vote_empty_input():
    res = focal_vote(
        np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros((0, 2)), W, H
    )
    assert res["focal_px"] is None
    assert res["family"] is None
    assert res["n_epipolar"] == 0
    assert res["n_rotation"] == 0
