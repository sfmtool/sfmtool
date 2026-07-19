# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the far-field rotation initialization Rust binding
(``sfmtool._sfmtool.geometry.rotation_init``; see
``specs/core/rotation-init.md``).

Synthetic scenes are built in the OpenCV/optical pixel convention (a point in
front has camera-frame ``z > 0``): cameras strung along +X panning across a
near cloud (strong parallax) plus a distant cloud whose parallax stays below
the homography gate. The kernel returns canonical world-to-camera poses (the
camera looks along −Z), so ground-truth rotations convert through
``S = diag(1, -1, -1)`` (``R_canonical = S @ R_optical``) for comparisons;
camera centers are convention-free.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import rotation_init

W, H = 1000, 1000
F0 = 800.0
S = np.diag([1.0, -1.0, -1.0])

EXPECTED_KEYS = {
    "image_indexes",
    "quaternions_wxyz",
    "translations",
    "points",
    "inlier_fractions",
    "far_cluster_indexes",
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _ry(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _project(r, c, x):
    """Optical-convention projection of world point x for camera (R, center)."""
    xc = r @ (x - c)
    if xc[2] <= 1e-6:
        return None
    u = F0 * xc[0] / xc[2] + W / 2.0
    v = F0 * xc[1] / xc[2] + H / 2.0
    if not (0 <= u < W and 0 <= v < H):
        return None
    return np.array([u, v])


def _far_field_scene(seed, n_img=10, n_near=130, n_far=160, noise=0.2):
    rng = np.random.default_rng(seed)
    mid = (n_img - 1) / 2.0
    rots = [_rx(rng.uniform(-0.01, 0.01)) @ _ry((i - mid) * 0.04) for i in range(n_img)]
    centers = [
        np.array([(i - mid) * 0.4, rng.uniform(-0.05, 0.05), 0.0]) for i in range(n_img)
    ]

    world = [rng.uniform([-4, -4, 4], [4, 4, 9]) for _ in range(n_near)]
    for _ in range(n_far):
        d = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), 1.0])
        world.append(d / np.linalg.norm(d) * rng.uniform(4000, 6000))

    cluster, image, pos = [], [], []
    far_start = None
    cid = 0
    for w_idx, x in enumerate(world):
        members = []
        for i in range(n_img):
            p = _project(rots[i], centers[i], x)
            if p is not None:
                members.append((i, p))
        if len(members) < 2:
            continue
        if w_idx >= n_near and far_start is None:
            far_start = cid
        for i, p in members:
            cluster.append(cid)
            image.append(i)
            pos.append(p + noise * rng.standard_normal(2))
        cid += 1
    return {
        "cluster": np.asarray(cluster, np.uint32),
        "image": np.asarray(image, np.uint32),
        "pos": np.asarray(pos, np.float64),
        "rots": rots,
        "centers": centers,
        "far_start": far_start,
    }


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _aligned_rotation_errors_deg(gt, est):
    acc = np.zeros((3, 3))
    for g, e in zip(gt, est):
        acc += g.T @ e
    u, _s, vt = np.linalg.svd(acc)
    gauge = u @ vt
    if np.linalg.det(gauge) < 0:
        gauge = -gauge
    errs = []
    for g, e in zip(gt, est):
        rel = e @ (g @ gauge).T
        errs.append(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
    return np.asarray(errs)


# ── Result layout and quality ──────────────────────────────────────────────


def test_dict_layout_and_rotations():
    sc = _far_field_scene(11)
    res = rotation_init(sc["cluster"], sc["image"], sc["pos"], W, H, F0, seed=0)
    assert res is not None
    assert set(res) == EXPECTED_KEYS
    n_posed = res["image_indexes"].shape[0]
    assert n_posed >= 8
    assert res["image_indexes"].dtype == np.uint32
    assert res["quaternions_wxyz"].shape == (n_posed, 4)
    assert res["translations"].shape == (n_posed, 3)
    n_cl = int(sc["cluster"].max()) + 1
    assert res["points"].shape == (n_cl, 3)
    assert res["inlier_fractions"].shape == (n_posed,)
    assert res["far_cluster_indexes"].dtype == np.uint32

    # Sub-degree rotations after averaging: canonical GT is S @ R_optical.
    gt = [S @ sc["rots"][i] for i in res["image_indexes"]]
    est = [_quat_to_mat(q) for q in res["quaternions_wxyz"]]
    errs = _aligned_rotation_errors_deg(gt, est)
    assert errs.max() < 1.0, f"rotation errors {errs}"


def test_translations_match_up_to_similarity():
    sc = _far_field_scene(11)
    res = rotation_init(sc["cluster"], sc["image"], sc["pos"], W, H, F0, seed=0)
    assert res is not None
    est_c = np.array(
        [
            -(_quat_to_mat(q).T @ t)
            for q, t in zip(res["quaternions_wxyz"], res["translations"])
        ]
    )
    gt_c = np.array([sc["centers"][i] for i in res["image_indexes"]])
    xc = est_c - est_c.mean(0)
    yc = gt_c - gt_c.mean(0)
    cov = yc.T @ xc
    u, _s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = -r
    scale = np.trace(r.T @ cov) / max(np.einsum("ij,ij", xc, xc), 1e-300)
    resid = np.linalg.norm(yc - scale * xc @ r.T, axis=1)
    spread = np.sqrt((yc**2).sum(1).mean())
    assert resid.max() < 0.05 * spread, f"residuals {resid} vs spread {spread}"

    # Far cluster ids are dominated by the far cloud, and their point rows are
    # unit directions (the finishing adjustment models them at infinity).
    far = res["far_cluster_indexes"]
    assert far.size > 0
    assert (far >= sc["far_start"]).mean() > 0.8
    far_rows = res["points"][far]
    finite = np.isfinite(far_rows[:, 0])
    norms = np.linalg.norm(far_rows[finite], axis=1)
    npt.assert_allclose(norms, 1.0, atol=1e-9)


# ── Failure modes and budgets ──────────────────────────────────────────────


def test_min_images_returns_none():
    sc = _far_field_scene(11)
    assert (
        rotation_init(
            sc["cluster"], sc["image"], sc["pos"], W, H, F0, seed=0, min_images=20
        )
        is None
    )


def test_max_images_caps_growth():
    sc = _far_field_scene(11)
    res = rotation_init(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        W,
        H,
        F0,
        seed=0,
        min_images=6,
        max_images=6,
    )
    assert res is not None
    assert res["image_indexes"].shape[0] == 6


def test_all_parallax_scene_returns_none():
    rng = np.random.default_rng(5)
    n_img = 10
    mid = (n_img - 1) / 2.0
    rots = [_rx(rng.uniform(-0.02, 0.02)) @ _ry((i - mid) * 0.05) for i in range(n_img)]
    centers = [
        np.array([(i - mid) * 0.8, rng.uniform(-0.1, 0.1), 0.0]) for i in range(n_img)
    ]
    cluster, image, pos = [], [], []
    cid = 0
    for _ in range(250):
        x = rng.uniform([-4, -4, 3], [4, 4, 8])
        members = []
        for i in range(n_img):
            p = _project(rots[i], centers[i], x)
            if p is not None:
                members.append((i, p))
        if len(members) < 2:
            continue
        for i, p in members:
            cluster.append(cid)
            image.append(i)
            pos.append(p + 0.2 * rng.standard_normal(2))
        cid += 1
    assert (
        rotation_init(
            np.asarray(cluster, np.uint32),
            np.asarray(image, np.uint32),
            np.asarray(pos, np.float64),
            W,
            H,
            F0,
            seed=0,
        )
        is None
    )


# ── Determinism, layout guards, validation ─────────────────────────────────


def test_determinism_same_seed():
    sc = _far_field_scene(11)
    a = rotation_init(sc["cluster"], sc["image"], sc["pos"], W, H, F0, seed=42)
    b = rotation_init(sc["cluster"], sc["image"], sc["pos"], W, H, F0, seed=42)
    assert a is not None and b is not None
    npt.assert_array_equal(a["image_indexes"], b["image_indexes"])
    npt.assert_array_equal(a["quaternions_wxyz"], b["quaternions_wxyz"])
    npt.assert_array_equal(a["translations"], b["translations"])
    npt.assert_array_equal(a["points"], b["points"])
    npt.assert_array_equal(a["inlier_fractions"], b["inlier_fractions"])
    npt.assert_array_equal(a["far_cluster_indexes"], b["far_cluster_indexes"])


def test_fortran_order_positions_match_c_order():
    sc = _far_field_scene(11)
    pos_f = np.asfortranarray(sc["pos"])
    assert not pos_f.flags["C_CONTIGUOUS"]
    a = rotation_init(sc["cluster"], sc["image"], pos_f, W, H, F0, seed=0)
    b = rotation_init(
        sc["cluster"], sc["image"], np.ascontiguousarray(sc["pos"]), W, H, F0, seed=0
    )
    assert a is not None and b is not None
    npt.assert_array_equal(a["image_indexes"], b["image_indexes"])
    npt.assert_array_equal(a["quaternions_wxyz"], b["quaternions_wxyz"])
    npt.assert_array_equal(a["translations"], b["translations"])
    npt.assert_array_equal(a["points"], b["points"])


def test_shape_validation():
    with pytest.raises(ValueError):
        rotation_init(
            np.zeros(10, np.uint32),
            np.zeros(10, np.uint32),
            np.zeros((10, 3)),
            W,
            H,
            F0,
        )
    with pytest.raises(ValueError):
        rotation_init(
            np.zeros(10, np.uint32), np.zeros(9, np.uint32), np.zeros((10, 2)), W, H, F0
        )
    # Non-monotone cluster ids are rejected.
    with pytest.raises(ValueError):
        rotation_init(
            np.array([0, 2, 1], np.uint32),
            np.array([0, 1, 2], np.uint32),
            np.zeros((3, 2)),
            W,
            H,
            F0,
        )
