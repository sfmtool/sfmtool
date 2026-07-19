# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reconstruction growth Rust bindings
(``sfmtool._sfmtool.geometry.grow_reconstruction`` /
``resect_images_batch``; see ``specs/core/reconstruction-growth.md``).

Synthetic orbit scenes in the canonical camera frame (the camera looks along
-Z): cameras on a circle looking at the origin, world points on a jittered
cylinder whose visibility is limited to front-facing cameras, so covisibility
is local and growth genuinely propagates around the orbit from a small seed.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import (
    CameraIntrinsics,
    grow_reconstruction,
    resect_images_batch,
)

W, H = 800, 800
F0 = 700.0

GROW_KEYS = {
    "quaternions_wxyz",
    "translations",
    "posed",
    "points",
    "focal",
    "residual_norms",
}
RESECT_KEYS = {"quaternions_wxyz", "translations", "inlier_fractions", "accepted"}


def _make_cam(f=F0):
    return CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": W,
            "height": H,
            "parameters": {
                "focal_length": float(f),
                "principal_point_x": W / 2.0,
                "principal_point_y": H / 2.0,
            },
        }
    )


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_quat(m):
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = np.copysign(
        np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2]
    )
    y = np.copysign(
        np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0]
    )
    z = np.copysign(
        np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1]
    )
    return np.array([w, x, y, z])


def _orbit_scene(seed, n_img=12, n_pts=200, noise=0.2, vis_cos=0.4):
    rng = np.random.default_rng(seed)
    r_orbit = 10.0
    thetas = 2 * np.pi * np.arange(n_img) / n_img
    rots, centers = [], []
    for th in thetas:
        c = np.array(
            [r_orbit * np.sin(th), rng.uniform(-0.3, 0.3), r_orbit * np.cos(th)]
        )
        target = rng.uniform(-0.2, 0.2, 3)
        z_cam = c - target
        z_cam /= np.linalg.norm(z_cam)
        x_cam = np.cross([0.0, 1.0, 0.0], z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        rots.append(np.stack([x_cam, y_cam, z_cam]))
        centers.append(c)

    cluster, image, pos, world = [], [], [], []
    cid = 0
    for _ in range(n_pts):
        phi = rng.uniform(0, 2 * np.pi)
        r_cyl = 4.0 + rng.uniform(-1.0, 1.0)
        x = np.array([r_cyl * np.sin(phi), rng.uniform(-3, 3), r_cyl * np.cos(phi)])
        members = []
        for i in range(n_img):
            if np.cos(thetas[i] - phi) <= vis_cos:
                continue
            pc = rots[i] @ (x - centers[i])
            if pc[2] >= -1e-6:
                continue
            u = F0 * pc[0] / -pc[2] + W / 2.0
            v = F0 * -pc[1] / -pc[2] + H / 2.0
            if not (0 <= u < W and 0 <= v < H):
                continue
            members.append((i, [u, v]))
        if len(members) < 2:
            continue
        for i, p in members:
            cluster.append(cid)
            image.append(i)
            pos.append(p + noise * rng.standard_normal(2))
        world.append(x)
        cid += 1
    return {
        "cluster": np.asarray(cluster, np.uint32),
        "image": np.asarray(image, np.uint32),
        "pos": np.asarray(pos, np.float64),
        "rots": rots,
        "centers": centers,
        "world": np.asarray(world, np.float64),
    }


def _seed_arrays(sc, n=3):
    q = np.stack([_mat_to_quat(sc["rots"][i]) for i in range(n)])
    t = np.stack([-(sc["rots"][i] @ sc["centers"][i]) for i in range(n)])
    return (
        np.ascontiguousarray(q),
        np.ascontiguousarray(t),
        np.arange(n, dtype=np.uint32),
    )


def _grow(sc, cam=None, n_seed=3, **kwargs):
    q, t, idx = _seed_arrays(sc, n_seed)
    return grow_reconstruction(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        cam if cam is not None else _make_cam(),
        q,
        t,
        idx,
        **kwargs,
    )


# ── grow_reconstruction ────────────────────────────────────────────────────


def test_grow_dict_layout_and_full_registration():
    sc = _orbit_scene(3)
    res = _grow(sc)
    assert set(res) == GROW_KEYS
    n_img = 12
    n_cl = int(sc["cluster"].max()) + 1
    assert res["quaternions_wxyz"].shape == (n_img, 4)
    assert res["translations"].shape == (n_img, 3)
    assert res["posed"].shape == (n_img,)
    assert res["posed"].dtype == np.bool_
    assert res["points"].shape == (n_cl, 3)
    assert res["residual_norms"].shape == (len(sc["cluster"]),)
    assert isinstance(res["focal"], float)

    # A 3-image seed grows the whole orbit.
    assert res["posed"].all()
    assert abs(res["focal"] - F0) / F0 < 0.02

    # Camera centers match ground truth (canonical world-to-camera poses).
    for i in range(n_img):
        r = _quat_to_mat(res["quaternions_wxyz"][i])
        c_est = -(r.T @ res["translations"][i])
        assert np.linalg.norm(c_est - sc["centers"][i]) < 0.2, f"image {i}"

    # Residuals at the final state are tight for most observations.
    finite = res["residual_norms"][np.isfinite(res["residual_norms"])]
    assert len(finite) >= 0.9 * len(sc["cluster"])
    assert np.median(finite) < 1.0


def test_grow_determinism():
    sc = _orbit_scene(3)
    a = _grow(sc, seed=7)
    b = _grow(sc, seed=7)
    npt.assert_array_equal(a["quaternions_wxyz"], b["quaternions_wxyz"])
    npt.assert_array_equal(a["translations"], b["translations"])
    npt.assert_array_equal(a["posed"], b["posed"])
    npt.assert_array_equal(a["points"], b["points"])
    npt.assert_array_equal(a["residual_norms"], b["residual_norms"])
    assert a["focal"] == b["focal"]


def test_grow_ba_window_at_posed_count_matches_unbounded():
    sc = _orbit_scene(3)
    unbounded = _grow(sc)
    windowed = _grow(sc, ba_window=12)
    npt.assert_array_equal(unbounded["quaternions_wxyz"], windowed["quaternions_wxyz"])
    npt.assert_array_equal(unbounded["translations"], windowed["translations"])
    npt.assert_array_equal(unbounded["points"], windowed["points"])
    assert unbounded["focal"] == windowed["focal"]


def test_grow_bounded_window_and_anchor_still_register():
    sc = _orbit_scene(3)
    res = _grow(sc, ba_window=4, anchor_every=3)
    assert res["posed"].all()
    for i in range(12):
        r = _quat_to_mat(res["quaternions_wxyz"][i])
        c_est = -(r.T @ res["translations"][i])
        assert np.linalg.norm(c_est - sc["centers"][i]) < 0.4, f"image {i}"


def test_grow_degenerate_no_seed_returns_input_state():
    sc = _orbit_scene(3)
    empty_q = np.zeros((0, 4))
    empty_t = np.zeros((0, 3))
    empty_i = np.zeros(0, np.uint32)
    res = grow_reconstruction(
        sc["cluster"], sc["image"], sc["pos"], _make_cam(), empty_q, empty_t, empty_i
    )
    assert not res["posed"].any()
    assert np.isnan(res["points"]).all()
    assert np.isinf(res["residual_norms"]).all()
    assert res["focal"] == F0


def test_grow_degenerate_min_obs_returns_seed_state():
    sc = _orbit_scene(3)
    q, t, idx = _seed_arrays(sc, 3)
    res = _grow(sc, min_obs=10**6)
    assert res["posed"].sum() == 3
    npt.assert_allclose(res["quaternions_wxyz"][:3], q, atol=1e-14)
    npt.assert_array_equal(res["translations"][:3], t)
    assert res["focal"] == F0


def test_grow_shape_validation():
    sc = _orbit_scene(3)
    q, t, idx = _seed_arrays(sc, 3)
    cam = _make_cam()
    with pytest.raises(ValueError):
        grow_reconstruction(sc["cluster"], sc["image"][:-1], sc["pos"], cam, q, t, idx)
    with pytest.raises(ValueError):
        grow_reconstruction(
            sc["cluster"],
            sc["image"],
            np.zeros((len(sc["cluster"]), 3)),
            cam,
            q,
            t,
            idx,
        )
    with pytest.raises(ValueError):
        grow_reconstruction(sc["cluster"], sc["image"], sc["pos"], cam, q, t, idx[:2])
    # Non-monotone cluster ids are rejected.
    bad_cluster = sc["cluster"].copy()
    bad_cluster[0] = bad_cluster.max()
    with pytest.raises(ValueError):
        grow_reconstruction(bad_cluster, sc["image"], sc["pos"], cam, q, t, idx)


# ── resect_images_batch ────────────────────────────────────────────────────


def test_resect_batch_registers_against_gt_structure():
    sc = _orbit_scene(3)
    q, t, idx = _seed_arrays(sc, 3)
    image_list = np.arange(12, dtype=np.uint32)
    res = resect_images_batch(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        _make_cam(),
        np.ascontiguousarray(sc["world"]),
        image_list,
        posed_quaternions_wxyz=q,
        posed_translations=t,
        posed_indexes=idx,
    )
    assert set(res) == RESECT_KEYS
    assert res["accepted"].shape == (12,)
    assert res["accepted"].dtype == np.bool_
    assert res["accepted"].all()
    assert (res["inlier_fractions"] > 0.9).all()
    for k in range(12):
        r = _quat_to_mat(res["quaternions_wxyz"][k])
        c_est = -(r.T @ res["translations"][k])
        assert np.linalg.norm(c_est - sc["centers"][k]) < 0.05, f"image {k}"


def test_resect_batch_deterministic_and_matches_single():
    sc = _orbit_scene(3)
    image_list = np.arange(12, dtype=np.uint32)
    run = lambda lst: resect_images_batch(  # noqa: E731
        sc["cluster"],
        sc["image"],
        sc["pos"],
        _make_cam(),
        np.ascontiguousarray(sc["world"]),
        np.asarray(lst, np.uint32),
        seed=9,
    )
    a = run(image_list)
    b = run(image_list)
    npt.assert_array_equal(a["quaternions_wxyz"], b["quaternions_wxyz"])
    npt.assert_array_equal(a["translations"], b["translations"])
    npt.assert_array_equal(a["inlier_fractions"], b["inlier_fractions"])
    npt.assert_array_equal(a["accepted"], b["accepted"])

    # Per-image RANSAC seeding is a pure function of (seed, image index), so
    # a one-image call matches its batch row exactly.
    for k in (0, 5, 11):
        single = run([k])
        npt.assert_array_equal(single["quaternions_wxyz"][0], a["quaternions_wxyz"][k])
        npt.assert_array_equal(single["translations"][0], a["translations"][k])
        assert single["inlier_fractions"][0] == a["inlier_fractions"][k]
        assert single["accepted"][0] == a["accepted"][k]


def test_resect_batch_gates_and_degenerate():
    sc = _orbit_scene(3)
    # Corrupt one image's observations entirely.
    rng = np.random.default_rng(0)
    junk = sc["pos"].copy()
    mask = sc["image"] == 7
    junk[mask] = rng.uniform(0, W, size=(mask.sum(), 2))
    res = resect_images_batch(
        sc["cluster"],
        sc["image"],
        junk,
        _make_cam(),
        np.ascontiguousarray(sc["world"]),
        np.asarray([6, 7], np.uint32),
    )
    assert res["accepted"][0]
    assert not res["accepted"][1]

    # min_obs above every candidate count skips the image with the identity
    # pose.
    skipped = resect_images_batch(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        _make_cam(),
        np.ascontiguousarray(sc["world"]),
        np.asarray([6], np.uint32),
        min_obs=10**6,
    )
    assert not skipped["accepted"][0]
    npt.assert_array_equal(skipped["quaternions_wxyz"][0], [1.0, 0.0, 0.0, 0.0])

    # All-NaN structure: nothing to resect against.
    none = resect_images_batch(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        _make_cam(),
        np.full_like(sc["world"], np.nan),
        np.asarray([6], np.uint32),
    )
    assert not none["accepted"][0]


def test_resect_batch_validation():
    sc = _orbit_scene(3)
    cam = _make_cam()
    pts = np.ascontiguousarray(sc["world"])
    lst = np.asarray([0], np.uint32)
    # Points row count below the cluster id range.
    with pytest.raises(ValueError):
        resect_images_batch(sc["cluster"], sc["image"], sc["pos"], cam, pts[:1], lst)
    # Posed pose arrays must come together.
    with pytest.raises(ValueError):
        resect_images_batch(
            sc["cluster"],
            sc["image"],
            sc["pos"],
            cam,
            pts,
            lst,
            posed_indexes=np.asarray([0], np.uint32),
        )
    # Points must be (n, 3).
    with pytest.raises(ValueError):
        resect_images_batch(
            sc["cluster"], sc["image"], sc["pos"], cam, pts[:, :2].copy(), lst
        )
