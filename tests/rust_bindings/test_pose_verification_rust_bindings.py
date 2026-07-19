# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pose-verification Rust bindings
(``sfmtool._sfmtool.geometry.verify_poses`` / ``repair_poses``; see
``specs/core/pose-verification.md``).

Synthetic *station* scenes in the canonical camera frame (the camera looks
along -Z): groups of cameras share each orbit position with small rotation
offsets, so same-station pairs are pure rotations — genuine near-duplicate
viewpoints where screen B's conjugate-homography model holds — while
cross-station pairs carry real parallax. The displacement-neighborhood
substrate travels as the compact pair arrays produced by
``ClusterCovisibility.neighborhood_arrays()``.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import (
    CameraIntrinsics,
    repair_poses,
    verify_poses,
)
from sfmtool._sfmtool.matching import ClusterCovisibility

W, H = 800, 800
F0 = 700.0

VERIFY_KEYS = {
    "resect_flags",
    "resect_inlier_fractions",
    "rotation_flags",
    "rotation_scores_deg",
    "flagged",
}
REPAIR_KEYS = VERIFY_KEYS | {
    "quaternions_wxyz",
    "translations",
    "repaired",
    "inlier_before",
    "inlier_after",
}


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


def _rot_from_rotvec(v):
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    kx = np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64
    )
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * (kx @ kx)


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


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _station_scene(seed, n_stations=4, per_station=5, n_pts=400, noise=0.2, vis_cos=0.4):
    rng = np.random.default_rng(seed)
    r_orbit = 10.0
    n_img = n_stations * per_station
    thetas = 2 * np.pi * np.arange(n_stations) / n_stations
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
        r_base = np.stack([x_cam, y_cam, z_cam])
        for m in range(per_station):
            yaw = np.deg2rad((m - (per_station - 1) / 2.0) * 2.0)
            pitch = np.deg2rad(rng.uniform(-1.0, 1.0))
            rots.append(_rot_from_rotvec([pitch, yaw, 0.0]) @ r_base)
            centers.append(c)

    cluster, image, pos, world = [], [], [], []
    cid = 0
    for _ in range(n_pts):
        phi = rng.uniform(0, 2 * np.pi)
        r_cyl = 4.0 + rng.uniform(-1.0, 1.0)
        x = np.array([r_cyl * np.sin(phi), rng.uniform(-3, 3), r_cyl * np.cos(phi)])
        members = []
        for i in range(n_img):
            if np.cos(thetas[i // per_station] - phi) <= vis_cos:
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
        "pos": np.ascontiguousarray(pos, np.float64),
        "rots": rots,
        "centers": centers,
        "world": np.ascontiguousarray(world, np.float64),
        "n_img": n_img,
    }


def _pose_arrays(sc):
    q = np.stack([_mat_to_quat(r) for r in sc["rots"]])
    t = np.stack([-(r @ c) for r, c in zip(sc["rots"], sc["centers"])])
    return (
        np.ascontiguousarray(q),
        np.ascontiguousarray(t),
        np.arange(sc["n_img"], dtype=np.uint32),
    )


def _neighborhood_arrays(sc):
    """The substrate's compact serialization, via the covisibility pyclass."""
    counts = np.bincount(sc["cluster"], minlength=int(sc["cluster"].max()) + 1)
    starts = np.zeros(len(counts) + 1, np.uint32)
    starts[1:] = np.cumsum(counts)
    covis = ClusterCovisibility.from_arrays(
        starts, sc["image"], sc["n_img"], positions_xy=sc["pos"]
    )
    arrs = covis.neighborhood_arrays()
    return arrs["i"], arrs["j"], arrs["count"], arrs["mean_disp"]


def _verify(sc, **kwargs):
    q, t, idx = _pose_arrays(sc)
    pi, pj, pc, pd = _neighborhood_arrays(sc)
    return verify_poses(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        _make_cam(),
        sc["world"],
        q,
        t,
        idx,
        pi,
        pj,
        pc,
        pd,
        **kwargs,
    )


def _repair(sc, **kwargs):
    q, t, idx = _pose_arrays(sc)
    pi, pj, pc, pd = _neighborhood_arrays(sc)
    return repair_poses(
        sc["cluster"],
        sc["image"],
        sc["pos"],
        _make_cam(),
        sc["world"],
        q,
        t,
        idx,
        pi,
        pj,
        pc,
        pd,
        **kwargs,
    )


def _corrupt_pose(sc, img, angle_deg=8.0):
    sc["rots"][img] = _rot_from_rotvec([0.0, np.deg2rad(angle_deg), 0.0]) @ sc["rots"][
        img
    ]
    sc["centers"][img] = sc["centers"][img] + np.array([0.5, 0.3, -0.4])


# ── verify_poses ───────────────────────────────────────────────────────────


def test_verify_dict_layout_and_clean_scene_yields_no_flags():
    sc = _station_scene(3)
    res = _verify(sc)
    assert set(res) == VERIFY_KEYS
    n = sc["n_img"]
    for key in ("resect_flags", "rotation_flags", "flagged"):
        assert res[key].shape == (n,)
        assert res[key].dtype == np.bool_
        assert not res[key].any(), key
    assert res["resect_inlier_fractions"].shape == (n,)
    assert (res["resect_inlier_fractions"] > 0.9).all()
    # Every camera has four same-station neighbours: screen B measures
    # everywhere and the pure-rotation homographies agree with the poses.
    assert np.isfinite(res["rotation_scores_deg"]).all()
    assert (res["rotation_scores_deg"] < 1.0).all()


def test_verify_flags_wrong_pose_via_rotation_screen():
    sc = _station_scene(3)
    victim = 12
    _corrupt_pose(sc, victim)
    res = _verify(sc)
    assert res["rotation_flags"][victim]
    assert res["flagged"][victim]
    assert res["rotation_scores_deg"][victim] > 3.0
    # The per-image median keeps every neighbour clean: each sees exactly one
    # discrepant pair (with the victim) among four.
    others = np.arange(sc["n_img"]) != victim
    assert not res["flagged"][others].any()


def test_verify_flags_junk_support_via_resection_screen():
    sc = _station_scene(3)
    victim = 7
    rng = np.random.default_rng(0)
    mask = sc["image"] == victim
    sc["pos"][mask] = rng.uniform(0, W, size=(int(mask.sum()), 2))
    res = _verify(sc)
    assert res["resect_flags"][victim]
    assert res["flagged"][victim]
    others = np.arange(sc["n_img"]) != victim
    assert not res["resect_flags"][others].any()


# ── repair_poses ───────────────────────────────────────────────────────────


def test_repair_restores_wrong_pose_and_leaves_others_untouched():
    sc = _station_scene(3)
    victim = 12
    true_rot = sc["rots"][victim].copy()
    true_center = sc["centers"][victim].copy()
    _corrupt_pose(sc, victim)
    q_in, t_in, _ = _pose_arrays(sc)

    res = _repair(sc)
    assert set(res) == REPAIR_KEYS
    assert res["flagged"][victim]
    assert res["repaired"][victim]
    assert res["inlier_before"][victim] < 0.1
    assert res["inlier_after"][victim] > 0.9

    # The repaired pose is restored to within tight bounds of truth.
    r = _quat_to_mat(res["quaternions_wxyz"][victim])
    c_est = -(r.T @ res["translations"][victim])
    angle = np.degrees(
        np.arccos(np.clip((np.trace(r @ true_rot.T) - 1) / 2, -1.0, 1.0))
    )
    assert angle < 0.3, f"rotation off by {angle} deg"
    assert np.linalg.norm(c_est - true_center) < 0.05

    # Every other camera passes through bit for bit; no repair attempted.
    others = np.arange(sc["n_img"]) != victim
    npt.assert_array_equal(res["quaternions_wxyz"][others], q_in[others])
    npt.assert_array_equal(res["translations"][others], t_in[others])
    assert not res["repaired"][others].any()
    assert np.isnan(res["inlier_before"][others]).all()
    assert np.isnan(res["inlier_after"][others]).all()


def test_repair_rejected_when_cluster_points_corrupted():
    sc = _station_scene(3)
    # Corrupt the world points of every cluster station 0 observes: its
    # cameras are flagged by screen A but pose-only repair cannot fix broken
    # structure — every repair is rejected and the state passes through.
    rng = np.random.default_rng(1)
    station0 = sc["image"] < 5
    broken = np.zeros(len(sc["world"]), bool)
    broken[sc["cluster"][station0]] = True
    sc["world"][broken] = rng.uniform(-20, 20, size=(int(broken.sum()), 3))
    q_in, t_in, _ = _pose_arrays(sc)

    res = _repair(sc)
    assert res["resect_flags"][:5].all()
    assert res["flagged"][:5].all()
    assert not res["repaired"].any()
    npt.assert_array_equal(res["quaternions_wxyz"], q_in)
    npt.assert_array_equal(res["translations"], t_in)


# ── Determinism and validation ─────────────────────────────────────────────


def test_verify_and_repair_deterministic():
    sc = _station_scene(3, n_stations=3, n_pts=300)
    _corrupt_pose(sc, 7)
    a = _repair(sc, seed=42)
    b = _repair(sc, seed=42)
    for key in sorted(REPAIR_KEYS):
        npt.assert_array_equal(a[key], b[key], err_msg=key)


def test_verify_validation_errors():
    sc = _station_scene(3, n_stations=2, per_station=2, n_pts=60)
    q, t, idx = _pose_arrays(sc)
    pi, pj, pc, pd = _neighborhood_arrays(sc)
    cam = _make_cam()
    args = [sc["cluster"], sc["image"], sc["pos"], cam, sc["world"], q, t, idx]
    # Diagonal pair in the substrate arrays.
    with pytest.raises(ValueError):
        verify_poses(*args, np.array([1], np.uint32), np.array([1], np.uint32),
                     np.array([9], np.uint32), np.array([1.0]))
    # Pair arrays not parallel.
    with pytest.raises(ValueError):
        verify_poses(*args, pi[:-1], pj, pc, pd)
    # Observation arrays not parallel.
    with pytest.raises(ValueError):
        verify_poses(sc["cluster"], sc["image"][:-1], sc["pos"], cam, sc["world"],
                     q, t, idx, pi, pj, pc, pd)
    # Poses not parallel to posed_indexes.
    with pytest.raises(ValueError):
        verify_poses(sc["cluster"], sc["image"], sc["pos"], cam, sc["world"],
                     q[:-1], t, idx, pi, pj, pc, pd)
    # Cluster ids out of range for the points array.
    with pytest.raises(ValueError):
        verify_poses(sc["cluster"], sc["image"], sc["pos"], cam, sc["world"][:1],
                     q, t, idx, pi, pj, pc, pd)
