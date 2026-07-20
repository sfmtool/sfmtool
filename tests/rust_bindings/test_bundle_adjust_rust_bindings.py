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


# ── Points at infinity ──────────────────────────────────────────────────


def _lowpar_scene(n_img=6, n_pt=80, f=500.0, noise=0.3, seed=11):
    """A low-parallax, near-planar scene with observation noise: cameras on a
    shallow look-at arc over a thin central cloud whose depth relief is ~1%
    of the viewing distance — the regime where a wrong focal trades against
    rotation bends under ``opt_f`` (the finite focal signal drowns in the
    pixel noise)."""
    rng = np.random.default_rng(seed)
    cam = _cam(f)
    rots, trans = [], []
    for i in range(n_img):
        ang = 0.04 * (i - (n_img - 1) / 2)
        c = np.array([8.0 * np.sin(ang), 0.1 * rng.uniform(-1, 1), 8.0 * np.cos(ang)])
        r = _look_at_origin(c)
        rots.append(r)
        trans.append(-r @ c)
    pts = rng.uniform(-1.0, 1.0, size=(n_pt, 3)) * np.array([1.2, 0.9, 0.06])
    uv, oi, op = [], [], []
    for p in range(n_pt):
        for i in range(n_img):
            c = rots[i] @ pts[p] + trans[i]
            if c[2] >= -0.5:
                continue
            u = f * (-c[0] / c[2]) + 320.0
            v = f * (c[1] / c[2]) + 240.0
            if not (0.0 <= u < 640.0 and 0.0 <= v < 480.0):
                continue
            uv.append([u, v])
            oi.append(i)
            op.append(p)
    rays = np.array(
        [rots[i] @ pts[p] + trans[i] for i, p in zip(oi, op)], dtype=np.float64
    )
    uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    uv = uv + rng.uniform(-noise, noise, uv.shape)
    quats = np.array([_matrix_to_quat_wxyz(r) for r in rots])
    return {
        "cam": cam,
        "quats": quats,
        "trans": np.array(trans),
        "points": pts,
        "uv": np.ascontiguousarray(uv, dtype=np.float64),
        "obs_image": np.array(oi, dtype=np.uint32),
        "obs_point": np.array(op, dtype=np.uint32),
    }


def _add_direction_tracks(s, n_dir, rng, f=500.0, noise=0.0):
    """Append far-field direction tracks (world-frame unit directions, mostly
    along world −Z) observed by exact projection — plus optional pixel noise
    — in every camera where they land in-image and in front; returns the
    ``point_at_infinity`` mask."""
    rots = [_quat_to_matrix(q) for q in s["quats"]]
    n_pt = len(s["points"])
    dirs, rays, oi, op = [], [], [], []
    for j in range(n_dir):
        d = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.25, 0.25), -1.0])
        d /= np.linalg.norm(d)
        dirs.append(d)
        n_obs = 0
        for i, r in enumerate(rots):
            c = r @ d
            if c[2] >= 0.0:
                continue
            u = f * (-c[0] / c[2]) + 320.0
            v = f * (c[1] / c[2]) + 240.0
            if not (0.0 <= u < 640.0 and 0.0 <= v < 480.0):
                continue
            rays.append(c)
            oi.append(i)
            op.append(n_pt + j)
            n_obs += 1
        assert n_obs >= 2, f"direction {j} observed only {n_obs} times"
    uv = np.asarray(s["cam"].ray_to_pixel_batch(np.ascontiguousarray(np.array(rays))))
    uv = uv + rng.uniform(-noise, noise, uv.shape)
    s["points"] = np.vstack([s["points"], np.array(dirs)])
    s["uv"] = np.vstack([s["uv"], np.ascontiguousarray(uv, dtype=np.float64)])
    s["obs_image"] = np.concatenate([s["obs_image"], np.array(oi, dtype=np.uint32)])
    s["obs_point"] = np.concatenate([s["obs_point"], np.array(op, dtype=np.uint32)])
    mask = np.zeros(len(s["points"]), dtype=bool)
    mask[n_pt:] = True
    return mask


def test_point_at_infinity_absent_none_and_all_false_match_bitwise():
    # The finite-only kernel must be reproduced bit for bit by an absent
    # kwarg, an explicit None, and an all-False mask.
    runs = []
    for kw in (
        {},
        {"point_at_infinity": None},
        {"point_at_infinity": np.zeros(60, dtype=bool)},
    ):
        s = _scene(n_img=8, n_pt=60)
        s["cam"] = _cam(600.0)
        runs.append(_run(s, opt_f=True, **kw))
    ref = runs[0]
    for out in runs[1:]:
        assert out["focal"] == ref["focal"]
        npt.assert_array_equal(out["quaternions_wxyz"], ref["quaternions_wxyz"])
        npt.assert_array_equal(out["translations"], ref["translations"])
        npt.assert_array_equal(out["points"], ref["points"])
        npt.assert_array_equal(out["residual_norms"], ref["residual_norms"])


def test_direction_rows_returned_unit_at_fixpoint():
    s = _scene(n_img=6, n_pt=40)
    mask = _add_direction_tracks(s, 12, np.random.default_rng(5))
    d_true = s["points"][mask].copy()
    out = _run(s, point_at_infinity=mask)
    assert np.max(out["residual_norms"]) < 1e-5
    d = out["points"][mask]
    npt.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)
    npt.assert_allclose(d, d_true, atol=1e-6)


def test_directions_recover_focal_on_low_parallax_scene():
    # Binding parity of the kernel's rotation-lock test: finite-only opt_f
    # converges far from the true focal on the noisy low-parallax
    # near-planar scene; far-field direction tracks recover it.
    plain = _lowpar_scene()
    plain["cam"] = _cam(650.0)
    out_plain = _run(plain, opt_f=True, schedule=[(300.0, 2.0)], max_iters=150)
    assert abs(out_plain["focal"] - 500.0) > 25.0, out_plain["focal"]

    s = _lowpar_scene()
    mask = _add_direction_tracks(s, 20, np.random.default_rng(17), noise=0.3)
    s["cam"] = _cam(650.0)
    out = _run(
        s,
        point_at_infinity=mask,
        opt_f=True,
        schedule=[(300.0, 2.0)],
        max_iters=150,
    )
    assert abs(out["focal"] - 500.0) < 5.0, (out["focal"], out_plain["focal"])


def test_translation_frozen_for_direction_only_image():
    s = _scene(n_img=6, n_pt=40)
    # An extra image on the arc that observes only directions.
    c = np.array([8.0 * np.sin(0.5), 0.2, 8.0 * np.cos(0.5)])
    r = _look_at_origin(c)
    s["quats"] = np.vstack([s["quats"], _matrix_to_quat_wxyz(r)])
    s["trans"] = np.vstack([s["trans"], -r @ c])
    extra = len(s["quats"]) - 1
    mask = _add_direction_tracks(s, 15, np.random.default_rng(9))
    assert np.any(s["obs_image"] == extra)
    # Perturb its pose: the rotation must refine back while the translation
    # (unconstrained by directions) passes through bit-identical.
    s["quats"][extra] = _matrix_to_quat_wxyz(
        _rotvec_to_matrix(np.array([0.01, -0.008, 0.006])) @ r
    )
    s["trans"][extra] = s["trans"][extra] + np.array([0.3, -0.2, 0.1])
    t_frozen = s["trans"][extra].copy()
    r0_true = _quat_to_matrix(s["quats"][0])
    out = _run(s, point_at_infinity=mask, schedule=[(50.0, 1.0)])
    npt.assert_array_equal(out["translations"][extra], t_frozen)
    # The global rotation gauge is free (translations are invariant under a
    # world rotation about the origin), so compare the relative rotation
    # against image 0.
    rel_est = (
        _quat_to_matrix(out["quaternions_wxyz"][extra])
        @ _quat_to_matrix(out["quaternions_wxyz"][0]).T
    )
    rel = rel_est @ (r @ r0_true.T).T
    ang = np.arccos(min(1.0, (np.trace(rel) - 1) / 2))
    assert ang < 1e-5, f"extra relative rotation err {ang}"


def test_nan_direction_reborn_by_reestimation():
    s = _scene(n_img=5, n_pt=40)
    mask = _add_direction_tracks(s, 8, np.random.default_rng(13))
    victim = np.nonzero(mask)[0][3]
    d_true = s["points"][victim].copy()
    s["points"] = s["points"].copy()
    s["points"][victim] = np.nan
    out = _run(s, point_at_infinity=mask, schedule=[(50.0, 2.0), (4.0, 1.0)])
    d = out["points"][victim]
    assert np.all(np.isfinite(d))
    npt.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-9)
    ang = np.arccos(min(1.0, float(np.dot(d, d_true))))
    assert ang < 1e-6, f"reborn direction off truth by {ang}"
    rows = s["obs_point"] == victim
    assert np.max(out["residual_norms"][rows]) < 0.1


def test_fortran_order_with_mask_matches_c_order():
    def build():
        s = _scene(n_img=6, n_pt=40)
        mask = _add_direction_tracks(s, 10, np.random.default_rng(23))
        return s, mask

    s, mask = build()
    ref = _run(s, point_at_infinity=mask)
    f, fmask = build()
    for key in ("quats", "trans", "points", "uv"):
        f[key] = np.asfortranarray(f[key])
    out = _run(f, point_at_infinity=fmask)
    npt.assert_allclose(out["points"], ref["points"], atol=1e-9)
    npt.assert_allclose(out["translations"], ref["translations"], atol=1e-9)
    npt.assert_allclose(out["residual_norms"], ref["residual_norms"], atol=1e-9)


def test_point_at_infinity_shape_validation():
    s = _scene()
    with pytest.raises(ValueError, match="point_at_infinity"):
        _run(s, point_at_infinity=np.zeros(3, dtype=bool))


# ── Protected observations ──────────────────────────────────────────────


def test_protected_absent_none_and_all_false_match_bitwise():
    # The unprotected kernel must be reproduced bit for bit by an absent
    # kwarg, an explicit None, and an all-False mask.
    runs = []
    for kw_of_n_obs in (
        lambda n_obs: {},
        lambda n_obs: {"protected": None},
        lambda n_obs: {"protected": np.zeros(n_obs, dtype=bool)},
    ):
        s = _scene(n_img=8, n_pt=60)
        s["cam"] = _cam(600.0)
        runs.append(_run(s, opt_f=True, **kw_of_n_obs(len(s["uv"]))))
    ref = runs[0]
    for out in runs[1:]:
        assert out["focal"] == ref["focal"]
        npt.assert_array_equal(out["quaternions_wxyz"], ref["quaternions_wxyz"])
        npt.assert_array_equal(out["translations"], ref["translations"])
        npt.assert_array_equal(out["points"], ref["points"])
        npt.assert_array_equal(out["residual_norms"], ref["residual_norms"])


def _corrupt_track(s, victim, rng):
    """Mutually inconsistent beyond-trim offsets on one track. The signs
    alternate deterministically per observation so no common component exists
    for the track's free point to absorb (random signs can come out
    near-consistent and be fitted away); the magnitude stays moderate so the
    junk's soft-L1 cost — linear in the offset — cannot outweigh sacrificing
    the clean fit."""
    rows = np.nonzero(s["obs_point"] == victim)[0]
    s["uv"] = s["uv"].copy()
    for j, k in enumerate(rows):
        sx = 1.0 if j % 2 == 0 else -1.0
        sy = 1.0 if (j // 2) % 2 == 0 else -1.0
        s["uv"][k] += np.array([sx, sy]) * rng.uniform(40.0, 60.0, 2)
    return rows


def test_protected_track_survives_trim_unprotected_is_dropped():
    def build():
        s = _scene(n_img=6, n_pt=40)
        victim = int(s["obs_point"][0])
        _corrupt_track(s, victim, np.random.default_rng(29))
        return s, victim

    schedule = [(25.0, 1.0)]
    # Unprotected: the corrupted track is fully trimmed; under a single-round
    # schedule its point passes through bit-identical.
    s, victim = build()
    before = s["points"][victim].copy()
    out_plain = _run(s, schedule=schedule)
    npt.assert_array_equal(out_plain["points"][victim], before)
    # Protected: the corrupted observations stay in the solve and the point
    # moves; the junk is pulled toward, never fitted.
    s, victim = build()
    protected = s["obs_point"] == victim
    out = _run(s, protected=protected, schedule=schedule)
    assert np.any(out["points"][victim] != before)
    # The junk stays an outlier beyond the trim gate (pulled toward, not
    # fitted) and the clean majority still fits.
    assert np.max(out["residual_norms"][protected]) > 25.0
    assert np.median(out["residual_norms"][~protected]) < 1.0


def test_protected_composes_with_point_at_infinity():
    # Both masks apply with no special casing (smoke): a protected corrupted
    # direction observation stays in the solve and pulls its direction off
    # the truth that the clean observations pin.
    def build():
        s = _scene(n_img=6, n_pt=40)
        mask = _add_direction_tracks(s, 8, np.random.default_rng(31))
        victim = np.nonzero(mask)[0][2]
        k = int(np.nonzero(s["obs_point"] == victim)[0][0])
        s["uv"] = s["uv"].copy()
        s["uv"][k] += np.array([120.0, -90.0])
        return s, mask, victim, k

    schedule = [(25.0, 1.0)]
    s, mask, victim, _k = build()
    d_true = s["points"][victim].copy()
    out_plain = _run(s, point_at_infinity=mask, schedule=schedule)
    npt.assert_allclose(out_plain["points"][victim], d_true, atol=1e-9)

    s, mask, victim, k = build()
    protected = np.zeros(len(s["uv"]), dtype=bool)
    protected[k] = True
    out = _run(s, point_at_infinity=mask, protected=protected, schedule=schedule)
    d = out["points"][victim]
    npt.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-9)
    ang = np.arccos(min(1.0, float(np.dot(d, d_true))))
    assert ang > 1e-6, "protected corrupted direction obs left no trace"


def test_protected_shape_validation():
    s = _scene()
    with pytest.raises(ValueError, match="protected"):
        _run(s, protected=np.zeros(3, dtype=bool))


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, np.inf, np.nan])
def test_protected_loss_scale_validation(bad_scale):
    s = _scene()
    with pytest.raises(ValueError, match="protected_loss_scale"):
        _run(s, protected_loss_scale=bad_scale)


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
