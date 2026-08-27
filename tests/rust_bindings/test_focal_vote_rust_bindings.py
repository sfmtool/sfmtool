# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the structure-free focal-vote Rust bindings
(``sfmtool._sfmtool.geometry.focal_vote`` and ``estimate_homography``; see
``specs/core/geometry/focal-vote.md``).

Synthetic scenes are built in the OpenCV/optical pixel convention (a point in
front has camera-frame ``z > 0``). A pure-rotation rig (all camera centres at
the world origin) produces parallax-free pairs that vote through rotation
self-calibration; a baseline camera track over finite structure produces
parallax-rich pairs that vote through the Bougnoux focal of a fundamental
matrix. Both families pool into one log-space median (unless their medians
disagree beyond 0.25 in log-focal, when the majority family's median stands
alone); the majority contributor of each scene's pool is the family its
parallax regime can observe.
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
        "n_pool",
        "pool_spread",
        "family_disagreement",
        "parallax_poverty",
        "epipolar_spread",
        "rotation_spread",
        "epipolar_votes",
        "rotation_votes",
        "n_h_dominated",
        "n_estimator_failed",
        "n_band_rejected",
        "n_degenerate",
        "n_inconsistent_pairs",
        "camera_model",
        "columns",
    }
    assert isinstance(res["n_epipolar"], int)
    assert isinstance(res["n_rotation"], int)
    assert isinstance(res["n_pool"], int)
    assert isinstance(res["n_inconsistent_pairs"], int)
    assert isinstance(res["n_degenerate"], int)
    assert isinstance(res["parallax_poverty"], float)
    assert isinstance(res["pool_spread"], float)
    assert res["family"] in ("Epipolar", "Rotation", None)
    # n_pool is exactly the two families' pooled contributions.
    assert res["n_pool"] == res["n_epipolar"] + res["n_rotation"]
    # The inter-family gap exists exactly when both families voted.
    both_voted = res["n_epipolar"] > 0 and res["n_rotation"] > 0
    assert (res["family_disagreement"] is not None) == both_voted
    # The per-vote detail lists are the diagnostic layer: epipolar entries are
    # per DIRECTION (both F and Ft), so they outnumber the pooled pair votes.
    assert len(res["rotation_votes"]) == res["n_rotation"]
    for v in res["epipolar_votes"]:
        assert set(v) == {
            "image_a",
            "image_b",
            "shared_clusters",
            "mean_disp_px",
            "n_f_inliers",
            "n_h_inliers",
            "transposed",
            "focal_px",
        }
    for v in res["rotation_votes"]:
        assert set(v) == {
            "image",
            "partner",
            "mean_disp_px",
            "n_inliers",
            "focal_px",
        }
    # The default column set is pinhole-only: nothing to arbitrate, no scans.
    assert res["camera_model"] == "Pinhole"
    assert res["columns"] == []


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
    # Parallax-free rig: the epipolar candidates are homography-dominated, so
    # the pool is rotation votes and Rotation is its majority contributor.
    cl, im, pos = _rotation_scene(2024)
    res = focal_vote(cl, im, pos, W, H, seed=0)
    assert res["family"] == "Rotation", res
    assert res["n_rotation"] > res["n_epipolar"]
    assert res["n_pool"] == res["n_epipolar"] + res["n_rotation"]
    assert res["focal_px"] is not None
    assert abs(res["focal_px"] - F_TRUE) / F_TRUE < 0.1
    assert res["parallax_poverty"] >= 0.55
    # Each unordered image pair votes at most once, so the scan's mutual
    # widest-partner pairs appear a single time.
    pairs = {
        (min(v["image"], v["partner"]), max(v["image"], v["partner"]))
        for v in res["rotation_votes"]
    }
    assert len(pairs) == res["n_rotation"]
    # No epipolar votes survive the homography gate, so the pool is the
    # rotation family alone and its spread is that family's.
    assert res["n_epipolar"] == 0
    assert res["family_disagreement"] is None
    assert res["pool_spread"] == res["rotation_spread"]


def test_focal_vote_parallax_scene():
    # Baseline track over finite structure: direction-consistent epipolar pairs
    # dominate the pool.
    cl, im, pos = _parallax_scene(4048)
    res = focal_vote(cl, im, pos, W, H, seed=0)
    assert res["family"] == "Epipolar", res
    assert res["n_epipolar"] > res["n_rotation"]
    assert res["n_pool"] >= 2
    # Detail entries are per direction; pooled votes are per pair.
    assert len(res["epipolar_votes"]) >= res["n_epipolar"]
    assert res["focal_px"] is not None
    assert abs(res["focal_px"] - F_TRUE) / F_TRUE < 0.15
    assert res["parallax_poverty"] < 0.55
    # One stray rotation vote (1188 px) joins the 7 epipolar pair votes, and
    # the two family medians are 0.40 apart in log-focal — past the 0.25 band —
    # so the majority family's median stands alone rather than blending, and
    # pool_spread describes that family's votes.
    assert res["n_rotation"] == 1
    assert res["family_disagreement"] > 0.25
    assert res["focal_px"] == res["epipolar_focal_px"]
    assert res["pool_spread"] == res["epipolar_spread"]


def test_focal_vote_seed_reproducibility():
    cl, im, pos = _rotation_scene(2024)
    a = focal_vote(cl, im, pos, W, H, seed=42)
    b = focal_vote(cl, im, pos, W, H, seed=42)
    assert a["focal_px"] == b["focal_px"]
    assert a["family"] == b["family"]
    assert a["n_epipolar"] == b["n_epipolar"]
    assert a["n_rotation"] == b["n_rotation"]
    assert a["n_pool"] == b["n_pool"]
    assert a["n_inconsistent_pairs"] == b["n_inconsistent_pairs"]
    assert a["n_degenerate"] == b["n_degenerate"]
    assert a["parallax_poverty"] == b["parallax_poverty"]
    assert a["pool_spread"] == b["pool_spread"]
    assert a["family_disagreement"] == b["family_disagreement"]
    assert a["epipolar_votes"] == b["epipolar_votes"]
    assert a["rotation_votes"] == b["rotation_votes"]


def test_focal_vote_noncontiguous_input():
    cl, im, pos = _rotation_scene(2024)
    pos_nc = np.repeat(pos, 2, axis=1)[:, ::2]
    assert not pos_nc.flags["C_CONTIGUOUS"]
    a = focal_vote(cl, im, pos_nc, W, H, seed=0)
    b = focal_vote(cl, im, np.ascontiguousarray(pos), W, H, seed=0)
    assert a["focal_px"] == b["focal_px"]
    assert a["family"] == b["family"]


# ── Camera-model columns ────────────────────────────────────────────────────

F_FISH = 320.0


def _project_fisheye(r, t, x, f=F_FISH):
    """Equidistant fisheye projection ``theta = r / f``, image centre at CX/CY."""
    xc = r @ x + t
    rho = np.hypot(xc[0], xc[1])
    if rho < 1e-12:
        return None
    th = np.arctan2(rho, xc[2])
    if not 0.0 <= th < 0.98 * np.pi:
        return None
    rr = f * th
    u, v = CX + rr * xc[0] / rho, CY + rr * xc[1] / rho
    if not (0 <= u < W and 0 <= v < H):
        return None
    return np.array([u, v])


def _emit_fisheye_pair(obs, cams, ia, ib, m, rng, depth=None):
    done, guard = 0, 0
    while done < m and guard < m * 2000:
        guard += 1
        th, ph = rng.uniform(0.25, 1.45), rng.uniform(0, 2 * np.pi)
        d = 30.0 if depth is None else rng.uniform(*depth)
        x = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]) * d
        pa = _project_fisheye(*cams[ia], x)
        pb = _project_fisheye(*cams[ib], x)
        if pa is not None and pb is not None:
            obs.push(
                ia,
                pa + 0.2 * rng.standard_normal(2),
                ib,
                pb + 0.2 * rng.standard_normal(2),
            )
            done += 1


def _fisheye_scene(seed):
    """A ~179 deg fisheye capture: a rotation rig plus a baseline sub-capture, so
    each cell of the equidistant column has its own ground."""
    rng = np.random.default_rng(seed)
    rot_n, bl_n = 8, 8
    cams = _rotation_cams(rot_n, 1.4, rng) + _baseline_cams(bl_n, 1.0, rng)
    obs = _Obs()
    for i in range(rot_n - 1):
        _emit_fisheye_pair(obs, cams, i, i + 1, 60, rng)
    for i in range(rot_n - 2):
        _emit_fisheye_pair(obs, cams, i, i + 2, 60, rng)
    for i in range(bl_n - 1):
        _emit_fisheye_pair(obs, cams, rot_n + i, rot_n + i + 1, 60, rng, (5.0, 12.0))
    for i in range(bl_n - 2):
        _emit_fisheye_pair(obs, cams, rot_n + i, rot_n + i + 2, 60, rng, (5.0, 12.0))
    return obs.arrays()


def test_focal_vote_default_columns_match_explicit_pinhole_only():
    # The new parameter defaults to pinhole-only, and asking for pinhole-only
    # explicitly reproduces the default call exactly — every pre-existing key
    # included.
    cl, im, pos = _rotation_scene(2024)
    default = focal_vote(cl, im, pos, W, H, seed=0)
    explicit = focal_vote(cl, im, pos, W, H, seed=0, columns=("pinhole",))
    assert default == explicit
    assert default["camera_model"] == "Pinhole"
    assert default["columns"] == []


def test_focal_vote_rejects_unknown_column():
    cl, im, pos = _rotation_scene(2024)
    with pytest.raises(ValueError):
        focal_vote(cl, im, pos, W, H, columns=("brown-conrady",))


def test_focal_vote_fisheye_capture_is_arbitrated_equidistant():
    cl, im, pos = _fisheye_scene(2718)
    res = focal_vote(cl, im, pos, W, H, seed=0, columns=("pinhole", "equidistant"))
    assert res["camera_model"] == "EquidistantFisheye", res["columns"]
    assert [c["camera_model"] for c in res["columns"]] == [
        "Pinhole",
        "EquidistantFisheye",
    ]
    pin, fish = res["columns"]
    assert fish["n_informative"] > pin["n_informative"]
    # The top level reports the winning column's consensus, never a blend of
    # the two columns' focals.
    assert res["focal_px"] == fish["focal_px"]
    assert abs(res["focal_px"] - F_FISH) / F_FISH < 0.05
    assert (res["n_epipolar"], res["n_rotation"]) == (
        fish["n_epipolar"],
        fish["n_rotation"],
    )
    # Per-column diagnostics mirror the per-family ones and add the certificate
    # counts the arbitration reads.
    assert set(pin) == {
        "camera_model",
        "focal_px",
        "family",
        "epipolar_focal_px",
        "rotation_focal_px",
        "n_epipolar",
        "n_rotation",
        "n_pool",
        "pool_spread",
        "family_disagreement",
        "epipolar_spread",
        "rotation_spread",
        "parallax_poverty",
        "n_rotation_dominated",
        "n_scanned_epipolar",
        "n_scanned_rotation",
        "n_certified_epipolar",
        "n_certified_rotation",
        "n_informative_epipolar",
        "n_informative_rotation",
        "n_certified",
        "n_informative",
        "scan_votes",
    }
    for c in res["columns"]:
        assert c["n_certified"] == c["n_certified_epipolar"] + c["n_certified_rotation"]
        assert (
            c["n_informative"]
            == c["n_informative_epipolar"] + c["n_informative_rotation"]
        )
        assert len(c["scan_votes"]) == c["n_scanned_epipolar"] + c["n_scanned_rotation"]
        for v in c["scan_votes"]:
            assert set(v) == {
                "cell",
                "image_a",
                "image_b",
                "focal_px",
                "cost",
                "sharpness",
                "dir_disagreement",
                "rotation_dominated",
                "rotation_ratio",
                "coverage_p90",
                "n_inliers",
                "in_fov_band",
                "at_grid_edge",
                "angular_focal_px",
                "certified",
                "model_informative",
            }
            assert v["cell"] in ("Epipolar", "Rotation")
            # Only certified votes can be model-informative, and an edge-pinned
            # or rotation-dominated scan is never certified.
            assert not (v["model_informative"] and not v["certified"])
            assert not (v["certified"] and v["at_grid_edge"])
            assert not (v["certified"] and v["rotation_dominated"])
            # Rotation domination is an epipolar-cell verdict only.
            if v["cell"] == "Rotation":
                assert not v["rotation_dominated"]
                assert v["rotation_ratio"] is None
        assert c["n_rotation_dominated"] == sum(
            v["rotation_dominated"] for v in c["scan_votes"]
        )


def test_focal_vote_pinhole_capture_is_arbitrated_pinhole():
    # The converse, and the compatibility guarantee: when pinhole wins, the
    # top-level fields are exactly the pinhole-only answer.
    cl, im, pos = _parallax_scene(4048)
    both = focal_vote(cl, im, pos, W, H, seed=0, columns=("pinhole", "equidistant"))
    only = focal_vote(cl, im, pos, W, H, seed=0)
    assert both["camera_model"] == "Pinhole", both["columns"]
    for key in (
        "focal_px",
        "family",
        "epipolar_focal_px",
        "rotation_focal_px",
        "n_epipolar",
        "n_rotation",
        "n_pool",
        "pool_spread",
        "family_disagreement",
    ):
        assert both[key] == only[key], key


def test_focal_vote_columns_seed_reproducibility():
    cl, im, pos = _fisheye_scene(2718)
    cols = ("pinhole", "equidistant")
    a = focal_vote(cl, im, pos, W, H, seed=11, columns=cols)
    b = focal_vote(cl, im, pos, W, H, seed=11, columns=cols)
    assert a == b


def test_focal_vote_empty_input():
    res = focal_vote(
        np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros((0, 2)), W, H
    )
    assert res["focal_px"] is None
    assert res["family"] is None
    assert res["n_epipolar"] == 0
    assert res["n_rotation"] == 0
    assert res["n_pool"] == 0
    assert res["n_inconsistent_pairs"] == 0
    assert res["n_degenerate"] == 0
    assert res["pool_spread"] == 0.0
    assert res["family_disagreement"] is None
