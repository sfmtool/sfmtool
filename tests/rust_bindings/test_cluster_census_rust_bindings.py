# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster match census Rust binding
(``sfmtool._sfmtool.analysis.cluster_census``; see
``specs/core/cluster-census.md``).

Synthetic two-group captures in the canonical camera frame (the camera looks
along -Z): two well-separated groups of cameras looking at a cloud around the
origin, per-group clusters that build the intra-group covisibility the community
detection needs, and bridge clusters spanning both groups — the cross-group
evidence the census scores. A candidate that shifts one whole group vertically
still explains that group's own clusters perfectly, so only the bridges disagree:
exactly the "internally consistent and wrong" shape the census exists to catch.

The last test is the parity bar: the reference Python prototype
(``scripts/seed_census.py``'s ``census_score``, vendored below so the test needs
no cross-checkout import) and the native binding must agree on the score and
produce the identical viewpoint-group partition.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import cluster_census
from sfmtool._sfmtool.geometry import CameraIntrinsics

W, H = 800, 800
F0 = 700.0
SAT_PX = 2.0
HI_PARA = 5.0
QUALITY_PCTL = 95

REPORT_KEYS = {"score", "n_groups", "group_of", "pairs", "sat_pct", "group_consistency"}
PAIR_KEYS = {"group_a", "group_b", "n_eligible_hi", "n_unsatisfied_hi", "wilson_lb"}
CONSISTENCY_KEYS = {
    "corrections",
    "explained_pct",
    "n_explained",
    "n_unsatisfied_before",
    "net_before",
    "net_after",
}
CORRECTION_KEYS = {"group", "rotation_wxyz", "translation", "log_scale"}


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


def _look_at(c):
    """World-to-camera rotation of a camera at `c` looking at the origin."""
    z = c.copy()
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _mat_to_quat_wxyz(r):
    """Rotation matrix -> WXYZ quaternion (Shepperd's method)."""
    t = r[0, 0] + r[1, 1] + r[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        ]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        ]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        ]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        ]
    return np.asarray(q, np.float64)


def _project(rot, center, x, f=F0):
    """Canonical pinhole projection (-Z forward); None when behind the camera
    or outside the image rectangle."""
    xc = rot @ (x - center)
    if xc[2] >= -1e-9:
        return None
    u = -f * xc[0] / xc[2] + W / 2.0
    v = f * xc[1] / xc[2] + H / 2.0
    if not (0.0 <= u < W and 0.0 <= v < H):
        return None
    return np.array([u, v])


class Scene:
    """A two-group synthetic capture plus the arrays both implementations eat."""

    def __init__(self, rots, centers, cluster, image, uv, warp, n_posed, n_bridge):
        self.rots = rots
        self.centers = centers
        self.cluster = np.asarray(cluster, np.uint32)
        self.image = np.asarray(image, np.uint32)
        self.uv = np.ascontiguousarray(np.asarray(uv, np.float64))
        self.warp = np.asarray(warp, np.float64)
        self.n_img = len(rots)
        self.n_posed = n_posed
        self.n_bridge = n_bridge

    def candidate(self, shift, n_a):
        """Poses with group B (images `n_a`..`n_posed`) shifted `shift` world
        units along +Y — a rigid move, so group B still explains its own clusters
        and only the bridges disagree. Returns (R, C, posed mask)."""
        rots = self.rots.copy()
        centers = self.centers.copy()
        centers[n_a : self.n_posed, 1] += shift
        posed = np.zeros(self.n_img, bool)
        posed[: self.n_posed] = True
        return rots, centers, posed

    def native(self, shift, n_a, warp=None, **kwargs):
        rots, centers, posed = self.candidate(shift, n_a)
        idx = np.nonzero(posed)[0].astype(np.uint32)
        quats = np.ascontiguousarray(
            np.stack([_mat_to_quat_wxyz(rots[i]) for i in idx]), np.float64
        )
        trans = np.ascontiguousarray(
            np.stack([-rots[i] @ centers[i] for i in idx]), np.float64
        )
        return cluster_census(
            self.cluster,
            self.image,
            self.uv,
            self.warp if warp is None else np.asarray(warp, np.float64),
            _make_cam(),
            quats,
            trans,
            idx,
            **kwargs,
        )

    def prototype_data(self, warp=None):
        return {
            "obs_c": self.cluster.astype(np.int64),
            "obs_i": self.image.astype(np.int64),
            "obs_uv": self.uv,
            "n_img": self.n_img,
            "n_cl": len(self.warp),
            "dims": [(W, H)] * self.n_img,
            "cl_quality": self.warp if warp is None else np.asarray(warp, np.float64),
        }


def _two_group_scene(
    seed=7,
    n_a=12,
    n_b=12,
    n_unposed=2,
    n_intra=1200,
    n_bridge=400,
    n_phantom=200,
    n_no_warp=40,
):
    """Two groups of cameras plus `n_unposed` trailing images that the candidate
    never poses. Clusters: per-group (satisfied whatever the candidate does),
    genuine bridges (the evidence), phantom bridges (geometrically incoherent,
    poor warp consistency), and a few clusters with no warp-consistency value
    at all."""
    rng = np.random.default_rng(seed)
    rots, centers = [], []
    for n, lo, hi in ((n_a, -25.0, 25.0), (n_b, 65.0, 115.0), (n_unposed, -20.0, 20.0)):
        for i in range(n):
            frac = i / (n - 1) if n > 1 else 0.5
            az = np.deg2rad(lo + (hi - lo) * frac)
            c = np.array([10.0 * np.sin(az), rng.uniform(-0.3, 0.3), 10.0 * np.cos(az)])
            rots.append(_look_at(c))
            centers.append(c)
    rots = np.stack(rots)
    centers = np.stack(centers)
    n_posed = n_a + n_b
    arc_a = list(range(n_a)) + list(range(n_posed, n_posed + n_unposed))
    arc_b = list(range(n_a, n_posed))

    cluster, image, uv, warp = [], [], [], []
    bridge_ids = []

    def add(members, points, quality):
        rows = []
        for i, x in zip(members, points):
            p = _project(rots[i], centers[i], x)
            if p is None:
                return None
            rows.append((i, p))
        cid = len(warp)
        for i, p in rows:
            cluster.append(cid)
            image.append(i)
            uv.append(p)
        warp.append(quality)
        return cid

    def random_point():
        return rng.uniform([-2.5, -2.0, -2.5], [2.5, 2.0, 2.5])

    def fill(count, pick, coherent, quality):
        made = 0
        guard = 0
        while made < count:
            guard += 1
            assert guard < 100 * count + 100, "scene generation stalled"
            members = pick()
            pts = (
                [random_point()] * len(members)
                if coherent
                else [random_point() for _ in members]
            )
            cid = add(members, pts, quality())
            if cid is not None:
                made += 1
                yield cid

    genuine = lambda: float(rng.uniform(0.05, 0.5))  # noqa: E731
    phantom = lambda: float(rng.uniform(3.0, 6.0))  # noqa: E731

    for _ in fill(
        n_intra, lambda: sorted(rng.choice(arc_a, 5, replace=False)), True, genuine
    ):
        pass
    if arc_b:
        for _ in fill(
            n_intra, lambda: sorted(rng.choice(arc_b, 5, replace=False)), True, genuine
        ):
            pass
    for cid in fill(
        n_bridge,
        lambda: (
            sorted(rng.choice(arc_a[:n_a], 3, replace=False))
            + sorted(rng.choice(arc_b, 3, replace=False))
        ),
        True,
        genuine,
    ):
        bridge_ids.append(cid)
    for _ in fill(
        n_phantom,
        lambda: (
            sorted(rng.choice(arc_a[:n_a], 3, replace=False))
            + sorted(rng.choice(arc_b, 3, replace=False))
        ),
        False,
        phantom,
    ):
        pass
    for _ in fill(
        n_no_warp,
        lambda: sorted(rng.choice(arc_a, 4, replace=False)),
        True,
        lambda: float("inf"),
    ):
        pass

    scene = Scene(rots, centers, cluster, image, uv, warp, n_posed, n_bridge)
    scene.bridge_ids = bridge_ids
    scene.n_a = n_a
    return scene


# ── The reference prototype, vendored ────────────────────────────────────
# Verbatim (modulo formatting) from ``scripts/seed_census.py`` so the parity
# test compares against the logic the fleet campaign validated, without
# importing across checkouts.


def _proto_wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d)


def _proto_modularity_groups(W_):
    n = len(W_)
    k = W_.sum(1)
    two_m = k.sum()
    if two_m == 0 or n <= 2:
        return np.zeros(n, int)
    comms = [{i} for i in range(n)]

    def dq(a, b):
        w_ab = sum(W_[i, j] for i in comms[a] for j in comms[b])
        ka = sum(k[i] for i in comms[a])
        kb = sum(k[i] for i in comms[b])
        return 2 * (w_ab / two_m - ka * kb / (two_m * two_m))

    def q_of(partition):
        q = 0.0
        for com in partition:
            for i in com:
                for j in com:
                    q += W_[i, j] / two_m - k[i] * k[j] / (two_m * two_m)
        return q

    best_q, best = q_of(comms), [set(c) for c in comms]
    while len(comms) > 1:
        pairs = [
            (dq(a, b), a, b)
            for a in range(len(comms))
            for b in range(a + 1, len(comms))
        ]
        gain, a, b = max(pairs)
        comms[a] |= comms[b]
        del comms[b]
        q = q_of(comms)
        if q > best_q:
            best_q, best = q, [set(c) for c in comms]
    grp = np.zeros(n, int)
    for g, com in enumerate(best):
        for i in com:
            grp[i] = g
    return grp


def _proto_census(data, posed, R, C, f):
    """``scripts/seed_census.py``'s ``census_score``, extended to also return
    the per-image group labels, the per-pair counts, and ``sat_pct`` so the
    whole report can be compared."""
    from sfmtool._sfmtool.analysis import triangulate_batch
    from sfmtool._sfmtool.matching import ClusterCovisibility

    oc, oi, ouv = data["obs_c"], data["obs_i"], data["obs_uv"]
    n_img, n_cl = data["n_img"], data["n_cl"]
    w, h = data["dims"][0]
    cx, cy = w / 2.0, h / 2.0

    starts = np.searchsorted(oc, np.arange(n_cl + 1)).astype(np.uint32)
    cov = ClusterCovisibility.from_arrays(
        starts,
        oi.astype(np.uint32),
        n_img,
        member_accepted=np.ascontiguousarray(posed[oi]),
    )
    W_ = np.asarray(cov.counts, np.float64)
    pidx = np.nonzero(posed)[0]
    pgrp = _proto_modularity_groups(W_[np.ix_(pidx, pidx)])
    grp = np.full(n_img, -1, int)
    grp[pidx] = pgrp
    n_groups = len(set(pgrp.tolist()))
    if n_groups < 2:
        return 0.0, n_groups, grp, {}, None

    m = posed[oi]
    oc_p, oi_p, uv_p = oc[m], oi[m], ouv[m]
    order = np.argsort(oc_p, kind="stable")
    oc_s, oi_s, uv_s = oc_p[order], oi_p[order], uv_p[order]
    uniq, counts = np.unique(oc_s, return_counts=True)
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    cam = _make_cam(f)
    rc = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv_s, np.float64)))
    rays = np.einsum("nji,nj->ni", R[oi_s], rc)
    tri = triangulate_batch(
        np.ascontiguousarray(rays), np.ascontiguousarray(C[oi_s]), offs
    )
    P = np.asarray(tri["points"])
    seg_of_obs = np.repeat(np.arange(len(uniq)), counts)

    xc = np.einsum("nij,nj->ni", R[oi_s], P[seg_of_obs] - C[oi_s])
    z = np.where(np.abs(xc[:, 2]) < 1e-9, -1e-9, xc[:, 2])
    resn = np.linalg.norm(
        np.stack(
            [
                -f * xc[:, 0] / z + cx - uv_s[:, 0],
                f * xc[:, 1] / z + cy - uv_s[:, 1],
            ],
            1,
        ),
        axis=1,
    )
    resn[~np.isfinite(resn) | (z > -1e-9)] = 1e6

    so = np.lexsort((resn, seg_of_obs))
    rs = resn[so]
    lo, hi = offs[:-1], offs[1:]
    med = 0.5 * (rs[(lo + hi - 1) // 2] + rs[(lo + hi) // 2])
    fin_pt = np.isfinite(P).all(1)
    measurable = (counts >= 2) & fin_pt & np.isfinite(med)

    g_of_obs = grp[oi_s]
    key = seg_of_obs.astype(np.int64) * (n_groups + 1) + g_of_obs
    uk = np.unique(key)
    uk_seg = (uk // (n_groups + 1)).astype(np.int64)
    uk_grp = (uk % (n_groups + 1)).astype(np.int64)
    ngrp_per = np.bincount(uk_seg, minlength=len(uniq))
    bridge = measurable & (ngrp_per >= 2)
    grp_of_seg = {}
    for s, g in zip(uk_seg, uk_grp):
        grp_of_seg.setdefault(int(s), []).append(int(g))

    para = np.full(len(uniq), np.nan)
    for s in np.nonzero(bridge)[0]:
        X = P[s]
        imgs = oi_s[offs[s] : offs[s + 1]]
        v = X[None, :] - C[imgs]
        nv = np.linalg.norm(v, axis=1)
        good = nv > 0
        if good.sum() < 2:
            continue
        u = v[good] / nv[good, None]
        para[s] = np.degrees(np.arccos(np.clip((u @ u.T).min(), -1, 1)))

    qual = np.asarray(data["cl_quality"])[uniq]
    sat = measurable & (med < SAT_PX)
    qs = qual[sat & np.isfinite(qual)]
    q_eligible = float(np.percentile(qs, QUALITY_PCTL)) if len(qs) else np.inf
    eligible = np.isfinite(qual) & (qual <= q_eligible)
    brr = bridge & eligible
    hi_p = np.isfinite(para) & (para >= HI_PARA)

    pair_stats = {}
    for s in np.nonzero(bridge)[0]:
        gs = grp_of_seg[int(s)]
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                pk = (min(gs[a], gs[b]), max(gs[a], gs[b]))
                st = pair_stats.setdefault(pk, [0, 0])
                if brr[s] and hi_p[s]:
                    st[0] += 1
                    if med[s] >= SAT_PX:
                        st[1] += 1
    score = 0.0
    for pk, (nrh, nuh) in sorted(pair_stats.items()):
        score = max(score, _proto_wilson_lb(nuh, nrh))
    # ``census_one.py``'s global-satisfaction companion, over the same masks.
    rl = measurable & eligible
    sat_pct = 100.0 * (med[rl] < SAT_PX).mean() if rl.any() else 0.0
    return float(score), n_groups, grp, pair_stats, float(sat_pct)


# ── Binding surface ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def scene():
    return _two_group_scene()


def test_report_shape(scene):
    """The report carries every documented field, with `group_consistency`
    absent unless the companion is asked for."""
    report = scene.native(0.0, scene.n_a)
    assert set(report) == REPORT_KEYS
    assert isinstance(report["score"], float)
    assert isinstance(report["n_groups"], int)
    assert report["group_consistency"] is None
    group_of = np.asarray(report["group_of"])
    assert group_of.shape == (scene.n_img,)
    assert group_of.dtype == np.int32
    # The trailing images were never posed.
    assert (group_of[scene.n_posed :] == -1).all()
    assert (group_of[: scene.n_posed] >= 0).all()
    for pair in report["pairs"]:
        assert set(pair) == PAIR_KEYS
        assert pair["group_a"] < pair["group_b"]
        assert pair["n_unsatisfied_hi"] <= pair["n_eligible_hi"]


def test_clean_candidate_scores_zero(scene):
    """A candidate at the truth satisfies every bridge."""
    report = scene.native(0.0, scene.n_a)
    assert report["n_groups"] == 2
    # The groups land in different viewpoint groups.
    group_of = np.asarray(report["group_of"])
    assert len(set(group_of[: scene.n_a].tolist())) == 1
    assert len(set(group_of[scene.n_a : scene.n_posed].tolist())) == 1
    assert group_of[0] != group_of[scene.n_a]
    # A zero numerator over a large denominator can leave float dust in the
    # Wilson bound, not an exact zero.
    assert report["score"] < 1e-12
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["n_eligible_hi"] > 100
    assert report["pairs"][0]["n_unsatisfied_hi"] == 0
    assert report["sat_pct"] > 95.0


def test_misplaced_arc_is_flagged(scene):
    """One group shifted off its true place cannot satisfy the cross-group
    evidence, even though it still explains its own clusters."""
    report = scene.native(0.5, scene.n_a)
    assert report["n_groups"] == 2
    pair = report["pairs"][0]
    assert pair["n_unsatisfied_hi"] == pair["n_eligible_hi"] > 100
    assert report["score"] > 0.9
    # Local failure: global satisfaction stays high.
    assert report["sat_pct"] > 70.0


def test_phantom_bridges_are_not_evidence(scene):
    """Bridges whose warp consistency falls outside the satisfied clusters'
    P95 carry no weight — the same seam scores zero once the genuine bridges
    are relabelled as phantoms."""
    warp = scene.warp.copy()
    warp[scene.bridge_ids] = 8.0
    report = scene.native(0.5, scene.n_a, warp=warp)
    assert report["pairs"][0]["n_eligible_hi"] == 0
    assert report["score"] == 0.0


def test_single_arc_is_unverifiable():
    """One viewpoint group means there is no group structure to census."""
    scene = _two_group_scene(
        n_a=10, n_b=0, n_unposed=0, n_intra=300, n_bridge=0, n_phantom=0
    )
    report = scene.native(0.0, scene.n_a)
    assert report["n_groups"] == 1
    assert report["pairs"] == []
    assert report["score"] == 0.0
    assert report["sat_pct"] > 95.0


def test_parameters_are_honoured(scene):
    """A parallax floor above every bridge's parallax removes the evidence."""
    strict = scene.native(0.5, scene.n_a, hi_parallax_deg=179.0)
    assert strict["pairs"][0]["n_eligible_hi"] == 0
    assert strict["score"] == 0.0
    # A satisfied bar wide enough to admit the misplaced bridges clears them.
    # (A zero numerator over a large denominator leaves float dust, not an
    # exact zero — the prototype's formulation behaves identically.)
    loose = scene.native(0.5, scene.n_a, sat_px=1e5)
    assert loose["pairs"][0]["n_unsatisfied_hi"] == 0
    assert loose["score"] < 1e-12


def test_malformed_parameters_are_rejected(scene):
    """A non-finite threshold must raise, not leak through every comparison
    as "no cluster qualifies" and report a clean 0.0."""
    for kwargs in (
        {"sat_px": float("nan")},
        {"sat_px": 0.0},
        {"hi_parallax_deg": float("nan")},
        {"hi_parallax_deg": -1.0},
        {"wilson_z": float("nan")},
        {"wilson_z": -1.96},
        {"warp_percentile": float("nan")},
        {"warp_percentile": 101.0},
    ):
        with pytest.raises(ValueError):
            scene.native(0.5, scene.n_a, **kwargs)


def test_determinism(scene):
    a = scene.native(0.5, scene.n_a)
    b = scene.native(0.5, scene.n_a)
    assert a["score"] == b["score"]
    assert a["sat_pct"] == b["sat_pct"]
    np.testing.assert_array_equal(np.asarray(a["group_of"]), np.asarray(b["group_of"]))
    assert a["pairs"] == b["pairs"]


def test_input_validation(scene):
    cam = _make_cam()
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    t = np.zeros((1, 3))
    with pytest.raises(ValueError, match="nondecreasing"):
        cluster_census(
            np.array([1, 0], np.uint32),
            np.array([0, 0], np.uint32),
            np.zeros((2, 2)),
            np.array([0.1, 0.1]),
            cam,
            q,
            t,
            np.array([0], np.uint32),
        )
    with pytest.raises(ValueError, match="out of range"):
        cluster_census(
            np.array([0, 5], np.uint32),
            np.array([0, 0], np.uint32),
            np.zeros((2, 2)),
            np.array([0.1, 0.1]),
            cam,
            q,
            t,
            np.array([0], np.uint32),
        )
    with pytest.raises(ValueError, match="n_posed"):
        cluster_census(
            np.array([0, 0], np.uint32),
            np.array([0, 0], np.uint32),
            np.zeros((2, 2)),
            np.array([0.1]),
            cam,
            q,
            np.zeros((2, 3)),
            np.array([0], np.uint32),
        )


# ── Group consistency (the opt-in companion) ─────────────────────────────


def _correction(report, group):
    (c,) = [
        c for c in report["group_consistency"]["corrections"] if c["group"] == group
    ]
    return c


def _angle_deg(q):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.asarray(q)[0]))))))


def test_group_consistency_shape(scene):
    """Asking for the companion populates it with every documented field, and
    leaves every phase-1 field bit-identical."""
    off = scene.native(0.5, scene.n_a)
    on = scene.native(0.5, scene.n_a, compute_group_consistency=True)
    assert off["group_consistency"] is None
    assert set(on) == REPORT_KEYS
    assert on["score"] == off["score"]
    assert on["n_groups"] == off["n_groups"]
    assert on["sat_pct"] == off["sat_pct"]
    assert on["pairs"] == off["pairs"]
    np.testing.assert_array_equal(
        np.asarray(on["group_of"]), np.asarray(off["group_of"])
    )

    gc = on["group_consistency"]
    assert set(gc) == CONSISTENCY_KEYS
    assert isinstance(gc["explained_pct"], float)
    assert isinstance(gc["n_explained"], int)
    assert isinstance(gc["n_unsatisfied_before"], int)
    assert isinstance(gc["net_before"], int)
    assert isinstance(gc["net_after"], int)
    assert len(gc["corrections"]) == on["n_groups"]
    assert [c["group"] for c in gc["corrections"]] == list(range(on["n_groups"]))
    for c in gc["corrections"]:
        assert set(c) == CORRECTION_KEYS
        assert np.asarray(c["rotation_wxyz"]).shape == (4,)
        assert np.asarray(c["rotation_wxyz"]).dtype == np.float64
        assert np.asarray(c["translation"]).shape == (3,)
        assert isinstance(c["log_scale"], float)


def test_a_rigidly_misplaced_group_is_coherent(scene):
    """One group shifted +0.5 in Y: the correction that re-glues the seam is
    exactly that shift, so the two groups' corrections must differ by it —
    whichever of them ended up holding the gauge."""
    report = scene.native(0.5, scene.n_a, compute_group_consistency=True)
    group_of = np.asarray(report["group_of"])
    ga, gb = int(group_of[0]), int(group_of[scene.n_a])
    assert ga != gb
    ta = np.asarray(_correction(report, ga)["translation"])
    tb = np.asarray(_correction(report, gb)["translation"])
    np.testing.assert_allclose(ta - tb, [0.0, 0.5, 0.0], atol=0.05)
    for g in (ga, gb):
        c = _correction(report, g)
        assert _angle_deg(c["rotation_wxyz"]) < 1.0
        assert abs(c["log_scale"]) < 0.02

    gc = report["group_consistency"]
    assert gc["explained_pct"] > 90.0
    # The percentage is the reported ratio, with a denominator big enough
    # to mean something.
    assert gc["n_unsatisfied_before"] > 40
    assert gc["explained_pct"] == pytest.approx(
        100.0 * gc["n_explained"] / gc["n_unsatisfied_before"]
    )
    assert gc["net_after"] > gc["net_before"]


def test_a_truthful_candidate_needs_no_correction(scene):
    """Nothing to correct and nothing to explain."""
    report = scene.native(0.0, scene.n_a, compute_group_consistency=True)
    gc = report["group_consistency"]
    for c in gc["corrections"]:
        assert _angle_deg(c["rotation_wxyz"]) < 0.05
        assert abs(c["log_scale"]) < 1e-3
        assert np.linalg.norm(np.asarray(c["translation"])) < 0.02
    assert gc["explained_pct"] == 0.0
    assert gc["net_after"] == gc["net_before"]


def test_junk_bridges_that_slipped_the_screen_are_incoherent():
    """A candidate at the truth whose only cross-group evidence is false
    matches. The census flags it, but no rigid correction can place unrelated
    physical points on top of each other, so the flag is incoherent."""
    scene = _two_group_scene(
        n_intra=600, n_bridge=0, n_phantom=300, n_no_warp=0, n_unposed=0
    )
    # The phantoms are the capture's only bridges; relabel their warp
    # consistency so the eligibility screen admits them as evidence.
    warp = scene.warp.copy()
    warp[warp > 1.0] = 0.3
    report = scene.native(0.0, scene.n_a, warp=warp, compute_group_consistency=True)
    assert report["n_groups"] == 2
    assert report["score"] > 0.9
    assert report["group_consistency"]["explained_pct"] < 10.0


def test_group_consistency_declines_without_evidence(scene):
    """Fewer than two groups, or no eligible bridge, means the companion has
    nothing to say — `None`, not a vacuous identity."""
    single = _two_group_scene(
        n_a=10, n_b=0, n_unposed=0, n_intra=300, n_bridge=0, n_phantom=0
    )
    report = single.native(0.0, single.n_a, compute_group_consistency=True)
    assert report["n_groups"] == 1
    assert report["group_consistency"] is None

    warp = scene.warp.copy()
    warp[scene.bridge_ids] = 8.0
    starved = scene.native(0.5, scene.n_a, warp=warp, compute_group_consistency=True)
    assert starved["n_groups"] == 2
    assert starved["group_consistency"] is None


def test_group_consistency_determinism(scene):
    a = scene.native(0.5, scene.n_a, compute_group_consistency=True)
    b = scene.native(0.5, scene.n_a, compute_group_consistency=True)
    ga, gb = a["group_consistency"], b["group_consistency"]
    assert ga["explained_pct"] == gb["explained_pct"]
    assert ga["net_before"] == gb["net_before"]
    assert ga["net_after"] == gb["net_after"]
    for ca, cb in zip(ga["corrections"], gb["corrections"]):
        assert ca["group"] == cb["group"]
        assert ca["log_scale"] == cb["log_scale"]
        np.testing.assert_array_equal(
            np.asarray(ca["rotation_wxyz"]), np.asarray(cb["rotation_wxyz"])
        )
        np.testing.assert_array_equal(
            np.asarray(ca["translation"]), np.asarray(cb["translation"])
        )


# ── Parity with the reference prototype ──────────────────────────────────


@pytest.mark.parametrize(
    "shift, floor, ceiling",
    [
        # A clean candidate, a partially-failing seam, and a clearly bad one —
        # so parity is checked across the whole response range, not just at 0.
        (0.0, 0.0, 0.0),
        (0.06, 0.2, 0.9),
        (0.5, 0.9, 1.0),
    ],
)
def test_parity_with_prototype(scene, shift, floor, ceiling):
    """The native census and the vendored prototype must agree on the score,
    the per-pair counts, `sat_pct`, and the viewpoint-group partition."""
    rots, centers, posed = scene.candidate(shift, scene.n_a)
    proto_score, proto_groups, proto_group_of, proto_pairs, proto_sat = _proto_census(
        scene.prototype_data(), posed, rots, centers, F0
    )
    report = scene.native(shift, scene.n_a)

    assert floor <= proto_score <= ceiling, f"unexpected regime: {proto_score}"
    assert report["n_groups"] == proto_groups
    np.testing.assert_array_equal(
        np.asarray(report["group_of"]), proto_group_of.astype(np.int32)
    )
    assert report["score"] == pytest.approx(proto_score, abs=1e-9)
    assert report["sat_pct"] == pytest.approx(proto_sat, abs=1e-9)
    native_pairs = {
        (p["group_a"], p["group_b"]): [p["n_eligible_hi"], p["n_unsatisfied_hi"]]
        for p in report["pairs"]
    }
    assert native_pairs == {tuple(k): list(v) for k, v in proto_pairs.items()}
