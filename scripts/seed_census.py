# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""GT-free cluster census score for an in-memory seed solve.

Ported from the workspace-prep echo-census detector (census_one.py): partition
the posed frames into viewpoint groups by greedy-modularity communities of the
raw cluster-covisibility graph, triangulate every raw cluster at the solve's
poses, and score each group pair by the Wilson lower bound of the fraction of
eligible, high-parallax bridge clusters the solve cannot satisfy (eligible =
warp-consistency residual within the satisfied clusters' own P95).  A solve whose
cross-group evidence disagrees with it scores high; the score falls monotonically
as the focal/placement approach truth (validated on the DnDTabletop GT), which
makes it a structure-independent arbiter between candidate solves of the SAME
capture.  Needs only the cluster arrays (``load_clusters`` with ``cl_quality``)
and the candidate poses + focal — no reference reconstruction.
"""

import numpy as np

SAT_PX = 2.0  # satisfied / unsatisfied reprojection bar (px)
HI_PARA = 5.0  # "high parallax" floor (deg)
QUALITY_PCTL = 95  # eligibility threshold pctile (satisfied-cluster warp consistency)


def wilson_lb(k, n, z=1.96):
    """Wilson lower confidence bound of a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d)


def modularity_groups(W):
    """Greedy (CNM) modularity communities of a weighted graph; returns a
    group id per node.  Deterministic; best-Q partition over the merge path."""
    n = len(W)
    k = W.sum(1)
    two_m = k.sum()
    if two_m == 0 or n <= 2:
        return np.zeros(n, int)
    comms = [{i} for i in range(n)]

    def dq(a, b):
        w_ab = sum(W[i, j] for i in comms[a] for j in comms[b])
        ka = sum(k[i] for i in comms[a])
        kb = sum(k[i] for i in comms[b])
        return 2 * (w_ab / two_m - ka * kb / (two_m * two_m))

    def q_of(partition):
        q = 0.0
        for com in partition:
            for i in com:
                for j in com:
                    q += W[i, j] / two_m - k[i] * k[j] / (two_m * two_m)
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


def census_score(data, posed, R, C, f, camera=None):
    """Cluster census score of one candidate solve.

    ``data`` is the ``load_clusters`` dict (requires ``cl_quality``); ``posed``
    a bool mask over its images; ``R``/``C`` world->camera rotations and camera
    centers per image (canonical frame, -Z forward); ``f`` the focal in px.
    Returns ``(score, n_groups)`` — score 0.0 when fewer than two viewpoint
    groups exist (nothing to census).

    ``camera`` is the candidate's own ``CameraIntrinsics``.  The census
    explains bridge clusters by TRIANGULATING them at the candidate poses and
    REPROJECTING; both steps read the camera model, so scoring a fisheye solve
    through the default centred pinhole at ``f`` would measure the model
    mismatch rather than the solve.  Omitted, it is that pinhole — the model
    every caller had before the camera context existed.

    Dispatches to the native core operation
    (``analysis.cluster_census``, specs/core/analysis/cluster-census.md) when the
    installed binding has it; the pure-Python implementation below is the
    fallback for older builds and stays as the parity reference."""
    try:
        return _census_score_native(data, posed, R, C, f, camera)
    except (ImportError, AttributeError):
        return _census_score_py(data, posed, R, C, f, camera)


def census_report(data, posed, R, C, f, camera=None):
    """Full native census report of one candidate solve, with the
    group-consistency companion (specs/core/analysis/cluster-census.md § 6): the
    per-group corrections and how much of the cross-group disagreement they
    explain (``explained_pct`` with ``n_explained``/``n_unsatisfied_before``,
    ``net_before``/``net_after``). Same inputs as :func:`census_score`.
    Native only — requires a binding with ``compute_group_consistency``."""
    return _native_report(data, posed, R, C, f, camera, compute_group_consistency=True)


# The models whose map images directions past 90 deg off-axis, so canonical
# depth is not their domain test: the core's ``needs_ray_path`` (fisheye or
# equirectangular, ``camera/intrinsics.rs``) transcribed model by model,
# because the binding exposes no predicate to read it from.
#
# Spelled out rather than matched on the "FISHEYE" substring, which is a
# naming accident in both directions: it happens to catch SFMTOOL_FISHEYE, and
# it would go on missing any future ray-path model that is not named for one.
# SFMTOOL_PINHOLE is the case that makes the difference visible — a
# spline-corrected PINHOLE, ``is_fisheye = false`` in the core, whose map
# divides by rz and whose domain test is exactly the canonical depth.
_RAY_PATH_MODELS = frozenset(
    {
        "EQUIDISTANT_FISHEYE",
        "SFMTOOL_FISHEYE",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "OPENCV_FISHEYE",
        "THIN_PRISM_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
        "EQUIRECTANGULAR",
    }
)


def _needs_ray_path(cam):
    """Whether ``cam`` images directions past 90 deg off-axis, so canonical
    depth is not its domain test (see ``_RAY_PATH_MODELS``)."""
    return cam.model in _RAY_PATH_MODELS


def _default_camera(data, f):
    """The centred SIMPLE_PINHOLE at ``f`` — the model this census assumed
    before its callers could carry a camera context."""
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    w, h = data["dims"][0]
    return CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": int(w),
            "height": int(h),
            "parameters": {
                "focal_length": float(f),
                "principal_point_x": w / 2.0,
                "principal_point_y": h / 2.0,
            },
        }
    )


def _census_score_native(data, posed, R, C, f, camera=None):
    rep = _native_report(data, posed, R, C, f, camera)
    return float(rep["score"]), int(rep["n_groups"])


def _native_report(data, posed, R, C, f, camera=None, **kwargs):
    from scipy.spatial.transform import Rotation

    from sfmtool._sfmtool.analysis import cluster_census

    # The native kernel is model-generic: it unprojects the members with
    # ``pixel_to_ray`` and scores the explanation with ``ray_to_pixel``, so
    # handing it the candidate's own camera is the whole of what makes the
    # census model-correct.
    cam = _default_camera(data, f) if camera is None else camera
    posed_idx = np.ascontiguousarray(np.nonzero(posed)[0], np.uint32)
    Rp = np.asarray(R, np.float64)[posed_idx]
    q = Rotation.from_matrix(Rp).as_quat()[:, [3, 0, 1, 2]]
    t = -np.einsum("nij,nj->ni", Rp, np.asarray(C, np.float64)[posed_idx])
    rep = cluster_census(
        np.ascontiguousarray(data["obs_c"], np.uint32),
        np.ascontiguousarray(data["obs_i"], np.uint32),
        np.ascontiguousarray(data["obs_uv"], np.float64),
        np.ascontiguousarray(data["cl_quality"], np.float64),
        cam,
        np.ascontiguousarray(q, np.float64),
        np.ascontiguousarray(t, np.float64),
        posed_idx,
        sat_px=SAT_PX,
        hi_parallax_deg=HI_PARA,
        warp_percentile=float(QUALITY_PCTL),
        **kwargs,
    )
    return rep


def _census_score_py(data, posed, R, C, f, camera=None):
    from sfmtool._sfmtool.analysis import triangulate_batch
    from sfmtool._sfmtool.matching import ClusterCovisibility

    oc, oi, ouv = data["obs_c"], data["obs_i"], data["obs_uv"]
    n_img, n_cl = data["n_img"], data["n_cl"]
    starts = np.searchsorted(oc, np.arange(n_cl + 1)).astype(np.uint32)
    cov = ClusterCovisibility.from_arrays(
        starts,
        oi.astype(np.uint32),
        n_img,
        member_accepted=np.ascontiguousarray(posed[oi]),
    )
    W = np.asarray(cov.counts, np.float64)
    pidx = np.nonzero(posed)[0]
    pgrp = modularity_groups(W[np.ix_(pidx, pidx)])
    grp = np.full(n_img, -1, int)
    grp[pidx] = pgrp
    n_groups = len(set(pgrp.tolist()))
    if n_groups < 2:
        return 0.0, n_groups

    m = posed[oi]
    oc_p, oi_p, uv_p = oc[m], oi[m], ouv[m]
    order = np.argsort(oc_p, kind="stable")
    oc_s, oi_s, uv_s = oc_p[order], oi_p[order], uv_p[order]
    uniq, counts = np.unique(oc_s, return_counts=True)
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    cam = _default_camera(data, f) if camera is None else camera
    rc = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv_s, np.float64)))
    rays = np.einsum("nji,nj->ni", R[oi_s], rc)
    tri = triangulate_batch(
        np.ascontiguousarray(rays), np.ascontiguousarray(C[oi_s]), offs
    )
    P = np.asarray(tri["points"])
    seg_of_obs = np.repeat(np.arange(len(uniq)), counts)

    # Reprojection through the CAMERA, not a hand-written pinhole division:
    # `-f*x/z + cx` is `ray_to_pixel` for SIMPLE_PINHOLE only, and a ray-path
    # model images the whole `z >= 0` half-space this expression rejects.  The
    # in-front test follows the model the same way — canonical depth for the
    # perspective family, the batch projector's own validity otherwise.
    xc = np.einsum("nij,nj->ni", R[oi_s], P[seg_of_obs] - C[oi_s])
    proj = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc, np.float64)))
    resn = np.linalg.norm(proj - uv_s, axis=1)
    invalid = ~np.isfinite(resn)
    if not _needs_ray_path(cam):
        invalid |= xc[:, 2] > -1e-9
    resn[invalid] = 1e6

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
        score = max(score, wilson_lb(nuh, nrh))
    return float(score), n_groups
