# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 experiment: photometric track widening -- THE RECIPE.

Phase 1 (`exp_track_widen.py`) proved the localization machinery and found
that post-hoc widening + flat robust BA cannot repair the drift gauge: the
manufactured wide observations are a 1-2% minority whose large residuals ARE
the corrective signal, and the staged trim votes them out.  The fragment
tables then showed the constraints DO merge rigid fragments (ws2 components
15 -> 5, dominant 87 -> 194 cams) and what remains is a LOW-DIMENSIONAL
inter-fragment gauge problem.  This script implements the sharpened recipe:

  1. REACH-DEFICIT selection (the key change from Phase 1): candidate tracks
     are finite, consistent tracks whose observers span a SMALL fraction of
     the TRAJECTORY (registration/sequence order), not tracks of small
     angular span.  Extension views are chosen for TRAJECTORY DISTANCE --
     temporally distant, frustum-visible anchor views plus covisibility
     cross-sweep revisit partners (covisible-by-features but temporally far)
     -- subject to the Phase-1 degeneracy gates (span >= 10x the angular
     noise floor atan(reproj/f), implied depth <= 3 rig diameters, grazing
     < 60 deg vs the refined patch normal).  Purpose: cross-time coupling,
     not angle.
  2. Localize with the Phase-1 machinery (PatchCloud.from_tracks, refined
     normals K=8, feature-scaled extents, chunked pyramids).  Accept by
     LOO-ZNCC quality (data-derived floor); record each observation's
     reprojection delta but do NOT gate on it -- an obs landing in a
     different fragment is EXPECTED to have a large delta; that is the
     signal the Phase-1 <4px gate threw away.
  3. Inject accepted obs as PROTECTED into the staged native bundle_adjust
     (protected=mask, default protected_loss_scale): protected obs are never
     trimmed and pull through a widened robust loss.  Arms: control (no new
     obs), unprot (new obs, no protection), prot (new obs, protected).
  4. FRAGMENT-ALIGN: decompose the solve GT-free by RANSAC-similarity
     decomposition of bootstrap-vs-protected-BA poses (cameras that the
     protected pull moved as a rigid group = one fragment); estimate the few
     inter-fragment similarities FROM the protected spanning observations
     (each links a 3D point owned by one fragment to a 2D ray in another;
     >= 3 non-collinear links determine a similarity); apply to non-dominant
     fragments, re-triangulate, final protected BA.
  5. Score every stage with `sfm compare <GT> <arm>.sfmr --fragments`
     (subprocess; stage .sfmr files saved via clone_with_changes) plus the
     Phase-1 global/piecewise center-error metrics.  Success: components ->
     1 (or dominant -> ~100%), global -> piecewise, no regression on the
     clean-sweep guard.

Phase-1 machinery is REUSED BY IMPORT (`import exp_track_widen as p1`):
loading, triangulation, track stats, localization, evaluation.  Only
`load_clusters` is copied (extended to return the matches-file image list and
the covisible posed-image pairs derived from the same read).

Usage:
  pixi run -e dev python scripts/exp_widen_growth.py <ws> [<ws> ...]
      [--max-tracks 500] [--no-compare] [--out DIR]
"""

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_track_widen as p1  # noqa: E402  (Phase-1 machinery)

from sfmtool._sfmtool.analysis import (  # noqa: E402
    apply_se3_to_camera_poses_py,
    estimate_alignment_rs,
)
from sfmtool._sfmtool.geometry import RotQuaternion, bundle_adjust  # noqa: E402
from sfmtool._sfmtool.io import read_matches  # noqa: E402
from sfmtool._sfmtool.reconstruction import SfmrReconstruction  # noqa: E402
from sfmtool._compare_fragments import decompose_fragments  # noqa: E402
from sfmtool.align.core import ImageMatch  # noqa: E402

GRAZE_DEG = p1.GRAZE_DEG  # 60 deg grazing ceiling vs the refined patch normal
SHARED_COUNT_FLOOR = 50  # shared-cluster floor per covisible pair (073131082)
N_UNIFORM_ANCHORS = 20  # temporally-spread candidate anchor views
N_REVISIT_ANCHORS = 12  # top covisibility revisit views (by track votes)
CANDS_PER_TRACK = 12  # candidate anchor views offered per track (dt-bucketed)
MIN_LINKS = 3  # similarity floor per fragment pair (#228 spec: >= 3 non-collinear)
TRIM_PX = p1.TRIM_PX


def log(*a):
    print(*a, flush=True)


# -- Loading (load_clusters copied from Phase 1 + matches_names return) -------


def load_clusters_ext(ws, posed_names):
    """Phase-1 `load_clusters` with the matches-file image list kept, plus the
    covisible image pairs derived from the same read.

    The covisibility evidence is indexed by the matches file's image_names, so
    the mapping matches-index -> pose-index must survive; everything else is
    the Phase-1 universe (status-0/1 members, valid reference, posed span
    >= 2, best ACTIVE_CAP clusters by span)."""
    files = sorted(glob.glob(str(ws / "matches" / "*-clusters-patches.matches")))
    if not files:
        raise FileNotFoundError("no clusters-patches file")
    pose_set = set(posed_names)
    best, best_ov, best_f = None, -1, None
    for fp in files:
        d = read_matches(fp)
        ov = len(set(d["image_names"]) & pose_set)
        if ov > best_ov or (
            ov == best_ov and len(d["image_names"]) > len(best["image_names"])
        ):
            best, best_ov, best_f = d, ov, fp
    d = best
    log(f"  clusters file: {Path(best_f).name} ({best_ov} posed images)")
    names = list(d["image_names"])
    pose_of = {n: j for j, n in enumerate(posed_names)}
    img_pose = np.array([pose_of.get(n, -1) for n in names])
    starts = np.asarray(d["cluster_starts"]).astype(np.int64)
    mi = np.asarray(d["member_images"])
    st = np.asarray(d["member_status"])
    refs = np.asarray(d["reference_members"])
    aff = np.asarray(d["member_affines"])
    feat = np.asarray(d["member_features"])
    zncc = np.asarray(d["member_zncc"])
    usable = []
    for c in range(len(starts) - 1):
        lo, hi = int(starts[c]), int(starts[c + 1])
        if refs[c] == np.iinfo(np.uint32).max:
            continue
        sel = (
            np.nonzero(
                ((st[lo:hi] == 0) | (st[lo:hi] == 1)) & (img_pose[mi[lo:hi]] >= 0)
            )[0]
            + lo
        )
        span = len(np.unique(mi[sel]))
        if span >= 2:
            usable.append((span, sel))
    # Covisible pairs of POSED views: the shared-cluster count over the whole
    # usable cluster set (not the ACTIVE_CAP working subset), derived here from
    # the file just read.  At most one status-0/1 member per (cluster, image),
    # so the binary image x cluster incidence Gram IS the per-pair count.
    from scipy import sparse

    rows = (
        np.concatenate([sel for _span, sel in usable])
        if usable
        else np.zeros(0, np.int64)
    )
    cl_of = np.repeat(
        np.arange(len(usable), dtype=np.int64),
        [len(sel) for _span, sel in usable],
    )
    inc = sparse.csr_matrix(
        (
            np.ones(len(rows), np.int32),
            (img_pose[mi[rows]].astype(np.int64), cl_of),
        ),
        shape=(len(posed_names), max(len(usable), 1)),
    )
    inc.data[:] = 1
    gram = sparse.triu(inc @ inc.T, k=1).tocoo()
    strong = gram.data >= SHARED_COUNT_FLOOR
    covis_pairs = np.stack([gram.row[strong], gram.col[strong]], axis=1)

    order = sorted(range(len(usable)), key=lambda k: -usable[k][0])
    keep = sorted(order[: p1.ACTIVE_CAP])
    obs_c, obs_i, obs_uv, obs_feat, obs_img, obs_zncc = [], [], [], [], [], []
    for n_cl, k in enumerate(keep):
        for m in usable[k][1]:
            obs_c.append(n_cl)
            obs_i.append(img_pose[mi[m]])
            obs_uv.append(aff[m, :, 2])
            obs_feat.append(int(feat[m]))
            obs_img.append(names[mi[m]])
            obs_zncc.append(float(zncc[m]))
    dims = np.asarray(d["image_dims"])
    return dict(
        obs_c=np.asarray(obs_c, np.int64),
        obs_i=np.asarray(obs_i, np.int64),
        obs_uv=np.asarray(obs_uv, np.float64),
        obs_feat=np.asarray(obs_feat, np.int64),
        obs_img=np.asarray(obs_img),
        obs_zncc=np.asarray(obs_zncc),
        n_cl=len(keep),
        wh=(int(dims[0][0]), int(dims[0][1])),
        matches_names=names,
        img_pose=img_pose,
        # Pure-2D covisibility evidence (shared-cluster count per posed image
        # pair), reconstruction-independent -- exactly what frustum tests at
        # drifted poses are not.
        covis_pairs=covis_pairs,
    )


# -- Reach-deficit selection --------------------------------------------------


def temporal_ranks(names):
    """Sequence order = name-sorted rank (video frame naming)."""
    rank = np.empty(len(names), np.int64)
    rank[np.argsort(np.asarray(names))] = np.arange(len(names))
    return rank


def track_reach(obs_c, obs_i, rank, n_cl, n_img):
    """Per-track observer reach: fraction of the trajectory (rank span) the
    member views cover, plus the temporal midpoint (rank units)."""
    reach = np.full(n_cl, np.nan)
    mid = np.full(n_cl, np.nan)
    lo_arr = np.full(n_cl, np.iinfo(np.int64).max, np.int64)
    hi_arr = np.full(n_cl, -1, np.int64)
    np.minimum.at(lo_arr, obs_c, rank[obs_i])
    np.maximum.at(hi_arr, obs_c, rank[obs_i])
    have = hi_arr >= 0
    denom = max(n_img - 1, 1)
    reach[have] = (hi_arr[have] - lo_arr[have]) / denom
    mid[have] = 0.5 * (hi_arr[have] + lo_arr[have])
    return reach, mid, lo_arr, hi_arr


def select_reach_deficit(
    span, count, med_zncc, med_reproj, depth, rig_diam, f0, reach, mid, max_tracks
):
    """Reach-deficit widening candidates.

    Phase-1 degeneracy gates stay (member floor, data-derived ZNCC floor,
    member reproj < gate, implied depth <= 3 rig diameters, span >= 10x the
    angular noise floor -- a track below its noise floor triangulates to
    arbitrary depth and every candidate computation from it is fiction).
    The angular-narrowness gate is REPLACED by the reach-deficit gate:
    observer reach at/below the qualifying set's median (data-derived).
    Selected tracks are sampled evenly across the TEMPORAL MIDPOINT of the
    qualifying list so protected coupling lands everywhere along the
    trajectory (fragment junctions included), not just in one segment."""
    valid = np.isfinite(span) & np.isfinite(reach)
    zfloor = float(np.nanpercentile(med_zncc[valid], 25)) if valid.any() else 0.0
    theta_noise = np.degrees(np.arctan(np.maximum(med_reproj, 1.0) / f0))
    qual0 = (
        valid
        & (count >= p1.MIN_MEMBERS)
        & (span >= p1.SPAN_NOISE_MULT * theta_noise)
        & (med_zncc >= zfloor)
        & (med_reproj < p1.ACCEPT_PX)
        & (depth <= p1.DEPTH_RIG_MAX * rig_diam)
    )
    if not qual0.any():
        return np.array([], np.int64), zfloor, np.nan, np.nan
    reach_med = float(np.median(reach[qual0]))
    reach_p75 = float(np.percentile(reach[qual0], 75))
    qual = np.nonzero(qual0 & (reach <= reach_med))[0]
    qual = qual[np.argsort(mid[qual])]
    if len(qual) > max_tracks:
        qual = qual[np.round(np.linspace(0, len(qual) - 1, max_tracks)).astype(int)]
    return qual, zfloor, reach_med, reach_p75


# -- Trajectory-distance candidates (anchor views) ----------------------------


def build_anchor_set(track_ids, obs_c, obs_i, rank, covis_pairs, n_img, td_floor):
    """Global candidate anchor views: a temporally-spread uniform skeleton
    plus the covisibility revisit views most voted by the selected
    tracks (covisible-by-features with a member but temporally distant --
    cross-sweep revisits the drifted frustum test can miss).  A small global
    anchor set keeps pyramid chunks tractable AND concentrates the protected
    constraints into few views, which is what fragment alignment wants."""
    order = np.argsort(rank)
    uni = order[
        np.round(np.linspace(0, n_img - 1, min(N_UNIFORM_ANCHORS, n_img))).astype(int)
    ]
    votes = np.zeros(n_img, np.int64)
    if covis_pairs is not None and len(covis_pairs):
        bounds = np.searchsorted(obs_c, np.arange(obs_c[-1] + 2))
        adj = {}
        for a, b in covis_pairs:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        denom = max(n_img - 1, 1)
        for c in track_ids:
            members = np.unique(obs_i[bounds[c] : bounds[c + 1]])
            mr = rank[members]
            mlo, mhi = int(mr.min()), int(mr.max())
            seen = set()
            for m in members:
                for q in adj.get(int(m), ()):
                    if q in seen:
                        continue
                    dt = max(0, mlo - rank[q], rank[q] - mhi) / denom
                    if dt >= td_floor:
                        seen.add(q)
            for q in seen:
                votes[q] += 1
    rev = np.argsort(-votes)[:N_REVISIT_ANCHORS]
    rev = rev[votes[rev] > 0]
    anchors = np.asarray(sorted(set(uni.tolist()) | set(rev.tolist())), np.int64)
    return anchors, votes


def find_reach_candidates(
    track_ids,
    obs_c,
    obs_i,
    pts,
    mean_ray,
    rot,
    t,
    cam,
    wh,
    rank,
    anchors,
    covis_pairs,
    td_floor,
):
    """Per selected track: extension anchor views at TRAJECTORY DISTANCE.

    A view qualifies if it is an anchor, not a member, temporally >= td_floor
    of the trajectory away from the member rank interval, the point projects
    in front and in bounds (5% margin; covisible revisit partners of the
    track's members get a loose bound -- projection within max_shift of the
    frame -- because at drifted poses the frustum test itself is suspect),
    and the viewing ray stays under the grazing ceiling vs the patch normal.
    Up to CANDS_PER_TRACK views, spread across trajectory-distance buckets.
    No angle-vs-span requirement: cross-time coupling is the currency."""
    w, h = wh
    margin = 0.05 * min(w, h)
    loose = 60.0  # max_shift_px: a projection this far out can still localize
    n_img = len(rot)
    denom = max(n_img - 1, 1)
    centers = -np.einsum("nij,ni->nj", rot, t)
    bounds = np.searchsorted(obs_c, np.arange(obs_c[-1] + 2))
    adj = {}
    if covis_pairs is not None:
        for a, b in covis_pairs:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))
    anchor_mask = np.zeros(n_img, bool)
    anchor_mask[anchors] = True
    buckets = [(td_floor, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1.01)]
    buckets = [(lo, hi) for lo, hi in buckets if hi > td_floor]
    out = {}
    for c in track_ids:
        p = pts[c]
        members = np.unique(obs_i[bounds[c] : bounds[c + 1]])
        mr = rank[members]
        mlo, mhi = int(mr.min()), int(mr.max())
        dt = np.maximum(0, np.maximum(mlo - rank, rank - mhi)) / denom
        rays = p[None, :] - centers
        dist = np.linalg.norm(rays, axis=1)
        ok = dist > 1e-12
        rays = np.where(ok[:, None], rays / np.maximum(dist, 1e-12)[:, None], 0.0)
        ang = np.degrees(np.arccos(np.clip(rays @ mean_ray[c], -1.0, 1.0)))
        x_cam = np.einsum("nij,nj->ni", rot, np.broadcast_to(p, (n_img, 3))) + t
        uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
        infront = x_cam[:, 2] < 0
        finite = np.isfinite(uv).all(axis=1)
        strict = (
            finite
            & (uv[:, 0] >= margin)
            & (uv[:, 0] <= w - margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 1] <= h - margin)
        )
        loose_b = (
            finite
            & (uv[:, 0] >= -loose)
            & (uv[:, 0] <= w + loose)
            & (uv[:, 1] >= -loose)
            & (uv[:, 1] <= h + loose)
        )
        revisit = np.zeros(n_img, bool)
        for m in members:
            for qv in adj.get(int(m), ()):
                revisit[qv] = True
        base = (
            ok
            & anchor_mask
            & infront
            & (ang < GRAZE_DEG)
            & (dt >= td_floor)
            & (strict | (revisit & loose_b))
        )
        base[members] = False
        cidx = np.nonzero(base)[0]
        if len(cidx) == 0:
            continue
        per_bucket = max(1, CANDS_PER_TRACK // max(len(buckets), 1))
        picked = []
        for blo, bhi in buckets:
            b = cidx[(dt[cidx] >= blo) & (dt[cidx] < bhi)]
            if len(b) == 0:
                continue
            b = b[np.argsort(dt[b])]
            k = min(per_bucket, len(b))
            picked.extend(b[np.round(np.linspace(0, len(b) - 1, k)).astype(int)])
        if not picked:
            continue
        out[int(c)] = dict(
            views=np.asarray(sorted(set(int(v) for v in picked)), np.int64),
            ang=ang,
            proj=uv,
            dtn=dt,
            revisit=revisit,
        )
    return out


# -- Protected bundle adjustment ----------------------------------------------


def run_ba(cam0, q, t, pts, obs_c, obs_i, uv, wh, f0, protected=None, opt_f=True):
    """Phase-1 staged native BA (opt_f, FRAC_DIAG-scaled schedule) plus the
    protected-observation mask (permuted with the same lexsort as the obs)."""
    pxs = max(1.0, float(np.hypot(*wh)) / 550.6)
    order = np.lexsort((obs_i, obs_c))
    kwargs = {}
    if protected is not None and protected.any():
        kwargs["protected"] = np.ascontiguousarray(protected[order])
    out = bundle_adjust(
        p1.make_cam(f0, *wh),
        np.ascontiguousarray(q),
        np.ascontiguousarray(t),
        np.ascontiguousarray(pts),
        np.ascontiguousarray(uv[order]),
        obs_i[order].astype(np.uint32),
        obs_c[order].astype(np.uint32),
        opt_f=opt_f,
        schedule=[(50.0 * pxs, 5.0 * pxs), (12.0 * pxs, 2.0 * pxs), (TRIM_PX, 1.0)],
        max_iters=60,
        min_track=2,
        min_obs=12,
        **kwargs,
    )
    res = np.full(len(obs_c), np.inf)
    res[order] = np.asarray(out["residual_norms"])
    return (
        float(out["focal"]),
        np.asarray(out["quaternions_wxyz"]),
        np.asarray(out["translations"]),
        np.asarray(out["points"]),
        res,
    )


# -- GT-free fragment decomposition + alignment -------------------------------


def poses_to_centers(q, t):
    rot = Rotation.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix()
    return -np.einsum("nij,ni->nj", rot, t), rot


def gt_free_decompose(names, q_a, t_a, q_b, t_b):
    """RANSAC similarity decomposition of state A vs state B (both ours --
    no GT).  Components = camera groups whose A->B motion is one similarity;
    with A = bootstrap and B = the protected-BA arm, the protected pull
    moves each rigid fragment by its own correction, so the components
    recover the fragment structure GT-free."""
    ca, _ = poses_to_centers(q_a, t_a)
    cb, _ = poses_to_centers(q_b, t_b)
    scene_scale = float(np.sqrt(np.mean(np.sum((cb - cb.mean(0)) ** 2, axis=1))))
    matches = [
        ImageMatch(
            image_name=str(names[k]),
            source_index=k,
            target_index=k,
            source_quat=RotQuaternion.from_wxyz_array(np.asarray(q_a[k])),
            source_camera_center=ca[k],
            target_quat=RotQuaternion.from_wxyz_array(np.asarray(q_b[k])),
            target_camera_center=cb[k],
        )
        for k in range(len(names))
    ]
    dec = decompose_fragments(matches, scene_scale=scene_scale)
    frag = np.full(len(names), -1, np.int64)
    for k, comp in enumerate(dec.components):
        frag[comp.indices] = k
    return dec, frag, scene_scale


def owner_fragments(obs_c, obs_i, frag, n_cl):
    """Point ownership: the fragment holding the majority of member observers."""
    own = np.full(n_cl, -1, np.int64)
    bounds = np.searchsorted(obs_c, np.arange(n_cl + 1))
    for c in range(n_cl):
        f = frag[obs_i[bounds[c] : bounds[c + 1]]]
        f = f[f >= 0]
        if len(f) == 0:
            continue
        vals, cnts = np.unique(f, return_counts=True)
        own[c] = int(vals[np.argmax(cnts)])
    return own


def closest_on_ray(center, d, p):
    """Closest point to ``p`` on the ray ``center + s*d`` (s >= 0)."""
    s = max(0.0, float(np.dot(p - center, d)))
    return center + s * d


def fragment_align(
    names, q_x, t_x, f_x, pts_x, cl, frag, own, accepted, wh, scene_scale
):
    """Estimate + apply inter-fragment similarities from the protected
    spanning observations.

    Each accepted wide obs linking fragment B to the dominant fragment 0
    yields a 3D-3D correspondence attached to B: either the B-owned point
    (dst = closest point on the dominant camera's observed ray) or the point
    on the B camera's observed ray (dst = the dominant-owned point).  With
    >= MIN_LINKS non-collinear links, a trimmed Umeyama similarity for B is
    estimated (estimate_alignment_rs, rounds=3, keep 0.8) and applied to B's
    cameras.  Fragments without enough links to fragment 0 stay (logged)."""
    cam = p1.make_cam(f_x, *wh)
    centers, rot = poses_to_centers(q_x, t_x)
    links_by_frag = {}
    n_dir = {"b_owned": 0, "dom_owned": 0}
    for r in accepted:
        c, v = int(r["track"]), int(r["view"])
        fa, fb = int(own[c]), int(frag[v])
        if fa < 0 or fb < 0 or fa == fb or (fa != 0 and fb != 0):
            continue
        p = pts_x[c]
        if not np.isfinite(p).all():
            continue
        d_loc = np.asarray(
            cam.pixel_to_ray_batch(
                np.ascontiguousarray(np.asarray(r["uv"], np.float64).reshape(1, 2))
            )
        )[0]
        d_world = rot[v].T @ d_loc
        nrm = np.linalg.norm(d_world)
        if nrm < 1e-12:
            continue
        d_world = d_world / nrm
        if fa != 0 and fb == 0:
            # B-owned point seen from a dominant camera: move p onto the ray.
            links_by_frag.setdefault(fa, []).append(("b_owned", centers[v], d_world, p))
            n_dir["b_owned"] += 1
        else:
            # dominant-owned point seen from a B camera: move the ray point
            # (attached to B) onto p.
            links_by_frag.setdefault(fb, []).append(
                ("dom_owned", centers[v], d_world, p)
            )
            n_dir["dom_owned"] += 1
    q_new = np.array(q_x, dtype=np.float64, copy=True)
    t_new = np.array(t_x, dtype=np.float64, copy=True)
    applied = []
    for fb in sorted(links_by_frag):
        links = links_by_frag[fb]
        if len(links) < MIN_LINKS:
            applied.append(dict(frag=fb, n_links=int(len(links)), status="too_few"))
            continue

        # ICP-style iteration: the perpendicular-foot correspondence is
        # biased (a synthetic 1.3x-scale displacement recovers 0.85 instead
        # of 0.77 one-shot); re-dropping the foot under the current estimate
        # converges to the exact similarity (the true one is a fixed point:
        # T^-1(p) lies ON the observed ray).
        def correspondences(tf):
            src, dst = [], []
            for kind, cv, d, p in links:
                if kind == "dom_owned":
                    tgt = (
                        p
                        if tf is None
                        else np.asarray(tf.inverse().apply_to_points(p[None]))[0]
                    )
                    src.append(closest_on_ray(cv, d, tgt))
                    dst.append(p)
                else:
                    moved = (
                        p if tf is None else np.asarray(tf.apply_to_points(p[None]))[0]
                    )
                    src.append(p)
                    dst.append(closest_on_ray(cv, d, moved))
            return np.asarray(src, np.float64), np.asarray(dst, np.float64)

        src, _ = correspondences(None)
        sv = np.linalg.svd(src - src.mean(0), compute_uv=False)
        if sv[1] < 1e-6 * max(sv[0], 1e-12):
            applied.append(dict(frag=fb, n_links=int(len(links)), status="collinear"))
            continue
        rounds = 3 if len(links) >= 2 * MIN_LINKS else 1
        tf = None
        for _it in range(12):
            src, dst = correspondences(tf)
            tf = estimate_alignment_rs(
                np.ascontiguousarray(src),
                np.ascontiguousarray(dst),
                rounds=rounds,
                keep_fraction=0.8,
            )
        res = np.linalg.norm(
            tf.apply_to_points(np.ascontiguousarray(src)) - dst, axis=1
        )
        sel = np.nonzero(frag == fb)[0]
        disp = np.linalg.norm(
            tf.apply_to_points(np.ascontiguousarray(centers[sel])) - centers[sel],
            axis=1,
        )
        rot_wxyz = np.asarray(tf.rotation.to_wxyz_array(), np.float64)
        rot_deg = float(
            np.degrees(2.0 * np.arccos(np.clip(abs(rot_wxyz[0]), 0.0, 1.0)))
        )
        qs, ts = apply_se3_to_camera_poses_py(
            np.ascontiguousarray(rot_wxyz),
            np.ascontiguousarray(np.asarray(tf.translation, np.float64)),
            float(tf.scale),
            np.ascontiguousarray(q_new[sel]),
            np.ascontiguousarray(t_new[sel]),
        )
        q_new[sel] = np.asarray(qs)
        t_new[sel] = np.asarray(ts)
        applied.append(
            dict(
                frag=fb,
                n_links=int(len(src)),
                n_cams=int(len(sel)),
                scale=float(tf.scale),
                rot_deg=rot_deg,
                disp_pct=float(100 * np.mean(disp) / scene_scale),
                link_res_med_pct=float(100 * np.median(res) / scene_scale),
                status="applied",
            )
        )
    return q_new, t_new, applied, n_dir


# -- Scoring ------------------------------------------------------------------


def save_arm(ws, names, q_x, t_x, tag):
    bt = SfmrReconstruction.load(ws / "sfmr" / "bootstrap-pinhole.sfmr")
    arm = bt.clone_with_changes(
        quaternions_wxyz=np.ascontiguousarray(q_x, dtype=np.float64),
        translations=np.ascontiguousarray(t_x, dtype=np.float64),
    )
    path = ws / "sfmr" / f"widen2-{tag}.sfmr"
    arm.save(path, operation=f"widen2-{tag}")
    return path


def gt_path_of(ws):
    return sorted(
        p for p in (ws / "sfmr").glob("*-solve-*.sfmr") if "bootstrap" not in p.name
    )[0]


def run_compare(gt, arm_path, out_dir, tag):
    """`sfm compare <GT> <arm> --fragments` via subprocess; full output saved,
    fragment section parsed to a summary dict."""
    cmd = [
        sys.executable,
        "-c",
        "from sfmtool.cli import main; main()",
        "compare",
        str(gt),
        str(arm_path),
        "--fragments",
    ]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        text = (r.stdout or "") + "\n" + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return dict(error="compare timeout")
    out_file = out_dir / f"compare-{tag}.txt"
    out_file.write_text(text, encoding="utf-8")
    summ = parse_fragments(text)
    summ["wall_s"] = round(time.perf_counter() - t0, 1)
    return summ


def parse_fragments(text):
    m = re.search(r"Components: (\d+), outlier frames: (\d+)", text)
    if not m:
        return dict(error="no fragment section")
    out = dict(components=int(m.group(1)), outliers=int(m.group(2)))
    sizes = [int(s) for s in re.findall(r"Component \d+: (\d+) cameras", text)]
    out["sizes_top3"] = sizes[:3]
    out["n_in_components"] = int(np.sum(sizes)) if sizes else 0
    dom = re.search(
        r"Component 1: .*?Position error \(% of scene scale\): mean ([\d.]+), "
        r"median ([\d.]+), max ([\d.]+).*?Rotation error \(deg\): mean ([\d.]+), "
        r"median ([\d.]+), max ([\d.]+)",
        text,
        re.S,
    )
    if dom:
        out["dom_pos_mean"] = float(dom.group(1))
        out["dom_pos_med"] = float(dom.group(2))
        out["dom_rot_mean"] = float(dom.group(4))
        out["dom_rot_med"] = float(dom.group(5))
    return out


def fmt_frag(s):
    if "error" in s:
        return s["error"]
    dom = (
        f" dom pos {s.get('dom_pos_mean', float('nan')):.2f}/"
        f"{s.get('dom_pos_med', float('nan')):.2f}% "
        f"rot {s.get('dom_rot_mean', float('nan')):.2f}/"
        f"{s.get('dom_rot_med', float('nan')):.2f}deg"
        if "dom_pos_mean" in s
        else ""
    )
    return (
        f"comps {s['components']} top3 {s.get('sizes_top3')} "
        f"outliers {s['outliers']}{dom}"
    )


# -- Main ---------------------------------------------------------------------


def run_workspace(ws, max_tracks, out_dir, do_compare=True):
    log(f"== {ws}")
    t_start = time.perf_counter()
    out_dir = out_dir / ws.name
    out_dir.mkdir(parents=True, exist_ok=True)
    names, q, t, rot, f0 = p1.load_bootstrap(ws)
    cl = load_clusters_ext(ws, names)
    w, h = cl["wh"]
    cam = p1.make_cam(f0, w, h)
    n_img = len(names)
    centers = -np.einsum("nij,ni->nj", rot, t)
    pts = p1.triangulate(cl, rot, t, cam)
    oc, oi = cl["obs_c"], cl["obs_i"]

    # member reprojection consistency at the bootstrap poses
    res_obs = np.full(len(oc), np.nan)
    fin = np.isfinite(pts[oc]).all(axis=1)
    x_cam = np.einsum("nij,nj->ni", rot[oi[fin]], pts[oc[fin]]) + t[oi[fin]]
    proj = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
    res_obs[fin] = np.linalg.norm(proj - cl["obs_uv"][fin], axis=1)

    span, mean_ray, count, depth = p1.track_stats(oc, oi, centers, pts, cl["n_cl"])
    med_zncc = np.full(cl["n_cl"], np.nan)
    med_reproj = np.full(cl["n_cl"], np.inf)
    bounds = np.searchsorted(oc, np.arange(cl["n_cl"] + 1))
    for c in range(cl["n_cl"]):
        med_zncc[c] = np.median(cl["obs_zncc"][bounds[c] : bounds[c + 1]])
        r = res_obs[bounds[c] : bounds[c + 1]]
        if np.isfinite(r).any():
            med_reproj[c] = np.nanmedian(r)
    rig_diam = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    log(
        f"  universe: {cl['n_cl']} clusters, {len(oc)} obs, {n_img} posed; "
        f"member reproj med {np.nanmedian(res_obs):.2f}px"
    )

    # reach-deficit selection
    rank = temporal_ranks(names)
    reach, mid, _, _ = track_reach(oc, oi, rank, cl["n_cl"], n_img)
    sel, zfloor, reach_med, reach_p75 = select_reach_deficit(
        span, count, med_zncc, med_reproj, depth, rig_diam, f0, reach, mid, max_tracks
    )
    td_floor = reach_p75  # far = beyond most qualifying tracks' own reach
    log(
        f"  reach: qualifying med {reach_med:.4f} p75 {reach_p75:.4f} of trajectory; "
        f"selected {len(sel)} reach-deficit tracks (zncc>={zfloor:.2f}); "
        f"td_floor {td_floor:.4f}"
    )
    if len(sel) == 0:
        log("  nothing to widen")
        return None

    covis_pairs = cl["covis_pairs"]
    log(
        f"  covisible pairs (count>={SHARED_COUNT_FLOOR}, both posed): "
        f"{len(covis_pairs)}"
    )
    anchors, votes = build_anchor_set(sel, oc, oi, rank, covis_pairs, n_img, td_floor)
    n_rev = int((votes[anchors] > 0).sum())
    log(f"  anchors: {len(anchors)} views ({n_rev} revisit-voted, rest uniform)")

    cands = find_reach_candidates(
        sel,
        oc,
        oi,
        pts,
        mean_ray,
        rot,
        t,
        cam,
        (w, h),
        rank,
        anchors,
        covis_pairs,
        td_floor,
    )
    offered = [(c, v) for c in cands for v in cands[c]["views"]]
    dt_off = np.asarray([cands[c]["dtn"][v] for c, v in offered])
    log(
        f"  candidates: {len(offered)} views over {len(cands)} tracks; "
        f"dt p25/50/75 {np.percentile(dt_off, [25, 50, 75]).round(3).tolist() if len(offered) else '-'}"
    )
    if not cands:
        log("  no candidates")
        return None

    # localization (Phase-1 machinery, refined normals)
    records, _ = p1.localize_widen(
        ws, names, q, t, cam, cl, pts, cands, sel, refine_normals=True
    )
    for r in records:
        r["dtn"] = float(cands[r["track"]]["dtn"][r["view"]])
        r["revisit"] = bool(cands[r["track"]]["revisit"][r["view"]])
    # LOO-ZNCC acceptance floor: data-derived from the LOCALIZED population
    # (the member-ZNCC floor is miscalibrated for wide-baseline views -- member
    # patches congeal near-identically, ~0.96+, while genuinely-correct
    # wide localizations sit near 0.87), with the localizer's own relative
    # consensus floor (0.6) as the absolute sanity bound.
    loo_all = np.asarray([r["loo_zncc"] for r in records], np.float64)
    loo_floor = max(0.6, float(np.nanpercentile(loo_all, 25))) if len(loo_all) else 0.6
    log(
        f"  loo-zncc localized p25/50/75 "
        f"{np.nanpercentile(loo_all, [25, 50, 75]).round(3).tolist() if len(loo_all) else '-'}"
        f"; acceptance floor {loo_floor:.3f}"
    )
    accepted = [r for r in records if r["loo_zncc"] >= loo_floor]
    noop = sum(1 for r in accepted if r["delta_px"] < p1.NOOP_PX)
    deltas = np.asarray([r["delta_px"] for r in accepted])
    log(
        f"  localized {len(records)}, accepted (loo>={loo_floor:.2f}) {len(accepted)} "
        f"on {len({r['track'] for r in accepted})} tracks; noop {noop}; "
        f"delta_px p50/p90 "
        f"{np.percentile(deltas, [50, 90]).round(1).tolist() if len(deltas) else '-'}"
    )
    # acceptance by trajectory-distance bucket
    dt_buckets = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1.01)]
    curve = []
    for lo, hi in dt_buckets:
        rs = [r for r in records if lo <= r["dtn"] < hi]
        ac = [r for r in rs if r["loo_zncc"] >= loo_floor]
        d = np.asarray([r["delta_px"] for r in ac])
        row = dict(
            lo=lo,
            hi=hi,
            localized=len(rs),
            accepted=len(ac),
            noop=sum(1 for r in ac if r["delta_px"] < p1.NOOP_PX),
            delta_p50=float(np.percentile(d, 50)) if len(d) else None,
            delta_p90=float(np.percentile(d, 90)) if len(d) else None,
        )
        curve.append(row)
        log(f"    dt {lo:.2f}-{hi:.2f}: {row}")

    evals = {"bootstrap": p1.eval_vs_gt(ws, names, q, t, f0)}
    log(f"  bootstrap vs GT: {evals['bootstrap']}")

    # BA arms
    uv_all = cl["obs_uv"]
    new_c = np.asarray([r["track"] for r in accepted], np.int64)
    new_i = np.asarray([r["view"] for r in accepted], np.int64)
    new_uv = np.asarray([r["uv"] for r in accepted], np.float64).reshape(-1, 2)
    full_c = np.concatenate([oc, new_c])
    full_i = np.concatenate([oi, new_i])
    full_uv = np.concatenate([uv_all, new_uv])
    prot_mask = np.zeros(len(full_c), bool)
    prot_mask[len(oc) :] = True

    states = {}
    arm_defs = {
        "control": (oc, oi, uv_all, None),
        "unprot": (full_c, full_i, full_uv, None),
        "prot": (full_c, full_i, full_uv, prot_mask),
    }
    for tag, (oc2, oi2, uv2, mask) in arm_defs.items():
        t0 = time.perf_counter()
        f_x, q_x, t_x, p_x, res_x = run_ba(
            cam, q, t, pts.copy(), oc2, oi2, uv2, (w, h), f0, protected=mask
        )
        ev = p1.eval_vs_gt(ws, names, q_x, t_x, f_x)
        n_new = len(oc2) - len(oc)
        if n_new:
            new_res = res_x[len(oc) :]
            ev["n_new_obs"] = int(n_new)
            ev["n_new_surviving"] = int((new_res < TRIM_PX).sum())
            ev["new_res_p50_p90"] = (
                np.percentile(new_res[np.isfinite(new_res)], [50, 90]).round(2).tolist()
                if np.isfinite(new_res).any()
                else None
            )
        evals[tag] = ev
        states[tag] = (f_x, q_x, t_x, p_x, res_x)
        log(f"  {tag}_ba ({time.perf_counter() - t0:.0f}s) vs GT: {ev}")
        save_arm(ws, names, q_x, t_x, tag)

    # GT-free fragment decomposition: bootstrap vs protected arm
    f_p, q_p, t_p, p_p, _ = states["prot"]
    dec, frag, scene_scale = gt_free_decompose(names, q, t, q_p, t_p)
    frag_sizes = [len(cmp.indices) for cmp in dec.components]
    log(
        f"  GT-free decomposition (bootstrap vs prot): {len(dec.components)} comps "
        f"{frag_sizes[:5]} + {len(dec.outlier_indices)} outliers "
        f"(scene_scale {scene_scale:.3g})"
    )
    # diagnostic: control vs prot (isolates the protected pull from generic BA motion)
    dec2, _, _ = gt_free_decompose(
        names, states["control"][1], states["control"][2], q_p, t_p
    )
    log(
        f"  GT-free decomposition (control vs prot): {len(dec2.components)} comps "
        f"{[len(cmp.indices) for cmp in dec2.components][:5]} + "
        f"{len(dec2.outlier_indices)} outliers"
    )

    align_info = dict(n_fragments=len(dec.components), applied=[])
    if len(dec.components) > 1:
        own = owner_fragments(oc, oi, frag, cl["n_cl"])
        q_al, t_al, applied, n_dir = fragment_align(
            names, q_p, t_p, f_p, p_p, cl, frag, own, accepted, (w, h), scene_scale
        )
        align_info["applied"] = applied
        align_info["link_directions"] = n_dir
        for a in applied:
            log(f"  align frag {a}")
        if any(a["status"] == "applied" for a in applied):
            rot_al = Rotation.from_quat(q_al[:, [1, 2, 3, 0]]).as_matrix()
            cam_al = p1.make_cam(f_p, w, h)
            pts_al = p1.triangulate(cl, rot_al, t_al, cam_al)
            t0 = time.perf_counter()
            f_a, q_a2, t_a2, p_a2, res_a2 = run_ba(
                cam_al,
                q_al,
                t_al,
                pts_al,
                full_c,
                full_i,
                full_uv,
                (w, h),
                f_p,
                protected=prot_mask,
            )
            ev = p1.eval_vs_gt(ws, names, q_a2, t_a2, f_a)
            new_res = res_a2[len(oc) :]
            ev["n_new_obs"] = int(len(full_c) - len(oc))
            ev["n_new_surviving"] = int((new_res < TRIM_PX).sum())
            evals["aligned"] = ev
            log(f"  aligned_ba ({time.perf_counter() - t0:.0f}s) vs GT: {ev}")
            save_arm(ws, names, q_a2, t_a2, "aligned")
        else:
            log("  fragment-align: no transform applied (no fragment had enough links)")
    else:
        log("  fragment-align: single GT-free component; skipping")

    # CLI fragment scoring per stage
    frag_scores = {}
    if do_compare:
        gt = gt_path_of(ws)
        stage_paths = {"bootstrap": ws / "sfmr" / "bootstrap-pinhole.sfmr"}
        for tag in ("control", "unprot", "prot", "aligned"):
            path = ws / "sfmr" / f"widen2-{tag}.sfmr"
            if path.exists() and (tag in evals):
                stage_paths[tag] = path
        for tag, path in stage_paths.items():
            frag_scores[tag] = run_compare(gt, path, out_dir, tag)
            log(f"  fragments[{tag}]: {fmt_frag(frag_scores[tag])}")

    out = dict(
        ws=str(ws),
        n_cl=int(cl["n_cl"]),
        n_obs=int(len(oc)),
        n_img=int(n_img),
        reach_med=reach_med,
        reach_p75=reach_p75,
        td_floor=td_floor,
        zncc_floor=zfloor,
        n_selected=int(len(sel)),
        n_anchors=int(len(anchors)),
        n_offered=len(offered),
        n_localized=len(records),
        n_accepted=len(accepted),
        acceptance_curve=curve,
        evals=evals,
        gt_free_fragments=dict(
            boot_vs_prot=dict(
                components=frag_sizes, outliers=int(len(dec.outlier_indices))
            ),
            control_vs_prot=dict(
                components=[len(cmp.indices) for cmp in dec2.components],
                outliers=int(len(dec2.outlier_indices)),
            ),
        ),
        align=align_info,
        fragment_scores=frag_scores,
        wall_s=float(time.perf_counter() - t_start),
    )
    (out_dir / "widen2.json").write_text(json.dumps(out, indent=2))
    log(f"  done in {out['wall_s']:.0f}s -> {out_dir / 'widen2.json'}")
    return out


# -- Stage B: widen + protect INSIDE growth at anchors (prototype) ------------
#
# The recipe's final form is widening DURING growth (photo-anchor v6
# prevention: the photometric radius is only guaranteed while drift is
# small).  `exp_hier_ba.grow_loop` already fires a covisibility-spread
# anchor BA every SFMTOOL_ANCHOR_EVERY windowed BAs and delegates it to the
# module-global `_photo_anchor_ba` when SFMTOOL_PHOTO_ANCHOR=1 -- so the
# prototype MONKEYPATCHES that hook (no edits to exp_hier_ba): at each
# anchor, reach-deficit tracks of the CURRENTLY-POSED subgraph are widened
# into the anchor subset's trajectory-distant views, the accepted obs
# accumulate OUTSIDE grow_loop's observation arrays (its windowed BAs can
# never trim them), and the anchor BA re-runs with the accumulated wide obs
# PROTECTED.  After growth, one final protected BA over the full active set.


def make_widen_anchor(ws, hb, state, budget):
    """Anchor hook with `_photo_anchor_ba`'s signature: widen + protected
    anchor BA; returns (rvec, tvec) or None (fall back to the raw anchor)."""

    def hook(rvec, tvec, pts, posed, win, f0, obs_c, obs_i, u, n_img, n_cl):
        t0 = time.perf_counter()
        names = list(hb._PH_CTX["names"])
        obs_f = np.asarray(hb._PH_CTX["obs_f"])
        wh = hb._CAM_WH
        cam = p1.make_cam(f0, *wh)
        rot = Rotation.from_rotvec(rvec).as_matrix()
        q = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
        live = posed[obs_i] & ~np.isnan(pts[obs_c, 0])
        oc, oi, uv = obs_c[live], obs_i[live], u[live]
        if len(oc) < 100:
            return None
        name_arr = np.asarray(names)
        cl2 = dict(
            obs_c=oc,
            obs_i=oi,
            obs_uv=uv,
            obs_feat=obs_f[live],
            obs_img=name_arr[oi],
            n_cl=n_cl,
        )
        centers = -np.einsum("nij,ni->nj", rot, tvec)
        span, mean_ray, count, depth = p1.track_stats(oc, oi, centers, pts, n_cl)
        x_cam = np.einsum("nij,nj->ni", rot[oi], pts[oc]) + tvec[oi]
        proj = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
        res_obs = np.linalg.norm(proj - uv, axis=1)
        bounds = np.searchsorted(oc, np.arange(n_cl + 1))
        med_reproj = np.full(n_cl, np.inf)
        for c in np.unique(oc):
            med_reproj[c] = np.median(res_obs[bounds[c] : bounds[c + 1]])
        rank = temporal_ranks(names)
        reach, mid, _, _ = track_reach(oc, oi, rank, n_cl, n_img)
        pc = centers[posed]
        rig_diam = float(np.linalg.norm(pc.max(axis=0) - pc.min(axis=0)))
        theta_noise = np.degrees(np.arctan(np.maximum(med_reproj, 1.0) / f0))
        qual0 = (
            np.isfinite(span)
            & np.isfinite(reach)
            & (count >= p1.MIN_MEMBERS)
            & (span >= p1.SPAN_NOISE_MULT * theta_noise)
            & (med_reproj < p1.ACCEPT_PX)
            & (depth <= p1.DEPTH_RIG_MAX * rig_diam)
        )
        if qual0.sum() < 10:
            return None
        reach_med = float(np.median(reach[qual0]))
        td_floor = float(np.percentile(reach[qual0], 75))
        sel = np.nonzero(qual0 & (reach <= reach_med))[0]
        sel = sel[np.argsort(mid[sel])]
        if len(sel) > budget:
            sel = sel[np.round(np.linspace(0, len(sel) - 1, budget)).astype(int)]
        anchors_pool = np.nonzero(win)[0]
        cands = find_reach_candidates(
            sel,
            oc,
            oi,
            pts,
            mean_ray,
            rot,
            tvec,
            cam,
            wh,
            rank,
            anchors_pool,
            None,
            td_floor,
        )
        if not cands:
            return None
        records, _ = p1.localize_widen(
            ws, names, q, tvec, cam, cl2, pts, cands, sel, refine_normals=True
        )
        loo = np.asarray([r["loo_zncc"] for r in records], np.float64)
        floor = max(0.6, float(np.nanpercentile(loo, 25))) if len(loo) else 0.6
        acc = [r for r in records if r["loo_zncc"] >= floor]
        state["obs"].extend(
            dict(track=int(r["track"]), view=int(r["view"]), uv=r["uv"]) for r in acc
        )
        # protected anchor BA: the anchor subset's obs + every accumulated
        # wide obs on a posed view
        win_live = live.copy()
        win_live[np.nonzero(live)[0]] = win[oi]
        oc_w, oi_w, uv_w = obs_c[win_live], obs_i[win_live], u[win_live]
        wc = np.asarray([o["track"] for o in state["obs"]], np.int64)
        wi = np.asarray([o["view"] for o in state["obs"]], np.int64)
        wuv = np.asarray([o["uv"] for o in state["obs"]], np.float64).reshape(-1, 2)
        kw = posed[wi]
        full_c = np.concatenate([oc_w, wc[kw]])
        full_i = np.concatenate([oi_w, wi[kw]])
        full_uv = np.concatenate([uv_w, wuv[kw]])
        mask = np.zeros(len(full_c), bool)
        mask[len(oc_w) :] = True
        _, q_n, t_n, _, res_n = run_ba(
            cam,
            q,
            tvec,
            pts.copy(),
            full_c,
            full_i,
            full_uv,
            wh,
            f0,
            protected=mask,
            opt_f=False,
        )
        n_surv = int((res_n[len(oc_w) :] < TRIM_PX).sum())
        state["anchors"] += 1
        state["wall"] = state.get("wall", 0.0) + time.perf_counter() - t0
        log(
            f"    [widen-anchor {state['anchors']}: {len(sel)} tracks, "
            f"{len(records)} localized, +{len(acc)} accepted "
            f"(total {len(state['obs'])} wide obs, {n_surv} <{TRIM_PX:.0f}px), "
            f"{time.perf_counter() - t0:.0f}s]"
        )
        rvec_n = Rotation.from_quat(np.asarray(q_n)[:, [1, 2, 3, 0]]).as_rotvec()
        return rvec_n, np.asarray(t_n)

    return hook


def stage_b(ws, out_dir, budget=80, do_compare=True):
    """Grow the reconstruction from the fast-pinhole seed with widening at
    anchors, then a final protected BA; score like Stage A."""
    import os

    log(f"== stage B (widen during growth): {ws}")
    t_start = time.perf_counter()
    out_dir = out_dir / ws.name
    out_dir.mkdir(parents=True, exist_ok=True)
    argv_save = sys.argv
    sys.argv = [sys.argv[0], str(ws)]
    import exp_hier_ba as hb

    sys.argv = argv_save
    hb.WS = Path(ws)
    hb.REF = None
    os.environ["SFMTOOL_BA_WINDOW"] = "40"
    os.environ["SFMTOOL_ANCHOR_EVERY"] = "4"
    os.environ["SFMTOOL_PHOTO_ANCHOR"] = "1"
    # experiment seeds carry confidence flags; force completion (we are
    # measuring the widen-anchor path, not gating on seed trust)
    os.environ.setdefault("SFMTOOL_FORCE", "1")
    os.environ.setdefault("SFMTOOL_COMPLETE_MAX_CL", "15000")
    os.environ.setdefault("SFMTOOL_FRAC_DIAG", "1")
    data = hb.load_clusters()
    all_c, all_i = data["obs_c"], data["obs_i"]
    n_img, n_cl = data["n_img"], data["n_cl"]
    dims = np.asarray(data["dims"], dtype=np.float64)
    hb._CAM_WH = tuple(data["dims"][0])
    all_u = data["obs_uv"]
    ba_cl = data["adm_rank"] < hb.MAX_CLUSTERS
    ds_all = np.sqrt(np.maximum(np.abs(np.linalg.det(data["obs_warp"])), 1e-12))
    ref_img = np.full(n_cl, -1, np.int64)
    ref_img[all_c[data["obs_ref"]]] = all_i[data["obs_ref"]]
    active_cl = np.zeros(n_cl, bool)
    state = dict(obs=[], anchors=0)
    hb._photo_anchor_ba = make_widen_anchor(ws, hb, state, budget)
    seed = ws / "fast-pinhole.json"
    r = hb.external_seed_complete(
        data,
        str(seed),
        active_cl,
        all_c,
        all_i,
        all_u,
        n_img,
        n_cl,
        ds_all,
        ref_img,
        dims,
        ba_cl,
    )
    if r is None:
        log("  stage B: seed refused (flagged)")
        return None
    f, rvec, tvec, pts, posed, ok, keep, res = r
    log(
        f"  growth done: {int(posed.sum())}/{n_img} posed, f={f:.1f}, "
        f"{state['anchors']} widen-anchors, {len(state['obs'])} wide obs"
    )
    names = list(data["names"])
    q = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
    wh = hb._CAM_WH
    cam = p1.make_cam(f, *wh)
    # final protected BA over the ACTIVE observation set + wide obs
    act = active_cl[all_c]
    live = act & posed[all_i]
    oc, oi, uv = all_c[live], all_i[live], all_u[live]
    order = np.argsort(oc, kind="stable")
    oc, oi, uv = oc[order], oi[order], uv[order]
    pts_t = p1.triangulate(
        dict(obs_c=oc, obs_i=oi, obs_uv=uv, n_cl=n_cl),
        Rotation.from_rotvec(rvec).as_matrix(),
        tvec,
        cam,
    )
    wc = np.asarray([o["track"] for o in state["obs"]], np.int64)
    wi = np.asarray([o["view"] for o in state["obs"]], np.int64)
    wuv = np.asarray([o["uv"] for o in state["obs"]], np.float64).reshape(-1, 2)
    kw = posed[wi] if len(wi) else np.zeros(0, bool)
    full_c = np.concatenate([oc, wc[kw]])
    full_i = np.concatenate([oi, wi[kw]])
    full_uv = np.concatenate([uv, wuv[kw]])
    mask = np.zeros(len(full_c), bool)
    mask[len(oc) :] = True
    f_x, q_x, t_x, p_x, res_x = run_ba(
        cam, q, tvec, pts_t, full_c, full_i, full_uv, wh, f, protected=mask
    )
    pi = np.nonzero(posed)[0]
    names_p = [names[j] for j in pi]
    evals = {
        "growth": p1.eval_vs_gt(ws, names_p, q[pi], tvec[pi], f),
        "final_prot": p1.eval_vs_gt(ws, names_p, q_x[pi], t_x[pi], f_x),
    }
    log(f"  growth vs GT: {evals['growth']}")
    log(f"  final_prot vs GT: {evals['final_prot']}")
    # save via the bootstrap sfmr's name order (unposed keep bootstrap poses)
    frag_scores = {}
    if do_compare:
        bt = SfmrReconstruction.load(ws / "sfmr" / "bootstrap-pinhole.sfmr")
        bnames = list(bt.image_names)
        b_of = {n: j for j, n in enumerate(bnames)}
        for tag, (q_s, t_s) in {
            "growthB": (q, tvec),
            "finalB": (q_x, t_x),
        }.items():
            q_all = np.array(bt.quaternions_wxyz, dtype=np.float64, copy=True)
            t_all = np.array(bt.translations, dtype=np.float64, copy=True)
            n_hit = 0
            for j in pi:
                k = b_of.get(names[j])
                if k is None:
                    continue
                q_all[k], t_all[k] = q_s[j], t_s[j]
                n_hit += 1
            arm = bt.clone_with_changes(
                quaternions_wxyz=np.ascontiguousarray(q_all),
                translations=np.ascontiguousarray(t_all),
            )
            path = ws / "sfmr" / f"widen2-{tag}.sfmr"
            arm.save(path, operation=f"widen2-{tag}")
            frag_scores[tag] = run_compare(gt_path_of(ws), path, out_dir, tag)
            log(
                f"  fragments[{tag}] ({n_hit} posed mapped): {fmt_frag(frag_scores[tag])}"
            )
    out = dict(
        ws=str(ws),
        stage="B",
        n_posed=int(posed.sum()),
        n_img=int(n_img),
        focal=float(f_x),
        n_anchors=int(state["anchors"]),
        n_wide_obs=int(len(state["obs"])),
        n_wide_surviving=int((res_x[len(oc) :] < TRIM_PX).sum()),
        evals=evals,
        fragment_scores=frag_scores,
        wall_s=float(time.perf_counter() - t_start),
    )
    (out_dir / "widen2-stageB.json").write_text(json.dumps(out, indent=2))
    log(f"  stage B done in {out['wall_s']:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspaces", nargs="+", type=Path)
    ap.add_argument("--max-tracks", type=int, default=500)
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--stage-b", action="store_true")
    ap.add_argument("--stage-b-budget", type=int, default=80)
    ap.add_argument("--out", type=Path, default=Path("widen2-out"))
    args = ap.parse_args()
    for ws in args.workspaces:
        try:
            if args.stage_b:
                stage_b(
                    ws,
                    args.out,
                    budget=args.stage_b_budget,
                    do_compare=not args.no_compare,
                )
            else:
                run_workspace(
                    ws, args.max_tracks, args.out, do_compare=not args.no_compare
                )
        except Exception as ex:  # keep the sweep going
            import traceback

            traceback.print_exc()
            log(f"{ws.name}\tERROR {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
