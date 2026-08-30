# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 experiment: photometric track WIDENING.

THESIS: reconstruction drift is caused by NARROW tracks -- small ANGULAR span
of observing rays (a long video track can span 2 deg: long by count, narrow by
parallax -> depth/scale ill-conditioned).  A triangulated patch with a normal
can be LOCALIZED photometrically (warp-compensated ZNCC) in posed frames SIFT
never matched, manufacturing wide-parallax observations.  This script widens
narrow tracks post-hoc on a completed bootstrap reconstruction and measures
whether the long-range constraints reduce drift.

Pipeline (per workspace):
  1. Load poses/focal from sfmr/bootstrap-pinhole.sfmr and cluster
     observations from the matching *-clusters-patches.matches file (status
     0/1 members, valid reference, span >= 2, best 15000 by span -- the same
     universe the bootstrap completion optimized).
  2. Triangulate every cluster from the bootstrap poses; compute per-track
     angular span (max pairwise angle of the world rays camera-center ->
     point) and report the distribution.
  3. Select widening candidates: finite point, >= MIN_MEMBERS members, span
     < NARROW_DEG, member ZNCC above the data-derived floor; cap MAX_TRACKS
     evenly across the span-sorted qualifying list.
  4. Per selected track, find candidate extension views: posed frames not in
     the track whose frustum sees the point (in front, in bounds with margin)
     at a viewing angle vs the track mean ray LARGER than the current span
     (that is the widening) but below the grazing limit vs the patch normal.
     Up to CANDS_PER_BUCKET per angular bucket (0-5,5-10,10-20,20-40,40+ deg).
  5. Localize each track's patch in members + candidates via
     PatchCloud.from_tracks + localize_keypoints (keypoint scales from the
     .sift files; extent="feature_size"), chunked so no pyramid set exceeds
     VIEW_CAP images.
  6. Accept candidate-view localizations that pass a reprojection sanity gate
     (< ACCEPT_PX px from the point's projection at the current pose); report
     acceptance by angular-distance bucket.
  7. Re-run the staged native bundle_adjust (opt_f=True) from the bootstrap
     state in three variants -- control (original obs only), widened (+ the
     gated obs), widened_all (+ every localized candidate obs; BA's staged
     trim is the arbiter) -- and compare span distributions, camera errors vs
     the GT solve (global + piecewise-k8 center error, rotation median), and
     focal.

Usage:
  pixi run -e dev python scripts/exp_track_widen.py <ws> [<ws> ...]
      [--max-tracks 500] [--refine-normals] [--skip-ba] [--out DIR]
"""

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sfmtool._sfmtool.analysis import estimate_alignment_rs, triangulate_batch
from sfmtool._sfmtool.geometry import CameraIntrinsics, bundle_adjust
from sfmtool._sfmtool.io import read_matches, read_sift_partial
from sfmtool._sfmtool.patches import CameraViews, ImagePyramidSet, PatchCloud
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.sift.file import get_sift_path_for_image

ACTIVE_CAP = 15000  # cluster universe cap (matches SFMTOOL_COMPLETE_MAX_CL)
MIN_MEMBERS = 6  # member-count floor for widening candidates
NARROW_DEG = 5.0  # a track narrower than this is a widening candidate
GRAZE_DEG = 60.0  # candidate-ray grazing limit vs the patch normal
BUCKETS = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 180.0)]
CANDS_PER_BUCKET = 3  # candidate views offered per track per bucket
MEMBER_SUB = 12  # members fed to the localization consensus (evenly spread)
VIEW_CAP = 60  # max images per pyramid chunk (4K frames ~ 33 MB each)
ACCEPT_PX = 4.0  # reprojection sanity gate for new observations
NOOP_PX = 0.1  # a localization this close to the projection carries no signal
DEPTH_RIG_MAX = 3.0  # implied-depth cap (rig diameters) against infinity-like tracks
SPAN_NOISE_MULT = 10.0  # span >= this multiple of the angular noise (<=10% depth err)
TRIM_PX = 4.0  # final BA trim (same as the bootstrap)


def log(*a):
    print(*a, flush=True)


# -- Loading -----------------------------------------------------------------


def load_bootstrap(ws):
    """Poses + focal from bootstrap-pinhole.sfmr (canonical frame, -Z fwd)."""
    bt = SfmrReconstruction.load(ws / "sfmr" / "bootstrap-pinhole.sfmr")
    names = list(bt.image_names)
    q = np.asarray(bt.quaternions_wxyz, dtype=np.float64)
    t = np.asarray(bt.translations, dtype=np.float64)
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    cam_d = bt.cameras[0].to_dict()
    f0 = float(cam_d["parameters"]["focal_length"])
    return names, q, t, rot, f0


def make_cam(f, w, h):
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


def load_clusters(ws, posed_names):
    """Observation universe of the bootstrap completion.

    Picks the clusters-patches file with the best image overlap vs the
    bootstrap's posed set, keeps status-0/1 members of clusters with a valid
    reference and posed span >= 2, and caps to the best ACTIVE_CAP clusters
    by span (the same universe the completion optimized).  Returns
    pose-indexed observation arrays grouped by cluster, plus per-obs feature
    ids / matches image names (for .sift scale reads) and member ZNCC.
    """
    files = sorted(glob.glob(str(ws / "matches" / "*-clusters-patches.matches")))
    if not files:
        raise FileNotFoundError("no clusters-patches file")
    pose_set = set(posed_names)
    best, best_ov = None, -1
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
    usable = []  # (span, member row selection)
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
    order = sorted(range(len(usable)), key=lambda k: -usable[k][0])
    keep = sorted(order[:ACTIVE_CAP])
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
    )


# -- Geometry ----------------------------------------------------------------


def triangulate(cl, rot, t, cam):
    """Ray-midpoint triangulation of every cluster from the bootstrap poses."""
    pts = np.full((cl["n_cl"], 3), np.nan)
    oc, oi, uv = cl["obs_c"], cl["obs_i"], cl["obs_uv"]
    d_loc = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv)))
    dirs = np.einsum("nji,nj->ni", rot[oi], d_loc)
    centers = -np.einsum("nji,nj->ni", rot[oi], t[oi])
    uniq, counts = np.unique(oc, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    result = triangulate_batch(
        np.ascontiguousarray(dirs), np.ascontiguousarray(centers), offsets
    )
    good = counts >= 2
    pts[uniq[good]] = np.asarray(result["points"])[good]
    return pts


def track_stats(obs_c, obs_i, centers, pts, n_cl):
    """Per-track angular span (max pairwise ray angle, deg), mean world ray
    (unit, camera -> point), member count, and median member depth (distance
    camera center -> point).  NaN span for invalid points."""
    span = np.full(n_cl, np.nan)
    mean_ray = np.full((n_cl, 3), np.nan)
    count = np.zeros(n_cl, np.int64)
    depth = np.full(n_cl, np.nan)
    bounds = np.searchsorted(obs_c, np.arange(n_cl + 1))
    for c in range(n_cl):
        lo, hi = bounds[c], bounds[c + 1]
        count[c] = hi - lo
        p = pts[c]
        if hi - lo < 2 or not np.isfinite(p).all():
            continue
        r = p[None, :] - centers[obs_i[lo:hi]]
        nrm = np.linalg.norm(r, axis=1)
        ok = nrm > 1e-12
        if ok.sum() < 2:
            continue
        depth[c] = float(np.median(nrm[ok]))
        r = r[ok] / nrm[ok, None]
        cmin = np.clip((r @ r.T).min(), -1.0, 1.0)
        span[c] = np.degrees(np.arccos(cmin))
        m = r.sum(axis=0)
        mean_ray[c] = m / max(np.linalg.norm(m), 1e-12)
    return span, mean_ray, count, depth


def select_tracks(span, count, med_zncc, med_reproj, depth, rig_diam, f0, max_tracks):
    """Widening candidates: finite narrow tracks with enough members, a
    member-ZNCC at/above the data-derived floor (25th pct of valid tracks),
    a member reprojection median under the acceptance gate (a track the
    current geometry cannot reproject has no usable projection guidance),
    and an implied depth within DEPTH_RIG_MAX rig diameters -- narrowness
    must come from a short observing baseline, not from the point being
    (near-)infinitely far: an infinity-like track has no parallax at ANY
    baseline, its triangulated depth is fiction, and every candidate-angle /
    delta computation built on that depth is fiction too (0614's sky tracks
    localize with delta ~ 0 everywhere and then collapse under BA).
    Geometrically, widening to >= 20 deg needs depth <= rig_diam / (2
    tan 10 deg) ~ 2.8 rig diameters.

    The mirror-image degeneracy (0614): a track whose span is BELOW its own
    angular noise floor triangulates to an arbitrary depth -- on an open path
    the midpoint of near-parallel skew rays collapses onto the camera path
    (depth ~ 0), and every candidate angle computed from that depth is
    equally fictional (phantom 40-180 deg candidates on an open walk).
    Relative depth error ~ noise/span with the angular noise
    atan(member reproj px / f), so requiring span >= SPAN_NOISE_MULT * noise
    bounds the depth error to ~1/SPAN_NOISE_MULT.  Sampled evenly across the
    span-sorted qualifying list."""
    valid = np.isfinite(span)
    zfloor = float(np.nanpercentile(med_zncc[valid], 25)) if valid.any() else 0.0
    theta_noise = np.degrees(np.arctan(np.maximum(med_reproj, 1.0) / f0))
    qual = np.nonzero(
        valid
        & (count >= MIN_MEMBERS)
        & (span < NARROW_DEG)
        & (span >= SPAN_NOISE_MULT * theta_noise)
        & (med_zncc >= zfloor)
        & (med_reproj < ACCEPT_PX)
        & (depth <= DEPTH_RIG_MAX * rig_diam)
    )[0]
    qual = qual[np.argsort(span[qual])]
    if len(qual) > max_tracks:
        qual = qual[np.round(np.linspace(0, len(qual) - 1, max_tracks)).astype(int)]
    return qual, zfloor


def find_candidates(track_ids, obs_c, obs_i, pts, span, mean_ray, rot, t, cam, wh):
    """Per selected track: extension views (pose indexes) with their viewing
    angle vs the track mean ray.  A view qualifies if it is not a member, the
    point projects in front and in bounds (margin), its angle exceeds the
    track's span (widening) and stays below the grazing limit vs the patch
    normal (~ -mean_ray for a mean_viewing patch).  Up to CANDS_PER_BUCKET
    per angular bucket, spread within the bucket."""
    w, h = wh
    margin = 0.05 * min(w, h)
    n_img = len(rot)
    centers = -np.einsum("nij,ni->nj", rot, t)
    bounds = np.searchsorted(obs_c, np.arange(obs_c[-1] + 2))
    out = {}
    for c in track_ids:
        p = pts[c]
        members = np.unique(obs_i[bounds[c] : bounds[c + 1]])
        rays = p[None, :] - centers  # (n_img, 3)
        dist = np.linalg.norm(rays, axis=1)
        ok = dist > 1e-12
        rays = np.where(ok[:, None], rays / np.maximum(dist, 1e-12)[:, None], 0.0)
        ang = np.degrees(
            np.arccos(np.clip(rays @ mean_ray[c], -1.0, 1.0))
        )  # vs mean ray == incidence vs the mean_viewing normal
        x_cam = np.einsum("nij,nj->ni", rot, np.broadcast_to(p, (n_img, 3))) + t
        uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
        infront = x_cam[:, 2] < 0  # canonical camera looks along -Z
        inb = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= margin)
            & (uv[:, 0] <= w - margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 1] <= h - margin)
        )
        cand = ok & infront & inb & (ang > max(span[c], 1e-3)) & (ang < GRAZE_DEG)
        cand[members] = False
        cidx = np.nonzero(cand)[0]
        if len(cidx) == 0:
            continue
        picked = []
        for blo, bhi in BUCKETS:
            b = cidx[(ang[cidx] >= blo) & (ang[cidx] < bhi)]
            if len(b) == 0:
                continue
            b = b[np.argsort(ang[b])]
            k = min(CANDS_PER_BUCKET, len(b))
            picked.extend(b[np.round(np.linspace(0, len(b) - 1, k)).astype(int)])
        out[int(c)] = dict(
            views=np.asarray(sorted(set(picked)), np.int64),
            ang=ang,
            proj=uv,
        )
    return out


# -- Localization ------------------------------------------------------------


class SiftScaleCache:
    """Per-image keypoint scales (norm of the affine shape's first column --
    the value extent="feature_size" reads from the .sift files)."""

    def __init__(self, ws):
        self.ws = ws
        self.cache = {}

    def get(self, img_name, feat_ids):
        need = int(np.max(feat_ids)) + 1
        have = self.cache.get(img_name)
        if have is None or len(have) < need:
            a = read_sift_partial(get_sift_path_for_image(self.ws / img_name), need)[
                "affine_shapes"
            ].astype(np.float64)
            self.cache[img_name] = np.hypot(a[:, 0, 0], a[:, 1, 0])
        return self.cache[img_name][feat_ids]


def chunk_tracks(track_ids, obs_c, obs_i, cands, bounds):
    """Greedy chunks: consecutive selected tracks (span-ordered) whose union
    of member-subset + candidate views stays within VIEW_CAP."""
    chunks, cur, cur_views = [], [], set()
    for c in track_ids:
        if int(c) not in cands:
            continue
        members = np.unique(obs_i[bounds[c] : bounds[c + 1]])
        if len(members) > MEMBER_SUB:
            members = members[
                np.round(np.linspace(0, len(members) - 1, MEMBER_SUB)).astype(int)
            ]
        views = set(members.tolist()) | set(cands[int(c)]["views"].tolist())
        if cur and len(cur_views | views) > VIEW_CAP:
            chunks.append((cur, sorted(cur_views)))
            cur, cur_views = [], set()
        cur.append((int(c), members))
        cur_views |= views
    if cur:
        chunks.append((cur, sorted(cur_views)))
    return chunks


def localize_widen(
    ws, names, q, t, cam, cl, pts, cands, track_ids, refine_normals=False
):
    """Localize each selected track's patch in member + candidate views,
    chunked to VIEW_CAP-image pyramid sets.  Returns per-candidate-view
    records (track, pose idx, angle, uv, reproj delta px, loo zncc)."""
    obs_c, obs_i = cl["obs_c"], cl["obs_i"]
    bounds = np.searchsorted(obs_c, np.arange(cl["n_cl"] + 1))
    scales_cache = SiftScaleCache(ws)
    # prewarm: one partial .sift read per image at the global max feature id
    sel_obs = np.concatenate(
        [np.arange(bounds[c], bounds[c + 1]) for c in track_ids if int(c) in cands]
    )
    for img in np.unique(cl["obs_img"][sel_obs]):
        m = sel_obs[cl["obs_img"][sel_obs] == img]
        scales_cache.get(str(img), cl["obs_feat"][m])
    chunks = chunk_tracks(track_ids, obs_c, obs_i, cands, bounds)
    log(
        f"  localization: {sum(len(c[0]) for c in chunks)} tracks in {len(chunks)} chunks"
    )
    records = []
    for ci, (tracks, views) in enumerate(chunks):
        t0 = time.perf_counter()
        vmap = {v: k for k, v in enumerate(views)}
        cview = CameraViews([cam], q[views], t[views])
        # chunk-local tracks: member observations only, grouped by point
        tp, ti, tuv, tsc = [], [], [], []
        pos = []
        view_sets = {}
        member_views = []
        for local_pid, (c, members_sub) in enumerate(tracks):
            lo, hi = bounds[c], bounds[c + 1]
            sel = np.nonzero(np.isin(obs_i[lo:hi], views))[0] + lo
            tp.extend([local_pid] * len(sel))
            ti.extend(vmap[int(v)] for v in obs_i[sel])
            tuv.extend(cl["obs_uv"][sel])
            for s in sel:
                tsc.append(
                    float(
                        scales_cache.get(
                            str(cl["obs_img"][s]), np.asarray([cl["obs_feat"][s]])
                        )[0]
                    )
                )
            pos.append(pts[c])
            mset = {vmap[int(v)] for v in members_sub if int(v) in vmap}
            cset = {vmap[int(v)] for v in cands[c]["views"]}
            view_sets[local_pid] = sorted(mset | cset)
            member_views.append(mset)
        cloud = PatchCloud.from_tracks(
            cview,
            np.ascontiguousarray(np.c_[np.asarray(pos), np.ones(len(pos))]),
            np.asarray(tp, np.uint32),
            np.asarray(ti, np.uint32),
            keypoint_scales=np.asarray(tsc, np.float64),
            normal="mean_viewing",
            extent="feature_size",
            extent_value=2.5,
        )
        imgs = [
            np.ascontiguousarray(cv2.imread(str(ws / names[v]), cv2.IMREAD_COLOR))
            for v in views
        ]
        pyrset = ImagePyramidSet(cview, imgs)
        if refine_normals:
            # view-subset refinement over K=8 members, evenly spread
            vind = []
            for k in range(len(tracks)):
                mv = sorted(member_views[k])
                if len(mv) > 8:
                    mv = [
                        mv[j]
                        for j in np.round(np.linspace(0, len(mv) - 1, 8)).astype(int)
                    ]
                vind.append(mv)
            cloud.refine_normals(cview, pyrset, view_indices=vind)
        results = cloud.localize_keypoints(
            cview,
            pyrset,
            view_sets=view_sets,
            max_shift_px=60.0,
            search=12.0,
            min_relative_zncc=0.6,
        )
        for r in results:
            pid = int(r["point_index"])
            c, _members = tracks[pid]
            kept = np.asarray(r["views"])
            kps = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 2)
            loo = np.asarray(r["loo_zncc"], dtype=np.float64)
            for k, v in enumerate(kept):
                gv = views[int(v)]
                if int(v) in member_views[pid]:
                    continue  # existing member -- keep the SIFT observation
                delta = float(np.linalg.norm(kps[k] - cands[c]["proj"][gv]))
                records.append(
                    dict(
                        track=int(c),
                        view=int(gv),
                        ang=float(cands[c]["ang"][gv]),
                        uv=kps[k].tolist(),
                        delta_px=delta,
                        loo_zncc=float(loo[k]),
                    )
                )
        log(
            f"    chunk {ci + 1}/{len(chunks)}: {len(tracks)} tracks, "
            f"{len(views)} views, {time.perf_counter() - t0:.1f}s"
        )
    return records, chunks


# -- Bundle adjustment + evaluation ------------------------------------------


def run_ba(cam0, q, t, pts, obs_c, obs_i, uv, wh, f0):
    """Staged native BA (opt_f), same schedule shape as the bootstrap."""
    pxs = max(1.0, float(np.hypot(*wh)) / 550.6)
    order = np.lexsort((obs_i, obs_c))
    out = bundle_adjust(
        make_cam(f0, *wh),
        np.ascontiguousarray(q),
        np.ascontiguousarray(t),
        np.ascontiguousarray(pts),
        np.ascontiguousarray(uv[order]),
        obs_i[order].astype(np.uint32),
        obs_c[order].astype(np.uint32),
        opt_f=True,
        schedule=[(50.0 * pxs, 5.0 * pxs), (12.0 * pxs, 2.0 * pxs), (TRIM_PX, 1.0)],
        max_iters=60,
        min_track=2,
        min_obs=12,
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


def eval_vs_gt(ws, names, q, t, f_est):
    """Rotation median, global + piecewise-k8 center error vs the GT solve
    (first *-solve-*.sfmr), focal error.  Piecewise chunks follow the
    name-sorted (temporal) order."""
    gt_f = sorted(
        p for p in (ws / "sfmr").glob("*-solve-*.sfmr") if "bootstrap" not in p.name
    )[0]
    gt = SfmrReconstruction.load(gt_f)
    gnames = list(gt.image_names)
    common = sorted(set(names) & set(gnames))
    ei = np.array([names.index(n) for n in common])
    gi = np.array([gnames.index(n) for n in common])
    r_est = Rotation.from_quat(q[ei][:, [1, 2, 3, 0]]).as_matrix()
    rg = Rotation.from_quat(
        np.asarray(gt.quaternions_wxyz)[gi][:, [1, 2, 3, 0]]
    ).as_matrix()
    tg = np.asarray(gt.translations)[gi]
    ce = -np.einsum("nij,ni->nj", r_est, t[ei])
    cg = -np.einsum("nij,ni->nj", rg, tg)
    u_svd, _s, vt = np.linalg.svd(np.einsum("nji,njk->ik", r_est, rg))
    if np.linalg.det(u_svd @ vt) < 0:
        u_svd[:, 2] *= -1.0
    g = u_svd @ vt
    rot_err = Rotation.from_matrix(
        np.einsum("nij,nkj->nik", rg, np.einsum("nij,jk->nik", r_est, g))
    ).magnitude() * (180 / np.pi)
    diam = np.max(np.linalg.norm(cg[:, None] - cg[None, :], axis=2))

    def aerr(a, b):
        tf = estimate_alignment_rs(np.ascontiguousarray(a), np.ascontiguousarray(b))
        return (
            np.linalg.norm(tf.apply_to_points(np.ascontiguousarray(a)) - b, axis=1)
            / diam
        )

    ge = aerr(ce, cg)
    pw = []
    for ch in np.array_split(np.arange(len(common)), 8):
        if len(ch) >= 8:
            pw.append(aerr(ce[ch], cg[ch]))
    pe = np.concatenate(pw) if pw else ge
    f_gt = float(np.mean(gt.cameras[0].focal_lengths))
    return dict(
        n_common=len(common),
        rot_med=float(np.median(rot_err)),
        global_pct=float(100 * np.median(ge)),
        piecewise_pct=float(100 * np.median(pe)),
        focal_err_pct=float(100 * (f_est / f_gt - 1.0)),
    )


def span_summary(span, sel=None):
    s = span[np.isfinite(span)] if sel is None else span[sel][np.isfinite(span[sel])]
    if len(s) == 0:
        return {}
    pct = np.percentile(s, [10, 25, 50, 75, 90])
    return dict(
        n=int(len(s)),
        p10=float(pct[0]),
        p25=float(pct[1]),
        med=float(pct[2]),
        p75=float(pct[3]),
        p90=float(pct[4]),
    )


def bucket_of(ang):
    for k, (lo, hi) in enumerate(BUCKETS):
        if lo <= ang < hi:
            return k
    return len(BUCKETS) - 1


# -- Main --------------------------------------------------------------------


def run_workspace(ws, max_tracks, refine_normals, out_dir, skip_ba=False):
    log(f"== {ws}")
    t_start = time.perf_counter()
    names, q, t, rot, f0 = load_bootstrap(ws)
    cl = load_clusters(ws, names)
    w, h = cl["wh"]
    cam = make_cam(f0, w, h)
    centers = -np.einsum("nij,ni->nj", rot, t)
    pts = triangulate(cl, rot, t, cam)

    # projection sanity: member obs must reproject tightly at the current pose
    oc, oi = cl["obs_c"], cl["obs_i"]
    res_obs = np.full(len(oc), np.nan)
    fin = np.isfinite(pts[oc]).all(axis=1)
    x_cam = np.einsum("nij,nj->ni", rot[oi[fin]], pts[oc[fin]]) + t[oi[fin]]
    proj = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam)))
    res_obs[fin] = np.linalg.norm(proj - cl["obs_uv"][fin], axis=1)
    log(
        f"  universe: {cl['n_cl']} clusters, {len(oc)} obs; member reproj med "
        f"{np.nanmedian(res_obs):.2f} px (wide clusters drift-inconsistent by design)"
    )

    span, mean_ray, count, depth = track_stats(oc, oi, centers, pts, cl["n_cl"])
    med_zncc = np.full(cl["n_cl"], np.nan)
    med_reproj = np.full(cl["n_cl"], np.inf)
    bounds = np.searchsorted(oc, np.arange(cl["n_cl"] + 1))
    for c in range(cl["n_cl"]):
        med_zncc[c] = np.median(cl["obs_zncc"][bounds[c] : bounds[c + 1]])
        r = res_obs[bounds[c] : bounds[c + 1]]
        if np.isfinite(r).any():
            med_reproj[c] = np.nanmedian(r)
    log(f"  span (all universe tracks): {span_summary(span)}")
    rig_diam = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    narrow = np.isfinite(span) & (span < NARROW_DEG) & np.isfinite(depth)
    if narrow.any():
        dq = np.percentile(depth[narrow] / rig_diam, [25, 50, 75, 90])
        log(
            f"  narrow-track implied depth / rig diam: p25 {dq[0]:.2f} "
            f"med {dq[1]:.2f} p75 {dq[2]:.2f} p90 {dq[3]:.2f} "
            f"(gate at {DEPTH_RIG_MAX})"
        )

    sel, zfloor = select_tracks(
        span, count, med_zncc, med_reproj, depth, rig_diam, f0, max_tracks
    )
    log(
        f"  selected {len(sel)} widening tracks "
        f"(count>={MIN_MEMBERS}, span<{NARROW_DEG} deg, zncc>={zfloor:.2f}, "
        f"depth<={DEPTH_RIG_MAX}*rig, span>={SPAN_NOISE_MULT}x noise floor)"
    )
    log(f"  span (selected): {span_summary(span, sel)}")

    cands = find_candidates(sel, oc, oi, pts, span, mean_ray, rot, t, cam, (w, h))
    offered = [r for c in cands.values() for r in c["ang"][c["views"]]]
    n_off_bucket = np.zeros(len(BUCKETS), np.int64)
    for a in offered:
        n_off_bucket[bucket_of(a)] += 1
    log(
        f"  candidates: {len(offered)} views over {len(cands)} tracks; "
        f"per bucket {n_off_bucket.tolist()}"
    )

    records, _ = localize_widen(
        ws, names, q, t, cam, cl, pts, cands, sel, refine_normals=refine_normals
    )
    kept_b = np.zeros(len(BUCKETS), np.int64)
    acc_b = np.zeros(len(BUCKETS), np.int64)
    noop_b = np.zeros(len(BUCKETS), np.int64)
    deltas_b = [[] for _ in BUCKETS]
    accepted = []
    for r in records:
        b = bucket_of(r["ang"])
        kept_b[b] += 1
        deltas_b[b].append(r["delta_px"])
        if r["delta_px"] < NOOP_PX:
            noop_b[b] += 1  # indistinguishable from the projection: no signal
        if r["delta_px"] < ACCEPT_PX:
            acc_b[b] += 1
            accepted.append(r)
    log(
        "  bucket\toffered\tlocalized\taccepted(<%.0fpx)\tnoop(<%.1fpx)"
        "\tdelta_px p50/p90" % (ACCEPT_PX, NOOP_PX)
    )
    curve = []
    for k, (lo, hi) in enumerate(BUCKETS):
        d = np.asarray(deltas_b[k])
        dp = f"{np.percentile(d, 50):.1f}/{np.percentile(d, 90):.1f}" if len(d) else "-"
        log(
            f"  {lo:.0f}-{hi:.0f}\t{n_off_bucket[k]}\t{kept_b[k]}\t{acc_b[k]}"
            f"\t{noop_b[k]}\t{dp}"
        )
        curve.append(
            dict(
                lo=lo,
                hi=hi,
                offered=int(n_off_bucket[k]),
                localized=int(kept_b[k]),
                accepted=int(acc_b[k]),
                noop=int(noop_b[k]),
                delta_p50=float(np.percentile(d, 50)) if len(d) else None,
                delta_p90=float(np.percentile(d, 90)) if len(d) else None,
            )
        )

    # -- BA: control (original obs) vs treatment (+ widened obs).  Two
    # treatment flavours: the brief's < ACCEPT_PX reprojection gate, and
    # every localized candidate obs (delta p50 sits well above the gate --
    # if that shift is the drift-corrective signal rather than photometric
    # noise, only the ungated variant can deliver it; BA's own staged trim
    # is the arbiter).
    uv_all = cl["obs_uv"]

    def new_obs(recs):
        c = np.asarray([r["track"] for r in recs], np.int64)
        i = np.asarray([r["view"] for r in recs], np.int64)
        u = np.asarray([r["uv"] for r in recs], np.float64).reshape(-1, 2)
        return c, i, u

    log(
        f"  accepted {len(accepted)} new obs (<{ACCEPT_PX:.0f}px) on "
        f"{len({r['track'] for r in accepted})} tracks; "
        f"{len(records)} localized total on {len({r['track'] for r in records})}"
    )

    evals = {"bootstrap": eval_vs_gt(ws, names, q, t, f0)}
    log(f"  bootstrap vs GT: {evals['bootstrap']}")
    if skip_ba:
        log("  --skip-ba: stopping after the acceptance curve")
        return None

    def span_after(oc2, oi2, q_x, t_x, p_x, res_x):
        rot_x = Rotation.from_quat(q_x[:, [1, 2, 3, 0]]).as_matrix()
        centers_x = -np.einsum("nij,ni->nj", rot_x, t_x)
        keep = res_x < TRIM_PX
        o = np.lexsort((oi2, oc2))
        s, _, _, _ = track_stats(
            oc2[o][keep[o]], oi2[o][keep[o]], centers_x, p_x, cl["n_cl"]
        )
        return s

    variants = {
        "control_ba": (oc, oi, uv_all, 0),
        "widened_ba": (
            *[np.concatenate([a, b]) for a, b in zip((oc, oi), new_obs(accepted)[:2])],
            np.concatenate([uv_all, new_obs(accepted)[2]]),
            len(accepted),
        ),
        "widened_all_ba": (
            *[np.concatenate([a, b]) for a, b in zip((oc, oi), new_obs(records)[:2])],
            np.concatenate([uv_all, new_obs(records)[2]]),
            len(records),
        ),
    }
    spans_out = {}
    for tag, (oc2, oi2, uv2, n_new) in variants.items():
        t0 = time.perf_counter()
        f_x, q_x, t_x, p_x, res_x = run_ba(
            cam, q, t, pts.copy(), oc2, oi2, uv2, (w, h), f0
        )
        evals[tag] = eval_vs_gt(ws, names, q_x, t_x, f_x)
        surv = int((res_x[len(oc) :] < TRIM_PX).sum()) if n_new else 0
        evals[tag]["n_new_obs"] = int(n_new)
        evals[tag]["n_new_surviving"] = surv
        s_after = span_after(oc2, oi2, q_x, t_x, p_x, res_x)
        spans_out[tag] = dict(
            selected=span_summary(s_after, sel), all=span_summary(s_after)
        )
        log(f"  {tag} ({time.perf_counter() - t0:.0f}s) vs GT: {evals[tag]}")
        log(f"    span after (selected): {spans_out[tag]['selected']}")
        log(f"    span after (all universe): {spans_out[tag]['all']}")
        # Save the arm's poses as a .sfmr next to the bootstrap so external
        # tooling (`sfm compare --fragments`) can score it.  Points stay the
        # bootstrap's (stale): the fragment decomposition uses camera poses
        # and the reference's scene scale only.
        bt = SfmrReconstruction.load(ws / "sfmr" / "bootstrap-pinhole.sfmr")
        arm = bt.clone_with_changes(
            quaternions_wxyz=np.ascontiguousarray(q_x, dtype=np.float64),
            translations=np.ascontiguousarray(t_x, dtype=np.float64),
        )
        arm.save(
            ws / "sfmr" / f"track-widen-{tag}.sfmr", operation=f"track-widen-{tag}"
        )

    loo = np.asarray([r["loo_zncc"] for r in records], dtype=np.float64)
    loo_acc = np.asarray([r["loo_zncc"] for r in accepted], dtype=np.float64)
    out = dict(
        ws=str(ws),
        n_cl=int(cl["n_cl"]),
        n_obs=int(len(oc)),
        span_all=span_summary(span),
        span_selected=span_summary(span, sel),
        spans_after=spans_out,
        n_selected=int(len(sel)),
        zncc_floor=zfloor,
        rig_diam=rig_diam,
        depth_over_rig_selected=(
            np.percentile(depth[sel] / rig_diam, [25, 50, 75, 90]).tolist()
            if len(sel)
            else None
        ),
        acceptance_curve=curve,
        n_localized=int(len(records)),
        n_accepted=int(len(accepted)),
        loo_zncc_localized=(
            np.nanpercentile(loo, [25, 50, 75]).tolist() if len(loo) else None
        ),
        loo_zncc_accepted=(
            np.nanpercentile(loo_acc, [25, 50, 75]).tolist() if len(loo_acc) else None
        ),
        evals=evals,
        refine_normals=bool(refine_normals),
        wall_s=float(time.perf_counter() - t_start),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"track-widen-{ws.name}.json").write_text(json.dumps(out, indent=2))
    log(f"  done in {out['wall_s']:.0f}s -> {out_dir / f'track-widen-{ws.name}.json'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspaces", nargs="+", type=Path)
    ap.add_argument("--max-tracks", type=int, default=500)
    ap.add_argument("--refine-normals", action="store_true")
    ap.add_argument("--skip-ba", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    for ws in args.workspaces:
        try:
            run_workspace(
                ws, args.max_tracks, args.refine_normals, args.out, args.skip_ba
            )
        except Exception as ex:  # keep the sweep going
            import traceback

            traceback.print_exc()
            log(f"{ws.name}\tERROR {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
