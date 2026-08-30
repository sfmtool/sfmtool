# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment: pinhole-only coarse reconstruction from cluster patches.

Starting from a workspace holding images and a `*-clusters-patches.matches`
file (sift extraction -> cluster matching -> cluster-patches), and using only
a pinhole camera model, bootstrap a coarse 3D reconstruction and write it to
a `.sfmr` file — no COLMAP solver involved.

Pipeline:
  1. Load patch clusters; refined member positions are read directly from
     the stored affines' last column (`member_affines[k][:, 2]` holds the
     absolute keypoint position since .matches format version 4), and the
     image dimensions from the images section — no per-image .sift reads.
  2. Group images by cluster covisibility (shared-cluster counts) — no
     sequence order is assumed.  Affine (weak-perspective) ALS factorization
     of candidate seed groups (a single global factorization breaks on wide
     baselines) + Tomasi–Kanade metric upgrade, both reflection hypotheses.
  3. Seed a perspective solve on the best group (a small fixed-focal BA
     also resolves the reflection), then grow incrementally: the
     next-best-view image (most observations of valid points) is resected
     pose-only against the global structure (trimmed iterations, most-
     covisible posed poses as inits), new clusters are triangulated as
     they gain posed views, short global BAs run every few images.
  4. Steps 2–3 run per candidate focal on a small grid with f held FIXED —
     the focal is unobservable from a weak init (the residual decreases
     monotonically toward the affine limit), but with a converged geometry
     the inlier fraction peaks near the true focal.  The scan caps growth
     at ~20 images; the winner grows fully and its BA then releases f.
  5. Report reprojection stats and, when a reference solve exists in the
     workspace, camera errors after similarity alignment; save the result
     as `sfmr/bootstrap-pinhole.sfmr`.

Run: pixi run -e dev python scripts/exp_pinhole_bootstrap.py <workspace> [ref.sfmr]

The optional second argument names the reference solve to compare against
(it may live in another workspace, e.g. a full-sequence solve when
bootstrapping a frame subset — images are matched by workspace-relative
name).  Default: the first non-bootstrap .sfmr in the workspace.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sfmtool._sfmtool.geometry import (
    CameraIntrinsics,
    bundle_adjust as _native_ba,
    inlier_fraction as _inlier_fraction,
    refine_absolute_pose as _refine_absolute_pose,
    reprojection_residuals as _reprojection_residuals,
)

WS = Path(sys.argv[1] if len(sys.argv) > 1 else "e_seoul_ws")
REF = Path(sys.argv[2]) if len(sys.argv) > 2 else None
_T0 = time.perf_counter()
MIN_SPAN_BA = 2  # min distinct images for a cluster to become a point
MAX_CLUSTERS = int(os.environ.get("SFMTOOL_MAX_CLUSTERS", "10000"))  # BA-set size
F_GRID = [0.55, 0.7, 0.9, 1.2, 1.6]  # focal candidates, in units of max(w, h)
TRIM_PX = 4.0  # BA inter-round observation trim threshold
# Cluster ordering for the cap and the admission tiers: "cons" ranks by the
# stored warp-consistency residual (max over members, ascending — measured
# AUC 0.79-0.92 for junk prediction across the campaign datasets), "span" is
# the original highest-span-first ordering.
ORDER = os.environ.get("SFMTOOL_ORDER", "span")
# Resection init from warp-determinant depth ratios: each member warp's
# sqrt|det| predicts the point's depth in the new image from its depth in
# the (posed) reference image, giving camera-frame 3D points -> closed-form
# trimmed Kabsch pose init (no neighbor-pose inits needed when it works).
DEPTH_INIT = os.environ.get("SFMTOOL_DEPTH_INIT", "0") == "1"
# Diagnostics: trace per-resection inliers in growth; optionally disable the
# periodic growth BA to attribute damage between resection and BA.
TRACE = os.environ.get("SFMTOOL_TRACE", "0") == "1"
GROW_BA = os.environ.get("SFMTOOL_GROW_BA", "1") == "1"

# The whole script works in the CANONICAL camera frame (-Z forward, +Y up):
# poses are canonical world->camera, 3D points are world points, observations
# are FULL (un-centered) pixel coordinates, and every projection goes through
# the native `CameraIntrinsics` batch functions.  The world frame is the
# COLMAP-world gauge inherited from the affine factorization (irrelevant to
# the reprojection residuals and absorbed by the eval's similarity alignment);
# only the writer rotates it by W to reach the .sfmr canonical world.
_CAM_WH = None  # (w, h) of the shared pinhole; set in main() from the uniform image dims


def make_cam(f):
    """A SIMPLE_PINHOLE `CameraIntrinsics` at focal ``f`` (principal point at
    the image centre).  The images share one size (see main()), so one camera
    serves every projection; ``ray_to_pixel_batch`` / ``pixel_to_ray_batch``
    map canonical camera-space points <-> full pixels."""
    w, h = _CAM_WH
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


def reproj_res_one(cam, rvec_i, tvec_i, x_pts, uv, invalid=1e6):
    """(proj − obs) pixel residuals of one image's gathered points under a
    single canonical world->camera pose (rvec/tvec), via the native
    ``reprojection_residuals`` kernel.  Behind-camera observations get
    ``invalid`` on their x component (never an inlier), mirroring the old
    ``max(z, 1e-6)`` clamp.  Returns an (N, 2) array."""
    q = Rotation.from_rotvec(rvec_i).as_quat()[[3, 0, 1, 2]][None, :]
    n = len(uv)
    return _reprojection_residuals(
        cam,
        q,
        np.ascontiguousarray(tvec_i, dtype=np.float64)[None, :],
        np.ascontiguousarray(x_pts, dtype=np.float64),
        np.ascontiguousarray(uv, dtype=np.float64),
        np.zeros(n, np.uint32),
        np.arange(n, dtype=np.uint32),
        invalid,
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_clusters():
    """Patch clusters as flat observation arrays with refined positions.

    Everything geometric comes straight from the .matches file: image
    dimensions from the images section and member positions from the stored
    affines' last column (the absolute refined keypoint position).
    """
    from sfmtool._sfmtool.io import read_matches

    override = os.environ.get("SFMTOOL_MATCHES")
    patches = (
        [Path(override)]
        if override
        else sorted(WS.glob("matches/*-clusters-patches.matches"))
    )
    print(f"matches file: {patches[0]}")
    data = read_matches(patches[0])
    names = list(data["image_names"])
    dims = [(int(w), int(h)) for w, h in np.asarray(data["image_dims"])]

    starts = np.asarray(data["cluster_starts"])
    mi = np.asarray(data["member_images"])
    mf = np.asarray(data["member_features"])
    st = np.asarray(data["member_status"])
    refs = np.asarray(data["reference_members"])
    aff = np.asarray(data["member_affines"])
    cons = np.asarray(data["member_consistency_residual"], dtype=np.float64)

    # First pass: member selections, spans, and quality of every usable
    # cluster.  Quality is the worst (max) finite warp-consistency residual
    # over the selected members — lower is better; clusters where no member
    # entered the consistency fit rank last (inf).
    usable = []
    for c in range(len(starts) - 1):
        lo, hi = int(starts[c]), int(starts[c + 1])
        if refs[c] == np.iinfo(np.uint32).max:
            continue
        sel = np.nonzero((st[lo:hi] == 0) | (st[lo:hi] == 1))[0] + lo
        span = len(np.unique(mi[sel]))
        if span >= MIN_SPAN_BA:
            cq = cons[sel]
            cq = cq[np.isfinite(cq)]
            quality = float(cq.max()) if len(cq) else np.inf
            usable.append((span, c, sel, quality))

    # Admission order (best first) — used for both the cap and the tiers.
    # "span": highest span first (the original).  "cons": best consistency
    # first.  "cons_strat": best consistency first WITHIN each span stratum,
    # strata interleaved proportionally — any admission prefix then keeps
    # the span distribution (wide-baseline rigidity) while dropping the
    # worst-consistency clusters of every stratum first.
    spans = np.array([t[0] for t in usable])
    quals = np.array([t[3] for t in usable])
    cids = np.array([t[1] for t in usable])
    if ORDER == "span":
        order = np.lexsort((cids, -spans))
    elif ORDER == "cons":
        order = np.lexsort((cids, -spans, quals))
    elif ORDER == "cons_strat":
        p = np.empty(len(usable))
        for s in np.unique(spans):
            idx = np.nonzero(spans == s)[0]
            r = np.argsort(np.argsort(quals[idx], kind="stable"))
            p[idx] = (r + 0.5) / len(idx)
        order = np.lexsort((cids, quals, p))
    elif ORDER == "union":
        # Interleave the span backbone with the stratified-consistency
        # core: any prefix is half highest-span clusters (multi-view
        # rigidity + connectivity: south-building's quality lives here)
        # and half best-consistency-within-stratum (junk-poor: dino's
        # echo-free accuracy lives here).
        o_span = np.lexsort((cids, -spans))
        p = np.empty(len(usable))
        for s in np.unique(spans):
            idx = np.nonzero(spans == s)[0]
            r = np.argsort(np.argsort(quals[idx], kind="stable"))
            p[idx] = (r + 0.5) / len(idx)
        o_strat = np.lexsort((cids, quals, p))
        claimed = np.zeros(len(usable), bool)
        order = []
        ia = ib = 0
        while len(order) < len(usable):
            while ia < len(o_span) and claimed[o_span[ia]]:
                ia += 1
            if ia < len(o_span):
                claimed[o_span[ia]] = True
                order.append(o_span[ia])
            while ib < len(o_strat) and claimed[o_strat[ib]]:
                ib += 1
            if ib < len(o_strat):
                claimed[o_strat[ib]] = True
                order.append(o_strat[ib])
        order = np.asarray(order, dtype=np.int64)
    elif ORDER == "img_union":
        # Per-image top-K union: within each image, interleave its clusters
        # by span-desc and consistency-asc (the union rule per image), then
        # round-robin across images.  Any prefix gives EVERY image its
        # locally-best backbone + core — a global ordering starves images
        # whose matches are globally poor (dino img 52: 6 BA-set obs vs
        # median 375 under `union`), which are exactly the images that
        # need BA anchoring.
        by_img = {}
        for k, (_span, _c, sel, _q) in enumerate(usable):
            for im in np.unique(mi[sel]):
                by_img.setdefault(int(im), []).append(k)
        img_lists = {}
        for im, ks in by_img.items():
            ks = np.asarray(ks)
            o_span = ks[np.lexsort((cids[ks], -spans[ks]))]
            o_cons = ks[np.lexsort((cids[ks], quals[ks]))]
            seen = set()
            lst = []
            for a, b in zip(o_span, o_cons):
                for x in (int(a), int(b)):
                    if x not in seen:
                        seen.add(x)
                        lst.append(x)
            img_lists[im] = lst
        ptr = dict.fromkeys(img_lists, 0)
        claimed = np.zeros(len(usable), bool)
        order = []
        img_ids = sorted(img_lists)
        while len(order) < len(usable):
            progress = False
            for im in img_ids:
                lst = img_lists[im]
                p_i = ptr[im]
                while p_i < len(lst) and claimed[lst[p_i]]:
                    p_i += 1
                ptr[im] = p_i
                if p_i < len(lst):
                    claimed[lst[p_i]] = True
                    order.append(lst[p_i])
                    ptr[im] = p_i + 1
                    progress = True
            if not progress:
                break
        order = np.asarray(order, dtype=np.int64)
    elif ORDER == "cons_rr":
        # Per-image round-robin by quality: every image repeatedly claims
        # its best not-yet-claimed cluster, so any admission prefix gives
        # every image its locally-best clusters (balanced coverage — a
        # global quality or stratified cap can disconnect a chain-shaped
        # capture: south-building fragmented at 36/128 under cons_strat).
        by_img = {}
        for k, (_span, _c, sel, _q) in enumerate(usable):
            for im in np.unique(mi[sel]):
                by_img.setdefault(int(im), []).append(k)
        for im in by_img:
            by_img[im].sort(key=lambda k: (quals[k], cids[k]))
        ptr = dict.fromkeys(by_img, 0)
        claimed = np.zeros(len(usable), bool)
        order = []
        img_ids = sorted(by_img)
        while len(order) < len(usable):
            progress = False
            for im in img_ids:
                lst = by_img[im]
                p_i = ptr[im]
                while p_i < len(lst) and claimed[lst[p_i]]:
                    p_i += 1
                ptr[im] = p_i
                if p_i < len(lst):
                    claimed[lst[p_i]] = True
                    order.append(lst[p_i])
                    ptr[im] = p_i + 1
                    progress = True
            if not progress:
                break
        order = np.asarray(order, dtype=np.int64)
    else:
        raise SystemExit(f"unknown SFMTOOL_ORDER {ORDER!r}")

    # No admission cap: growth and triangulation see every usable cluster
    # (a capped set can disconnect a chain-shaped capture — south-building
    # fragmented at 36/128).  The ordering instead selects which clusters'
    # observations enter the BAs (the top MAX_CLUSTERS by adm_rank).
    pos = {int(k): i for i, k in enumerate(order)}
    keep_idx = sorted(range(len(usable)), key=lambda k: usable[k][1])

    obs_c, obs_i, obs_f, obs_uv, obs_warp, obs_ref = [], [], [], [], [], []
    adm_rank = []
    n_cl = 0
    for k in keep_idx:
        _span, c, sel, q = usable[k]
        adm_rank.append(pos[int(k)])
        # The affine's last column is the member's absolute refined keypoint
        # position; the 2x2 block is its ABSOLUTE affine shape S = W·S_ref
        # (.matches v5), with the reference row holding S_ref itself.  This
        # loader's consumers want the member<-reference WARP, so invert the
        # cluster's own reference row: W = S·S_ref⁻¹.  Clusters without a
        # reference never reach here (they are dropped as unusable above).
        s_ref_inv = np.linalg.inv(aff[int(refs[c]), :, :2])
        for k in sel:
            obs_c.append(n_cl)
            obs_i.append(int(mi[k]))
            obs_f.append(int(mf[k]))
            obs_uv.append(aff[k, :, 2])
            obs_warp.append(aff[k, :, :2] @ s_ref_inv)
            obs_ref.append(st[k] == 0)
        n_cl += 1

    return {
        "names": names,
        "dims": dims,
        "obs_c": np.asarray(obs_c),
        "obs_i": np.asarray(obs_i),
        "obs_f": np.asarray(obs_f),
        "obs_uv": np.asarray(obs_uv, dtype=np.float64),
        "obs_warp": np.asarray(obs_warp, dtype=np.float64),
        "obs_ref": np.asarray(obs_ref, dtype=bool),
        "adm_rank": np.asarray(adm_rank, dtype=np.int64),
        "refine_radius": float(data["refine_options"]["radius"]),
        "n_img": len(names),
        "n_cl": n_cl,
    }


# ── Covisibility grouping ────────────────────────────────────────────────────
#
# No sequence order is assumed: the natural grouping is how many clusters a
# pair of images shares.  High mutual covisibility implies nearby viewpoints,
# which is exactly what the weak-perspective factorization needs from a seed
# group, and the same counts drive the growth order and the resection inits.
# The counting and grouping live in the ClusterCovisibility binding; it is
# built from the loaded (span-filtered, capped) observation arrays rather
# than from the file so it sees exactly the clusters the bootstrap uses.


def build_covisibility(obs_c, obs_i, n_img, n_cl):
    """ClusterCovisibility over the loaded observation arrays."""
    from sfmtool._sfmtool.matching import ClusterCovisibility

    # obs_c is grouped by cluster in ascending order — derive the CSR starts.
    starts = np.searchsorted(obs_c, np.arange(n_cl + 1)).astype(np.uint32)
    return ClusterCovisibility.from_arrays(starts, obs_i.astype(np.uint32), n_img)


def kabsch_trimmed(x_world, x_cam, rounds=3, keep_q=0.6):
    """Rigid R, t with x_cam ~ R·x_world + t, trimmed to the best-fitting
    fraction each round (the depth predictions include junk members) — the
    native ``estimate_alignment_rs`` (estimate_scale=False for a rigid fit)."""
    from sfmtool._sfmtool.analysis import estimate_alignment_rs

    tf = estimate_alignment_rs(
        np.ascontiguousarray(x_world, dtype=np.float64),
        np.ascontiguousarray(x_cam, dtype=np.float64),
        rounds,
        keep_q,
        False,
    )
    qd = tf.to_dict()["rotation"]
    r_fit = Rotation.from_quat([qd["x"], qd["y"], qd["z"], qd["w"]]).as_matrix()
    return r_fit, np.asarray(tf.translation, dtype=np.float64)


# Per-image warp-depth coherence measured at resection acceptance
# (image, median |log(z_pose / z_warp_predicted)|, resection inlier frac).
_DEPTH_COH = []


def depth_init(s, obs_c, u, pts, rvec, tvec, posed, f0, i, aux):
    """Closed-form pose init for image ``i`` from warp-predicted depths.

    Each observation's sqrt|det warp| is the reference->member magnification,
    so the point's depth in image i is its depth in the (posed) reference
    image divided by it; backprojecting at those depths gives camera-frame
    points and a trimmed Kabsch solve gives the pose.  Returns (rvec0,
    tvec0, obs index array, predicted depths) or None when too few
    observations have a posed reference view."""
    ds, ref_img = aux[0], aux[1]
    si = np.nonzero(s)[0]
    rc = ref_img[obs_c[si]]
    okd = (rc >= 0) & (rc != i) & posed[np.maximum(rc, 0)]
    if okd.sum() < 8:
        return None
    x_w = pts[obs_c[si[okd]]]
    r_ref = Rotation.from_rotvec(rvec[rc[okd]]).as_matrix()
    # Canonical camera z is NEGATIVE in front, so in-front depths are < 0.
    z_ref = np.einsum("nij,nj->ni", r_ref, x_w)[:, 2] + tvec[rc[okd], 2]
    z_pred = z_ref / ds[si[okd]]
    good = z_pred < -1e-6
    if good.sum() < 8:
        return None
    sel = si[okd][good]
    # Backproject the full pixels to canonical unit rays and place each at its
    # predicted depth (ray z < 0, z_pred < 0 -> positive scale).
    rays = make_cam(f0).pixel_to_ray_batch(np.ascontiguousarray(u[sel]))
    x_cam = rays * (z_pred[good] / rays[:, 2])[:, None]
    r_fit, t_fit = kabsch_trimmed(x_w[good], x_cam)
    return Rotation.from_matrix(r_fit).as_rotvec(), t_fit, sel, z_pred[good]


def p3p_resect(uv, x_pts, f0, wh):
    """Minimal-sample absolute pose: RANSAC P3P over 2D-3D candidates.

    The trimmed-LS ``pose_refine`` needs a decent inlier fraction; a
    junk-match-dominated image (dino img 52: ~7-10% true 2D-3D pairs from a
    4x physical scale gap) defeats it, while minimal 3-point sampling finds
    the consensus routinely.  Uses the native Lambda Twist estimator
    (specs/core/geometry/absolute-pose.md); a tight 4 px threshold matches the
    bootstrap's TRIM_PX (a loose consensus is mostly junk on a
    wrong-match-heavy image and anchoring the verification BA on it drags
    the pose).  ``uv`` are full pixels.  Returns (rvec, tvec, inlier mask
    over the given obs) or None."""
    from sfmtool._sfmtool.geometry import estimate_absolute_pose

    ans = estimate_absolute_pose(
        np.ascontiguousarray(uv),
        np.ascontiguousarray(x_pts),
        camera=make_cam(f0),
        max_error_px=4.0,
        seed=0,
    )
    if ans is None:
        return None
    # The estimator already returns a canonical world-to-camera pose, which
    # is the frame the whole script works in — no flip.
    q = np.asarray(ans["quaternion_wxyz"])
    rv = Rotation.from_quat(q[[1, 2, 3, 0]]).as_rotvec()
    tv = np.asarray(ans["translation"], dtype=np.float64)
    return rv, tv, np.asarray(ans["inliers"], dtype=bool)


def pose_refine(uv, x_pts, rv0, tv0, f):
    """Pose-only resection of one image against known 3D points.

    Trimmed iterations (native ``refine_absolute_pose``): repeatedly refit L2
    on the best-fitting 60% of the observations, then a final refit on the
    < 3 px inliers.  A plain L2 warm-up is dragged by the junk observations'
    leverage, and a robust loss has near-zero gradient when every residual
    starts as a 100 px "outlier" — trimming from a decent init has neither
    problem.  Canonical world->camera pose in, canonical pose out."""
    q0 = Rotation.from_rotvec(rv0).as_quat()[[3, 0, 1, 2]]
    out = _refine_absolute_pose(
        make_cam(f),
        np.ascontiguousarray(uv, dtype=np.float64),
        np.ascontiguousarray(x_pts, dtype=np.float64),
        q0,
        np.ascontiguousarray(tv0, dtype=np.float64),
        5,  # trim rounds
        0.6,  # keep fraction
        3.0,  # final inlier px
    )
    q = np.asarray(out["quaternion_wxyz"])
    rv = Rotation.from_quat(q[[1, 2, 3, 0]]).as_rotvec()
    tv = np.asarray(out["translation"], dtype=np.float64)
    return rv, tv, float(out["inlier_fraction"])


def fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f):
    """DLT-triangulate clusters that lack a point but now have >= 2 posed
    observations.  Existing points are left untouched."""
    need = np.isnan(pts[:, 0])[obs_c] & posed[obs_i]
    if not need.any():
        return pts
    uniq, c2 = np.unique(obs_c[need], return_inverse=True)
    rot = Rotation.from_rotvec(rvec).as_matrix()
    newp = triangulate(c2, obs_i[need], u[need], rot, tvec, posed, len(uniq), f)
    out = pts.copy()
    out[uniq] = newp
    return out


def grow_loop(
    rvec, tvec, pts, posed, f0, obs_c, obs_i, u, n_img, n_cl, covis,
    max_images=None, aux=None, ba=None,
):
    """Next-best-view growth from an existing state (resumable: tier
    admission re-enters here after activating more clusters)."""
    grow_schedule = [(30.0, 3.0), (8.0, 1.5)]
    ba_every = max(3, min(8, n_img // 10))
    cam0 = make_cam(f0)
    # Local/windowed growth BA: refine only the most-recently-posed frontier
    # (SFMTOOL_BA_WINDOW frames) instead of every posed camera, so the BA cost
    # stays bounded as the reconstruction grows around a long orbit — the fix
    # that lets full next-best-view growth scale to thousands of frames (a
    # global growth BA is superlinear in posed frames and hangs at 2600).
    ba_window = int(os.environ.get("SFMTOOL_BA_WINDOW", "0"))
    # Periodic anchor BA: every ANCHOR_EVERY windowed BAs, refine a covisibility-
    # SPREAD subset of the posed frames instead of the frontier window. The
    # spread subset includes sequence-distant but space-near (loop-closing)
    # frames, so it pulls back the drift that a pure frontier window accumulates
    # around a long orbit.
    anchor_every = int(os.environ.get("SFMTOOL_ANCHOR_EVERY", "0"))
    posed_order = list(np.nonzero(posed)[0])
    ba_calls = [0]

    def run_grow_ba(rvec, tvec, pts):
        ba_calls[0] += 1
        live = posed[obs_i] & ~np.isnan(pts[obs_c, 0])
        if ba is not None:
            live &= ba
        anchor_fired = False
        if ba_window > 0:
            win = np.zeros(n_img, bool)
            if (
                anchor_every > 0
                and ba_calls[0] % anchor_every == 0
                and int(posed.sum()) > ba_window
            ):
                anchor_fired = True
                win[np.asarray(
                    covis.thin_to(min(int(posed.sum()), 150)), dtype=np.int64
                )] = True
                win &= posed
            elif len(posed_order) > ba_window:
                win[np.asarray(posed_order[-ba_window:], dtype=np.int64)] = True
            if win.any():
                live &= win[obs_i]
        # Photometric anchor: congeal the anchor subset's representative tracks
        # and run the anchor BA on the congealed observations.  Drift between
        # anchors is small, so the current poses sit INSIDE the photometric
        # search radius (unlike post-hoc correction) — each anchor re-couples
        # the loop on clean sub-pixel data before drift compounds.
        if anchor_fired and _PH_CTX is not None:
            ph = _photo_anchor_ba(
                rvec, tvec, pts, posed, win, f0, obs_c, obs_i, u, n_img, n_cl
            )
            if ph is not None:
                rvec, tvec = ph
                pts = fill_new_points(
                    np.full_like(pts, np.nan), obs_c, obs_i, u, rvec, tvec,
                    posed, f0,
                )
                return rvec, tvec, pts
        rot = Rotation.from_rotvec(rvec).as_matrix()
        out = bundle_adjust(
            obs_c[live], obs_i[live], u[live], rot, tvec, pts, f0,
            n_img, n_cl, opt_f=False, verbose=False, schedule=grow_schedule,
        )
        # The BA retriangulates only the observations it was given, wiping
        # every other cluster's point to NaN — refill them from the full
        # observation set at the updated poses, or the next-best-view count
        # sees only BA-set connectivity and growth stalls at its boundary.
        pts = fill_new_points(out[3], obs_c, obs_i, u, out[1], out[2], posed, f0)
        return out[1], out[2], pts

    def image_inl(i, rvec, tvec, pts):
        s = (obs_i == i) & ~np.isnan(pts[obs_c, 0])
        if not s.any():
            return 0.0
        res = reproj_res_one(cam0, rvec[i], tvec[i], pts[obs_c[s]], u[s])
        return _inlier_fraction(res, 3.0)

    since_ba = 0
    accepted_inl = []
    blocked = set()
    force_tried = set()
    ba_retry = True
    while max_images is None or posed.sum() < max_images:
        # Next-best-view: most observations of currently-valid points.
        cand = ~posed[obs_i] & ~np.isnan(pts[obs_c, 0])
        if not cand.any():
            break
        cnt = np.bincount(obs_i[cand], minlength=n_img)
        cnt_all = cnt.copy()
        for j in blocked:
            cnt[j] = 0
        i = int(np.argmax(cnt))
        if cnt[i] < 6:
            # Every eligible image is blocked or too weak.  One BA +
            # retriangulation pass may repair the frontier; afterwards the
            # blocked images get a second chance.  (Ranking-only scan
            # growth skips the retry like it skips force-accept: it does
            # not need completion and each retry costs a BA.)
            if blocked and ba_retry and max_images is None:
                ba_retry = False
                blocked.clear()
                rvec, tvec, pts = run_grow_ba(rvec, tvec, pts)
                pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f0)
                since_ba = 0
                continue
            # Verified force-accept: low-inlier resections are often
            # BA-recoverable (ungated seoul carried imgs 0-5 to <= 6°
            # final error this way).  Accept the strongest blocked
            # candidate WITHOUT building points from it, BA, then verify:
            # keep it only if its inliers rose into the accepted band,
            # else unpose it for good.  Damage is bounded to one BA whose
            # trims already suppress a single wrong camera.  Skipped in
            # capped (focal-scan) growth: the scan ranks candidates, it
            # does not need completion, and each trial costs a BA.
            if max_images is not None:
                break
            trial = [j for j in blocked if j not in force_tried and cnt_all[j] >= 6]
            if trial:
                j = max(trial, key=lambda k: cnt_all[k])
                force_tried.add(j)
                blocked.discard(j)
                sj = (obs_i == j) & ~np.isnan(pts[obs_c, 0])
                sj_idx = np.nonzero(sj)[0]
                # RANSAC P3P first: a junk-dominated image (wrong matches
                # from a scale gap) can hold a small true consensus that
                # trimmed-LS can never find (dino img 52: ~7-10% inliers,
                # P3P registers at 0.33 deg vs 0% from every LS init).
                consensus = None
                p3p = (
                    p3p_resect(u[sj], pts[obs_c[sj]], f0, aux[2][j])
                    if aux is not None and len(aux) > 2
                    else None
                )
                if p3p is not None and int(p3p[2].sum()) >= 12:
                    rv0, tv0, mask = p3p
                    # polish on the consensus subset only (mostly inliers)
                    rv, tv, inl_c = pose_refine(
                        u[sj][mask], pts[obs_c[sj]][mask], rv0, tv0, f0
                    )
                    best_j = (float(inl_c), rv, tv)
                    consensus = sj_idx[mask]
                    if TRACE:
                        print(f"    p3p img {j}: {int(mask.sum())}/"
                              f"{int(sj.sum())} RANSAC inliers, "
                              f"consensus refit inl {inl_c:.0%}")
                else:
                    posed_idx = np.nonzero(posed)[0].astype(np.uint32)
                    inits = covis.rank_by_covisibility(j, posed_idx)[:3]
                    best_j = None
                    for k in inits:
                        rv, tv, inl = pose_refine(
                            u[sj], pts[obs_c[sj]], rvec[k], tvec[k], f0
                        )
                        if best_j is None or inl > best_j[0]:
                            best_j = (inl, rv, tv)
                _, rvec[j], tvec[j] = best_j
                posed[j] = True
                posed_order.append(int(j))
                # A P3P-registered image's clusters are mostly junk matches
                # and mostly absent from the BA set, so the growth BA would
                # leave its pose anchored on almost nothing (dino img 52:
                # registered at 47% inl, then dragged to 10% by a BA that
                # held ~7 of its obs).  Anchor it on its own verified
                # evidence: consensus obs enter the BA set, its junk obs
                # leave it (restored if verification rejects).
                ba_saved = None
                if consensus is not None and ba is not None:
                    ba_saved = ba.copy()
                    # Promote the WHOLE consensus clusters (all members'
                    # obs), not just image j's rows: with only j's obs in
                    # the BA, each anchored point has a single participating
                    # observation, the inter-round retriangulation wipes it
                    # to NaN, and the image saves with a pose but zero kept
                    # features.  Then quarantine j's non-consensus (junk)
                    # obs out of the BA.
                    cons_cl = np.unique(obs_c[consensus])
                    ba[np.isin(obs_c, cons_cl)] = True
                    ba[sj_idx] = False
                    ba[consensus] = True
                rvec, tvec, pts = run_grow_ba(rvec, tvec, pts)
                since_ba = 0
                inl_after = image_inl(j, rvec, tvec, pts)
                bar = 0.35 * float(np.median(accepted_inl)) if accepted_inl else 0.0
                # Verification: the all-obs inlier bar, OR — for a
                # P3P-registered image whose observations are mostly wrong
                # MATCHES — survival of the P3P consensus set through the
                # BA (the registration claim is those obs, not the junk).
                surv = np.nan
                if consensus is not None:
                    res = reproj_res_one(
                        cam0, rvec[j], tvec[j], pts[obs_c[consensus]], u[consensus]
                    )
                    surv = _inlier_fraction(res, 3.0)
                if inl_after >= bar or (consensus is not None and surv >= 0.5):
                    accepted_inl.append(max(inl_after, bar))
                    pts = fill_new_points(
                        pts, obs_c, obs_i, u, rvec, tvec, posed, f0
                    )
                    ba_retry = True
                    blocked.clear()
                    if TRACE:
                        print(f"    force-accept img {j}: {best_j[0]:.0%} -> "
                              f"{inl_after:.0%} after BA"
                              f"{'' if consensus is None else f', consensus surv {surv:.0%}'}"
                              f" (kept)")
                else:
                    posed[j] = False
                    if posed_order and posed_order[-1] == int(j):
                        posed_order.pop()
                    if ba_saved is not None:
                        ba[:] = ba_saved
                    if TRACE:
                        print(f"    force-reject img {j}: {best_j[0]:.0%} -> "
                              f"{inl_after:.0%} after BA"
                              f"{'' if consensus is None else f', consensus surv {surv:.0%}'}"
                              f" (unposed)")
                continue
            break
        s = (obs_i == i) & ~np.isnan(pts[obs_c, 0])
        # Warp-depth Kabsch init (when enabled) — also feeds the
        # post-acceptance depth-coherence diagnostic below, so compute it
        # regardless of which resection path wins.
        di = depth_init(s, obs_c, u, pts, rvec, tvec, posed, f0, i, aux) \
            if aux is not None and DEPTH_INIT else None
        # Primary resection: minimal-sample RANSAC P3P over the 2D-3D
        # candidates, then a trimmed-LS polish on the consensus subset.
        # Minimal 3-point sampling finds the true pose without a from-init
        # warm-up, so it leads whenever the native estimator is available.
        # The value fed to the gate is the ALL-obs inlier fraction (not the
        # consensus-only fraction) so it stays coherent with the covis path
        # and the median-based bar: a junk-dominated image (dino img 52:
        # ~7-10% true pairs) then yields a correct pose whose all-obs inl is
        # still low, defers here, and lands in the verified force-accept path
        # below — exactly as it did before P3P became primary.
        found = None
        p3p = p3p_resect(u[s], pts[obs_c[s]], f0, aux[2][i]) \
            if aux is not None and len(aux) > 2 else None
        if p3p is not None and int(p3p[2].sum()) >= 12:
            rv0, tv0, mask = p3p
            rv, tv, _ = pose_refine(u[s][mask], pts[obs_c[s]][mask], rv0, tv0, f0)
            res = reproj_res_one(cam0, rv, tv, pts[obs_c[s]], u[s])
            found = (_inlier_fraction(res, 3.0), rv, tv)
            if TRACE:
                print(f"    p3p    img {i}: {int(mask.sum())}/{int(s.sum())} "
                      f"RANSAC inliers, all-obs inl {found[0]:.0%}")
        # Fallback: warp-depth Kabsch + most-covisible posed poses as inits
        # for a from-init trimmed-LS.  Carries images whose 2D-3D candidate
        # set is too thin for a minimal solver (the seed-adjacent path).
        # First init clearing 40% inliers wins.
        if found is None:
            init_poses = [] if di is None else [(di[0], di[1])]
            posed_idx = np.nonzero(posed)[0].astype(np.uint32)
            inits = covis.rank_by_covisibility(i, posed_idx)[:3]
            if len(inits) == 0:
                inits = posed_idx[:1]
            init_poses += [(rvec[j], tvec[j]) for j in inits]
            for rv0, tv0 in init_poses:
                rv, tv, inl = pose_refine(u[s], pts[obs_c[s]], rv0, tv0, f0)
                if found is None or inl > found[0]:
                    found = (inl, rv, tv)
                if inl > 0.4:
                    break
        # Acceptance gate: a resection far below the accepted-so-far level
        # is a misregistration in the making (the no-gate trace showed 0-7%
        # resections cascading into an 80° wreck), but the marginal band is
        # recoverable by the periodic BAs and carries the growth chain, so
        # the bar sits well below the median (seoul full-data trace:
        # accepted 49-81%, recoverable boundary 22%, poison 0-10%).  Defer
        # the image; it gets another chance after the frontier improves.
        if accepted_inl and found[0] < 0.35 * float(np.median(accepted_inl)):
            blocked.add(i)
            if TRACE:
                print(f"    defer  img {i}: inl {found[0]:.0%} on "
                      f"{int(s.sum())} obs (median accepted "
                      f"{float(np.median(accepted_inl)):.0%})")
            continue
        accepted_inl.append(found[0])
        _, rvec[i], tvec[i] = found
        posed[i] = True
        posed_order.append(int(i))
        ba_retry = True
        if TRACE:
            print(f"    resect img {i}: inl {found[0]:.0%} on {int(s.sum())} obs")
        if di is not None:
            # Warp-depth coherence of the accepted pose (echo diagnostics):
            # a misregistered camera can look reprojection-consistent while
            # its pose-implied depths disagree with the warp-predicted ones.
            _, _, sel, z_pred = di
            xc = Rotation.from_rotvec(rvec[i]).apply(pts[obs_c[sel]]) + tvec[i]
            # Canonical camera z is < 0 in front; the ratio of two negatives.
            ok_z = (xc[:, 2] < -1e-6) & (z_pred < -1e-6)
            if ok_z.sum() >= 6:
                coh = float(np.median(np.abs(np.log(xc[ok_z, 2] / z_pred[ok_z]))))
                _DEPTH_COH.append((i, coh, found[0]))
        pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f0)
        since_ba += 1
        if GROW_BA and since_ba >= ba_every:
            since_ba = 0
            rvec, tvec, pts = run_grow_ba(rvec, tvec, pts)
    return rvec, tvec, pts, posed


def batch_resect(
    rvec, tvec, pts, posed, f0, obs_c, obs_i, u, n_img, n_cl, covis, aux,
    gate=0.30, rounds=6,
):
    """Register the remaining un-posed images by pose-only resection against
    the structure — NO bundle adjustment, ever.  Batched-iterative: each round
    resects every frame with >= 8 valid-point observations, then re-triangulates
    new clusters from the newly-posed frames so the next round can reach further
    (a frame's own structure becomes available once a couple of its neighbours
    are in).  This is the "smart data" completion — grow a backbone with the
    expensive BA loop, then propagate poses over the redundant remainder with
    cheap independent resections; cost is ~linear in frames, no global BA."""
    cam0 = make_cam(f0)
    dims = aux[2] if aux is not None else None
    total = 0
    for _ in range(rounds):
        added = 0
        for j in np.nonzero(~posed)[0]:
            s = (obs_i == j) & ~np.isnan(pts[obs_c, 0])
            if int(s.sum()) < 8:
                continue
            uj, xj = u[s], pts[obs_c[s]]
            wh = dims[j] if dims is not None else np.asarray(_CAM_WH, float)
            found = None
            p3p = p3p_resect(uj, xj, f0, wh)
            if p3p is not None and int(p3p[2].sum()) >= 8:
                m = p3p[2]
                rv, tv, _ = pose_refine(uj[m], xj[m], p3p[0], p3p[1], f0)
                res = reproj_res_one(cam0, rv, tv, xj, uj)
                found = (_inlier_fraction(res, 3.0), rv, tv)
            if found is None:
                posed_idx = np.nonzero(posed)[0].astype(np.uint32)
                for k in covis.rank_by_covisibility(j, posed_idx)[:2]:
                    rv, tv, inl = pose_refine(uj, xj, rvec[k], tvec[k], f0)
                    if found is None or inl > found[0]:
                        found = (inl, rv, tv)
            if found is not None and found[0] >= gate:
                _, rvec[j], tvec[j] = found
                posed[j] = True
                added += 1
        total += added
        if added == 0:
            break
        pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f0)
    return rvec, tvec, pts, posed, total


# ── Hierarchical (multigrid-style) BA: restriction, coarse solve, prolongation ─


def restrict_tracks(obs_c, obs_i, adm_rank, coarse, n_cl, m_per_cam=30, cap=12000):
    """Coverage-constrained representative track selection (restriction).

    Greedily admit clusters in admission-rank order, but only clusters with
    >= 2 coarse-camera observations, until every coarse camera holds at least
    ``m_per_cam`` selected observations (or clusters run out / cap reached).
    Unlike a global best-N cap, the constraint keeps the coarse problem
    well-conditioned at every camera."""
    on_coarse = coarse[obs_i]
    # per cluster: number of coarse cams observing it (span on the coarse set)
    span_c = np.zeros(n_cl, np.int64)
    np.add.at(span_c, obs_c[on_coarse], 1)
    order = np.argsort(adm_rank, kind="stable")
    need = np.full(len(coarse), m_per_cam, np.int64)
    need[~coarse] = 0
    sel = np.zeros(n_cl, bool)
    # cluster -> its coarse cams (via sorted obs walk)
    have = np.zeros(len(coarse), np.int64)
    n_sel = 0
    for c in order:
        if span_c[c] < 2 or n_sel >= cap:
            continue
        rows = np.nonzero((obs_c == c) & on_coarse)[0]
        cams = obs_i[rows]
        if not (have[cams] < m_per_cam).any() and (have[coarse] >= m_per_cam).all():
            break
        sel[c] = True
        n_sel += 1
        np.add.at(have, cams, 1)
    return sel


def prolongate(rvec, tvec, posed, coarse, rvec_c, tvec_c, W):
    """Transfer the coarse solve's pose corrections to the fine cameras.

    Each coarse camera's correction is the world-frame motion of its
    camera-to-world pose; a fine camera receives the covisibility-weighted
    blend of its coarse neighbours' corrections, anchored at each neighbour's
    centre (rotation about the neighbour, plus its centre displacement), which
    keeps the transfer local and lever-arm free."""
    rc_old = Rotation.from_rotvec(rvec).as_matrix().transpose(0, 2, 1)
    cen_old = np.einsum("nij,nj->ni", rc_old, -tvec)
    rc_new = Rotation.from_rotvec(rvec_c).as_matrix().transpose(0, 2, 1)
    cen_new = np.einsum("nij,nj->ni", rc_new, -tvec_c)
    ck = np.nonzero(coarse)[0]
    rw = rc_new[ck] @ rc_old[ck].transpose(0, 2, 1)  # world-frame delta rot
    out_r, out_t = rvec.copy(), tvec.copy()
    for i in np.nonzero(posed & ~coarse)[0]:
        w = W[i, ck].astype(np.float64)
        if w.sum() <= 0:
            continue
        top = np.argsort(-w)[:6]
        w = w[top] / w[top].sum()
        kk = ck[top]
        rot_i = Rotation.from_matrix(rw[top]).mean(weights=w)
        c_pred = np.einsum(
            "k,kj->j",
            w,
            cen_new[kk]
            + np.einsum("kij,kj->ki", rw[top], cen_old[i] - cen_old[kk]),
        )
        rc_i = rot_i.as_matrix() @ rc_old[i]
        out_r[i] = Rotation.from_matrix(rc_i.T).as_rotvec()
        out_t[i] = -rc_i.T @ c_pred
    # coarse cameras take their solved poses directly
    out_r[coarse] = rvec_c[coarse]
    out_t[coarse] = tvec_c[coarse]
    return out_r, out_t


_S3 = np.diag([1.0, -1.0, -1.0])


def relative_rotation(f, wh, ua, ub, seed=0):
    """Relative rotation between two views from pixel correspondences.

    Native fundamental estimate -> essential at the known focal -> standard
    (+Z-forward) decomposition with midpoint-cheirality disambiguation ->
    conjugate back to the canonical (-Z-forward) frame.  Returns
    (R_rel canonical with x_b = R_rel x_a, inlier count) or None."""
    from sfmtool._sfmtool.geometry import estimate_fundamental

    res = estimate_fundamental(
        np.ascontiguousarray(ua), np.ascontiguousarray(ub), seed=seed
    )
    if res is None:
        return None
    inl = np.asarray(res["inliers"], bool)
    if int(inl.sum()) < 25:
        return None
    k_m = np.array([[f, 0.0, wh[0] / 2], [0.0, f, wh[1] / 2], [0.0, 0.0, 1.0]])
    e_m = k_m.T @ np.asarray(res["f_matrix"]) @ k_m
    uu, _s, vt = np.linalg.svd(e_m)
    e_m = uu @ np.diag([1.0, 1.0, 0.0]) @ vt
    uu, _s, vt = np.linalg.svd(e_m)
    if np.linalg.det(uu) < 0:
        uu = -uu
    if np.linalg.det(vt) < 0:
        vt = -vt
    w_m = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cands = [(uu @ w_m @ vt, uu[:, 2]), (uu @ w_m.T @ vt, uu[:, 2]),
             (uu @ w_m @ vt, -uu[:, 2]), (uu @ w_m.T @ vt, -uu[:, 2])]
    # cheirality on a sample of inliers (+Z-forward normalized coords)
    sel = np.nonzero(inl)[0][:60]
    xn1 = np.c_[(ua[sel] - wh / 2) / f, np.ones(len(sel))]
    xn2 = np.c_[(ub[sel] - wh / 2) / f, np.ones(len(sel))]
    best, best_n = None, -1
    for r_m, t_v in cands:
        # midpoint depths: rays d1 = xn1, d2 = R^T xn2 from centers 0, -R^T t
        d2 = xn2 @ r_m
        c2 = -(r_m.T @ t_v)
        n_ok = 0
        for a_r, b_r in zip(xn1, d2):
            m_a = np.array([a_r, -b_r]).T
            try:
                s_ab = np.linalg.lstsq(m_a, c2, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            if s_ab[0] > 0 and s_ab[1] > 0:
                n_ok += 1
        if n_ok > best_n:
            best, best_n = r_m, n_ok
    if best is None or best_n < 0.5 * len(sel):
        return None
    return _S3 @ best @ _S3, int(inl.sum())


def vcycle_rot(
    rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis, adm_rank,
    n_coarse=120, m_per_cam=30,
):
    """Rotation-first V-cycle: measure relative rotations on coarse
    covisibility edges directly from image correspondences, chordally average
    them (anchored to the current gauge), re-solve coarse translations with
    rotations locked, then prolong + smooth.  Global rotation methods see
    long-wavelength drift directly; a robust-trimmed point adjustment cannot
    (drift residuals get trimmed as outliers)."""
    from sfmtool._sfmtool.geometry import resect_translation

    coarse = np.zeros(n_img, bool)
    coarse[np.asarray(covis.thin_to(min(n_coarse, int(posed.sum()))),
                      dtype=np.int64)] = True
    coarse &= posed
    ck = np.nonzero(coarse)[0]
    if len(ck) < 8:
        return rvec, tvec, pts
    W = np.asarray(covis.counts, dtype=np.int64)
    wh = np.asarray(_CAM_WH, float)

    # cluster -> observation rows (obs_c is cluster-sorted)
    starts = np.searchsorted(obs_c, np.arange(n_cl + 1))

    def pair_uv(a, b):
        rows_a, rows_b = [], []
        ca = np.nonzero((obs_i == a))[0]
        cb = np.nonzero((obs_i == b))[0]
        cl_a = {int(obs_c[r]): r for r in ca}
        for r in cb:
            ra = cl_a.get(int(obs_c[r]))
            if ra is not None:
                rows_a.append(ra)
                rows_b.append(r)
        return np.asarray(rows_a, np.int64), np.asarray(rows_b, np.int64)

    # edges: top covisible coarse partners per coarse camera
    edges = {}
    for i in ck:
        part = ck[np.argsort(-W[i, ck])]
        n_e = 0
        for j in part:
            if j == i or W[i, j] < 25 or n_e >= 8:
                continue
            key = (min(i, j), max(i, j))
            if key not in edges:
                edges[key] = None
            n_e += 1
    # Edge gating: E is degenerate on near-coincident (low-baseline) pairs —
    # skip them; and drift between covisible spatial neighbours is small by
    # construction, so a measured relative rotation far from the current one
    # is a mirror/degeneracy artifact, not a correction (trust region).
    r_cur_m = Rotation.from_rotvec(rvec).as_matrix()
    cen_cur = np.einsum("nij,nj->ni", r_cur_m.transpose(0, 2, 1), -tvec)
    scene = float(np.linalg.norm(
        cen_cur[posed].max(0) - cen_cur[posed].min(0)
    )) or 1.0
    meas = {}
    n_base = n_trust = 0
    for (a, b) in edges:
        if np.linalg.norm(cen_cur[a] - cen_cur[b]) < 0.01 * scene:
            n_base += 1
            continue
        ra, rb = pair_uv(a, b)
        if len(ra) < 30:
            continue
        rr = relative_rotation(f, wh, u[ra], u[rb], seed=a * 100003 + b)
        if rr is None:
            continue
        r_ab, w_e = rr
        delta = Rotation.from_matrix(
            r_ab @ (r_cur_m[b] @ r_cur_m[a].T).T
        ).magnitude() * 180 / np.pi
        if delta > 25.0:
            n_trust += 1
            continue
        meas[(a, b)] = (r_ab, w_e)
    print(f"    [edge gates: {n_base} low-baseline, {n_trust} out-of-trust]")
    if len(meas) < len(ck):
        print(f"    [vcycle-rot: only {len(meas)} edges; skipping]")
        return rvec, tvec, pts
    print(f"    [vcycle-rot: {len(meas)} rotation edges over {len(ck)} coarse cams]")

    # chordal averaging anchored at the max-degree camera's current rotation
    r_cur = Rotation.from_rotvec(rvec).as_matrix()
    r_new = {int(k): r_cur[k].copy() for k in ck}
    deg = {int(k): 0 for k in ck}
    for (a, b), (_r, w_e) in meas.items():
        deg[int(a)] += w_e
        deg[int(b)] += w_e
    anchor = max(deg, key=deg.get)
    for _ in range(30):
        for k in ck:
            k = int(k)
            if k == anchor:
                continue
            acc = np.zeros((3, 3))
            for (a, b), (r_ab, w_e) in meas.items():
                if int(a) == k:
                    acc += w_e * (r_ab.T @ r_new[int(b)])
                elif int(b) == k:
                    acc += w_e * (r_ab @ r_new[int(a)])
            if not np.isfinite(acc).all() or np.abs(acc).sum() == 0:
                continue
            uu, _s, vt = np.linalg.svd(acc)
            if np.linalg.det(uu @ vt) < 0:
                uu[:, -1] *= -1
            r_new[k] = uu @ vt
    rvec_c = rvec.copy()
    for k in ck:
        rvec_c[int(k)] = Rotation.from_matrix(r_new[int(k)]).as_rotvec()

    # translations: keep centers, re-derive t from the averaged rotations,
    # then alternate triangulation and rotation-locked linear resection.
    tvec_c = tvec.copy()
    cen = np.einsum("nij,nj->ni", r_cur.transpose(0, 2, 1), -tvec)
    for k in ck:
        k = int(k)
        tvec_c[k] = -(r_new[k] @ cen[k])
    sel = restrict_tracks(obs_c, obs_i, adm_rank, coarse, n_cl, m_per_cam)
    cam = make_cam(f)
    for _ in range(4):
        rot_c = Rotation.from_rotvec(rvec_c).as_matrix()
        pts_c = triangulate(
            obs_c, obs_i, u, rot_c, tvec_c, coarse, n_cl, f
        )
        for k in ck:
            k = int(k)
            s = (obs_i == k) & sel[obs_c] & ~np.isnan(pts_c[obs_c, 0])
            if int(s.sum()) < 12:
                continue
            q_k = Rotation.from_rotvec(rvec_c[k]).as_quat()[[3, 0, 1, 2]]
            out = resect_translation(
                cam, q_k, np.ascontiguousarray(pts_c[obs_c[s]]),
                np.ascontiguousarray(u[s]),
            )
            if out is not None:
                tvec_c[k] = np.asarray(out["translation"])

    rvec_o, tvec_o = prolongate(rvec, tvec, posed, coarse, rvec_c, tvec_c, W)
    pts_o = fill_new_points(
        np.full_like(pts, np.nan), obs_c, obs_i, u, rvec_o, tvec_o, posed, f
    )
    cam0 = make_cam(f)
    for i in np.nonzero(posed & ~coarse)[0]:
        s = (obs_i == i) & ~np.isnan(pts_o[obs_c, 0])
        if int(s.sum()) < 8:
            continue
        uj, xj = u[s], pts_o[obs_c[s]]
        inl0 = _inlier_fraction(reproj_res_one(cam0, rvec_o[i], tvec_o[i], xj, uj), 3.0)
        rv, tv, _ = pose_refine(uj, xj, rvec_o[i], tvec_o[i], f)
        if _inlier_fraction(reproj_res_one(cam0, rv, tv, xj, uj), 3.0) >= inl0:
            rvec_o[i], tvec_o[i] = rv, tv
    pts_o = fill_new_points(
        np.full_like(pts_o, np.nan), obs_c, obs_i, u, rvec_o, tvec_o, posed, f
    )
    return rvec_o, tvec_o, pts_o


# Photometric-anchor context (names / obs_f / adm_rank), set by
# external_seed_complete when SFMTOOL_PHOTO_ANCHOR=1.
_PH_CTX = None


def _photo_anchor_ba(rvec, tvec, pts, posed, win, f0, obs_c, obs_i, u, n_img, n_cl):
    """Anchor BA on photometrically congealed observations of the anchor
    subset's representative tracks.  Returns (rvec, tvec) or None (too few
    congealed observations — caller falls back to the raw anchor BA)."""
    sub_mask = win.copy()
    if int(sub_mask.sum()) > 80:
        keep80 = np.zeros(n_img, bool)
        # thin within the anchor subset for the congeal budget (image loads)
        idx = np.nonzero(sub_mask)[0]
        keep80[idx[:: max(1, len(idx) // 80)][:80]] = True
        sub_mask &= keep80
    sel = restrict_tracks(
        obs_c, obs_i, _PH_CTX["adm_rank"], sub_mask, n_cl, 30, cap=3000
    )
    live = sub_mask[obs_i] & sel[obs_c] & ~np.isnan(pts[obs_c, 0])
    span_c = np.zeros(n_cl, np.int64)
    np.add.at(span_c, obs_c[live], 1)
    live &= (span_c >= 3)[obs_c]
    tr_ids = np.unique(obs_c[live])
    if len(tr_ids) < 60:
        return None
    rows = (obs_c[live], np.nonzero(live)[0])
    try:
        cc, ic, uvc = congeal_tracks(
            _PH_CTX["names"], np.nonzero(sub_mask)[0], rvec, tvec, f0,
            pts[tr_ids], tr_ids, rows, obs_i, _PH_CTX["obs_f"], u,
        )
    except Exception as ex:  # experiment-grade: fall back to the raw anchor
        print(f"    [photo-anchor congeal failed: {ex}]")
        return None
    if len(cc) < 400:
        print(f"    [photo-anchor: only {len(cc)} congealed obs; raw anchor]")
        return None
    print(f"    [photo-anchor: {len(cc)} congealed obs, {len(tr_ids)} tracks, "
          f"{int(sub_mask.sum())} cams]")
    pts_t = pts[tr_ids].copy()
    rot = Rotation.from_rotvec(rvec).as_matrix()
    _, rvec_n, tvec_n, _, _, _, _ = bundle_adjust(
        cc, ic, uvc, rot, tvec, pts_t, f0,
        n_img, len(tr_ids), opt_f=False, verbose=False,
        schedule=((30.0, 3.0), (8.0, 1.5), (4.0, 1.0)),
    )
    return rvec_n, tvec_n


def congeal_tracks(names, sub, rvec, tvec, f, pts_t, tr_ids, rows, obs_i, obs_f, u):
    """Photometric congealing of the coarse tracks' observations.

    Builds CameraViews over the coarse cameras, a feature-scaled patch cloud
    over the coarse tracks, and localizes each track in exactly the views that
    observe it (leave-one-out ZNCC consensus).  Returns congealed observation
    arrays (track-local id, full image id, corrected uv) — junk members fail
    the consensus and simply drop out."""
    import cv2

    from sfmtool._sfmtool.patches import CameraViews, ImagePyramidSet, PatchCloud
    from sfmtool._sfmtool.io import read_sift_partial
    from sfmtool.sift.file import get_sift_path_for_image

    sub_names = [names[int(g)] for g in sub]
    sub_of_full = np.full(len(names), -1, np.int64)
    sub_of_full[sub] = np.arange(len(sub))
    q = Rotation.from_rotvec(rvec[sub]).as_quat()[:, [3, 0, 1, 2]]
    views = CameraViews([make_cam(f)], q, tvec[sub])
    ws = WS.resolve()
    t_local = {int(c): k for k, c in enumerate(tr_ids)}
    tr_a = np.asarray([t_local[int(c)] for c in rows[0]], np.uint32)
    tr_img = sub_of_full[obs_i[rows[1]]].astype(np.uint32)
    tr_feat = obs_f[rows[1]]
    scales = np.full(len(tr_a), np.nan)
    for j, name in enumerate(sub_names):
        m = tr_img == j
        if not m.any():
            continue
        aff = read_sift_partial(
            get_sift_path_for_image(ws / name), int(tr_feat[m].max()) + 1
        )["affine_shapes"].astype(np.float64)
        scales[m] = np.hypot(aff[tr_feat[m], 0, 0], aff[tr_feat[m], 1, 0])
    cloud = PatchCloud.from_tracks(
        views,
        np.c_[pts_t, np.ones(len(pts_t))],
        tr_a,
        tr_img,
        keypoint_scales=scales,
        normal="mean_viewing",
        extent="feature_size",
        extent_value=2.5,
    )
    imgs = [
        np.ascontiguousarray(cv2.imread(str(ws / n), cv2.IMREAD_COLOR))
        for n in sub_names
    ]
    pyrset = ImagePyramidSet(views, imgs)
    view_sets = {}
    for k in range(len(tr_ids)):
        view_sets[k] = sorted({int(v) for v in tr_img[tr_a == k]})
    results = cloud.localize_keypoints(
        views, pyrset, view_sets=view_sets,
        max_shift_px=60.0, search=12.0, min_relative_zncc=0.6,
    )
    cc, ic, uvc = [], [], []
    for r in results:
        pid = int(r["point_index"])
        for k, v in enumerate(np.asarray(r["views"])):
            cc.append(pid)
            ic.append(int(sub[int(v)]))
            uvc.append(np.asarray(r["keypoints"], np.float64)[k])
    return (
        np.asarray(cc, np.int64),
        np.asarray(ic, np.int64),
        np.asarray(uvc, np.float64).reshape(-1, 2),
    )


def vcycle_photo(
    rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis, adm_rank,
    n_coarse=80, m_per_cam=30, obs_f=None, names=None,
):
    """Photometric V-cycle: the restriction operator produces a MORE ACCURATE
    coarse problem, not a noisy subsample — the representative tracks'
    observations are photometrically congealed (junk drops out, positions
    sharpen to sub-pixel) before the wide-schedule coarse solve, so the loop
    signal is no longer buried under the SIFT junk floor."""
    coarse = np.zeros(n_img, bool)
    coarse[np.asarray(covis.thin_to(min(n_coarse, int(posed.sum()))),
                      dtype=np.int64)] = True
    coarse &= posed
    sel = restrict_tracks(obs_c, obs_i, adm_rank, coarse, n_cl, m_per_cam, cap=4000)
    live = coarse[obs_i] & sel[obs_c] & ~np.isnan(pts[obs_c, 0])
    span_c = np.zeros(n_cl, np.int64)
    np.add.at(span_c, obs_c[live], 1)
    good = span_c >= 3
    live &= good[obs_c]
    tr_ids = np.unique(obs_c[live])
    if len(tr_ids) < 100:
        return rvec, tvec, pts
    rows = (obs_c[live], np.nonzero(live)[0])
    cc, ic, uvc = congeal_tracks(
        names, np.nonzero(coarse)[0], rvec, tvec, f, pts[tr_ids], tr_ids,
        rows, obs_i, obs_f, u,
    )
    if len(cc) < 500:
        print("    [vcycle-photo: too few congealed obs; skipping]")
        return rvec, tvec, pts
    print(f"    [vcycle-photo: {len(cc)} congealed obs on {len(tr_ids)} tracks "
          f"x {int(coarse.sum())} cams]")
    # coarse solve on the congealed observations, drift-wide schedule
    pts_t = np.full((len(tr_ids), 3), np.nan)
    pts_t[:] = pts[tr_ids]
    rot = Rotation.from_rotvec(rvec).as_matrix()
    _, rvec_c, tvec_c, _, _, _, _ = bundle_adjust(
        cc.astype(np.int64), ic, uvc, rot, tvec, pts_t, f,
        n_img, len(tr_ids), opt_f=False, verbose=False,
        schedule=((300.0, 30.0), (80.0, 8.0), (20.0, 2.0), (4.0, 1.0)),
    )
    W = np.asarray(covis.counts, dtype=np.int64)
    rvec_o, tvec_o = prolongate(rvec, tvec, posed, coarse, rvec_c, tvec_c, W)
    pts_o = fill_new_points(
        np.full_like(pts, np.nan), obs_c, obs_i, u, rvec_o, tvec_o, posed, f
    )
    cam0 = make_cam(f)
    for i in np.nonzero(posed & ~coarse)[0]:
        s = (obs_i == i) & ~np.isnan(pts_o[obs_c, 0])
        if int(s.sum()) < 8:
            continue
        uj, xj = u[s], pts_o[obs_c[s]]
        inl0 = _inlier_fraction(reproj_res_one(cam0, rvec_o[i], tvec_o[i], xj, uj), 3.0)
        rv, tv, _ = pose_refine(uj, xj, rvec_o[i], tvec_o[i], f)
        if _inlier_fraction(reproj_res_one(cam0, rv, tv, xj, uj), 3.0) >= inl0:
            rvec_o[i], tvec_o[i] = rv, tv
    pts_o = fill_new_points(
        np.full_like(pts_o, np.nan), obs_c, obs_i, u, rvec_o, tvec_o, posed, f
    )
    return rvec_o, tvec_o, pts_o


def vcycle(
    rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis, adm_rank,
    n_coarse=150, m_per_cam=30,
):
    """One two-level V-cycle: restrict -> coarse solve -> prolong -> smooth.

    The coarse level is a covisibility-spread camera subset with a
    coverage-constrained representative track set; its staged BA corrects the
    long-wavelength (drift) error the windowed smoother cannot see; the
    prolongation carries those corrections to every fine camera; a pose-only
    polish per camera then re-couples poses to the re-triangulated structure."""
    coarse = np.zeros(n_img, bool)
    coarse[np.asarray(covis.thin_to(min(n_coarse, int(posed.sum()))),
                      dtype=np.int64)] = True
    coarse &= posed
    sel = restrict_tracks(obs_c, obs_i, adm_rank, coarse, n_cl, m_per_cam)
    live = coarse[obs_i] & sel[obs_c] & ~np.isnan(pts[obs_c, 0])
    if live.sum() < 100:
        return rvec, tvec, pts
    rot = Rotation.from_rotvec(rvec).as_matrix()
    # Drift-aware coarse solve: a WIDE leading schedule. The default staged
    # trim (50 -> 12 -> 4 px) classifies drift-scale residuals on loop-closing
    # observations as outliers and removes exactly the signal the coarse level
    # exists to correct; the wide rounds keep them in the problem so the loop
    # can pull closed, then tighten.
    _, rvec_c, tvec_c, pts_c, _, _, _ = bundle_adjust(
        obs_c[live], obs_i[live], u[live], rot, tvec, pts, f,
        n_img, n_cl, opt_f=False, verbose=False,
        schedule=((300.0, 30.0), (80.0, 8.0), (20.0, 2.0), (4.0, 1.0)),
    )
    W = np.asarray(covis.counts, dtype=np.int64)
    rvec, tvec = prolongate(rvec, tvec, posed, coarse, rvec_c, tvec_c, W)
    pts = fill_new_points(
        np.full_like(pts, np.nan), obs_c, obs_i, u, rvec, tvec, posed, f
    )
    # smoother: GATED pose-only polish of every fine camera — keep a refined
    # pose only if its inlier fraction does not degrade (an ungated polish
    # mirror-flips weak cameras).
    cam0 = make_cam(f)
    for i in np.nonzero(posed & ~coarse)[0]:
        s = (obs_i == i) & ~np.isnan(pts[obs_c, 0])
        if int(s.sum()) < 8:
            continue
        uj, xj = u[s], pts[obs_c[s]]
        inl0 = _inlier_fraction(reproj_res_one(cam0, rvec[i], tvec[i], xj, uj), 3.0)
        rv, tv, _ = pose_refine(uj, xj, rvec[i], tvec[i], f)
        if _inlier_fraction(reproj_res_one(cam0, rv, tv, xj, uj), 3.0) >= inl0:
            rvec[i], tvec[i] = rv, tv
    pts = fill_new_points(
        np.full_like(pts, np.nan), obs_c, obs_i, u, rvec, tvec, posed, f
    )
    return rvec, tvec, pts


# ── Perspective conversion + triangulation ───────────────────────────────────


def triangulate(obs_c, obs_i, u, rot, trans, used, n_cl, f):
    """Ray-midpoint triangulation of every cluster from the posed images,
    via the batch triangulation binding (clusters with < 2 posed
    observations stay NaN)."""
    from sfmtool._sfmtool.analysis import triangulate_batch

    pts = np.full((n_cl, 3), np.nan)
    sel = used[obs_i]
    if not sel.any():
        return pts
    oc, oi, uv = obs_c[sel], obs_i[sel], u[sel]
    # World-space unit rays and camera centers: x_cam = R x + t, so the world
    # ray is Rᵀ·(canonical camera ray of the full pixel) and the center -Rᵀ t.
    d_loc = make_cam(f).pixel_to_ray_batch(np.ascontiguousarray(uv))
    dirs = np.einsum("nji,nj->ni", rot[oi], d_loc)
    centers = -np.einsum("nji,nj->ni", rot[oi], trans[oi])
    # obs_c is cluster-sorted, so the selection is CSR-ready.
    uniq, counts = np.unique(oc, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    result = triangulate_batch(
        np.ascontiguousarray(dirs), np.ascontiguousarray(centers), offsets
    )
    good = counts >= 2
    pts[uniq[good]] = np.asarray(result["points"])[good]
    return pts


# ── Bundle adjustment ────────────────────────────────────────────────────────


def bundle_adjust(
    obs_c,
    obs_i,
    u,
    rot,
    trans,
    pts,
    f0,
    n_img,
    n_cl,
    opt_f,
    verbose=True,
    schedule=None,
):
    """Staged robust BA via the native kernel (analytic Jacobian in Rust) —
    the same ``geometry.bundle_adjust`` the fast bootstrap uses.  The kernel
    trims gross outliers and behind-camera observations before each solve and
    re-triangulates every cluster from the refined cameras between rounds;
    per-observation residual norms come back in input order (inf for dropped
    obs).  Returns the tiered path's 7-tuple (keep = inlier obs at TRIM_PX).

    The scipy finite-difference version this replaced cost one native residual
    eval per free parameter per iteration — untenable on a full completion
    (tens of thousands of clusters); the analytic-Jacobian kernel is the whole
    reason stage 1 is seconds, not minutes."""
    if schedule is None:
        schedule = [(50.0, 5.0), (12.0, 2.0), (TRIM_PX, 1.0)]
    rvec = Rotation.from_matrix(rot).as_rotvec()
    q = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
    out = _native_ba(
        make_cam(f0),
        np.ascontiguousarray(q),
        np.ascontiguousarray(trans, dtype=np.float64),
        np.ascontiguousarray(pts, dtype=np.float64),
        np.ascontiguousarray(u, dtype=np.float64),
        obs_i.astype(np.uint32),
        obs_c.astype(np.uint32),
        opt_f=opt_f,
        schedule=[(float(t), float(s)) for t, s in schedule],
        max_iters=60,
        min_track=MIN_SPAN_BA,
        min_obs=12,
    )
    f = float(out["focal"])
    rvec = Rotation.from_quat(
        np.asarray(out["quaternions_wxyz"])[:, [1, 2, 3, 0]]
    ).as_rotvec()
    tvec = np.asarray(out["translations"])
    pts = np.asarray(out["points"])
    res = np.asarray(out["residual_norms"])
    keep = res < TRIM_PX
    inlier2 = float((res < 2.0).mean())
    if verbose:
        finite = res[np.isfinite(res)]
        med = float(np.median(finite)) if len(finite) else float("nan")
        print(
            f"  BA: f {f:.1f}, median reproj {med:.2f} px on "
            f"{int(keep.sum())} inlier obs"
        )
    return f, rvec, tvec, pts, keep, res, inlier2


# ── Evaluation against a reference solve ─────────────────────────────────────


def compare_to_reference(names, rvec, tvec, f_est, mask=None):
    """Compare against the first non-bootstrap solve in the workspace.

    Our poses and the reference ``.sfmr`` are both canonical camera frame; the
    world frames differ only by a global rotation, which the similarity
    alignment below absorbs, so the poses feed straight in.  ``mask``
    restricts to a subset of images (e.g. the posed ones).
    """
    if mask is not None:
        names = [n for j, n in enumerate(names) if mask[j]]
        rvec, tvec = rvec[np.asarray(mask)], tvec[np.asarray(mask)]
    if REF is not None:
        ref_files = [REF]
    else:
        ref_files = sorted(
            p for p in WS.glob("sfmr/*.sfmr") if p.name != "bootstrap-pinhole.sfmr"
        )
    if not ref_files:
        print("no reference solve found; skipping comparison")
        return
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    q_wxyz = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
    t_xyz = tvec

    ref = SfmrReconstruction.load(ref_files[0])
    ref_names = list(ref.image_names)
    common = [n for n in names if n in ref_names]
    if len(common) < 3:
        # Cross-workspace fallback: match by basename against the ref
        # directory with the most unique matches (e.g. the bootstrap's
        # frames/ against a rig reference's fisheye_left/).
        from collections import defaultdict
        from pathlib import PurePosixPath

        groups = defaultdict(dict)
        for rn in ref_names:
            pp = PurePosixPath(rn)
            groups[str(pp.parent)][pp.name] = rn
        best = {}
        for g in groups.values():
            mm = {
                n: g[PurePosixPath(n).name]
                for n in names
                if PurePosixPath(n).name in g
            }
            if len(mm) > len(best):
                best = mm
        if len(best) >= 3:
            print(f"matched {len(best)} images by basename fallback")
            names = [n if n not in best else best[n] for n in names]
            common = [best[n] for n in best]
    if len(common) < 3:
        print(f"only {len(common)} common images with {ref_files[0].name}; skipping")
        return

    def centers_rots(qs, ts, order):
        rs = Rotation.from_quat(np.asarray(qs)[order][:, [1, 2, 3, 0]]).as_matrix()
        cs = -np.einsum("nij,ni->nj", rs, np.asarray(ts)[order])
        return cs, rs

    ei = np.array([names.index(n) for n in common])
    ri = np.array([ref_names.index(n) for n in common])
    c_est, r_est = centers_rots(q_wxyz, t_xyz, ei)
    c_ref, r_ref = centers_rots(ref.quaternions_wxyz, ref.translations, ri)

    # A posed SUBSET can have nearly-degenerate camera centers (a short arc of
    # a long orbit), leaving the center-based similarity a free rotation about
    # the arc — so the ROTATION-error gauge is fitted from the camera rotations
    # (well-conditioned always: argmin_g sum ||R_est_i g - R_ref_i||), while the
    # center error uses the free similarity (its own best case).
    from sfmtool._sfmtool.analysis import estimate_alignment_rs

    u_svd, _s, vt = np.linalg.svd(np.einsum("nji,njk->ik", r_est, r_ref))
    if np.linalg.det(u_svd @ vt) < 0:
        u_svd[:, 2] *= -1.0
    g = u_svd @ vt
    rot_err = Rotation.from_matrix(
        np.einsum("nij,nkj->nik", r_ref, np.einsum("nij,jk->nik", r_est, g))
    ).magnitude() * (180 / np.pi)

    tf = estimate_alignment_rs(
        np.ascontiguousarray(c_est, dtype=np.float64),
        np.ascontiguousarray(c_ref, dtype=np.float64),
    )
    c_fit = tf.apply_to_points(np.ascontiguousarray(c_est, dtype=np.float64))
    diam = np.max(np.linalg.norm(c_ref[:, None, :] - c_ref[None, :, :], axis=2))
    cen_err = np.linalg.norm(c_fit - c_ref, axis=1) / diam

    cam0 = ref.cameras[0].to_dict()
    f_ref = ref.cameras[0].focal_lengths[0]
    print(f"\nvs reference {ref_files[0].name} ({len(common)} common images):")
    print(
        f"  camera rotation err: mean {rot_err.mean():.2f}, "
        f"median {np.median(rot_err):.2f}, max {rot_err.max():.2f} deg; "
        f"{(rot_err > 10).sum()} cams > 10 deg"
    )
    print(
        f"  camera center err:   mean {100 * cen_err.mean():.2f}%, "
        f"median {100 * np.median(cen_err):.2f}%, "
        f"max {100 * cen_err.max():.2f}% of scene diameter"
    )
    print(
        f"  focal: bootstrap {f_est:.1f} px vs reference {f_ref:.1f} px "
        f"({cam0['model']})"
    )


# ── Save as .sfmr ────────────────────────────────────────────────────────────


def save_sfmr(data, f, rvec, tvec, pts, keep, res, out_path):
    """Write the bootstrap as an ``embedded_patches`` reconstruction.

    The bootstrap's observations are the cluster patches' *refined*
    positions, not the SIFT detections, so they are stored inline as
    ``keypoints_xy`` rather than as feature indexes into the ``.sift``
    files (which would silently resolve back to the unrefined seeds).
    """
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction
    from sfmtool._workspace import load_workspace_config
    from sfmtool.colmap.convention import world_rotate_w
    from sfmtool.colmap.io import (
        _build_sfmr_data_dict,
        _resolve_workspace_and_sift,
        build_metadata,
        finite_positions_xyzw,
    )

    out_path = Path(out_path).resolve()
    names, dims = data["names"], data["dims"]
    obs_c, obs_i, obs_f = data["obs_c"], data["obs_i"], data["obs_f"]
    w, h = dims[0]

    # Surviving points, renumbered densely; observations grouped by point.
    alive = np.nonzero(np.bincount(obs_c[keep], minlength=len(pts)) >= 2)[0]
    remap = {int(c): k for k, c in enumerate(alive)}
    order = np.argsort(obs_c[keep], kind="stable")
    ko = np.nonzero(keep)[0][order]
    ko = ko[np.isin(obs_c[ko], alive)]

    track_img = obs_i[ko]
    track_feat = obs_f[ko]
    keypoints_xy = data["obs_uv"][ko].astype(np.float32)
    point_idx = np.array([remap[int(c)] for c in obs_c[ko]])
    obs_counts = np.bincount(point_idx, minlength=len(alive))

    positions = pts[alive]
    per_point_err = np.zeros(len(alive), dtype=np.float32)
    np.add.at(per_point_err, point_idx, res[ko].astype(np.float32))
    per_point_err /= np.maximum(obs_counts, 1)

    # The internal poses are already canonical camera frame, in the COLMAP-world
    # gauge; only the world rotation W remains to reach the .sfmr canonical
    # world.  W rotates the point positions and, applied to each rotation row,
    # right-multiplies the world->camera rotations (R_int·Wᵀ); the camera-frame
    # translation is unchanged.
    rot_int = Rotation.from_rotvec(rvec).as_matrix()
    q_can = Rotation.from_matrix(
        world_rotate_w(rot_int.reshape(-1, 3)).reshape(-1, 3, 3)
    ).as_quat()[:, [3, 0, 1, 2]]
    t_can = tvec
    p_can = world_rotate_w(positions)

    (
        workspace_dir,
        _contents,
        resolved_names,
        ft_hashes,
        sc_hashes,
        thumbnails,
    ) = _resolve_workspace_and_sift(names, WS.resolve())

    # Colors from the .sift thumbnails at the (scaled) observation position.
    colors = np.zeros((len(alive), 3), dtype=np.uint8)
    uv = data["obs_uv"][ko]
    for k in range(len(ko)):
        th = np.asarray(thumbnails[track_img[k]])
        ty = int(np.clip(uv[k, 1] * th.shape[0] / h, 0, th.shape[0] - 1))
        tx = int(np.clip(uv[k, 0] * th.shape[1] / w, 0, th.shape[1] - 1))
        colors[point_idx[k]] = th[ty, tx]

    camera = CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": w,
            "height": h,
            "parameters": {
                "focal_length": float(f),
                "principal_point_x": w / 2,
                "principal_point_y": h / 2,
            },
        }
    )

    metadata = build_metadata(
        workspace_dir=workspace_dir,
        output_path=out_path,
        workspace_config=load_workspace_config(workspace_dir),
        operation="cluster_bootstrap",
        tool_name="sfmtool",
        tool_options={"camera_model": "SIMPLE_PINHOLE", "focal_grid": F_GRID},
        image_count=len(names),
        point_count=len(alive),
        observation_count=int(obs_counts.sum()),
        camera_count=1,
    )

    sfmr_dict = _build_sfmr_data_dict(
        cameras=[camera],
        image_names=resolved_names,
        camera_indexes=np.zeros(len(names), dtype=np.uint32),
        quaternions_wxyz=q_can,
        translations_xyz=t_can,
        positions_xyzw=finite_positions_xyzw(p_can),
        colors_rgb=colors,
        reprojection_errors=per_point_err,
        track_image_indexes=track_img,
        track_feature_indexes=track_feat,
        point_indexes=point_idx,
        observation_counts=obs_counts,
        feature_tool_hashes=ft_hashes,
        sift_content_hashes=sc_hashes,
        thumbnails=thumbnails,
        metadata=metadata,
    )

    recon = SfmrReconstruction.from_data(workspace_dir, sfmr_dict)

    # ── Surfel frames copied from the cluster patches ────────────────────
    # Each member's stored 2x2 warp is the projection of the cluster's
    # common surfel into that image, so the 3D patch frame is recoverable:
    # solve J_k·B = A_k per point (J_k the projection Jacobian at the
    # point, B the 3x2 map from reference-image pixels to 3D on the surfel
    # plane; the reference row contributes J_ref·B = I), then
    # u = B·(r, 0), v = B·(0, r) with r the refinement radius in reference
    # pixels (keypoint-frame radius x the reference feature's scale).
    from sfmtool._sfmtool.patches import PatchCloud
    from sfmtool._sfmtool.io import read_sift, read_sift_metadata
    from sfmtool.colmap.convention import world_rotate_w
    from sfmtool.sift.file import get_sift_path_for_image

    feature_scales = {}
    image_file_hashes = []
    for i, name in enumerate(names):
        sp = get_sift_path_for_image(workspace_dir / name)
        meta = read_sift_metadata(sp)["metadata"]
        image_file_hashes.append(bytes.fromhex(meta["image_file_xxh128"]))
        shapes = np.asarray(read_sift(sp)["affine_shapes"], dtype=np.float64)
        feature_scales[i] = 0.5 * (
            np.linalg.norm(shapes[:, :, 0], axis=1)
            + np.linalg.norm(shapes[:, :, 1], axis=1)
        )

    # The surfel-frame solve below is written in the COLMAP +Z-forward camera
    # model (the pinhole projection Jacobian and z > 0 in front), so it runs on
    # the poses flipped back to that frame by S = diag(1, -1, -1); its
    # world-space u/v/normal outputs convert to the canonical world by the same
    # W as the points, at the end.  Positions stay in the COLMAP-world gauge.
    s_flip = np.array([1.0, -1.0, -1.0])
    rot_all = Rotation.from_rotvec(rvec).as_matrix() * s_flip[None, :, None]
    tvec_col = tvec * s_flip
    warps = data["obs_warp"][ko]
    is_ref = data["obs_ref"][ko]
    radius_kf = data["refine_radius"]
    half_u = np.zeros((len(alive), 3), dtype=np.float64)
    half_v = np.zeros((len(alive), 3), dtype=np.float64)
    normals = np.zeros((len(alive), 3), dtype=np.float64)
    p_starts = np.searchsorted(point_idx, np.arange(len(alive) + 1))
    # The reference constraint J_ref·B = I determines B up to a 2-vector
    # b_z — the surfel's out-of-plane slope in the reference camera frame
    # (B = R_refᵀ·[(z_r/f)·I + p_r·b_z ; b_z] with p_r the normalized ref
    # coords).  Each other member contributes A_k − (z_r/f)·M2 = c_k·b_z
    # with M = J_k·R_refᵀ = [M2 | m3] and c_k = M2·p_r + m3.  The tilt is
    # exactly the depth-like weakly-observed direction, so the solve gets a
    # fronto-parallel Tikhonov prior (weight relative to the members'
    # leverage) and a hard obliquity cap; these are what the photometric
    # normal refinement later polishes.
    tan_cap = np.tan(np.radians(80.0))
    for p in range(len(alive)):
        lo, hi = int(p_starts[p]), int(p_starts[p + 1])
        refs_here = np.nonzero(is_ref[lo:hi])[0]
        if len(refs_here) == 0:
            continue  # reference member trimmed: leave the zero (no-patch) frame
        k_ref = lo + int(refs_here[0])
        i_ref = int(track_img[k_ref])
        x_pt = positions[p]
        r_ref = rot_all[i_ref]
        xc_ref = r_ref @ x_pt + tvec_col[i_ref]
        z_ref = max(xc_ref[2], 1e-6)
        p_r = xc_ref[:2] / z_ref
        rows, rhs = [], []
        for k in range(lo, hi):
            if k == k_ref:
                continue
            i = int(track_img[k])
            xc = rot_all[i] @ x_pt + tvec_col[i]
            z = max(xc[2], 1e-6)
            j_proj = (f / z) * np.array(
                [[1.0, 0.0, -xc[0] / z], [0.0, 1.0, -xc[1] / z]]
            )
            m = j_proj @ rot_all[i] @ r_ref.T
            c_k = m[:, :2] @ p_r + m[:, 2]
            resid = warps[k] - (z_ref / f) * m[:, :2]
            for j in range(2):
                rows.append([c_k[0] * (1 - j), c_k[0] * j])
                rows.append([c_k[1] * (1 - j), c_k[1] * j])
                rhs.append(resid[0, j])
                rhs.append(resid[1, j])
        if not rows:
            continue
        rows = np.asarray(rows)
        rhs = np.asarray(rhs)
        # Fronto prior: damping rows scaled to a fraction of member leverage.
        lam = 0.3 * np.sqrt((rows**2).sum() / max(len(rows), 1))
        rows = np.vstack([rows, [[lam, 0.0], [0.0, lam]]])
        rhs = np.concatenate([rhs, [0.0, 0.0]])
        b_z = np.linalg.lstsq(rows, rhs, rcond=None)[0]
        # Obliquity cap: tan(tilt) = |b_z| / (z_ref / f).
        b_norm = np.linalg.norm(b_z)
        max_bz = tan_cap * z_ref / f
        if b_norm > max_bz:
            b_z *= max_bz / b_norm
        b_map = r_ref.T @ np.vstack(
            [(z_ref / f) * np.eye(2) + np.outer(p_r, b_z), b_z[None, :]]
        )
        r_px = radius_kf * feature_scales[i_ref][int(track_feat[k_ref])]
        u3 = b_map @ np.array([r_px, 0.0])
        v3 = b_map @ np.array([0.0, r_px])
        n3 = np.cross(u3, v3)
        norm = np.linalg.norm(n3)
        if norm < 1e-12:
            continue
        n3 /= norm
        cam_c = -r_ref.T @ tvec_col[i_ref]
        if np.dot(n3, cam_c - x_pt) < 0:
            u3, v3, n3 = v3, u3, -n3  # keep normal = normalize(u x v), front-facing
        half_u[p], half_v[p], normals[p] = u3, v3, n3

    # COLMAP -> canonical for the direction quantities (same W as the points).
    half_u = np.asarray(world_rotate_w(half_u), dtype=np.float32)
    half_v = np.asarray(world_rotate_w(half_v), dtype=np.float32)
    normals = np.asarray(world_rotate_w(normals), dtype=np.float32)

    cloud = PatchCloud.from_halfvec_arrays(half_u, half_v, np.asarray(p_can))
    recon = recon.clone_with_changes(
        feature_source="embedded_patches",
        keypoints_xy=keypoints_xy,
        image_file_hashes=image_file_hashes,
        normals=normals,
        patches=cloud,
    )
    recon.save(out_path)
    n_patched = int(np.count_nonzero(np.linalg.norm(half_u, axis=1) > 0))
    print(
        f"\nwrote {out_path} ({len(alive)} points, {int(obs_counts.sum())} obs, "
        f"{recon.feature_source}, {n_patched} warp-derived patch frames)"
    )
    return recon


# ── Main ─────────────────────────────────────────────────────────────────────


def external_seed_complete(
    data, seed_path, active_cl, all_c, all_i, all_u,
    n_img, n_cl, ds_all, ref_img, dims, ba_cl,
):
    """Complete a stage-1 fast-pinhole seed instead of searching for one.

    Activates every cluster up front (the external seed replaces the tier-0
    search), plants the seed poses at ``focal_structure_px``, triangulates,
    grows the rest with one next-best-view pass at fixed f, then releases f.
    Returns the same tuple the tiered path leaves for the assemble tail
    (f, rvec, tvec, pts, posed, ok, keep, res), or None to abort (a flagged
    seed without SFMTOOL_FORCE=1)."""
    seed = json.loads(Path(seed_path).read_text())
    flags = seed.get("confidence_flags") or []
    if flags and os.environ.get("SFMTOOL_FORCE") != "1":
        print(
            f"\nseed {seed_path} is FLAGGED {flags}; refusing to complete an "
            f"untrustworthy seed (set SFMTOOL_FORCE=1 to override)."
        )
        return None
    if flags:
        print(f"\nseed FLAGGED {flags} but SFMTOOL_FORCE=1 — completing anyway")

    # Growth cluster set. By default every cluster is active; on a large
    # (full-res) capture that is the completion bottleneck — grow_loop's
    # next-best-view + periodic BAs over 100k+ clusters do not scale. The
    # coarse-to-fine lever SFMTOOL_COMPLETE_MAX_CL grows on only the best-N
    # clusters by span (the robust multi-view backbone); the finer clusters
    # can be admitted in a later pass.
    cap = int(os.environ.get("SFMTOOL_COMPLETE_MAX_CL", "0"))
    if cap > 0 and cap < n_cl:
        active_cl[:] = data["adm_rank"] < cap
        print(
            f"coarse completion: growing on best {int(active_cl.sum())} of "
            f"{n_cl} clusters by span"
        )
    else:
        active_cl[:] = True
    act = active_cl[all_c]
    obs_c, obs_i, u = all_c[act], all_i[act], all_u[act]
    aux = (ds_all[act], ref_img, dims)
    bam = ba_cl[all_c][act]
    covis = build_covisibility(obs_c, obs_i, n_img, n_cl)
    global _PH_CTX
    if os.environ.get("SFMTOOL_PHOTO_ANCHOR") == "1":
        _PH_CTX = {
            "names": data["names"],
            "obs_f": data["obs_f"][act],
            "adm_rank": data["adm_rank"],
        }

    # Plant the seed poses.  Image index space is data["names"] order (the
    # raw matches-file image order both scripts share); map each posed name
    # back to its index.
    name_to_idx = {n: j for j, n in enumerate(data["names"])}
    f = float(seed["focal_structure_px"])
    rvec = np.zeros((n_img, 3))
    tvec = np.tile([0.0, 0.0, -f], (n_img, 1))  # canonical: -Z is in front
    posed = np.zeros(n_img, bool)
    seed_rvec = np.asarray(seed["rvec"], dtype=np.float64).reshape(-1, 3)
    seed_tvec = np.asarray(seed["tvec"], dtype=np.float64).reshape(-1, 3)
    missing = []
    for k, name in enumerate(seed["posed_images"]):
        j = name_to_idx.get(name)
        if j is None:
            missing.append(name)
            continue
        rvec[j], tvec[j], posed[j] = seed_rvec[k], seed_tvec[k], True
    if missing:
        print(f"WARNING: {len(missing)} seed images absent from matches names: "
              f"{missing[:5]}")
    print(
        f"\nexternal seed: f = {f:.1f} px, planted {int(posed.sum())}/"
        f"{len(seed['posed_images'])} seed poses onto {n_img} images "
        f"[{int(active_cl.sum())} clusters, {len(obs_c)} observations]"
    )
    compare_to_reference(data["names"], rvec, tvec, f, mask=posed)

    # Triangulate the initial structure from the planted poses at fixed f.
    rot = Rotation.from_rotvec(rvec).as_matrix()
    pts = triangulate(obs_c, obs_i, u, rot, tvec, posed, n_cl, f)
    print(f"  triangulated {int((~np.isnan(pts[:, 0])).sum())}/{n_cl} clusters "
          f"from the seed poses")

    # Growth. SFMTOOL_BACKBONE=N grows only a covisibility-SPREAD N-image
    # backbone (thinned to span the whole capture, not a contiguous next-best-
    # view arc), then batch-resects the redundant remainder against the frozen
    # structure with no BA — the "smart data" completion that decouples cost
    # from total frame count on a dense capture (grow_loop's periodic BAs are
    # superlinear in posed frames).
    backbone = int(os.environ.get("SFMTOOL_BACKBONE", "0"))
    ba_window = int(os.environ.get("SFMTOOL_BA_WINDOW", "0"))
    posed_before = int(posed.sum())
    if 0 < backbone < n_img:
        bb = np.zeros(n_img, bool)
        bb[np.asarray(covis.thin_to(backbone), dtype=np.int64)] = True
        bb |= posed  # keep the seed frames
        bbm = bb[obs_i]
        aux_bb = (aux[0][bbm], aux[1], aux[2]) if aux is not None else None
        print(f"  spread backbone: growing {int(bb.sum())} covisibility-thinned "
              f"images (of {n_img})")
        rvec, tvec, pts, posed = grow_loop(
            rvec, tvec, pts, posed, f, obs_c[bbm], obs_i[bbm], u[bbm],
            n_img, n_cl, covis, aux=aux_bb,
            ba=(bam[bbm] if bam is not None else None),
        )
    else:
        rvec, tvec, pts, posed = grow_loop(
            rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis,
            aux=aux, ba=bam,
        )
    print(f"  [after grow: {int(posed.sum())}/{n_img} posed "
          f"(+{int(posed.sum()) - posed_before}) at "
          f"{time.perf_counter() - _T0:.0f}s]")
    compare_to_reference(data["names"], rvec, tvec, f, mask=posed)

    # Release f. A global BA over every posed frame becomes the bottleneck once
    # windowed growth scales to thousands of frames; since the focal is global,
    # refine it (and re-triangulate) on a covisibility-SPREAD bounded subset,
    # leaving the windowed-grown poses elsewhere.
    rot = Rotation.from_rotvec(rvec).as_matrix()
    if ba_window > 0 and int(posed.sum()) > 120:
        sub = np.zeros(n_img, bool)
        sub[np.asarray(covis.thin_to(120), dtype=np.int64)] = True
        sub &= posed
        okb = sub[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        f, rvec, tvec, pts, _, _, _ = bundle_adjust(
            obs_c[okb], obs_i[okb], u[okb], rot, tvec, pts, f,
            n_img, n_cl, opt_f=True,
        )
        pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f)
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        xc = Rotation.from_rotvec(rvec[obs_i[ok]]).apply(pts[obs_c[ok]]) + tvec[obs_i[ok]]
        rn = np.linalg.norm(
            make_cam(f).ray_to_pixel_batch(np.ascontiguousarray(xc)) - u[ok], axis=1
        )
        keep = rn < TRIM_PX
        res = np.where(np.isnan(rn), np.inf, rn)
        inl = float((res < 2.0).mean())
    else:
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        f, rvec, tvec, pts, keep, res, inl = bundle_adjust(
            obs_c[ok], obs_i[ok], u[ok], rot, tvec, pts, f,
            n_img, n_cl, opt_f=True,
        )

    if backbone > 0 and int(posed.sum()) < n_img:
        # Complete the structure at the backbone poses, then batch-resect the
        # remaining frames against it (no BA).
        pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f)
        pb = int(posed.sum())
        rvec, tvec, pts, posed, nacc = batch_resect(
            rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis, aux
        )
        print(f"  [batch-resect: +{nacc} frames ({pb} backbone -> "
              f"{int(posed.sum())}/{n_img}) at {time.perf_counter() - _T0:.0f}s]")
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        xc = Rotation.from_rotvec(rvec[obs_i[ok]]).apply(pts[obs_c[ok]]) + tvec[obs_i[ok]]
        proj = make_cam(f).ray_to_pixel_batch(np.ascontiguousarray(xc))
        rn = np.linalg.norm(proj - u[ok], axis=1)
        keep = rn < TRIM_PX
        res = np.where(np.isnan(rn), np.inf, rn)
        inl = float((res < 2.0).mean())
    # Hierarchical (multigrid-style) V-cycles: coarse solve on a spread camera
    # subset with a coverage-constrained representative track set, prolongation
    # of the corrections to every fine camera, pose-only smoothing.  Corrects
    # the long-wavelength drift the windowed growth BAs cannot see.
    vc_mode = os.environ.get("SFMTOOL_VC_MODE", "")
    vc_fn = {"rot": vcycle_rot, "photo": vcycle_photo}.get(vc_mode, vcycle)
    vc_kw = (
        {"obs_f": data["obs_f"][act], "names": data["names"]}
        if vc_mode == "photo"
        else {}
    )
    for cy in range(int(os.environ.get("SFMTOOL_VCYCLES", "0"))):
        t_cy = time.perf_counter()
        rvec, tvec, pts = vc_fn(
            rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis,
            data["adm_rank"], **vc_kw,
        )
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        xc = Rotation.from_rotvec(rvec[obs_i[ok]]).apply(pts[obs_c[ok]]) + tvec[obs_i[ok]]
        rn = np.linalg.norm(
            make_cam(f).ray_to_pixel_batch(np.ascontiguousarray(xc)) - u[ok], axis=1
        )
        keep = rn < TRIM_PX
        res = np.where(np.isnan(rn), np.inf, rn)
        inl = float((res < 2.0).mean())
        print(f"  [v-cycle {cy + 1}: {time.perf_counter() - t_cy:.1f}s, "
              f"inlier<2px {100 * inl:.1f}%]")
        compare_to_reference(data["names"], rvec, tvec, f, mask=posed)

    print(
        f"[external seed completed at {time.perf_counter() - _T0:.0f}s: "
        f"f {f:.1f}, inlier<2px {100 * inl:.1f}% of its {int(ok.sum())} obs, "
        f"{int(posed.sum())}/{n_img} posed]"
    )
    return f, rvec, tvec, pts, posed, ok, keep, res


def main():
    global _CAM_WH
    data = load_clusters()
    all_c, all_i = data["obs_c"], data["obs_i"]
    n_img, n_cl = data["n_img"], data["n_cl"]
    dims = np.asarray(data["dims"], dtype=np.float64)
    # The images share one size, so one shared pinhole serves every projection;
    # observations are the FULL (un-centered) pixel positions throughout.
    _CAM_WH = tuple(data["dims"][0])
    all_u = data["obs_uv"]
    print(
        f"{WS}: {n_img} images, {n_cl} clusters (span >= {MIN_SPAN_BA}), "
        f"{len(all_c)} observations"
    )

    # BA working set: the best MAX_CLUSTERS clusters in admission order.
    # Growth, resection, and triangulation always see every cluster
    # (connectivity must not starve); only the BAs are restricted to the
    # representative subset.
    ba_cl = data["adm_rank"] < MAX_CLUSTERS
    if n_cl > MAX_CLUSTERS:
        print(f"BA set: best {MAX_CLUSTERS} of {n_cl} clusters by {ORDER}")
    # Warp-depth aux data: per-obs sqrt|det| magnification and each
    # cluster's reference image (for the depth-ratio resection init).
    ds_all = np.sqrt(np.maximum(np.abs(np.linalg.det(data["obs_warp"])), 1e-12))
    ref_img = np.full(n_cl, -1, np.int64)
    ref_img[all_c[data["obs_ref"]]] = all_i[data["obs_ref"]]
    active_cl = np.zeros(n_cl, bool)

    # Seed-required: this stage COMPLETES a stage-1 fast-bootstrap seed; it no
    # longer searches for its own (that half — tier search, affine
    # factorization, f-grid focal scan — is superseded by the more robust,
    # all-native fast bootstrap).  The seed (fast-pinhole.json: focal +
    # order-free initial poses + confidence flags) comes from
    # scripts/exp_fast_seed.py.  Pass SFMTOOL_SEED=<path>, or drop a
    # fast-pinhole.json in the workspace.
    seed = os.environ.get("SFMTOOL_SEED")
    if not seed and (WS / "fast-pinhole.json").exists():
        seed = str(WS / "fast-pinhole.json")
    if not seed:
        sys.exit(
            "no seed found: run scripts/exp_fast_seed.py on this workspace "
            "first (it writes fast-pinhole.json), or pass SFMTOOL_SEED=<path>"
        )
    r = external_seed_complete(
        data, seed, active_cl, all_c, all_i, all_u,
        n_img, n_cl, ds_all, ref_img, dims, ba_cl,
    )
    if r is None:
        return
    f, rvec, tvec, pts, posed, ok, keep, res = r

    act = active_cl[all_c]
    act_idx = np.nonzero(act)[0]
    full_keep = np.zeros(len(all_c), bool)
    full_keep[act_idx[np.nonzero(ok)[0]]] = keep
    full_res = np.full(len(all_c), np.inf)
    full_res[act_idx[np.nonzero(ok)[0]]] = res
    keep, res = full_keep, full_res
    rk = res[keep]
    n_pts = len(np.unique(all_c[keep]))
    print(
        f"\nbootstrap result: f = {f:.1f} px, {n_pts} points, "
        f"{keep.sum()}/{len(all_c)} observations kept, "
        f"{int(posed.sum())}/{n_img} images posed"
    )
    print(
        f"reprojection (kept): rms {np.sqrt((rk**2).mean()):.2f} px, "
        f"median {np.median(rk):.2f} px; inlier<2px {100 * (res < 2).mean():.1f}% "
        f"of all obs"
    )

    if _DEPTH_COH:
        # Warp-depth coherence at resection time (final-growth resections
        # only appear once each; scan-phase entries repeat per focal).
        coh = np.array([c for _, c, _ in _DEPTH_COH])
        worst = sorted(_DEPTH_COH, key=lambda t: -t[1])[:5]
        print(
            f"\nwarp-depth coherence at resection ({len(coh)} resections): "
            f"median {np.median(coh):.3f}, p90 {np.percentile(coh, 90):.3f} "
            f"|log depth ratio|"
        )
        print(
            "  worst: "
            + ", ".join(f"img {i} {c:.2f} (inl {v:.0%})" for i, c, v in worst)
        )

    compare_to_reference(data["names"], rvec, tvec, f, mask=posed)

    out = WS / "sfmr" / os.environ.get("SFMTOOL_OUT", "bootstrap-pinhole.sfmr")
    save_sfmr(data, f, rvec, tvec, pts, keep, res, out)


if __name__ == "__main__":
    main()
