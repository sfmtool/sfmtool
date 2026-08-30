# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment: rapid pinhole-camera estimate from cluster patches.

The first stage of a divide-and-conquer bootstrap: from a workspace holding
a `*-clusters-patches.matches` file, get to a good shared SIMPLE_PINHOLE
estimate (focal + a small set of posed views) as fast as possible.

  1. Pairwise focal vote: wide-baseline image pairs each estimate a
     fundamental matrix (native RANSAC) and cast a Bougnoux focal vote;
     the median picks the BASIN — no structure, so no bas-relief trap.
  2. Covisibility seed groups (parallax-gated: a video's most-covisible
     frames are its most static ones) -> affine ALS factorization +
     Tomasi-Kanade metric upgrade; grow an 8-image core by P3P resection
     at a probe focal from the vote.
  3. Ladder-widen (far-first verified resections), then one photometric
     verification pass (feature-scaled patch localization) that un-poses
     junk rungs the geometric gates missed.
  4. Fixed-focal scan across a vote-centred grid on the widened geometry
     (coarse ranking + heavy refits, arbitrated toward the bias-corrected
     vote when the structure has no opinion), then an iterated release
     with an anti-affine basin guard.
  5. When the best consensus stays below the healthy band (non-rigid or
     f-degenerate capture), report the structure-free vote instead of the
     structure estimate — the flagged cases are exactly where structure
     is a lottery and the vote is not.

Run: pixi run -e dev python scripts/exp_fast_seed.py <workspace> [ref.sfmr]

Prints the focal + camera errors vs the reference solve (when one exists)
and writes `<workspace>/fast-pinhole.json` with the estimate for later
stages to consume.
"""

import dataclasses
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sfmtool._sfmtool.geometry import (
    CameraIntrinsics,
    bundle_adjust as _bundle_adjust,
    estimate_absolute_pose,
    factorize_affine,
    inlier_fraction as _inlier_fraction,
    refine_absolute_pose as _refine_absolute_pose,
    reprojection_residuals as _reprojection_residuals,
)

WS = Path(sys.argv[1] if len(sys.argv) > 1 else "e_seoul_ws")
REF = Path(sys.argv[2]) if len(sys.argv) > 2 else None
_T0 = time.perf_counter()
# The cluster SELECTION stage 1 actually solves on — the derivation
# `load_clusters` admitted (reference/kept members, optional thinning
# restriction, span filter).  The seed's restriction stage narrows THIS handle
# to the seed images, so the restricted file's clusters are a subsequence of
# the ones stage 1 certified evidence against.
_SEL_MATCHES = None
# The loader's own context (image table, admission span, warp passthrough), so
# the hypothesis loop can repackage a derived selection exactly like the
# admission it came from.
_LOAD_CTX = {}

# BA row budget: each bundle adjustment keeps rows from at most MAX_CL
# clusters, selected among the clusters VISIBLE in that BA's own window (>= 1
# observation in a currently-posed image) and ordered by the GLOBAL admission
# rank (span-major, cluster-id tiebreak).  Restrict first, truncate second: a
# global rank prefix is a fixed set of clusters, and a seed window is a small
# covisibility neighbourhood whose local tracks can all rank below the budget
# — that window's mask is then EMPTY and every measurement taken on it (probe
# consensus, widen rungs, the photometric floor, the focal scan) runs on zero
# rows.  Window-local truncation guarantees a non-empty mask wherever the
# window has observations at all, at the same row ceiling.  The global rank
# stays the ordering: it encodes the trusted quality order, whereas ranking by
# IN-WINDOW span promotes junk-dominated high-span prefixes.  Ranks are a
# unique permutation over clusters, so the k-th smallest visible rank is an
# exact, order-free threshold.  Covisibility, factorization, resection, and
# triangulation are never budgeted — a capped working set starves the seed
# window (dino's high-span prefix is junk-dominated and a 2000-cluster load cap
# collapsed its factorization outright).
MAX_CL = int(os.environ.get("SFMTOOL_MAX_CL", "3000"))
SCAN_CAP = int(os.environ.get("SFMTOOL_SCAN_CAP", "8"))  # core images per try
# Wide-baseline shell: after the covisibility-driven core, resect up to this
# many far images (widest viewpoint angles).  The focal is unobservable on a
# near-affine core — a sliver of a long orbit fits ANY focal at high inlier
# fraction (living-room: 86% inliers at f 45% off) — and the shell is what
# makes the scan discriminate and the released f converge.
SHELL = int(os.environ.get("SFMTOOL_SHELL", "6"))
# Per-image BA row cap: each image keeps only its best observations (by
# admission rank), so BA cost stays flat as the shell widens the image set.
OBS_PER_IMG = int(os.environ.get("SFMTOOL_OBS_PER_IMG", "250"))
# Anchor-cluster budget for the photometric verification pass.
N_ANCHORS = int(os.environ.get("SFMTOOL_ANCHORS", "400"))
# NEAR-STATIC GATE.  A probed core whose translation parallax is under this
# many degrees is rejected as a seed: a video's most-mutually-covisible frames
# are where the camera moved least, and such a core fits ANY focal at high
# inlier fraction while its depths are unusable (the affine-band study's
# bas-relief edge).  The gate is advisory in the attempt (a rejected seed is
# still finished when nothing else survives) and its verdict is RECORDED on the
# release, but it does not rank: the parallax of a seed says what its geometry
# could observe, not whether the solve that grew from it is right, and on
# 20250712_195736354 the class rule promoted the wrong attempt on exactly that
# confusion (a 0.71 deg seed grew the solve a human called good, and the
# 1.15 deg seed beside it grew the one they called bad).
NEAR_STATIC_DEG = 1.0

# ROTATION-DISAGREEMENT INDICTMENT.  An attempt's poses are checked against the
# INDEPENDENT rotation-only fit of the same frames (the far-field layer's own
# skeleton, fit from pair evidence alone).  A solve whose camera motion is
# wrong disagrees with that fit; a solve whose motion is right agrees with it
# whatever its parallax was.  The human review that produced these bounds read
# a good attempt at 0.03 deg and a bad one at 1.36 deg against the canonical
# solve -- a 45x separation spanning more than a degree -- so an indictment
# needs BOTH an absolute floor and a wide margin over the competitor, set far
# inside that separation so only outliers are cut.
ROT_DISAGREE_ABS_DEG = 0.5
ROT_DISAGREE_RATIO = 3.0

# ABSORPTION INDICTMENT (secondary, indictment-only).  A finite solve that
# swallows the far field into a fictitious depth shell leaves its own far layer
# with little to say that the solve does not already claim, so a LOW
# `obs_beyond` beside a competitor's high one is evidence of absorption.  Two
# confirmed kills sat at 492-vs-2780 and 1087-vs-2029; one miss sat at
# 1768-vs-3201 on a parallax-poverty 0.96 capture.  The middle two are 1.87x
# and 1.81x, which no ratio can separate, so the ratio bound is set ABOVE both
# and the conjunct fires only on the extreme end -- when in doubt, no
# indictment.  A far-field-dominated capture is excluded outright: there the
# far layer is most of the scene and a large `obs_beyond` says nothing about
# absorption.
ABSORPTION_RATIO = 3.0
FAR_DOMINANT_POVERTY = 0.85
# Probe measurability gate.  A probe reports its consensus over its own BA row
# mask; a window that measures at or near zero consensus is UNMEASURED, not
# good, and must not commit on parallax alone (a mask with too few rows to
# constrain anything, a degenerate window, a BA that never took a step).  Both
# bars are per-attempt, over the covisibility group chunks only (the rotation
# core is a different seed class and does not set the bar).
#   ABS is the standing floor: the lowest healthy committed probe on the fleet
#     is 40.5% (Daegu), so a near-zero consensus is never a real window.
#   REL kills a window measured far below its own attempt's siblings: 38.1%
#     against 85.0%/92.3% siblings (20250706_204855796), while healthy commits
#     sit at ratio 1.00 (Daegu) and 0.83 (vid2) of their attempt's max.
PROBE_GATE_ABS = float(os.environ.get("SFMTOOL_PG_ABS", "0.15"))
PROBE_GATE_REL = float(os.environ.get("SFMTOOL_PG_REL", "0.5"))
F_GRID = [0.55, 0.7, 0.9, 1.2, 1.6]  # focal candidates, units of max(w, h)

# ── Camera-model escalation (specs/core/geometry/focal-vote.md § Camera-Model Columns) ─
#
# The whole pipeline is pinhole-only.  Handed a fisheye capture it does not
# announce that; it degrades quietly (kerry: a center-out solve at 59% inliers
# that every per-image gate passes).  The vote kernel can arbitrate the camera
# MODEL, but only by paying for two self-consistency scans per pair per column.
# So the pinhole vote runs FIRST and alone, exactly as before, and the second
# column is added only when that vote is itself LOW CONFIDENCE — the second
# column then doubles as a cheap cross-check on a shaky pinhole consensus,
# which is the only situation where paying for it buys anything.
#
# The trigger is a disjunction over the pinhole vote's OWN diagnostics.  Each
# arm says "this consensus is not to be trusted", and each cut point comes
# from the kernel's own constants or from the fleet's measured distribution —
# no fitted magic numbers:
#
#   no_consensus         focal_px is None.  Fewer than MIN_POOL votes pooled;
#                        there is no pinhole answer to trust.  (Fires on 0/40
#                        of the current fleet; present because it is the
#                        definitional case.)
#   rotation_railed      The pooled focal came from the rotation family and
#                        that family's answer sits within ONE GRID STEP of the
#                        bottom of the orthogonality scan's grid.  The kernel
#                        scans 48 log-spaced points over [0.3, 4.0] x max(w,h)
#                        (ORTHO_GRID_{LO,HI,N}), so one step is
#                        (4/0.3)^(1/47) = 1.0566; an answer there is a scan
#                        that ran out of grid rather than one that found an
#                        interior minimum.  This is the known fisheye rail:
#                        kerry at 480 px lands exactly on the floor (ratio
#                        1.000) while the nearest pinhole capture sits 2.2
#                        grid steps above it (1.153).
#   family_disagreement  The kernel's own FAMILY_DISAGREEMENT_BAND (0.25 in
#                        log-focal) fired: the epipolar and rotation families
#                        measured incompatible focals and the reported
#                        consensus is one family's median with the other
#                        thrown away.  Two independent estimators disagreeing
#                        beyond the kernel's own bimodality band IS low
#                        confidence, by the kernel's own definition.
#   thin_pool            n_pool <= 9.  The one arm without a kernel constant
#                        behind it: 9 is the tightest bar that still reaches
#                        every fisheye capture on the fleet (KerryPark360 and
#                        OmniTemple1 both pool exactly 9 and no other arm
#                        catches them).  It happens to be half the kernel's
#                        MAX_EPIPOLAR_PAIRS budget of 18.  The pinhole
#                        datasets it additionally admits all pool 3-8, i.e.
#                        strictly thinner still, so it is not admitting
#                        healthy votes.
#
# Measured over the 40-dataset fleet: fires on 4/4 fisheye captures and 9/36
# pinhole ones, at +0.4-0.8 s each on the pinhole side (5.3 s total, against a
# 30-60 s seed run per dataset).  Every one of those 9 is a genuinely weak
# pinhole vote — 7 tripped the kernel's own bimodality band, 5 pooled 8 or
# fewer votes — so the extra column is a confidence cross-check there, not
# waste.
#
# VERDICT CONFIRMATION.  The arbitration's bare verdict is NOT trustworthy on
# its own: run unconditionally it returns EquidistantFisheye on three
# rectilinear fleet captures (BadlandPanorama, 20240614_224244438,
# MossyRailing), and its certified-mass margin (1.08x-3.60x on those three)
# overlaps a true fisheye capture's (OmniTemple1, 1.46x), so no margin cut
# separates them.  What does separate them, with no threshold at all, is WHICH
# CELLS carry the equidistant column's evidence: all three false verdicts win
# on the epipolar cell with EXACTLY ZERO model-informative rotation-cell mass,
# while all four true fisheye captures earn mass in both cells (rotation-cell
# mass 4 / 26 / 41 / 44).  That asymmetry is structural rather than lucky —
# the rotation cell fits a pure rotation of unit RAYS, which a wrong ray map
# cannot fake, whereas the epipolar cell's essentialness residual is a weaker
# statement a wrong map can partially satisfy.  So the fisheye verdict is
# recorded only when BOTH cells corroborate it.
FISHEYE_THIN_POOL = int(os.environ.get("SFMTOOL_FISHEYE_THIN_POOL", "9"))
# Mirrored from crates/sfmtool-core/src/geometry/focal_vote.rs.
_ORTHO_GRID_LO, _ORTHO_GRID_HI, _ORTHO_GRID_N = 0.3, 4.0, 48
_ORTHO_GRID_STEP = (_ORTHO_GRID_HI / _ORTHO_GRID_LO) ** (1.0 / (_ORTHO_GRID_N - 1))
_FAMILY_DISAGREEMENT_BAND = 0.25

# Canonical camera frame (-Z forward, +Y up) throughout; full-pixel
# observations; shared SIMPLE_PINHOLE with the principal point at the centre.
_CAM_WH = None
_VOTE_POVERTY = 0.0  # set by focal_vote: median H/F inlier ratio over pairs
_VOTE_ROT_N = 0  # set by focal_vote: number of rotation self-calibration votes
# Set by focal_vote: the pinhole vote pool's own log-focal IQR — the
# structure-free measurement's dispersion, i.e. how tightly the pairwise
# evidence pins the focal with no structure involved.  Read as the vote's
# PRECISION BAND by the finalization's arbitration (see `vote_band`).
_VOTE_SPREAD = 0.0
# Set by focal_vote when the escalated two-column vote returns a CONFIRMED
# EquidistantFisheye verdict: {"focal_px", "margin", "mass_pinhole",
# "mass_equidistant", "mass_epipolar", "mass_rotation", "trigger"}.  None
# whenever the escalation did not run, the verdict was Pinhole, or the verdict
# was not corroborated by both cells.
_VOTE_FISHEYE = None


def elapsed():
    return time.perf_counter() - _T0


# ── Fisheye seed camera context (scripts/notes-fisheye-seed.md, Phases 1-6) ──
#
# A per-run CAMERA CONTEXT — (model, focal) — behind every camera this script
# builds.  Default: SIMPLE_PINHOLE, which is the code path this script has
# always run and stays byte-identical.
#
# ROUTING.  A CONFIRMED equidistant verdict from ``_escalate_camera_model`` —
# the both-cells corroboration — installs the fisheye context, and it does so BY
# DEFAULT: routing needs no opt-in.  SFMTOOL_FISHEYE_SEED is an OVERRIDE on that
# decision, not a precondition for it:
#
#   unset (or anything but "0")  route on a confirmed verdict — the default.
#   "0"                          never route: the verdict stays an annotation,
#                                the seed solves pinhole and degrades gracefully
#                                (center-out).
#
# What no setting can do is route WITHOUT a confirmed verdict.  An unconfirmed
# verdict stays annotation-only exactly as before, and that is load-bearing
# rather than conservative: run unconditionally the arbitration claims three
# rectilinear fleet captures (see VERDICT CONFIRMATION above), and all three win
# on the epipolar cell with zero rotation-cell mass.  A capture with no verdict
# has no equidistant focal to build a context from at all.
#
# The fisheye model is EQUIDISTANT_FISHEYE — the exact SIMPLE_PINHOLE analog
# (one focal, centred principal point) parameterizing `theta = r/f` — at the
# verdict's EQUIDISTANT focal, which is a different quantity from the pinhole
# vote and not commensurable with it.  (Phase 1-2 carried the same map as
# SIMPLE_RADIAL_FISHEYE with k1 = 0 by hand; Phase 3a made it a native model
# with closed-form projection both ways and an analytic pixel Jacobian, so the
# native BA and pose refinement linearize analytically instead of
# central-differencing.)
#
# What Phase 1 buys: every geometric primitive here reaches its camera through
# ``make_cam``, so ``triangulate``, ``p3p_resect``, ``pose_refine``,
# ``reproj_res_one`` and the native BA all become equidistant under the
# context, and the native kernels behind them are model-generic including at
# theta >= 90 deg (audited, with Rust tests).  Phase 2 replaced the
# pinhole-specific ESTIMATORS (affine factorization, pair parallax, rotation
# core) with ray-space equivalents, and Phase 3b the last one: the focal is
# scanned over an equidistant band and RELEASED, so the commit bar's
# focal-observability term and the scan-derived confidence flags apply to a
# fisheye capture exactly as they do to a pinhole one.
def fisheye_routing_override(value):
    """The routing override's tri-state, factored out so it is testable.

    ``"0"`` forces the fisheye branch off; anything else — unset, or any other
    value — leaves the routing decision to the confirmed verdict.  See the block
    comment above."""
    return value != "0"


FISHEYE_SEED = fisheye_routing_override(os.environ.get("SFMTOOL_FISHEYE_SEED"))
_CAM_CONTEXT = {"model": "SIMPLE_PINHOLE", "focal": None}

# ── Equidistant focal band (scripts/notes-fisheye-seed.md, Phase 3) ──────────
#
# Under `theta = r/f` the focal and the field of view are tied by
# `f = r_edge / theta_edge`, so a beyond-180-degree capture's plausible focals
# run well below the pinhole plausibility floor (0.3 x max(w, h)) — a 480 px
# sensor at 200 deg implies f ~ 137 px.  The band is the focal-vote kernel's
# own FOV-derived one (specs/core/geometry/focal-vote.md, Camera-Model Columns), reused
# verbatim so the scan cannot rank a focal the vote would not have scanned.
FISHEYE_BAND = (0.075, 3.0)
# Per-step ratio of the five-point equidistant scan grid.  The pinhole grid
# skews UPWARD because Bougnoux votes run consistently low; the equidistant
# column carries no such measured directional bias, so its grid is log-symmetric
# about the verdict.  The span (0.756x .. 1.323x) matches the pinhole grid's.
FISHEYE_SCAN_RATIO = 1.15

# Floor on the structure-free vote's PRECISION BAND (log-focal), the band the
# finalization's arbitration reads (Phase 6, Item 2).  The band itself is the
# vote pool's own log-focal IQR — an instrument reporting its own dispersion,
# data-derived per capture rather than a constant.  A pool can be unanimous and
# still carry a systematic error, though, so the band never closes below the
# accuracy the column has been MEASURED to have: the equidistant column reads
# +2.06% / +1.93% against the two rig-calibrated captures' best-fit equidistant
# focals (KerryPark480 / KerryPark360), and on uncontaminated support a
# free-focal release lands within 0.7% of the vote (the coverage-complement
# experiment).  2% is the largest of those, i.e. the choice most permissive to
# the structure candidate.  On the pinhole branch the measured pool IQR runs
# 0.04-0.28, so the floor never binds there.
VOTE_BAND_FLOOR_LOG = float(np.log(1.02))

# Trim bar (px) on the rotation core's rotation-locked translation resection,
# and a per-resection trace of the residual distribution it trims.  Both are
# experiment handles for re-deriving the bar (Phase 6, Item 3): Phase 5's
# resection fix put the whole periphery back into that residual population, so
# the bar is worth re-measuring rather than inheriting.  Default unchanged.
RESECT_MAX_PX = float(os.environ.get("SFMTOOL_RESECT_MAX_PX", "8.0"))
RESECT_TRACE = os.environ.get("SFMTOOL_RESECT_TRACE", "0") == "1"


def fisheye_focal_band():
    """The FOV-derived plausible equidistant focal band `(lo, hi)` in px."""
    m = max(_CAM_WH)
    return FISHEYE_BAND[0] * m, FISHEYE_BAND[1] * m


def focal_floor():
    """Absolute plausibility floor on a released focal, px — FOV-derived under
    an equidistant context, the pinhole band's 0.3 x max(w, h) otherwise."""
    if fisheye_stage1():
        return fisheye_focal_band()[0]
    return 0.3 * max(_CAM_WH)


def fisheye_focal_grid(f_center, n=5):
    """The equidistant scan grid: ``n`` log-spaced candidates centred on
    ``f_center``, clipped to the FOV-derived band."""
    lo, hi = fisheye_focal_band()
    k = np.arange(n) - (n - 1) // 2
    return np.clip(f_center * FISHEYE_SCAN_RATIO**k, lo, hi)


def set_camera_context(model, focal=None):
    """Install the per-run camera context (see the block comment above).

    ``model`` is a COLMAP model name; ``focal`` is the context focal used by
    ``make_cam()`` when no explicit focal is passed."""
    _CAM_CONTEXT["model"] = model
    _CAM_CONTEXT["focal"] = None if focal is None else float(focal)


def camera_context():
    """The active ``(model, focal)`` context as a plain dict (a copy)."""
    return dict(_CAM_CONTEXT)


def bootstrap_module():
    """``exp_pinhole_bootstrap`` with THIS run's workspace and CAMERA CONTEXT
    installed — the only way anything here reaches the bootstrap's writers.

    The bootstrap keeps its own camera context, and its writers read it: the
    densification (``dense_structure``) triangulates through ``make_cam`` and
    the writer (``save_sfmr``) stamps that same camera on the file.  A
    fisheye-routed run that skips this step therefore writes equidistant poses
    with a PINHOLE densification under a ``SIMPLE_PINHOLE`` stamp — the poses
    survive the mistake and the structure does not (measured on OmniHilltop's
    hypothesis releases: own-track reprojection 9-110 px read as equidistant
    against 1.0-2.0 px read as pinhole, i.e. a cloud that was never solved).
    """
    import exp_pinhole_bootstrap as B

    B.WS = WS
    # Stage 1 owns the model and the focal; it never owns a distortion (the
    # spline rung is the finalization's, and it promotes the bootstrap's own
    # context in place).  Re-installing here must therefore not demote a
    # promotion that already happened — a later call would otherwise reset
    # the camera the finalized reconstruction was written with.  A promoted
    # model is anything stage 1 itself never installs, so the check is over
    # stage 1's own two models rather than a list of promotions.
    if B.camera_context()["model"] in ("SIMPLE_PINHOLE", "EQUIDISTANT_FISHEYE"):
        B.set_camera_context(_CAM_CONTEXT["model"], _CAM_CONTEXT["focal"])
    return B


def make_cam(f=None):
    """The context camera at focal ``f`` (the context focal when omitted).

    SIMPLE_PINHOLE by default; EQUIDISTANT_FISHEYE under a confirmed,
    opted-in fisheye context.  Both models take the same three parameters —
    one focal and a centred principal point — so this builds one dict."""
    w, h = _CAM_WH
    if f is None:
        f = _CAM_CONTEXT["focal"]
    params = {
        "focal_length": float(f),
        "principal_point_x": w / 2.0,
        "principal_point_y": h / 2.0,
    }
    return CameraIntrinsics.from_dict(
        {
            "model": _CAM_CONTEXT["model"],
            "width": int(w),
            "height": int(h),
            "parameters": params,
        }
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def repackage_selection(sel_h, names, dims, want_warp=False, refine_radius=None):
    """A cluster selection handle as the flat observation dict stage 1 solves
    on: one row per member, cluster ids dense in the handle's own numbering,
    plus the global admission rank (span-major, cluster-id tiebreak).

    Every selection reaches the pipeline through here — the loader's initial
    admission and the hypothesis loop's coverage complements alike — so a
    complement is loaded exactly like the admission it was derived from.
    ``names``/``dims`` come from the file the chain started at: an id
    restriction never touches the image table, so every handle in the chain
    shares that image frame.
    """
    starts_s = np.asarray(sel_h.cluster_starts, dtype=np.int64)
    sizes = np.diff(starts_s)
    n_sel = len(sizes)
    order = np.lexsort((np.arange(n_sel), -sizes))
    rank = np.empty(n_sel, np.int64)
    rank[order] = np.arange(n_sel)
    aff_s = np.asarray(sel_h.member_affines)
    # The affine's leading 2x2 is the member's ABSOLUTE affine shape S
    # (.matches v5): the map from the detector's canonical unit frame onto
    # this member's image pixels.  Only the warp passthrough reads it; the
    # geometry the seed solves on is the last column (the keypoint), so no
    # `.sift` file is opened here at all.
    shapes = aff_s[:, :, :2]
    obs_c = np.repeat(np.arange(n_sel, dtype=np.int64), sizes)
    out = {
        "names": names,
        "dims": dims,
        "obs_c": obs_c,
        # The selection keeps the file's own image table, so its member
        # image indexes are already in the loader's frame.
        "obs_i": np.asarray(sel_h.member_images, dtype=np.int64),
        "obs_f": np.asarray(sel_h.member_features, dtype=np.int64),
        "obs_uv": np.ascontiguousarray(aff_s[:, :, 2], dtype=np.float64),
        "adm_rank": rank,
        "cl_quality": np.asarray(sel_h.cluster_worst_consistency(), dtype=np.float64),
        "n_img": len(names),
        "n_cl": n_sel,
    }
    if seed_eval_on():
        # The candidate battery's warp channel reads the member's ABSOLUTE
        # shape S, which is this same leading 2x2: two members of one cluster
        # carry the measured pixel-to-pixel warp `S_j S_i^-1` between their
        # images.  Carried here rather than re-parsed, so the battery costs the
        # file no second read; the rows are the observation rows, so it indexes
        # exactly like `obs_uv`.
        out["obs_shape"] = np.ascontiguousarray(shapes, dtype=np.float64)
    if want_warp:
        # The surfel writer wants the REFERENCE-RELATIVE warp, which v5 no
        # longer stores: recover it as W = S.S_ref^-1 through each cluster's
        # own reference row (the bootstrap's shared helper, so this loader
        # and B.load_clusters cannot drift apart).
        import exp_pinhole_bootstrap as B

        out["obs_warp"] = np.ascontiguousarray(
            B.relative_warps(shapes, obs_c, sel_h.reference_members),
            dtype=np.float64,
        )
        out["obs_ref"] = np.asarray(sel_h.member_status) == 0
        out["refine_radius"] = refine_radius
    return out


# ── Rung 1: alias-free basin exploration ────────────────────────────────────
#
# The seed does not have to answer the capture in one pass, and its first pass
# should not try to.  Rung 1 asks one question only: WHICH BASINS does this
# capture's geometry support?  It answers it on the coarsest clusters alone,
# and it stops before the photometric finalization.
#
# Coarse features are the alias-free evidence.  On a repeating scene texture --
# a tiled floor, a brick wall, a railing -- fine features match
# self-consistently at FALSE lattice offsets, and the aliased basin is
# internally clean and OUTNUMBERS the true one, so a pass that admits
# everything commits to the lie (20250906_211742965's tile-grid floor: the
# unfiltered run never found the floor-nearest basin at all; admitting only the
# clusters whose support is wider than the tile period surfaced it as
# hypothesis 1).  A feature wider than the period cannot alias that way, so the
# coarse admission is not a speed trick -- it is the only admission on which
# the basin structure is legible.
#
# The rung's OUTPUT IS THE HYPOTHESIS SET.  Every committed hypothesis is
# written and kept; the arbitration ranks them and names a winner for whatever
# runs next, but it discards nothing, because the set is worth more than the
# stamped winner: on this fleet the winner is unqualified far more often than
# the set is empty, and the losing hypothesis is repeatedly the one that
# explains the part of the capture the winner cannot reach.  A rung that threw
# the alternatives away would be spending the coarse admission's whole point.
#
# N = 3000 is the fleet's SEEDABILITY floor, measured, not chosen: sweeping the
# fraction bars first showed a scene-scale threshold lets the kept population
# span 40x across a fleet (so "one bar" is not one working set), and sweeping
# counts then bracketed the floor -- N = 1000 seeded only 25 of 42 captures at
# 34.5% median focal error, N = 5000 seeded 42 of 42 at 5.0%, and the interior
# sweep put the knee at 3000.  That is a floor for FINDING THE BASIN, not for
# measuring the lens: metric anchoring wants far more evidence than 3000
# clusters carry, and it belongs to a later rung that starts from the basin
# this one hands it.
@dataclasses.dataclass
class Rung:
    """Everything rung 1 carries between its phases, in one object.

    The rung used to live in three module dicts that each phase reached into,
    which made the order the phases ran in part of the contract and invisible.
    It is one value now, created by the loader when the rung is armed, passed
    to the seams that need it, and None everywhere the rung is not.

    ``SFMTOOL_SEED_RUNG1=N`` arms the WHOLE validated stack: the coarse top-N
    admission, the vote measured on the full admission, the per-group local
    re-admission at the same N, the far-field layers, the evidence ranking with
    its runner-up commits, the stage-1 stop, and the ``candidate_solves``
    product.  There are no sub-switches: the pieces were separated while each
    was being validated, and separately switchable pieces of one validated
    stack are combinations nobody measured."""

    #: The cluster budget, capture-wide and (unless overridden) per group.
    n: int
    #: The per-group budget.  ``SFMTOOL_SEED_LOCAL_ADMISSION`` overrides it as a
    #: plain integer -- the only knob left, kept for the N_local sweep, which
    #: needs to move the group budget without moving the capture's.
    n_local: int
    #: The run's identity.  It lives in the manifest rather than in file names,
    #: so the product's paths are stable and a new run replaces it wholesale.
    stamp: str
    #: Where releases accumulate before the product is swapped into place.
    stage: Path
    # ── the global (stage A) admission, as the manifest reports it ──
    n_clusters_total: int = 0
    n_clusters_kept: int = 0
    min_kept_radius_px: float = 0.0
    # ── the referee's evidence: the FULL admission's observation arrays,
    # held only until the vote has read them ──
    vote_obs: tuple = ()
    vote_clusters: int = 0
    vote_observations: int = 0
    #: The same arrays, kept past the vote for the evaluation battery's
    #: two-view witness and released as soon as it has read them.  Empty
    #: whenever the battery is off, and whenever the coarse cut was a no-op
    #: (there the solve's own admission IS the full one).
    eval_obs: tuple = ()
    #: The committed FAR-FIELD layers' own arrays, in commit order, so the
    #: battery can measure a rotation-only member on its own model instead of
    #: recording it unmeasurable.  Each entry is what
    #: ``rotation_only_hypothesis`` handed back under ``_eval``.
    far_layers: list = dataclasses.field(default_factory=list)
    #: The RELAXED members committed off those layers, in commit order, so the
    #: battery measures each one on the finite channels its model now answers.
    #: Each entry is ``{"idx", "res", "result"}`` -- the manifest index, the
    #: committed ``res`` dict and the relaxation's own result.
    relaxed_layers: list = dataclasses.field(default_factory=list)
    # ── the pre-restriction handle and per-member radii the group-local
    # re-admission draws on ──
    handle: object = None
    m_cl: object = None
    m_img: object = None
    radius: object = None
    n_full: int = 0
    # ── the loader context a re-admission needs to repackage a selection ──
    min_span: int = 2
    names: tuple = ()
    dims: tuple = ()
    want_warp: bool = False
    refine_radius: float = 0.0
    # ── exploration cost controls, both defaulting to the measured behaviour ──
    #: Photometric verification of a widened window: ``full`` localizes anchors
    #: in every posed view and un-poses the frames that keep too little support;
    #: ``skip`` trusts the geometric gates alone.  It decides which frames
    #: survive the widen, so changing it is content-changing by construction --
    #: but under the coarse rung it is measurably inert: the 42-entry fleet at
    #: N=3000 logged ZERO un-posing events (rung1-eval-c-20260823), and the
    #: four-entry A/B produced bit-equal products, while the verify was the
    #: rung's largest cost bucket (0.6-3.3s/entry).  Wide clusters give every
    #: widened frame ample photometric support; the check earns its cost on
    #: fine-feature rungs, so ``skip`` is the COARSE rung's default and the
    #: machinery stays for the rungs that need it.
    verify: str = "skip"
    #: Cap on the covisibility-thinning ladder's levels (0 = uncapped, which is
    #: the measured behaviour).  A cap trades later, thinner working sets for
    #: time; whether those levels still find basins the earlier ones miss is a
    #: measurement, not an assumption.
    max_levels: int = 0
    # ── the committed set ──
    hypotheses: list = dataclasses.field(default_factory=list)
    # ── memoization across hypothesis explorations ──
    #
    # The complement exploration re-probes seed groups the first hypothesis
    # already probed, and the coverage claim it derives from does not touch what
    # a probe reads: the group's frames, the local re-admission over their
    # covisible neighbourhood, and the level's image restriction.  Given those
    # three, the whole factorize -> grow -> adjust chain is a deterministic
    # function of the matches file, so the second probe reproduces the first
    # number for number (measured on every arm-B and arm-C log).  Both memos are
    # keyed on exactly those three, so a hit is an identity rather than an
    # approximation.
    workset_memo: dict = dataclasses.field(default_factory=dict)
    probe_memo: dict = dataclasses.field(default_factory=dict)
    finish_memo: dict = dataclasses.field(default_factory=dict)
    memo_hits: int = 0

    def vote_admission(self, obs_c, obs_i, u):
        """The observations the pairwise focal vote is measured on: the FULL
        admission, since the coarse cut narrowed what the solve sees.

        The vote is the capture-level independent REFEREE
        (`specs/core/geometry/seed-hypothesis-loop.md`): it is measured once over the
        whole capture's pair graph, every hypothesis reads it rather than
        re-deriving one from its own restricted graph, and the arbitration
        measures each release against it.  A referee that reads only the
        evidence the contestants were given is not independent of them -- and
        the coarse restriction starves it hardest exactly where it matters,
        because dropping to a few thousand of a capture's clusters can leave
        too few surviving pairs to vote at all (six fleet entries returned NO
        vote under the rung's admission while voting perfectly well on the full
        one).

        Which clusters the SOLVE explores is a statement about what geometry to
        develop; it is not a statement about how much evidence the capture
        offers for its lens."""
        return self.vote_obs if self.vote_obs else (obs_c, obs_i, u)

    def local_admission(self, images):
        """The working admission for an attempt's IMAGE SET: the ``n_local``
        coarsest clusters carried by those images, as an ordinary derived
        selection.

        Coarseness is measured ON THOSE IMAGES (a cluster's widest member
        there), not capture-wide, so the ranking answers "what does this window
        see coarsely", which is the question the window's own solve depends on.
        Eligibility is the loader's span bar counted on the same images.

        The selection is derived from the PRE-restriction handle, so the group
        can reach clusters the global cut dropped -- that is the entire point.
        Its members are then the cluster's full member list, every image
        included, so the widen ladder still has structure to resect against
        wherever the admitted clusters are seen.

        Returns the repackaged observation dict (carrying ``local_admission``
        for the manifest), or None when the image set carries nothing."""
        on = np.isin(self.m_img, images)
        if not on.any():
            return None
        m_cl_on = self.m_cl[on]
        cnt = np.bincount(m_cl_on, minlength=self.n_full)
        r_loc = np.zeros(self.n_full)
        np.maximum.at(r_loc, m_cl_on, self.radius[on])
        elig = np.nonzero(cnt >= self.min_span)[0]
        if len(elig) < 2:
            return None
        # Radius descending, cluster id ascending among ties -- the same stable
        # ordering the global cut uses, so the two are one rule at two scopes.
        top = np.sort(elig[np.argsort(-r_loc[elig], kind="stable")[: self.n_local]])
        sel = self.handle.select_clusters(
            min_span=self.min_span, restrict_cluster_ids=[int(c) for c in top]
        )
        d = repackage_selection(
            sel,
            self.names,
            self.dims,
            want_warp=self.want_warp,
            refine_radius=self.refine_radius,
        )
        d["local_admission"] = {
            "n_clusters": int(d["n_cl"]),
            "min_radius_px": float(r_loc[top].min()),
            "n_images": int(len(images)),
        }
        return d


# ── The 2-knot spline lens release ──────────────────────────────────────────
#
# A committed hypothesis ships a LENS, not just a focal.  The upstream
# radial-spline models (SFMTOOL_PINHOLE over the perspective chart,
# SFMTOOL_FISHEYE over the equidistant one) add a monotone cubic B-spline
# correction to the base model's radial coordinate before the focal scales it
# to pixels, with dimensionless coefficients on a center-anchored gauge, so
# `f` stays the central scale and the spline can only bend the map away from
# the axis.  TWO KNOTS is the point: this rung is a STARTING distortion
# correction handed to whatever refines the hypothesis later, not a lens fit.
#
# The pattern is the finalization's spline rung miniaturized: a control arm
# with the coefficients frozen at zero, a spline arm co-releasing f with them
# (the gauge means the spline cannot express a central-scale correction, so the
# two must move together), and keep-best against the control on median and
# peripheral reprojection.
#
# A REFUSAL SHIPS THE SPLINE MODEL WITH ZERO COEFFICIENTS.  A zero spline is
# the base map bit for bit -- projection, unprojection and pixel Jacobian alike
# -- so a refused hypothesis's structure is what it always was, while its
# camera record is one a later rung can release coefficients on with no model
# switch.  Falling back to the base MODEL NAME would close that door for
# nothing.
BSPLINE_KNOTS = 2
# Acceptance bars, taken from the finalization rung unchanged: the released arm
# may not be worse than the better of the control and the pre-rung state by
# more than 2% on either the median or the peripheral median, and may not lose
# valid observations.  Ratios rather than absolutes, because the two arms are
# the same observations under two lenses.
BSPLINE_GAIN_BAR = 1.02
# Headroom past the outermost observation for the spline's domain end, so the
# basis covers the field it was fitted on rather than ending inside it.
BSPLINE_DOMAIN_HEADROOM = 1.02
# The outer fraction of the OBSERVED radial distribution that counts as
# PERIPHERAL: the field where a radial correction can do anything, and where a
# lens error shows.  The finalization rung states this as a fixed 60 degrees of
# incidence, which is right for a fisheye and vacuous for a pinhole capture --
# 60 degrees is rho = 1.73, i.e. 1.73 x f in pixels, several times outside the
# frame of every pinhole entry measured here, so the mask selected nothing and
# the peripheral bar silently stopped being a bar at all.  A quantile of the
# field the capture actually imaged is the same statement without the
# assumption: it is the periphery of THIS lens, on either base.
BSPLINE_PERIPHERAL_Q = 0.75


def spline_scores(res, peripheral):
    """``(median residual, peripheral median, finite count)`` of one arm.

    The peripheral mask is fixed at the PRE-RUNG camera so every arm is scored
    on the same observations; a lens that improves the median by dropping the
    field it cannot model would otherwise score as an improvement."""
    fin = np.isfinite(res)
    med = float(np.median(res[fin])) if fin.any() else np.inf
    pf = fin & peripheral
    return med, (float(np.median(res[pf])) if pf.any() else np.nan), int(fin.sum())


def equivalent_focal(B, f_chart, coeffs, d_max, r_obs, f0):
    """The EQUIVALENT base focal of a composite map: the single ``f_eq`` with
    ``r = f_eq * d`` that best fits the released map over the OBSERVED radial
    distribution, least squares, ``f_eq = sum(r * d) / sum(d^2)``.

    This is the quantity hypotheses are compared on once coefficients exist,
    and the one commensurable with the capture-level vote.  The raw chart focal
    is not: under the center-anchored gauge `f` and the coefficients trade
    against each other, so two lenses with the same map can carry different
    `f`, and comparing those numbers compares parameterizations rather than
    optics."""
    cam = B.make_cam_bspline(f_chart, coeffs, d_max)
    dd = r_obs / f0
    th = B._theta_of_d(dd)
    rays = np.stack([np.sin(th), np.zeros_like(th), -np.cos(th)], axis=1)
    px = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    r_map = px[:, 0] - float(_CAM_WH[0]) / 2.0
    ok = np.isfinite(r_map) & (dd > 0)
    if not ok.any():
        return float("nan")
    return float((r_map[ok] * dd[ok]).sum() / (dd[ok] ** 2).sum())


def spline_release(obs_c, obs_i, u, rvec, tvec, pts, f0, n_img, n_cl, live):
    """The hypothesis's lens: a 2-knot spline arm against a spline-free
    control, keep-best.

    On ACCEPTANCE the arm's own poses and structure ship with its lens: they
    were solved under it, and a released map paired with structure triangulated
    under a different one is not a reconstruction.  On REFUSAL nothing is
    adopted and the coefficients are all zero, which is the base map bit for
    bit, so the release is what it would have been without this rung.

    Returns the lens record, or ``None`` where there is no domain to promote
    onto at all."""
    B = bootstrap_module()
    B._CAM_WH = _CAM_WH
    cx, cy = float(_CAM_WH[0]) / 2.0, float(_CAM_WH[1]) / 2.0
    r_obs = np.hypot(u[live][:, 0] - cx, u[live][:, 1] - cy)
    field_r = float(r_obs.max()) if r_obs.size else 0.0
    d_max = BSPLINE_DOMAIN_HEADROOM * field_r / f0
    if not (np.isfinite(d_max) and d_max > 0.0):
        return None
    zero = np.zeros(BSPLINE_KNOTS)
    peripheral = r_obs >= np.quantile(r_obs, BSPLINE_PERIPHERAL_Q)

    def arm(opt_bspline):
        q = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
        return _bundle_adjust(
            B.make_cam_bspline(f0, zero, d_max),
            np.ascontiguousarray(q),
            np.ascontiguousarray(tvec, dtype=np.float64),
            np.ascontiguousarray(pts, dtype=np.float64),
            np.ascontiguousarray(u[live], dtype=np.float64),
            obs_i[live].astype(np.uint32),
            obs_c[live].astype(np.uint32),
            opt_f=True,
            opt_bspline=opt_bspline,
            schedule=[(12.0, 2.0), (4.0, 1.0)],
            max_iters=30,
        )

    # The CONTROL arm's own state, for the evolution corpus alone: the survey
    # scores both arms of the keep-best against ground truth, so the arm that
    # did not ship has to leave the function.  Unset off the corpus, so what a
    # production run carries is unchanged.
    control = {}

    def refused(reason, report, arm_out=None, **scores):
        """A refusal, WITH the arm it refused.

        The verdict is the zero spline the hypothesis ships, but the arm that
        lost is the evidence for the verdict, and a bar nobody can look behind
        is a bar nobody can revise.  So the refused arm travels with the
        refusal and is written beside the release."""
        rec = {
            "coeffs": zero,
            "d_max": d_max,
            "f_chart": f0,
            "f_eq": f0,
            "report": report,
            "accepted": False,
            "refused": {"reason": reason, "report": report, **scores},
        }
        if control:
            rec["control"] = control
        if arm_out is not None:
            rec["refused"].update(
                rvec=Rotation.from_quat(
                    np.asarray(arm_out["quaternions_wxyz"])[:, [1, 2, 3, 0]]
                ).as_rotvec(),
                tvec=np.asarray(arm_out["translations"]),
                pts=np.asarray(arm_out["points"]),
            )
        return rec

    try:
        ctl, out = arm(False), arm(True)
    except Exception as exc:  # noqa: BLE001 — a lens rung never kills the run
        print(f"  spline rung FAILED ({type(exc).__name__}: {exc}); zero spline")
        return refused("failed", "the spline arm raised")
    if evo_on():
        control.update(
            f=float(ctl["focal"]),
            rvec=Rotation.from_quat(
                np.asarray(ctl["quaternions_wxyz"])[:, [1, 2, 3, 0]]
            ).as_rotvec(),
            tvec=np.asarray(ctl["translations"]),
            pts=np.asarray(ctl["points"]),
        )
    coeffs = np.asarray(out["bspline_coefficients"], dtype=np.float64)
    f_chart = float(out["focal"])
    f_eq = equivalent_focal(B, f_chart, coeffs, d_max, r_obs, f0)
    medc, perc, nfinc = spline_scores(
        np.asarray(ctl["residual_norms"], dtype=np.float64), peripheral
    )
    med1, per1, nfin1 = spline_scores(
        np.asarray(out["residual_norms"], dtype=np.float64), peripheral
    )
    report = (
        f"median {medc:.3f} -> {med1:.3f} px, peripheral {perc:.3f} -> "
        f"{per1:.3f} px, valid {nfinc} -> {nfin1}, f {f0:.1f} -> {f_chart:.1f} "
        f"chart / {f_eq:.1f} equivalent"
    )
    scores = {
        "camera": {
            "model": B.spline_model()[0],
            "params": {
                "focal_length": f_chart,
                B.spline_model()[1]: float(d_max),
                "coefficients": [float(c) for c in coeffs],
            },
        },
        "f_chart": f_chart,
        "f_equiv": None if not np.isfinite(f_eq) else f_eq,
        "median_before": medc,
        "median_after": med1,
        "peripheral_before": None if not np.isfinite(perc) else perc,
        "peripheral_after": None if not np.isfinite(per1) else per1,
        "valid_before": nfinc,
        "valid_after": nfin1,
    }
    # The composite map has to stay in the basin the scan committed to; the
    # spline's freedom is radial shape, not a second chance at focal.
    if not (np.isfinite(f_eq) and focal_floor() <= f_eq <= 1.15 * f0):
        print(f"  spline rung REFUSED (composite map outside the basin): {report}")
        return refused("basin", report, out, **scores)
    # Two distinct verdicts, kept apart because they mean different things: a
    # lens that fits no better, and a lens that fits better by pushing
    # observations out of its own domain.
    if nfin1 < nfinc:
        print(f"  spline rung REFUSED (lost valid observations): {report}")
        return refused("domain", report, out, **scores)
    if med1 > BSPLINE_GAIN_BAR * medc or (
        np.isfinite(perc) and np.isfinite(per1) and per1 > BSPLINE_GAIN_BAR * perc
    ):
        print(f"  spline rung REFUSED (no gain): {report}")
        return refused("no_gain", report, out, **scores)
    print(f"  spline rung: {report}")
    print(f"  spline coefficients: {np.array2string(coeffs, precision=6)}")
    return {
        "coeffs": coeffs,
        "d_max": d_max,
        "f_chart": f_chart,
        "f_eq": f_eq,
        "report": report,
        "accepted": True,
        # The corpus's copies of the two arms' scores and of the arm that lost.
        **({"scores": scores, "control": control} if evo_on() else {}),
        "rvec": Rotation.from_quat(
            np.asarray(out["quaternions_wxyz"])[:, [1, 2, 3, 0]]
        ).as_rotvec(),
        "tvec": np.asarray(out["translations"]),
        "pts": np.asarray(out["points"]),
        "res": np.asarray(out["residual_norms"]),
    }


def copy_probe(cand):
    """A private copy of a probe outcome ``(inl, par, rvec, tvec, pts, posed,
    med_inl)``.

    The stages downstream of a probe MUTATE its arrays in place -- the widen
    ladder refills `pts` and the photometric verify un-poses frames -- so a
    memoized probe is stored and returned as copies.  Handing out the stored
    arrays would let the first use rewrite what the second one reads."""
    inl, par, rvec, tvec, pts, posed, med_inl = cand
    return (inl, par, rvec.copy(), tvec.copy(), pts.copy(), posed.copy(), med_inl)


def rung1_n():
    """The rung-1 cluster budget (``SFMTOOL_SEED_RUNG1``), or 0 when the rung is
    not armed.  Unset, every path the rung touches is the one that ran before it
    existed."""
    try:
        return int(os.environ.get("SFMTOOL_SEED_RUNG1", "0") or 0)
    except ValueError:
        return 0


def make_rung(
    handle, m_cl, m_img, radius, n_full, names, dims, min_span, want_warp, rr
):
    """Build the run's `Rung`, staging directory included.

    The staging directory is emptied on creation: a previous run that died
    between writing a release and swapping the product in leaves one behind, and
    it is scratch, never evidence."""
    from datetime import datetime

    n = rung1_n()
    try:
        override = int(os.environ.get("SFMTOOL_SEED_LOCAL_ADMISSION", "0") or 0)
    except ValueError:
        override = 0
    stage = WS / "sfmr" / "candidate_solves.partial"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    return Rung(
        n=n,
        n_local=override if override > 0 else n,
        stamp=os.environ.get("SFMTOOL_ROUND_STAMP")
        or datetime.now().strftime("%Y%m%dT%H%M"),
        stage=stage,
        handle=handle,
        m_cl=m_cl,
        m_img=m_img,
        radius=radius,
        n_full=n_full,
        min_span=min_span,
        names=names,
        dims=dims,
        want_warp=want_warp,
        refine_radius=rr,
    )


def load_clusters():
    """Flat observation arrays of every usable cluster, plus each cluster's
    global admission rank (span-major, cluster-id tiebreak) — the quality
    ordering the per-BA row budget truncates against.

    Everything geometric comes off the stored affines: the last column is the
    member's absolute keypoint position (.matches v4+) and the leading 2x2 its
    absolute affine shape (v5), whose column norms give the member's feature
    extent -- so no `.sift` file is read here at all.  Admission
    (reference/kept members, span filter) runs as the matches-format crate's
    ``select_clusters`` derivation.
    """
    from sfmtool._sfmtool.io import MatchesFile

    override = os.environ.get("SFMTOOL_MATCHES")
    patches = (
        [Path(override)]
        if override
        else sorted(WS.glob("matches/*-clusters-patches.matches"))
    )
    mfile = MatchesFile(patches[0])
    names = list(mfile.image_names)
    dims = [(int(w), int(h)) for w, h in np.asarray(mfile.image_dims)]

    min_span = max(2, int(os.environ.get("SFMTOOL_MIN_SPAN", "2")))

    # Patch-frame passthrough, snapshot-gated: the debug checkpoints are written
    # through the bootstrap's `save_sfmr`, which builds each point's surfel frame
    # from the member's stored 2x2 warp (the affine's leading block).  With
    # snapshots off (the default) the arrays are never built and the loader is
    # byte-identical to before.
    want_warp = bool(os.environ.get("SFMTOOL_SEED_SNAPSHOT_DIR"))

    # Native admission: the matches-format select_clusters derivation is
    # exactly this loader's predicate (drop unrefinable clusters, keep
    # reference/kept members on selected images, span-filter), with
    # cluster and member order preserved.  The selected file keeps at
    # most one reference/kept member per (cluster, image), so each
    # cluster's span IS its member count.
    sel_h = mfile.select_clusters(min_span=min_span)

    # THE COARSE ADMISSION, armed by the rung alone.  On a repeating scene
    # texture -- a tiled floor -- fine features match self-consistently at FALSE
    # lattice offsets: the aliased basin is internally clean and outnumbers the
    # true one, so attempt selection commits to it.  A feature whose support is
    # wider than the tile period cannot alias that way, so restricting stage 1
    # to the coarse clusters buys an unaliased basin before any commitment.
    #
    # The radius is read off the stored affines (no `.sift` read): the leading
    # 2x2 is the member's absolute shape S, whose two column norms average to
    # the detector's affine SCALE.  The patch half-extent in image pixels is
    # `refine_radius x scale` -- the same reading `save_sfmr` sizes a surfel by
    # (see `_cluster_detection_sizes` in the bootstrap).  ANY member over the
    # bar qualifies the cluster: a coarse feature seen coarsely once is a coarse
    # feature, and the cluster's other members are its own matches, not
    # independent evidence.
    #
    # The bar is stated as a POPULATION (keep the N coarsest), not as a
    # threshold.  The scene-scale forms this replaced (a pixel bar and a
    # fraction-of-frame bar) let the kept population span two orders of
    # magnitude across a fleet, which means runs at "one bar" were never one
    # working set; they were the instruments of the sweep that found N, and the
    # sweep is done.
    rung = None
    if rung1_n() > 0:
        aff_s = np.asarray(sel_h.member_affines)
        radius = (
            0.5
            * float(mfile.refine_radius)
            * (
                np.linalg.norm(aff_s[:, :, 0], axis=1)
                + np.linalg.norm(aff_s[:, :, 1], axis=1)
            )
        )
        starts_s = np.asarray(sel_h.cluster_starts, dtype=np.int64)
        m_cl = np.repeat(
            np.arange(len(starts_s) - 1, dtype=np.int64), np.diff(starts_s)
        )
        m_img = np.asarray(sel_h.member_images, dtype=np.int64)
        n_before = len(starts_s) - 1
        # The cluster's own coarseness: its widest member.
        cl_radius = np.zeros(n_before)
        np.maximum.at(cl_radius, m_cl, radius)
        # The rung carries the UNRESTRICTED handle and these radii, because two
        # of its phases read them: the referee reads the full observation arrays
        # (the vote must not shrink with the solve's working set), and a
        # group-local re-admission re-derives the coarse population over its own
        # images out of the whole capture rather than out of stage A's survivors
        # (which could only ever narrow it further).
        rung = make_rung(
            sel_h,
            m_cl,
            m_img,
            radius,
            n_before,
            names,
            dims,
            min_span,
            want_warp,
            mfile.refine_radius,
        )
        rung.n_clusters_total = n_before
        rung.vote_clusters = n_before
        print(
            f"group-local admission armed: top {rung.n_local} per attempt image "
            f"set, drawn from {n_before} clusters"
        )
        if rung.n >= n_before:
            # Nothing to drop: the whole admission IS the top N.  Skip the
            # re-selection entirely so the handle stays the one an unfiltered
            # run would carry, bit for bit.
            rung.n_clusters_kept = n_before
            rung.min_kept_radius_px = float(cl_radius.min())
            print(
                f"coarse admission: top {rung.n} -> kept {n_before}/{n_before} "
                f"(min radius in kept set: {cl_radius.min():.1f} px) "
                f"[no-op: N >= cluster count]"
            )
        else:
            # Radius descending, cluster id ascending among ties: a stable sort
            # of the negated radii is exactly that, so the kept set is a
            # function of the file and N alone.
            keep = np.sort(np.argsort(-cl_radius, kind="stable")[: rung.n])
            rung.vote_obs = (
                m_cl,
                m_img,
                np.ascontiguousarray(aff_s[:, :, 2], dtype=np.float64),
            )
            # Restrict the HANDLE, not the file: `restrict_cluster_ids` names
            # ids of the file it is called on, and the radii above are in
            # `sel_h`'s own dense numbering.  Re-selecting also re-runs the full
            # admission derivation, so the narrowed handle is an ordinary
            # selection that the hypothesis loop and the restriction stage read
            # exactly like the unrestricted one.
            sel_h = sel_h.select_clusters(
                min_span=min_span, restrict_cluster_ids=[int(c) for c in keep]
            )
            n_kept = len(np.asarray(sel_h.cluster_starts)) - 1
            rung.n_clusters_kept = n_kept
            rung.min_kept_radius_px = float(cl_radius[keep].min())
            print(
                f"coarse admission: top {rung.n} -> kept {n_kept}/{n_before} "
                f"(min radius in kept set: {cl_radius[keep].min():.1f} px)"
            )

    # The selection stage 1 solves on: the seed's restriction stage narrows
    # this handle, so every cluster id stage 1 produces is an id in it.  The
    # hypothesis loop derives its complements FROM this handle, and the winning
    # hypothesis's handle is the one the restriction stage sees.
    global _SEL_MATCHES, _LOAD_CTX
    _SEL_MATCHES = sel_h
    _LOAD_CTX = {
        "names": names,
        "dims": dims,
        "min_span": min_span,
        "want_warp": want_warp,
        "refine_radius": mfile.refine_radius,
    }
    return (
        repackage_selection(
            sel_h, names, dims, want_warp=want_warp, refine_radius=mfile.refine_radius
        ),
        rung,
    )


# ── Pairwise focal vote (Bougnoux + rotation self-calibration) ───────────────
#
# Each image pair casts an independent focal vote through whichever estimator
# its geometry supports (native kernel, `sfmtool-core`): parallax-rich pairs
# vote Bougnoux (fundamental-matrix RANSAC on the pair's cluster
# correspondences, principal point at the centre); rotation-dominated pairs
# vote by self-calibration from the conjugate homography K·R·K⁻¹ that
# explains them.  Each family is degenerate on the other's pairs.  The
# families pool into one log-space median unless their medians disagree
# beyond 0.25 log-focal (majority family then stands alone).  The pooled
# median is a consensus that needs NO
# reconstruction — immune to the bas-relief warp that traps every
# structure-based estimate on hard captures (swivel-chair, dino-ledge: votes
# within 5% where the full pipeline sat 17-84% off).  It is coarse (-10..+5%
# observed), so it picks the BASIN: the probe focal and the scan grid centre.
# Structure-based release does the refinement.


def _low_confidence_vote(res):
    """Reasons the pinhole-only vote is low confidence — the escalation
    trigger.  Empty means the vote stands on its own and the camera-model
    columns are never asked for.  See the FISHEYE_THIN_POOL block above for
    where each cut point comes from."""
    w, h = _CAM_WH
    why = []
    if res["focal_px"] is None:
        why.append("no_consensus")
    rot = res["rotation_focal_px"]
    if (
        res["family"] == "Rotation"
        and rot is not None
        and rot <= _ORTHO_GRID_STEP * _ORTHO_GRID_LO * max(w, h)
    ):
        why.append("rotation_railed")
    fd = res["family_disagreement"]
    if fd is not None and fd > _FAMILY_DISAGREEMENT_BAND:
        why.append("family_disagreement")
    if res["n_pool"] <= FISHEYE_THIN_POOL:
        why.append("thin_pool")
    return why


def _escalate_camera_model(obs_c, obs_i, u, res, why):
    """Escalated camera-model arbitration, run only on a low-confidence vote.

    Re-votes with both camera-model columns (pinhole + equidistant fisheye) and
    records a CONFIRMED EquidistantFisheye verdict in ``_VOTE_FISHEYE``.  That
    verdict ROUTES by default: it installs the equidistant camera context, so
    stage 1 and the finalization solve the map the capture actually obeys
    instead of a pinhole approximation of it.  ``SFMTOOL_FISHEYE_SEED=0``
    refuses the routing and leaves the verdict an annotation (see the
    camera-context block above).

    Two things did NOT change, and they are what make routing safe:

      * The CONFIRMATION rule.  A verdict is recorded only when the epipolar
        AND the rotation cell carry model-informative mass.  Run unconditionally
        the bare verdict claims three rectilinear fleet captures, and all three
        win on the epipolar cell alone (see VERDICT CONFIRMATION above).  An
        unconfirmed verdict is annotation-only under every override setting, so
        a wrong verdict still costs at most a wrong label.
      * An equidistant focal is not a pinhole focal.  It parameterizes
        `theta = r/f`, not `r = f*tan(theta)`; the two agree only near the axis
        and diverge without bound at the periphery (kerry: equidistant 138 px
        against a pinhole vote of 144 px).  A routed capture therefore does not
        keep the pinhole vote as a fallback anywhere — the probe, the scan grid,
        the divergence guard and the finalization's arbitration all read the
        verdict's own equidistant focal instead.  Handing the equidistant number
        to a pinhole probe would still be a units error; what changed is that
        the probe is no longer a pinhole one.
    """
    from sfmtool._sfmtool.geometry import focal_vote as _focal_vote

    global _VOTE_FISHEYE
    w, h = _CAM_WH
    two = _focal_vote(
        np.ascontiguousarray(obs_c, dtype=np.uint32),
        np.ascontiguousarray(obs_i, dtype=np.uint32),
        np.ascontiguousarray(u, dtype=np.float64),
        int(w),
        int(h),
        seed=0,
        columns=["pinhole", "equidistant"],
    )
    cols = {c["camera_model"]: c for c in two["columns"]}
    ph, eq = cols.get("Pinhole", {}), cols.get("EquidistantFisheye", {})
    m_p, m_e = int(ph.get("n_informative", 0)), int(eq.get("n_informative", 0))
    m_epi = int(eq.get("n_informative_epipolar", 0))
    m_rot = int(eq.get("n_informative_rotation", 0))
    lo, hi = min(m_p, m_e), max(m_p, m_e)
    margin = (hi / lo) if lo else float("inf") if hi else float("nan")
    f_eq = eq.get("focal_px")
    print(
        f"  camera-model arbitration: verdict {two['camera_model']}, "
        f"certified model-informative mass pinhole {m_p} / equidistant {m_e} "
        f"(margin {margin:.2f}x, equidistant cells epipolar {m_epi} / "
        f"rotation {m_rot}), equidistant focal "
        f"{'none' if f_eq is None else f'{f_eq:.1f}'} px over "
        f"{eq.get('n_pool', 0)} votes at log-focal IQR "
        f"{float(eq.get('pool_spread') or 0.0):.4f}"
    )
    if two["camera_model"] != "EquidistantFisheye":
        return
    if not (m_epi > 0 and m_rot > 0):
        # Verdict NOT corroborated by both cells — the false-positive
        # signature.  Report it and leave the run unannotated.
        print(
            "  camera-model verdict NOT confirmed: the equidistant column's "
            f"evidence is single-cell (epipolar {m_epi}, rotation {m_rot}); a "
            "capture this pipeline genuinely cannot model earns mass in both. "
            "Treating the capture as pinhole."
        )
        return
    _VOTE_FISHEYE = {
        "focal_px": f_eq,
        "margin": margin,
        "mass_pinhole": m_p,
        "mass_equidistant": m_e,
        "mass_epipolar": m_epi,
        "mass_rotation": m_rot,
        # The equidistant column's own log-focal IQR — how tightly the pairwise
        # evidence pins the focal WITHOUT any structure.  The finalization's
        # vote-vs-structure arbitration reads it as the vote's precision band
        # (see `vote_band` in run_pipeline).
        "pool_spread": float(eq.get("pool_spread") or 0.0),
        "trigger": ",".join(why),
    }
    pin_txt = "none" if res["focal_px"] is None else f"{res['focal_px']:.1f} px"
    eq_txt = "none" if f_eq is None else f"{f_eq:.1f} px"
    detected = (
        "FISHEYE DETECTED: the capture arbitrates to an equidistant-fisheye "
        f"model, corroborated by both cells (f_equidistant ~ {eq_txt} vs the "
        f"pinhole vote {pin_txt})."
    )
    # Routing.  The confirmed verdict routes by default; the override can
    # refuse, and a verdict with no equidistant focal has nothing to route to.
    # Either way the run stays pinhole and says so.
    if not FISHEYE_SEED or f_eq is None:
        why_not = (
            "SFMTOOL_FISHEYE_SEED=0"
            if not FISHEYE_SEED
            else "the confirmed verdict carries no equidistant focal"
        )
        print(
            f"{detected}  NOT ROUTED ({why_not}): the seed solve proceeds on "
            "the pinhole vote and will degrade gracefully (center-out), which "
            "is a KNOWN outcome rather than a silent one.  Do not read the "
            "released focal as a lens calibration."
        )
        return
    print(detected)
    set_camera_context("EQUIDISTANT_FISHEYE", f_eq)
    # The parallax diagnostics that steer stage 1's seed-class choice must be
    # read off the WINNING column: `parallax_poverty` is the median
    # rotation/essential consensus ratio, and the pinhole column's version of
    # it is measured through a ray map this capture does not obey.  The
    # equidistant column measured the same quantity correctly.
    global _VOTE_POVERTY, _VOTE_ROT_N
    _VOTE_POVERTY = float(eq.get("parallax_poverty") or 0.0)
    _VOTE_ROT_N = int(eq.get("n_rotation") or 0)
    print(
        "  ROUTED to the fisheye seed: camera context -> EQUIDISTANT_FISHEYE "
        f"at the equidistant focal {f_eq:.1f} px.  Stage 1 now runs "
        "its ray-space seed (Phase 2): ray-space two-view pair init instead "
        "of the affine factorization, ray parallax, a ray-rotation far-field "
        "core, and the focal held FIXED at this value (the equidistant scan "
        "and the free-focal release are Phase 3).  Column diagnostics: "
        f"parallax-poverty {_VOTE_POVERTY:.2f}, rotation votes {_VOTE_ROT_N}."
    )


def focal_vote(obs_c, obs_i, u, n_img):
    """Median focal over image pairs via the native structure-free vote kernel
    (specs/core/geometry/focal-vote.md): Bougnoux votes from parallax-rich pairs,
    rotation-self-calibration votes from rotation-dominated pairs (each
    estimator is degenerate on the other's pairs).  One vote per image pair;
    both families pool into one log-space median, unless their medians disagree
    beyond 0.25 in log-focal and the majority family's median stands alone.
    Returns (focal, n_pooled_votes), or None.

    When that vote comes back LOW CONFIDENCE (``_low_confidence_vote``) it is
    escalated to the two-column camera-model arbitration, which ANNOTATES the
    run and never changes it — see ``_escalate_camera_model``."""
    from sfmtool._sfmtool.geometry import focal_vote as _focal_vote

    w, h = _CAM_WH
    res = _focal_vote(
        np.ascontiguousarray(obs_c, dtype=np.uint32),
        np.ascontiguousarray(obs_i, dtype=np.uint32),
        np.ascontiguousarray(u, dtype=np.float64),
        int(w),
        int(h),
        seed=0,
    )

    global _VOTE_POVERTY, _VOTE_ROT_N, _VOTE_SPREAD
    _VOTE_POVERTY = float(res["parallax_poverty"])
    _VOTE_ROT_N = int(res["n_rotation"])
    _VOTE_SPREAD = float(res["pool_spread"] or 0.0)
    if res["epipolar_focal_px"] is not None and res["rotation_focal_px"] is not None:
        print(
            f"  vote split: epipolar {res['epipolar_focal_px']:.0f} "
            f"({res['n_epipolar']} pair votes), "
            f"rotation {res['rotation_focal_px']:.0f} "
            f"({res['n_rotation']} pair votes), pool {res['n_pool']}, "
            f"parallax-poverty {_VOTE_POVERTY:.2f}"
        )
    print(
        f"  vote pool: {res['n_pool']} votes, log-focal IQR "
        f"{_VOTE_SPREAD:.4f} (the vote's own precision)"
    )
    why = _low_confidence_vote(res)
    if why:
        print(
            f"  low-confidence pinhole vote ({','.join(why)}): escalating to "
            f"the camera-model columns"
        )
        _escalate_camera_model(obs_c, obs_i, u, res, why)
    if res["focal_px"] is None:
        return None
    return float(res["focal_px"]), int(res["n_pool"])


def rotation_core(o_c, o_i, o_u, nw, n_cl, f0):
    """Core hypothesis from the native far-field rotation initializer, or None.

    Thin wrapper over ``sfmtool._sfmtool.geometry.rotation_init``
    (specs/core/geometry/rotation-init.md): far-field conjugate homographies fix a
    rotation skeleton (spanning tree + chordal-mean averaging), the near
    field seeds the baseline and structure, translation grows by
    rotation-locked resection, and the finishing staged BA models the far
    field at infinity (which is what keeps the scale gauge from collapsing
    the core to a panorama).  Adapts the kernel's posed-subset dict to
    grow_to_cap's per-hypothesis candidate tuple.
    """
    from sfmtool._sfmtool.geometry import rotation_init

    w, h = _CAM_WH
    res = rotation_init(
        np.ascontiguousarray(o_c, dtype=np.uint32),
        np.ascontiguousarray(o_i, dtype=np.uint32),
        np.ascontiguousarray(o_u, dtype=np.float64),
        int(w),
        int(h),
        float(f0),
        seed=0,
        min_images=SCAN_CAP,
        max_images=min(nw, SCAN_CAP + 6),
    )
    if res is None:
        return None
    posed_idx = np.asarray(res["image_indexes"], dtype=np.int64)
    if len(posed_idx) < 3:
        return None
    quats = np.asarray(res["quaternions_wxyz"], dtype=np.float64)
    trans = np.asarray(res["translations"], dtype=np.float64)
    frac = np.asarray(res["inlier_fractions"], dtype=np.float64)
    rvec = np.zeros((nw, 3))
    tv_arr = np.zeros((nw, 3))
    rvec[posed_idx] = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_rotvec()
    tv_arr[posed_idx] = trans
    pts = np.asarray(res["points"], dtype=np.float64)
    if len(pts) < n_cl:
        pad = np.full((n_cl - len(pts), 3), np.nan)
        pts = np.vstack([pts, pad])
    pm = np.zeros(nw, bool)
    pm[posed_idx] = True
    med_inl = float(np.median(frac)) if len(frac) else 1.0
    inl = float(np.mean(frac)) if len(frac) else 0.0
    par = core_parallax(rvec, tv_arr, pts, pm, o_c, o_i)
    return inl, par, rvec, tv_arr, pts, pm, med_inl


# ── Seed: covisibility grouping + affine factorization ───────────────────────


def build_covisibility(obs_c, obs_i, n_img, n_cl):
    from sfmtool._sfmtool.matching import ClusterCovisibility

    starts = np.searchsorted(obs_c, np.arange(n_cl + 1)).astype(np.uint32)
    return ClusterCovisibility.from_arrays(starts, obs_i.astype(np.uint32), n_img)


# The CAPTURE-level covisibility graph: built once, from the FULL admission,
# before any hypothesis runs.  Coverage reach asks how much of the capture a
# solve connects to — a capture-level question, exactly like the focal vote the
# loop already holds capture-level — so the arbitration's gates read reach on
# THIS graph for every hypothesis alike, and a complement's smaller admission
# cannot deflate the answer for a solve that genuinely spans the capture (the
# claim removes clusters, the removal costs edges near the >= 8-shared-cluster
# threshold, and two fleet rescues died on reach alone: 70% -> 59% and
# 62% -> 49% against a 60% bar).  A pass's OWN graph stays the exploration's:
# the thinning ladder is choosing among working sets of the admission it is
# actually solving.
_COVIS_CAPTURE = {}


def capture_covisibility(obs_c, obs_i, n_img, n_cl):
    """Resolve the capture-level covisibility graph, building it on first call
    from the full admission's observation arrays."""
    if "v" not in _COVIS_CAPTURE:
        _COVIS_CAPTURE["v"] = build_covisibility(obs_c, obs_i, n_img, n_cl)
        _COVIS_CAPTURE["obs_c"] = obs_c
    return _COVIS_CAPTURE["v"]


def capture_reach(images):
    """Coverage reach of a posed image set on the capture-level graph."""
    return float(_COVIS_CAPTURE["v"].reach(np.ascontiguousarray(images, np.uint32), 8))


def window_spans(obs_c, obs_i, imgs, min_span):
    """Observation selection for clusters seen in >= min_span window images."""
    inw = np.isin(obs_i, imgs)
    cl, il = obs_c[inw], np.searchsorted(imgs, obs_i[inw])
    span = np.zeros(cl.max() + 1 if len(cl) else 1, int)
    for c in np.unique(cl):
        span[c] = len(np.unique(il[cl == c]))
    sel = inw.copy()
    sel[inw] = np.isin(cl, np.nonzero(span >= min_span)[0])
    uniq, c2 = np.unique(obs_c[sel], return_inverse=True)
    return sel, np.searchsorted(imgs, obs_i[sel]), uniq, c2


def factorize_window(obs_c, obs_i, u, imgs, min_span=3):
    """ALS affine factorization + metric upgrade of a candidate window.

    Returns (metric hypotheses as (rot, scale, t_aff), used mask, span-2
    selection for the window mini-BA) or None when too sparse.
    """
    sel, il, uniq, c2 = window_spans(obs_c, obs_i, imgs, min_span)
    if sel.sum() < 30:
        return None
    fac = factorize_affine(
        c2.astype(np.uint32),
        il.astype(np.uint32),
        np.ascontiguousarray(u[sel]),
        len(imgs),
        len(uniq),
    )
    used = np.asarray(fac.used_images)
    t_aff = np.asarray(fac.translations)
    upgraded = fac.metric_upgrade()
    hyps = []
    if upgraded is not None:
        # The factorization's own 3D points ride along with each hypothesis, in
        # that hypothesis' metric frame (M_i·A = s_i·R_i[:2], so Y = A^-1·X) —
        # the debug snapshot needs the structure the factorization produced, not
        # a re-derivation of it.  Off (no snapshots) nothing is converted.
        x_aff = np.asarray(fac.points) if snapshots_on() else None
        for hyp in upgraded:
            rot = np.asarray(hyp.rotations)
            scale = np.asarray(hyp.scales)
            if (scale[used] > 0).all():
                pts_m = None
                if x_aff is not None:
                    pts_m = x_aff @ np.linalg.inv(np.asarray(hyp.gauge)).T
                hyps.append((rot, scale, t_aff, pts_m))
    return hyps, used, window_spans(obs_c, obs_i, imgs, 2)


def perspective_init(rot, scale, t_cam, used, f0):
    """Weak-perspective -> pinhole poses in the canonical camera frame.

    PINHOLE-ONLY, by construction: the `-f0/scale` conversion is the
    weak-perspective depth of a rectilinear projection, and the affine
    factorization it consumes assumes a locally-linear image map.  The fisheye
    branch never calls it — `fisheye_window_seed` replaces the whole
    factorization + metric-upgrade + perspective-init chain with ray-space
    two-view geometry (scripts/notes-fisheye-seed.md, Phase 2)."""
    rot_can = rot * np.array([1.0, -1.0, -1.0])[None, :, None]
    trans = np.zeros((len(rot), 3))
    for i in np.nonzero(used)[0]:
        trans[i] = [t_cam[i, 0] / scale[i], -t_cam[i, 1] / scale[i], -f0 / scale[i]]
    return rot_can, trans


# ── Fisheye seed: ray-space two-view geometry (Phase 2) ──────────────────────
#
# The fisheye branch's replacement for the pinhole seed's affine
# factorization.  Weak-perspective factorization assumes a locally-linear
# projection, which a >180-degree equidistant map is not; the ray-space
# equivalent is the focal-vote column scan's own estimators, run ONCE at the
# verdict's camera instead of scanned over candidate focals
# (specs/core/geometry/focal-vote.md, "Camera-Model Columns"; the kernel is
# sfmtool_core::geometry::relative_pose, exposed as
# `estimate_essential_rays` / `fit_ray_rotation`).
#
# Everything below lives on the sphere.  A ray with theta >= 90 degrees has no
# planar projection at all, and on these captures ~20% of the correspondences
# are past that line, so:
#   * consensus bounds are ANGLES, derived per camera from a pixel tolerance
#     through the map's local dr/dtheta (`fisheye_ray_tol`);
#   * the four-way chirality disambiguation of an essential decomposition
#     counts depth ALONG THE RAY, never `z > 0` — a `z > 0` test discards the
#     entire periphery, which is the same defect Phase 1 fixed in the native
#     bundle adjuster's in-front gate.

# Keypoint localization tolerance the angular bounds derive from (the column
# scan's SCAN_TOL_PX).
FISHEYE_TOL_PX = float(os.environ.get("SFMTOOL_FISHEYE_TOL_PX", "3.0"))
# Correspondences per pair fed to the ray estimators (deterministic stride
# subsample); the column scan caps at the same 600.
FISHEYE_PAIR_CORR = int(os.environ.get("SFMTOOL_FISHEYE_PAIR_CORR", "600"))
# Minimal samples drawn per ray estimate.
FISHEYE_PAIR_SAMPLES = int(os.environ.get("SFMTOOL_FISHEYE_SAMPLES", "256"))
# A pair needs this many shared clusters before it is worth an estimate.
FISHEYE_PAIR_MIN_CORR = 30
# Cheiral support a ray-space pair init must muster to seed a window.
FISHEYE_PAIR_MIN_INL = 20
# Translation-parallax floor (deg) a seed pair should clear.  Below it the pair
# is preferred only when nothing else is available — the 1.0 deg bas-relief
# edge of the affine-band study.
FISHEYE_PAIR_PARALLAX_DEG = 1.0
# Rotation-cell support: a far-field edge needs this many rotation inliers.
FISHEYE_ROT_MIN_INL = 20


def fisheye_stage1():
    """True when stage 1 runs its fisheye-native path.

    The camera context is installed only on a CONFIRMED equidistant verdict
    (which routes by default, unless SFMTOOL_FISHEYE_SEED=0 refuses it), so
    this single test is the whole gate: a capture the
    arbitration did not confirm as fisheye can never reach any code below."""
    return _CAM_CONTEXT["model"] != "SIMPLE_PINHOLE"


def fisheye_ray_tol(f):
    """Angular consensus bound (rad) from a pixel tolerance, through the map.

    A fixed angular threshold does not transfer across lenses and resolutions;
    the focal-vote spec's convention is to carry a pixel tolerance through the
    map's local `dr/dtheta`, which for the equidistant map is the focal itself
    at every radius."""
    return float(FISHEYE_TOL_PX / f)


def _image_cluster_index(obs_c, obs_i, u, n_img):
    """Per-image (cluster ids, uv), cluster-sorted — the pair-correspondence
    index the ray estimators query.

    A dict-per-pair form is O(n_obs) per pair, which the rotation core's
    ~100-pair sweep cannot afford on a 360k-observation capture; this pays
    O(n_obs log n_obs) once."""
    order = np.lexsort((obs_c, obs_i))
    oc, oi, ou = obs_c[order], obs_i[order], u[order]
    starts = np.searchsorted(oi, np.arange(n_img + 1))
    return [
        (oc[starts[i] : starts[i + 1]], ou[starts[i] : starts[i + 1]])
        for i in range(n_img)
    ]


def _pair_from_index(index, a, b):
    """Shared-cluster correspondences of images ``a``/``b`` plus their ids."""
    ca, ua = index[a]
    cb, ub = index[b]
    common, ia, ib = np.intersect1d(ca, cb, return_indices=True)
    return (
        np.ascontiguousarray(ua[ia]),
        np.ascontiguousarray(ub[ib]),
        common,
    )


def _cap_corr(*arrays, cap=None):
    """Deterministic stride subsample of parallel correspondence arrays."""
    cap = FISHEYE_PAIR_CORR if cap is None else cap
    n = len(arrays[0])
    if n <= cap:
        return arrays
    idx = np.linspace(0, n - 1, cap).astype(np.int64)
    return tuple(np.ascontiguousarray(a[idx]) for a in arrays)


def _tri_ray_pair(d1, d2, r_rel, t_rel):
    """Native ray-midpoint triangulation of a pair, in camera 1's frame.

    Camera 1 is the gauge (identity pose), camera 2 is at ``(r_rel, t_rel)``.
    The Phase-1 primitive (`triangulate_batch`) does the work, so this is the
    same midpoint solve the rest of the pipeline uses — no DLT on a `z = 1`
    plane, which a beyond-hemisphere ray has no representation on."""
    from sfmtool._sfmtool.analysis import triangulate_batch

    n = len(d1)
    c2 = -r_rel.T @ t_rel
    dirs = np.empty((2 * n, 3))
    dirs[0::2] = d1
    dirs[1::2] = d2 @ r_rel  # rowwise r_rel.T @ d2: camera-2 rays in frame 1
    ctr = np.zeros((2 * n, 3))
    ctr[1::2] = c2
    off = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    out = triangulate_batch(np.ascontiguousarray(dirs), np.ascontiguousarray(ctr), off)
    return np.asarray(out["points"]), c2


def ray_pair_pose(x1, x2, f, seed=0, min_inliers=FISHEYE_PAIR_MIN_INL):
    """Ray-space two-view init of an image pair, or None.

    Estimates the epipolar matrix on UNIT RAYS (the column scan's estimator at
    a known camera), decomposes it four ways, and picks the hypothesis with the
    most RAY-NATIVE cheiral support: depth along the ray positive in BOTH
    cameras, `x . d1 > 0` and `(R x + t) . d2 > 0`.  The familiar `z > 0` test
    is wrong here by construction — every observation past 90 degrees has
    `z <= 0` while sitting perfectly in front of the lens.

    Returns a dict with the relative pose (world = camera 1), the epipolar
    consensus mask, the cheiral count and the pair's median translation-
    parallax angle in degrees (rotation removed by the decomposition)."""
    from sfmtool._sfmtool.geometry import estimate_essential_rays

    if len(x1) < FISHEYE_PAIR_MIN_CORR:
        return None
    cam = make_cam(f)
    d1 = np.ascontiguousarray(cam.pixel_to_ray_batch(np.ascontiguousarray(x1)))
    d2 = np.ascontiguousarray(cam.pixel_to_ray_batch(np.ascontiguousarray(x2)))
    est = estimate_essential_rays(
        d1,
        d2,
        max_angle_rad=fisheye_ray_tol(f),
        min_inliers=min_inliers,
        samples=FISHEYE_PAIR_SAMPLES,
        seed=seed,
    )
    if est is None:
        return None
    inl = np.asarray(est["inliers"], dtype=bool)
    a1, a2 = np.ascontiguousarray(d1[inl]), np.ascontiguousarray(d2[inl])
    best = None
    for r_rel, t_rel in _decompose_essential(np.asarray(est["e_matrix"])):
        x, c2 = _tri_ray_pair(a1, a2, r_rel, t_rel)
        good = np.isfinite(x).all(axis=1)
        good &= np.einsum("ij,ij->i", x, a1) > 0.0
        good &= np.einsum("ij,ij->i", x @ r_rel.T + t_rel, a2) > 0.0
        ng = int(good.sum())
        if best is None or ng > best[0]:
            best = (ng, r_rel, t_rel, x, good, c2)
    ng, r_rel, t_rel, x, good, c2 = best
    if ng < min_inliers:
        return None
    xa = x[good]
    v1 = xa / (np.linalg.norm(xa, axis=1, keepdims=True) + 1e-12)
    v2 = xa - c2
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    par = float(np.degrees(np.median(np.arccos(np.clip((v1 * v2).sum(1), -1.0, 1.0)))))
    return {
        "rot": r_rel,
        "tvec": t_rel,
        "inliers": inl,
        "n_cheiral": ng,
        "parallax_deg": par,
        "essentialness": float(est["essentialness"]),
    }


def ray_pair_parallax(x1, x2, f, seed=0):
    """Median translation-parallax angle (deg) of a pair, ray-native.

    Rotation is removed by an essential decomposition, so a pure-rotation pair
    scores ~0 however large its raw displacement; the measurement runs through
    the ray map rather than through `K`, `E = K^T F K` and a `z = 1`
    normalization, neither of which means anything past the hemisphere."""
    pose = ray_pair_pose(x1, x2, f, seed=seed)
    if pose is None:
        return np.nan, 0
    return pose["parallax_deg"], int(np.asarray(pose["inliers"]).sum())


def _best_ray_pair(index, pairs, f, seed=0):
    """Best ray-space pair init over ``pairs``, or None.

    Ranking, in order: clears the parallax floor, then cheiral support, then
    parallax.  A pair with real baseline and broad support is what seeds
    structure; a pair below the floor is taken only when nothing else exists
    (the caller's own low-parallax fallbacks then arbitrate)."""
    best = None
    for a, b in pairs:
        x1, x2, _ids = _pair_from_index(index, a, b)
        if len(x1) < FISHEYE_PAIR_MIN_CORR:
            continue
        px1, px2 = _cap_corr(x1, x2)
        pose = ray_pair_pose(px1, px2, f, seed=seed)
        if pose is None:
            continue
        key = (
            pose["parallax_deg"] >= FISHEYE_PAIR_PARALLAX_DEG,
            pose["n_cheiral"],
            pose["parallax_deg"],
        )
        if best is None or key > best[0]:
            best = (key, a, b, pose)
    return best


def _ray_grow_local(lc, li, lu, rvec, tvec, used, ncl, f, gate=0.25):
    """Grow a posed pair across a local image set by ray-P3P + pose refine.

    Both are Phase-1 primitives and already model-correct under the camera
    context (Lambda Twist is bearing-vector native and the refinement's
    residual goes through the map), so the only fisheye-specific thing here is
    that the structure it resects against came from ray-space geometry.
    Mutates and returns ``(rvec, tvec, used, pts)``."""
    cam = make_cam(f)
    rot = Rotation.from_rotvec(rvec).as_matrix()
    pts = triangulate(lc, li, lu, rot, tvec, used, ncl, f)
    blocked = set()
    while True:
        cand = ~used[li] & ~np.isnan(pts[lc, 0])
        if not cand.any():
            break
        cnt = np.bincount(li[cand], minlength=len(used))
        for k in blocked:
            cnt[k] = 0
        j = int(np.argmax(cnt))
        if cnt[j] < 12:
            break
        s = (li == j) & ~np.isnan(pts[lc, 0])
        p3p = p3p_resect(lu[s], pts[lc[s]], f)
        if p3p is None or int(p3p[2].sum()) < 12:
            blocked.add(j)
            continue
        rv0, tv0, mask = p3p
        rv, tv, _ = pose_refine(lu[s][mask], pts[lc[s]][mask], rv0, tv0, f)
        res = reproj_res_one(cam, rv, tv, pts[lc[s]], lu[s])
        if _inlier_fraction(res, 3.0) < gate:
            blocked.add(j)
            continue
        rvec[j], tvec[j], used[j] = rv, tv, True
        rot = Rotation.from_rotvec(rvec).as_matrix()
        pts = triangulate(lc, li, lu, rot, tvec, used, ncl, f)
    return rvec, tvec, used, pts


def fisheye_window_seed(obs_c, obs_i, u, imgs, f):
    """Ray-space two-view seed of one covisibility window, or None.

    The fisheye replacement for ``factorize_window`` -> ``metric_upgrade`` ->
    ``perspective_init``: the window's best covisible pair is initialized from
    ray-space epipolar geometry, the window's clusters are triangulated from
    it, the rest of the window joins by ray-P3P, and a fixed-focal mini-BA
    settles it.  There is no reflection twin to carry — the pinhole path grows
    two hypotheses because the metric upgrade's mirror fits a near-affine
    window equally well, whereas ray cheirality picks one of the four essential
    decompositions outright.

    Returns the candidate tuple ``grow_to_cap`` builds — (window inlier
    fraction, image ids, used mask, cluster ids, rvec, tvec, points) — so
    ``_grow_one`` consumes it unchanged."""
    imgs = np.asarray(sorted(int(x) for x in imgs), dtype=np.int64)
    sel, il, cl_ids, c2 = window_spans(obs_c, obs_i, imgs, 2)
    if sel.sum() < 30:
        return None
    lu = np.ascontiguousarray(u[sel])
    nw, ncl = len(imgs), len(cl_ids)
    index = _image_cluster_index(c2, il, lu, nw)
    best = _best_ray_pair(index, itertools.combinations(range(nw), 2), f)
    if best is None:
        print(f"seed group {[int(k) for k in imgs]}: no ray-space pair init")
        return None
    _key, a, b, pose = best
    print(
        f"seed group {[int(k) for k in imgs]}: ray pair "
        f"({int(imgs[a])}, {int(imgs[b])}) {pose['n_cheiral']} cheiral, "
        f"parallax {pose['parallax_deg']:.2f} deg, essentialness "
        f"{pose['essentialness']:.4f} [{elapsed():.1f}s]"
    )
    rvec = np.zeros((nw, 3))
    tvec = np.zeros((nw, 3))
    used = np.zeros(nw, bool)
    used[a] = used[b] = True
    rvec[b] = Rotation.from_matrix(pose["rot"]).as_rotvec()
    tvec[b] = pose["tvec"]
    rvec, tvec, used, pts = _ray_grow_local(c2, il, lu, rvec, tvec, used, ncl, f)
    ok = ~np.isnan(pts[:, 0])[c2] & used[il]
    if ok.sum() < 30:
        return None
    _, rvw, tvw, p_w, _, inl = bundle_adjust(
        c2[ok],
        il[ok],
        lu[ok],
        rvec,
        tvec,
        pts,
        f,
        nw,
        ncl,
        opt_f=False,
        schedule=((30.0, 3.0), (8.0, 1.5)),
        max_nfev=30,
    )
    return inl, imgs, used, cl_ids, rvw, tvw, p_w


def grow_to_cap_rays(groups, f0, obs_c, obs_i, u, n_img, n_cl, cap, rank):
    """Fisheye analog of ``grow_to_cap``: ray-space window seeds, then the
    shared P3P growth loop.

    No snapshot hook: checkpoint ``00-affine`` records an affine factorization
    the fisheye branch does not perform.  Checkpoints 01-04 are unchanged."""
    cands = []
    for imgs in groups:
        cand = fisheye_window_seed(obs_c, obs_i, u, imgs, f0)
        if cand is not None:
            cands.append(cand)
    cands.sort(key=lambda t: -t[0])
    return [
        _grow_one(c, f0, obs_c, obs_i, u, n_img, n_cl, cap, rank) for c in cands[:2]
    ]


def _pair_counts(obs_c, obs_i, n_img, n_cl):
    """Shared-cluster counts over every image pair (upper triangle)."""
    from scipy import sparse

    m = sparse.csr_matrix(
        (
            np.ones(len(obs_i), np.int32),
            (obs_i.astype(np.int64), obs_c.astype(np.int64)),
        ),
        shape=(n_img, n_cl),
    )
    m.data[:] = 1
    counts = np.asarray((m @ m.T).todense())
    np.fill_diagonal(counts, 0)
    return np.triu(counts)


def ray_rotation_floors(obs_c, obs_i, n_img, n_cl, max_pairs=120):
    """The pair-support floors ``(min shared clusters, min rotation inliers)``
    this ADMISSION can actually meet, never above the full-admission bars.

    `FISHEYE_PAIR_MIN_CORR` and `FISHEYE_ROT_MIN_INL` are stated for a capture's
    whole cluster population.  A coarse admission is a different population: the
    rung's few thousand widest clusters spread over hundreds of frames leave
    single-digit overlap on almost every pair (BadlandPanorama at rung 3000:
    p95 = 10 shared clusters, three pairs of 8638 reach 30), so the fixed bars
    do not thin the pair graph, they empty it.  So the bar is read off the
    admission -- the support the best `max_pairs` pairs actually carry -- and
    floored at four times the two-ray rotation sample, which is the point below
    which a fit stops being a measurement whatever the population."""
    counts = _pair_counts(obs_c, obs_i, n_img, n_cl).ravel()
    k = min(max_pairs, len(counts))
    have = int(np.partition(counts, -k)[-k]) if k else 0
    min_corr = max(8, min(FISHEYE_PAIR_MIN_CORR, have))
    return min_corr, max(6, min(FISHEYE_ROT_MIN_INL, min_corr // 2))


def ray_rotation_edges(
    obs_c,
    obs_i,
    u,
    n_img,
    n_cl,
    f,
    max_pairs=120,
    min_corr=FISHEYE_PAIR_MIN_CORR,
    min_inliers=FISHEYE_ROT_MIN_INL,
):
    """The capture's FAR-FIELD ROTATION EDGES: for each of the best-covisible
    image pairs, the rotation that explains its correspondences as a pure
    turn of unit rays, with the count that supports it.

    Ray space is what makes this camera-model-agnostic: a parallax-free pair
    simply IS a rotation of unit rays under whatever map the context installs,
    so nothing here constructs a conjugate homography and nothing assumes a
    locally-linear image.  The angular consensus bound comes from a pixel
    tolerance through the map's local ``dr/dtheta`` (`fisheye_ray_tol`), which
    is the focal for an equidistant map and a lower bound on it for a pinhole
    one -- conservative in the direction that costs edges rather than invents
    them.

    Returns ``(per-image cluster index, candidate pairs, edges)`` with each
    edge ``(inliers, a, b, R_ab)``, or None when the pair graph is too thin to
    produce any."""
    from sfmtool._sfmtool.geometry import fit_ray_rotation

    counts = _pair_counts(obs_c, obs_i, n_img, n_cl)
    ia, ib = np.nonzero(counts >= min_corr)
    if len(ia) < 2:
        return None
    order = np.argsort(-counts[ia, ib], kind="stable")[:max_pairs]
    pairs = [(int(ia[k]), int(ib[k])) for k in order]
    index = _image_cluster_index(obs_c, obs_i, u, n_img)

    tol = fisheye_ray_tol(f)
    cam = make_cam(f)
    edges = []
    for a, b in pairs:
        x1, x2, _ids = _pair_from_index(index, a, b)
        px1, px2 = _cap_corr(x1, x2)
        r1 = np.ascontiguousarray(cam.pixel_to_ray_batch(px1))
        r2 = np.ascontiguousarray(cam.pixel_to_ray_batch(px2))
        fit = fit_ray_rotation(
            r1,
            r2,
            max_angle_rad=tol,
            min_inliers=min_inliers,
            samples=FISHEYE_PAIR_SAMPLES,
            seed=0,
        )
        if fit is None:
            continue
        inl = np.asarray(fit["inliers"], dtype=bool)
        edges.append((int(inl.sum()), a, b, np.asarray(fit["rotation"])))
    if not edges:
        return None
    return index, pairs, edges


def rotation_spanning_tree(edges, n_img, gauge=None):
    """Absolute rotations from rotation edges: a maximum-consensus spanning
    tree, chained breadth-first from its root.

    ``gauge`` is ``(a, b, R_ab)`` -- an edge forced into the tree before any
    other and used as the root, so a caller that fixes its metric gauge on a
    particular pair gets a skeleton that agrees with it by construction.
    Without one no pair is privileged, because none carries a baseline: the
    forest is then rooted in its LARGEST component, which is the most of the
    capture one common rotation frame can reach.

    Returns ``(rvec (n_img, 3), {image: R})`` over the chained component, or
    None when fewer than three images chain."""
    parent = list(range(n_img))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = {}

    def add_edge(a, b, rot_ab):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[ra] = rb
        tree.setdefault(a, []).append((b, rot_ab))
        tree.setdefault(b, []).append((a, rot_ab.T))

    ranked = sorted(edges, key=lambda e: -e[0])
    if gauge is not None:
        add_edge(gauge[0], gauge[1], gauge[2])
    for _n, a, b, rot_ab in ranked:
        add_edge(a, b, rot_ab)

    if gauge is not None:
        root = gauge[0]
    else:
        members = {}
        for i in sorted(tree):
            members.setdefault(find(i), []).append(i)
        root = max(members.values(), key=lambda ms: (len(ms), -ms[0]))[0]

    rvec = np.zeros((n_img, 3))
    abs_rot = {root: np.eye(3)}
    queue = [root]
    while queue:
        i = queue.pop(0)
        for j, rot_ij in tree.get(i, ()):
            if j in abs_rot:
                continue
            abs_rot[j] = rot_ij @ abs_rot[i]
            queue.append(j)
    if len(abs_rot) < 3:
        return None
    for i, r in abs_rot.items():
        rvec[i] = Rotation.from_matrix(r).as_rotvec()
    return rvec, abs_rot


def rotation_core_rays(obs_c, obs_i, u, n_img, n_cl, f, max_pairs=120):
    """Fisheye rotation core: a far-field skeleton from ray-rotation fits.

    The pinhole ``rotation_core`` reads its skeleton off far-field CONJUGATE
    HOMOGRAPHIES, which a fisheye map induces at no focal — a rotating fisheye
    camera produces no pixel homography.  The equidistant analog is the focal
    vote's rotation cell: under the right ray map a parallax-free pair simply
    IS a pure rotation of unit rays, needing no conjugacy construction and
    valid over the whole sphere (theta >= 90 degrees rays participate like any
    others).  So:

      * fit a ray rotation on every candidate covisible pair; the pairs a
        rotation explains are the far-field edges;
      * chain a maximum-consensus spanning tree into absolute rotations;
      * fix the gauge with the best ray-space epipolar pair (that edge is
        forced into the tree, so the skeleton and the baseline agree by
        construction), triangulate, and grow the remaining skeleton images by
        ROTATION-LOCKED resection — the native `resect_translation`, whose
        ray-space rows are camera-model-agnostic.

    Returns ``rotation_core``'s tuple (inlier fraction, parallax, rvec, tvec,
    points, posed mask, median inlier fraction), or None."""
    from sfmtool._sfmtool.geometry import resect_translation

    found = ray_rotation_edges(obs_c, obs_i, u, n_img, n_cl, f, max_pairs)
    if found is None:
        return None
    index, pairs, edges = found
    seed_pair = _best_ray_pair(index, pairs, f)
    if seed_pair is None:
        return None
    _key, sa, sb, pose = seed_pair
    chained = rotation_spanning_tree(edges, n_img, gauge=(sa, sb, pose["rot"]))
    if chained is None:
        return None
    rvec, abs_rot = chained
    tvec = np.zeros((n_img, 3))
    posed = np.zeros(n_img, bool)
    posed[sa] = posed[sb] = True
    tvec[sb] = pose["tvec"]

    rot = Rotation.from_rotvec(rvec).as_matrix()
    pts = triangulate(obs_c, obs_i, u, rot, tvec, posed, n_cl, f)
    # Rotation-locked translation growth over the skeleton, most-supported
    # first; an image whose locked resection fails simply stays unposed and
    # the caller's P3P growth may still reach it.
    accepted = []
    for _round in range(len(abs_rot)):
        skeleton = np.ones(n_img, bool)
        for i in abs_rot:
            skeleton[i] = False
        cand = ~posed[obs_i] & ~np.isnan(pts[obs_c, 0])
        cnt = np.bincount(obs_i[cand], minlength=n_img)
        cnt[skeleton] = 0
        j = int(np.argmax(cnt))
        if cnt[j] < 12:
            break
        s = (obs_i == j) & ~np.isnan(pts[obs_c, 0])
        q = Rotation.from_rotvec(rvec[j]).as_quat()[[3, 0, 1, 2]]
        out = resect_translation(
            make_cam(f),
            [float(x) for x in q],
            np.ascontiguousarray(pts[obs_c[s]], dtype=np.float64),
            np.ascontiguousarray(u[s], dtype=np.float64),
            RESECT_MAX_PX,
            10,
        )
        if out is None:
            del abs_rot[j]
            continue
        if RESECT_TRACE:
            rn = np.asarray(out["residual_norms"], dtype=np.float64)
            keep_r = np.asarray(out["inliers"], dtype=bool)
            fin = rn[np.isfinite(rn) & (rn < 1e5)]
            print(
                f"    [resect img {j}: {int(keep_r.sum())}/{len(keep_r)} kept at "
                f"{RESECT_MAX_PX:g} px; residual p50 "
                f"{np.median(fin) if len(fin) else float('nan'):.2f} p90 "
                f"{np.percentile(fin, 90) if len(fin) else float('nan'):.2f} "
                f"max {fin.max() if len(fin) else float('nan'):.2f}; "
                f"{int((~np.isfinite(rn) | (rn >= 1e5)).sum())} out of domain]"
            )
        tvec[j] = np.asarray(out["translation"])
        posed[j] = True
        res = reproj_res_one(make_cam(f), rvec[j], tvec[j], pts[obs_c[s]], u[s])
        accepted.append(_inlier_fraction(res, 3.0))
        rot = Rotation.from_rotvec(rvec).as_matrix()
        pts = triangulate(obs_c, obs_i, u, rot, tvec, posed, n_cl, f)
    if int(posed.sum()) < 3:
        return None

    live = ~np.isnan(pts[obs_c, 0]) & posed[obs_i]
    if live.sum() < 30:
        return None
    _, rvec, tvec, pts, res, inl = bundle_adjust(
        obs_c[live],
        obs_i[live],
        u[live],
        rvec,
        tvec,
        pts,
        f,
        n_img,
        n_cl,
        opt_f=False,
        schedule=((30.0, 3.0), (8.0, 1.5)),
        max_nfev=30,
    )
    par = core_parallax(rvec, tvec, pts, posed, obs_c, obs_i)
    med_inl = float(np.median(accepted)) if accepted else 1.0
    return inl, par, rvec, tvec, pts, posed, med_inl


# ── Geometry kernels ─────────────────────────────────────────────────────────


def triangulate(obs_c, obs_i, u, rot, trans, used, n_cl, f, cam=None):
    """Ray-midpoint triangulation from the posed images (< 2 views: NaN).

    ``cam`` overrides the context camera at ``f``, for geometry solved under a
    lens the context does not carry: a structure triangulated through a
    different map than the one that placed the poses is not the same structure,
    and the writer that reprojects it will cull most of it."""
    from sfmtool._sfmtool.analysis import triangulate_batch

    pts = np.full((n_cl, 3), np.nan)
    sel = used[obs_i]
    if not sel.any():
        return pts
    oc, oi, uv = obs_c[sel], obs_i[sel], u[sel]
    d_loc = (make_cam(f) if cam is None else cam).pixel_to_ray_batch(
        np.ascontiguousarray(uv)
    )
    dirs = np.einsum("nji,nj->ni", rot[oi], d_loc)
    centers = -np.einsum("nji,nj->ni", rot[oi], trans[oi])
    uniq, counts = np.unique(oc, return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    result = triangulate_batch(
        np.ascontiguousarray(dirs), np.ascontiguousarray(centers), offsets
    )
    good = counts >= 2
    pts[uniq[good]] = np.asarray(result["points"])[good]
    return pts


# ── Seed snapshots (debug) ───────────────────────────────────────────────────
#
# SFMTOOL_SEED_SNAPSHOT_DIR names a directory; when it is set the seed stage
# writes a .sfmr at each pipeline checkpoint so every intermediate state can be
# opened in the SfM Explorer.  Unset (the default) every hook below returns
# immediately and the run is byte-identical.  Stage 1 owns checkpoints 0-4:
#
#   00-affine-<pass>-<n>   the raw affine factorization (far-pinhole embedding)
#   01-probe-<pass>-<a>    the perspective-initialized probe core at f_probe
#   02-widen-<pass>-<a>    after the widen ladder
#   03-verify-<pass>-<a>   after the photometric verify un-posing
#   04-release-<pass>      the pass's released estimate
#
# <pass> is the run_pipeline pass tag — always "nosel", kept as a literal so
# existing snapshot names do not move.
# <a> counts probe attempts within the pass and <n> the factorization
# hypotheses (two seed groups x two reflections per chunk).  Checkpoints 5-7
# (dense / embed / culled) are written by exp_pinhole_bootstrap's finalization.
_SNAP = {"tag": "nosel", "n_affine": 0, "n_attempt": 0}


def snapshots_on():
    return bool(os.environ.get("SFMTOOL_SEED_SNAPSHOT_DIR"))


def _snap_full_frame(data, keep, rvec, tvec, posed, f):
    """Working-set per-image arrays -> the full ``data`` image frame.

    ``attempt``'s working set renumbers IMAGE indexes whenever the covisibility
    thinning is active (``keep`` maps working index -> data image index; it is
    ``arange`` at the unthinned level).  Cluster ids and therefore the ``pts``
    array are never renumbered, so only the per-image arrays move frames.
    """
    n_all = data["n_img"]
    idx = np.asarray(keep, dtype=np.int64)
    rv = np.zeros((n_all, 3))
    tv = np.tile([0.0, 0.0, -f], (n_all, 1))
    pd = np.zeros(n_all, bool)
    rv[idx] = rvec
    tv[idx] = tvec
    pd[idx] = posed
    return rv, tv, pd


def stage1_census(tag, data, keep, posed):
    """Per-image cluster-observation census at a stage-1 checkpoint.

    The finalization's per-image observation waterfall (`SFMTOOL_STAGE_DUMPS`
    in ``exp_pinhole_bootstrap``) starts at the embed; this is its CEILING —
    what the matcher's clusters offer each image before any photometric pass
    has had a say.  Two counts per image: every cluster observation it carries,
    and the subset on clusters at least two POSED frames see (the ones dense
    triangulation can place, i.e. what actually reaches the embed).  No-op
    unless the stage dumps are on."""
    if not os.environ.get("SFMTOOL_STAGE_DUMPS"):
        return
    try:
        idx = np.asarray(keep, dtype=np.int64)
        pd = np.zeros(data["n_img"], bool)
        pd[idx] = np.asarray(posed, bool)
        oc, oi = data["obs_c"], data["obs_i"]
        n_img, n_cl = data["n_img"], data["n_cl"]
        total = np.bincount(oi, minlength=n_img)
        sel = pd[oi]
        n_posed_views = np.bincount(oc[sel], minlength=n_cl)
        usable = np.bincount(oi[sel & (n_posed_views[oc] >= 2)], minlength=n_img)
        print(f"  stage1 {tag}: posed {pd.tolist()}")
        print(f"  stage1 {tag}: cluster obs per image {total.tolist()}")
        print(f"  stage1 {tag}: triangulable obs per image {usable.tolist()}")
    except Exception as exc:
        print(f"  [stage1-census {tag} FAILED: {type(exc).__name__}: {exc}]")


def seed_snap(tag, data, keep, f, rvec, tvec, pts, posed, extra=None):
    """One stage-1 checkpoint (no-op unless snapshots are enabled)."""
    stage1_census(tag, data, keep, posed)
    if not snapshots_on():
        return
    B = bootstrap_module()
    rv, tv, pd = _snap_full_frame(data, keep, rvec, tvec, posed, f)
    B.seed_snapshot(tag, data, f, rv, tv, pts, pd, extra_tool_options=extra)


# ── The evolution corpus (SFMTOOL_SEED_EVO_DUMP) ────────────────────────────
#
# `SFMTOOL_SEED_EVO_DUMP=<dir>` records the rung's WHOLE EXPLORATION, not just
# the set it ships: every candidate the ladder produced, at every stage of its
# evolution that has poses and points, plus one `evolution.json` holding every
# cheap evidence channel the run already computed about each of them.
#
# The corpus exists to be SCORED against ground truth, and a corpus of
# survivors alone cannot be: a criterion that separates good candidates from
# bad ones needs the bad ones.  So the drops are in it too — probes the
# measurability gate refused, near-static seeds skipped for a better window,
# widened outcomes a later level displaced, finalists that collapsed as
# duplicates, candidates past the budget — each carrying the reason it never
# reached the product.  Nothing here decides anything; every field is a
# candidate criterion for a survey that has not been run yet, which is why the
# rule is to record more rather than less.
#
# Stage tags are open-ended strings: later rungs will add stages, and a reader
# keyed on a closed set would have to change with each one.  A candidate's
# serial is allocated when its PROBE lands and never reused, so the same number
# names it whether it went on to commit or was dropped two lines later.
#
# The dump is instrumentation: it never steers.  It requires the rung (there is
# no candidate set off it), it does not touch `snapshots_on()` (which disables
# the memos and turns on the warp passthrough), and with the variable unset
# every hook returns before it reads anything.
_EVO = {"resolved": False, "dir": None, "serial": 0, "cands": {}, "order": []}


def evo_dir():
    """The evolution-dump directory, or None when the corpus is not armed."""
    if not _EVO["resolved"]:
        d = os.environ.get("SFMTOOL_SEED_EVO_DUMP")
        out = Path(d) if (d and rung1_n()) else None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        _EVO["dir"] = out
        _EVO["resolved"] = True
    return _EVO["dir"]


def evo_on():
    return evo_dir() is not None


def _jsonable(v):
    """Numpy scalars, arrays and masks as plain JSON values."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, float):
        return None if not np.isfinite(v) else v
    if isinstance(v, Path):
        return str(v)
    return v


def evo_candidate(kind, **channels):
    """Open a record for one candidate and return its serial (None when off).

    ``kind`` says how the candidate was seeded — a covisibility group's probe,
    the rotation core, or a rotation-only layer — so the survey can break the
    corpus out by seeding pathway."""
    if not evo_on():
        return None
    s = _EVO["serial"]
    _EVO["serial"] += 1
    _EVO["cands"][s] = {
        "serial": s,
        "kind": kind,
        "dropped_reason": None,
        "hypothesis_index": None,
        "stages": {},
        **_jsonable(channels),
    }
    _EVO["order"].append(s)
    return s


def evo_note(serial, **channels):
    """Add evidence channels to a candidate's record."""
    if serial is None or not evo_on():
        return
    _EVO["cands"][serial].update(_jsonable(channels))


def evo_reason(serial, reason):
    """Record why a candidate did not reach the product."""
    if serial is None or not evo_on():
        return
    _EVO["cands"][serial]["dropped_reason"] = reason


def evo_clear(serial):
    """Withdraw a drop reason: the candidate was returned after all."""
    evo_reason(serial, None)


def evo_link(serial, idx):
    """Bind a candidate to the hypothesis index it committed as."""
    if serial is None or not evo_on():
        return
    _EVO["cands"][serial]["hypothesis_index"] = int(idx)


def stage_metrics(obs_c, obs_i, u, rvec, tvec, pts, posed, f, cam=None):
    """One stage's reprojection census over the observations it holds.

    Measured on the working set's own arrays rather than on the written
    artifact: the numbers are the same and this costs one projection pass.
    Both inlier bars are reported (2 px is what the pipeline scores on, 4 px
    what its trims use) alongside the residual quantiles, because which of them
    predicts quality is exactly what the survey has to find out."""
    cam = make_cam(float(f)) if cam is None else cam
    posed = np.asarray(posed, bool)
    finite = np.isfinite(pts[:, 0])
    live = posed[obs_i] & finite[obs_c]
    seen = np.bincount(obs_c[posed[obs_i]], minlength=len(pts))
    out = {
        "n_posed": int(posed.sum()),
        "n_points": int(finite.sum()),
        "n_points_multiview": int((finite & (seen >= 2)).sum()),
        "n_obs": int(live.sum()),
        "median_px": None,
        "mean_px": None,
        "p90_px": None,
        "inlier_2px": None,
        "inlier_4px": None,
    }
    if not live.any():
        return out
    rows = np.nonzero(live)[0]
    xc = (
        Rotation.from_rotvec(rvec[obs_i[rows]]).apply(pts[obs_c[rows]])
        + tvec[obs_i[rows]]
    )
    proj = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)))
    res = np.linalg.norm(proj - u[rows], axis=1)
    fin = np.isfinite(res)
    if not fin.any():
        return out
    r = res[fin]
    out.update(
        median_px=float(np.median(r)),
        mean_px=float(np.mean(r)),
        p90_px=float(np.percentile(r, 90)),
        inlier_2px=float((r < 2.0).mean()),
        inlier_4px=float((r < 4.0).mean()),
    )
    return out


def evo_lens_cam(lens):
    """The camera one stage's geometry was solved under, when it carries a
    spline lens; None means the base map, which is what a zero spline is."""
    if lens is None:
        return None
    return bootstrap_module().make_cam_bspline(
        float(lens["f_chart"]),
        np.asarray(lens["coeffs"], dtype=np.float64),
        float(lens["d_max"]),
    )


def evo_stage(
    serial, stage, data, keep, f, rvec, tvec, pts, posed, wk=None, lens=None, **channels
):
    """Write one candidate's stage: the release-grade ``.sfmr`` and its row.

    ``wk`` is the working set the stage's arrays are stated in, for the
    reprojection census; ``lens`` is the spline the geometry was solved under,
    installed for the write exactly as a release installs it."""
    if serial is None or not evo_on():
        return None
    row = dict(_jsonable(channels))
    row["f"] = float(f)
    try:
        if wk is not None:
            row.update(
                _jsonable(
                    stage_metrics(
                        wk[0],
                        wk[1],
                        wk[2],
                        rvec,
                        tvec,
                        pts,
                        posed,
                        f,
                        cam=evo_lens_cam(lens),
                    )
                )
            )
    except Exception as exc:  # noqa: BLE001 -- instrumentation never kills the run
        row["metrics_error"] = f"{type(exc).__name__}: {exc}"
    B = bootstrap_module()
    prior = B.camera_context()
    tag = f"cand{serial:03d}-{stage}"
    try:
        if lens is not None:
            B.set_camera_context(
                B.spline_model()[0],
                float(lens["f_chart"]),
                bspline=np.asarray(lens["coeffs"], dtype=np.float64),
                theta_max=float(lens["d_max"]),
            )
        rv, tv, pd = _snap_full_frame(data, keep, rvec, tvec, posed, f)
        written = B.seed_snapshot(
            tag,
            data,
            f,
            rv,
            tv,
            pts,
            pd,
            extra_tool_options={"evo_candidate": str(serial), "evo_stage": stage},
            path=evo_dir() / f"{tag}.sfmr",
            release_grade=True,
        )
        row["file"] = None if written is None else Path(written).name
    except Exception as exc:  # noqa: BLE001 -- instrumentation never kills the run
        row["file"] = None
        row["write_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [evo-dump {tag} FAILED: {type(exc).__name__}: {exc}]")
    finally:
        B.set_camera_context(
            prior["model"],
            prior["focal"],
            bspline=prior["bspline"],
            theta_max=prior["theta_max"],
        )
    _EVO["cands"][serial]["stages"][stage] = row
    return row


def evo_copy_stage(serial, stage, src, **channels):
    """Record a stage whose artifact the run already wrote (the rotation-only
    layers): copy it under the candidate's name rather than re-deriving it."""
    if serial is None or not evo_on():
        return None
    row = dict(_jsonable(channels))
    row["file"] = None
    try:
        if src is not None and Path(src).is_file():
            dst = evo_dir() / f"cand{serial:03d}-{stage}.sfmr"
            shutil.copy2(src, dst)
            row["file"] = dst.name
    except Exception as exc:  # noqa: BLE001 -- instrumentation never kills the run
        row["copy_error"] = f"{type(exc).__name__}: {exc}"
    _EVO["cands"][serial]["stages"][stage] = row
    return row


def evo_write(**capture):
    """Write ``evolution.json``: every candidate, every stage, every channel."""
    if not evo_on():
        return None
    doc = {
        "workspace": str(WS),
        "candidates": [_EVO["cands"][s] for s in _EVO["order"]],
        **_jsonable(capture),
    }
    out = evo_dir() / "evolution.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    n_files = sum(
        1 for c in _EVO["cands"].values() for r in c["stages"].values() if r.get("file")
    )
    print(
        f"\nevolution corpus: {len(_EVO['cands'])} candidates, {n_files} stage "
        f"artifacts -> {out.parent} [{elapsed():.1f}s]"
    )
    return out


def _affine_depth_span(rot, used, pts_metric):
    """Depth span of the factorized points, in factorization world units.

    ``pts_metric`` is the ALS structure in the hypothesis' metric frame
    (Y = A^-1·X), so the span is the widest per-view depth (R_i[2]·Y) extent
    over the points the factorization itself produced — including, on a window
    whose depth direction is unobservable, the enormous relief the solve put
    there (which is exactly what the snapshot is meant to show).  Measured
    1st..99th percentile so a few stray points cannot inflate the focal.
    """
    if pts_metric is None:
        return np.nan
    y = np.asarray(pts_metric, dtype=np.float64)
    y = y[np.isfinite(y).all(axis=1)]
    if len(y) < 8:
        return np.nan
    z = np.asarray(rot, dtype=np.float64)[np.asarray(used, dtype=bool)][:, 2, :] @ y.T
    return float(np.max(np.percentile(z, 99, axis=1) - np.percentile(z, 1, axis=1)))


def snapshot_affine(
    data, keep, u, n_cl, imgs, used, rot, scale, t_aff, pts_m, span2, hyp
):
    """Checkpoint 0: the RAW affine factorization, as a far-pinhole recon.

    Weak perspective IS a pinhole placed far away: with the view's factorization
    scale s_i (pixels per world unit) and a SHARED synthetic focal f_synth, view
    i stands at distance d_i = f_synth / s_i — exactly the conversion
    ``perspective_init`` performs.  f_synth is picked so every view stands at
    100x the factorized structure's depth span (foreshortening error < 1%, i.e.
    the affine limit the factorization assumes):

        f_synth = 100 * (depth span) * max_i(s_i)   =>   d_i >= 100 * span.

    The saved file declares what it is (``synthetic_weak_perspective``) and
    carries the true affine numbers (the per-view scales) in its tool_options.
    """
    if not snapshots_on():
        return
    try:
        sel, il, cl_ids, c2 = span2
        used = np.asarray(used, dtype=bool)
        uv = u[sel]
        if int(used.sum()) < 2:
            return
        span = _affine_depth_span(rot, used, pts_m)
        s_max = float(np.max(np.asarray(scale)[used]))
        if not (np.isfinite(span) and span > 0.0 and np.isfinite(s_max) and s_max > 0):
            print(
                f"  [seed-snapshot 00-affine: degenerate factorization "
                f"(depth span {span}); skipped]"
            )
            return
        f_synth = 100.0 * span * s_max
        rot_can, trans = perspective_init(rot, scale, t_aff, used, f_synth)
        pts_w = triangulate(c2, il, uv, rot_can, trans, used, len(cl_ids), f_synth)
        pts = np.full((n_cl, 3), np.nan)
        pts[cl_ids] = pts_w
        idx_full = np.asarray(keep, dtype=np.int64)[np.asarray(imgs, dtype=np.int64)]
        n_all = data["n_img"]
        rv = np.zeros((n_all, 3))
        tv = np.tile([0.0, 0.0, -f_synth], (n_all, 1))
        pd = np.zeros(n_all, bool)
        rv[idx_full] = Rotation.from_matrix(rot_can).as_rotvec()
        tv[idx_full] = trans
        pd[idx_full] = used

        B = bootstrap_module()
        n = _SNAP["n_affine"]
        _SNAP["n_affine"] += 1
        B.seed_snapshot(
            f"00-affine-{_SNAP['tag']}-{n:02d}",
            data,
            f_synth,
            rv,
            tv,
            pts,
            pd,
            extra_tool_options={
                "synthetic_weak_perspective": "true",
                "factorization_scales": ",".join(
                    f"{s:.6g}" for s in np.asarray(scale)[used]
                ),
                "synthetic_focal_px": f"{f_synth:.6g}",
                "affine_depth_span": f"{span:.6g}",
                "reflection_hypothesis": str(int(hyp)),
                "seed_window": ",".join(str(int(j)) for j in idx_full),
            },
        )
    except Exception as exc:
        print(f"  [seed-snapshot 00-affine FAILED: {type(exc).__name__}: {exc}]")


# ── Two-view decomposition and pose primitives ───────────────────────────────


def _decompose_essential(e):
    """The four (R, t) candidates of an essential matrix (Hartley-Zisserman)."""
    u_svd, _s, vt = np.linalg.svd(e)
    if np.linalg.det(u_svd) < 0:
        u_svd = -u_svd
    if np.linalg.det(vt) < 0:
        vt = -vt
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r1, r2, t = u_svd @ w @ vt, u_svd @ w.T @ vt, u_svd[:, 2]
    return [(r1, t), (r1, -t), (r2, t), (r2, -t)]


def p3p_resect(uv, x_pts, f0):
    """RANSAC P3P absolute pose; returns (rvec, tvec, inlier mask) or None."""
    ans = estimate_absolute_pose(
        np.ascontiguousarray(uv),
        np.ascontiguousarray(x_pts),
        camera=make_cam(f0),
        max_error_px=4.0,
        seed=0,
    )
    if ans is None:
        return None
    q = np.asarray(ans["quaternion_wxyz"])
    rv = Rotation.from_quat(q[[1, 2, 3, 0]]).as_rotvec()
    return rv, np.asarray(ans["translation"]), np.asarray(ans["inliers"], dtype=bool)


def pose_refine(uv, x_pts, rv0, tv0, f):
    """Trimmed pose-only refinement (native)."""
    q0 = Rotation.from_rotvec(rv0).as_quat()[[3, 0, 1, 2]]
    out = _refine_absolute_pose(
        make_cam(f),
        np.ascontiguousarray(uv, dtype=np.float64),
        np.ascontiguousarray(x_pts, dtype=np.float64),
        q0,
        np.ascontiguousarray(tv0, dtype=np.float64),
        5,
        0.6,
        3.0,
    )
    q = np.asarray(out["quaternion_wxyz"])
    rv = Rotation.from_quat(q[[1, 2, 3, 0]]).as_rotvec()
    return rv, np.asarray(out["translation"]), float(out["inlier_fraction"])


def reproj_res_one(cam, rvec_i, tvec_i, x_pts, uv):
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
        1e6,
    )


# ── Bundle adjustment ────────────────────────────────────────────────────────

# Free-focal BA admits both of this script's camera models: SIMPLE_PINHOLE and
# EQUIDISTANT_FISHEYE.  Both are single-focal and distortion-free, so the
# kernel's analytic focal column `d(u, v)/df = (u - cx)/f` is EXACT for each
# (specs/core/geometry/bundle-adjustment.md) — the Phase-3b kernel widening removed the
# fixed-focal clamp this wrapper used to apply under a fisheye context.


def bundle_adjust(
    obs_c,
    obs_i,
    u,
    rvec,
    tvec,
    pts,
    f0,
    n_img,
    n_cl,
    opt_f,
    schedule=((50.0, 5.0), (12.0, 2.0), (4.0, 1.0)),
    max_nfev=60,
):
    """Staged robust BA via the native kernel: trim, solve, re-triangulate
    between rounds (specs/core/geometry/bundle-adjustment.md — same semantics as the
    scipy original it replaced, including the < 12-survivors degenerate exit
    with all-inf residuals)."""
    q = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
    out = _bundle_adjust(
        make_cam(f0),
        np.ascontiguousarray(q),
        np.ascontiguousarray(tvec, dtype=np.float64),
        np.ascontiguousarray(pts, dtype=np.float64),
        np.ascontiguousarray(u, dtype=np.float64),
        obs_i.astype(np.uint32),
        obs_c.astype(np.uint32),
        opt_f=opt_f,
        schedule=[(float(t), float(s)) for t, s in schedule],
        max_iters=max_nfev,
        min_track=2,
        min_obs=12,
    )
    rv = Rotation.from_quat(
        np.asarray(out["quaternions_wxyz"])[:, [1, 2, 3, 0]]
    ).as_rotvec()
    res = np.asarray(out["residual_norms"])
    if os.environ.get("SFMTOOL_BA_DEBUG", "0") == "1":
        print(
            f"    [BA opt_f={int(opt_f)} n_obs={len(obs_c)} "
            f"f {f0:.2f} -> {out['focal']:.2f}, "
            f"inl<2 {(res < 2.0).mean():.3f}]"
        )
    return (
        float(out["focal"]),
        rv,
        np.asarray(out["translations"]),
        np.asarray(out["points"]),
        res,
        float((res < 2.0).mean()),
    )


# ── Growth (scan-sized) ──────────────────────────────────────────────────────

_RANK_O = None  # per-observation admission rank (set in main)


def budget_mask(posed, obs_i, rank):
    """Per-observation BA row mask: the MAX_CL cluster budget applied to the
    clusters VISIBLE in this window, ordered by global admission rank.

    Visible = at least one observation in a currently-posed image, so the mask
    tracks the posed set and must be recomputed wherever that set changes
    between adjustments.  Ranks are a unique permutation over clusters, so
    truncation is a threshold on the MAX_CL-th smallest visible rank — a pure
    function of (posed set, ranks), with no RNG and no dict-order dependence.
    Below the budget the whole visible window is in."""
    vis = posed[obs_i]
    if not vis.any():
        return vis
    vr = np.unique(rank[vis])
    if len(vr) <= MAX_CL:
        return vis
    return vis & (rank <= vr[MAX_CL - 1])


def ba_rows(live, obs_i):
    """Per-image cap on BA rows: keep each image's best OBS_PER_IMG
    observations by admission rank, so BA cost stays flat in image count."""
    idx = np.nonzero(live)[0]
    keep = live.copy()
    for i in np.unique(obs_i[idx]):
        rows = idx[obs_i[idx] == i]
        if len(rows) > OBS_PER_IMG:
            keep[rows[np.argsort(_RANK_O[rows], kind="stable")[OBS_PER_IMG:]]] = False
    return keep


def grow_to_cap(seed, f0, obs_c, obs_i, u, n_img, n_cl, cap, rank, snap=None):
    """Seed perspective solves and P3P-grow each to ``cap`` images.

    Both reflection hypotheses of the metric upgrade fit a near-affine seed
    window almost equally well, so the top two seed candidates (by the seed
    mini-BA inlier fraction) BOTH grow — the mirror solution falls behind
    once wider-baseline views join.  Yields the grown states for the caller
    to rank.

    Minimal next-best-view loop: no force-accept or retry machinery — an
    image that fails resection or the inlier gate is blocked for good (the
    scan ranks candidates; it does not need completion).
    """
    cands = []
    for imgs, wd in seed:
        if wd is None:
            continue
        hyps, used, (sel, il, cl_ids, c2) = wd
        for hyp, (rot0, scale, t_aff, pts_m) in enumerate(hyps):
            # Checkpoint 0: the factorization itself, before any perspective
            # init at the probe focal (``snap`` is the caller's index-frame-aware
            # hook; None outside a snapshot run).
            if snap is not None:
                snap(imgs, used, rot0, scale, t_aff, pts_m, (sel, il, cl_ids, c2), hyp)
            rot_can, trans0 = perspective_init(rot0, scale, t_aff, used, f0)
            pts_w = triangulate(c2, il, u[sel], rot_can, trans0, used, len(cl_ids), f0)
            ok = ~np.isnan(pts_w[:, 0])[c2] & used[il]
            if ok.sum() < 30:
                continue
            rvec_w = Rotation.from_matrix(rot_can).as_rotvec()
            _, rvw, tvw, p_w, _, inl = bundle_adjust(
                c2[ok],
                il[ok],
                u[sel][ok],
                rvec_w,
                trans0,
                pts_w,
                f0,
                len(imgs),
                len(cl_ids),
                opt_f=False,
                schedule=((30.0, 3.0), (8.0, 1.5)),
                max_nfev=30,
            )
            cands.append((inl, imgs, used, cl_ids, rvw, tvw, p_w))
    cands.sort(key=lambda t: -t[0])
    return [
        _grow_one(c, f0, obs_c, obs_i, u, n_img, n_cl, cap, rank) for c in cands[:2]
    ]


def core_parallax(rvec, tvec, pts, posed, obs_c, obs_i):
    """Median over triangulated points of the widest ray angle between the
    posed views observing them, in degrees.

    A covisibility-picked seed can be a zero-baseline segment (a video's
    most-mutually-covisible frames are where the camera moved LEAST —
    DinoLedge's seed was a near-static clip at the end of the walk).  Such
    a core fits any focal at high inlier fraction while its depths are
    unusable, so growth and the focal scan need a parallax gate, not a
    reprojection one."""
    valid = ~np.isnan(pts[:, 0])[obs_c] & posed[obs_i]
    if not valid.any():
        return 0.0
    oc, oi = obs_c[valid], obs_i[valid]
    rot = Rotation.from_rotvec(rvec).as_matrix()
    centers = -np.einsum("nji,nj->ni", rot[oi], tvec[oi])
    d = pts[oc] - centers
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    order = np.argsort(oc, kind="stable")
    oc_s, d_s = oc[order], d[order]
    uniq, starts = np.unique(oc_s, return_index=True)
    first = np.repeat(d_s[starts], np.diff(np.append(starts, len(oc_s))), axis=0)
    cosang = np.clip((d_s * first).sum(1), -1.0, 1.0)
    widest = np.minimum.reduceat(cosang, starts)
    return float(np.degrees(np.median(np.arccos(widest))))


def _grow_one(cand, f0, obs_c, obs_i, u, n_img, n_cl, cap, rank):
    _, imgs, used, cl_ids, rvw, tvw, p_w = cand

    rvec = np.zeros((n_img, 3))
    tvec = np.tile([0.0, 0.0, -f0], (n_img, 1))
    posed = np.zeros(n_img, bool)
    pts = np.full((n_cl, 3), np.nan)
    for k, i in enumerate(imgs):
        if used[k]:
            rvec[i], tvec[i], posed[i] = rvw[k], tvw[k], True
    pts[cl_ids] = p_w

    cam0 = make_cam(f0)

    def refill(pts, rvec, tvec):
        # Triangulate clusters that lack a point but have >= 2 posed views
        # (also restores the points a budget-restricted BA wiped to NaN).
        need = np.isnan(pts[:, 0])[obs_c] & posed[obs_i]
        if need.any():
            uniq, c2n = np.unique(obs_c[need], return_inverse=True)
            rot = Rotation.from_rotvec(rvec).as_matrix()
            pts[uniq] = triangulate(
                c2n, obs_i[need], u[need], rot, tvec, posed, len(uniq), f0
            )
        return pts

    accepted, blocked, since_ba = [], set(), 0
    while posed.sum() < cap:
        cand = ~posed[obs_i] & ~np.isnan(pts[obs_c, 0])
        cnt = np.bincount(obs_i[cand], minlength=n_img)
        for j in blocked:
            cnt[j] = 0
        i = int(np.argmax(cnt))
        if cnt[i] < 6:
            break
        s = (obs_i == i) & ~np.isnan(pts[obs_c, 0])
        found = None
        p3p = p3p_resect(u[s], pts[obs_c[s]], f0)
        if p3p is not None and int(p3p[2].sum()) >= 12:
            rv0, tv0, mask = p3p
            rv, tv, _ = pose_refine(u[s][mask], pts[obs_c[s]][mask], rv0, tv0, f0)
            res = reproj_res_one(cam0, rv, tv, pts[obs_c[s]], u[s])
            found = (_inlier_fraction(res, 3.0), rv, tv)
        if found is None or (accepted and found[0] < 0.35 * float(np.median(accepted))):
            blocked.add(i)
            continue
        accepted.append(found[0])
        rvec[i], tvec[i], posed[i] = found[1], found[2], True
        pts = refill(pts, rvec, tvec)
        since_ba += 1
        if since_ba >= 3 and posed.sum() < cap:
            since_ba = 0
            # The posed set grew since the last adjustment: re-truncate.
            bm = budget_mask(posed, obs_i, rank)
            live = ba_rows(bm & ~np.isnan(pts[obs_c, 0]), obs_i)
            _, rvec, tvec, pts, _, _ = bundle_adjust(
                obs_c[live],
                obs_i[live],
                u[live],
                rvec,
                tvec,
                pts,
                f0,
                n_img,
                n_cl,
                opt_f=False,
                schedule=((30.0, 3.0), (8.0, 1.5)),
                max_nfev=30,
            )
            pts = refill(pts, rvec, tvec)

    med_inl = float(np.median(accepted)) if accepted else 1.0
    return rvec, tvec, pts, posed, med_inl


def widen(
    rvec,
    tvec,
    pts,
    posed,
    f0,
    obs_c,
    obs_i,
    u,
    n_img,
    n_cl,
    rank,
    gate,
    allow=None,
    rungs=SHELL,
    order=None,
):
    """Ladder-widen a converged fixed-f state for focal observability.

    Structure triangulated from a near-affine core has depth errors too
    large for far views to resect against directly (a 12-image jump probe
    on the console orbit failed at every focal).  Instead each rung resects
    the FARTHEST currently-viable image (weakest covisibility link that
    passes the inlier gate), then re-triangulates and bundle-adjusts — the
    reachable arc grows with every rung, so a handful of rungs spans an
    orbit that incremental most-covisible growth needs dozens of images to
    cross.  The focal stays fixed; the caller releases it afterwards.

    ``allow`` restricts the candidate pool to a boolean image mask (the
    hypothesis loop's combination stage nominates the frames another committed
    hypothesis certified by posing them, plus the bridges it certified, and
    nothing else); ``rungs`` bounds the ladder.  ``order(pool, cnt, posed)``
    replaces the ladder's own farthest-first ordering of a rung's pool (the
    combination puts the donor frames nearest an ACCEPTED bridge first, which
    only its donor-gauge certificates can say).  All three default to the
    exploration's behavior — every unposed image, ``SHELL`` rungs,
    farthest-first.
    """
    cam0 = make_cam(f0)

    def refill(pts):
        need = np.isnan(pts[:, 0])[obs_c] & posed[obs_i]
        if need.any():
            uniq, c2n = np.unique(obs_c[need], return_inverse=True)
            rot = Rotation.from_rotvec(rvec).as_matrix()
            pts[uniq] = triangulate(
                c2n, obs_i[need], u[need], rot, tvec, posed, len(uniq), f0
            )
        return pts

    rejected = set()
    accepted = []
    for _rung in range(rungs):
        valid = ~np.isnan(pts[obs_c, 0])
        cnt = np.bincount(obs_i[~posed[obs_i] & valid], minlength=n_img)
        for j in rejected:
            cnt[j] = 0
        pool = np.nonzero((cnt >= POOL_FLOOR) & ~posed)[0]
        if allow is not None:
            pool = pool[allow[pool]]
        # Weakest link = farthest first, falling back toward nearer views
        # until one resects (the farthest reachable image extends the arc,
        # and the reachable radius grows as rungs accumulate).  Log-spaced
        # sampling keeps the near end of the pool in reach — a rung whose
        # far candidates are all junk-connected must still make progress.
        pool = pool[np.argsort(cnt[pool])] if order is None else order(pool, cnt, posed)
        if len(pool) > 20:
            pool = pool[np.unique(np.geomspace(1, len(pool), 20).astype(int) - 1)]
        hit = None
        for j in pool:
            s = (obs_i == j) & valid
            p3p = p3p_resect(u[s], pts[obs_c[s]], f0)
            if p3p is None or int(p3p[2].sum()) < 12:
                continue
            rv0, tv0, mask = p3p
            rv, tv, _ = pose_refine(u[s][mask], pts[obs_c[s]][mask], rv0, tv0, f0)
            res = reproj_res_one(cam0, rv, tv, pts[obs_c[s]], u[s])
            # Relative gate (like core growth): an absolute floor rejects
            # every candidate when the probe focal is far from true and the
            # whole fit sits at a low but consistent inlier level.
            if _inlier_fraction(res, 3.0) < gate:
                continue
            hit = (j, rv, tv)
            break
        if hit is None:
            break
        # Verified acceptance: a far P3P against depth-noisy points can
        # find a junk consensus that wrecks the BA (wide span, broken
        # geometry).  Accept, BA, then require the image to have SURVIVED
        # the adjustment; revert and blacklist it otherwise.  (A global
        # "did the old images keep their fit" check does NOT work here: a
        # legitimate far rung worsens the old fit by design — breaking the
        # bas-relief compensation is what it is for.)
        saved = (rvec.copy(), tvec.copy(), pts.copy(), posed.copy())
        j, rvec[j], tvec[j] = hit
        posed[j] = True
        pts = refill(pts)
        # Every rung adds an image, so the visible cluster set — and with it
        # the truncation — is re-derived per rung.
        bm = budget_mask(posed, obs_i, rank)
        live = ba_rows(bm & ~np.isnan(pts[obs_c, 0]), obs_i)
        _, rvec, tvec, pts, _, _ = bundle_adjust(
            obs_c[live],
            obs_i[live],
            u[live],
            rvec,
            tvec,
            pts,
            f0,
            n_img,
            n_cl,
            opt_f=False,
            schedule=((12.0, 2.0), (4.0, 1.0)),
            max_nfev=30,
        )
        pts = refill(pts)
        s = (obs_i == j) & ~np.isnan(pts[obs_c, 0])
        res = reproj_res_one(cam0, rvec[j], tvec[j], pts[obs_c[s]], u[s])
        if _inlier_fraction(res, 3.0) < gate:
            rvec, tvec, pts, posed = saved
            rejected.add(int(j))
        else:
            accepted.append(int(j))
    return rvec, tvec, pts, posed, accepted


# ── Photometric verification (embed-patches machinery) ──────────────────────


def localize_anchors(names, sub, rvec, tvec, f0, pts_a, tr_a, tr_img, tr_feat):
    """Congeal-localize anchor keypoints across an image subset.

    Builds a ``CameraViews`` over ``sub`` (image indexes, all posed), a
    feature-scaled mean-viewing-normal patch cloud over the anchor points
    via ``PatchCloud.from_tracks`` (keypoint scales read from the subset
    images' ``.sift`` files), and an in-memory pyramid set, then localizes
    every anchor in EVERY subset view (a patch tile is registered against
    the leave-one-out consensus of the other views' tiles).  Returns
    (anchor idx, full image idx, keypoint xy) arrays of the KEPT views —
    appearance-verified observations.  Cost: ~1-2 s for 400 anchors over
    ~15 4K frames, pyramids included.
    """
    import cv2

    from sfmtool._sfmtool.patches import CameraViews, ImagePyramidSet, PatchCloud
    from sfmtool._sfmtool.io import read_sift_partial
    from sfmtool.sift.file import get_sift_path_for_image

    sub_names = [names[int(g)] for g in sub]
    q = Rotation.from_rotvec(rvec[sub]).as_quat()[:, [3, 0, 1, 2]]
    views = CameraViews([make_cam(f0)], q, tvec[sub])
    ws = WS.resolve()
    # Per-observation keypoint scale (the norm of the affine shape's first
    # column — the same value extent="feature_size" reads from the .sift
    # files in reconstruction mode).
    scales = np.full(len(tr_a), np.nan)
    for j, name in enumerate(sub_names):
        m = tr_img == j
        if not m.any():
            continue
        aff = read_sift_partial(
            get_sift_path_for_image(ws / name), int(tr_feat[m].max()) + 1
        )["affine_shapes"].astype(np.float64)
        scales[m] = np.hypot(aff[tr_feat[m], 0, 0], aff[tr_feat[m], 1, 0])
    # Patch size must track the FEATURE scale: a fixed pixel radius that is
    # fine on a 480 px frame is hopelessly small on 4K (a 12 px tile has no
    # discriminative texture and a patch-grid search budget of a few source
    # px, so localization can neither reach nor reject anything).
    cloud = PatchCloud.from_tracks(
        views,
        np.c_[pts_a, np.ones(len(pts_a))],
        tr_a.astype(np.uint32),
        tr_img.astype(np.uint32),
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
    all_views = list(range(len(sub_names)))
    results = cloud.localize_keypoints(
        views,
        pyrset,
        view_sets=dict.fromkeys(range(len(pts_a)), all_views),
        max_shift_px=60.0,
        search=12.0,
        min_relative_zncc=0.6,
    )
    a_idx, i_idx, uv = [], [], []
    for r in results:
        pid = int(r["point_index"])
        views = np.asarray(r["views"])
        kps = np.asarray(r["keypoints"], dtype=np.float64)
        for k, v in enumerate(views):
            a_idx.append(pid)
            i_idx.append(int(sub[int(v)]))
            uv.append(kps[k])
    return (
        np.asarray(a_idx, dtype=np.int64),
        np.asarray(i_idx, dtype=np.int64),
        np.asarray(uv, dtype=np.float64).reshape(-1, 2),
    )


# ── Evaluation ───────────────────────────────────────────────────────────────


def compare_to_reference(names, rvec, tvec, f_est, mask):
    """Camera errors vs the first non-bootstrap solve in the workspace.

    A posed SUBSET can have nearly-degenerate camera centers (a short arc of
    a long orbit), which leaves the center-based similarity alignment a free
    rotation about the arc — so the gauge rotation for the ROTATION errors
    is fitted from the camera rotations (well-conditioned always), and the
    center errors use the free similarity (its own best case).
    """
    names = [n for j, n in enumerate(names) if mask[j]]
    rvec, tvec = rvec[mask], tvec[mask]
    ref_files = (
        [REF]
        if REF is not None
        else sorted(
            p
            for p in WS.glob("sfmr/*.sfmr")
            if "bootstrap" not in p.name and "fast-pinhole" not in p.name
        )
    )
    if not ref_files:
        print("no reference solve found; skipping comparison")
        return
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction
    from sfmtool._sfmtool.analysis import estimate_alignment_rs

    # A workspace can hold reconstructions this build cannot read (e.g. a
    # beta-model file from before a parameterization change); skip those
    # rather than dying before the seed is written.
    ref = None
    for ref_file in ref_files:
        try:
            ref = SfmrReconstruction.load(ref_file)
            break
        except OSError as e:
            print(f"reference {ref_file.name} unreadable ({e}); trying next")
    if ref is None:
        print("no readable reference solve found; skipping comparison")
        return
    ref_names = list(ref.image_names)
    common = [n for n in names if n in ref_names]
    if len(common) < 3:
        print(f"only {len(common)} common images with {ref_file.name}; skipping")
        return

    def centers_rots(qs, ts, order):
        rs = Rotation.from_quat(np.asarray(qs)[order][:, [1, 2, 3, 0]]).as_matrix()
        return -np.einsum("nij,ni->nj", rs, np.asarray(ts)[order]), rs

    q_wxyz = Rotation.from_rotvec(rvec).as_quat()[:, [3, 0, 1, 2]]
    ei = np.array([names.index(n) for n in common])
    ri = np.array([ref_names.index(n) for n in common])
    c_est, r_est = centers_rots(q_wxyz, tvec, ei)
    c_ref, r_ref = centers_rots(ref.quaternions_wxyz, ref.translations, ri)

    # Gauge rotation from the rotations: argmin_g sum ||R_est_i g - R_ref_i||.
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
    c_all, _ = centers_rots(
        ref.quaternions_wxyz, ref.translations, np.arange(len(ref_names))
    )
    diam_all = np.max(np.linalg.norm(c_all[:, None, :] - c_all[None, :, :], axis=2))
    f_ref = ref.cameras[0].focal_lengths[0]
    print(
        f"vs reference {ref_files[0].name} ({len(common)} common images; "
        f"subset spans {100 * diam / diam_all:.0f}% of the reference rig):"
    )
    print(
        f"  camera rotation err: mean {rot_err.mean():.2f}, "
        f"median {np.median(rot_err):.2f}, max {rot_err.max():.2f} deg"
    )
    print(
        f"  camera center err:   mean {100 * cen_err.mean():.2f}%, "
        f"median {100 * np.median(cen_err):.2f}%, "
        f"max {100 * cen_err.max():.2f}% of subset diameter"
    )
    print(
        f"  focal: fast {f_est:.1f} px vs reference {f_ref:.1f} px "
        f"({ref.cameras[0].to_dict()['model']}) — "
        f"{100 * (f_est / f_ref - 1):+.1f}%"
    )


# ── Seed hypothesis loop ─────────────────────────────────────────────────────
#
# The seed stage develops structure hypotheses while the ones it commits stay
# trustworthy AND their coverage claims leave real evidence unexplained, then
# commits the one the capture-level measurements support.  A hypothesis is one
# full exploration (probe -> widen -> verify -> scan -> release) over an
# admitted cluster selection; the first admits the whole selection, each later
# one admits the COVERAGE COMPLEMENT of everything the committed hypotheses
# before it claimed.  See `specs/core/geometry/seed-hypothesis-loop.md`.
#
# A capture one hypothesis explains produces exactly the single-hypothesis
# result: its complement retains most of the admission, the materiality gate
# refuses to explore it, and the loop ends leaving h0 alone with the metadata
# it wrote before the loop existed.

# The pose-noise scale two hypotheses must disagree by, over the frames they
# both pose, to be two WORLDS rather than one world seeded from two windows.
POSE_DISAGREE_DEG = 5.0
# A complement is explored only when the claim bit: it must retain less than
# this fraction of the clusters the pass that produced it admitted.
MATERIAL_RETENTION = 0.5

# How many FINITE candidates one capture may commit.  A resource cap, not a
# judgment: the generator has no opinion about which candidates are worth
# keeping, and this only bounds what a pathological capture can cost.  It awaits
# fleet evidence -- if captures routinely hit it, the cap is what is wrong, not
# the set.
CANDIDATE_BUDGET = 8

# The run's round identity, resolved once (see `round_stamp`).
_ROUND_STAMP = {}


def round_stamp():
    """The run's round identity: one ``SFMTOOL_ROUND_STAMP`` across a fleet
    invocation, or a self-stamp for a bare run.  Resolved ONCE, so every
    artifact a run accumulates — the per-hypothesis releases and the round copy
    of the final — carries the same stamp."""
    if "v" not in _ROUND_STAMP:
        from datetime import datetime

        _ROUND_STAMP["v"] = os.environ.get("SFMTOOL_ROUND_STAMP") or (
            datetime.now().strftime("%Y%m%dT%H%M")
        )
    return _ROUND_STAMP["v"]


def image_rows(data):
    """Observation rows grouped by image: ``(order, bounds)`` such that
    ``order[bounds[g]:bounds[g + 1]]`` are image ``g``'s rows.  Memoized on the
    selection's own dict — the claim and the cluster test both walk every image
    of a large observation array, and a per-image mask scan is quadratic."""
    if "_img_rows" not in data:
        obs_i = data["obs_i"]
        order = np.argsort(obs_i, kind="stable")
        data["_img_rows"] = (
            order,
            np.searchsorted(obs_i[order], np.arange(data["n_img"] + 1)),
        )
    return data["_img_rows"]


def cell_size(xy):
    """One image's claim-grid cell: the MEDIAN NEAREST-NEIGHBOUR distance among
    the retained members' keypoints there.

    Coverage measured at the spacing the matcher actually sampled the scene at
    — fine on dense texture, coarse on sparse — with the capture's pixel scale
    divided out.  Returns 0 when the image resolves no spacing at all."""
    from scipy.spatial import cKDTree

    d = cKDTree(xy).query(xy, k=2)[0][:, 1]
    cell = float(np.median(d))
    if cell > 0:
        return cell
    # More than half the members coincide (duplicate detections on one
    # keypoint), so the median spacing is zero and the grid would degenerate to
    # the keypoints themselves.  The smallest spacing the image DOES resolve is
    # still a measurement of its sampling; when every member coincides there is
    # no spacing to measure and the image claims nothing.
    pos = d[d > 0]
    return float(pos.min()) if len(pos) else 0.0


def cell_codes(xy, cell, cols):
    """Grid cell index per keypoint, in one flat code per (row, column).

    Columns are clamped into the grid so the code stays injective; rows are
    not, so a keypoint past the last stamped row simply matches no claimed
    cell.  Stamping and testing share this function — a grid is only a claim
    if both sides read it the same way."""
    cx = np.clip((xy[:, 0] / cell).astype(np.int64), 0, cols - 1)
    cy = np.maximum((xy[:, 1] / cell).astype(np.int64), 0)
    return cy * cols + cx


def claim_coverage(res, data, claims):
    """Stamp a committed hypothesis's claim into ``claims``.

    The retained structure is every cluster with a finite position in the
    released geometry's FULL triangulation (``claim_pts``, not the BA row
    budget's subset — the budget is a solver-cost control, and the
    observations it leaves out are read by the same geometry all the same).
    The claim is TRANSITIVE over cluster membership: a retained cluster is an
    explained 3D point, so its members stamp in EVERY image they appear in,
    posed or not.  The members come off the selection handle's own arrays,
    which cover the whole image table, so a hypothesis that poses a handful of
    a long capture's frames still claims its structure's footprint
    capture-wide.

    The claim is an OCCUPANCY GRID per image, not a pixel bitmap: the cell size
    is that image's median nearest-neighbour keypoint spacing among the
    retained members (`cell_size`), and a cell holding at least one of them is
    claimed.  An image with fewer than two retained members has no spacing to
    measure and claims nothing.  ``claims`` accumulates per data image index as
    a LIST of grids — each hypothesis stamps into its own grid geometry, and
    the cluster test evaluates a member against every grid its image carries.
    """
    t0 = time.perf_counter()
    obs_c, obs_uv = data["obs_c"], data["obs_uv"]
    order, bounds = image_rows(data)
    finite = np.isfinite(res["claim_pts"][:, 0])
    n_img_claimed = n_members = 0
    areas = []
    for g in range(data["n_img"]):
        rows = order[bounds[g] : bounds[g + 1]]
        sel = rows[finite[obs_c[rows]]]
        if len(sel) < 2:
            continue
        xy = obs_uv[sel]
        cell = cell_size(xy)
        if cell <= 0:
            continue
        w, h = data["dims"][g]
        cols = max(1, int(np.ceil(w / cell)))
        codes = np.unique(cell_codes(xy, cell, cols))
        claims.setdefault(g, []).append((cell, cols, codes))
        n_img_claimed += 1
        n_members += len(sel)
        areas.append(len(codes) * cell * cell / (w * h))
    print(
        f"coverage claim: {int(finite.sum())} retained clusters, {n_members} "
        f"members stamped over {n_img_claimed} images at "
        f"{100 * float(np.mean(areas)) if areas else 0.0:.1f}% mean claimed "
        f"area in {time.perf_counter() - t0:.1f}s; {len(claims)} images now "
        f"carry a claim [{elapsed():.1f}s]"
    )


def unclaimed_clusters(data, claims):
    """The clusters the accumulated claim leaves unexplained, plus the number
    of claimed members.

    A cluster is claimed when MORE THAN HALF of its members fall in claimed
    cells of their images, tested against every grid the member's image
    carries."""
    obs_c, obs_uv = data["obs_c"], data["obs_uv"]
    order, bounds = image_rows(data)
    claimed = np.zeros(len(obs_c), bool)
    for g, grids in claims.items():
        rows = order[bounds[g] : bounds[g + 1]]
        if not len(rows):
            continue
        xy = obs_uv[rows]
        hit = np.zeros(len(rows), bool)
        for cell, cols, codes in grids:
            code = cell_codes(xy, cell, cols)
            k = np.minimum(np.searchsorted(codes, code), len(codes) - 1)
            hit |= codes[k] == code
        claimed[rows] = hit
    n_cl = data["n_cl"]
    sizes = np.bincount(obs_c, minlength=n_cl).astype(np.float64)
    hits = np.bincount(obs_c, weights=claimed.astype(np.float64), minlength=n_cl)
    return np.nonzero(hits <= 0.5 * sizes)[0], int(claimed.sum())


def complement_selection(handle, survivors):
    """The next hypothesis's admission: ``handle`` minus the claimed clusters,
    as a cluster-id restriction of the stage's own selection handle.

    A complement is therefore an ordinary derived selection — it carries
    provenance, and the downstream stages read it exactly like the unrestricted
    admission.  No stage applies a claim predicate of its own."""
    nxt = handle.select_clusters(
        min_span=_LOAD_CTX["min_span"],
        restrict_cluster_ids=[int(c) for c in survivors],
    )
    return nxt, repackage_selection(
        nxt,
        _LOAD_CTX["names"],
        _LOAD_CTX["dims"],
        want_warp=_LOAD_CTX["want_warp"],
        refine_radius=_LOAD_CTX["refine_radius"],
    )


def release_path(rung, idx):
    """Where a committed hypothesis's release is written.

    Under the rung it is ``h<NN>.sfmr`` in the STAGING directory, so the
    product's file names carry no stamp and a crashed run cannot leave a
    half-written product where a reader looks.  Off the rung it is the legacy
    stamped path, untouched."""
    if rung is not None:
        return rung.stage / f"h{idx:02d}.sfmr"
    out = WS / "sfmr" / "seed-hypotheses" / f"{round_stamp()}-h{idx}.sfmr"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_finite_release(idx, res, data, out):
    """A finite hypothesis's release: a RELEASE-GRADE reconstruction of its
    poses and its retained structure.

    Every committed hypothesis, the winner included, so the alternatives the
    loop developed stay inspectable after every run.  Release-grade is poses
    and points: no consensus bitmaps and no patch frames, which is also what
    keeps it cheap (the surfel-frame solve is what makes the writer expensive,
    and it opens every posed image's `.sift` affine array).  Instrumentation
    must never kill the run it instruments, so a failure is a one-line warning.

    The artifact is written under the CAPTURE'S OWN camera model
    (`bootstrap_module`): a fisheye capture's hypotheses densify and reproject
    through the equidistant context, never the pinhole default.

    Returns the path written, or None when there was no reconstruction to write
    (a hypothesis with no multi-view point) — the manifest records which."""
    B = bootstrap_module()
    # The hypothesis's own LENS installed for the write, and taken back out
    # after: the writer stamps whatever camera the context carries, and each
    # hypothesis carries its own.  A refusal installs the spline model with
    # zero coefficients, which is the base map bit for bit, so the artifact is
    # what it always was apart from the camera record.
    lens, prior = res.get("lens"), B.camera_context()
    try:
        if lens is not None:
            B.set_camera_context(
                B.spline_model()[0],
                float(lens["f_chart"]),
                bspline=lens["coeffs"],
                theta_max=lens["d_max"],
            )
        written = write_finite_body(B, idx, res, data, out)
        # THE REFUSED ARM, beside the release it lost to.  The verdict is the
        # zero spline in the release; this is the lens that verdict rejected,
        # written as the full reconstruction acceptance would have shipped so a
        # human can open the two side by side and say whether the bar was
        # right.  A rung that keeps only its winners cannot have its bars
        # revised, and these bars are new.
        ref = (lens or {}).get("refused") or {}
        if ref.get("release_pts") is not None:
            arm_out = out.with_name(f"{out.stem}-spline-refused.sfmr")
            cam = ref["camera"]
            B.set_camera_context(
                cam["model"],
                float(ref["f_chart"]),
                bspline=np.asarray(cam["params"]["coefficients"], dtype=np.float64),
                theta_max=float(lens["d_max"]),
            )
            rv, tv, pd = _snap_full_frame(
                data,
                res["keep"],
                ref["rvec"],
                ref["tvec"],
                res["posed"],
                float(ref["f_chart"]),
            )
            arm_res = dict(
                res,
                f=float(ref["f_chart"]),
                f_released=float(ref["f_chart"]),
                rvec_full=rv,
                tvec_full=tv,
                posed_full=pd,
                release_pts=ref["release_pts"],
                flags=[*res["flags"], f"spline_refused_{ref['reason']}"],
            )
            ref["file"] = getattr(
                write_finite_body(B, f"{idx}-spline-refused", arm_res, data, arm_out),
                "name",
                None,
            )
        return written
    finally:
        B.set_camera_context(
            prior["model"],
            prior["focal"],
            bspline=prior["bspline"],
            theta_max=prior["theta_max"],
        )


def write_finite_body(B, idx, res, data, out):
    """`write_finite_release` with the camera context already installed."""
    return B.seed_snapshot(
        f"hypothesis-{idx}",
        data,
        res["f"],
        res["rvec_full"],
        res["tvec_full"],
        # The structure in the space the SOLVE produced it (`release_data`
        # goes with it), which is the pass's own unless a group-local
        # re-admission gave this hypothesis a denser working set.
        res.get("release_pts", res["claim_pts"]),
        res["posed_full"],
        extra_tool_options={
            "hypothesis_index": str(idx),
            "focal_released_px": f"{res['f_released']:.3f}",
            "inlier_fraction": f"{res['inl']:.4f}",
            "confidence_flags": "|".join(res["flags"]) or "ok",
            "qualified": str(qualifies(res)),
        },
        path=out,
        release_grade=True,
    )


def write_rotation_release(
    idx, data, f_rot, res_obs, keep, rvec, tvec, dirs, f_src, out
):
    """A rotation-only layer's release: the same release-grade writer, with the
    point block promoted to INFINITY afterwards.

    The writer stores Euclidean positions, so the directions go in as unit
    vectors at ``w = 1`` and the saved cloud is re-stated homogeneously with
    ``w = 0`` (the world rotation the writer applies is a rotation, so a unit
    direction reaches the canonical frame still a unit direction).  That is the
    whole difference between this artifact and a finite one: same poses, same
    tracks, same colors, and a cloud that claims bearing without depth."""
    B = bootstrap_module()
    try:
        recon = B.save_sfmr(
            data,
            f_rot,
            rvec,
            tvec,
            dirs,
            keep,
            res_obs,
            out,
            tool_options={
                "hypothesis_index": str(idx),
                "focal_released_px": f"{f_rot:.3f}",
                "structure_model": "rotation_only",
                "focal_source": f_src,
                "confidence_flags": "rotation_only",
                "qualified": "False",
            },
            quiet=True,
            release_grade=True,
        )
        xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
        norm = np.linalg.norm(xyzw[:, :3], axis=1, keepdims=True)
        xyzw[:, :3] /= np.maximum(norm, 1e-12)
        xyzw[:, 3] = 0.0
        recon.clone_with_changes(positions=np.ascontiguousarray(xyzw)).save(str(out))
        print(f"  [hypothesis-{idx} release: {len(xyzw)} points, all at infinity]")
        return out
    except Exception as exc:  # noqa: BLE001 -- instrumentation never kills the run
        print(f"  [rotation-only release FAILED: {type(exc).__name__}: {exc}]")
        return None


def write_relaxed_release(idx, res, tool_options, out):
    """A relaxed member's release: the same release-grade writer, under the
    lens the relaxation settled on, with the at-infinity rows promoted
    afterwards.

    A relaxed member is a MIXED cloud -- finite positions where its baselines
    priced a depth, unit bearings where they could not -- so unlike a
    rotation-only release only part of the point block is re-stated at
    ``w = 0``.  ``alive`` maps a written row back to the cluster it came from,
    which is what says which part.

    The lens is installed in the bootstrap context for the write and taken back
    out in ``finally``: the writer stamps whatever camera the context carries,
    and the next hypothesis must not write under this one's."""
    B = bootstrap_module()
    lens, prior = res.get("lens"), B.camera_context()
    try:
        if lens is not None:
            B.set_camera_context(
                lens["model"],
                float(lens["f_chart"]),
                bspline=np.asarray(lens["coeffs"], dtype=np.float64),
                theta_max=float(lens["d_max"]),
            )
        # The CHART focal, which is what the installed model is parameterized
        # by; the equivalent focal the manifest reports is a reading of the
        # composite map, not a parameter of it.
        f_write = float(res["f_released"] if lens is None else lens["f_chart"])
        recon, alive, _posed_img = B.save_sfmr(
            res["data"],
            f_write,
            res["rvec_full"],
            res["tvec_full"],
            res["release_pts"],
            res["keep"],
            res["res_obs"],
            out,
            return_alive=True,
            tool_options=tool_options,
            quiet=True,
            release_grade=True,
            operation="seed-relaxed",
        )
        inf = np.asarray(res["at_inf"], dtype=bool)[alive]
        if inf.any():
            xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
            norm = np.linalg.norm(xyzw[inf, :3], axis=1, keepdims=True)
            xyzw[inf, :3] /= np.maximum(norm, 1e-12)
            xyzw[inf, 3] = 0.0
            recon.clone_with_changes(positions=np.ascontiguousarray(xyzw)).save(
                str(out)
            )
        print(
            f"  [hypothesis-{idx} release: {len(alive)} points, "
            f"{int(inf.sum())} at infinity]"
        )
        return out
    except Exception as exc:  # noqa: BLE001 -- instrumentation never kills the run
        print(f"  [relaxed release FAILED: {type(exc).__name__}: {exc}]")
        return None
    finally:
        B.set_camera_context(
            prior["model"],
            prior["focal"],
            bspline=prior["bspline"],
            theta_max=prior["theta_max"],
        )


def commit_hypothesis(rung, idx, res, write, model="finite", f_source=None, extra=None):
    """THE seam every committed hypothesis passes through: finite winners,
    ladder runner-ups and rotation-only layers alike.

    It allocates the release path, calls ``write(out)`` to produce the
    artifact, and records the manifest entry.  One seam means one place where a
    hypothesis becomes part of the product, so a model that joins the set later
    inherits the naming, the provenance and the schema instead of restating
    them.

    The entry carries what rung 2 needs from this hypothesis alone: its model
    and (for the far layers) its scope and the sibling it pairs with, its
    camera, its metrics and flags, the admission its solve ran on, and the
    PROVENANCE of its geometry -- the frames it was seeded from and the frames
    it actually posed.  ``f`` stays beside the camera block for one transition.

    A no-op off the rung except for the release itself: the point count costs a
    read of the artifact that a legacy run has no reason to pay."""
    out = release_path(rung, idx)
    release = write(out)
    if rung is None:
        return release
    points = None
    if release is not None:
        try:
            from sfmtool._sfmtool.reconstruction import SfmrReconstruction

            points = int(SfmrReconstruction.load(str(release)).point_count)
        except Exception as exc:  # noqa: BLE001 — a manifest field, never the run
            print(f"  [rung1: could not read {Path(release).name}: {exc}]")
    # THE CAMERA the hypothesis shipped, and the two focals that describe it.
    # `f` is the EQUIVALENT focal -- the best base-model fit of the composite
    # map over the observed field -- because that is the quantity comparable
    # across hypotheses and against the capture-level vote once coefficients
    # exist.  `f_chart` is the raw chart focal the model carries; under the
    # center-anchored gauge it trades against the coefficients, so two lenses
    # with the same optics can report different chart focals and comparing
    # those compares parameterizations.  They are equal, and `f_chart` absent,
    # whenever the spline is all zeros.
    lens = res.get("lens")
    f = float(res["f_released"] if lens is None else lens["f_eq"])
    if lens is None:
        camera = {
            "model": "EQUIDISTANT_FISHEYE" if fisheye_stage1() else "PINHOLE",
            "params": {"focal_length": f},
        }
    else:
        cam_model, d_key, _ = bootstrap_module().spline_model()
        camera = {
            "model": cam_model,
            "params": {
                "focal_length": float(lens["f_chart"]),
                d_key: float(lens["d_max"]),
                "coefficients": [float(c) for c in lens["coeffs"]],
            },
            "focal_length_is": "chart focal; the entry's `f` is the equivalent",
            "accepted": bool(lens["accepted"]),
        }
    posed = res.get("posed_full")
    posed = res["posed"] if posed is None else posed
    seed = res.get("seed_frames")
    entry = {
        "idx": int(idx),
        "model": model,
        "f": f,
        # The lens this hypothesis was released under, whole.
        "camera": camera,
        "inlier_pct": 100.0 * float(res["inl"]),
        "posed": int(np.asarray(posed).sum()),
        "reach": float(res["reach"]),
        "flags": list(res["flags"]),
        "qualified": bool(qualifies(res)),
        "points": points,
        "release_file": None if release is None else Path(release).name,
        # PROVENANCE: which frames this geometry came from and which it reached.
        "seed_frames": None if seed is None else [int(k) for k in seed],
        "posed_frames": [int(k) for k in np.nonzero(np.asarray(posed))[0]],
    }
    # THE LADDER RECORD, advisory throughout: where the score put this
    # candidate, how many duplicate finalists collapsed into it, and the two
    # evidence channels the retired indictments used to gate on.  They are
    # corroboration for whatever reads the set, not verdicts taken here.
    entry["ladder_rank"] = int(res.get("ladder_rank", 0))
    if res.get("collapsed"):
        entry["ladder_collapsed"] = int(res["collapsed"])
    if res.get("rotation_disagreement_deg") is not None:
        entry["rotation_disagreement_deg"] = float(res["rotation_disagreement_deg"])
    if res.get("far_beyond_at_ladder") is not None:
        entry["far_beyond_at_ladder"] = int(res["far_beyond_at_ladder"])
    # The arm the keep-best refused, and why: the camera it would have shipped,
    # the file it was written to, and the numbers the verdict was taken on.  A
    # reader can open that file against the release and judge the bar.
    ref = (lens or {}).get("refused")
    if ref is not None:
        entry["spline_refused"] = {
            k: v
            for k, v in ref.items()
            if k not in ("rvec", "tvec", "pts", "release_pts")
        }
    if lens is not None and lens["f_chart"] != f:
        # Only where the two differ, which is only where coefficients live.
        entry["f_chart"] = float(lens["f_chart"])
    if f_source is not None:
        entry["f_source"] = f_source
    # What the SOLVE's own admission was, when it was not the global one: the
    # global block describes the capture's cut, this describes the set this
    # hypothesis actually worked with.
    if res.get("local_admission"):
        entry["local_admission"] = res["local_admission"]
    entry.update(extra or {})
    rung.hypotheses.append(entry)
    return release


# ── The candidate-evaluation battery (SFMTOOL_SEED_EVAL) ────────────────────
#
# THE SET IS THE PRODUCT, and a set without evidence about its members is a
# list.  When the loop has committed everything it is going to commit, every
# member is measured once, here, and the readings ride in the member's own
# manifest entry: a later selection pass ranks, refuses and trims on stored
# evidence instead of re-opening artifacts and re-deriving it.
#
# The battery lives in `seed_candidate_eval`, which sees only finished arrays
# and its own fixed seeds.  It NEVER perturbs the run: it draws from no RNG the
# reconstruction path consumes, mutates nothing it is handed, and writes only
# manifest and evolution fields, so a released `.sfmr` is byte-identical
# whether it ran or not.  `SFMTOOL_SEED_EVAL=0` switches the whole thing off.


def seed_eval_on():
    """Whether the candidate-evaluation battery runs this session.

    Read here rather than imported so the loader can ask the question before
    anything has imported the battery."""
    return (os.environ.get("SFMTOOL_SEED_EVAL", "1") or "1").strip() != "0"


def relax_on():
    """Whether the RELAXATION rung runs this session (``SFMTOOL_RELAX``).

    The question is the package's own; it is asked through here so the
    hypothesis loop can hold the far layers for the rung without importing the
    rest of it."""
    import seed_relax

    return seed_relax.relax_on()


def capture_floors(blocks):
    """The capture's own conditioning floors, off the members already measured.

    The battery's hold-out gates are quantiles of the CAPTURE's own per-frame
    readings pooled over its members, so a second call measuring more members
    would re-derive them on a different population and move the readings of the
    members already measured.  The relaxed members are therefore measured with
    these passed in -- conditioning respected, never re-derived, which is
    exactly what ``evaluate``'s ``floors`` argument is for."""
    out = {
        "inlier_floor": None,
        "rot_inlier_floor": None,
        "rot_support_spread_bar": None,
    }
    for b in blocks.values():
        sr = b.get("self_resection") or {}
        rr = b.get("rot_self_resection") or {}
        if out["inlier_floor"] is None and sr.get("inlier_floor") is not None:
            out["inlier_floor"] = float(sr["inlier_floor"])
        if out["rot_inlier_floor"] is None and rr.get("inlier_floor") is not None:
            out["rot_inlier_floor"] = float(rr["inlier_floor"])
        if (
            out["rot_support_spread_bar"] is None
            and rr.get("support_spread_bar_deg") is not None
        ):
            out["rot_support_spread_bar"] = float(rr["support_spread_bar_deg"])
    return out


def relax_far_layers(rung, data_full):
    """THE RELAXATION RUNG: a finite sibling for every rotation-only member.

    A rotation-only member claims bearing without range, and the observations
    its model REFUSED are the near points that carry its baselines.  The rung
    turns those rows into camera centres and finite depths, fills the result in
    from the source clusters the member's admission never held, reads a lens on
    it, and commits it as a member BESIDE the original
    (``specs/core/geometry/seed-relaxation.md``).

    It runs here, as rung 1's last exploration phase, because this is the only
    place that holds both halves of what it needs: the far layers' own arrays,
    which ``attach_evaluation`` clears as soon as it has read them, and the
    pre-restriction selection handle the fill-in reads its clusters from.  The
    relaxed member enters the set through the one seam every hypothesis passes
    through, so it inherits the naming, the release path and the schema.

    Nothing about the member it read changes: the rotation-only entry, its
    release file and its channels stand, and it gains only a back-pointer to
    the sibling -- or, where the chain refused, the reason."""
    import seed_candidate_eval as EV
    import seed_relax
    from seed_relax import release as RELEASE

    opts = seed_relax.options()
    by_idx = {int(h["idx"]): h for h in rung.hypotheses}
    idx = len(rung.hypotheses)
    t0 = elapsed()
    n_ok = n_ref = 0
    for layer in list(rung.far_layers):
        src = by_idx.get(int(layer["idx"]), {})
        ld = layer["data"]
        scope = src.get("scope", "capture")
        print(
            f"\n=== hypothesis {idx}: the relaxation of h{layer['idx']} "
            f"({scope}, {int(np.asarray(layer['posed']).sum())} posed frames) ==="
        )
        # A FRESH member: the chain extends the one it is handed with the
        # clusters the fill-in admits, and the battery's own far-layer member
        # is built later from these same arrays.
        m = EV.Member(
            int(layer["idx"]),
            "rotation_only",
            data_full["names"],
            make_cam(float(layer["f"])),
            float(layer["f"]),
            layer["rvec"],
            layer["tvec"],
            layer["posed"],
            layer["dirs"],
            (ld["obs_c"], ld["obs_i"], ld["obs_uv"], ld.get("obs_f")),
            shapes=ld.get("obs_shape"),
            keep=layer["keep"],
        )
        try:
            result = seed_relax.run(m, rung.handle, opts)
        except Exception as exc:  # noqa: BLE001 -- a rung never kills the run
            print(f"  [relaxation FAILED: {type(exc).__name__}: {exc}]")
            result = seed_relax.pipeline.RelaxResult(
                refused=f"{type(exc).__name__}: {exc}"
            )
        block = _jsonable(RELEASE.relaxation_block(result))
        if not result.ok:
            src["relaxation"] = block
            n_ref += 1
            print(
                f"hypothesis {idx} (relaxed): the chain refused "
                f"({result.refused}); nothing committed [{elapsed():.1f}s]"
            )
            continue
        frames = np.sort(np.asarray([int(j) for j in result.state["frames"]], np.int64))
        res = RELEASE.relaxed_res(
            result, ld, capture_reach(frames), f_source=src.get("f_source")
        )
        tool_opts = RELEASE.tool_options(
            result,
            idx,
            paired_with=int(layer["idx"]),
            scope=scope,
            f_source=src.get("f_source"),
        )
        res["release_file"] = commit_hypothesis(
            rung,
            idx,
            res,
            lambda out, i=idx, r=res, t=tool_opts: write_relaxed_release(i, r, t, out),
            model="relaxed",
            f_source=src.get("f_source"),
            extra={
                "paired_with": int(layer["idx"]),
                "scope": scope,
                "relaxation": block,
            },
        )
        src["relaxed_as"] = int(idx)
        rung.relaxed_layers.append({"idx": int(idx), "res": res, "result": result})
        c = result.census
        print(
            f"hypothesis {idx} committed: f {res['f_released']:.1f} px "
            f"(relaxed from h{layer['idx']}), inlier<2px {100 * res['inl']:.1f}%, "
            f"{res['kept']} posed, reach {100 * res['reach']:.0f}% "
            f"(capture-level), flags relaxed, qualified False"
        )
        print(
            f"  relaxed ({scope}, paired with h{layer['idx']}): "
            f"{c['n_finite_final']} finite of {c['n_points_final']} points "
            f"(+{(c.get('fill') or {}).get('n_added', 0)} clusters filled in), "
            f"early release {c.get('early_release')}, late release "
            + ("applied" if (c.get("late_release") or {}).get("applied") else "held")
            + f", reproj med {(c.get('retri') or {}).get('reproj_med_px')} px "
            f"[{elapsed():.1f}s]"
        )
        idx += 1
        n_ok += 1
    print(
        f"\nseed relaxation: {n_ok} relaxed members committed, {n_ref} refused "
        f"[{elapsed():.1f}s, +{elapsed() - t0:.1f}s]"
    )


def member_camera(res):
    """The camera a committed hypothesis was released under.

    The same one `write_finite_release` installs for the artifact: the
    hypothesis's own spline lens where it carries coefficients, the capture's
    base model otherwise.  A refusal ships a zero spline, which is the base map
    bit for bit."""
    lens = res.get("lens")
    return make_cam(float(res["f_released"])) if lens is None else evo_lens_cam(lens)


def member_focal_eq(res):
    """The hypothesis's EQUIVALENT focal — what the manifest reports as `f`."""
    lens = res.get("lens")
    return float(res["f_released"] if lens is None else lens["f_eq"])


def peer_records(hyps):
    """Peer corroboration: each member against every sibling on shared frames.

    Gauge-free by construction (relative rotations only) and the one channel
    that says anything about a member's RELATION to its rivals — how far the
    two stand apart where they overlap, and how much of each other's posed sets
    they cover.  Cheap: the set is capped at the candidate budget."""
    out = {k: [] for k in range(len(hyps))}
    for a, b in itertools.combinations(range(len(hyps)), 2):
        n_sh, deg = rotation_disagreement(hyps[a], hyps[b])
        pa = np.asarray(hyps[a]["posed_full"])
        pb = np.asarray(hyps[b]["posed_full"])
        jac = float((pa & pb).sum()) / max(int((pa | pb).sum()), 1)
        for x, y in ((a, b), (b, a)):
            out[x].append(
                {
                    "vs_hypothesis": int(y),
                    "vs_serial": hyps[y].get("evo"),
                    "shared_frames": int(n_sh),
                    "rot_disagreement_deg": deg,
                    "posed_jaccard": jac,
                    "distinct": bool(deg is not None and deg > POSE_DISAGREE_DEG),
                }
            )
    agg = {}
    for k, rows in out.items():
        deg = [r["rot_disagreement_deg"] for r in rows if r["rot_disagreement_deg"]]
        agg[k] = {
            "n_peers": len(rows),
            "rot_disagreement_min": min(deg) if deg else None,
            "rot_disagreement_max": max(deg) if deg else None,
            "n_distinct": sum(1 for r in rows if r["distinct"]),
            "peers": rows,
        }
    return agg


def eval_image_source():
    """Read one of the capture's images as grayscale, by its relative path.

    The photometric witness is the one channel that consults the pixels rather
    than the arrays; everything else the battery reads is already in memory.
    A capture whose images cannot be opened records the channel unmeasurable."""
    try:
        import cv2
    except ImportError:
        return None
    ws = WS.resolve()

    def load(name):
        img = cv2.imread(str(ws / name), cv2.IMREAD_GRAYSCALE)
        return None if img is None else img.astype(np.float32)

    return load


def write_member_arrays(stage, members):
    """Ship each measured member's own arrays beside the release set.

    The battery's channels are aggregates; a selection pass that cuts frames
    out of a member has to measure what is left, and an aggregate taken before
    the cut cannot answer that.  So the arrays the battery was handed ride
    along in one compressed sidecar -- keypoints, shapes, structure, poses,
    membership -- and a later pass states the surviving core as a member and
    measures it.  Written beside the product, never into a release: no
    `.sfmr` gains a byte from this."""
    import zlib

    import seed_candidate_eval as EV

    blob = {}
    meta = {}
    for m in members:
        d = EV.member_arrays(m)
        key = f"m{d['idx']:04d}"
        meta[key] = {
            "idx": d["idx"],
            "model": d["model"],
            "camera": d["camera"],
            "f_eq": d["f_eq"],
        }
        blob[f"{key}__rvec"] = np.asarray(d["rvec"], np.float64)
        blob[f"{key}__tvec"] = np.asarray(d["tvec"], np.float64)
        blob[f"{key}__posed"] = np.asarray(d["posed"], bool)
        if d["pts"] is not None:
            blob[f"{key}__pts"] = np.asarray(d["pts"], np.float64)
        blob[f"{key}__obs_c"] = np.asarray(d["obs_c"], np.int32)
        blob[f"{key}__obs_i"] = np.asarray(d["obs_i"], np.int32)
        blob[f"{key}__obs_uv"] = np.asarray(d["obs_uv"], np.float64)
        if d["obs_f"] is not None:
            blob[f"{key}__obs_f"] = np.asarray(d["obs_f"], np.int32)
        if d["obs_shape"] is not None:
            blob[f"{key}__obs_shape"] = np.asarray(d["obs_shape"], np.float32)
        blob[f"{key}__keep"] = np.asarray(d["keep"], bool)
    names = list(members[0].names) if members else []
    blob["_meta"] = np.frombuffer(
        zlib.compress(json.dumps({"names": names, "members": meta}).encode("utf-8"), 6),
        dtype=np.uint8,
    )
    np.savez_compressed(Path(stage) / "member_arrays.npz", **blob)


def attach_evaluation(rung, hyps, data_full, f_vote):
    """Measure every committed member and attach the channels to its records.

    The manifest entry gains an ``evaluation`` block with the per-frame and
    per-held-out-image detail nested under it; the evolution record gains the
    same block, so the corpus and the product carry one story.  Instrumentation
    never kills the run it instruments: a failure is a one-line warning and an
    ``error`` field."""
    import seed_candidate_eval as EV

    by_idx = {}
    if not EV.eval_on():
        for h in rung.hypotheses:
            h["evaluation"] = EV.disabled_block()
        return
    t0 = elapsed()
    # THE WITNESS'S EVIDENCE: the capture's full pair graph where the coarse cut
    # dropped part of it, the solve's own admission where it did not.
    pair_obs = tuple(rung.eval_obs) or (
        data_full["obs_c"],
        data_full["obs_i"],
        data_full["obs_uv"],
    )
    rung.eval_obs = ()
    peers = peer_records(hyps)
    try:
        members = []
        for k, res in enumerate(hyps):
            d = res.get("release_data") or res["data"]
            members.append(
                EV.Member(
                    k,
                    "finite",
                    data_full["names"],
                    member_camera(res),
                    member_focal_eq(res),
                    res["rvec_full"],
                    res["tvec_full"],
                    res["posed_full"],
                    res.get("release_pts"),
                    (d["obs_c"], d["obs_i"], d["obs_uv"], d.get("obs_f")),
                    shapes=d.get("obs_shape"),
                )
            )
        # THE FAR-FIELD LAYERS, measured on their own model.  A rotation-only
        # member is a candidate like any other -- on a panorama capture it is
        # the RIGHT one -- so it is judged, not waved through.
        for layer in rung.far_layers if rung is not None else ():
            ld = layer["data"]
            members.append(
                EV.Member(
                    layer["idx"],
                    "rotation_only",
                    data_full["names"],
                    make_cam(layer["f"]),
                    layer["f"],
                    layer["rvec"],
                    layer["tvec"],
                    layer["posed"],
                    layer["dirs"],
                    (ld["obs_c"], ld["obs_i"], ld["obs_uv"], ld.get("obs_f")),
                    shapes=ld.get("obs_shape"),
                    keep=layer["keep"],
                )
            )
        # THE RELAXED MEMBERS, on the FINITE channels.  A relaxed member has
        # depth where its baselines priced one, so the questions a rotation
        # cannot answer are exactly the ones it now can; its remaining bearings
        # are stated as no finite position, so a channel that reads structure
        # reads only what the member actually placed.
        relaxed = []
        for rl in rung.relaxed_layers if rung is not None else ():
            r = rl["res"]
            d = r["data"]
            pts = np.asarray(r["release_pts"], dtype=np.float64).copy()
            pts[np.asarray(r["at_inf"], dtype=bool)] = np.nan
            relaxed.append(
                EV.Member(
                    rl["idx"],
                    "finite",
                    data_full["names"],
                    member_camera(r),
                    member_focal_eq(r),
                    r["rvec_full"],
                    r["tvec_full"],
                    r["posed_full"],
                    pts,
                    (d["obs_c"], d["obs_i"], d["obs_uv"], d.get("obs_f")),
                    shapes=d.get("obs_shape"),
                    keep=r["keep"],
                )
            )
        by_idx = EV.evaluate(
            members, f_vote, pair_obs=pair_obs, images=eval_image_source()
        )
        # Measured in a SECOND call, on the floors the first one drew.  The
        # hold-out gates are quantiles of the capture's own pooled per-frame
        # readings, so measuring the relaxed members in the same call would
        # re-draw those floors and move every member already measured; the
        # relaxed members are a later population of the same capture, and they
        # are conditioned on it rather than allowed to redefine it.
        if relaxed:
            by_idx.update(
                EV.evaluate(
                    relaxed,
                    f_vote,
                    pair_obs=pair_obs,
                    images=eval_image_source(),
                    floors=capture_floors(by_idx),
                )
            )
        # The arrays the battery was handed, beside the product: what a
        # selection pass needs to measure a core it cut rather than to
        # re-aggregate readings taken before the cut.
        write_member_arrays(rung.stage, members + relaxed)
    except Exception as exc:  # noqa: BLE001 — evaluation never kills the run
        print(f"  [candidate evaluation FAILED: {type(exc).__name__}: {exc}]")
        by_idx = {}
        for h in rung.hypotheses:
            h["evaluation"] = {
                "enabled": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return
    for h in rung.hypotheses:
        idx = int(h["idx"])
        block = by_idx.get(idx)
        if block is None:
            # A far-field layer: the geometry channels ask questions a model
            # with no depth cannot answer, and the focal one it can.
            block = {"enabled": True, "model": h.get("model", "rotation_only")}
            block["focal_vote"] = EV.focal_vote_channel(h.get("f"), f_vote)
            for name in EV.GEOMETRY_CHANNELS + EV.ROTATION_CHANNELS:
                block[name] = {
                    "measurable": False,
                    "unmeasurable_reason": "layer_not_measured",
                }
        if idx in peers:
            block["peer_corroboration"] = peers[idx]
        h["evaluation"] = _jsonable(block)
    # The same channels into the corpus, under the candidate they belong to.
    if evo_on():
        for k, res in enumerate(hyps):
            c = _EVO["cands"].get(res.get("evo"))
            if c is not None and k in by_idx:
                blk = dict(by_idx[k])
                blk["peer_corroboration"] = peers[k]
                c["evaluation"] = _jsonable(blk)
                c["peers"] = _jsonable(peers[k]["peers"])
    if rung is not None:
        rung.far_layers = []
        rung.relaxed_layers = []
    n_ok = sum(
        1
        for h in rung.hypotheses
        if (h.get("evaluation") or {}).get("self_resection", {}).get("n_measured")
    )
    n_rot = sum(
        1
        for h in rung.hypotheses
        if (h.get("evaluation") or {}).get("rot_self_resection", {}).get("n_measured")
    )
    print(
        f"candidate evaluation: {len(rung.hypotheses)} members measured "
        f"({n_ok} finite with a hold-out reading, {n_rot} rotation-only) "
        f"[{elapsed():.1f}s, +{elapsed() - t0:.1f}s]"
    )


def write_candidate_solves(rung, win, f_vote, n_votes):
    """Swap the rung's product into place at ``sfmr/candidate_solves/``.

    THE DIRECTORY IS THE PRODUCT: a manifest naming every committed hypothesis
    and one release per entry, self-contained, with no other file to read and no
    stamp in any path.  The stamp is a field inside the manifest, so a new run
    replaces the whole directory rather than accumulating beside it.

    ATOMIC by construction: the releases were written into a staging sibling as
    they were committed, the manifest joins them there, and only then is the
    destination removed and the staging directory renamed onto it.  A reader
    therefore sees the previous product, or nothing, or this one -- never a
    partial set, and never a manifest naming a release that is not there."""
    manifest = {
        "stamp": rung.stamp,
        "top_n": rung.n,
        "n_clusters_total": rung.n_clusters_total,
        "n_clusters_kept": rung.n_clusters_kept,
        "min_kept_radius_px": rung.min_kept_radius_px,
        # The vote block is the REFEREE's reading, so it reports the admission
        # the referee actually measured: the full one, which the two
        # `measured_on_*` fields name.
        "vote": {
            "f": None if f_vote is None else float(f_vote),
            "n_votes": None if n_votes is None else int(n_votes),
            "parallax_poverty": float(_VOTE_POVERTY),
            "measured_on_clusters": rung.vote_clusters,
            "measured_on_observations": rung.vote_observations,
        },
        "hypotheses": rung.hypotheses,
        # ADVISORY.  The ladder ranks, it does not decide: this names the entry
        # the rank put first, and nothing downstream is entitled to read it as
        # the answer.  The set is the product.
        "ladder_first": int(win),
        "elapsed_s": round(elapsed(), 3),
    }
    (rung.stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    dest = WS / "sfmr" / "candidate_solves"
    if dest.exists():
        shutil.rmtree(dest)
    rung.stage.replace(dest)
    return dest


def finite_priced_rows(rvec, tvec, pts, posed, f, obs_c, obs_i, u, rows):
    """Which of ``rows`` a FINITE geometry prices within 2 px.

    The cross-layer materiality number is built on this: a far-field row the
    finite solve also explains is a shared observation, and one it does not is
    evidence the near layer had no way to hold.  All of ``pts``, ``obs_c`` and
    the pose arrays must be stated in one cluster and image space."""
    out = np.zeros(len(rows), bool)
    if pts is None or not len(rows):
        return out
    ok = posed[obs_i[rows]] & np.isfinite(pts[obs_c[rows], 0])
    if not ok.any():
        return out
    r = np.nonzero(ok)[0]
    sel = rows[r]
    xc = (
        Rotation.from_rotvec(rvec[obs_i[sel]]).apply(pts[obs_c[sel]]) + tvec[obs_i[sel]]
    )
    proj = make_cam(float(f)).ray_to_pixel_batch(np.ascontiguousarray(xc))
    res = np.linalg.norm(proj - u[sel], axis=1)
    out[r] = np.isfinite(res) & (res < 2.0)
    return out


def sibling_priced_rows(sib, data, rows):
    """`finite_priced_rows` for a committed hypothesis's released geometry.

    ``data`` must be the selection the sibling solved on, so the two read one
    cluster space; a sibling with no retained structure prices nothing."""
    return finite_priced_rows(
        sib["rvec_full"],
        sib["tvec_full"],
        sib.get("release_pts"),
        sib["posed_full"],
        sib["f"],
        data["obs_c"],
        data["obs_i"],
        data["obs_uv"],
        rows,
    )


def rotation_only_hypothesis(
    rung,
    idx,
    data,
    f_rot,
    f_source,
    scope="capture",
    images=None,
    paired_with=None,
    sib=None,
):
    """Commit a ROTATION-ONLY (far-field) reading as hypothesis ``idx``, or
    return None when the rotation fit has nothing to say.

    Two scopes, one mechanism.  ``capture`` reads the whole admission and asks
    what the capture's far field is; ``group`` is restricted to one finite
    hypothesis's own frames (``images``) and its own working set (``data``),
    and asks the same question of that seed's neighbourhood.  Every set of
    images has a near-field reading and a far-field one, and rung 1's job is to
    commit both cheaply rather than to choose.

    A group layer is fit INDEPENDENTLY, from the pair evidence alone -- never
    derived from its sibling's rotations.  That independence is the whole of
    its later value: a far layer whose rotations disagree with the finite
    solve's is evidence against that finite basin, and a layer inherited from
    the basin it is meant to referee could never say so.

    The model is that the camera turned and did not travel: one rotation per
    frame, every camera center at the same point, and one point per admitted
    cluster the model EXPLAINS, stored at INFINITY as the world-frame direction
    of its reference ray.  No baseline is fabricated and no depth is claimed.

    This is not a rival reading of the whole capture so much as a reading of the
    part of it the finite hypotheses cannot price.  Anything far enough away has
    no measurable baseline no matter how far the camera walked -- horizon,
    skyline, distant ridge -- and a triangulating pass must either drop it or
    stamp it at a fictitious depth (BadlandPanorama's releases put a shell at
    depth ~6900 from ~0.2 deg of implied parallax).  A rotation-only pass prices
    exactly that layer, and prices it as bearing without range, which is what it
    is.  Membership is therefore SELF-SELECTING and must stay so: the points are
    the clusters the rotation model explains at the same 2 px bar the finite
    hypotheses are scored on, and the clusters it cannot explain are the near
    objects, which are the finite hypotheses' to hold.  The two layers share a
    rotation frame and the release's own track identity, so a later rung can
    correlate them rather than choose between them.

    The rotation skeleton is the one the rotation core already builds -- the
    same far-field ray-rotation edges and the same maximum-consensus spanning
    tree (`ray_rotation_edges`, `rotation_spanning_tree`) -- taken WITHOUT the
    metric half.  That half is why neither `rotation_core` nor its ray twin can
    serve here: both fix a gauge on an epipolar pair, triangulate, and grow
    translations, and where the near field is thin or absent they return None
    rather than the skeleton they already hold.  The skeleton needs no gauge
    pair, because a model with no baseline has no scale to fix, and it is rooted
    in its largest component rather than at a chosen frame.

    The focal is the capture-level vote's, not a scan's: the rotation model
    carries no depth for a focal scan to trade against, so the referee's number
    is the honest one and the manifest says where it came from.

    Metrics are the finite hypotheses': the same 2 px reprojection inlier
    fraction (over the member observations of the posed frames, under this
    model), the posed count, and coverage reach on the capture-level graph."""
    obs_c, obs_i, u = data["obs_c"], data["obs_i"], data["obs_uv"]
    n_img, n_cl = data["n_img"], data["n_cl"]
    # The frames the fit may pose.  A group layer sees only its sibling
    # attempt's frames; the capture layer sees the whole image table.
    fit_c, fit_i, fit_u = obs_c, obs_i, u
    if images is None:
        # The far field is a CAPTURE-WIDE layer, so the pair budget is stated
        # against the capture rather than at the rotation core's fixed 120: a
        # skeleton spanning n frames needs n - 1 surviving edges, and a graph
        # whose candidate pairs are concentrated in one neighbourhood chains one
        # neighbourhood (BadlandPanorama's top 120 pairs reached 7 of 370).
        max_pairs = max(120, 4 * n_img)
    else:
        allowed = np.zeros(n_img, bool)
        allowed[np.asarray(images, dtype=np.int64)] = True
        m = allowed[obs_i]
        fit_c, fit_i, fit_u = obs_c[m], obs_i[m], u[m]
        # Same rule, stated against the frames THIS fit may reach: the group's
        # working set carries evidence on a neighbourhood, not on the capture.
        max_pairs = max(120, 4 * int(len(np.unique(fit_i))))
    min_corr, min_inl = ray_rotation_floors(fit_c, fit_i, n_img, n_cl, max_pairs)
    found = ray_rotation_edges(
        fit_c,
        fit_i,
        fit_u,
        n_img,
        n_cl,
        f_rot,
        max_pairs=max_pairs,
        min_corr=min_corr,
        min_inliers=min_inl,
    )
    chained = None if found is None else rotation_spanning_tree(found[2], n_img)
    print(
        f"rotation-only skeleton ({scope}): {max_pairs} pair budget at floors "
        f"{min_corr} shared clusters / {min_inl} rotation inliers, "
        + ("no edges" if found is None else f"{len(found[2])} rotation edges")
        + f" [{elapsed():.1f}s]"
    )
    if chained is None:
        print(
            f"hypothesis {idx} (rotation-only, {scope}): the rotation fit "
            f"produced no skeleton; not committed [{elapsed():.1f}s]"
        )
        return None
    rvec, abs_rot = chained
    posed = np.zeros(n_img, bool)
    posed[list(abs_rot)] = True
    if int(posed.sum()) < 3:
        print(
            f"hypothesis {idx} (rotation-only, {scope}): the rotation fit posed "
            f"{int(posed.sum())} frames (< 3); not committed [{elapsed():.1f}s]"
        )
        return None
    # Poses: the fit's rotations, every camera center at the common origin.  A
    # rotation-only model has no baseline, so none is invented.
    tvec = np.zeros((n_img, 3))
    rot = Rotation.from_rotvec(rvec).as_matrix()
    cam = make_cam(f_rot)

    # One direction per cluster: the world-frame ray of its REFERENCE
    # observation (the first one a posed frame carries; the observation arrays
    # are cluster-major, so that is well defined and order-free).
    live = posed[obs_i]
    rows = np.nonzero(live)[0]
    d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(u[rows]))
    d_w = np.einsum("nji,nj->ni", rot[obs_i[rows]], d_loc)
    d_w /= np.maximum(np.linalg.norm(d_w, axis=1, keepdims=True), 1e-12)
    uniq, first = np.unique(obs_c[rows], return_index=True)
    dirs = np.full((n_cl, 3), np.nan)
    dirs[uniq] = d_w[first]

    # Reprojection under the rotation-only model: a direction is transported by
    # the rotation alone (translation cannot move a point at infinity).
    x_cam = np.einsum("nij,nj->ni", rot[obs_i[rows]], dirs[obs_c[rows]])
    proj = cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam))
    res_live = np.linalg.norm(proj - u[rows], axis=1)
    ok = np.isfinite(res_live)
    if not fisheye_stage1():
        # A direction on the far side of a pinhole camera reprojects into the
        # image as its own mirror image, at a residual that means nothing.  The
        # optical axis is taken FROM THE MODEL (the principal point's own ray)
        # rather than assumed to be a signed z, so this states cheirality in the
        # camera frame the context actually installed.
        axis = np.asarray(
            cam.pixel_to_ray_batch(
                np.ascontiguousarray([[_CAM_WH[0] / 2.0, _CAM_WH[1] / 2.0]], np.float64)
            )
        )[0]
        ok &= x_cam @ axis > 0
    res_live = np.where(ok, res_live, np.inf)
    res_obs = np.full(len(obs_c), np.inf)
    res_obs[rows] = res_live
    # MEMBERSHIP IS THE MODEL'S OWN INLIER SET, at the same 2 px bar every
    # hypothesis is scored on.  A finite pass trims at 4 px because it has just
    # optimized the structure it is trimming; nothing is optimized here, and a
    # looser bar would let near objects -- which are the finite hypotheses' to
    # hold, and which a rotation cannot place -- into the far-field layer.
    keep = np.zeros(len(obs_c), bool)
    keep[rows] = res_live < 2.0
    n_alive = int((np.bincount(obs_c[keep], minlength=n_cl) >= 2).sum())
    if not n_alive:
        print(
            f"hypothesis {idx} (rotation-only, {scope}): no cluster is explained "
            f"in two posed frames; not committed [{elapsed():.1f}s]"
        )
        return None

    inl = float((res_live < 2.0).sum() / max(len(res_live), 1))
    # The reference observation of each cluster fits by construction, so the
    # fraction over the REMAINING rows is the part of the number the model
    # actually earned.  Reported beside it rather than in place of it: the
    # manifest's `inlier_pct` stays the finite hypotheses' measure, over every
    # member observation of the posed frames.
    is_ref = np.zeros(len(rows), bool)
    is_ref[first] = True
    earned = res_live[~is_ref]
    inl_earned = float((earned < 2.0).mean()) if len(earned) else 0.0
    reach = capture_reach(np.nonzero(posed)[0])
    # CROSS-LAYER MATERIALITY: how much of what this layer prices its finite
    # sibling could not.  It is the number that says whether the far layer is
    # new evidence or a second reading of the same observations, and it is
    # reported rather than thresholded -- rung 1 commits the pair and lets the
    # measurement speak.
    extra = {"scope": scope}
    if paired_with is not None:
        extra["paired_with"] = int(paired_with)
    if sib is not None:
        extra["obs_beyond_sibling"] = int(
            ((res_live < 2.0) & ~sibling_priced_rows(sib, data, rows)).sum()
        )
    res = {
        "f_released": float(f_rot),
        "inl": inl,
        "posed": posed,
        "reach": float(reach),
        "flags": ["rotation_only"],
        "kept": int(posed.sum()),
        # No focal scan runs on this model (it has no depth to trade against
        # focal), so it declares no scan spread -- which is also why
        # `qualifies` reads it as unqualified, as the rotation-only member is.
        "spread": 0.0,
        # The frames this layer was restricted to, when it was: a group layer's
        # provenance is its sibling's frame set.
        "seed_frames": None if images is None else images,
        # Evidence channels the evolution corpus reads off the layer: its size,
        # the part of its inlier fraction it earned past the reference rays, and
        # the far field it prices that its finite sibling does not.
        "n_points_infinity": int(n_alive),
        "inl_earned": float(inl_earned),
        "far_rows": int(len(rows)),
        "obs_beyond_sibling": extra.get("obs_beyond_sibling"),
    }
    # THE LAYER'S OWN ARRAYS, for the evaluation battery.  A rotation-only
    # member is measured on its own model -- hold-out rotation resection, a
    # rotation-only witness, the full-form warp, the parallax it left behind --
    # and the battery needs the same arrays this fit produced.  Held on the
    # rung, never written: `commit_hypothesis` builds the manifest entry from
    # named fields, so nothing here reaches the product.  The RELAXATION rung
    # reads the same arrays, so the layer is held whenever either wants it.
    if (seed_eval_on() or relax_on()) and rung is not None:
        rung.far_layers.append(
            {
                "idx": int(idx),
                "data": data,
                "f": float(f_rot),
                "rvec": rvec,
                "tvec": tvec,
                "posed": posed,
                "dirs": dirs,
                "keep": keep,
            }
        )
    res["release_file"] = commit_hypothesis(
        rung,
        idx,
        res,
        lambda out: write_rotation_release(
            idx, data, f_rot, res_obs, keep, rvec, tvec, dirs, f_source, out
        ),
        model="rotation_only",
        f_source=f_source,
        extra=extra,
    )
    print(
        f"hypothesis {idx} committed: f {f_rot:.1f} px ({f_source}), "
        f"inlier<2px {100 * inl:.1f}%, {int(posed.sum())} posed, "
        f"reach {100 * reach:.0f}% (capture-level), "
        f"flags rotation_only, qualified False"
    )
    print(
        f"  rotation-only ({scope}"
        + ("" if paired_with is None else f", paired with h{paired_with}")
        + f"): {n_alive} points at infinity, {len(rows)} member "
        f"observations of which {int(is_ref.sum())} are reference rays "
        f"(exact by construction); inlier<2px over the rest "
        f"{100 * inl_earned:.1f}%"
        + (
            ""
            if "obs_beyond_sibling" not in extra
            else f"; {extra['obs_beyond_sibling']} priced observations its "
            f"sibling does not"
        )
        + f" [{elapsed():.1f}s]"
    )
    return res


def qualifies(res):
    """Whether a committed hypothesis clears the structure-trust gates the
    arbitration ranks on: the commit bar (posed count, coverage reach, focal
    observability), a release inside the corrected vote band, and no
    flat-scan / edge-scan / near-static-seed verdict."""
    if res["kept"] < 8 or res["reach"] < 0.60 or res["spread"] < 0.05:
        return False
    blocking = {
        "vote_divergence",
        "flat_scan",
        "edge_scan",
        "near_static_seed",
        "ladder_runner_up",
    }
    return not (blocking & set(res["flags"]))


def relative_rotation_disagreement(rv_a, pd_a, rv_b, pd_b):
    """``(shared posed images, median relative-rotation disagreement in
    degrees)`` between two rotation sets stated in one image frame, the
    disagreement None when they share fewer than two posed images.

    Compared through RELATIVE rotations only, which is what makes the
    comparison GAUGE-FREE without an alignment step: each solve fixes its own
    world frame, so absolute rotations are incommensurable, while the rotation
    between two frames both posed is a statement about the same geometry and is
    invariant to any global rotation applied to either side."""
    shared = np.nonzero(pd_a & pd_b)[0]
    if len(shared) < 2:
        return len(shared), None
    ra = Rotation.from_rotvec(rv_a[shared]).as_matrix()
    rb = Rotation.from_rotvec(rv_b[shared]).as_matrix()
    i, j = np.triu_indices(len(shared), 1)
    rel_a = np.einsum("nij,nkj->nik", ra[j], ra[i])
    rel_b = np.einsum("nij,nkj->nik", rb[j], rb[i])
    delta = Rotation.from_matrix(np.einsum("nji,njk->nik", rel_a, rel_b))
    return len(shared), float(np.degrees(np.median(delta.magnitude())))


def rotation_disagreement(a, b):
    """``(shared posed images, median relative-rotation disagreement in
    degrees)`` between two hypotheses.

    The poses are lifted into the loader's image frame, which every hypothesis
    in a chain shares, so the masks index the same images."""
    return relative_rotation_disagreement(
        a["rvec_full"], a["posed_full"], b["rvec_full"], b["posed_full"]
    )


def distinct(a, b):
    """Two hypotheses are DISTINCT when they share at least two posed images
    and disagree about the geometry there — median relative-rotation
    disagreement over the shared image pairs above the seed stage's own
    pose-noise scale.  Hypotheses with disjoint posed sets, or shared frames in
    agreement, are the same world seeded from different windows."""
    _, deg = rotation_disagreement(a, b)
    return deg is not None and deg > POSE_DISAGREE_DEG


def arbitrate(hyps):
    """``(shipping index, number qualified, any qualified pair distinct)``.

    The earliest qualified hypothesis is the INCUMBENT; a qualified challenger
    displaces it only when the two are distinct AND the challenger ranks higher
    (released inlier fraction, coverage reach as the tiebreak).  A non-distinct
    challenger never displaces a qualified incumbent, whatever its numbers —
    inlier fractions measured on different admissions of the same world reward
    the smaller solve.  With no qualifier at all the first hypothesis ships
    with its confidence flags (the single-hypothesis behavior, unchanged); when
    only a later one qualifies, it ships (rescue)."""
    ok = [i for i, h in enumerate(hyps) if qualifies(h)]
    if not ok:
        return 0, 0, False
    pairs = {}
    if len(ok) > 1:
        print()
    for a, b in itertools.combinations(ok, 2):
        n_shared, deg = rotation_disagreement(hyps[a], hyps[b])
        pairs[(a, b)] = deg is not None and deg > POSE_DISAGREE_DEG
        print(
            f"  distinctness h{a} vs h{b}: {n_shared} shared posed images, "
            + (
                "no pair to measure"
                if deg is None
                else f"median relative-rotation disagreement {deg:.1f} deg"
            )
            + f" -> {'distinct' if pairs[(a, b)] else 'same world'}"
        )
    win = ok[0]
    for c in ok[1:]:
        if pairs[(win, c)] and (hyps[c]["inl"], hyps[c]["reach"]) > (
            hyps[win]["inl"],
            hyps[win]["reach"],
        ):
            win = c
    return win, len(ok), any(pairs.values())


# ── Combination ──────────────────────────────────────────────────────────────
#
# Losing hypotheses are working capital.  Their point clouds are never merged
# across gauges — the prototype measured why: the clusters two same-world
# hypotheses share carry 0.2-6.6 deg of triangulation angle inside each window,
# a 12-dof affine fits them no better than a 7-dof similarity, and forcing the
# highest-link sibling in took one seed from 66.2% inliers to 43.7% and 22.7 deg
# of camera-rotation error.  There is no depth agreement to align because
# neither narrow window measures depth.  Their FRAMES are a different matter:
# they are viewpoints another exploration certified by POSING them, and
# resecting them into the winner's own structure needs no cross-gauge alignment
# at all (the prototype's arm C: 10 -> 43 posed at 66% -> 79% inliers with the
# camera errors against a full solve flat).

# The acceptance gate as a fraction of the base's OWN consensus — the widen
# ladder's rule (`gate = 0.35 * med_inl`), reused rather than restated, because
# a constant gate is what fails: 0.5 admits cleanly on a pinhole capture and
# over-admits into collapse (76.3% -> 11.3% inliers) on a fisheye one.
COMBINE_GATE = 0.35

# The ladder's pool floor: the observation count a frame needs against a
# structure before a resection into it is attempted at all (`widen`).  The
# combination reads it on BOTH sides — a candidate that cannot clear it toward
# a structure cannot resect there — so membership counting alone is the whole
# pre-filter on the bridge candidates, and no candidate costs a solve until it
# has cleared it twice.
POOL_FLOOR = 30

# How many bridge candidates one donor pair certifies.  Candidates are ranked
# by the WEAKER of their two membership counts (most bridge-like first) and the
# budget bounds the certification stage's cost at two resections each.
BRIDGE_ATTEMPTS = 48

# Significance of the one-sidedness test that reclassifies a pair DISTINCT.
# The test is the exact McNemar over the DISCORDANT certificates: a candidate
# that clears the floor toward both structures and resects into one but not the
# other is one discordant trial, and two windows of ONE world have no reason to
# put them all on the same side (p = 0.5 under that null, so the population
# floor is the test's own — six all-one-sided discordants are the first count
# that reaches this alpha two-sided).  It fires only when one side certified
# NOTHING; see `certify_bridges` for why a rate difference cannot.
BRIDGE_ALPHA = 0.05


def base_consensus(rvec, tvec, pts, posed, f, obs_c, obs_i, u):
    """The base's own resection consensus: the median, over its posed images,
    of the fraction of that image's observations of the base's structure
    reprojecting within 3 px.

    This is the quantity the widen ladder's ``med_inl`` is a median of (there,
    over the resections the core growth accepted), measured on a released state
    instead of a growing one — so the combination's gate scales with the base
    exactly as the ladder's does."""
    cam = make_cam(f)
    valid = ~np.isnan(pts[obs_c, 0])
    frac = []
    for j in np.nonzero(posed)[0]:
        s = (obs_i == j) & valid
        if int(s.sum()) < 12:
            continue
        res = reproj_res_one(cam, rvec[j], tvec[j], pts[obs_c[s]], u[s])
        frac.append(_inlier_fraction(res, 3.0))
    return float(np.median(frac)) if frac else 1.0


def _certificate(data, pts, j, f, gate):
    """Resect one frame against one hypothesis's structure, in that
    hypothesis's own gauge and at its own released focal.

    ``(passed, inlier fraction, rvec, tvec)`` from exactly the ladder's rung
    measurement — p3p, trimmed pose refine, 3 px consensus against the
    consensus-scaled gate — so a certificate and a rung are the same test made
    against different structure.  A frame carrying fewer than the pool floor of
    finite points, a p3p that finds no consensus, or a refined pose under the
    gate all fail."""
    order, bounds = image_rows(data)
    rows = order[bounds[j] : bounds[j + 1]]
    rows = rows[np.isfinite(pts[data["obs_c"][rows], 0])]
    if len(rows) < POOL_FLOOR:
        return False, 0.0, None, None
    uv, x = data["obs_uv"][rows], pts[data["obs_c"][rows]]
    p3p = p3p_resect(uv, x, f)
    if p3p is None or int(p3p[2].sum()) < 12:
        return False, 0.0, None, None
    rv0, tv0, mask = p3p
    rv, tv, _ = pose_refine(uv[mask], x[mask], rv0, tv0, f)
    frac = _inlier_fraction(reproj_res_one(make_cam(f), rv, tv, x, uv), 3.0)
    return bool(frac >= gate), float(frac), rv, tv


def camera_centres(rvec, tvec, idx):
    """``-R^T t`` for the given images, in the poses' own gauge."""
    rot = Rotation.from_rotvec(rvec[idx]).as_matrix()
    return -np.einsum("nji,nj->ni", rot, tvec[idx])


def certify_bridges(w, pts_w, gate_w, h, posed_any):
    """The BRIDGES of one (winner, donor) pair, and the pair's distinctness
    verdict from their certificates.

    A bridge is a frame posed by NEITHER hypothesis that is covisible with both
    retained cluster sets.  Candidacy is membership counting and nothing else:
    a frame must carry the ladder's pool floor of observations of each
    hypothesis's retained structure, because a frame that cannot clear the
    floor toward a structure cannot resect there whatever a solve would say.
    Candidates are then ranked by the WEAKER of the two counts — the most
    bridge-like first — and the strongest ``BRIDGE_ATTEMPTS`` of them are
    resected into BOTH structures, in each structure's own gauge and at its own
    focal.

    The roles are asymmetric.  The donor-side resection is a CERTIFICATE: it
    proves the frame genuinely views the donor's world, its donor-gauge pose
    orders the walk, and it is then discarded — the donor's depths never
    contribute a measurement.  That certificate is what puts the frame in the
    pool.  The winner-side resection is the LOAD-BEARING one and the walk makes
    it itself, against the current structure and through the ladder's own gate
    and per-rung verification; the copy taken here is a measurement for the
    test below, never a pose, because a candidate the released structure cannot
    reach yet is exactly what a rung of growth is for.

    The certificates are also the pair's second, stronger distinctness test,
    and the DONOR'S OWN FRAMES are in that population: a donor frame carries the
    donor-side certificate already (the donor posed it), so one that clears the
    winner-side floor is a candidate whose two certificates can disagree exactly
    like a bridge's.  Every member of the population cleared the floor toward
    both structures, so a member that resects into one and not the other is a
    DISCORDANT trial.  The verdict needs one side to certify NOTHING while the
    other certifies — systematic failure is TOTAL failure, because a frame that
    does not view a world resects into it never rather than rarely — with the
    discordants numerous enough for the one-sidedness to beat chance (exact
    McNemar, p = 0.5, two-sided at ``BRIDGE_ALPHA``, so six).  A rate
    DIFFERENCE cannot carry the verdict at any threshold: the two sides are not
    exchangeable trials, and the fleet measures the asymmetry a same-world pair
    produces on its own (a complement's thin structure certifies more readily
    than the full admission's rich one).

    Returns ``(bridges, donor-gauge centres, distinct, stats)``."""
    from scipy.stats import binom

    data_w, data_d = w["data"], h["data"]
    n_img = data_w["n_img"]
    f_w, f_d = float(w["f_released"]), float(h["f_released"])
    # The donor's own released geometry, retriangulated in the DONOR's gauge at
    # the DONOR's focal: the structure its certificates are measured against.
    pts_d = triangulate(
        data_d["obs_c"],
        data_d["obs_i"],
        data_d["obs_uv"],
        Rotation.from_rotvec(h["rvec_full"]).as_matrix(),
        h["tvec_full"],
        h["posed_full"],
        data_d["n_cl"],
        f_d,
    )
    med_d = base_consensus(
        h["rvec_full"],
        h["tvec_full"],
        pts_d,
        h["posed_full"],
        f_d,
        data_d["obs_c"],
        data_d["obs_i"],
        data_d["obs_uv"],
    )
    gate_d = COMBINE_GATE * med_d
    seen_w = np.bincount(
        data_w["obs_i"][np.isfinite(pts_w[data_w["obs_c"], 0])], minlength=n_img
    )
    seen_d = np.bincount(
        data_d["obs_i"][np.isfinite(pts_d[data_d["obs_c"], 0])], minlength=n_img
    )
    cand = np.nonzero((seen_w >= POOL_FLOOR) & (seen_d >= POOL_FLOOR) & ~posed_any)[0]
    n_cand = len(cand)
    cand = cand[np.argsort(-np.minimum(seen_w[cand], seen_d[cand]), kind="stable")]
    cand = cand[:BRIDGE_ATTEMPTS]

    bridges, centres = [], {}
    n_w = n_d = only_w = only_d = 0
    for j in cand:
        j = int(j)
        ok_w = _certificate(data_w, pts_w, j, f_w, gate_w)[0]
        ok_d, _, rv_d, tv_d = _certificate(data_d, pts_d, j, f_d, gate_d)
        n_w += ok_w
        n_d += ok_d
        only_w += ok_w and not ok_d
        only_d += ok_d and not ok_w
        if ok_d:
            # The DONOR-side certificate is what a bridge carries into the pool.
            # Its winner side is the load-bearing resection, and the walk makes
            # that one itself against the CURRENT structure — measured here for
            # the one-sidedness test above, but never spent: a candidate the
            # winner's released structure cannot reach yet is exactly what a
            # rung or two of growth is for.
            bridges.append(j)
            centres[j] = camera_centres(rv_d[None, :], tv_d[None, :], [0])[0]
    # The DONOR'S OWN FRAMES, on the same population.  Their donor-side
    # certificate is the donor's pose — a certificate the donor already issued —
    # so the winner side is all there is to measure, and a donor frame that
    # clears the winner-side floor and does not resect there is a discordant
    # trial of exactly the same kind as a bridge's.
    d_pool = np.nonzero(h["posed_full"] & ~w["posed_full"] & (seen_w >= POOL_FLOOR))[0]
    d_only_w = 0
    for j in d_pool:
        d_only_w += not _certificate(data_w, pts_w, int(j), f_w, gate_w)[0]
    # The population's certificates per side: a donor frame's donor-side one is
    # the donor's own pose, so it never fails there.
    pass_w = n_w + len(d_pool) - d_only_w
    pass_d = n_d + len(d_pool)
    only_d += d_only_w
    disc = only_w + only_d
    p = (
        1.0
        if not disc
        else float(min(1.0, 2 * binom.cdf(min(only_w, only_d), disc, 0.5)))
    )
    # SYSTEMATIC means TOTAL.  The two sides' certificates are not exchangeable
    # trials — the fleet measures a complement hypothesis's thin structure
    # certifying a candidate more readily than the full admission's rich one
    # (41/41 against 35/41 on one healthy same-world pair), because an inlier
    # fraction over a hundred points is a looser test than the same fraction
    # over thousands.  A rate DIFFERENCE therefore cannot carry this verdict at
    # any threshold.  Total failure can: a frame that does not view a world
    # resects into it never, not rarely.  So the verdict needs one side to
    # certify NOTHING while the other certifies, with the discordants numerous
    # enough for the one-sidedness itself to beat chance.
    one_sided = (pass_w == 0) != (pass_d == 0)
    stats = {
        "candidates": n_cand,
        "attempted": len(cand),
        "cert_w": n_w,
        "cert_d": n_d,
        "both": n_w - only_w,
        "bridges": len(bridges),
        "donor_frames": len(d_pool),
        "donor_frames_failing": d_only_w,
        "pass_w": pass_w,
        "pass_d": pass_d,
        "discordant_w": only_w,
        "discordant_d": only_d,
        "p": p,
        "gate_d": gate_d,
        "consensus_d": med_d,
    }
    return bridges, centres, one_sided and p < BRIDGE_ALPHA, stats


def combine(hyps, win):
    """Grow the winning hypothesis by resection over ONE pool of certified
    frames: the other committed non-distinct hypotheses' posed frames, plus the
    BRIDGES that weld them on.  Returns ``(release, reclassified)`` — the
    combined release or None when the stage added nothing (in which case it
    left no trace at all: no BA runs and the winner ships exactly as it was
    arbitrated), and the hypothesis indexes the certificates reclassified
    DISTINCT.

    Every frame posed by a NON-DISTINCT committed hypothesis and not by the
    winner is donor fuel — it carries its certificate already, since the donor
    posed it.  Distinct hypotheses' frames are never resected in, because they
    belong to another world.  A bridge is a frame posed by neither hypothesis
    that resects into both structures (`certify_bridges`), and it is what opens
    a donor window the winner's own structure never reaches: the walk is one
    growth loop, and the structure GROWS between rungs, so a bridge's clusters
    triangulate in and the donor frames behind it clear the floor a rung later.
    Frames covisible with neither hypothesis are not in the pool at all —
    welding committed hypotheses is this stage's job, growing the capture is
    the completion's.

    Frames the winner's own exploration blacklisted stay blacklisted: the pool
    is what another hypothesis certified, not the winner's own rejects.  The
    ladder's machinery does the work — resection into the current structure,
    the consensus-scaled gate, and the per-rung accept/BA/verify/revert — and
    the grown state is then retriangulated and released at the winner's own
    focal under the unchanged basin guard.  The winner must still qualify
    afterwards; it keeps the better of the two states otherwise."""
    global _RANK_O
    w = hyps[win]
    data_w = w["data"]
    obs_c, obs_i, u = data_w["obs_c"], data_w["obs_i"], data_w["obs_uv"]
    n_img, n_cl = data_w["n_img"], data_w["n_cl"]
    rank = data_w["adm_rank"][obs_c]
    _RANK_O = rank

    others = []
    for k, h in enumerate(hyps):
        if k == win:
            continue
        if distinct(w, h):
            print(
                f"combination: hypothesis {k} is DISTINCT from the winner; its "
                f"{int(h['posed_full'].sum())} frames belong to another world "
                f"and are not resected in"
            )
            continue
        others.append((k, h))
    donors = np.zeros(n_img, bool)
    for _, h in others:
        donors |= h["posed_full"]
    donors &= ~w["posed_full"]
    n_donor = int(donors.sum())
    if not n_donor:
        print(
            f"combination: no donor frames (0 frames posed by a non-distinct "
            f"committed hypothesis and not by hypothesis {win}); the winner "
            f"ships as arbitrated"
        )
        return None, []

    f0 = float(w["f_released"])
    rvec, tvec = w["rvec_full"].copy(), w["tvec_full"].copy()
    posed = w["posed_full"].copy()
    pts = triangulate(
        obs_c, obs_i, u, Rotation.from_rotvec(rvec).as_matrix(), tvec, posed, n_cl, f0
    )
    med_inl = base_consensus(rvec, tvec, pts, posed, f0, obs_c, obs_i, u)
    gate = COMBINE_GATE * med_inl
    # How many donors the ladder can even LOOK at: a candidate needs POOL_FLOOR
    # observations of the base's triangulated structure.  Two windows of one
    # world that never overlap leave this at zero — that is what the bridges
    # below are for, and it is worth reading in the log either way.
    seen = np.bincount(obs_i[donors[obs_i] & ~np.isnan(pts[obs_c, 0])], minlength=n_img)
    # The same count with the triangulation dropped: how many members of the
    # winner's ADMISSION a donor frame carries at all.  Growth can triangulate a
    # cluster the winner's structure is missing; it cannot admit one the
    # winner's selection never contained, so this is the ceiling the floor is
    # being measured against — and on a complement winner it is the whole story.
    members = np.bincount(obs_i, minlength=n_img)[donors]
    print(
        f"\ncombination: {n_donor} donor frames from {len(hyps) - 1} other "
        f"committed hypotheses, resecting into hypothesis {win}'s structure at "
        f"f {f0:.1f} px; base consensus {100 * med_inl:.1f}%, gate "
        f"{100 * gate:.1f}%; {int((seen[donors] >= POOL_FLOOR).sum())} donors "
        f"carry >= {POOL_FLOOR} observations of it (median "
        f"{int(np.median(seen[donors]))} of a median {int(np.median(members))} "
        f"members of the winner's ADMISSION — the ceiling growth can reach) "
        f"[{elapsed():.1f}s]"
    )

    # BRIDGES.  One certification pass per (winner, donor) pair, over the frames
    # neither hypothesis posed.
    posed_any = np.zeros(n_img, bool)
    for h in hyps:
        posed_any |= h["posed_full"]
    reclassified, ctx = [], []
    bridge_mask = np.zeros(n_img, bool)
    for k, h in others:
        bridges, centres, is_distinct, st = certify_bridges(w, pts, gate, h, posed_any)
        d_frames = h["posed_full"] & ~w["posed_full"]
        print(
            f"bridges h{k}: {st['candidates']} candidates carry >= {POOL_FLOOR} "
            f"observations of BOTH retained structures, {st['attempted']} "
            f"resected both ways (donor gate {100 * st['gate_d']:.1f}% of a "
            f"{100 * st['consensus_d']:.1f}% consensus); winner side "
            f"{st['cert_w']}, donor side {st['cert_d']}, both {st['both']}; "
            f"{st['donor_frames']} donor frames clear the winner-side floor, "
            f"{st['donor_frames_failing']} of them fail it; population "
            f"certificates {st['pass_w']} winner / {st['pass_d']} donor, "
            f"discordant {st['discordant_w']}/{st['discordant_d']} "
            f"(winner-only/donor-only), McNemar p {st['p']:.4f}"
        )
        if is_distinct:
            # SYSTEMATIC one-sided certificate failure.  The pair is two worlds
            # whatever the shared-frame rotation test said; the weld aborts for
            # it, its frames are withdrawn, and the verdict feeds the flag —
            # the arbitration is NOT re-run (the incumbent already won among
            # the qualified).
            reclassified.append(k)
            print(
                f"bridges h{k}: one side certified NOTHING and the other "
                f"certified, over {st['discordant_w'] + st['discordant_d']} "
                f"discordants (p < {BRIDGE_ALPHA}); hypothesis {k} is "
                f"RECLASSIFIED DISTINCT "
                f"from the winner — its {int(d_frames.sum())} frames are "
                f"withdrawn and its bridges are not welded"
            )
            continue
        # At most one bridge per donor frame: a bridge earns its rung by
        # opening the donor's window, and beyond the size of that window the
        # additions are capture growth, which is the completion's job.
        keep = bridges[: int(d_frames.sum())]
        bridge_mask[keep] = True
        idx = np.nonzero(d_frames)[0]
        ctx.append(
            {
                "frames": d_frames,
                "cen": dict(
                    zip(
                        (int(j) for j in idx),
                        camera_centres(h["rvec_full"], h["tvec_full"], idx),
                    )
                ),
                "bridge_centres": {j: centres[j] for j in keep},
            }
        )
    if reclassified:
        donors[:] = False
        for k, h in others:
            if k not in reclassified:
                donors |= h["posed_full"]
        donors &= ~w["posed_full"]
        n_donor = int(donors.sum())
    if not n_donor:
        print(
            f"combination: every donor pair was reclassified DISTINCT; the weld "
            f"is abandoned and the winner ships as arbitrated [{elapsed():.1f}s]"
        )
        return None, reclassified
    n_bridge = int(bridge_mask.sum())
    pool = donors | bridge_mask
    print(
        f"combination: pool {int(pool.sum())} frames = {n_donor} donor + "
        f"{n_bridge} bridge [{elapsed():.1f}s]"
    )

    def walk_order(pool_j, cnt, posed_now):
        """The ladder's rung ordering under the certificates: donor frames
        NEAREST an already-accepted bridge first (donor-gauge distance, the one
        thing the discarded donor-side pose is kept for), then the ladder's own
        farthest-first order over everything else."""
        key = np.full(len(pool_j), np.inf)
        for c in ctx:
            acc = [v for j, v in c["bridge_centres"].items() if posed_now[j]]
            if not acc:
                continue
            a = np.asarray(acc)
            for i, j in enumerate(pool_j):
                if c["frames"][j]:
                    d = float(np.linalg.norm(a - c["cen"][int(j)], axis=1).min())
                    key[i] = min(key[i], d)
        near = np.isfinite(key)
        rest = pool_j[~near]
        return np.concatenate(
            [
                pool_j[near][np.argsort(key[near], kind="stable")],
                rest[np.argsort(cnt[rest], kind="stable")],
            ]
        )

    rvec, tvec, pts, posed, accepted = widen(
        rvec,
        tvec,
        pts,
        posed,
        f0,
        obs_c,
        obs_i,
        u,
        n_img,
        n_cl,
        rank,
        gate,
        allow=pool,
        rungs=int(pool.sum()),
        # No bridge, no reordering: an entry the certificates found nothing for
        # walks exactly the ladder it walked before.
        order=walk_order if n_bridge else None,
    )
    if not accepted:
        # The ladder reverts a rejected rung whole, so nothing moved: no
        # release runs and the winner is untouched.
        print(
            f"combination: no pool frame cleared the gate; the winner ships "
            f"as arbitrated [{elapsed():.1f}s]"
        )
        return None, reclassified
    n_add_b = int(sum(1 for j in accepted if bridge_mask[j]))
    # What the growth did to the donors' reach: the same count the pool floor
    # tests, re-measured on the structure the walk left behind.  A weld that
    # accepted bridges and no donor frame says so here.
    left = donors & ~posed
    seen_after = np.bincount(
        obs_i[left[obs_i] & ~np.isnan(pts[obs_c, 0])], minlength=n_img
    )
    print(
        f"combination: resected {len(accepted)}/{int(pool.sum())} pool frames "
        f"({len(accepted) - n_add_b} donor, {n_add_b} bridge); posed "
        f"{int(w['posed_full'].sum())} -> {int(posed.sum())}; the "
        f"{int(left.sum())} donor frames left out now carry a median "
        f"{int(np.median(seen_after[left])) if left.any() else 0} observations "
        f"of the grown structure (was {int(np.median(seen[donors]))}) "
        f"[{elapsed():.1f}s]"
    )

    # Retriangulate and release at the winner's focal, under the same iterated
    # schedule and the same basin guard the exploration releases with (anchored
    # on the winner's released focal, which is what the combination grew from).
    bam = budget_mask(posed, obs_i, rank)
    denom = ba_rows(bam, obs_i)
    pts = triangulate(
        obs_c, obs_i, u, Rotation.from_rotvec(rvec).as_matrix(), tvec, posed, n_cl, f0
    )
    live = ba_rows(bam & ~np.isnan(pts[obs_c, 0]), obs_i)
    _, rvec, tvec, pts, res, _ = bundle_adjust(
        obs_c[live],
        obs_i[live],
        u[live],
        rvec,
        tvec,
        pts,
        f0,
        n_img,
        n_cl,
        opt_f=False,
        max_nfev=60,
    )
    inl = float((res < 2.0).sum() / max(int(denom.sum()), 1))
    # The fixed-f state at the winner's own released focal is the baseline the
    # free-f release has to beat, so a release that walks out of the basin
    # costs the combination its focal, not its frames.
    kept_state = (inl, f0, rvec.copy(), tvec.copy(), pts.copy())
    f, f_prev = f0, f0
    for _ in range(3):
        live = ba_rows(bam & ~np.isnan(pts[obs_c, 0]), obs_i)
        f, rvec, tvec, pts, res, _ = bundle_adjust(
            obs_c[live],
            obs_i[live],
            u[live],
            rvec,
            tvec,
            pts,
            f,
            n_img,
            n_cl,
            opt_f=True,
            max_nfev=30,
        )
        inl = float((res < 2.0).sum() / max(int(denom.sum()), 1))
        if not (f0 / 1.15 <= f <= 1.15 * f0) or f < focal_floor():
            print(f"combined release left the basin (f = {f:.0f}); keeping previous")
            break
        if inl > kept_state[0]:
            kept_state = (inl, f, rvec.copy(), tvec.copy(), pts.copy())
        if abs(f - f_prev) < 0.01 * f_prev:
            break
        f_prev = f
    inl, f, rvec, tvec, pts = kept_state

    # The gates, re-measured on the combined release.  Posed count and reach
    # can only grow; the scan verdicts are properties of the focal scan the
    # exploration ran and carry over unchanged; the consensus and the
    # vote-divergence guard are re-derived from the new release.
    kept = int(posed.sum())
    reach = capture_reach(np.nonzero(posed)[0])
    f_bias = 1.0 if fisheye_stage1() else 1.1
    f_indep = w["f_indep"]
    flags = ["low_consensus"] if inl < 0.60 else []
    if f_indep is not None and abs(np.log(f / (f_bias * f_indep))) > np.log(1.35):
        flags.append("vote_divergence")
    if reach < 0.30:
        flags.append("narrow_reach")
    flags += [t for t in ("flat_scan", "edge_scan") if t in w["flags"]]
    combined = {
        "kept": kept,
        "reach": reach,
        "spread": w["spread"],
        "inl": inl,
        "flags": flags,
    }
    gained = set(flags) - set(w["flags"])
    print(
        f"combination: released f {f:.1f} px, inlier<2px {100 * inl:.1f}% "
        f"(was {100 * w['inl']:.1f}%), reach {100 * reach:.0f}% (was "
        f"{100 * w['reach']:.0f}%), flags {','.join(flags) or 'ok'}, "
        f"qualified {qualifies(combined)} [{elapsed():.1f}s]"
    )
    # KEEP-BEST.  The combination is an addition to a state the arbitration
    # already chose, so it has to leave that state no worse on the gates: a
    # combined release that stops qualifying (where the winner did qualify), or
    # that picks up a confidence flag the winner did not carry, is reverted
    # whole and the arbitrated winner ships.  A flag is a gate here — a
    # collapse that keeps f in its basin shows up as `low_consensus` and
    # nowhere else, and that is exactly the failure the prototype measured
    # (76.3% -> 11.3% inliers on an over-admitting gate).
    lost_qual = qualifies(w) and not qualifies(combined)
    if lost_qual or gained:
        print(
            "combination REVERTED (keep-best): the combined release "
            + (
                f"raised {','.join(sorted(gained))}"
                if gained
                else "no longer clears the qualification gates"
            )
            + "; the winner ships as arbitrated"
        )
        return None, reclassified
    f_report = f_indep if (flags and f_indep is not None) else f
    return {
        "names": list(data_w["names"]),
        "rvec": rvec,
        "tvec": tvec,
        "posed": posed,
        "f": float(f),
        "f_report": float(f_report),
        "f_released": float(f),
        "inl": float(inl),
        "flags": flags,
        "n_donor": n_donor,
        "n_bridge": n_bridge,
        "n_added": len(accepted),
        "n_added_bridge": n_add_b,
    }, reclassified


# ── Main ─────────────────────────────────────────────────────────────────────


def explore(hyp, args):
    """One exploration pass over one admitted selection: EVERY distinct
    candidate its ladder generated, rank-ordered.

    Empty when the admission produced no reconstruction at all (only possible
    past the first pass -- the first keeps the fatal behavior, since a capture
    whose full admission cannot seed has no seed stage at all).  A candidate
    that posed no image is dropped: that is the exploration's starved fallback,
    not a reconstruction."""
    got = run_pipeline(*args, hyp=hyp, required=hyp == 0)
    if not got:
        return []
    out = [r for r in got if int(r["posed"].sum())]
    for r in got:
        if not int(r["posed"].sum()):
            evo_reason(r.get("evo"), "no_posed_image")
    if not out:
        print(f"pass {hyp}: no candidate posed an image")
    return out


def main():
    global _CAM_WH, _RANK_O, _SEL_MATCHES
    data, rung = load_clusters()
    # The FULL admission, held past the loop's complement rebinds: the
    # rotation-only hypothesis below is a reading of the whole capture, not of
    # whatever the last complement left.
    data_full = data
    obs_c, obs_i, u = data["obs_c"], data["obs_i"], data["obs_uv"]
    n_img, n_cl = data["n_img"], data["n_cl"]
    _CAM_WH = tuple(data["dims"][0])
    _RANK_O = data["adm_rank"][obs_c]
    print(
        f"{WS}: {n_img} images, {n_cl} clusters "
        f"(<= {MAX_CL} per BA window), "
        f"{len(obs_c)} observations [{elapsed():.1f}s]"
    )

    # Probe growth at a nominal focal: the core is near-affine, so growth
    # succeeds across a wide f range and its only job here is to build the
    # core geometry and resolve the reflection hypothesis (the two grown
    # candidates' inlier fractions).  The focal itself is chosen later, on
    # geometry wide enough to observe it.
    #
    # Seed groups are tried in covisibility order with a PARALLAX gate: a
    # video's most-mutually-covisible frames are where the camera moved
    # least, and a near-static core fits any focal at high inlier fraction
    # while its depths are unusable (DinoLedge seeded on a static clip).
    if os.environ.get("SFMTOOL_F0"):  # debug: skip the vote stage
        vote = (float(os.environ["SFMTOOL_F0"]), 0)
    else:
        # Vote on the FULL admission, solve on the restricted one: the referee
        # keeps the whole capture's pair graph even when the rung hands the
        # solve a few thousand clusters (`vote_admission`).
        vote_c, vote_i, vote_u = (
            rung.vote_admission(obs_c, obs_i, u)
            if rung is not None
            else (obs_c, obs_i, u)
        )
        if rung is not None and vote_c is not obs_c:
            print(
                f"focal vote measured on the full admission "
                f"({rung.vote_clusters} clusters, {len(vote_c)} "
                f"observations), not the restricted {n_cl}"
            )
            rung.vote_observations = int(len(vote_c))
        vote = focal_vote(vote_c, vote_i, vote_u, n_img)
        # The referee has spoken; the arrays are not needed again, and a big
        # capture's full admission is not worth carrying through the loop.
        # THE ONE EXCEPTION is the evaluation battery's two-view witness, which
        # needs the capture's whole pair graph rather than the coarse cut the
        # solve explores: a held-out image and a member image share a few tens
        # of top-N clusters and hundreds of full-admission ones, and an
        # epipolar estimate on the former is not an estimate.  Held under the
        # battery's own switch and released the moment it has read them.
        if rung is not None:
            rung.eval_obs = rung.vote_obs if seed_eval_on() else ()
            rung.vote_obs = ()
    if vote is not None:
        f_vote, n_votes = vote
        # Bougnoux votes from noisy F run consistently LOW on this campaign
        # (-1..-10%, one -22%): probe above the vote and skew the scan grid
        # upward so the true focal stays inside it.
        f_probe = 1.1 * f_vote
        f_grid = np.array([0.8, 0.95, 1.1, 1.3, 1.55]) * f_vote
        print(
            f"pairwise focal vote: f ~ {f_vote:.1f} ({n_votes} votes) "
            f"[{elapsed():.1f}s]"
        )
    else:
        f_vote = None
        f_probe = 0.9 * max(_CAM_WH)
        f_grid = np.asarray(F_GRID) * max(_CAM_WH)
        print(f"no focal vote (sparse pairs); probing at {f_probe:.1f}")
    if fisheye_stage1():
        # Under a confirmed, opted-in fisheye verdict stage 1 probes at the
        # verdict's EQUIDISTANT focal.  The pinhole vote (and its +10% bias
        # correction, and the grid centred on it) parameterizes
        # r = f*tan(theta) and has no meaning for a theta = r/f map; probing
        # there is what made the Phase-1 gated runs mixed geometry.  The scan
        # grid is the log-symmetric equidistant one about that verdict, inside
        # the FOV-derived band.
        f_probe = float(_VOTE_FISHEYE["focal_px"])
        f_grid = fisheye_focal_grid(f_probe)
        lo, hi = fisheye_focal_band()
        print(
            f"fisheye seed: probing at the verdict's equidistant focal "
            f"{f_probe:.1f} px; scan grid "
            f"[{', '.join(f'{v:.1f}' for v in f_grid)}] px "
            f"inside the FOV band [{lo:.1f}, {hi:.1f}]"
        )
    cap = min(n_img, SCAN_CAP)

    # The hypothesis loop.  Explore the admission; claim the committed
    # hypothesis's coverage, and while the claim leaves real evidence
    # unexplained (or the hypothesis is untrusted and the capture is owed a
    # rescue look), admit the complement and explore again
    # (`specs/core/geometry/seed-hypothesis-loop.md`).  The
    # capture-level vote above is measured ONCE over the full admission's pair
    # graph and read by every hypothesis — it is the independent referee the
    # arbitration measures each release against, so no hypothesis re-derives it
    # from its own restricted pair graph.
    # THE CANDIDATE SET: every distinct finalist of every pass, in commit order.
    hyps = []
    npass = 0
    claims = {}
    handle = _SEL_MATCHES
    rescue_spent = False
    # The generator is the RUNG's.  The legacy full-seed path keeps the
    # single-winner loop it was made with, materiality stop and all: it feeds a
    # finalization that wants one seed, not a set.
    label = "pass" if rung is not None else "hypothesis"
    # The capture-level covisibility graph, built ONCE from the full admission
    # (as the vote above is measured once over the full pair graph), so every
    # hypothesis's coverage reach is measured against the same capture.
    capture_covisibility(obs_c, obs_i, n_img, n_cl)
    while True:
        if len(hyps) >= CANDIDATE_BUDGET:
            print(
                f"\ncandidate budget reached ({CANDIDATE_BUDGET} finite "
                f"candidates); the generator stops [{elapsed():.1f}s]"
            )
            break
        if npass:
            print(f"\n=== {label} {npass}: exploring the coverage complement ===")
        args = (obs_c, obs_i, u, data, n_img, n_cl, cap, f_probe, f_vote, f_grid, rung)
        if npass == 0:
            got = explore(npass, args)
        else:
            # A complement admission is a THIN, unvetted working set: too few
            # clusters to factorize, a window that degenerates mid-solve.  Any
            # way it fails to produce ends the generator — a failed later pass
            # must never cost the capture the candidates it already has.
            try:
                got = explore(npass, args)
            except Exception as exc:  # noqa: BLE001 — see above
                print(f"{label} {npass} failed: {type(exc).__name__}: {exc}")
                got = []
        if not got:
            print(
                f"{label} {npass}: the complement produced no reconstruction; "
                f"the capture's cluster evidence is exhausted [{elapsed():.1f}s]"
            )
            break
        npass += 1
        # EVERY candidate this pass generated commits.  The ladder's rank rides
        # along as metadata and decides nothing; what used to be a winner and a
        # runner-up is just the first two entries of a list.
        first = None
        for n_got, res in enumerate(got if rung is not None else got[:1]):
            if len(hyps) >= CANDIDATE_BUDGET:
                print(
                    f"candidate budget reached; {len(got) - len(hyps)} further "
                    f"candidates from this pass are not released"
                )
                for r in got[n_got:]:
                    evo_reason(r.get("evo"), "budget_overflow")
                break
            idx = len(hyps)
            res["handle"] = handle
            # The release lifted into the LOADER's image frame — the one frame
            # every candidate in the chain shares.  It is what the artifact
            # writes and what the distinctness test compares.
            res["rvec_full"], res["tvec_full"], res["posed_full"] = _snap_full_frame(
                data, res["keep"], res["rvec"], res["tvec"], res["posed"], res["f"]
            )
            res["data"] = data
            hyps.append(res)
            first = first or res
            print(
                (
                    f"candidate {idx} committed (ladder rank {res['ladder_rank']}): "
                    if rung is not None
                    else f"hypothesis {idx} committed: "
                )
                + f"f {res['f_released']:.1f} px, "
                f"inlier<2px {100 * res['inl']:.1f}%, {int(res['posed'].sum())} posed, "
                f"reach {100 * res['reach']:.0f}% (capture-level; own admission "
                f"{100 * res['reach_pass']:.0f}%), "
                f"flags {','.join(res['flags']) or 'ok'}, "
                f"qualified {qualifies(res)}"
            )
            evo_link(res.get("evo"), idx)
            evo_note(
                res.get("evo"),
                committed=True,
                f_released=res["f_released"],
                f_report=res["f_report"],
                f_structure=res["f"],
                f_vote=res["f_vote"],
                f_indep=res["f_indep"],
                f_center=res["f_center"],
                f_band=res["f_band"],
                vote_divergence_log=(
                    None
                    if not res["f_center"]
                    else float(np.log(res["f_released"] / res["f_center"]))
                ),
                inlier_2px=res["inl"],
                posed=int(res["posed"].sum()),
                posed_frames=[int(k) for k in np.nonzero(res["posed_full"])[0]],
                reach=res["reach"],
                reach_pass=res["reach_pass"],
                spread=res["spread"],
                flags=list(res["flags"]),
                qualified=bool(qualifies(res)),
                ladder_rank=res.get("ladder_rank", 0),
                ladder_collapsed=res.get("collapsed", 0),
                claim_clusters=(
                    None
                    if res.get("claim_pts") is None
                    else int(np.isfinite(res["claim_pts"][:, 0]).sum())
                ),
                claim_clusters_total=int(data["n_cl"]),
                release_clusters=(
                    None
                    if res.get("release_pts") is None
                    else int(np.isfinite(res["release_pts"][:, 0]).sum())
                ),
                local_admission=res.get("local_admission"),
            )
            d_rel = res.get("release_data") or data
            rel = commit_hypothesis(
                rung,
                idx,
                res,
                lambda out, i=idx, r=res, dd=d_rel: write_finite_release(i, r, dd, out),
            )
            evo_note(
                res.get("evo"),
                release_file=None if rel is None else Path(rel).name,
            )
        if first is None:
            break
        # THE CLAIM ORDERS THE QUEUE, it does not gate it.  Every candidate this
        # pass produced stamps its footprint, so the next admission starts where
        # the least has been explained; nothing about the claim can stop the
        # generator, and a complement that looks immaterial is explored anyway.
        # Sharp selection was wrong in every direction we tried it, and a
        # winner's claim used to gate exploration, so a wrong winner did not
        # merely mis-rank the set -- it impoverished it (20240618_001255975~2
        # lost its +1.3% member to a claim stamped by the wrong candidate).
        for res in got if rung is not None else got[:1]:
            if res.get("claim_pts") is not None:
                claim_coverage(res, data, claims)
            res.pop("claim_pts", None)
        survivors, n_claimed = unclaimed_clusters(data, claims)
        n_cl = data["n_cl"]
        print(
            f"coverage complement: {n_claimed}/{len(data['obs_c'])} members in "
            f"claimed cells; clusters {n_cl} -> {len(survivors)} "
            f"({100 * len(survivors) / max(n_cl, 1):.1f}% retained) "
            f"[{elapsed():.1f}s]"
        )
        if not len(survivors) or len(survivors) == n_cl:
            # Nothing left, or nothing claimed at all — either way the
            # complement is not a new admission and the generator is done.
            print("the complement is not a new admission; the generator stops")
            break
        if rung is None:
            # MATERIALITY, the legacy stop.  A complement that is most of the
            # admission again means the committed hypothesis's structure barely
            # overlaps the evidence pool, and exploring it enumerates
            # independently-seedable frame WINDOWS of the same world.  The
            # exception is rescue: an untrusted hypothesis's claim is not
            # evidence about the rest of the capture, and it is spent when used,
            # because a chain of untrusted hypotheses has no brake otherwise.
            # The rung retires all of it -- a claim there orders the queue and
            # can no longer veto a group.
            qual = qualifies(first)
            retention = len(survivors) / max(n_cl, 1)
            material = retention < MATERIAL_RETENTION
            rescue = not qual and not rescue_spent
            if material:
                decision = f"material (< {100 * MATERIAL_RETENTION:.0f}%) — exploring"
            elif rescue:
                decision = (
                    f"immaterial but hypothesis {npass - 1} does not qualify — "
                    f"exploring (rescue, the capture's one look)"
                )
            elif not qual:
                decision = (
                    f"immaterial and hypothesis {npass - 1} does not qualify, but "
                    f"the rescue look is spent — the loop stops here"
                )
            else:
                decision = (
                    f"immaterial (>= {100 * MATERIAL_RETENTION:.0f}% retained "
                    f"under a qualified hypothesis) — the loop stops here"
                )
            print(f"materiality: {decision} [{elapsed():.1f}s]")
            if not (material or rescue):
                break
            rescue_spent = rescue_spent or rescue
        handle, data = complement_selection(handle, survivors)
        obs_c, obs_i, u = data["obs_c"], data["obs_i"], data["obs_uv"]
        n_cl = data["n_cl"]
        _RANK_O = data["adm_rank"][obs_c]

    # THE FAR-FIELD LAYERS.  The exploration above is finite by construction:
    # every pass triangulates, so whatever a set of images holds beyond the
    # reach of its own baseline is either dropped or stamped at a fictitious
    # depth (BadlandPanorama, parallax-poverty 0.998: a shell at depth ~6900
    # from ~0.2 deg of implied parallax, and not one point at infinity).  The
    # rotation-only reading is already in the run as the vote's second family
    # and as a seeding pathway; here it becomes a COMMITTED member, so the
    # observations the finite passes cannot price are priced by the model that
    # can -- bearing without range.
    #
    # EVERY SET OF IMAGES HAS BOTH READINGS.  Each committed finite hypothesis
    # gets a far-field sibling on its own frames and its own working set, and
    # the capture gets one over the whole admission.  Neither is a rival of the
    # near layer: on most captures the finite hypotheses hold the near objects
    # and these hold the horizon, in a shared rotation frame, and a later rung
    # merges them.  Rung 1's job is to commit both cheaply, not to choose.
    #
    # Unconditional, because the far field is a LAYER OF MOST SCENES rather
    # than a degenerate-capture fallback.  A trigger on parallax poverty made
    # every low-poverty capture silent about its far field, when the honest
    # answer there is a small layer, and a small layer measured is a fleet-wide
    # far-field mass census that a skipped one is not.
    #
    # A group layer is fit INDEPENDENTLY of its sibling's rotations: its later
    # value is as a referee of that finite basin, and a layer inherited from the
    # basin it referees could never indict it.
    #
    # All of them land after the complement is exhausted and before the
    # combination and the arbitration; none stamps a coverage claim, donates or
    # receives frames, or enters the ranking (qualification stays a
    # finite-structure verdict).  They join the set AFTER the finite
    # hypotheses, which are exactly what they were.
    # Every finite candidate, in commit order, for the far layers to pair with.
    layered = list(enumerate(hyps))
    if rung is not None:
        # The vote's focal, in the SOLVE's own parameterization: the pinhole
        # pool for a pinhole run, the equidistant verdict under a fisheye
        # context.  The rotation model has no depth for a scan to trade against
        # focal, so the referee's number is the honest one here.
        if fisheye_stage1() or f_vote is None:
            f_rot, f_src = f_probe, ("vote" if fisheye_stage1() else "probe")
        else:
            f_rot, f_src = f_vote, "vote"
        far_idx = len(layered)
        for k, h in layered:
            print(
                f"\n=== hypothesis {far_idx}: the far-field layer of h{k} "
                f"({int(h['posed'].sum())} posed frames) ==="
            )
            built = rotation_only_hypothesis(
                rung,
                far_idx,
                h.get("release_data") or data_full,
                float(f_rot),
                f_src,
                scope="group",
                images=h["keep"],
                paired_with=k,
                sib=h,
            )
            if built is not None:
                evo_copy_stage(
                    h.get("evo"),
                    "far-field",
                    built.get("release_file"),
                    hypothesis_index=far_idx,
                    scope="group",
                    f=float(f_rot),
                    f_source=f_src,
                    n_posed=int(built["posed"].sum()),
                    n_points=built["n_points_infinity"],
                    inlier_2px=built["inl"],
                    inlier_2px_earned=built["inl_earned"],
                    reach=built["reach"],
                    far_rows=built["far_rows"],
                    obs_beyond_sibling=built["obs_beyond_sibling"],
                    obs_beyond_sibling_frac=(
                        None
                        if not built["far_rows"] or built["obs_beyond_sibling"] is None
                        else built["obs_beyond_sibling"] / built["far_rows"]
                    ),
                )
            far_idx += built is not None
        print(
            f"\n=== hypothesis {far_idx}: the capture's far-field layer "
            f"(parallax-poverty {_VOTE_POVERTY:.2f}) ==="
        )
        cap_far = rotation_only_hypothesis(
            rung, far_idx, data_full, float(f_rot), f_src
        )
        if cap_far is not None and evo_on():
            # The capture's own far field is a reading of no single candidate,
            # so it gets a record of its own rather than a stage of someone's.
            s_cap = evo_candidate(
                "capture_rotation",
                scope="capture",
                parallax_poverty=float(_VOTE_POVERTY),
            )
            evo_link(s_cap, far_idx)
            evo_copy_stage(
                s_cap,
                "capture-hrot",
                cap_far.get("release_file"),
                scope="capture",
                f=float(f_rot),
                f_source=f_src,
                n_posed=int(cap_far["posed"].sum()),
                n_points=cap_far["n_points_infinity"],
                inlier_2px=cap_far["inl"],
                inlier_2px_earned=cap_far["inl_earned"],
                reach=cap_far["reach"],
                far_rows=cap_far["far_rows"],
            )

    # THE RELAXATION RUNG.  Every rotation-only member the phase above
    # committed is relaxed into a finite sibling and committed beside it: the
    # rows its model refused carry the baselines its model has no place for, so
    # the same evidence prices camera centres and depths.  It lands here, as
    # rung 1's LAST exploration phase, because it needs the far layers' arrays
    # (which the battery clears) and the pre-restriction handle (which the
    # restriction stage replaces), and because a member measured before the
    # relaxation would be measured on rotations the relaxation moves.
    if rung is not None and relax_on():
        relax_far_layers(rung, data_full)

    # THE RANK.  Under the rung this is all the arbitration is: the recorded
    # order of the set, qualified first and commit order otherwise.  The
    # distinctness test and the combination that read it are statements ABOUT
    # THE SET -- which committed hypotheses are one world and how they merge --
    # and the set is not finished until rung 1 has shipped it, so both belong to
    # whatever reads the product, not to the pass that writes it.  The legacy
    # full-seed path still needs a single winner grown over its rivals before
    # the finalization sees it, so it keeps both, untouched.
    if rung is not None:
        win = next(
            (k for k, h in enumerate(hyps) if qualifies(h)),
            0,
        )
        n_qual = sum(1 for h in hyps if qualifies(h))
        any_distinct = False
    else:
        win, n_qual, any_distinct = arbitrate(hyps)
    res = hyps[win]
    # The winner's own admission is what the restriction stage narrows: cluster
    # ids downstream are the winning hypothesis's selection's ids.
    _SEL_MATCHES = res["handle"]
    if len(hyps) > 1:
        print(
            f"arbitration: {len(hyps)} hypotheses committed, {n_qual} "
            f"qualified, "
            + ("a distinct pair" if any_distinct else "no distinct pair")
            + f"; hypothesis {win} ships"
            + ("" if n_qual else " (no qualifier — the first hypothesis ships)")
        )
    # COMBINATION.  The losing hypotheses' frames are certified-seedable
    # viewpoints of the winner's own world, so the winner grows over them before
    # anything downstream sees it: the combined posed set is what the
    # restriction stage narrows to and what the seed dict carries.  A capture
    # with no donor frame — every single-hypothesis capture, and any capture
    # whose other hypotheses are distinct — leaves the stage a strict no-op.
    comb, reclassified = (None, []) if rung is not None else combine(hyps, win)
    # DISTINCTNESS FEEDBACK.  The combination's cross-resection certificates are
    # the second, stronger source of the arbitration's distinctness verdict: a
    # pair whose linking frames systematically resect into one structure but not
    # the other is two worlds, whatever the shared-frame rotation test said.  The
    # verdict withdrew the donor's frames above; here it feeds the flag.  It
    # never re-runs the arbitration — the incumbent already won among the
    # qualified, and a reclassification only says the loser was a different
    # world, not a better one.
    if reclassified and qualifies(res):
        newly = [k for k in reclassified if qualifies(hyps[k])]
        if newly and not any_distinct:
            print(
                f"arbitration: the combination's certificates reclassify "
                f"{', '.join(f'h{k}' for k in newly)} DISTINCT from the winner; "
                f"the capture carries more than one rigid world"
            )
        any_distinct = any_distinct or bool(newly)
    f_released = res["f_released"]
    if comb is None:
        names = res["names"]
        rvec, tvec, posed = res["rvec"], res["tvec"], res["posed"]
        f, f_report, inl, flags = res["f"], res["f_report"], res["inl"], res["flags"]
    else:
        # The combined release is stated in the LOADER's image frame (the
        # winner's working set was a thinned subset of it), so the names, poses
        # and posed mask all come off the combination.
        names = comb["names"]
        rvec, tvec, posed = comb["rvec"], comb["tvec"], comb["posed"]
        f, f_report, inl, flags = (
            comb["f"],
            comb["f_report"],
            comb["inl"],
            comb["flags"],
        )
        f_released = comb["f_released"]
    compare_to_reference(names, rvec, tvec, f_report, posed)
    # The seed stage's data product is the finalized reconstruction, NOT a JSON.
    # Build the seed in memory and run the photometric finalization in-process as
    # the mandatory terminal step (embed-patches expand + congeal + consensus
    # bitmaps -> drop length-2 -> native BA), writing sfmr/seed-final.sfmr.  No
    # fast-pinhole.json is ever written; the stage-1 focal estimates and
    # confidence flags are recorded in the sfmr's tool_options metadata instead.
    # Camera-model verdict (escalation only; None on every pinhole capture).
    # Without the opt-in it is a CONFIDENCE flag only — the seed above was
    # produced by the pinhole pipeline on the pinhole vote, and the flag marks
    # the reconstruction as one that pipeline cannot model.  With the opt-in
    # (fisheye_stage1()) the seed IS equidistant and the flag records which
    # model produced it.
    if _VOTE_FISHEYE is not None:
        flags = [*flags, "fisheye_detected"]
    # Two or more hypotheses clearing the structure-trust gates AND disagreeing
    # about the frames they both pose means the capture's cluster evidence
    # supports more than one rigid world: the unchosen one is real structure,
    # not noise, and the flag says so.  Qualified-but-not-distinct is one world
    # seeded from two windows, which the flag must not claim.
    if n_qual >= 2 and any_distinct:
        flags = [*flags, "multiple_hypotheses"]
    # Hypothesis records, present ONLY when the loop committed more than one —
    # a single-hypothesis capture's metadata is byte-identical to a run without
    # the loop.
    hyp_opts = {}
    if len(hyps) > 1:
        hyp_opts["hypothesis_count"] = str(len(hyps))
        hyp_opts["hypothesis_winner"] = str(win)
        for i, h in enumerate(hyps):
            hyp_opts[f"hypothesis_{i}"] = (
                f"focal_released_px={h['f_released']:.3f},"
                f"inlier_fraction={h['inl']:.4f},"
                f"posed_count={int(h['posed'].sum())},"
                f"flags={'|'.join(h['flags']) or 'ok'}"
            )
    # THE PRODUCT.  Everything above this line is stage 1: the focal vote, the
    # probe/widen/verify/release attempts, the hypothesis loop, the far-field
    # layers and the rank.  Everything below is the finalization — the
    # restriction stage's matches artifact, the photometric embed/congeal, and
    # sfmr/seed-final.sfmr — which belongs to the legacy full-seed path.
    #
    # An armed rung stops here BY DEFINITION: exploration ends when the basin
    # structure is on disk.  It ships `sfmr/candidate_solves/`, and that
    # directory IS the product: the whole set, its manifest, nothing else to
    # read, no stamp in any path.
    if rung is not None:
        # THE EVIDENCE, before the product goes out: every committed member
        # measured by the evaluation battery, attached to its own manifest
        # entry (and to its corpus record), so what ranks and refuses the set
        # later does not have to re-derive any of it.  Peer corroboration --
        # how far each candidate stands from its rivals where they overlap --
        # rides in the same block.
        attach_evaluation(rung, hyps, data_full, f_vote)
        path = write_candidate_solves(
            rung, win, f_vote, None if vote is None else vote[1]
        )
        n_rel = sum(1 for h in rung.hypotheses if h["release_file"] is not None)
        print(
            f"\nRUNG 1 COMPLETE (top {rung.n} clusters): "
            f"{len(rung.hypotheses)} hypotheses committed, {n_qual} qualified, "
            f"{n_rel} released, {rung.memo_hits} probes memoized; "
            f"the rank puts h{win} first "
            f"(the set is the product, nothing is discarded); "
            f"{path.relative_to(WS).as_posix()} [{elapsed():.1f}s]"
        )
        evo_write(
            stamp=rung.stamp,
            top_n=rung.n,
            n_images=int(data_full["n_img"]),
            n_clusters=int(data_full["n_cl"]),
            n_clusters_total=rung.n_clusters_total,
            n_clusters_kept=rung.n_clusters_kept,
            min_kept_radius_px=rung.min_kept_radius_px,
            image_names=[Path(str(s)).name for s in data_full["names"]],
            image_dims=[[int(w), int(h)] for w, h in data_full["dims"]],
            candidate_budget=CANDIDATE_BUDGET,
            ladder_first=int(win),
            n_qualified=int(n_qual),
            vote_f=None if f_vote is None else float(f_vote),
            vote_n=None if vote is None else int(vote[1]),
            vote_parallax_poverty=float(_VOTE_POVERTY),
            vote_rotation_votes=int(_VOTE_ROT_N),
            vote_spread_log=float(_VOTE_SPREAD),
            fisheye_verdict=(
                None
                if _VOTE_FISHEYE is None
                else {
                    k: v
                    for k, v in _VOTE_FISHEYE.items()
                    if isinstance(v, (int, float, str))
                }
            ),
            fisheye_stage1=bool(fisheye_stage1()),
            probe_focal=float(f_probe),
            elapsed_s=round(elapsed(), 3),
        )
        return
    seed = {
        "focal_structure_px": float(f),
        # The structure-free focal the finalization arbitrates against — in
        # the SOLVE's parameterization, so a fisheye seed carries the
        # equidistant verdict here rather than the (incommensurable) pinhole
        # vote it would otherwise hand a fisheye camera.
        "focal_vote_px": res["f_indep"],
        # The same measurement bias-corrected, and its own measured precision
        # (log-focal half-width): the finalization refuses a structure candidate
        # that this independent measurement contradicts beyond its own band.
        "focal_vote_center_px": res["f_center"],
        "focal_vote_band_log": res["f_band"],
        "posed_images": [n for j, n in enumerate(names) if posed[j]],
        "rvec": rvec[posed].tolist(),
        "tvec": tvec[posed].tolist(),
        "confidence_flags": flags,
    }
    _write_seed_sfmr(seed, f_report, f_vote, inl, f_released, hyp_opts)


# ── Cluster restriction stage ────────────────────────────────────────────────
#
# Stage 1 explores over the whole admitted cluster selection; the seed it
# commits to spans a handful of images.  Narrowing the clusters to those images
# is a STAGE with a file artifact — matches/seed-restricted.matches — and
# everything after it reads that file.  Cluster ids downstream are the
# restricted file's own; no stage past this one names a cluster of the
# workspace's matches file.

RESTRICTED_MATCHES = "matches/seed-restricted.matches"


def restrict_clusters(posed_images):
    """Run the restriction stage: derive stage 1's cluster selection again on
    the seed images alone and write it as the stage artifact.  Returns the
    path."""
    import exp_pinhole_bootstrap as B

    sel = _SEL_MATCHES
    names = list(sel.image_names)
    wanted = set(posed_images)
    keep = np.array([n in wanted for n in names], bool)
    restricted = sel.select_clusters(
        min_span=B.MIN_SPAN_BA,
        restrict_images=[n for j, n in enumerate(names) if keep[j]],
    )
    out_path = WS / RESTRICTED_MATCHES
    out_path.parent.mkdir(parents=True, exist_ok=True)
    restricted.save(str(out_path))
    starts_in = np.asarray(sel.cluster_starts, dtype=np.int64)
    starts_out = np.asarray(restricted.cluster_starts, dtype=np.int64)
    print(
        f"cluster restriction: {len(starts_in) - 1} -> {len(starts_out) - 1} "
        f"clusters on {int(keep.sum())}/{len(names)} seed images "
        f"-> {out_path.name}"
    )
    return out_path


def _write_seed_sfmr(seed, f_report, f_vote, inl, f_released, hyp_opts=None):
    """Finalize the in-memory seed into sfmr/seed-final.sfmr — the seed stage's
    terminal, mandatory step.  Runs the bootstrap's photometric finalization
    in-process (no JSON is written); the confidence flags and the stage-1 focal
    estimates are recorded in the sfmr's tool_options metadata."""
    from sfmtool._sfmtool.io import MatchesFile

    # The stage-1 camera context reaches the finalization through
    # `bootstrap_module` (a no-op restating of the pinhole default on every
    # capture without a confirmed, opted-in fisheye verdict).
    B = bootstrap_module()
    # Cluster admission is decoupled from image selection: the restriction
    # stage narrows the stage-1 selection to the seed's posed images, so span
    # and the admission population are computed over the seed-image population
    # instead of the whole matches file.  It writes the restricted cluster set
    # as the stage artifact and everything below reads THAT — cluster ids from
    # here on are the restricted file's own.
    restricted_path = restrict_clusters(seed["posed_images"])
    data = B.load_clusters(matches_data=MatchesFile(restricted_path), preselected=True)
    final = B.finalize_seed_from_dict(data, seed)
    out_path = WS / "sfmr" / "seed-final.sfmr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        "confidence_flags": ",".join(seed["confidence_flags"]) or "ok",
        "focal_report_px": f"{f_report:.3f}",
        "focal_vote_px": "none" if f_vote is None else f"{f_vote:.3f}",
        "stage1_inlier_fraction": f"{inl:.4f}",
    }
    # Hypothesis-loop records.  Empty (the keys absent entirely) unless more
    # than one hypothesis committed, so a single-hypothesis capture writes the
    # metadata it wrote before the loop existed.
    opts.update(hyp_opts or {})
    # Escalated camera-model verdict.  These keys describe the DETECTION, so
    # they appear only on a fisheye verdict and keep their fisheye names.
    # Under the opt-in every focal here parameterizes the EQUIDISTANT map
    # (theta = r/f); without it focal_report_px / focal_vote_px stay pinhole
    # and fisheye_focal_equidistant_px is the one equidistant number.
    if _VOTE_FISHEYE is not None:
        opts["camera_model_verdict"] = "EquidistantFisheye"
        opts["fisheye_focal_equidistant_px"] = (
            "none"
            if _VOTE_FISHEYE["focal_px"] is None
            else f"{_VOTE_FISHEYE['focal_px']:.3f}"
        )
        # The RELEASED stage-1 focal, recorded whether or not a confidence
        # flag sent focal_report_px back to the structure-free verdict — the
        # release is the measurement Phase 3b exists to produce.
        opts["focal_released_px"] = f"{f_released:.3f}"
        opts["fisheye_verdict_margin"] = f"{_VOTE_FISHEYE['margin']:.2f}"
        opts["fisheye_verdict_mass"] = (
            f"pinhole={_VOTE_FISHEYE['mass_pinhole']},"
            f"equidistant={_VOTE_FISHEYE['mass_equidistant']}"
            f",equidistant_epipolar={_VOTE_FISHEYE['mass_epipolar']}"
            f",equidistant_rotation={_VOTE_FISHEYE['mass_rotation']}"
        )
        opts["fisheye_escalation_trigger"] = _VOTE_FISHEYE["trigger"]
    # The finalization's spline rung: the promoted model, the shipped radial
    # spline and its domain.  The rung runs on EITHER base and the record is
    # named for the spline, not for the base -- `bspline_d_max` is the end of
    # the spline's domain in whichever radial coordinate the base measures,
    # radians of incidence angle under the equidistant one and the normalized
    # image-plane radius rho = tan(theta) under the pinhole one
    # (specs/formats/sfmtool-camera-models.md); `camera_model_final` is the
    # promoted model, which says which.
    #
    # The rung promotes on a REFUSAL too, at the pre-rung focal with an
    # all-zero spline, so these keys record a refusal as naturally as a
    # release: an all-zero `bspline` IS the refusal, and it is the base map
    # bit for bit.  All three go absent only where nothing promoted the
    # context at all -- the SFMTOOL_BSPLINE_RUNG=0 kill switch, or a capture
    # with no spline domain to promote onto -- and the camera table then
    # carries the base model, exactly as it did before the rung existed.
    cam_ctx = B.camera_context()
    promoted = {
        "SFMTOOL_FISHEYE": "SfmtoolFisheye",
        "SFMTOOL_PINHOLE": "SfmtoolPinhole",
    }.get(cam_ctx["model"])
    if promoted is not None:
        opts["camera_model_final"] = promoted
        opts["bspline"] = ",".join(
            f"{c:.8f}" for c in np.asarray(cam_ctx["bspline"], dtype=np.float64)
        )
        opts["bspline_d_max"] = f"{cam_ctx['theta_max']:.6f}"
    final.save(str(out_path), operation="seed-finalized", tool_options=opts)
    print(
        f"\nwrote {out_path} ({final.point_count} points, seed-finalized "
        f"w/ bitmaps; no JSON written)"
    )
    # Timestamped round copy: every run leaves a byte-identical snapshot under
    # sfmr/seed-rounds/, so successive rounds accumulate and `sfm compare` can
    # diagnose any two of them.  A fleet runner exports one SFMTOOL_ROUND_STAMP
    # for the whole invocation so all its datasets share one round identity;
    # a bare run stamps itself.  Copy, not re-save: a second save would
    # recompute content_xxh128 and the round artifact would no longer be the
    # file the canonical path holds.
    import shutil

    round_path = WS / "sfmr" / "seed-rounds" / f"{round_stamp()}-seed-final.sfmr"
    round_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, round_path)
    print(f"round snapshot: {round_path}")


def run_pipeline(
    obs_c,
    obs_i,
    u,
    data,
    n_img,
    n_cl,
    cap,
    f_probe,
    f_vote,
    f_grid,
    rung,
    hyp=0,
    required=True,
):
    """One full seed-exploration -> focal-scan -> release pass.

    Returns the released estimate (poses, focal, released inlier fraction,
    flags, and the working set it was measured on) for ``main`` to arbitrate.

    ``hyp`` is the hypothesis index the pass belongs to; it only tags the
    snapshots, so hypothesis 0's checkpoints keep the names they always had.
    ``required`` is the exploration's contract: the first hypothesis MUST
    produce a reconstruction (a capture whose full admission cannot seed has no
    seed stage), while a complement that explores nothing simply returns None
    and ends the loop.
    """
    global _RANK_O
    _SNAP["tag"] = "nosel" + (f"-h{hyp}" if hyp else "")
    _SNAP["n_affine"] = 0
    _SNAP["n_attempt"] = 0
    # The pass index, bound here because ``hyp`` is rebound by a loop inside
    # ``attempt`` and the evolution records are stated per pass.
    pass_idx = int(hyp)

    # The structure-free focal measurement COMMENSURABLE with what this pass
    # solves, plus the bias correction that turns it into the comparator the
    # scan tiebreak, the divergence guard and the flagged fallback all read.
    # Pinhole: the pairwise vote, which runs ~10% low from noisy F.
    # Equidistant: the verdict's own focal — same map as the solve, and no
    # measured directional bias, so it is used raw.  The pinhole vote is NOT a
    # fallback under a fisheye context; the two parameterize different maps.
    if fisheye_stage1():
        f_indep, f_bias = float(_VOTE_FISHEYE["focal_px"]), 1.0
        vote_iqr = float(_VOTE_FISHEYE.get("pool_spread") or 0.0)
    else:
        f_indep, f_bias = f_vote, 1.1
        vote_iqr = _VOTE_SPREAD
    # The structure-free measurement AND ITS PRECISION, for the finalization's
    # vote-vs-structure arbitration (see `_finalize_seed`'s contradiction test).
    # `f_center` is the measurement in the solve's parameterization, bias
    # corrected; `f_band` is a multiplicative half-width in log-focal.
    f_center = None if f_indep is None else f_bias * float(f_indep)
    f_band = None if f_center is None else max(vote_iqr, VOTE_BAND_FLOOR_LOG)

    def attempt(wk, nw, keep, rc_only=False, allow_rc=True):
        """One working set through probe -> widen -> photometric verify,
        arbitrating SEED CHUNKS by their widened outcome.

        rc_only runs just the rotation-core hypothesis (the parallax-poor
        fast path: covis factorization chunks are what starve there, and
        the native-level ones are the most expensive stage of the whole
        pipeline).  allow_rc=False skips rotation-core entirely (thinned
        working sets have had their redundancy — the H-edge fuel —
        deliberately removed).

        Ladder-widen for focal observability: a sliver of a long orbit fits
        ANY focal at high inlier fraction (bas-relief compensation), so both
        the focal scan and the release only mean something on a wide arc.
        The photometric pass localizes anchors (valid points, >= 3 posed
        views, best stored warp-consistency first) in every posed view; an
        image that keeps too little photometric support is a junk rung the
        geometric gates missed — un-pose it before the focal scan.  The
        verified keypoints themselves stay OUT of the estimation: the
        localization renders through the current geometry, so they carry a
        bias toward the probe focal (seoul's scan moved from 336 to 432 when
        fed them) — appearance VERIFIES, geometry decides.

        A chunk whose core probes with parallax can still starve the widen
        ladder (the widen gate scales with the core's own consensus, so a
        tight core can reject every rung another chunk's core accepts) —
        so a starved widen tries the next seed chunk before the caller
        escalates thinning.
        """
        o_c, o_i, o_u, o_f, nms, rank = wk
        # The set the CURRENT try solves on.  It is the attempt's own working
        # set everywhere except inside a group-local re-admission (stage B),
        # which swaps in a denser set derived for that group's images; every
        # stage below reads it from here so the swap is one assignment rather
        # than a parameter threaded through the probe, the widen and the
        # verify.  `wk` itself stays the GLOBAL (stage-A) working set, which is
        # what the coverage claim and the complement are stated in.
        cur = {
            "wk": wk,
            "n_cl": n_cl,
            "data": data,
            "seed": None,
            "nbr_key": None,
            # The evolution serial of the candidate the current try is
            # developing, so the stages downstream of a probe attribute
            # themselves to it without threading it through every call.
            "evo": None,
        }
        # This level's image restriction, as a key: it is the other half of a
        # working set's identity beside the re-admitted neighbourhood.
        keep_key = tuple(int(x) for x in keep)

        def use_workset(w=None, ncl=None, d=None, seed=None):
            """Point the current try at a working set (the attempt's own when
            called with nothing), including the per-observation admission rank
            the BA row budget reads off a module global."""
            global _RANK_O
            cur["wk"] = wk if w is None else w
            cur["n_cl"] = n_cl if ncl is None else ncl
            cur["data"] = data if d is None else d
            cur["evo"] = None
            # The seed group this try grew from, in the DATA image frame: the
            # committed hypothesis records where its geometry came from.
            cur["seed"] = seed
            _RANK_O = cur["wk"][5]

        def snap_affine(imgs, used, rot, scale, t_aff, pts_m, span2, hyp):
            # Bind the working set's index frames (`keep` maps working image
            # index -> data image index; `o_u` holds the working observations).
            snapshot_affine(
                cur["data"],
                keep,
                cur["wk"][2],
                cur["n_cl"],
                imgs,
                used,
                rot,
                scale,
                t_aff,
                pts_m,
                span2,
                hyp,
            )

        def finish(cand, near_static=False):
            _, par, rvec, tvec, pts, posed, med_inl = cand
            # The current try's set (the attempt's own, or a group-local one).
            o_c, o_i, o_u, o_f, nms, rank = cur["wk"]
            n_cl_w, data_w = cur["n_cl"], cur["data"]
            att_n = _SNAP["n_attempt"]
            _SNAP["n_attempt"] += 1
            seed_snap(
                f"01-probe-{_SNAP['tag']}-{att_n:02d}",
                data_w,
                keep,
                f_probe,
                rvec,
                tvec,
                pts,
                posed,
            )
            print(f"probe done (parallax {par:.2f} deg) [{elapsed():.1f}s]; widening")
            rvec, tvec, pts, posed, _accepted = widen(
                rvec,
                tvec,
                pts,
                posed,
                f_probe,
                o_c,
                o_i,
                o_u,
                nw,
                n_cl_w,
                rank,
                gate=0.35 * med_inl,
            )
            print(f"widened to {int(posed.sum())}/{nw} images [{elapsed():.1f}s]")
            evo_stage(
                cur["evo"],
                "post-widen",
                data_w,
                keep,
                f_probe,
                rvec,
                tvec,
                pts,
                posed,
                wk=cur["wk"],
                widen_gate=0.35 * med_inl,
                probe_med_inl=med_inl,
            )
            seed_snap(
                f"02-widen-{_SNAP['tag']}-{att_n:02d}",
                data_w,
                keep,
                f_probe,
                rvec,
                tvec,
                pts,
                posed,
            )
            # PHOTOMETRIC VERIFY.  The anchors are localized in every posed
            # view and a frame that keeps too little support is un-posed: an
            # image the geometric gates admitted on depth-noisy structure is
            # a junk rung the focal scan must not read.  The verified
            # keypoints themselves stay OUT of the estimation -- the
            # localization renders through the current geometry, so they
            # carry a bias toward the probe focal.  Appearance VERIFIES,
            # geometry decides.  It is also the rung's largest cost bucket,
            # which is why it is a mode rather than a constant.
            if rung is None or rung.verify == "full":
                pv = np.bincount(o_c[posed[o_i]], minlength=n_cl_w)
                cand_a = np.nonzero(~np.isnan(pts[:, 0]) & (pv >= 3))[0]
                cand_a = cand_a[np.argsort(data_w["cl_quality"][cand_a], kind="stable")]
                anchors = cand_a[:N_ANCHORS]
                a_of_cl = np.full(n_cl_w, -1, np.int64)
                a_of_cl[anchors] = np.arange(len(anchors))
                sub = np.nonzero(posed)[0]
                rows = np.nonzero((a_of_cl[o_c] >= 0) & posed[o_i])[0]
                ai = a_of_cl[o_c[rows]]
                order = np.argsort(ai, kind="stable")
                rows, ai = rows[order], ai[order]
                sub_of_full = np.full(nw, -1, np.int64)
                sub_of_full[sub] = np.arange(len(sub))
                ph_a, a_i, ph_uv = localize_anchors(
                    nms,
                    sub,
                    rvec,
                    tvec,
                    f_probe,
                    pts[anchors],
                    ai,
                    sub_of_full[o_i[rows]],
                    o_f[rows],
                )
                kept_per_img = np.bincount(a_i, minlength=nw)
                floor = max(15, 0.2 * np.median(kept_per_img[sub]))
                for j in sub:
                    if kept_per_img[j] < floor:
                        posed[j] = False
                        print(
                            f"  un-posing image {j}: {kept_per_img[j]} verified obs "
                            f"(floor {floor:.0f})"
                        )
                print(
                    f"photometric verify kept {int(posed.sum())}/{len(sub)} images "
                    f"[{elapsed():.1f}s]"
                )
                evo_stage(
                    cur["evo"],
                    "post-verify",
                    data_w,
                    keep,
                    f_probe,
                    rvec,
                    tvec,
                    pts,
                    posed,
                    wk=cur["wk"],
                    verify_floor=float(floor),
                )
            seed_snap(
                f"03-verify-{_SNAP['tag']}-{att_n:02d}",
                data_w,
                keep,
                f_probe,
                rvec,
                tvec,
                pts,
                posed,
            )
            pi = np.nonzero(posed)[0]
            return {
                "wk": cur["wk"],
                # The attempt's GLOBAL working set, kept beside the solved one:
                # the coverage claim and the complement are stated in stage A's
                # cluster space whatever the solve ran on.  The same object
                # when no re-admission happened.
                "wk_global": wk,
                "n_cl": n_cl_w,
                "data": data_w,
                "nw": nw,
                "keep": keep,
                "rvec": rvec,
                "tvec": tvec,
                "pts": pts,
                "posed": posed,
                "kept": int(posed.sum()),
                # Exploration reach (this pass's graph) and QUALIFICATION reach
                # (the capture's).  The exploration compares working sets of
                # one admission; the arbitration compares hypotheses across
                # admissions, and only the capture-level graph is common to
                # them.
                "reach": float(reach_of(keep[pi])),
                "reach_capture": float(capture_reach_of(keep[pi])),
                # Whether this outcome's SEED was one the near-static gate
                # rejected and the attempt finished anyway once nothing else
                # survived.  The gate's verdict travels with the outcome, so
                # the ladder can rank on it and the release can flag it.
                "near_static_seed": bool(near_static),
                "seed_frames": cur["seed"],
                # The evolution record this outcome develops.
                "evo": cur["evo"],
            }

        covis = build_covisibility(o_c, o_i, nw, n_cl)
        groups = list(itertools.islice(covis.seed_groups(), 8))
        best = None  # best starved widened outcome across chunks
        flat = None  # best reach-healthy but focal-blind outcome
        # Deferred PROBES carry the working set they were probed on: under a
        # group-local re-admission two chunks are two cluster spaces, and a
        # probe's pts/posed arrays only mean anything in their own.  (Deferred
        # OUTCOMES already carry theirs on the outcome dict.)
        low_par = None  # max-parallax probe fallback if no chunk clears 1 deg
        gated = []  # probes rejected as unmeasurable (last-resort fallback)
        probe_max = 0.0  # running max group-chunk probe consensus this attempt
        tried_rc = False

        def workset():
            return (cur["wk"], cur["n_cl"], cur["data"])

        def finish_on(held, near_static=False):
            cand, ws, serial = held
            use_workset(*ws)
            cur["evo"] = serial
            return finish(cand, near_static=near_static)

        def committable(outcome):
            """The full commit bar — posed count, coverage reach AND focal
            observability, identical to the level loop's break bar.

            Posed count and reach say a window GREW; they say nothing about
            whether its geometry can read focal, and a flat-scan window
            commits the whole pipeline to a focal the structure has no
            opinion about (20240614_224422531 committed a 0.0pp-spread
            window while observable alternatives were never tried).  The
            spread is measured here and kept on the outcome, so the level
            loop reuses the measurement instead of repeating it."""
            if outcome["kept"] < 8 or outcome["reach"] < 0.60:
                return False
            outcome["spread"] = scan_spread(outcome)
            print(
                f"outcome scan spread {100 * outcome['spread']:.1f}pp "
                f"[{elapsed():.1f}s]"
            )
            return outcome["spread"] >= 0.05

        def defer(outcome, nxt):
            """Below-bar outcome: report it and keep it as a fallback.

            Reach-healthy but focal-blind outcomes are held apart from
            starved ones and arbitrated by the level loop's own comparator
            (spread first, coverage as tiebreak); a widened, verified,
            reach-healthy solve outranks any starved one, so the flat slot
            wins over the starved slot at the end of the attempt."""
            nonlocal best, flat
            if "spread" in outcome:
                print(
                    f"scan flat ({100 * outcome['spread']:.1f}pp); focal not "
                    f"observable on this window, trying {nxt}"
                )
                evo_note(outcome.get("evo"), spread=outcome["spread"])
                evo_reason(outcome.get("evo"), "deferred_flat_scan")
                if flat is None or score(outcome) > score(flat):
                    flat = outcome
            else:
                print(f"widen starved; trying {nxt}")
                evo_reason(outcome.get("evo"), "deferred_starved_widen")
                if best is None or outcome["kept"] > best["kept"]:
                    best = outcome

        def try_rotation_core():
            rc = (
                rotation_core_rays(o_c, o_i, o_u, nw, n_cl, f_probe)
                if fisheye_stage1()
                else rotation_core(o_c, o_i, o_u, nw, n_cl, f_probe)
            )
            if rc is None:
                return None
            print(
                f"rotation core: poses {int(rc[5].sum())}/{nw}, "
                f"inlier<2px {100 * rc[0]:5.1f}%, parallax {rc[1]:.2f} deg "
                f"[{elapsed():.1f}s]"
            )
            return rc

        def open_evo(cand, kind, **channels):
            """Open the evolution record of a probe outcome and dump its
            post-probe state.  Returns the serial (None off the corpus)."""
            if not evo_on():
                return None
            serial = evo_candidate(
                kind,
                pass_index=pass_idx,
                level=level,
                probe_inlier_2px=float(cand[0]),
                probe_parallax_deg=float(cand[1]),
                probe_posed=int(cand[5].sum()),
                probe_med_inl=float(cand[6]),
                near_static_seed_probe=bool(cand[1] < NEAR_STATIC_DEG),
                working_n_clusters=int(cur["n_cl"]),
                working_n_obs=int(len(cur["wk"][0])),
                working_n_images=int(nw),
                local_admission=(cur["data"] or {}).get("local_admission"),
                seed_frames=cur["seed"],
                **channels,
            )
            cur["evo"] = serial
            evo_stage(
                serial,
                "post-probe",
                cur["data"],
                keep,
                f_probe,
                cand[2],
                cand[3],
                cand[4],
                cand[5],
                wk=cur["wk"],
            )
            return serial

        # On parallax-poor captures the far-field rotation skeleton is the
        # seed that works — try it before the covis factorization chunks.
        # It reads the attempt's OWN (stage-A) set: it is a capture-wide
        # far-field fit, not a seed group.
        if allow_rc and (rc_only or _VOTE_POVERTY >= 0.55):
            tried_rc = True
            use_workset()
            cand = try_rotation_core()
            if cand is not None:
                serial = open_evo(cand, "rotation_core")
                if cand[1] >= NEAR_STATIC_DEG:
                    outcome = finish(cand)
                    if committable(outcome):
                        return outcome
                    defer(outcome, "covis seed groups")
                elif low_par is None or cand[1] > low_par[0][1]:
                    evo_reason(serial, "near_static_seed_skipped")
                    low_par = (cand, workset(), serial)
                else:
                    evo_reason(serial, "near_static_seed_skipped")
        if rc_only:
            if flat is not None:
                return flat
            if best is not None:
                return best
            # The near-static fallback: a seed the gate rejected, finished
            # because nothing else survived.  It carries the verdict forward.
            return finish_on(low_par, near_static=True) if low_par is not None else None

        def group_workset(group_list):
            """Point the current try at the working set for ``group_list``'s
            images, re-admitted locally (stage B), or leave it at the
            attempt's own when the re-admission is off or produces nothing.

            The image set is the groups' frames PLUS everything covisible with
            them in the attempt's own graph.  A group of five frames on its own
            would rank coarseness off five viewpoints and leave the widen
            ladder nothing beyond them; its covisible neighbourhood is the part
            of the capture this seed can actually grow into, and it is what the
            ladder resects against rung by rung.

            The re-admitted clusters keep their FULL member lists, so a frame
            outside the neighbourhood still contributes wherever it sees one --
            the neighbourhood bounds what the ranking is measured on, not what
            the solve may reach."""
            if rung is None:
                return None
            imgs = np.unique(np.concatenate([np.asarray(g) for g in group_list]))
            seen = np.unique(o_c[np.isin(o_i, imgs)])
            nbr = np.unique(np.concatenate([imgs, o_i[np.isin(o_c, seen)]]))
            # The re-admission's ONLY input, so it is the working set's identity:
            # `local_admission` is a pure function of this image set given the
            # rung, and the restriction below is a pure function of `keep`.
            nbr_key = tuple(int(x) for x in keep[nbr])
            cur["nbr_key"] = None
            hit = rung.workset_memo.get((nbr_key, keep_key))
            if hit is not None:
                w, n_cl_w, d = hit
                cur["nbr_key"] = nbr_key
                use_workset(w, n_cl_w, d, seed=keep[imgs])
                return w
            d = rung.local_admission(keep[nbr])
            if d is None:
                return None
            # Down to this level's image frame, exactly as the level loop
            # narrows the global set (`keep` maps working -> data image).
            imap = np.full(len(d["names"]), -1, np.int64)
            imap[keep] = np.arange(len(keep))
            m = imap[d["obs_i"]] >= 0
            w = (
                d["obs_c"][m],
                imap[d["obs_i"][m]],
                d["obs_uv"][m],
                d["obs_f"][m],
                nms,
                d["adm_rank"][d["obs_c"][m]],
            )
            la = d["local_admission"]
            print(
                f"group-local admission: {len(nbr)} images "
                f"({len(imgs)} seed + covisible), {la['n_clusters']} clusters "
                f"(min radius {la['min_radius_px']:.1f} px), {int(m.sum())} "
                f"members vs {int(np.isin(o_i, nbr).sum())} global "
                f"[{elapsed():.1f}s]"
            )
            cur["nbr_key"] = nbr_key
            if not snapshots_on():
                rung.workset_memo[(nbr_key, keep_key)] = (w, d["n_cl"], d)
            use_workset(w, d["n_cl"], d, seed=keep[imgs])
            return w

        def try_group_list(group_list):
            """Probe -> widen -> verify each seed group (two at a time, sharing
            factorization); return a HEALTHY outcome or None.

            Each sub-healthy result folds into the enclosing ``best`` /
            ``low_par`` fallbacks the attempt arbitrates at its end.

            Under a group-local re-admission the chunk is ONE group: the whole
            point is a working set derived for the images being seeded, and two
            groups in one chunk would split N_local between them."""
            nonlocal best, low_par, probe_max
            step = 1 if rung is not None else 2
            for chunk in range(0, len(group_list), step):
                grp = group_list[chunk : chunk + step]
                gk = tuple(
                    int(x)
                    for x in keep[
                        np.unique(np.concatenate([np.asarray(g) for g in grp]))
                    ]
                )
                use_workset(seed=np.asarray(gk, dtype=np.int64))
                local = group_workset(grp)
                o_c, o_i, o_u, _o_f, _nms, rank = cur["wk"]
                n_cl = cur["n_cl"]
                if local is None and rung is not None:
                    print("group-local admission: nothing to re-admit; global set")
                # A probe is memoizable exactly when its working set has an
                # identity in the key: the re-admission ran (so the arrays are a
                # function of the neighbourhood and `keep` rather than of the
                # pass's own admission, which the complement changes under it),
                # and no snapshot hook is watching the run it would skip.
                p_key = (
                    None
                    if (cur["nbr_key"] is None or snapshots_on())
                    else (gk, cur["nbr_key"], keep_key, float(f_probe))
                )
                memo = None if p_key is None else rung.probe_memo.get(p_key)
                if memo is not None:
                    rung.memo_hits += 1
                    cand = copy_probe(memo)
                    print(
                        f"seed group {[int(k) for k in gk]}: probe memoized "
                        f"(inlier<2px {100 * cand[0]:5.1f}%, parallax "
                        f"{cand[1]:.2f} deg) [{elapsed():.1f}s]"
                    )
                elif fisheye_stage1():
                    # Ray-space two-view init replaces the affine
                    # factorization outright (Phase 2): weak-perspective
                    # factorization assumes a locally-linear image map, which
                    # is exactly what an equidistant fisheye is not.
                    grown_list = grow_to_cap_rays(
                        [np.asarray(g) for g in grp],
                        f_probe,
                        o_c,
                        o_i,
                        o_u,
                        nw,
                        n_cl,
                        cap,
                        rank,
                    )
                else:
                    seed = []
                    for group in grp:
                        imgs = np.asarray(group)
                        wd = factorize_window(o_c, o_i, o_u, imgs)
                        seed.append((imgs, wd))
                        state = (
                            "sparse"
                            if wd is None
                            else f"{len(wd[2][2])} span-2 clusters"
                        )
                        print(
                            f"seed group {[int(k) for k in imgs]}: {state} "
                            f"[{elapsed():.1f}s]"
                        )
                    grown_list = grow_to_cap(
                        seed,
                        f_probe,
                        o_c,
                        o_i,
                        o_u,
                        nw,
                        n_cl,
                        cap,
                        rank,
                        snap=snap_affine if snapshots_on() else None,
                    )
                if memo is None:
                    cand = None
                    for hyp, grown in enumerate(grown_list):
                        rvec, tvec, pts, posed, med_inl = grown
                        # One posed set for this probe: mask once, and report the
                        # consensus over the SAME rows the adjustment used.
                        bm = budget_mask(posed, o_i, rank)
                        live = ba_rows(bm & ~np.isnan(pts[o_c, 0]), o_i)
                        _, rvec, tvec, pts, res, _ = bundle_adjust(
                            o_c[live],
                            o_i[live],
                            o_u[live],
                            rvec,
                            tvec,
                            pts,
                            f_probe,
                            nw,
                            n_cl,
                            opt_f=False,
                            schedule=((12.0, 2.0), (4.0, 1.0)),
                            max_nfev=30,
                        )
                        denom = ba_rows(bm, o_i)
                        inl = float((res < 2.0).sum() / max(int(denom.sum()), 1))
                        par = core_parallax(rvec, tvec, pts, posed, o_c, o_i)
                        print(
                            f"probe hyp {hyp}: poses {int(posed.sum())}/{nw}, "
                            f"inlier<2px {100 * inl:5.1f}%, parallax {par:.2f} deg "
                            f"[{elapsed():.1f}s]"
                        )
                        if cand is None or inl > cand[0]:
                            cand = (inl, par, rvec, tvec, pts, posed, med_inl)
                    if p_key is not None and cand is not None:
                        rung.probe_memo[p_key] = copy_probe(cand)
                if cand is None:
                    continue
                probe_max = max(probe_max, cand[0])
                gate = max(PROBE_GATE_ABS, PROBE_GATE_REL * probe_max)
                serial = open_evo(
                    cand,
                    "covis_group",
                    seed_group=[int(x) for x in gk],
                    probe_gate=float(gate),
                    probe_max_this_attempt=float(probe_max),
                    probe_memoized=memo is not None,
                )
                if cand[0] < gate:
                    print(
                        f"probe consensus {100 * cand[0]:5.1f}% below the "
                        f"measurability gate {100 * gate:5.1f}% (parallax "
                        f"{cand[1]:.2f} deg); trying next groups"
                    )
                    evo_reason(serial, "probe_gated")
                    gated.append((cand, workset(), serial))
                    continue
                if cand[1] < NEAR_STATIC_DEG:
                    print(
                        "core parallax too low (near-static seed); trying next groups"
                    )
                    evo_reason(serial, "near_static_seed_skipped")
                    if low_par is None or cand[1] > low_par[0][1]:
                        low_par = (cand, workset(), serial)
                    continue
                # The WIDEN and the VERIFY are the other four fifths of a
                # repeated group's cost, and they are the same function of the
                # same key: `finish` reads the memoized probe, the memoized
                # working set and this level's frames, and nothing else that a
                # complement changes.  Two fields do come off the PASS -- the
                # exploration reach is measured on the pass's own covisibility
                # graph, and `wk_global` is the pass's stage-A arrays -- so they
                # are recomputed on the hit rather than replayed.
                held = None if p_key is None else rung.finish_memo.get(p_key)
                if held is not None:
                    rung.memo_hits += 1
                    outcome = dict(held)
                    for k in ("rvec", "tvec", "pts", "posed"):
                        outcome[k] = held[k].copy()
                    outcome["wk_global"] = wk
                    outcome["evo"] = serial
                    pi = np.nonzero(outcome["posed"])[0]
                    outcome["reach"] = float(reach_of(keep[pi]))
                    outcome["reach_capture"] = float(capture_reach_of(keep[pi]))
                    evo_stage(
                        serial,
                        "post-widen",
                        outcome["data"],
                        outcome["keep"],
                        f_probe,
                        outcome["rvec"],
                        outcome["tvec"],
                        outcome["pts"],
                        outcome["posed"],
                        wk=outcome["wk"],
                        widen_memoized=True,
                    )
                    print(
                        f"widen and verify memoized: {outcome['kept']}/{nw} "
                        f"images, reach {100 * outcome['reach']:.0f}% "
                        f"[{elapsed():.1f}s]"
                    )
                else:
                    outcome = finish(cand)
                    if p_key is not None:
                        rung.finish_memo[p_key] = {
                            k: (v.copy() if isinstance(v, np.ndarray) else v)
                            for k, v in outcome.items()
                        }
                if committable(outcome):
                    return outcome
                defer(outcome, "next seed groups")
            return None

        outcome = try_group_list(groups)
        if outcome is not None:
            return outcome
        if allow_rc and not tried_rc and best is None:
            use_workset()
            cand = try_rotation_core()
            if cand is not None:
                serial = open_evo(cand, "rotation_core")
                if cand[1] >= NEAR_STATIC_DEG:
                    outcome = finish(cand)
                    if committable(outcome):
                        return outcome
                    defer(outcome, "the attempt fallbacks")
                elif low_par is None or cand[1] > low_par[0][1]:
                    evo_reason(serial, "near_static_seed_skipped")
                    low_par = (cand, workset(), serial)
                else:
                    evo_reason(serial, "near_static_seed_skipped")
        if flat is not None:
            return flat
        if best is not None:
            return best
        if low_par is not None:
            # The near-static fallback: a seed the gate rejected, finished
            # because nothing else survived.  It carries the verdict forward.
            return finish_on(low_par, near_static=True)
        if gated:
            # Last resort only: an unmeasurable probe still beats no seed at
            # all, but it loses to any widened outcome and to the parallax
            # fallback (20240614_224422531: deferring to a 98.4%/0.45 deg
            # low_par window moved the final from +8.3% to -7.1%).
            held = max(gated, key=lambda c: c[0][0])
            print(
                f"no measurable window; falling back to the best gated probe "
                f"({100 * held[0][0]:.1f}%, parallax {held[0][1]:.2f} deg)"
            )
            return finish_on(held)
        return None

    # Adaptive covisibility thinning: casual video often moves too slowly
    # for adjacent-image parallax (GT medians 0.08-0.93 deg per step across
    # the phone-video fleet), and a parallax-starved seed strands the whole
    # ladder at a sliver of the capture.  The working set is thinned by
    # COVISIBILITY, not frame order (the bootstrap is order-free end to
    # end): a greedy sweep keeps an image only when its shared-cluster
    # count with every already-kept image is below tau, so dense redundant
    # viewpoint blobs thin out while coverage survives — on video this
    # reproduces frame striding (covisibility decays with frame distance),
    # on unordered sets it thins burst-duplicates striding cannot see.
    # Each tau level runs the full probe/widen/verify attempt; a HEALTHY
    # outcome commits immediately, otherwise tau drops 3x (all on the SAME
    # clusters file — no re-matching) and the best-covering attempt wins.
    # Probe parallax alone cannot arbitrate this: cc bootstraps fine from a
    # 1.26-deg probe while the parallax-starved phone videos die from
    # identical numbers — the downstream outcome is the only reliable
    # signal.  The pairwise vote above stays at the full native set on
    # purpose: thinned pair graphs degraded it badly (telephoto case: -10%
    # native, -52% at a x16-equivalent thinning).
    # Covisibility-banded thinning (ClusterCovisibility.thin_to) and reach
    # (>= 8 shared clusters) are native kernels — order-free redundancy
    # thinning and the sliver-vs-wide reach signal live in the core covis
    # object, no Python re-derivation of the counts matrix.
    #
    # Two graphs, deliberately: the pass's OWN covisibility drives the thinning
    # (`thin_to`) and the exploration's reach signals — the ladder is choosing
    # among working sets of the admission it is solving — while the
    # CAPTURE-level graph, the full admission's, is what the arbitration's
    # gates and the narrow-reach flag read.  On the first hypothesis the two
    # ARE the same graph (the pass admission is the full one), so it is built
    # once and shared.
    covis_capture = _COVIS_CAPTURE["v"]
    covis_full = (
        covis_capture
        if obs_c is _COVIS_CAPTURE["obs_c"]
        else build_covisibility(obs_c, obs_i, n_img, n_cl)
    )
    n0 = n_img

    def reach_of(orig_posed):
        return covis_full.reach(np.ascontiguousarray(orig_posed, np.uint32), 8)

    def capture_reach_of(orig_posed):
        """Coverage reach on the CAPTURE-level graph — what qualification, the
        commit bar the arbitration ranks on, and the narrow-reach flag mean by
        reach.  Identical to ``reach_of`` on the first hypothesis."""
        if covis_full is covis_capture:
            return reach_of(orig_posed)
        return capture_reach(orig_posed)

    # Coverage reach saturates as a quality signal (graph coverage cannot
    # separate a focal-blind video sliver from a sparse-but-wide photo
    # solve: dl-sliver 53% vs south-building-healthy 61%), so the health
    # bar below is an EXPLORATION threshold, not a verdict — below it we
    # keep trying thinner working sets.  Attempts are arbitrated by FOCAL
    # OBSERVABILITY first: the coarse fixed-f scan of an observable solve
    # peaks (dl thinned: 67->85->67), a bas-relief solve fits every
    # candidate equally (204251146: 71.8-72.8% across the grid), so the
    # scan's inlier spread separates them where coverage cannot; coverage
    # only breaks ties among equally-observable attempts.
    def coverage(att):
        return att["kept"] * min(att["reach"], 0.60)

    def scan_spread(att):
        """Spread (max - min inlier fraction) of the coarse fixed-f scan
        over the attempt's geometry — the focal-observability score.

        Measured in the attempt's OWN working set: an outcome carries the
        cluster space it was solved in (a group-local re-admission is not the
        pass's), so the cluster count and the per-observation rank the BA row
        budget reads both come off the outcome rather than the pass."""
        global _RANK_O
        o_c, o_i, o_u = att["wk"][0], att["wk"][1], att["wk"][2]
        nw, posed = att["nw"], att["posed"]
        n_cl_w = att["n_cl"]
        _RANK_O = att["wk"][5]
        # The posed set is fixed across the grid, so one mask serves every
        # candidate's adjustment and the common denominator.
        bm = budget_mask(posed, o_i, att["wk"][5])
        denom = ba_rows(bm, o_i)
        inls = []
        for f_try in f_grid:
            rv, tv = att["rvec"].copy(), att["tvec"] * (f_try / f_probe)
            rot = Rotation.from_rotvec(rv).as_matrix()
            p_t = triangulate(o_c, o_i, o_u, rot, tv, posed, n_cl_w, f_try)
            live = ba_rows(bm & ~np.isnan(p_t[o_c, 0]), o_i)
            _, rv, tv, p_t, res, _ = bundle_adjust(
                o_c[live],
                o_i[live],
                o_u[live],
                rv,
                tv,
                p_t,
                f_try,
                nw,
                n_cl_w,
                opt_f=False,
                max_nfev=25,
            )
            inls.append(float((res < 2.0).sum() / max(int(denom.sum()), 1)))
        # The grid's own peak, kept on the outcome: the ladder's evidence needs
        # a focal for an attempt that has not released one yet, and this is the
        # attempt's best fixed-f answer measured on its own geometry.
        att["f_peak"] = float(f_grid[int(np.argmax(inls))])
        return max(inls) - min(inls)

    def score(att):
        spread = att["spread"] if att["spread"] >= 0.05 else 0.0
        return (spread, coverage(att))

    # The focal the ladder's far fits are measured at: the capture-level vote,
    # in the solve's own parameterization.  Fixed for the whole pass, so two
    # attempts are always compared through the same lens.
    f_far = f_probe if (fisheye_stage1() or f_vote is None) else float(f_vote)

    def attempt_evidence(att):
        """The attempt's own INDEPENDENT far-field fit, and what it says about
        the attempt: how far the attempt's camera motion is from the rotation
        the pair evidence supports, and how much of the far field the attempt
        leaves its own far layer to price.

        Memoized on the outcome, and computed only for attempts the ladder
        actually compares -- a pass with one attempt makes no ranking decision
        and pays nothing.  The fit is the far layer's own machinery
        (`ray_rotation_edges`, `rotation_spanning_tree`) on the attempt's
        working set and frames, so what ranks the attempt here is the same
        object that will later be committed beside it."""
        if "evidence" in att:
            return att["evidence"]
        o_c, o_i, o_u = att["wk"][0], att["wk"][1], att["wk"][2]
        nw, n_cl_w = att["nw"], att["n_cl"]
        ev = {"dis": None, "shared": 0, "beyond": None, "far_rows": 0}
        att["evidence"] = ev
        n_live = int(len(np.unique(o_i)))
        max_pairs = max(120, 4 * n_live)
        min_corr, min_inl = ray_rotation_floors(o_c, o_i, nw, n_cl_w, max_pairs)
        found = ray_rotation_edges(
            o_c,
            o_i,
            o_u,
            nw,
            n_cl_w,
            f_far,
            max_pairs=max_pairs,
            min_corr=min_corr,
            min_inliers=min_inl,
        )
        chained = None if found is None else rotation_spanning_tree(found[2], nw)
        if chained is None:
            return ev
        rv_rot, abs_rot = chained
        pd_rot = np.zeros(nw, bool)
        pd_rot[list(abs_rot)] = True
        ev["shared"], ev["dis"] = relative_rotation_disagreement(
            att["rvec"], att["posed"], rv_rot, pd_rot
        )
        # Absorption: of the far layer's own priced rows, how many the
        # attempt's finite structure does not also price.  Measured at the
        # attempt's probe focal, which is the geometry it actually holds.
        rot = Rotation.from_rotvec(rv_rot).as_matrix()
        cam = make_cam(f_far)
        rows = np.nonzero(pd_rot[o_i])[0]
        if not len(rows):
            return ev
        d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(o_u[rows]))
        d_w = np.einsum("nji,nj->ni", rot[o_i[rows]], d_loc)
        d_w /= np.maximum(np.linalg.norm(d_w, axis=1, keepdims=True), 1e-12)
        uniq, first = np.unique(o_c[rows], return_index=True)
        dirs = np.full((n_cl_w, 3), np.nan)
        dirs[uniq] = d_w[first]
        x_cam = np.einsum("nij,nj->ni", rot[o_i[rows]], dirs[o_c[rows]])
        proj = cam.ray_to_pixel_batch(np.ascontiguousarray(x_cam))
        r_far = np.linalg.norm(proj - o_u[rows], axis=1)
        far_in = np.isfinite(r_far) & (r_far < 2.0)
        priced = finite_priced_rows(
            att["rvec"],
            att["tvec"],
            att["pts"],
            att["posed"],
            f_probe,
            o_c,
            o_i,
            o_u,
            rows,
        )
        ev["far_rows"] = int(far_in.sum())
        ev["beyond"] = int((far_in & ~priced).sum())
        return ev

    def indicts(a, b):
        """Does the evidence indict attempt ``a`` in favour of ``b``?

        Two conjuncts, primary first.  ROTATION: ``a``'s camera motion is an
        outlier against the rotation its own pair evidence supports, while
        ``b``'s is not -- the direct reading of "the motion is not accurate".
        ABSORPTION: ``b``'s far layer prices far more than ``a``'s does beyond
        its finite sibling, which says ``a`` swallowed the far field into a
        depth shell.  The second only speaks where the first is silent, is
        never exculpatory, and stands down entirely on a far-field-dominated
        capture, where a big far layer is the scene rather than a symptom."""
        ea, eb = attempt_evidence(a), attempt_evidence(b)
        if ea["dis"] is not None and eb["dis"] is not None:
            if ea["dis"] > ROT_DISAGREE_ABS_DEG and ea[
                "dis"
            ] > ROT_DISAGREE_RATIO * max(eb["dis"], 1e-6):
                return f"rotation {ea['dis']:.2f} deg vs {eb['dis']:.2f} deg"
            if eb["dis"] > ROT_DISAGREE_ABS_DEG and eb[
                "dis"
            ] > ROT_DISAGREE_RATIO * max(ea["dis"], 1e-6):
                return None  # the competitor is the outlier, not this one
        if _VOTE_POVERTY >= FAR_DOMINANT_POVERTY:
            return None
        ba, bb = ea["beyond"], eb["beyond"]
        if ba is None or bb is None or ba <= 0:
            return None
        if bb > ABSORPTION_RATIO * ba and (
            a.get("f_peak") is None
            or b.get("f_peak") is None
            or f_center is None
            or abs(np.log(a["f_peak"] / f_center)) > abs(np.log(b["f_peak"] / f_center))
        ):
            return f"absorption, far-beyond {ba} vs {bb}"
        return None

    def legacy_rank(att):
        """The parallax-CLASS ranking this one replaces, kept only to name the
        finalist the change displaces (never to decide anything)."""
        return (not att.get("near_static_seed", False), *score(att))

    def att_pose_frame(a):
        """An attempt's rotations and posed mask in the DATA image frame, so
        two attempts from different thinning levels can be compared."""
        rv, _tv, pd = _snap_full_frame(
            data, a["keep"], a["rvec"], a["tvec"], a["posed"], f_probe
        )
        return rv, pd

    def finalists(atts):
        """THE LADDER'S CANDIDATES: every distinct finalist, score-ranked.

        The ladder used to end in a verdict, and every verdict we tried was
        wrong somewhere -- the parallax class demoted the good attempt on
        20250712_195736354, the rotation conjunct promoted the bad one on
        20240618_001255975~2, and the bare score prefers whatever scan spread
        degenerate geometry produces.  A stage this cheap does not have to
        choose: it can hand the next rung several readings and the evidence
        each was judged on, and let corroboration do what a threshold cannot.

        So the ladder generates.  Every finalist commits, the score becomes an
        ADVISORY rank, and the three retired verdicts (the near-static class,
        the rotation-disagreement indictment and the absorption indictment)
        stay on each candidate as recorded evidence rather than as gates.

        The one thing still removed is a DUPLICATE: two finalists that pose
        overlapping frames and agree about the geometry there are one answer
        found twice (`distinct`), and the better-scored copy stands for both.
        That is dedup, not judgment -- it collapses identical answers and never
        chooses between different ones."""
        ranked = sorted(atts, key=score, reverse=True)
        if rung is None:
            # The legacy full-seed path keeps the single-winner ladder it was
            # made with: one attempt out, no evidence pass, no extra line.
            return ranked[:1]
        for a in ranked:
            ev = attempt_evidence(a)
            a["evidence_summary"] = ev
            evo_note(
                a.get("evo"),
                rotation_disagreement_deg=ev["dis"],
                rotation_shared_frames=ev["shared"],
                far_beyond=ev["beyond"],
                far_rows=ev["far_rows"],
                far_beyond_frac=(
                    None
                    if not ev["far_rows"] or ev["beyond"] is None
                    else ev["beyond"] / ev["far_rows"]
                ),
            )
            dis = "n/a" if ev["dis"] is None else f"{ev['dis']:.2f} deg"
            print(
                f"  ladder candidate: level {a['level']} kept {a['kept']} "
                f"spread {100 * a['spread']:.1f}pp, rotation disagreement {dis} "
                f"over {ev['shared']} shared frames, far-beyond {ev['beyond']}"
                + (", near-static seed" if a.get("near_static_seed") else "")
            )
        out, frames, collapsed = [], [], 0
        for a in ranked:
            rv_a, pd_a = att_pose_frame(a)
            dup = None
            margins = []
            for k, (rv_b, pd_b) in enumerate(frames):
                n_sh, deg = relative_rotation_disagreement(rv_a, pd_a, rv_b, pd_b)
                margins.append(
                    {
                        "vs_serial": out[k].get("evo"),
                        "shared_frames": int(n_sh),
                        "rot_disagreement_deg": deg,
                    }
                )
                if deg is not None and deg <= POSE_DISAGREE_DEG:
                    dup = k
                    break
            evo_note(a.get("evo"), ladder_margins=margins)
            if dup is not None:
                collapsed += 1
                out[dup]["collapsed"] = out[dup].get("collapsed", 0) + 1
                print(
                    f"    level {a['level']} collapses into level "
                    f"{out[dup]['level']} (same basin on shared frames)"
                )
                evo_reason(a.get("evo"), "distinctness_collapse")
                evo_note(a.get("evo"), collapsed_into_serial=out[dup].get("evo"))
                continue
            a["ladder_rank"] = len(out)
            evo_note(a.get("evo"), ladder_rank=len(out))
            out.append(a)
            frames.append((rv_a, pd_a))
        print(
            f"  ladder: {len(out)} distinct candidate"
            + ("" if len(out) == 1 else "s")
            + (f", {collapsed} collapsed as duplicates" if collapsed else "")
            + f" [{elapsed():.1f}s]"
        )
        return out

    atts = []
    level = 0  # native set first; then targets n0/3, n0/9, ...
    while True:
        if level == 0:
            keep = np.arange(n0)
            wk = (
                obs_c,
                obs_i,
                u,
                data["obs_f"],
                list(data["names"]),
                data["adm_rank"][obs_c],
            )
        else:
            if rung is not None and rung.max_levels and level >= rung.max_levels:
                # The ladder is capped: with the displaced finalists preserved,
                # the later, thinner levels may only re-derive basins the
                # earlier ones already committed.  Whether they do is a
                # measurement, so the cap is a field with the measured
                # behaviour (uncapped) as its default.
                break
            target = n0 // 3**level
            if target < 20:
                break
            keep = np.asarray(covis_full.thin_to(target), dtype=np.int64)
            if len(keep) >= n0 or len(keep) < 20:
                break
            print(
                f"thinning working set to {len(keep)} images "
                f"(target {target}) [{elapsed():.1f}s]"
            )
            imap = np.full(n0, -1, np.int64)
            imap[keep] = np.arange(len(keep))
            m = imap[obs_i] >= 0
            wk = (
                obs_c[m],
                imap[obs_i[m]],
                u[m],
                data["obs_f"][m],
                [data["names"][j] for j in keep],
                data["adm_rank"][obs_c[m]],
            )
        _RANK_O = wk[5]
        # Parallax-poor fast path: level 0 runs ONLY the rotation core (its
        # covis chunks are both the most expensive stage and the one that
        # starves there); thinned levels never run it (no redundancy left).
        # The path needs actual rotation votes (a poverty-high capture with
        # none has a weak far field — dino_dog_toy's orbit background) and
        # falls back to the full native attempt when the core produces
        # nothing.
        rc_first = _VOTE_POVERTY >= 0.6 and _VOTE_ROT_N >= 3 and n0 >= 60
        att = attempt(
            wk,
            len(keep),
            keep,
            rc_only=(level == 0 and rc_first),
            allow_rc=(level == 0),
        )
        if att is None and level == 0 and rc_first:
            att = attempt(wk, len(keep), keep, allow_rc=False)
        if att is not None:
            att["level"] = level
            if "spread" not in att:  # attempts measure their own commit bar
                att["spread"] = scan_spread(att) if att["kept"] >= 8 else 0.0
            # This level's outcome was RETURNED, whatever fallback slot it came
            # out of, so any drop reason a defer stamped on it is withdrawn.
            evo_clear(att.get("evo"))
            evo_note(
                att.get("evo"),
                level=level,
                kept=att["kept"],
                reach=att["reach"],
                reach_capture=att["reach_capture"],
                spread=att["spread"],
                f_peak=att.get("f_peak"),
                near_static_seed=bool(att.get("near_static_seed", False)),
            )
            print(
                f"attempt: kept {att['kept']}, reach {100 * att['reach']:.0f}%"
                + (
                    ""
                    if covis_full is covis_capture
                    else f" (capture-level {100 * att['reach_capture']:.0f}%)"
                )
                + f", scan spread {100 * att['spread']:.1f}pp [{elapsed():.1f}s]"
            )
            atts.append(att)
            if att["kept"] >= 8 and att["reach"] >= 0.60 and att["spread"] >= 0.05:
                # Stop exploring: observable + ample coverage.  Under the rung
                # this is the only reason the ladder stops early, and it stops
                # a GENERATOR rather than a search for one answer -- the
                # attempts already found still all commit.
                break
        level += 1
    if not atts:
        if not required:
            return None
        raise SystemExit("no seed group produced a reconstruction")
    cands = finalists(atts)

    def release_from(chosen):
        """The chosen attempt through the focal scan, the release and the
        coverage claim, as a committed hypothesis's record.

        Taken as a function of the attempt so the ladder's DISPLACED
        finalist can be released the same way: nothing an attempt loses a
        ranking by makes it unreleasable, and the set is the product."""
        global _RANK_O
        obs_c, obs_i, u, _obs_f, names, _RANK_O = chosen["wk"]
        # The SOLVE's cluster space, which is the pass's own unless a group-local
        # re-admission supplied a denser one.  `n_cl` stays the PASS's: the
        # coverage claim below is stated in it, whatever the solve ran on.
        n_cl_w = chosen["n_cl"]
        n_img = chosen["nw"]
        keep_w = chosen["keep"]
        rvec, tvec, pts, posed = (
            chosen["rvec"],
            chosen["tvec"],
            chosen["pts"],
            chosen["posed"],
        )
        if chosen["level"] > 0:
            print(f"committed thinned working set ({n_img} images)")
        # The committed posed set is frozen from here on (the scan and the release
        # move focal and structure, never the posed set), so the budget resolves
        # once and every scan/release adjustment and denominator shares it.
        bam = budget_mask(posed, obs_i, _RANK_O)
        denom = ba_rows(bam, obs_i)

        # Focal scan on the widened geometry: per candidate, rescale the
        # translations (depth scale ~ f), retriangulate, staged fixed-f BA.
        # Two phases: a capped-iteration pass ranks all candidates cheaply, and
        # heavier refits decide between the top two (neighbouring candidates
        # can rank within a point of each other — DinoLedge flips at 52 vs 54 —
        # and the light pass is not reliable at that margin).
        def scan_candidate(f_try, nfev):
            scale = f_try / f_probe
            rv_t, tv_t = rvec.copy(), tvec * scale
            rot = Rotation.from_rotvec(rv_t).as_matrix()
            p_t = triangulate(obs_c, obs_i, u, rot, tv_t, posed, n_cl_w, f_try)
            live = ba_rows(bam & ~np.isnan(p_t[obs_c, 0]), obs_i)
            _, rv_t, tv_t, p_t, res, _ = bundle_adjust(
                obs_c[live],
                obs_i[live],
                u[live],
                rv_t,
                tv_t,
                p_t,
                f_try,
                n_img,
                n_cl_w,
                opt_f=False,
                max_nfev=nfev,
            )
            inl = float((res < 2.0).sum() / max(int(denom.sum()), 1))
            return inl, f_try, rv_t, tv_t, p_t

        coarse = []
        for f_try in f_grid:
            cand = scan_candidate(f_try, 25)
            coarse.append(cand)
            print(
                f"f={cand[1]:6.1f}: inlier<2px {100 * cand[0]:5.1f}% [{elapsed():.1f}s]"
            )
        inls_grid = [c[0] for c in coarse]  # grid order, for the edge-scan check
        coarse.sort(key=lambda t: -t[0])
        pick = [coarse[0], coarse[1]]
        if f_indep is not None:
            # A flat scan is f-degenerate structure with no opinion of its own
            # — marginal captures then flip basins on run-to-run noise.  The
            # structure-free vote is an INDEPENDENT measurement, so the candidate
            # nearest it (bias-corrected: the pinhole vote runs ~10% low, the
            # equidistant verdict is used raw) always earns a refit slot when it
            # ranks within noise of the leader, and it wins outright when the
            # refits tie.
            near = min(coarse, key=lambda t: abs(np.log(t[1] / (f_bias * f_indep))))
            if (
                near[1] not in (pick[0][1], pick[1][1])
                and near[0] >= coarse[0][0] - 0.05
            ):
                pick[1] = near
        finals = [scan_candidate(c[1], 60) for c in pick]
        for c in finals:
            print(
                f"f={c[1]:6.1f} (refit): inlier<2px {100 * c[0]:5.1f}% [{elapsed():.1f}s]"
            )
        best = max(finals, key=lambda t: t[0])
        if f_indep is not None and abs(finals[0][0] - finals[1][0]) < 0.05:
            best = min(finals, key=lambda t: abs(np.log(t[1] / (f_bias * f_indep))))

        inl0, f, rvec, tvec, pts = best
        print(f"scan winner: f = {f:.1f} [{elapsed():.1f}s]; releasing f")
        # Iterated release: full schedule (the wide first trim + inter-round
        # retriangulation is what lets f keep walking — the structure absorbs a
        # wrong f and must be re-formed as f moves).  Stop when f stabilizes;
        # keep the best-fit state seen.
        inl, f_prev = inl0, f
        kept = (inl0, f, rvec, tvec, pts)
        for _ in range(3):
            live = ba_rows(bam & ~np.isnan(pts[obs_c, 0]), obs_i)
            f, rvec, tvec, pts, res, _ = bundle_adjust(
                obs_c[live],
                obs_i[live],
                u[live],
                rvec,
                tvec,
                pts,
                f,
                n_img,
                n_cl_w,
                opt_f=True,
                max_nfev=30,
            )
            inl = float((res < 2.0).sum() / max(int(denom.sum()), 1))
            # The release has to stay in the scan's basin, in BOTH directions.
            # Upward it is the affine collapse: on narrow or shallow geometry
            # f -> inf keeps fitting better (rising inlier fraction — which
            # also means the keep-best rule cannot be trusted upward).
            # Downward the same bas-relief freedom lets the release walk a long
            # way below the scan winner (20240918_074134864: winner 2878.9 at
            # +3.3% released to 2078.1 at -25%).  Either way the walk
            # contradicts the scan's trim-consistent fixed-f ranking, which is
            # the more trustworthy measurement, so the band is symmetric; every
            # legitimate walk observed stayed within +/-10%.  The absolute
            # floor stays as a plausibility bound — FOV-derived under an
            # equidistant context, where the pinhole floor would reject the
            # capture's own true focal outright (kerry: f ~ 138 against a
            # 0.3 x 480 = 144 pinhole floor).
            if not (best[1] / 1.15 <= f <= 1.15 * best[1]) or f < focal_floor():
                print(f"release left the scan basin (f = {f:.0f}); keeping previous")
                break
            if inl > kept[0]:
                kept = (inl, f, rvec, tvec, pts)
            if abs(f - f_prev) < 0.01 * f_prev:
                break
            f_prev = f
        inl, f, rvec, tvec, pts = kept
        serial = chosen.get("evo")
        evo_stage(
            serial,
            "pre-spline",
            chosen["data"],
            keep_w,
            f,
            rvec,
            tvec,
            pts,
            posed,
            wk=chosen["wk"],
            scan_grid_f=[float(v) for v in f_grid],
            scan_grid_inliers=[float(v) for v in inls_grid],
            scan_winner_f=float(best[1]),
            released_inlier_2px=float(inl),
        )
        # THE LENS.  The focal is released; the radial shape is not, and this is
        # where the hypothesis gets a starting correction for it.  Refusals cost
        # the release nothing -- an all-zero spline is the base map bit for bit
        # -- so the structure below is what it was either way, and only the
        # camera record changes.
        lens = None
        if rung is not None:
            live_l = ba_rows(bam & ~np.isnan(pts[obs_c, 0]), obs_i)
            lens = spline_release(
                obs_c, obs_i, u, rvec, tvec, pts, f, n_img, n_cl_w, live_l
            )
            if lens is not None and lens["accepted"]:
                # The arm's own state ships with its lens.
                rvec, tvec, pts = lens["rvec"], lens["tvec"], lens["pts"]
                f = lens["f_chart"]
                inl = float((lens["res"] < 2.0).sum() / max(int(denom.sum()), 1))
            elif lens is not None and "rvec" in lens.get("refused", {}):
                # The REFUSED arm, completed exactly as an accepted one would
                # have been: the full triangulation of its own geometry, so the
                # artifact beside the release is what acceptance would have
                # shipped and not a partial view of it.
                ref = lens["refused"]
                ref["release_pts"] = triangulate(
                    obs_c,
                    obs_i,
                    u,
                    Rotation.from_rotvec(ref["rvec"]).as_matrix(),
                    ref["tvec"],
                    posed,
                    n_cl_w,
                    ref["f_chart"],
                    cam=bootstrap_module().make_cam_bspline(
                        ref["f_chart"],
                        ref["camera"]["params"]["coefficients"],
                        lens["d_max"],
                    ),
                )
        # BOTH ARMS OF THE KEEP-BEST into the corpus: the survey has to score
        # the arm that shipped against the one that did not, and a refusal is a
        # verdict about which of two reconstructions is better -- exactly the
        # kind of verdict this study exists to check.
        if lens is not None and evo_on():
            ctl = lens.pop("control", None) or None
            if ctl is not None:
                evo_stage(
                    serial,
                    "spline-control",
                    chosen["data"],
                    keep_w,
                    ctl["f"],
                    ctl["rvec"],
                    ctl["tvec"],
                    ctl["pts"],
                    posed,
                    wk=chosen["wk"],
                    spline_arm="control",
                )
            ref = lens.get("refused") or {}
            if lens["accepted"]:
                evo_stage(
                    serial,
                    "spline-treatment",
                    chosen["data"],
                    keep_w,
                    lens["f_chart"],
                    rvec,
                    tvec,
                    pts,
                    posed,
                    wk=chosen["wk"],
                    lens=lens,
                    spline_arm="treatment",
                    spline_accepted=True,
                )
            elif "rvec" in ref:
                arm_lens = {
                    "f_chart": ref["f_chart"],
                    "coeffs": ref["camera"]["params"]["coefficients"],
                    "d_max": lens["d_max"],
                }
                evo_stage(
                    serial,
                    "spline-treatment",
                    chosen["data"],
                    keep_w,
                    ref["f_chart"],
                    ref["rvec"],
                    ref["tvec"],
                    ref.get("release_pts", ref["pts"]),
                    posed,
                    wk=chosen["wk"],
                    lens=arm_lens,
                    spline_arm="treatment",
                    spline_accepted=False,
                    refused_reason=ref["reason"],
                )
            scores = lens.pop("scores", None) or {
                k: v
                for k, v in ref.items()
                if k not in ("rvec", "tvec", "pts", "release_pts", "camera")
            }
            evo_note(
                serial,
                spline_accepted=bool(lens["accepted"]),
                spline_refused_reason=ref.get("reason"),
                spline_f_chart=lens["f_chart"],
                spline_f_eq=lens["f_eq"],
                spline_d_max=lens["d_max"],
                spline_coeffs=[float(c) for c in np.asarray(lens["coeffs"])],
                spline_scores={
                    k: v for k, v in _jsonable(scores).items() if k != "camera"
                },
            )
        # Checkpoint 4: this pass's released estimate.  Both passes snapshot (main
        # arbitrates between them), so the rejected alternative stays inspectable.
        seed_snap(
            f"04-release-{_SNAP['tag']}",
            chosen["data"],
            keep_w,
            f,
            rvec,
            tvec,
            pts,
            posed,
        )

        # Structure for the COVERAGE CLAIM (the hypothesis loop's, see
        # `specs/core/geometry/seed-hypothesis-loop.md`): every cluster the posed set sees in
        # two or more views, placed by the released geometry.  Not `pts`: that array
        # carries only the clusters the BA ROW BUDGET admitted (MAX_CL clusters,
        # OBS_PER_IMG per image), and the budget is a solver-cost control — the
        # observations it leaves out are read by the same posed images all the same,
        # so a claim built from it states a fraction of the area the hypothesis
        # actually explains.  One triangulation of the full cluster set, the same
        # call the focal scan already makes per candidate.
        #
        # The claim is stated in the PASS's cluster space whatever the solve ran
        # on, because that is the space the complement, the materiality test and
        # the next hypothesis's admission are all derived in.  Poses are poses, so
        # the same released geometry places both sets; the RELEASE artifact then
        # shows the structure the hypothesis actually solved, which under a
        # group-local re-admission is the denser one.
        rot_rel = Rotation.from_rotvec(rvec).as_matrix()
        g_c, g_i, g_u = (
            chosen["wk_global"][0],
            chosen["wk_global"][1],
            chosen["wk_global"][2],
        )
        # Triangulated through the LENS the poses were solved under.  An
        # accepted spline is a different map from the base one at the same
        # focal, and structure placed through the wrong map is culled by the
        # writer that reprojects it through the right one.  A refusal ships the
        # zero spline, which IS the base map, so `lens_cam` is None there and
        # nothing moves.
        lens_cam = (
            bootstrap_module().make_cam_bspline(
                lens["f_chart"], lens["coeffs"], lens["d_max"]
            )
            if lens is not None and lens["accepted"]
            else None
        )
        claim_pts = triangulate(
            g_c, g_i, g_u, rot_rel, tvec, posed, n_cl, f, cam=lens_cam
        )
        release_pts = (
            claim_pts
            if chosen["wk_global"] is chosen["wk"]
            else triangulate(
                obs_c, obs_i, u, rot_rel, tvec, posed, n_cl_w, f, cam=lens_cam
            )
        )

        # Confidence flag for the failure classes rigid pinhole SfM cannot fix
        # (diagnosed on the validation campaign): non-rigid / scene-changed
        # captures (swivel-chair: people and a dog in frame, props moved
        # mid-capture) and f-degenerate near-planar close-ups (dino-ledge)
        # both leave NO focal fitting well — the best scan/release inlier
        # fraction stays below the healthy band (broken: 17-54%, healthy: 62%+
        # across the campaign) while the poses bend smoothly (bas-relief warp),
        # so the per-image geometric and photometric checks all pass.  (A
        # structure-thickness planarity test does NOT work: orbit captures
        # produce thin relief shells of surface points, and the healthiest
        # datasets measure as "planar" as the broken ones.)
        #
        # The 0.60 cut is recalibrated for the native BA (the scipy-era band was
        # broken 17-54% / healthy 62%+ with a 0.58 cut): the native optimizer
        # squeezes more inliers out of a broken capture (kerry fisheye: 43% ->
        # 58.7%), compressing the separation — on the in-repo datasets the gap is
        # now 58.7% (broken) vs 61.8%+ (healthy).  Knife-edge; re-validate the
        # band on the full campaign datasets.
        flags = ["low_consensus"] if inl < 0.60 else []
        # Runtime guards from the casual-video fleet (no reference needed).  A
        # release that leaves the bias-corrected vote's error band is bas-relief
        # walking (ws2: vote +0.2% vs release -35% at 86% inliers), and a posed
        # set clustered in a sliver of the input sequence cannot observe focal
        # at all (scans rank meaninglessly below ~30% rig span).  Both bands are
        # first-pass; recalibrate on the full campaign.  The 1.35 band clears
        # seoul's known vote-low jitter (release/1.1-vote hits 1.30 there).
        # The guard compares the released focal with the structure-free vote in
        # the SOLVE'S OWN parameterization (`f_indep`): the pinhole vote for a
        # pinhole solve, the equidistant verdict for an equidistant one.  Reading
        # the pinhole vote against an equidistant release would be a units error
        # rather than a guard — the two maps agree only near the axis.
        if f_indep is not None and abs(np.log(f / (f_bias * f_indep))) > np.log(1.35):
            flags.append("vote_divergence")
            print(
                f"VOTE DIVERGENCE: release f = {f:.1f} left the corrected vote "
                f"band ({f_bias * f_indep:.1f} +/- 35%); structure is bas-relief "
                f"suspect"
            )
        final_reach = capture_reach_of(keep_w[np.nonzero(posed)[0]])
        if final_reach < 0.30:
            flags.append("narrow_reach")
            print(
                f"NARROW REACH: posed set is covisibility-connected to only "
                f"{100 * final_reach:.0f}% of the input images (<30%); focal is "
                f"not observable on a sliver"
            )
        if rung is not None and chosen.get("near_static_seed", False):
            # The ladder had nothing the near-static gate passed, so the capture is
            # seeded from a window the gate rejected.  Nothing is discarded, but the
            # release says so and cannot qualify on it.
            flags.append("near_static_seed")
            print(
                f"NEAR-STATIC SEED: every viable window came from a seed the "
                f"parallax gate rejected (core parallax under "
                f"{NEAR_STATIC_DEG:.1f} deg); the geometry cannot observe depth "
                f"and the released focal is not evidence"
            )
        if chosen["spread"] < 0.05:
            flags.append("flat_scan")
            print(
                f"FLAT SCAN: the fixed-f scan varied only "
                f"{100 * chosen['spread']:.1f}pp across the grid — this geometry "
                f"does not observe focal (bas-relief); do not trust the "
                f"structure estimate"
            )
        # A scan that rises MONOTONICALLY to the top edge of the grid is the
        # upward affine escape wearing an "observable" spread: the inlier fraction
        # keeps improving as f grows without bound, so no interior optimum exists
        # and the structure has no downward opinion on focal at all (DnDTabletop:
        # 72->83% rising into the edge; the release then tried f=4800).  Spread
        # cannot see this (it measures variation, not peakedness).  Flag it: the
        # released f is a guard artifact, and the structure-free vote is the
        # reliable estimate.
        if (
            "flat_scan" not in flags
            and int(np.argmax(inls_grid)) == len(inls_grid) - 1
            and np.all(np.diff(inls_grid) >= -0.005)
        ):
            flags.append("edge_scan")
            print(
                f"EDGE SCAN: the fixed-f scan rises monotonically into the top of "
                f"the grid ({100 * inls_grid[0]:.1f}% -> {100 * inls_grid[-1]:.1f}%) "
                f"— no interior optimum; the structure does not bound focal from "
                f"above; do not trust the structure estimate"
            )
        if flags:
            print(
                f"LOW CONFIDENCE: best inlier fraction {100 * inl:.1f}% — "
                f"no focal fits this capture well (non-rigid scene or "
                f"f-degenerate structure); do not trust the structure estimate"
            )
            # When the structure pipeline disqualifies itself, its focal is a
            # lottery over run-to-run noise (marginal captures flip basins on a
            # 2% probe shift).  The structure-free vote never touched structure,
            # so it is the more reliable focal on exactly these captures — report
            # it as the estimate and keep the structure value alongside.  Under a
            # fisheye context that vote is the EQUIDISTANT verdict, not the
            # pinhole one: only the former parameterizes the map the geometry was
            # solved in.
            if f_indep is not None:
                print(f"falling back to the structure-free vote: f = {f_indep:.1f}")

        f_report = f_indep if (flags and f_indep is not None) else f
        print(
            f"\nfast pinhole estimate: f = {f_report:.1f} px on "
            f"{int(posed.sum())}/{n_img} images, inlier<2px {100 * inl:.1f}% "
            f"[{elapsed():.1f}s]"
        )
        return {
            "names": names,
            "posed": posed,
            "rvec": rvec,
            "tvec": tvec,
            "f": float(f),
            "f_report": float(f_report),
            "f_vote": None if f_vote is None else float(f_vote),
            # The structure-free focal COMMENSURABLE with the solve (the pinhole
            # vote, or the equidistant verdict under a fisheye context) — what the
            # finalization arbitrates the released focal against.
            "f_indep": None if f_indep is None else float(f_indep),
            # That same measurement bias-corrected, plus its own precision band —
            # what the finalization tests a structure candidate against.
            "f_center": None if f_center is None else float(f_center),
            "f_band": None if f_band is None else float(f_band),
            "f_released": float(f),
            "inl": float(inl),
            "flags": flags,
            "n_img": n_img,
            # The working image frame the poses are stated in (``keep`` maps working
            # image index -> data image index), for the full-frame lift the
            # hypothesis artifact and the distinctness test both take.
            "keep": keep_w,
            # The released geometry's FULL triangulation, by PASS cluster id — the
            # coverage claim's retained structure.
            "claim_pts": claim_pts,
            # The same geometry's structure in the SOLVE's own cluster space, with
            # the selection it belongs to — what the release artifact is written
            # from.  Both are the pass's own unless a group-local re-admission
            # supplied a denser working set, in which case the artifact shows what
            # the hypothesis actually solved while the claim stays comparable
            # across the loop.
            "release_pts": release_pts,
            "release_data": chosen["data"],
            # What that re-admission selected, for the rung manifest; None when
            # the solve ran on the pass's own admission.
            "local_admission": chosen["data"].get("local_admission"),
            # The commit-bar measurements of the attempt this pass committed, for
            # the hypothesis arbitration's qualification gates.  Reach is the
            # CAPTURE-level one: every hypothesis's reach is measured on the full
            # admission's covisibility graph, so the gates compare like with like.
            "kept": chosen["kept"],
            "reach": float(chosen["reach_capture"]),
            "reach_pass": float(chosen["reach"]),
            "spread": float(chosen["spread"]),
            # The lens: ``(coefficients, domain end, chart focal, equivalent
            # focal, report)``, all-zero coefficients on a refusal and None
            # where there was no domain to promote onto.
            "lens": lens,
            # PROVENANCE: the seed group this hypothesis's geometry grew from,
            # in the data image frame.  None when the seed was the rotation
            # core rather than a covisibility group.
            "seed_frames": chosen.get("seed_frames"),
            # The evolution record this release belongs to.
            "evo": chosen.get("evo"),
        }

    # Every candidate through the release, rank order.  The rank is advisory:
    # it orders the list a reader sees and nothing else.
    out = []
    for a in cands:
        if a is not cands[0]:
            print(
                f"\nladder candidate {a.get('ladder_rank')}: releasing "
                f"(level {a['level']}, kept {a['kept']}, "
                f"spread {100 * a['spread']:.1f}pp) [{elapsed():.1f}s]"
            )
        r = release_from(a)
        r["ladder_rank"] = a.get("ladder_rank", 0)
        r["collapsed"] = a.get("collapsed", 0)
        ev = a.get("evidence_summary") or {}
        r["rotation_disagreement_deg"] = ev.get("dis")
        r["far_beyond_at_ladder"] = ev.get("beyond")
        out.append(r)
    return out


if __name__ == "__main__":
    main()
