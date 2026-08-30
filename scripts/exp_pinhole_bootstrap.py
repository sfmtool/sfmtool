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
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
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
MIN_SPAN_BA = int(  # min distinct images for a cluster to become a point
    os.environ.get("SFMTOOL_MIN_SPAN_BA", "2")
)
MAX_CLUSTERS = int(os.environ.get("SFMTOOL_MAX_CLUSTERS", "10000"))  # BA-set size
F_GRID = [0.55, 0.7, 0.9, 1.2, 1.6]  # focal candidates, in units of max(w, h)
TRIM_PX = 4.0  # BA inter-round observation trim threshold
# Fraction-of-diagonal threshold scaling (SFMTOOL_FRAC_DIAG=1): the COARSE BA
# trims are geometry-dominated and should scale with framing (fractions of the
# image diagonal, referenced to the 270x480 seoul diagonal where they were
# validated); the FINAL trims stay in pixels (keypoint noise is ~constant in
# pixels regardless of resolution).  threshold = max(frac * diag, floor_px).
_PXS = 1.0  # set in main() once the image size is known
# Clusters are ordered for the BA cap and the admission tiers by span
# (highest-span-first); see load_clusters.
# Resection init from warp-determinant depth ratios: each member warp's
# sqrt|det| predicts the point's depth in the new image from its depth in
# the (posed) reference image, giving camera-frame 3D points -> closed-form
# trimmed Kabsch pose init (no neighbor-pose inits needed when it works).
DEPTH_INIT = os.environ.get("SFMTOOL_DEPTH_INIT", "0") == "1"
# Diagnostics: trace per-resection inliers in growth; optionally disable the
# periodic growth BA to attribute damage between resection and BA.
TRACE = os.environ.get("SFMTOOL_TRACE", "0") == "1"
GROW_BA = os.environ.get("SFMTOOL_GROW_BA", "1") == "1"
# Census endorsement threshold for the finalization's dual-candidate
# arbitration.  The census score is the Wilson LOWER BOUND on the fraction of
# bridge clusters the solve cannot explain, so it is a disagreement measure —
# lower is better — and 0.5 reads as "a majority of bridge clusters are left
# unexplained".  A structure candidate scoring at or above it is one the census
# cannot ENDORSE, and an unendorsed structure candidate must never override the
# structure-free vote, however much less it disagrees than the vote candidate:
# when both candidates are sick the comparison between their scores carries no
# information, and the vote NUMBER is the only estimate still standing on
# evidence the structure error cannot touch.  Calibration: endorsed-and-right
# sits at 0.117, while unendorsed-and-wrong sits at 0.654 / 0.841.
CENSUS_ENDORSE_MAX = float(os.environ.get("SFMTOOL_CENSUS_ENDORSE_MAX", "0.5"))
# Census flag threshold for the `census_echo` seed confidence flag
# (specs/core/analysis/cluster-census.md).  Same disagreement measure as above, read
# against a far lower bar: the endorsement gate asks whether a candidate is
# explicable enough to take a DECISION on, while the flag only reports what the
# raw cross-group evidence says about the seed that shipped.  A quarter of the
# eligible high-parallax bridges of one group pair left unexplained — with 95%
# confidence, the score being a Wilson lower bound — is more disagreement than a
# correctly-placed group pair produces, and it is the failure axis the focal
# flags cannot see: correct focal, wrong placement.
CENSUS_ECHO_FLAG_MIN = float(os.environ.get("SFMTOOL_CENSUS_ECHO_FLAG", "0.25"))
# Duplicate-point collapse (seed finalization).  A pair's collapse radius is
# THE SMALLER of the two points' DETECTION-TIME feature sizes — two detections
# farther apart than the extent of the finer of the two features are distinct
# features, not one feature found twice — floored at _COLLAPSE_R_MIN_PX
# (keypoint localization noise: below it, "same place" is not a claim the
# keypoints can support) and NOT capped: a cap contradicts the very principle
# the radius states.  A pair of 37.1 px / 14.9 px features asks for 14.9 px, and
# a 6 px cap answered 6.0, splitting a pair whose three remaining views sit at
# 6.0-6.2 px (20250907_000240907 points 1314/1605: 7/10 views inside the capped
# radius, 10/10 inside the patch-derived one).
#
# RADIUS CONVENTION: a "size" here is a SEMI-AXIS MEAN — a RADIUS, not a
# diameter.  _cluster_detection_sizes reads the `.sift` affine frame as
# 0.5 * (|col0| + |col1|), the mean of the frame's two SEMI-axes (COLMAP's own
# scale formula), and multiplies it by the patch refinement radius; the same
# half-extent convention holds for the patch frames the recon carries, whose
# columns project the patch HALF-vectors (observation_affine_shape).  So the
# number already IS the feature's half-extent — which is why _ARS_A_LO calls ONE
# det_size "the same feature found twice".  The pair radius is therefore
# min(size_a, size_b) with no further halving: an extra 0.5 would make the bound
# a QUARTER of the finer feature's footprint, twice as tight as the principle
# above states.  It was that tight, and it split pairs a human reads as one
# feature: on DaeguArtMuseumTreeStumpExhibit the pair behind seed-final points
# (188, 32) sits 20.6-26.2 px apart across its three shared views and agrees
# about depth to 0.16%, and its pair radius is 19.0 px halved against 38.0 px
# whole — the halving is the whole of why it did not merge (the pair behind
# (15, 12), 28-58 px apart and agreeing to 1.3%, is the same story one step
# removed: halved, each member is absorbed into a DIFFERENT cluster, and the
# MIN-over-members shrink below then puts the two clusters out of each other's
# reach).  SFMTOOL_COLLAPSE_RADIUS_SCALE multiplies min(size_a, size_b) before
# the floor (1.0 = the extent principle; 0.5 = the old quarter-footprint bound).
#
# THE DEFAULT STAYS 0.5: a five-dataset A/B (2026-08-01) rejected 1.0 —
# agglomeration transfers the pair radius to CHAINS (worst group span 24 ->
# 115 px on Daegu against a 26 px median detection radius), manufacturing
# aliased tracks (malignant conflicted pairs 3 -> 31 on 20240614_224422531
# with reproj-cull debris 85 -> 416 and focal +1.16%; 3 -> 35 on DnDTabletop)
# and roughly halving every shipped cloud, even though the two Daegu pairs do
# merge and the duplicate-heavy 20250907_000240907 improves monotonically
# (malignant 114 -> 75).  0.75 is graded, not safe.  The extent principle per
# PAIR needs a matching cap on the GROUP: a span bound at the finer member's
# extent would admit both Daegu pairs (spans 26 / 58 px within radii 38 / 95)
# while forbidding the chains — untried.
#
# The size MUST come from the detection (see _cluster_detection_sizes), never
# from `max_embedded_feature_size_per_point`.  That accessor projects the
# point's WORLD patch frame at the point's CURRENT depth, so it measures
# fs ~= f * patch_world_size / depth (measured against that model: ratio 0.851
# on 20250907_000240907, 0.957 on DnDTabletop) — a geometry read-out, not an
# image measurement.  Using it here closes a FEEDBACK LOOP: merging
# re-triangulates, which changes depth, which changes the radius, which reopens
# candidates, so the pass has no fixpoint even in principle.  Worse, it is
# self-amplifying on exactly the points that need care — on 20250907_000240907
# the 25 points with fs > 500 px sit at 0.11x the cloud median depth (the
# foreground-spike population), and the radii they claimed (up to 30306 px)
# chained 36-member groups spanning 471 px, manufacturing aliased tracks
# (malignant conflicted pairs 72 -> 175; DnDTabletop 0 -> 49).  The
# detection-time size is depth-independent and immune to all of it.
#
# SFMTOOL_COLLAPSE is a plain on/off switch, never a length: the radius itself
# is data-derived per pair (above), so there is nothing here for a caller to set
# but whether the pass runs at all.
COLLAPSE_ENABLED = os.environ.get("SFMTOOL_COLLAPSE", "1") == "1"
COLLAPSE_RADIUS_SCALE = float(os.environ.get("SFMTOOL_COLLAPSE_RADIUS_SCALE", "0.5"))
_COLLAPSE_R_MIN_PX = 3.0
# Safety bound on the collapse's outer fixpoint sweeps (each one must merge at
# least one pair or the loop stops; the incremental re-detection normally
# reaches the fixpoint inside the first sweep and the second only confirms it).
_COLLAPSE_MAX_ROUNDS = 8
# Aliased-track reconciliation (see _conflicted_pairs / _reconcile_aliased_tracks).
# A conflicted pair splits BENIGN / MALIGNANT at this median-depth mismatch: two
# distinct neighbors grazing each other's radius agree about depth to within a
# few percent, while an aliased pair — two tracks each stitched across different
# instances of a repeated pattern — cannot.
_ALIAS_MALIGNANT_MISMATCH = 0.05
# Assign -> retriangulate iterations.  The reassignment moves the points, which
# can change who wins the next round; in practice it settles in one or two.
_ALIAS_REPAIR_ROUNDS = 3
# Severity gate for the CULL of last resort: a pair that is still malignant after
# repair loses both members only if it disagrees about depth by at least this
# much.  Calibration is the observed severity spread, and the gate has to sit
# between its two regimes: the MILD regime (20240906_081206935, 743 malignant
# pairs at median 9% mismatch, 16.8% of the cloud involved, solve visually fine)
# must survive untouched, and the CATASTROPHIC regime (20250907_000240907, 71
# pairs at median 81%, p90 8503%) must not.  0.5 is an order of magnitude clear
# of the first and well below the second.
ALIAS_CULL_MISMATCH = float(os.environ.get("SFMTOOL_ALIAS_CULL_MISMATCH", "0.5"))
# CONTAINED-INCONSISTENT points (see _cull_contained_inconsistent).  Every
# pairwise test above scales its overlap bound by the SMALLER of the two
# features' radii, so a COARSE feature whose footprint swallows several fine
# ones whole is invisible to all of them: the fine points sit far inside the
# coarse footprint yet well outside the fine radius the bound is measured in.
# This pass scales the bound by the LARGER member instead and asks the one
# question the containment geometry makes answerable — do the two agree about
# range?  A patch is a piece of SURFACE, so a point whose keypoints land inside
# another patch's footprint in the images and whose range is GREATER sits BEHIND
# that surface while both are visible in the same pictures, which is impossible.
# The container is the member the pass drops, and only when SEVERAL independent
# contained points say the same thing (see _contained_cull_set).
CONTAINED_CULL_ENABLED = os.environ.get("SFMTOOL_CONTAINED_CULL", "1") == "1"
# Containment is GEOMETRIC and STRICT: the small member's whole footprint must
# fit inside the large one's, sep + r_small <= scale x r_large, measured at the
# MEDIAN separation over the shared images.  The strict form is what separates
# this pass from the collapse's regime — two features of similar radius satisfy
# it only at sep ~ 0, which is the duplicate/alias case the collapse and the
# reconciliation already own.  Measured on the six-dataset fleet's saved seeds:
# relaxing the test to "sep <= r_large" (mere overlap, no containment) multiplies
# DnDTabletop's flagged pairs by five and drags in 5.9% of the cloud, almost all
# of it similar-radius pairs that are duplicates, not containments.
CONTAINED_SCALE = float(os.environ.get("SFMTOOL_CONTAINED_SCALE", "1.0"))
# Range disagreement, same relative-mismatch convention as the alias
# reconciliation (|d_a - d_b| / min(d_a, d_b), medians over the SHARED images —
# the shared views are the only ones both points answer to).  It cannot be the
# 0.05 that decides "next to each other on the surface" elsewhere in this file
# (_ARS_DEPTH_TOL / _ALIAS_MALIGNANT_MISMATCH): those bound the spread between
# ADJACENT points, and a container's footprint is not adjacency — it is a whole
# angular extent, across which an OBLIQUE surface legitimately changes range by
# tan(tilt) x r_large / f.  On DnDTabletop's 115 px container at f = 2805 that is
# 4.1% for a 45 deg surface, i.e. 0.05 sits inside what obliquity alone explains.
# 0.10 clears it, and the flagged population starts far above it (the human-
# inspected container disagrees by 22%).
CONTAINED_MISMATCH = float(os.environ.get("SFMTOOL_CONTAINED_MISMATCH", "0.10"))
# A pair needs at least this many shared images: one shared image cannot tell a
# containment from a chance projection alignment (the same reason
# _ARS_MIN_SHARED is 2).
_CONTAINED_MIN_SHARED = 2
# OCCLUSION DIRECTION.  The violation is only a violation one way round: a small
# point NEARER than a large patch legitimately sits in front of it (a detail on
# top of a broad surface), while a small point FARTHER than the patch that
# contains it is hidden behind an opaque surface both cameras nonetheless
# imaged.  Requiring the container to be the nearer member keeps only the
# impossible half — and it is load-bearing, not decoration: without it the pass
# reaches 2.35% of 20250907_000240907's cloud against 1.64% with it.
CONTAINED_OCCLUSION = os.environ.get("SFMTOOL_CONTAINED_OCCLUSION", "1") == "1"
# Corroboration: how many contained points must contradict a container before it
# is dropped.  See _contained_cull_set — one witness names no culprit.
_CONTAINED_MIN_WITNESSES = 2
# Print the gate's full parameter sweep instead of one line — for A/B only.
CONTAINED_DEBUG = os.environ.get("SFMTOOL_CONTAINED_DEBUG", "0") == "1"
# TRACK-VIEW EVICTION.  Photometric view selection
# (specs/core/patch/patch-view-selection.md) vets CANDIDATE views against the track's
# reference appearance but admits the track's OWN members unconditionally — the
# matcher's word is taken as given.  So a member the photometry disowns rides
# through the whole finalization: it drags the reference, it feeds the congealing
# and the BA, and it ships.  Human inspection of a 20250907_000240907 seed found
# five points whose image-0 member sits on a flat-black region and scores ZNCC
# 0.07 / -0.09 / 0.19 / 0.19 / 0.37 against its own track's reference, where a
# CANDIDATE for the same points would have needed 0.41-0.62 to get in.
# This pass closes that asymmetry: a member that could not get in should not stay
# in.  See _evict_track_views for the two estimators the bar is taken against.
TRACK_EVICT_ENABLED = os.environ.get("SFMTOOL_TRACK_EVICT", "1") == "1"
# The two selection parameters the eviction bar is built from, held at
# select_views' own defaults so the bar a member must clear is exactly the bar a
# candidate faces (the embed runs the same machinery at the same defaults).
_EVICT_REL_ZNCC = float(os.environ.get("SFMTOOL_TRACK_EVICT_REL", "0.7"))
_EVICT_MIN_SELF_AGREEMENT = 0.3
# Members needed before the ROBUST bar (the member-score median) is trusted.  The
# median identifies the typical member only while a majority of members are good:
# at 5 it still survives two corrupted members (sorted, the 3rd is clean), at 4 it
# does not.  It is also the point at which a track can afford the pass — losing
# two members of five leaves the three the length-3 cull below requires.
# Set it above the longest track to disable the robust half and leave the
# self-agreement bar alone (the A/B arm the default was chosen against).
_EVICT_MIN_MEMBERS = int(os.environ.get("SFMTOOL_TRACK_EVICT_MIN_MEMBERS", "5"))
# DISAGREEMENT-RATIO GATE.  The relative bar above is a fraction of a ZNCC, so it
# can never demand more than _EVICT_REL_ZNCC (0.7) however crisp the track is: a
# member that coincidentally correlates at 0.82 with a reference its twelve
# siblings match at 0.995 is out of reach of the bar entirely.  Read as
# DISAGREEMENT (1 - zncc) the same member is a 35x outlier against a sibling
# median of 0.005 — a spread the bar cannot see because it is denominated in
# agreement, where every good member is crowded against 1.0.
# So: evict a member whose disagreement exceeds K x the track's median member
# disagreement.  A pure ratio would evict on microscopic spreads (a track at
# median 0.002 would disown a 0.98 member), so the test is against
# ``max(K x med_dis, _EVICT_DIS_FLOOR)`` — the floor is an absolute disagreement
# below which no member is ever an outlier, and it is what sets the gate's
# CEILING: no member above ``1 - floor`` ZNCC can be evicted by this gate at all.
# Calibrated at this site on the 20250907_000240907 seed against four
# human-verified tracks (23 members) and the junk members human inspection
# named.  DISAGREEMENT: the verified-good members top out at 0.131, the junk
# members sit at 0.184 / 0.189, so the floor is the midpoint of that gap.
# RATIO: the verified-good members top out at 7.9x, the junk member the gate has
# to reach is at 35x.  K sits 3x above the former and 1.4x below the latter —
# deliberately asymmetric, because the gate's population grows fast as K falls
# (0.57% of observations at 24, 0.79% at 20 on that seed, against 0.00-0.26%
# everywhere else) and a false eviction costs more than a missed one.
# Requires the same _EVICT_MIN_MEMBERS quorum as the median bar — a spread needs
# members to be estimated from.
_EVICT_DIS_RATIO = float(os.environ.get("SFMTOOL_TRACK_EVICT_DIS_RATIO", "24"))
_EVICT_DIS_FLOOR = float(os.environ.get("SFMTOOL_TRACK_EVICT_DIS_FLOOR", "0.16"))
# PER-VIEW FLATNESS GATE.  Both bars above are RELATIVE to the reference, and ZNCC
# normalizes contrast away, so a view holding a smooth gradient can score 0.75
# against a richly textured consensus purely on the sign of that gradient — a
# soft correlation carrying no localizable structure.  Nothing else in the
# finalization sees this: the patch-localizability score reads the CONSENSUS
# bitmap, which is exactly what such a member is not.  This gate reads the
# MEMBER'S OWN source content: the std of the image luma inside the member's
# projected patch footprint, over the std of the track's consensus bitmap.
# The ratio, not an absolute luma threshold, is the quantity — an absolute one
# would fail every legitimately dark or low-contrast scene, whereas a view that
# genuinely sees the surface carries the contrast that surface's own consensus
# was fused from.  Measured on the same seed: the verified-good members run
# 0.95-1.29 (median 1.11) and never fall below 0.95, while the featureless
# member reads 0.14 and an occlusion alias onto a low-contrast surface reads
# 0.21; the threshold is the geometric midpoint of that gap.  It reaches nothing
# but that tail — 0.00% of the members on three of the six A/B datasets, 0.003%
# and 0.06% on two more, 0.30% on the seed it was calibrated against (whose
# first frame looks at a different part of the scene from every other frame, and
# which the base bar already reads as 4.9% junk).
# Implemented over per-image integral images: O(1) per
# member after one summed-area pass per source image, and the ZNCC's own
# footprint (the projected surfel half-vectors) is the window, so the reading is
# of exactly the content that was scored.
_EVICT_FLAT_FRAC = float(os.environ.get("SFMTOOL_TRACK_EVICT_FLAT", "0.35"))
# Consensus bitmap pixels needed before its contrast is a usable denominator.
_EVICT_FLAT_MIN_PIX = 16
# Fronto-parallel priors — the two terms that pull a surfel normal toward facing
# its cameras.  WRITER_FRONTO_LAM damps the seed writer's own tilt solve (the
# out-of-plane slope b_z is the weakly observed, depth-like direction, so an
# undamped solve on a narrow-baseline window is noise); EMBED_FRONTO_PRIOR is
# the additive prior the photometric normal refinement carries, and since the
# refinement re-solves the plane it is that term, not the writer's, that the
# saved normals ultimately answer to.  Both are exposed for sweeping.
WRITER_FRONTO_LAM = float(os.environ.get("SFMTOOL_WRITER_FRONTO_LAM", "0.3"))
EMBED_FRONTO_PRIOR = float(os.environ.get("SFMTOOL_EMBED_FRONTO_PRIOR", "0.05"))
# Reprojection cull (seed finalization).  A finite point whose MEDIAN observation
# reprojection error exceeds max(_REPROJ_CULL_FLOOR_PX, mult x the cloud's median)
# is debris: the retriangulations in the collapse and the reconciliation are ray
# midpoints of whatever track they are handed, so a track that is still partly
# wrong produces a point that satisfies no view.  Nothing else in the chain looks
# at reprojection — the culls above test track length, patch size, depth sign and
# image overlap — so such a point ships (human inspection: point 2675 of a
# reconciled 20250907_000240907 seed, 4 views, depth 0.1 against a cloud median
# of 3.0, median reprojection 18-47 px).  Calibration is the observed separation:
# a healthy cloud sits at a 0.55 px median with p99 2.50 px, and the debris
# population starts around 3 px and runs to 388 px, so the floor alone separates
# the two on a well-conditioned solve and the multiple carries the bound up on a
# loose one.  0 disables the cull.
REPROJ_CULL_MULT = float(os.environ.get("SFMTOOL_REPROJ_CULL_MULT", "5.0"))
_REPROJ_CULL_FLOOR_PX = 3.0
# Keypoint-localization noise floor (px) the seed finalization hands the native
# points-at-infinity classifier.  It is the measurement noise a track's depth
# uncertainty is calibrated against: the classifier takes
# max(point reprojection error, this) / f as each ray's angular noise.  1.0 px
# is the core's own DEFAULT_NOISE_FLOOR_PX; the knob exists so a run can probe
# the classifier's sensitivity, not because the seed wants a different value.
SEED_INFINITY_NOISE_PX = float(os.environ.get("SFMTOOL_SEED_INFINITY_NOISE_PX", "1.0"))
# INFINITY DEMOTION GATE (SFMTOOL_INF_DEMOTION_GATE=1, default ON).  Vetoes a
# demotion whose track the BEARING MODEL cannot itself explain, and at the
# decisive post-BA site only when the finite position it would ship is also
# depth-plausible — see _inf_gate_veto.  0 restores the ungated behaviour (the
# gate still measures and reports what it would have vetoed, so an A/B keeps the
# same evidence lines).
INF_DEMOTION_GATE = os.environ.get("SFMTOOL_INF_DEMOTION_GATE", "1") == "1"
# Keypoint RE-LOCALIZATION (seed finalization, the last step before the final BA;
# see _relocalize_keypoints).  A patch's keypoints are localized THROUGH its warp
# and the warp is derived from its NORMAL, so a keypoint answers to the geometry
# it was localized under; nothing upstream re-localizes once the frames settle,
# and the resulting stale keypoints drag the triangulated position.  Measured on
# DnDTabletop's shipped seed: point 721 carried four of its eight keypoints ~6.2
# px off, and re-localizing them moved its depth 2706.3 -> 2599.4 against a
# 12-neighbour median of 2596.5 — its offset from the local surface went +109.8 ->
# +2.9 — while a healthy neighbour whose patch overlaps it in image space (747)
# moved 0.75 px and 2.3 depth units.  The pass is inert where the geometry was
# already right, which is why it can be applied to the whole cloud.  0 disables it.
RELOCALIZE_ENABLED = os.environ.get("SFMTOOL_SEED_RELOCALIZE", "1") == "1"
# The discrete localizer's drop gate: a view whose re-localized keypoint lands
# more than this many SOURCE px from the point's own projection is not written
# back (it keeps its stored keypoint).  The gate is a correction CEILING as much
# as an outlier test, and the corrections this pass exists to make measured
# 6.1-6.3 px on DnDTabletop — the production default of 3.0 rejects precisely the
# population the pass is for.  8.0 clears the measured corrections with margin
# while staying inside the localizer's own search window (6 grid px, which is
# ~14-27 source px at the seed's patch sizes), so a keypoint that jumped to a
# different feature entirely is still refused.
RELOCALIZE_MAX_SHIFT_PX = float(
    os.environ.get("SFMTOOL_SEED_RELOCALIZE_SHIFT_PX", "8.0")
)
# Re-localization closes a loop (normals <- positions <- keypoints <- normals), so
# it could be iterated.  It is not, on measurement: a second pass on DnDTabletop
# accepted a further 1153 points but moved only 5 observations past 3 px and the
# median position by 0.02% of the cloud's depth (pass 1: 933 observations, 0.38%),
# leaving the focal identical and every quality metric at the fourth decimal
# (depth-outlier fraction 7.29 -> 7.26%, reprojection median 0.604 -> 0.602 px);
# vid2 behaved the same.  One pass takes the corrections; the second pays a full
# localize + refine for noise.  The knob exists to re-measure that claim.
RELOCALIZE_PASSES = int(os.environ.get("SFMTOOL_SEED_RELOCALIZE_PASSES", "1"))
# Acceptance floor on the view side: a candidate needs at least this many
# re-localized views AND at least this fraction of the point's observations.  3
# views is the seed's own standing floor (the length-2 cull refuses to ship a
# point triangulated from fewer), and the half-of-the-track fraction keeps the
# re-triangulation from being decided by a minority while the majority of the
# track keeps keypoints that answer to the old geometry.
_RELOCALIZE_MIN_VIEWS = 3
_RELOCALIZE_MIN_KEEP_FRAC = 0.5
# ADAPTIVE ROBUST SURFEL (ARS) normals — the seed's normal estimator, REPLACING
# the photometric five-patch path for the saved normals (see _ars_normals).
#
# The support is the point cloud's own IMAGE-SPACE adjacency: two finite points
# are neighbours when, in a majority of the images that see both, their keypoints
# sit inside an annulus measured in DETECTION-radius units (depth-free — the
# radius is the detection-time patch half-extent _cluster_detection_sizes already
# carries, never f x patch_world / depth, which would read out the very geometry
# the estimator is trying to find).  The plane is p-ANCHORED (surfel semantics —
# p's tangent plane, not the neighbourhood's best plane) and fitted on the UNIT
# neighbour directions, so a neighbour contributes its angular deviation once
# rather than in proportion to its distance.
#
# Calibration (four seed-like artifacts, held-out tilt + the independent
# photometric normal + DnDTabletop's dominant plane, every method scored over the
# SAME population of finite points).  On DnDTabletop's on-table points — the one
# knowable answer — the median angle to the dominant plane runs stored 22.27 deg
# / five-patch-where-it-applied 5.70 deg / ARS 2.35 deg, worst decile 62.32 /
# 46.24 / 17.07.  ARS is best or tied-best on held-out agreement on all four
# artifacts and beats five-patch's worst decile on all four.  It is un-estimable
# on 0.2-2.0% of finite points against five-patch's 15-26% (which silently keeps
# the STORED normal there and is therefore not an estimate at all).
ARS_ENABLED = os.environ.get("SFMTOOL_ARS_NORMALS", "1") == "1"
# Support radius, in detection radii.  ONE graph, and the fit uses ALL of it: the
# robust term below is what makes a wide support safe, and wide support is what
# the estimate needs — per-point ADAPTIVE support (bandwidth from the K-th
# nearest neighbour, so dense neighbourhoods fit locally) was swept and LOST,
# 2-7x worse on the dense artifact (dominant-plane error 27.5 / 18.6 deg at
# K = 8 / 16 against 3.7 deg fixed-wide) and no better anywhere else: small
# supports are dominated by the triangulation noise of the neighbours' own
# positions, and it is angular leverage, not locality, that carries the fit.
# Widening further for the sparse points ("reach") was measured too and does not
# earn its cost — it moves the medians by <= 0.1 deg while being the single
# largest wall-clock item (11.6 s on DnDTabletop to serve 48 points).  The
# redescender also makes the estimator far less sensitive to this number than a
# hard annulus is: over B = 8..20 a hard annulus loses 3.8 deg of held-out tilt
# on DinoDogToy while ARS loses 1.2.
ARS_B_MAX = float(os.environ.get("SFMTOOL_ARS_B_MAX", "10.0"))
# Inner annulus bound: closer than one detection radius is the SAME feature found
# twice, which the duplicate collapse already merges.
_ARS_A_LO = 1.0
# The calibration's radius unit: 5.5 x the SIFT affine scale (embed-patches
# --patch-size 11, halved).  `det_size` is refine_radius x the same scale, so it
# is converted into this unit before B_MAX is applied — otherwise a workspace
# whose cluster-patches ran at a different patch size would silently be using a
# different support radius than the one the numbers above were measured at.
_ARS_RADIUS_MULT = 5.5
# A pair must share at least this many images (one shared image cannot tell "next
# to each other on the surface" from "one behind the other along the viewing
# ray"), hit the annulus in a majority of them, and agree about range from the
# shared cameras to within the depth tolerance — which drops the aliased class.
_ARS_MIN_SHARED = 2
_ARS_MAJORITY = 0.5
_ARS_DEPTH_TOL = 0.05
# The graph itself is built by the native observation-adjacency primitive
# (`sfmtool._sfmtool.analysis.build_observation_adjacency`, see
# `specs/core/analysis/observation-adjacency-graph.md`); the fit comes from the native
# surfel kernel (`sfmtool._sfmtool.analysis.estimate_adjacency_surfel_normals`,
# see `specs/core/analysis/adjacency-surfel-normals.md`).
# Robust fit: Tukey redescender on the SCALE-FREE off-plane residual
# r = |d . n| / |d| (the sine of the tilt), robust scale
# sigma = 1.4826 x median(r) floored at sin(2 deg) — the floor is keypoint /
# triangulation noise, below which "the neighbourhood is planar to within
# nothing" is not a claim the data supports and an unfloored sigma would start
# rejecting the whole support.  A neighbour that leaves p's tangent plane
# (curvature, a different surface, a mismatch) redescends to ZERO weight instead
# of tilting the plane.  Three IRLS passes: the weights are settled by then.
ARS_IRLS_ITERS = int(os.environ.get("SFMTOOL_ARS_IRLS_ITERS", "3"))
ARS_TUKEY_C = float(os.environ.get("SFMTOOL_ARS_TUKEY_C", "4.685"))
ARS_SIGMA_FLOOR_DEG = float(os.environ.get("SFMTOOL_ARS_SIGMA_FLOOR_DEG", "2.0"))
# DETERMINACY.  N_eff = (sum w)^2 / sum w^2 is the effective neighbour count
# after redescending; the occupied-sector count and the IN-PLANE ANISOTROPY
# lam1/lam2 of the weighted scatter catch what a raw count cannot see — a
# COLLINEAR neighbourhood leaves the normal free to rotate about the line, and
# fits it perfectly while determining nothing.  A point failing the predicate is
# handed to the expansion stage, never quietly kept.
ARS_DET_NEFF = float(os.environ.get("SFMTOOL_ARS_DET_NEFF", "4.0"))
ARS_DET_SECTORS = int(os.environ.get("SFMTOOL_ARS_DET_SECTORS", "3"))
ARS_DET_ANISO = float(os.environ.get("SFMTOOL_ARS_DET_ANISO", "0.10"))
# Tangent sectors.  The basis is orthogonal to the point's MEAN VIEWING
# DIRECTION, not to any normal estimate, so nothing here is circular.
_ARS_N_SECTORS = 8
# EXPANSION — the five-patch machinery, AIMED rather than blanket.  For an
# under-determined point only, one synthetic patch is congealed into each EMPTY
# tangent sector at one patch diameter out, and the fit is redone with the
# survivors included.  On the calibration artifacts it congealed 0.5-27.9% as
# many patches as the blanket estimator's four-per-point and took the
# un-estimable rate from 0.5-4.5% down to 0.2-2.0%.
#
# The budget caps the stage at this many patches per finite point (blanket is
# 4.0), spent most-under-determined-first; 0 disables the stage.  It BINDS on a
# sparse seed: the calibration artifacts asked for at most 0.87 patches per
# point, but a seed whose points mostly sit alone asks for 2.8-2.9
# (DaeguArtMuseumTreeStumpExhibit 755/263, 20250907_000240907 7133/2514 — nearly
# every sector empty at nearly every point).  Measured against a lifted budget,
# what a cut costs is DETERMINACY, not coverage: on those two seeds it moved 31
# and 383 points from expansion-resolved to weak — both classes keep a FITTED
# normal — and left the fronto-parallel fallback count identical at 35 and 39.
# The fallback rate is set by the adjacency graph, not by this number.
ARS_EXPAND_BUDGET = float(os.environ.get("SFMTOOL_ARS_EXPAND_BUDGET", "2.0"))
_ARS_EXPAND_DIAM = 2.0  # patch diameters from p, along the empty sector's bisector
_ARS_EXPAND_SEARCH = 6.0  # grid px; the localizer's own default
_ARS_EXPAND_MAX_SHIFT_PX = 8.0
_ARS_EXPAND_MAX_REPROJ_PX = 2.0
_ARS_EXPAND_MIN_VIEWS = 3
# The reach is AIMED AT UNCLAIMED IMAGE AREA.  An image-space direction that is
# already claimed by observations — but whose owning tracks did NOT become graph
# neighbours of p — was almost certainly rejected by the adjacency range vet,
# i.e. it is another surface at a different depth.  Congealing a synthetic patch
# into that direction manufactures exactly the off-surface contaminant the Tukey
# loop then has to reject, so a candidate whose direction lands on claimed area
# in a majority of p's views is dropped before it costs budget or a congeal.
# The claim map is the observation-coverage atlas
# (`sfmtool._sfmtool.analysis.ObservationCoverage`, see
# `specs/core/analysis/observation-coverage.md`).  Set to 0 for the un-gated behaviour.
ARS_EXPAND_COVERAGE = os.environ.get("SFMTOOL_ARS_EXPAND_COVERAGE", "1") == "1"
# The congealing of the aimed candidates comes from the native candidate-track
# spawner (`sfmtool._sfmtool.patches.spawn_candidate_tracks`, see
# `specs/core/patch/candidate-track-spawning.md`) — one batch call owning the
# localize -> refine -> triangulate pipeline and the acceptance gates.
# A re-oriented point's baked consensus bitmap stops describing it once its frame
# turns, so it is marked MUTATED for the end-of-finalization re-fuse.  The gate is
# the frame contract's own resolution (_check_frame_contract calls a normal and
# its frame in agreement below 1 deg): a turn smaller than that is not a change
# the stored geometry can even represent, so it does not buy a re-render.
_ARS_MUTATE_MIN_DEG = 1.0
# DENSIFICATION BY PROMOTION — the expansion's survivors become REAL points.
#
# The expansion congeals a synthetic patch exactly as a real track is congealed
# and then throws the result away, keeping only its 3D position as a fit
# neighbour.  That makes the support INVISIBLE: a candidate is seeded in the
# parent's own plane hypothesis, and at weak parallax the reprojection gate is
# nearly blind along the ray, so a candidate that never moved off the plane
# triangulates back onto it and CONFIRMS the hypothesis that placed it.  Measured
# on 20250907_000240907, 44% of the grazing-confident normals have fewer than
# four real graph neighbours and 5.5% have none at all — their determinacy is
# carried entirely by scaffolding nothing else can see.
#
# Promotion converts that support into evidence the rest of the pipeline can
# audit: a promoted candidate carries its OWN observations into the
# reconstruction and then faces the identical downstream gauntlet as every other
# point — keypoint re-localization, track-view eviction, the BA, the infinity
# classifier and every cull.  Support that cannot survive that is support the fit
# should not have had, and support that does survive is no longer synthetic.
# Set to 0 for the expansion's ephemeral-extras behaviour (bit-identical).
ARS_PROMOTE = os.environ.get("SFMTOOL_ARS_PROMOTE", "1") == "1"
# Promotion gates, ON TOP of the spawn primitive's own acceptance (>= 3 views,
# finite triangulation in front of every camera, RMS <= _ARS_EXPAND_MAX_REPROJ_PX).
# Being good enough to inform a robust plane fit is a lower bar than being a
# shipped landmark: a fit neighbour contributes one redescendable direction among
# many, while a point is adjusted, re-classified and rendered on its own.
#
# VIEWS — one above the spawn floor.  A 3-view candidate is exactly what the
# post-eviction length-3 cull retires the moment any single view is disowned, so
# promoting one buys a point with no margin at all.
_ARS_PROMOTE_MIN_VIEWS = int(os.environ.get("SFMTOOL_ARS_PROMOTE_MIN_VIEWS", "4"))
# REPROJECTION — half the spawn gate.  The spawn bound is a "this congealed at
# all" test; the seed's own reprojection cull judges shipped points against the
# cloud's median (typically well under a pixel), so a 2 px candidate would be
# admitted here only to be culled there.
_ARS_PROMOTE_MAX_REPROJ_PX = float(
    os.environ.get("SFMTOOL_ARS_PROMOTE_MAX_REPROJ", "1.0")
)
# PARALLAX — the gate the reprojection bound structurally cannot be.  RMS
# reprojection is measured against the rays that PRODUCED the position, so a
# candidate whose views are nearly collinear reprojects perfectly wherever along
# the ray it happens to sit, including exactly where the parent's plane
# hypothesis put it.  The widest angle between any two of its own observation
# rays is what says a triangulation happened; below the bar the position is the
# seeding, not a measurement.
_ARS_PROMOTE_MIN_PARALLAX_DEG = float(
    os.environ.get("SFMTOOL_ARS_PROMOTE_MIN_PARALLAX", "1.0")
)
# LOCALIZABILITY — the gate none of the three geometric ones can be.  Views,
# reprojection and parallax all judge the candidate ALONG ITS RAYS; none of them
# looks at the image content the candidate was congealed on.  A patch congealed
# on a line or on a flat wash slides LATERALLY at no photometric cost, so it
# triangulates cleanly from many views at wide parallax and still cannot pin a
# keypoint: promoted, it becomes a real track whose observations the BA is free
# to drag along the weak axis, and a real graph neighbour whose position is a
# smear.  The structure tensor of the CONGEALED consensus is what says whether
# the patch can hold a keypoint at all, and it is the same reading real points
# already had to pass at embed time (`_cull_by_localizability`, same tau, same
# grid-px unit, same NaN-keeps convention) — promotion was the one door into the
# cloud that skipped it.  In patch-grid px; 0 disables the gate (bit-identical
# to the pre-gate order).
#
# PROMOTION ONLY, not the fit's ephemeral extras.  Under ARS_LATE_FIT (the
# default) the extras are not consumed at all — the early stage returns straight
# after promotion and the extras re-fit below it never runs — so gating them
# would change nothing by construction; the aiming they might have influenced is
# decided BEFORE the congealing, by the graph fit that precedes the expansion.
# With the late fit off they do feed the write, and they are still left ungated
# on the same asymmetry the other three gates rest on: a laterally-slideable
# patch still carries a usable DIRECTION from its triangulated centre to its
# parent, which is all the plane fit reads, and the fit's IRLS redescends what
# disagrees.  Scoring them would also mean fusing a consensus for every congealed
# survivor rather than for the promoted subset (on 20250907_000240907, 4639
# against 738).
_ARS_PROMOTE_MAX_SIGMA_GRID = float(os.environ.get("SFMTOOL_ARS_PROMOTE_LOC", "0.35"))
# ANISOTROPY BOUND on the localizability criterion (SFMTOOL_LOC_MAX_ANISO), the
# second half of that criterion and the one sigma_pos structurally cannot be.
# sigma_pos = sigma_noise / sqrt(lam2) reads the WEAK axis in ABSOLUTE terms, so
# it answers "is this patch pinned well enough", and a high-contrast streak
# answers yes: its lam2 is small relative to lam1 but large in absolute terms,
# because the contrast that steepens the strong axis leaks into the weak one.
# Measured on the 20250907_000240907 seed: the point whose every member bitmap is
# a same-direction streak scores sigma 0.253 against a 0.35 threshold — an honest
# pass — at lam1/lam2 = 73, while four human-verified good tracks on the same
# seed read 1.3-4.3.  What the bound adds is the SHAPE: a
# patch stretched into a line has one usable direction, its keypoint slides along
# the other at no photometric cost, and no amount of contrast fixes that.
# The reading is scale-free (a ratio of eigenvalues of the same tensor), so one
# number transfers across datasets and resolutions the way the grid-px tau does.
# Applied wherever the localizability criterion is: the ARS promotion gate at
# congeal time and the late re-cull on the shipped bitmaps.  0 (or inf) disables
# it — bit-identical to the sigma-only criterion.
#
# CALIBRATION, read off the distributions the way the eviction gates' were.
# ABOVE: the four human-verified good tracks on 20250907_000240907 top out at 5.6
# at final geometry (1.7 / 2.3 / 3.5 / 5.6), and a healthy cloud's whole
# distribution sits below: DnDTabletop runs p50 2.3 / p90 6.2 / p99 23.7.
# BELOW: the streak exemplar reads 73 on the pre-vet artifact and 87.7 once the
# eviction replay has re-fused it from its surviving members.  75 sits 13x above
# the widest verified-good track, 3x above the healthy p99 and just above the p99
# of the healthiest video capture in the A/B set (vid2, 73.1) — so on a healthy
# cloud it reaches the top ~1% (measured: 0.3% of DnDTabletop, 1.0% of vid2,
# 3.3% of 20250907_000422554) — while taking the exemplar and the streak
# population of the two captures that carry one (5.4% of 20240614_224422531,
# 12.8% of 20250907_000240907).
#
# It is deliberately NOT tighter, and that is a casualty judgement rather than a
# taste one.  Two things were measured.  (1) The cost: at 50 the worst capture
# loses 25.0% of its finite cloud to the late vet — exactly the campaign's stop
# threshold — against 21.1% at 75.  (2) What the extra casualty BUYS: a grazing
# frame samples a compressed strip of source image and stretches it across the
# consensus grid, so a high shipped ratio can be a frame smear rather than a line
# feature, and the frame-free control (the SOURCE-image structure tensor over an
# isotropic window at the member's own keypoint) says the 20-100 band is
# dominated by exactly that — on 20240614_224422531 it is 98-100% grazing frames
# at a source ratio of 1.6-1.8, i.e. no genuine line features at all.  A smeared
# shipped patch is unreliable whatever put the smear there, so the bound culls it
# either way; but the increment below 75 spends its casualty on the confound, and
# the confound's real fix is upstream of this pass.
LOC_MAX_ANISO = float(os.environ.get("SFMTOOL_LOC_MAX_ANISO", "75"))
# Matches the core scorer's own lam2 floor, so a flat patch's ratio is the same
# (very large) number here as the sigma_pos it produces there.
_LOC_LAM2_FLOOR = 1e-12
# WHERE THE FIT WRITES FROM.  ON (default): the plane fit, the frame/normal write
# and the confidence run ONCE, at the END of the finalization, on the cloud that
# actually ships.
#
# The fit is a NEIGHBOURHOOD estimator, so what it is worth is exactly what its
# neighbourhood is worth.  Run before the gauntlet it consumes geometry the
# gauntlet is about to reject — points the BA has not adjusted, tracks the
# eviction is about to disown, positions the post-BA collapse is about to merge —
# and it never sees the promoted candidates' support at all, because promotion
# happens after the fit that the promotion exists to feed (measured on
# 20250907_000240907: the >80 deg-with-confidence-255 population was unchanged
# within a run).  Worse, a promoted point is born wearing its parent's FITTED
# frame, so a grazing parent ships a grazing twin beside it at confidence 0 and
# nothing ever refits either.
#
# Under the late fit the expansion and the promotion still run early — they
# create the tracks the gauntlet has to vet — but they are aimed by a SCRATCH
# fit whose normals, frames and determinacy are never written anywhere; the only
# thing that reaches the cloud from the early site is the promoted tracks
# themselves, wearing the embed-stage frame they were congealed under.  The one
# fit that writes then runs on post-BA positions, post-cull membership and the
# surviving promoted points as ordinary graph neighbours, and gives every one of
# them — promoted included — its own normal and its own honest confidence.
#
# 0 restores the single pre-BA site (bit-identical to the pre-late-fit order).
ARS_LATE_FIT = os.environ.get("SFMTOOL_ARS_LATE_FIT", "1") == "1"
# LATE VETTING (SFMTOOL_LATE_VET=1, default ON).  Every content gate in the
# finalization runs BEFORE the BA — it has to, because an eviction only reaches
# the adjustment from there — and therefore judges the EMBED stage's frames and
# the bitmaps fused under them.  What ships is a different artifact: the BA moves
# every point, the late ARS fit re-aims every frame off the adjusted cloud, and
# the closing re-fuse re-renders each mutated consensus THROUGH those new frames.
# A normal that turned takes the sampling window with it, so a bitmap that was
# crisp when it was vetted can ship as a smear, and a member that agreed with its
# track under the old frame can disagree under the new one.
#
# Measured on the shipped 20250907_000240907 seed (locgate round, 110e63e2):
# 362/2906 finite points (12.5%) fail the embed cull's own localizability
# threshold on their SHIPPED bitmaps, and re-scoring the pre-BA eviction's own
# gates at final geometry disowns members it had cleared — three of the points
# human inspection flagged carry an image-0 member at ZNCC 0.48 / 0.58 / 0.07
# against an eviction bar of 0.54 / 0.69 / 0.63.
#
# So the two vettings run a SECOND time, at the end, on the frames and the
# bitmaps that will actually ship: the track-view eviction replay and the
# localizability re-cull (see _late_vet).  Nothing here reaches the BA — that is
# the point; this is the pass that judges the file rather than the input to the
# adjustment.  0 restores the single pre-BA site (bit-identical).
LATE_VET = os.environ.get("SFMTOOL_LATE_VET", "1") == "1"
# The late re-cull's sigma_pos threshold, in patch-grid px.  Held at the embed
# cull's and the promotion gate's own tau: the whole point is that this is the
# SAME criterion those two applied, re-applied to what ships, so a different
# number here would make it a different (and unmotivated) test.  0 disables the
# sigma half of the re-cull, leaving the eviction replay and the aniso bound.
_LATE_VET_MAX_SIGMA_GRID = float(os.environ.get("SFMTOOL_LATE_VET_LOC", "0.35"))
# MEMBER COHERENCE in the late vet (SFMTOOL_MEMBER_COHERENCE = the pairwise-ZNCC
# BAR; default 0.65, the core primitive's own calibrated default; 0 disables the
# stage, bit-identical to the pre-coherence order).
#
# Every membership test above this line scores a member against the track's FUSED
# CONSENSUS, and that test is structurally blind to a balanced chimera: the
# consensus is built FROM the members, so a track whose members image two
# surfaces gets a compromise blend that flatters both sides, and the more even
# the split the better it flatters them.  The disagreement is only visible
# BETWEEN members, which is what the pairwise matrix reads
# (`specs/core/patch/member-coherence-validation.md`): a k x k ZNCC table over the
# point's own frame, then the max-support block, the separation margin and a
# verdict of keep / split / retire.
#
# ADDITIVE, not a replacement.  The two tests act on almost disjoint populations
# — measured Jaccard 0.049 between the members the eviction replay disowns and
# the ones the coherence vet rejects — because they are asking different
# questions of the same track.  The evictor stays exactly as it was.
MEMBER_COHERENCE_BAR = float(os.environ.get("SFMTOOL_MEMBER_COHERENCE", "0.65"))

# SELF-NORMALIZED ADMISSION BAR (SFMTOOL_MC_SELF_K, 0 disables and restores the
# absolute rule byte-for-byte).
#
# MEMBER_COHERENCE_BAR is absolute, and an absolute threshold can only be
# calibrated against one kind of disagreement.  It was calibrated on CROSS-SURFACE
# chimeras, whose members correlate 0.2-0.5 against the core, and 0.65 separates
# those cleanly.  It structurally cannot reach the other family: an OCCLUDING
# member on a repeating-texture surface (a railing in front of siding) shares the
# core's dominant streak structure and correlates 0.85-0.95 with it, while the
# core agrees with ITSELF at 0.98-1.00.  The block structure is real and sits
# entirely above the bar.
#
# So each track re-derives its own thresholds from its own coherence — see
# `specs/core/patch/member-coherence-validation.md`.  The kernel sweeps the block at the
# absolute bar, measures that block's intra-pair centre c and (one-sided) scatter
# sigma, and re-runs at max(bar, min(c - K*sigma, 0.99)) with the separation
# margin floored at min(margin_gate, sigma).  A noisy or drifting track has a
# large sigma and collapses back to the absolute pair, which is what keeps this
# off drift chains and off the human-approved flat-tabletop tracks.
MEMBER_COHERENCE_SELF_K = float(os.environ.get("SFMTOOL_MC_SELF_K", "1.5"))

# MULTI-SCALE EXONERATION (SFMTOOL_MC_EXON, 0 disables and leaves the raw
# self-normalized rule; inert whenever SFMTOOL_MC_SELF_K is 0).
#
# The self-normalized bar cannot help evicting some innocents along with the
# occluders: a member that trails a tight core because its frame is SOFT trails it
# by the same amount as a member that trails it because it is looking at a
# railing, and the full-resolution pairwise ZNCC reports one number for both.
# Measured on this capture set, that collateral is almost entirely blur — on
# DnDTabletop the self-bar's on-table casualties are the soft frames of the
# capture, image 11 above all.
#
# The two stop looking alike as soon as the fine detail is taken away.  An
# occluder's disagreement is STRUCTURAL and survives blurring both sides; a soft
# frame's is SPECTRAL and evaporates, because its low frequencies ARE the core's.
# So the kernel re-measures each relative-flagged member's agreement deficit on a
# HALF-scale copy of the same renders and spares the ones whose deficit does not
# survive — see `specs/core/patch/member-coherence-validation.md`.  Only the relative
# term's evictions are exonerable; a member the absolute bar rejects images a
# different thing and blur is no defence against that.
#
# The threshold is high (0.90) because the test is SURVIVAL, not decay: across one
# halving an occluder keeps 0.85-1.00 of its deficit while a soft frame's already
# slips.  Measured on the two labelled populations, the half scale separates them
# and the quarter scale does not -- a 6x6 grid washes out the occluder's structure
# along with the blur.
MEMBER_COHERENCE_EXON = float(os.environ.get("SFMTOOL_MC_EXON", "0.90"))

# PER-OBSERVATION SHARPNESS CONFIDENCE (SFMTOOL_MC_SHARP_SCALE, 0 disables the
# whole column and ships a file without it).
#
# The multi-scale machinery above measures, for every scored member and not just
# the suspects, how much of that member's disagreement with its track's consensus
# exists ONLY at fine scale.  That quantity is a sharpness reading: it is ~0 for a
# member whose disagreement is scale-free (an occluder is SHARP, just wrong) and
# grows for a member that agrees coarsely and not finely, which is what defocus
# and motion blur do.  It is written to `tracks/observation_confidence` and
# nothing downstream reads it — see `specs/formats/sfmr-file-format.md`.
#
# The scale is ABSOLUTE and fixed, not per-track, because the column has to be
# comparable across points and across files; a per-track normalization would make
# every track's softest member look equally soft.  0.05 ZNCC of fine-scale-only
# deficit is the zero point, calibrated on a capture whose soft frames are known:
# its crisp frames' 99th percentile lands near 200 and its visibly soft frame's
# median near 50, i.e. the bottom fifth.
MC_SHARP_SCALE = float(os.environ.get("SFMTOOL_MC_SHARP_SCALE", "0.05"))


def _sharpness_to_confidence(deficit):
    """Quantize the fine-scale-only agreement deficit to the format's uint8.

    `0` is RESERVED for "no data-derived support" — an observation nothing
    measured — so a measured value never lands there: the measured range is
    1..255, worst to best.  Anything at or below zero deficit is fully sharp."""
    d = np.asarray(deficit, dtype=float)
    frac = np.clip(d / max(MC_SHARP_SCALE, 1e-9), 0.0, 1.0)
    out = np.rint(255.0 - 254.0 * frac).astype(np.uint8)
    return np.where(np.isfinite(d), out, 0).astype(np.uint8)


# KEYPOINT-ANCHORED VETTING PHOTOMETRY.
#
# Every photometric membership test in the vetting chain renders a member's patch
# somewhere in that member's source image, and there are two candidate places: the
# point's REPROJECTION (where the current geometry says it should be) and the
# member's STORED KEYPOINT (where the feature actually is, as the matcher matched
# it and the congealing/localization refined it).  The two differ by the member's
# reprojection residual.
#
# Sampling at the reprojection carries that residual into the render — the window
# slides off the member's own content by exactly it — so the ZNCC comes back
# depressed by MISALIGNMENT rather than by disagreement about what is imaged.
# That is a geometric error being charged to a photometric measure, and the
# reprojection cull and the BA already own it.  Measured on the 907 seed: pt
# 1149's image-14 member carries a 0.92 px residual against a 7.9 px patch
# half-width (12%) and scores 0.28-0.73 pairwise against siblings imaging exactly
# the content it does.
#
# So every site that samples a MEMBER — the eviction replay's reference and its
# member scores, the coherence matrix, and the flatness gate's footprint box — is
# anchored at the stored keypoint.  CANDIDATE views in the eviction replay stay
# projection-anchored by necessity: a candidate has no observation, so it has no
# keypoint.  The bitmap re-fuse and everything downstream of it (the
# localizability re-cull, the promotion gate) were already keypoint-anchored — the
# render-only refiner is seeded at the stored keypoints — so this brings the
# member tests into line with the consensus they are judged beside.

# The whole script works in the CANONICAL camera frame (-Z forward, +Y up):
# poses are canonical world->camera, 3D points are world points, observations
# are FULL (un-centered) pixel coordinates, and every projection goes through
# the native `CameraIntrinsics` batch functions.  The world frame is the
# COLMAP-world gauge inherited from the affine factorization (irrelevant to
# the reprojection residuals and absorbed by the eval's similarity alignment);
# only the writer rotates it by W to reach the .sfmr canonical world.
_CAM_WH = (
    None  # (w, h) of the shared pinhole; set in main() from the uniform image dims
)


# ── Fisheye seed camera context (scripts/notes-fisheye-seed.md, Phase 1) ─────
#
# The finalization's twin of ``exp_fast_seed``'s camera context: a per-run
# (model, focal) pair behind every camera this script builds.  Default
# SIMPLE_PINHOLE — the code path this script has always run, byte-identical.
#
# The context is INSTALLED BY THE CALLER, never inferred here: the confirmed
# equidistant verdict lives in stage 1, and ``exp_fast_seed`` hands it over
# through ``set_camera_context`` before ``finalize_seed_from_dict``.  A
# standalone bootstrap run therefore stays pinhole unless the caller says
# otherwise.  As in stage 1, the fisheye model is EQUIDISTANT_FISHEYE (the
# SIMPLE_PINHOLE analog for `theta = r/f`, closed form both ways with an
# analytic pixel Jacobian), and only the primitives that build their camera
# through ``make_cam`` become equidistant — the photometric embed's patch
# geometry is Phase 4.
_CAM_CONTEXT = {
    "model": "SIMPLE_PINHOLE",
    "focal": None,
    "bspline": None,
    "theta_max": None,
}


def set_camera_context(model, focal=None, bspline=None, theta_max=None):
    """Install the per-run camera context (see the block comment above).

    ``bspline`` / ``theta_max`` are the radial-spline coefficients and the END
    OF THEIR DOMAIN, ignored by every other model.  The domain end is measured
    in the base model's own radial coordinate — incidence angle under the
    equidistant base, normalized image-plane radius under the pinhole one (see
    ``_SPLINE_MODEL``) — and the keyword keeps its fisheye name over both.  The
    finalization's spline rung is the only caller that ever passes them, and it
    passes whichever coordinate its own base measures; stage 1 stays
    base-model."""
    _CAM_CONTEXT["model"] = model
    _CAM_CONTEXT["focal"] = None if focal is None else float(focal)
    _CAM_CONTEXT["bspline"] = (
        None if bspline is None else np.ascontiguousarray(bspline, dtype=np.float64)
    )
    _CAM_CONTEXT["theta_max"] = None if theta_max is None else float(theta_max)


def camera_context():
    """The active ``(model, focal)`` context as a plain dict (a copy)."""
    return dict(_CAM_CONTEXT)


# ── The radial-spline models, keyed off the base model ──────────────────────
#
# SFMTOOL_FISHEYE and SFMTOOL_PINHOLE are ONE model with two bases: a monotone
# cubic B-spline over the base's own radial coordinate ``d``, added to ``d``
# before the focal scales it to pixels
# (specs/formats/sfmtool-camera-models.md).  ``d`` is the incidence angle
# ``theta`` under the equidistant base and the normalized image-plane radius
# ``rho = tan(theta)`` under the pinhole one, and that is the ONLY difference:
# same parameter head, same coefficients, same gauge, same monotonicity
# invariant.  Everything this script does with a spline camera is therefore
# written in ``d``, and this table is the single point where the base model
# picks which coordinate that is.
_SPLINE_MODEL = {
    # base model -> (spline model, domain-end parameter, fisheye base?)
    "SIMPLE_PINHOLE": ("SFMTOOL_PINHOLE", "bspline_rho_max", False),
    "SFMTOOL_PINHOLE": ("SFMTOOL_PINHOLE", "bspline_rho_max", False),
    "EQUIDISTANT_FISHEYE": ("SFMTOOL_FISHEYE", "bspline_theta_max", True),
    "SIMPLE_RADIAL_FISHEYE": ("SFMTOOL_FISHEYE", "bspline_theta_max", True),
    "SFMTOOL_FISHEYE": ("SFMTOOL_FISHEYE", "bspline_theta_max", True),
}


def spline_model(model=None):
    """``(spline model, domain-end parameter, fisheye base?)`` of the
    radial-spline promotion of ``model`` — the installed context's model when
    omitted.  A model the table does not carry is read as a fisheye base, the
    one this script promoted before the pinhole spline existed."""
    m = _CAM_CONTEXT["model"] if model is None else model
    return _SPLINE_MODEL.get(m, _SPLINE_MODEL["EQUIDISTANT_FISHEYE"])


def _d_of_theta(theta, fisheye_base=None):
    """The base model's radial coordinate at incidence angle ``theta``: the
    angle itself under a fisheye base, ``tan(theta)`` under a pinhole one."""
    if fisheye_base is None:
        fisheye_base = spline_model()[2]
    return theta if fisheye_base else np.tan(theta)


def _theta_of_d(d, fisheye_base=None):
    """The incidence angle at radial coordinate ``d`` — inverse of
    :func:`_d_of_theta`."""
    if fisheye_base is None:
        fisheye_base = spline_model()[2]
    return d if fisheye_base else np.arctan(d)


def fisheye_stage1():
    """Whether a FISHEYE-BASE camera context is installed — the finalization's
    twin of ``exp_fast_seed.fisheye_stage1``, and the single test every Phase-4
    branch is gated on.  A fisheye base only ever arrives through
    ``set_camera_context``, which stage 1 calls on a CONFIRMED both-cells verdict
    (routing by default, unless ``SFMTOOL_FISHEYE_SEED=0`` refuses it), so no
    capture the arbitration did not confirm as fisheye can reach any fisheye
    branch.

    Read off ``_SPLINE_MODEL`` rather than as "not the pinhole default": the
    spline rung promotes a PINHOLE capture too, and SFMTOOL_PINHOLE is a
    non-default model whose base is still the pinhole one.  What the branches
    gated here ask — the ray-range depth measure, the imaged-cone field test,
    the writer's model-generic surfel arm — is a question about the base, not
    about whether a distortion rung ran.  Every model reachable before that
    promotion answers exactly as the old test did."""
    return spline_model()[2]


# Absolute plausibility floor on a released focal, as a multiple of
# max(w, h).  The pinhole value is the long-standing bound; the equidistant
# one is the low end of the focal-vote kernel's FOV-derived band
# (specs/core/geometry/focal-vote.md), because `theta = r/f` ties focal to field of
# view — a >180 deg capture's own focal sits BELOW the pinhole floor (kerry:
# f ~ 138 px against 0.3 x 480 = 144), which would reject every honest solve.
_FOCAL_FLOOR_MULT = {
    "SIMPLE_PINHOLE": 0.3,
    "EQUIDISTANT_FISHEYE": 0.075,
    # The spline rung's promotion of the same lens: same floor as its base.
    "SFMTOOL_FISHEYE": 0.075,
    "SFMTOOL_PINHOLE": 0.3,
}


def focal_floor():
    """The context's absolute focal plausibility floor, px."""
    return _FOCAL_FLOOR_MULT.get(_CAM_CONTEXT["model"], 0.075) * max(_CAM_WH)


def make_cam(f=None):
    """The context camera at focal ``f`` (the context focal when omitted).

    SIMPLE_PINHOLE by default (principal point at the image centre);
    EQUIDISTANT_FISHEYE under an installed fisheye context, or the matching
    radial-spline model once the finalization's spline rung has promoted it
    (same map plus the context's spline).  All of them share the same three
    base parameters — a promoted model adds its distortion — so
    this builds one dict.  The images share one size (see main()), so one
    camera serves every projection; ``ray_to_pixel_batch`` /
    ``pixel_to_ray_batch`` map canonical camera-space points <-> full
    pixels."""
    w, h = _CAM_WH
    if f is None:
        f = _CAM_CONTEXT["focal"]
    params = {
        "focal_length": float(f),
        "principal_point_x": w / 2.0,
        "principal_point_y": h / 2.0,
    }
    model = _CAM_CONTEXT["model"]
    if model in ("SFMTOOL_FISHEYE", "SFMTOOL_PINHOLE"):
        coeffs = np.asarray(_CAM_CONTEXT["bspline"], dtype=np.float64)
        params[spline_model(model)[1]] = float(_CAM_CONTEXT["theta_max"])
        params["bspline_coeff_count"] = float(len(coeffs))
        for i, c in enumerate(coeffs):
            params[f"bspline_c{i}"] = float(c)
    return CameraIntrinsics.from_dict(
        {
            "model": model,
            "width": int(w),
            "height": int(h),
            "parameters": params,
        }
    )


def make_cam_bspline(f, coeffs, d_max):
    """The context camera promoted to its radial-spline model at
    ``(f, coeffs)`` on the spline domain ``[0, d_max]`` — SFMTOOL_FISHEYE
    under an equidistant base, SFMTOOL_PINHOLE under a pinhole one, with
    ``d_max`` in whichever radial coordinate that base measures
    (see ``_SPLINE_MODEL``).

    An all-zero ``coeffs`` is the base model's own map — bit for bit,
    projection, inverse and pixel Jacobian alike (the model's zero-spline
    identity) — so the promotion itself moves nothing; the coefficients are
    the dimensionless radial correction the base map cannot
    express, with ``f`` staying the central scale under the model's
    center-anchored gauge."""
    w, h = _CAM_WH
    model, d_key, _ = spline_model()
    cc = np.asarray(coeffs, dtype=np.float64)
    params = {
        "focal_length": float(f),
        "principal_point_x": w / 2.0,
        "principal_point_y": h / 2.0,
        d_key: float(d_max),
        "bspline_coeff_count": float(len(cc)),
    }
    for i, c in enumerate(cc):
        params[f"bspline_c{i}"] = float(c)
    return CameraIntrinsics.from_dict(
        {
            "model": model,
            "width": int(w),
            "height": int(h),
            "parameters": params,
        }
    )


def _cam_depth(p_cam):
    """Model-aware "distance in front" of CANONICAL camera-frame points.

    ``-z`` for the perspective family (the canonical camera looks down -Z), the
    ray RANGE ``|p|`` under a fisheye context.  Past 90 degrees off axis ``-z``
    is negative, so every ordering, median and ratio taken over it inverts on
    exactly the periphery a >180 degree capture exists to image; the range is
    the distance those observations actually sit at, and the two agree on axis.
    Broadcasts over the last axis, so it takes an ``(n, 3)`` block or one row."""
    if fisheye_stage1():
        return np.linalg.norm(p_cam, axis=-1)
    return -np.asarray(p_cam)[..., 2]


def _in_field(cam, p_cam):
    """Whether CANONICAL camera-frame points lie inside the cone ``cam`` images.

    Perspective family: the half-space in front, ``-z > 0`` — equivalently
    ``theta < 90 deg``.  A fisheye's imaged cone is not that half-space, so the
    same statement is made model-generically: ``theta <= r_max / f``, with
    ``r_max`` the INSCRIBED image-circle radius (half the smaller image
    dimension), i.e. the largest off-axis angle the sensor carries.  For a
    pinhole the two coincide; for the equidistant map the pinhole form would
    discard the whole 90-degree-plus annulus, which is 18-37% of every fisheye
    entry's detected features."""
    p = np.asarray(p_cam, dtype=np.float64)
    if not fisheye_stage1():
        return -p[..., 2] > 0
    rng = np.linalg.norm(p, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.clip(-p[..., 2] / np.maximum(rng, 1e-300), -1.0, 1.0))
    return (rng > 0) & (theta <= _field_theta_max(cam))


def _field_d_max(cam):
    """The largest radial coordinate ``cam`` images: the base model's own
    coordinate at the inscribed image circle's rim — the incidence angle
    ``theta`` under a fisheye base, the normalized image-plane radius
    ``rho = tan(theta)`` under a pinhole one.

    Distortion-free the pixel radius is ``f * d`` either way, so the rim is
    ``r_max / f`` outright.  Once a distortion rung has promoted the camera
    (the spline rung's SFMTOOL_FISHEYE / SFMTOOL_PINHOLE, or a legacy
    SIMPLE_RADIAL_FISHEYE) the pixel radius is ``f * (d + delta(d))``, not
    ``f * d``, so the rim is read back through the model's own inverse rather
    than divided out."""
    r_max = 0.5 * min(cam.width, cam.height)
    if cam.model in ("SIMPLE_RADIAL_FISHEYE", "SFMTOOL_FISHEYE", "SFMTOOL_PINHOLE"):
        cx, cy = cam.principal_point
        ray = cam.pixel_to_ray(cx + r_max, cy)
        theta = float(np.arccos(np.clip(-ray[2], -1.0, 1.0)))
        return _d_of_theta(theta, spline_model(cam.model)[2])
    return r_max / float(cam.focal_lengths[0])


# The fisheye entry point: under a fisheye base the radial coordinate IS the
# incidence angle, which is the reading ``_in_field`` and its callers take.
_field_theta_max = _field_d_max


def _colmap_proj_jacobian(cam, xc_col, s_flip, rel_step=1e-5):
    """Per-row 2x3 pixel Jacobian ``d(u, v)/d x`` of ``cam`` at the COLMAP-frame
    (+Z forward) camera-space points ``xc_col``.

    The camera object projects CANONICAL (-Z forward) rays, and the two frames
    differ by the involution ``S = diag(1, -1, -1)``, so ``x_can = S x_col`` and
    the chain rule is a column-wise sign flip: ``J_col = J_can · S``.  ``J_can``
    is a central difference of ``ray_to_pixel_batch``, which is what makes this
    model-generic — the same measure ``WarpMap`` takes its warp Jacobian by,
    valid for the equidistant map at every theta including past 90 degrees,
    where no image-plane form exists.  The step is relative to each row's own
    range so it is scale-free (the projection is degree-0 homogeneous in the
    ray, so only the direction matters).

    Returns an ``(n, 2, 3)`` array."""
    xc_can = np.ascontiguousarray(xc_col * s_flip)
    h = rel_step * np.maximum(np.linalg.norm(xc_can, axis=1), 1e-12)
    out = np.zeros((len(xc_can), 2, 3))
    for j in range(3):
        step = np.zeros_like(xc_can)
        step[:, j] = h
        plus = np.asarray(
            cam.ray_to_pixel_batch(np.ascontiguousarray(xc_can + step)),
            dtype=np.float64,
        )
        minus = np.asarray(
            cam.ray_to_pixel_batch(np.ascontiguousarray(xc_can - step)),
            dtype=np.float64,
        )
        out[:, :, j] = (plus - minus) / (2.0 * h)[:, None] * s_flip[j]
    return out


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


def relative_warps(shapes, obs_c, reference_members):
    """Reference->member warps ``W = S.S_ref^-1`` from stored absolute shapes.

    Since .matches format version 5 the affine's leading 2x2 is the member's
    ABSOLUTE affine shape ``S = W.S_ref`` -- the detector's canonical unit
    frame mapped onto that member's image pixels -- not the reference-relative
    warp.  Consumers that want image-space SIZE read ``S`` directly; consumers
    that want the warp between two views of one patch (the surfel writer's
    tilt solve) invert the cluster's own reference row ``S_ref`` to get it
    back.

    ``shapes`` is ``(K, 2, 2)``, ``obs_c`` the per-member cluster id, and
    ``reference_members`` the per-cluster global member index (``0xFFFFFFFF``
    where the cluster carries no reference).  Members of a reference-less
    cluster have no recoverable warp -- a derived file whose reference fell
    outside its restriction -- and come back as zeros, exactly like the
    all-zero rows of members that were never evaluated.
    """
    refs = np.asarray(reference_members)
    have = refs != np.iinfo(np.uint32).max
    out = np.zeros_like(shapes)
    rows = np.nonzero(have[obs_c])[0]
    if len(rows):
        s_ref = shapes[refs[obs_c[rows]].astype(np.int64)]
        out[rows] = shapes[rows] @ np.linalg.inv(s_ref)
    return out


def load_clusters(matches_data=None, preselected=False):
    """Patch clusters as flat observation arrays with refined positions.

    Everything geometric comes straight from the .matches file: image
    dimensions from the images section and member positions from the stored
    affines' last column (the absolute refined keypoint position).  The
    admission itself — drop unrefinable clusters, keep reference/kept
    members, restrict to selected images, span-filter — is the
    matches-format crate's ``select_clusters`` derivation; this function
    just reshapes the selected file into the flat observation arrays.

    ``matches_data`` (optional): an already-open ``MatchesFile`` handle to
    reuse.  Default None: read the workspace's clusters-patches file.

    ``preselected``: the handle IS the admission — a restriction stage already
    ran the selection and wrote it, so this load reshapes its arrays as they
    stand.  Re-selecting such a file would drop every cluster whose reference
    member fell outside the restriction (the derived file records the
    absent-reference sentinel, which the selection reads as unrefinable), so a
    preselected load never re-selects.  Its cluster ids are the file's own,
    which is what the stages upstream of it name.

    Restricting to a subset of images is NOT an option here: it is the
    restriction stage's job, and its artifact is what a ``preselected`` load
    reads.  Two independent restrictions of one source renumber independently,
    which is the coupling that stage exists to remove.
    """
    _t_load = time.perf_counter()
    if matches_data is not None:
        data = matches_data
    else:
        from sfmtool._sfmtool.io import MatchesFile

        override = os.environ.get("SFMTOOL_MATCHES")
        patches = (
            [Path(override)]
            if override
            else sorted(WS.glob("matches/*-clusters-patches.matches"))
        )
        print(f"matches file: {patches[0]}")
        data = MatchesFile(patches[0])
    names = list(data.image_names)
    dims = [(int(w), int(h)) for w, h in np.asarray(data.image_dims)]

    # File-level selection (native): reference/kept members, clusters spanning
    # >= MIN_SPAN_BA distinct images.  Cluster order (by source id) and member
    # order are preserved, so the observation stream below matches the
    # selection's CSR layout directly.
    sel = data if preselected else data.select_clusters(min_span=MIN_SPAN_BA)
    starts = np.asarray(sel.cluster_starts, dtype=np.int64)
    sizes = np.diff(starts)
    n_cl = len(sizes)
    aff = np.asarray(sel.member_affines)

    obs_i = np.asarray(sel.member_images, dtype=np.int64)
    if preselected:
        n2r = int((sizes == 2).sum())
        print(
            f"preselected admission over {len(names)} images: span-2 {n2r}, "
            f"span>=3 {n_cl - n2r} usable clusters"
        )

    # Admission order (best first) — used for both the cap and the tiers:
    # highest span first (ties broken by cluster id for determinism).  The
    # selected file keeps at most one reference/kept member per (cluster,
    # image), so each cluster's span IS its member count.
    #
    # No admission cap: growth and triangulation see every usable cluster
    # (a capped set can disconnect a chain-shaped capture — south-building
    # fragmented at 36/128).  The ordering instead selects which clusters'
    # observations enter the BAs (the top MAX_CLUSTERS by adm_rank).
    order = np.lexsort((np.arange(n_cl), -sizes))
    adm_rank = np.empty(n_cl, dtype=np.int64)
    adm_rank[order] = np.arange(n_cl)

    print(f"load_clusters: {time.perf_counter() - _t_load:.2f}s")
    obs_c = np.repeat(np.arange(n_cl, dtype=np.int64), sizes)
    # The affine's last column is the member's absolute refined keypoint
    # position; the 2x2 block is its ABSOLUTE affine shape (`S_ref | x_ref`
    # for the reference row).  The surfel writer wants the reference-relative
    # warp, so recover it through each cluster's reference row.
    out = {
        "names": names,
        "dims": dims,
        "obs_c": obs_c,
        "obs_i": obs_i,
        "obs_f": np.asarray(sel.member_features, dtype=np.int64),
        "obs_uv": np.ascontiguousarray(aff[:, :, 2], dtype=np.float64),
        "obs_warp": np.ascontiguousarray(
            relative_warps(aff[:, :, :2], obs_c, sel.reference_members),
            dtype=np.float64,
        ),
        "obs_ref": np.asarray(sel.member_status) == 0,
        "adm_rank": adm_rank,
        # Worst (max) finite warp-consistency residual over the selected
        # members — lower is better; clusters where no member entered the
        # consistency fit rank last (inf).
        "cl_quality": np.asarray(sel.cluster_worst_consistency(), dtype=np.float64),
        "refine_radius": data.refine_radius,
        "n_img": len(names),
        "n_cl": n_cl,
    }
    return out


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

    STILL PINHOLE, and deliberately left so: the `rays * (z_pred / rays.z)`
    scaling below places a ray at a predicted DEPTH ALONG -Z, which is
    meaningless for a ray at theta >= 90 degrees (`rays.z` crosses zero) — the
    ray-native form would scale by range along the ray instead.  It is not
    fixed here because nothing reaches it from the fisheye seed: `depth_init`
    is opt-in (SFMTOOL_DEPTH_INIT=1, default off) and lives inside
    ``grow_loop``, which only `exp_pinhole_bootstrap.main()`'s growth stage
    calls — the seed's `finalize_seed_from_dict` path never does.  Any future
    fisheye GROWTH stage must ray-fix it first
    (scripts/notes-fisheye-seed.md, Phase 2 item 5).

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
    covis,
    max_images=None,
    aux=None,
    ba=None,
    snap=None,
):
    """Next-best-view growth from an existing state (resumable: tier
    admission re-enters here after activating more clusters)."""
    snap_every = int(os.environ.get("SFMTOOL_SNAPSHOT_EVERY", "1"))
    grow_schedule = [(30.0 * _PXS, 3.0 * _PXS), (8.0 * _PXS, 1.5 * _PXS)]
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
        if ba_window > 0:
            win = np.zeros(n_img, bool)
            if (
                anchor_every > 0
                and ba_calls[0] % anchor_every == 0
                and int(posed.sum()) > ba_window
            ):
                win[
                    np.asarray(
                        covis.thin_to(min(int(posed.sum()), 150)), dtype=np.int64
                    )
                ] = True
                win &= posed
            elif len(posed_order) > ba_window:
                win[np.asarray(posed_order[-ba_window:], dtype=np.int64)] = True
            if win.any():
                live &= win[obs_i]
        rot = Rotation.from_rotvec(rvec).as_matrix()
        out = bundle_adjust(
            obs_c[live],
            obs_i[live],
            u[live],
            rot,
            tvec,
            pts,
            f0,
            n_img,
            n_cl,
            opt_f=False,
            verbose=False,
            schedule=grow_schedule,
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
                        print(
                            f"    p3p img {j}: {int(mask.sum())}/"
                            f"{int(sj.sum())} RANSAC inliers, "
                            f"consensus refit inl {inl_c:.0%}"
                        )
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
                    pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f0)
                    ba_retry = True
                    blocked.clear()
                    if TRACE:
                        print(
                            f"    force-accept img {j}: {best_j[0]:.0%} -> "
                            f"{inl_after:.0%} after BA"
                            f"{'' if consensus is None else f', consensus surv {surv:.0%}'}"
                            f" (kept)"
                        )
                else:
                    posed[j] = False
                    if posed_order and posed_order[-1] == int(j):
                        posed_order.pop()
                    if ba_saved is not None:
                        ba[:] = ba_saved
                    if TRACE:
                        print(
                            f"    force-reject img {j}: {best_j[0]:.0%} -> "
                            f"{inl_after:.0%} after BA"
                            f"{'' if consensus is None else f', consensus surv {surv:.0%}'}"
                            f" (unposed)"
                        )
                continue
            break
        s = (obs_i == i) & ~np.isnan(pts[obs_c, 0])
        # Warp-depth Kabsch init (when enabled) — also feeds the
        # post-acceptance depth-coherence diagnostic below, so compute it
        # regardless of which resection path wins.
        di = (
            depth_init(s, obs_c, u, pts, rvec, tvec, posed, f0, i, aux)
            if aux is not None and DEPTH_INIT
            else None
        )
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
        p3p = (
            p3p_resect(u[s], pts[obs_c[s]], f0, aux[2][i])
            if aux is not None and len(aux) > 2
            else None
        )
        if p3p is not None and int(p3p[2].sum()) >= 12:
            rv0, tv0, mask = p3p
            rv, tv, _ = pose_refine(u[s][mask], pts[obs_c[s]][mask], rv0, tv0, f0)
            res = reproj_res_one(cam0, rv, tv, pts[obs_c[s]], u[s])
            found = (_inlier_fraction(res, 3.0), rv, tv)
            if TRACE:
                print(
                    f"    p3p    img {i}: {int(mask.sum())}/{int(s.sum())} "
                    f"RANSAC inliers, all-obs inl {found[0]:.0%}"
                )
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
                print(
                    f"    defer  img {i}: inl {found[0]:.0%} on "
                    f"{int(s.sum())} obs (median accepted "
                    f"{float(np.median(accepted_inl)):.0%})"
                )
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
        if snap is not None and int(posed.sum()) % snap_every == 0:
            snap(f"grow-{int(posed.sum()):03d}-img{i}", f0, rvec, tvec, pts, posed)
        since_ba += 1
        if GROW_BA and since_ba >= ba_every:
            since_ba = 0
            rvec, tvec, pts = run_grow_ba(rvec, tvec, pts)
            if snap is not None:
                snap(f"grow-{int(posed.sum()):03d}-ba", f0, rvec, tvec, pts, posed)
    return rvec, tvec, pts, posed


def batch_resect(
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
    covis,
    aux,
    gate=0.30,
    rounds=6,
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


# ── Perspective conversion + triangulation ───────────────────────────────────


def dense_structure(all_c, all_i, all_u, f, rvec, tvec, pts, posed, quiet=False):
    """The fine structure pass the coarse growth defers.

    Growth and the BAs run on the capped cluster subset (the scaling lever), so
    the structure they leave behind covers only that subset — a frame whose
    covisibility lives entirely in capped-out clusters ends up posed but with
    zero observations (a pose-only skeleton). The poses are final here, so
    triangulate EVERY cluster at those poses and keep each posed-frame inlier.
    Returns ``(pts, keep, res)`` over the full observation array; a one-line
    reprojection-inlier summary is printed as the only structure-quality signal
    (the pose-only save had none). A low inlier fraction means the poses/focal
    do not yield consistent multi-view structure even where the poses
    similarity-align to a reference."""
    pts = fill_new_points(pts, all_c, all_i, all_u, rvec, tvec, posed, f)
    ok = posed[all_i] & ~np.isnan(pts[:, 0])[all_c]
    res = np.full(len(all_c), np.inf)
    if ok.any():
        # (empty-input guard: ray_to_pixel_batch returns shape (0, 0) for an
        # empty batch, which does not broadcast against all_u[ok]'s (0, 2))
        xc = (
            Rotation.from_rotvec(rvec[all_i[ok]]).apply(pts[all_c[ok]])
            + tvec[all_i[ok]]
        )
        proj = make_cam(f).ray_to_pixel_batch(np.ascontiguousarray(xc))
        res[ok] = np.linalg.norm(proj - all_u[ok], axis=1)
    keep = res < TRIM_PX
    if not quiet:
        r = res[ok & np.isfinite(res)]
        if len(r):
            print(
                f"structure: {int(ok.sum())} dense obs triangulated; reproj "
                f"<2px {100 * (r < 2).mean():.1f}% <4px {100 * (r < 4).mean():.1f}% "
                f"<10px {100 * (r < 10).mean():.1f}% (median kept "
                f"{np.median(res[keep]):.2f} px)"
            )
        else:
            print("structure: 0 dense obs triangulated")
    return pts, keep, res


# ── Step-by-step snapshots (debug) ───────────────────────────────────────────
# When SFMTOOL_SNAPSHOT_DIR is set, the completion writes a numbered dense
# .sfmr at each growth step so a run can be replayed frame by frame against
# expectation. Unset (the default) every hook is a no-op and the run is
# byte-identical to production. SFMTOOL_SNAPSHOT_EVERY thins the per-image
# stream (key events — seed, every BA, release-f, batch-resect — always save).


def make_snapshotter(data):
    """Return a `snap(tag, f, rvec, tvec, pts, posed)` that writes a dense
    .sfmr per call, or None when snapshots are disabled."""
    snap_dir = os.environ.get("SFMTOOL_SNAPSHOT_DIR")
    if not snap_dir:
        return None
    out = Path(snap_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_c, all_i, all_u = data["obs_c"], data["obs_i"], data["obs_uv"]
    n = [0]

    def snap(tag, f, rvec, tvec, pts, posed):
        p2, keep, res = dense_structure(
            all_c, all_i, all_u, f, rvec, tvec, pts.copy(), posed, quiet=True
        )
        path = out / f"{n[0]:03d}-{tag}.sfmr"
        save_sfmr(data, f, rvec, tvec, p2, keep, res, path)
        npt = len(np.unique(all_c[keep]))
        print(
            f"  [snapshot {n[0]:03d}-{tag}: {int(posed.sum())} posed, "
            f"{npt} pts, {int(keep.sum())} obs]"
        )
        n[0] += 1

    return snap


# ── Seed-stage snapshots (debug) ─────────────────────────────────────────────
# When SFMTOOL_SEED_SNAPSHOT_DIR is set, the SEED stage writes a numbered .sfmr
# at every checkpoint of its pipeline — stage 1's affine factorization, probe,
# widen and photometric verify (exp_fast_seed), the released estimate of each
# pass, and this module's photometric finalization (dense / embed / culled) — so
# every intermediate state can be opened in the SfM Explorer.  Unset (the
# default) every hook is a no-op and the run is byte-identical to production.
# Files are named `NN-<stage>[-<pass>-<attempt>].sfmr` with NN the checkpoint
# index, so a directory listing reads in pipeline order.


def seed_snapshot_path(tag):
    """Output path for a seed snapshot, or None when snapshots are disabled."""
    snap_dir = os.environ.get("SFMTOOL_SEED_SNAPSHOT_DIR")
    if not snap_dir:
        return None
    out = Path(snap_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{tag}.sfmr"


def seed_snapshot(
    tag,
    data,
    f,
    rvec,
    tvec,
    pts,
    posed,
    extra_tool_options=None,
    path=None,
    release_grade=False,
):
    """Write one seed checkpoint as a dense, Explorer-loadable .sfmr.

    ``rvec``/``tvec``/``posed`` are indexed by ``data``'s image index and
    ``pts`` by its cluster index — a caller holding a thinned working set maps
    back to that frame first.  The structure is completed exactly as the final
    save does it (``dense_structure`` at the given poses, then ``save_sfmr``),
    so the file shows the state as the pipeline holds it.  Debug instrumentation
    must never kill the run it instruments, so every failure is caught and
    reported as a one-line warning.

    ``path`` names an explicit destination instead of the env-gated snapshot
    directory, for the artifacts a run writes unconditionally (the seed's
    per-hypothesis releases); ``release_grade`` writes poses and points only.
    """
    if path is None:
        path = seed_snapshot_path(tag)
    if path is None:
        return None
    try:
        global _CAM_WH
        if _CAM_WH is None:
            _CAM_WH = tuple(data["dims"][0])
        all_c, all_i, all_u = data["obs_c"], data["obs_i"], data["obs_uv"]
        pts_dense, keep, res = dense_structure(
            all_c,
            all_i,
            all_u,
            f,
            rvec,
            tvec,
            np.array(pts, dtype=np.float64, copy=True),
            posed,
            quiet=True,
        )
        # `save_sfmr` keeps points with >= 2 surviving observations; when none
        # does (typically a state with 0-1 posed frames) there is no
        # reconstruction to write, and the writer's empty arrays raise instead
        # of saying so.  The missing file IS the signal.
        n_alive = int((np.bincount(all_c[keep], minlength=len(pts_dense)) >= 2).sum())
        if n_alive == 0:
            print(
                f"  [seed-snapshot {tag}: {int(np.asarray(posed).sum())} posed, "
                f"{int(keep.sum())} obs, no multi-view points; skipped]"
            )
            return None
        save_sfmr(
            data,
            f,
            rvec,
            tvec,
            pts_dense,
            keep,
            res,
            path,
            tool_options=extra_tool_options,
            quiet=True,
            release_grade=release_grade,
        )
        print(
            f"  [seed-snapshot {path.name}: {int(np.asarray(posed).sum())} posed, "
            f"{len(np.unique(all_c[keep]))} pts, {int(keep.sum())} obs, f={f:.1f}]"
        )
        return path
    except Exception as exc:
        print(f"  [seed-snapshot {tag} FAILED: {type(exc).__name__}: {exc}]")
        return None


def seed_snapshot_recon(tag, recon):
    """Write an already-built reconstruction as a seed checkpoint."""
    path = seed_snapshot_path(tag)
    if path is None:
        return None
    try:
        recon.save(str(path))
        print(
            f"  [seed-snapshot {path.name}: {len(recon.image_names)} images, "
            f"{recon.point_count} pts]"
        )
        return path
    except Exception as exc:
        print(f"  [seed-snapshot {tag} FAILED: {type(exc).__name__}: {exc}]")
        return None


# ── Stage dumps (per-image observation waterfall) ────────────────────────────
# `SFMTOOL_STAGE_DUMPS=<dir>` turns every major pass of the photometric
# finalization into a checkpoint: the reconstruction as that pass LEAVES it,
# saved to `<dir>/NN-<stage>.sfmr`, plus two log lines — the per-image
# observation census, and the per-image DELTA since the previous stage.  The
# delta is what makes the chain readable as a waterfall without diffing files:
# a frame that ends the finalization empty lost its observations at specific
# stages, and the delta line names them.
#
# Stages are numbered in EXECUTION order across the whole finalization,
# including the passes nested inside the late vetting and the ARS promotion,
# so a directory listing reads in pipeline order.  Unset (the default) every
# hook returns before touching the reconstruction and the run is byte-identical
# to production.
_STAGE_DUMP = {"n": 0, "prev": None, "dir": None, "resolved": False}


def stage_dump_dir():
    """Output directory for stage dumps, or None when they are disabled."""
    if not _STAGE_DUMP["resolved"]:
        d = os.environ.get("SFMTOOL_STAGE_DUMPS")
        out = Path(d) if d else None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        _STAGE_DUMP["dir"] = out
        _STAGE_DUMP["resolved"] = True
    return _STAGE_DUMP["dir"]


def stage_dump(name, recon, save=True):
    """Record one finalization checkpoint: census, delta, and the .sfmr.

    ``save=False`` logs the census without writing the file, for the cheap
    intermediate points where the geometry has not changed.  Instrumentation
    must never kill the run it instruments, so every failure is a warning."""
    out = stage_dump_dir()
    if out is None or recon is None:
        return
    try:
        n = _STAGE_DUMP["n"]
        _STAGE_DUMP["n"] = n + 1
        names = [Path(s).name for s in recon.image_names]
        ti = np.asarray(recon.track_image_indexes)
        obs = np.bincount(np.asarray(ti, np.int64), minlength=len(names))[: len(names)]
        census = dict(zip(names, obs.tolist()))
        prev = _STAGE_DUMP["prev"]
        _STAGE_DUMP["prev"] = census
        print(
            f"  [stage {n:02d} {name}: {recon.point_count} pts, "
            f"{len(ti)} obs, {len(names)} images]"
        )
        print(f"    stage {n:02d} {name}: obs per image {obs.tolist()}")
        if prev is not None:
            delta = [census[k] - prev.get(k, 0) for k in names]
            print(f"    stage {n:02d} {name}: delta per image {delta}")
            gone = [k for k in prev if k not in census]
            if gone:
                print(f"    stage {n:02d} {name}: images dropped {gone}")
        if save:
            recon.save(str(out / f"{n:02d}-{name}.sfmr"))
    except Exception as exc:
        print(f"  [stage-dump {name} FAILED: {type(exc).__name__}: {exc}]")


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

# Free-focal BA admits both camera models this script builds: SIMPLE_PINHOLE
# and EQUIDISTANT_FISHEYE.  Both are single-focal and distortion-free, so the
# kernel's analytic focal column `d(u, v)/df = (u - cx)/f` is EXACT for each
# (specs/core/geometry/bundle-adjustment.md) — the Phase-3b kernel widening removed the
# fixed-focal clamp this wrapper used to apply under a fisheye context.


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
        schedule = [
            (50.0 * _PXS, 5.0 * _PXS),
            (12.0 * _PXS, 2.0 * _PXS),
            (TRIM_PX, 1.0),
        ]
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
                n: g[PurePosixPath(n).name] for n in names if PurePosixPath(n).name in g
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


def save_sfmr(
    data,
    f,
    rvec,
    tvec,
    pts,
    keep,
    res,
    out_path,
    return_alive=False,
    tool_options=None,
    quiet=False,
    release_grade=False,
    operation="cluster_bootstrap",
):
    """Write the bootstrap as an ``embedded_patches`` reconstruction.

    The bootstrap's observations are the cluster patches' *refined*
    positions, not the SIFT detections, so they are stored inline as
    ``keypoints_xy`` rather than as feature indexes into the ``.sift``
    files (which would silently resolve back to the unrefined seeds).

    ``tool_options`` merges extra entries into the file's metadata (a debug
    snapshot declares what state it holds); ``operation`` names the stage that
    produced the artifact, which a reader takes the file's provenance from, and
    stays the bootstrap's own unless the caller is a different stage;
    ``quiet`` suppresses the summary
    line so an instrumented run stays readable.  ``release_grade`` stops at
    poses and points — no keypoint passthrough and no patch frames — which is
    what an inspectable side artifact needs and is far cheaper: the surfel
    solve below opens every posed image's `.sift` affine array.
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

    # Write only the posed cameras. Every kept observation belongs to a posed
    # frame, so the images that carry one are exactly the posed set; the rest
    # hold the shared default seed pose (identical camera centers), which both
    # misrepresents the reconstruction (phantom cameras at the origin) and, when
    # many frames are unposed (e.g. an early growth snapshot), overflows the
    # viewer's k-d tree spatial index. Compact the per-image arrays and remap
    # the observation image indexes; everything downstream then works in the
    # posed-only image space (embedded_patches cannot be image-subset later).
    posed_img = np.unique(track_img)
    if len(posed_img) < len(names):
        img_remap = np.full(len(names), -1, np.int64)
        img_remap[posed_img] = np.arange(len(posed_img))
        names = [names[j] for j in posed_img]
        rvec = rvec[posed_img]
        tvec = tvec[posed_img]
        track_img = img_remap[track_img].astype(track_img.dtype)

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

    # The context camera, NOT a hardcoded SIMPLE_PINHOLE.  The structure this
    # writes was triangulated through ``make_cam`` (equidistant rays under a
    # fisheye context), so stamping a pinhole model on it hands every
    # downstream consumer — the photometric embed, the reprojection culls, the
    # finalization BA — a camera that does not describe the observations.
    # Identical to the previous literal on the pinhole default.
    camera = make_cam(float(f))

    opts = {"camera_model": _CAM_CONTEXT["model"], "focal_grid": F_GRID}
    opts.update(tool_options or {})
    metadata = build_metadata(
        workspace_dir=workspace_dir,
        output_path=out_path,
        workspace_config=load_workspace_config(workspace_dir),
        operation=operation,
        tool_name="sfmtool",
        tool_options=opts,
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

    if release_grade:
        # Poses and points, and stop: no keypoint passthrough, no patch cloud,
        # no normals.  The surfel solve below is the expensive half of this
        # writer (a `.sift` read per posed image plus a per-point least
        # squares), and a side artifact meant for inspection carries none of it.
        recon.save(out_path)
        if not quiet:
            print(f"\nwrote {out_path} ({len(alive)} points, release-grade)")
        return (recon, alive, posed_img) if return_alive else recon

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

    # The surfel-frame solve below runs in the COLMAP +Z-forward camera frame,
    # so it runs on the poses flipped back to that frame by S = diag(1, -1, -1);
    # its world-space u/v/normal outputs convert to the canonical world by the
    # same W as the points, at the end.  Positions stay in the COLMAP-world
    # gauge.
    #
    # There are two arms.  The PERSPECTIVE arm is the historical one, written
    # against the pinhole projection Jacobian `(f/z)[I | -p_r]` and its
    # `z > 0` in-front test, with the reference right-inverse `(z/f)[I; 0]` and
    # the null direction `X/z`.  Under a RAY-PATH context (fisheye) both are
    # wrong: `z` is not a distance past 90 deg off axis (it crosses zero and
    # then goes negative over the very periphery a >180 deg capture exists to
    # image), and the pinhole Jacobian describes a different map.  The fisheye
    # arm is the same solve stated model-generically -- the camera's own 2x3
    # Jacobian, its minimum-norm right inverse, and the unit viewing ray as the
    # null direction -- which reduces to the perspective arm up to a
    # reparameterization of the out-of-plane slope.  Both are kept because the
    # Tikhonov prior and the obliquity cap live on that slope, so the pinhole
    # path stays bit-identical only if its own parameterization does.
    fisheye_frames = fisheye_stage1()
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
    # Under the fisheye context, every kept observation's camera-frame point and
    # the camera model's own 2x3 pixel Jacobian there, taken ONCE for the whole
    # writer by central difference of ``ray_to_pixel`` — the same measure
    # ``WarpMap`` takes its warp Jacobian by, so the frames agree with the render.
    xc_obs = j_obs = None
    if fisheye_frames:
        xc_obs = (
            np.einsum("nij,nj->ni", rot_all[track_img], positions[point_idx])
            + tvec_col[track_img]
        )
        j_obs = _colmap_proj_jacobian(make_cam(float(f)), xc_obs, s_flip)
    # Patch EXTENT for every point at once, through THE CAMERA'S OWN sizing rule
    # (``CameraIntrinsics.pixel_radius_to_world_batch``): the world size that
    # subtends the detection's pixel radius at the reference view.  One rule for
    # every model — it reduces to `r_px*|z|/f` for a pinhole and `r_px*d/f` under
    # the equidistant map — and, more to the point, it is the SAME
    # implementation ``PatchExtent::FeatureSize`` uses, so this writer and the
    # core cannot drift apart.  (They did once: the writer carried a hand-copied
    # pinhole formula through Phase 3b, which is exactly how a fisheye seed came
    # to be sized through a pinhole camera.)
    ref_row = np.full(len(alive), -1, np.int64)
    for p in range(len(alive)):
        lo, hi = int(p_starts[p]), int(p_starts[p + 1])
        here = np.nonzero(is_ref[lo:hi])[0]
        if len(here):
            ref_row[p] = lo + int(here[0])
    ext_all = np.zeros(len(alive))
    has_ref = ref_row >= 0
    if has_ref.any():
        rr = ref_row[has_ref]
        ri = track_img[rr]
        xc_ref_all = (
            np.einsum("nij,nj->ni", rot_all[ri], positions[has_ref]) + tvec_col[ri]
        )
        r_px_all = np.array(
            [
                radius_kf * feature_scales[int(i)][int(track_feat[k])]
                for i, k in zip(ri.tolist(), rr.tolist())
            ]
        )
        ext_all[has_ref] = make_cam(float(f)).pixel_radius_to_world_batch(
            # The camera projects CANONICAL rays; the writer works in the COLMAP
            # frame, and the two differ by the involution S = diag(1, -1, -1).
            np.ascontiguousarray(xc_ref_all * s_flip),
            np.ascontiguousarray(r_px_all),
        )
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
        if fisheye_frames:
            # RANGE, not optical-axis depth: the distance an angular size
            # ``r_px / f`` actually spans, and the one quantity that stays
            # positive over a >180 deg field.
            d_ref = max(float(np.linalg.norm(xc_ref)), 1e-6)
            n_ref = xc_ref / d_ref  # unit viewing ray == null direction of J_ref
            scale_ref = d_ref / f
            b_perp = np.linalg.pinv(j_obs[k_ref])  # 3x2, columns ⊥ n_ref
        else:
            z_ref = max(xc_ref[2], 1e-6)
            p_r = xc_ref[:2] / z_ref
            scale_ref = z_ref / f
        rows, rhs = [], []
        for k in range(lo, hi):
            if k == k_ref:
                continue
            i = int(track_img[k])
            if fisheye_frames:
                m = j_obs[k] @ rot_all[i] @ r_ref.T
                c_k = m @ n_ref
                resid = warps[k] - m @ b_perp
            else:
                xc = rot_all[i] @ x_pt + tvec_col[i]
                z = max(xc[2], 1e-6)
                j_proj = (f / z) * np.array(
                    [[1.0, 0.0, -xc[0] / z], [0.0, 1.0, -xc[1] / z]]
                )
                m = j_proj @ rot_all[i] @ r_ref.T
                c_k = m[:, :2] @ p_r + m[:, 2]
                resid = warps[k] - scale_ref * m[:, :2]
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
        lam = WRITER_FRONTO_LAM * np.sqrt((rows**2).sum() / max(len(rows), 1))
        rows = np.vstack([rows, [[lam, 0.0], [0.0, lam]]])
        rhs = np.concatenate([rhs, [0.0, 0.0]])
        b_z = np.linalg.lstsq(rows, rhs, rcond=None)[0]
        # Obliquity cap: tan(tilt) = |b_z| / (the in-plane scale of the
        # reference right-inverse) — z_r/f for the pinhole, d_r/f (the radial
        # singular value of the equidistant map's inverse) for the fisheye arm.
        b_norm = np.linalg.norm(b_z)
        max_bz = tan_cap * scale_ref
        if b_norm > max_bz:
            b_z *= max_bz / b_norm
        if fisheye_frames:
            b_map = r_ref.T @ (b_perp + np.outer(n_ref, b_z))
        else:
            b_map = r_ref.T @ np.vstack(
                [scale_ref * np.eye(2) + np.outer(p_r, b_z), b_z[None, :]]
            )
        # The tilt solve above determines the surfel's NORMAL and nothing else.
        # The frame itself is a SQUARE: a SIFT detection is a round region of the
        # image, so the surface element it seeds is a square patch that
        # orientation may TILT but nothing may DISTORT.  Foreshortening is the
        # projection's business — it belongs in the render, not in the stored
        # frame.  (b_map's own two columns are a sheared, anisotropic
        # parallelogram: they carry the tilt's foreshortening inside the extents.
        # Storing those made the seed's patches stretched — median |u|/|v| 2.6 on
        # 20240614_224422531 with 22% beyond 4x — and worse, `refine_normals`
        # later re-solves the normal to near-fronto while KEEPING the extents, so
        # the patch ends up facing the camera and still stretched: the measured
        # tilt is 0.8-7.6 deg median across the fleet while the elongation stays.)
        n3 = np.cross(b_map[:, 0], b_map[:, 1])
        norm = np.linalg.norm(n3)
        if norm < 1e-12:
            continue
        n3 /= norm
        cam_c = -r_ref.T @ tvec_col[i_ref]
        if np.dot(n3, cam_c - x_pt) < 0:
            n3 = -n3  # front-facing; the frame below is built around the normal
        # In-plane u: the reference image's x direction projected onto the plane.
        # This keeps the upright-bitmap convention structural rather than
        # corrective — bitmap columns run along reference-image x, and
        # v = n x u then comes out along -image-y, which is exactly what the
        # raster's row reversal (rows step along -v) expects.
        ax = r_ref.T @ np.array([1.0, 0.0, 0.0])
        u3 = ax - np.dot(ax, n3) * n3
        nu = np.linalg.norm(u3)
        if nu < 1e-9:  # plane edge-on to image x: fall back to image y
            ay = r_ref.T @ np.array([0.0, 1.0, 0.0])
            u3 = ay - np.dot(ay, n3) * n3
            nu = np.linalg.norm(u3)
            if nu < 1e-9:
                continue
        u3 /= nu
        v3 = np.cross(n3, u3)  # u x v == n exactly, for a unit u perpendicular to n
        # ONE scalar extent: the detection's fronto world size at the reference
        # view, taken from the camera itself in the batch above (`ext_all`).
        # b_map's own fronto case (b_z = 0) is r_ref^T (z/f) I, so a detection of
        # r_px reference pixels spans exactly the camera's own
        # `pixel_radius_to_world` at that point — the same detection-time
        # quantity the duplicate collapse sizes its radius from, now the surfel's
        # only size parameter.  Under the equidistant map that is the RANGE form
        # (r_px pixels subtend r_px / f radians everywhere, spanning
        # r_px * d_ref / f at range d_ref); the optical-axis form is that times
        # cos(theta_ref), which shrinks the patch two-fold at 60 deg off axis and
        # to nothing at 90 — a zero-extent frame is written as NO patch at all
        # (`PatchCloud.from_halfvec_arrays` drops a zero `u` row).
        ext = float(ext_all[p])
        half_u[p], half_v[p], normals[p] = u3 * ext, v3 * ext, n3

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
    if not quiet:
        print(
            f"\nwrote {out_path} ({len(alive)} points, {int(obs_counts.sum())} obs, "
            f"{recon.feature_source}, {n_patched} warp-derived patch frames)"
        )
    # `alive` maps output point index -> source cluster index; `posed_img` maps
    # output image index -> data image index (the writer keeps only posed
    # frames). The seed finalization maps refined points/views back through them.
    return (recon, alive, posed_img) if return_alive else recon


# ── Main ─────────────────────────────────────────────────────────────────────


def _cheirality_keep(recon):
    """Per-point mask: False where the point is behind ANY camera observing it.

    Canonical frame is -Z forward, so the camera-frame depth of world point p
    seen by image (R, t) is ``-(R @ p + t).z`` and must be strictly positive.
    Points at infinity carry a DIRECTION in ``positions``, so the translation
    does not apply to them — the direction itself must point in front.  One
    einsum over the observation list; no Python loop.

    Under a fisheye context "in front" is the camera's own imaged CONE rather
    than the half-space (see ``_in_field``): a >180 degree capture legitimately
    sees points at ``-z <= 0``, so the half-space test would cull every point
    with a single peripheral observation — which is what the cull was doing on
    every fisheye entry (their shipped seeds stop at theta = 90.0 deg exactly)."""
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    if len(q) == 0 or recon.point_count == 0:
        return np.ones(recon.point_count, bool)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    t = np.asarray(recon.translations, dtype=np.float64)
    pts = np.asarray(recon.positions, dtype=np.float64)
    ti = np.asarray(recon.track_point_indexes)
    ii = np.asarray(recon.track_image_indexes)
    inf = np.asarray(recon.point_is_at_infinity)
    pc = np.einsum("nij,nj->ni", rots[ii], pts[ti])
    pc += np.where(inf[ti][:, None], 0.0, t[ii])
    bad = ~_in_field(recon.cameras[0], pc)
    keep = np.ones(recon.point_count, bool)
    np.logical_and.at(keep, ti, ~bad)
    return keep


def _refresh_errors(recon):
    """Recompute the STORED per-point reprojection errors in place.

    Every pass below that moves a point (the collapse's and the reconciliation's
    retriangulations, the BA) leaves the errors array describing the geometry the
    point USED to have.  The array is derived data, so it is refreshed at each
    such step and unconditionally before the caller saves — a stored error that
    disagrees with the current geometry is worse than no error at all, because
    every downstream consumer (the viewer's quality shading, the infinity
    classifier's noise calibration, any error-based filter) reads it as fact.
    Measured on a reconciled 20250907_000240907 seed before this was done: 121
    finite points (5.4%) whose actual reprojection ran past 3 px — to 388 px —
    while the stored array claimed a file-wide maximum of 3.8 px."""
    recon.recompute_point_errors()
    return recon


def _patch_halfvecs(recon):
    """``(half_u, half_v)`` as dense ``(n, 3)`` arrays, zero where no patch."""
    n = recon.point_count
    hu = np.zeros((n, 3))
    hv = np.zeros((n, 3))
    cloud = recon.patches
    if cloud is None:
        return hu, hv
    pidx = np.asarray(cloud.point_indexes).astype(np.int64)
    for k in range(len(cloud)):
        patch = cloud[k]
        he = patch.half_extent
        hu[pidx[k]] = np.asarray(patch.u_axis, dtype=np.float64) * he[0]
        hv[pidx[k]] = np.asarray(patch.v_axis, dtype=np.float64) * he[1]
    return hu, hv


def _resized_patch_frames(recon, pos4, rots, tv, cam, targets, ref_imgs, radii_px):
    """``(half_u, half_v)`` for ``recon`` with the ``targets`` rows RE-SIZED at
    their current position, directions untouched.

    A patch's world extent is not a property of the surfel alone: it is the world
    size the feature's DETECTION radius subtends at the point's depth in its
    reference view — exactly what ``save_sfmr`` writes, through the camera's own
    ``pixel_radius_to_world_batch`` (see the sizing comment there).  So anything
    that MOVES a point along its ray invalidates the LENGTH of its half-vectors,
    and nothing downstream re-derives it: the BA moves positions, the normal
    passes rewrite directions, the patch-size cull runs earlier.  A far-born point
    re-triangulated near keeps a frame inflated by ``d_old / d_new`` — measured
    2.58x on point 1170 of 20240614_224244438, a container three times the size
    its detection can justify, with 5-20% of a fleet entry's points off the
    writer's sizing invariant ``r_world ~= det_px * depth / f`` by more than 1.3x.

    Hence: every re-triangulation re-derives the length from the SAME
    detection-time radius through the SAME camera rule at the NEW position, and
    keeps the DIRECTIONS.  The frame is square by contract — a SIFT detection is a
    round region, so the surfel it seeds is a square that orientation may tilt but
    nothing may distort (see ``_check_frame_contract``) — so both half-vectors are
    set to the one target radius, which is also what the writer does
    (``half_u, half_v = u3 * ext, v3 * ext``).

    ``targets`` are point indexes, ``ref_imgs`` their parallel reference image
    indexes and ``radii_px`` their parallel detection radii in reference pixels
    (the carried detection-size ledger, unfloored: the collapse's
    ``_COLLAPSE_R_MIN_PX`` floor is a claim about how far apart two features may
    sit, not about how big one IS, and flooring the extent here would write a size
    the detection never had).  Finite points only — a point at infinity carries an
    ANGULAR extent that no depth enters, and takes no part in either
    re-triangulating pass.  A row with no frame, or with no usable detection
    radius, is left exactly as it stands."""
    hu, hv = _patch_halfvecs(recon)
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) == 0:
        return hu, hv
    ref_imgs = np.asarray(ref_imgs, dtype=np.int64)
    radii_px = np.asarray(radii_px, dtype=np.float64)
    lu = np.linalg.norm(hu[targets], axis=1)
    lv = np.linalg.norm(hv[targets], axis=1)
    # CANONICAL camera space (-Z forward) — what the reconstruction stores and
    # what ``pixel_radius_to_world_batch`` takes.  (The writer flips through
    # S = diag(1, -1, -1) only because IT works in the COLMAP frame.)
    xc = np.einsum("nij,nj->ni", rots[ref_imgs], pos4[targets, :3]) + tv[ref_imgs]
    r_world = np.asarray(
        cam.pixel_radius_to_world_batch(
            np.ascontiguousarray(xc), np.ascontiguousarray(radii_px)
        )
    )
    ok = (
        (lu > 0)
        & (lv > 0)
        & np.isfinite(radii_px)
        & (radii_px > 0)
        & np.isfinite(r_world)
        & (r_world > 0)
    )
    t = targets[ok]
    hu[t] *= (r_world[ok] / lu[ok])[:, None]
    hv[t] *= (r_world[ok] / lv[ok])[:, None]
    return hu, hv


def _refresh_normals(recon):
    """Re-derive the stored normals array from the patch frame it must agree with.

    The format's contract is that a finite point's ``normals_xyz`` agrees with its
    frame's ``normalize(u x v)``, and a point at infinity carries ``(0, 0, 0)``.
    The photometric pipeline breaks that: ``refine_normals`` rotates the patch
    CLOUD's frames, but ``compact_to_embedded_patches`` carries the INPUT recon's
    normals array through verbatim
    (``src/sfmtool/_patch_compaction.py``: ``normals = recon.normals[survivors]``
    beside ``kwargs["patches"] = culled_cloud``), so the array keeps describing
    the plane the refinement moved away from.  Measured on shipped seeds: 98.7%
    of DnDTabletop's points and 98.8% of 20250907_000240907's disagreed with
    their own frame, by a median of 31.6 and 77.2 degrees.  The frame is the
    refined quantity, so the frame wins."""
    n = recon.point_count
    if n == 0 or recon.patches is None:
        return recon
    hu, hv = _patch_halfvecs(recon)
    nrm = np.cross(hu, hv)
    ln = np.linalg.norm(nrm, axis=1)
    inf = np.asarray(recon.point_is_at_infinity)
    out = np.zeros((n, 3), np.float32)
    ok = (ln > 0) & ~inf
    out[ok] = (nrm[ok] / ln[ok, None]).astype(np.float32)
    return recon.clone_with_changes(normals=np.ascontiguousarray(out))


def _check_frame_contract(recon, tag):
    """Count violations of the square-frame contract and print them.

    Two invariants, over FINITE patched points: the stored normal agrees with
    ``normalize(u x v)`` to within a degree, and the frame is square
    (``| |u| - |v| | / max <= 1%``).  Points at infinity are exempt from
    squareness — their frame is the world square PROJECTED onto the tangent plane
    of the direction and divided by the demotion distance, and a projection is
    not a similarity, so an infinity patch is legitimately non-square."""
    n = recon.point_count
    if n == 0 or recon.patches is None:
        return
    hu, hv = _patch_halfvecs(recon)
    inf = np.asarray(recon.point_is_at_infinity)
    lu, lv = np.linalg.norm(hu, axis=1), np.linalg.norm(hv, axis=1)
    nrm = np.cross(hu, hv)
    ln = np.linalg.norm(nrm, axis=1)
    stored = np.asarray(recon.normals, dtype=np.float64) if recon.has_normals else None
    live = (lu > 0) & (lv > 0) & ~inf
    n_sq = 0
    if live.any():
        n_sq = int(
            (np.abs(lu[live] - lv[live]) / np.maximum(lu[live], lv[live]) > 0.01).sum()
        )
    n_ang = 0
    if stored is not None:
        sn = np.linalg.norm(stored, axis=1)
        m = live & (ln > 0) & (sn > 0)
        if m.any():
            cos = np.clip(
                (nrm[m] / ln[m, None] * (stored[m] / sn[m, None])).sum(1), -1, 1
            )
            n_ang = int((np.degrees(np.arccos(cos)) > 1.0).sum())
    print(
        f"  frame contract ({tag}): {int(live.sum())} finite patched points; "
        f"{n_sq} non-square (>1% extent mismatch), {n_ang} normal/frame "
        f"disagreements (>1 deg), {int((inf & (lu > 0)).sum())} infinity frames "
        f"(exempt from squareness)"
    )


def _reprojection_medians(recon):
    """Per-point MEDIAN observation reprojection error, ``nan`` where a point has
    no observation.  The median (the stored array carries the MEAN, which the
    format defines) is what the cull judges on: one wild view in an otherwise
    sound track is a bad observation, while a majority of wild views is a bad
    point, and only the median tells those apart."""
    n_pts = recon.point_count
    out = np.full(n_pts, np.nan)
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    if n_pts == 0 or len(ti) == 0:
        return out
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.ascontiguousarray(np.asarray(recon.keypoints_xy, dtype=np.float64))
    q = np.ascontiguousarray(recon.quaternions_wxyz, dtype=np.float64)
    tv = np.ascontiguousarray(recon.translations, dtype=np.float64)
    pos = np.ascontiguousarray(np.asarray(recon.positions, dtype=np.float64))
    res = _reprojection_residuals(
        recon.cameras[0],
        q,
        tv,
        pos,
        uv,
        np.ascontiguousarray(ii.astype(np.uint32)),
        np.ascontiguousarray(ti.astype(np.uint32)),
        1e6,
    )
    err = np.hypot(res[:, 0], res[:, 1])
    order = np.argsort(ti, kind="stable")
    ti_s, err_s = ti[order], err[order]
    starts = np.searchsorted(ti_s, np.arange(n_pts))
    ends = np.searchsorted(ti_s, np.arange(n_pts) + 1)
    for p_k in np.nonzero(ends > starts)[0].tolist():
        out[p_k] = float(np.median(err_s[starts[p_k] : ends[p_k]]))
    return out


def _reprojection_cull_mask(recon):
    """``(keep, bound, n_culled)`` for the reprojection cull.

    The bound is data-derived with an absolute floor:
    ``max(_REPROJ_CULL_FLOOR_PX, REPROJ_CULL_MULT x the finite cloud's median)``.
    Points at infinity are exempt — their residual is a bearing disagreement, not
    a landmark's, and the classifier owns them."""
    keep = np.ones(recon.point_count, bool)
    if REPROJ_CULL_MULT <= 0 or recon.point_count == 0:
        return keep, float("inf"), 0
    med = _reprojection_medians(recon)
    inf = np.asarray(recon.point_is_at_infinity)
    scorable = np.isfinite(med) & ~inf
    if not scorable.any():
        return keep, float("inf"), 0
    bound = max(
        _REPROJ_CULL_FLOOR_PX, REPROJ_CULL_MULT * float(np.median(med[scorable]))
    )
    keep[scorable & (med > bound)] = False
    return keep, bound, int((~keep).sum())


def _rerender_mutated_bitmaps(recon, images, mutated):
    """Re-fuse the consensus patch bitmap of every MUTATED point from its CURRENT
    track, returning ``(recon, n_rendered, n_cleared)``.

    A patch bitmap is baked once at the embed step and then rides verbatim
    through every later mutation, so a point whose track membership changed keeps
    a texture fused from observations it no longer owns — human inspection found
    mostly-black bitmaps with slivers of live content, including pixels from
    observations the reconciliation reassigned away and from images the covis cull
    removed.  Stale is worse than absent, so a point whose re-render produces no
    valid consensus has its bitmap CLEARED rather than left as it was.

    Only mutated points are re-rendered: the pass is the render-only mode of the
    same sub-pixel refiner ``embed_patches`` uses (``sweeps=0`` keeps every seed
    and takes no Gauss-Newton step), seeded at the point's current keypoints, so
    unmutated points keep their bitmaps byte-identical and are not paid for."""
    from sfmtool._embed_patches import _localizations_from_recon, _refine_subpixel

    bmp = recon.patch_bitmaps
    cloud = recon.patches
    if bmp is None or cloud is None or not mutated.any():
        return recon, 0, 0
    locs = [
        loc
        for loc in _localizations_from_recon(recon)
        if mutated[int(loc["point_index"])]
    ]
    if not locs:
        return recon, 0, 0
    # The stored block defines the grid the rest of the cloud is on; a re-render
    # at any other resolution could not be scattered back into it.
    resolution = int(np.asarray(bmp).shape[1])
    _locs, bitmaps, valid = _refine_subpixel(
        cloud,
        recon,
        images,
        locs,
        sweeps=0,
        resolution=resolution,
        render_bitmaps=True,
    )
    out = np.asarray(bmp, dtype=np.uint8).copy()
    if bitmaps is None:
        out[mutated] = 0
        return _reattach_bitmaps(recon, out), 0, int(mutated.sum())
    fresh = mutated & np.asarray(valid, bool)
    stale = mutated & ~fresh
    out[fresh] = np.asarray(bitmaps, dtype=np.uint8)[fresh]
    out[stale] = 0
    return _reattach_bitmaps(recon, out), int(fresh.sum()), int(stale.sum())


def _relocalize_keypoints(recon, images, mutated):
    """Re-localize every eligible point's keypoints through its CURRENT surfel
    frame, re-triangulate the point from them, and write both back wherever the
    result is demonstrably no worse.  Returns ``(recon, stats, mutated)``.

    A keypoint is localized THROUGH the patch warp, and the warp is the surfel
    frame, so a keypoint answers to the normal it was localized under: with the
    normal wrong the sampling window is sheared wrongly and the congealed
    keypoint lands somewhere else — and the wrong keypoints then drag the
    triangulated position.  Nothing upstream re-localizes once the frames
    settle.  ``embed_patches`` runs its discrete localizer ONCE, in round 1,
    against the round-1 normals; every later round re-refines the normals and
    then only SUB-PIXEL-refines the keypoints, and that refiner is bounded at
    ``max_offset_px = 2.0`` PATCH-GRID px from its seed (its own contract asks
    for a seed already inside ~1 px), so it cannot walk a keypoint back to a
    normal that moved under it.  The finalization then moves the patch CENTRES
    again — the duplicate collapse and the alias reconciliation both end in a
    re-triangulation — while leaving every keypoint exactly where it was.

    Per point: the discrete localizer seeded at the STORED keypoints.  The keypoint
    marks where the feature's centre IS in the image — the centre of the observed
    bitmap content — and that 2D location is independent of the normal, which
    only shears the sampling window around it.  What goes stale when a frame
    moves is the WARP, not the location, so the fresh answer starts the search at
    the location that was actually measured and re-localizes through the current
    warp; a projection seed instead injects the triangulation's own error into
    the search (measured on displaced centres: keypoint seeds recover to
    ~0.03 px where projection seeds land 6.5-7 px off).  Then the same sub-pixel
    refiner ``embed_patches`` ends each round with, then a re-triangulation from
    the refined keypoints.  A view the localizer drops keeps its stored
    keypoint: membership is not changed here, and the post-BA reprojection cull
    is what judges a point whose remaining views disagree.

    ACCEPTANCE is per point, never blanket.  The re-triangulation must be finite,
    in front of EVERY camera that observes the point, backed by enough
    re-localized views, and its MEDIAN reprojection — over the point's full
    observation set, each candidate scored against the keypoints IT owns — no
    worse than the stored geometry's.  Scoring the new position against the
    STORED keypoints is meaningless once the keypoints have moved, so the guard
    scores against the refined ones.  A point that fails keeps its keypoints and
    its position untouched.

    Points at infinity are skipped: their ``positions`` row is a direction, so
    there is no triangulation to redo, and their frame is the fixed tangent-plane
    one the format's infinity convention fixes rather than a refined normal."""
    from sfmtool._embed_patches import _refine_subpixel
    from sfmtool._sfmtool.analysis import triangulate_batch
    from sfmtool._sfmtool.patches import ImagePyramidSet

    st = {
        "eligible": 0,
        "candidates": 0,
        "accepted": 0,
        "rej_views": 0,
        "rej_degenerate": 0,
        "rej_cheirality": 0,
        "rej_reproj": 0,
        "moves": np.zeros(0),
        "move3d_med": 0.0,
        "move3d_rel": 0.0,
        "secs": 0.0,
    }
    n = recon.point_count
    cloud = recon.patches
    if not RELOCALIZE_ENABLED or cloud is None or n == 0:
        return recon, st, mutated
    t0 = time.perf_counter()
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, dtype=np.float64)
    pos = np.asarray(recon.positions, dtype=np.float64)
    inf = np.asarray(recon.point_is_at_infinity)
    hu, hv = _patch_halfvecs(recon)
    tl = np.bincount(ti, minlength=n)
    eligible = (
        ~inf
        & (tl >= _RELOCALIZE_MIN_VIEWS)
        & (np.linalg.norm(hu, axis=1) > 0)
        & (np.linalg.norm(hv, axis=1) > 0)
    )
    ids = np.nonzero(eligible)[0]
    st["eligible"] = len(ids)
    if len(ids) == 0:
        return recon, st, mutated

    # Per-point observation rows, image-sorted, so a localized view maps back to
    # its row by search rather than by a dict over every observation.
    order = np.argsort(ti, kind="stable")
    ti_s = ti[order]
    starts = np.searchsorted(ti_s, np.arange(n))
    ends = np.searchsorted(ti_s, np.arange(n) + 1)
    rows_of, view_sets = {}, {}
    for p in ids.tolist():
        rws = order[starts[p] : ends[p]]
        rws = rws[np.argsort(ii[rws], kind="stable")]
        rows_of[p] = rws
        view_sets[p] = ii[rws].astype(np.uint32).tolist()

    # Localize + sub-pixel refine on the grid the stored consensus textures were
    # fused on (the embed default when a recon carries none).  Everything else —
    # search radius, ZNCC bar, consensus-basis cap — is the embed pipeline's own
    # default, so this is the same kernel pair under the same settings, differing
    # only in the shift gate and in the geometry it answers to.
    bmp = recon.patch_bitmaps
    resolution = int(bmp.shape[1]) if bmp is not None else 24
    pyramids = ImagePyramidSet(recon, images)
    seeds = {int(p): uv[rows_of[p]].tolist() for p in ids.tolist()}
    locs = cloud.localize_keypoints(
        recon,
        pyramids,
        view_sets=view_sets,
        point_indexes=[int(p) for p in ids],
        max_shift_px=RELOCALIZE_MAX_SHIFT_PX,
        resolution=resolution,
        starting_keypoints=seeds,
    )
    locs = [loc for loc in locs if len(np.asarray(loc["views"]))]
    locs, _bitmaps, _valid = _refine_subpixel(
        cloud, recon, pyramids, locs, sweeps=1, resolution=resolution
    )

    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    tv = np.asarray(recon.translations, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    centers = -np.einsum("nij,ni->nj", rots, tv)
    cam = recon.cameras[0]

    uv_new = uv.copy()
    cand, cand_rows, kept_rows, dirs, ctrs, counts = [], [], [], [], [], []
    for loc in locs:
        p = int(loc["point_index"])
        rws = rows_of.get(p)
        if rws is None:
            continue
        v = np.asarray(loc["views"], dtype=np.int64)
        k = np.asarray(loc["keypoints"], dtype=np.float64).reshape(-1, 2)
        imgs = ii[rws]
        at = np.searchsorted(imgs, v)
        hit = (at < len(imgs)) & (imgs[np.minimum(at, len(imgs) - 1)] == v)
        v, k, at = v[hit], k[hit], at[hit]
        if len(v) < _RELOCALIZE_MIN_VIEWS or len(v) < _RELOCALIZE_MIN_KEEP_FRAC * len(
            rws
        ):
            continue
        kept = rws[at]
        uv_new[kept] = k
        d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(k))
        dirs.append(np.einsum("nji,nj->ni", rots[imgs[at]], d_loc))
        ctrs.append(centers[imgs[at]])
        counts.append(len(kept))
        cand.append(p)
        cand_rows.append(rws)
        kept_rows.append(kept)
    st["candidates"] = len(cand)
    st["rej_views"] = st["eligible"] - st["candidates"]
    if not cand:
        st["secs"] = time.perf_counter() - t0
        return recon, st, mutated

    # Re-triangulate every candidate from its refined keypoints (ray midpoint,
    # the same batch kernel the seed structure and both merge passes use).
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    X = np.asarray(
        triangulate_batch(
            np.ascontiguousarray(np.vstack(dirs)),
            np.ascontiguousarray(np.vstack(ctrs)),
            offs,
        )["points"],
        dtype=np.float64,
    )
    cand = np.asarray(cand, dtype=np.int64)
    accept = np.isfinite(X).all(axis=1)
    st["rej_degenerate"] = int((~accept).sum())
    pos_new = pos.copy()
    pos_new[cand[accept]] = X[accept]

    # Both candidates over the SAME observations: the stored geometry against the
    # stored keypoints, the re-localized one against the refined keypoints it
    # owns (a view the localizer dropped contributes its stored keypoint to both,
    # so a point whose correction leaves the rest of its track behind is judged
    # for it rather than rewarded).
    rows_all = np.concatenate(cand_rows)
    grp = np.concatenate([[0], np.cumsum([len(r) for r in cand_rows])]).astype(np.int64)
    im_all = np.ascontiguousarray(ii[rows_all].astype(np.uint32))
    pt_all = np.ascontiguousarray(ti[rows_all].astype(np.uint32))

    def _errs(points, keypoints):
        res = _reprojection_residuals(
            cam, q, tv, points, np.ascontiguousarray(keypoints), im_all, pt_all, 1e6
        )
        return np.hypot(res[:, 0], res[:, 1])

    e_old = _errs(pos, uv[rows_all])
    e_new = _errs(pos_new, uv_new[rows_all])
    # Cheirality over the FULL observation set (canonical -Z forward), not just
    # the views the localizer kept: a point must sit in front of every camera
    # that claims to see it, exactly as the standing cull demands — and "in
    # front" is the MODEL's imaged cone (``_in_field``), so a fisheye's
    # periphery is not read as a violation.
    p_cam_all = (
        np.einsum("nij,nj->ni", rots[ii[rows_all]], pos_new[ti[rows_all]])
        + tv[ii[rows_all]]
    )
    depth_ok = (
        np.minimum.reduceat(_in_field(cam, p_cam_all).astype(np.int8), grp[:-1]) > 0
    )
    for j in range(len(cand)):
        if not accept[j]:
            continue
        if not depth_ok[j]:
            accept[j] = False
            st["rej_cheirality"] += 1
        elif np.median(e_new[grp[j] : grp[j + 1]]) > np.median(
            e_old[grp[j] : grp[j + 1]]
        ):
            accept[j] = False
            st["rej_reproj"] += 1
    st["accepted"] = int(accept.sum())
    if not accept.any():
        st["secs"] = time.perf_counter() - t0
        return recon, st, mutated

    acc_ids = cand[accept]
    acc_rows = np.concatenate([kept_rows[j] for j in np.nonzero(accept)[0]])
    st["moves"] = np.linalg.norm(uv_new[acc_rows] - uv[acc_rows], axis=1)
    d3 = np.linalg.norm(X[accept] - pos[acc_ids], axis=1)
    st["move3d_med"] = float(np.median(d3))
    depth_all = _cam_depth(p_cam_all)
    med_depth = (
        float(np.median(depth_all[depth_all > 0])) if (depth_all > 0).any() else 0.0
    )
    st["move3d_rel"] = st["move3d_med"] / med_depth if med_depth > 0 else 0.0

    # Homogeneous write-back: a bare (n, 3) positions block would re-promote every
    # w = 0 point to finite (none is a candidate, but the block spans the cloud).
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    xyzw[acc_ids, :3] = X[accept]
    uv_out = uv.copy()
    uv_out[acc_rows] = uv_new[acc_rows]
    out = recon.clone_with_changes(
        positions=np.ascontiguousarray(xyzw),
        keypoints_xy=np.ascontiguousarray(uv_out.astype(np.float32)),
    )
    # An accepted point's keypoints and its centre both moved, so its baked
    # consensus bitmap no longer describes it: re-render it with the rest.
    mutated = mutated.copy()
    mutated[acc_ids] = True
    st["secs"] = time.perf_counter() - t0
    return out, st, mutated


def _print_relocalize(tag, st):
    """One line of keypoint re-localization evidence."""
    mv = st["moves"]
    med, p90, mx = (
        (float(np.median(mv)), float(np.percentile(mv, 90)), float(mv.max()))
        if len(mv)
        else (0.0, 0.0, 0.0)
    )
    rate = 100.0 * st["accepted"] / max(st["candidates"], 1)
    print(
        f"  keypoint re-localization ({tag}): {st['eligible']} eligible, "
        f"{st['candidates']} candidates, {st['accepted']} accepted ({rate:.0f}%); "
        f"rejected {st['rej_views']} too-few-views / {st['rej_degenerate']} "
        f"degenerate / {st['rej_cheirality']} cheirality / {st['rej_reproj']} "
        f"worse-reprojection; keypoint move med {med:.2f} px, p90 {p90:.2f} px, "
        f"max {mx:.2f} px ({int((mv > 3.0).sum())} obs past the production 3 px "
        f"gate); position move med {st['move3d_med']:.4f} "
        f"({100 * st['move3d_rel']:.2f}% of the median depth); {st['secs']:.1f}s"
    )


def _reattach_bitmaps(recon, bitmaps):
    """Write a full bitmap block back.  The patch FRAME is left alone — passing
    ``patches=`` would clear the bitmaps in the same call — so this is only valid
    on a recon that already carries a frame, which is the only case it is used in."""
    return recon.clone_with_changes(
        patch_bitmaps=np.ascontiguousarray(np.asarray(bitmaps, dtype=np.uint8))
    )


def _cluster_detection_sizes(data):
    """Per-cluster DETECTION-TIME patch half-extent, in pixels, shape ``(n_cl,)``.

    This is the depth-independent quantity the seed's own surfel frames were
    built from: ``save_sfmr`` sizes each patch by ``refine_radius x the
    reference feature's scale``, with the scale read from the image's ``.sift``
    affine shapes as the mean of the two column norms.  Reading it back the same
    way gives the extent the feature was DETECTED at — a property of the image,
    not of the current triangulation — where
    ``max_embedded_feature_size_per_point`` gives that extent reprojected at the
    point's current depth (see COLLAPSE_ENABLED for why the difference matters).

    A cluster's size is the MEDIAN over its member observations: the members are
    the same feature seen at slightly different scales, and the median is the
    robust representative of the family.  ``obs_f`` holds each member's feature
    INDEX into its image's ``.sift``, so only the shapes array is read, and only
    up to the highest index an image actually contributes."""
    from sfmtool._sfmtool.io import read_sift_partial
    from sfmtool.sift.file import get_sift_path_for_image

    obs_c = np.asarray(data["obs_c"], dtype=np.int64)
    obs_i = np.asarray(data["obs_i"], dtype=np.int64)
    obs_f = np.asarray(data["obs_f"], dtype=np.int64)
    n_cl = int(data["n_cl"])
    r_kf = float(data["refine_radius"])
    ws = WS.resolve()
    size = np.full(len(obs_c), np.nan)
    for j, name in enumerate(data["names"]):
        m = obs_i == j
        if not m.any():
            continue
        shapes = np.asarray(
            read_sift_partial(
                get_sift_path_for_image(ws / name), int(obs_f[m].max()) + 1
            )["affine_shapes"],
            dtype=np.float64,
        )
        scale = 0.5 * (
            np.linalg.norm(shapes[:, :, 0], axis=1)
            + np.linalg.norm(shapes[:, :, 1], axis=1)
        )
        size[m] = r_kf * scale[obs_f[m]]
    order = np.argsort(obs_c, kind="stable")
    bounds = np.searchsorted(obs_c[order], np.arange(n_cl + 1))
    out = np.zeros(n_cl)
    for c in range(n_cl):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        if hi > lo:
            block = size[order[lo:hi]]
            block = block[np.isfinite(block)]
            if len(block):
                out[c] = float(np.median(block))
    return out


def _seed_index_map(seed_positions, dst_positions):
    """Per-destination-point index into ``seed_positions``, or -1.

    The photometric pipeline between the seed recon and its refined descendants
    only ever CULLS points and copies their positions verbatim, so a seed point
    is identified downstream by its exact float64 position.  Anything that MOVES
    a point (the collapse's re-triangulation, the BA) breaks the identity, which
    is why every caller must build the map before such a step and carry it
    forward through the masks it applies itself."""
    src = np.ascontiguousarray(np.asarray(seed_positions, dtype=np.float64))
    dst = np.ascontiguousarray(np.asarray(dst_positions, dtype=np.float64))
    key = {row.tobytes(): k for k, row in enumerate(src)}
    return np.array([key.get(dst[i].tobytes(), -1) for i in range(len(dst))], np.int64)


def _collapse_duplicate_points(emb, sizes, mutated, uid):
    """Merge near-duplicate points: ONE surface feature detected several times
    in the same images, never joined into a single track, each fragment
    triangulated on its own narrow per-fragment baseline — so the fragments
    scatter in depth ALONG the ray while sitting on top of each other in the
    image (human inspection found laterally-overlapping points at wildly
    different depths; on 20250907_000240907, 70.5% of the seed's points sat in
    such a pair and 57% of those pairs disagreed by >5% in depth).

    A CLUSTER is a set of points being treated as one feature; it carries one
    REPRESENTATIVE observation per image (the lowest current reprojection error,
    the longer-track member's on a tie), which is exactly the track the merge
    would write, so every test below reads the merged geometry rather than the
    pre-merge fragments.  Every point starts as its own cluster.

    Positive linkage — the criterion the overlap diagnostic validated: two
    clusters may merge when their representatives sit within the pair's collapse
    radius of each other in at least half of the images they share, and they
    share at least 2 images (see COLLAPSE_ENABLED for the radius policy).

    CANNOT-LINK: if the two clusters are co-observed in ANY shared image with
    any pair of their MEMBER observations at least 2x the collapse radius apart,
    the merge is forbidden.  That image is direct negative evidence of
    distinctness — one feature cannot be in two places in one picture — and it
    outranks any amount of positive evidence elsewhere.  It is what makes the
    uncapped radius safe: single-linkage components would chain A-B-C together on
    transitive overlap while A and C sit far apart in a shared view (measured on
    a shipped 20250907_000240907 seed: 3083 qualifying pairs single-link into a
    528-point component spanning 1490 px).  The constraint is checked against the
    FULL member-observation lists, not the representatives: the merged track
    keeps one observation per image, so a representative can move up to 2x the
    radius per merge, and a chain checked only against representatives can DRIFT
    arbitrarily far while never presenting a single blocking image (measured
    before the fix: 56-member groups spanning 1106 px).  Keeping the history
    makes the check strictly stricter as a cluster grows.

    Agglomeration to a FIXPOINT, best candidate first (highest in-radius
    fraction; ties by cluster index): merge, rebuild the merged cluster's
    representative track, re-detect only in the merged cluster's own
    neighborhood, repeat.  Merge products are therefore re-examined — the defect
    the old one-sweep union-find left behind (1314/1605 above are both survivors
    of the first sweep, and their merged tracks then satisfy the criterion in
    10/10 shared images).  An outer sweep re-detects globally and confirms the
    fixpoint.

    Each group's merged track lands on its longest-track member (ties by point
    index), is re-triangulated from the current poses, and the other members are
    dropped.  The survivor's patch frame is RE-SIZED at the merged depth — it was
    born at the survivor's own pre-merge depth, and a re-triangulation that moves
    the point along its ray leaves the extent describing a depth the point no
    longer has (see ``_resized_patch_frames``).  A merged cluster's feature size
    is the MIN over its members, so the radius can only SHRINK as a cluster
    grows — a merge can never widen the net it casts.  Points at infinity carry
    a direction rather than a depth — no split-depth failure mode, and no
    midpoint re-triangulation — so they stay out of the pass entirely.

    ``sizes`` is the per-point detection-time feature size (see
    ``_cluster_detection_sizes``); a point with no size (0 or non-finite) gets
    the floor radius.  Returns ``(recon, stats, sizes, mutated, uid)`` with the
    carried arrays of the surviving points, so the caller can keep carrying them
    — a merged group's survivor keeps its own row (and so its uid) and the
    members it absorbed drop.  stats keys:
    ``merged``, ``groups``, ``blocked``, ``rounds``, ``max_group``,
    ``max_group_sep``, ``max_sep``.  Deterministic: no RNG, every candidate
    ordering breaks ties on indices, and every group and member list is
    index-sorted."""
    import heapq

    from sfmtool._sfmtool.analysis import triangulate_batch
    from sfmtool._sfmtool.patches import PatchCloud

    stats = {
        "merged": 0,
        "groups": 0,
        "blocked": 0,
        "rounds": 0,
        "max_group": 0,
        "max_group_sep": 0.0,
        "max_sep": 0.0,
    }
    n_pts = emb.point_count
    ti = np.asarray(emb.track_point_indexes).astype(np.int64)
    ii = np.asarray(emb.track_image_indexes).astype(np.int64)
    uv = np.asarray(emb.keypoints_xy, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if not COLLAPSE_ENABLED or n_pts < 2 or len(ti) < 2:
        return emb, stats, sizes, mutated, uid

    # Per-point search radius: the point's own detection size (a SEMI-AXIS mean,
    # i.e. already a radius — see the COLLAPSE_RADIUS_SCALE convention note) times
    # the scale, floored.  The PAIR radius is min(R_a, R_b), which is identically
    # max(floor, scale * min(size_a, size_b)) — so one per-point number both
    # defines the pair criterion and bounds the neighborhood a point has to search
    # (nothing outside R_a can pair with a, whatever its own size).
    def _radius(size):
        return (
            _COLLAPSE_R_MIN_PX
            if not np.isfinite(size) or size <= 0
            else max(_COLLAPSE_R_MIN_PX, COLLAPSE_RADIUS_SCALE * float(size))
        )

    rad = np.maximum(_COLLAPSE_R_MIN_PX, COLLAPSE_RADIUS_SCALE * sizes)
    rad[~np.isfinite(sizes) | (sizes <= 0)] = _COLLAPSE_R_MIN_PX

    inf = np.asarray(emb.point_is_at_infinity)
    if int((~inf).sum()) < 2:
        return emb, stats, sizes, mutated, uid

    # Per-observation reprojection error (the merge's evidence for which of two
    # observations of the same image to keep), read ONCE at the pass's input
    # state so the representative rule cannot depend on merge order.
    cam = emb.cameras[0]
    q = np.ascontiguousarray(emb.quaternions_wxyz, dtype=np.float64)
    tv = np.ascontiguousarray(emb.translations, dtype=np.float64)
    # Homogeneous positions: a bare (n, 3) block would re-promote every w = 0
    # point to finite when this pass runs after the infinity classification.
    pos4 = np.asarray(emb.positions_xyzw, dtype=np.float64).copy()
    res = _reprojection_residuals(
        cam,
        q,
        tv,
        np.ascontiguousarray(pos4[:, :3]),
        np.ascontiguousarray(uv),
        np.ascontiguousarray(ii.astype(np.uint32)),
        np.ascontiguousarray(ti.astype(np.uint32)),
        1e6,
    )
    err = np.hypot(res[:, 0], res[:, 1])
    err = np.where(np.isfinite(err), err, 1e9)
    tlen = np.bincount(ti, minlength=n_pts)
    row_order = np.argsort(ti, kind="stable")
    row_start = np.searchsorted(ti[row_order], np.arange(n_pts + 1))
    err_l, ux, uy = err.tolist(), uv[:, 0].tolist(), uv[:, 1].tolist()
    tlen_l, ti_l, ii_l = tlen.tolist(), ti.tolist(), ii.tolist()

    def _rep_key(r):
        """Ordering of two observations of the same image within a cluster."""
        p_k = ti_l[r]
        return (err_l[r], -tlen_l[p_k], p_k, r)

    # ── Cluster state ────────────────────────────────────────────────────────
    # A cluster is keyed by the point index it was created from and holds:
    # reps[c] (image -> representative row), allobs[c] (image -> EVERY member
    # observation, the cannot-link constraint set), members[c], surv[c] (the
    # point that will carry the merged track), csize[c]/crad[c] (its feature size
    # and search radius), version[c] (any change invalidates queued candidates).
    n_img = len(emb.image_names)
    reps = [None] * n_pts
    allobs = [None] * n_pts
    for r in np.nonzero(~inf[ti])[0].tolist():
        c, g = ti_l[r], ii_l[r]
        d = reps[c]
        if d is None:
            d, allobs[c] = {}, {}
            reps[c] = d
        cur = d.get(g)
        if cur is None or _rep_key(r) < _rep_key(cur):
            d[g] = r
        allobs[c].setdefault(g, []).append(r)
    active = np.zeros(n_pts, bool)
    img_rep = [dict() for _ in range(n_img)]
    for c in range(n_pts):
        if reps[c]:
            active[c] = True
            for g, r in reps[c].items():
                img_rep[g][c] = r
    members = {c: [c] for c in np.nonzero(active)[0].tolist()}
    surv = {c: c for c in members}
    csize = sizes.copy()
    crad = rad.copy()
    version = np.zeros(n_pts, np.int64)
    cache = [None] * n_img

    def _img_arrays(g):
        """(cluster ids, representative rows) of one image, cached until it
        changes.  Keys and values of a dict iterate in the same order."""
        c = cache[g]
        if c is None:
            d = img_rep[g]
            c = cache[g] = (
                np.fromiter(d.keys(), np.int64, len(d)),
                np.fromiter(d.values(), np.int64, len(d)),
            )
        return c

    def _profile(a, b):
        """``(in-radius fraction, cannot_link)`` for a cluster pair, or None
        when the positive criterion does not hold.  The positive test reads the
        REPRESENTATIVES (the track the merge would write); the cannot-link test
        reads EVERY member observation (no drift past the constraint)."""
        ra, rb = reps[a], reps[b]
        shared = sorted(ra.keys() & rb.keys())
        n_sh = len(shared)
        if n_sh < 2:
            return None
        rr = float(min(crad[a], crad[b]))
        near, far = rr * rr, 4.0 * rr * rr
        oa, ob = allobs[a], allobs[b]
        n_in, block = 0, False
        for g in shared:
            ka, kb = ra[g], rb[g]
            dx, dy = ux[ka] - ux[kb], uy[ka] - uy[kb]
            if dx * dx + dy * dy <= near:
                n_in += 1
            if not block:
                for ja in oa[g]:
                    xa, ya = ux[ja], uy[ja]
                    for jb in ob[g]:
                        dx, dy = xa - ux[jb], ya - uy[jb]
                        if dx * dx + dy * dy >= far:
                            block = True
                            break
                    if block:
                        break
        if n_in < max(2, 0.5 * n_sh):
            return None
        return n_in / n_sh, block

    def _neighbors(c):
        """Cluster ids sharing an image with ``c`` inside the pair radius (a
        superset of c's candidates: the box test bounds the disk test)."""
        out = set()
        rc = crad[c]
        for g, row in reps[c].items():
            ks, vs = _img_arrays(g)
            if len(ks) < 2:
                continue
            lim = np.minimum(crad[ks], rc)
            m = (
                (np.abs(uv[vs, 0] - ux[row]) <= lim)
                & (np.abs(uv[vs, 1] - uy[row]) <= lim)
                & (ks != c)
            )
            out.update(ks[m].tolist())
        return out

    def _detect_all():
        """Every candidate cluster pair, globally: per image a sort-by-x sliding
        window whose width is the LEFT member's own radius (never the O(n^2)
        all-pairs product; the pair radius is the min, so a window of R_a finds
        every pair a can be in)."""
        cl, rw = [], []
        for c in np.nonzero(active)[0].tolist():
            for r in reps[c].values():
                cl.append(c)
                rw.append(r)
        cl, rw = np.asarray(cl, np.int64), np.asarray(rw, np.int64)
        im = ii[rw]
        srt = np.lexsort((cl, uv[rw, 1], uv[rw, 0], im))
        cl, rw, im = cl[srt], rw[srt], im[srt]
        pairs = set()
        _, starts_i, counts_i = np.unique(im, return_index=True, return_counts=True)
        for st, cnt in zip(starts_i.tolist(), counts_i.tolist()):
            if cnt < 2:
                continue
            x, y = uv[rw[st : st + cnt], 0], uv[rw[st : st + cnt], 1]
            p = cl[st : st + cnt]
            rp = crad[p]
            k = np.searchsorted(x, x + rp, side="right") - np.arange(cnt) - 1
            tot = int(k.sum())
            if tot == 0:
                continue
            a = np.repeat(np.arange(cnt), k)
            b = a + 1 + (np.arange(tot) - np.repeat(np.cumsum(k) - k, k))
            rr = np.minimum(rp[a], rp[b])
            m = (x[b] - x[a] <= rr) & (np.abs(y[b] - y[a]) <= rr)
            pairs.update(
                zip(
                    np.minimum(p[a[m]], p[b[m]]).tolist(),
                    np.maximum(p[a[m]], p[b[m]]).tolist(),
                )
            )
        return pairs

    def _merge(a, b):
        """Absorb ``b`` into ``a`` (the lower index keeps the cluster) and
        rebuild the merged representative track.  Returns the surviving id."""
        keep, gone = (a, b) if a < b else (b, a)
        mem = sorted(members[keep] + members[gone])
        ra, rb = reps[keep], reps[gone]
        new = {}
        for g in ra.keys() | rb.keys():
            ka, kb = ra.get(g), rb.get(g)
            if ka is None or kb is None:
                new[g] = ka if kb is None else kb
            else:
                new[g] = ka if _rep_key(ka) <= _rep_key(kb) else kb
        for g in rb:
            del img_rep[g][gone]
            cache[g] = None
        for g, r in new.items():
            if img_rep[g].get(keep) != r:
                img_rep[g][keep] = r
                cache[g] = None
        oa, ob = allobs[keep], allobs[gone]
        for g, rows in ob.items():
            oa.setdefault(g, []).extend(rows)
        for rows in oa.values():
            rows.sort()
        reps[keep], reps[gone] = new, None
        allobs[gone] = None
        members[keep] = mem
        del members[gone]
        surv[keep] = min(mem, key=lambda p_k: (-tlen_l[p_k], p_k))
        del surv[gone]
        # MIN over members: a merge may only narrow the cluster's net, never
        # widen it (a size is a claim about ONE feature's extent, and the
        # smallest member's claim is the one that has to hold).
        s_a, s_b = csize[keep], csize[gone]
        csize[keep] = min(s_a, s_b) if (s_a > 0 and s_b > 0) else max(s_a, s_b)
        crad[keep] = _radius(csize[keep])
        active[gone] = False
        version[keep] += 1
        return keep

    # ── Best-first agglomeration to the fixpoint ─────────────────────────────
    blocked = set()

    def _consider(a, b, heap):
        pr = _profile(a, b)
        if pr is None:
            return
        frac, block = pr
        entry = (a, b, int(version[a]), int(version[b]))
        if block:
            blocked.add(entry)
        else:
            heapq.heappush(heap, (-frac,) + entry)

    while stats["rounds"] < _COLLAPSE_MAX_ROUNDS:
        heap = []
        for a, b in sorted(_detect_all()):
            _consider(a, b, heap)
        if not heap:
            break
        stats["rounds"] += 1
        heapq.heapify(heap)
        while heap:
            _negf, a, b, va, vb = heapq.heappop(heap)
            if not (active[a] and active[b]):
                continue
            if version[a] != va or version[b] != vb:
                continue  # stale: re-detected in the merge that changed it
            keep = _merge(a, b)
            for x in sorted(_neighbors(keep)):
                _consider(min(keep, x), max(keep, x), heap)
    stats["blocked"] = len(blocked)
    groups = [
        (c, members[c]) for c in sorted(members) if active[c] and len(members[c]) > 1
    ]
    if not groups:
        return emb, stats, sizes, mutated, uid

    # ── Materialize: one track per group, re-triangulated ────────────────────
    out_sizes = sizes.copy()
    drop = np.zeros(n_pts, bool)
    in_group = np.zeros(n_pts, bool)
    kept_rows, tri_counts, survivors = [], [], []
    for c, mem in groups:
        s = surv[c]
        out_sizes[s] = csize[c]
        for p_k in mem:
            in_group[p_k] = True
            if p_k != s:
                drop[p_k] = True
        sel = [reps[c][g] for g in sorted(reps[c])]
        survivors.append(s)
        kept_rows.extend(sel)
        tri_counts.append(len(sel))
        # Chaining audit: how far apart do this group's MEMBER observations sit
        # in a single image?  A healthy merge stays at patch scale.
        per_img = {}
        for p_k in mem:
            for r in row_order[row_start[p_k] : row_start[p_k + 1]].tolist():
                per_img.setdefault(ii_l[r], []).append(r)
        sep = 0.0
        for rr in per_img.values():
            if len(rr) < 2:
                continue
            pxy = uv[rr]
            dd = np.hypot(
                pxy[:, None, 0] - pxy[None, :, 0], pxy[:, None, 1] - pxy[None, :, 1]
            )
            sep = max(sep, float(dd.max()))
        stats["max_sep"] = max(stats["max_sep"], sep)
        if len(mem) > stats["max_group"]:
            stats["max_group"], stats["max_group_sep"] = len(mem), sep

    # Re-triangulate every merged track from the current poses (ray midpoint,
    # the same batch kernel the seed structure is built with).
    tri_rows = np.asarray(kept_rows, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(tri_counts)]).astype(np.int64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(uv[tri_rows]))
    img_of = ii[tri_rows]
    dirs = np.einsum("nji,nj->ni", rots[img_of], d_loc)
    centers = -np.einsum("nji,nj->ni", rots[img_of], tv[img_of])
    tri = np.asarray(
        triangulate_batch(
            np.ascontiguousarray(dirs), np.ascontiguousarray(centers), offsets
        )["points"]
    )
    surv_idx = np.asarray(survivors, dtype=np.int64)
    good = np.isfinite(tri).all(axis=1)
    pos4[surv_idx[good], :3] = tri[good]

    # Rebuild the track arrays: every non-group observation as it stands, plus
    # the merged groups' kept observations re-pointed at their survivor.
    new_ti = ti.copy()
    new_ti[tri_rows] = np.repeat(surv_idx, tri_counts)
    keep_row = ~in_group[ti]
    keep_row[tri_rows] = True
    fin = np.nonzero(keep_row)[0]
    fin = fin[np.lexsort((ii[fin], new_ti[fin]))]  # tracks stay grouped by point
    changes = {
        "positions": np.ascontiguousarray(pos4),
        "track_point_indexes": np.ascontiguousarray(new_ti[fin].astype(np.uint32)),
        "track_image_indexes": np.ascontiguousarray(ii[fin].astype(np.uint32)),
        # unused: embedded
        "track_feature_indexes": np.zeros(len(fin), np.uint32),
        "keypoints_xy": np.ascontiguousarray(uv[fin].astype(np.float32)),
    }
    # The survivor kept the frame it was BORN with, whose extent encodes its
    # PRE-merge depth, and the line above just re-triangulated it somewhere else.
    # Re-derive the extent at the merged depth (see ``_resized_patch_frames``).
    # The reference view is the one the merged track itself names: the lowest
    # ``_rep_key`` — lowest reprojection error, longer track then lower index on a
    # tie — among the observations the merge KEPT, the same ordering that picked
    # each of them.  The detection radius is the merged cluster's own size, the
    # MIN over its members, already written into ``out_sizes``.
    if emb.patches is not None:
        ref_rows = np.array(
            [
                min(kept_rows[int(offsets[g]) : int(offsets[g + 1])], key=_rep_key)
                for g in range(len(surv_idx))
            ],
            dtype=np.int64,
        )
        hu_out, hv_out = _resized_patch_frames(
            emb,
            pos4,
            rots,
            tv,
            cam,
            surv_idx[good],
            ii[ref_rows[good]],
            out_sizes[surv_idx[good]],
        )
        changes["patches"] = PatchCloud.from_halfvec_arrays(
            np.ascontiguousarray(hu_out, dtype=np.float32),
            np.ascontiguousarray(hv_out, dtype=np.float32),
            np.ascontiguousarray(pos4[:, :3]),
        )
        bmp = emb.patch_bitmaps
        if bmp is not None:
            # ``patches=`` alone would clear the consensus textures.
            changes["patch_bitmaps"] = np.ascontiguousarray(bmp)
    out = emb.clone_with_changes(**changes)
    out = out.filter_points_by_mask(np.ascontiguousarray(~drop))
    stats["merged"], stats["groups"] = int(drop.sum()), len(groups)
    mutated = mutated.copy()
    mutated[surv_idx] = True
    return out, stats, out_sizes[~drop], mutated[~drop], uid[~drop]


def _print_collapse(tag, st):
    """One line of collapse evidence, including the chaining audit."""
    print(
        f"  collapse ({tag}): {st['merged']} points merged away in "
        f"{st['groups']} groups over {st['rounds']} round(s), "
        f"{st['blocked']} cannot-link blocks; largest group {st['max_group']} "
        f"members spanning {st['max_group_sep']:.1f} px in one image "
        f"(worst group {st['max_sep']:.1f} px)"
    )


def _conflicted_pairs(recon, sizes):
    """CONFLICTED point pairs and their depth disagreement.

    A pair is conflicted when the two points are observed within their proximity
    radius of each other in at least one shared image AND at least 2x that radius
    apart in another.  Both facts about the same pair cannot describe two
    surfaces: something in the pair's observation set does not belong to it.

    The proximity radius is HALF the finer detection size — deliberately tighter
    than the collapse's merge radius (the full semi-axis extent; see the
    COLLAPSE_RADIUS_SCALE convention note) and held fixed independently of it.
    This is a DETECTOR, not a merge criterion: the benign/malignant split and the
    cull gate below are calibrated against the pair population it selects, so
    widening it would move the reconciliation's thresholds off their calibration.

    The split is the median camera-frame DEPTH mismatch, ``|d_a - d_b| /
    min(d_a, d_b)``:

    - BENIGN (< _ALIAS_MALIGNANT_MISMATCH) is two distinct NEIGHBORS grazing —
      the same surface patch of scene at nearly the same distance, drifting in and
      out of each other's radius with viewpoint.  Harmless; the collapse must
      still refuse to merge them (that is what its cannot-link is for), but there
      is nothing to fix.
    - MALIGNANT is repeated-pattern ALIASING (railings, roof tiles, brickwork):
      the tracks have stitched observations of DIFFERENT physical features
      together, and each triangulation splits the difference, so the pair
      disagrees about depth by far more than its lateral separation can explain.

    Returns ``(pairs, mism, depth_med)`` — an index-sorted list of (a, b), the
    matching depth mismatches, and the per-point median depth.  Points at
    infinity carry no depth and take no part."""
    n_pts = recon.point_count
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, dtype=np.float64)
    inf = np.asarray(recon.point_is_at_infinity)
    depth_med = np.zeros(n_pts)
    if n_pts < 2 or len(ti) < 2:
        return [], np.zeros(0), depth_med
    rad = np.maximum(_COLLAPSE_R_MIN_PX, 0.5 * np.asarray(sizes, dtype=np.float64))
    rad[~np.isfinite(sizes) | (np.asarray(sizes) <= 0)] = _COLLAPSE_R_MIN_PX

    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    tv = np.asarray(recon.translations, dtype=np.float64)
    pc = np.einsum("nij,nj->ni", rots[ii], np.asarray(recon.positions)[ti]) + tv[ii]
    depth = _cam_depth(pc)

    # One observation per (point, image) — the same representative view the
    # collapse reasons about; a conflicted pair is a statement about tracks.
    obs_of = [None] * n_pts
    for k in np.nonzero(~inf[ti])[0].tolist():
        d = obs_of[ti[k]]
        if d is None:
            d = obs_of[ti[k]] = {}
        d.setdefault(ii[k], k)
    for p_k in range(n_pts):
        if obs_of[p_k]:
            depth_med[p_k] = float(np.median(depth[sorted(obs_of[p_k].values())]))

    # Candidates: pairs within one radius somewhere, found by the same per-image
    # forward x-window the collapse uses (width = the left point's own radius).
    cand = set()
    rows = np.concatenate(
        [np.fromiter(d.values(), np.int64, len(d)) for d in obs_of if d]
        or [np.zeros(0, np.int64)]
    )
    if len(rows) < 2:
        return [], np.zeros(0), depth_med
    srt = rows[np.lexsort((ti[rows], uv[rows, 1], uv[rows, 0], ii[rows]))]
    _, starts_i, counts_i = np.unique(ii[srt], return_index=True, return_counts=True)
    for st, cnt in zip(starts_i.tolist(), counts_i.tolist()):
        if cnt < 2:
            continue
        blk = srt[st : st + cnt]
        x, y, p = uv[blk, 0], uv[blk, 1], ti[blk]
        rp = rad[p]
        k = np.searchsorted(x, x + rp, side="right") - np.arange(cnt) - 1
        tot = int(k.sum())
        if tot == 0:
            continue
        a = np.repeat(np.arange(cnt), k)
        b = a + 1 + (np.arange(tot) - np.repeat(np.cumsum(k) - k, k))
        rr = np.minimum(rp[a], rp[b])
        m = (x[b] - x[a] <= rr) & (np.abs(y[b] - y[a]) <= rr)
        cand.update(
            zip(
                np.minimum(p[a[m]], p[b[m]]).tolist(),
                np.maximum(p[a[m]], p[b[m]]).tolist(),
            )
        )

    ux, uy = uv[:, 0].tolist(), uv[:, 1].tolist()
    pairs, mism = [], []
    for a_p, b_p in sorted(cand):
        oa, ob = obs_of[a_p], obs_of[b_p]
        shared = sorted(oa.keys() & ob.keys())
        if len(shared) < 2:
            continue
        rr = float(min(rad[a_p], rad[b_p]))
        near, far = rr * rr, 4.0 * rr * rr
        hit = apart = False
        for g in shared:
            ka, kb = oa[g], ob[g]
            dx, dy = ux[ka] - ux[kb], uy[ka] - uy[kb]
            d2 = dx * dx + dy * dy
            if d2 <= near:
                hit = True
            elif d2 >= far:
                apart = True
        if not (hit and apart):
            continue
        d_a, d_b = depth_med[a_p], depth_med[b_p]
        pairs.append((a_p, b_p))
        mism.append(abs(d_a - d_b) / max(min(d_a, d_b), 1e-9))
    return pairs, np.asarray(mism), depth_med


def _reconcile_aliased_tracks(recon, sizes, mutated, uid):
    """REPAIR the malignant conflicted pairs, then cull only what repair failed.

    A malignant pair (see ``_conflicted_pairs``) is two tracks that have stitched
    observations of DIFFERENT physical features together — a repeated pattern
    matched across instances — so each one's triangulation splits the difference
    between two real surfaces.  The observations themselves are good pixels; it
    is their ASSIGNMENT that is wrong, and an assignment can be repaired:

    1. Union-find over the malignant pairs only (benign pairs never take part —
       distinct neighbors have nothing to reassign).  Inside each component, pool
       the member tracks' observations and, per image, give each observation to
       the component point it reprojects closest to under the current poses and
       focal, greedily by error with index tie-breaks and at most one observation
       per point per image.  Re-triangulate the members from their new tracks and
       iterate to convergence (3 rounds max) — the retriangulation moves the
       points, which can change who wins the next round.
    2. A member left under 3 observations is dropped; its observations stay with
       whoever won them.
    2b. A repaired member's patch frame is RE-SIZED at its new depth, for the same
       reason the collapse's survivor's is (see ``_resized_patch_frames``).
    3. Only then a severity-gated cull: a pair still malignant AND disagreeing by
       at least ALIAS_CULL_MISMATCH loses BOTH members, because the conflict
       cannot say which of the two tracks is the corrupt one.

    Repair-first is not a preference, it is forced by the severity spread: on
    20240906_081206935 the malignant population is 743 pairs at a median 9%
    mismatch involving 16.8% of the cloud (mild fuzz on a roof; the solve is
    visually fine), while on 20250907_000240907 it is 71 pairs at median 81% /
    p90 8503% (foreground spikes).  A blanket cull would delete a sixth of an
    essentially correct reconstruction to fix the first case.

    Returns ``(recon, stats, sizes, mutated, uid)`` — the carried arrays with the
    culled rows removed.  Deterministic: components, members, images and
    observations are all visited in index order."""
    from sfmtool._sfmtool.analysis import triangulate_batch
    from sfmtool._sfmtool.patches import PatchCloud

    stats = {
        "components": 0,
        "reassigned": 0,
        "dropped_obs": 0,
        "retriangulated": 0,
        "culled": 0,
        "rounds": 0,
        "mal_before": 0,
        "mal_after": 0,
        "benign": 0,
    }
    sizes = np.asarray(sizes, dtype=np.float64)
    if not COLLAPSE_ENABLED or recon.point_count < 2:
        return recon, stats, sizes, mutated, uid
    pairs, mism, _ = _conflicted_pairs(recon, sizes)
    if not pairs:
        return recon, stats, sizes, mutated, uid
    malignant = [pr for pr, ms in zip(pairs, mism) if ms >= _ALIAS_MALIGNANT_MISMATCH]
    stats["mal_before"] = len(malignant)
    stats["benign"] = len(pairs) - len(malignant)
    if not malignant:
        return recon, stats, sizes, mutated, uid

    n_pts = recon.point_count
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, dtype=np.float64)
    cam = recon.cameras[0]
    q = np.ascontiguousarray(recon.quaternions_wxyz, dtype=np.float64)
    tv = np.ascontiguousarray(recon.translations, dtype=np.float64)
    pos4 = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    # Ray directions and camera centers, per observation row (the retriangulation
    # input; both are fixed by the poses, so they are computed once).
    d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(uv))
    dirs = np.ascontiguousarray(np.einsum("nji,nj->ni", rots[ii], d_loc))
    centers = np.ascontiguousarray(-np.einsum("nji,nj->ni", rots[ii], tv[ii]))

    # Components over the malignant pairs (union to the LOWER root).
    parent = list(range(n_pts))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_p, b_p in malignant:
        ra, rb = _find(a_p), _find(b_p)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comps = {}
    for a_p, b_p in malignant:
        for x in (a_p, b_p):
            comps.setdefault(_find(x), set()).add(x)
    components = [sorted(mem) for _root, mem in sorted(comps.items())]
    stats["components"] = len(components)

    # Rows of every component member, grouped by image.  Points at infinity are
    # never members (they cannot be malignant), so every row here is finite.
    row_order = np.argsort(ti, kind="stable")
    row_start = np.searchsorted(ti[row_order], np.arange(n_pts + 1))
    assign = ti.copy()
    # Only a component member can die here.  Points at infinity are never members
    # (they carry no depth, so they cannot be malignant) and must not be swept up.
    alive = np.ones(n_pts, bool)
    dead_rows = np.zeros(len(ti), bool)
    # Every member the repair actually re-triangulates: its patch frame was sized
    # at the depth it USED to have (see ``_resized_patch_frames``).
    moved = np.zeros(n_pts, bool)
    n_reassigned = n_dropped_obs = n_retri = 0

    for mem in components:
        by_img = {}
        for p_k in mem:
            for r in row_order[row_start[p_k] : row_start[p_k + 1]].tolist():
                by_img.setdefault(int(ii[r]), []).append(r)
        for rows in by_img.values():
            rows.sort()
        live = list(mem)
        for _round in range(_ALIAS_REPAIR_ROUNDS):
            stats["rounds"] = max(stats["rounds"], _round + 1)
            # Reprojection error of every (row, candidate point) combination in
            # this component, in one batched kernel call.
            rr_list, pp_list = [], []
            for g, rows in sorted(by_img.items()):
                for r in rows:
                    for p_k in live:
                        rr_list.append(r)
                        pp_list.append(p_k)
            if not rr_list:
                break
            rr_arr = np.asarray(rr_list, np.int64)
            pp_arr = np.asarray(pp_list, np.int64)
            res = _reprojection_residuals(
                cam,
                q,
                tv,
                np.ascontiguousarray(pos4[:, :3]),
                np.ascontiguousarray(uv[rr_arr]),
                np.ascontiguousarray(ii[rr_arr].astype(np.uint32)),
                np.ascontiguousarray(pp_arr.astype(np.uint32)),
                1e6,
            )
            cost = np.hypot(res[:, 0], res[:, 1])
            cost = np.where(np.isfinite(cost), cost, 1e9)
            cost_of = {}
            for k in range(len(rr_arr)):
                cost_of[(int(rr_arr[k]), int(pp_arr[k]))] = float(cost[k])

            # Greedy assignment per image: lowest error first, one observation
            # per point, ties by (point, row) index.
            changed = False
            new_assign = {}
            for g, rows in sorted(by_img.items()):
                order = sorted(
                    ((cost_of[(r, p_k)], p_k, r) for r in rows for p_k in live),
                )
                taken_pt, taken_row = set(), set()
                for _c, p_k, r in order:
                    if p_k in taken_pt or r in taken_row:
                        continue
                    taken_pt.add(p_k)
                    taken_row.add(r)
                    new_assign[r] = p_k
                for r in rows:
                    tgt = new_assign.get(r, -1)
                    if tgt < 0:
                        if not dead_rows[r]:
                            changed = True
                            n_dropped_obs += 1
                        dead_rows[r] = True
                    else:
                        if dead_rows[r]:
                            dead_rows[r] = False
                            changed = True
                        if assign[r] != tgt:
                            n_reassigned += 1
                            changed = True
                        assign[r] = tgt

            # Re-triangulate every live member from its new track; a member left
            # under 3 observations is no longer a point.
            tri_rows, counts, targets = [], [], []
            for p_k in live:
                rws = sorted(
                    r
                    for r, t_p in new_assign.items()
                    if t_p == p_k and not dead_rows[r]
                )
                if len(rws) < 3:
                    continue
                tri_rows.extend(rws)
                counts.append(len(rws))
                targets.append(p_k)
            if targets:
                offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
                sel = np.asarray(tri_rows, np.int64)
                tri = np.asarray(
                    triangulate_batch(
                        np.ascontiguousarray(dirs[sel]),
                        np.ascontiguousarray(centers[sel]),
                        offs,
                    )["points"]
                )
                good = np.isfinite(tri).all(axis=1)
                tgt = np.asarray(targets, np.int64)
                pos4[tgt[good], :3] = tri[good]
                moved[tgt[good]] = True
                n_retri += int(good.sum())
            drop_now = [p_k for p_k in live if p_k not in targets]
            for p_k in drop_now:
                alive[p_k] = False
            live = list(targets)
            if not changed or len(live) < 2:
                break
    stats["reassigned"] = n_reassigned
    stats["dropped_obs"] = n_dropped_obs
    stats["retriangulated"] = n_retri

    # Materialize the repair, then re-detect and cull only the unrepaired.
    fin = np.nonzero(~dead_rows & alive[assign])[0]
    fin = fin[np.lexsort((ii[fin], assign[fin]))]
    changes = {
        "positions": np.ascontiguousarray(pos4),
        "track_point_indexes": np.ascontiguousarray(assign[fin].astype(np.uint32)),
        "track_image_indexes": np.ascontiguousarray(ii[fin].astype(np.uint32)),
        # unused: embedded
        "track_feature_indexes": np.zeros(len(fin), np.uint32),
        "keypoints_xy": np.ascontiguousarray(uv[fin].astype(np.float32)),
    }
    # This pass re-triangulates too, so it inherits the collapse's defect: a
    # repaired member's patch frame still encodes the depth it was born at.  Same
    # remedy (``_resized_patch_frames``), with the reference view this pass's own
    # criterion names — the repaired track's lowest-reprojection observation AT
    # THE NEW POSITION, which is the same quantity the greedy reassignment above
    # ranks by, evaluated once the last retriangulation has landed.
    moved_live = moved & alive
    if recon.patches is not None and moved_live.any():
        sel = fin[moved_live[assign[fin]]]
        e_sel = np.full(len(sel), 1e9)
        if len(sel):
            res_f = _reprojection_residuals(
                cam,
                q,
                tv,
                np.ascontiguousarray(pos4[:, :3]),
                np.ascontiguousarray(uv[sel]),
                np.ascontiguousarray(ii[sel].astype(np.uint32)),
                np.ascontiguousarray(assign[sel].astype(np.uint32)),
                1e6,
            )
            e_sel = np.hypot(res_f[:, 0], res_f[:, 1])
            e_sel = np.where(np.isfinite(e_sel), e_sel, 1e9)
        best = {}
        for k in np.lexsort((sel, e_sel))[::-1].tolist():  # best row lands last
            best[int(assign[sel[k]])] = int(sel[k])
        tgts = np.array(sorted(best), dtype=np.int64)
        rrows = np.array([best[int(p_k)] for p_k in tgts.tolist()], dtype=np.int64)
        hu_out, hv_out = _resized_patch_frames(
            recon, pos4, rots, tv, cam, tgts, ii[rrows], sizes[tgts]
        )
        changes["patches"] = PatchCloud.from_halfvec_arrays(
            np.ascontiguousarray(hu_out, dtype=np.float32),
            np.ascontiguousarray(hv_out, dtype=np.float32),
            np.ascontiguousarray(pos4[:, :3]),
        )
        bmp = recon.patch_bitmaps
        if bmp is not None:
            # ``patches=`` alone would clear the consensus textures.
            changes["patch_bitmaps"] = np.ascontiguousarray(bmp)
    out = recon.clone_with_changes(**changes)
    # A point that lost every observation must go too, or it ships as a
    # zero-track point (filter_points_by_mask renumbers, so do it in one pass).
    keep = alive & (np.bincount(assign[fin], minlength=n_pts) > 0)
    out = out.filter_points_by_mask(np.ascontiguousarray(keep))
    sizes = sizes[keep]
    uid = uid[keep]
    stats["culled"] = int((~keep).sum())
    # Every component member's track and position may have moved: its bitmap is
    # now stale and its stored error describes the old geometry.
    mutated = mutated.copy()
    for mem in components:
        mutated[mem] = True
    mutated = mutated[keep]

    pairs2, mism2, _ = _conflicted_pairs(out, sizes)
    mal2 = [pr for pr, ms in zip(pairs2, mism2) if ms >= _ALIAS_MALIGNANT_MISMATCH]
    stats["mal_after"] = len(mal2)
    cull = np.zeros(out.point_count, bool)
    for (a_p, b_p), ms in zip(pairs2, mism2):
        if ms >= ALIAS_CULL_MISMATCH:
            cull[a_p] = cull[b_p] = True
    if cull.any():
        out = out.filter_points_by_mask(np.ascontiguousarray(~cull))
        sizes = sizes[~cull]
        mutated = mutated[~cull]
        uid = uid[~cull]
        stats["culled"] += int(cull.sum())
    return out, stats, sizes, mutated, uid


def _print_reconcile(tag, st):
    """One line of reconciliation evidence."""
    print(
        f"  alias reconcile ({tag}): {st['mal_before']} malignant + "
        f"{st['benign']} benign pairs; {st['components']} components, "
        f"{st['reassigned']} observations reassigned ({st['dropped_obs']} "
        f"orphaned) over {st['rounds']} round(s), {st['retriangulated']} points "
        f"retriangulated, {st['culled']} points culled; malignant "
        f"{st['mal_before']} -> {st['mal_after']}"
    )


def _contained_pairs(recon, sizes):
    """Every co-observed pair whose members' DETECTION footprints nest, one
    inside the other, with the pair's range disagreement.

    ``(a, b, big, n_shared, sep, r_lo, r_hi, mismatch, container_nearer,
    d_big, d_small)`` per pair, index-sorted; ``big`` is the member carrying
    the LARGER radius and ``d_big``/``d_small`` the members' median camera
    ranges over the shared views.  Containment is measured
    at the loosest bound the callers use (``sep + r_lo <= r_hi``), so a caller
    tightening ``CONTAINED_SCALE`` only filters this list.

    The radius is the DETECTION-time patch half-extent (``_cluster_detection_sizes``,
    already a radius — see the COLLAPSE_RADIUS_SCALE convention note), floored at
    ``_COLLAPSE_R_MIN_PX``, and NOT scaled: the claim under test is that one
    feature's whole image footprint swallows another's, which is a statement
    about the full extent.  It is depth-independent for the same reason the
    collapse insists on it — ``max_embedded_feature_size_per_point`` would make a
    point's radius grow as its depth shrinks, so exactly the wrong-depth
    foreground points this pass judges would claim the largest footprints and
    swallow the cloud (on 20250907_000240907 the projected radii run to 30306 px).

    Candidates come from one ball query per observation at that observation's
    OWN radius: a containment pair is always within the CONTAINER's radius, so
    querying every point at its own radius finds every pair from the container's
    side, and no query is wider than a real feature.  Points at infinity carry no
    range and take no part."""
    n_pts = recon.point_count
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, dtype=np.float64)
    inf = np.asarray(recon.point_is_at_infinity)
    if n_pts < 2 or len(ti) < 2:
        return []
    sizes = np.asarray(sizes, dtype=np.float64)
    rad = np.where(
        np.isfinite(sizes) & (sizes > 0),
        np.maximum(_COLLAPSE_R_MIN_PX, sizes),
        _COLLAPSE_R_MIN_PX,
    )

    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    tv = np.asarray(recon.translations, dtype=np.float64)
    pc = np.einsum("nij,nj->ni", rots[ii], np.asarray(recon.positions)[ti]) + tv[ii]
    depth = _cam_depth(pc)

    # One observation per (point, image) — a containment is a statement about
    # tracks, the same representative view the collapse reasons about.
    obs_of = [None] * n_pts
    for k in np.nonzero(~inf[ti])[0].tolist():
        d = obs_of[ti[k]]
        if d is None:
            d = obs_of[ti[k]] = {}
        d.setdefault(ii[k], k)

    cand = set()
    for g in range(len(recon.image_names)):
        rows = np.array(
            sorted(d[g] for d in obs_of if d is not None and g in d), dtype=np.int64
        )
        if len(rows) < 2:
            continue
        pts_g = ti[rows]
        tree = cKDTree(uv[rows])
        for j, hits in enumerate(tree.query_ball_point(uv[rows], rad[pts_g])):
            a_p = int(pts_g[j])
            for h in hits:
                b_p = int(pts_g[h])
                if a_p != b_p:
                    cand.add((min(a_p, b_p), max(a_p, b_p)))

    out = []
    for a_p, b_p in sorted(cand):
        oa, ob = obs_of[a_p], obs_of[b_p]
        shared = sorted(oa.keys() & ob.keys())
        if len(shared) < _CONTAINED_MIN_SHARED:
            continue
        ka = [oa[g] for g in shared]
        kb = [ob[g] for g in shared]
        sep = float(np.median(np.hypot(*(uv[ka] - uv[kb]).T)))
        r_a, r_b = float(rad[a_p]), float(rad[b_p])
        r_lo, r_hi = min(r_a, r_b), max(r_a, r_b)
        if sep + r_lo > r_hi:
            continue
        d_a = float(np.median(depth[ka]))
        d_b = float(np.median(depth[kb]))
        mism = abs(d_a - d_b) / max(min(d_a, d_b), 1e-9)
        big = a_p if r_a >= r_b else b_p
        d_big = d_a if big == a_p else d_b
        d_small = d_b if big == a_p else d_a
        out.append(
            (
                a_p,
                b_p,
                big,
                len(shared),
                sep,
                r_lo,
                r_hi,
                mism,
                d_big < d_small,
                d_big,
                d_small,
            )
        )
    return out


def _contained_cull_set(pairs):
    """The points to drop, given the flagged pairs: every CONTAINER that is
    contradicted by at least ``_CONTAINED_MIN_WITNESSES`` contained points.

    The cull is one-sided and CORROBORATED, and both halves of that are
    measurements rather than preferences:

    ONE-SIDED — only the container is ever dropped.  Resolving a flagged pair by
    SUPPORT instead ("cull the member with fewer views, ties by reprojection")
    was tried and it culls the CONTAINED point in about half the fleet's pairs,
    which is the wrong direction on the evidence: containment is the coarse
    feature's claim about a whole region, and the fine points inside it are each
    carried by their own independent tracks.  Support does not even identify the
    culprit — the containers this pass drops on 20250907_000240907 carry 6-11
    views, more than several of the points they contradict.  Dropping the fine
    member also feeds transitivity, since one coarse feature typically contains
    several fine ones and they then accuse each other through it.

    CORROBORATED — a single contradicted neighbour can be a bad triangulation on
    EITHER side, so it names no culprit.  Two independent contained points
    disagreeing the same way do: they were triangulated from different tracks and
    the only thing they share is the container.  Corroboration is what keeps the
    pass rare where the cloud is merely noisy — accepting a single witness takes
    20250907_000240907 from 51 to 113 culls (1.6% -> 3.6% of the cloud) and
    DnDTabletop from 3 to 17 — while costing nothing on the case the pass was
    built for, whose container is contradicted ten times over."""
    victims = {}
    for _a_p, _b_p, big, *_rest in pairs:
        victims[big] = victims.get(big, 0) + 1
    return sorted(v for v, n in victims.items() if n >= _CONTAINED_MIN_WITNESSES)


def _cull_contained_inconsistent(recon, sizes, mutated, uid):
    """Drop each point whose detection footprint CONTAINS several points that
    disagree with it about range (see CONTAINED_CULL_ENABLED).

    Human inspection of a DnDTabletop seed found point 3562 — 3 views, a 115 px
    detection radius, depth 2192 — sitting on top of ten well-supported gameboard
    points at depth 2650-2700 whose keypoints land 24-88 px away, i.e. entirely
    inside its own footprint.  Its rays pass straight through their patches.
    Nothing in the chain saw it: the collapse and the reconciliation both measure
    overlap in the SMALLER member's radius (24 px against a 30 px feature is not
    overlap), and the point's own reprojection error, track length and patch size
    are all unremarkable.

    Returns ``(recon, stats, sizes, mutated, uid)``; the carried arrays come back
    with the culled rows removed.  Deterministic: the
    pair list and the cull set are index-ordered throughout."""
    stats = {"examined": 0, "flagged": 0, "culled": 0, "worst": 0.0}
    if (not CONTAINED_CULL_ENABLED and not CONTAINED_DEBUG) or recon.point_count < 2:
        return recon, stats, sizes, mutated, uid
    sizes = np.asarray(sizes, dtype=np.float64)
    pairs = _contained_pairs(recon, sizes)
    stats["examined"] = len(pairs)
    if CONTAINED_DEBUG:
        # A/B scaffolding: the whole nested-pair table plus the per-point support
        # a gate might key on, so a candidate rule can be swept offline against a
        # run's own geometry instead of re-running the seed six times.
        err_med = _reprojection_medians(recon)
        # Per-point triangulation-noise ingredients, so an offline sweep can
        # judge a pair's range mismatch against the depth uncertainty the
        # geometry actually supports (sigma_rel ~ err_px * depth / (f * B))
        # instead of a fixed fraction.  Positions identify points across the
        # stage's indexing and the final seed's.
        n_pts = recon.point_count
        tp_d = np.asarray(recon.track_point_indexes).astype(np.int64)
        ii_d = np.asarray(recon.track_image_indexes).astype(np.int64)
        q_d = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
        rots_d = Rotation.from_quat(q_d[:, [1, 2, 3, 0]]).as_matrix()
        tv_d = np.asarray(recon.translations, dtype=np.float64)
        centers = -np.einsum("nji,nj->ni", rots_d, tv_d)
        pos_d = np.asarray(recon.positions, dtype=np.float64)
        pc_d = np.einsum("nij,nj->ni", rots_d[ii_d], pos_d[tp_d]) + tv_d[ii_d]
        depth_d = _cam_depth(pc_d)
        depth_median = np.full(n_pts, np.nan)
        baseline_span = np.zeros(n_pts)
        for p in range(n_pts):
            m = tp_d == p
            if not m.any():
                continue
            depth_median[p] = float(np.median(depth_d[m]))
            c = centers[np.unique(ii_d[m])]
            if len(c) > 1:
                baseline_span[p] = float(
                    np.max(np.linalg.norm(c[:, None] - c[None, :], axis=2))
                )
        dump = WS / "sfmr" / "contained-pairs.npz"
        dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump,
            pairs=np.asarray(pairs, dtype=np.float64).reshape(len(pairs), -1),
            track_lengths=np.bincount(
                np.asarray(recon.track_point_indexes), minlength=recon.point_count
            ),
            error_median=np.where(np.isfinite(err_med), err_med, 0.0),
            radius=np.where(
                np.isfinite(sizes) & (sizes > 0),
                np.maximum(_COLLAPSE_R_MIN_PX, sizes),
                _COLLAPSE_R_MIN_PX,
            ),
            positions=pos_d,
            normals=np.asarray(recon.normals, dtype=np.float64),
            depth_median=depth_median,
            baseline_span=baseline_span,
            focal=np.float64(recon.cameras[0].parameters["focal_length"]),
        )
        print(f"  [contained-pairs dumped: {len(pairs)} -> {dump}]")
    flagged = [
        p
        for p in pairs
        if CONTAINED_CULL_ENABLED
        and p[7] > CONTAINED_MISMATCH
        and p[4] + p[5] <= CONTAINED_SCALE * p[6]
        and (p[8] or not CONTAINED_OCCLUSION)
    ]
    stats["flagged"] = len(flagged)
    if not flagged:
        return recon, stats, sizes, mutated, uid
    stats["worst"] = max(p[7] for p in flagged)
    cull_idx = _contained_cull_set(flagged)
    stats["culled"] = len(cull_idx)
    keep = np.ones(recon.point_count, bool)
    keep[np.asarray(cull_idx, dtype=np.int64)] = False
    keep = np.ascontiguousarray(keep)
    recon = recon.filter_points_by_mask(keep)
    return (
        recon,
        stats,
        np.asarray(sizes)[keep],
        np.asarray(mutated)[keep],
        np.asarray(uid)[keep],
    )


def _print_contained(tag, st):
    """One line of contained-inconsistent evidence."""
    print(
        f"  contained-inconsistent cull ({tag}): {st['examined']} nested pairs "
        f"examined, {st['flagged']} range-inconsistent, {st['culled']} "
        f"corroborated containers culled (worst range mismatch "
        f"{100.0 * st['worst']:.1f}%)"
    )


def _member_view_structure(recon, images, pts, imgs):
    """Per-member view structure, as a fraction of the track's consensus contrast.

    For member ``(pts[i], imgs[i])``: the standard deviation of the SOURCE image
    luma inside the patch's projected footprint in that image, over the standard
    deviation of the point's consensus bitmap luma.  See _EVICT_FLAT_FRAC for why
    the reading is a ratio rather than an absolute.

    The window is the axis-aligned box that contains the projected surfel — its
    half-width from the four corners ``X ± hu ± hv`` carried through the same
    projection the patch render uses — so it covers the content the ZNCC was
    scored over (an oblique view's box is larger than its parallelogram, which
    only ever ADDS surrounding contrast, i.e. errs toward keeping the member).  A
    box statistic is exact and O(1) per member off one summed-area pass per image,
    so the whole gate costs a couple of integral images per source frame and
    nothing per observation.

    The box is CENTRED on the member's stored keypoint, not
    on the reprojection: the gate asks whether this member's own source content
    carries structure, and the content it must ask about is the content at the
    keypoint — the same place the ZNCC beside it now samples.  Its EXTENT still
    comes from the projected surfel, which is the only thing that knows the
    patch's footprint in this image.  A member with no stored keypoint falls back
    to the projection.

    Returns a float array parallel to ``pts``: NaN wherever the reading cannot be
    made (no consensus bitmap, a point at infinity, a projection behind the
    camera, a window off the image edge) — and NaN never evicts."""
    import cv2

    n = len(pts)
    out = np.full(n, np.nan)
    bmp = recon.patch_bitmaps
    if n == 0 or bmp is None:
        return out
    # Consensus contrast per point, over the covered (alpha > 0) pixels only:
    # an uncovered pixel is absence, not black.
    bmp = np.asarray(bmp)
    gl = 0.299 * bmp[..., 0] + 0.587 * bmp[..., 1] + 0.114 * bmp[..., 2]
    al = bmp[..., 3] > 0
    cnt = al.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cmu = np.where(al, gl, 0.0).sum(axis=(1, 2)) / cnt
        cvar = np.where(al, gl * gl, 0.0).sum(axis=(1, 2)) / cnt - cmu * cmu
        cstd = np.sqrt(np.maximum(cvar, 0.0))
    cstd = np.where(cnt >= _EVICT_FLAT_MIN_PIX, cstd, np.nan)

    hu, hv = _patch_halfvecs(recon)
    xyzw = np.asarray(recon.positions_xyzw, np.float64)
    rots = Rotation.from_quat(
        np.asarray(recon.quaternions_wxyz, np.float64)[:, [1, 2, 3, 0]]
    ).as_matrix()
    tvec = np.asarray(recon.translations, np.float64)
    cam = recon.cameras[0]
    # (point, image) -> stored keypoint, for the box centre.
    kp_at = None
    if recon.keypoints_xy is not None:
        ti = np.asarray(recon.track_point_indexes).astype(np.int64)
        ii = np.asarray(recon.track_image_indexes).astype(np.int64)
        uv = np.asarray(recon.keypoints_xy, np.float64)
        n_img = len(recon.image_names)
        keys = ti * n_img + ii
        order = np.argsort(keys, kind="stable")
        kp_at = (keys[order], uv[order])

    for g in np.unique(imgs):
        m = np.nonzero(imgs == g)[0]
        p = pts[m]
        rot, tr = rots[g], tvec[g]

        def proj(xw, rot=rot, tr=tr):
            # In-field, not `z < 0`: the pinhole half-space test NaNs every
            # observation past 90 degrees, and a NaN reading never evicts — so
            # under a fisheye the whole periphery was silently exempt from the
            # gate rather than passing it.
            pc = xw @ rot.T + tr
            px = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(pc)))
            return np.where(_in_field(cam, pc)[:, None], px, np.nan)

        x = xyzw[p, :3]
        c = proj(x)
        c = np.where((xyzw[p, 3] != 0)[:, None], c, np.nan)
        rad = np.zeros(len(p))
        for dv in (hu[p], hv[p], hu[p] + hv[p], hu[p] - hv[p]):
            e = proj(x + dv)
            rad = np.maximum(rad, np.hypot(e[:, 0] - c[:, 0], e[:, 1] - c[:, 1]))
        rad = np.clip(rad, 2.0, None)
        if kp_at is not None:
            # Re-centre on the stored keypoint where the member has one; the
            # extent (rad) stays the projected footprint's.  Only members the
            # projection could already read are moved, so the gate's readable
            # population — and therefore who it can evict — is unchanged.
            skeys, suv = kp_at
            want = p * len(recon.image_names) + int(g)
            at = np.minimum(np.searchsorted(skeys, want), len(skeys) - 1)
            hit = (skeys[at] == want) & np.isfinite(c[:, 0]) & np.isfinite(c[:, 1])
            c = np.where(hit[:, None], suv[at], c)

        src = np.asarray(images[g])
        gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        s1, s2 = cv2.integral2(gray)
        h, w = gray.shape
        x0 = np.clip(np.floor(c[:, 0] - rad), 0, w - 1)
        x1 = np.clip(np.ceil(c[:, 0] + rad) + 1, 1, w)
        y0 = np.clip(np.floor(c[:, 1] - rad), 0, h - 1)
        y1 = np.clip(np.ceil(c[:, 1] + rad) + 1, 1, h)
        ok = np.isfinite(x0) & np.isfinite(y0) & (x1 > x0 + 2) & (y1 > y0 + 2)
        xa, xb = (
            np.where(ok, x0, 0).astype(np.int64),
            np.where(ok, x1, 1).astype(np.int64),
        )
        ya, yb = (
            np.where(ok, y0, 0).astype(np.int64),
            np.where(ok, y1, 1).astype(np.int64),
        )
        npx = (xb - xa) * (yb - ya)
        su = s1[yb, xb] - s1[ya, xb] - s1[yb, xa] + s1[ya, xa]
        sq = s2[yb, xb] - s2[ya, xb] - s2[yb, xa] + s2[ya, xa]
        mu = su / npx
        sd = np.sqrt(np.maximum(sq / npx - mu * mu, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = np.where(ok, sd / cstd[p], np.nan)
    return out


def _dump_evict_members(recon, pts, imgs, sc, med, nsc, bar, ratio, frac):
    """Write the eviction pass's whole per-member table when
    ``SFMTOOL_TRACK_EVICT_DUMP`` names a path.  Instrumentation only — every
    gate threshold above was converged off this table, and the raw columns
    (score, track median, member count, structure fraction) are what a
    re-calibration needs, taken at the site the gates actually run at rather
    than replayed on a shipped artifact whose geometry the BA has since moved.
    The keypoint columns give each member a cross-artifact identity."""
    path = os.environ.get("SFMTOOL_TRACK_EVICT_DUMP")
    if not path:
        return
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, np.float64)
    at = {}
    for r in range(len(ti)):
        at.setdefault((ti[r], ii[r]), (uv[r, 0], uv[r, 1]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("pt\timg\tzncc\tmed\tnsc\tbar_bad\tratio_bad\tfrac\tu\tv\n")
        for i in range(len(pts)):
            u, v = at.get((pts[i], imgs[i]), (float("nan"),) * 2)
            fh.write(
                f"{pts[i]}\t{imgs[i]}\t{sc[i]:.5f}\t{med[i]:.5f}\t{nsc[i]}\t"
                f"{int(bar[i])}\t{int(ratio[i])}\t{frac[i]:.5f}\t{u:.3f}\t{v:.3f}\n"
            )
    print(f"  [track-view eviction dump: {len(pts)} members -> {path}]")


def _drop_observations(recon, drop_row, sizes, mutated, uid):
    """Remove the observation rows ``drop_row`` marks, retiring what that leaves
    under three views.  Returns ``(recon, sizes, mutated, uid, n_culled)``.

    The track-array surgery every photometric MEMBERSHIP pass ends in, factored
    out so they all do it identically — the eviction replay and the
    member-coherence vet remove members on different evidence but must renumber,
    re-mark and re-cull the same way.  Four steps: rebuild the four track arrays
    without the dropped rows (re-sorted so a track's rows stay contiguous by
    point); mark every track a row was taken from MUTATED, since its baked
    consensus bitmap no longer describes the track that produced it; cull the
    points the drop leaves under three views, because two views cannot tell a
    real point from a repeated pattern and nothing downstream of the late vet
    re-checks track length; and refresh the stored errors, because a surviving
    track is a different measurement.  The carried per-point arrays follow the
    renumbering.

    A point ALL of whose rows are dropped falls out through the same length cull,
    so a caller that wants a point gone entirely can say so by dropping its whole
    track rather than by a second filtering pass."""
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy)
    hit = np.unique(ti[drop_row])
    rows = np.nonzero(~drop_row)[0]
    rows = rows[np.lexsort((ii[rows], ti[rows]))]  # tracks stay grouped by point
    # Every per-OBSERVATION column has to take the same `rows` selection the
    # track arrays take, or it silently describes the wrong observations from
    # here on.  `observation_confidence` is optional, so it is passed only when
    # the recon carries one (omitting the kwarg preserves; passing None clears).
    kw = {}
    conf = recon.observation_confidence
    if conf is not None:
        kw["observation_confidence"] = np.ascontiguousarray(
            np.asarray(conf)[rows].astype(np.uint8)
        )
    out = recon.clone_with_changes(
        track_point_indexes=np.ascontiguousarray(ti[rows].astype(np.uint32)),
        track_image_indexes=np.ascontiguousarray(ii[rows].astype(np.uint32)),
        track_feature_indexes=np.zeros(len(rows), np.uint32),  # unused: embedded
        keypoints_xy=np.ascontiguousarray(uv[rows].astype(np.float32)),
        **kw,
    )
    mutated = mutated.copy()
    mutated[hit] = True
    keep = np.ascontiguousarray(np.bincount(ti[rows], minlength=recon.point_count) >= 3)
    n_culled = int((~keep).sum())
    if n_culled:
        out = out.filter_points_by_mask(keep)
        sizes, mutated, uid = (
            np.asarray(sizes)[keep],
            np.asarray(mutated)[keep],
            np.asarray(uid)[keep],
        )
    out = _refresh_errors(out)
    return out, sizes, mutated, uid, n_culled


def _evict_track_views(recon, images, sizes, mutated, uid):
    """Score each track's OWN members photometrically and evict the disowned.

    View selection admits track views unconditionally (see TRACK_EVICT_ENABLED),
    so this replays it over the whole embedded cloud and applies the CANDIDATE
    admission rule to the members themselves: a member whose windowed ZNCC to the
    track's reference appearance falls below ``min_relative_zncc x`` the track's
    typical agreement is evicted.  "Typical agreement" is read through TWO
    estimators and the bar is taken against the LARGER:

    * ``self_agreement`` — selection's own statistic, the MEAN member ZNCC.  This
      is the exact bar a candidate faces, so the eviction is that rule's mirror.
    * the member-score MEDIAN, for tracks with ``_EVICT_MIN_MEMBERS`` members.
      The mean is computed over the members under test, so a junk tail lowers the
      very bar meant to catch it; the median is the robust reading of the same
      quantity and only ever RAISES the bar (hence the ``max``).  It is what
      reaches the members that are individually plausible but flatly inconsistent
      with their siblings — an occlusion alias whose view sees a different
      surface entirely, scoring 0.64 where the track's other eleven views run
      0.79-0.98.

    That bar is a FRACTION of a ZNCC, so it saturates: at
    ``_EVICT_REL_ZNCC`` = 0.7 no member above 0.70 is reachable however crisp its
    track, and coincidental correlations cluster just above that ceiling.  Two
    further per-member gates, each judging a quantity the bar structurally
    cannot, evict on top of it (a member fails if ANY gate fails it):

    * DISAGREEMENT RATIO (``_EVICT_DIS_RATIO`` / ``_EVICT_DIS_FLOOR``) — the same
      scores read as ``1 - zncc``, where the good members spread out instead of
      crowding against 1.0, so a member 17x-35x its siblings' median disagreement
      is visible as the outlier it is.
    * PER-VIEW FLATNESS (``_EVICT_FLAT_FRAC``) — the member's OWN source content,
      which no ZNCC gate reads: a smooth gradient correlates softly with anything
      and carries no localizable structure to correlate with.

    A track selection could not build a trustworthy reference for is skipped
    outright (``self_agreement`` NaN or below ``min_self_agreement``): the
    selection did not vet candidates there either, and no evidence means no
    eviction.  Members whose score is NaN (unscorable view) are likewise left
    alone.

    PLACEMENT is load-bearing, and measured.  Run straight after ``embed_patches``
    the test is CIRCULAR: the localizer has just congealed every member's window
    onto the reference it is then scored against, so a member sitting on a
    different surface still agrees with it.  On the 20250907_000240907 seed the
    five flat-black members score 0.27 / 0.73 / 0.23 / 0.89 / 0.79 there — four of
    the five invisible.  The frames only become independent of that congealing
    once the ARS normals REPLACE them (a 74 deg median frame turn on that seed)
    and the keypoints are re-localized through the new frames; scored at that
    geometry the same five read 0.14 / 0.36 / 0.01 / 0.35 / 0.08 and every one of
    them is disowned.  So the pass runs where ``_ars_normals`` and
    ``_relocalize_keypoints`` have already run — which is also the last site at
    which an eviction still reaches the BA, so the adjustment never sees the
    observations the photometry rejected.

    Eviction removes OBSERVATIONS, not points.  Points it leaves under three
    views are retired here (nothing downstream of this site re-checks track
    length), so the carried arrays come back with those rows dropped.  Returns
    ``(recon, stats, sizes, mutated, uid)``, with every evicted-from survivor
    marked mutated — its baked consensus bitmap no longer describes the track
    that produced it."""
    st = {
        "scored": 0,
        "untrusted": 0,
        "evicted": 0,
        "bar": 0,
        "ratio": 0,
        "flat": 0,
        "tracks": 0,
        "culled": 0,
        "zncc_med": float("nan"),
        "secs": 0.0,
    }
    cloud = recon.patches
    if not TRACK_EVICT_ENABLED or recon.point_count == 0 or cloud is None:
        return recon, st, sizes, mutated, uid
    from sfmtool._sfmtool.patches import ImagePyramidSet

    t0 = time.time()
    # One selection pass over the whole cloud (it parallelizes internally); the
    # shared pyramid set keeps the decode off the per-call path, as the embed does.
    # Keypoint-anchored on both halves of what this pass reads: the reference is
    # fused from the members at their own keypoints, and each member's score is
    # taken through the same anchored render, so a member's reprojection residual
    # neither smears the reference nor deflates the score the bar reads (see
    # KEYPOINT-ANCHORED VETTING PHOTOMETRY above).  Candidate scoring is
    # untouched — this pass never looks at candidates, and a candidate has no
    # keypoint to anchor at anyway.
    sel = cloud.select_views(
        recon,
        ImagePyramidSet(recon, images),
        keypoint_anchor=True,
    )

    n_img = len(recon.image_names)
    # Gather every scorable member of every trusted track, with the two ZNCC-only
    # verdicts already taken; the flatness gate then reads all of them at once
    # (its cost is per SOURCE IMAGE, not per member — see _member_view_structure).
    m_pt, m_img, m_sc, m_bar, m_ratio = [], [], [], [], []
    m_med, m_n = [], []  # carried for _dump_evict_members; no gate reads them
    for s in sel:
        k = int(s["track_view_count"])
        if k == 0:
            continue
        agree = float(s["self_agreement"])
        if not np.isfinite(agree) or agree < _EVICT_MIN_SELF_AGREEMENT:
            st["untrusted"] += 1
            continue
        imgs = np.asarray(s["admitted"], dtype=np.int64)[:k]
        sc = np.asarray(s["scores"], dtype=np.float64)[:k]
        ok = np.isfinite(sc)
        if not ok.any():
            continue
        st["scored"] += 1
        sc, imgs = sc[ok], imgs[ok]
        ref = agree
        quorum = len(sc) >= _EVICT_MIN_MEMBERS
        med = float(np.median(sc))
        if quorum:
            ref = max(ref, med)
        m_pt.append(np.full(len(sc), int(s["point_index"]), np.int64))
        m_med.append(np.full(len(sc), med))
        m_n.append(np.full(len(sc), len(sc), np.int64))
        m_img.append(imgs)
        m_sc.append(sc)
        m_bar.append(sc < _EVICT_REL_ZNCC * ref)
        # The disagreement-ratio gate needs a spread, so it needs the same member
        # quorum the robust bar does.
        m_ratio.append(
            (1.0 - sc) > max(_EVICT_DIS_RATIO * (1.0 - med), _EVICT_DIS_FLOOR)
            if quorum
            else np.zeros(len(sc), bool)
        )
    if not m_pt:
        st["secs"] = time.time() - t0
        return recon, st, sizes, mutated, uid
    m_pt = np.concatenate(m_pt)
    m_img = np.concatenate(m_img)
    m_sc = np.concatenate(m_sc)
    m_bar = np.concatenate(m_bar)
    m_ratio = np.concatenate(m_ratio)
    m_med = np.concatenate(m_med)
    m_n = np.concatenate(m_n)
    frac = _member_view_structure(recon, images, m_pt, m_img)
    _dump_evict_members(recon, m_pt, m_img, m_sc, m_med, m_n, m_bar, m_ratio, frac)
    m_flat = np.nan_to_num(frac, nan=np.inf) < _EVICT_FLAT_FRAC
    bad = m_bar | m_ratio | m_flat
    st["secs"] = time.time() - t0
    if not bad.any():
        return recon, st, sizes, mutated, uid

    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    # (point, image) is selection's granularity — its track view list is deduped,
    # so a point observed twice in one image loses both rows together, which is
    # the honest reading of "the photometry disowns this view of this point".
    obs = ti * n_img + ii
    key = m_pt * n_img + m_img
    drop_row = np.isin(obs, key[bad])
    st["evicted"] = int(drop_row.sum())
    # Attribution, in observations and in gate priority: what the committed bar
    # already caught, then what each new gate adds on top of it.
    st["bar"] = int(np.isin(obs, key[m_bar]).sum())
    st["ratio"] = int(np.isin(obs, key[m_ratio & ~m_bar]).sum())
    st["flat"] = int(np.isin(obs, key[m_flat & ~m_bar & ~m_ratio]).sum())
    st["tracks"] = len(np.unique(ti[drop_row]))
    st["zncc_med"] = float(np.median(m_sc[bad]))

    # The renumbering, the mutation marking and the length-3 retirement are the
    # shared surgery (see _drop_observations).
    out, sizes, mutated, uid, n_culled = _drop_observations(
        recon, drop_row, sizes, mutated, uid
    )
    st["culled"] = n_culled
    return out, st, sizes, mutated, uid


def _print_evict(tag, st, n_obs):
    """One line of track-view eviction evidence."""
    pct = 100.0 * st["evicted"] / n_obs if n_obs else 0.0
    print(
        f"  track-view eviction ({tag}): {st['scored']} tracks scored "
        f"({st['untrusted']} skipped, no trustworthy reference), "
        f"{st['evicted']}/{n_obs} observations ({pct:.2f}%) evicted from "
        f"{st['tracks']} tracks (bar {st['bar']} + ratio {st['ratio']} + "
        f"flat {st['flat']}), {st['culled']} points culled under 3 views, "
        f"eviction ZNCC median {st['zncc_med']:.3f}; {st['secs']:.1f}s"
    )


# ── Late vetting (SFMTOOL_LATE_VET) ──────────────────────────────────────────


def _localizability_verdict(sigma, lam1, lam2, tau, max_aniso):
    """The localizability criterion, as two independent per-point verdicts:
    ``(bad_sigma, bad_aniso)``.

    ``bad_sigma`` is the committed test — the noise-normalized weak-axis
    uncertainty ``sigma_pos`` (patch-grid px) over ``tau``.  ``bad_aniso`` is the
    SHAPE bound the first one structurally cannot be (see LOC_MAX_ANISO): the
    structure tensor's eigenvalue ratio over ``max_aniso``.

    NaN keeps, on both halves and for the same reason the embed cull keeps it
    (`_cull_by_localizability`): an unscorable consensus is an absence of
    evidence, not evidence of un-localizability.  A zero threshold disables its
    half.  Shared by the promotion gate and the late re-cull so the two apply one
    criterion rather than two that drift apart."""
    sigma = np.asarray(sigma, dtype=float)
    n = len(sigma)
    bad_sigma = sigma > tau if tau > 0 else np.zeros(n, bool)
    if max_aniso > 0 and np.isfinite(max_aniso):
        ratio = np.asarray(lam1, dtype=float) / np.maximum(
            np.asarray(lam2, dtype=float), _LOC_LAM2_FLOOR
        )
        bad_aniso = ratio > max_aniso
    else:
        bad_aniso = np.zeros(n, bool)
    return np.asarray(bad_sigma, bool), np.asarray(bad_aniso, bool)


def _member_coherence_vet(recon, images, sizes, mutated, uid):
    """Read each track's PAIRWISE member agreement and act on the verdict.
    Returns ``(recon, stats, sizes, mutated, uid)``.

    The core primitive does the work
    (``PatchCloud.validate_member_coherence``,
    ``specs/core/patch/member-coherence-validation.md``): for each point it renders
    every member's patch from that member's OWN source image through the point's
    own frame — the identical photometry ``select_views`` uses, so the numbers
    live in the same metric the eviction bar is taken in — correlates every pair
    over one frozen common support, and reads a verdict off the k x k matrix via
    the max-support block and the separation margin.  Three outcomes, applied
    here:

    * ``keep_all`` — nothing to do.  This is also where the MARGIN GATE lands
      the tracks it refuses to cut: a single surfel swept across a wide baseline
      gives a BANDED matrix whose every threshold-block is a strict subset, and
      cutting one picks an arbitrary place along a continuum.  Those keeps are
      counted separately (``margin_gated``) because the gate is load-bearing —
      it is the whole reason the pass can be run on a drift-heavy video capture.
      A gate REFUSAL is a finite margin at or below the gate; a NaN margin means
      no cut was ever on the table (nothing scored, or every block a singleton),
      which is an absence of evidence and is counted apart as ``unscored_keep``.
      Both show ``support < k``, so the margin is the only thing that separates
      them and crediting the gate for both overstates what it is doing.
    * ``split`` — the rejected members' OBSERVATIONS are removed, exactly as the
      eviction replay removes the ones it disowns, through the same
      ``_drop_observations`` surgery; the point survives on the block, and if the
      cut leaves it under three views that surgery's own length rule retires it.
    * ``retire`` — the track's members split into two comparably-supported and
      mutually incompatible groups and NOTHING in the matrix says which one is
      the point, so it ships nothing.  Expressed as dropping the point's whole
      track, which the same length rule then culls: one code path removes
      members and points, so there is no second renumbering to keep in step.

    ELIGIBILITY is any point with at least three observations, at infinity or
    not.  The length floor costs nothing (a k < 3 track is kept whole by the
    rule anyway) and keeps the render off two-member tracks.  A w = 0 point's
    frame projects translation-invariantly — every member samples the same
    angular window around the bearing at the same scale — so the matrix reads
    the identical photometric quantity it reads for finite points.  Member
    disagreement on an infinity track has one extra cause (content that is not
    actually at infinity drifts with the baseline), but either reading — an
    occluding member, or a misclassified depth — describes a track that should
    not ship as it stands, so the verdicts apply unchanged.

    Every rejected member is a member the fused consensus AGREED with, so every
    touched survivor comes back marked mutated and its bitmap must be re-fused
    before anything downstream scores it."""
    st = {
        "checked": 0,
        "keep_all": 0,
        "split": 0,
        "retire": 0,
        "margin_gated": 0,
        "unscored_keep": 0,
        "evicted": 0,
        "retired": 0,
        "short_culled": 0,
        # How often the SELF-NORMALIZED bar actually bound: `engaged` counts the
        # points whose effective bar came out above the absolute one, and
        # `engaged_split` how many of the splits are its doing.  Without the
        # split share the engagement rate reads as an effect when it is mostly
        # just "a tight core was measured and nothing came of it".
        "engaged": 0,
        "engaged_split": 0,
        "eff_bar_sum": 0.0,
        # MULTI-SCALE EXONERATION accounting.  `flagged` counts the members the
        # relative term alone put outside the block -- the only ones exoneration
        # may reach -- and `exonerated` how many of those it spared.  Their
        # difference is what the self-bar actually cost after the refund, which
        # is the number the collateral argument is about; reporting only the
        # spared count would hide how much was flagged in the first place.
        "flagged": 0,
        "exonerated": 0,
        "exon_points": 0,
        "exon_ratio_sum": 0.0,
        "conf_written": 0,
        "conf_median": 0.0,
        "n_before": recon.point_count,
        "n_after": recon.point_count,
        "secs": 0.0,
    }
    cloud = recon.patches
    if MEMBER_COHERENCE_BAR <= 0 or recon.point_count == 0 or cloud is None:
        return recon, st, sizes, mutated, uid
    from sfmtool._sfmtool.patches import ImagePyramidSet

    t0 = time.perf_counter()
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    n_views = np.bincount(ti, minlength=recon.point_count)
    eligible = n_views >= 3
    pts = np.nonzero(eligible)[0]
    if len(pts) == 0:
        st["secs"] = time.perf_counter() - t0
        return recon, st, sizes, mutated, uid
    res = cloud.validate_member_coherence(
        recon,
        ImagePyramidSet(recon, images),
        bar=MEMBER_COHERENCE_BAR,
        self_bar_k=MEMBER_COHERENCE_SELF_K,
        exoneration_ratio=MEMBER_COHERENCE_EXON,
        point_indexes=pts.tolist(),
        keypoint_anchor=True,
    )

    n_img = len(recon.image_names)
    # Per-observation sharpness, keyed by (point, image) so it survives the row
    # re-sort `_drop_observations` does.  Collected for every scored member of
    # every checked point, whatever the verdict: it describes the observations
    # the point SHIPS, not the ones under suspicion.  A point the eligibility
    # floor skipped, and a member the matrix could not score, keep the reserved
    # `0` -- no data-derived support, which is not a claim that they are soft.
    sharp_key, sharp_val = [], []
    split_keys, retire_pts = [], []
    for d in res:
        st["checked"] += 1
        verdict = d["verdict"]
        p = int(d["point_index"])
        if MC_SHARP_SCALE > 0:
            mem_all = np.asarray(d["members"]).astype(np.int64)
            sd = np.asarray(d["sharpness_deficit"], float)
            ok = np.isfinite(sd)
            if ok.any():
                sharp_key.append(p * n_img + mem_all[ok])
                sharp_val.append(_sharpness_to_confidence(sd[ok]))
        eff = float(d["effective_bar"])
        engaged = np.isfinite(eff) and eff > MEMBER_COHERENCE_BAR
        if engaged:
            st["engaged"] += 1
            st["eff_bar_sum"] += eff
            if verdict != "keep_all":
                st["engaged_split"] += 1
        # Exoneration accounting.  `flagged` is what the relative term put outside
        # the block -- everything exoneration was allowed to look at -- and
        # `exonerated` what it spared; their difference is the self-bar's cost
        # after the refund.  Counted on every verdict, because a track whose whole
        # rejected side is spared comes back keep_all and would otherwise vanish
        # from the accounting entirely.
        fl = np.asarray(d["relative_flagged"], bool)
        ex = np.asarray(d["exonerated"], bool)
        if fl.any():
            st["flagged"] += int(fl.sum())
            n_ex = int(ex.sum())
            st["exonerated"] += n_ex
            if n_ex:
                st["exon_points"] += 1
            rd = np.asarray(d["retained_deficit"], float)[fl]
            rd = rd[np.isfinite(rd)]
            st["exon_ratio_sum"] += float(rd.sum())
            st["exon_ratio_n"] = st.get("exon_ratio_n", 0) + len(rd)
        if verdict == "keep_all":
            st["keep_all"] += 1
            # KeepAll has three sources: every member in the block (support == k,
            # counted in neither bucket below), the margin gate REFUSING a cut it
            # could otherwise have made, and no cut having been on the table at
            # all.  The last two both leave support < k; only the margin tells
            # them apart, and a NaN one is an absence of evidence -- an unscored
            # member, or every block a singleton -- not a gate refusal.
            if int(d["support"]) < len(np.asarray(d["members"])):
                if np.isnan(d["margin"]):
                    st["unscored_keep"] += 1
                else:
                    st["margin_gated"] += 1
            continue
        if verdict == "retire":
            st["retire"] += 1
            retire_pts.append(p)
            continue
        st["split"] += 1
        mem = np.asarray(d["members"]).astype(np.int64)
        kept = np.asarray(d["kept"], bool)
        split_keys.append(p * n_img + mem[~kept])
    # Attach the column BEFORE the surgery below, so the one code path that
    # renumbers observations is also the one that keeps it aligned -- rather than
    # writing it afterwards against a numbering that has already moved.
    obs_key = ti * n_img + ii
    if sharp_key:
        conf = np.zeros(len(obs_key), np.uint8)
        keys = np.concatenate(sharp_key)
        vals = np.concatenate(sharp_val)
        order = np.argsort(obs_key, kind="stable")
        at = np.searchsorted(obs_key[order], keys)
        hit = (at < len(order)) & (
            obs_key[order][np.minimum(at, len(order) - 1)] == keys
        )
        conf[order[at[hit]]] = vals[hit]
        recon = recon.clone_with_changes(observation_confidence=conf)
        st["conf_written"] = int((conf > 0).sum())
        st["conf_median"] = (
            float(np.median(conf[conf > 0])) if (conf > 0).any() else 0.0
        )
    st["secs"] = time.perf_counter() - t0
    if split_keys or retire_pts:
        # (point, image) is the primitive's granularity — its member list is
        # deduped first-seen-wins — so a point observed twice in one image loses
        # both rows together, the same reading the eviction replay takes.
        obs = obs_key
        drop_split = (
            np.isin(obs, np.concatenate(split_keys))
            if split_keys
            else np.zeros(len(obs), bool)
        )
        drop_retire = np.isin(ti, np.asarray(retire_pts, np.int64))
        st["evicted"] = int(drop_split.sum())
        recon, sizes, mutated, uid, n_culled = _drop_observations(
            recon, drop_split | drop_retire, sizes, mutated, uid
        )
        st["retired"] = len(retire_pts)
        # Everything the length rule took beyond the retirements is a split that
        # left its point under three views.
        st["short_culled"] = n_culled - st["retired"]
        st["n_after"] = recon.point_count
    # The pass renumbers, so it registers with the provenance trace like every
    # other renumbering stage — unconditionally once it has run, so a round in
    # which it cut nothing still says so.  It flips nothing by construction (it
    # neither moves a position nor relabels a w), so an event here is a leak.
    return recon, st, sizes, mutated, uid


def _print_member_coherence(st):
    """The member-coherence vet's per-verdict casualty accounting."""
    n = max(st["n_before"], 1)
    if st.get("conf_written"):
        print(
            f"  member coherence sharpness: observation_confidence written for "
            f"{st['conf_written']} observations (median {st['conf_median']:.0f} of "
            f"255; 0 = no data-derived support), scale "
            f"{MC_SHARP_SCALE:g} ZNCC of fine-scale-only deficit"
        )
    if MEMBER_COHERENCE_SELF_K > 0 and MEMBER_COHERENCE_EXON > 0:
        fl, ex = st["flagged"], st["exonerated"]
        mean_r = st["exon_ratio_sum"] / max(st.get("exon_ratio_n", 0), 1)
        print(
            f"  member coherence exoneration (tau {MEMBER_COHERENCE_EXON:g}): "
            f"{fl} members relative-flagged, {ex} exonerated "
            f"({100.0 * ex / max(fl, 1):.1f}%) across {st['exon_points']} points; "
            f"mean retained deficit {mean_r:.3f}; the self-bar's net cost is "
            f"{fl - ex} members"
        )
    if MEMBER_COHERENCE_SELF_K > 0:
        eng = st["engaged"]
        mean_bar = st["eff_bar_sum"] / eng if eng else float("nan")
        print(
            f"  member coherence self-bar (K {MEMBER_COHERENCE_SELF_K:g}): "
            f"{eng}/{max(st['checked'], 1)} points tightened above "
            f"{MEMBER_COHERENCE_BAR:g} ({100.0 * eng / max(st['checked'], 1):.1f}%), "
            f"mean effective bar {mean_bar:.3f}; {st['engaged_split']} of them "
            f"were cut"
        )
    print(
        f"  member coherence (bar {MEMBER_COHERENCE_BAR:g}): {st['checked']} "
        f"points checked ({st['keep_all']} keep_all incl. {st['margin_gated']} "
        f"margin-gated + {st['unscored_keep']} no-evidence, {st['split']} split, "
        f"{st['retire']} retire), "
        f"{st['evicted']} members evicted, {st['retired']} points retired "
        f"(+{st['short_culled']} split under 3 views); {st['n_before']} -> "
        f"{st['n_after']} points "
        f"({100.0 * (st['n_before'] - st['n_after']) / n:.2f}% culled); "
        f"{st['secs']:.1f}s"
    )


def _late_vet(recon, images, sizes, mutated, uid):
    """Replay the member and content vetting at FINAL geometry.  Returns
    ``(recon, stats, sizes, mutated, uid)``.

    Runs LAST — after the BA, after the late ARS fit, after the infinity reframe
    and after the closing bitmap re-fuse — because that is the first moment at
    which the frames and the bitmaps under test are the ones that will ship.
    Three passes, in this order:

    1. EVICTION REPLAY.  ``_evict_track_views`` again, unchanged: the same
       relative-ZNCC bar, the same disagreement-ratio and per-view flatness
       gates, at the same thresholds, against the final frames, positions and
       consensus contrast.  It is the identical machinery, so a member disowned
       here is disowned by exactly the rule that admitted it, and the points it
       leaves under three views are retired by that pass's own length cull.
       The reference it scores against is re-fused from the source images by
       ``select_views`` on every call (it never reads the stored bitmap), so the
       replay is a fresh measurement rather than a re-reading of the pre-BA one.

    2. MEMBER COHERENCE.  The pairwise member-ZNCC matrix and its verdict
       (``_member_coherence_vet``, MEMBER_COHERENCE_BAR), on the membership pass
       1 leaves.  It is here rather than beside the evictor because it asks the
       question the evictor cannot: agreement with the fused consensus is
       agreement with a blend of the members themselves, and a track whose
       members image two surfaces balances that blend.

       ORDER against pass 3 is decided, and measured (the two orders were run
       against each other).  It is decided on STALENESS: a split changes its
       track's consensus, and only this order re-fuses that consensus before the
       re-cull — the one pass that reads the stored bitmaps — scores it.  Run
       last instead, coherence corrects tracks the re-cull has already judged and
       ships them re-fused but never re-scored: on 20250907_000240907 that is 6
       tracks in this order against 2 vetted-then-changed in the other.  The
       casualty totals barely move (2696 points against 2695 there, 2341 either
       way on 20240614_224422531), so the choice is not a yield argument — but
       the POPULATION does move, and in the direction that says which test gets
       to state the reason: coherence-first reaches 19 of 907's points and 9 of
       20240614_224422531's, coherence-last only 11 and 4, because the re-cull
       has already taken roughly half of them on their blended texture rather
       than on their incoherence.

    3. LOCALIZABILITY RE-CULL.  The same scorer, unit and threshold the embed
       cull and the promotion gate use (``PatchCloud.score_localizability``,
       sigma_noise 3.0, tau in patch-grid px), plus the anisotropy bound
       (LOC_MAX_ANISO), applied to the consensus each point actually carries.

    NO STALE-BITMAP WINDOW, and it is checked at each boundary.  Passes 1 and 2
    read the SOURCE images through the frames (``select_views`` and
    ``validate_member_coherence`` both re-fuse or re-render on every call and
    neither reads ``patch_bitmaps``), so neither can be misled by a bitmap the
    other invalidated; each nonetheless re-fuses the tracks it took a member from
    before control leaves it, so the only consumer that DOES read the stored
    bitmaps — pass 3 — scores a consensus fused from exactly the membership that
    ships.  The re-fuse after pass 2 is what makes the ordering above sound.

    ONE ROUND, not a loop.  The eviction changes the consensus of the tracks it
    touches, so their bitmaps are re-fused between the passes and the
    re-cull judges the fused-from-survivors texture.  A second eviction round
    scores against a reference that is already robust to what the first round
    removed — ``select_views`` builds it by IRLS, which redescends the outlying
    members rather than averaging them in — so the second round's bar is the
    first round's bar.  Measured by replaying the evictor on what this pass
    ships (``latevet_round2``): a second round reaches 0.16% of
    20250907_000240907's observations, 0.05% of 20240614_224422531's and 0.01%
    of DnDTabletop's, and retires no point at all on any of the three.  That is
    a residue, not a round.

    The ARS fit is NOT re-run over the culled graph, and the reason is not that
    a refit would change nothing — measured on 20250907_000240907, it turns 1239
    of 2201 normals past a degree, median 2.2 and p90 24.  The reason is what
    those turns
    would MEAN here.  A turned frame is a frame the shipped bitmap was not
    rendered through, so a refit at this site re-creates exactly the staleness
    this pass exists to catch: it would demand another re-fuse, and a re-fuse
    demands another vet, which is the iteration the design refuses.  The refit is
    also a WEAKER estimator, not a better-informed one — its graph is the culled
    cloud (2201 neighbours against 2779), and a point whose own patch cannot pin
    a keypoint still carries a perfectly usable DIRECTION to its neighbours,
    which is all the plane fit reads.  So the one fit that writes runs before the
    vet, on the adjusted cloud, and the vet is the terminal judge of the file.

    ``mutated`` arrives DISCHARGED (the caller's re-fuse has just rendered every
    point that carried a stale bitmap), so this pass tracks its own and returns
    it discharged too."""
    st = {
        "evict": None,
        "mc": None,
        "mc_refused": 0,
        "mc_cleared": 0,
        "scored": 0,
        "sigma_culled": 0,
        "aniso_culled": 0,
        "both": 0,
        "no_consensus": 0,
        "sigma_med": float("nan"),
        "ratio_med": float("nan"),
        "inf_culled": 0,
        "refused": 0,
        "cleared": 0,
        "n_before": recon.point_count,
        "n_after": recon.point_count,
        "secs": 0.0,
    }
    if not LATE_VET or recon.point_count == 0 or recon.patches is None:
        return recon, st, sizes, mutated, uid
    t0 = time.perf_counter()

    # 1. Eviction replay.  Its own `mutated` output is what the re-fuse below
    # consumes: the pass marks every track it took a member from.
    touched = np.zeros(recon.point_count, bool)
    n_obs = len(np.asarray(recon.track_point_indexes))
    recon, ev, sizes, touched, uid = _evict_track_views(
        recon, images, sizes, touched, uid
    )
    st["evict"] = ev
    st["evict_obs"] = n_obs
    if touched.any():
        recon, n_fresh, n_clear = _rerender_mutated_bitmaps(recon, images, touched)
        st["refused"], st["cleared"] = n_fresh, n_clear
    stage_dump("late-evict-replay", recon)

    # 2. Member coherence, on the membership the eviction leaves.  Its own
    # `mutated` output drives the second re-fuse: a split track's consensus was
    # fused from members it no longer owns, and pass 3 reads that consensus.
    touched = np.zeros(recon.point_count, bool)
    recon, mc, sizes, touched, uid = _member_coherence_vet(
        recon, images, sizes, touched, uid
    )
    st["mc"] = mc
    if touched.any():
        recon, n_fresh, n_clear = _rerender_mutated_bitmaps(recon, images, touched)
        st["mc_refused"], st["mc_cleared"] = n_fresh, n_clear
    mutated = np.zeros(recon.point_count, bool)
    stage_dump("late-member-coherence", recon)

    # 3. Localizability re-cull, on the bitmaps as they now stand.
    bmp = recon.patch_bitmaps
    cloud = recon.patches
    if bmp is None or cloud is None:
        st["secs"] = time.perf_counter() - t0
        st["n_after"] = recon.point_count
        return recon, st, sizes, mutated, uid
    scored = cloud.score_localizability(
        recon, np.ascontiguousarray(np.asarray(bmp, dtype=np.uint8)), sigma_noise=3.0
    )
    sigma = np.asarray(scored["sigma_pos_grid"], dtype=float)
    lam1 = np.asarray(scored["lam1"], dtype=float)
    lam2 = np.asarray(scored["lam2"], dtype=float)
    bad_sigma, bad_aniso = _localizability_verdict(
        sigma, lam1, lam2, _LATE_VET_MAX_SIGMA_GRID, LOC_MAX_ANISO
    )
    # POINTS AT INFINITY are scored on the same footing as finite ones, and this
    # is a decision rather than an oversight.  Their frame is synthetic, but it
    # is a REAL frame: `_normalize_infinity_frames` gives every one of them an
    # orthonormal tangent basis with |hu| = |hv|, so the grid is isotropic in
    # angle, and the frame faces the bearing by construction, so it carries none
    # of the obliquity smear the finite frames do.  The reading is therefore the
    # same quantity, and it says the same thing: a bearing whose patch slides
    # laterally is an unpinned direction.  Measured on 20250907_000240907's 511
    # infinity points: median sigma 0.039 and median ratio 3.8, against 0.073 and
    # 11.1 for the finite cloud — they are the HEALTHY half, and the uniform rule
    # costs 2 of them on sigma.
    scorable = np.isfinite(sigma)
    st["scored"] = int(scorable.sum())
    st["no_consensus"] = int((~scorable).sum())
    if scorable.any():
        st["sigma_med"] = float(np.median(sigma[scorable]))
        st["ratio_med"] = float(
            np.median(lam1[scorable] / np.maximum(lam2[scorable], _LOC_LAM2_FLOOR))
        )
    bad = bad_sigma | bad_aniso
    st["sigma_culled"] = int((bad_sigma & ~bad_aniso).sum())
    st["aniso_culled"] = int((bad_aniso & ~bad_sigma).sum())
    st["both"] = int((bad_sigma & bad_aniso).sum())
    st["inf_culled"] = int((bad & np.asarray(recon.point_is_at_infinity)).sum())
    if bad.any():
        keep = np.ascontiguousarray(~bad)
        recon = recon.filter_points_by_mask(keep)
        sizes, mutated, uid = (
            np.asarray(sizes)[keep],
            np.asarray(mutated)[keep],
            np.asarray(uid)[keep],
        )
        # Culling a point changes no survivor's track, so no consensus needs
        # re-fusing here; only the stored errors describe a different cloud.
        recon = _refresh_errors(recon)
    stage_dump("late-localizability-recull", recon)
    st["n_after"] = recon.point_count
    st["secs"] = time.perf_counter() - t0
    return recon, st, sizes, mutated, uid


def _print_late_vet(st):
    """The late vetting's casualty accounting, per gate."""
    if st["evict"] is not None:
        _print_evict("late", st["evict"], st.get("evict_obs", 0))
    if st.get("mc") is not None and MEMBER_COHERENCE_BAR > 0:
        _print_member_coherence(st["mc"])
    n = max(st["n_before"], 1)
    culled = st["sigma_culled"] + st["aniso_culled"] + st["both"]
    print(
        f"  late localizability re-cull: {st['scored']} points scored "
        f"({st['no_consensus']} no consensus, kept), {culled} culled "
        f"({st['sigma_culled']} sigma > {_LATE_VET_MAX_SIGMA_GRID:g} + "
        f"{st['aniso_culled']} lam1/lam2 > {LOC_MAX_ANISO:g} + {st['both']} both; "
        f"{st['inf_culled']} at infinity); shipped consensus median sigma "
        f"{st['sigma_med']:.3f} grid px, median ratio {st['ratio_med']:.2f}"
    )
    print(
        f"  late vet: {st['n_before']} -> {st['n_after']} points "
        f"({100.0 * (st['n_before'] - st['n_after']) / n:.1f}% culled); "
        f"{st['refused']} consensus re-fused ({st['cleared']} cleared) after the "
        f"eviction replay, {st['mc_refused']} ({st['mc_cleared']} cleared) after "
        f"the coherence vet; {st['secs']:.1f}s"
    )


# ── Point identity and infinity evidence ─────────────────────────────────────
#
# A per-point UID carried through the finalization exactly as ``det_size`` and
# ``mutated`` are — every renumbering pass returns the uid array with the same
# rows dropped, and a collapse survivor keeps its uid while the merged-away
# members end theirs.  That is what lets a later pass ask what became of a point
# the passes above it renumbered (``_n_promoted_alive``).

_UID_STATE = {"next_uid": 0}


def _seed_point_uids(n):
    """A fresh block of ``n`` point uids — stable per-point identity carried
    across the finalization's renumbering passes."""
    base = _UID_STATE["next_uid"]
    _UID_STATE["next_uid"] = base + int(n)
    return np.arange(base, base + int(n), dtype=np.int64)


def _inf_proj_err(cam, p_cam, uv):
    """Pixel error of camera-frame points ``p_cam`` against the observations
    ``uv``.  Canonical frame is -Z forward, so a row outside the camera's
    imaged cone is behind it and scores 1e6 — never an agreement, and never a
    nan that would silently drop out of a median.  ``_in_field`` is what makes
    the cone the MODEL's (the half-space for a pinhole, theta <= r_max/f for a
    fisheye); the pinhole reading would score every peripheral observation of a
    >180 degree capture as a maximal disagreement, which is how the infinity
    gate came to see the periphery as unexplainable."""
    pred = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(p_cam)))
    e = np.hypot(pred[:, 0] - uv[:, 0], pred[:, 1] - uv[:, 1])
    return np.where(np.isfinite(e) & _in_field(cam, p_cam), e, 1e6)


def _infinity_fit_px(recon, idx):
    """How well the BEARING MODEL explains each track of ``idx``, at ``recon``'s
    current geometry, as an array parallel to ``idx``.

    Per point: the median reprojection of the track's bearing mean — its own
    observations read as a single DIRECTION, the rule the native classifier
    installs — over those same observations.  It is what the point becomes when
    demoted, so a caller measuring a DEMOTION must pass the reconstruction as it
    stood BEFORE the demoting call.  A track with no observations comes back
    nan."""
    n = len(idx)
    out = np.full(n, np.nan)
    if n == 0 or recon.point_count == 0:
        return out
    idx = np.asarray(idx, dtype=np.int64)
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, dtype=np.float64)
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    cam = recon.cameras[0]
    order = np.argsort(ti, kind="stable")
    start = np.searchsorted(ti[order], np.arange(recon.point_count + 1))
    per = [order[start[p] : start[p + 1]] for p in idx.tolist()]
    cnt = np.array([len(r) for r in per], dtype=np.int64)
    sel = np.nonzero(cnt > 0)[0]
    if len(sel) == 0:
        return out
    rows = np.concatenate([per[k] for k in sel.tolist()])
    cnt_s = cnt[sel]
    grp = np.concatenate([[0], np.cumsum(cnt_s)]).astype(np.int64)
    # World-space observation rays: pixel_to_ray gives the camera-frame ray and
    # R^T carries it to the world (the convention every other pass here uses).
    d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(uv[rows]))
    rays = np.einsum("nji,nj->ni", rots[ii[rows]], d_loc)
    rays = rays / np.maximum(np.linalg.norm(rays, axis=1), 1e-300)[:, None]
    acc = np.add.reduceat(rays, grp[:-1], axis=0)
    nrm = np.maximum(np.linalg.norm(acc, axis=1), 1e-300)
    dbar = acc / nrm[:, None]
    p_inf = np.einsum("nij,nj->ni", rots[ii[rows]], np.repeat(dbar, cnt_s, axis=0))
    e_inf = _inf_proj_err(cam, p_inf, uv[rows])
    for k in range(len(sel)):
        out[int(sel[k])] = float(np.median(e_inf[int(grp[k]) : int(grp[k + 1])]))
    return out


# ── Infinity demotion gate (SFMTOOL_INF_DEMOTION_GATE) ───────────────────────
#
# Demotion is relabel-only and irreversible in practice, and it REPLACES the
# track's position with a single direction — so it is only ever a faithful
# description of a track the BEARING MODEL can explain.  A provenance trace over
# the demotions the finalization actually makes found a population it cannot:
# on 20250907_000240907, 383 of the 791 points that
# shipped at infinity had a demotion-time bearing fit ABOVE the classifier's own
# noise floor — the model chosen for them disagreed with their own observations
# by more than the measurement noise the same classifier was calibrated with —
# while their finite positions fit (median 0.44 px).
#
# The test is that bearing fit: veto a demotion whose track's median bearing-fit
# reprojection exceeds SEED_INFINITY_NOISE_PX, at the poses current at that site.
# It is one-sided — nothing new is ever demoted — and conservative in the right
# direction: a genuinely far point's rays ARE a bearing, so it fits (median
# 0.58 px on the same dataset) and demotes exactly as before.
#
# The gate is ASYMMETRIC across the two sites, because they do not decide the
# same thing.  The PRE-BA classify is upstream of the adjustment: a point kept
# finite there is a LANDMARK the BA gets to use and to move, and the classify
# that runs after the BA sees it again.  Keeping a questionable point finite
# there costs one round of doubt and buys the BA a constraint, so that site
# vetoes on the bearing fit alone.  The POST-BA classify is the LAST word —
# whatever it labels ships — so it is the decisive site and it has to answer
# for the position a veto ships, not only for the label.  There it vetoes only
# when the misfit is joined by DEPTH PLAUSIBILITY: the point's own median
# camera-frame depth must lie inside
# INF_GATE_DEPTH_LO_MULT * p10 .. INF_GATE_DEPTH_HI_MULT * p90 of the finite
# cloud's per-point median depths, measured on that same reconstruction, so the
# band is derived per dataset from the scene the BA just produced.
#
# The archived 907 evidence (workspace-prep/inf-trace-907.npz, inf-rescue-907.npz)
# is what sizes this.  At the post-BA site the bearing test alone vetoes 62
# rescuable and 23 far demotions — 73% precision, and the rescuable ones are
# points whose finite positions fit their own observations.  The plausibility
# conjunct costs exactly one of those 62, and caps the depth any veto can ship at
# about 12 against a cloud median of 3.7.  What it excludes is the population the
# bearing test cannot see: a point the BA ran away along its own ray normally
# still demotes (its rays ARE a bearing), but not always — one point on 907 and
# two on 20240614_224422531 failed the bearing test by a hair (1.12-1.40 px), and
# the uniform gate shipped them finite at depth 8.3e4, 8.7e8 and 5524 against
# cloud medians of 3.7 and 52.  Those depths are 1e4-1e8 times the band; they
# cannot pass it, and they demote as they should.


# Depth-plausibility band at the post-BA site, as multiples of the finite
# cloud's own per-point median-depth percentiles.  Wide by construction: the
# question a veto has to answer is not "is this depth typical" but "is this a
# position the scene could contain at all", and the population it must exclude
# misses by four to eight orders of magnitude.
INF_GATE_DEPTH_LO_MULT = 0.5
INF_GATE_DEPTH_HI_MULT = 2.0


def _inf_gate_median_depths(recon, idx):
    """Median camera-frame depth of each point of ``idx``, over its own
    observations, at ``recon``'s current geometry.

    Canonical frame is -Z forward, so the returned depth is positive in front of
    the camera and a negative median is a cheirality violation (the sign is
    kept).  A point with no observations comes back nan; a point stored AT
    INFINITY holds a direction rather than a position, so its "depth" is
    meaningless and callers must exclude it."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.full(len(idx), np.nan)
    if len(idx) == 0 or recon.point_count == 0:
        return out
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    tv = np.asarray(recon.translations, dtype=np.float64)
    pos = np.asarray(recon.positions, dtype=np.float64)
    order = np.argsort(ti, kind="stable")
    start = np.searchsorted(ti[order], np.arange(recon.point_count + 1))
    cnt = (start[idx + 1] - start[idx]).astype(np.int64)
    sel = np.nonzero(cnt > 0)[0]
    if len(sel) == 0:
        return out
    rows = np.concatenate([order[start[p] : start[p + 1]] for p in idx[sel].tolist()])
    cnt_s = cnt[sel]
    p_cam = (
        np.einsum("nij,nj->ni", rots[ii[rows]], pos[np.repeat(idx[sel], cnt_s)])
        + tv[ii[rows]]
    )
    depth = _cam_depth(p_cam)
    # Per-track median without a per-track call: sort by (track, depth) once and
    # index the middle element(s) of each run.
    grp = np.repeat(np.arange(len(sel)), cnt_s)
    ds = depth[np.lexsort((depth, grp))]
    st = np.concatenate([[0], np.cumsum(cnt_s)]).astype(np.int64)
    out[sel] = 0.5 * (ds[st[:-1] + (cnt_s - 1) // 2] + ds[st[:-1] + cnt_s // 2])
    return out


def _inf_gate_veto(tag, before, idx, depth_plausibility=False):
    """Which of the demotions proposed at ``tag`` the gate refuses, as a bool
    array parallel to ``idx``.

    ``before`` is the reconstruction as it stood BEFORE the demoting call, in
    the same point indexing, so both readings below are taken at the geometry
    current at that site.  The bearing is the track's RAY MEAN — the same
    direction the native classifier installs — and the fit is the median
    reprojection of that direction over the track's own observations, through
    the camera's own projection (``_infinity_fit_px``).

    ``depth_plausibility`` adds the second conjunct the decisive post-BA site
    requires: the point's finite position must also be one the scene could hold,
    its median camera-frame depth inside the band the finite cloud's own p10/p90
    define at that moment (the cloud excludes the proposals themselves, so a
    large batch of runaways cannot widen the band that judges them).  A misfit
    demotion whose position is implausible goes through — the label is wrong for
    it either way, and at this site the alternative is shipping the position.
    If the cloud is too small to define a band, nothing is vetoed: plausibility
    that cannot be established is not a reason to keep a position.

    Always measures and prints, so the gate-off arm of an A/B still reports the
    counterfactual; only the returned mask is emptied when the gate is off."""
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(0, bool)
    fit = _infinity_fit_px(before, idx)
    misfit = np.isfinite(fit) & (fit > SEED_INFINITY_NOISE_PX)
    veto, dep, band = misfit, None, None
    if depth_plausibility:
        dep = _inf_gate_median_depths(before, idx)
        cloud = ~np.asarray(before.point_is_at_infinity, bool)
        cloud[idx] = False
        cd = _inf_gate_median_depths(before, np.nonzero(cloud)[0])
        cd = cd[np.isfinite(cd) & (cd > 0)]
        if len(cd) >= 10:
            p10, p90 = (float(x) for x in np.percentile(cd, [10.0, 90.0]))
            band = (INF_GATE_DEPTH_LO_MULT * p10, INF_GATE_DEPTH_HI_MULT * p90)
            plausible = np.isfinite(dep) & (dep >= band[0]) & (dep <= band[1])
        else:
            p10 = p90 = float("nan")
            plausible = np.zeros(len(idx), bool)
        veto = misfit & plausible

    def _med(v):
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    msg = (
        f"  infinity demotion gate ({tag}): {len(idx)} demotions proposed, "
        f"{int(misfit.sum())} over the {SEED_INFINITY_NOISE_PX:.2f} px noise "
        f"floor (bearing-fit median {_med(fit[misfit]):.2f} px, kept "
        f"{_med(fit[~misfit]):.2f} px)"
    )
    if depth_plausibility:
        drop = misfit & ~veto
        d_drop = dep[drop]
        d_fin = d_drop[np.isfinite(d_drop)]
        d_max = float(d_fin.max()) if len(d_fin) else float("nan")
        if band is None:
            # Finite cloud under 10 points: no band exists, so nothing is
            # vetoed (the docstring's "plausibility that cannot be established
            # is not a reason to keep a position").  Report that rather than
            # formatting a band that was never computed.
            msg += (
                f"; no depth band (finite cloud under 10 points): "
                f"{int(veto.sum())} vetoed, {int(drop.sum())} demoted despite "
                f"misfit with plausibility unestablished (their median depths "
                f"{_med(d_drop):.4g}, max {d_max:.4g})"
            )
        else:
            msg += (
                f"; depth band [{band[0]:.3g}, {band[1]:.3g}] from cloud "
                f"p10/p90 {p10:.3g}/{p90:.3g}: {int(veto.sum())} vetoed, "
                f"{int(drop.sum())} demoted despite misfit as depth-implausible "
                f"(their median depths {_med(d_drop):.4g}, max {d_max:.4g})"
            )
    else:
        msg += f"; {int(veto.sum())} vetoed"
    print(msg + ("" if INF_DEMOTION_GATE else " [GATE OFF: advisory only]"))
    return veto if INF_DEMOTION_GATE else np.zeros(len(idx), bool)


def _inf_gate_restore(before, after, rows):
    """Put the per-point state of ``rows`` back the way ``before`` held it.

    The native classifier does not renumber, so a vetoed row is restored in
    place: the HOMOGENEOUS position (the w it rewrote is the whole point), the
    normal and its confidence (it zeroes both for a demoted row) and the patch
    frame with its bitmap (it converts the frame to angular extents, or clears
    it outright).  Frames and bitmaps go back in ONE ``clone_with_changes`` —
    passing ``patches=`` alone drops every stored bitmap."""
    from sfmtool._sfmtool.patches import PatchCloud

    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return after
    xyzw = np.asarray(after.positions_xyzw, dtype=np.float64).copy()
    xyzw[rows] = np.asarray(before.positions_xyzw, dtype=np.float64)[rows]
    changes = {"positions": np.ascontiguousarray(xyzw)}
    if after.has_normals and before.has_normals:
        nrm = np.asarray(after.normals, dtype=np.float32).copy()
        nrm[rows] = np.asarray(before.normals, dtype=np.float32)[rows]
        changes["normals"] = np.ascontiguousarray(nrm)
    conf_a, conf_b = after.normal_confidence, before.normal_confidence
    if conf_a is not None and conf_b is not None:
        conf = np.asarray(conf_a, dtype=np.uint8).copy()
        conf[rows] = np.asarray(conf_b, dtype=np.uint8)[rows]
        changes["normal_confidence"] = np.ascontiguousarray(conf)
    if after.patches is not None:
        hu, hv = _patch_halfvecs(after)
        hu_b, hv_b = _patch_halfvecs(before)
        hu[rows], hv[rows] = hu_b[rows], hv_b[rows]
        changes["patches"] = PatchCloud.from_halfvec_arrays(
            np.ascontiguousarray(hu, dtype=np.float32),
            np.ascontiguousarray(hv, dtype=np.float32),
            np.ascontiguousarray(xyzw[:, :3]),
        )
        bmp = after.patch_bitmaps
        if bmp is not None:
            bmp = np.asarray(bmp).copy()
            bmp_b = before.patch_bitmaps
            if bmp_b is not None:
                bmp[rows] = np.asarray(bmp_b)[rows]
            changes["patch_bitmaps"] = np.ascontiguousarray(bmp)
    return after.clone_with_changes(**changes)


def _infinity_angular_extent(recon, rad_px):
    """Per-point angular half-extent for the points at infinity: the angle that
    subtends ``rad_px`` pixels at the point's BEARING in each observing view,
    minimised across views (the ``PixelRadius`` ``Min`` reduce — the patch stays
    inside its pixel budget in every view).

    The rule is the camera's own: ``rad_px / (|p_cam| * sigma_min)``, the local
    pixels-per-radian in the least-magnified tangent direction
    (``CameraIntrinsics.pixel_radius_to_angle_batch``).  Dividing by ``f`` — the
    reading this replaced — is that quantity only on the optical axis and only
    for an equidistant fisheye at every angle; for a pinhole the pixels-per-radian
    is ``f*sec(theta)``, so ``rad_px/f`` oversizes an off-axis bearing by
    ``1/cos(theta)`` (2x at 60 deg).  Points with no observation keep the on-axis
    reading.  Returns an array of angles parallel to the points."""
    n = recon.point_count
    ang = np.asarray(rad_px, dtype=np.float64) / max(
        float(recon.cameras[0].focal_lengths[0]), 1e-9
    )
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    if n == 0 or len(ti) == 0:
        return ang
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    ci = np.asarray(recon.camera_indexes).astype(np.int64)
    q = np.ascontiguousarray(recon.quaternions_wxyz, dtype=np.float64)
    d = np.asarray(recon.positions, dtype=np.float64)
    dn = np.linalg.norm(d, axis=1)
    d = np.where((dn > 1e-12)[:, None], d / np.maximum(dn, 1e-30)[:, None], d)
    # Rotate each observed bearing into its camera's frame (w = 0: no translation).
    qw, qx, qy, qz = q[ii, 0], q[ii, 1], q[ii, 2], q[ii, 3]
    v = d[ti]
    t = 2.0 * np.cross(np.stack([qx, qy, qz], 1), v)
    rays = np.ascontiguousarray(
        v + qw[:, None] * t + np.cross(np.stack([qx, qy, qz], 1), t)
    )
    per_obs = np.full(len(ti), np.inf)
    px = np.asarray(rad_px, dtype=np.float64)[ti]
    for cam_id in np.unique(ci[ii]):
        m = ci[ii] == cam_id
        per_obs[m] = recon.cameras[int(cam_id)].pixel_radius_to_angle_batch(
            np.ascontiguousarray(rays[m]), np.ascontiguousarray(px[m])
        )
    out = np.full(n, np.inf)
    np.minimum.at(out, ti, per_obs)
    return np.where(np.isfinite(out), out, ang)


def _normalize_infinity_frames(recon, det_size, refine_radius):
    """Rebuild every infinity point's patch frame from its DETECTION size:
    orthonormal tangent basis, ``|hu| = |hv|`` the angle that detection
    footprint subtends at the point's bearing (see
    ``_infinity_angular_extent``), ``u x v`` along ``-d``.

    IDEMPOTENT and depth-free, unlike deriving angular extents by dividing the
    stored world extents by the demotion-time distance — that rule silently
    assumes the stored extent matches the CURRENT distance, which a far-field
    point that ran away between embedding and demotion violates, and it
    compounds when a point round-trips finite<->infinite (an intermediate step
    that re-triangulates and re-promotes leaves ANGULAR extents behind, and the
    next demotion divides them by the distance AGAIN — observed at 1e-10 after
    two ~3400x divisions).  Returns ``(recon, n_rebuilt)``."""
    from sfmtool._sfmtool.patches import PatchCloud

    n = recon.point_count
    if n == 0 or recon.patches is None or recon.infinity_point_count == 0:
        return recon, 0
    inf = np.asarray(recon.point_is_at_infinity)
    xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64)
    d = xyzw[:, :3].copy()
    dn = np.linalg.norm(d, axis=1)
    tgt = inf & (dn > 1e-12)
    if not tgt.any():
        return recon, 0
    d[dn > 1e-12] /= dn[dn > 1e-12, None]

    rad = np.asarray(det_size, dtype=np.float64) * (
        _ARS_RADIUS_MULT / max(float(refine_radius), 1e-9)
    )
    if (rad > 0).any():
        rad = np.where(rad > 0, rad, float(np.median(rad[rad > 0])))
    ang = _infinity_angular_extent(recon, rad)

    hu, hv = _patch_halfvecs(recon)
    lu = np.linalg.norm(hu, axis=1)
    has = tgt & (lu > 0)
    if not has.any():
        return recon, 0
    # v direction: current v projected onto the tangent plane; deterministic
    # axis fallback when it is degenerate (matches OrientedPatch's rule).
    v0 = hv.astype(np.float64)
    vt = v0 - (v0 * d).sum(1)[:, None] * d
    vl = np.linalg.norm(vt, axis=1)
    a = np.where((np.abs(d[:, 0]) < 0.9)[:, None], (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    alt = a - (a * d).sum(1)[:, None] * d
    alt /= np.maximum(np.linalg.norm(alt, axis=1), 1e-30)[:, None]
    vhat = np.where((vl > 1e-12)[:, None], vt / np.maximum(vl, 1e-30)[:, None], alt)
    # u = v x n with n = -d, so u x v runs along -d per the format convention.
    uhat = np.cross(vhat, -d)
    hu_out = np.where(has[:, None], (uhat * ang[:, None]), hu).astype(np.float32)
    hv_out = np.where(has[:, None], (vhat * ang[:, None]), hv).astype(np.float32)

    changes = {
        "patches": PatchCloud.from_halfvec_arrays(
            np.ascontiguousarray(hu_out),
            np.ascontiguousarray(hv_out),
            np.ascontiguousarray(xyzw[:, :3]),
        )
    }
    bmp = recon.patch_bitmaps
    if bmp is not None:
        changes["patch_bitmaps"] = np.ascontiguousarray(bmp)
    return recon.clone_with_changes(**changes), int(has.sum())


# ── Adaptive robust surfel normals ───────────────────────────────────────────


def _ars_tangent_basis(view):
    """Row-wise orthonormal basis of the plane orthogonal to ``view``.

    NORMAL-FREE by construction — a pure function of the point's mean viewing
    direction — so the sector bookkeeping it supports cannot be circular with the
    normal it is used to judge.  A shift along this plane is also the direction of
    maximum image-space leverage."""
    v = np.array(view, dtype=np.float64, copy=True).reshape(-1, 3)
    ln = np.linalg.norm(v, axis=1)
    v[ln < 1e-9] = (0.0, 0.0, 1.0)
    v /= np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
    w = np.tile(np.array([0.0, 0.0, 1.0]), (len(v), 1))
    w[np.abs(v[:, 2]) > 0.95] = (1.0, 0.0, 0.0)
    e1 = w - (w * v).sum(1)[:, None] * v
    e1 /= np.maximum(np.linalg.norm(e1, axis=1), 1e-12)[:, None]
    return e1, np.cross(v, e1)


def _ars_sector_ids(d, e1, e2):
    """Tangent-sector index (0 .. _ARS_N_SECTORS-1) of each row of ``d``."""
    ang = np.arctan2((d * e2).sum(1), (d * e1).sum(1))
    return ((ang / (2 * np.pi) + 1.0) * _ARS_N_SECTORS).astype(
        np.int64
    ) % _ARS_N_SECTORS


def _ars_edges(recon, rad_pt):
    """Image-space adjacency of the finite points: ``(ptr, dst, s, stats)``.

    Two points are ADJACENT when, in a majority of the images that see both, their
    keypoints sit inside an annulus running from ``_ARS_A_LO`` to ``ARS_B_MAX``
    times the pair's detection radius (the smaller of the two points'), the pair
    shares at least ``_ARS_MIN_SHARED`` images, and their RANGES from the shared
    cameras agree to within ``_ARS_DEPTH_TOL`` — which is what separates "next to
    each other on the surface" from "one behind the other along the viewing ray".

    Returned as symmetric CSR, each point's neighbours ordered by ``s``, the pair's
    MEDIAN image-space separation in detection-radius units.

    Built by the native observation-adjacency builder: the criterion, the
    parameters and the symmetric-CSR contract are the spec's
    (`specs/core/analysis/observation-adjacency-graph.md`); the builder derives the camera
    centres from the poses itself.  ``stats['cand']`` is 0 because the native
    builder does not report a candidate count."""
    from sfmtool._sfmtool.analysis import build_observation_adjacency

    t0 = time.perf_counter()
    g = build_observation_adjacency(
        np.asarray(recon.keypoints_xy, dtype=np.float64),
        np.asarray(recon.track_point_indexes, dtype=np.uint32),
        np.asarray(recon.track_image_indexes, dtype=np.uint32),
        np.asarray(rad_pt, dtype=np.float32),
        np.asarray(recon.point_is_at_infinity, dtype=bool),
        np.asarray(recon.positions, dtype=np.float64),
        np.asarray(recon.quaternions_wxyz, dtype=np.float64),
        np.asarray(recon.translations, dtype=np.float64),
        b_max=ARS_B_MAX,
        a_lo=_ARS_A_LO,
        min_shared_images=_ARS_MIN_SHARED,
        majority=_ARS_MAJORITY,
        range_tol=_ARS_DEPTH_TOL,
    )
    ptr = np.asarray(g["offsets"], dtype=np.int64)
    dst = np.asarray(g["neighbours"], dtype=np.int64)
    sv = np.asarray(g["separation_med"], dtype=np.float64)
    deg = np.diff(ptr)
    stats = {
        "edges": len(dst) // 2,
        "cand": 0,
        "deg_med": float(np.median(deg[deg > 0])) if (deg > 0).any() else 0.0,
        "secs": time.perf_counter() - t0,
    }
    return ptr, dst, sv, stats


def _ars_fit(pos, view_dir, ptr, dst, sel, extras=None):
    """p-anchored robust surfel fit over the points selected by ``sel``.

    The plane passes through p itself (surfel semantics) and is fitted on the UNIT
    neighbour directions, so each neighbour contributes its angular deviation once
    rather than in proportion to its distance.  ``extras`` maps a selected point to
    an ``(k, 3)`` block of synthesised neighbour POSITIONS (the expansion stage's
    survivors), admitted at weight 1 — they were placed at one patch diameter,
    i.e. inside any support that admits anything at all.

    Returns ``(normals, diag, determined)``, all dense over the cloud, with
    ``nan`` / ``False`` for anything not fitted.

    The fit, the diagnostics and the determinacy predicate are the native
    adjacency-surfel kernel's (`specs/core/analysis/adjacency-surfel-normals.md`).  The
    kernel reports NOTHING for a point it could not fit (fewer than two usable
    neighbour directions): every diagnostic comes back ``nan``.  Only ``n_eff``
    is read downstream."""
    from sfmtool._sfmtool.analysis import estimate_adjacency_surfel_normals

    blocks = {
        int(p): np.ascontiguousarray(np.asarray(b, dtype=np.float64).reshape(-1, 3))
        for p, b in (extras or {}).items()
    }
    out = estimate_adjacency_surfel_normals(
        np.ascontiguousarray(pos, dtype=np.float64),
        np.ascontiguousarray(ptr, dtype=np.uint32),
        np.ascontiguousarray(dst, dtype=np.uint32),
        np.ascontiguousarray(view_dir, dtype=np.float64),
        np.ascontiguousarray(sel, dtype=bool),
        extras=blocks or None,
        irls_iters=ARS_IRLS_ITERS,
        tukey_c=ARS_TUKEY_C,
        sigma_floor_deg=ARS_SIGMA_FLOOR_DEG,
        n_sectors=_ARS_N_SECTORS,
        det_n_eff=ARS_DET_NEFF,
        det_sectors=ARS_DET_SECTORS,
        det_aniso=ARS_DET_ANISO,
    )
    diag = {
        "n_eff": np.asarray(out["n_eff"], dtype=np.float64),
        "sectors": np.asarray(out["sectors"], dtype=np.float64),
        "aniso": np.asarray(out["anisotropy"], dtype=np.float64),
        "sigma_deg": np.asarray(out["sigma_deg"], dtype=np.float64),
        "resid_deg": np.asarray(out["resid_deg"], dtype=np.float64),
        "n_support": np.asarray(out["n_support"], dtype=np.float64),
    }
    return (
        np.asarray(out["normals"], dtype=np.float64),
        diag,
        np.asarray(out["determined"], dtype=bool),
    )


def _ars_expand_coverage_gate(
    coverage, recon, rad_pt, cand_p, ctr, order, starts, ends
):
    """Boolean over expansion candidates: does the reach land on UNCLAIMED image
    area?

    Coverage is what tells the aiming apart from the graph.  A direction that the
    atlas reports as already claimed, yet whose owning tracks are NOT neighbours
    of p in the adjacency graph, was almost certainly thrown out by that graph's
    RANGE vet — those observations sit at a different depth, i.e. on a different
    surface, and only project next to p.  Congealing a synthetic patch that way
    manufactures precisely the off-surface neighbour the robust fit then has to
    redescend to zero, at full congealing cost.  So the candidate has to reach
    into image area no observation claims.

    Per candidate view: p and the synthetic centre are both projected, the
    image-space direction between them is binned into the kernel's sector
    convention, and the atlas is asked whether that sector still holds uncovered
    cells within the reach radius of p's own keypoint.  The candidate survives on
    a majority of the views where the direction is computable; a candidate with
    no computable view survives, since that is no evidence against it."""
    n_c = len(cand_p)
    keep = np.ones(n_c, bool)
    if n_c == 0:
        return keep
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    uv = np.asarray(recon.keypoints_xy, dtype=np.float64)
    pos = np.asarray(recon.positions, dtype=np.float64)
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    tv = np.asarray(recon.translations, dtype=np.float64)
    cam = recon.cameras[0]

    # One flat row per (candidate, view of its point) — no per-candidate call.
    deg = (ends - starts)[cand_p]
    row = np.repeat(np.arange(n_c, dtype=np.int64), deg)
    off = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
    slot = np.arange(int(deg.sum()), dtype=np.int64) - off[row] + starts[cand_p][row]
    obs = order[slot]
    img = ii[obs]
    p_row = cand_p[row]

    xc = np.einsum("nij,nj->ni", rots[img], ctr[row]) + tv[img]
    xp = np.einsum("nij,nj->ni", rots[img], pos[p_row]) + tv[img]
    d = cam.ray_to_pixel_batch(np.ascontiguousarray(xc)) - cam.ray_to_pixel_batch(
        np.ascontiguousarray(xp)
    )
    ok = np.isfinite(d).all(axis=1) & (np.hypot(d[:, 0], d[:, 1]) > 1e-9)
    if not ok.any():
        return keep
    row_ok, d_ok = row[ok], d[ok]
    # The kernel's binning, in image space (see `_ars_sector_ids`).
    sec = (
        (np.arctan2(d_ok[:, 1], d_ok[:, 0]) / (2 * np.pi) + 1.0) * _ARS_N_SECTORS
    ).astype(np.int64) % _ARS_N_SECTORS
    masks = np.asarray(
        coverage.uncovered_sectors(
            img[ok].astype(np.uint32),
            np.ascontiguousarray(uv[obs[ok]]),
            np.ascontiguousarray(
                (_ARS_EXPAND_DIAM * 2.0 * rad_pt[p_row[ok]]).astype(np.float32)
            ),
            _ARS_N_SECTORS,
        )
    ).astype(np.int64)
    hit = ((masks >> sec) & 1) != 0
    votes = np.bincount(row_ok, minlength=n_c)
    hits = np.bincount(row_ok, weights=hit.astype(np.float64), minlength=n_c)
    return (votes == 0) | (2 * hits >= votes)


def _ars_expand(
    recon,
    images,
    ptr,
    dst,
    need,
    view_dir,
    hu,
    hv,
    n_eff,
    budget,
    rad_pt=None,
    coverage=None,
):
    """Congeal ONE synthetic patch into each EMPTY tangent sector of the
    under-determined points; return ``(extras, spawned, stats)``.

    This is the five-patch machinery aimed rather than blanket: a candidate
    inherits p's frame and view set, is centred ``_ARS_EXPAND_DIAM`` patch widths
    out along the empty sector's bisector, and is congealed (discrete localize ->
    sub-pixel refine) and triangulated exactly as a real track would be.  Only the
    survivors — enough views, a finite triangulation in front of every camera, and
    a reprojection RMS inside ``_ARS_EXPAND_MAX_REPROJ_PX`` — are handed back as
    extra neighbours; the rest simply never existed.

    ``spawned`` carries the survivors WHOLE (parent, position, surviving views,
    refined keypoints, view count, reprojection RMS) for ``_ars_promote``, or is
    ``None`` when the congealing path reports no observations.  The fit reads
    ``extras`` and nothing else, so the two consumers cannot drift apart.

    ``coverage`` (an ``ObservationCoverage`` atlas over the same detection radii as
    the adjacency graph) aims the reach at image area no observation claims: see
    ``_ars_expand_coverage_gate``.  Gated-out candidates cost neither budget nor
    congealing time.

    Aiming and budgeting stay here; the congealing itself is
    ``_ars_expand_congeal``."""
    from sfmtool._sfmtool.patches import CameraViews, ImagePyramidSet

    st = {
        "candidates": int(need.sum()),
        "usable": 0,
        "attempts": 0,
        "kept": 0,
        "points_gaining": 0,
        "budget_dropped": 0,
        "coverage_skipped": 0,
        "secs": 0.0,
    }
    t0 = time.perf_counter()
    n = recon.point_count
    pos = np.asarray(recon.positions, dtype=np.float64)
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    lu, lv = np.linalg.norm(hu, axis=1), np.linalg.norm(hv, axis=1)
    n_obs = np.bincount(ti, minlength=n)
    usable = np.flatnonzero(need & (lu > 0) & (lv > 0) & (n_obs >= 4))
    st["usable"] = len(usable)
    if not len(usable) or budget <= 0:
        st["secs"] = time.perf_counter() - t0
        return {}, None, st

    order = np.argsort(ti, kind="stable")
    starts = np.searchsorted(ti[order], np.arange(n))
    ends = np.searchsorted(ti[order], np.arange(n) + 1)
    e1, e2 = _ars_tangent_basis(view_dir[usable])

    # Most under-determined first, so a budget cut takes the least needy.
    rank = np.argsort(np.nan_to_num(n_eff[usable], nan=0.0), kind="stable")
    cand_p, cand_ctr, groups = [], [], []
    for j in rank.tolist():
        p = int(usable[j])
        nb = dst[ptr[p] : ptr[p + 1]]
        occ = np.zeros(_ARS_N_SECTORS, bool)
        if len(nb):
            d = pos[nb] - pos[p]
            occ[
                _ars_sector_ids(
                    d, np.tile(e1[j], (len(d), 1)), np.tile(e2[j], (len(d), 1))
                )
            ] = True
        free = np.flatnonzero(~occ)
        if not len(free):
            continue
        groups.append((p, len(cand_p), len(cand_p) + len(free)))
        for s in free.tolist():
            th = 2 * np.pi * (s + 0.5) / _ARS_N_SECTORS
            bis = np.cos(th) * e1[j] + np.sin(th) * e2[j]
            cand_p.append(p)
            cand_ctr.append(pos[p] + _ARS_EXPAND_DIAM * lu[p] * bis)

    # Aim the reach at unclaimed image area BEFORE anything is spent on it.
    keep_c = np.ones(len(cand_p), bool)
    if coverage is not None and rad_pt is not None and len(cand_p):
        keep_c = _ars_expand_coverage_gate(
            coverage,
            recon,
            np.asarray(rad_pt, dtype=np.float64),
            np.asarray(cand_p, dtype=np.int64),
            np.asarray(cand_ctr, dtype=np.float64).reshape(-1, 3),
            order,
            starts,
            ends,
        )
    st["coverage_skipped"] = int((~keep_c).sum())

    attempts = []
    for p, lo, hi in groups:
        surv = [k for k in range(lo, hi) if keep_c[k]]
        if not surv:
            continue
        if len(attempts) + len(surv) > budget:
            st["budget_dropped"] += 1
            continue
        attempts.extend((p, cand_ctr[k]) for k in surv)
    st["attempts"] = len(attempts)
    if not attempts:
        st["secs"] = time.perf_counter() - t0
        return {}, None, st

    pidx = np.array([a[0] for a in attempts], np.int64)
    ctr = np.array([a[1] for a in attempts], np.float64)
    view_sets = [
        ii[order[starts[p] : ends[p]]].astype(np.uint32).tolist() for p in pidx.tolist()
    ]
    views = CameraViews(
        list(recon.cameras),
        np.ascontiguousarray(np.asarray(recon.quaternions_wxyz, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(recon.translations, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(recon.camera_indexes).astype(np.uint32)),
    )
    pyramids = ImagePyramidSet(views, images)
    bmp = recon.patch_bitmaps
    resolution = int(np.asarray(bmp).shape[1]) if bmp is not None else 24
    X, good, obs = _ars_expand_congeal(
        recon, views, pyramids, pos, hu, hv, lu, lv, pidx, ctr, view_sets, resolution
    )

    extras = {}
    for k in np.flatnonzero(good).tolist():
        extras.setdefault(int(pidx[k]), []).append(X[k])
    extras = {p: np.asarray(v, dtype=np.float64) for p, v in extras.items()}
    spawned = None
    if obs is not None:
        spawned = dict(obs, parents=pidx, positions=X, good=good)
    st.update(
        kept=int(good.sum()),
        points_gaining=len(extras),
        secs=time.perf_counter() - t0,
    )
    return extras, spawned, st


def _ars_expand_congeal(
    recon, views, pyramids, pos, hu, hv, lu, lv, pidx, ctr, view_sets, resolution
):
    """``_ars_expand``'s congealing via the native candidate-track spawner.

    The construction, the pipeline (one discrete localization, one sub-pixel
    sweep, one triangulation) and the acceptance gates are the spec's
    (`specs/core/patch/candidate-track-spawning.md`), with the whole batch spawned in
    one call.

    The primitive owns the placement: it takes each offset in units of the
    parent's OWN half-extent vectors and builds ``X_c = X_p + du.hu + dv.hv``,
    so a candidate always lands IN the parent's plane.  The aim is computed in
    the tangent basis of the mean viewing direction instead — the plane the
    empty-sector bookkeeping lives in — so the requested centre can carry an
    extra component along the patch normal.  The offsets handed over here are
    that displacement resolved in the frame's own basis — the same in-plane aim
    and the same reach, with the out-of-plane component the contract does not
    admit dropped."""
    from sfmtool._sfmtool.patches import PatchCloud, spawn_candidate_tracks

    cloud = PatchCloud.from_halfvec_arrays(
        np.ascontiguousarray(hu[pidx], dtype=np.float32),
        np.ascontiguousarray(hv[pidx], dtype=np.float32),
        np.ascontiguousarray(pos[pidx]),
    )
    d = ctr - pos[pidx]
    offsets = np.stack(
        [
            (d * hu[pidx]).sum(1) / np.maximum(lu[pidx] ** 2, 1e-30),
            (d * hv[pidx]).sum(1) / np.maximum(lv[pidx] ** 2, 1e-30),
        ],
        axis=1,
    )
    out = spawn_candidate_tracks(
        views,
        pyramids,
        cloud,
        np.arange(len(pidx), dtype=np.uint32),
        np.ascontiguousarray(offsets),
        view_sets,
        resolution=resolution,
        search=_ARS_EXPAND_SEARCH,
        max_shift_px=_ARS_EXPAND_MAX_SHIFT_PX,
        subpixel_sweeps=1,
        min_views=_ARS_EXPAND_MIN_VIEWS,
        max_reproj_rms_px=_ARS_EXPAND_MAX_REPROJ_PX,
    )
    # The primitive reports what it FOUND as well as where it put it — the
    # surviving views and their refined keypoints, CSR over candidates.  The fit
    # needs only the position; promotion needs the whole track (see _ars_promote).
    obs = {
        "n_views": np.asarray(out["n_views"]).astype(np.int64),
        "reproj_rms_px": np.asarray(out["reproj_rms_px"], dtype=np.float64),
        "offsets": np.asarray(out["obs_offsets"]).astype(np.int64),
        "images": np.asarray(out["obs_view_indexes"]).astype(np.int64),
        "keypoints": np.asarray(out["obs_keypoints_xy"], dtype=np.float64).reshape(
            -1, 2
        ),
    }
    return (
        np.asarray(out["positions"], dtype=np.float64),
        np.asarray(out["status"]) == 0,
        obs,
    )


def _dump_promote_candidates(surv, parents, n_views, rms, par_deg):
    """Every expansion SURVIVOR with the three quantities promotion gates on, to
    ``SFMTOOL_ARS_PROMOTE_DUMP``.

    The gates are thresholds on distributions, so they are chosen by reading the
    distributions rather than by taste; this is what makes that possible without
    re-running the congealing.  Off by default and read by no decision."""
    path = os.environ.get("SFMTOOL_ARS_PROMOTE_DUMP")
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("cand\tparent\tviews\treproj_px\tparallax_deg\n")
        for i in range(len(surv)):
            fh.write(
                f"{surv[i]}\t{parents[surv[i]]}\t{n_views[i]}\t"
                f"{rms[i]:.5f}\t{par_deg[i]:.5f}\n"
            )
    print(f"  [promotion candidate dump: {len(surv)} survivors -> {path}]")


def _dump_promote_localizability(rows):
    """Every scored promotion candidate with its localizability reading, to
    ``SFMTOOL_ARS_PROMOTE_LOC_DUMP``.

    Same purpose as ``_dump_promote_candidates``: the threshold is a cut through
    a distribution, so it gets chosen by reading the distribution.  Off by
    default and read by no decision.

    Each row also carries the candidate's FIRST observation (image index and
    keypoint).  Point indexes do not survive the downstream gauntlet, so that
    observation is what lets a shipped point be traced back to its congeal-time
    score — the same keypoint-signature matching the artifact probes use."""
    path = os.environ.get("SFMTOOL_ARS_PROMOTE_LOC_DUMP")
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "point\tparent\tsigma_grid\tsigma_px\tlam1\tlam2\tconsensus\t"
            "image0\tu0\tv0\n"
        )
        for r in rows:
            fh.write(
                f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t"
                f"{r[4]:.6g}\t{r[5]:.6g}\t{int(r[6])}\t"
                f"{int(r[7])}\t{r[8]:.3f}\t{r[9]:.3f}\n"
            )
    print(f"  [promotion localizability dump: {len(rows)} candidates -> {path}]")


def _promote_localizability(recon, images, n_before, parents, det_size, mutated, uid):
    """Drop the just-promoted points whose CONGEALED patch cannot pin a keypoint.
    Returns ``(recon, stats, det_size, mutated, uid)``.

    The promoted block is ``[n_before, point_count)``, carrying the tracks the
    spawner congealed.  Their consensus patches are fused here — the render-only
    mode of the same sub-pixel refiner (``sweeps=0``: every seed kept, no
    Gauss-Newton step), at the keypoints they were promoted with — and scored
    with the SAME scorer the embed stage culls on
    (``PatchCloud.score_localizability``, ``specs/core/patch/patch-localizability.md``):
    the noise-normalized structure tensor's weak axis, as a keypoint positional
    uncertainty ``sigma_pos`` in patch-grid px, plus the ANISOTROPY bound the
    same criterion carries everywhere (see LOC_MAX_ANISO — a high-contrast
    streak passes ``tau`` on absolute weak-axis curvature while being a line).
    Failing either, the point is dropped before anything downstream ever sees it.

    Why here and not at congeal time: the reading needs the fused consensus, and
    the spawn primitive returns positions and keypoints, not textures.  Rendering
    the block after the append costs one render-only pass over the promoted
    points alone and puts the gate exactly where its casualties are free — ahead
    of the re-localization, the eviction, the BA and every cull.

    The fused bitmaps are NOT written back: a promoted point is marked mutated
    and its texture is re-fused at the end of the finalization from the track it
    actually ends up with, which is a different (and later) track than this one.

    ``sigma_pos`` is NaN where the consensus is empty — no texture could be
    fused at all — and ``NaN > tau`` is False, so such a point is KEPT.  That is
    the embed-time convention (`_cull_by_localizability`): the score is evidence
    of un-localizability, and its absence is not evidence.  Counted separately so
    the population is visible rather than assumed."""
    st = {
        "scored": 0,
        "rejected": 0,
        "rejected_aniso": 0,
        "no_consensus": 0,
        "med": 0.0,
        "secs": 0.0,
    }
    n = recon.point_count
    m = n - n_before
    tau = _ARS_PROMOTE_MAX_SIGMA_GRID
    aniso = LOC_MAX_ANISO
    dumping = bool(os.environ.get("SFMTOOL_ARS_PROMOTE_LOC_DUMP"))
    if m <= 0 or (tau <= 0 and aniso <= 0 and not dumping):
        return recon, st, det_size, mutated, uid
    from sfmtool._embed_patches import _localizations_from_recon, _refine_subpixel

    t0 = time.perf_counter()
    bmp = recon.patch_bitmaps
    cloud = recon.patches
    if bmp is None or cloud is None:
        return recon, st, det_size, mutated, uid
    # The stored block defines the grid the rest of the cloud is on, and the
    # score is in that grid's px — scoring on any other resolution would not be
    # the same quantity the embed cull thresholds.
    resolution = int(np.asarray(bmp).shape[1])
    locs = [
        loc
        for loc in _localizations_from_recon(recon)
        if int(loc["point_index"]) >= n_before
    ]
    if not locs:
        return recon, st, det_size, mutated, uid
    _l, bitmaps, valid = _refine_subpixel(
        cloud, recon, images, locs, sweeps=0, resolution=resolution, render_bitmaps=True
    )
    if bitmaps is None:
        return recon, st, det_size, mutated, uid
    scored = cloud.score_localizability(recon, np.ascontiguousarray(bitmaps))
    sigma = np.asarray(scored["sigma_pos_grid"], dtype=float)[n_before:]
    st["scored"] = m
    st["no_consensus"] = int((~np.asarray(valid, bool)[n_before:]).sum())
    fin = np.isfinite(sigma)
    if fin.any():
        st["med"] = float(np.median(sigma[fin]))
    s_px = np.asarray(scored["sigma_pos_px"], dtype=float)
    lam1 = np.asarray(scored["lam1"], dtype=float)
    lam2 = np.asarray(scored["lam2"], dtype=float)
    ok = np.asarray(valid, bool)
    first = {int(loc["point_index"]): loc for loc in locs}
    _dump_promote_localizability(
        [
            (
                n_before + k,
                int(parents[k]),
                float(sigma[k]),
                float(s_px[n_before + k]),
                float(lam1[n_before + k]),
                float(lam2[n_before + k]),
                bool(ok[n_before + k]),
                int(np.asarray(first[n_before + k]["views"])[0]),
                float(
                    np.asarray(first[n_before + k]["keypoints"]).reshape(-1, 2)[0, 0]
                ),
                float(
                    np.asarray(first[n_before + k]["keypoints"]).reshape(-1, 2)[0, 1]
                ),
            )
            for k in range(m)
            if n_before + k in first
        ]
    )
    if tau <= 0 and aniso <= 0:
        st["secs"] = time.perf_counter() - t0
        return recon, st, det_size, mutated, uid
    bad_sigma, bad_aniso = _localizability_verdict(
        sigma, lam1[n_before:], lam2[n_before:], tau, aniso
    )
    reject = bad_sigma | bad_aniso  # NaN -> False -> kept, on both halves
    st["rejected"] = int(reject.sum())
    st["rejected_aniso"] = int((bad_aniso & ~bad_sigma).sum())
    st["keep"] = ~reject
    st["secs"] = time.perf_counter() - t0
    if not reject.any():
        return recon, st, det_size, mutated, uid
    keep = np.ones(n, bool)
    keep[n_before:] = ~reject
    keep = np.ascontiguousarray(keep)
    return (
        recon.filter_points_by_mask(keep),
        st,
        np.asarray(det_size)[keep],
        np.asarray(mutated)[keep],
        np.asarray(uid)[keep],
    )


def _ars_promote(recon, images, spawned, hu, hv, det_size, mutated, uid):
    """Promote the well-congealed expansion survivors into REAL points.  Returns
    ``(recon, stats, det_size, mutated, uid)``.

    A promoted candidate arrives as an ordinary track — its own surviving views,
    its own sub-pixel-refined keypoints, its own triangulated position — appended
    to the cloud with the carried per-point bookkeeping (``det_size``,
    ``mutated``, ``uid``) extended to match.  Nothing downstream is told it is
    synthetic, which is the whole point: the seed finalization re-localizes
    keypoints, evicts disowned track views, adjusts, re-classifies at infinity
    and culls after this site, so a promoted point earns its place by surviving
    the same passes as everything else, or it does not survive them.

    WHAT EACH FIELD IS AND WHY.

    * Position — the spawner's own triangulation of the refined keypoints (the
      same number the fit was handed).  Written homogeneous with ``w = 1``: a
      candidate is finite by construction (see the spawning spec), and the
      infinity classifier downstream gets to say otherwise.
    * Track — the surviving ``(image, keypoint)`` rows verbatim.  These are real
      photometric observations of real image content; they are what the point IS.
    * Frame — the PARENT's frame at the candidate's own centre.  Under
      ``ARS_LATE_FIT`` that is the parent's EMBED-stage frame, which is the frame
      the candidate was actually congealed under, and it is provisional: the late
      fit gives the survivor its own plane from its own neighbourhood.  With the
      late fit off it is the parent's post-ARS frame instead, and it is what the
      point SHIPS — which is the birth defect the late fit exists to remove, a
      grazing parent spawning a grazing twin beside it that nothing ever refits.
    * Normal / confidence — the frame's own ``normalize(u x v)``, i.e. the
      parent's, at confidence ZERO.  A promoted point has no adjacency support of
      its own at birth: it was created after the graph was built and no fit has
      ever been run on it.  Confidence 0 is the honest statement of that, and it
      is the marker the culls and the viewer already read.  Under the late fit it
      is also temporary — a survivor is fitted with everything else at the end.
    * Bitmap — zero-filled and marked MUTATED, so the end-of-finalization re-fuse
      renders it from the track it actually ends up with.  Rendering it here
      would bake a texture from a track the eviction and the BA have not touched
      yet; nothing between here and there reads a STORED bitmap.
    * ``det_size`` — the PARENT's detection half-extent.  A promoted point has no
      SIFT detection to read a scale from, and it is not free to have none: the
      duplicate collapse and the contained-inconsistent cull both measure image
      footprints in this radius, and a zero would exempt it from the collapse's
      overlap test while a large one would let it swallow real points.  The
      parent's is the defensible answer — the candidate is a patch of the
      parent's own frame, congealed at the parent's scale and one patch diameter
      away, so it claims the same image footprint the parent does.
    * ``uid`` — a FRESH block from the trace's counter, never the parent's.  The
      trace's identity is per-point history; a promoted point has none.

    GATES.  Beyond the spawn primitive's own (see ``_ARS_PROMOTE_MIN_VIEWS`` /
    ``_ARS_PROMOTE_MAX_REPROJ_PX`` / ``_ARS_PROMOTE_MIN_PARALLAX_DEG``), plus the
    photometric one those three cannot be: the congealed patch has to be able to
    pin a keypoint (``_promote_localizability``, ``_ARS_PROMOTE_MAX_SIGMA_GRID``),
    which is applied on the appended block because the reading needs the fused
    consensus.  Fewer candidates are promoted than the fit uses, and that
    asymmetry is deliberate: the fit's bar is "informative about a direction",
    promotion's is "defensible as a landmark"."""
    st = {
        "survivors": 0,
        "no_obs": spawned is None,
        "cut_views": 0,
        "cut_reproj": 0,
        "cut_parallax": 0,
        "cut_localizability": 0,
        "loc": {},
        "promoted": 0,
        "views_med": 0.0,
        "parallax_med": 0.0,
        # First uid of the promoted block.  The uids are a fresh contiguous
        # range, so "uid >= this" identifies the promoted points at any later
        # stage without any extra bookkeeping — which is how the casualty
        # accounting through the downstream gauntlet is read off.
        "uid_base": -1,
        "secs": 0.0,
    }
    if spawned is None or not ARS_PROMOTE:
        return recon, st, det_size, mutated, uid
    from sfmtool._sfmtool.patches import PatchCloud

    t0 = time.perf_counter()
    good = np.asarray(spawned["good"], bool)
    surv = np.flatnonzero(good)
    st["survivors"] = len(surv)
    if not len(surv):
        st["secs"] = time.perf_counter() - t0
        return recon, st, det_size, mutated, uid

    off = np.asarray(spawned["offsets"], dtype=np.int64)
    o_img = np.asarray(spawned["images"], dtype=np.int64)
    o_kp = np.asarray(spawned["keypoints"], dtype=np.float64)
    n_v = np.asarray(spawned["n_views"], dtype=np.int64)
    rms = np.asarray(spawned["reproj_rms_px"], dtype=np.float64)
    X = np.asarray(spawned["positions"], dtype=np.float64)
    par = np.asarray(spawned["parents"], dtype=np.int64)

    # PARALLAX, from the candidate's own observation rays: the widest angle
    # between any two of them — the parallax a triangulation has to work with.
    # Measured on EVERY survivor, not just on what the cheaper gates left, so the
    # calibration dump shows the whole population each gate is choosing from.
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    cam = recon.cameras[0]
    rows = np.concatenate([np.arange(off[k], off[k + 1]) for k in surv.tolist()])
    cnt = (off[surv + 1] - off[surv]).astype(np.int64)
    d_loc = cam.pixel_to_ray_batch(np.ascontiguousarray(o_kp[rows]))
    rays = np.einsum("nji,nj->ni", rots[o_img[rows]], d_loc)
    rays /= np.maximum(np.linalg.norm(rays, axis=1), 1e-300)[:, None]
    grp = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    par_deg = np.zeros(len(surv))
    for k in range(len(surv)):
        r = rays[grp[k] : grp[k + 1]]
        par_deg[k] = np.degrees(np.arccos(np.clip(float((r @ r.T).min()), -1.0, 1.0)))

    # Gates in priority order, each counted against what the previous ones left,
    # so the casualties add up to the survivors.
    ok_v = n_v[surv] >= _ARS_PROMOTE_MIN_VIEWS
    ok_r = rms[surv] <= _ARS_PROMOTE_MAX_REPROJ_PX
    ok_p = par_deg >= _ARS_PROMOTE_MIN_PARALLAX_DEG
    st["cut_views"] = int((~ok_v).sum())
    st["cut_reproj"] = int((ok_v & ~ok_r).sum())
    st["cut_parallax"] = int((ok_v & ok_r & ~ok_p).sum())
    _dump_promote_candidates(surv, par, n_v[surv], rms[surv], par_deg)
    keep = ok_v & ok_r & ok_p
    sel = surv[keep]
    st["promoted"] = len(sel)
    if len(sel):
        st["views_med"] = float(np.median(n_v[sel]))
        st["parallax_med"] = float(np.median(par_deg[keep]))
    if not len(sel):
        st["secs"] = time.perf_counter() - t0
        return recon, st, det_size, mutated, uid

    n = recon.point_count
    m = len(sel)
    p_of = par[sel]
    rows = np.concatenate([np.arange(off[k], off[k + 1]) for k in sel.tolist()])
    cnt = (off[sel + 1] - off[sel]).astype(np.int64)
    new_pt = np.repeat(np.arange(n, n + m, dtype=np.int64), cnt)

    # Colours the way the seed writer takes them — the source image at the
    # observation — read at full resolution here (the images are already in hand)
    # and MEDIANed over the track, which is the same fusion the consensus bitmap
    # applies to the same pixels.
    col = np.zeros((m, 3), np.uint8)
    samp = np.zeros((len(rows), 3), np.float64)
    for g in np.unique(o_img[rows]).tolist():
        r = np.flatnonzero(o_img[rows] == g)
        im = np.asarray(images[g])
        yy = np.clip(np.rint(o_kp[rows[r], 1]).astype(np.int64), 0, im.shape[0] - 1)
        xx = np.clip(np.rint(o_kp[rows[r], 0]).astype(np.int64), 0, im.shape[1] - 1)
        samp[r] = im[yy, xx, :3]
    for k in range(m):
        lo, hi = (
            int(np.searchsorted(new_pt, n + k)),
            int(np.searchsorted(new_pt, n + k + 1)),
        )
        col[k] = np.clip(np.median(samp[lo:hi], axis=0), 0, 255).astype(np.uint8)

    pos_new = np.vstack(
        [
            np.asarray(recon.positions_xyzw, dtype=np.float64),
            np.hstack([X[sel], np.ones((m, 1))]),
        ]
    )
    hu_new = np.vstack([hu, hu[p_of]])
    hv_new = np.vstack([hv, hv[p_of]])
    nrm = np.cross(hu[p_of], hv[p_of])
    nl = np.linalg.norm(nrm, axis=1)
    nrm = np.where((nl > 0)[:, None], nrm / np.maximum(nl, 1e-30)[:, None], 0.0)
    normals = np.vstack(
        [np.asarray(recon.normals, dtype=np.float32), nrm.astype(np.float32)]
    )
    # The channel is absent until something writes it, and under ARS_LATE_FIT
    # nothing has by the time promotion runs — so the None test has to happen on
    # the attribute, before np.asarray turns it into a 0-d object array.
    conf = recon.normal_confidence
    conf = np.zeros(n, np.uint8) if conf is None else np.asarray(conf, np.uint8)
    conf_new = np.concatenate([conf, np.zeros(m, np.uint8)])

    ti = np.concatenate(
        [np.asarray(recon.track_point_indexes).astype(np.int64), new_pt]
    )
    ii = np.concatenate(
        [np.asarray(recon.track_image_indexes).astype(np.int64), o_img[rows]]
    )
    uv = np.vstack([np.asarray(recon.keypoints_xy, dtype=np.float64), o_kp[rows]])
    # The observation list is CSR by point (`observation_offsets` is a prefix sum
    # over the per-point counts), so it has to come out grouped.  Appending the
    # new block already leaves it so — new points carry the highest indexes — but
    # the sort makes that a property of this code rather than of its caller.
    order = np.lexsort((ii, ti))
    ti, ii, uv = ti[order], ii[order], uv[order]

    changes = {
        "positions": np.ascontiguousarray(pos_new),
        "colors": np.ascontiguousarray(np.vstack([np.asarray(recon.colors), col])),
        "normals": np.ascontiguousarray(normals),
        "normal_confidence": np.ascontiguousarray(conf_new),
        "patches": PatchCloud.from_halfvec_arrays(
            np.ascontiguousarray(hu_new, dtype=np.float32),
            np.ascontiguousarray(hv_new, dtype=np.float32),
            np.ascontiguousarray(pos_new[:, :3]),
        ),
        "track_point_indexes": np.ascontiguousarray(ti.astype(np.uint32)),
        "track_image_indexes": np.ascontiguousarray(ii.astype(np.uint32)),
        "track_feature_indexes": np.zeros(len(ti), np.uint32),  # unused: embedded
        "keypoints_xy": np.ascontiguousarray(uv.astype(np.float32)),
    }
    bmp = recon.patch_bitmaps
    if bmp is not None:
        bmp = np.asarray(bmp, dtype=np.uint8)
        changes["patch_bitmaps"] = np.ascontiguousarray(
            np.concatenate([bmp, np.zeros((m, *bmp.shape[1:]), np.uint8)])
        )
    out = recon.clone_with_changes(**changes)
    out = _refresh_errors(out)  # the appended rows arrive with a zero error
    stage_dump("ars-promote-append", out)

    det_size = np.concatenate([np.asarray(det_size, dtype=np.float64), det_size[p_of]])
    mutated = np.concatenate([np.asarray(mutated, bool), np.ones(m, bool)])
    new_uid = _seed_point_uids(m)
    st["uid_base"] = int(new_uid[0])
    uid = np.concatenate([np.asarray(uid, dtype=np.int64), new_uid])

    # LOCALIZABILITY, the one gate that reads the congealed IMAGE CONTENT rather
    # than the geometry (see _promote_localizability).  It runs on the appended
    # block because the reading needs the fused consensus; a rejected candidate
    # is removed here, before anything downstream has seen it.
    out, l_st, det_size, mutated, uid = _promote_localizability(
        out, images, n, p_of, det_size, mutated, uid
    )
    st["loc"] = l_st
    st["cut_localizability"] = l_st["rejected"]
    st["promoted"] -= l_st["rejected"]
    # The medians describe what was actually promoted, so they answer to the
    # last gate as well as to the first three.
    alive = np.asarray(l_st.get("keep", np.ones(m, bool)), bool)
    if alive.any():
        st["views_med"] = float(np.median(n_v[sel][alive]))
        st["parallax_med"] = float(np.median(par_deg[keep][alive]))
    st["secs"] = time.perf_counter() - t0
    return out, st, det_size, mutated, uid


def _print_promote(st):
    """One line of promotion evidence."""
    if st.get("no_obs"):
        print("    promotion: unavailable (congealing path reports no observations)")
        return
    s = max(st["survivors"], 1)
    print(
        f"    promotion: {st['promoted']}/{st['survivors']} survivors promoted to "
        f"real tracks ({100 * st['promoted'] / s:.1f}%; cut "
        f"{st['cut_views']} views + {st['cut_reproj']} reproj + "
        f"{st['cut_parallax']} parallax + "
        f"{st.get('cut_localizability', 0)} localizability), median "
        f"{st['views_med']:.0f} views / "
        f"{st['parallax_med']:.2f} deg parallax; {st['secs']:.1f}s"
    )
    loc = st.get("loc") or {}
    if loc.get("scored"):
        print(
            f"      localizability: {loc['scored']} congealed patches scored, "
            f"{loc['rejected']} rejected (tau={_ARS_PROMOTE_MAX_SIGMA_GRID:.2f} "
            f"grid px, of which {loc.get('rejected_aniso', 0)} on the "
            f"lam1/lam2 > {LOC_MAX_ANISO:g} bound alone) "
            f"({loc['no_consensus']} with no consensus, kept), median sigma "
            f"{loc['med']:.3f}; {loc['secs']:.1f}s"
        )


def _ars_normals(recon, images, det_size, refine_radius, mutated, uid, mode="all"):
    """Write the ARS normals and the frames that agree with them.  Returns
    ``(recon, stats, mutated, det_size, uid)``.

    ``mode`` splits the stage across the downstream gauntlet (see
    ``ARS_LATE_FIT``); it is ``"all"`` — everything at one site, the pre-late-fit
    order — unless the caller says otherwise.

    * ``"expand"`` — the EARLY half.  Builds the graph and runs the fit as
      SCRATCH, purely to aim the expansion (the empty-sector bookkeeping needs
      the neighbour directions, the budget order needs ``n_eff``), congeals the
      candidates and promotes the survivors.  Writes no normal, no frame and no
      confidence: promoted points arrive wearing their parent's EMBED-stage
      frame, which is the frame they were congealed under.
    * ``"fit"`` — the LATE half.  Rebuilds the graph over whatever cloud the
      gauntlet left and writes the normals, frames and confidences from that one
      fit.  No expansion, no promotion, no ``extras``.

    The saved normal of every finite patched point comes from here; the
    photometric estimate the embed pipeline left in the patch frames is REPLACED,
    not blended with.  Writing a normal means ROTATING the frame onto its plane
    the way the core's own ``refine_normals`` does — the stored ``v_axis``
    reprojected onto the new plane, ``u = v x n``, both half-extents kept, so the
    frame stays SQUARE and ``normalize(u x v)`` is the normal that was written.
    Normals, frames and bitmaps go back in ONE ``clone_with_changes``: passing
    ``patches=`` without ``patch_bitmaps=`` would clear the consensus textures.

    Points at infinity keep everything they have — their frame is the tangent-plane
    one the format's infinity convention fixes, not a refined normal — as do finite
    points that carry no patch frame to rotate.

    The expansion stage's survivors are then PROMOTED into the cloud as real
    points (``_ars_promote``, ``SFMTOOL_ARS_PROMOTE``), which is why the carried
    per-point arrays come back from here: the cloud this returns can be longer
    than the one it was handed."""
    from sfmtool._sfmtool.patches import PatchCloud

    st = {
        "mode": mode,
        "eligible": 0,
        "geom": 0,
        "expand": 0,
        "weak": 0,
        "fallback": 0,
        "no_radius": 0,
        "turn_med": 0.0,
        "turn_p90": 0.0,
        "mutated": 0,
        "graph": {},
        "expansion": {},
        "promotion": {},
        "secs": 0.0,
    }
    n = recon.point_count
    if not ARS_ENABLED or n == 0 or recon.patches is None:
        return recon, st, mutated, det_size, uid
    t0 = time.perf_counter()
    hu, hv = _patch_halfvecs(recon)
    lu, lv = np.linalg.norm(hu, axis=1), np.linalg.norm(hv, axis=1)
    inf = np.asarray(recon.point_is_at_infinity)
    elig = (~inf) & (lu > 0) & (lv > 0)
    st["eligible"] = int(elig.sum())
    if not elig.any():
        st["secs"] = time.perf_counter() - t0
        return recon, st, mutated, det_size, uid

    # Detection radius, converted into the unit ARS_B_MAX was calibrated in.  A
    # point whose cluster carried no detection size falls back to the cloud's
    # median rather than dropping out of the graph — it still has neighbours, and
    # the annulus only needs a length scale, not this point's own.
    rad = np.asarray(det_size, dtype=np.float64) * (
        _ARS_RADIUS_MULT / max(float(refine_radius), 1e-9)
    )
    st["no_radius"] = int((elig & (rad <= 0)).sum())
    if (rad > 0).any():
        rad = np.where(rad > 0, rad, float(np.median(rad[rad > 0])))

    pos = np.asarray(recon.positions, dtype=np.float64)
    ti = np.asarray(recon.track_point_indexes).astype(np.int64)
    ii = np.asarray(recon.track_image_indexes).astype(np.int64)
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    rots = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    cams = -np.einsum(
        "nij,ni->nj", rots, np.asarray(recon.translations, dtype=np.float64)
    )
    # Mean observing direction: the normal-free reference the tangent basis, the
    # sign convention and the fronto-parallel fallback all answer to.
    cnt = np.bincount(ti, minlength=n).astype(np.float64)
    acc = np.stack(
        [np.bincount(ti, weights=cams[ii][:, k], minlength=n) for k in range(3)], axis=1
    )
    view = np.zeros((n, 3))
    seen = cnt > 0
    view[seen] = acc[seen] / cnt[seen, None] - pos[seen]
    vn = np.linalg.norm(view, axis=1)
    view[vn > 1e-12] /= vn[vn > 1e-12, None]

    ptr, dst, _sv, g_st = _ars_edges(recon, rad)
    st["graph"] = g_st
    nrm, diag, det = _ars_fit(pos, view, ptr, dst, elig)
    st["geom"] = int((elig & det).sum())

    # Only what geometry cannot serve goes to the expansion stage.
    need = elig & ~det
    extras, spawned, x_st = ({}, None, {})
    if mode != "fit" and need.any() and ARS_EXPAND_BUDGET > 0:
        # What the images already account for, in the SAME detection-radius
        # footprints the adjacency graph vets pairs in — built once for the whole
        # stage, queried in batch by the aiming gate.
        coverage = None
        if ARS_EXPAND_COVERAGE:
            from sfmtool._sfmtool.analysis import ObservationCoverage

            wh = np.array(
                [[int(c.width), int(c.height)] for c in recon.cameras], np.uint32
            )
            coverage = ObservationCoverage(
                np.ascontiguousarray(wh[np.asarray(recon.camera_indexes)]),
                np.ascontiguousarray(ii.astype(np.uint32)),
                np.ascontiguousarray(np.asarray(recon.keypoints_xy, dtype=np.float64)),
                np.ascontiguousarray(rad[ti].astype(np.float32)),
                cell_px=4,
            )
        extras, spawned, x_st = _ars_expand(
            recon,
            images,
            ptr,
            dst,
            need,
            view,
            hu,
            hv,
            diag["n_eff"],
            int(ARS_EXPAND_BUDGET * st["eligible"]),
            rad_pt=rad,
            coverage=coverage,
        )
    st["expansion"] = x_st
    if mode == "expand":
        # Nothing the scratch fit produced reaches the cloud.  The extras re-fit
        # below only ever served the write, and the write belongs to the late
        # site; the promoted tracks are this half's entire output, and they wear
        # the parent's embed-stage frame — the one they were congealed under.
        out, p_st, det_size, mutated, uid = _ars_promote(
            recon, images, spawned, hu, hv, det_size, mutated, uid
        )
        st["promotion"] = p_st
        _dump_promote_graph(out, ptr, det_size, refine_radius, np.zeros(n, np.uint8), n)
        st["secs"] = time.perf_counter() - t0
        return out, st, mutated, det_size, uid
    if extras:
        touched = np.zeros(n, bool)
        touched[np.fromiter(extras, np.int64, len(extras))] = True
        n2, d2, det2 = _ars_fit(pos, view, ptr, dst, touched, extras)
        upd = touched & np.isfinite(n2).all(axis=1)
        nrm[upd] = n2[upd]
        det[upd] = det2[upd]
        for k in diag:
            diag[k][upd] = d2[k][upd]
        st["expand"] = int((need & det).sum())

    # Fronto-parallel where even that failed: the honest "no information" answer,
    # counted rather than hidden behind a silently-kept photometric normal.
    fitted = np.isfinite(nrm).all(axis=1)
    fallback = elig & ~fitted
    nrm[fallback] = view[fallback]
    st["fallback"] = int(fallback.sum())
    st["weak"] = st["eligible"] - st["geom"] - st["expand"] - st["fallback"]

    # Binary normal confidence: 1 where the determinacy predicate held (the
    # geometry or expansion paths), 0 for weak fits, the fronto-parallel
    # fallback, and points that carry no fitted normal at all (infinity points,
    # frameless points).  A fallback normal LOOKS as definite as a fitted one in
    # the saved arrays; this is the marker that says it is not.  Binary for now
    # — the slot widens to a continuous score when something downstream can
    # grade on it.  Persisted through the format's per-point channel below
    # ({0,1} -> {0,255} on its monotonic scale); the native passes downstream
    # keep the column aligned (the culls row-select it, the infinity classifier
    # zeroes demoted rows).
    conf = np.zeros(n, dtype=np.uint8)
    conf[elig & det] = 1
    st["confidence"] = conf

    # Rotate each frame onto its new plane (the core's `repose_patch` rule: the
    # stored v_axis reprojected onto the plane of n, u = v x n, extents kept).
    apply = elig & np.isfinite(nrm).all(axis=1)
    nl = np.linalg.norm(nrm[apply], axis=1)
    nrm[apply] /= np.maximum(nl, 1e-30)[:, None]
    v_old = np.zeros((n, 3))
    v_old[elig] = hv[elig] / lv[elig, None]
    proj = v_old - (v_old * nrm).sum(1)[:, None] * nrm
    pn = np.linalg.norm(proj, axis=1)
    # Degenerate only when the new normal is parallel to the stored v_axis; the
    # deterministic axis fallback matches OrientedPatch::from_center_normal.
    a = np.where((np.abs(nrm[:, 0]) < 0.9)[:, None], (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    alt = a - (a * nrm).sum(1)[:, None] * nrm
    alt /= np.maximum(np.linalg.norm(alt, axis=1), 1e-30)[:, None]
    v_new = np.where((pn > 1e-9)[:, None], proj / np.maximum(pn, 1e-30)[:, None], alt)
    u_new = np.cross(v_new, nrm)
    hu_out = np.where(apply[:, None], u_new * lu[:, None], hu)
    hv_out = np.where(apply[:, None], v_new * lv[:, None], hv)

    old_n = np.cross(hu, hv)
    on = np.linalg.norm(old_n, axis=1)
    turn = np.zeros(n)
    live = apply & (on > 0)
    turn[live] = np.degrees(
        np.arccos(np.clip((old_n[live] / on[live, None] * nrm[live]).sum(1), -1.0, 1.0))
    )
    tv_ = turn[apply]
    if len(tv_):
        st["turn_med"] = float(np.median(tv_))
        st["turn_p90"] = float(np.percentile(tv_, 90))

    normals = np.asarray(recon.normals, dtype=np.float32).copy()
    normals[apply] = nrm[apply].astype(np.float32)
    changes = {
        "normals": np.ascontiguousarray(normals),
        "patches": PatchCloud.from_halfvec_arrays(
            np.ascontiguousarray(hu_out, dtype=np.float32),
            np.ascontiguousarray(hv_out, dtype=np.float32),
            np.ascontiguousarray(pos),
        ),
    }
    bmp = recon.patch_bitmaps
    if bmp is not None:
        changes["patch_bitmaps"] = np.ascontiguousarray(bmp)
    changes["normal_confidence"] = np.ascontiguousarray(
        np.where(conf == 1, np.uint8(255), np.uint8(0))
    )
    out = recon.clone_with_changes(**changes)

    # A turned frame invalidates the consensus bitmap fused through the old one.
    mutated = mutated.copy()
    mutated[apply & (turn > _ARS_MUTATE_MIN_DEG)] = True
    st["mutated"] = int((apply & (turn > _ARS_MUTATE_MIN_DEG)).sum())

    # DENSIFICATION BY PROMOTION.  Last, on the reconstruction the normals have
    # already been written into, so a promoted point inherits the frame its
    # parent SHIPS rather than the one the expansion congealed against — and
    # early enough that the whole downstream gauntlet still runs over it.
    # (Under ``mode="fit"`` this half already ran, at the early site.)
    if mode != "fit":
        out, p_st, det_size, mutated, uid = _ars_promote(
            out, images, spawned, hu_out, hv_out, det_size, mutated, uid
        )
        st["promotion"] = p_st
        _dump_promote_graph(out, ptr, det_size, refine_radius, conf, n)
    st["secs"] = time.perf_counter() - t0
    return out, st, mutated, det_size, uid


def _dump_promote_graph(recon, ptr, det_size, refine_radius, conf, n_before):
    """Per-point adjacency degree BEFORE and AFTER promotion, to
    ``SFMTOOL_ARS_PROMOTE_GRAPH_DUMP``.

    The claim promotion rests on is that it turns invisible scaffolding into
    support the graph can see, so the graph is rebuilt over the extended cloud
    and its degrees are put beside the ones the fit actually ran on.  A point
    whose ``before`` degree was 0 and whose ``after`` degree is not is a point
    whose determinacy stopped being self-referential.  Off by default; read by
    no decision."""
    path = os.environ.get("SFMTOOL_ARS_PROMOTE_GRAPH_DUMP")
    if not path or recon.point_count == n_before:
        return
    rad = np.asarray(det_size, dtype=np.float64) * (
        _ARS_RADIUS_MULT / max(float(refine_radius), 1e-9)
    )
    if (rad > 0).any():
        rad = np.where(rad > 0, rad, float(np.median(rad[rad > 0])))
    ptr2, _dst2, _sv2, _st2 = _ars_edges(recon, rad)
    deg_before = np.diff(np.asarray(ptr))
    deg_after = np.diff(np.asarray(ptr2))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("pt\tpromoted\tconf\tdeg_before\tdeg_after\n")
        for p in range(recon.point_count):
            b = int(deg_before[p]) if p < n_before else -1
            c = int(conf[p]) if p < n_before else 0
            fh.write(f"{p}\t{int(p >= n_before)}\t{c}\t{b}\t{int(deg_after[p])}\n")
    print(
        f"  [promotion graph dump: {n_before} -> {recon.point_count} points -> {path}]"
    )


def _print_ars(tag, st):
    """One block of ARS evidence."""
    g, x = st.get("graph") or {}, st.get("expansion") or {}
    mode = st.get("mode", "all")
    e = max(st["eligible"], 1)
    # The native builder reports no candidate count; say nothing rather than 0.
    cand = g.get("cand", 0)
    from_cand = f" from {cand} candidates" if cand else ""
    what = {"expand": "ARS aiming", "fit": "ARS normals"}.get(mode, "ARS normals")
    resolved = (
        f"resolved {st['geom']} geometry ({100 * st['geom'] / e:.1f}%) / "
        f"{st['eligible'] - st['geom']} under-determined (scratch fit, aiming "
        f"only — nothing written)"
        if mode == "expand"
        else (
            f"resolved {st['geom']} geometry ({100 * st['geom'] / e:.1f}%) / "
            f"{st['expand']} expansion / {st['weak']} weak / {st['fallback']} "
            f"fronto-parallel fallback ({100 * st['fallback'] / e:.1f}%)"
        )
    )
    print(
        f"  {what} ({tag}): {st['eligible']} finite patched points; "
        f"adjacency {g.get('edges', 0)} edges{from_cand} "
        f"(degree median {g.get('deg_med', 0):.0f}) in {g.get('secs', 0.0):.1f}s; "
        f"{resolved}"
    )
    if x:
        print(
            f"    expansion: {x['usable']}/{x['candidates']} under-determined "
            f"points expandable, {x['attempts']} synthetic patches congealed "
            f"({x.get('coverage_skipped', 0)} candidates skipped as already "
            f"covered, {x['budget_dropped']} points cut by the budget), "
            f"{x['kept']} kept, "
            f"{x['points_gaining']} points gained a neighbour; {x['secs']:.1f}s"
        )
    if x and x.get("kept") and st.get("promotion"):
        _print_promote(st["promotion"])
    if mode == "expand":
        print(
            f"    no normal, frame or confidence written here; {st['no_radius']} "
            f"points with no detection size; {st['secs']:.1f}s"
        )
        return
    conf = st.get("confidence")
    n1 = int(conf.sum()) if conf is not None else 0
    print(
        f"    frame turn from the photometric normal: median {st['turn_med']:.2f} "
        f"deg, p90 {st['turn_p90']:.2f} deg; {st['mutated']} frames past the "
        f"{_ARS_MUTATE_MIN_DEG:g} deg re-render gate; {st['no_radius']} points "
        f"with no detection size; normal confidence 1 on {n1}/{e} "
        f"({100 * n1 / e:.1f}%); {st['secs']:.1f}s"
    )


def _finalize_seed(
    data,
    active_cl,
    rvec,
    tvec,
    f,
    pts,
    posed,
    f_final=None,
    arbitrate_vote=None,
    arbitrate_center=None,
    arbitrate_band=None,
    flags=None,
):
    """Full photometric seed finalization -> a bitmap-bearing embedded recon.

    Builds the seed as embedded_patches (surfel frames from the cluster warps),
    runs the whole embed-patches photometric pipeline over the posed frames —
    refine normals, EXPAND the view set (select_views admits frames the matcher
    missed), CONGEAL + SUB-PIXEL-refine the per-view keypoints (localize +
    refine_keypoints), rendering the consensus patch bitmaps — CULLS points left
    at <= 2 views (a real point extends to a third; a repeated-pattern match
    cannot), and bundle-adjusts with the NATIVE core BA on the refined keypoints
    (canonical frame, so no convention round-trip). Returns the finalized
    reconstruction with bitmaps, ready to save.

    Before the BA the depth-unconstrained tracks are RECLASSIFIED AT INFINITY
    (``classify_points_at_infinity``): a far-field feature has no depth for the
    triangulation to find, and forcing one to a finite position parks it at an
    invented depth that then pollutes the adjustment.  At ``w = 0`` it keeps its
    rotational (and focal) constraint without claiming a distance — the native
    BA carries such a point as a direction.

    ``arbitrate_center`` / ``arbitrate_band`` are the structure-free vote in the
    solve's own parameterization (bias corrected) and ITS OWN measured precision
    as a log-focal half-width; the arbitration refuses a structure candidate
    they contradict (see the contradiction test below).

    ``flags`` is the seed's confidence-flag list; the census endorsement gate
    appends ``census_guard`` to it IN PLACE when it holds the arbitration to the
    vote, and the contradiction test appends ``vote_contradiction``; both are
    appended when both hold, so the caller records every reason the structure
    candidate lost alongside the artifact.  The accepted candidate's own census
    appends ``census_echo`` the same way — a report on the shipped seed, not a
    reason a candidate lost, and the one flag that can fire on a seed stage 1
    was sure about."""
    from concurrent.futures import ThreadPoolExecutor

    from sfmtool._embed_patches import embed_patches
    from sfmtool._sfmtool.geometry import bundle_adjust as _nba
    from sfmtool._workspace_image import read_workspace_image

    all_c, all_i, all_u = data["obs_c"], data["obs_i"], data["obs_uv"]
    # A seed with fewer than two posed frames has nothing to finalize — fail
    # loudly and intelligibly here rather than as a shape error in the dense
    # triangulation (seen when the seed stage plants 0 poses).
    n_posed = int(np.asarray(posed).sum())
    if n_posed < 2:
        raise RuntimeError(
            f"seed finalization: only {n_posed} posed frame(s) — the seed "
            "stage failed to plant a usable seed; nothing to finalize"
        )
    # Dense seed structure: triangulate EVERY cluster at the seed poses (not just
    # the capped active set) so the finalization sees the full seed, matching the
    # dense-save snapshot.
    pts_dense, keep, res = dense_structure(
        all_c, all_i, all_u, f, rvec, tvec, pts.copy(), posed, quiet=True
    )
    tmp = WS / "sfmr" / "_seedfinal_tmp.sfmr"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # `alive` maps seed point -> source cluster; the far-field evidence arrives
    # in cluster ids, so it is the only bridge between the two index spaces.
    recon, alive, _posed_img = save_sfmr(
        data, f, rvec, tvec, pts_dense, keep, res, tmp, return_alive=True
    )
    seed_positions = np.asarray(recon.positions, dtype=np.float64)
    # Checkpoint 5: the dense seed the finalization starts from.  It is already
    # materialized here, so the snapshot is a file copy, not a re-save.
    snap = seed_snapshot_path("05-dense")
    if snap is not None:
        try:
            shutil.copyfile(tmp, snap)
            print(
                f"  [seed-snapshot {snap.name}: {len(recon.image_names)} images, "
                f"{recon.point_count} pts]"
            )
        except OSError as exc:
            print(f"  [seed-snapshot 05-dense FAILED: {type(exc).__name__}: {exc}]")
    try:
        tmp.unlink()
    except OSError:
        pass

    names = list(recon.image_names)
    with ThreadPoolExecutor() as pool:
        images = list(
            pool.map(lambda n: read_workspace_image(recon.workspace_dir, n), names)
        )
    n0 = recon.point_count
    emb = embed_patches(
        recon, images, progress=None, fronto_prior_weight=EMBED_FRONTO_PRIOR
    )
    n_emb = emb.point_count
    # The refinement rotated the patch frames; the normals array did not follow
    # it (see _refresh_normals).  Re-derive it before anything reads either.
    emb = _refresh_normals(emb)
    seed_snapshot_recon("06-embed", emb)  # checkpoint 6: refined, before the culls
    stage_dump("embed", emb)

    # DETECTION-TIME feature size per surviving point, for the duplicate collapse.
    # It is bound to the points HERE, while the photometric pipeline's
    # position-verbatim identity still holds, and then carried through every cull
    # below by applying the same masks — the collapse's re-triangulation and the
    # BA both break the position identity, and re-deriving the size from the
    # geometry afterwards is precisely the feedback loop COLLAPSE_ENABLED warns
    # about.
    cl_size = _cluster_detection_sizes(data)
    seed_size = cl_size[np.asarray(alive, dtype=np.int64)]
    src_of = _seed_index_map(seed_positions, emb.positions)
    det_size = np.where(src_of >= 0, seed_size[np.maximum(src_of, 0)], 0.0)
    # Points whose track membership or position a later pass changes.  Their
    # baked-at-embed bitmap and their stored error both stop describing them, so
    # the mask travels with det_size and drives the re-render below.
    mutated = np.zeros(emb.point_count, bool)
    # Stable per-point identity, carried the same way: the renumbering passes
    # below drop uid rows with the points they cull, so a later pass can still
    # ask what became of a point the passes above it renumbered.
    uid = _seed_point_uids(emb.point_count)
    print(
        f"  detection-time patch sizes: median {np.median(seed_size):.1f} px "
        f"(cluster scale x refine radius {data['refine_radius']:.1f}); "
        f"{int((src_of < 0).sum())}/{n_emb} embedded points unmatched"
    )

    # Drop length-2 survivors (repeated-pattern / spurious matches the expansion
    # could not validate to a third view).
    tl = np.bincount(np.asarray(emb.track_point_indexes), minlength=emb.point_count)
    keep_l3 = np.ascontiguousarray(tl >= 3, dtype=bool)
    emb = emb.filter_points_by_mask(keep_l3)
    det_size, mutated, uid = det_size[keep_l3], mutated[keep_l3], uid[keep_l3]
    n_len3 = emb.point_count
    stage_dump("len3", emb)

    # Cull coarse (large world-space) patches before the final BA.  Under the
    # feature_size extent policy a patch's world half-extent tracks its keypoint
    # SIFT scale, so the largest patches are the coarsest, least precisely
    # localizable features (their reprojection error runs higher) — real and
    # in-scene, but noise for the fine accuracy of the solve.  Drop any patch
    # whose characteristic world size exceeds SFMTOOL_PATCH_SIZE_MAX_MULT x the
    # median (data-derived per dataset; 0 disables).
    n_sizecull = n_len3
    mult = float(os.environ.get("SFMTOOL_PATCH_SIZE_MAX_MULT", "3.0"))
    pc = emb.patches
    if mult > 0 and pc is not None and len(pc) > 0:
        sz = np.array(
            [
                np.sqrt(abs(pc[i].half_extent[0]) * abs(pc[i].half_extent[1]))
                for i in range(len(pc))
            ]
        )
        keep_sz = np.ascontiguousarray(sz <= mult * np.median(sz), dtype=bool)
        emb = emb.filter_points_by_mask(keep_sz)
        det_size, mutated, uid = det_size[keep_sz], mutated[keep_sz], uid[keep_sz]
        n_sizecull = emb.point_count
    stage_dump("size-cull", emb)

    # Cheirality cull: drop any point that sits BEHIND one of the cameras that
    # claims to see it.  Photometric expansion + congealing can lock a patch
    # onto a repeated texture whose triangulation lands on the wrong side of a
    # camera plane; such a point is geometrically impossible, yet none of the
    # culls above look at depth, so it survives the whole chain and ships in
    # the seed (human inspection found 65/3433 points with non-positive depth
    # in EVERY observing view, down to -7155).  A single negative-depth
    # observation is disqualifying — the point cannot be both in front of one
    # camera and behind another and still be the thing both cameras imaged.
    n_cheir = emb.point_count
    keep_ch = _cheirality_keep(emb)
    if not keep_ch.all():
        keep_ch = np.ascontiguousarray(keep_ch, dtype=bool)
        emb = emb.filter_points_by_mask(keep_ch)
        det_size, mutated, uid = det_size[keep_ch], mutated[keep_ch], uid[keep_ch]
        n_cheir = emb.point_count
    stage_dump("cheirality", emb)

    # Collapse duplicate points: the same surface feature detected repeatedly in
    # the same images and left as several unmerged tracks, each triangulated on
    # its own narrow baseline into its own depth.  None of the culls above sees
    # this — every fragment is individually well-formed — so the seed ships a
    # cloud of laterally-coincident points at inconsistent depths, and the
    # census reads the resulting structure as evidence.  Merge before the BA so
    # the adjustment sees one point per feature with its full track.
    emb, coll, det_size, mutated, uid = _collapse_duplicate_points(
        emb, det_size, mutated, uid
    )
    emb = _refresh_errors(emb)
    n_collapse = emb.point_count
    _print_collapse("pre-BA", coll)
    stage_dump("collapse-preBA", emb)

    # Reconcile ALIASED tracks: the collapse's cannot-link refuses to merge the
    # pairs that overlap in one view and stand apart in another, which is correct
    # — but refusing to merge them leaves them as they are, and the malignant
    # ones are corrupt (two tracks each stitched across different instances of a
    # repeated pattern, both triangulated between the two real surfaces).  Repair
    # the assignment first, cull only what repair could not fix.
    emb, rec, det_size, mutated, uid = _reconcile_aliased_tracks(
        emb, det_size, mutated, uid
    )
    emb = _refresh_errors(emb)
    n_alias = emb.point_count
    _print_reconcile("pre-BA", rec)
    stage_dump("alias-preBA", emb)

    # Reprojection cull: the two passes above both END in a retriangulation, which
    # is a ray midpoint of whatever track it is handed — a track that is still
    # partly wrong yields a point that satisfies no view at all.  See
    # REPROJ_CULL_MULT; nothing earlier in the chain looks at reprojection.
    keep_rp, rp_bound, n_rp = _reprojection_cull_mask(emb)
    if n_rp:
        keep_rp = np.ascontiguousarray(keep_rp)
        emb = emb.filter_points_by_mask(keep_rp)
        det_size, mutated, uid = det_size[keep_rp], mutated[keep_rp], uid[keep_rp]
    print(
        f"  reprojection cull (pre-BA): {n_rp} points over "
        f"{rp_bound:.2f} px median reprojection"
    )
    n_reproj = emb.point_count
    stage_dump("reproj-cull-preBA", emb)

    # Cull unposeable cameras before the final BA.  A posed camera that shares
    # fewer than SFMTOOL_MIN_COVIS points with EVERY other camera resected on
    # almost nothing — its pose is underdetermined and it flies far off (the
    # catastrophic echo mode: a couple of near-empty cameras placed scene-scales
    # away).  The floor targets the truly-degenerate (covis 0-3) without touching
    # merely-weak cameras (covis 5-9), which are legitimately-but-loosely
    # connected — culling those drops real constraints and destabilizes the BA.
    # Drop cameras below the floor along with any now-orphaned points.
    # (default 5; 0 disables).
    n_camcull = len(emb.image_names)
    min_covis = int(os.environ.get("SFMTOOL_MIN_COVIS", "5"))
    if min_covis > 0 and n_camcull > 2:
        tii = np.asarray(emb.track_image_indexes)
        tpi = np.asarray(emb.track_point_indexes)
        pts_of = [set(tpi[tii == i].tolist()) for i in range(n_camcull)]
        maxcov = np.array(
            [
                max(
                    (len(pts_of[i] & pts_of[j]) for j in range(n_camcull) if j != i),
                    default=0,
                )
                for i in range(n_camcull)
            ]
        )
        keep_img = np.nonzero(maxcov >= min_covis)[0].astype(np.uint32)
        if 0 < len(keep_img) < n_camcull:
            # drop_orphaned_points keeps, in order, every point with a surviving
            # observation — mirror that mask onto the carried sizes.
            keep_pt = np.zeros(emb.point_count, bool)
            keep_pt[tpi[np.isin(tii, keep_img)]] = True
            emb = emb.subset_by_image_indices(np.ascontiguousarray(keep_img), True)
            det_size, mutated, uid = det_size[keep_pt], mutated[keep_pt], uid[keep_pt]
            images = [images[j] for j in keep_img.tolist()]
            n_camcull = len(keep_img)
    stage_dump("covis-camera-cull", emb)

    # Points at infinity: represent the depth-unconstrained tracks as DIRECTIONS
    # rather than forcing every track through finite triangulation.  A far-field
    # feature has almost no parallax, so its triangulated depth is noise — human
    # inspection found 3-view tracks at 0.29 deg parked at foreground depth, and
    # a whole <0.3 deg sub-population scattering to depth 103376 against a cloud
    # median of 2.9 (ray-smeared visual fuzz).  None of the culls above sees it:
    # each such point is individually well-formed, in front of every camera, and
    # reprojects fine.  The native classifier reads the triangulation's own
    # observability diagnostics (condition number, then the noise-calibrated
    # inverse-depth z-score) and relabels the unresolvable ones w = 0, keeping
    # their rotational and focal constraint without inventing a distance; the
    # native BA below carries a w = 0 point as a direction.  Relabel-only: no
    # point is dropped, and a track the baseline cannot adjudicate stays finite.
    #
    # Ordering: AFTER every cull and the collapse pass (they reason about depth,
    # image overlap and re-triangulation, none of which is defined for a
    # direction), and BEFORE the BA and the census, whose consumers are w-aware
    # (the census scores poses and focal against the raw cluster evidence and
    # never reads the point cloud; the cheirality cull already special-cases
    # directions).
    #
    # The pre-demotion state is kept whole (not just its mask): a demoted row
    # holds a direction afterwards, so the evidence the trace records for it —
    # depth and finite-fit reprojection — can only be measured on this.
    before_cls = emb
    was_inf = np.asarray(emb.point_is_at_infinity)
    emb = emb.classify_points_at_infinity(SEED_INFINITY_NOISE_PX)
    native_new = np.asarray(emb.point_is_at_infinity) & ~was_inf
    # Gate: undo the demotions whose own track the bearing model cannot explain.
    # Bearing test alone here — this is upstream of the BA, so a point kept
    # finite is a landmark the adjustment gets to use, and the post-BA classify
    # gets to reconsider it with the poses it produced.
    prop = np.nonzero(native_new)[0]
    veto = _inf_gate_veto("classify (pre-BA)", before_cls, prop)
    if veto.any():
        emb = _inf_gate_restore(before_cls, emb, prop[veto])
        native_new[prop[veto]] = False
    n_native = int(native_new.sum())
    n_inf = emb.infinity_point_count
    emb, n_reframed = _normalize_infinity_frames(emb, det_size, data["refine_radius"])
    print(
        f"  points at infinity: {n_inf}/{emb.point_count} "
        f"({n_native} native classifier); "
        f"{n_reframed} frames rebuilt at detection angular size"
    )
    stage_dump("infinity-preBA", emb)

    # ADAPTIVE ROBUST SURFEL NORMALS — the seed's normal estimate, replacing the
    # photometric one outright (see _ars_normals).
    #
    # Placement.  Under ARS_LATE_FIT (default) this site runs the EXPANSION and
    # the PROMOTION only: they manufacture tracks, and a manufactured track has
    # to face the same gauntlet as every other one — re-localization, eviction,
    # the BA, the infinity classifier, every cull — so it has to exist before
    # them.  The aiming needs neighbour directions and a determinacy ranking, so
    # the graph is built and the fit is run here as SCRATCH; none of it is
    # written.  The fit that writes runs at the end of the finalization, on the
    # cloud the gauntlet left.
    #
    # With ARS_LATE_FIT=0 the whole stage runs here, which is where it used to
    # live: the earliest point at which every membership and position cull has
    # run and the finite/infinity split is settled, and the last point at which a
    # new normal still reaches the BA through _relocalize_keypoints.  That
    # ordering bought the re-localization ARS frames at the price of fitting
    # unvetted geometry and never refitting after promotion.
    n_pre_promote = emb.point_count
    emb, ars, mutated, det_size, uid = _ars_normals(
        emb,
        images,
        det_size,
        data["refine_radius"],
        mutated,
        uid,
        mode="expand" if ARS_LATE_FIT else "all",
    )
    n_promoted = emb.point_count - n_pre_promote
    _print_ars("pre-BA", ars)
    emb = _refresh_errors(emb)
    stage_dump("ars-expand-promote", emb)

    # Re-localize the keypoints through the frames the cloud currently carries
    # (see _relocalize_keypoints).  The view sets, the track membership and the
    # finite/infinity split are all settled here, and this is the last moment at
    # which a keypoint correction still reaches the BA, which is the point of
    # running it here: the adjustment then optimizes against corrected
    # observations.  The post-BA re-checks (infinity re-classification,
    # cheirality, reprojection cull) stay where they are; they judge what the
    # adjustment did, not what it was given.
    #
    # Under ARS_LATE_FIT those frames are the EMBED stage's — the photometric
    # normals the congealing fitted, plus whatever the expansion's promoted
    # points inherited — not ARS ones, because the ARS write is now behind the
    # BA.  That is the acknowledged cost of fitting on the vetted cloud: the
    # re-localization corrects the keypoints against the same frames the embed
    # congealing used, so it is measuring drift the congealing left rather than
    # drift a new normal introduced.  What it buys is that the normals that SHIP
    # are fitted on adjusted, vetted geometry.
    if RELOCALIZE_ENABLED:
        for i_rl in range(max(RELOCALIZE_PASSES, 1)):
            emb, rl, mutated = _relocalize_keypoints(emb, images, mutated)
            _print_relocalize(f"pass {i_rl + 1}/{RELOCALIZE_PASSES}", rl)
            stage_dump(f"relocalize-pass{i_rl + 1}", emb)
            if not rl["accepted"]:
                break
        emb = _refresh_errors(emb)  # the pass moved points and keypoints

    # Evict the track members the photometry disowns.  Selection vetted the views
    # it ADDED and took the matcher's own on trust; this applies the same bar to
    # the members themselves.  It runs HERE, not at the embed, because by now the
    # keypoints have been re-localized, the merges and repairs have rewritten the
    # track membership, and the promoted candidates have arrived — so the members
    # being judged are the ones the BA will see — and it is still ahead of the
    # BA, so the adjustment never sees a rejected observation.  (Under
    # ARS_LATE_FIT the frames it measures against are the embed stage's; see the
    # re-localization note above.)
    n_obs_pre = len(np.asarray(emb.track_point_indexes))
    emb, ev, det_size, mutated, uid = _evict_track_views(
        emb, images, det_size, mutated, uid
    )
    if TRACK_EVICT_ENABLED:
        _print_evict("pre-BA", ev, n_obs_pre)
    stage_dump("evict-preBA", emb)
    # Promoted points are identified from here on by their uid block (see
    # _ars_promote): "how many of them the gauntlet keeps" is the question the
    # feature answers for, so it is counted rather than inferred from totals.
    prom_base = (ars.get("promotion") or {}).get("uid_base", -1)

    def _n_promoted_alive(u):
        return 0 if prom_base < 0 else int((np.asarray(u) >= prom_base).sum())

    n_prom_evict = _n_promoted_alive(uid)

    seed_snapshot_recon("07-culled", emb)  # checkpoint 7: culled, before the BA

    # Native core BA on the congealed, sub-pixel-refined keypoints.
    #
    # FOCAL GUARD (SFMTOOL_FINAL_F_MODE = guarded|fixed, default guarded):
    # this BA re-releases f on DENSE structure with none of the release stage's
    # protections, and on echo-contaminated seeds it takes the upward affine
    # escape the release explicitly refused (DnDTabletop: stage-1 +10.4% ->
    # finalize +36.3%).  "guarded" accepts the released f only inside the
    # stage-1 basin (<= 1.15x the seed f, >= the plausibility floor), else
    # refits with f frozen; "fixed" always freezes.  When the caller passes
    # ``f_final`` (a flagged seed's structure-free vote focal), the BA runs
    # frozen at that value — the structure disqualified itself, so it does not
    # get to re-release.
    mode = os.environ.get("SFMTOOL_FINAL_F_MODE", "guarded")

    def _run_ba(cam0, opt_f, state=None, opt_bspline=False, sel=None):
        """One BA over the embed's observations.

        ``state`` continues from a previous result's poses and points (the
        staged release: the spline rung starts where the focal release
        stopped); omitted, it starts from the embed.

        ``sel`` restricts the adjustment to a SUBSET of the observations
        (row indices into the track arrays).  The poses, points and the
        finite/infinity split stay whole — only the evidence changes — which
        is what makes two disjoint observation subsets two independent
        measurements of the same lens (the spline rung's candidate
        sources)."""
        if state is None:
            quats = np.ascontiguousarray(emb.quaternions_wxyz, np.float64)
            trans = np.ascontiguousarray(emb.translations, np.float64)
            pts = np.ascontiguousarray(emb.positions, np.float64)
        else:
            quats = np.ascontiguousarray(state["quaternions_wxyz"], np.float64)
            trans = np.ascontiguousarray(state["translations"], np.float64)
            pts = np.ascontiguousarray(state["points"], np.float64)
        uv_o = np.asarray(emb.keypoints_xy, np.float64)
        oi_o = np.asarray(emb.track_image_indexes, np.uint32)
        op_o = np.asarray(emb.track_point_indexes, np.uint32)
        if sel is not None:
            uv_o, oi_o, op_o = uv_o[sel], oi_o[sel], op_o[sel]
        return _nba(
            camera=cam0,
            quaternions_wxyz=quats,
            translations=trans,
            points=pts,
            uv=np.ascontiguousarray(uv_o, np.float64),
            obs_image=np.ascontiguousarray(oi_o, np.uint32),
            obs_point=np.ascontiguousarray(op_o, np.uint32),
            point_at_infinity=np.ascontiguousarray(emb.point_is_at_infinity, bool),
            opt_f=opt_f,
            opt_bspline=opt_bspline,
        )

    def _plant(out):
        """Map a BA result's poses onto the cluster data's image index space."""
        q = np.asarray(out["quaternions_wxyz"])
        t = np.asarray(out["translations"])
        Rm = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        Cm = -np.einsum("nij,ni->nj", Rm, t)
        name_to_j = {Path(n).name: j for j, n in enumerate(data["names"])}
        n_all = data["n_img"]
        Rfull = np.tile(np.eye(3), (n_all, 1, 1))
        Cfull = np.zeros((n_all, 3))
        posed_all = np.zeros(n_all, bool)
        for s, nm in enumerate(emb.image_names):
            j = name_to_j.get(Path(nm).name)
            if j is not None:
                Rfull[j], Cfull[j], posed_all[j] = Rm[s], Cm[s], True
        return posed_all, Rfull, Cfull

    def _census_of(out, f_c):
        """Cluster census score of a candidate BA result (GT-free).

        The census triangulates and reprojects the raw clusters itself, so it
        gets the CONTEXT camera at the candidate focal; scored through a
        pinhole it would be measuring the model mismatch of a fisheye solve
        rather than the solve."""
        import seed_census as SC

        posed_all, Rfull, Cfull = _plant(out)
        score, _ = SC.census_score(
            data, posed_all, Rfull, Cfull, float(f_c), camera=make_cam(float(f_c))
        )
        return score

    def _bspline_rung(out0, opt_f_rung):
        """Release the lens's radial spline on top of the settled focal.

        A real lens's ``r(d)`` is not exactly ``f * d`` in its base model's
        radial coordinate — the incidence angle ``theta`` under the
        equidistant base, the normalized image-plane radius ``rho`` under the
        pinhole one — and a single cubic term (the superseded k1 rung) cannot
        fit a lens that flattens at the periphery: its genuine optimum on
        such a lens is a near-base compromise with the residual curvature
        smeared everywhere.  Promoting the camera to the base's spline model
        with a ZERO radial spline — the same map, bit for bit (the model's
        zero-spline identity) — gives the whole residual radial field an
        N-knot monotone spline to land on instead, and lets far structure
        walk back out to infinity.  Everything below is written in ``d``;
        ``spline_model()`` is where the base picks which coordinate that is.

        FOCAL SEMANTICS.  The rung CO-RELEASES ``f`` with the spline even
        where the finalization froze it: a frozen-at-vote f is model-charged
        (the vote measured the best BASE-MODEL focal, not the central scale
        of a spline-corrected map), and under the model's center-anchored gauge
        (``delta(0) = delta'(0) = 0``) the spline cannot express a
        central-scale correction on its own.  The freeze is honored instead
        in the quantity the vote actually measures: the EQUIVALENT BASE
        FOCAL ``f_eq`` of the released composite map — the
        least-squares fit of ``r = f_eq * d`` over the observed ``d``
        distribution — must stay inside the stage-1 basin, and, where the
        finalization froze f against a structure-free vote with a measured
        band, inside that band around the frozen f as well.  The test runs
        as a CANDIDATE GATE: a candidate whose map escapes never reaches the
        arbitration, so what ships is in band by construction (the shipped
        map restates it, and a rung with nothing left in band refuses).

        CANDIDATES AND ARBITRATION.  The release is not one continuation but
        a POOL, arbitrated across two independent measurements of the same
        lens.  Four facts force this:

        1. Basin membership depends on the STARTING GEOMETRY, not on the
           schedule — the same cold start converges from one state and lands
           in a wrong basin from another, so no single start is reliable and
           a spread of starts is.  ``SFMTOOL_BSPLINE_POOL`` (default
           ``0.9,1.0,1.1``) is that spread, crossed with cold (co-release
           f + spline from a zero spline) and warm (release f alone first,
           then co-release from where it stopped).  The multipliers are
           clamped to what the kernel's first trim can absorb — see the
           ``mult_cap`` note below.
        2. A candidate's OWN residual median is anti-correlated with map
           truth — a wrong basin can fit its own subset beautifully — so the
           pool cannot be ranked on its own residuals.
        3. Popularity inside one pool is gamed by near-duplicate candidates
           (six near-identical members outvote a lone correct one), so
           candidates agreeing to ``0.5`` px of composite map are folded to
           one before anything counts them.
        4. A degenerate kernel exit returns its INPUT state with all-inf
           residuals, and two such exits "agree" perfectly.  The validity
           gate is therefore both a finite residual median AND a composite
           map that moved at least ``0.05`` px from the candidate's own
           start map.

        The sources are the two halves of an INTERLEAVED split of the posed
        frames (by image index): disjoint evidence, both spanning the whole
        capture, the same poses and points underneath.  The winner is the
        cross-source pair with the smallest median composite-map
        disagreement over the comparison band at the image's own pixel scale —
        agreement between measurements that share no observation is the one
        signal a shared wrong basin cannot manufacture.  A pair further
        apart than the agreement bar (1% of the field radius by default;
        ``SFMTOOL_BSPLINE_AGREE_PX`` overrides it in absolute pixels)
        refuses the rung outright.  The pair member from
        the richer-support source is what ships: both halves measured the
        same lens, and the one that saw more observations measured it
        better.

        THE FULL-SET PASS then fits the GEOMETRY to that camera, with the
        intrinsics held fixed and starting from the embed state.  It does
        not re-release f or the spline: doing so hands the decision back to
        the evidence whose own optimum is the basin the arbitration just
        rejected (see the block comment at the call).

        KEEP-BEST is measured against a CONTROL that runs the same full-set
        pass on the strongest SPLINE-FREE camera — the base model, f
        released, same start, same schedule, same trims and
        retriangulations, N + 1
        parameters fewer.  Comparing against ``out0`` alone would only ask
        whether the rung beat a frozen base map; the bar here is that
        it beats the best one.  Monotonicity needs no end-state assertion —
        the kernel's step guard rejects any folding step, so an accepted
        spline is monotone on [0, d_max] by construction.

        The last release stage, on either base: everything after
        it (the infinity re-classification included) then judges the
        spline-refined geometry.

        A REFUSAL GOVERNS THE MAP, NOT THE MODEL.  Every guard below is a
        verdict about the radial CORRECTION — there is none this evidence
        can stand behind — and none of them is a verdict about which model
        describes the lens.  So a refusal keeps the base map exactly and
        ships it AS THE BASE'S SPLINE MODEL, at the refusal's own focal with
        a ZERO spline over the field-derived domain ``[0, d_max]``.  That is
        the same map bit for bit (the model's zero-spline identity:
        projection, inverse and pixel Jacobian alike), with the coefficient
        slots allocated, so a later stage can release ``opt_bspline`` on the
        shipped camera without a model switch — where shipping the base model
        would have foreclosed the lens's evolution downstream.  The
        conservative decision lives in the COEFFICIENTS, never in the model.

        Returns ``(result, (coefficients, d_max))``: the released result and
        its spline on acceptance, the PRE-RUNG result and a ZERO spline on
        every refusal below.  ``(result, None)`` — the base model left
        untouched — only where there is no spline domain to promote onto at
        all."""
        f0 = float(out0["focal"])
        uv_all = np.asarray(emb.keypoints_xy, dtype=np.float64)
        cxp, cyp = float(_CAM_WH[0]) / 2.0, float(_CAM_WH[1]) / 2.0
        r_obs = np.hypot(uv_all[:, 0] - cxp, uv_all[:, 1] - cyp)
        field_r = float(r_obs.max()) if r_obs.size else 0.0
        n_knots = int(os.environ.get("SFMTOOL_BSPLINE_KNOTS", "8"))
        fisheye_base = spline_model()[2]
        # The spline's domain: the radial coordinate of the outermost
        # observation through the PRE-RUNG base map (d = r / f0, the base
        # model being distortion-free either way), with 2%
        # headroom so that observation sits strictly inside the live knot
        # spans rather than exactly on the boundary of the held-constant
        # tail.  d_max is fixed by the input camera from here on — the
        # BA solves coefficients on this domain, and the promoted camera
        # carries it verbatim.
        d_max = 1.02 * field_r / f0
        # The one exit that keeps the BASE model.  It is not a refusal of a
        # correction — it is the absence of anything to promote onto: no
        # coefficient slots to allocate (``n_knots < 2``), or no observed
        # field to measure a domain over.  A spline model needs a domain end
        # and a coefficient count, and neither exists here.
        if n_knots < 2 or not (np.isfinite(d_max) and d_max > 0.0):
            return out0, None
        # theta > 60 deg is where a radial spline is worth anything — and
        # where the base fit was paying for it in geometry.  The mask
        # is taken at the PRE-rung camera so every arm is scored on the same
        # observations, and the bound is carried into d so it is the same
        # 60 degrees of incidence under either base.
        peripheral = r_obs > f0 * _d_of_theta(np.pi / 3.0, fisheye_base)

        def _score(o):
            res = np.asarray(o["residual_norms"], dtype=np.float64)
            fin = np.isfinite(res)
            med = float(np.median(res[fin])) if fin.any() else np.inf
            pf = fin & peripheral
            per = float(np.median(res[pf])) if pf.any() else np.nan
            return med, per, int(fin.sum())

        def _f_eq_of_map(fc, coeffs_c):
            """Equivalent BASE focal of a composite map: ``r(d) = f_eq * d``
            least-squares fitted over the
            OBSERVED ``d`` distribution (each observation's radial coordinate
            at the pre-rung camera), i.e. ``f_eq = sum(r * d) / sum(d^2)``
            with ``r`` the released map evaluated at those ``d``."""
            cam_c = make_cam_bspline(fc, coeffs_c, d_max)
            dd = r_obs / f0
            th = _theta_of_d(dd, fisheye_base)
            rays = np.stack([np.sin(th), np.zeros_like(th), -np.cos(th)], axis=1)
            px = np.asarray(cam_c.ray_to_pixel_batch(np.ascontiguousarray(rays)))
            r_map = px[:, 0] - cxp
            ok = np.isfinite(r_map) & (dd > 0)
            if not ok.any():
                return float("nan")
            return float((r_map[ok] * dd[ok]).sum() / (dd[ok] ** 2).sum())

        def _f_eq_of(o):
            """``_f_eq_of_map`` for a BA result's camera."""
            return _f_eq_of_map(
                float(o["focal"]),
                np.asarray(o["bspline_coefficients"], dtype=np.float64),
            )

        def _f_eq_ok(fe):
            """The stage-1 basin on the composite map's base-model reading
            (the same bounds the focal release holds raw f to), plus — where
            the finalization froze f against a structure-free vote with a
            measured precision band — that band around the frozen f."""
            if not np.isfinite(fe) or not (focal_floor() <= fe <= 1.15 * f):
                return False
            if not opt_f_rung and arbitrate_center and arbitrate_band:
                return abs(float(np.log(fe / f0))) <= float(arbitrate_band)
            return True

        # ── composite-map comparison, at the image's own pixel scale ──────
        # Every candidate comparison in this rung is made on the MAP, not on
        # parameters: f and the coefficients trade against each other under
        # the center-anchored gauge, so two cameras with visibly different
        # focals can describe the same lens (Kerry's pool spans f 127-143 at
        # 0.2 px of map).  The band runs in ``d`` from 5 deg of incidence —
        # inside it every map is its own central scale by construction — out
        # to the far end of the data: theta 95 deg under the fisheye base,
        # and the spline domain's own end under the pinhole base, whose map
        # is defined below 90 deg only and whose rho grows without bound
        # toward it.
        d_cmp = np.linspace(
            _d_of_theta(np.deg2rad(5.0), fisheye_base),
            _d_of_theta(np.deg2rad(95.0), fisheye_base) if fisheye_base else d_max,
            60,
        )
        th_cmp = _theta_of_d(d_cmp, fisheye_base)
        ray_cmp = np.ascontiguousarray(
            np.stack([np.sin(th_cmp), np.zeros_like(th_cmp), -np.cos(th_cmp)], axis=1)
        )

        def _map_curve(fc, coeffs_c):
            px = np.asarray(
                make_cam_bspline(fc, coeffs_c, d_max).ray_to_pixel_batch(ray_cmp)
            )
            return np.hypot(px[:, 0] - cxp, px[:, 1] - cyp)

        def _map_dist(a, b):
            d = np.abs(_map_curve(*a) - _map_curve(*b))
            fin = np.isfinite(d)
            return float(np.median(d[fin])) if fin.any() else np.inf

        zero = np.zeros(n_knots)

        def _refuse(why, detail):
            """What a refusal ships: the PRE-RUNG result, carried by the
            base's spline model with a ZERO spline on ``[0, d_max]``.

            See the docstring — the guards decide the correction, not the
            model, and an all-zero spline is the base map bit for bit.  The
            result returned is the one the base-model refusal always
            returned, so every array the caller writes is unchanged; the
            camera table and the metadata are the whole of the difference."""
            print(f"  spline rung REFUSED ({why}): {detail}")
            return out0, (zero, d_max)

        # Start spread, CLAMPED TO WHAT THE KERNEL CAN ABSORB.  A start
        # multiplier displaces the outermost observation by |1 - m| * field_r
        # pixels; past the widest trim of the BA's first round (50 px) the
        # round trims essentially everything and exits degenerate, returning
        # the start verbatim.  That is a pure image-scale effect and it makes
        # the same nominal spread useless on large frames: +/-10% is 24 px on
        # Kerry's 480 px frames (fine) and 192 px on FirstInstaTest's 3840 px
        # ones, where 8 of 12 candidates died on arrival.  Cap the deviation
        # at half the first trim so every requested start is a live candidate.
        mult_cap = 25.0 / field_r if field_r > 0 else 1.0
        pool_mults = [
            1.0 + float(np.clip(float(x) - 1.0, -mult_cap, mult_cap))
            for x in os.environ.get("SFMTOOL_BSPLINE_POOL", "0.9,1.0,1.1").split(",")
            if x.strip()
        ]
        # Cross-source agreement bar.  It is a MAP distance in pixels, so it
        # scales with the frame: the true pairs measured here agreed to 0.03%
        # of the field radius on Kerry and 0.23-0.28% on FirstInstaTest, while
        # wrong basins sat 3-8% of it apart.  1% of the field radius sits in
        # that gap with margin on both sides (2.4 px on Kerry, 19 px on
        # FirstInstaTest); a fixed 5 px bar is 2% of one field and 0.26% of the
        # other and cannot serve both.  SFMTOOL_BSPLINE_AGREE_PX overrides it
        # with an absolute pixel bar.
        agree_bar = float(
            os.environ.get("SFMTOOL_BSPLINE_AGREE_PX", "") or 0.01 * field_r
        )

        def _candidate(sel, f_start, warm):
            """One pool member: a continuation from ``out0``'s state over the
            source's observations.  Returns ``(result, start_camera)`` — the
            start is what the validity gate measures movement against, and
            for the warm schedule that is the f-only stage's own converged
            camera, not the multiplier."""
            if warm:
                pre = _run_ba(
                    make_cam_bspline(f_start, zero, d_max),
                    True,
                    state=out0,
                    sel=sel,
                )
                f_pre = float(pre["focal"])
                return (
                    _run_ba(
                        make_cam_bspline(f_pre, zero, d_max),
                        True,
                        state=pre,
                        opt_bspline=True,
                        sel=sel,
                    ),
                    (f_pre, zero),
                )
            return (
                _run_ba(
                    make_cam_bspline(f_start, zero, d_max),
                    True,
                    state=out0,
                    opt_bspline=True,
                    sel=sel,
                ),
                (f_start, zero),
            )

        # Interleaved split of the POSED FRAMES: disjoint observations, both
        # halves spanning the capture, one underlying geometry.
        img_of_obs = np.asarray(emb.track_image_indexes)
        pool_by_src, src_stats = {}, {}
        for sname in ("A", "B"):
            mask = (img_of_obs % 2) == (0 if sname == "A" else 1)
            sel = np.nonzero(mask)[0]
            kept, n_deg, n_still, n_band, n_dup = [], 0, 0, 0, 0
            for mult in pool_mults:
                for warm in (False, True):
                    o, start = _candidate(sel, f0 * mult, warm)
                    fc = float(o["focal"])
                    cc = np.asarray(o["bspline_coefficients"], dtype=np.float64)
                    res_c = np.asarray(o["residual_norms"], dtype=np.float64)
                    fin_c = np.isfinite(res_c)
                    if not fin_c.any():
                        n_deg += 1
                        continue
                    if _map_dist((fc, cc), start) < 0.05:
                        n_still += 1
                        continue
                    if not _f_eq_ok(_f_eq_of_map(fc, cc)):
                        n_band += 1
                        continue
                    if any(_map_dist((fc, cc), (f2, c2)) < 0.5 for _, f2, c2 in kept):
                        n_dup += 1
                        continue
                    tag = f"{sname}{mult:.2f}{'w' if warm else 'c'}"
                    kept.append((tag, fc, cc))
            pool_by_src[sname] = kept
            src_stats[sname] = (int(mask.sum()), n_deg, n_still, n_band, n_dup)
        n_pool = 2 * 2 * len(pool_mults)
        srcs = (
            f"sources A/B {src_stats['A'][0]}/{src_stats['B'][0]} obs "
            f"({len(np.unique(img_of_obs))} posed frames); pool {n_pool} -> "
            f"valid {len(pool_by_src['A'])}/{len(pool_by_src['B'])} "
            f"(rejected: degenerate {src_stats['A'][1] + src_stats['B'][1]}, "
            f"unmoved {src_stats['A'][2] + src_stats['B'][2]}, "
            f"out-of-band {src_stats['A'][3] + src_stats['B'][3]}, "
            f"duplicate {src_stats['A'][4] + src_stats['B'][4]})"
        )
        best = None
        for la, fa, ca in pool_by_src["A"]:
            for lb, fb, cb in pool_by_src["B"]:
                d = _map_dist((fa, ca), (fb, cb))
                if best is None or d < best[0]:
                    best = (d, (la, fa, ca), (lb, fb, cb))
        if best is None:
            return _refuse("no cross-source pair", srcs)
        agree, cand_a, cand_b = best
        if not np.isfinite(agree) or agree > agree_bar:
            return _refuse(
                f"best cross-source pair {cand_a[0]} ~ {cand_b[0]} disagrees "
                f"by {agree:.2f} px > {agree_bar:g} px bar",
                srcs,
            )
        # The richer-support member ships: both halves measured the same
        # lens, and the one that saw more observations measured it better.
        win = cand_a if src_stats["A"][0] >= src_stats["B"][0] else cand_b
        pair = (
            f"pair {cand_a[0]} ~ {cand_b[0]} agree {agree:.2f} px "
            f"(bar {agree_bar:.2f}), ship {win[0]} "
            f"(f {win[1]:.1f}, f_eq {_f_eq_of_map(win[1], win[2]):.1f})"
        )

        # ── the full-set pass: geometry, at the ARBITRATED camera ─────────
        # The intrinsics are decided by now, and they are NOT re-opened here.
        # Two measured reasons:
        #
        # - Re-releasing f/spline on the full set hands the decision back to
        #   the very evidence whose own residual optimum is the basin the
        #   arbitration rejected: on Kerry it walks the arbitrated map (0.4 px
        #   from the transferred truth) out to 11-14 px, because the whole
        #   capture's least-squares optimum IS the contaminated one.  The
        #   cross-source agreement is the finding; the full set does not get a
        #   veto over it.
        # - The pass starts from the EMBED state, not from ``out0``.  ``out0``
        #   is a converged fit to the pre-rung base map, and re-fitting
        #   it under a different map only perturbs it inside that basin:
        #   Kerry's arbitrated camera scores 3.8 px median continued from
        #   ``out0`` and 0.36 px started from the embed — the same start the
        #   pre-rung BA itself used.  Finding 1 (basin membership is set by the
        #   starting geometry) restated on the geometry rather than the lens.
        #
        # The control is the strongest SPLINE-FREE alternative from that same
        # start — the base-model camera with f released — so the bar the
        # spline has to clear is "better than any base map could do
        # here", not merely "better than the continuation".
        cam0 = make_cam_bspline(f0, zero, d_max)
        cam_w = make_cam_bspline(win[1], win[2], d_max)
        ctl = _run_ba(cam0, True)
        out1 = _run_ba(cam_w, False)
        # The basin guard, restated on the quantity the vote measures: the
        # composite map's own base-model reading, not raw f (which is now a
        # central scale the spline shares the field with).  It already ran as
        # a candidate gate, so this is the shipped map's own restatement of it
        # — no candidate that fails it can reach here.
        f_eq = _f_eq_of(out1)
        f1 = float(out1["focal"])
        coeffs = np.asarray(out1["bspline_coefficients"], dtype=np.float64)
        med0, per0, nfin0 = _score(out0)
        medc, perc, nfinc = _score(ctl)
        med1, per1, nfin1 = _score(out1)
        d_txt = (
            f"theta_max={np.degrees(d_max):.1f}deg"
            if fisheye_base
            else f"rho_max={d_max:.4f}"
        )
        report = (
            f"{srcs}; {pair}; "
            f"spline N={len(coeffs)} {d_txt} "
            f"|c|inf={float(np.abs(coeffs).max()) if coeffs.size else 0.0:.5f}; "
            f"f {f0:.1f} -> {f1:.1f} ({100 * (f1 / f0 - 1):+.2f}%), "
            f"f_eq {f_eq:.1f} ({100 * (f_eq / f0 - 1):+.2f}%); "
            f"median {med1:.3f} px vs control {medc:.3f} / pre-rung {med0:.3f}; "
            f"theta>60 {per1:.3f} vs {perc:.3f} / {per0:.3f} px "
            f"({int(peripheral.sum())} obs); valid {nfin1} vs {nfinc} / {nfin0}"
        )
        # KEEP-BEST.  Three ways to refuse: a composite map whose base-model
        # reading escapes the basin/band, a residual field the spline did
        # not improve, or observations that fell out of the model's domain.
        # The bar is BOTH comparisons: better than the control (the spline
        # earned its coefficients against the best base-model map) and no
        # worse than the pre-rung result (nothing shipped degrades).
        if not _f_eq_ok(f_eq):
            return _refuse("composite map outside the basin/band", report)
        med_bar, per_bar = min(medc, med0), min(perc, per0)
        worse = med1 > 1.02 * med_bar or (
            np.isfinite(per_bar) and np.isfinite(per1) and per1 > 1.02 * per_bar
        )
        if worse or nfin1 < min(nfinc, nfin0):
            return _refuse("no gain", report)
        print(f"  spline rung: {report}")
        return out1, (coeffs, d_max)

    if arbitrate_vote is not None:
        # Dual-candidate finalization for flagged (non-edge) seeds: the flags
        # say stage 1 was unsure, not WHICH of vote vs structure to trust —
        # fleet A/B showed blanket freeze-at-vote fixes vote-good captures and
        # breaks vote-bad ones the free BA had been rescuing.  Run the BA both
        # ways and let the cluster census (structure-independent, tracks
        # focal error) pick the candidate the raw evidence disagrees with
        # least.  Ties prefer the vote (the flag's own semantics).
        out_v = _run_ba(make_cam(float(arbitrate_vote)), False)
        out_s = _run_ba(emb.cameras[0], True)
        f_s = float(out_s["focal"])
        s_released = True
        if not (focal_floor() <= f_s <= 1.15 * f):
            out_s = _run_ba(emb.cameras[0], False)
            f_s = f
            s_released = False
        s_v = _census_of(out_v, arbitrate_vote)
        s_s = _census_of(out_s, f_s)
        # CONTRADICTION TEST.  The census is computed FROM THE SAME STRUCTURE
        # that produced the structure candidate — it triangulates the raw
        # clusters at that candidate's poses — so on support that is internally
        # self-consistent without being the static scene (false parallax: moving
        # cloud, wind-blown foliage) it endorses the wrong focal.  Measured on
        # KerryPark480, whose rig gives a best-fit equidistant optimum of
        # 135.52 px: the census scores the +13.1% structure candidate 0.416
        # (endorsed) against 0.865 for the +2.06% vote.  The structure-free vote
        # never touches structure, and its own precision is measurable at
        # runtime, so a structure candidate outside the vote's band is
        # contradicted by an INDEPENDENT measurement and the census — dependent
        # evidence — is not admissible against that.  The band is the vote
        # pool's own log-focal IQR floored at the column's measured accuracy
        # (see exp_fast_seed.VOTE_BAND_FLOOR_LOG); on uncontaminated support
        # release and vote agree to 0.7%, so the test is inert there.
        contradicted = False
        if arbitrate_center and arbitrate_band:
            d = abs(float(np.log(f_s / float(arbitrate_center))))
            contradicted = d > float(arbitrate_band)
        # Structure may only take the decision when the census ENDORSES it
        # (see CENSUS_ENDORSE_MAX): the relative comparison s_v vs s_s only
        # means something while the structure candidate is itself explicable
        # to the census.  When it is not, the fact that it disagrees less than
        # the vote does is not evidence for it — both candidates are sick, and
        # the vote is the one estimate that does not depend on the structure.
        endorsed = s_s < CENSUS_ENDORSE_MAX and not contradicted
        guarded = s_v > s_s + 0.005 and not endorsed
        if s_v <= s_s + 0.005 or guarded:
            out = out_v
            f_was_released = False
            pick = f"vote {float(arbitrate_vote):.1f}"
        else:
            out = out_s
            f_was_released = s_released
            pick = f"structure {f_s:.1f}"
        print(
            f"  finalize census arbitration: vote {float(arbitrate_vote):.1f} "
            f"-> {s_v:.3f} vs structure {f_s:.1f} -> {s_s:.3f}; keeping {pick}"
        )
        # Both reasons are recorded when both hold: they are different facts
        # about the structure candidate (the census cannot explain it / an
        # independent measurement contradicts it), and a run that drops one
        # would report a weaker case than it made.
        if guarded and s_s >= CENSUS_ENDORSE_MAX:
            print(
                f"  census guard: structure {f_s:.1f} unendorsed "
                f"(census {s_s:.3f} >= {CENSUS_ENDORSE_MAX:g}); keeping vote"
            )
            if flags is not None and "census_guard" not in flags:
                flags.append("census_guard")
        if guarded and contradicted:
            print(
                f"  vote contradiction: structure {f_s:.1f} is "
                f"{100 * (f_s / float(arbitrate_center) - 1):+.1f}% off the "
                f"structure-free measurement {float(arbitrate_center):.1f} "
                f"(its own precision band +/-"
                f"{100 * (np.exp(float(arbitrate_band)) - 1):.1f}%); the census "
                f"is structure-derived and cannot endorse against it"
            )
            if flags is not None and "vote_contradiction" not in flags:
                flags.append("vote_contradiction")
    elif f_final is not None:
        print(f"  finalize BA: f frozen at the structure-free vote {f_final:.1f}")
        out = _run_ba(make_cam(float(f_final)), False)
        f_was_released = False
    elif mode == "fixed":
        out = _run_ba(emb.cameras[0], False)
        f_was_released = False
    else:
        out = _run_ba(emb.cameras[0], True)
        f_was_released = True
        f_ba = float(out["focal"])
        if mode == "guarded" and not (focal_floor() <= f_ba <= 1.15 * f):
            print(
                f"  finalize BA left the stage-1 basin (f {f:.1f} -> {f_ba:.1f}); "
                f"refitting with f frozen"
            )
            out = _run_ba(emb.cameras[0], False)
            f_was_released = False
    # THE SPLINE RUNG, after the focal release has settled — on EITHER BASE.
    # A real lens's radial map is not exactly its base model's, and that is as
    # true of a pinhole capture as of a fisheye one; the rung is written in
    # whichever radial coordinate the base measures, so the base decides only
    # which spline model the promotion installs (SFMTOOL_FISHEYE over the
    # equidistant base, SFMTOOL_PINHOLE over the pinhole one).  The context
    # itself is promoted, so every later stage — the census below, the
    # infinity re-classification, the reprojection cull — and the camera the
    # writer stamps all read the rung's camera through make_cam.
    #
    # A REFUSAL PROMOTES TOO, at the pre-rung focal with a ZERO spline over
    # the rung's own field-derived domain.  The rung refuses a radial
    # CORRECTION; that is no reason to foreclose the lens's evolution
    # downstream, which shipping the base model would.  A zero spline is the
    # base map bit for bit, so a refused seed's poses, structure, residuals
    # and every other archive entry are exactly what they were before this
    # promotion existed — the camera table and the metadata are the whole of
    # the difference — and a later stage can release opt_bspline on the
    # shipped camera with no model switch.
    #
    # SFMTOOL_BSPLINE_RUNG=0 is the deliberate exception: the kill switch
    # exists to reproduce the PRE-RUNG pipeline, so it skips the promotion
    # along with the rung and writes the base model, as it always did.
    bspline_final = None
    if os.environ.get("SFMTOOL_BSPLINE_RUNG", "1") != "0":
        promoted_model = spline_model()[0]
        out, bspline_final = _bspline_rung(out, f_was_released)
        if bspline_final is not None:
            set_camera_context(
                promoted_model,
                float(out["focal"]),
                bspline=bspline_final[0],
                theta_max=bspline_final[1],
            )
    # Full census report of the accepted candidate, group-consistency
    # companion included: does the raw cross-group evidence disagree, and if
    # so is the disagreement coherent (a group-level pose error a similarity
    # could fix) or junk-driven?  Analysis only — logged for the caller.
    import seed_census as SC

    posed_all, Rfull, Cfull = _plant(out)
    f_acc = float(out["focal"])
    rep = SC.census_report(data, posed_all, Rfull, Cfull, f_acc, camera=make_cam(f_acc))
    gc = rep["group_consistency"]
    gc_txt = (
        "gc=none"
        if gc is None
        else (
            f"gc explained={gc['explained_pct']:.1f}% "
            f"({gc['n_explained']}/{gc['n_unsatisfied_before']}) "
            f"net {gc['net_before']}->{gc['net_after']}"
        )
    )
    print(
        f"  finalize census: score={rep['score']:.3f} groups={rep['n_groups']} "
        f"sat={rep['sat_pct']:.1f}% {gc_txt}"
    )
    # The `census_echo` confidence flag.  METADATA ONLY — nothing downstream
    # reads it, and the finalization ships the same reconstruction either way;
    # it records that the raw cross-group evidence disagrees with the seed that
    # shipped.  Fewer than two viewpoint groups is UNVERIFIABLE, not clean (the
    # score is 0 there by construction), so the flag states the group condition
    # rather than leaning on the vacuous score to stay under the bar.
    if int(rep["n_groups"]) >= 2 and float(rep["score"]) >= CENSUS_ECHO_FLAG_MIN:
        print(
            f"  census echo: score {float(rep['score']):.3f} >= "
            f"{CENSUS_ECHO_FLAG_MIN:g} over {int(rep['n_groups'])} viewpoint "
            f"groups; flagging the seed"
        )
        if flags is not None and "census_echo" not in flags:
            flags.append("census_echo")
    # The BA returns a (n_pt, 3) block whose infinity rows are unit DIRECTIONS,
    # so the write-back must be homogeneous: a bare (n, 3) `positions` is taken
    # as Euclidean and would silently re-promote every w = 0 point to finite.
    inf_out = np.asarray(emb.point_is_at_infinity)
    pos_out = np.hstack(
        [
            np.asarray(out["points"], dtype=np.float64),
            np.where(inf_out, 0.0, 1.0)[:, None],
        ]
    )
    final = emb.clone_with_changes(
        cameras=[make_cam(float(out["focal"]))],  # write the BA-refined focal back
        quaternions_wxyz=np.ascontiguousarray(out["quaternions_wxyz"]),
        translations=np.ascontiguousarray(out["translations"]),
        positions=np.ascontiguousarray(pos_out),
    )
    final = _refresh_errors(final)  # the BA moved every point
    stage_dump("ba", final)
    # Collapse again at the post-BA state.  The adjustment moves both structure
    # and poses, so pairs that were still distinguishable going in can come out
    # coincident: the BA leaves a small duplicate residue the pre-BA pass could
    # not have seen (a prior measurement found 12 further mergeable pairs
    # appearing only after the final BA).  Same criterion, same cannot-link, on
    # the adjusted geometry — before the infinity re-classification and the
    # cheirality re-check, both of which should judge the merged tracks.
    final, coll_ba, det_size, mutated, uid = _collapse_duplicate_points(
        final, det_size, mutated, uid
    )
    final = _refresh_errors(final)
    _print_collapse("post-BA", coll_ba)
    stage_dump("collapse-postBA", final)
    final, rec_ba, det_size, mutated, uid = _reconcile_aliased_tracks(
        final, det_size, mutated, uid
    )
    final = _refresh_errors(final)
    _print_reconcile("post-BA", rec_ba)
    stage_dump("alias-postBA", final)

    # Contained-inconsistent cull.  Runs AFTER the collapse and the
    # reconciliation, on the adjusted geometry: both of those measure overlap in
    # the SMALLER member's radius, so what reaches here is what a min-radius test
    # cannot see, and the BA is the last thing that moves a range.  It culls
    # rather than repairs — a coarse feature that swallows fine points at a
    # different range has no track to reassign; its observations are its own.
    final, cont_ba, det_size, mutated, uid = _cull_contained_inconsistent(
        final, det_size, mutated, uid
    )
    final = _refresh_errors(final)
    _print_contained("post-BA", cont_ba)
    stage_dump("contained-postBA", final)

    # Re-classify AFTER the BA, for the same reason the cheirality check below
    # repeats here: the adjustment is the last thing that can move a point, and
    # it moves a low-parallax one FURTHEST — along its own ray, the direction it
    # is least constrained in.  A track whose depth the pre-BA geometry could
    # still adjudicate can come out the far side unresolvable (on
    # 20250907_000240907 the post-BA finite tail ran to depth 680000 against a
    # cloud median of 2.6, and every one of those points classified at infinity
    # at the final poses).  Relabel-only, so nothing is dropped; it runs before
    # the cheirality cull because a depth the BA blew up along the ray is not
    # evidence of a cheirality violation — the direction is what survived.
    n_inf_ba = final.infinity_point_count
    before_cls2 = final  # see before_cls above: the demotion evidence lives here
    was_inf2 = np.asarray(final.point_is_at_infinity)
    final = final.classify_points_at_infinity(SEED_INFINITY_NOISE_PX)
    # Gate — and this is the DECISIVE site, so it carries the depth-plausibility
    # conjunct the two pre-BA sites do not: whatever this classify labels ships,
    # and a veto here ships the position too.  A point the BA blew up along its
    # own ray normally still demotes (its rays ARE a bearing, so the fit passes
    # and the gate is silent), but not always: on 20250907_000240907 one and on
    # 20240614_224422531 two of the runaways failed the bearing test by a hair
    # (1.12-1.40 px), and depth-blind the gate shipped them finite at 8.3e4 /
    # 8.7e8 / 5524 against cloud medians of 3.7 / 52.  Plausibility is what those
    # cannot pass.
    prop2 = np.nonzero(np.asarray(final.point_is_at_infinity) & ~was_inf2)[0]
    veto2 = _inf_gate_veto(
        "classify (post-BA)", before_cls2, prop2, depth_plausibility=True
    )
    if veto2.any():
        final = _inf_gate_restore(before_cls2, final, prop2[veto2])
    n_inf_final = final.infinity_point_count
    # Rebuild ALL infinity frames at the detection angular size: idempotent, so
    # it does not matter how many demotion passes (or finite<->infinite round
    # trips) a point went through on the way here.
    final, n_reframed_ba = _normalize_infinity_frames(
        final, det_size, data["refine_radius"]
    )
    stage_dump("infinity-postBA", final)

    # Re-check cheirality AFTER the BA: the adjustment is the last thing that
    # can push a point across a camera plane (it moves both the structure and
    # the poses), so a set that was clean going in is not guaranteed clean
    # coming out.  Cull again before the caller saves.
    n_post = final.point_count
    keep_ch2 = _cheirality_keep(final)
    if not keep_ch2.all():
        keep_ch2 = np.ascontiguousarray(keep_ch2, dtype=bool)
        final = final.filter_points_by_mask(keep_ch2)
        det_size, mutated, uid = det_size[keep_ch2], mutated[keep_ch2], uid[keep_ch2]
        n_post = final.point_count
    stage_dump("cheirality-postBA", final)

    # Reprojection cull, re-checked at the final geometry alongside the infinity
    # and cheirality re-checks and for the same reason: the BA is the last thing
    # that moves a point, and a point it could not satisfy is debris however well
    # it scored going in.
    keep_rp2, rp_bound2, n_rp2 = _reprojection_cull_mask(final)
    if n_rp2:
        keep_rp2 = np.ascontiguousarray(keep_rp2)
        final = final.filter_points_by_mask(keep_rp2)
        det_size, mutated, uid = det_size[keep_rp2], mutated[keep_rp2], uid[keep_rp2]
        n_post = final.point_count
    print(
        f"  reprojection cull (post-BA): {n_rp2} points over "
        f"{rp_bound2:.2f} px median reprojection"
    )
    stage_dump("reproj-cull-postBA", final)

    # THE ARS FIT, on the cloud that ships (ARS_LATE_FIT).  Every membership
    # decision is behind it — the BA has adjusted the positions, the eviction has
    # disowned what the photometry disowned, the collapse/alias/contained passes
    # have merged and culled, and the cheirality and reprojection re-checks have
    # taken what the adjustment broke — so the adjacency graph is built over
    # exactly the neighbours that exist in the file, and the promoted candidates
    # that survived all of that take part in it as ordinary points and get their
    # own fitted normal and their own confidence rather than the parent frame
    # they were born with.  It is deliberately NOT given extras: the expansion is
    # a congealing pass over images and the early site already spent it, so what
    # the late graph cannot determine is reported as weak or fronto-parallel
    # rather than propped up a second time.
    #
    # The two consumers that still follow are the infinity reframe (idempotent,
    # and disjoint from the fit — the fit writes finite rows only) and the bitmap
    # re-fuse, which is what the frame turns mark mutated for.
    if ARS_LATE_FIT:
        final, ars_late, mutated, det_size, uid = _ars_normals(
            final, images, det_size, data["refine_radius"], mutated, uid, mode="fit"
        )
        _print_ars("post-BA", ars_late)
        final, _ = _normalize_infinity_frames(final, det_size, data["refine_radius"])
        stage_dump("ars-late-fit", final)

    # Re-fuse the consensus bitmaps of every point a merge or a reassignment
    # touched, from its CURRENT track.  Last, so one pass covers both sites.
    t_bmp = time.perf_counter()
    n_mut = int(mutated.sum())
    final, n_fresh, n_clear = _rerender_mutated_bitmaps(final, images, mutated)
    print(
        f"  patch bitmaps: {n_mut}/{final.point_count} mutated points re-rendered "
        f"({n_fresh} re-fused, {n_clear} cleared - no valid consensus) in "
        f"{time.perf_counter() - t_bmp:.1f}s; {final.point_count - n_mut} kept "
        f"byte-identical"
    )
    stage_dump("refuse-bitmaps", final)

    # LATE VETTING, on the artifact as it now stands.  Everything above this line
    # was vetted before the BA, against frames the late fit has since re-aimed and
    # bitmaps the re-fuse has since re-rendered through them; this is the first
    # site at which the frames, the positions and the consensus under test are the
    # ones that ship.  See _late_vet — the track-view eviction replayed at final
    # geometry, then the localizability criterion (sigma_pos plus the anisotropy
    # bound) re-applied to the shipped consensus.
    n_pre_vet = final.point_count
    final, lv, det_size, mutated, uid = _late_vet(final, images, det_size, mutated, uid)
    if LATE_VET:
        _print_late_vet(lv)

    # Derived data is refreshed before the caller can save it, unconditionally.
    final = _refresh_normals(final)
    final = _refresh_errors(final)
    _check_frame_contract(final, "save")
    stage_dump("final", final)
    rk = np.asarray(out["residual_norms"])
    rk = rk[np.isfinite(rk)]
    print(
        f"  seed finalize: {n0} -> embed {n_emb} -> len3 {n_len3} -> "
        f"size-cull {n_sizecull} -> cheirality {n_cheir} -> collapse "
        f"{n_collapse} pts ({coll['merged']} merged away in "
        f"{coll['groups']} groups) -> alias-reconcile {n_alias} "
        f"({rec['culled']} culled) -> reproj-cull {n_reproj} -> "
        f"infinity {n_inf}; {n_camcull} cams "
        f"(covis-culled); ARS promotion +{n_promoted} -> "
        f"{n_pre_promote + n_promoted} pts (promoted survivors "
        f"{n_prom_evict} post-eviction, {_n_promoted_alive(uid)} shipped); "
        f"native BA f={out['focal']:.1f}, reproj median "
        f"{np.median(rk):.2f} px -> post-BA collapse -{coll_ba['merged']} "
        f"/ alias -{rec_ba['culled']} / contained -{cont_ba['culled']} "
        f"-> infinity {n_inf_ba} -> "
        f"{n_inf_final} -> cheirality/reproj {n_post} pts "
        f"({final.infinity_point_count} at infinity); bitmaps refreshed"
        + (f" -> late vet {n_pre_vet} -> {final.point_count} pts" if LATE_VET else "")
    )
    return final


def finalize_seed_from_dict(data, seed):
    """Finalize an in-memory stage-1 seed into a bitmap-bearing reconstruction,
    with NO JSON read or written.  ``seed`` is a dict with ``focal_structure_px``,
    ``posed_images`` (image names) and ``rvec``/``tvec`` (one canonical
    world->camera row per posed image).  Plants the poses by name onto the
    cluster data, triangulates the initial structure, and runs the photometric
    finalization (embed-patches expand + congeal + consensus bitmaps -> drop
    length-2 -> native BA).  This is the seed stage's terminal step: it always
    finalizes — a confidence flag does not abort here; the caller records the
    flag alongside the artifact.  Returns the finalized reconstruction."""
    global _CAM_WH
    _CAM_WH = tuple(data["dims"][0])
    all_c, all_i, all_u = data["obs_c"], data["obs_i"], data["obs_uv"]
    n_img, n_cl = data["n_img"], data["n_cl"]
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
        print(f"WARNING: {len(missing)} seed images absent from matches names")
    rot = Rotation.from_rotvec(rvec).as_matrix()
    pts = triangulate(all_c, all_i, all_u, rot, tvec, posed, n_cl, f)
    print(
        f"\nseed: f = {f:.1f} px, {int(posed.sum())} poses planted, "
        f"{int((~np.isnan(pts[:, 0])).sum())}/{n_cl} clusters triangulated"
    )
    # Flag-aware finalization focal policy.  edge_scan means the structure
    # provably has no interior optimum (monotone scan) — freeze at the
    # structure-free vote outright.  Any OTHER flag means stage 1 was unsure
    # but not which side to trust: run the dual-candidate census arbitration
    # (frozen-at-vote vs guarded-free BA) inside _finalize_seed.
    f_final = None
    arb = None
    vote = seed.get("focal_vote_px")
    # THE SEED DICT'S OWN LIST, installed when absent: an UNFLAGGED seed must
    # still be able to come back flagged, and `or []` would hand the
    # finalization a throwaway copy whose appends the caller never sees.
    flags = seed.get("confidence_flags")
    if not isinstance(flags, list):
        flags = list(flags or [])
        seed["confidence_flags"] = flags
    if vote and flags:
        if "edge_scan" in flags:
            f_final = float(vote)
        else:
            arb = float(vote)
    # ``flags`` is the seed dict's own list — the census endorsement gate
    # appends ``census_guard`` to it in place, and the accepted candidate's
    # census appends ``census_echo``, so the caller's metadata write sees both
    # without any return-value plumbing.
    return _finalize_seed(
        data,
        None,
        rvec,
        tvec,
        f,
        pts,
        posed,
        f_final=f_final,
        arbitrate_vote=arb,
        arbitrate_center=seed.get("focal_vote_center_px"),
        arbitrate_band=seed.get("focal_vote_band_log"),
        flags=flags,
    )


def load_seed(path):
    """Load a stage-1 seed as a dict {focal_structure_px, posed_images, rvec,
    tvec, confidence_flags} from the finalized reconstruction
    (sfmr/seed-final.sfmr — the current artifact) or, for a not-yet-migrated
    workspace, a legacy fast-pinhole.json.  Poses come back as canonical
    world->camera rvec/tvec — the frame external_seed_complete plants.  From the
    .sfmr the seed is the FINALIZED pose set (embed-expanded, congealed, BA'd at
    the refined focal), a stronger starting point than the raw stage-1 seed."""
    p = Path(path)
    if p.suffix == ".json":
        s = json.loads(p.read_text())
        return {
            "focal_structure_px": float(s["focal_structure_px"]),
            "posed_images": list(s["posed_images"]),
            "rvec": s["rvec"],
            "tvec": s["tvec"],
            "confidence_flags": s.get("confidence_flags") or [],
        }
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    rec = SfmrReconstruction.load(str(p))
    q = np.asarray(rec.quaternions_wxyz)
    rvec = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec()
    meta = rec.metadata()
    cf = (meta.get("tool_options") or {}).get("confidence_flags", "") if meta else ""
    flags = [] if cf in ("", "ok") else cf.split(",")
    return {
        "focal_structure_px": float(rec.cameras[0].focal_lengths[0]),
        "posed_images": list(rec.image_names),
        "rvec": rvec.tolist(),
        "tvec": np.asarray(rec.translations).tolist(),
        "confidence_flags": flags,
    }


def external_seed_complete(
    data,
    seed,
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
):
    """Complete a stage-1 fast-pinhole seed instead of searching for one.

    Activates every cluster up front (the external seed replaces the tier-0
    search), plants the seed poses at ``focal_structure_px``, triangulates,
    grows the rest with one next-best-view pass at fixed f, then releases f.
    Returns the same tuple the tiered path leaves for the assemble tail
    (f, rvec, tvec, pts, posed, ok, keep, res), or None to abort (a flagged
    seed without SFMTOOL_FORCE=1)."""
    flags = seed.get("confidence_flags") or []
    if flags and os.environ.get("SFMTOOL_FORCE") != "1":
        print(
            f"\nseed is FLAGGED {flags}; refusing to complete an "
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
        print(
            f"WARNING: {len(missing)} seed images absent from matches names: "
            f"{missing[:5]}"
        )
    print(
        f"\nexternal seed: f = {f:.1f} px, planted {int(posed.sum())}/"
        f"{len(seed['posed_images'])} seed poses onto {n_img} images "
        f"[{int(active_cl.sum())} clusters, {len(obs_c)} observations]"
    )
    compare_to_reference(data["names"], rvec, tvec, f, mask=posed)

    # Triangulate the initial structure from the planted poses at fixed f.
    rot = Rotation.from_rotvec(rvec).as_matrix()
    pts = triangulate(obs_c, obs_i, u, rot, tvec, posed, n_cl, f)
    print(
        f"  triangulated {int((~np.isnan(pts[:, 0])).sum())}/{n_cl} clusters "
        f"from the seed poses"
    )
    snap = make_snapshotter(data)
    if snap is not None:
        snap("seed", f, rvec, tvec, pts, posed)

    # Seed finalization (opt-in, SFMTOOL_SEED_CULL=1): run the full embed-patches
    # photometric pipeline on the seed — expand the view set, congeal + sub-pixel
    # refine the keypoints (rendering consensus bitmaps), cull length-2 survivors,
    # and native-BA on the refined keypoints. This is terminal: the finalized
    # bitmap-bearing reconstruction is the output, so growth is skipped (the
    # growth integration is future work; for now this produces the seed artifact
    # for inspection).
    if os.environ.get("SFMTOOL_SEED_CULL") == "1":
        final = _finalize_seed(data, active_cl, rvec, tvec, f, pts, posed)
        out = WS / "sfmr" / os.environ.get("SFMTOOL_OUT", "bootstrap-pinhole.sfmr")
        final.save(str(out), operation="seed-finalized")
        print(f"\nwrote {out} ({final.point_count} points, seed-finalized w/ bitmaps)")
        return None

    # Seed-only (SFMTOOL_SEED_ONLY=1): return the seed straight to the dense save,
    # skipping growth — for inspecting the initial bootstrap (optionally
    # finalized) across the fleet. main() overwrites keep/res via
    # dense_structure, so placeholders suffice.
    if os.environ.get("SFMTOOL_SEED_ONLY") == "1":
        print(f"  seed-only: {int(posed.sum())} posed, skipping growth")
        dummy = np.zeros(len(all_c), bool)
        return f, rvec, tvec, pts, posed, dummy, dummy, np.full(len(all_c), np.inf)

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
        print(
            f"  spread backbone: growing {int(bb.sum())} covisibility-thinned "
            f"images (of {n_img})"
        )
        rvec, tvec, pts, posed = grow_loop(
            rvec,
            tvec,
            pts,
            posed,
            f,
            obs_c[bbm],
            obs_i[bbm],
            u[bbm],
            n_img,
            n_cl,
            covis,
            aux=aux_bb,
            ba=(bam[bbm] if bam is not None else None),
            snap=snap,
        )
    else:
        rvec, tvec, pts, posed = grow_loop(
            rvec,
            tvec,
            pts,
            posed,
            f,
            obs_c,
            obs_i,
            u,
            n_img,
            n_cl,
            covis,
            aux=aux,
            ba=bam,
            snap=snap,
        )
    if snap is not None:
        snap(f"grow-{int(posed.sum()):03d}-done", f, rvec, tvec, pts, posed)
    print(
        f"  [after grow: {int(posed.sum())}/{n_img} posed "
        f"(+{int(posed.sum()) - posed_before}) at "
        f"{time.perf_counter() - _T0:.0f}s]"
    )
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
            obs_c[okb],
            obs_i[okb],
            u[okb],
            rot,
            tvec,
            pts,
            f,
            n_img,
            n_cl,
            opt_f=True,
        )
        pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f)
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        xc = (
            Rotation.from_rotvec(rvec[obs_i[ok]]).apply(pts[obs_c[ok]])
            + tvec[obs_i[ok]]
        )
        rn = np.linalg.norm(
            make_cam(f).ray_to_pixel_batch(np.ascontiguousarray(xc)) - u[ok], axis=1
        )
        keep = rn < TRIM_PX
        res = np.where(np.isnan(rn), np.inf, rn)
        inl = float((res < 2.0).mean())
    else:
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        f, rvec, tvec, pts, keep, res, inl = bundle_adjust(
            obs_c[ok],
            obs_i[ok],
            u[ok],
            rot,
            tvec,
            pts,
            f,
            n_img,
            n_cl,
            opt_f=True,
        )
    if snap is not None:
        snap(f"release-f-{f:.0f}", f, rvec, tvec, pts, posed)

    if backbone > 0 and int(posed.sum()) < n_img:
        # Complete the structure at the backbone poses, then batch-resect the
        # remaining frames against it (no BA).
        pts = fill_new_points(pts, obs_c, obs_i, u, rvec, tvec, posed, f)
        pb = int(posed.sum())
        rvec, tvec, pts, posed, nacc = batch_resect(
            rvec, tvec, pts, posed, f, obs_c, obs_i, u, n_img, n_cl, covis, aux
        )
        print(
            f"  [batch-resect: +{nacc} frames ({pb} backbone -> "
            f"{int(posed.sum())}/{n_img}) at {time.perf_counter() - _T0:.0f}s]"
        )
        ok = posed[obs_i] & ~np.isnan(pts[:, 0])[obs_c] & bam
        xc = (
            Rotation.from_rotvec(rvec[obs_i[ok]]).apply(pts[obs_c[ok]])
            + tvec[obs_i[ok]]
        )
        proj = make_cam(f).ray_to_pixel_batch(np.ascontiguousarray(xc))
        rn = np.linalg.norm(proj - u[ok], axis=1)
        keep = rn < TRIM_PX
        res = np.where(np.isnan(rn), np.inf, rn)
        inl = float((res < 2.0).mean())
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
    global _PXS
    if os.environ.get("SFMTOOL_FRAC_DIAG") == "1":
        _PXS = max(1.0, float(np.hypot(*_CAM_WH)) / 550.6)
        print(f"fractional thresholds: scale {_PXS:.2f} (diag ref 550.6)")
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
        print(f"BA set: best {MAX_CLUSTERS} of {n_cl} clusters by span")
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
    seed_src = os.environ.get("SFMTOOL_SEED")
    if not seed_src:
        cand = WS / "sfmr" / "seed-final.sfmr"
        if cand.exists():
            seed_src = str(cand)
        elif (WS / "fast-pinhole.json").exists():
            seed_src = str(WS / "fast-pinhole.json")  # legacy, pre-migration
    if not seed_src:
        sys.exit(
            "no seed found: run scripts/exp_fast_seed.py on this workspace "
            "first (it writes sfmr/seed-final.sfmr), or pass SFMTOOL_SEED=<path>"
        )
    r = external_seed_complete(
        data,
        load_seed(seed_src),
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
        return
    f, rvec, tvec, pts, posed, ok, keep, res = r

    pts, keep, res = dense_structure(all_c, all_i, all_u, f, rvec, tvec, pts, posed)
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
