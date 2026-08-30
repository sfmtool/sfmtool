# Photometric track widening — phase-1 experiment notes (2026-07-19)

`scripts/exp_track_widen.py` — post-growth widening pass on a completed
bootstrap reconstruction, testing the thesis that drift is caused by NARROW
tracks (small angular span of observing rays), and that photometric patch
localization (warp-compensated ZNCC) can manufacture wide-parallax
observations in frames SIFT never matched, giving BA long-range constraints
that shrink global (gauge) error.

## Setup

- Datasets (GT = first `*-solve-*.sfmr` in `ws/sfmr/`):
  - `SpainSoapmaker/ws2` — loop-rich, residual drift (bootstrap global 16.4%)
  - `PhotogrammetryVids/20240614_224244438` — open path, pure gauge wander
    (global 34.9%, piecewise 0.21%)
  - `PhotogrammetryVids/20240915_073131082` — clean sweep, regression guard
    (global 0.48%)
- Reconstruction reused: `sfmr/bootstrap-pinhole.sfmr` (poses + shared
  SIMPLE_PINHOLE focal) + the matching `*-clusters-patches.matches`
  observation universe (status-0/1 members, valid reference, posed span >= 2,
  best 15000 clusters by span — the completion's active set); points
  re-triangulated from the bootstrap poses (ray-midpoint batch kernel).
- Track narrowness: angular span = max pairwise angle of the world rays
  camera-center -> point over the members.
- Selection (500 tracks, evenly sampled across the span-sorted qualifiers):
  finite point, >= 6 members, span < 5 deg, member ZNCC >= data-derived floor
  (25th pct), member reproj median < 4 px, implied depth <= 3 rig diameters
  (anti-infinity), and span >= 10x the track's angular noise floor
  `atan(max(reproj_med, 1 px) / f)` (below it the triangulated depth — and
  every candidate angle computed from it — is fiction; relative depth error
  ~ noise/span).
- Candidate extension views: posed frames not in the track, point projects
  in front + in bounds (5% margin), viewing angle vs the track mean ray >
  the track's span (that is the widening) and < 60 deg grazing vs the
  mean-viewing patch normal; up to 3 per angular bucket
  (0-5 / 5-10 / 10-20 / 20-40 / 40+ deg).
- Localization: `PatchCloud.from_tracks` (keypoint scales = `.sift` affine
  first-column norms, `extent="feature_size"`, 2.5) + per-track view-subset
  normal refinement (K=8 members, evenly spread; it measurably improved
  wide-angle keeps in an A/B on ws2) + `localize_keypoints` over members
  (<= 12, evenly spread) + candidates, `max_shift_px=60`, `search=12`,
  `min_relative_zncc=0.6`; pyramid sets chunked at <= 60 4K images
  (~1.4-2.2 s / chunk including decode; ~11-29 min / dataset end to end).
- BA variants (staged native `bundle_adjust`, `opt_f=True`, schedule
  `[(50s, 5s), (12s, 2s), (4, 1)]`, `s = diag/550.6`, all from the bootstrap
  state):
  - `control_ba` — original observations only
  - `widened_ba` — + localized candidate obs passing the < 4 px reprojection
    gate at the bootstrap pose
  - `widened_all_ba` — + every localized candidate obs (no delta gate; BA's
    staged trim is the arbiter)
- Metrics vs GT: rotation median (rotation-fitted gauge), global similarity
  center error and piecewise-k8 center error (medians, % of GT rig
  diameter), focal error. Spans "after" recomputed at the refined geometry
  over the obs surviving the 4 px trim.

## Results

### Acceptance vs angular distance (offered -> localized -> accepted <4 px; noop = delta < 0.1 px)

ws2 (delta_px p50/p90 in parens):

| bucket | offered | localized | accepted | noop |
|--------|---------|-----------|----------|------|
| 0-5    | 1261 | 763 (61%) | 196 (16%) | 1 | (7.0/20.5) |
| 5-10   | 1433 | 550 (38%) | 89 (6%)   | 0 | (8.8/25.4) |
| 10-20  | 1356 | 378 (28%) | 56 (4%)   | 0 | (9.3/30.0) |
| 20-40  | 1161 | 270 (23%) | 43 (4%)   | 0 | (9.2/26.1) |
| 40+    |  885 | 197 (22%) | 57 (6%)   | 1 | (7.2/25.6) |

20240915 mirrors ws2 (localized 88% -> 42%, accepted 29% -> 4%, delta p50
rising 6.2 -> 13.5 px with angle; noop ~ 0). 20240614 is qualitatively
different: localized ~85%, accepted ~85%, but **80-99% of localizations are
no-ops** (delta exactly 0 at the search lattice) — see failure analysis.

So: warp-compensated ZNCC **can** localize a refined patch 20-60 deg off the
track's mean ray (roughly a fifth to a quarter of geometrically-visible
candidates on real drifted video, LOO ZNCC ~ 0.87 median), and on drifted
data the localized position sits 7-14 px from the current projection —
which is precisely the corrective signal, and precisely what the < 4 px
acceptance gate throws away (the gate passes ~5%, mostly near-angle).

### Camera errors vs GT (medians; global/piecewise as % of rig diameter)

ws2 (loop-rich, drifted):

| state | rot med | global % | piecewise % | focal % | new obs (survive 4 px) |
|-------|---------|----------|-------------|---------|------------------------|
| bootstrap    | 4.33 | 16.4 | 4.05 | -1.9 | |
| control_ba   | 4.43 | 29.0 | 0.82 | -0.9 | |
| widened_ba   | 2.90 | 30.0 | 0.32 | -0.9 | 441 (173) |
| widened_all  | 2.74 | 30.1 | 0.18 | +0.7 | 2158 (241) |

20240614 (open path):

| state | rot med | global % | piecewise % | focal % | new obs (survive) |
|-------|---------|----------|-------------|---------|-------------------|
| bootstrap    | 4.16 | 34.9 | 0.21 | -5.5 | |
| control_ba   | 3.00 | 34.9 | 0.15 | -5.5 | |
| widened_ba   | 3.67 | 34.9 | 0.15 | -6.1 | 5580 (1474) |
| widened_all  | 4.37 | 34.9 | 0.19 | -6.4 | 5782 (1483) |

20240915 (clean sweep):

| state | rot med | global % | piecewise % | focal % | new obs (survive) |
|-------|---------|----------|-------------|---------|-------------------|
| bootstrap    | 0.36 | 0.48 | 0.18  | -0.17 | |
| control_ba   | 0.13 | 0.29 | 0.007 | -0.13 | |
| widened_ba   | 0.13 | 0.29 | 0.007 | -0.13 | 980 (776 = 79%) |
| widened_all  | 0.13 | 0.28 | 0.008 | -0.15 | 4818 (2026 = 42%) |

### Span distributions (selected tracks, deg)

| dataset | before | control after | widened after | widened_all after |
|---------|--------|---------------|---------------|-------------------|
| ws2      | med 1.7, p90 4.1 | 1.3 / 3.2 | 1.4 / 6.1 | 1.7 / 10.7 |
| 20240614 | med 1.7, p90 4.4 | 1.4 / 4.1 | 74.5 / 178 (!) | 33.5 / 109 (!) |
| 20240915 | med 3.4, p90 4.7 | 3.4 / 4.7 | 6.3 / 24.4 | 15.1 / 52.2 |

The whole-universe span distribution barely moves anywhere (15000 tracks,
500 widened): med 3.4 -> 5.0-5.8 (ws2), 11.7 -> 11.8-12.0 (0915).

## Verdict: thesis NOT supported in the post-hoc (phase-1) form

The machinery works; the drift correction does not materialize.

1. **Widening is real where geometry is good.** On 20240915, 79% of the
   gated new observations survive BA's 4 px trim and the selected tracks'
   span really does shift right at flat count (med 3.4 -> 6.3 -> 15.1 deg).
   No regression on any metric (global 0.287 -> 0.280). The
   localize-at-wide-angle machinery is validated end to end.
2. **On the drifted loop-rich capture (ws2) BA rejects the corrective
   signal.** The wide-angle localizations disagree with the current
   geometry by 7-14 px (that disagreement IS the drift information), but
   they are ~1-2% of observations: the robust staged trim treats them as
   outliers (11-40% survival) and re-converges within the drift gauge.
   Global error does not improve (29.0 -> 30.0); the gains are piecewise
   (0.82 -> 0.18, 4.6x) and rotation (4.43 -> 2.74 deg median), i.e. local
   rigidity, not gauge repair.
3. **On the open path (20240614) there is nothing to widen against.** The
   narrow high-count tracks are CLOSE points being walked past (implied
   depth ~ 0 relative to the rig bbox — which is itself inflated by scale
   drift). Wide VIEWING ANGLES on close points come from temporally-near
   frames, where the geometry is already locally perfect: 80-99% of
   localizations land exactly on the projection (noop), carrying zero
   corrective information. The surviving no-op obs actively hurt: focal
   -5.5% -> -6.4%, rotation 3.0 -> 4.4 deg, and the selected points get
   dragged into physically impossible 90-180 deg spans (near-field
   collapse) while camera metrics stay flat.
4. **The angular span is the wrong currency for long-range constraints.**
   What BA needs to kill gauge wander is coupling between cameras far apart
   along the trajectory (observer-baseline span / temporal separation), not
   rays far apart in angle. Close points give wide angles over seconds of
   video; distant points give long temporal reach at near-zero angle (pure
   rotation anchors, no scale). Angle-bucketed widening optimizes the wrong
   axis on open paths, and on loop-rich paths the wide-angle obs it does
   manufacture are the minority that robust BA votes out.
5. **A flat full re-BA is itself gauge-destructive on drifted data** —
   control alone moved ws2's global error 16.4% -> 29.0% while halving the
   local error. The windowed+anchored growth state was globally better than
   its own flat re-optimization at 4 px trim. Any phase-2 scheme that
   re-optimizes globally must reckon with this: local tightness and global
   correctness are different objectives once drift exists.

### Selection lessons (paid for in reruns)

- Span-vs-noise conditioning is mandatory: a track whose span is below its
  angular noise floor (`atan(reproj_px/f)`) triangulates to arbitrary depth
  — on an open path the ray-midpoint collapses onto the camera path, and
  every "candidate angle" computed from that depth is fiction. Gate:
  span >= 10x noise floor (~<=10% depth error). An implied-depth cap
  (<= 3 rig diameters) guards the mirror infinity case; neither gate alone
  rescued 20240614 because its close-point/no-op failure is geometric, not
  a triangulation artifact.
- The < 4 px acceptance gate is self-defeating on drifted data: it accepts
  exactly the observations that already agree with the drifted geometry.
  The ungated variant (BA-trim as arbiter) widened more and improved
  piecewise/rotation more on ws2 — but still could not move the global
  gauge.

### Where this leaves the program

Post-hoc widening + flat re-BA is falsified as a drift cure. The surviving
hypotheses, in order of plausibility: (a) extend DURING growth near anchors
(phase 2) so wide constraints exist before drift accrues and are never a
trimmed minority; (b) select for observer-baseline (temporal) reach rather
than angular buckets — on loops that means explicit cross-sweep revisit
candidates (displacement k-NN), on open paths accept that scale gauge is
unobservable and target rotation anchoring via distant points instead;
(c) if post-hoc repair is wanted, the widened obs need a non-uniform trim
(e.g. exempt appearance-verified wide obs from the robust vote, or
gauge-level solves like similarity-per-window on the widened graph) rather
than competing 1:100 in a flat robust BA.

> _Status (2026-08-12): The revisit candidates in (b) no longer come from a
> `displacement-knn.npz` sidecar — that cache was measured unread, stale on
> 36/41 fleet workspaces and removed. `exp_widen_growth.load_clusters_ext`
> derives the same shared-cluster counts per posed image pair in memory from
> the `.matches` read it already does; the hypothesis is unchanged._

Artifacts: `exp_track_widen.py` prints all tables; per-dataset JSON dumps
(span distributions, acceptance curve with noop counts, LOO-ZNCC, evals)
land in `--out`. Runs here: 500 tracks/dataset, `--refine-normals`, wall
11-29 min/dataset alongside a fleet eval.

## Follow-up (2026-07-19): fragment decomposition of the BA arms

Upstream `sfm compare` gained RANSAC similarity decomposition into rigid
fragments (#227). Each arm's poses were saved as
`ws/sfmr/track-widen-<arm>.sfmr` (clone of the bootstrap with
`quaternions_wxyz`/`translations` replaced) and scored with
`sfm compare <GT> <arm> --fragments` (defaults: pos < 3.5% scene scale,
rot < 5 deg, min component 5).

ws2 (301 shared cameras) — dominant-component stats are pos mean/med (% of
scene scale) and rot mean/med (deg):

| arm | comps | top-3 sizes | outliers | dom pos | dom rot |
|-----|-------|-------------|----------|---------|---------|
| bootstrap    | 15 | 87/44/26  | 54 | 1.50/1.53 | 0.52/0.32 |
| control_ba   | 10 | 68/57/36  | 26 | 1.23/0.95 | 0.41/0.35 |
| widened_ba   |  7 | 92/83/56  | 21 | 0.77/0.69 | 0.26/0.20 |
| widened_all  |  5 | 194/60/13 | 17 | 1.77/1.73 | 0.41/0.39 |

20240614 (292):

| arm | comps | top-3 sizes | outliers | dom pos | dom rot |
|-----|-------|-------------|----------|---------|---------|
| bootstrap    | 8 | 133/38/16 | 57 | 0.93/0.88 | 1.04/0.76 |
| control_ba   | 6 | 88/59/51  | 46 | 1.27/1.08 | 0.31/0.23 |
| widened_ba   | 6 | 89/85/19  | 66 | 0.86/0.70 | 0.27/0.19 |
| widened_all  | 8 | 90/53/34  | 63 | 0.74/0.60 | 0.22/0.17 |

20240915 (368):

| arm | comps | top-3 sizes | outliers | dom pos | dom rot |
|-----|-------|-------------|----------|---------|---------|
| bootstrap    | 2 | 356/9 | 3 | 1.18/1.10 | 0.33/0.23 |
| control_ba   | 1 | 367   | 1 | 0.09/0.07 | 0.06/0.05 |
| widened_ba   | 1 | 367   | 1 | 0.09/0.07 | 0.06/0.05 |
| widened_all  | 1 | 367   | 1 | 0.19/0.18 | 0.05/0.04 |

Reading:

1. **On the loop-rich capture, widening consolidates rigid structure — the
   medians hid it.** ws2 components fall monotonically with injected
   observations (15 -> 10 -> 7 -> 5), outliers 54 -> 17, and the dominant
   rigid fragment grows 87 -> 194 cameras (29% -> 64% of the shared set)
   under widened_all, at healthy internal stats (pos mean 1.77%, rot
   0.41 deg). The whole-set global median (~30% in every arm) is dominated
   by the remaining inter-fragment scale/displacement breaks, so it was
   blind to this. The long-range photometric constraints ARE gluing
   fragments together; they fail only to close the last inter-fragment
   gauge gaps. This partially rehabilitates the widening thesis: the effect
   is real and structural, and the earlier "BA rejects the minority"
   verdict applies to the residual breaks, not to the whole solve.
2. **The flat control re-BA does not multiply ws2's fragments — it evens
   them out.** Components drop 15 -> 10 and outliers halve, but the
   dominant fragment SHRINKS (87 -> 68; 29% -> 23%) and the top three
   become near-equal (68/57/36). That is the fragment-level signature of
   the gauge-destructive-BA finding: local tightening (dom pos med 1.53 ->
   0.95) at the cost of breaking the largest rigid group — and with more
   equal fragments no single gauge fits most cameras, which is exactly why
   the global median jumped 16.4% -> 29.0%.
3. **Open path: no consolidation, mild damage.** 20240614's dominant
   fragment shrinks under every re-BA arm (133 -> ~89) and widening adds
   nothing structural (6-8 components throughout) while pushing outliers up
   (46 -> 66/63) — consistent with its no-op observations carrying no
   long-range information.
4. **Regression guard clean.** 20240915: control alone reaches one
   component with near-perfect internals; widened matches it bit-for-near;
   widened_all only slightly loosens position internals (0.09 -> 0.19%).

Phase-2 implication sharpened: the widening constraints demonstrably merge
rigid fragments post-hoc on loop-rich captures; the remaining failure is
inter-fragment scale/gauge alignment, which is a low-dimensional problem
(a handful of similarity transforms), not a per-camera one. Widen-then-
fragment-align (or widening during growth) looks more promising than
widen-then-flat-BA.
