# Cluster pinhole bootstrap — experiment notes

**Status: idea-stage notes.** Experiments with
`scripts/exp_pinhole_bootstrap.py` (first run: seoul bull sculpture; a
multi-dataset campaign log is at the bottom). Nothing here is production
design; numbers are from single runs.

## Problem

Starting from images and a cluster-patch `.matches` file (sift extraction →
cluster matching → cluster-patches), and using only a pinhole camera model:
how effectively can we bootstrap a coarse 3D reconstruction into a `.sfmr`
file — with no COLMAP solver, no pairwise two-view geometry, and no prior
intrinsics?

The input observations are the patch clusters' refined member positions
(the stored affine warp applied to the reference keypoint,
`x_m = A·x_ref + t`). Each cluster becomes one candidate track; members with
status `reference`/`kept` become its observations.

## Result (seoul bull, 17 images @ 270×480, ~150° orbit)

2,889 clusters (span ≥ 2), 7,193 observations. Reference: the incremental
COLMAP solve in the same workspace (944 points, 3,191 obs, SIMPLE_RADIAL
f = 332.6, k1 = +0.013).

A useful ceiling: triangulating every cluster against the *reference*
cameras explains 72% of the observations at < 2 px (median 0.29 px) — the
other ~28% of cluster members are junk that no cameras can explain. Per-image
ceilings range 54–83%.

The bootstrap (single run, ~3.3 min in scipy, of which ~2.7 min is a
5-candidate focal scan):

- **78.5% of all observations < 2 px** (kept-set rms 0.39 px, median
  0.17 px) — at the ceiling; the small excess over 72% is the pinhole
  absorbing part of the reference's k1.
- Cameras vs reference after similarity alignment: **rotation err mean 2.9°
  (max 5.1°), center err mean 1.6% of scene diameter (max 2.9%)**.
- Focal recovered **345.7 px vs 332.6 px reference** (+3.9%, plausibly the
  k1 trade-off) from a blind five-point grid over 0.55–1.6 × max(w, h).
- 2,319 points / 5,674 observations written as a valid
  `cluster_bootstrap` `.sfmr` (SIMPLE_PINHOLE, canonical convention,
  integrity OK).

## Method that survived

(As first built for seoul this used the video's frame order — consecutive
windows and constant-velocity resection inits; the campaign version below
replaced every use of sequence order with cluster covisibility, with
identical results on seoul.)

1. **Seed-group affine factorization.** Candidate seed groups grow greedily
   from the strongest image-covisibility edge (shared-cluster counts),
   maximizing the minimum shared count against the group. ALS
   (missing-data-tolerant, trimmed) weak-perspective factorization +
   Tomasi–Kanade metric upgrade per group, both reflection hypotheses kept.
2. **Seed + incremental growth.** Seed a perspective solve on the best
   group with a small fixed-focal BA (its inlier fraction also resolves
   the reflection: 93.7% vs 66.4% on the seoul seed). Then repeatedly pose
   the next-best-view image (most observations of valid points): trimmed
   pose-only resection initialised from its most-covisible posed images →
   DLT for clusters that now have ≥ 2 posed views → a short global BA
   every few images.
3. **Focal by outer scan, not by BA.** Run the whole growth at each
   candidate focal with f held fixed; the all-observation inlier fraction
   peaks near the true focal (41.8 / 67.4 / **78.2** / 66.6 / 54.6% across
   the grid). Release f only in the final BA of the winner — it then moves
   432 → 345.7 and stays put.
4. **Staged robust BA throughout**: trim gross outliers and behind-camera
   observations before each solve, re-triangulate every cluster from the
   refined cameras between rounds (re-admission), thresholds 50 → 12 → 4 px.

## What failed on the way (the actual findings)

- **A single global factorization over the orbit**: rotations 50–90° wrong
  before BA (the sequence spans ~150°; weak-perspective holds only over
  ~40° windows). Known from the earlier bootstrap experiments; confirmed.
- **Free-focal BA from a weak init escapes to the affine limit.** With
  median ~14 px init error, the reprojection residual decreases
  monotonically as f → ∞ at init, and a free-f BA slides 576 → 2435 px while
  *improving* its kept-set rms to 1.1 px — a self-consistent telephoto
  collapse fitting 48% of the data. The wrong reflection hypothesis
  survives this way too (reflection is unobservable in the affine limit).
  Hence the fixed-f outer scan.
- **Chaining independently-solved windows drifts.** Registering windows by
  similarity on shared cluster points accumulated 83° mean rotation error
  across the orbit — the sparse middle windows (~90–150 factorizable
  clusters) are 17–37° wrong even after a window BA, and everything after
  them inherits the error. Growth by resection against the *global*
  structure replaced it.
- **Plain robust resection fails from a one-frame-away init.** With ~100
  observations, an adjacent-frame init (~10° off, median residual ~50 px)
  and ~17% junk: an L2 warm-up is dragged by the junk's leverage, and
  soft_l1 has near-zero gradient when every residual starts as an
  "outlier" — both land ~21° wrong with ~1% inliers. Trimmed iterations
  (refit L2 on the best-fitting 60%, five times) reach 0.05° / 83% inliers
  on the same input. (COLMAP uses RANSAC P3P here for the same reason.)

## Campaign log (2026-07-12)

Running log of the multi-dataset campaign; entries appended as tried.

- **Campaign datasets** (picked from `test-data/` and `C:/DataSets` for
  contrast): seoul bull (17-frame 270×480 phone video, ~150° orbit —
  baseline), dino_dog_toy (85 unordered 2040×1536 photos, tabletop object),
  COLMAP south-building (128 photos 3072×2304, building + vegetation,
  reference = existing v4 glomap solve), DinoLedge subset (120 of 1196 4K
  video frames, stride 2, outdoor walk — reference = existing full-sequence
  solve, cross-workspace by image name), Swivel_Chair subset (106 of 632
  portrait-4K video frames — the existing April solve is .sfmr v1 with
  failed integrity, so a fresh subset solve is the reference), Kerry Park
  fisheye
  (24 frames, OPENCV_FISHEYE ground truth from the rig config — the
  deliberate failure-mode probe for pinhole-only). Also added
  `MAX_CLUSTERS = 10000` (keep highest-span clusters) — dino_dog_toy
  produces 122k clusters / 368k obs, 50× seoul, unusable in scipy BAs.
- **Covisibility grouping replaces sequence order.** All uses of the frame
  index (consecutive-frame windows, orbit growth order, constant-velocity
  resection init) replaced by shared-cluster counts: seed groups grow
  greedily from the strongest covisibility edge maximizing the *minimum*
  shared count against the group (mutual covisibility, not hub-and-spokes);
  growth picks the unposed image with the most observations of valid points
  (next-best-view); resection inits from the top-3 most-covisible posed
  images' poses (early-accept at >40% inliers). Focal scan now caps growth
  at ~20 images and only the winner grows fully. Seoul parity run: 78.4%
  inliers, 3.5° mean rotation, f 344.9 (vs 78.5% / 2.9° / 345.7 with the
  video-order version — same result, and the two seed groups it picked by
  covisibility alone were exactly the two ends of the orbit).

- **dino_dog_toy (85 unordered photos, 2040×1536), 18.7 min.** Focal scan
  peaked sharply and alone at 1428 (86.6% vs 59.5/61.8% neighbours on the
  20-image scan subset); released f = 1431.9 vs reference 1475.0 (−2.9%).
  Full growth posed all 85 images: 74.8% of 80,814 observations < 2 px
  (kept rms 1.34 px), 9,164 points. Camera errors vs the fresh incremental
  solve: rotation mean 17.9°, max 112.7°; centers mean 9.5%, max 41.2% —
  i.e. the bulk is right but a tail of cameras is badly wrong (unordered
  photos have weakly-connected views where trimmed resection can lock onto
  a locally-consistent wrong pose that the global 4 px trim then never
  revisits). Compare printout now includes medians + a >10° count to size
  that tail on later runs. Cluster cap engaged: kept 10,000 of 122,769 by
  span.

- **COLMAP south-building (128 photos, 3072×2304), 12.6 min.** The
  strongest run of the campaign: all 128 posed, **95.5% of 85,534
  observations < 2 px** (kept rms 0.53 px), rotation mean 2.33° / max 6.55°
  (zero cameras > 10°), centers mean 1.17%, f 2622.9 vs reference 2561.2
  (+2.4%). The focal scan again peaked decisively (95.9 vs 77.5/63.1%).
  Notable: seed groups picked by covisibility straddle non-consecutive
  file-order images ([24,25,26,125,126]) — the capture loops back, and the
  covisibility grouping finds it where a frame-index window could not.
  Cluster observations here are unusually clean (95.5% explainable —
  building texture localizes well).

- **DinoLedge subset (120 of 1196 4K video frames, stride 2, forward walk),
  15.1 min.** Reference = the existing full-sequence solve (cross-workspace
  by image name). All 120 posed, 84.6% of 88,890 observations < 2 px (kept
  rms 1.24 px), rotation mean 2.61° / max 3.20°, centers mean 0.36% / max
  0.60%. Focal essentially exact: 2751.3 vs 2746.8 (**+0.16%**) — this
  camera is nearly distortion-free, so pinhole-only has no k1 to absorb and
  the recovered f matches. Walking-forward motion (not an orbit) works the
  same as orbits under the covisibility machinery.

- **Swivel_Chair subset (106 of 632 portrait-4K (2160×3840) video frames,
  stride 6, indoor object orbit), 12.7 min.** The existing April solve is
  .sfmr v1 with failed integrity — reference is a fresh incremental solve
  on the same subset. Sharp focal peak again (84.1% at 2688). 78.9% of
  80,615 observations < 2 px, 9,609 points; f 2709.9 vs 2740.4 (−1.1%).
  The most accurate cameras of the campaign: rotation mean **0.26°** / max
  0.70°, centers mean 0.24% / max 0.60%. A dense indoor orbit of a
  texture-rich object at video frame rate is the easy case. (Logged as
  1080p during the campaign; the v4 `image_dims` array caught the error.)

- **Kerry Park fisheye (24 frames 480×480, OPENCV_FISHEYE fx = 129.1),
  6.3 min — the failure-mode probe.** Pinhole-only degrades gracefully
  rather than crashing: 43.4% of 13,217 observations < 2 px, 2,471 points,
  f settling at 340.7 (not comparable to the equidistant fx). Inliers by
  radial band: 44/46/43% out to 0.5·rmax (matching this capture's ~50%
  contamination floor even at the center), 34% (median 9 px) at 0.5–0.7,
  and 5% (median 164 px) past 0.7. So the bootstrap silently keeps the
  central ~half-field where pinhole ≈ equidistant and drops the rim —
  graceful and radially ordered, which is exactly the structure a
  camera-correction stage (or a center-out unlock) could pick up from.

  > _Status (2026-07-12): Falsified by visual inspection — the kerry
  > reconstruction is a complete geometric failure._ The GUI shows a
  > tangle; diagnostics confirm: consecutive-frame rotation deltas of a
  > steady walking capture swing 2.5°–157°, camera spread is 10× the
  > scene scale, and camera 0 sees the *median* 3D point at negative
  > depth (most structure behind the camera — a mirror/degenerate
  > collapse). The paragraph above stands as a lesson in metric
  > circularity, not as a result: kerry has no pose reference, so its
  > inlier and radial-band numbers were computed against the bootstrap's
  > own broken cameras — a self-referential score that locally-consistent
  > wrong geometry passes. Pinhole-only on this fisheye fails outright,
  > and the pipeline's internal metrics cannot detect it. Cheap
  > self-diagnostics that would have caught it: cheirality fraction over
  > all observations, and per-camera structure-depth sign stats.

  > _Status (2026-08-08): the failure is now NAMED, not repaired._ The
  > seed stage (`scripts/exp_fast_seed.py`) runs its pinhole focal vote
  > first and alone, exactly as before, and escalates to the camera-model
  > columns of `specs/core/geometry/focal-vote.md` only when that vote is itself
  > **low confidence**. A confirmed `EquidistantFisheye` verdict is
  > recorded as the `fisheye_detected` confidence flag plus the
  > equidistant focal, verdict margin, per-cell certified mass and the
  > trigger reason in the seed `.sfmr`'s `tool_options`. The verdict does
  > not touch the solve: the probe focal, the scan grid and every
  > downstream stage keep the pinhole vote's answer, because an
  > equidistant focal parameterizes `θ = r/f` and is not a pinhole focal
  > (kerry: equidistant 138.3 px against a pinhole vote of 144.0 px, and
  > the pinhole solve that converges is a center-out one whose effective
  > focal is neither). So the outcome is unchanged and still broken — it
  > is now a *knowing* failure a consumer can filter on, which is what the
  > self-referential 43% could not provide.
  >
  > **Escalation trigger** — a disjunction over the pinhole vote's own
  > diagnostics, each arm meaning "this consensus is not to be trusted":
  > no consensus at all (`focal_px` is `None`); the pooled focal came from
  > the rotation family and sits within one grid step
  > (`(4/0.3)^(1/47) = 1.0566`) of the bottom of the orthogonality scan's
  > `[0.3, 4.0] × max(w,h)` grid, i.e. the scan ran out of grid rather
  > than finding an interior minimum; the kernel's own
  > `FAMILY_DISAGREEMENT_BAND` of `0.25` fired, so the two independent
  > estimator families measured incompatible focals and one was discarded;
  > or the pool is thin (`n_pool ≤ 9`). Only the last has no kernel
  > constant behind it: `9` is the tightest bar reaching every fisheye
  > capture on the fleet (KerryPark360 and OmniTemple1 both pool exactly
  > 9 and no other arm catches them), and it coincides with half the
  > kernel's `MAX_EPIPOLAR_PAIRS` budget of 18. Measured over the
  > 40-dataset fleet the trigger fires on 4/4 fisheye captures and 9/36
  > pinhole ones, at +0.4–0.8 s each against a 30–60 s seed run. Those 9
  > are genuinely weak votes (7 tripped the bimodality band, 5 pooled ≤ 8
  > votes), so the second column is a confidence cross-check there rather
  > than waste.
  >
  > **Verdict confirmation** — the bare verdict is not trustworthy alone.
  > Run unconditionally, the two-column vote returns `EquidistantFisheye`
  > on three rectilinear fleet captures (BadlandPanorama, margin 3.60×;
  > 20240614_224244438, 2.25×; MossyRailing, 1.08×), and that margin range
  > overlaps a true fisheye capture's (OmniPhotos Temple1, 1.46×), so no
  > margin cut separates them. What separates them with no threshold at
  > all is **which cells carry the equidistant column's evidence**: all
  > three false verdicts win on the epipolar cell with exactly zero
  > model-informative rotation-cell mass, while all four true fisheye
  > captures earn mass in both cells (rotation-cell mass 4 / 26 / 41 / 44).
  > The asymmetry is structural: the rotation cell fits a pure rotation of
  > unit *rays*, which a wrong ray map cannot fake, whereas the epipolar
  > cell's essentialness residual is a weaker statement a wrong map can
  > partially satisfy. The flag is therefore set only when both cells
  > corroborate the verdict — 4/4 on the fisheye captures, 0/36 on the
  > pinhole fleet.

- **Visual inspection of dino_dog_toy (2026-07-12).** The misregistration
  tail is visible as a partial duplicate ("echo") of the dino in the point
  cloud: the >10° cameras are not randomly wrong but form a coherent
  wrongly-registered subset that re-triangulates its own copy of the
  object — consistent with resection locking onto a locally-consistent
  wrong pose and the global 4 px trim then keeping the echo's
  self-consistent observations.

### Cross-dataset summary

| dataset | input | imgs | obs | inlier<2px | rot err mean/max (deg) | center err mean/max (% diam) | f vs ref | time |
|---|---|---|---|---|---|---|---|---|
| seoul bull | 270×480 video orbit | 17 | 7,193 | 78.4% | 3.46 / 6.00 | 1.8 / 3.2 | +3.7% | 4 min |
| dino_dog_toy | 2040×1536 photos, unordered | 85 | 80,814 | 74.8% | 17.9 / 112.7 | 9.5 / 41.2 | −2.9% | 19 min |
| south-building | 3072×2304 photos | 128 | 85,534 | 95.5% | 2.33 / 6.55 | 1.2 / 4.5 | +2.4% | 13 min |
| DinoLedge (subset) | 4K video, forward walk | 120 | 88,890 | 84.6% | 2.61 / 3.20 | 0.4 / 0.6 | +0.16% | 15 min |
| Swivel_Chair (subset) | portrait-4K video orbit | 106 | 80,615 | 78.9% | 0.26 / 0.70 | 0.2 / 0.6 | −1.1% | 13 min |
| Kerry fisheye | 480×480 fisheye | 24 | 13,217 | 43.4%† | (no reference) | — | n/a | 6 min |

† Self-referential (no pose reference); visual inspection shows a complete
geometric failure — see the kerry status note above.

Cross-dataset observations:

- The fixed-f focal scan peaked **decisively and uniquely on every
  dataset** — the inlier-fraction-at-fixed-f signal appears robust across
  scene types, resolutions, and motion patterns.
- The recovered focal lands within ±4% of the reference everywhere, and the
  deviation tracks the reference's k1 (DinoLedge, nearly distortion-free:
  +0.16%; seoul, largest k1 relative to f: +3.7%).
- The one weak spot is **unordered photo sets** (dino_dog_toy): a tail of
  weakly-connected cameras resects onto locally-consistent wrong poses that
  the global trim never revisits. Videos and the loop-closing
  south-building set have no such tail (0 cameras > 10° on all four).
- Wall-clock is dominated by the 5-candidate focal scan and the scipy BAs;
  the per-dataset ~13–19 min is throwaway-prototype speed, not a statement
  about the method.

## Campaign log (2026-07-13): quality-ordered BA sets + gated growth

Goals: fewer incorrect registrations / phantom points, and faster
convergence to first-order-correct geometry, using the stored per-member
quality signals and the warp determinants.

### Signal diagnostics (all six datasets, vs reference-pose triangulation)

- **`member_consistency_residual` is the junk signal everywhere**: AUC
  0.79–0.92 for good-vs-bad clusters. Among a top-10k admission set:
  dino 1.3% junk (consistency-ordered) vs 30.1% (span-ordered), DinoLedge
  0.2% vs 19.1%, Swivel 1.5% vs 35.2%. Span ordering is *anti*-predictive
  of junk on four of six datasets — but south-building's high-span
  clusters are clean (0.5–4.4%), which matters below. ZNCC adds little
  (already gated at match time); feature scale is slightly anti-predictive;
  `.sift` features are scale-sorted, so the feature index is a free size
  rank.
- **`sqrt(|det warp|)` tracks the reference depth ratio `z_ref/z_k`**
  where depth structure exists: dino corr 0.910 (5.5% median error vs
  22.4% actual spread), south-building 0.846 (2.9% vs 6.7%). On
  near-constant-depth captures (seoul, DinoLedge) the foreshortening
  noise exceeds the tiny depth spread — no usable signal, but also little
  to correct.

### Architecture changes that survived

1. **Admission is not binary.** Growth, resection, and triangulation see
   every usable cluster; only the BAs are restricted to the best
   `MAX_CLUSTERS` clusters in admission order. Every capped-admission
   variant disconnected some capture: a 10k consistency-stratified cap
   stranded south-building at 36/128 (and dino's 12-image tail starved
   before ever reaching the gate); per-image round-robin admission
   destroyed seed-window density (7/128). A related trap: the masked
   growth BA's inter-round retriangulation rebuilds points from only the
   observations it was given, wiping non-BA clusters — the points must be
   refilled from the full observation set after each BA or the
   next-best-view count sees only BA-set connectivity.
2. **BA-set ordering = `union`**: interleave the span-descending backbone
   with the best-consistency-within-span-stratum core, half/half in any
   prefix. Span-only carries the junk that produced the dino echo;
   consistency-only lacks multi-view rigidity (south 5.65° mean, 6 cams
   > 10°; round-robin 7.8°, 27 > 10°). The union restores both.
3. **Resection acceptance gate**: a resection under 0.35× the median
   accepted inlier fraction is deferred (one BA + retriangulation retry
   re-arms the blocked images). Without it, 0–7%-inlier resections at a
   thin frontier cascade — each wrong pose builds phantom points that
   justify the next wrong pose — and wreck even a previously-good core
   (measured: 0.63° core → 81° after ungated regrowth).
4. **Verified force-accept**: when growth stalls with everything
   deferred, accept the strongest blocked candidate *without building
   points from it*, run the growth BA, and keep it only if its inlier
   fraction rises into the accepted band — else unpose it permanently.
   This restores the accept-then-BA-repair path that completes marginal
   frontiers (seoul imgs 0–5: 0–10% → 36–51% after BA, all kept) while
   refusals stay honest (south imgs 13/115: 0% → 4–8%, left unposed).

### Cross-dataset results (union BA set + gate + force-accept)

Inlier fractions are over the BA-candidate observations (the baseline
table's denominators were the full admitted set — not comparable).

| dataset | posed | rot err mean/max (deg) | center err mean/max (% diam) | f vs ref | time | baseline (2026-07-12) |
|---|---|---|---|---|---|---|
| seoul bull | 17/17 | 3.37 / 5.89 | 1.7 / 3.1 | +3.1% | 2.1 min | 17/17, 3.46 / 6.00, 4 min |
| dino_dog_toy | 84/85 | **0.65 / 4.06** | 0.2 / 0.7 | −3.0% | 18.3 min | 85/85, **17.9 / 112.7** (echo), 19 min |
| south-building | 126/128 | 3.68 / 7.94 | 2.4 / 6.7 | +2.7% | 16.6 min | 128/128, 2.33 / 6.55, 13 min |
| DinoLedge (subset) | 120/120 | 2.36 / 2.93 | 0.35 / 0.62 | **+0.03%** | 17 min | 120/120, 2.61 / 3.20, 15 min |
| Swivel_Chair (subset) | 106/106 | 0.31 / 0.75 | 0.35 / 0.74 | −1.3% | 15.4 min | 106/106, 0.26 / 0.70, 13 min |
| Kerry fisheye | 24/24 | 178.0 (mirror) | 24.1 / 51.2 | n/a | 6.3 min | complete failure (43% self-ref) |

- **The dino echo is eliminated**: mean rotation error 17.9° → 0.65°, max
  112.7° → 4.06°, zero cameras over 10°. Two tail images were recovered
  by verified force-accept (15% → 56%, 12% → 30%); one was honestly
  refused (0% → 3%).
- Kerry now has an external reference (48-image fisheye rig solve) and
  the pinhole bootstrap's failure is quantified: a uniform ~178° mirror
  solution — the metric the earlier self-referential inlier fraction
  could not provide.
- south-building pays a small accuracy premium vs its span-only baseline
  (3.68 vs 2.33 mean) for using the one-config union set; its two
  refusals are genuine (0% resections after BA repair).
- Wall-clock is roughly flat vs baseline at equal coverage. The measured
  speed lever is the tiered scan (35% coarse tier: seoul scan 178 s →
  47 s, 3.8×, with a 0.63°-mean 8-image core — better than any full
  solve here), currently parked: frontier force-accepts off a stranded
  coarse tier verify at marginal levels and settle ~12° off, so the
  coarse tier needs either a connectivity-aware tier or a stricter
  verification bar before it becomes the default. The apex-solver BA
  backend (1.9–3.1×, separate evaluation) composes with all of this.
- The warp-depth Kabsch resection init (`SFMTOOL_DEPTH_INIT`) is
  implemented and exact on synthetic data but is not part of this
  campaign's config; its measured per-resection warp-depth coherence is
  logged for the future registration-verification channel.

### Addendum (2026-07-13): RANSAC P3P last-chance registration

Root cause of the refused dino image (index 52, the dataset's closest
camera): its kept features are 3–4× physically finer than every far
image's extractable band, so ~60–70% have no legitimate counterpart and
~90% of its cluster memberships are wrong matches (reference-arbitrated).
Descriptor-based 2D–3D rematching cannot rescue this (measured 1%
candidate precision — wrong matches score 0.96+ cosine on this texture),
but the surviving ~7–10% true correspondences are ample for
minimal-sample estimation. The last-chance path (when growth stalls with
every candidate deferred) now runs RANSAC P3P with four load-bearing
pieces, each forced by a measured failure:

1. **Tight 4 px RANSAC threshold** (the 12 px default's consensus was 75%
   loose junk and anchoring on it dragged the pose).
2. **Consensus polish**: the pose is refit on the RANSAC consensus only.
3. **Whole-cluster BA promotion**: the consensus clusters (all members'
   observations) enter the BA working set and the image's non-consensus
   observations are quarantined out — single-observation anchoring lets
   the BA's inter-round retriangulation wipe the anchored points, saving
   a pose with zero kept features.
4. **Consensus-survival verification**: accept when the consensus set
   stays ≥ 50% inliers through the growth BA — the all-obs inlier bar
   can never certify an image whose observations are mostly wrong
   *matches* even under a correct pose.

Six-dataset result vs the pre-P3P config (same union/gate config,
scipy):

| dataset | posed | rot err mean/max (deg) | pre-P3P |
|---|---|---|---|
| seoul bull | 17/17 | 2.99 / 5.27 | 17/17, 3.37 / 5.89 |
| dino_dog_toy | **85/85** | **0.57 / 1.27** | 84/85, 0.65 / 4.06 |
| south-building | **128/128** | **2.63 / 4.53** | 126/128, 3.68 / 7.94 |
| DinoLedge (subset) | 120/120 | 2.36 / 2.93 | identical (P3P never fired) |
| Swivel_Chair (subset) | 106/106 | 0.31 / 0.75 | identical (P3P never fired) |
| Kerry fisheye | 24/24 | mirror (~176) | mirror (~178), unchanged |

Every dataset where the path fired gained full coverage AND accuracy
(south-building now beats its span-only baseline's max, 6.55°; all its
three previously-refused images registered at 99–100% consensus
survival); where it did not fire the results are unchanged. The estimator
is pycolmap's (experiment-grade); a native Lambda-Twist P3P belongs with
the planned geometric verifier.

## Open questions

- The dino_dog_toy misregistration tail: weakly-connected views in
  unordered photo sets need either a resection acceptance gate (reject and
  retry later when more structure exists), a re-resection pass after the
  final BA, or RANSAC P3P. Everything else in the campaign has no tail.
  > _Status (2026-07-13): Done — acceptance gate + verified force-accept
  > (see the 2026-07-13 campaign log); dino 17.9° → 0.65° mean with zero
  > cameras over 10°. Then RANSAC P3P last-chance (addendum above):
  > 85/85 at 0.57 / 1.27._
- The junk-observation floor (~5–25% by dataset, ~50% on kerry) matches the
  contamination floor seen in the grid-distortion experiments; per-member
  vetting signals (ZNCC, consistency residual) are stored in the `.matches`
  file and are not yet used to pre-filter here.
  > _Status (2026-07-13): Done for the consistency residual — it now
  > orders the BA working set (union ordering, 2026-07-13 campaign log);
  > ZNCC measured near-uninformative post-gating (AUC ≈ 0.55)._
- Runtime is dominated by the focal scan (5 × capped growth) and the scipy
  BAs. A coarse-to-fine scan (2 candidates + golden-section refine) would
  cut most of it. All of this is throwaway scipy; a production version
  would be a Rust kernel.
- Whether the `cluster_bootstrap` `.sfmr` is good enough to seed `sfm solve`
  (as a triangulation/pose prior) or the planned cluster-level geometric
  verifier — not tried.
- Self-diagnostics: kerry produced a completely broken reconstruction
  while its internal inlier metric read 43% — with no external reference,
  the bootstrap cannot currently tell success from locally-consistent
  failure. Cheap candidates that would have caught it: cheirality fraction
  over all observations, per-camera structure-depth sign statistics, and
  pose-path coherence on sequential captures.
  > _Status (2026-07-13): Partially done — kerry now has an external rig
  > reference (uniform ~178° mirror, quantified), and the warp-depth
  > coherence measured at resection time is a stored pose-free
  > verification channel; neither is wired into a self-diagnostic verdict
  > yet._
  > _Status (2026-08-08): Partially done — the CAUSE is now self-diagnosed
  > on this capture class._ The seed stage flags `fisheye_detected` from
  > the escalated camera-model vote (see the kerry entry above), which is
  > a pose-free verdict on the camera MODEL and is exactly the signal that
  > separates "the solver failed" from "the model cannot represent this
  > capture". It does not close the general item: a pinhole capture that
  > collapses for any other reason still reads only as `low_consensus` /
  > the geometric flags, and the cheirality and depth-sign statistics
  > remain unimplemented.
