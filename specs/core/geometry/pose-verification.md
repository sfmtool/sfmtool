# Displacement-Neighborhood Pose Verification

## Purpose

Find the cameras in a finished reconstruction whose poses are wrong, and
put them back — using only the 2D tracks, with no reference solve, image
ordering, or motion model to check against. The ruler is a
2D structure computed once from the cluster tracks — which images are
near-duplicate viewpoints of which, measured by keypoint displacement —
and the tests hold the current poses against it. Because the substrate
never reads poses, it is computed before any reconstruction exists and
stays valid through seeding, growth, and refinement; the same structure
serves pair selection, thinning, neighbour initialization, and
verification at every stage.

## Substrate: the displacement neighborhood

Per covisible image pair: the shared-cluster count and the mean pixel
displacement of shared-cluster keypoints. One pass over clusters emits
each cluster's member pairs (`span·(span−1)/2` of them); under the
cluster matcher's size cap the total is linear in observations. The two
statistics are aggregated differently on purpose. The shared count is
deduplicated per cluster, so it agrees with `ClusterCovisibility.count`:
a cluster votes at most once for a pair however many members it holds in
either image. The mean displacement averages over *every* accepted
cross-image member pair of those clusters — exhaustive, not the seeded
one-sample-per-cluster estimate behind
`ClusterCovisibility.pair_displacement`. The two aggregations coincide
wherever clusters hold one member per image, which is the common case.
Storage is sparse — only realized pairs, itself linear under the cap —
with per-image queries:

- `nearest(i, k, min_shared)` — the k lowest-mean-displacement partners
  with at least `min_shared` shared clusters (near-duplicate viewpoints);
- `farthest(i, k, min_shared)` — the k highest-displacement partners over
  the same shared-count floor (wide-baseline pairs, e.g. for focal
  estimation);
- pair stats lookup.

A cluster-member acceptance mask (as elsewhere on `ClusterCovisibility`)
is honored at construction. The build itself is not sparse: it addresses
a pair's accumulator through a transient dense slot index over
`num_images²` (`4·N²` bytes, the same order as the count matrix
`ClusterCovisibility` holds permanently), which is why
`DisplacementNeighborhood::from_clusters` refuses `num_images` above the
dense bound the count matrix uses. Only realized pairs get an
accumulator, and the index is dropped once the sparse adjacency is
assembled. Persistence is the substrate's own:
`DisplacementNeighborhood::to_arrays` emits parallel per-pair arrays
`(i, j, shared count, mean displacement)` with `i < j`, and
`from_arrays` rebuilds it, so one computation serves a multi-stage
pipeline. The round trip is on the neighborhood alone — the
`ClusterCovisibility` it was built from is not recoverable from those
arrays, and the kernels only ever need the neighborhood.

## Screen A: self-resection

Re-resect every registered camera's own observations against the shared
structure with the batch registration primitive
(`resect_images_batch`). A camera whose pose cannot be re-derived from
its own 2D–3D support is flagged: fewer than `resect_min_obs`
observations of valid points, or a re-resection whose all-observation
inlier fraction falls below `resect_accept_gate` — that gate is what "no
acceptable consensus" means here. Catches junk-consensus registrations
and cameras whose support collapsed under later refinement.

The screen tests support, not agreement with the stored pose: it never
compares the re-derived pose against the one on record. A camera whose
stored pose is wrong but whose observations are healthy re-derives
correctly and passes A, while Screen B flags it. The two screens are
complementary rather than redundant, and the reported `flagged` array is
their union.

## Screen B: measured-versus-posed relative rotation

For each registered camera and each of its `max_neighbors` `nearest`
neighbours (the low-parallax regime, where the conjugate-homography model
holds): estimate the homography over the pair's shared-cluster
correspondences — skipping a pair with fewer than
`min_pair_correspondences` of them, or a homography carrying fewer than
`min_h_inliers` — extract the relative rotation `K⁻¹HK`, and compare with
the pose-implied relative rotation. Orthonormalization is the polar
factor with the whole-sign fix used everywhere in this codebase, which
resolves the `H ≃ −R` sign ambiguity a per-column fix would leave open;
`K⁻¹HK` is an *optical*-frame rotation, so it is conjugated by
`S = diag(1, −1, −1)` on both sides to reach the canonical frame the
poses live in. The per-image score is the **median** angular discrepancy
over its neighbours; flag at or above `rotation_threshold_deg`
(default 3°). A camera with fewer than `min_rotation_measurements`
usable neighbours is not scored at all: the screen abstains, reporting
`NaN` and no flag, rather than judging a camera on one measurement.

Two properties are load-bearing. The comparison must be restricted to
low-displacement neighbours: at wider baselines the displacement carries
parallax and a small-angle rotation model misattributes it (measured
relative rotation via the homography stays valid only where parallax is
small). And the aggregation must be a per-image median: a single
discrepant pair is noise or parallax, while a misregistered camera is
implicated consistently by every neighbour that overlaps it.

## Repair

Flagged cameras are repaired in ascending image order, and an accepted
repair joins the working pose state, so a later camera can initialize
from a neighbour this pass just fixed. Repairs are therefore order
dependent, and sorting on the image index rather than on flag discovery
order is what keeps the pass deterministic.

For each flagged camera: build an initial pose from its top-2 `nearest`
registered neighbours over the same `min_shared` floor — chordal mean of
their rotations, mean of their centres — then trimmed pose-only
refinement against the current structure (5 trim rounds keeping the best
0.6 of observations each, final inliers at 3 px). A camera with fewer
than two registered near neighbours, or with fewer than `min_obs`
observations of valid points, is skipped and its flag stands. Accept only
when the all-observation inlier fraction reaches
`max(inlier_floor, before + inlier_margin)` (defaults 0.10 and 0.05): an
"improvement" below the absolute floor means the camera's neighbourhood
structure is itself broken, which pose-only repair cannot fix (re-posing
plus re-triangulation of the segment is a separate concern). Rejected
repairs leave the pose untouched and the flag standing.

## Inputs and outputs

The kernels live in
[pose_verification.rs](../../../crates/sfmtool-core/src/geometry/pose_verification.rs),
bound under `sfmtool._sfmtool.geometry` as `verify_poses` and
`repair_poses`; the displacement-neighborhood substrate they read is
[displacement.rs](../../../crates/sfmtool-core/src/features/cluster_match/covisibility/displacement.rs),
whose queries and compact array serialization hang off the
`ClusterCovisibility` pyclass
([cluster-covisibility.md](../features/cluster-covisibility.md)). The
kernels build on homography estimation ([focal-vote.md](focal-vote.md)),
batch registration ([reconstruction-growth.md](reconstruction-growth.md)),
and absolute-pose refinement ([absolute-pose.md](absolute-pose.md)).

Kernels take the flat cluster-observation arrays, the shared camera, the
current poses and points, and the substrate (or construct it on the
fly). `verify_poses` returns per-image flags and scores from both
screens; `repair_poses` additionally returns updated poses and the
repaired/rejected lists. Both are read-only on the observation data;
images are independent in both screens and parallelize.

## Parameters

`VerifyOptions` and `RepairOptions` (`pose_verification.rs`) carry the
tunables; the four fixed constants below them are module-level in the
same file.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `resect_min_obs` | `8` | Screen A: a camera observing fewer valid points than this is flagged without resecting |
| `resect_accept_gate` | `0.30` | Screen A: all-observation inlier fraction a re-resection must reach to clear the screen |
| `max_neighbors` | `4` | Screen B: lowest-displacement registered neighbours examined per camera |
| `min_shared` | `50` | Shared-cluster floor for a pair to count as a neighbour, in both the screen and the repair init |
| `min_pair_correspondences` | `30` | Screen B: a pair with fewer shared-cluster correspondences is skipped |
| `min_h_inliers` | `20` | Screen B: a homography with fewer inliers is skipped |
| `min_rotation_measurements` | `2` | Screen B: neighbour measurements a camera needs to be scored; below it the screen abstains with `NaN` |
| `rotation_threshold_deg` | `3.0` | Screen B: flag at or above this median angular discrepancy |
| `seed` | `0` | Base seed for the per-image resection and per-pair homography RANSACs |
| `min_obs` (`RepairOptions`) | `12` | Skip repairing a flagged camera observing fewer valid points |
| `inlier_floor` | `0.10` | Absolute inlier-fraction floor an accepted repair must reach |
| `inlier_margin` | `0.05` | Improvement over the pre-repair inlier fraction an accepted repair must reach |
| `INLIER_PX` | `3.0` | Final-inlier pixel bound shared by the screens and repair acceptance (matches the growth kernel) |
| `REFINE_TRIM_ROUNDS` | `5` | Trim rounds in the repair's pose-only refinement |
| `REFINE_KEEP_FRACTION` | `0.6` | Observations retained per trim round |
| `REPAIR_INIT_NEIGHBORS` | `2` | Registered neighbours a repair blends its initial pose from |

The absolute thresholds are calibrated for clean captures. On messy
handheld fleets they over-flag, and the right response is not to raise
them globally but to derive them from the dataset's own score
distribution — flagging relative to the median of the per-image medians,
for instance. Both kernels return the raw per-image scores
(`resect_inlier_fractions`, `rotation_scores_deg`) alongside the flags so
a caller can do that without re-running either screen.

## Testing requirements

- Substrate: construction cost linear in observations under the span
  cap; `nearest`/`farthest` exact against a dense reference on a small
  scene; mask honored; serialization round-trips; the slot-indexed
  accumulation reproduces a hash-map reference exactly -- pair set,
  counts, mean-displacement bit patterns and CSR rows -- on a 300-image,
  20 000-cluster synthetic scene, masked and unmasked.
- Screens on a synthetic scene with implanted misregistrations: a
  wrong-pose camera with healthy neighbours is flagged by both screens;
  an unflagged scene yields no flags at the default thresholds; a
  translation-rich (high-parallax) pair alone never flags (screen B's
  low-parallax gate).
- Repair: an implanted wrong pose with intact structure is restored to
  within tight bounds of truth; a camera whose cluster points are
  corrupted is flagged but its repair is rejected and state unchanged.
- Determinism: fixed seed reproduces flags, scores, and repairs bitwise.

## Non-goals

Structure-level repair (re-posing plus re-triangulation of a broken
segment), capture-shape classification from the substrate's off-diagonal
mass, ordering- or motion-model-based checks, and photometric
verification (a complementary, stricter tier for registrations that are
geometrically self-consistent on wrong content).
