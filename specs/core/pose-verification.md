# Displacement-Neighborhood Pose Verification

**Status:** Implemented (2026-07-19) — the `DisplacementNeighborhood`
substrate on `ClusterCovisibility`
(`crates/sfmtool-core/src/features/cluster_match/covisibility.rs`; see
specs/core/cluster-covisibility.md) plus the `verify_poses` / `repair_poses`
kernels (`crates/sfmtool-core/src/geometry/pose_verification.rs`), bound
under `sfmtool._sfmtool.geometry` (the kernels take the substrate's compact
array serialization; the substrate queries and serialization live on the
`ClusterCovisibility` pyclass). Depends on homography estimation
(specs/core/focal-vote.md), batch registration
(specs/core/reconstruction-growth.md), and absolute-pose refinement
(specs/core/absolute-pose.md).

## Purpose

Detect and repair misregistered cameras in a reconstruction without a
reference solve, an image ordering, or a motion model. The ruler is a
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
cluster matcher's size cap the total is linear in observations. Storage
is sparse — only realized pairs, itself linear under the cap — with
per-image queries:

- `nearest(i, k)` — the k lowest-mean-displacement partners above a
  shared-count floor (near-duplicate viewpoints);
- `farthest(i, k)` — the k highest-displacement partners (wide-baseline
  pairs, e.g. for focal estimation);
- pair stats lookup.

A cluster-member acceptance mask (as elsewhere on `ClusterCovisibility`)
is honored at construction. The structure is serializable so one
computation serves a multi-stage pipeline.

## Screen A: self-resection

Re-resect every registered camera's own observations against the shared
structure with the batch registration primitive
(`resect_images_batch`). A camera whose pose cannot be re-derived from
its own 2D–3D support — no acceptable consensus — is flagged. Catches
junk-consensus registrations and cameras whose support collapsed under
later refinement.

## Screen B: measured-versus-posed relative rotation

For each registered camera and each of its `nearest` neighbours (the
low-parallax regime, where the conjugate-homography model holds):
estimate the homography over the pair's shared-cluster correspondences,
extract the relative rotation `R = K⁻¹HK` (orthonormalized, conjugated
to the canonical frame), and compare with the pose-implied relative
rotation. The per-image score is the **median** angular discrepancy over
its neighbours; flag at or above a threshold (default 3°).

Two properties are load-bearing. The comparison must be restricted to
low-displacement neighbours: at wider baselines the displacement carries
parallax and a small-angle rotation model misattributes it (measured
relative rotation via the homography stays valid only where parallax is
small). And the aggregation must be a per-image median: a single
discrepant pair is noise or parallax, while a misregistered camera is
implicated consistently by every neighbour that overlaps it.

## Repair

For each flagged camera: build an initial pose from its top-2 `nearest`
registered neighbours — chordal mean of their rotations, mean of their
centres — then trimmed pose-only refinement against the current
structure. Accept only when the all-observation inlier fraction reaches
`max(floor, before + margin)` (defaults 0.10 and 0.05): an "improvement"
below the absolute floor means the camera's neighbourhood structure is
itself broken, which pose-only repair cannot fix (re-posing plus
re-triangulation of the segment is a separate concern). Rejected repairs
leave the pose untouched and the flag standing.

## Inputs and outputs

Kernels take the flat cluster-observation arrays, the shared camera, the
current poses and points, and the substrate (or construct it on the
fly). `verify_poses` returns per-image flags and scores from both
screens; `repair_poses` additionally returns updated poses and the
repaired/rejected lists. Both are read-only on the observation data;
images are independent in both screens and parallelize.

## Testing requirements

- Substrate: construction cost linear in observations under the span
  cap; `nearest`/`farthest` exact against a dense reference on a small
  scene; mask honored; serialization round-trips.
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
