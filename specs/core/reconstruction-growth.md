# Reconstruction Growth

**Status:** Implemented — `grow_reconstruction` and `resect_images_batch`
(`crates/sfmtool-core/src/geometry/reconstruction_growth.rs`, bound as
`sfmtool._sfmtool.geometry.{grow_reconstruction, resect_images_batch}` in
`crates/sfmtool-py/src/geometry/reconstruction_growth.rs`). Depends on
absolute-pose estimation and refinement (specs/core/absolute-pose.md),
batch triangulation (specs/core/batch-triangulation-api.md), the staged
bundle adjustment (specs/core/bundle-adjustment.md), and cluster
covisibility (specs/core/cluster-covisibility.md).

> _Status (2026-07-19): Implemented, with notes against the text below.
> (1) `resect_images_batch` takes optional `posed_quaternions_wxyz` /
> `posed_translations` / `posed_indexes` (all-or-none): the
> neighbour-initialized fallback needs the registered poses, which the
> binding line below omitted; without them the primitive is P3P-only.
> (2) A rejected force-accept restores poses, points, and the adjustment
> set fully (stronger than the reference implementation, which left the
> verification adjustment's side effects in place). (3) Consensus
> promotion/quarantine maintains an adjustment-set mask unconditionally.
> (4) The growth candidate floor is `min_obs` (default 8) throughout.
> (5) The finishing "full re-triangulation" refills only clusters the
> finishing adjustment wiped; clusters it retained keep their refined
> positions. (6) Above the dense-covisibility image bound the kernel
> degrades gracefully: neighbour ranking falls back to first-posed,
> anchor and finishing subsets to frontier/all-posed. (7) The first
> resection after the seed is ungated — the gate's median has no
> accepted samples yet. (8) The anchored-beats-frontier testing
> requirement is only observable on loops longer than the finishing
> adjustment's ~120-camera spread subset (below that the finishing pass
> is effectively global and both configurations converge identically, to
> platform float noise); the default-run unit test therefore asserts
> anchoring does not degrade a converged loop, and the discriminative
> 140-camera comparison is an `#[ignore]`d minutes-scale test run
> manually._

## Purpose

Register the un-posed images of a cluster-track set against a seeded
reconstruction. The caller supplies poses for a few images and a focal;
the kernel grows the reconstruction image by image — resecting each new
camera against the current structure, triangulating clusters as they gain
posed views, and interleaving bounded bundle adjustments — until no
further image clears its acceptance gate. Per-step cost is bounded by
construction (each adjustment sees a bounded camera subset), so growth
over thousands of images runs at a per-image cost independent of how many
are already posed.

`resect_images_batch` is the standalone registration primitive: pose-only
resection of many images against fixed structure, each image independent
(no adjustment, no cross-image coupling), parallelized across images.

## Inputs

Flat cluster-observation arrays (`cluster_indexes`, `image_indexes`,
`positions_xy` in full pixels), the shared camera intrinsics at a fixed
focal, seed poses (`quaternions_wxyz`, `translations`, and the posed-image
index list), and options:

- `ba_window` (default 0 = unbounded): the number of most-recently-posed
  cameras each growth adjustment refines. 0 refines every posed camera.
- `anchor_every` (default 0 = never): every `anchor_every`-th growth
  adjustment instead refines a covisibility-spread subset of all posed
  cameras (capped ~150), so cameras far apart in registration order but
  covisible in space are periodically re-coupled and accumulated drift is
  pulled back.
- `ba_cluster_cap` (default 0 = all): the adjustments are restricted to
  the best-`cap` clusters by span; resection, triangulation, and the
  next-best-view count always see every cluster.
- `min_obs` (default 8), `accept_gate` (default 0.35), `seed` for the
  RANSAC.

## Mechanism

### 1. Batch registration (`resect_images_batch`)

For each requested image independently: gather its observations of
clusters that currently have a finite point; below `min_obs` skip.
Estimate the pose by RANSAC P3P over the 2D–3D candidates and polish by
trimmed pose-only refinement on the consensus subset; if the minimal
estimate fails, fall back to trimmed refinement initialized from the
poses of the image's most-covisible registered neighbours. Score the
result by the all-observation inlier fraction at 3 px; accept at or above
the gate. Images are independent — the kernel runs them in parallel and
returns poses, per-image inlier fractions, and the accepted mask.

### 2. Next-best-view growth (`grow_reconstruction`)

Repeatedly pose the un-posed image with the most observations of valid
points, by the same estimate-then-refine ladder as batch registration
with one coupling addition: an image whose inlier fraction falls below
`accept_gate ×` the median accepted-so-far fraction is deferred rather
than rejected. When every candidate is deferred, one adjustment +
retriangulation pass re-arms them; if growth still stalls, the strongest
deferred candidate is force-accepted without building points from it,
the adjustment runs, and the candidate is kept only if its inliers rose
into the accepted band (its P3P consensus clusters are promoted whole
into the adjustment set, and its non-consensus observations quarantined,
so verification measures the registration claim rather than the junk).
After each accepted image, clusters that now have two or more posed views
are triangulated in.

### 3. Bounded adjustments

A growth adjustment runs every few accepted images (staged robust
schedule, fixed focal). Its camera set is bounded:

- **Frontier:** the `ba_window` most-recently-posed cameras (registration
  order, force-rejected images removed). Refines where growth is
  happening; cost is constant per adjustment regardless of total posed
  count.
- **Anchor:** every `anchor_every`-th adjustment refines a
  covisibility-spread subset of all posed cameras instead (the
  covisibility thinning's banded selection, capped ~150). Spread cameras
  include pairs that are far apart in registration order but observe the
  same clusters, which is what re-couples a long loop and bounds drift.

After every adjustment, points for clusters outside the adjustment's
observation set are re-triangulated from the full observation set at the
updated poses (the adjustment re-triangulates only what it was given, and
the next-best-view count must see full connectivity).

### 4. Finishing

A final adjustment releases the focal on a covisibility-spread subset of
the posed cameras (capped ~120; the focal is a single global parameter,
so a spread subset conditions it), followed by a full re-triangulation at
the released focal. Outputs are computed from the full observation set.

## Output

`quaternions_wxyz`, `translations`, the posed mask, `points` (NaN rows
for never-triangulated clusters), the released `focal`, and per-observation
residual norms at the final state (inf where invalid). `resect_images_batch`
returns per-image poses, inlier fractions, and the accepted mask.

## Binding

`sfmtool._sfmtool.geometry.grow_reconstruction(cluster_indexes,
image_indexes, positions_xy, camera, quaternions_wxyz, translations,
posed_indexes, *, ba_window=0, anchor_every=0, ba_cluster_cap=0,
min_obs=8, accept_gate=0.35, seed=0)` and
`resect_images_batch(cluster_indexes, image_indexes, positions_xy,
camera, points, image_list, *, min_obs=8, accept_gate=0.30, seed=0)`,
NumPy in/out, following the geometry submodule's conventions.

## Testing requirements

- Synthetic orbit (known poses, known focal): a small seed grows to full
  registration; camera errors under similarity alignment within tight
  bounds; released focal near truth.
- `ba_window` at or above the posed count reproduces the unbounded
  adjustment's result on the same input; a bounded window still registers
  the full synthetic orbit.
- `anchor_every` on a long synthetic loop yields lower final camera error
  than frontier-only windows at the same window size.
- `resect_images_batch` matches one-at-a-time resection of the same
  images against the same fixed structure, and its parallel execution is
  deterministic for a fixed seed.
- Gates: images with junk-dominated observations are deferred, then
  force-accepted only when verification passes; a rejected force-accept
  leaves poses and structure unchanged.
- Degenerate inputs (no seed poses, no triangulable clusters, all images
  below `min_obs`) return the input state with empty growth, not an error.

## Non-goals

Seed construction (rotation initialization and factorization seeding are
separate kernels), focal estimation (the caller supplies it), confidence
flagging and attempt arbitration (caller policy), and loop-closure-style
global relaxation beyond the anchor adjustments (a final global solve is
the downstream consumer's concern).
