# Reconstruction Growth

## Purpose

Register the un-posed images of a cluster-track set against a seeded
reconstruction. The caller supplies poses for a few images and a focal;
the kernel grows the reconstruction image by image — resecting each new
camera against the current structure, triangulating clusters as they gain
posed views, and interleaving bounded bundle adjustments — until no
further image clears its acceptance gate. With a frontier window set
each adjustment sees a bounded camera subset, so growth over thousands
of images runs at a per-image cost independent of how many are already
posed.

`resect_images_batch` is the standalone registration primitive: pose-only
resection of many images against fixed structure, each image independent
(no adjustment, no cross-image coupling), parallelized across images.
The held-out resection behind the viewer's *Resect Image*
(`geometry::resect_images`, [../../gui/edits/resect-image.md](../../gui/edits/resect-image.md))
does not call this primitive. It re-estimates poses against two sources whose
roles differ, the reconstruction's tracks and the clusters of a
cluster-patches `.matches` file: its RANSAC draws minimal samples from the
tracks alone when a target has at least three finite ones, and scores points
at infinity by angle. This primitive treats every observation alike and reads
finite points only, so the resection carries its own estimator. That
estimator keeps this primitive's P3P consensus floor (8), its trimmed
refinement schedule (five rounds keeping 60%) and its 3 px inlier bound, so an
inlier means the same thing in both.

## Inputs

Flat cluster-observation arrays (`cluster_indexes`, `image_indexes`,
`positions_xy` in full pixels), the shared camera intrinsics at a fixed
focal, seed poses (`quaternions_wxyz`, `translations`, and the posed-image
index list), and options:

- `ba_window` (default 0 = unbounded): the number of most-recently-posed
  cameras each growth adjustment refines. 0 refines every posed camera.
- `anchor_every` (default 0 = never): every `anchor_every`-th growth
  adjustment instead refines a covisibility-spread subset of all posed
  cameras (capped at 150), so cameras far apart in registration order but
  covisible in space are periodically re-coupled and accumulated drift is
  pulled back. Anchoring modifies the frontier window, so it only bites
  where there is one: at `ba_window = 0`, or while the posed count is
  still inside the window, every adjustment already refines every posed
  camera and `anchor_every` has nothing to change.
- `ba_cluster_cap` (default 0 = all): the adjustments are restricted to
  the best-`cap` clusters by span (distinct observing images, ties to the
  lower cluster id); resection, triangulation, and the next-best-view
  count always see every cluster. The per-observation adjustment-set mask
  the cap produces is maintained whether or not the cap binds — at
  `cap = 0` it simply starts all-true — because the force-accept path
  below edits it.
- `min_obs` (default 8), `accept_gate` (default 0.35), `seed` for the
  RANSAC.
- `free_points` (default `FreePointPolicy::CROSS`): whether the adjustments'
  storage decision may store a point as a direction, and a direction as a
  position again ([bundle-adjustment.md](bundle-adjustment.md) § "Free points:
  inverse depth and the storage decision"). `FreePointPolicy::NO_CROSS` keeps every
  triangulated point a position.

## Mechanism

### 1. Batch registration (`resect_images_batch`)

For each requested image independently: gather its observations of
clusters that currently have a finite point; below `min_obs` skip.
Estimate the pose by RANSAC P3P over the 2D–3D candidates and polish by
trimmed pose-only refinement on the consensus subset; if the minimal
estimate fails, fall back to trimmed refinement initialized from the
poses of the image's most-covisible registered neighbours. Those
neighbour poses are an input — the all-or-none `posed_quaternions_wxyz` /
`posed_translations` / `posed_indexes` triple — and with them omitted the
primitive is P3P-only, since the fallback has nothing to initialize from.
Score the result by the all-observation inlier fraction at 3 px; accept
at or above the gate. Images are independent — the kernel runs them in
parallel and returns poses, per-image inlier fractions, and the accepted
mask.

### 2. Next-best-view growth (`grow_reconstruction`)

Repeatedly pose the un-posed image with the most observations of valid
points — at least `min_obs` of them, the same floor the batch primitive
applies — by the same estimate-then-refine ladder as batch registration
with one coupling addition: an image whose inlier fraction falls below
`accept_gate ×` the median accepted-so-far fraction is deferred rather
than rejected. That bar is a median over the growth images accepted so
far, which leaves the first resection after the seed ungated: there are
no accepted samples to take a median of yet.

When every candidate is deferred, one adjustment + retriangulation pass
re-arms them; if growth still stalls, the strongest deferred candidate is
force-accepted without building points from it, the adjustment runs, and
the candidate is kept if either its all-observation inlier fraction rose
into the accepted band or at least half of its P3P consensus
observations survive the adjustment as inliers. The consensus test is
what makes verification meaningful for an image whose observations are
mostly wrong matches: its consensus clusters are promoted whole into the
adjustment set and its non-consensus observations quarantined out of it,
so the adjustment measures the registration claim rather than the junk.
A rejected force-accept restores the poses, the points, the
adjustment-set mask and the adjustment cadence exactly as they were — the
verification adjustment leaves nothing behind — and the image stays
un-posed for good.

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
  covisibility thinning's banded selection, capped at 150), once the
  posed count has outgrown the window. Spread cameras include pairs that
  are far apart in registration order but observe the same clusters,
  which is what re-couples a long loop and bounds drift.

After every adjustment, points for clusters outside the adjustment's
observation set are re-triangulated from the full observation set at the
updated poses (the adjustment re-triangulates only what it was given, and
the next-best-view count must see full connectivity).

Each cluster carries a representation beside its row, a position or a
direction. Triangulation writes positions; an adjustment hands each point in
its observation set in at the representation it holds, and under the default
crossing its storage decision stores it as a direction where its rays give no
depth at the noise the adjustment measures, and as a position again where a
later adjustment finds one. A direction is a unit world-frame row: a
resection reads positions only, so resection, the next-best-view count and the
acceptance inlier fractions leave the directions out, while every adjustment
keeps them, and the refill does not overwrite them. A residual of a direction
is read through the rotation alone, as the kernel reads it. The crossing
reaches only the clusters inside an adjustment's observation set: a direction
outside it is wiped by the adjustment and refilled as a midpoint position, as
under `FreePointPolicy::NO_CROSS`. With a binding `ba_cluster_cap`, the crossing
decides only the clusters the cap keeps, which are the ones seen by the most
images; the far clusters it leaves out are triangulated as positions by the
refill and never decided. With `ba_window` 4, `anchor_every` 2 and a cap of 150
on a scene of about 280 clusters, the cap keeps no far cluster, so windowed
growth stores no far point as a direction, as with the crossing off; with a cap
of 280 it keeps nearly all of them and they are decided. With far points stored as directions, an image whose
observations are mostly of the far field has fewer correspondences to resect
against, which is what the crossing costs growth; on a synthetic orbit with a
far field 10⁶ units out every image still registers, as many as with the
crossing off. Every adjustment growth runs ends with a converged final round on
the real tracks measured (49 adjustments over the seoul bull, seattle, dino
and Kerry Park solves and the Kerry Park ground truth), so the storage decision
is read at a converged level, and growth with the crossing takes 0.43 to 0.87 of the time it takes
with the crossing off
([bundle-adjustment-measurements.md](bundle-adjustment-measurements.md#cost)).

Everything covisibility-driven here rests on the dense cluster
covisibility, which is only built up to `MAX_DENSE_IMAGES` (4096) images.
Past that bound the kernel degrades instead of failing: neighbour ranking
for the resection fallback falls back to the first posed image, the
anchor subset to the frontier window, and the finishing subset to every
posed camera.

### 4. Finishing

A final adjustment releases the focal on a covisibility-spread subset of
the posed cameras (capped at 120; the focal is a single global parameter,
so a spread subset conditions it) — at or below that many posed cameras,
and wherever the covisibility is unavailable, the subset is every posed
camera. It is followed by the same refill the growth adjustments use, at
the released focal: only the clusters the finishing adjustment wiped are
re-triangulated, and the clusters inside its observation set keep the
positions it refined. Outputs are computed from the full observation set.

## Output

`quaternions_wxyz`, `translations`, the posed mask, `points` (NaN rows
for never-triangulated clusters), `point_at_infinity` (which rows are unit
directions), the released `focal`, and per-observation residual norms at the
final state (inf where invalid). `resect_images_batch`
returns per-image poses, inlier fractions, and the accepted mask.

## Binding

`grow_reconstruction` lives in
[reconstruction_growth.rs](../../../crates/sfmtool-core/src/geometry/reconstruction_growth.rs)
and `resect_images_batch` in
[batch_resection.rs](../../../crates/sfmtool-core/src/geometry/batch_resection.rs),
bound by
[reconstruction_growth.rs](../../../crates/sfmtool-py/src/geometry/reconstruction_growth.rs).
They build on absolute-pose estimation and refinement
([absolute-pose.md](absolute-pose.md)), batch triangulation
([batch-triangulation-api.md](../reconstruction/batch-triangulation-api.md)),
the staged bundle adjustment ([bundle-adjustment.md](bundle-adjustment.md)),
and cluster covisibility
([cluster-covisibility.md](../features/cluster-covisibility.md)).

`sfmtool.geometry.grow_reconstruction(cluster_indexes,
image_indexes, positions_xy, camera, quaternions_wxyz, translations,
posed_indexes, *, ba_window=0, anchor_every=0, ba_cluster_cap=0,
min_obs=8, accept_gate=0.35, seed=0, free_points_cross=True)` (its dict
carrying `point_at_infinity`) and
`resect_images_batch(cluster_indexes, image_indexes, positions_xy,
camera, points, image_list, *, posed_quaternions_wxyz=None,
posed_translations=None, posed_indexes=None, min_obs=8,
accept_gate=0.30, seed=0)`, NumPy in/out, following the geometry
submodule's conventions. The three `posed_*` arguments are all-or-none:
passing some but not all of them is a `ValueError`. `resect_images_batch`
reads every finite `points` row as a position, so a caller passing growth's
`points` to it sets the rows `point_at_infinity` marks to `NaN` first.

## Testing requirements

- Synthetic orbit (known poses, known focal): a small seed grows to full
  registration; camera errors under similarity alignment within tight
  bounds; released focal near truth.
- `ba_window` at or above the posed count reproduces the unbounded
  adjustment's result on the same input; a bounded window still registers
  the full synthetic orbit.
- `anchor_every` on a synthetic loop long enough to outrun the finishing
  adjustment's 120-camera spread subset yields lower final camera error
  than a frontier-only window of the same size. Below that length the
  finishing pass is effectively global and the two configurations
  converge to the same answer up to platform float noise, so the
  default-run test asserts only that anchoring does not degrade an
  already-converged loop; the discriminative 140-camera comparison is a
  minutes-scale `#[ignore]`d test run by hand.
- `resect_images_batch` matches one-at-a-time resection of the same
  images against the same fixed structure, and its parallel execution is
  deterministic for a fixed seed.
- Gates: images with junk-dominated observations are deferred, then
  force-accepted only when verification passes; a rejected force-accept
  leaves poses, structure and the adjustment set unchanged.
- Degenerate inputs (no seed poses, no triangulable clusters, all images
  below `min_obs`) return the input state with empty growth, not an error.
- The crossing: on the synthetic orbit with a far field 10⁶ units out, every
  image registers, as with the crossing off; at least 80% of the far clusters
  come back unit directions with finite residuals under a pixel, and no near
  point does; with the crossing off no point is a direction. On the orbit
  alone no point is a direction.
- Directions and resection: the one test of what a resection can use (a
  position, not a direction or a `NaN` row) holds for the gathered
  correspondences, the next-best-view count and the acceptance inlier
  fraction, each checked on a position, a direction and a `NaN` row; an image
  that sees only far clusters the adjustments stored as directions is not
  registered while every other image is, over five draws; and a rejected
  force-accept restores each cluster's representation with the poses, the
  structure and the adjustment set, checked on the saved state directly and
  through growth with a far field present.
- The representation that comes back is the one the last adjustment left: on
  a far field 10⁵ units out where the finishing adjustment stores a point in
  the other representation than growth left it in, and on every far-field
  test above, a row is a unit direction exactly where `point_at_infinity` is
  set.

## Non-goals

Seed construction (rotation initialization and factorization seeding are
separate kernels), focal estimation (the caller supplies it), confidence
flagging and attempt arbitration (caller policy), and loop-closure-style
global relaxation beyond the anchor adjustments (a final global solve is
the downstream consumer's concern).
