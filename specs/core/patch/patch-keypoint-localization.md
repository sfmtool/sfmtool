# Patch-Keypoint Localization (Congealing)

Patch-keypoint localization refines, for a single 3D point, exactly where that
point's piece of surface appears in each image that sees it. Given the point's
patch — a small oriented square of surface carrying a reference bitmap — a set of
views, and a rough starting keypoint in each of them, it registers all the views
against a shared consensus of the patch's appearance, moves every keypoint to
sub-pixel, and **drops the views that will not co-register** so the rest agree
against a cleaner consensus; it reports which views it kept. It is the per-point
refinement step of the [sift-based → patch-based reconstruction
pipeline](sift-to-patch-reconstruction.md). The keypoints it produces are defined
geometrically in [sfmr-file-format.md](../../formats/sfmr-file-format.md)
("Observation source"); this spec is one way to obtain them, not part of their
definition.

## Problem

An `embedded_patches` `.sfmr` stores, per observation, a 2D `keypoints_xy[j]`
that **anchors that observation's patch**: the surfel re-anchored within its
plane so its centre projects to the keypoint in that view (the relationship is
defined in the v4 format spec). We *refine* those keypoints for a point that
already carries an oriented patch (the v3 `(u, v)` frame + normal), starting from
a seed the caller supplies.

The naive keypoint is the projection of the point, `project_i(X_p)`. That can sit
slightly off where the surfel's appearance actually lands in a view, so
refinement searches a per-view sub-pixel offset `δ_j` that maximizes cross-view
patch correlation; the refined keypoint `project_i(X_p) + δ_j` then sits on the
image content.

## What the keypoint encodes

The keypoint and its anchor relationship are defined in
[sfmr-file-format.md](../../formats/sfmr-file-format.md). For observation `j`
(`i = image_indexes[j]`, `p = point_indexes[j]`):

```
keypoint_j = project_i(X_p) + δ_j        # image px
```

`δ_j` is the in-plane shift of the surfel centre for view `i`. The reader
recovers it by unprojecting the keypoint onto the patch plane.

## Inputs

- One 3D point `X_p` with its **patch frame**: the half-vectors `u_p`, `v_p` and
  normal `n_p = normalize(u_p × v_p)`.
- A **view set** `G`: the `nv` views (camera pose + intrinsics + source image) to
  refine.
- A **starting keypoint** per view — it should already be approximately right,
  since the refinement only nudges it; the 3D point's projection
  `project_i(X_p)` is a good seed, and is the seed a view supplies no keypoint
  for. The seed is **per view, independently optional**: a view set can mix views
  that carry a keypoint (an observation's stored position — the appearance that
  was actually matched) with views that carry none (a view added by [view
  selection](patch-view-selection.md) observes nothing, so it has no keypoint),
  and each is seeded from what it has. A view set in which no view carries a
  keypoint is the all-projection case, identical to supplying no seeds at all.
- **Drop thresholds** — the per-view gates the refiner uses to drop a view
  in-loop (below): `max_shift_px`, `min_relative_zncc`, and the grazing cutoff. The caller
  supplies them; the refiner stops dropping once only the LOO floor of two views
  remains and reports what survived (the per-point `min_views` cull is the
  caller's).

## Algorithm: group-wise translation registration (congealing)

For one point with view set `G` (the `nv` views to refine), first **pre-filter
grazing views** — drop any whose ray is near-parallel to `Π_p` (`|d · n_p|` below
the grazing cutoff), where the in-plane anchor is ill-conditioned and the view
would only contaminate the consensus. The surviving views are the consensus
membership; optionally that membership is *capped* at `K` and the views past the
cap register once against the finished consensus instead of joining it — see
[keypoint-localization-consensus-basis.md](keypoint-localization-consensus-basis.md).
Then maintain a per-view in-plane
coordinate `acc[v]` (patch-grid units) for the patch centre on `Π_p`, measured
from `X_p` and **initialized by unprojecting the starting keypoint onto `Π_p`**
(zero when the seed is the point's own projection). Each round:

1. **Render** every view's patch tile from its source image at its accumulated
   offset `acc[v]` — a *single* resample of the source, with the patch centre
   translated in-plane by `acc[v]` (never re-sampling an already-warped tile, so
   applying offsets across rounds cannot compound blur). Tiles are rendered onto
   a **context tile** larger than the scored `PATCH×PATCH` core so the shift
   search can slide without running off the edge.
2. **Consensus.** Build the robust (IRLS) z-normalized weighted-mean template
   over the stack — the same robust photometric consensus used by [patch-normal
   refinement](patch-normal-refinement.md).
3. **Per-view shift.** For each view `v`, search the residual in-plane shift that
   maximizes windowed ZNCC against the **leave-one-out** consensus of the *other*
   views (so a view is never aligned to a template its own pixels polluted): a
   full-res integer search then a separable parabolic sub-pixel fit. That integer
   search runs one of two strategies (`search_strategy`): by **default**,
   `PlusDescent` — a local "+"-descent that seeds at the current cell, steps to
   the best of its 4 axis neighbors, and stops when none improves (~6 cells per
   call), *not* a scan of the whole grid; `Exhaustive` scores every cell of the
   `(2·margin+1)²` grid and is retained as the global-argmax fallback (no
   local-optima risk). See
   [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md)
   for the strategy trade-off. The parabolic
   fit is a cheap estimate to converge and seed on; an accurate sub-pixel keypoint
   is a separate continuous-photometric algorithm
   ([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md)) that runs
   after this converges.
4. **Accumulate** `acc[v] += δ_v`, clipping the total move from the point's
   projection (`acc[v]`, which `project_i(X_p)` sets to zero) to `±search` — the
   same anchor the `max_shift_px` gate below uses.
5. **Drop failing views.** Remove any view whose keypoint has left the frame,
   whose keypoint sits more than `max_shift_px` from the point's projection
   (`|acc[v]|` mapped to source-image px — an *absolute* distance from
   `project_i(X_p)`, not the move from the seed), or whose leave-one-out ZNCC
   falls below `min_relative_zncc` of the views' median LOO ZNCC (relative, so a
   low-texture patch isn't over-dropped); the next round's consensus is rebuilt
   from the survivors, so the remaining views register against a cleaner
   template. Stop dropping once only two views (the LOO floor) remain.
6. **Repeat** to convergence or a small iteration cap (default 5). Convergence
   is the mean **round-over-round change** of each view's refined position
   (integer accumulator + sub-pixel residual, this round vs the previous one)
   dropping below `convergence_px` (`~0.05` px), **and** the view set having
   survived the round unchanged — a round that dropped a view changed the
   consensus the survivors registered against, so they always get at least one
   more round against the survivor-only template before the stationarity test
   can fire. (The raw per-round search output is *not* the metric: it includes
   the freshly recomputed parabolic residual, which never moves the read
   position, so its magnitude has a fraction-of-a-pixel floor even once the
   search stops moving.)

The converged `acc[v]`, mapped from patch-grid units back to image pixels via the
view's projection, is `δ_j`; the emitted keypoint is `project_i(X_p) + δ_j`.

### Why these guards matter

- **No compounding blur.** Each round re-renders the tile from the source at the
  accumulated offset instead of re-warping an already-warped tile, so
  interpolation blur can't compound across iterations. Re-rendering from the
  source every round is the simple thing to start with; rendering one expanded
  cache per view up front and reading each round's core from *that* removes the
  redundant renders — and because an in-plane shift maps to an integer index
  shift in the cache, the read is **exact** for integer offsets (no extra blur),
  not an approximation. That cache, plus a SIMD search over it, is specified in
  [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md).
- **Leave-one-out scoring** is the honest "did it register?" signal: mean per-view
  ZNCC against the consensus of the *others* can only rise if the views genuinely
  co-register; a template fitting its own noise would inflate self-agreement but
  not LOO.
- **In-loop dropping** removes a contaminating view as soon as it reveals itself,
  so the consensus the *other* views refine against keeps improving — better than
  refining the whole set and discarding afterward.

## Mapping a shift to a keypoint

`acc[v]` locates the patch centre on `Π_p` in patch-grid units, measured from
`X_p`; the starting keypoint sets its initial value and refinement moves it. The
keypoint is the image projection of that centre:

```
center_v = X_p + acc[v].s · û_p + acc[v].t · v̂_p     # patch centre on Π_p
keypoint_j = ray_to_pixel_i(R_i · center_v + t_i)
```

Unprojecting the emitted keypoint back onto `Π_p` recovers `center_v` — the
inverse of the format spec's reader relationship (`keypoint → anchor` by
ray∩plane), so a producer and a reader round-trip.

## Outputs

The algorithm returns:

- the **kept views** — a mask over the input `G` of which views survived the
  in-loop drops (grazing, out-of-frame, large-shift, low-agreement);
- per kept view, its **refined keypoint** (`project_i(X_p) + δ_j` in the format's
  terms) and **quality signals** — its offset from the point's projection
  (`acc[v]` mapped to source-image px) and the final leave-one-out ZNCC against
  the other views' consensus.

## Implementation

The refiner is `localize_patch_keypoints` / `localize_patch_cloud_keypoints` in
[keypoint_localize.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize.rs),
bound as `PatchCloud.localize_keypoints` and called per point by the pipeline in
[_embed_patches.py](../../../src/sfmtool/_embed_patches.py). It reuses the
existing patch machinery:

- Patch rendering per view reuses `WarpMap::from_patch`
  and the patch cloud ([patch-cloud.md](patch-cloud.md)) — already camera-model
  agnostic via `ray_to_pixel`.
- The robust consensus reuses the IRLS template from patch-normal refinement.
- The per-view sub-pixel ZNCC shift search is the new kernel it
  adds, in the same crate alongside the rendering and consensus it sits between.

## Parameters (defaults)

| parameter | default | meaning |
|---|---|---|
| `max_iters` | 5 | max congealing rounds (stops early at convergence) |
| `search` | 6 px | max total per-view drift from the projection (patch-grid px), bounds runaway; also the context-tile margin |
| `max_shift_px` | ~3 | drop a view whose keypoint sits more than this from the point's projection (source-image px) |
| `min_relative_zncc` | ~0.7 | drop a view whose LOO ZNCC falls below this fraction of the views' median LOO ZNCC |
| `min_grazing_cos` | 0.1 | pre-filter a view whose ray is near-parallel to the plane (`|d̂·n̂|` below this) |
| `resolution` | 24 | the `R×R` patch grid the consensus / ZNCC are scored on |
| `robust_iters` | 3 | IRLS passes for the robust consensus |
| `convergence_px` | 0.05 | stop once a round's mean round-over-round change of the per-view refined positions is below this (patch-grid px) |
| `search_strategy` | `PlusDescent` | per-(view, round) shift-grid traversal: `PlusDescent` (default, local "+"-descent) or `Exhaustive` (full-grid global-argmax fallback); see [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) |
| `search_resolution_multiplier` | 1.0 | discrete-search resolution multiplier `m` (`1.0` = no-op); see [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md) |

(plus `window` and `sampler`, shared with [normal refinement](patch-normal-refinement.md).)

The patch size is carried by the frame the algorithm is handed (the `(u, v)`
half-vectors).

## Implementation details

`PatchCloud.localize_keypoints(recon, images, *, view_sets=None,
max_iters=5, search=6.0, max_shift_px=3.0, min_relative_zncc=0.7,
min_grazing_cos=0.1, resolution=24, …, point_indexes=None)` returns a per-point
`{point_index, views, keypoints, offsets_px, loo_zncc}`. Each round renders a
**context tile** per view (the scored `R×R` core extended by `±⌈search⌉` px so the
shift search slides without re-warping), z-normalizes the cores into a shared
compacted channel space (a channel flat in any view is dropped, as in normal
refinement), builds the leave-one-out IRLS consensus of the *other* views, and
runs a full-res integer windowed-ZNCC search refined by a separable parabolic fit;
the per-view offset accumulates and is clipped to `±search`. A view is dropped when
its core leaves the frame (any window-support pixel out of frame), its keypoint
sits more than `max_shift_px` from the projection, or its leave-one-out ZNCC falls
below `min_relative_zncc ×` the views' median — all subject to the two-view
leave-one-out floor (when the gates would leave fewer than two, the two
best-agreeing views are kept, so a kept pair can exceed `max_shift_px`). Grazing
views (`|d̂·n̂| < min_grazing_cos`) are pre-filtered. The view set is deduped
order-preserving, and seeds default to the point's own projection (`acc = 0`); a
supplied starting keypoint is unprojected onto the plane to initialize `acc`. The
render → z-normalize → robust-consensus primitives are shared with `normal_refine`
(`pub(super)`), not duplicated. Defaults match the table above with
`min_grazing_cos = 0.1`, `robust_iters = 3`, and `convergence_px = 0.05`. The
starting-keypoint seed is a per-view `Option` parallel to the view set
(`localize_patch_keypoints(..., starting_keypoints, ...)`, exposed on the PyO3
binding as `starting_keypoints={point_index: [[x, y] | None, ...]}`): `None` for
one view seeds that view at its projection while its siblings keep their explicit
seeds, and an all-`None` list is bit-identical to supplying no seeds.
