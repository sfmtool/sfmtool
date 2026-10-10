# Patch-Keypoint Localization

Patch-keypoint localization finds, for a single 3D point, exactly where that
point's piece of surface appears in each image that sees it. Given the point's
patch (a small oriented square of surface), a set of views, a rough starting
keypoint in each of them, and which of those views is the point's reference
observation, it renders the reference observation's patch tile at its own
keypoint, slides every other view's tile over a small window to where it
matches that render best, and moves each keypoint there to sub-pixel. It
**drops the views that do not match**, and reports the views it kept. It is the
per-point keypoint step of the [sift-based → patch-based reconstruction
pipeline](sift-to-patch-reconstruction.md), of `sfm xform --refine-keypoints`,
of the bench's *Fit* and of Track at Pixel. The keypoints it produces are
defined geometrically in [sfmr-file-format.md](../../formats/sfmr-file-format.md)
("Observation source"); this spec is one way to obtain them, not part of their
definition.

## Rust API

The kernel is in
[keypoint_localize.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize.rs),
with the one-pass alignment in
[keypoint_localize/align.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/align.rs),
the parameters and result in
[keypoint_localize/params.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/params.rs),
the shift search in
[keypoint_localize/search.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/search.rs)
(described in [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md)),
and the keypoint-to-grid mapping in
[keypoint_localize/seed.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/seed.rs).

```rust
pub fn localize_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],          // one per reconstruction image
    view_set: &[u32],                      // the views to localize
    starting_keypoints: Option<&[Option<[f64; 2]>]>, // parallel to view_set
    reference: Option<usize>,              // position in view_set of the reference
    params: &KeypointLocalizeParams,
) -> KeypointLocalization;

pub fn try_localize_patch_keypoints(
    /* the same arguments */, progress: &Progress<'_>,
) -> Result<KeypointLocalization, LocalizeError>;

pub fn localize_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    references: Option<&[Option<usize>]>,  // parallel to the cloud
    params: &KeypointLocalizeParams,
    done: Option<&AtomicUsize>,
    progress: &Progress<'_>,
) -> Result<Vec<KeypointLocalization>, Cancelled>;

pub fn keypoint_grid_offset(
    patch: &OrientedPatch, view: &ProjectedImage<'_>,
    keypoint: [f64; 2], params: &KeypointLocalizeParams,
) -> Option<[f64; 2]>;

pub fn view_cache_bytes(params: &KeypointLocalizeParams, channels: usize) -> usize;
```

**Why it is shaped this way.**

- **The reference is an argument, not something the kernel keeps.** The
  reference observation is stored with the point (`tracks/reference_observations`
  in the `.sfmr` file, an index within the point's own track) or held by a
  bench track (`TrackPayload::reference`), so the caller is the one that knows
  it. It passes the position of that observation in `view_set`. `None` has the
  kernel ask the [reference-view rule](reference-view.md) instead, so a caller
  with no reference gets the same template a writer of the stored bitmap would
  pick.
- **Each starting keypoint is independently optional.** A view set can mix
  views that carry a keypoint (an observation's stored position) with views
  that carry none (a view added by [view selection](patch-view-selection.md)
  observes nothing, so it has no keypoint); `None` starts that view at the
  point's projection `project_i(X_p)`. An all-`None` list is the same as no
  list.
- **The fallible entry point exists because the buffers are the caller's to
  size.** The context tile grows as the square of `search` (below), so a caller
  that takes the radius from a person runs `try_localize_patch_keypoints`, which
  reports `LocalizeError::OutOfMemory` and `LocalizeError::Cancelled` instead of
  aborting the process. `localize_patch_keypoints` is the same call for callers
  whose radius is a constant.
- **The batch form is parallel over points** (rayon) and takes a `done` counter
  for a Python poller and a `Progress` that counts `patches`, is polled for
  cancellation before each patch and between a patch's views, and, when
  detailed, times each sampler's renders in its own detail phase.

The kernel is bound as `PatchCloud.localize_keypoints` and called per point by
the pipeline in [_embed_patches.py](../../../src/sfmtool/_embed_patches.py).

**Example.** The first view of the set is the point's reference observation.
It is the reference reported unless the grazing pre-filter turns it away or its
tile cannot be rendered at its keypoint:

```rust
let localized = try_localize_patch_keypoints(
    patch,
    views,
    view_set,
    None,      // every view starts at the point's projection
    Some(0),   // view_set[0] is the reference observation
    &KeypointLocalizeParams::default(),
    &Progress::none(),
)?;
if let Some(reference) = localized.reference {
    let k = localized.views.iter().position(|&v| v == reference).unwrap();
    assert_eq!(localized.zncc[k], 1.0);
}
```

## What the keypoint encodes

The keypoint and its anchor relationship are defined in
[sfmr-file-format.md](../../formats/sfmr-file-format.md). For observation `j`
(`i = image_indexes[j]`, `p = point_indexes[j]`):

```
keypoint_j = project_i(X_p) + δ_j        # image px
```

`δ_j` is the in-plane shift of the surfel centre for view `i`. The reader
recovers it by unprojecting the keypoint onto the patch plane. The naive
keypoint, `project_i(X_p)`, can sit slightly off where the surfel's appearance
actually lands in a view; localization chooses `δ_j` so the keypoint sits on the
image content.

## The template: the reference render

Every view is aligned to one template, the **reference render**: the `R×R`
render of the point's reference observation at its own starting keypoint, with
the sampler the [sampler rule](../camera/image-warping.md) picks for that
observation. It is the tile a `.sfmr` stores as the point's patch bitmap
([reference-view.md](reference-view.md#the-stored-bitmap)), so the views are
placed against the same pixels every score on the bench reads them against.

**Which reference.**

1. The caller's: the position in `view_set` of the point's stored reference
   observation, or on the bench the reference the track holds. It is matched
   by image, so a reference given at the dropped slot of a repeated image still
   counts.
2. Where the caller passes none, or the one it passes is turned away by the
   grazing pre-filter, the reference-view rule picks one from the views' renders
   at their starting keypoints, as every render of the stored bitmap does.
3. Where the rule picks none it would store (it picks no view, or reaches its
   pick only through its last fallback), the template is the **fused mean** of
   the views at their starting keypoints (the IRLS-weighted mean of
   [keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md#the-fused-mean)),
   which is then the point's stored bitmap. No view is held fixed, and every
   view is searched. Where no fused mean renders either, the rule's
   last-fallback pick is the reference after all.

A view set of one view is its own reference. The resolution is one function,
`resolve_reference`, which the sub-pixel refiner calls too, with the same
grazing pre-filter.

**The reference is not moved.** Aligning it to its own render would return it
where it is. Whatever error the
reference's keypoint carries is shared by every view aligned to it, so it moves
the triangulated point rather than adding reprojection error, and the
per-observation confidence, not the alignment, is where its uncertainty belongs.
The reference is returned at exactly the keypoint it was given, with ZNCC
`1.0`, and faces no gate after the grazing pre-filter.

**The template is never blurred.** A blurred template places views no closer
and lowers the curvature of the correlation peak, so the peak is placed less
precisely ([sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md)
Part 6). A blur-matched score may be read afterwards, for judging a view, but
the alignment reads the render as it is.

**One pass.** The template does not change while the views are aligned, so each
view is searched once. There are no rounds, no convergence test and no weights
that depend on the other views.

**When the reference changes** (*Set as reference* on the bench, unpinning the
reference row, a writer that picks again), the views are aligned to the new
reference at the next localization. A low score is not a reason to move a view;
only a localization moves it.

## Algorithm

For one point with view set `G`:

1. **Deduplicate** the view set in order (a point can carry two observations in
   one image), and **pre-filter grazing views**: drop a view whose ray is
   near-parallel to the plane (`|d̂ · n̂|` below `min_grazing_cos`), where the
   in-plane anchor is ill-conditioned, and a view the point does not project
   into.
2. **Seed.** Each remaining view's starting keypoint is unprojected onto the
   patch plane to its starting offset `start[v]`, in patch-grid px from `X_p`
   (zero for a view started at the projection). The sampler for every render of
   the view is chosen once, at that keypoint.
3. **Resolve the reference** and render the template, as above. Where nothing
   renders to align to (the reference's tile leaves the frame or is flat at its
   keypoint, its keypoint does not map onto the patch plane, or no fused mean
   renders), there is no reference: every view, the given reference included,
   keeps its starting keypoint exactly as given, unscored, and still faces the
   `max_shift_px` gate. The refiner does the
   same, so both report no reference for such a point.
4. **Per view other than the reference:**
   - render one **context tile**, the `R×R` core extended by `margin =
     ⌈search⌉` grid px on every side, centred on `start[v]`;
   - read the **member self-similarity gate** on the tile's core (below), and
     drop the view if it fails;
   - **search** the integer shift over `±margin` that maximizes the windowed
     ZNCC of the view's core against the template, by `search_strategy`
     (`Exhaustive` by default, which scores every cell and takes the
     highest; `PlusDescent` climbs from the zero shift to the nearest peak),
     then refine it to sub-pixel by the vertex of a quadratic fitted to the
     3×3 cells around the peak (see
     [Sub-pixel step](keypoint-localization-search-cache.md#sub-pixel-hand-off)).
     The view's
     final offset is `start[v]` plus that shift, and its score is the ZNCC at
     the integer peak. A view no shift could be scored for (its core out of
     frame at every shift) is dropped.
5. **Gates.** A searched view is dropped when its keypoint leaves the frame,
   when its keypoint sits more than `max_shift_px` from the point's projection
   (an absolute distance from `project_i(X_p)` in source px, not the move from
   its start), when its ZNCC is finite and below `min_absolute_zncc`, or when it
   is below `min_relative_zncc` times the median ZNCC of the views other than
   the reference.

The search, the context tile and the kernels that make them cheap are in
[keypoint-localization-search-cache.md](keypoint-localization-search-cache.md).
The quadratic step is an estimate; an accurate sub-pixel keypoint is the job of
the continuous refiner
([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md)), which runs
after this. It resolves the reference by the same function, with the same
grazing pre-filter, so given the reference this reports it aligns to the same
observation's render; where this aligned to the fused mean, it resolves again
from the kept views at their new keypoints.

**Every render is from the source photograph.** The context tile is rendered
once per view from the source image, and every candidate shift reads it at an
integer offset, which is exactly the render at that shift, so no tile is ever
re-warped from another tile and interpolation blur cannot compound.

**A starting keypoint is not clipped.** The tile is centred on it, so the search
reaches `±search` grid px around wherever it starts, and `max_shift_px` then
judges where the view ended. `keypoint_grid_offset` gives the same offset the
seeding computes, on the params' own grid, for a caller that wants to know how
far a keypoint sits from the projection in grid px.

## Mapping a shift to a keypoint

`off[v] = start[v] + shift` locates the patch centre on the plane in patch-grid
units, measured from `X_p`. The keypoint is the image projection of that centre:

```
center_v = X_p + off[v].s · wpp_u · û_p − off[v].t · wpp_v · v̂_p   # patch centre on the plane
keypoint_j = ray_to_pixel_i(R_i · center_v + t_i)
```

Grid rows count downward from `+v̂`, so a positive `t` steps along `−v̂`, as in
`WarpMap::from_patch`. Unprojecting the emitted keypoint back onto the plane
recovers `center_v`: the inverse of the format spec's reader relationship
(`keypoint → anchor` by ray∩plane), so a producer and a reader round-trip.

## Outputs

`KeypointLocalization` holds parallel arrays over the **kept** views, in the
input order:

- `views`: the kept image indices. The reference observation is always kept
  when its render is the template. Every other view can be dropped, so the set
  can hold one view, or none, for the caller's `min_views` cull to remove.
  Where there was nothing to align to, every view is kept at its start unless
  the `max_shift_px` gate drops it;
- `keypoints`: the localized keypoint per kept view, source px; the reference's
  is its starting keypoint, exactly;
- `offsets_px`: the keypoint's distance from the point's projection, source px;
- `zncc`: the plain ZNCC against the template at the search's integer peak:
  `1.0` for the reference, `NaN` for a view that was not searched because there
  was no template;
- `reference`: the image index of the reference observation the views were
  aligned to, or `None` where they were aligned to the fused mean or to nothing.

## Parameters (defaults)

| parameter | default | meaning |
|---|---|---|
| `search` | 6 | the reach of each view's search around its starting keypoint, patch-grid px: the context tile's margin on every side, and the `±search` shift window |
| `max_shift_px` | 3 | drop a view whose keypoint sits more than this from the point's projection (source px); never applied to the reference |
| `min_relative_zncc` | 0.7 | drop a view whose ZNCC against the template falls below this fraction of the median over the views other than the reference; `0` disables |
| `min_absolute_zncc` | 0.5 | drop a view whose ZNCC against the template is finite and below this floor, however many views remain; `0` disables |
| `max_member_zncc_self_similarity_radius` | 2.5 | drop a view whose own core's [ZNCC self-similarity radius](zncc-self-similarity-radius.md) is above this (patch-grid px); `0` disables, and `3` or more rejects nothing; see [The member gate's default](#the-member-gates-default) |
| `min_grazing_cos` | 0.1 | pre-filter a view whose ray is near-parallel to the plane (`|d̂·n̂|` below this) |
| `resolution` | 24 | the `R×R` patch grid the template and the ZNCC are scored on |
| `window` | `GaussianDisk { sigma: 0.6 }` | per-pixel scoring weight and support, shared with [normal refinement](patch-normal-refinement.md) |
| `sampler` | the sampler rule | which sampler renders each view's tiles, chosen once per view at its starting keypoint ([image-warping.md](../camera/image-warping.md) § "Choosing the sampler per view"), so the localizer reads a view through the sampler the bench and the stored bitmap read it through |
| `robust_iters` | 3 | IRLS passes for the fused mean, where it is the template |
| `search_strategy` | `Exhaustive` | how each view's shift grid is traversed: `Exhaustive` or `PlusDescent`; see [How the alignment was measured](#how-the-alignment-was-measured) |

The patch size is carried by the frame the kernel is handed (the `(u, v)`
half-vectors). Setting `min_absolute_zncc`, `min_relative_zncc` or
`max_member_zncc_self_similarity_radius` to `0` (or a non-finite value) disables
that gate exactly: the relative bar reads `0` as "off" rather than as a bar at
zero, so a caller that asked for no gate keeps even an anti-correlated view and
its number, which is what a *reading* of a track rather than a fit of one needs
([editable-track.md](../bench/editable-track.md)).

## The member self-similarity gate

A view whose own tile fixes no 2D position (a flat sky or water crop, a lone
straight edge) matches itself a few pixels away, so its ZNCC against anything
cannot place it. The member gate refuses such a view before it is searched.

It reads each view's own `R×R` core at its starting offset, cut from the context
tile rendered for its search, with the default `SelfSimilarityParams`
(`max_radius` `r = 3`), read
[the overlap way](zncc-self-similarity-radius.md#the-overlap-reading): at each
shift only the samples both windows hold inside the core are correlated, so no
pixel of the tile around the core enters the reading, and the radius does not
depend on the search radius. Every sample counts as data; a pixel out of frame
reads as the black it was rendered as, as every other read of the tile does.
The reference is not read: it is not moved, so it is not placed by its own
texture.

A view passes when its radius is at or below the bar
(`KeypointLocalizeParams::admits_member_zncc_self_similarity_radius`). A `NaN`
radius fails an active gate. A bar of `0` or a non-finite bar is off, and the
radius is then not read at all. The radius reads at most `r`, which stands for
"that far or further", so a bar of `r` or more reads every view and turns none
out. A flat tile reads `r` and so does a straight edge, since each matches
itself along the whole search.

### The member gate's default

The default, `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`, is `2.5` patch-grid
px, the same bar as the bench's `BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`. The
user chose it from a sweep on the seoul_bull and kerry_park ground truths: two
measurements run with the gate off and with the gate at bars from 1 to 2.9. The
sweep was run when the localizer aligned views to a leave-one-out consensus of
the other views; the gate reads each view's own tile at its starting keypoint,
which did not change with the alignment, so the radii it judges are the same.

The sweeps read the radius of a tile rendered `r` px wider than the core, with
the core as the template. The overlap reading the gate uses reads a few
hundredths of a grid px shorter on average and gives a different verdict on 1%
to 3% of views at 2.5, nearly all a pass where the wider tile's reading fails,
so the bar means nearly the same on it
([zncc-self-similarity-radius.md](zncc-self-similarity-radius.md#why-only-the-bitmap-is-read)).

The [add-image-to-tracks harness](../../../scripts/add_image_to_tracks/README.md),
default rule, resected pose, summed over every image. *Recall* is the known
tracks rejoined, *> 2 px* the rejoined keypoints over 2 px from the ground
truth's, *extra* the tracks joined that the image was not in, *x bad* those whose
new observation's residual after retriangulation is over 2 px, and *x worse*
those whose largest residual grew by more than 1 px:

| member gate | seoul recall | > 2 px | extra | x bad | kerry recall | > 2 px | extra | x bad | x worse |
|---|---|---|---|---|---|---|---|---|---|
| off | 91.2% | 20 | 126 | 0 | 90.0% | 14 | 1432 | 2 | 18 |
| radius ≤ 1 | 39.7% | 10 | 50 | 0 | 24.4% | 10 | 299 | 3 | 13 |
| radius ≤ 1.5 | 66.7% | 17 | 74 | 0 | 46.7% | 12 | 625 | 5 | 15 |
| radius ≤ 2 | 76.8% | 19 | 94 | 0 | 66.2% | 12 | 846 | 0 | 7 |
| **radius ≤ 2.5 (default)** | **80.4%** | **19** | **106** | **0** | **73.8%** | **12** | **985** | **0** | **6** |
| radius ≤ 2.9 | 81.2% | 19 | 104 | 0 | 75.7% | 13 | 1010 | 0 | 6 |

The localizer over each ground truth's own tracks, every finite point, each view
seeded at the point's projection, each kept view's keypoint against the ground
truth's observation (1233 observations on seoul_bull, 2193 on kerry_park):

| member gate | seoul kept | err med / p90 px | > 2 px | points under 2 views | kerry kept | err med / p90 px | > 2 px | points under 2 views |
|---|---|---|---|---|---|---|---|---|
| off | 97.8% | 0.185 / 0.789 | 35 | 0 | 98.5% | 0.170 / 0.664 | 51 | 1 |
| radius ≤ 1.5 | 72.5% | 0.156 / 0.692 | 32 | 53 | 58.7% | 0.149 / 0.585 | 32 | 99 |
| radius ≤ 2 | 83.7% | 0.169 / 0.823 | 41 | 29 | 75.0% | 0.160 / 0.624 | 42 | 54 |
| **radius ≤ 2.5 (default)** | **87.0%** | **0.173 / 0.796** | **39** | **21** | **82.4%** | **0.161 / 0.602** | **43** | **39** |
| radius ≤ 2.9 | 88.0% | 0.173 / 0.792 | 39 | 17 | 84.4% | 0.164 / 0.633 | 45 | 33 |

At 2.5 the harness recovers 80.4% of the known tracks on seoul_bull and 73.8% on
kerry_park, against 91.2% and 90.0% with the gate off. On kerry_park it adds 0 bad extra
observations and 6 that make a track's largest residual worse, against 2 and 18
with the gate off. Against the current ground truths the sightings it turns
out are about as accurate as the ones it keeps: over the known tracks the
harness's default rule accepts with the gate off, those whose new view's core
reads the largest radius land within 1 px of the ground truth 96.6%
(seoul_bull, 119) and 98.7% (kerry_park, 318) of the time, against 97.9% and
98.5% for a radius under 1.

The lost recall is not taken as a reason to loosen the bar. On seoul_bull the
ground-truth sightings that read over 2.5 are mostly smooth surfaces (the
bull's bronze) seen where the patch covers only a few image pixels: at a texel
scale under 0.5 image px per patch-grid px, 48.6% of sightings read over 2.5,
against about 1% at 0.75 and above. Those patches should have been bigger, and
the ground truths are to be revised with SfM Explorer showing the radius. The
radius is measured in patch-grid px, so a grid finer than the image pixels it
samples renders a smooth tile that matches itself a few grid px away; whether
a radius read in image px would separate the views better is the open question
in [zncc-self-similarity-radius.md](zncc-self-similarity-radius.md).

## How the alignment was measured

Aligning to the reference render replaced congealing, which aligned each view
over several rounds to the robust (IRLS) mean of all the other views and scored
it by that leave-one-out ZNCC. The two are compared on the seoul_bull (259
tracks) and kerry_park (380 tracks) ground truths, every point with at least
three views, by the harness in
[`scripts/keypoint_localization/`](../../../scripts/keypoint_localization/README.md),
which has the commands, the full tables and the views kept. Congealing is built
from commit `83ffb08e`, the last with it; the alignment includes the 3×3
quadratic sub-pixel step. These figures were recorded on 2026-10-09.

**Protocol.** The poses the ground truth stores are taken as correct, and a
view's ground-truth keypoint is the projection of the patch centre. At a
displacement of 0 every view starts at its stored keypoint; at `d` px every
view starts `d` px from its ground-truth keypoint in a random direction. The
agreement bars and `max_shift_px` are off and the member gate is at its
default. Two variants differ in the reference:

- **Reference displaced**, the comparison on equal terms: the reference starts
  `d` px off like every other view, as its keypoint does in real use.
- **Reference stored**: the reference keeps its stored keypoint, which gives
  the alignment a template rendered at an almost correct keypoint and
  congealing none. This favours the alignment; it is how the alignment was
  first measured.

Two biases remain in both. The ground-truth points were triangulated from
keypoints that congealing placed, which favours congealing at 0 px. And the
ground-truth poses are themselves solved, so a residual of about 0.15 to 0.3 px
is the floor of either method.

Two errors are read per kept view:

- the **raw error**, its distance from the ground-truth keypoint. A view
  aligned to a displaced reference carries the reference's offset with it, so
  with the reference displaced the alignment's raw error is close to `d` even
  where the views agree perfectly with each other;
- the **re-triangulated residual**, its reprojection residual after the track
  is re-triangulated from its kept keypoints at the ground-truth poses. This
  leaves out an offset all views share, so it is the measure that compares the
  two methods with the reference displaced. A few tracks per run fail to
  re-triangulate (residual over 5 px; 0 to 5 per cell) and weigh heavily on
  the mean, so the median is the steadier figure.

**Re-triangulated residual, mean / median (px), reference displaced:**

| dataset | disp px | congealing | "+"-descent | exhaustive |
|---|---|---|---|---|
| seoul_bull | 0 | **0.357 / 0.261** | 0.428 / 0.263 | 0.436 / 0.264 |
| seoul_bull | 0.5 | 0.349 / 0.271 | **0.340 / 0.265** | 0.369 / 0.272 |
| seoul_bull | 1 | 0.355 / 0.279 | **0.346 / 0.262** | 0.373 / 0.263 |
| seoul_bull | 2 | 0.522 / 0.316 | 0.623 / 0.331 | **0.439 / 0.292** |
| seoul_bull | 3 | 0.954 / 0.427 | 1.276 / 0.633 | **0.796 / 0.412** |
| kerry_park | 0 | 0.207 / 0.160 | **0.202 / 0.156** | 0.232 / 0.157 |
| kerry_park | 0.5 | 0.216 / 0.164 | **0.204 / 0.158** | 0.236 / 0.159 |
| kerry_park | 1 | 0.252 / 0.177 | 2.469 / 0.163 | **0.240 / 0.160** |
| kerry_park | 2 | 0.672 / 0.290 | 4.387 / 0.437 | **0.527 / 0.254** |
| kerry_park | 3 | 2.185 / 0.751 | 1.942 / 1.469 | **1.297 / 0.712** |

The "+"-descent figures here and below are from before its walk moved on to a
diagonal neighbour that beats the cell it stopped at
([keypoint-localization-search-cache.md](keypoint-localization-search-cache.md#-descent)).
Run again since, its medians in this table move by at most 0.022 px.

**Raw error, mean / median (px), and the share of views more than 1.5 px from
the ground truth, reference displaced:**

| dataset | disp px | congealing | "+"-descent | exhaustive |
|---|---|---|---|---|
| seoul_bull | 0 | 0.559 / 0.306, 5.9% | 0.509 / 0.284, 5.9% | 0.607 / 0.284, 6.7% |
| seoul_bull | 1 | 0.904 / 0.754, 10.2% | 1.135 / 1.000, 13.1% | 1.176 / 1.000, 14.0% |
| seoul_bull | 3 | 2.727 / 2.644, 88.5% | 3.158 / 3.000, 95.9% | 3.022 / 3.000, 95.9% |
| kerry_park | 0 | 0.250 / 0.180, 0.6% | 0.247 / 0.174, 0.9% | 0.271 / 0.175, 1.4% |
| kerry_park | 1 | 0.746 / 0.659, 7.2% | 0.915 / 0.894, 7.2% | 0.931 / 0.897, 7.2% |
| kerry_park | 3 | 2.380 / 2.255, 73.9% | 2.984 / 2.970, 86.7% | 2.781 / 2.698, 83.8% |

**Reference stored, mean / median (px):**

| dataset | disp px | method | raw error | re-triangulated | over 1.5 px |
|---|---|---|---|---|---|
| seoul_bull | 1 | congealing | 0.796 / 0.614 | 0.363 / 0.278 | 8.3% |
| seoul_bull | 1 | "+"-descent | **0.510** / 0.290 | 0.363 / **0.265** | **5.8%** |
| seoul_bull | 1 | exhaustive | 0.604 / **0.289** | **0.359** / 0.269 | 6.6% |
| seoul_bull | 3 | congealing | 1.945 / 1.799 | 0.720 / 0.377 | 60.3% |
| seoul_bull | 3 | "+"-descent | 0.759 / 0.318 | 0.635 / 0.329 | 11.7% |
| seoul_bull | 3 | exhaustive | **0.636 / 0.306** | **0.407 / 0.275** | **7.4%** |
| kerry_park | 1 | congealing | 0.645 / 0.554 | 0.254 / 0.177 | 5.0% |
| kerry_park | 1 | "+"-descent | **0.263 / 0.182** | **0.210 / 0.158** | **1.0%** |
| kerry_park | 1 | exhaustive | 0.292 / **0.182** | 0.242 / 0.160 | 1.4% |
| kerry_park | 3 | congealing | 2.005 / 1.859 | 4.894 / 0.550 | 61.0% |
| kerry_park | 3 | "+"-descent | 1.156 / 0.332 | 1.205 / 0.571 | 24.6% |
| kerry_park | 3 | exhaustive | **0.681 / 0.278** | **0.635 / 0.301** | **10.7%** |

**Time per track** with the reference given, from the stored keypoints, by
track length, in ms: the best of three single-threaded runs of the harness
(`--threads 1`), each method run on a point straight after the others, so the
ratios hold better than the absolute times on this machine's mixed cores. The
long tracks are 10 tracks per length range of a DnDTabletop reconstruction
and 10, 10 and 20 of a DinoLedge one, spread evenly over each range by the
harness's `--bins` option:

| dataset | views | congealing | "+"-descent | exhaustive | exhaustive / "+"-descent | exhaustive / congealing |
|---|---|---|---|---|---|---|
| seoul_bull | 3-5 | 1.34 | **0.46** | 0.56 | 1.21 | 0.42 |
| seoul_bull | 6-10 | 2.65 | **0.87** | 1.05 | 1.21 | 0.40 |
| kerry_park | 3-5 | 1.91 | **1.02** | 1.13 | 1.12 | 0.60 |
| kerry_park | 6-10 | 3.57 | **1.82** | 2.08 | 1.15 | 0.58 |
| kerry_park | 11-20 | 5.08 | **2.98** | 3.47 | 1.16 | 0.68 |
| kerry_park | 21+ | 6.29 | **4.83** | 5.69 | 1.18 | 0.90 |
| DnDTabletop | 20-49 | 10.89 | **8.02** | 9.16 | 1.14 | 0.84 |
| DnDTabletop | 50-99 | 17.39 | **13.56** | 15.36 | 1.13 | 0.88 |
| DnDTabletop | 100-199 | 38.00 | **31.65** | 36.33 | 1.15 | 0.96 |
| DnDTabletop | 200-399 | **49.22** | 50.36 | 57.98 | 1.15 | 1.18 |
| DinoLedge | 20-49 | **7.55** | 7.57 | 8.44 | 1.11 | 1.12 |
| DinoLedge | 50-99 | **12.47** | 13.83 | 15.62 | 1.13 | 1.25 |
| DinoLedge | 100-199 | **8.15** | 22.81 | 23.16 | 1.02 | 2.84 |

At the default `search` of 6 (a 13×13 shift grid) the exhaustive search costs
1.1 to 1.2 times the "+"-descent per track at every length. Per search it
takes 113 µs against the descent's 50 µs, but rendering the views and the
reference is about 90% of the localizer's time either way, and a whole
`embed_patches` run takes the same time with either: the best of two runs is
0.84 to 0.87 s on kerry_park and 0.24 to 0.25 s on seoul_bull with all
threads. The cost grows with the window: at `search` 9 (19×19, wider than the
AVX2 grid kernel takes, so the scalar kernel runs) the exhaustive search costs
1.2 to 2.0 times the descent per track (1.7 to 2.0 on DnDTabletop, 1.2 to 1.8
on DinoLedge). That matters
to the bench, whose evaluation widens the window by the farthest starting
keypoint's offset: its `search` is the track's `max_shift_px` (6 by default)
plus that offset, so a seed more than 1 grid px off takes it past 7 and the
whole window is scored in the scalar kernel.

Where the point has no reference and the reference-view rule picks one, the
pick adds a mean of 1.4 ms per track on seoul_bull and 3.9 ms on kerry_park.

What it shows:

- **With the exhaustive search, the default, the alignment is as accurate as
  congealing or more by the median, on equal terms, and less accurate by the
  mean at small displacements.** With the reference displaced, its
  median re-triangulated residual is level with congealing's from the stored
  keypoints (0.264 against 0.261 px on seoul_bull, 0.157 against 0.160 on
  kerry_park) and at 0.5 px on seoul_bull (0.272 against 0.271), and lower at
  every other displacement: on seoul_bull 0.263, 0.292 and 0.412 against
  0.279, 0.316 and 0.427 at 1, 2 and 3 px; on kerry_park 0.159, 0.160, 0.254
  and 0.712 against 0.164, 0.177, 0.290 and 0.751 at 0.5 to 3 px. Its means
  are higher than congealing's from 0 to 0.5 px on both captures and at 1 px
  on seoul_bull (0.436 against 0.357 px on seoul_bull from the stored
  keypoints, 0.369 and 0.373 against 0.349 and 0.355 at 0.5 and 1 px), and
  lower from 2 px. The higher means do not come from a few tracks that fail
  to re-triangulate: at 0.5 and 1 px on seoul_bull no exhaustive track is
  over 5 px; with every track whose mean residual is over 2 px left out, the
  exhaustive search's mean is still higher (0.343 against 0.323 px on
  seoul_bull at 0 px, 0.212 against 0.200 and 0.225 against 0.209 on
  kerry_park at 0 and 0.5 px); and from 0 to 1 px it has more tracks whose
  mean residual is over 1 px, 7 to 12 per cell against 4 to 9. Its residuals
  have a heavier tail, spread over more tracks than congealing's.
- **The "+"-descent is level with the exhaustive search by the median near
  the truth, slightly better by the mean and the share of views on a side
  peak, and worse farther off.** From 0 to 1 px its median is within 0.007 px
  of the exhaustive search's; its mean is lower (0.340 and 0.346 against 0.369
  and 0.373 px on seoul_bull at 0.5 and 1 px, 0.202 and 0.204 against 0.232
  and 0.236 on kerry_park at 0 and 0.5 px) except on kerry_park at 1 px; and
  fewer of its views are more than 1.5 px off (from the stored keypoints 5.9%
  against 6.7% on seoul_bull and 0.9% against 1.4% on kerry_park, and with the
  reference stored at 1 px 5.8% and 1.0% against 6.6% and 1.4%). From 2 px its
  median is higher (0.331 and 0.633 on seoul_bull, 0.437 and 1.469 on
  kerry_park), where it stops at a nearer, lower peak; at 2 and 3 px it is
  above congealing too. On kerry_park at 1 and 2 px it also has tracks that
  fail to re-triangulate (means 2.469 and 4.387 px).
- **With the reference stored the alignment places views far closer**, from
  0.5 px up: at 3 px 7.4% (seoul_bull) and 10.7% (kerry_park) of the
  exhaustive search's views are more than 1.5 px off against congealing's
  60.3% and 61.0%. That advantage comes from the reference's correct keypoint,
  which in real use carries error too.
- **The raw errors with the reference displaced** favour congealing from
  0.5 px, by about the displacement, which is the reference's offset that every
  aligned view shares and that moves the point rather than adding reprojection
  error.
- **Why the alignment replaced congealing.** It scores and places each view
  against the same render the point stores, the one the bench, the scores and
  the stored bitmap read, so a view's keypoint and its score mean the same
  thing everywhere; it has no rounds, no weights and no basis cap; and with
  the exhaustive search it is as accurate or more by the median residual. It
  is not faster everywhere: per track it takes 0.40 to 0.90 times congealing's
  time on the ground truths and 0.84 to 0.96 times on DnDTabletop tracks of 20
  to 199 views, but 1.18 times on DnDTabletop tracks of 200 views or more and
  1.12 to 2.84 times on DinoLedge tracks of 20 to 199 views.
- **There is no coarse level.** Before the 3×3 fit, with the reference stored,
  a half-resolution coarse level lowered the error from starting keypoints
  3 px off, and 2 px off on kerry_park, and raised it from 0 to 1 px on both
  captures and from 2 px on seoul_bull (mean raw error 0.63 against 0.48 px
  from the stored keypoints on seoul_bull, 0.68 against 0.54 px at 2 px; 0.37
  against 0.29 px from the stored keypoints on kerry_park), at 1.5 to 1.7
  times the descent's time per track.
- **The exhaustive search is the default.** It finds the highest peak in the
  window, which on these captures is the true one more often than the peak
  nearest a start 2 px or more off is. Near the truth it is level with the
  descent by the median and slightly worse by the mean and the share of views
  on a side peak; that is the cost of the default. The
  "+"-descent remains as `SearchStrategy::PlusDescent` for a caller that wants
  the 11 to 18% of the time per track it saves.

### The agreement gates on the plain score

The gates `min_absolute_zncc` and `min_relative_zncc` read the plain ZNCC
against the reference render. Their bars were measured with the reference
stored, one table per starting displacement rather than the runs pooled, over
the views kept by both methods (the reference left out): a **good** view is
within 1 px of the ground-truth keypoint, a **bad** one more than 1.5 px from
it, each by the error of the method whose score is gated. Each cell is the
share of good / bad views a bar drops, for the plain score of the exhaustive
search and for congealing's leave-one-out score it replaced:

| dataset, disp | views | good / bad, plain | good / bad, leave-one-out | absolute 0.5, plain | absolute 0.5, leave-one-out | relative 0.7, plain | relative 0.7, leave-one-out |
|---|---|---|---|---|---|---|---|
| seoul_bull, 0 px | 872 | 781 / 67 | 781 / 49 | 0.6% / 13.4% | 0.4% / 4.1% | 0.6% / 7.5% | 0.5% / 2.0% |
| seoul_bull, 0.5 px | 878 | 786 / 67 | 776 / 49 | 0.6% / 9.0% | 0.4% / 2.0% | 0.5% / 4.5% | 0.5% / 2.0% |
| seoul_bull, 1 px | 871 | 775 / 66 | 666 / 70 | 0.6% / 7.6% | 0.5% / 0.0% | 0.5% / 3.0% | 0.6% / 0.0% |
| kerry_park, 0 px | 2877 | 2810 / 45 | 2841 / 18 | 0.4% / 15.6% | 0.5% / 0.0% | 0.3% / 13.3% | 0.6% / 5.6% |
| kerry_park, 0.5 px | 2863 | 2789 / 46 | 2784 / 26 | 0.4% / 15.2% | 0.4% / 7.7% | 0.3% / 17.4% | 0.6% / 7.7% |
| kerry_park, 1 px | 2843 | 2759 / 46 | 2430 / 121 | 0.4% / 15.2% | 0.5% / 1.7% | 0.3% / 10.9% | 0.5% / 2.5% |

The absolute bars at 0.4 and 0.6 and the relative bars at 0.6 and 0.8 are in
the README. The bad views are few (18 to 121 per table), so one view moves a
share by up to 6 points. At the same bars the plain score drops about as many
good views as the leave-one-out score did and, in every table, more of the
bad ones, by up to several times. The defaults stay at 0.5 and 0.7, where each
bar drops at most 0.6% of good views. The median plain score of a good view is
lower than its leave-one-out score was, since a single sharp render correlates
less with a view than the mean of the other views did.

### A spot check on four other reconstructions

400 tracks of each, from their stored keypoints with the default gates, the
reference given, before the 3×3 sub-pixel fit and while the "+"-descent was the
default search. The score column is each
method's own: congealing's leave-one-out ZNCC and the "+"-descent's plain
ZNCC against the reference, so the two are not the same measure:

| reconstruction | method | views kept | tracks with 2+ views | median score | ms per track |
|---|---|---|---|---|---|
| dino_dog_toy, 13 images | congealing | 90.3% | 383 | 0.950 | 2.41 |
| dino_dog_toy, 13 images | "+"-descent | 91.7% | 383 | 0.943 | 0.95 |
| dino_dog_toy, 85 images | congealing | 66.2% | 311 | 0.907 | 4.90 |
| dino_dog_toy, 85 images | "+"-descent | 69.8% | 310 | 0.856 | 1.81 |
| MossyRailing | congealing | 94.3% | 398 | 0.921 | 3.82 |
| MossyRailing | "+"-descent | 96.2% | 398 | 0.886 | 1.31 |
| ChristmasTreeWithPresents | congealing | 94.0% | 398 | 0.951 | 4.48 |
| ChristmasTreeWithPresents | "+"-descent | 95.0% | 397 | 0.935 | 1.97 |

The alignment keeps 1 to 4 percentage points more of the views and the same
number of tracks with two or more views to within one, and is 2.3 to 2.9 times
faster.

## Implementation details

**The render path is shared.** Each view's context tile is rendered by
`WarpMap::from_patch` and the sampler's remap, over the patch cloud's frame
([patch-cloud.md](patch-cloud.md)), which is camera-model agnostic through
`ray_to_pixel`. The z-normalization and the fused mean's robust consensus are
shared with [normal refinement](patch-normal-refinement.md) (`pub(super)`), not
duplicated, and the reference-view rule's render is
`stored_bitmap::render_reference`, the one every writer of the stored bitmap
calls.

**Mixed channel counts.** A view can be narrower than the template (a grayscale
frame beside a colour reference). Its search reads the template's leading kept
channels, which is all a narrower tile can line up with.

**Widening is quadratic, so the big buffers are asked for fallibly.** The context
tile is `R + 2 · margin` on a side and the shift grids are `(2 · margin + 1)²`
cells, so a caller that widens `search` asks for memory that grows as the
square of that radius, and an allocation the global allocator cannot make
**aborts the process**, which in a window takes everything unsaved with it.
`view_cache_bytes(params, channels)` is what one view's tile and the render
scratch behind it cost, so a caller can decide before it asks; the views are
searched one after another, so one tile is held at a time.
`try_localize_patch_keypoints` reserves the tile planes and the shift grids
through `try_reserve_exact`, and probes the render scratch at its own size
before a tile over 16 MB is built, reporting `LocalizeError::OutOfMemory` with
the size it asked for.

**The fallible call is also the cancellable one.** `try_localize_patch_keypoints`
polls its `Progress` after resolving the reference and before each view's
render, which is where a widened search spends its time, and reports
`LocalizeError::Cancelled`. Nothing else about the two paths differs: with an
uncancelled `Progress::none()` and buffers that fit, they compute the same
numbers.

**Profiling.** `SFMTOOL_PROFILE=1` times the reference resolution, the renders,
the searches and their sub-phases, and counts the views each gate drops
([keypoint_localize/prof.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/prof.rs)).
