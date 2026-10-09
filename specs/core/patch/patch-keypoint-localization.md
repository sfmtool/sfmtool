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
     (`PlusDescent` by default, a local "+"-descent from the zero shift;
     `Exhaustive` scores every cell), then refine it to sub-pixel by the
     vertex of a quadratic fitted to the 3×3 cells around the peak (see
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
| `search_strategy` | `PlusDescent` | how each view's shift grid is traversed: `PlusDescent` or `Exhaustive`; see [How the alignment was measured](#how-the-alignment-was-measured) |

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
it by that leave-one-out ZNCC. The two were compared on the seoul_bull (259
tracks) and kerry_park (380 tracks) ground truths, which are metric
reconstructions with known poses. Each view's starting keypoint was displaced
from its ground-truth projection by 0, 0.5, 1, 2 or 3 px in a random direction,
the reference kept at its stored keypoint; the agreement bars and
`max_shift_px` were off and the member gate at its default. Four variants of the
alignment were run: the "+"-descent and the exhaustive search, each alone and
each after an exhaustive search at half resolution (a **coarse level**, from
which the full-resolution search starts).

Two errors were read per kept view:

- the **raw error**, its distance from the ground-truth projection;
- the **re-triangulated residual**, its reprojection residual after each track
  is re-triangulated from its aligned keypoints at the ground-truth poses. This
  leaves out the offset the reference shares with every view aligned to it,
  which moves the point rather than adding reprojection error.

A view more than 1.5 px from the ground-truth projection is counted as **locked
to a side peak**.

**Mean raw error (px) / side-peak share**, by starting displacement:

| dataset | disp px | congealing | "+"-descent | exhaustive | coarse, then descent | coarse, then exhaustive |
|---|---|---|---|---|---|---|
| seoul_bull | 0 | 0.53 / 5.7% | **0.48 / 6.7%** | 0.59 / 7.9% | 0.63 / 7.8% | 0.62 / 7.6% |
| seoul_bull | 0.5 | 0.58 / 5.6% | **0.50 / 6.7%** | 0.60 / 7.9% | 0.65 / 7.8% | 0.64 / 7.6% |
| seoul_bull | 1 | 0.78 / 8.0% | **0.51 / 6.5%** | 0.60 / 7.5% | 0.66 / 7.6% | 0.65 / 7.6% |
| seoul_bull | 2 | 1.29 / 30.0% | **0.54 / 8.2%** | 0.63 / 8.2% | 0.68 / 8.9% | 0.67 / 8.4% |
| seoul_bull | 3 | 1.79 / 60.1% | **0.90 / 14.8%** | 0.69 / 9.2% | 0.69 / 8.9% | 0.69 / 9.1% |
| kerry_park | 0 | 0.29 / 0.6% | **0.29 / 1.0%** | 0.32 / 1.6% | 0.37 / 1.9% | 0.36 / 1.8% |
| kerry_park | 0.5 | 0.38 / 0.9% | **0.31 / 1.0%** | 0.34 / 1.7% | 0.40 / 2.5% | 0.37 / 2.1% |
| kerry_park | 1 | 0.64 / 4.2% | **0.31 / 1.1%** | 0.35 / 1.7% | 0.41 / 2.6% | 0.40 / 2.3% |
| kerry_park | 2 | 1.26 / 38.2% | **0.54 / 9.1%** | 0.43 / 4.2% | 0.47 / 5.1% | 0.44 / 4.6% |
| kerry_park | 3 | 1.86 / 60.3% | **1.22 / 27.8%** | 0.77 / 12.1% | 0.69 / 10.6% | 0.65 / 9.3% |

**Median re-triangulated residual (px):**

| dataset | disp px | congealing | "+"-descent | exhaustive | coarse, then descent | coarse, then exhaustive |
|---|---|---|---|---|---|---|
| seoul_bull | 0 | 0.27 | **0.26** | 0.26 | 0.26 | 0.26 |
| seoul_bull | 1 | 0.29 | **0.28** | 0.27 | 0.27 | 0.27 |
| seoul_bull | 3 | 0.36 | **0.34** | 0.29 | 0.28 | 0.28 |
| kerry_park | 0 | 0.17 | **0.17** | 0.17 | 0.17 | 0.17 |
| kerry_park | 1 | 0.19 | **0.18** | 0.18 | 0.17 | 0.17 |
| kerry_park | 3 | 0.55 | **0.86** | 0.37 | 0.24 | 0.22 |

**Time per track** with the reference given, from the stored keypoints, by
track length (ms):

| dataset | method | 3-5 views | 6-10 | 11-20 | 21+ |
|---|---|---|---|---|---|
| seoul_bull | congealing | 1.21 | 2.19 | 3.74 | - |
| seoul_bull | "+"-descent | **0.42** | **0.77** | **1.44** | - |
| seoul_bull | exhaustive | 0.52 | 0.97 | 1.76 | - |
| seoul_bull | coarse, then descent | 0.66 | 1.26 | 2.20 | - |
| kerry_park | congealing | 1.59 | 3.03 | 4.29 | 5.19 |
| kerry_park | "+"-descent | **0.61** | **1.29** | **2.27** | **3.42** |
| kerry_park | exhaustive | 0.73 | 1.52 | 2.76 | 4.25 |
| kerry_park | coarse, then descent | 1.04 | 2.13 | 3.72 | 5.86 |

Where the point has no reference and the reference-view rule picks one, the
pick adds a mean of 1.4 ms per track on seoul_bull and 3.9 ms on kerry_park.

What it decided:

- **Alignment to the reference replaced congealing.** It is 1.5 to 2.9 times
  faster, and from starting keypoints 1 px or more off it places views far
  closer: congealing's share of views locked to a side peak rises to 30-60% at 2
  and 3 px, against 8-28% for the "+"-descent. From the stored keypoints the two
  are close (mean raw error 0.48 against 0.53 px on seoul_bull, 0.29 against
  0.29 on kerry_park), with congealing leaving slightly fewer views past 1.5 px
  (5.7% against 6.7%, 0.6% against 1.0%).
- **There is no coarse level.** It lowered the error only from starting
  keypoints 2 to 3 px off, and raised it from 0 to 1 px: from the stored keypoints its mean raw error is
  0.63 against 0.48 px on seoul_bull and 0.37 against 0.29 px on kerry_park. Its
  time per track is 1.6 to 1.7 times the descent's.
- **The "+"-descent is the default search.** From starting keypoints within
  1 px of the truth it places views closer than the exhaustive search, because
  a side peak of a repeated texture can score higher than the true peak, and
  the descent stops at the peak nearest the start, which is the evidence for
  which peak is meant. From 2 to 3 px off the exhaustive search does better, so
  a caller whose starting keypoints may be that far off can choose
  `SearchStrategy::Exhaustive`.

### The agreement gates on the plain score

The gates `min_absolute_zncc` and `min_relative_zncc` read the plain ZNCC
against the reference render. Their bars were measured on the views of the 0,
0.5 and 1 px runs: a **good** view is within 1 px of the ground-truth
projection, a **bad** one more than 1.5 px from it. The table gives the share of
each a bar drops, for the plain score against the reference and for congealing's
leave-one-out score it replaced:

| bar | seoul_bull plain: good / bad | seoul_bull leave-one-out | kerry_park plain | kerry_park leave-one-out |
|---|---|---|---|---|
| absolute 0.4 | 0.5% / 10.9% | 0.0% / 0.6% | 0.3% / 4.4% | 0.3% / 1.2% |
| **absolute 0.5** | **0.9% / 13.1%** | 0.4% / 1.8% | **0.6% / 7.7%** | 0.5% / 2.4% |
| absolute 0.6 | 4.2% / 18.3% | 1.8% / 3.0% | 1.5% / 16.5% | 0.8% / 7.3% |
| relative 0.6 | 0.3% / 0.6% | 0.1% / 0.6% | 0.3% / 3.3% | 0.3% / 1.2% |
| **relative 0.7** | **0.7% / 5.7%** | 0.5% / 1.2% | **0.5% / 7.7%** | 0.5% / 3.6% |
| relative 0.8 | 2.7% / 9.1% | 2.0% / 1.8% | 1.8% / 8.8% | 1.4% / 7.9% |

(2632 views on seoul_bull, 2361 good and 175 bad; 8660 on kerry_park, 8465
good and 91 bad.) The plain score separates the two better than the
leave-one-out score did: at the same bars it drops slightly more good views and
2 to 7 times as many bad ones. The defaults stay at 0.5 and 0.7, where each bar
drops under 1% of good views. The median plain score of a good view is lower
than its leave-one-out score was (0.85 and 0.88, against 0.90 and 0.93), since
a single sharp render correlates less with a view than the mean of the other
views did.

### A spot check on four other reconstructions

400 tracks of each, from their stored keypoints with the default gates, the
reference given:

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
