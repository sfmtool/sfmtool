# View-region pyramid

**Status:** Draft

The required companion of
[out-of-core-photograph-pyramid.md](out-of-core-photograph-pyramid.md), which
links here. Decided by the user: every kernel reads every photograph,
whatever its size, through view-region pyramids, so its memory grows with
the patches it renders rather than with the photographs (open question 1).
It is kept as a draft of its own because what it specifies, the samplers'
regions, is separate from the files and caches the main draft specifies; its
steps 1 and 2 are step 7 of the main draft, and its steps 3 to 6 are the main
draft's steps 9 to 12. The main draft's step 8, between them, limits the
cluster members' elongation (§ "The members' elongation limit"). The user
has decided every open question (§ "Open questions"). It amends
[core/camera/photograph-cache.md](../core/camera/photograph-cache.md) and the
samplers of [core/camera/image-warping.md](../core/camera/image-warping.md).

When a patch kernel renders a view's tile, it reads a few hundred to a few
thousand pixels of one or two levels of the photograph's pyramid, near where
the patch projects. Today it is handed the whole pyramid of every photograph
it reads: 762 MiB for a 200-megapixel photograph, 12.6 GiB for a 17-image
track of them, and a `get_many` hands back every photograph's pyramid at once,
which the caller holds while its kernel runs. A view-region pyramid holds,
for one render of one patch in one photograph, only the rectangle of each level
that the render can reach, assembled from photograph tiles, with each level's
offset in the full level. The samplers read it at the same coordinates as a
whole pyramid and get the same values, so a kernel's results do not change, and
the memory a kernel touches grows with the number of patches it renders instead
of with the size of the photographs.

The name: in this tree *match* means a feature correspondence (`.matches`
files, `sfm match`), so a name such as *per-match local pyramid* would point
at the wrong thing; the unit here is one view of one patch, the unit the
glossary's "Rendering a view's tile" section names, so the term is
**view-region pyramid**: the pyramid of the region one view's render reads.

## What it holds

```rust
/// One level of a pyramid that may hold only a rectangle of the full level.
pub struct PyramidLevel {
    /// The pixels held: the whole level, or a rectangle of it.
    pub image: ImageU8,
    /// The held rectangle's top-left pixel in the full level; `[0, 0]` for a
    /// whole level.
    pub origin: [u32; 2],
    /// The full level's size, which clamping at the image's edge uses.
    pub full_size: [u32; 2],
}
```

`ImageU8Pyramid` keeps its interface for whole pyramids and gains
`from_regions(levels: Vec<Option<PyramidLevel>>)`, where `None` is a level the
render does not read, and `level_origin(l)` beside `level(l)`. A whole pyramid
has every origin at zero, so every existing caller is unchanged.

```rust
/// How far a held region extends past the region a render requires, as a
/// share of the required region's width (left and right) and height (above
/// and below), each rounded up to whole pixels, before clipping to the level
/// (§ "Where the pixels come from", "Reuse across renders").
pub const REGION_REUSE_MARGIN: f64 = 0.25; // in camera/sampler.rs
```

It is one of the three values that size what a kernel holds, beside the
kernel call's region pool and the tile path's window of 16 pinned tiles
(§ "Bounding a render's memory"). Like them, it changes memory and speed,
never a result.

**Why the values are equal.** Every sampler reads level `l` at the level-0
position divided by `2^l` with `bilinear_taps` in
[remap.rs](../../crates/sfmtool-core/src/camera/remap.rs), which takes the
pixel `floor(x − 0.5)` and the one after it on each axis (pixel centres at
half-integers) and clamps both to the level's bounds (`bilinear_geometry`).
With a region, the sampler subtracts the origin from the level coordinate and
reads the held image. The subtraction is exact in `f32`: the coordinate is a
multiple of its own unit in the last place, which is at most 1 for any
coordinate below 2^24, and the origin is an integer, so the difference is
representable and the fractions `fx`, `fy` are the same bits as the whole
level's. The two agree for every sample when:

1. the region is clipped to the full level, so a region edge that is an image
   edge clamps exactly where the whole level clamps; and
2. every sample's two pixels on each axis, after the whole level's clamp, are
   inside the region, so a region edge that is not an image edge is never
   reached.

The region of a level is therefore the bounding box, over every sample the
render takes at that level, of `floor(x − 0.5)` to `floor(x − 0.5) + 1` and the
same for `y`, clamped to the level. A sample outside the image clamps to the
edge pixel in both, and the clamped pixel is inside the box because the box is
clamped the same way.

## Which samples a render takes

The region is computed from the warp map before the render, by the same rules
the render applies, so it is exact rather than estimated:

- **`Bilinear`**: level 0 at each valid texel's `(sx, sy)`.
- **`BilinearMip`**: level `mip_level_for_sigma(σ_major)` at `(sx, sy) / 2^l`
  for each valid texel.
- **`Anisotropic`**: levels `level_lo` and `level_hi` of `aniso_footprint`,
  with the `n` samples along the major axis at offsets `t·σ_major` for
  `t = (i + 0.5)/n − 0.5`, so within `σ_major / 2` of the texel's position on
  either side; a texel with `σ_major ≤ 1` takes one level-0 sample.
- **Value and gradient renders** (`remap_*_with_grad_into`) read the same
  pixels as the value renders.
- **The view selection's affine fast path** and **the cluster refinement**
  (`affine_core_map` in
  [view_selection.rs](../../crates/sfmtool-core/src/patch/view_selection.rs),
  `level_for_map` in
  [cluster_refine.rs](../../crates/sfmtool-core/src/patch/cluster_refine.rs))
  take one level per map and sample the map's grid, so the region is the
  map's image of the grid at that level. Under `Anisotropic` the affine fast
  path samples the support with `aniso_sample` instead
  (`sample_support_affine_aniso`), whose region is the `Anisotropic` rule's
  over the support's positions.
- **`clipped_share`** in
  [reference_view/tile.rs](../../crates/sfmtool-core/src/patch/reference_view/tile.rs)
  reads every level-0 pixel inside the tile's outline. It becomes one of the
  render entry points (§ "Where the pixels come from") and takes no region
  pyramid: it walks the level-0 tiles the outline's bounding box covers,
  one at a time, applying the same even-odd rule to the pixel centres of each and summing the counts, so it holds one tile at a time
  and gives the same share. A footprint of `F` level-0 pixels covers up to
  `(ceil(F / 512) + 1)^2` tiles; this is the one read whose cost grows with the
  footprint rather than with the patch resolution, as it does today.
- **The grey reads of *Find Nearby Tracks*** (`GreyImages` in
  [bench/nearby/grey.rs](../../crates/sfmtool-core/src/bench/nearby/grey.rs))
  blur level 0 with a nine-tap Gaussian, reflecting at the image's edges: a
  read asks `read_source_rect` for the bounding box of its samples plus 4 px
  for the blur and 1 px for the bilinear tap, clipped to the image, and the
  blur reflects only at the image's edges, never at an inner edge of the
  rectangle.

A function per sampler, `required_regions(map, sampler, layout) ->
LevelRegions`, walks the map once, as the render does; the region of a view's
tile costs about what the render's own address arithmetic costs, and is a
fraction of the sampling. It lives beside the samplers in
[sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs) so a change to a
sampler's footprint and to its region is one change. A read at given
positions rather than through a warp map has its regions from
`LevelRegions::for_samples(level, positions, border)`: the bounding box of
each position's tap pair widened by `border` pixels on every side (0 for a
bilinear read, 1 for a read of the 4 x 4 pixels about it), clamped to the
level. Both are called by the render entry points and the panels, never by
a kernel (§ "Where the pixels come from").

## Readers that sample through their own code

Every function in `crates/` that reads a photograph's pyramid pixels is one
of three kinds. Most render through `render_tile`, `render_tiles`,
`render_tile_with_grad_into` or `render_view_tile`, and the rules above give
their regions from the map they render. A few sample through code of their
own, at a level and positions they choose; each has a region rule of its own
below, worked out from the quantities it already computes before it samples,
which the render entry point it reads through applies (§ "Where the pixels
come from"). The panels' crops are not kernels and apply theirs themselves.
The rest read a level only for its size or its channels, which is a layout
read (§ "Reads of the size and channels only"). The steps are this draft's,
with the main draft's in brackets.

**Through the general rules.** These need nothing more than the rules above,
the region pool and the tile path:

- `render_tile` with the map's own sampler: Track View's patch tile and its
  track-stage hover context (`track_view/body/patch.rs`
  `patch_frame_color_image`, a 64 x 64 tile and a 192 x 192 context over
  three times the patch frame, both at the tile's sampler), step 3 [9]; the
  bench's tiles, the keypoint localizer's context tiles of side
  `R + 2 · margin` (`keypoint_localize.rs`), the sub-pixel refiner's value
  and gradient renders (`keypoint_subpixel/kernels/render.rs`), the stored
  bitmap's and the reference picks' renders (`render_view_tile`), the
  observation readings, *Add image to tracks*, and normal refinement's
  renders (`patch/normal_refine/znorm.rs`, `render_tiles`), step 4 [10];
- the normal kernels that render through `render_tile` (below);
- `clipped_share`, the cluster grid samplers `sample_grid_window` and
  `build_template` (Track View's cluster-stage tile and its hover context
  through `sample_member_grid`, at 3 times the radius and resolution for the
  context; the bench's tile self-similarity; the piecewise refinement's
  working patch of side `R + 2 · margin`), the affine fast path, and
  `GreyImages`, as their rules above say.

Track View's *Zoom* cell reads only the warp's Jacobian, and its *ZNCC*
hover (`track_view/body/zncc_hover.rs`) draws the stored patch bitmap and the
row's tile already rendered; neither reads a photograph.

**Normal estimation.** Every normal kernel reads photographs only through
the renders below; what each does after a render reads the render, never the
photograph.

| Kernel | Photograph reads | Level and region | Typical region | Step |
|--------|------------------|------------------|----------------|------|
| `pairwise_flow_normal` (`bench/normal/pairwise_flow/crops.rs` `render_crop`) | one `render_tile` per view per round, of the crop: a square of side `2 · half_width` samples on the round's plane, at least 48 px of the finest view on each side of the point and at least 3 patch half-extents, at most `max_crop_samples` (512 by default, 4096 at most) | the general rule for the view's sampler (`SamplerChoice::per_view`) over the crop's map | the finest view reads about `side + 2` px a side at level 0: 98 px (29 KB) for a patch of 24 px, 194 px (113 KB) for one of 64 px; any facing view at most about `√2 · side` px at its level, 725 px (1.6 MB) at the 512 cap; copied within the pool. A view up to the 80° limit reads a stretched footprint, and a crop of 4096 samples up to about 100 MB; these take the tile path | 4 [10] |
| | after the renders: the crop's grey image (`matcher_gray_from_rgb`), its Gaussian pyramid (`CropPyramid`, the flow's `ImagePyramid`), the DIS flow between crops and the noise fields of `texture.rs`, which are synthetic | no photograph read | — | — |
| `grid_plane_normal` (`bench/normal/grid_plane.rs` `render`) | one `render_tile` per view per pass of the working grid, `r` samples a side (about one sample per photograph pixel, clamped to `R ..= 2R`), widened by `margin = ceil(cell_shift_bound_px)`, 2 by default, so `r + 4` a side | the general rule for the view's sampler | up to 52 samples a side at `R = 24` and 132 at `R = 64`: about 76 px (17 KB) and 190 px (108 KB) at the level read; copied | 4 [10] |
| `grown_plane_normal` (`bench/normal/grown_plane/render.rs` `ViewImage`) | its own sampler: one level per view, `bilinear_mip_level(σ_major)` of the anchored patch frame's Jacobian at the working grid's centre, clamped to the levels, chosen once; each render (`ViewImage::sample`) reads the value at each grid sample's projected pixel plus the view's image shift, divided by `2^l`, with `sample_bilinear_u8`, and `ViewImage::gradients` reads the Sobel gradient there from the 4 x 4 pixels about it (`sobel_gradient`) | per render, the kernel hands its positions `(pixels + shift) / 2^l` at level `l` to `read_source_samples` with a border of 1, which computes the region with `LevelRegions::for_samples` and returns each position's 4 x 4 pixels and fractions; the kernel blends the value from the middle 2 x 2 with `sample_bilinear_u8`'s expression and the Sobel gradient from the 4 x 4 as `sobel_gradient` does. The pixels are projected before the read, so the region is exact | the grid is `R ..= 2R` a side and the level keeps the major axis at most `√2` px a sample, so a frame near the starting one reads at most `(√2 · side + 4)` px a side: about 72 px (15 KB) at `R = 24` and 185 px (103 KB) at `R = 64`; the tilt the fit moves to can stretch it further, which the per-render region follows. Copied, with the reuse margin across the fit's iterations | 4 [10] |
| `finite_difference_normal` (`bench/normal.rs`) | the bench's `fit` for each piece and `evaluate_rendering_bitmap` at the turned normal (`turn_and_read`) | the general rules, through those kernels | as the bench's tiles | 4 [10] |
| `refine_normals` (`bench/refine_normals.rs`) | `pairwise_flow_normal` or `grown_plane_normal` per point, then `evaluate_rendering_bitmap` | as those kernels; the pool its caller made is passed to each point | as above | 4 [10]; *Refine all normals*' fetch and point order 5 [11] |
| normal refinement (`patch/normal_refine/znorm.rs`) | `render_tiles` over the views | the general rules | as the bench's tiles | 4 [10] |

On the tile path `read_source_samples` reads each window by address. A 4 x 4
window reaches one pixel further than a photograph tile's border, so it can
straddle up to four tiles: the entry point clamps each of its 16 pixel
addresses to the level as `sobel_gradient` does, which
can move an address several pixels when the position lies off the image, and
reads the pixel through the same tile lookup from the tile whose core holds
the clamped address. Every pixel is the one the whole-level read takes and
the arithmetic is unchanged, so the gradients are bit-identical.

Tests, in step 4: on a small synthetic reconstruction, each of the four
normal kernels and `refine_normals` gives the identical normal and report
with whole pyramids and with tiled sources of 32 px tiles, once with a pool
large enough for every region and once with a pool so small that every
render takes the tile path; the grown plane's value and Sobel reads through a
copied region and through tiles equal the whole-level reads at positions
whose 4 x 4 window straddles a tile corner and an image corner.

**Crops and their hover views.** `patch_pictures/crop.rs` `picture` cuts
Track View's crop cell and its hover (`crop::image`, `crop::context`, via
`track_view/body/crop.rs`) and the Track Table's crop column and its hover
(`track_table_panel.rs` `crop_picture`, `track_table_panel/pictures.rs`
`crop_context`). It samples a rectangle `region` of level 0, the bounding
box of the patch's outline plus `CROP_MARGIN_PX` (1 px), at `texels` per
side: one texel per pixel, or the longer side capped at `CROP_MAX_TEXELS`
(128), with a step of `size / texels`. The context is 3 times the region
(`CONTEXT_FACTOR`) at 3 times the texels, so at the same step, up to 384
texels. The level is `floor(log2(min step))`, clamped to the levels, one per
crop, and each texel inside the photograph is `sample_bilinear_u8` of that
level at its centre divided by `2^l`, cast to `f32`. Its region is
`LevelRegions::for_samples(l, centres, 0)` over the texel centres that fall
on the photograph, the same `f32` positions it samples. A texel step is
under twice the level's pixel, so the region is at most `2 · texels + 2` px
a side at its level: about 27 px (2.2 KB) for a patch of 24 px and 67 px
(13 KB) for one of 64 px at level 0, and at most 258 px (200 KB) for any
crop; a context is 79 px (19 KB), 199 px (120 KB) and at most 770 px
(1.8 MB). Both are copied from the panel's pool (§ "Bounding a render's
memory") and released when the picture is cut, since the cell and the hover
keep the picture, not the region; a context of a large patch while the pool
is busy takes the tile path, which reads each texel's tap pair as the other
samplers do. `picture` reads `level(0)` only for the photograph's size and
`num_levels()` for the clamp: layout reads. Step 3 [9]. Test: the crop and
the context of a synthetic photograph at steps of 1, 3 and 40 px per texel,
over outlines that cross tile edges and the image's edges, equal the
whole-pyramid pictures byte for byte, through a copied region and through
tiles of 32 px.

**The cluster refinement's evaluation tiles.** The cascade's objective
(`cluster_refine/kernels.rs` `eval_zncc`, `eval_zncc_parts`, through
`refine_member`, `read_member_at` and `member_zncc_at`) does not sample the
level directly. It reads a `LevelTile`, a planar `f32` copy of a rectangle of
the level `level_for_map` picks, with each channel's mean over the rectangle
subtracted, held per member in a `TileCache`. The rectangle is the affine
image of the `R x R` grid's corners at that level, floored and ceiled and
widened by 2 px (`grid_bbox`), joined with the rectangle already held and
widened by `TILE_MARGIN` (8 px) when it does not cover the request, clipped
to the level. The rectangle depends only on the maps and the level's size,
and the scores depend on it (the evaluation's coordinates are relative to its
corner and its values to its mean), so it must stay exactly as it is: the
rectangle is the algorithm's, not a choice of how to read. `LevelTile::build`
therefore hands that rectangle to `read_source_rect`, which reserves its
bytes from the kernel call's pool and returns its pixels, copied from a
region the pool holds or read out of the photograph's tiles; the kernel then
converts and centres them, row by row in the same order as today, so its
pixels, its mean and every score are the same. The kernel never touches a
tile or a region. At `R = 25` and a SIFT shape near a
similarity, the grid covers at most 50 px a side at its level, a rectangle of
about 70 px: 59 KB in `f32` RGB. Step 4 [10], through the bench's evaluation
of a cluster-stage track; the cluster-patches build moves its fetch in step 5
[11]. Test: `refine_cluster_patches` gives identical shapes, positions and
scores with whole pyramids and with tiled sources of 32 px tiles, including
members whose rectangle crosses tile edges and is grown during the cascade.
The evaluation has no tile path, since it must stay bit-identical and its
AVX2 form agrees with the scalar one only to about `1e-4`. So
`read_source_rect` reserves a `LevelTile`'s bytes from the kernel call's
region pool and copies the rectangle whether or not the reservation
succeeds; the members' elongation limit below bounds
each of its requests (open question 6).

### The members' elongation limit

The **elongation** of a member's affine shape, or of an evaluation's map, is
the ratio of its larger singular value to its smaller, `s_max / s_min`, from
the closed form `level_for_map` already uses (`s_max = q + r`,
`s_min = |q − r|`); the map is the shape scaled by the grid's step, so the
two have the same elongation. `MAX_PATCH_ELONGATION` in
`patch/cluster_refine/params.rs`, `MAX_ANISOTROPY as f64` (16), bounds it:

- **At validation**, beside the determinant test, a member whose detected
  shape's elongation is over the limit, or not finite, is not usable: its
  status is `not_evaluated`, and it takes no part in the member gate, the
  choice of reference or the cascade.
- **At every evaluation** (`refine_member`'s rotation starts, its cascade
  and its final parts read, `read_member_at`, `member_zncc_at`), a map whose
  elongation is over the limit scores as a support out of frame does,
  before any `LevelTile` is asked for: the simplex scores it worst (+1.0)
  and retreats. A rotation start turns the detected shape, which keeps its
  singular values, so no start of a member that passed validation is past
  the limit. The shape the refinement returns is one it evaluated, so it is
  within the limit too.

The value is the samplers' own: `MAX_ANISOTROPY` is the most samples the
anisotropic sampler takes along a footprint's major axis, so the renders
already resample a footprint stretched past 16 times with aliasing. A circle
on the surface foreshortened 16 times is seen at `acos(1/16)`, 86.4°, from
its normal, past the 80° beyond which the pairwise-flow normal kernel takes
no view (`max_view_angle_deg`). It is a named constant, not a setting, because the
memory bound below rests on it and a setting would let a caller remove the
bound; tying it to `MAX_ANISOTROPY` keeps one number for how stretched a
footprint the tree reads. A member past it keeps the status `not_evaluated`
(the `.matches` format's description of that status gains "or an affine
shape elongated past `MAX_PATCH_ELONGATION`") rather than a status of its
own, since a new status is a new `.matches` version and a reader does nothing
different with it. The reason is stated where the members are counted:
`ClusterRefineResult` counts the members refused for their elongation and
the evaluations scored out of frame for it, and `sfm cluster-patches`'
summary, the viewer's cluster-patches build and the bench's evaluation of a
cluster-stage track print how many of their `not_evaluated` members were
elongated past the limit.

**The bound on a `LevelTile`'s request.** With the level from `s_min`, the
map's singular values at its level are `s_min / 2^l` under 2 and
`s_max / 2^l` under `2 · MAX_PATCH_ELONGATION`, so their sum is under 34
(also at level 0, where `s_min` is under 2). The affine image of the `R x R`
grid spans `W` by `H` pixels with `W + H ≤ 2 (R − 1)(s_max + s_min) / 2^l`,
so `W · H ≤ ((R − 1)(s_max + s_min) / 2^l)²`; `grid_bbox`'s flooring,
ceiling and 2 px add at most 6 px to the width and to the height and
`TILE_MARGIN` 16 more, so one evaluation's request holds at most
`((R − 1) · 34 + 22)²` pixels, 12 bytes each for RGB:

| Resolution `R` | A member near a similarity | A member at the limit |
|----------------|----------------------------|-----------------------|
| 24 | about 68 px a side, 55 KB | 804² pixels, 7.8 MB |
| 25 (the default) | about 70 px a side, 59 KB | 838² pixels, 8.4 MB |
| 64 | about 148 px a side, 263 KB | 2164² pixels, 56 MB |

A level so coarse that the grid reaches past it is clipped to it, and the top
levels are a few pixels. The limit bounds each request, not what a member's
`TileCache` holds. The rectangle it holds at a level is the union of the
requests made there, widened by `TILE_MARGIN` at each rebuild, as today, so
it grows as far as the member's cascade moves the grid at that level, which
nothing caps; capping it would change the rectangle, and so the scores, of
the members whose cascades wander. A member that wanders far almost
certainly fails the shift gate (`max_shift_scales`, 2 of the member's scales
from its seed, a sixth of the grid's width at the default radius of 6), but
its tile has been built by then. A member holds one `LevelTile` per level
its cascade evaluates at, one or two since the scale moves little within a
cascade, and the rotation starts' own tiles are dropped before the cascade.
`refine_member` runs no parallel loop, so a thread holds one member's tiles
at a time. `read_source_rect` reserves each tile's bytes from the kernel
call's pool and copies the tile when the pool is full too. A kernel call that runs the cluster refinement therefore
builds past the pool, on each of its 33 rendering threads, tiles of at most
8.4 MB per request at the default resolution, 280 MB for one level on all of
them while no cascade wanders; a cascade that wanders holds more, by the
area its grid swept at that level. This is a bound on each request, not a
hard bound on what one member's cache holds. Members near a similarity, the
common case, stay within the pool. The phase note counts the tiles built past
the pool, the largest tile one member held and the most levels one member
touched.

The limit changes results whether or not tiles are used, so it is the
main draft's step 8, before the cluster refinement reads tiles in step 4
[10]. They change a member refused at validation, a member whose cascade or
piecewise stage evaluated a map past the limit, and,
through the cluster's own rules, the members those decide between: every
member of a cluster whose reference was a refused member (the reference is the
usable member of largest scale) or that a refusal leaves with fewer than two
usable members, and the other members of an image whose kept member changed,
since one member is kept per image. Every other member's result is unchanged.

### Reads of the size and channels only

These read `level(0)` for its width, height or channels, or `num_levels()`,
and no pixel: the reference picks' size check (`track_table_panel/picks.rs`),
the display patch bitmaps' (`patch/display_bitmaps.rs`), the crops above, the
bench's localizer budget (`bench/evaluate.rs`), the keypoint kernels'
channel counts (`keypoint_localize.rs`, `keypoint_subpixel.rs`), normal
refinement's (`normal_refine/znorm.rs`), the bindings' size checks
(`sfmtool-py/src/patches/views.rs`, `view_tile.rs`), and the grown plane's
in-frame test (`ViewImage::sample`). Each becomes a read of the
`TiledPhotograph`'s layout, `layout().level_size(l)`, `layout().levels()` and
`channels()`, which the layout read already knows, in the step that moves the
reader.

**Not photograph readers.** The optical flow's pyramids
(`features/optical_flow`) are of the grey images they are given: the
pairwise-flow crops, or arrays (`sfmtool-py/src/flow/warp.rs`). The
spherical tile source stacks (`spherical/per_tile_source_stack.rs`, through
`sfmtool-py/src/spherical/tile_source_stack.rs`) warp the arrays the caller
passes with `remap_bilinear`. Image Detail, the 3D view's background and the
thumbnails read whole tiles or levels for display, as the main draft's
§ "Display" says, not regions.

## Where the pixels come from

`PhotographCache::view_region_pyramid(path, &LevelRegions)` assembles
the region of each level from photograph tiles (`get_tile`, decoding any tile
not yet cached) and copies the rectangles out. The tiles come from the
mip level files, and level 0 from the original if it is region-readable or from its
`-mip0` otherwise, as the main draft describes; every one is a tile
decode, so no region of any level needs a whole decode. A region pyramid
therefore holds the same pixels as a whole pyramid of the stored levels would.
An in-core photograph is read the same way: its tiles are cut from a whole
decode of the original, made by a kernel's `get_many` in its read phase (its
misses in groups sized by the build budget), or by a panel's one decode task
for the photograph, and again by a render only for a tile evicted during its
kernel; only the tiles stay (the main draft's § "The caches in memory"), so a region pyramid of it
holds today's exact levels. Tiles cut from an array a caller already has
(`TiledPhotograph::from_image`) are read the same way too.

A kernel's `ProjectedImage` holds a `&TiledPhotograph`, for every photograph,
and a kernel reads pixels only through the **render entry points** in
[sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs). Each takes the
source and the region pool the kernel was given, works out the regions the
read needs, takes them from the pool as copied regions or reads them through
tiles, and returns what the kernel asked for. No kernel computes a region,
assembles a region pyramid or touches a photograph tile:

| Entry point | What the kernel passes | What it returns | Kernels |
|-------------|------------------------|-----------------|---------|
| `render_source_tile`, `render_source_tile_with_grad_into` | a warp map and a sampler | the rendered tile, and its gradients | every render through the general rules |
| `render_source_affine` | the view selection's affine map, its level, the support's grid pixels and the sampler | the support's raw samples, by today's `sample_support_affine` or `sample_support_affine_aniso`, whose arithmetic moves behind it unchanged | the view selection's affine fast path |
| `sample_source_grid` | a cluster grid's affine map and the grid points it reads (a window of the grid, or the support's pixels) with the edge rule | the raw samples, by today's `sample_grid_window` and `build_template` sampling, whose arithmetic moves behind it unchanged | the cluster refinement's grid samplers, `sample_member_grid` |
| `read_source_samples` | a level, positions on it, and a border of 0 or 1 | for each position, the `(2 + 2 · border)²` pixels from `floor(x − 0.5) − border` and `floor(y − 0.5) − border`, each address clamped to the level as `bilinear_geometry` and `sobel_gradient` clamp it, with `bilinear_geometry`'s fractions | the grown plane's value and Sobel reads |
| `read_source_rect` | a level and a rectangle of it that the kernel's own arithmetic defines | the rectangle's pixels, reserved from the pool and copied whether or not the reservation succeeds | the cluster refinement's `LevelTile`, the grey reads of *Find Nearby Tracks* |
| `clipped_share` | the outline | the share of level-0 pixel centres inside it, walking the level-0 tiles under the outline one at a time | the reference view's tile readings |

`required_regions` and `LevelRegions::for_samples` (§ "Which samples a render
takes") are the entry points' own: the entry points call them, and so do the
panels, which assemble regions from peeked tiles on the GUI thread
(§ "Bounding a render's memory"); no kernel calls them. The two rectangle
reads take a rectangle rather than positions because the rectangle is the
algorithm's: a `LevelTile`'s scores depend on its corner and its mean, and
the grey read's blur on how far past its samples it reads. The entry point
still decides how the rectangle is read. The unchanged `render_tile`
samples a region pyramid already assembled, for the panels and the tests.
The entry points return `Result<_, PhotographReadError>`, since a tiled read can find its
files changed, gone stale or unreadable, and the
kernels pass the error up with `?` through their loops, the parallel ones
included; the main draft's § "Errors from reading pixels" lists the
signatures that change and the step each changes in. The kernels' sampling
arithmetic does not change.

**Reuse across renders.** A kernel that optimizes a patch renders the same
view many times with a slightly moved frame (the bench's fit rounds, the
sub-pixel refinement, normal refinement). The region pyramid is assembled with
the **reuse margin**, `REGION_REUSE_MARGIN = 0.25` in
[sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs), beside
`required_regions` and `LevelRegions`: a level's held region is its required
region, `w` by `h` pixels, grown by `ceil(REGION_REUSE_MARGIN · w)` pixels on
the left and on the right and `ceil(REGION_REUSE_MARGIN · h)` pixels above
and below, then clipped to the level. Clipping changes no value, since
sampling clamps at the level's edge the same way (§ "What it holds"). A
later render whose required regions lie inside the held ones reuses it; one
that does not assembles a new one. Before clipping, the held region is 1.5
times as wide and as tall as the required one when its sides are multiples
of 4, so 2.25 times the area. Rounding up adds at most 1.5 px to each side,
so in general the area is at most `(1.5 + 1.5 / w)(1.5 + 1.5 / h)` times,
a bound of about 2.3 for square regions of about 60 px and 2.5 for ones of
about 20 px. For
regions of a few pixels it is well above 2.25, since a 1 px side grows to
3 px and a 2 px side to 4 px. Reuse does not change any value, only how often regions
are copied. A kernel keeps a region pyramid for one view of the patch it is
working on, and drops them all when it moves to the next patch (a fit's
rounds, a refinement's steps and an evaluation's views of one track share
them; the next track starts with none). So what it keeps is at most one region
pyramid per view of one patch, grown by `REGION_REUSE_MARGIN`, about 2.25
times the area of the region itself, and every copied region of the kernel call reserves from one pool
(§ "Bounding a render's memory").

## Bounding a render's memory

The region of a level is a bounding box, and the box of one render is not
bounded by the patch resolution `R`:

- **Anisotropic.** The level comes from `σ_minor`, but the `n` samples, capped
  at 16 (`MAX_ANISOTROPY` in
  [sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs), applied by
  `aniso_footprint` in
  [remap.rs](../../crates/sfmtool-core/src/camera/remap.rs)), still spread
  over `±σ_major / 2`. The cap bounds the number of samples, not their reach.
  A footprint stretched `a = σ_major / σ_minor` times along a diagonal has a
  bounding box of about `(R·a)² / 2` pixels at its level: `a = 100` at
  `R = 24` is about 8.7 MB for one render, 280 MB for 32 in flight, and a
  grazing view can approach a whole level 0.
- **One level per grid** (`level_for_map` in
  [cluster_refine.rs](../../crates/sfmtool-core/src/patch/cluster_refine.rs))
  picks its level from `s_min`, with the same effect for a stretched map, up
  to the cluster members' elongation limit of 16.
- **`Bilinear`**, which the kernels' sampler argument can select, reads level 0
  over the whole footprint whatever `R` is: a 24 x 24 tile of a patch that
  covers 2000 px reads a 2000 x 2000 box, 12 MB.

So a render samples one of two ways, and both give the same values:

- **Copied regions**, as above, when the render can reserve its regions,
  with the reuse margin, from the kernel call's **region pool**. The pool holds
  four tiles' worth of pixels (4 x 512 x 512 x 3 bytes, 3.1 MB) per thread
  of the rayon pool, 99 MB on 32 threads, for all the call's patches
  together, however many are in progress: a kernel's patches do not map one
  to a worker, since *Refine all normals* runs a parallel loop over points
  whose normal kernels run parallel loops over views
  (`bench/refine_normals.rs`, `bench/normal/pairwise_flow/crops.rs`), the
  cluster refinement nests parallel loops too, and a worker waiting in an
  inner loop takes other points from the outer one. A reservation is one
  atomic compare-and-add that either succeeds or fails at once; it never
  waits, so no worker can block on the pool, and nesting or work stealing
  cannot deadlock it. A region's bytes go back to the pool when the region is
  dropped (when its patch is done, or at once for a render that keeps
  nothing). A 151-view evaluation whose regions are 7 KB each keeps all of
  them; a kernel call whose views are stretched fills the pool and sends the rest
  down the tile path. Sampling a copied region is the fast path: one
  contiguous image per level, and the AVX2 anisotropic kernel runs on it
  unchanged. **One rule decides who owns a pool: the caller of a kernel
  makes one `RegionPool` for that kernel call and passes it in**, beside the
  views. The pool is not tied to how the sources were fetched: a caller that
  fetched its views with one `get_many`, or with one `get` per view as the
  Track Table's reference picks do (`pick_reference` in
  `track_table_panel/picks.rs` calls `get` once per observation and hands all
  the views to one `render_reference` call), makes one pool for the one call,
  so a 151-view point shares 99 MB rather than taking 3.1 MB per view.
  `RegionPool::for_kernel_call()` sizes it at 3.1 MB per rayon thread. A
  kernel call is a call into the kernels from code that is not itself a
  kernel: a viewer job, a panel, a command or a binding. A kernel that calls
  another kernel passes its own pool down and never makes one: `refine_normals`
  calls `pairwise_flow_normal` or `grown_plane_normal` for each point from
  `into_par_iter` and gives every point the one pool its caller made, and
  every other bulk kernel does the same for its per-point work, so the pool
  is not multiplied by the points in progress. Track
  View's renders on the GUI thread (tiles, crops and their hover views),
  which are not one kernel call but many small ones a frame, share one pool
  of 3.1 MB that the panel owns, and the Track Table's crop column and hover
  share another. No other
  caller builds one kernel call from several `get`s: the photograph cache's
  other `get` callers are the panels' decode requests
  (`state/photograph_requests.rs`), which fetch without rendering.
- **Tile-addressed sampling** otherwise, for a render whose reservation
  fails. It keeps nothing for reuse: its pinned
  tiles are released when it ends, and the next render of the same view
  computes its regions again. The render does not copy a region. It
  reads each sample's taps from the photograph tile that holds them: the tap
  pair `floor(x − 0.5)` and `floor(x − 0.5) + 1` (clamped to the level) always
  lies within one tile's core and its one-pixel border, so the tile is the one
  whose core holds the first tap, and the border pixel equals the neighbour's
  pixel, or the edge pixel at the image's edge, which is what clamping reads.
  The fractions are computed from the level coordinate as in the whole-level
  sampler, so every value is bit-identical. The render keeps at most 16 tiles
  pinned (12.7 MB), least recently used out, and fetches any other through
  the cache (`get_tile`, a lookup when the tile is cached and a decode of a few
  milliseconds when not, or, for an in-core photograph whose tiles were
  evicted after the kernel's read phase decoded it, one whole decode of the
  photograph, as the main draft's § "The caches in memory" says). For the samplers of `remap.rs` this path runs the
  scalar per-pixel code, which the AVX2 kernel already equals bit for bit
  (`remap/aniso_avx2.rs`), with a tile lookup only when consecutive samples
  change tile. The cluster refinement's grid samplers, `sample_grid_window`
  and `build_template` in
  [cluster_refine.rs](../../crates/sfmtool-core/src/patch/cluster_refine.rs),
  and the view selection's affine fast path in
  [view_selection.rs](../../crates/sfmtool-core/src/patch/view_selection.rs)
  are their own arithmetic (in `f64` for the cluster grids), not those
  samplers. Their sampling moves unchanged behind their entry points,
  `sample_source_grid` and `render_source_affine`, and each has a
  tile-addressed version there that reads its taps through the same tile
  lookup and keeps its own arithmetic, so its values are unchanged.

The bound has two parts. **Copied regions** are bounded by the kernel call's pool,
99 MB on 32 threads, whatever the number of patches in progress. **Pinned
tiles** are bounded per thread that renders: a thread runs one render at a
time, because a render never waits inside itself (its rows run serially on a
rayon worker, `parallelize_rows` in `camera/warp_map.rs` checking
`current_thread_index`, and its tile reads decode on the calling thread
without rayon), so work stealing cannot start a second render on a thread
whose first still holds tiles. On a thread that is not a rayon worker,
`parallelize_rows` hands a render of more than 2048 pixels
(`REMAP_PAR_MIN_PIXELS`) to rayon row by row, as it does Track View's
64 x 64 tiles on the GUI thread; the tile path does not, and keeps its rows
serial on every thread, so its window has one owner. A copied-region render
whose rows go to rayon pins no tiles while it samples. The threads that render are the rayon pool's
32, the job thread that calls the kernel, and in the viewer the other
queue's job thread and the GUI thread. One kernel call on the command line
holds at most 99 MB + 33 x 12.7 MB, about 520 MB, on top of the cache's budget;
the viewer, with at most two kernel calls at once (one per background queue)
and the two panels' renders, at most 2 x 99 MB + 2 x 3.1 MB + 35 x 12.7 MB, about
650 MB. A kernel call that runs the cluster refinement adds the evaluation
tiles it builds past the pool: at most 8.4 MB per request on each rendering
thread at the default resolution, 280 MB for one level on all of them, but a
member's tile is the union of its requests at a level and grows as far as its
cascade wanders, so this bounds each request, not what one member holds
(§ "The members' elongation limit"). The common case, footprints that are not stretched, stays at a few
kilobytes per view. No view is refused and no result changes:
which path a render takes depends on its boxes and on what its patch already keeps, and both read the same
pixels. `clipped_share` already walks tiles one at a time (above). The
pool's four tiles per thread and the window of 16 are decided (open
question 4).

**On the GUI thread** (Track View's tiles, crops and context, the Track
Table's crop column), the panel acts on the photograph's layout state as the
main draft's § "Readers on the GUI thread" says: while it is `Building` the
panel adds the photograph to the front of the *Build mipmaps* quiet task's
list once, draws a spinner with *Building mipmap…*, and requests no tiles;
once it is `Known` the panel computes the regions, peeks the tiles
(`peek_tile`), and draws the view's tile only when all of them are there,
requesting the missing ones: one tile task per missing tile out of core, and
one decode task per in-core photograph with all the tiles it is missing
(`try_decode_tiles`), which ends at once when another caller holds the
photograph's slot or the build budget refuses its reservation, the panel
peeking again on a later frame. The GUI thread never decodes and never waits,
no task a panel starts waits on a slot or the build budget, and no panel
makes a mipmap.

## Memory

- **Track View, a 17-image track of 200-megapixel photographs**: a 64 x 64
  tile whose patch covers `F` level-0 pixels reads level `≈ log2(F / 64)` over
  about 66 x 66 pixels, 13 KB, and a quarter of that at the next level for
  anisotropic blending; the photograph tiles it is cut from, one to four per
  level and one or two levels, are 0.8 to 6.3 MB in the tile cache. Seventeen
  views: 13 to 108 MB, against 12.6 GiB of whole pyramids. These are estimates from the sizes, not
  measurements.
- **Bulk kernels** (the display patch bitmaps at open, *Refine all normals*,
  the web export) touch every photograph at the levels their patches need.
  What they hold is the regions of the patches in progress, a few kilobytes
  each, grown by `REGION_REUSE_MARGIN` to about 2.25 times the area, all
  within the kernel call's region pool (99 MB on 32 threads), plus at most
  12.7 MB of pinned tiles per rendering thread (§ "Bounding a render's
  memory"), plus the tile cache, which its budget
  bounds whatever the number of photographs; the tiles they touch can be many
  more (level 2 of one 200-megapixel photograph is 48 tiles, 38 MB; level 0
  is 768 tiles, 607 MB), and the ones past the budget are decoded again when
  read again, at 2.3 to 6 ms a tile out of core, or by one whole decode of
  an in-core photograph (about 0.05 s at 3840 x 2160). Level 0 is always read
  by tiles, from the original, from `-mip0`, or cut from an in-core
  photograph's decode, which is dropped once cut, so no photograph's level 0
  stays whole.
  The order a kernel takes its renders in decides how often a tile is decoded
  again. **A bulk kernel takes its points in one order**: by the image of
  each point's reference observation, then by that observation's keypoint,
  its pixel row `floor(y)` and then its `x`, with ties broken by point
  index; the points with no reference observation come last, by point
  index. The renders of one photograph then follow each other, and so do
  those of points near each other in it and, through the surface, near each
  other in the views that also see them. The order uses only the
  reconstruction, so it does not depend on how photographs are cut into
  tiles. Each kernel's phase note reports the tiles decoded and reused.
- **`GreyImages`** today holds a blurred `f32` grey copy of each view's
  level 0, 800 MB for a 200-megapixel view. With regions it holds a few
  kilobytes per read.

## Implementation steps

Steps 1 and 2 are step 7 of the main draft, exercised on photographs cut
into tiles in memory. Steps 3 to 6 are the main draft's steps 9 to 12, after
its step 8, the members' elongation limit, each its own pull request, made
while the cache still hands out whole pyramids:
each moved reader wraps the pyramid it gets in a `TiledPhotograph`
(`from_pyramid`), so every reader works at every step. Once all have moved,
the main draft's step 13 makes the cache hand out tiled photographs only, and
there is no whole pyramid for any reader to fall back on.

1. **Regions in the pyramid and the samplers**: `PyramidLevel`,
   `ImageU8Pyramid::from_regions`, the origin subtraction in every sampler and
   gradient sampler (the AVX2 anisotropic kernel included), the
   tile-addressed path with its window of pinned tiles for the `remap.rs`
   samplers, the call's region pool that chooses between the paths, and the
   render entry points of § "Where the pixels come from", each returning
   `PhotographReadError`: `render_source_tile` and its gradient form;
   `render_source_affine` and `sample_source_grid`, with the sampling of
   the affine fast path, `sample_grid_window` and `build_template` moved
   behind them unchanged and a tile-addressed version of each;
   `read_source_samples`, with the tile-addressed read of a tap pair and of a
   4 x 4 window; `read_source_rect`. Behind them, `required_regions` for each
   sampler and `LevelRegions::for_samples`, which only the entry points and
   the panels call. Also `TiledPhotograph::layout` and `channels`. Tests: for
   random warps over a synthetic photograph, every sampler, every value and
   gradient render through a region pyramid built from `required_regions`, and
   through the tile-addressed path with a tile of 32 px, is bit-identical to
   the whole-pyramid render, including maps that run off every edge of the
   image, NaN texels, views shrunk past the top level, a footprint stretched
   100 times on a diagonal and a `Bilinear` render over a large footprint
   (which take the tile path); the cluster grid samplers and the affine fast
   path through tiles equal their whole-level results on stretched maps;
   the value blended and the Sobel gradient computed from
   `read_source_samples`' windows, through copied regions and through tiles,
   equal `sample_bilinear_u8` and `sobel_gradient` on the whole level;
   `read_source_rect` returns the rectangle's pixels through both paths; a
   kernel call whose regions exceed its pool (sized small for the test) never holds
   more than the pool, each render reserving its required regions grown by
   `REGION_REUSE_MARGIN`, the renders past it taking the tile path, including
   from nested parallel loops on many threads at once, and no reservation
   waits; the tile path pins no more than its window; a
   region one pixel smaller on an inner edge makes the comparison fail, so
   the test can fail.
2. **View-region pyramids assembled from a `TiledPhotograph`'s tiles**,
   with the reuse margin, and `clipped_share` as an entry point over tiles, on photographs cut
   into tiles in memory (`from_image` and `from_pyramid`); the cache's
   `view_region_pyramid` comes with the main draft's step 13. Tests: the assembled regions equal the whole pyramid's pixels;
   `clipped_share` over tiles equals it over the whole level 0, for outlines
   that cross tile edges and the image's edges; each held region is its
   required region grown by `ceil(REGION_REUSE_MARGIN · w)` and
   `ceil(REGION_REUSE_MARGIN · h)` px on each side and clipped to the level,
   for regions of 1, 2, 3 and 61 px a side and at every edge of the image;
   reuse when inside, reassembly when not.
3. **Track View and the Track Table** on region pyramids, peeking the
   cache's whole pyramids as today and reading them through `from_pyramid`;
   they peek tiles once the main draft's step 13 adds `peek_tile`. Track
   View's tiles and track-stage hover contexts call `render_tile`, and its
   cluster-stage tiles and contexts `sample_member_grid`, on the region
   pyramid they assembled, so no kernel signature changes here; the crops
   and their hover views of both panels cut their region as § "Readers that
   sample through their own code" says; each panel owns its pool. Tests: the
   drawn tiles and hover contexts of a synthetic track at both stages equal
   the whole-pyramid ones, and the crop test of that section.
4. **The bench's jobs and the reference picks**: `ViewSources` hands the
   kernels a tiled source, and every read goes through the render entry
   points, which work out its regions; each bench step and each reference pick makes one
   region pool for its kernel call and passes it in, the picks for every
   `render_reference` call over the views their per-view `get`s fetched.
   Every kernel these steps run moves here, with every kernel that calls
   one of them, since a kernel passes its caller's pool down: the normal
   kernels the bench's normal steps run and `refine_normals`, the grown
   plane's reads through `read_source_samples`, the cluster refinement that
   the bench's evaluation of a cluster-stage track runs, with its grid
   samplers through `sample_source_grid` and `LevelTile` through
   `read_source_rect`, the view selection's affine fast path through
   `render_source_affine`, and the
   stored bitmap's, `refit_points`' and the display bitmaps' kernels. A
   caller that step 5 moves keeps its fetch until then: it wraps the whole
   pyramids it holds with `from_pyramid` (or the cluster-patches build its
   decoded images with `from_image`), makes the region pool for its call,
   and reports the error as a failure of its step, though sources built in
   memory never return one.
   Tests: each step's result is identical with the two sources on a
   small-threshold synthetic reconstruction; a reference pick over many views
   whose regions exceed a small pool holds no more than that one pool; the
   normal kernels' and the cluster refinement's tests of § "Readers that
   sample through their own code". Each bench step's phase note counts the
   region pyramids its renders reused and the ones they assembled again, so
   the pull request reports how often the bench's fits reassemble at
   `REGION_REUSE_MARGIN`. It also counts the renders that took the tile
   path, so the pull request reports how often renders take it on the test
   data and on a real capture.
5. **Bulk kernels**: the display patch bitmaps, the conversion to embedded
   patches, the web export, *Refine all normals*, *Refit all points* and the
   cluster-patches build fetch tiled photographs, each taking its points in
   the order § "Memory" gives (by reference image, then the keypoint's row
   and column). Tests: identical results; a cache budget smaller than one
   photograph's level 0 still completes; with a budget smaller than the
   reconstruction's tiles but holding the tiles of every photograph one
   reference image's points are seen in, each in-core photograph is decoded
   at most once in the read phase and once more for each reference image
   whose points it sees, and the same points in shuffled order decode more,
   so the test can fail; with a small pool, the
   regions of all the points in progress together never exceed the one pool
   the caller made. Each bulk kernel's phase note counts the renders that
   took the tile path, and the pull request reports that count for each
   kernel on the test data and on a real capture.
6. **`GreyImages` by region.** Tests: the nearby-track search's results are
   identical.

## Open questions

The user has decided every question below; the section records the
decisions.

1. **In-core photographs.** *Decided by the user:* every photograph is read
   through tiles and view-region pyramids, whatever its size. An in-core
   photograph is decoded once, in a kernel's read phase or by a panel's one
   decode task for it, its pyramid cut into tiles, and only the tiles stay,
   under the cache's budget; a render that needs a tile evicted during its
   kernel decodes the photograph again, about 0.05 s at 3840 x 2160, which
   the budget and the bulk kernels' point order (question 3) keep rare.
2. **The reuse margin.** *Decided by the user:* a quarter of the region on
   each side, `REGION_REUSE_MARGIN = 0.25`, as
   [§ "Where the pixels come from"](#where-the-pixels-come-from) defines it
   under "Reuse across renders".
3. **The point order of bulk kernels.** *Decided by the user:* grouped by
   reference image and then by the reference keypoint's row and column, not
   by tile, because a point order that does not depend on how photographs
   are cut into tiles is better (§ "Memory").
4. **The pool and the tile window.** *Decided by the user:* four tiles'
   worth of copied regions per rayon thread in the kernel call's region pool,
   16 pinned tiles per render on the tile path, and no cap on the patches in
   progress. Where the memory involved is small, here hundreds of megabytes
   at worst against budgets of several gigabytes, speed comes before tighter
   memory limits. The values change memory and speed, never a result.
5. **An interface for kernels.** *Decided by the user:* a kernel holds a
   `&TiledPhotograph` in `ProjectedImage`, and the render entry points
   resolve each render's region pyramid, copied from the pool or read
   through tiles; kernels never compute or request regions. The region logic
   lives in one place, and the tiling details stay out of the kernels, as
   with the point order of question 3.
6. **The cluster refinement's evaluation tile.** *Decided by the user:* a
   cluster member's elongation is limited to `MAX_PATCH_ELONGATION`, 16,
   since the extreme cases are rarely correct photometric matches; a member
   past it is `not_evaluated`, and an evaluation past it scores as out of
   frame (§ "The members' elongation limit"). The limit is the main
   draft's step 8.
