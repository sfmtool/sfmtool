# View-region pyramid

**Status:** Draft

An optional optimization on top of
[out-of-core-photograph-pyramid.md](out-of-core-photograph-pyramid.md), which
does not depend on it and links here. Decided by the user: it is a separate
draft and not a prerequisite. Open: the questions at the end. It amends
[core/camera/photograph-cache.md](../core/camera/photograph-cache.md) and the
samplers of [core/camera/image-warping.md](../core/camera/image-warping.md).

When a patch kernel renders a view's tile, it reads a few hundred to a few
thousand pixels of one or two levels of the photograph's pyramid, near where
the patch projects. Today, and after the out-of-core design, it is handed the
whole pyramid of every photograph it reads: 762 MiB for a 200-megapixel
photograph, 12.6 GiB for a 17-image track of them. A view-region pyramid holds,
for one render of one patch in one photograph, only the rectangle of each level
that the render can reach, assembled from photograph tiles, with each level's
offset in the full level. The samplers read it at the same coordinates as a
whole pyramid and get the same values, so a kernel's results do not change, and
the memory a kernel touches grows with the number of patches it renders instead
of with the size of the photographs.

The name: the request called this the *per-match local pyramid*. In this tree
*match* means a feature correspondence (`.matches` files, `sfm match`), and the
unit here is one view of one patch, the unit the glossary's "Rendering a view's
tile" section names, so the proposed term is **view-region pyramid**: the
pyramid of the region one view's render reads.

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
  map's image of the grid at that level.
- **`clipped_share`** in
  [reference_view/tile.rs](../../crates/sfmtool-core/src/patch/reference_view/tile.rs)
  reads level 0 inside the tile's outline: the outline's bounding box.
- **The grey reads of *Find Nearby Tracks*** (`GreyImages` in
  [bench/nearby/grey.rs](../../crates/sfmtool-core/src/bench/nearby/grey.rs))
  blur level 0 with a nine-tap Gaussian, reflecting at the image's edges: the
  region of a read is the bounding box of its samples plus 4 px for the blur
  and 1 px for the bilinear tap, clipped to the image, and the blur reflects
  only at the image's edges, never at an inner region edge.

A function per sampler, `required_regions(map, sampler, layout) ->
LevelRegions`, walks the map once, as the render does; the region of a view's
tile costs about what the render's own address arithmetic costs, and is a
fraction of the sampling. It lives beside the samplers in
[sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs) so a change to a
sampler's footprint and to its region is one change.

## Where the pixels come from

`PhotographCache::view_region_pyramid(path, &LevelRegions, progress)` assembles
the region of each level from photograph tiles (`get_tile`, waiting on any
tile not yet decoded) and copies the rectangles out. The tiles come from the
mipmaps and level 0 as the out-of-core design describes, so a region pyramid
holds the same pixels as the whole pyramid the cache would have built. An
in-core photograph is not tiled for this: its whole pyramid is already in
memory, and its renders read it as today.

**Reuse across renders.** A kernel that optimizes a patch renders the same
view many times with a slightly moved frame (the bench's fit rounds, the
sub-pixel refinement, normal refinement). The region pyramid is assembled with
a margin of a quarter of each level's region on every side, and a later render
whose required regions lie inside the held ones reuses it; one that does not
assembles a new one. Reuse does not change any value, only how often regions
are copied.

**On the GUI thread** (Track View's tiles, crops and context, the Track
Table's crop column), the panel computes the regions, peeks the tiles
(`peek_tile`), and draws the view's tile only when all of them are there,
requesting the missing ones, as it waits for a whole pyramid today. The GUI
thread never decodes.

## Memory

- **Track View, a 17-image track of 200-megapixel photographs**: a 64 x 64
  tile whose patch covers `F` level-0 pixels reads level `≈ log2(F / 64)` over
  about 66 x 66 pixels, 13 KB, and a quarter of that at the next level for
  anisotropic blending; the photograph tiles it is cut from, one to four per
  level, are 0.8 to 3.2 MB in the tile cache. Seventeen views: about 50 MB,
  against 12.6 GiB of whole pyramids. These are estimates from the sizes, not
  measurements.
- **Bulk kernels** (the display patch bitmaps at open, *Refine all normals*,
  the web export) touch every photograph at the levels their patches need.
  The bound is then the tiles of those levels: level 2 of one 200-megapixel
  photograph is 48 tiles (38 MB), level 0 768 tiles (607 MB). Whether that
  fits depends on the order the kernel takes its points in; rendering points
  grouped by their reference image and position keeps the tiles each group
  needs together. Level 0 of an original the fast path does not cover is one
  whole decode while that photograph's level 0 is read, so bulk kernels take
  such photographs one at a time.
- **`GreyImages`** today holds a blurred `f32` grey copy of each view's
  level 0, 800 MB for a 200-megapixel view. With regions it holds a few
  kilobytes per read.

## Implementation steps

After step 6 of the out-of-core design.

1. **Regions in the pyramid and the samplers**: `PyramidLevel`,
   `ImageU8Pyramid::from_regions`, the origin subtraction in every sampler and
   gradient sampler (the AVX2 anisotropic kernel included), and
   `required_regions` for each sampler. Tests: for random warps over a
   synthetic photograph, every sampler, every value and gradient render
   through a region pyramid built from `required_regions` is bit-identical to
   the whole-pyramid render, including maps that run off every edge of the
   image, NaN texels, and views shrunk past the top level; a region one pixel
   smaller on an inner edge makes the comparison fail, so the test can fail.
2. **`view_region_pyramid` in the cache**, with the reuse margin. Tests: the
   assembled regions equal the whole pyramid's pixels; reuse when inside,
   reassembly when not.
3. **Track View and the Track Table** on region pyramids, peeking tiles.
   Tests: a track of synthetic out-of-core photographs draws its tiles with no
   pyramid entry in the cache; the drawn tiles equal the whole-pyramid ones.
4. **The bench's jobs and the reference picks**: `ViewSources` hands the
   kernels a tiled source and each render asks for its regions; `clipped_share`
   reads a level-0 region. Tests: each step's result is identical with the two
   sources on a small-threshold synthetic reconstruction.
5. **Bulk kernels**: the display patch bitmaps, the conversion to embedded
   patches, the web export, *Refine all normals*, *Refit all points*, the
   cluster-patches build and `ImagePyramidSet.read`, each with a point order
   that keeps tile locality. Tests: identical results; a tile budget smaller
   than one photograph's level 0 still completes.
6. **`GreyImages` by region.** Tests: the nearby-track search's results are
   identical.

## Open questions

1. **In-core photographs.** 368 frames of 3840 x 2160 are 12.2 GB of whole
   pyramids. Options: leave them whole (as designed); tile them too, from
   in-memory decodes kept as tiles. Recommendation: leave them whole until a
   capture of in-core photographs is measured to need it; their decode is
   0.05 s, not 1 s.
2. **The reuse margin.** A quarter of the region (as designed), a fixed number
   of pixels, or none. Recommendation: a quarter, then measure how often the
   bench's fits reassemble.
3. **The point order of bulk kernels.** Grouped by reference image and
   position (as designed), or by tile. Recommendation: by reference image and
   position, which needs no knowledge of the tile grid in the kernel.
4. **An interface for kernels.** `ProjectedImage` holds `&ImageU8Pyramid`.
   Options: a `pyramid` field that is an enum of a whole pyramid and a tiled
   source, resolved to a region pyramid inside `render_tile`; or a
   `ProjectedImage` built per render with the region pyramid already in it.
   Recommendation: the enum, so the kernels' loops do not change and the region
   is computed where the map is.
