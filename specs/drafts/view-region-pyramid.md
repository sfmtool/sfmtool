# View-region pyramid

**Status:** Draft

The required companion of
[out-of-core-photograph-pyramid.md](out-of-core-photograph-pyramid.md), which
links here. Decided by the user: every kernel that reads a photograph that
uses the out-of-core pyramid reads it through view-region pyramids, so its
memory grows with the patches it renders rather than with the photographs;
photographs under the size threshold keep whole in-memory pyramids. It is
kept as a draft of its own because what it specifies, the samplers' regions,
is separate from the files and caches the main draft specifies; its steps 1
and 2 are step 7 of the main draft, and its steps 3 to 6 are the main draft's
steps 9 to 12. Open: the questions at the end. It amends
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
  reads every level-0 pixel inside the tile's outline. It does not take a
  region pyramid: for a tiled source it walks the level-0 tiles the outline's
  bounding box covers, one at a time, applying the same even-odd rule to the
  pixel centres of each and summing the counts, so it holds one tile at a time
  and gives the same share. A footprint of `F` level-0 pixels covers up to
  `(ceil(F / 512) + 1)^2` tiles; this is the one read whose cost grows with the
  footprint rather than with the patch resolution, as it does today.
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
the region of each level from photograph tiles (`get_tile`, decoding any tile
not yet cached) and copies the rectangles out. The tiles come from the
mipmaps, and level 0 from the original if it is region-readable or from its
`-mipmap1` otherwise, as the main draft describes; every one is a tile
decode, so no region of any level needs a whole decode. A region pyramid
therefore holds the same pixels as a whole pyramid of the stored levels would.
An in-core photograph is not tiled for this: its whole pyramid is already in
memory, and its renders read it as today (`PhotographSource::Whole`).

A kernel's `ProjectedImage` holds a `PhotographSource`. `render_tile` and the
other render entry points compute the regions from the map and, for a `Tiled`
source, ask the cache for the region pyramid before sampling; for a `Whole`
one they sample as today. The kernels' own loops do not change.

**Reuse across renders.** A kernel that optimizes a patch renders the same
view many times with a slightly moved frame (the bench's fit rounds, the
sub-pixel refinement, normal refinement). The region pyramid is assembled with
a margin of a quarter of each level's region on every side, and a later render
whose required regions lie inside the held ones reuses it; one that does not
assembles a new one. Reuse does not change any value, only how often regions
are copied. A kernel keeps a region pyramid for one view of the patch it is
working on, and drops them all when it moves to the next patch (a fit's
rounds, a refinement's steps and an evaluation's views of one track share
them; the next track starts with none). So what it keeps is at most one region
pyramid per view of one patch, with the margin, 2.25 times the area of the
region itself.

## Bounding a render's memory

The region of a level is a bounding box, and the box of one render is not
bounded by the patch resolution `R`:

- **Anisotropic.** The level comes from `σ_minor`, but the `n` samples, capped
  at 16 (`MAX_ANISOTROPY`, `aniso_footprint` in
  [remap.rs](../../crates/sfmtool-core/src/camera/remap.rs)), still spread
  over `±σ_major / 2`. The cap bounds the number of samples, not their reach.
  A footprint stretched `a = σ_major / σ_minor` times along a diagonal has a
  bounding box of about `(R·a)² / 2` pixels at its level: `a = 100` at
  `R = 24` is about 8.7 MB for one render, 280 MB for 32 in flight, and a
  grazing view can approach a whole level 0.
- **One level per grid** (`level_for_map` in
  [cluster_refine.rs](../../crates/sfmtool-core/src/patch/cluster_refine.rs))
  picks its level from `s_min`, with the same effect for a stretched map.
- **`Bilinear`**, which the kernels' sampler argument can select, reads level 0
  over the whole footprint whatever `R` is: a 24 x 24 tile of a patch that
  covers 2000 px reads a 2000 x 2000 box, 12 MB.

So a render samples one of two ways, and both give the same values:

- **Copied regions**, as above, when the boxes of all the levels the render
  reads, with the reuse margin, total at most four tiles' worth of pixels
  (4 x 512 x 512 x 3 bytes, 3.1 MB). Sampling a copied region is the fast
  path: one contiguous image per level, and the AVX2 anisotropic kernel runs
  on it unchanged.
- **Tile-addressed sampling** otherwise. The render does not copy a region. It
  reads each sample's taps from the photograph tile that holds them: the tap
  pair `floor(x − 0.5)` and `floor(x − 0.5) + 1` (clamped to the level) always
  lies within one tile's core and its one-pixel border, so the tile is the one
  whose core holds the first tap, and the border pixel equals the neighbour's
  pixel, or the edge pixel at the image's edge, which is what clamping reads.
  The fractions are computed from the level coordinate as in the whole-level
  sampler, so every value is bit-identical. The render keeps at most 16 tiles
  pinned (12.7 MB), least recently used out, and fetches any other through
  the cache (`get_tile`, a lookup when the tile is cached and a decode of a few
  milliseconds when not). This path runs the scalar sampler, which the AVX2
  kernel already equals bit for bit (`remap/aniso_avx2.rs`), with a tile
  lookup only when consecutive samples change tile.

The worst case of one render is then 3.1 MB of copied regions or 12.7 MB of
pinned tiles, whatever the view; 32 renders in flight hold at most about
500 MB on top of the tile budget, and the common case, a footprint that is not
stretched, stays at a few kilobytes. No view is refused and no result changes:
which path a render takes depends only on its boxes, and both read the same
pixels. `clipped_share` already walks tiles one at a time (above). The cap of
four tiles and the window of 16 are open question 4.

**On the GUI thread** (Track View's tiles, crops and context, the Track
Table's crop column), the panel acts on the photograph's layout state as the
main draft's § "Readers on the GUI thread" says: while it is `Building` the
panel asks the level builder once to move it forward and requests no tiles;
once it is `Known` the panel computes the regions, peeks the tiles
(`peek_tile`), and draws the view's tile only when all of them are there,
requesting the missing ones. The GUI thread never decodes and never waits.

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
  What they hold is the region pyramids of the renders in flight (one per
  worker thread, a few kilobytes each, at most 3.1 MB copied or 12.7 MB of
  pinned tiles each, § "Bounding a render's memory") plus the tile cache, which its budget
  bounds whatever the number of photographs; the tiles they touch can be many
  more (level 2 of one 200-megapixel photograph is 48 tiles, 38 MB; level 0
  is 768 tiles, 607 MB), and the ones past the budget are decoded again when
  read again, at 2.3 to 6 ms a tile. Level 0 is always read by tiles, from the
  original or from `-mipmap1`, so no photograph's level 0 is ever held whole.
  The order a kernel takes its renders in decides how often a tile is decoded
  again: rendering points grouped by their reference image and position keeps
  the tiles each group needs together, and each kernel's phase note reports
  the tiles decoded and reused.
- **`GreyImages`** today holds a blurred `f32` grey copy of each view's
  level 0, 800 MB for a 200-megapixel view. With regions it holds a few
  kilobytes per read.

## Implementation steps

Steps 1 and 2 are step 7 of the main draft, landed before the tier exists
and exercised on in-core photographs cut into tiles in memory. Steps 3 to 6
are the main draft's steps 9 to 12, each its own pull request after the tier
lands switched off (main step 8) and before it is switched on (main step 15),
because once photographs are out of core there is no whole pyramid of one for
any of these readers to fall back on; until a reader moves, it refuses a tiled
source.

1. **Regions in the pyramid and the samplers**: `PyramidLevel`,
   `ImageU8Pyramid::from_regions`, the origin subtraction in every sampler and
   gradient sampler (the AVX2 anisotropic kernel included), the
   tile-addressed path with its window of pinned tiles, the choice between
   them by the region cap, and `required_regions` for each sampler. Tests: for
   random warps over a synthetic photograph, every sampler, every value and
   gradient render through a region pyramid built from `required_regions`, and
   through the tile-addressed path with a tile of 32 px, is bit-identical to
   the whole-pyramid render, including maps that run off every edge of the
   image, NaN texels, views shrunk past the top level, a footprint stretched
   100 times on a diagonal and a `Bilinear` render over a large footprint
   (which take the tile path); the tile path pins no more than its window; a
   region one pixel smaller on an inner edge makes the comparison fail, so
   the test can fail.
2. **`view_region_pyramid` in the cache**, with the reuse margin, and
   `clipped_share` over tiles, on in-core photographs cut into tiles in
   memory. Tests: the assembled regions equal the whole pyramid's pixels;
   `clipped_share` over tiles equals it over the whole level 0, for outlines
   that cross tile edges and the image's edges; reuse when inside, reassembly
   when not.
3. **Track View and the Track Table** on region pyramids, peeking tiles.
   Tests: a track of synthetic out-of-core photographs draws its tiles with no
   pyramid entry in the cache; the drawn tiles equal the whole-pyramid ones.
4. **The bench's jobs and the reference picks**: `ViewSources` hands the
   kernels a tiled source and each render asks for its regions; `clipped_share`
   reads a level-0 region. Tests: each step's result is identical with the two
   sources on a small-threshold synthetic reconstruction.
5. **Bulk kernels**: the display patch bitmaps, the conversion to embedded
   patches, the web export, *Refine all normals*, *Refit all points* and the
   cluster-patches build, each with a point order that keeps tile locality. Tests: identical results; a tile budget smaller
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
4. **The region cap and the tile window.** Four tiles of copied regions per
   render and 16 pinned tiles for the tile path (as designed), or other
   values. Recommendation: keep them, and count in each kernel's phase note
   how many renders took the tile path, so the cap is tuned on real captures;
   the values change memory and speed, never a result.
5. **An interface for kernels.** Designed as a `PhotographSource` enum in
   `ProjectedImage`, resolved to a region pyramid inside the render entry
   points. The other option is a `ProjectedImage` built per render with the
   region pyramid already in it, which would move the region computation into
   every kernel. Recommendation: the enum, as designed.
