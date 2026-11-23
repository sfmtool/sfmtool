# View-region pyramids and the render entry points

When a patch kernel renders a view's tile, it reads a few hundred to a few
thousand pixels of one or two levels of a photograph's pyramid, near where the
patch projects. A view-region pyramid holds, for one such read, only the
rectangle of each level the read can reach, with each rectangle's offset in
the full level, and the samplers read it at the same coordinates as a whole
pyramid and get the same values, bit for bit. The render entry points are how
a kernel reads a photograph this way: each works out the rectangles its read
needs, assembles them from the photograph's tiles when the kernel call's
region pool can hold them and keeps them for the view's next read, and
otherwise reads every tap straight out of the photograph's tiles, so the
memory a read holds grows with the patch it renders rather than with the
photograph. This spec covers the regions, the two ways a read reaches its
pixels, the reuse of kept regions across renders, and the entry points,
`source_clipped_share` among them. The draft
[out-of-core-photograph-pyramid.md](../../drafts/out-of-core-photograph-pyramid.md)
and its companion [view-region-pyramid.md](../../drafts/view-region-pyramid.md)
propose moving the kernels and panels onto them. Today the entry points read
photographs made in memory. Track View, the Track Table's crops and
Measurement View's endpoint crops read a photograph made of the photograph
tiles they peeked in the photograph cache, through
`TiledPhotograph::from_tiles`, and draw every tile, crop and hover view
through the entry points
([gui/track-view.md](../../gui/track-view.md),
[gui/track-table.md](../../gui/track-table.md),
[gui/measurement-view.md](../../gui/measurement-view.md)). Every patch
kernel reads a `TiledPhotograph` through its `ProjectedImage`, and the
bench's own reads go through the entry points: the cluster stage's member
grids, the grid-plane
normal's context renders, the pairwise-flow crops and the grown plane's value
and Sobel reads, each bench step making the region pool for its kernel call
([core/bench/editable-track.md](../bench/editable-track.md) § "Each step reads
through the region pool its caller made"). So do the patch kernels beneath
them, with the pool their caller passes down: the cluster refinement, its
grids through `sample_source_grid` and its evaluation tiles through
`read_source_rect`
([patch/cluster-patch-refinement.md](../patch/cluster-patch-refinement.md));
the view selection, its affine fast path through `render_source_affine` and
its exact renders through `render_source_tile`
([patch/patch-view-selection.md](../patch/patch-view-selection.md)); and the
reference view and every kernel that reads through it, the tile through
`render_source_tile` and its clipped share through `source_clipped_share`
([patch/reference-view.md](../patch/reference-view.md)): member coherence,
the stored bitmap, the observation readings, the keypoint localizer and
sub-pixel refiner, the candidate spawn and *Add image to tracks*. The grey
reads of the nearby-track search read the rectangle each samples through
`read_source_rect`
([bench/far-field-sweep.md](../bench/far-field-sweep.md)), so no kernel reads
the whole pyramid a tiled photograph holds.

## Rust API

The regions live in [image/regions.rs](../../../crates/sfmtool-core/src/camera/image/regions.rs)
and [sampler/regions.rs](../../../crates/sfmtool-core/src/camera/sampler/regions.rs),
the samplers' level reads in
[remap/texels.rs](../../../crates/sfmtool-core/src/camera/remap/texels.rs),
the tiled photograph in
[tiled_photograph.rs](../../../crates/sfmtool-core/src/camera/tiled_photograph.rs),
the entry points, the kept regions, the pool and the tile path in
[sampler/source.rs](../../../crates/sfmtool-core/src/camera/sampler/source.rs),
[sampler/pool.rs](../../../crates/sfmtool-core/src/camera/sampler/pool.rs) and
[sampler/tile_window.rs](../../../crates/sfmtool-core/src/camera/sampler/tile_window.rs),
`source_clipped_share` beside `clipped_share` in
[reference_view/tile.rs](../../../crates/sfmtool-core/src/patch/reference_view/tile.rs),
and `sample_source_member_grid` beside `sample_member_grid` in
[cluster_refine.rs](../../../crates/sfmtool-core/src/patch/cluster_refine.rs).

```rust
/// One level of a pyramid that may hold only a rectangle of the full level.
pub struct PyramidLevel { pub image: ImageU8, pub origin: [u32; 2], pub full_size: [u32; 2] }
impl ImageU8Pyramid {
    /// `None` for a level the read does not take; refuses no held level, an
    /// empty level, a rectangle past its level, mixed channel counts and full
    /// sizes that are not each other halved.
    pub fn from_regions(levels: Vec<Option<PyramidLevel>>) -> Result<Self, PyramidRegionsError>;
    pub fn level_origin(&self, l: usize) -> [u32; 2];   // [0, 0] for a whole level
    pub fn level_full_size(&self, l: usize) -> [u32; 2];
}

/// What a kernel reads a photograph through: its layout and channels with no
/// pixel read, and its pixels a photograph tile or a rectangle at a time.
pub struct TiledPhotograph { /* … */ }
impl TiledPhotograph {
    pub fn from_image(image: ImageU8) -> Self;              // every tile of its full-depth pyramid
    pub fn from_pyramid(pyramid: Arc<ImageU8Pyramid>) -> Self; // a whole pyramid, read in place
    /// The photograph tiles `tiles` of `layout`'s photograph, held, and no
    /// other pixel; refuses no tiles, tiles of different channels and a tile
    /// the layout does not have or place as the tile says.
    pub fn from_tiles(layout: PyramidLayout,
        tiles: impl IntoIterator<Item = Arc<PhotographTile>>) -> Result<Self, PhotographReadError>;
    pub fn layout(&self) -> &PyramidLayout;
    pub fn channels(&self) -> u32;
}
pub enum PhotographReadError {
    Cancelled,
    OutOfRange { path: Option<PathBuf>, why: String },
    /// Every tile the read needs, which a photograph made by `from_tiles`
    /// does not all hold.
    NeedsTiles { keys: Vec<TileKey> },
}

pub const REGION_REUSE_MARGIN: f64 = 0.25;
pub const REGION_POOL_BYTES_PER_THREAD: u64 = 4 * 512 * 512 * 3;
pub const TILE_WINDOW_TILES: usize = 16;
pub struct RegionPool { /* … */ }
impl RegionPool {
    pub fn for_kernel_call() -> Self;        // REGION_POOL_BYTES_PER_THREAD per rayon thread
    pub fn with_capacity(bytes: u64) -> Self;
    pub fn held_bytes(&self) -> u64;
    pub fn stats(&self) -> RegionPoolStats;  // the most bytes held at once; reads copied, reused and on tiles
}

/// One view of the patch a kernel is working on: its photograph, the kernel
/// call's pool, and the view-region pyramid kept from its last copied read.
pub struct ViewRegions<'a> { /* … */ }
impl<'a> ViewRegions<'a> {
    pub fn new(source: &'a TiledPhotograph, pool: &'a RegionPool) -> Self;
    pub fn source(&self) -> &'a TiledPhotograph;
    pub fn pool(&self) -> &'a RegionPool;
    pub fn kept_regions(&self) -> Option<&LevelRegions>;
    pub fn kept_bytes(&self) -> u64;
}

pub fn required_regions(map: &mut WarpMap, sampler: Sampler, layout: &PyramidLayout) -> LevelRegions;
impl LevelRegions {
    pub fn for_samples(layout: &PyramidLayout, level: usize, positions: &[[f32; 2]], border: u32) -> Self;
    pub fn grown(&self, margin: f64) -> Self;
}

pub fn render_source_tile(view: &mut ViewRegions<'_>, map: &mut WarpMap,
    sampler: Sampler) -> Result<ImageU8, PhotographReadError>;
pub fn render_source_tile_with_grad_into(view, map, sampler,
    out: &mut ImageF32WithGrad) -> Result<(), PhotographReadError>;
pub fn render_source_affine(view, a: &[f64; 6], level: usize, pixels: &[usize],
    resolution: usize, sampler: Sampler, out: &mut [f32]) -> Result<(), PhotographReadError>;
pub fn sample_source_grid(view, a: &[f64; 6], level: usize, points: GridPoints<'_>,
    edge: GridEdge) -> Result<Option<Vec<f32>>, PhotographReadError>;
pub fn read_source_samples(view, level: usize, positions: &[[f32; 2]],
    border: u32) -> Result<SourceSamples, PhotographReadError>;
pub fn read_source_rect<'p>(source: &TiledPhotograph, pool: &'p RegionPool, level: usize,
    rect: LevelRect) -> Result<SourceRect<'p>, PhotographReadError>;
pub fn source_clipped_share(source: &TiledPhotograph,
    outline: &[[f64; 2]]) -> Result<Option<f64>, PhotographReadError>;
pub fn sample_source_member_grid(view, position: [f64; 2], affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams) -> Result<Option<Vec<f32>>, PhotographReadError>;
```

| Entry point | Reads as | Returns |
|-------------|----------|---------|
| `render_source_tile`, `render_source_tile_with_grad_into` | [`render_tile`](image-warping.md#choosing-the-sampler-per-view) and `render_tile_with_grad_into` | the tile, or the tile and its gradients |
| `render_source_affine` | the view selection's affine fast path: one bilinear sample of a level per support pixel with `u8` rounding, or under `Anisotropic` the anisotropic sample of the map's constant SVD at level-0 positions | the support's raw samples, planar |
| `sample_source_grid` | the cluster refinement's grid samplers (`sample_grid_window`, `build_template`): an `f64` map, one level, and an edge rule (`GridEdge::Clamp`, `Missing` or `InFrame`) over a window of the grid or a list of its pixels | the raw samples, interleaved, or `None` where the rule refuses the grid |
| `read_source_samples` | for each position, the `(2 + 2 · border)²` pixels from `floor(x − 0.5) − border` and `floor(y − 0.5) − border`, each address clamped to the level, and the bilinear in-cell fractions | the windows, from which the grown plane's value and Sobel gradient blend as `sample_bilinear_u8` and `sobel_gradient` do |
| `read_source_rect` | a rectangle the kernel's own arithmetic defines | its pixels, copied whether or not the reservation succeeds |
| `source_clipped_share` | [`clipped_share`](../patch/reference-view.md): the share of level-0 pixel centres inside a tile's outline that are clipped | the same share as over the whole level 0, or `None` for an outline of fewer than three points |
| `sample_source_member_grid` | `sample_member_grid`: a cluster member's `R x R` grid at its position and shape, its level and map by the cluster refinement's own rules, read through `sample_source_grid` with `GridEdge::Clamp` | the raw samples, interleaved, or `None` where `sample_member_grid` gives `None` |

**Why it is shaped this way.** A kernel never computes a region or touches a
tile: it hands the entry point what it would have handed the sampler, and the
entry point decides how to reach the pixels. So the region logic lives in one
place, beside the samplers whose footprints it follows, and a change to a
sampler's footprint and to its region is one change. The kernels keep their
level rules (`level_for_map`, `affine_core_map`), so the grid and affine entry
points take the level and the map in that level's pixels. Every entry point
returns `PhotographReadError`, because a photograph read from files can fail
mid-read; one made in memory fails only for a level, rectangle or tile it does
not have, or for having no pixels. The whole-pyramid samplers (`render_tile`,
the `remap_*` functions) keep their signatures and stay infallible: a pyramid
already assembled cannot fail.

The pool belongs to a kernel call, not to a thread or a view: the caller of a
kernel makes one for the call and passes it in, so a call over 151 views shares
one pool rather than taking one per view, and a kernel that calls another
passes its own pool down. A `ViewRegions` belongs to one view of one patch: a
kernel that optimizes a patch renders the same view many times with a slightly
moved frame (the bench's fit rounds, the sub-pixel refinement, normal
refinement), so it makes one `ViewRegions` per view of the patch it is working
on and passes it to every read of that view. A fit's rounds, a refinement's
steps and an evaluation's views of one track share their views' kept regions,
and the kernel drops them all when it moves to the next patch, so the next
starts with none and what a kernel keeps is at most one region pyramid per
view of the patches in progress, all reserved from the one pool. The kept
regions are opaque to the kernel: it never asks whether a read reused them,
and reuse changes no value. `read_source_rect` takes the pool itself, since
its rectangle is the kernel's and the `SourceRect` it returns is what the
kernel keeps; `source_clipped_share` takes no pool, since it holds one tile at
a time and copies nothing. The panels' reads on the GUI thread are not one
kernel call but many small ones a frame, so each panel owns one pool of
`REGION_POOL_BYTES_PER_THREAD` that all of them share, and makes a
`ViewRegions` for each picture it cuts, dropped once the picture is cut: a
panel keeps the picture, not the regions. A panel renders on the GUI thread,
which must never decode, from what the photograph cache holds, which another
thread can evict at any time, so Track View renders from a photograph made
by `from_tiles` of the tiles it peeked, which holds them for the render; a
read of it that needs a tile it does not hold is refused with `NeedsTiles`
before any pixel is read, naming every tile the read needs, so the panel
learns them from the read itself, with no second copy of the region logic,
and asks for them together. `sample_source_member_grid` is the
cluster member grid's read for a caller that is not the cluster refinement,
which keeps its own level rule private; Track View's cluster-stage tiles call
it.

**A bulk kernel takes its points in one order.** The kernels that loop over
a reconstruction's points in parallel, the stored bitmaps' whole-cloud render
`render_patch_cloud_bitmaps`, `refine_normals` and `refit_points`, take them
by the image of each point's reference observation, then by that
observation's keypoint, its pixel row `floor(y)` and then its `x`, with ties
broken by point index; the points with no reference observation (`-1`, none
within the track, or no column) come last, by point index, and a point set
with no keypoints orders one image's points by point index. The renders of
one photograph then follow each other, and so do those of points near each
other in it, so a read finds the photograph tiles the reads before it used.
The order uses only the reconstruction, never how a photograph is cut into
tiles. The parallel loop runs over the ordered list, which rayon splits into
runs of consecutive entries, so the points in progress together are in
about one reference image per worker; a worker that waits on a point's
nested parallel work, such as the pairwise-flow normal's, may take up another
point meanwhile. Each result is written at its point's
index, so the order changes no answer, and the whole-cloud render's error is
still the read of the first patch, in the cloud's order, that failed. It is
`BulkPointOrder` in
[reconstruction/bulk_point_order.rs](../../../crates/sfmtool-core/src/reconstruction/bulk_point_order.rs),
crate-private, the one place all three kernels take it from.

```rust
use sfmtool_core::camera::sampler::{render_source_tile, RegionPool, Sampler, ViewRegions};
use sfmtool_core::camera::TiledPhotograph;

let source = TiledPhotograph::from_image(image);
let pool = RegionPool::for_kernel_call();
let mut view = ViewRegions::new(&source, &pool);
let tile = render_source_tile(&mut view, &mut map, Sampler::Anisotropic)?;
// The next round's frame moved a little: this render reuses the kept regions.
let next = render_source_tile(&mut view, &mut moved_map, Sampler::Anisotropic)?;
```

## Theory

**Why a region gives the whole level's values.** Every sampler reads level `l`
at the level-0 position divided by `2^l`, takes the cell
`(floor(x − 0.5), floor(y − 0.5))` and its fractions with `bilinear_cell`, and
reads the taps `x0` and `x0 + 1` (and the same for `y`), each clamped to the
level. With a held rectangle at `origin`, the sampler subtracts the origin
from the integer tap addresses, after the `floor`, and clamps to the held
rectangle. The fractions are computed from the full-level coordinate, so they
are the whole level's bits whatever the coordinate, and the integer
subtraction is exact. The two reads take the same pixel for every tap when:

1. the rectangle is clipped to the full level, so a rectangle edge that is an
   image edge clamps where the whole level clamps; and
2. every tap, after the whole level's clamp, lies inside the rectangle, so a
   rectangle edge that is not an image edge is never reached.

A tap outside the rectangle reads the rectangle's nearest pixel, so a region
too small gives other values but never reads outside the held pixels.

**Which taps a read takes.** The regions are computed before the read by the
same rules the read applies, so they are exact rather than estimated:

- **`Bilinear`**: level 0 at each valid texel's `(sx, sy)`.
- **`BilinearMip`**: level `mip_level_for_sigma(σ_major)` at `(sx, sy) / 2^l`.
- **`Anisotropic`**: a texel with `σ_major ≤ 1` takes one level-0 sample;
  otherwise the `n` samples of `aniso_footprint` along the major axis
  (`aniso_tap`, the same positions the scalar and AVX2 paths take) at
  `level_lo`, and at `level_hi` when the blend fraction is above 0.
- **Value and gradient renders** read the same taps as the value renders.
- **The affine fast path**: the cell of `(a · (col, row) − 0.5)` cast to `f32`
  at its level, or under `Anisotropic` the anisotropic rule at the `f32`
  affine positions with the map's constant SVD.
- **The cluster grids**: the cell of each grid point the edge rule samples;
  a missing sample reads nothing, and a refused grid reads nothing at all.
- **A read at positions** (`LevelRegions::for_samples`): the bounding box of
  each position's tap pair widened by `border` pixels on every side, clamped
  to the level.

`required_regions` walks the warp map once, at about the cost of the render's
own address arithmetic.

**Why a tile holds every tap of a cell.** A photograph tile holds its core and
a one-pixel border of its neighbours' pixels, or the edge pixel repeated at
the image's edge. A cell's taps `x0` and `x0 + 1`, clamped to the level, lie in
the core of the tile whose core holds the clamped `x0` and in that tile's
right border; and the border pixel is what the whole level reads there,
including at the image's edge, where both taps clamp to the edge pixel. So a
cell is read from one tile, and a 4 x 4 window, which reaches one pixel
further, reads each of its clamped pixels from the tile whose core holds it.
A copied region is the cores of the tiles it covers, which partition the
level, so it holds the level's own pixels; a photograph in memory copies it
straight from the level it holds, which is the same pixels.

**What the reuse margin costs.** A kept region is its required region, `w` by
`h` pixels, grown by `ceil(REGION_REUSE_MARGIN · w)` pixels on the left and on
the right and `ceil(REGION_REUSE_MARGIN · h)` above and below, then clipped to
the level; clipping changes no value, since sampling clamps at the level's
edge the same way. Before clipping, a kept region is 1.5 times as wide and as
tall as the required one when its sides are multiples of 4, so 2.25 times the
area. Rounding up adds at most 1.5 px to the width and to the height, so in
general the area is at most `(1.5 + 1.5 / w)(1.5 + 1.5 / h)` times, a bound of
about 2.3 for square regions of about 60 px and 2.5 for ones of about 20 px.
For regions of a few pixels it is well above 2.25, since a 1 px side grows to
3 px and a 2 px side to 4 px.

**Why the clipped share over tiles equals the whole level's.** The even-odd
rule finds, for each row of pixel centres, where the row crosses the
outline's edges, and counts the centres between alternate crossings. The
crossings depend only on the outline and the row, not on which columns are
counted, so counting the columns of each tile's core separately and summing
counts each centre of the row exactly once, as the whole row does. The counts
are integers, so the share is the same number. `source_clipped_share` walks
the tiles under the outline's bounding box, which holds every centre inside it
(a crossing lies within the box up to rounding far under half a pixel), row of
tiles by row of tiles; an outline
smaller than one pixel, which holds no centre, reads the tile under each of
its points, as `clipped_share` reads the pixel under each.

## Implementation notes

**Two paths, one arithmetic.** The samplers' per-pixel code is generic over a
`LevelTexels` (one level) and a `PyramidTexels` (a pyramid): `HeldLevel` reads
a level held in memory, whole or a rectangle, and the tile path's `TileLevel`
reads through tiles. The row functions (`bilinear_row`, `bilinear_mip_row`,
`aniso_pixel`, and the gradient rows) are the same code for both, so the
arithmetic cannot drift between them. The AVX2 anisotropic kernel subtracts
the origin from its integer taps the same way; the tile path runs the scalar
per-pixel code, which the kernel equals bit for bit.

**Choosing the path.** An entry point computes its regions and first looks at
what its `ViewRegions` kept: when every required region lies inside the kept
region of the same level, it reads the kept pyramid, reserving nothing more.
Otherwise it drops the kept pyramid, which gives its bytes back to the pool,
grows each required region by `REGION_REUSE_MARGIN` (§ "What the reuse margin
costs"), and reserves the grown regions' bytes. It gives the old bytes back
before it reserves the new, so a view whose regions fill the pool can still
move. A reservation is a compare-and-exchange on one counter that succeeds or
fails at once and never waits, so nested parallel loops on many threads cannot
block on the pool or deadlock. On success the read assembles the grown regions
from the photograph's tiles into a region pyramid, reads it with the
whole-pyramid samplers, and keeps it, still reserved, in the `ViewRegions`
until a later read replaces it or the `ViewRegions` is dropped. On failure it
takes the tile path and the view keeps nothing, its old regions already given
back; a read of no pixel takes the tile path and leaves what the view kept as
it was. `read_source_rect` reserves the rectangle itself and assembles it
either way, from the reserved bytes or through the tile path's window; its
reservation lasts as long as the returned `SourceRect`. The pool counts the
reads that assembled their regions, the reads that reused kept ones and the
reads that took tiles, and the most bytes held at once.

**The tile path's window.** A read on the tile path keeps at most
`TILE_WINDOW_TILES` (16) tiles pinned, the least recently used out first, and
looks up a tile only when a tap leaves the last tile used. Its rows run in
order on the calling thread, so the window has one owner. A tile read that
fails is remembered, its taps read as 0, and the entry point returns the error
instead of the result.

**Photographs made of tiles.** `from_tiles` keeps its tiles by key and
answers a tile read with the tile it holds; so does `from_image`, which
holds every tile of its layout. A photograph that holds every tile of its
layout, as `from_image`'s does and as one `from_tiles` is handed all of them
does, needs no tile worked out before a read, so its reads skip the checks
below. Every entry point that goes
through the kept regions works out, before its reservation, the tiles under
its regions grown by `REGION_REUSE_MARGIN`, the tiles a copied read
assembles them from, and refuses with `NeedsTiles` when the photograph does
not hold all of them; a read that reuses kept regions reads no tile, and the
tile path's taps lie inside the ungrown regions, so a read that passes the
check takes either path with the tiles it holds. `read_source_rect` checks
the tiles under its rectangle, and `source_clipped_share` the level-0 tiles
under the outline's bounding box and, when the geometry alone says no pixel
centre lies inside the outline, those under its corners, which it then
reads. A rectangle is assembled from the part of each
tile's core inside it, so the pixels are the level's. `NeedsTiles` names
every tile the read needs, held or not, so a caller makes the photograph
again of those and the read runs; a photograph of no tiles is refused, since
its channels are its tiles', so a caller that does not know the tiles a read
needs yet makes one of the coarsest level's first tile.

**Photographs in memory.** `from_image` builds the image's pyramid of
`ImageU8Pyramid::full_levels` depth, cuts every tile of it, each with its
border, at `PHOTOGRAPH_TILE_SIZE`, holds them, and drops the pyramid, so a
read gives the pyramid's values bit for bit; the Python bindings make every
photograph they are handed this way. `from_pyramid` reads a whole pyramid in
place, and only the photograph cache still makes one so: it cuts a tile,
with its border, only when the tile path asks for it, copying the columns
that need no clamping as one span a row, and keeps none, and a rectangle is
copied row by row straight from the level, with no tile cut. Cutting a tile
for `from_image` and for `from_pyramid` is the same copy.
`from_pyramid` refuses with `OutOfRange`
a tile with pixels that a pyramid made by `from_regions` does not hold, rather
than read the black of a level it does not hold, and so refuses a rectangle
whose tiles it does not wholly hold. Their layout has the pyramid's own
levels, each level 0's size halved, and no stored level. The tests cut tiles
of 32 px, cutting a photograph again at that size, and cut a pyramid of
fewer levels than the full depth that a test builds with the crate-private
`TiledPhotograph::cut`, which `from_image` cuts its pyramid with.

**What a read costs.** `required_regions` walks a map of more than 2048
texels on the rayon pool, row by row, as the render does, and on the
calling thread when that thread is already a rayon worker, so a read made
inside a parallel loop takes no other work onto its thread; the bounding boxes
of the rows are joined in any order, which gives the same rectangles. A copied
read of a photograph in memory then costs the walk, the copy of its grown
regions and the render of the region pyramid. Measured in a release build on
a 6000 x 4000 photograph, 50 renders each with a new `ViewRegions`, as the
panels render: a 64 x 64 tile takes 0.30 to 0.44 ms through regions against
0.22 to 0.31 ms from the whole pyramid, and a 192 x 192 one 0.78 to 0.90 ms
against 0.63 to 0.67 ms under `BilinearMip`, and 1.33 ms against 0.49 ms
under `Anisotropic`, whose walk takes up to 16 samples a texel on the scalar
path where the render runs the AVX2 kernel.

**Testing.** [source/tests.rs](../../../crates/sfmtool-core/src/camera/sampler/source/tests.rs)
renders random warps over a synthetic photograph, including maps off every
edge, NaN texels, a view shrunk past the top level and a footprint stretched
100 times on a diagonal, with every sampler, value and gradient, through a
pool every read fits in and through one none does, and compares the bits with
the whole pyramid's; it does the same for the grid and affine samplers, the
sample windows and the rectangles, and checks that nested parallel renders on
eight threads never hold more than a small pool, that each copied read
reserves its grown regions, and that the tile path pins no more than its
window. [source/kept_tests.rs](../../../crates/sfmtool-core/src/camera/sampler/source/kept_tests.rs)
checks that regions assembled from tiles equal the whole pyramid's pixels at
every level, across tile edges and at the image's edges; that each kept region
is its required region grown by the margin and clipped to the level, for sides
of 1, 2, 3 and 61 px, inside the image and at and beside each of its edges;
that renders whose required regions lie inside the kept ones reuse them, with
the whole pyramid's values and a fresh view's, and renders outside reassemble,
as does a read with any one level outside;
and that kept regions are counted in the pool, given back before a reassembly
reserves, and released when their view is dropped, and that a view whose
render finds the pool full of another view's kept regions takes the tile path
and copies again once they are released.
[tiled_photograph/tests.rs](../../../crates/sfmtool-core/src/camera/tiled_photograph/tests.rs)
checks that a photograph `from_image` makes holds every tile of the image's
pyramid, each tile, with its border, the pyramid's pixels bit for bit, and
reads each whole level and rectangles across tile edges and at the level's
far corner as the pyramid holds them, cut at 512 px over a photograph two
tiles wide and high and cut again at 16 px.
[regions/tests.rs](../../../crates/sfmtool-core/src/camera/sampler/regions/tests.rs)
checks that a region pyramid of the required regions equals the whole
pyramid, that one a pixel smaller on any inner edge does not, and that a map
walked on the rayon pool gives the regions the serial walk gives. The grown
plane's value and Sobel reads through the windows are checked in
[source_read_tests.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/source_read_tests.rs).
The reference view's
[tests.rs](../../../crates/sfmtool-core/src/patch/reference_view/tests.rs)
checks that `source_clipped_share` over tiles of 32 px equals `clipped_share`
over the whole level 0 for outlines across tile edges, along a tile edge's
pixel centres, off each of the image's edges, concave, self-crossing, inside
one tile and smaller than a pixel at a tile corner. The cluster refinement's
[tests.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/tests.rs)
checks that `sample_source_member_grid` over tiles of 32 px equals
`sample_member_grid` over the whole pyramid, bit for bit, through a pool and
through tiles, at an interior position, across a tile corner, off three of the
image's edges and at a shape that reads a coarser level, at the tile's radius
and three times it, and that it refuses a degenerate shape and a non-finite
position as `sample_member_grid` does. The panels' pictures through regions
are checked against the whole pyramid's in their own tests
([gui/track-view.md](../../gui/track-view.md) § "Testing"), and the bench's
steps through regions against their results over a pool every read fits in
in [bench/tests/region_reads.rs](../../../crates/sfmtool-core/src/bench/tests/region_reads.rs),
the geometry search and the track stage's reading, fit and bitmap among them,
and the cluster refinement, the view selection and the kernels that read
through the reference view over the seoul_bull ground truth the same way in
[patch/region_read_tests.rs](../../../crates/sfmtool-core/src/patch/region_read_tests.rs).
