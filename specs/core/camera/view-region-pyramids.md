# View-region pyramids and the render entry points

When a patch kernel renders a view's tile, it reads a few hundred to a few
thousand pixels of one or two levels of a photograph's pyramid, near where the
patch projects. A view-region pyramid holds, for one such read, only the
rectangle of each level the read can reach, with each rectangle's offset in
the full level, and the samplers read it at the same coordinates as a whole
pyramid and get the same values, bit for bit. The render entry points are how
a kernel reads a photograph this way: each works out the rectangles its read
needs, copies them when the kernel call's region pool can hold them, and
otherwise reads every tap straight out of the photograph's tiles, so the
memory a read holds grows with the patch it renders rather than with the
photograph. This spec covers the regions, the two ways a read reaches its
pixels, and the entry points. The draft
[out-of-core-photograph-pyramid.md](../../drafts/out-of-core-photograph-pyramid.md)
and its companion [view-region-pyramid.md](../../drafts/view-region-pyramid.md)
propose moving the kernels and panels onto them; today the entry points read
photographs made in memory, and no kernel calls them yet.

## Rust API

The regions live in [image/regions.rs](../../../crates/sfmtool-core/src/camera/image/regions.rs)
and [sampler/regions.rs](../../../crates/sfmtool-core/src/camera/sampler/regions.rs),
the samplers' level reads in
[remap/texels.rs](../../../crates/sfmtool-core/src/camera/remap/texels.rs),
the tiled photograph in
[tiled_photograph.rs](../../../crates/sfmtool-core/src/camera/tiled_photograph.rs),
and the entry points, the pool and the tile path in
[sampler/source.rs](../../../crates/sfmtool-core/src/camera/sampler/source.rs),
[sampler/pool.rs](../../../crates/sfmtool-core/src/camera/sampler/pool.rs) and
[sampler/tile_window.rs](../../../crates/sfmtool-core/src/camera/sampler/tile_window.rs).

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
    pub fn from_image(image: ImageU8) -> Self;              // its full-depth pyramid
    pub fn from_pyramid(pyramid: Arc<ImageU8Pyramid>) -> Self; // a whole pyramid, read in place
    pub fn layout(&self) -> &PyramidLayout;
    pub fn channels(&self) -> u32;
}
pub enum PhotographReadError { Cancelled, OutOfRange { path: Option<PathBuf>, why: String } }

pub const REGION_REUSE_MARGIN: f64 = 0.25;
pub const REGION_POOL_BYTES_PER_THREAD: u64 = 4 * 512 * 512 * 3;
pub const TILE_WINDOW_TILES: usize = 16;
pub struct RegionPool { /* … */ }
impl RegionPool {
    pub fn for_kernel_call() -> Self;        // REGION_POOL_BYTES_PER_THREAD per rayon thread
    pub fn with_capacity(bytes: u64) -> Self;
    pub fn held_bytes(&self) -> u64;
    pub fn stats(&self) -> RegionPoolStats;  // the most bytes held at once, reads copied, reads on tiles
}

pub fn required_regions(map: &mut WarpMap, sampler: Sampler, layout: &PyramidLayout) -> LevelRegions;
impl LevelRegions {
    pub fn for_samples(layout: &PyramidLayout, level: usize, positions: &[[f32; 2]], border: u32) -> Self;
    pub fn grown(&self, margin: f64) -> Self;
}

pub fn render_source_tile(source: &TiledPhotograph, pool: &RegionPool, map: &mut WarpMap,
    sampler: Sampler) -> Result<ImageU8, PhotographReadError>;
pub fn render_source_tile_with_grad_into(source, pool, map, sampler,
    out: &mut ImageF32WithGrad) -> Result<(), PhotographReadError>;
pub fn render_source_affine(source, pool, a: &[f64; 6], level: usize, pixels: &[usize],
    resolution: usize, sampler: Sampler, out: &mut [f32]) -> Result<(), PhotographReadError>;
pub fn sample_source_grid(source, pool, a: &[f64; 6], level: usize, points: GridPoints<'_>,
    edge: GridEdge) -> Result<Option<Vec<f32>>, PhotographReadError>;
pub fn read_source_samples(source, pool, level: usize, positions: &[[f32; 2]],
    border: u32) -> Result<SourceSamples, PhotographReadError>;
pub fn read_source_rect<'p>(source, pool: &'p RegionPool, level: usize,
    rect: LevelRect) -> Result<SourceRect<'p>, PhotographReadError>;
```

| Entry point | Reads as | Returns |
|-------------|----------|---------|
| `render_source_tile`, `render_source_tile_with_grad_into` | [`render_tile`](image-warping.md#choosing-the-sampler-per-view) and `render_tile_with_grad_into` | the tile, or the tile and its gradients |
| `render_source_affine` | the view selection's affine fast path: one bilinear sample of a level per support pixel with `u8` rounding, or under `Anisotropic` the anisotropic sample of the map's constant SVD at level-0 positions | the support's raw samples, planar |
| `sample_source_grid` | the cluster refinement's grid samplers (`sample_grid_window`, `build_template`): an `f64` map, one level, and an edge rule (`GridEdge::Clamp`, `Missing` or `InFrame`) over a window of the grid or a list of its pixels | the raw samples, interleaved, or `None` where the rule refuses the grid |
| `read_source_samples` | for each position, the `(2 + 2 · border)²` pixels from `floor(x − 0.5) − border` and `floor(y − 0.5) − border`, each address clamped to the level, and the bilinear in-cell fractions | the windows, from which the grown plane's value and Sobel gradient blend as `sample_bilinear_u8` and `sobel_gradient` do |
| `read_source_rect` | a rectangle the kernel's own arithmetic defines | its pixels, copied whether or not the reservation succeeds |

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
passes its own pool down.

```rust
use sfmtool_core::camera::sampler::{render_source_tile, RegionPool, Sampler};
use sfmtool_core::camera::TiledPhotograph;

let source = TiledPhotograph::from_image(image);
let pool = RegionPool::for_kernel_call();
let tile = render_source_tile(&source, &pool, &mut map, Sampler::Anisotropic)?;
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

## Implementation notes

**Two paths, one arithmetic.** The samplers' per-pixel code is generic over a
`LevelTexels` (one level) and a `PyramidTexels` (a pyramid): `HeldLevel` reads
a level held in memory, whole or a rectangle, and the tile path's `TileLevel`
reads through tiles. The row functions (`bilinear_row`, `bilinear_mip_row`,
`aniso_pixel`, and the gradient rows) are the same code for both, so the
arithmetic cannot drift between them. The AVX2 anisotropic kernel subtracts
the origin from its integer taps the same way; the tile path runs the scalar
per-pixel code, which the kernel equals bit for bit.

**Choosing the path.** An entry point computes its regions, grows each by
`REGION_REUSE_MARGIN` (a `w` x `h` region by `ceil(0.25 · w)` pixels left and
right and `ceil(0.25 · h)` above and below, clipped to its level), and reserves
the grown regions' bytes from the pool. A reservation is a compare-and-exchange
on one counter that succeeds or fails at once and never waits, so nested
parallel loops on many threads cannot block on the pool or deadlock. On
success the read copies the grown regions into a region pyramid, reads it with
the whole-pyramid samplers, and gives the bytes back when it ends; on failure,
or when it reads no pixel, it takes the tile path. `read_source_rect` reserves
the rectangle itself, and copies it either way; its reservation lasts as long
as the returned `SourceRect`. The pool counts the reads that copied and the
reads that took tiles, and the most bytes held at once.

**The tile path's window.** A read on the tile path keeps at most
`TILE_WINDOW_TILES` (16) tiles pinned, the least recently used out first, and
looks up a tile only when a tap leaves the last tile used. Its rows run in
order on the calling thread, so the window has one owner. A tile read that
fails is remembered, its taps read as 0, and the entry point returns the error
instead of the result.

**Photographs in memory.** `from_image` builds the image's pyramid of
`ImageU8Pyramid::full_levels` depth; `from_pyramid` reads a whole pyramid in
place, and refuses with `OutOfRange` a tile or rectangle with pixels that a
pyramid made by `from_regions` does not hold, rather than read the black of a
level it does not hold. Both cut a tile, with its border, or copy a rectangle
only when a read asks for it, and keep neither. Their layout has the
pyramid's own levels, each level 0's size halved, and no stored level. The tests cut tiles of 32 px.

**Testing.** [source/tests.rs](../../../crates/sfmtool-core/src/camera/sampler/source/tests.rs)
renders random warps over a synthetic photograph, including maps off every
edge, NaN texels, a view shrunk past the top level and a footprint stretched
100 times on a diagonal, with every sampler, value and gradient, through a
pool every read fits in and through one none does, and compares the bits with
the whole pyramid's; it does the same for the grid and affine samplers, the
sample windows and the rectangles, and checks that nested parallel renders on
eight threads never hold more than a small pool, that each copied read
reserves its grown regions, and that the tile path pins no more than its
window. [regions/tests.rs](../../../crates/sfmtool-core/src/camera/sampler/regions/tests.rs)
checks that a region pyramid of the required regions equals the whole
pyramid, and that one a pixel smaller on any inner edge does not. The grown
plane's value and Sobel reads through the windows are checked in
[source_read_tests.rs](../../../crates/sfmtool-core/src/bench/normal/grown_plane/source_read_tests.rs).
