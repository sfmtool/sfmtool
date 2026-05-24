# Tile-batched consensus atlas: bounded-memory panorama compositing

**Status:** Implemented. Built on top of the existing
[`PerSphericalTileSourceStack`], [`SphericalTileRig`], and the
photometric RANSAC ([`refine_photometric_ransac`]) without changing any
of their algorithms. The orchestrator lives in
`crates/sfmtool-core/src/consensus_atlas.rs` (`render_consensus_atlas`),
the PyO3 binding in `crates/sfmtool-py/src/py_consensus_atlas.rs`, and
`sfm panorama` (`src/sfmtool/_panorama.py`) consumes it for the production
panorama render.

[`PerSphericalTileSourceStack`]: per-spherical-tile-source-stack.md
[`SphericalTileRig`]: spherical-tiles-rig.md
[`refine_photometric_ransac`]: ../drafts/photometric-subsets-ransac.md

## Problem statement

The current panorama path materialises one
`PerSphericalTileSourceStack<T>` over **every** source image at once,
then runs RANSAC over it, then collapses it to a consensus atlas. The
stack's level-0 buffer is `total_contrib_rows × base_patch_size² ×
channels × sizeof(T)`, where `total_contrib_rows = Σ_t n_contributors(t)
≈ n_tiles × N × f` and `f` is the fraction of the rig's tiles an average
source sees. For a full-resolution panorama this is the dominant
allocation and it grows linearly with the source count `N`.

The constant `f` is workload-dependent — a narrow-FOV capture has a
small `f`, while a 360° dual-fisheye rig sits near the worst case: each
lens covers roughly a hemisphere, so geometrically `f ≈ 0.5`, but on the
bundled Kerry Park set the *measured* `f ≈ 0.84` (the OPENCV_FISHEYE
lenses run slightly past 180°, the cull keeps any tile whose *centre* is
in-frame, and tiles overlap — so a hemispheric source ends up
contributing to most tiles). The table below is for that 360° rig
(`n_tiles = 320`, `base_patch_size = 128`, RGB):

| sources | rows (≈) | level-0 f32 | full pyramid f32 | full pyramid f16 |
|--------:|---------:|------------:|-----------------:|-----------------:|
| 320  | 86 k  | 17 GB | ~22 GB | ~11 GB |
| 640  | 172 k | 34 GB | ~45 GB | ~22 GB |
| 1280 | 345 k | 67 GB | ~89 GB | ~45 GB |
| 2078 | 560 k | 110 GB | ~146 GB | ~73 GB |

`f16` storage and skipping the unused mid-pyramid levels each shave a
constant factor, but the curve is still linear in `N` — past a few
hundred sources at production resolution the run OOMs on a workstation.

This spec adds a compositing path whose **peak memory is independent of
`N`** (modulo holding the decoded source images resident — see
[Memory](#memory)), produces a result **byte-identical to the monolithic
path**, and adds **no extra warp or RANSAC compute**.

## Key idea

### The picture, for someone new to the pieces

We have an **atlas of square tiles**. Each tile is a small pinhole
projection looking in one direction; the tiles overlap so that together
they cover the whole sphere of directions. ([`SphericalTileRig`] is this
set of tiles plus the rule for packing them into one 2-D atlas image.)

We load all the **source images**. Each one is a projection onto some
patch of the sphere — a fisheye frame covers roughly a hemisphere — so
each image only has pixels for the tiles whose direction it actually
sees, and conversely each tile draws on just the images pointed roughly
its way.

Now the batching. Pick a **set of tiles**. For each tile in the set,
warp every source image that sees it into that tile's local pinhole
frame — that gives, per tile, one warped patch per contributing source
(a *row*). Run the consensus step on those rows — for each tile, pick
the largest subset of sources that *agree* on the scene there and take
their per-pixel median (this is the photometric RANSAC; see
[`refine_photometric_ransac`]) — and write the resulting patch into the
destination atlas at that tile's slot. Drop everything, pick the next
set of tiles, repeat.

That's the whole shape: *build the warped rows for a handful of tiles →
consensus → write those atlas slots → throw it away → next handful.* The
destination atlas is the only thing that lives across iterations.

### Precisely

Process the rig's `n_tiles` tiles in **batches** of `B`. For each batch,
build a `PerSphericalTileSourceStack` over just those `B` tiles (all
contributing sources, all pyramid levels), run `refine_photometric_ransac`
on it, take the per-tile consensus patches, blit them into the full
atlas at the parent rig's tile slots, and drop the batch's stack before
the next batch:

```
# Precondition: full_rig already has its patch size fixed to a power of
# two — full_rig.set_patch_size(full_rig.patch_size().next_power_of_two())
# — exactly as for the monolithic build. tiles_subset inherits it.

decode all N source images                  # resident, once — see Memory
channels := sources.first().map_or(1, |s| s.image.channels())  # same rule as build_rotation_only
atlas := NaN-filled, full_rig.atlas_size() × channels, f32      # only buffer that persists
for batch in 0 .. ceil(n_tiles / B):
    tile_range := [batch·B .. min((batch+1)·B, n_tiles))   # global tile indices
    sub_rig    := full_rig.tiles_subset(tile_range)
    sub_stack  := PerSphericalTileSourceStack::<T>::build_rotation_only(
                      &sub_rig, &sources, &build_params)   # only this batch's buffers live
    ransac_b   := ransac_params.clone(); ransac_b.tile_index_base := tile_range.start
    ransac_out := refine_photometric_ransac(&sub_stack, &ransac_b)
    patches    := sub_stack.consensus_patches_per_tile(&ransac_out.primary_mask)  # Vec<Vec<f32>>, one per local tile
    for (local_t, patch) in patches.enumerate():
        blit patch into `atlas` at full_rig.tile_atlas_origin(tile_range.start + local_t)
    scatter ransac_out.tile_*_count / tile_*_lum_mad into the report at the global indices
    drop sub_stack, patches
return { atlas, tile_primary_count, tile_secondary_count, tile_primary_lum_mad, tile_secondary_lum_mad }
# caller resamples via SphericalTileRig::resample_atlas on the full rig + assembled atlas, unchanged
```

The atlas packs tile `t` at row `t / atlas_cols`, column `t %
atlas_cols`, each cell a `patch_size × patch_size × channels` block;
trailing cells beyond `n_tiles` stay `NaN`. A contiguous tile-index
batch is therefore a contiguous run of atlas cells (drawn here with
`atlas_cols = 5`, `B = 5`, `n_tiles = 14`, so each batch is one atlas
row — `B` is independent of the atlas width in general):

```
Full atlas — atlas_cols × atlas_rows cells, tile t at (t/cols, t%cols):

       col0    col1    col2    col3    col4
     ┌───────┬───────┬───────┬───────┬───────┐
 row0│  t0   │  t1   │  t2   │  t3   │  t4   │ ← batch 0: build a 5-tile sub-stack → RANSAC → write these cells
     ├───────┼───────┼───────┼───────┼───────┤
 row1│  t5   │  t6   │  t7   │  t8   │  t9   │ ← batch 1: … write these cells
     ├───────┼───────┼───────┼───────┼───────┤
 row2│  t10  │  t11  │  t12  │  t13  │  NaN  │ ← batch 2 (short): tiles 10..13; cell 14 has no tile → stays NaN
     └───────┴───────┴───────┴───────┴───────┘
   one batch's stack is live at a time;  the atlas is the only buffer that persists
```

(Note: atlas/index order is *not* sphere order — the tile relaxer
produces no spatial coherence — so a contiguous index batch is a
scattered set of *directions*. That is fine for correctness, and for the
360° workload this spec targets it barely matters for memory either
(each fisheye source sees ~84 % of all tiles regardless of how they're
grouped); see [Memory](#memory).)

### Why this is free

Each `(source, tile)` warp still happens exactly once — the work is the
same monolithic build, merely reordered so only one batch's patch
buffers are live at a time. The per-batch visibility culls also sum to
exactly the monolithic cull (each batch projects all `N` sources against
its `B` tiles; `Σ_batches B · N = n_tiles · N`). So there is **no extra
compute** of any kind — only the bookkeeping of slicing the rig and
re-entering `build_rotation_only` per batch. Peak memory is governed by
the row count of the heaviest single batch — see [Memory](#memory).

The result is **byte-identical** to the monolithic path for any `B`,
and the reason is narrow enough to state outright: `build_rotation_only`
uses no RNG (it is visibility cull + warp + downsample, all
deterministic), and every per-tile reduction downstream — the RANSAC
subset search in both passes, and the level-0 per-pixel **median** that
`consensus_patch_for_tile` takes — reads only its own tile's rows and is
order-independent. The *sole* batch-dependent input anywhere in the
chain is each per-tile RANSAC RNG, which is seeded from the tile's
*local* index; `tile_index_base` (see [Determinism](#determinism))
rewrites that seed to use the global index, closing the gap completely.
Nothing else in the path can observe how tiles were grouped.

The final equirectangular resample (`SphericalTileRig::resample_atlas`)
runs on the **full** rig and the assembled atlas, unchanged, and is not
memory-bound (the equirect output and the atlas are tens of MB) — it is
out of scope here.

## Non-goals

- **The equirect resample** stays as-is, on the full rig + assembled
  atlas. Not memory-bound, not touched.
- **Per-cluster / per-source debug panoramas** (rendering secondary,
  outlier, or single-source masks) re-run the consensus against the
  *full* stack with alternative masks; the batched path never
  materialises the full stack, so that kind of exploration stays on the
  monolithic path (`PerSphericalTileSourceStack` +
  `refine_photometric_ransac` + `primary_consensus_atlas`) — a dev aid on
  small data, where memory is not the constraint. Generalising the
  orchestrator to emit per-(tile, arbitrary-mask) patches is a possible
  later extension, not v1.
- **Spatially reordering the rig's tiles** so a contiguous batch is a
  contiguous patch of sphere (and thus touches fewer sources) would help
  *narrow-FOV* captures, where it isn't needed (small `f` ⇒ no memory
  pressure to begin with); for the 360° rig this spec targets it does
  almost nothing, since a hemispheric source contributes to most tiles
  no matter how they're ordered. Not pursued.
- **Streaming by source instead of by tile** is rejected outright — see
  [Why batch by tile, not by source](#why-batch-by-tile-not-by-source).

## Mechanism: `tiles_subset` + `consensus_patches_per_tile`

**Where the new code lives:** all of it in `sfmtool-core` —
`tiles_subset` on `SphericalTileRig` (in `spherical_tile_rig.rs`),
`consensus_patches_per_tile` on `PerSphericalTileSourceStack` (in
`per_spherical_tile_source_stack.rs`, alongside the
`primary_consensus_atlas` it factors out of), and the orchestrator plus
its `ConsensusAtlasBatch*` types in a new `consensus_atlas.rs`
re-exported from `lib.rs`. The PyO3 wrapper goes in a new
`crates/sfmtool-py/src/py_consensus_atlas.rs`. No new crates.

Two small additions to existing types, no algorithm changes:

```rust
impl SphericalTileRig {
    /// A rig containing only tiles `[range.start, range.end)` of `self`,
    /// re-indexed `0 .. range.len()`. Inherits `centre`, `half_fov_rad`,
    /// and `patch_size` (hence `tile_camera()`). The sub-rig's atlas is
    /// sized for `range.len()` tiles, so its `tile_atlas_origin` is *not*
    /// the parent's — the orchestrator never uses the sub-rig's atlas
    /// geometry, only the parent's `tile_atlas_origin(global_index)`.
    /// The direction KD-tree is rebuilt over the subset so `resample_atlas`
    /// / `warp_*` on the sub-rig stay correct (the orchestrator does not
    /// call those, but a sub-rig that silently mis-answers nearest-tile
    /// queries would be a footgun).
    pub fn tiles_subset(&self, range: std::ops::Range<usize>) -> SphericalTileRig;
}

impl<T: PatchPixel> PerSphericalTileSourceStack<T> {
    /// One consensus patch per tile (row-major `patch_size² × channels`
    /// f32, NaN where the primary cluster is empty or every primary
    /// contributor is invalid for a pixel) — the per-tile pieces that
    /// `primary_consensus_atlas` lays into an atlas, returned directly so
    /// the caller can place them into an atlas it owns. `primary_mask`
    /// length must equal `total_contrib_rows`.
    pub fn consensus_patches_per_tile(
        &self,
        primary_mask: &[bool],
    ) -> Result<Vec<Vec<f32>>, ConsensusAtlasError>;
}
```

`consensus_patches_per_tile` is just `primary_consensus_atlas`'s inner
loop (the existing `consensus_patch_for_tile` free function, run over all
tiles) without the atlas-layout step — `primary_consensus_atlas` then
becomes a thin wrapper over it. The orchestrator blits each returned
patch into the full atlas at `full_rig.tile_atlas_origin(tile_range.start
+ local_t)`, never materialising a per-batch atlas. (We avoid the
alternative — call the sub-rig's `primary_consensus_atlas`, then extract
tiles back out of a sub-sized atlas — because that round-trip works but
invites exactly the "which rig's origin?" confusion the doc comment
above warns about.)

`build_rotation_only` and `refine_photometric_ransac` need **no** changes
beyond the `tile_index_base` param in [Determinism](#determinism) — they
run on the sub-rig / sub-stack as-is. (`build_rotation_only` could
instead grow a `tile_range` parameter; the sub-rig route is preferred
because it keeps the builder's contract narrow.)

### Parallelism

Batches run **sequentially** — that is the whole point; peak memory is
one batch. Within a batch, `build_rotation_only` parallelises across
`(source, tile)` writes and `refine_photometric_ransac` /
`consensus_patches_per_tile` parallelise across the batch's tiles,
exactly as today. Running batches in parallel would multiply peak memory
by the number in flight and is explicitly **not** done.

## Why batch by tile, not by source

Streaming source *chunks* (rather than tile batches) cannot keep the
algorithm exact. Per-tile RANSAC needs a tile's *entire* contributor
population at once — the largest agreeing subset is defined over the
full set, so you cannot RANSAC a chunk and merge. You can accumulate
just the small target-level rows (size 4) across source chunks and
RANSAC once at the end, but then the level-0 consensus (per-pixel
**median** over the primary cluster) needs a second source pass, and a
streaming median needs either every primary value resident (worse than
the monolithic stack — a source touches many tiles, so you'd hold
medians for *all* tiles at once) or a per-pixel approximate-quantile
digest (a few GB of t-digests). The only O(1)-memory streaming reduction
is a running *mean*, which smears the per-pixel outliers — moving-object
edges on otherwise-agreeing sources — that the median exists to reject.
Tile-batching keeps the exact median for the price of holding the
decoded images resident; that is the better trade.

## Correctness and implementation notes

### The orchestrator must live in Rust

The batch loop is small enough to write in Python, but it must not be:
the PyO3 `build_rotation_only` binding's `parse_sources` **copies every
source image's pixel bytes** into a Rust `ImageU8`. A Python-side loop
would call `build_rotation_only` once per batch, re-marshalling all `N`
sources every time — `Σ Wᵢ·Hᵢ·Cᵢ × n_batches` of pure memcpy (~5.6 GB ×
≈`n_tiles/B` for the full Kerry Park set; tens of GB of wasted copying
at `B = 32`). A Rust `render_consensus_atlas` parses the Python source
list into Rust images **once** and reuses them across all batches. That,
not "hiding the intermediate stack", is the load-bearing reason the
orchestration is a single Rust call. (Mechanically: the `parse_sources`
helper currently lives in `py_per_spherical_tile_source_stack.rs` as a
private helper for the `build_rotation_only` binding — lift it to a
shared spot, e.g. a `py_sources.rs` module or `pub(crate)`, so both that
binding and the new `render_consensus_atlas` binding call it.)

### The visibility cull is per-batch, not a global setup phase

`build_rotation_only` runs its own visibility cull internally — for each
source, which of *that call's* tiles does it see — so the batched path
gets the cull "for free" per batch rather than computing one global
source↔tile bitmap up front. Total cull work is `n_tiles · N` ray
projections either way; the per-batch scratch (`B × N` bits/u32) is KBs.
Hoisting it to a true one-time step would need a new `build_rotation_only`
entry point and buys nothing — not done.

### Determinism

`refine_photometric_ransac` derives each tile's RNG state from
`(seed, tile_index)` so the result is independent of thread scheduling —
in **both** passes: the primary pass seeds tile `t` from
`seed.rotate_left(32) ^ (t as u64)`, the secondary pass from
`seed.rotate_left(32) ^ ((t as u64).wrapping_add(0xA5A5_A5A5_A5A5_A5A5))`.
On a sub-rig the local indices run `0 .. B`, not the global indices, so a
naïve batched run would seed local tile *b* with the *global* tile *b*'s
seed in **every** batch — different RANSAC candidate samples than the
monolithic run, hence (for tiles past the exhaustive-enumeration cap) a
different — though still valid — partition.

To keep the batched result **byte-identical** to the monolithic one,
`RansacPhotometricParams` gains:

```rust
/// Added to the local tile index before deriving per-tile RNG state, in
/// both the primary and secondary RANSAC passes. The monolithic path
/// leaves this 0; the tile-batched path sets it to the batch's starting
/// global tile index so the per-tile RNG streams match the monolithic
/// run regardless of batch size.
pub tile_index_base: usize,   // default 0
```

so the per-tile global index becomes `g = tile_index_base + local_t`,
and the two derivations become `seed.rotate_left(32) ^ (g as u64)` and
`seed.rotate_left(32) ^ ((g as u64).wrapping_add(0xA5A5_A5A5_A5A5_A5A5))`.
Batch *b* passes `tile_index_base = b·B`. With this the partition is a
pure function of `(sources, rig, params)` independent of `batch_size` —
a property the [validation plan](#validation-plan) asserts directly.
(`tile_index_base` is an internal plumbing field: the existing
`refine_photometric_ransac` PyO3 binding does **not** gain a kwarg for
it — it stays at the `0` default there; only the in-process Rust
orchestrator ever sets it.)
(Tiles whose contributor count `K` is small enough that `C(K, m) ≤
max_subsets_per_tile` enumerate all candidate subsets exhaustively and
use no RNG, so they are already batch-size-invariant without this fix;
`tile_index_base` only matters for the over-cap tiles.)

## Memory

**The point in one line.** Monolithic peak is the whole stack
(`total_contrib_rows × per_row_footprint`) held for the whole run;
batched peak is one batch's stack (`≈ rows-in-the-heaviest-batch ×
per_row_footprint`) plus the resident decoded images plus the assembled
atlas. Choose `B` to fit your budget. For the full 2078-source Kerry
Park set at production resolution with f16 storage and the unused
mid-pyramid levels dropped, `B = 32` lands the whole run around **13 GB**
(vs ~146 GB monolithic f32 / ~73 GB monolithic f16) — see the
[worked example](#worked-example) below.

```
mem │ monolithic ████████████████████████████████  ← one allocation, whole run
    │
    │ batched     ▟▖   ▟▖   ▟▖   ▟▖   ▟▖   ▟▖   ▟▖   ← peak = a single batch's stack
    └──────────────────────────────────────────────▶ time
                  b0   b1   b2   b3   b4   b5   b6
```

### The precise formula

The per-batch stack for a `B`-tile batch holds, per pyramid level `l`
(side `size_l = base_patch_size >> l`), a `patches` buffer of `R_b ·
size_l² · channels` elements of `T` and a `valid` buffer of `R_b ·
size_l²` bytes, where `R_b = Σ_{t ∈ batch} n_contributors(t)` is the
batch's row count. Define

```
pyramid_per_row := Σ_l size_l² · channels        # elements of T per row, all levels
valid_per_row   := Σ_l size_l²                   # bytes per row, all levels (≈ pyramid_per_row / channels)
per_batch_stack(R_b) := R_b · ( pyramid_per_row · sizeof(T) + valid_per_row )
```

(plus the per-batch visibility / position scratch, `B × N` bits / u32 —
KBs, negligible). Peak working set across the whole run is

```
peak ≈ Σᵢ Wᵢ·Hᵢ·Cᵢ                      (decoded sources, resident)
     + max_b per_batch_stack(R_b)        (heaviest single batch)
     + atlas_w · atlas_h · channels · 4  (the assembled atlas, tens of MB)
```

`Σ_b R_b = total_contrib_rows` exactly (every monolithic row belongs to
exactly one batch), so the average batch is `total_contrib_rows /
ceil(n_tiles / B) ≈ (B / n_tiles) · total_contrib_rows`. The *peak*
batch can exceed the average by the per-tile contributor skew — `max_b
R_b ≤ B · max_t n_contributors(t) = (B / n_tiles) · total_contrib_rows ·
(max_t n_contributors(t) / mean_t n_contributors(t))`. For
roughly-uniform fisheye coverage that "heaviest tile over mean tile"
ratio is ~1.2; a pathological rig where one tile is seen by every source
and the rest by few could push it higher. So the bound is *proportional
to* `B` with a rig-dependent constant ≥ 1 — not a clean `(B / n_tiles)`
fraction. (Spatially reordering tiles so a contiguous batch is a
contiguous patch of sphere would help only if many sources *don't* see a
given batch's directions — true for narrow-FOV captures, but those have
a small `f` and no memory problem; for the 360° rig here a hemispheric
source contributes to most tiles regardless of grouping, so reordering
buys little. Not pursued.)

### Decoded images must be resident

The orchestrator takes `sources: &[(CameraIntrinsics, RotQuaternion,
ImageU8)]` — already decoded, held in RAM for the duration. Cost is
`Σ Wᵢ·Hᵢ·Cᵢ` bytes: ~5.6 GB for the bundled 2078-image Kerry Park set
(960×960 RGB). On a 64 GB box this caps the path at ~22 k same-sized
sources — far beyond current need. If even that does not fit, the
fallback is to **page images from disk per batch** — re-decode each
source the first time a batch needs it, drop it at batch end — trading
I/O (≈`n_tiles/B` full decode passes) for truly `N`-unbounded memory;
v1 assumes residency and the paging variant is a later add behind the
same orchestrator signature.

### The other levers compose

`f16` storage (implemented) halves `sizeof(T)`; dropping the unused
mid-pyramid levels (proposed separately — only level 0 and the RANSAC
target level have consumers) takes another ~25 % off `pyramid_per_row`;
tile batching replaces `total_contrib_rows` with `max_b R_b`. All three
multiply.

#### Worked example

Full 2078-source Kerry Park set, `n_tiles = 320`, `base_patch_size =
128`, `total_contrib_rows ≈ 560 k`: monolithic f32 full pyramid ≈
146 GB. With f16 + skip-mid-levels the per-row footprint is ≈ `128²·3·2`
(level-0 patches) `+ 128²·1` (level-0 valid mask, ~16 % more) `+ 4²·3·2`
(target level) ≈ 112 KB, so the monolithic stack would be ≈ 63 GB. At
`batch_size = 32` (`max_b R_b ≈ 56 k · 1.2 ≈ 67 k`) the per-batch stack
is ≈ 7.5 GB, and total peak ≈ `5.6 GB (images) + 7.5 GB (batch stack) +
0.06 GB (atlas) ≈ 13 GB` — fits a workstation with headroom; halve the
batch term again at `batch_size = 16`.

## API

A single orchestration entry point, generic over the per-batch stack's
storage type `T` (the caller picks `f16` / `f32` at the type level,
exactly as with `build_rotation_only::<T>`; `u8` does not implement the
f32 read path the consensus and RANSAC need and is rejected by the
wrapper below):

```rust
/// Tunables for `render_consensus_atlas`.
pub struct ConsensusAtlasBatchParams {
    /// Tiles per batch. Smaller ⇒ lower peak memory, more per-batch
    /// fixed overhead. Must be ≥ 1 (0 is an error); a value above
    /// `n_tiles` acts as `n_tiles` (single batch = monolithic path).
    pub batch_size: usize,
    /// Photometric-RANSAC tunables. The orchestrator overwrites
    /// `ransac.tile_index_base` per batch; whatever the caller sets
    /// there is ignored. All other fields are forwarded unchanged.
    pub ransac: RansacPhotometricParams,
    /// Forwarded verbatim to `PerSphericalTileSourceStack::build_rotation_only`
    /// each batch. (Note: as of today `build_rotation_only` ignores its
    /// `BuildParams` argument — this field exists for forward-compat and
    /// to keep the call shape uniform.)
    pub build: BuildParams,
}

/// Atlas + per-tile validity signal, accumulated across batches. The
/// per-tile arrays are indexed by *global* tile index and are exactly
/// what a single monolithic `refine_photometric_ransac` would have
/// produced (the orchestrator just scatters each batch's slice into the
/// global arrays).
pub struct ConsensusAtlasReport {
    /// Row-major `atlas_h × atlas_w × channels` f32, NaN where no
    /// consensus (empty primary cluster, all-invalid pixel, or trailing
    /// atlas slot). Identical layout to
    /// `PerSphericalTileSourceStack::primary_consensus_atlas`.
    pub atlas: Vec<f32>,
    /// Per-tile primary / secondary cluster sizes, length `n_tiles`.
    pub tile_primary_count: Vec<i32>,
    pub tile_secondary_count: Vec<i32>,
    /// Per-tile primary / secondary luminance MADs (NaN where the
    /// cluster is below `min_inliers`), length `n_tiles` — same
    /// definition as `RansacPhotometricOutput`.
    pub tile_primary_lum_mad: Vec<f32>,
    pub tile_secondary_lum_mad: Vec<f32>,
}

/// Errors from `render_consensus_atlas`. The three `From`-style variants
/// just forward the inner call's error untouched (with the batch index
/// that produced it, for diagnostics); the first two are raised before
/// any batch work.
pub enum ConsensusAtlasBatchError {
    /// `params.batch_size == 0`.
    BatchSizeZero,
    /// `rig.patch_size()` is not a power of two. (Could equally be folded
    /// into `Build(BuildError::PatchSizeNotPowerOfTwo)`; called out
    /// separately here because the orchestrator can check it up front
    /// without constructing a sub-rig.)
    PatchSizeNotPowerOfTwo(u32),
    /// `PerSphericalTileSourceStack::build_rotation_only` failed for batch `batch`.
    Build { batch: usize, source: BuildError },
    /// `refine_photometric_ransac` failed for batch `batch`.
    Ransac { batch: usize, source: RansacPhotometricError },
    /// `consensus_patches_per_tile` (or the atlas blit) failed for batch `batch`.
    Consensus { batch: usize, source: ConsensusAtlasError },
}
// (The PyO3 `dtype="uint8"` rejection is a plain Python `ValueError` in
// the wrapper, not a variant here — the generic Rust path legitimately
// accepts any `T: PatchPixel`; see Edge cases.)

/// Composite a consensus atlas for `rig` over `sources` in tile batches,
/// never holding more than one batch's stack live.
///
/// Equivalent to building the monolithic `PerSphericalTileSourceStack<T>`,
/// running `refine_photometric_ransac`, and calling
/// `primary_consensus_atlas` — the `atlas` and the per-tile arrays are
/// byte-identical for any `batch_size` (see [Determinism](#determinism))
/// — but with peak per-batch stack memory governed by the heaviest
/// batch's row count (see [Memory](#memory)).
///
/// `rig.patch_size()` must already be a power of two (as for the
/// monolithic build). `u8`-backed batches are not supported — call with
/// `T = half::f16` or `T = f32`.
pub fn render_consensus_atlas<T: PatchPixel>(
    rig: &SphericalTileRig,
    sources: &[(CameraIntrinsics, RotQuaternion, ImageU8)],
    params: &ConsensusAtlasBatchParams,
) -> Result<ConsensusAtlasReport, ConsensusAtlasBatchError>;
```

**PyO3 surface:** a `render_consensus_atlas(rig, sources, batch_size=...,
dtype="float16", **ransac_kwargs)` free function. The binding does the
`match dtype { "float16" => render_consensus_atlas::<half::f16>, "float32"
=> render_consensus_atlas::<f32>, "uint8" => Err(...) }` dispatch — the
same pattern as `build_rotation_only`'s binding — and returns
`(atlas, tile_primary_count, tile_secondary_count, tile_primary_lum_mad,
tile_secondary_lum_mad)`. `sfm panorama` (`src/sfmtool/_panorama.py`)
drives its build→ransac→consensus trio through this single call, exposing
`--batch-size` (default chosen so the per-batch f16 stack is a few GB at
the command's default resolution). The command always takes the batched
path for the production render; per-cluster / per-source debug passes,
which need the full materialised stack, use the monolithic
`build_rotation_only` → `refine_photometric_ransac` →
`primary_consensus_atlas` trio directly (see [Non-goals](#non-goals)).

## Edge cases

- **`batch_size ≥ n_tiles`** — one batch; behaviour reduces exactly to
  the monolithic path (with `tile_index_base = 0`).
- **`batch_size = 0`** — rejected up front (a 0-tile batch is meaningless
  and would loop forever).
- **`batch_size` not dividing `n_tiles`** — last batch is short; handled
  by the `min(...)` in the range. No padding.
- **A batch with zero contributing rows** — every tile in it has an
  empty contributor list (no source sees those directions). The
  per-batch stack has `total_contrib_rows == 0`, RANSAC returns all-empty
  masks, every `consensus_patches_per_tile` entry is all-NaN; those atlas
  slots stay NaN. No crash, no special case.
- **`sources` empty** — every batch is a zero-row batch; the whole atlas
  is NaN, the count arrays all zero, the MAD arrays all NaN. The atlas is
  sized with `channels = 1` (the orchestrator's `sources.first()`
  fallback, matching `build_rotation_only`).
- **`u8` storage (`T = u8`)** — the PyO3 `dtype="uint8"` path errors up
  front, same rule as `primary_consensus_atlas` /
  `refine_photometric_ransac`. (The generic Rust signature accepts any
  `T: PatchPixel`, and `consensus_patches_per_tile` and the RANSAC gamma
  step both go through `PatchPixel::as_f32`, so a `u8` instantiation
  *would* run — it just range-promotes to `0.0–255.0` f32 like the
  others. The Python wrapper rejects it only for parity with the existing
  surface; nothing in the algorithm forbids a `u8` batched path if one
  is ever wanted.)
- **`rig.patch_size()` not a power of two** — rejected up front, same as
  `build_rotation_only`.

## Validation plan

1. **Batch-size invariance (the headline property).** On a reconstruction
   built from the `isolated_seoul_bull_17_images` fixture, run
   `render_consensus_atlas` with `batch_size ∈ {1, 3, 7, n_tiles}` and
   assert all `atlas` buffers are bitwise-equal, and equal to the
   monolithic `build_rotation_only::<f32>` → `refine_photometric_ransac`
   → `primary_consensus_atlas` result. Likewise assert
   `tile_primary_count`, `tile_secondary_count`, `tile_primary_lum_mad`,
   `tile_secondary_lum_mad` are identical across all of them and to the
   monolithic `RansacPhotometricOutput`. Repeat for `T = half::f16` (the
   three batched runs equal each other and the monolithic f16 run — *not*
   necessarily the f32 run).
2. **`tile_index_base` plumbing.** Unit test on a synthetic multi-tile
   stack large enough to have over-cap tiles (`C(K, m) >
   max_subsets_per_tile`): `refine_photometric_ransac` with
   `tile_index_base = k` on a stack of tiles `[0, m)` produces per-tile
   `primary_mask` / `secondary_mask` identical to running with
   `tile_index_base = 0` on the wider stack's tiles `[k, k+m)` — for
   *both* passes, since the secondary pass also seeds per-tile.
3. **Peak memory tracks the heaviest batch.** The orchestrator's
   sub-stacks are built and dropped internally, so the test can't weigh
   them directly. Instead: (a) reimplement the per-batch visibility cull
   in the test to get each batch's `R_b`, assert `Σ_b R_b ==
   total_contrib_rows` (the partition is exact) and that `max_b
   per_batch_stack(R_b)` equals the predicted peak from
   [Memory](#memory); (b) lean on `PerSphericalTileSourceStack`'s
   existing allocation test — `build_rotation_only` allocates exactly
   `Σ_l (total_contrib_rows · size_l² · channels) · sizeof(T) +
   total_contrib_rows · size_l²` with no slack — so a batch of `R_b` rows
   provably allocates `per_batch_stack(R_b)`. Together these pin the peak
   without needing a runtime hook. (If a direct check is wanted later,
   give `render_consensus_atlas` an optional `&mut dyn FnMut(usize
   /*batch*/, usize /*R_b*/)` progress/inspection callback — out of scope
   for v1.)
4. **`tiles_subset` correctness.** For a random rig and a random
   sub-range, assert the sub-rig's `direction(i)` / `basis(i)` /
   `tile_camera()` / `half_fov_rad()` / `patch_size()` match the
   parent's at `range.start + i`, that `len() == range.len()`, and that
   `atlas_size()` is sized for `range.len()` tiles (and so differs from
   the parent's whenever the column count changes).
5. **`consensus_patches_per_tile` ↔ `primary_consensus_atlas`.** For a
   small stack and a random primary mask, assert that laying the
   `consensus_patches_per_tile` outputs into a NaN-filled atlas at the
   rig's `tile_atlas_origin`s reproduces `primary_consensus_atlas`'s
   buffer bit-for-bit (the latter is meant to be a thin wrapper over the
   former).
6. **Empty batches.** A rig + source set engineered so some contiguous
   tile range has no contributors: assert those atlas slots are NaN, the
   corresponding count entries 0, the MAD entries NaN, and the run does
   not panic at any `batch_size` (including `batch_size = 1`).
7. **Rejections.** `batch_size = 0`, a non-power-of-two `rig.patch_size()`,
   and (via the PyO3 wrapper) `dtype="uint8"` each error out before any
   batch work.
8. **Wall-clock parity.** On the bundled `dino_dog_toy` set (85 imgs,
   2040×1536), assert the batched path's total time is within ~15 % of
   the monolithic path's (the reorder adds per-batch fixed overhead but
   no warp/RANSAC work) — a regression guard, not a tight bound.
