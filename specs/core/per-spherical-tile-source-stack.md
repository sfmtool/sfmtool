# Per-spherical-tile source patch stack

**Status:** Implemented in
`crates/sfmtool-core/src/per_spherical_tile_source_stack.rs` (the
rotation-only build) and exposed to Python as
`sfmtool._sfmtool.PerSphericalTileSourceStack`. The pose-aware variant
described under "Pose-aware variant" is still future work.

The build runs the outer source loop sequentially and parallelises across
each source's kept tiles via rayon — each tile-task writes to its own
unique CSR row in the per-level buffers, so the writes are race-free.
The `BuildParams::max_in_flight_sources` knob is reserved for future
parallel-source chunking; setting it has no effect today.

## Motivation

Many algorithms operating on a `SphericalTileRig` need the same input:
for each tile `t`, the set of source images whose camera observes the
tile's look direction, together with each source's view of that
direction warped into the tile's local pinhole frame **as an image
pyramid** (level 0 at the base resolution, halving at each step down to
1×1). The pyramid lets a single warp serve coarse-to-fine processing —
the same observation is consulted at multiple scales without re-running
the warp per scale. Concrete consumers:

- **Coarse-to-fine consensus / agreement tests** (cheap pass at level
  L, refine at level L−1, …).
- **Per-direction depth or parallax regression** (per-source pixel
  coordinates at the appropriate scale for the current refinement
  step).
- **Source-visibility classification** (which sources contribute to
  this direction, which are occluded, which are out-of-frame).
- **Atlas painting / panorama compositing** with multi-scale
  alpha-weighted aggregation.
- **Joint photometric refinement** (per-image colour parameters +
  per-tile soft cluster assignment, optimised under L-BFGS with autodiff
  via `argmin` + `burn-autodiff`).
- **GPU pipelines** that want a per-`(tile, source, level)` storage
  texture fed once and reused across many kernels.

Without a shared primitive each of these re-derives the same data —
the per-source visibility filter, the per-source warp, the per-tile
slice, and the pyramid downsample. Three concrete benefits to
extracting it:

1. **A single point of truth for the visibility filter.** "Does
   source `i` contribute to tile `t`?" is answered once, recorded
   once, and reused by every algorithm that builds on the same
   `(rig, sources)`.
2. **A canonical per-tile multi-source pyramid dataset.** Downstream
   code reads the contiguous per-tile slice (or the whole-level
   buffer) and applies its own logic — no ambiguity about which
   sources, in which order, with which valid masks, at which scale.
3. **A single rayon parallelisation surface.** Building patches and
   pyramids across `(tile, source)` is embarrassingly parallel and
   is typically the dominant per-level cost. Extracting it lets the
   parallel sweep be tuned once.

The caller chooses the **subset of source images** to feed in — the
entire reconstruction, a spatial neighbourhood of a reference camera,
a temporal window, the keyframes of a video sequence, or any other
slice. Selection is application-specific and out of scope.

## What it produces

For a `SphericalTileRig` of `n` tiles and a slice of `N` source images
(each with intrinsics, world-to-camera rotation, and decoded pixels),
a `PerSphericalTileSourceStack<T>` holds — per tile `t` — the ordered
list of contributing source images. Each contribution is a per-source
image pyramid of warped patches.

Storage is generic over a [`PatchPixel`] type — `u8` (compact, what
NCC / GPU-byte-texture consumers want) and `f32` (autodiff-ready) are
provided. The `u8` → `f32` conversion preserves **range**
(0–255 maps to `0.0–255.0`), not scale, so an `f32` stack is
byte-equivalent to a `u8` stack at level 0 modulo type cost.

**Pyramid sizing.** Let `B = rig.patch_size` (the *base patch size*)
and `L = log2(B) + 1` (the *level count*). For `B = 32`, `L = 6`,
with level sizes `[32, 16, 8, 4, 2, 1]`. Every observation in the
stack has the same `B` and the same `L`, by construction. The
primitive **requires `B` to be a power of two** so every level is a
perfect 2× downsample of the previous one — no fractional sizing, no
special end cases. The constructor's `patch_size` formula
(`ceil(2 · half_fov_rad / arc_per_pixel)`) does not generally produce
a power of two, so callers using this primitive should override after
construction:

```rust
let mut rig = SphericalTileRig::new(&rig_params)?;
rig.set_patch_size(rig.patch_size().next_power_of_two());
let stack = PerSphericalTileSourceStack::<u8>::build_rotation_only(&rig, &sources, &params);
```

`set_patch_size` keeps the rig's tile directions, bases, and half-FOV
unchanged; only the per-tile pixel resolution (and therefore
`tile_camera()` + `atlas_size()`) shifts. The build will return an
error if `rig.patch_size` is not a power of two when called.

**Visibility cull.** A source is considered to contribute to a tile
iff the **tile centre's world direction**, projected through that
source's pose, falls inside the source image (in front of the camera
and within `[0, W) × [0, H)`). Sources whose camera does not see the
tile centre are dropped from the list entirely; any per-direction
algorithm would discard them as a first step anyway.

**Per-pixel valid masks.** Each pyramid level carries its own valid
mask, stored as `Vec<u8>` with values strictly in `{0, 1}` (zero-copy
uploadable to a GPU `Bool` texture or a burn `Tensor<B, _, Bool>`
backing buffer). Level 0's mask comes directly from the warp (1 where
`map_x`, `map_y` are finite and in source bounds, else 0). Each
downsampled level uses the **all-four** rule: a level-L pixel is valid
iff all four level-(L−1) source pixels were valid. This is
conservative — boundaries shrink as resolution decreases — but it
matches what NCC / median / mean-style consumers expect (the
downsampled value reflects only fully-covered source content).

## Layout

Storage is **CSR-flat across all tiles**: each pyramid level holds a
single contiguous patch buffer and a single contiguous valid buffer
spanning every contributing `(tile, source)` row.

```
n_tiles                  (= rig.len())
n_sources                (= sources.len())
total_contrib_rows       (= sum over t of n_contributors(t))
channels                 (= 1, 3, or 4)
base_patch_size          (= rig.patch_size, a power of two)
pyramid_levels           (= log2(base_patch_size) + 1)

CSR row layout (one row per (tile, contributing source)):
  src_id:        Vec<u32>  length = total_contrib_rows
                 the source-list index of each row's source
  tile_id:       Vec<u32>  length = total_contrib_rows
                 the tile index of each row, materialised once
  tile_offsets:  Vec<u32>  length = n_tiles + 1
                 row range for tile t = [tile_offsets[t]..tile_offsets[t + 1])
                 tile_offsets[n_tiles] == total_contrib_rows

Per-level data (one PatchLevel<T> per pyramid level):
  size:          u32      = base_patch_size >> level_idx
  patches:       Vec<T>   length = total_contrib_rows · size² · channels
                 row r occupies [r · size² · C .. (r + 1) · size² · C)
                 row-major within each patch
  valid:         Vec<u8>  length = total_contrib_rows · size²
                 row r occupies [r · size² .. (r + 1) · size²)
                 0 = invalid, 1 = valid (no other values)
```

Within a tile's row range, contributors are sorted by ascending source
index. Within a row, pixels are row-major; within a pixel, channels are
interleaved.

The CSR layout supports three access patterns naturally:

- **Per-tile slice**, via `patches_for_tile(t, l)` /
  `valid_for_tile(t, l)` — what tile-major NCC / consensus consumers
  want.
- **Whole-level buffer**, via `level_patches(l)` /
  `level_valid(l)` — a single `Tensor::from_data` source for autodiff
  / GPU pipelines that build one computational graph per pyramid
  level.
- **Per-row gather**, via `tile_id()` / `src_id()` — segment-reduction
  by tile and gather-by-source, the access pattern joint-optimisation
  workloads want.

## API

```rust
/// Pixel-element storage trait.
pub trait PatchPixel: Copy + Default + Send + Sync + 'static {
    /// Convert a u8 source-image sample into storage type. Preserves
    /// **range** (0–255 maps to `0.0–255.0` for f32), not scale.
    fn from_u8(v: u8) -> Self;

    /// Mean of four storage-typed neighbours. For u8 this is the
    /// round-to-nearest u16 mean `(a + b + c + d + 2) / 4`; for f32
    /// it is exact arithmetic `0.25 · (a + b + c + d)`.
    fn box_avg_4(a: Self, b: Self, c: Self, d: Self) -> Self;
}

impl PatchPixel for u8 { /* … */ }
impl PatchPixel for f32 { /* … */ }

/// Per-pyramid-level CSR-packed storage.
pub struct PatchLevel<T> {
    pub size: u32,
    pub patches: Vec<T>,
    pub valid: Vec<u8>,
}

/// Per-spherical-tile source patch stack with CSR-packed per-level
/// storage. Generic over the pixel storage type `T`.
pub struct PerSphericalTileSourceStack<T: PatchPixel> { /* private */ }

impl<T: PatchPixel> PerSphericalTileSourceStack<T> {
    /// Build a rotation-only stack: every patch is the source's view
    /// of the tile assuming the scene is at infinite radial distance
    /// from the rig centre. Source positions are not used — only the
    /// relative orientation between each source and the rig.
    ///
    /// `sources[i]` is `(src_intrinsics_i, R_src_from_world_i,
    /// image_i)`. Image dtype is u8 with 1, 3, or 4 channels
    /// (consistent across the slice).
    ///
    /// Returns `BuildError::PatchSizeNotPowerOfTwo` if
    /// `rig.patch_size` is not a power of two; the caller is expected
    /// to round up via `rig.set_patch_size(rig.patch_size()
    /// .next_power_of_two())` first.
    pub fn build_rotation_only(
        rig: &SphericalTileRig,
        sources: &[(CameraIntrinsics, RotQuaternion, ImageU8)],
        params: &BuildParams,
    ) -> Result<Self, BuildError>;

    pub fn n_tiles(&self) -> usize;
    pub fn base_patch_size(&self) -> u32;
    pub fn pyramid_levels(&self) -> u32;
    pub fn channels(&self) -> u32;
    pub fn total_contrib_rows(&self) -> usize;

    // ── Per-tile slice accessors ────────────────────────────────────

    /// Source-list indices contributing to tile `t`, ascending. Slice
    /// of the flat `src_id` array.
    pub fn src_indices_for_tile(&self, t: usize) -> &[u32];

    /// Number of contributing sources for tile `t`.
    pub fn n_contributors(&self, t: usize) -> usize;

    /// Tile `t`'s patches at level `l`. Layout: row-major
    /// `n_contributors(t) × size² × channels`, contributors in
    /// `src_indices_for_tile(t)` order. Empty when `n_contributors(t)
    /// == 0`.
    pub fn patches_for_tile(&self, t: usize, l: usize) -> &[T];

    /// Tile `t`'s valid masks at level `l`. Layout: row-major
    /// `n_contributors(t) × size²`, contributors in the same order
    /// as `patches_for_tile`.
    pub fn valid_for_tile(&self, t: usize, l: usize) -> &[u8];

    // ── Whole-stack accessors ───────────────────────────────────────

    /// Per-row source index. Length `total_contrib_rows`.
    pub fn src_id(&self) -> &[u32];

    /// Per-row tile index. Length `total_contrib_rows`. Materialised
    /// at build time so consumers can use it as a gather index without
    /// reconstructing it from `tile_offsets`.
    pub fn tile_id(&self) -> &[u32];

    /// CSR offsets, length `n_tiles + 1`.
    pub fn tile_offsets(&self) -> &[u32];

    /// Pyramid level metadata + buffers as a struct.
    pub fn level(&self, l: usize) -> &PatchLevel<T>;

    /// Whole-level patches buffer at level `l`. Length
    /// `total_contrib_rows · size_l² · channels`.
    pub fn level_patches(&self, l: usize) -> &[T];

    /// Whole-level valid buffer at level `l`. Length
    /// `total_contrib_rows · size_l²`.
    pub fn level_valid(&self, l: usize) -> &[u8];
}

#[derive(Debug, Clone, Default)]
pub struct BuildParams {
    /// Reserved for future parallel-source chunking; currently a no-op.
    pub max_in_flight_sources: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    PatchSizeNotPowerOfTwo(u32),
    MixedSourceChannels { first: u32, offending: u32 },
    UnsupportedChannelCount(u32),
}
```

The Python wrapper exposes the same shape behind a single class
`sfmtool._sfmtool.PerSphericalTileSourceStack`, with pixel storage
selected via a `dtype="uint8" | "float32"` kwarg on
`build_rotation_only`. Per-tile and whole-level accessors return numpy
arrays of the appropriate dtype; `tile_offsets()`, `tile_id()`, and
`src_id()` return `np.uint32` arrays, and `valid_for_tile` /
`level_valid` return `np.bool_` views over the underlying `{0, 1}` u8
buffers.

### Joint photometric refinement consumer

The CSR layout was driven by the joint photometric refinement
workload — per-image colour parameters + per-tile soft cluster
assignment, optimised under L-BFGS with autodiff. The natural sketch:

```rust
// Once, at problem setup:
let stack: PerSphericalTileSourceStack<f32> =
    PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &params)?;
let level = stack.level(level_idx);
let patches: Tensor<B, 4> = Tensor::from_data(
    TensorData::new(level.patches.clone(), [
        stack.total_contrib_rows(), level.size as usize,
        level.size as usize, stack.channels() as usize,
    ]),
    &device,
);
let tile_id_t: Tensor<B, 1, Int> = Tensor::from_data(stack.tile_id(), &device);
let src_id_t:  Tensor<B, 1, Int> = Tensor::from_data(stack.src_id(),  &device);

// Each L-BFGS iteration:
//   colour_params: Tensor<B, 2>  shape [n_sources, 2]  — autodiff leaf
//   per-row colour = colour_params.gather(0, src_id_t)        // [rows, 2]
//   corrected      = apply_colour(patches, per_row_colour)    // [rows, s, s, c]
//   tile_means     = scatter_segment_mean(corrected, tile_id_t, n_tiles)
//   residuals      = corrected - tile_means.gather(0, tile_id_t)
//   loss           = robust(residuals).masked_sum(valid)
//   loss.backward();
//   gradient = colour_params.grad();   // hand to argmin
```

The whole-level patches buffer must be one allocation —
`Tensor::from_data` over per-tile pieces would either require an
allocation per iteration (concatenate into one buffer) or `n_tiles`
separate tensors and a manual segment-loop, defeating the point.

## Algorithm

Let `B = rig.patch_size` (validated to be a power of two on entry —
see `BuildError::PatchSizeNotPowerOfTwo`) and `L = log2(B) + 1`. The
per-tile pinhole is `rig.tile_camera()` directly — no scaling needed,
since the caller has already set `rig.patch_size` to the desired
power-of-two value.

The build is two passes so the SoA buffers can be allocated up front
and written to from parallel source-tasks at known positions, with no
per-tile locks and no thread-local-then-merge phase.

### Pass 1: visibility + CSR offsets

Sequentially over sources, in source-index order: for every tile
`t = 0..n-1`, compute `d_src_t = R_src_from_world_i · rig.direction(t)`,
project through `src_intrinsics_i`, and record
`visibility[t][i] = (finite & in_bounds & z > 0)` into a packed bitmap.
`O(n · N)` total.

From the matrix, derive:

- `n_contributors[t] = popcount(visibility[t][:])` for each tile.
- `tile_offsets`: prefix sum of `n_contributors`, length `n_tiles + 1`.
  `tile_offsets[n_tiles] == total_contrib_rows`.
- `src_id`: flat `Vec<u32>` of length `total_contrib_rows`. Filled by
  iterating tiles in order; for each tile, append source indices in
  ascending order for which `visibility[t][i] == true`.
- `tile_id`: flat `Vec<u32>` of length `total_contrib_rows`. For each
  tile `t`, fill `tile_id[tile_offsets[t]..tile_offsets[t + 1]]` with
  `t`.
- `position[t][i]`: the contributor's offset within tile `t`'s row
  range, used in Pass 3 to compute the global row index. Defined only
  where `visibility[t][i]`.

### Pass 2: per-level CSR allocation

For each pyramid level `l`, allocate one buffer of each kind:

```
levels[l].size    = base_patch_size >> l
levels[l].patches = vec![T::default(); total_contrib_rows · size² · channels]
levels[l].valid   = vec![0u8;          total_contrib_rows · size²]
```

Total memory across all levels is dominated by level 0; the
geometric series gives `≈ 4/3 · total_contrib_rows · B² · channels`
elements for the full pyramid (`× sizeof(T)` bytes for patches; +
`4/3 · total_contrib_rows · B²` bytes for valid).

### Pass 3: parallel per-source warp + pyramid write

Sequential outer loop over sources; rayon-parallel inner sweep over
each source's kept tiles. For each source `i` and every tile `t` with
`visibility[t][i] == true`:

1. Form `R_src_from_tile = R_src_from_world_i · rig.tile_rotation(t)`
   (the latter returns `R_world_from_tile`).
2. Build `WarpMap::from_cameras_with_rotation(src=src_intrinsics_i,
   dst=rig.tile_camera(), rot_src_from_dst=R_src_from_tile)`.
3. Apply `WarpMap::remap_bilinear` to `image_i` → a `B × B × C` u8
   patch in a per-task scratch buffer; the level-0 valid mask is the
   warp map's own `is_valid` bit per pixel.
4. **Convert + write level 0.** Compute the global row index
   `r = tile_offsets[t] + position[t][i]`. For each pixel and channel,
   convert the u8 sample to `T` via `T::from_u8` and copy into
   `levels[0].patches[r · B² · C ..]`; write `is_valid as u8` into
   `levels[0].valid[r · B² ..]`.
5. **Downsample for this source through every level.** For
   `l = 1..L`, with `s = B >> l` and `s_prev = s · 2`:
   - Box-filter downsample the previous level's patch via
     `T::box_avg_4` per channel.
   - Valid: the level-`l` pixel `(u, v)` is true iff all four
     level-`l − 1` source pixels were true (all-four rule), expressed
     as a bitwise AND of u8 `{0, 1}` values.
   - Write into `levels[l].patches[r · s² · C ..]` and
     `levels[l].valid[r · s² ..]`.

Doing the downsample in-line (right after the source's level-0 write)
keeps the source's patch in cache across all `L` levels and avoids a
second pass over the same memory.

**Race-free without atomics.** Two parallel tasks for a fixed source
`i` write to distinct global rows (each kept tile `t` produces a
unique `r = tile_offsets[t] + position[t][i]` because both
`tile_offsets[t]` is unique per `t` and `position[t][i]` is fixed per
`(t, i)`). The build expresses this with a small `SoaWriter` struct
holding raw `*mut T` / `*mut u8` pointers into each level's CSR
buffers; the SAFETY contract requires distinct `(level, row)` writes
per concurrent caller, satisfied by construction. The base-level
warp is built fresh for each kept `(source, tile)` pair —
typically `~0.1 · n · N` warps total at phone-FOV inputs.

### Why per-tile warps instead of one shared atlas warp

With `rig.patch_size` set to a power of two by the caller, both
`rig.warp_to_atlas_with_rotation` and per-tile
`WarpMap::from_cameras_with_rotation` produce a power-of-two base
patch. Per-tile is faster on the typical phone-FOV cull rate:

| Approach | Per-source warp pixels touched | Notes |
|---|---|---|
| **Per-tile warp at `B × B`** (this spec) | `n_kept · B²` | Skips unkept tiles entirely; needs `rig.tile_rotation(t)` (already shipped). |
| Atlas warp + slice | `atlas_w · atlas_h ≈ n · B² · 1.05` | Pays per-pixel work for every atlas slot regardless of visibility; reuses the already-shipped, already-validated atlas constructor. |

Concrete numbers at `n = 1280`, `B = 32`, visibility `f = 0.1`:

- Per-tile (this spec): `0.1 · 1280 · 32² = 131K` pixels per source.
- Atlas: `1280 · 32² · 1.05 = 1.38M` pixels per source.

Per-tile is ~10× cheaper at typical phone-FOV cull rates; the gap
narrows as `f` approaches 1 (a fully-covered 360° rig set), where
atlas wins on warp-construction overhead.

## Loop order and parallelism

Pass 1 (visibility) runs sequentially in source-index order so
`src_id` lands ascending within each tile's row range without a merge
step. Pass 3 (warp + pyramid write) carries the bulk of the per-source
work as **sequential outer over sources, parallel inner over that
source's kept tiles** (rayon `par_iter` on the tile range). Each
inner tile-task only writes to its own unique global row, so the
writes are race-free without atomics or locks.

A future memory-bounded variant could go source-major (parallel outer
over chunks of sources, each writing into per-tile CSR slots) when
sources are streamed in rather than handed in pre-loaded. That path
needs lock-free routing of per-`(tile, source)` slot references; the
`BuildParams::max_in_flight_sources` knob is reserved for that
extension and is a no-op today.

### Memory ceiling: `max_in_flight_sources` (reserved)

With the current sequential-source outer loop, only one source-image
is "active" at a time, so the resident source-image working set is
just one `ImageU8` regardless of the rayon thread count.
`BuildParams::max_in_flight_sources` is reserved for the future
source-major parallel variant. When that lands the knob will cap how
many source-images are simultaneously held by parallel source-tasks:
`None` (default) lets rayon decide; small integers (2–8) bound peak
memory when running on many-core boxes against high-resolution
captures, or when running multiple builds concurrently. Setting the
knob today has no effect.

## Memory

For `n` tiles, `N` sources, average `f` fraction of tiles each source
sees (`f` is determined by per-source FOV — roughly `0.10` for
phone-FOV captures, up to `0.50+` for wide-angle / panoramic captures
or dense 360° rigs), and base patch size `B`:

```
total_contrib_rows  ≈ f · n · N
elements per pyramid   =  B² + (B/2)² + (B/4)² + … + 1
                       ≈  4/3 · B²       (the geometric-series limit)

T = u8:
  total patch bytes (RGB)  ≈ f · n · N · 4 · B² · 3
T = f32:
  total patch bytes (RGB)  ≈ f · n · N · 4 · B² · 3 · 4   (4× u8)

valid bytes (T-independent)
                          ≈ f · n · N · 4 · B²            (1 byte/pixel)
```

Concrete sizes at `B = 32` (the caller has called `set_patch_size(32)`
on the rig; matches `n ∈ {320, 1280}` and the typical
`W ∈ {256 … 1024}` range, where the constructor's natural
`patch_size` lands around 21):

| n | N | f | u8 stack RGB bytes | f32 stack RGB bytes |
|---|---|---|---|---|
| 1280 | 26  | 0.10 | 14 MB  | 56 MB  |
| 1280 | 17  | 0.10 | 9 MB   | 36 MB  |
| 1280 | 124 | 0.50 | 325 MB | 1.3 GB |

The `f32` peak (`n = 1280, N = 124, f = 0.50`) is the load-bearing
constraint for the photometric refinement consumer's choice of how
many tiles / levels to materialise simultaneously.

## Testing

Rust:

- **Empty cull.** A rig with one source whose pose puts the rig
  centre out of view: every tile has `n_contributors(t) == 0`, all
  per-tile slices are empty, all whole-level buffers are empty.
- **Partial cull.** A 60° pinhole covers ~1/8 of the sphere; expect a
  non-trivial mix of kept and dropped tiles, every kept tile's
  `src_indices_for_tile` agrees with hand-computed
  centre-direction-projection rule.
- **Centre-cull symmetry.** Two sources with opposite look directions
  contribute to disjoint tile sets (no tile has both in
  `src_indices_for_tile`).
- **Power-of-two validation.** Calling `build_rotation_only` with a
  rig whose `patch_size` is not a power of two returns
  `Err(BuildError::PatchSizeNotPowerOfTwo(p))`. After
  `rig.set_patch_size(p.next_power_of_two())` the same call succeeds.
- **Pyramid level count and sizes.** For
  `rig.patch_size ∈ {8, 16, 32, 64, 128}`, `base_patch_size ==
  rig.patch_size`, `pyramid_levels == log2(base_patch_size) + 1`,
  every level's `size` is `base_patch_size >> l`, ending at 1.
- **CSR buffer sizing.** `level.patches.len() == total_contrib_rows
  · size² · channels`, `level.valid.len() == total_contrib_rows
  · size²`. The sum of per-tile slice lengths over all tiles equals
  the whole-level buffer length at every level.
- **CSR offsets well-formed.** `tile_offsets[0] == 0`,
  `tile_offsets[n_tiles] == total_contrib_rows`, monotone
  non-decreasing, `tile_offsets[t + 1] - tile_offsets[t] ==
  n_contributors(t)`.
- **`tile_id` / `src_id` consistency.** `tile_id[r] == t` iff
  `tile_offsets[t] <= r < tile_offsets[t + 1]`; `src_id` slice for
  tile `t` is strictly ascending and equals `src_indices_for_tile(t)`.
- **`valid` strictly `{0, 1}`.** Every byte of every level's
  `level_valid(l)` is in `{0, 1}` (no other values).
- **Pyramid downsample correctness.** Constant-colour input → every
  level reproduces the same constant in every kept (tile, source)
  slot; an invalid pixel propagates through the all-four AND rule
  (level-`L` pixel at `(x/2^L, y/2^L)` is invalid; the final 1×1
  entry is invalid).
- **Visibility = base-warp validity at the centre pixel.** A tile has
  source `i` in `src_indices_for_tile(t)` iff the level-0 warp's
  `map_x`, `map_y` at the patch centre is in-bounds for that source.
- **Round-trip vs a hand-rolled per-tile warp.** For each kept
  `(source, tile)`, compute the equivalent per-tile warp directly
  (`WarpMap::from_cameras_with_rotation` with the matching tile
  pinhole + `R_src_from_tile`), apply `remap_bilinear`, and assert
  byte-equality with the corresponding row of `level 0`'s patches.
- **Parallel-build determinism.** Two builds of the same `(rig,
  sources)` — one with `max_in_flight_sources = Some(1)`, one with
  the default — produce byte-equal `tile_offsets`, `src_id`,
  `tile_id`, and per-level patches/valid buffers.
- **u8 ↔ f32 level-0 byte equivalence.** Build the same `(rig,
  sources)` twice, once with `T = u8` and once with `T = f32`; assert
  that `f32_stack.level_patches(0)[i] == u8_stack.level_patches(0)[i]
  as f32` for every i.
- **u8 ↔ f32 deviation bound.** Beyond level 0 the rounded u8 mean
  and exact f32 mean diverge; bound the per-element deviation by
  `0.5 · level` (the round-to-nearest u8 mean adds at most 0.5
  rounding per level relative to the exact value).
- **f32 constant-colour means are exact.** With f32 storage, a
  constant input pyramid stays exactly constant at every level (no
  rounding drift).

Python (PyO3):

- **Smoke test on a real reconstruction.** Build a stack from a
  bundled dataset (`test-data/images/seoul_bull_sculpture` via the
  `Scene` loader) at `n = 320`. Assert pyramid sizing, per-tile
  per-level buffer shapes, and that every entry in
  `src_indices_for_tile` is a valid index into the input slice.
- **Pyramid round-trip in pure Python.** For one source and one
  kept tile, build the equivalent per-tile warp + `remap_bilinear`
  in Python, downsample with all-four valid, and assert byte-equality
  with the corresponding source-row at every level.
- **CSR offsets / ids match per-tile slices.** `tile_id[r] == t`
  inside tile `t`'s row range; `src_id[range]` matches
  `src_indices_for_tile(t)` element-wise.
- **Whole-level buffers match per-tile concatenation.**
  `level_patches(l)` equals the per-tile `patches_for_tile(t, l)`
  slices concatenated in tile order; same for `level_valid` /
  `valid_for_tile`.
- **dtype dispatch.** `dtype="float32"` produces a stack whose
  `level_patches(0)` is an `np.float32` array byte-equal (as
  `astype(np.float32)`) to the same stack built with `dtype="uint8"`.
  An invalid dtype (`"float64"`) raises `ValueError`.

## Non-goals

- **No depth handling.** The rotation-only build assumes scene
  content at infinity for the warp. A pose-aware extension lands when
  an application needs scene-finite-depth warps; until then,
  rotation-only is the contract.
- **No image decoding.** Caller decodes JPGs / EXRs and passes
  `ImageU8`s. Keeps the primitive pure-compute and lets the Python
  wrapper choose between eager and lazy decoding.
- **No source selection.** The caller picks which subset of source
  images to feed in; this primitive does not implement spatial,
  temporal, or angular subset selection rules.
- **No per-source colour normalisation.** Exposure / white-balance
  alignment is upstream of this primitive (caller pre-normalises) or
  downstream (consumer applies its own correction at aggregation
  time). The joint photometric refinement workload is the canonical
  downstream consumer.
- **No occlusion-aware visibility.** Centre-direction in-FOV is the
  only filter. Layered occlusion reasoning (e.g. using SfM 3D points
  to test whether a source's line-of-sight to a tile is blocked) is
  an application-level extension built on top of the observation
  list.
- **No per-tile pyramid depth knob.** Every tile's pyramid has the
  same `L = log2(B) + 1` levels by design — picking a subset would
  fragment the type and complicate downstream consumers. If a
  consumer only wants level 0, it reads `level_patches(0)`; the
  unused levels add ~33% memory but are simple box-filter passes,
  not significant compute.
- **No mid-build dtype switching.** A stack's `T` is fixed at
  construction. Cast helpers
  (`stack_u8.cast::<f32>() -> PerSphericalTileSourceStack<f32>`) are
  a follow-up if a consumer needs both representations from one
  build.
- **No `bool` storage for `valid`.** `Vec<u8>` with values strictly
  in `{0, 1}` is the contract — `Vec<bool>` is the same byte cost on
  CPU but isn't transmute-safe to `&[u8]` in Rust, so `u8` makes the
  zero-copy GPU / autodiff upload path lossless.

### Pose-aware variant

A `build_with_pose` sibling (taking each source's full
`RigidTransform` and either a single scene-radius scalar or a per-tile
inverse-depth array) lands when an application needs
scene-finite-depth warps rather than infinity-equivalent
rotation-only warps. The output shape is identical — same pyramid
structure, same valid-mask rule, same CSR layout; only the per-source
warp construction differs.

## Status / dependencies

Depends on:

- `SphericalTileRig` (`specs/core/spherical-tiles-rig.md`) — DONE.
  - `direction(t)`, `tile_rotation(t)`, `tile_camera()`,
    `patch_size`. All shipped.
- `WarpMap::from_cameras_with_rotation` and `WarpMap::remap_bilinear`
  (`specs/drafts/warpmap-pose-extension.md`) — DONE.

Independently testable; no external state beyond the rig + sources.

### Concrete consumers in this codebase

- The joint photometric refinement workload (per-image colour + soft
  cluster assignment, L-BFGS via `argmin` + `burn-autodiff`) consumes
  the `f32` build path; spec lands separately when implementation
  starts.
- New direction-space algorithms (panorama compositing, multi-source
  alpha estimation, GPU multi-scale shaders) consume the same
  observation list with no changes to this primitive.
