# Per-spherical-tile source patch stack

**Status:** Implemented in
`crates/sfmtool-core/src/per_spherical_tile_source_stack.rs` (the
rotation-only build) and exposed to Python as
`sfmtool._sfmtool.PerSphericalTileSourceStack`. The pose-aware variant
described under "Pose-aware variant" is still future work.

The implemented build runs the outer source loop sequentially and
parallelises across the source's kept tiles via rayon — each tile-task
writes to its own SoA buffer at a unique `pos` slot, so the writes are
race-free without atomics or locks. The
`BuildParams::max_in_flight_sources` knob is reserved for future
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
   code reads `tile(t).levels[L_idx]`, pulls the contiguous SoA
   buffer (or any source's slice within it), and applies its own
   logic — no ambiguity about which sources, in which order, with
   which valid masks, at which scale.
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
a `PerSphericalTileSourceStack` holds — per tile `t` — the ordered list of
contributing source images. Each contribution is a per-source image
pyramid of warped patches.

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
let stack = PerSphericalTileSourceStack::build_rotation_only(&rig, &sources, &params);
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
mask. Level 0's mask comes directly from the warp (true where
`map_x`, `map_y` are finite and in source bounds). Each downsampled
level uses the **all-four** rule: a level-L pixel is valid iff all
four level-(L−1) source pixels were valid. This is conservative —
boundaries shrink as resolution decreases — but it matches what
NCC / median / mean-style consumers expect (the downsampled value
reflects only fully-covered source content).

```
tile 0 → [(src=3, pyramid_3), (src=7, pyramid_7), …]
tile 1 → src_indices=[0, 3], levels=[lvl_0, lvl_1, …, lvl_L-1]
…
tile n-1 → src_indices=[14], levels=[lvl_0, lvl_1, …, lvl_L-1]
```

**SoA layout per (tile, level).** For tile `t` with `k =
n_contributors[t]` and pyramid level `L_idx` of side `s = B >> L_idx`,
the level holds `k` patches concatenated end-to-end in source-index
order:

```
tiles[t].levels[L_idx].patches  =  src_0_patch || src_1_patch || … || src_(k-1)_patch
                                   (length k · s² · C, row-major within each patch)
tiles[t].levels[L_idx].valid    =  src_0_valid || src_1_valid || … || src_(k-1)_valid
                                   (length k · s²)
```

`tiles[t].src_indices` (length `k`) names which input-slice index each
contributor came from. The same source order is used at every pyramid
level, so a downstream consumer that wants source `i`'s patch at level
`L_idx` can read `tiles[t].levels[L_idx].patches[pos · s² · C ..
(pos + 1) · s² · C]` where `pos = tiles[t].src_indices.iter().position(
|&x| x == i).unwrap()`. Per-(tile, level) iteration is the dominant
access pattern (NCC across all sources at one scale, GPU upload of a
single buffer per scale, etc.) and the SoA layout makes that one
contiguous slice.

## API

```rust
/// All contributing sources' patches for one tile at one pyramid level,
/// laid out struct-of-arrays so per-(tile, level) consumers get a
/// single contiguous buffer.
pub struct SphericalTilePatchLevel {
    /// Side length of this level in pixels (= `base_patch_size >> L_idx`).
    pub size: u32,
    /// Number of contributing sources (= `tiles[t].src_indices.len()`).
    pub n_contributors: u32,
    /// Channel count (matches the input image: 1, 3, or 4).
    pub channels: u32,
    /// All sources' patches concatenated in `src_indices` order. Each
    /// per-source patch is row-major `size × size × channels` u8;
    /// total length = `n_contributors · size · size · channels`.
    pub patches: Vec<u8>,
    /// All sources' valid masks, same SoA layout as `patches`. Level 0
    /// masks come from the warp's in-bounds test; level L > 0 uses the
    /// all-four rule (true iff every one of the four level-(L−1)
    /// source pixels was true).
    /// Length = `n_contributors · size · size`.
    pub valid: Vec<bool>,
}

/// All data for one spherical tile.
pub struct SphericalTileSourceStack {
    /// Source-list indices of this tile's contributors, in the order
    /// their patches appear in every level's SoA buffers. Sorted
    /// ascending by source index. Empty if no source sees the tile
    /// centre.
    pub src_indices: Vec<u32>,
    /// Pyramid levels in order: `levels[0]` is the base
    /// (`base_patch_size × base_patch_size`), `levels[L-1]` is `1 × 1`.
    /// Every level holds `src_indices.len()` patches in matching order.
    pub levels: Vec<SphericalTilePatchLevel>,
}

pub struct PerSphericalTileSourceStack {
    /// Tile count (mirrors `rig.len()`).
    pub n_tiles: usize,
    /// Side length of level 0 across every tile. Equals `rig.patch_size`
    /// at the time of `build_rotation_only`.
    pub base_patch_size: u32,
    /// Number of pyramid levels (= `log2(base_patch_size) + 1`).
    pub pyramid_levels: u32,
    /// Per-tile SoA bundles, length `n_tiles`.
    tiles: Vec<SphericalTileSourceStack>,
}

impl PerSphericalTileSourceStack {
    /// Build a rotation-only stack: every patch is the source's view of
    /// the tile assuming the scene is at infinite radial distance from
    /// the rig centre. Source positions are not used — only the
    /// relative orientation between each source and the rig. This is
    /// the right primitive when the per-direction question is
    /// orientation-invariant (sky / distant content / appearance
    /// agreement); applications that need to reason about finite-depth
    /// scene geometry want the pose-aware sibling.
    ///
    /// `sources[i]` is `(src_intrinsics_i, R_src_from_world_i, image_i)`.
    /// Image dtype is u8 with 1, 3, or 4 channels (consistent across
    /// the slice).
    ///
    /// Returns an error if `rig.patch_size` is not a power of two.
    /// The caller is expected to round up via `rig.set_patch_size(
    /// rig.patch_size().next_power_of_two())` before calling.
    pub fn build_rotation_only(
        rig: &SphericalTileRig,
        sources: &[(CameraIntrinsics, RotQuaternion, ImageU8)],
        params: &BuildParams,
    ) -> Result<Self, BuildError>;

    pub fn n_tiles(&self) -> usize;
    pub fn base_patch_size(&self) -> u32;
    pub fn pyramid_levels(&self) -> u32;

    /// All data for tile `idx`. `tile(idx).src_indices` lists the
    /// contributors; `tile(idx).levels[L_idx]` is the SoA bundle at
    /// pyramid level `L_idx`.
    pub fn tile(&self, idx: usize) -> &SphericalTileSourceStack;

    /// Number of contributing sources for tile `idx`
    /// (= `tile(idx).src_indices.len()`).
    pub fn n_contributors(&self, idx: usize) -> usize;
}

#[derive(Debug, Clone, Default)]
pub struct BuildParams {
    /// Cap on simultaneously-in-flight source tasks during the parallel
    /// outer loop. `None` (default) lets rayon's work-stealing pick — fine
    /// for small `n × base_patch_size² × num_threads`. Set to a finite
    /// value (typically 2–8) to bound peak memory when many large source
    /// images are held simultaneously; sources are processed in chunks
    /// of this size, with each chunk's sources rayon-parallel internally
    /// and source-image state released between chunks.
    pub max_in_flight_sources: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    /// `rig.patch_size()` is not a power of two.
    PatchSizeNotPowerOfTwo(u32),
    /// Sources disagree on channel count. The whole slice must use the
    /// same value.
    MixedSourceChannels { first: u32, offending: u32 },
    /// A source has an unsupported channel count (only 1, 3, 4 are
    /// accepted by the remap kernels).
    UnsupportedChannelCount(u32),
}
```

The Python wrapper exposes the same shape, with `sources` accepting an
iterator that yields `(CameraIntrinsics, RotQuaternion, np.ndarray)` so
callers can decode JPGs lazily and not hold all source images in memory
simultaneously.

### Adapters for downstream shapes

The SoA layout already gives most consumers what they want directly:
`tiles[t].levels[L_idx].patches` is the contiguous "all sources at
this scale" buffer that `multiview-consensus-cluster`'s `PatchStack`
type expects (modulo wrapping into a typed view), and likewise for
GPU upload. Adapters live as thin methods
(`fn as_patch_stack(&self, tile_idx: usize, level: usize) ->
PatchStack`, etc.) added next to the canonical `tile(idx)` accessor
as consumers land — pure aliasing of the underlying buffer, no copy.

### Pose-aware variant

A `build_with_pose` sibling (taking each source's full `RigidTransform`
and either a single scene-radius scalar or a per-tile inverse-depth
array) lands when an application needs scene-finite-depth warps rather
than infinity-equivalent rotation-only warps. The output shape is
identical — same pyramid structure, same valid-mask rule; only the
per-source warp construction differs.

## Algorithm

Let `B = rig.patch_size` (validated to be a power of two on entry —
see `BuildError::PatchSizeNotPowerOfTwo`) and `L = log2(B) + 1`. The
per-tile pinhole is `rig.tile_camera()` directly — no scaling needed,
since the caller has already set `rig.patch_size` to the desired
power-of-two value.

The build is two passes so the SoA buffers can be allocated up front
and written to from parallel source-tasks at known positions, with no
per-tile locks and no thread-local-then-merge phase.

### Pass 1: visibility matrix

Sequentially over sources, in source-index order: for every tile
`t = 0..n-1`, compute `d_src_t = R_src_from_world_i · rig.direction(t)`,
project through `src_intrinsics_i`, and record
`visibility[t][i] = (finite & in_bounds & z > 0)` into a packed bitmap.
`O(n · N)` total.

From the matrix, derive:

- `n_contributors[t] = popcount(visibility[t][:])` for each tile.
- `src_indices[t] = [i for i in 0..N if visibility[t][i]]` (sorted
  ascending, by construction).
- `position[t][i]` (defined only for `(t, i)` with `visibility[t][i]`)
  = the contributor's offset within tile `t`'s `src_indices` list,
  i.e. the popcount of `visibility[t][0..i]`.

### Pass 2: per-tile SoA allocation

For each tile `t`, allocate the per-level SoA buffers:

```
tiles[t].levels[L_idx].patches  =  vec![0u8;  n_contributors[t] · s² · C]
tiles[t].levels[L_idx].valid    =  vec![false; n_contributors[t] · s²]
```

where `s = B >> L_idx` and `C` is the input channel count. Total
per-tile bytes are dominated by level 0; the geometric series gives
`≈ 4/3 · n_contributors[t] · B² · C` for the full pyramid.

### Pass 3: parallel per-source warp + pyramid write

Process sources via rayon (`par_iter`, with the
`max_in_flight_sources` chunking from the parallelism section
applied). For each source `i` and every tile `t` with
`visibility[t][i] == true`:

1. Form `R_src_from_tile = R_src_from_world_i · rig.tile_rotation(t)`
   (the latter returns `R_world_from_tile`).
2. Build `WarpMap::from_cameras_with_rotation(src=src_intrinsics_i,
   dst=rig.tile_camera(), rot_src_from_dst=R_src_from_tile)`.
3. Apply `WarpMap::remap_bilinear` to `image_i` → a `B × B × C` patch
   in a per-task scratch buffer; the level-0 valid mask is the warp
   map's own `is_valid` bit per pixel (`WarpMap` already encodes
   "finite & in source bounds" by writing `(NaN, NaN)` into out-of-
   range pixels, so a single `!is_nan(map_x)` check is sufficient).
4. **Direct write to the SoA slot.** Let `pos = position[t][i]` and
   `s_0 = B`. Copy the patch into
   `tiles[t].levels[0].patches[pos · s_0² · C .. (pos + 1) · s_0² · C]`
   and the valid mask into the analogous slice of
   `tiles[t].levels[0].valid`.
5. **Downsample for this source through every level.** For
   `L_idx = 1 .. L`, with `s = B >> L_idx` and `s_prev = s · 2`:
   - Read the source's level-`L_idx − 1` patch from
     `tiles[t].levels[L_idx - 1].patches[pos · s_prev² · C .. ]`.
   - Each level-`L_idx` pixel `(u, v)` = per-channel u8 mean of the
     four level-`L_idx − 1` pixels at `(2u, 2v)`, `(2u+1, 2v)`,
     `(2u, 2v+1)`, `(2u+1, 2v+1)`.
   - Write into
     `tiles[t].levels[L_idx].patches[pos · s² · C .. (pos + 1) · s² · C]`.
   - Valid: the level-`L_idx` pixel `(u, v)` is true iff all four
     level-`L_idx − 1` source pixels were true (all-four rule); write
     into the corresponding valid slice.

Doing the downsample in-line (right after the source's level-0 write)
keeps the source's patch in cache across all `L` levels and avoids a
second pass over the same memory.

**No locking, no merging.** Two source-tasks writing to the same
tile's buffers touch disjoint slices (`pos_a ≠ pos_b` since each
contributor has a unique position in `src_indices`), so the parallel
writes are race-free without atomics. The base-level warp is built
fresh for each kept `(source, tile)` pair — typically `~0.1 · n · N`
warps total at phone-FOV inputs.

### Why per-tile warps instead of one shared atlas warp

With `rig.patch_size` set to a power of two by the caller, both
`rig.warp_to_atlas_with_rotation` and per-tile
`WarpMap::from_cameras_with_rotation` produce a power-of-two base
patch — the atlas approach by slicing `B × B` blocks out of the
shared atlas, the per-tile approach by warping directly into a `B × B`
target. Both are correct; the choice is about wasted work on tiles
that no source sees.

| Approach | Per-source warp pixels touched | Notes |
|---|---|---|
| **Per-tile warp at `B × B`** (this spec) | `n_kept · B²` | Skips unkept tiles entirely; needs `rig.tile_rotation(t)` (already shipped). |
| Atlas warp + slice | `atlas_w · atlas_h ≈ n · B² · 1.05` (5% trailing) | Pays per-pixel work for every atlas slot regardless of visibility; reuses the already-shipped, already-validated atlas constructor. |

Concrete numbers at `n = 1280`, `B = 32`, visibility `f = 0.1`:

- Per-tile (this spec): `0.1 · 1280 · 32² = 131K` pixels per source.
- Atlas: `1280 · 32² · 1.05 = 1.38M` pixels per source.

Per-tile is ~10× cheaper at typical phone-FOV cull rates; the gap
narrows as `f` approaches 1 (a fully-covered 360° rig set), where
atlas wins on warp-construction overhead. For dense panoramic
captures with high `f` the atlas pattern would be worth a future
benchmark; v1 picks per-tile because the cull-savings dominate on
every dataset we've measured so far.

## Loop order and parallelism

Pass 1 (visibility) runs sequentially in source-index order so
`src_indices[t]` lands ascending without a merge step. Pass 3 (warp +
pyramid write) is what carries the bulk of the per-source work.

The implemented v1 inverts the spec's original source-major plan and
runs Pass 3 as **sequential outer over sources, parallel inner over
that source's kept tiles** (rayon `par_iter_mut` on `tiles`). Each
inner tile-task only writes to its own tile's SoA buffer at the unique
`pos` slot for the current source, so the writes are race-free without
atomics, locks, or `unsafe` slice splitting. With all `N` source
images already in memory (the build's input), there is nothing to
amortise across kept tiles of a single source — the loaded image is
already resident and the outer-loop choice is free.

A future memory-bounded variant could go source-major (parallel outer
over chunks of sources, each writing into per-tile SoA slots) when
sources are streamed in rather than handed in pre-loaded. That path
needs lock-free routing of per-`(tile, source)` slot references; the
`BuildParams::max_in_flight_sources` knob is reserved for that
extension and is a no-op in v1.

### Memory ceiling: `max_in_flight_sources` (reserved)

Rayon's default work-stealing puts roughly `num_threads` source-tasks
in flight, each holding an `ImageU8` (the full source image, e.g.
~25 MB for a 2160 × 3840 RGB capture) plus its per-tile working set
(`n_kept · B² · (3 + 8)` bytes for patch + map data, well under
1 MB at typical configurations). With v1's tile-major Pass 3 only
*one* source is "active" at a time, so the resident source-image
working set is just one `ImageU8` regardless of `num_threads`.

`BuildParams::max_in_flight_sources` is reserved for the future
source-major parallel variant. When that lands the knob will cap
how many source-images are simultaneously held by parallel
source-tasks: `None` (default) lets rayon decide; small integers
(2–8) bound peak memory when running on many-core boxes against
high-resolution captures, or when running multiple builds
concurrently. Setting the knob today has no effect — the build
is sequential over sources by construction.

## Memory

For `n` tiles, `N` sources, average `f` fraction of tiles each source
sees (`f` is determined by per-source FOV — roughly `0.10` for
phone-FOV captures, up to `0.50+` for wide-angle / panoramic captures
or dense 360° rigs), and base patch size `B`:

```
total observations  ≈ f · n · N
bytes per pyramid (RGB)  =  3 · (B² + (B/2)² + (B/4)² + … + 1)
                         =  3 · B² · (1 − (1/4)^L) / (1 − 1/4)
                         ≈  4 · B²       (the geometric-series limit)
total bytes (RGB)   ≈ f · n · N · 4 · B²
plus valid bits     ≈ f · n · N · 4 · B² / 24   (~4% extra)
```

Concrete sizes at `B = 32` (the caller has called `set_patch_size(32)`
on the rig; matches `n ∈ {320, 1280}` and the typical
`W ∈ {256 … 1024}` range, where the constructor's natural
`patch_size` lands around 21):

| n | N | f | Stack RGB bytes |
|---|---|---|---|
| 1280 | 26  | 0.10 | 14 MB  |
| 1280 | 17  | 0.10 | 9 MB   |
| 1280 | 124 | 0.50 | 325 MB |

The 4× ratio between pyramid bytes and base-only bytes is the cost
of carrying every level. All three configurations remain in-RAM at
the largest planned `n`; the 124-source panoramic case is the
peak-memory scenario and the one where `max_in_flight_sources`
matters most for memory headroom during the build itself (peak adds
the in-flight source images on top of the resident stack).

## Testing

Rust:

- **Empty cull.** A rig at the origin with `n = 80` and a single
  source whose pose looks straight up: roughly 1/6 of the tiles (the
  upper hemisphere intersected with the source's FOV) have
  non-empty `tile(t).src_indices`; the rest have empty
  `src_indices` and zero-length level buffers.
- **Centre-cull symmetry.** Two sources with opposite look directions
  contribute to disjoint tile sets (no tile has both in
  `src_indices`).
- **Power-of-two validation.** Calling `build_rotation_only` with a
  rig whose `patch_size` is not a power of two (e.g. the
  constructor's natural value of 21) returns
  `Err(BuildError::PatchSizeNotPowerOfTwo(21))`. After
  `rig.set_patch_size(32)` the same call succeeds.
- **Pyramid level count and sizes.** For
  `rig.patch_size ∈ {8, 16, 32, 64, 128}`, `base_patch_size ==
  rig.patch_size`, `pyramid_levels == log2(base_patch_size) + 1`,
  every level's `size` is `base_patch_size >> L_idx`, ending at 1.
- **SoA buffer sizing.** For every tile `t` and every level `L_idx`,
  `levels[L_idx].patches.len() == n_contributors[t] · size² · channels`
  and `levels[L_idx].valid.len() == n_contributors[t] · size²`.
- **`src_indices` ordering.** `tile(t).src_indices` is strictly
  ascending, and every entry indexes a source whose centre-projection
  fell inside the source image.
- **Pyramid downsample correctness.** Synthetic level-0 patches
  written into a tile's SoA slot:
  - Constant colour → every level reproduces the same constant in
    that source's slice of every level's buffer.
  - Linear gradient `c(x, y) = ax + by + c` → level-L value at
    `(u, v)` equals the level-0 mean over the corresponding
    `2^L × 2^L` block (within u8 rounding).
- **All-four valid propagation.** With a level-0 mask slice
  containing a single invalid pixel at `(x, y)`, the same source's
  level-1 mask slice has the invalid pixel at `(x/2, y/2)`; level-2
  at `(x/4, y/4)`; …; the final 1×1 entry for that source is
  invalid.
- **Visibility = base-warp validity at the centre pixel.** A tile
  has source `i` in `tile(t).src_indices` iff the level-0 warp's
  `map_x`, `map_y` at the patch centre is in-bounds for that source.
- **Round-trip vs a hand-rolled per-tile warp.** For each kept
  `(source, tile)`, compute the equivalent per-tile warp directly
  (`WarpMap::from_cameras_with_rotation` with the matching tile
  pinhole + `R_src_from_tile`), apply `remap_bilinear`, and assert
  byte-equality with the corresponding `pos · B² · C` slice of
  `tile(t).levels[0].patches`. This is the load-bearing correctness
  check.
- **Parallel-build determinism.** Two builds of the same `(rig,
  sources)` — one with `max_in_flight_sources = Some(1)`, one with
  the default — produce byte-equal `tile(t)` results across every
  tile (same `src_indices`, byte-equal SoA buffers at every level).

Python (PyO3):

- **Smoke test on a real reconstruction.** Build a stack from a
  bundled dataset (e.g. `test-data/images/seoul_bull_sculpture` via
  the `Scene` loader the existing test scripts use) at `n = 320`.
  Assert `pyramid_levels == log2(base_patch_size) + 1`; per-tile
  per-level buffer lengths match the SoA-sizing rule; every entry
  in `src_indices` is a valid index into the input slice.
- **Pyramid round-trip in pure Python.** For one source and one
  kept tile, build the equivalent per-tile warp + `remap_bilinear`
  in Python, downsample with all-four valid, and assert byte-equality
  with the corresponding source-slice of every level's SoA buffer.

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
  time).
- **No occlusion-aware visibility.** Centre-direction in-FOV is the
  only filter. Layered occlusion reasoning (e.g. using SfM 3D points
  to test whether a source's line-of-sight to a tile is blocked) is
  an application-level extension built on top of the observation
  list.
- **No per-tile pyramid depth knob.** Every tile's pyramid has the
  same `L = log2(B) + 1` levels by design — picking a subset would
  fragment the type and complicate downstream consumers. If a
  consumer only wants level 0, it reads `tile(t).levels[0]`; the
  unused levels add ~33% memory but are simple box-filter passes,
  not significant compute.

## Status / dependencies

Depends on:

- `SphericalTileRig` (`specs/core/spherical-tiles-rig.md`) — DONE.
  - `direction(t)`, `tile_rotation(t)`, `tile_camera()`,
    `patch_size`. All shipped.
- `WarpMap::from_cameras_with_rotation` and `WarpMap::remap_bilinear`
  (`specs/drafts/warpmap-pose-extension.md`) — DONE.

Independently testable; no external state beyond the rig + sources.

### Concrete consumers in this codebase

- New direction-space algorithms (panorama compositing, multi-source
  alpha estimation, GPU multi-scale shaders) consume the same
  observation list with no changes to this primitive.
