# Spherical tile rig: discretizing the sphere as a rig of pinhole tiles

**Status:** Draft proposal.

## Motivation

For per-direction work on the sphere (infinity-consistency tests,
parallax-from-pose depth estimation, multi-view color aggregation) we need a
discretization that samples the sphere in small, nearly-distortion-free
patches. Three options were considered:

| Scheme | Distortion at patch edge | Patch count | Seam handling |
|--------|--------------------------|-------------|---------------|
| Raw equirectangular grid | Singular at poles (1/cos(lat)) | 1 | N/A |
| 6-face cubemap (90° FOV/face) | ~3× stretch at face corners | 6 | 12 seams |
| Tangent patches at evenly-distributed directions (few-° to few-tens-of-° FOV/patch) | <1.01× stretch over typical sizing | tunable (10²–10⁴) | overlapping patches |

A rig of small pinhole tiles distributed over the sphere is the right shape for this problem:

- **Per-direction work happens at the tile's look direction.** Infinity tests,
  depth regression, photometric consensus all evaluate at `d_tile`. A tile's
  peripheral pixels exist only as neighborhood support for NCC / gradient-based
  smoothness — they do not need to be the look direction of some other
  estimator.
- **Warping distortion matters for patch-match.** NCC between two warped
  patches is sensitive to the local stretch of the warp: if the warp is
  anisotropic or heavily stretched (cube-map corner), the signal degrades.
  At ~8° FOV tiles the local Jacobian deviates from identity by <1%.
- **Many small tasks parallelize well.** Each tile is independent and maps
  cleanly to a GPU dispatch.

## The tile rig, stored compactly

A `SphericalTileRig` is a **camera rig** of `n` pinhole tile cameras that all
share a single world-space optical centre `c`. Every
tile has identical intrinsics (same focal length, same half-FOV, same patch
resolution); only the rotation differs per tile, so the tiles look in
different directions on the unit sphere around `c`. Each tile's image plane
is locally tangent to that sphere at the tile's look direction.

A general rig would carry per-camera intrinsics and a full per-camera pose,
even though every tile here shares intrinsics and an optical centre. The
tile-rig representation drops the redundant fields:

- A single `centre: [f64; 3]` for the rig.
- A single `CameraIntrinsics` (returned by `tile_camera()`) shared by every
  tile.
- A flat `Vec<[f64; 3]>` of unit look directions in the world frame.
- A flat `Vec<[f64; 6]>` of `(e_right, e_up)` tangent-plane bases, also in
  the world frame, completing the per-tile rotation.

The per-tile rotation `R_world_from_tile` has columns `[e_right | e_up |
direction]`, so a tile-frame ray `v_tile` maps to world as
`v_world = R_world_from_tile · v_tile`. Storing `(direction, e_right, e_up)`
explicitly (rather than a quaternion or just `direction` with a derived
basis) keeps the data laid out for SoA / GPU consumption: each per-tile
shader invocation reads three contiguous unit vectors and assembles the
rotation matrix without a trig call.

## Tile layout

Tile look directions are produced by `evenly_distributed_sphere_points(n, …)`
(`crates/sfmtool-core/src/sphere_points.rs`), which
uniformly samples `n` points and then applies a KD-tree-accelerated
Thomson-style 1/r² repulsion relaxation. The result is a free-form,
near-uniform distribution: any `n` is allowed, the tile count is decoupled
from any fixed polyhedral subdivision, and the per-tile FOV / patch
resolution can be chosen to fit the desired output equirect width without
quantising to powers of 4.

### Sizing parameters

A `SphericalTileRig` is fully described by three knobs:

| Param | Meaning | Typical values |
|-------|---------|----------------|
| `n` | Number of tiles in the rig (i.e. distinct look directions on the sphere). Picked once for the finest angular resolution needed; reused across all pyramid levels (see § Coarse-to-fine pyramids). | 80 (small) up to ~20 000 (high-res) |
| `arc_per_pixel` | Angular size of one tile pixel, in radians. Chosen to match the target projected resolution: e.g. for an equirectangular width `W`, `arc_per_pixel = 2π/W`. | `2π/256` … `2π/4096` |
| `overlap_factor` | How much each tile's FOV exceeds the worst-case nearest-neighbour angular gap between tile directions on the sphere. | `1.15` (default; ~15% safety margin) |

From these, the half-FOV and patch grid size are derived directly from
the relaxer output, not from a closed-form approximation. The constructor:

1. Generates the `n` tile directions via
   `evenly_distributed_sphere_points`.
2. Measures the actual worst-case nearest-neighbour gap from those
   directions: chord distances come from
   `PointCloud3::nearest_neighbor_distances`, converted to angles via
   `angle = 2 · asin(chord / 2)`.
3. Sets `half_fov_rad = 0.5 · measured_max_nn_angle · overlap_factor`.
4. Sets `patch_size = ceil(2 · half_fov_rad / arc_per_pixel)`, clamped to
   a minimum (e.g. 5) so NCC / gradient kernels still have neighbourhood
   support at the smallest configurations.

The measured max NN angle is also stashed on the rig so callers can
inspect it. As a sanity check, on the current relaxer config it tracks
`≈ 1.05–1.15 · √(4π/n)` for `n ∈ [50, 10 000]`, with mean ≈ `0.95 ·
√(4π/n)` and `std/mean ≈ 0.045`. None of those numbers are baked into
the constructor — if relaxer quality changes, or `n` is small enough that
the empirical relation breaks, the FOV adapts automatically and the
coverage invariant ("every direction's nearest tile contains it") still
holds by construction.

For very small `n` (≲ 50), random init plus 50 iterations may not have
fully relaxed; the measured max NN angle will simply be larger and the
FOV correspondingly wider. If that's undesirable, raise `relax.iterations`
in `RelaxConfig` rather than tweaking a fudge factor.

### Sizing compared to equirectangular images

For `overlap_factor = 1.15`, the convention `arc_per_pixel = 2π/W`, and a
typical relaxer run (so `max_nn_angle ≈ 1.10 · √(4π/n)`):

| `n` | `W_equiv` | arc/pixel | tile FOV (approx) | `patch_size` (approx) |
|-----|-----------|-----------|-------------------|-----------------------|
| 80 | 256 | 1.41° | ~28.7° | ~21 |
| 320 | 512 | 0.70° | ~14.4° | ~21 |
| 1 280 | 1 024 | 0.35° | ~7.2° | ~21 |
| 5 120 | 2 048 | 0.18° | ~3.6° | ~21 |
| 20 000 | 4 096 | 0.09° | ~1.8° | ~21 |

The values above are illustrative — actual `half_fov_rad` and
`patch_size` come from the measured max NN gap of the constructed rig,
not the formula. The pattern is by design: `tile_FOV ∝ 1/√n`,
`arc_per_pixel ∝ 1/W`, so the ratio (and hence `patch_size`) is roughly
preserved when `n ∝ W²`. A `patch_size` near 21 (≈ 20×20 GPU kernel) is a
reasonable default; callers that want larger NCC support windows should
bump `n` down or `W` up. The `n ∝ W²` relation falls out of "spend the
same total work on the sphere as on a W×H equirectangular image": total
tile pixels `≈ n · patch_size² ≈ const · W²`. With these defaults
`n · patch_size² ≈ 0.54 · W²`, vs `W · H = 0.5 · W²` for the equirect —
about a 10% overdraw, which is the price of overlap.

### API

```rust
pub struct SphericalTileRig {
    /// World-space optical centre shared by every tile in the rig.
    centre: [f64; 3],
    /// Unit look direction per tile in world frame. (3·n f64)
    directions: Vec<[f64; 3]>,
    /// World-frame tangent-plane basis per tile: (e_right, e_up).
    /// e_right, e_up are unit, orthogonal to the tile's direction, and
    /// right-handed: `e_right × e_up = direction`.
    /// Together with `direction` they form the columns of
    /// `R_world_from_tile`: the tile-frame x axis maps to e_right in
    /// world, the tile-frame y axis maps to e_up, and the tile-frame z
    /// axis (the optical axis) maps to direction.
    bases: Vec<[f64; 6]>,
    /// Half-FOV of each tile in radians. Uniform across tiles.
    /// Set at construction to
    /// `0.5 · measured_max_nn_angle · overlap_factor`.
    half_fov_rad: f64,
    /// Measured worst-case nearest-neighbour angular gap across
    /// `directions`, in radians. Stored for diagnostics; `half_fov_rad`
    /// is derived from this.
    measured_max_nn_angle: f64,
    /// Per-tile patch grid size (pixels per side). Uniform across tiles.
    patch_size: u32,
    /// Number of tile columns in the packed atlas. The atlas height is
    /// `ceil(n / atlas_cols) * patch_size`; see § Tile-to-image mapping.
    atlas_cols: u32,
    /// KD-tree over `directions` (as unit vectors in R³) used by
    /// `warp_from_atlas_with_rotation` to pick the closest tile for any
    /// world-frame ray. Built once at construction from
    /// `crates/sfmtool-core/src/spatial.rs::PointCloud3`. Squared-Euclidean
    /// nearest-neighbour on unit vectors is monotone in angular distance,
    /// so this gives the angularly-closest tile.
    direction_tree: PointCloud3<f32>,
}

pub struct SphericalTileRigParams {
    /// Rig optical centre in world space.
    pub centre: [f64; 3],
    /// Number of tiles. Must be ≥ 2.
    pub n: usize,
    /// Angular size of one tile pixel, in radians. For a target equirect
    /// of width W, pass `2.0 * PI / (W as f64)`. Must be > 0.
    pub arc_per_pixel: f64,
    /// Multiplicative safety margin on the **measured** worst-case
    /// nearest-centre gap (not the analytic `√(4π/n)` baseline).
    /// 1.15 = 15% overlap; default. Must be ≥ 1.0.
    pub overlap_factor: f64,
    /// Optional override for the atlas column count. `None` ⇒
    /// `ceil(sqrt(n))`. Useful for matching texture-size limits or for
    /// tooling that streams tiles row-major.
    pub atlas_cols: Option<u32>,
    /// Forwarded to `evenly_distributed_sphere_points`. None ⇒ defaults.
    pub relax: Option<RelaxConfig>,
}

#[derive(Debug)]
pub enum SphericalTileRigError {
    /// `n < 2`.
    TooFewTiles,
    /// `arc_per_pixel <= 0` or non-finite.
    InvalidArcPerPixel,
    /// `overlap_factor < 1.0` or non-finite.
    InvalidOverlapFactor,
    /// `centre` contains a non-finite component.
    InvalidCentre,
}

impl SphericalTileRig {
    /// Build from explicit sizing parameters. Tile look directions come
    /// from the sphere-point relaxer; `half_fov_rad` is derived from the
    /// **measured** worst-case nearest-neighbour gap of the generated
    /// directions and `overlap_factor`; `patch_size` is then
    /// `max(MIN_PATCH_SIZE, ceil(2 · half_fov_rad / arc_per_pixel))`.
    ///
    /// Returns an error if any parameter fails the validity checks
    /// documented on `SphericalTileRigParams`.
    pub fn new(params: &SphericalTileRigParams) -> Result<Self, SphericalTileRigError>;

    pub fn len(&self) -> usize;

    /// World-space optical centre shared by every tile.
    pub fn centre(&self) -> [f64; 3];

    /// Unit look direction of tile `idx` in world frame.
    pub fn direction(&self, idx: usize) -> [f64; 3];
    pub fn basis(&self, idx: usize) -> ([f64; 3], [f64; 3]);

    /// Half-FOV of each tile, in radians.
    pub fn half_fov_rad(&self) -> f64;

    /// The measured worst-case nearest-neighbour angular gap across all
    /// tile directions. Diagnostic only; the constructor already used
    /// this to pick `half_fov_rad`.
    pub fn measured_max_nn_angle(&self) -> f64;

    /// Pinhole `CameraIntrinsics` shared by every tile. Concretely:
    /// - model: `Pinhole`
    /// - `width = height = patch_size`
    /// - `cx = cy = patch_size as f64 / 2.0`
    /// - `fx = fy = (patch_size as f64 / 2.0) / tan(half_fov_rad)`
    /// - no distortion
    ///
    /// Composed with `WarpMap::from_cameras_with_pose` this is the
    /// per-tile workhorse of the algorithm.
    pub fn tile_camera(&self) -> CameraIntrinsics;

    /// Build `R_world_from_tile` for tile `idx`. Columns are
    /// `[e_right | e_up | direction]`, so for a tile-frame ray
    /// `v_tile`, `R · v_tile` is the world-frame ray. The transpose
    /// gives `R_tile_from_world`.
    pub fn tile_rotation(&self, idx: usize) -> [f64; 9];

    /// Apply an `Se3Transform` to the rig: rotates and translates
    /// `centre`, rotates `directions` and `bases`, and rebuilds
    /// `direction_tree`. Scale is not consumed (the rig has no
    /// metric scale; the tile camera intrinsics are unitless).
    pub fn apply_transform(&mut self, t: &Se3Transform);

    // -- Tile-to-image (atlas) mapping --
    /// Atlas image size `(width, height)` in pixels. Width is
    /// `atlas_cols * patch_size`; height is
    /// `ceil(n / atlas_cols) * patch_size`.
    pub fn atlas_size(&self) -> (u32, u32);
    pub fn atlas_cols(&self) -> u32;
    /// Top-left pixel `(x, y)` of tile `idx`'s sub-image inside the
    /// atlas. Panics if `idx >= n`.
    pub fn tile_atlas_origin(&self, idx: usize) -> (u32, u32);

    // -- Atlas warp maps (whole-rig variants of `from_cameras_with_rotation`) --
    //
    // Note the asymmetry of the rotation parameters between these two
    // methods: each takes the rotation needed to move from its
    // *external* camera frame into the world frame (or vice versa).
    // `warp_to_atlas_with_rotation` reads from `src`, so it needs
    // `R_src_from_world` to land in `src`'s frame; `warp_from_atlas_with_rotation`
    // writes into `dst`, so it needs `R_world_from_dst` to leave
    // `dst`'s frame and reach the tile frame via the world.

    /// Build a `WarpMap` with the atlas as the **destination** image. For
    /// each atlas pixel `(u, v)`:
    ///
    /// 1. The owning tile and in-tile pixel are derived from
    ///    `tile_atlas_origin` / `patch_size`.
    /// 2. The in-tile pixel is unprojected through `tile_camera()` to a
    ///    ray in the tile's local frame.
    /// 3. The ray is rotated to world frame via
    ///    `R_world_from_tile = [e_right | e_up | direction]` (columns).
    /// 4. The ray is rotated to the src-camera frame via
    ///    `rot_src_from_world`.
    /// 5. The src-camera ray is projected through `src` to pixel
    ///    coordinates.
    ///
    /// The returned map has dimensions `atlas_size()`. Atlas slots that do
    /// not belong to any tile (the trailing slots when `atlas_cols` does
    /// not divide `n` evenly) are filled with `NaN`.
    ///
    /// This is the whole-rig analogue of
    /// `WarpMap::from_cameras_with_rotation(src, tile_camera(), R_st)`
    /// applied per tile and concatenated into the atlas — but it builds
    /// the atlas-sized map in a single pass.
    pub fn warp_to_atlas_with_rotation(
        &self,
        src: &CameraIntrinsics,
        rot_src_from_world: &RotQuaternion,
    ) -> WarpMap;

    /// Build a `WarpMap` with the atlas as the **source** image. For each
    /// `dst` pixel `(u, v)`:
    ///
    /// 1. `dst.pixel_to_ray(u, v)` gives a ray in the dst-camera frame.
    /// 2. `rot_world_from_dst` rotates it to the world frame.
    /// 3. The closest tile direction is found by querying
    ///    `direction_tree` (KD-tree, O(log n)).
    /// 4. The world ray is rotated into that tile's frame by
    ///    `R_tile_from_world = R_world_from_tile.transpose()`.
    /// 5. The tile-frame ray is projected through `tile_camera()` to an
    ///    in-tile pixel, which is offset by `tile_atlas_origin(tile_idx)`
    ///    to obtain atlas pixel coordinates.
    ///
    /// The returned map has dimensions `dst.size()`.
    ///
    /// **Why closest-tile is sufficient.** With
    /// `half_fov_rad = 0.5 · measured_max_nn_angle · overlap_factor`,
    /// every world direction's angularly-closest tile contains that
    /// direction inside its patch (validated by the coverage test below),
    /// so the projected pixel always lands inside the tile's
    /// `patch_size × patch_size` block.
    ///
    /// **Tile boundary discontinuities.** The output has visible seams
    /// along Voronoi cell boundaries between adjacent tiles, because
    /// each dst pixel samples a single tile. This is the right
    /// primitive for round-trip / diagnostic use. The smoother
    /// `k`-nearest-tile blend used for final equirectangular rendering is
    /// described in § Assembly into an equirectangular image; conceptually it is the
    /// `k > 1` extension of this method.
    pub fn warp_from_atlas_with_rotation(
        &self,
        dst: &CameraIntrinsics,
        rot_world_from_dst: &RotQuaternion,
    ) -> WarpMap;
}
```

The Python wrapper exposes `SphericalTileRig` along with a helper to resample
an assembled tile grid into an equirectangular image (bilinear trilateration
over the nearest overlapping tiles).

### Tile-to-image mapping

Per-tile arrays (color / alpha / depth / NCC scratch) live in a single
**atlas image**, not `n` independent buffers. Each tile occupies a
`patch_size × patch_size` block at origin
`tile_atlas_origin(idx) = (idx % atlas_cols * patch_size,
                          idx / atlas_cols * patch_size)`,
and the atlas as a whole has size
`(atlas_cols * patch_size, ceil(n / atlas_cols) * patch_size)`.

This matters for two reasons:

1. **GPU friendliness.** A future wgpu port wants every tile in one
   storage texture / one buffer, so a single bind-group + dispatch covers
   the entire set. `n` separate textures would multiply descriptor-set
   overhead and break batching. The CPU implementation uses the same
   layout for symmetry — Rust slices the atlas into per-tile views via
   `tile_atlas_origin` and `patch_size`.
2. **Cache locality and I/O.** Final equirectangular resampling and debug dumps
   become a single image read / write, and the atlas can be saved as one
   PNG / EXR for inspection.

`atlas_cols` defaults to `ceil(sqrt(n))` (a square-ish layout) but is
exposed as a knob: a wide atlas can be friendlier to texture-size limits,
and tooling that streams tiles row-major may want a fixed `atlas_cols`.

## Why evenly-distributed tile directions

Compared to other sphere-discretization options (icosahedron subdivision,
HEALPix, geodesic grid, cube sphere):

- **Thomson-relaxed point set** — what we use. Tile count is a free
  parameter; tile directions are isotropic to within ~5% NN-spacing
  variance; `n` is matched to the desired output resolution `W` without
  quantising to powers of 4.
- **Icosahedron subdivision** has uniform spatial density up to ~10%
  imbalance across subdivisions but quantises tile count to `20 · 4^L`
  (20, 80, 320, 1280, …). That forces the algorithm's resolution ladder
  to powers of 2 in `W` and makes intermediate tile counts impossible.
- **HEALPix** is uniform in solid angle but the patches aren't tangent-plane
  and require careful warping at the equatorial/polar-face boundary.
- **Cube sphere (6 faces)** has uniform sampling per face but up to 3×
  corner stretch. Rejected upstream.

### Coarse-to-fine pyramids are within tiles, not across rigs

Successive pyramid levels keep the **same** `SphericalTileRig` —
identical `centre`, identical `directions`, identical `bases`, identical
`half_fov_rad`. What changes is `arc_per_pixel`, and therefore
`patch_size`: a coarse level uses a smaller `patch_size`, a finer level
a larger one, both representing the same look directions. Building a
finer level from a coarser one is a per-tile bilinear / Gaussian
upsample of the atlas, with no re-association of directions.

This sidesteps the issue that two independent runs of the relaxer (or
the same run at different `n`) produce unrelated point sets — Thomson
relaxation does not nest. As long as the rig is constructed once at the
finest intended angular resolution and reused, that doesn't matter. If a
caller really does want a different `n` for a different stage, they
build a separate rig and treat it as unrelated.

## Assembly into an equirectangular image

Once per-tile arrays (color / alpha / depth) are computed, the equirectangular
output is produced by:

1. For each equirectangular pixel direction `d`, find the `k` nearest tile look
   directions (`k = 3` works; inverted-octree / BVH over the directions for
   O(log n) query).
2. Barycentric-weight the tiles' per-direction values (`remap_aniso`-sample
   each tile's atlas sub-image at the pixel corresponding to `d`'s
   projection through the (shared) tile camera with that tile's rotation).
3. Accept / reject by alpha, combine colors by confidence-weighted average.

The assembly kernel is analogous to `WarpMap::remap_aniso` but sourcing from
`k` tile images instead of one — a small variant of the existing resampler.

## Testing

- **Tile count is exactly `params.n`.** Trivial — but a regression guard
  against any future "snap to nearest icosahedral count" temptation.
- **Coverage (parameterised over `n`).** For each of `n ∈ {20, 80, 320,
  5_000}`, every unit vector's angular distance to its nearest tile
  direction is ≤ `half_fov_rad`. The "every unit vector" sample is a
  separate `evenly_distributed_sphere_points(50_000)` call (different
  RNG state from the rig under test) — that's the simplest sphere
  stratification we already have. Because `half_fov_rad` is derived from
  the rig's own measured max NN gap, this is mostly a check that
  `overlap_factor ≥ 1` and the asin/chord conversion is correct, but it
  also guards against the 50 000-sample probe being denser than the rig
  in regions the relaxer happened to leave sparse.
- **Half-FOV tracks measurement.** For a constructed rig with known `n`
  and `overlap_factor`, assert
  `|half_fov_rad - 0.5 · measured_max_nn_angle · overlap_factor| < eps`
  and `measured_max_nn_angle` matches `nn_angles.max()` recomputed from
  `directions`.
- **Direction uniformity.** Std/mean of NN angular distances over the
  generated tile directions is < 0.06 — guards against accidentally
  regressing the relaxer's `iterations` or `step_size` defaults.
- **Warp correctness.** A synthetic image of a known pattern warped
  through `(tile_camera(), from_cameras_with_rotation)` with each tile's
  rotation reproduces the pattern at the tile's look direction to within
  interpolation tolerance.
- **Seam behavior.** Two overlapping tiles, given the same source image,
  agree in the overlap region to within bilinear tolerance.
- **Atlas packing round-trip.** For every tile, writing a unique constant
  into its atlas sub-image and reading back via `tile_atlas_origin(idx)` +
  `patch_size` recovers the same constant — guards against off-by-one
  errors in the row/col packing.
- **Equirectangular ↔ atlas round-trip.** Build a smooth synthetic
  equirectangular image — a band-limited pattern such as
  `f(lon, lat) = 0.5 + 0.25·sin(3·lon)·cos(2·lat)`, sampled at a
  resolution that matches the rig's angular resolution
  (`n = 320`, `W = 512` is a good middle ground). Build an
  identity-rotation `equirect_camera` and a `SphericalTileRig` co-located
  at the origin. Then:
  ```text
  to_atlas    = tiles.warp_to_atlas_with_rotation(equirectangular, IDENTITY)
  atlas_img   = to_atlas.remap_bilinear(equirectangular_img)
  from_atlas  = tiles.warp_from_atlas_with_rotation(equirectangular, IDENTITY)
  recovered   = from_atlas.remap_bilinear(atlas_img)
  ```
  Assert two thresholds on a `[0, 1]` float image:
  - **Interior** (pixels whose corresponding direction is more than
    `0.5 · half_fov_rad` from the nearest Voronoi cell boundary):
    `mean_abs_error < 1e-3`, `max_abs_error < 5e-3`. This is a clean
    bilinear-interp-twice tolerance.
  - **Full image, including seams**: `mean_abs_error < 6e-3` (≈ 1.5/255
    on u8). Per-pixel max can spike to ~0.02 along seams where the two
    bilinear samples come from different tile centres; that's a property
    of the closest-tile primitive, not a bug.
  This single test exercises (a) the atlas-pixel → tile-frame mapping
  in `warp_to_atlas_with_rotation`, (b) the KD-tree closest-tile lookup
  in `warp_from_atlas_with_rotation`, (c) coverage (every equirectangular
  direction finds a tile that contains it), and (d) that the rotation
  conventions in both directions agree. The expected error ceiling
  drops as `n` and `W` increase; tighten if and when the assembly path
  (`§ Assembly into an equirectangular image`) replaces closest-tile
  with `k`-nearest blend.

## Open questions

- **Reproducibility of tile placement.** `random_sphere_points` currently
  uses `rand::rng()` (thread-local, unseeded). For deterministic
  reconstructions across runs the relaxer needs a seed. Plumb a
  `seed: Option<u64>` through `RelaxConfig` → `SphericalTileRigParams` before
  this lands as a public CLI surface.
- **Persistence.** The rig is described here as an in-memory compute
  structure; nothing in this spec serialises it. If a downstream `solve`
  / `densify` stage emits per-tile depth and we want resumable runs,
  decide whether to (a) re-generate the rig from `(centre, n, seed,
  arc_per_pixel, overlap_factor)` and trust determinism, or (b) write
  `directions` / `bases` to disk alongside the atlas.
- **Texture-size limits for very large `n`.** Default
  `atlas_cols = ceil(sqrt(n))` gives a near-square atlas; at
  `n = 20_000`, `patch_size ≈ 21` ⇒ atlas ≈ 2982 × 2961, well within
  any GPU's 8K limit. But a multi-channel atlas (RGB + alpha + depth +
  scratch) at `n = 80_000` starts brushing 8K; document the rule of
  thumb and let `params.atlas_cols` override.
- **Multi-buffer atlas layout.** Color, alpha, and depth all want
  different channel counts and bit depths. The current spec describes a
  single atlas; in practice each algorithm will keep several atlases
  (e.g. an `ImageU8` for color and a `Vec<f32>` for depth) sharing the
  same `tile_atlas_origin` layout. Worth a follow-up that defines an
  `Atlas<T>` helper rather than re-deriving the layout in every caller.
