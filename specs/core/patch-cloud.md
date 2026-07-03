# Patch Cloud and Patch-Projected Warp Maps

**Status:** Proposed. Motivated by the `scripts/patch_crossval.py` exploration,
which renders scale/orientation-normalized patches around matched keypoints by
warping each source image through the keypoint's 2D affine shape. That works per
image but is only a 2D approximation of a 3D surface element. This spec defines
the `sfmtool-core` types and routine that let callers express the patch in **3D**
— an oriented surfel — and render its appearance from any camera, so a track's
observations become geometrically-consistent views of one world patch.

Builds directly on `specs/core/image-warping.md` (`WarpMap`, `remap_bilinear`,
`remap_aniso`, `ImageU8Pyramid`) and the `CameraIntrinsics` ray API.

## Motivation

Given a reconstructed 3D point with a surface normal, every image that observes
it sees the same little planar surface patch — foreshortened, distorted, and
sampled differently per view. To compare or visualize those observations we want,
for each observing camera, the **canonical patch image**: a fixed `R×R` grid
aligned to the patch's own frame, filled by sampling that camera's source image.

Today `patch_crossval.py` builds this per-image from the keypoint's 2D affine
shape. That has three limitations the 3D formulation removes:

1. **Consistency.** Two views of the same point are normalized independently (each
   from its own affine shape), so the patches are only approximately aligned. A
   single 3D patch projected into both cameras aligns them by construction.
2. **Geometry.** The affine shape is a local linearization; the true mapping for a
   planar patch is the full perspective projection (a homography for a pinhole
   camera, and the distorted/fisheye projection in general). `ray_to_pixel`
   already models all 11 camera models, so projecting the patch is exact.
3. **Reuse.** The patch lives in world space, independent of any one image, so the
   same patch cloud drives cross-validation, track visualization, multi-view
   patch extraction, and (later) photometric refinement.

## Coordinate conventions

- World, camera, and pixel conventions follow the rest of the codebase: pixel
  centers at `(col + 0.5, row + 0.5)` (see `image-warping.md`), camera looks down
  +Z, extrinsics are **camera-from-world** (matching
  `SfmrReconstruction.quaternions_wxyz` / `translations`, constructible via
  `RigidTransform::from_wxyz_translation`).
- A patch has a local 2D frame `(s, t) ∈ [-1, 1]²`. The `R×R` patch grid samples
  it at pixel centers: grid pixel `(col, row)` maps to
  `s = 2·(col + 0.5)/R − 1`, `t = 2·(row + 0.5)/R − 1`.

## `OrientedPatch`

A planar surface element (surfel) in world space.

```rust
/// An oriented planar patch (surfel) in world space.
///
/// The patch plane is spanned by two orthonormal in-plane axes; `u_axis` and
/// `v_axis` define both the plane and its in-plane rotation. The outward normal
/// is `v_axis × u_axis`. `half_extent` is the world-space half-size along each
/// axis, so the patch covers the world points
/// `center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis` for
/// `(s, t) ∈ [-1, 1]²`.
///
/// The handedness is image-raster aligned: a `(col, row)` render (see
/// `WarpMap::from_patch`) steps `col` along `+u_axis` and `row` along `+v_axis`
/// with `row` increasing downward, so the front face renders un-mirrored — the
/// same chirality the camera sees. That is why the outward normal is
/// `v_axis × u_axis` (which points toward the camera for a front-facing patch)
/// and not `u_axis × v_axis` (which would render the back face, i.e. a mirror
/// image).
///
/// `w` is the homogeneous weight of the anchor: `1.0` for a finite point
/// (`center` is a Euclidean position) and `0.0` for a point at infinity
/// (`center` is a direction `d`, the patch is tangent to the unit sphere around
/// `d`, and the corners are themselves directions). Rendering and visibility
/// branch on it.
pub struct OrientedPatch {
    pub center: Point3<f64>,
    pub u_axis: Vector3<f64>,     // unit, in-plane
    pub v_axis: Vector3<f64>,     // unit, in-plane, ⟂ u_axis
    pub half_extent: [f64; 2],    // world units (or angular at infinity), along (u, v)
    pub w: f64,                   // 1.0 finite, 0.0 at infinity
}

impl OrientedPatch {
    /// Outward normal (`v_axis × u_axis`).
    pub fn normal(&self) -> Vector3<f64>;

    /// World point for a normalized patch coordinate `(s, t) ∈ [-1, 1]²`
    /// (Euclidean; meaningful for a finite patch only).
    pub fn to_world(&self, s: f64, t: f64) -> Point3<f64>;

    /// The patch corner for `(s, t)` as a homogeneous world point `(xyz, w)`:
    /// `(center + s·u + t·v, w)`. Finite (`w = 1`) gives a Euclidean point; at
    /// infinity (`w = 0`) `center` is a direction and the corner is again a
    /// direction. This is what rendering projects (see `WarpMap::from_patch`).
    pub fn corner_homogeneous(&self, s: f64, t: f64) -> (Vector3<f64>, f64);

    /// Build a **finite** patch (`w = 1`) from a center, a normal, and an
    /// `up_hint` used to pin the in-plane rotation: `u_axis` is `up_hint`
    /// projected onto the plane (Gram-Schmidt) and normalized,
    /// `v_axis = normal × u_axis`. `half_extent` may be a scalar (square) or
    /// per-axis.
    pub fn from_center_normal(
        center: Point3<f64>,
        normal: Vector3<f64>,
        up_hint: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self;

    /// Build the tangent-sphere frame for a **point at infinity** (`w = 0`) with
    /// direction `d`: outward normal `normalize(-d)`, `u, v ⊥ d`, in-plane
    /// rotation pinned by `up_hint`. `center` stores `d`.
    pub fn from_infinity_direction(
        direction: Point3<f64>,
        up_hint: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self;

    /// Whether `cam_from_world` looks at the front face. A point at infinity is
    /// always front-facing (its normal `normalize(-d)` faces every observer;
    /// cheirality is enforced by the projection).
    pub fn is_front_facing(&self, cam_from_world: &RigidTransform) -> bool;
}
```

The `up_hint` resolves the one remaining degree of freedom (rotation about the
normal). For visualization a stable, view-independent choice (e.g. world up, or
the first observing camera's up axis) keeps a track's patches mutually aligned.

## `PatchCloud`

A collection of oriented patches, stored struct-of-arrays to mirror the point
cloud representation and to batch well.

```rust
pub struct PatchCloud {
    pub patches: Vec<OrientedPatch>,
    /// Source reconstruction 3D-point index for each patch (parallel to
    /// `patches`).
    pub point_indexes: Vec<u32>,
}

impl PatchCloud {
    pub fn len(&self) -> usize;
    pub fn patch(&self, i: usize) -> &OrientedPatch;

    /// One patch per finite 3D point: center = position, in-plane up from the
    /// first observing camera, normal and half-size per the given policies.
    /// Errors with `PatchCloudError::MissingFeatureScale` under
    /// `PatchExtent::FeatureSize` when a point has no readable keypoint scale in
    /// any view (no silent size fallback).
    ///
    /// When `exclude_points_at_infinity` is `false` (the binding default — every
    /// patch operation handles infinity patches), each point at infinity also gets
    /// a tangent-sphere frame (`w = 0`) around its direction `d` — outward normal
    /// `normalize(-d)`, `u, v ⊥ d`, with an angular half-size from the
    /// distance-free form of `extent` (`FeatureSize`/`PixelRadius`: `σ_i/f_i` /
    /// `radius_px/f_i` reduced across views; `Fixed`/`RelativeToSpacing`: their
    /// scalar as the tangent magnitude). Pass `true` to emit finite points only —
    /// needed by an operation that scatters per-point results back and must leave
    /// infinity points untouched (normal refinement's normal write-back), or that
    /// wants the historical finite-only behavior (e.g. the strips viz).
    pub fn from_reconstruction(
        recon: &SfmrReconstruction,
        normal: PatchNormal,
        extent: PatchExtent,
        exclude_points_at_infinity: bool,
    ) -> Result<Self, PatchCloudError>;
}

/// How to choose each patch's surface normal.
pub enum PatchNormal {
    /// Use the reconstruction's stored estimated normal (whatever is in the
    /// `.sfmr`, not necessarily the mean viewing direction); falls back to the
    /// mean viewing direction if zero/degenerate.
    Stored,
    /// Normalized mean of the unit directions from the point to each observing
    /// camera centre.
    MeanViewing,
    /// Local plane fit (PCA) over the `k_neighbors` nearest 3D points, oriented
    /// toward the mean viewing direction.
    Geometric { k_neighbors: usize },
}

/// How to choose each patch's world-space half-size.
pub enum PatchExtent {
    /// Same world half-size for every patch.
    Fixed(f64),
    /// `factor` × the median nearest-neighbor spacing (denser regions -> smaller
    /// patches).
    RelativeToSpacing(f64),
    /// Back-project `radius_px` pixels to a world half-size in each observing
    /// view, reduced across views by `across`. `Min` keeps the patch within the
    /// pixel budget in every view (sized to it in the view where the point
    /// appears largest); `Max`/`Median`/`Mean` trade that off.
    PixelRadius { radius_px: f64, across: ViewReduce },
    /// `factor` × the projected world feature size — each observation's keypoint
    /// scale back-projected to world (`sigma_i · d_i / f_i`, where `d_i` is the
    /// ray distance `‖X − C_i‖`), reduced across views by `across` (`Median`
    /// recommended; the scales are consistent across a track's views). Using the
    /// ray distance rather than the optical-axis depth `z_i` makes this
    /// camera-agnostic: a fisheye (FoV > 180°) sees points past 90° off axis at
    /// `z_i ≤ 0`, which a pinhole `σ·z/f` could not size. On-axis `d = z`, so it
    /// reduces to the pinhole form. Reads the workspace `.sift` files. A point
    /// with no readable scale in any view is an error
    /// (`PatchCloudError::MissingFeatureScale`) — there is no silent size fallback.
    FeatureSize { factor: f64, across: ViewReduce },
}

// PatchExtent::default() == FeatureSize { factor: 5.0, across: Median }, and the
// Python binding's `extent` defaults to "feature_size" / `extent_value` to 5.0.

/// How to reduce a per-view quantity across a point's observing views.
pub enum ViewReduce { Min, Max, Median, Mean }
```

`SfmrReconstruction` exposes `positions`, `normals`, cameras and poses, so
`from_reconstruction` is the bridge from a solved model to a patch cloud.

**Points at infinity across patch operations.** `from_reconstruction` builds the
tangent-sphere frame for points at infinity when asked
(`exclude_points_at_infinity = false`), and every patch operation handles them per
its nature — a `w = 0` patch is first-class throughout:
- **Rendering.** `WarpMap::from_patch` projects each homogeneous corner
  ([`OrientedPatch::corner_homogeneous`]) — a finite corner translates and
  projects normally, a direction (`w = 0`) rotates without translating, then
  projects as a ray — so an infinity patch renders correctly (its projection is
  translation-invariant). The `recon.patches` getter re-flags the reloaded
  infinity rows (`w = 0`) so this survives a save/load round-trip.
- **Normal refinement leaves infinity-point frames untouched.** A point at
  infinity has a *fixed* outward normal (`normalize(-d)`), so there is nothing to
  refine: `refine_patch_normal` skips `w = 0` patches and returns them unchanged.
- **View selection** handles them: the cheirality gate (`is_in_front`) and
  front-facing test use the homogeneous direction (a `w = 0` patch is always
  front-facing; cheirality is `R·d` forward), and vetting renders through the
  homogeneous `WarpMap` path.
- **Keypoint localization** handles them: `project`, the grazing pre-filter,
  `render_context`, and `seed_offset` all branch on `w`. For `w = 0` the keypoint
  is the projection of the direction, the seed→offset inversion is angular
  (`a = (ray·û)/(ray·d)`), and re-centring shifts the direction within its tangent
  frame.
- **Included by default.** Since every operation handles them, the binding
  defaults `exclude_points_at_infinity = false`, so a cloud built from a
  reconstruction carries its points at infinity. Operations that are finite by
  nature opt out with `exclude_points_at_infinity = true`: **normal refinement**
  (its normal write-back scatters per-point results back and must leave the
  `(0, 0, 0)` normal of an infinity point untouched) and the **strips viz**.

### Serialization

A `PatchCloud` round-trips to the per-point patch frame in the `.sfmr`
`points3d/` section (format version 3+) via
`PatchCloud::to_halfvec_arrays(point_count)` /
`PatchCloud::from_halfvec_arrays(half_u_xyz, half_v_xyz, centers)`.
`to_halfvec_arrays` scatters the cloud's patches into per-point rows by
`point_indexes`, folding each unit axis and half-extent into one vector and leaving
zero rows elsewhere; `from_halfvec_arrays` takes the points' positions as
`centers`, keeps the present rows (non-zero `u`), and recovers their point
indices. The half-vector arrays don't encode the homogeneous weight (it lives in
the points' `w`), so `from_halfvec_arrays` builds every patch finite (`w = 1`);
the `recon.patches` getter, which knows each point's `w`, marks the infinity rows
(`w = 0`) afterward.

The two arrays (`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`) and the optional
`patch_bitmaps_y_x_rgba` are stored on `SfmrData`/`SfmrReconstruction` as plain
`Option` fields beside `normals_xyz` — there is no separate patch struct, and the
centre is not stored (it is the point's position). See
`specs/formats/sfmr-file-format.md` for the on-disk layout. On the Python side,
`recon.clone_with_changes(patches=cloud)` attaches a cloud and `recon.patches`
reads it back; `recon.save` writes the `patch_*` arrays beside the normals. Patch
bitmaps are attached separately via
`recon.clone_with_changes(patch_bitmaps=<(N, R, R, 4) uint8>)` (the frame must be
present, so pass `patches=` in the same call unless one is already attached); the
patch-normal refinement pipeline emits them when asked (see
`PatchCloud.refine_normals(render_bitmaps=True)` and
`specs/core/patch-normal-refinement.md`).

The per-point patch frame rides along with the reconstruction's editing
operations rather than being discarded: image subsetting and point-mask filtering
keep the rows of the surviving points, and an SE(3) similarity (`--rotate` /
`--translate` / `--scale`) reorients and rescales the half-vectors with the
geometry (rotation only for a point at infinity, whose patch is angular) while the
bitmaps — parameterised in the patch's own `(s, t)` frame — are carried unchanged.
The per-point `normal` rotates in step, so `normalize(v × u)` stays consistent
with it.

> _Status (2026-06-11): Implemented — `patch/cloud.rs`
> (`OrientedPatch`, `PatchCloud::from_reconstruction`, `PatchNormal`,
> `PatchExtent`, `mean_viewing_normal`, `pca_plane_normal`), `WarpMap::from_patch`,
> PyO3 bindings (`OrientedPatch`, `PatchCloud`, `WarpMap.from_patch`). The
> For the COLMAP-solved reconstructions tested here the stored normals happened
> to match the mean viewing direction (`|cos| = 1.0` for all points), but that is
> an empirical observation about the solver, not a property of `Stored`. A
> photometric normal-refinement prototype lives in `scripts/patch_crossval.py`
> (`--refine-normal`)._

## Routine: project an image onto a patch

The core operation. It fits the existing `WarpMap::from_*` constructor family
(`from_cameras`, `from_cameras_with_pose`), and produces a `WarpMap` whose
**destination is the patch grid** and whose source coordinates are where each
patch pixel lands in the given image.

```rust
impl WarpMap {
    /// Build an `resolution × resolution` warp map that samples `camera`'s image
    /// over `patch`. For each patch-grid pixel `(col, row)`:
    ///   1. `(s, t)`  -> world point `patch.to_world(s, t)`
    ///   2. world -> camera: `p_cam = cam_from_world · world`
    ///   3. `camera.ray_to_pixel(p_cam)` -> source `(x, y)`, or NaN
    ///
    /// A pixel is NaN (invalid) when the point is behind the camera or outside
    /// the camera model's domain (`ray_to_pixel` returns `None`) or projects
    /// outside the image bounds — identical to the other constructors.
    pub fn from_patch(
        patch: &OrientedPatch,
        camera: &CameraIntrinsics,
        cam_from_world: &RigidTransform,
        resolution: u32,
    ) -> Self;
}
```

Notes:

- **Generalizes `from_cameras_with_pose`.** That constructor traces each
  destination ray to a fronto-parallel surface at a fixed `depth`;
  `from_patch` replaces that surface with an arbitrary oriented plane and the
  destination raster with the patch's own frame. A patch placed fronto-parallel
  to the camera at distance `d` reproduces a centered crop.
- **All camera models.** Because projection goes through `ray_to_pixel`,
  distortion and fisheye/equirectangular models are handled with no special
  casing — the same reason the patch beats the 2D affine approximation.
- **Anti-aliasing.** The projection's per-pixel Jacobian varies (perspective
  foreshortening + distortion), so oblique patches compress the source. Callers
  that want clean samples call `compute_svd()` then `remap_aniso` (see
  `image-warping.md`); `from_patch` returns `svd: None` (lazy), matching the
  other constructors.
- **Back-facing patches.** A helper
  `OrientedPatch::is_front_facing(cam_from_world) -> bool` (camera sees the
  outward normal) lets visualizers skip or flag views that observe the back of
  the surfel. `from_patch` itself does not cull — it has no occlusion model and
  renders whatever projects into frame.

### Batch / convenience

Rendering a patch across the cameras that observe it is the common case, so a
thin convenience composes the routine with `remap`:

```rust
/// One `WarpMap` per (camera, pose); pair with `remap_bilinear`/`remap_aniso`
/// to produce the canonical patch image per view.
pub fn warp_maps_for_patch(
    patch: &OrientedPatch,
    views: &[(CameraIntrinsics, RigidTransform)],
    resolution: u32,
) -> Vec<WarpMap>;
```

## Python bindings

Mirror the Rust API in `flow/warp.rs` / a new `py_patch_cloud.rs`:

- `OrientedPatch(center, u_axis, v_axis, half_extent)` and
  `OrientedPatch.from_center_normal(center, normal, up_hint, half_extent)`.
- `WarpMap.from_patch(patch, camera, cam_from_world, resolution)` returning a
  `WarpMap` (then `remap_bilinear` / `remap_aniso` as today).
- `PatchCloud.from_reconstruction(recon, extent)` and indexing to `OrientedPatch`.

This collapses the visualization to: build one `OrientedPatch` per track point,
and for each observing image call `WarpMap.from_patch(...).remap_bilinear(image)`
— removing the bespoke 2D affine warp in `patch_crossval.py` and making every
patch in a strip the same world surfel.

## Open questions

- **Normal source and quality.** `normals` may be noisy or absent for
  some points. Fallbacks: orient the patch fronto-parallel to a reference view,
  or to the mean viewing direction of the observing cameras. Worth a quality flag
  per patch.
- **Extent selection.** `PatchExtent` lists four policies. `FeatureSize` is the
  default — it sizes each patch from the observing keypoints' scales, which keeps
  the patch tied to the real feature support. `PixelRadius` reproduces a
  "constant pixels" behavior; `RelativeToSpacing` is scene-adaptive; `Fixed` is a
  uniform world size. Whether `FeatureSize` is also the right default for
  matching (vs. visualization) is still open.
- **In-plane rotation default.** `up_hint` = world up is simple but degenerate for
  near-horizontal patches; per-track consistency may prefer the reference
  camera's up. Pick one default, allow override.
- **Occlusion.** `from_patch` ignores visibility. A patch may project into a view
  where it is actually occluded; consumers that care (e.g. photometric checks)
  must filter separately.
- **GPU / batching.** Building one small `R×R` map per (patch, view) is cheap, but
  a whole `PatchCloud × views` sweep could be batched; deferred until a caller
  needs it.
