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
/// is `u_axis × v_axis`. `half_extent` is the world-space half-size along each
/// axis, so the patch covers the world points
/// `center + s·half_extent[0]·u_axis + t·half_extent[1]·v_axis` for
/// `(s, t) ∈ [-1, 1]²`.
pub struct OrientedPatch {
    pub center: Point3<f64>,
    pub u_axis: Vector3<f64>,     // unit, in-plane
    pub v_axis: Vector3<f64>,     // unit, in-plane, ⟂ u_axis
    pub half_extent: [f64; 2],    // world units, along (u, v)
}

impl OrientedPatch {
    /// Outward normal (`u_axis × v_axis`).
    pub fn normal(&self) -> Vector3<f64>;

    /// World point for a normalized patch coordinate `(s, t) ∈ [-1, 1]²`.
    pub fn to_world(&self, s: f64, t: f64) -> Point3<f64>;

    /// Build from a center, a normal, and an `up_hint` used to pin the in-plane
    /// rotation: `u_axis` is `up_hint` projected onto the plane (Gram-Schmidt)
    /// and normalized, `v_axis = normal × u_axis`. `half_extent` may be a scalar
    /// (square) or per-axis.
    pub fn from_center_normal(
        center: Point3<f64>,
        normal: Vector3<f64>,
        up_hint: Vector3<f64>,
        half_extent: [f64; 2],
    ) -> Self;
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
    pub point_ids: Vec<u32>,
}

impl PatchCloud {
    pub fn len(&self) -> usize;
    pub fn patch(&self, i: usize) -> &OrientedPatch;

    /// One patch per finite 3D point: center = position, in-plane up from the
    /// first observing camera, normal and half-size per the given policies.
    pub fn from_reconstruction(
        recon: &SfmrReconstruction,
        normal: PatchNormal,
        extent: PatchExtent,
    ) -> Self;
}

/// How to choose each patch's surface normal.
pub enum PatchNormal {
    /// Use the reconstruction's stored normal (which is the mean viewing
    /// direction); falls back to recomputing it if zero/degenerate.
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
}

/// How to reduce a per-view quantity across a point's observing views.
pub enum ViewReduce { Min, Max, Median, Mean }
```

`SfmrReconstruction` exposes `positions`, `estimated_normals`, cameras and poses,
so `from_reconstruction` is the bridge from a solved model to a patch cloud.

> _Status (2026-06-11): Implemented — `patch_cloud.rs`
> (`OrientedPatch`, `PatchCloud::from_reconstruction`, `PatchNormal`,
> `PatchExtent`, `mean_viewing_normal`, `pca_plane_normal`), `WarpMap::from_patch`,
> PyO3 bindings (`OrientedPatch`, `PatchCloud`, `WarpMap.from_patch`). The
> reconstruction's stored normals are exactly the mean viewing direction; a
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

Mirror the Rust API in `py_warp_map.rs` / a new `py_patch_cloud.rs`:

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

- **Normal source and quality.** `estimated_normals` may be noisy or absent for
  some points. Fallbacks: orient the patch fronto-parallel to a reference view,
  or to the mean viewing direction of the observing cameras. Worth a quality flag
  per patch.
- **Extent selection.** `PatchExtent` lists three policies; which is the right
  default for visualization vs. matching is unresolved. `PixelRadiusInView`
  reproduces today's "constant pixels" behavior; `RelativeToSpacing` is more
  scene-adaptive.
- **In-plane rotation default.** `up_hint` = world up is simple but degenerate for
  near-horizontal patches; per-track consistency may prefer the reference
  camera's up. Pick one default, allow override.
- **Occlusion.** `from_patch` ignores visibility. A patch may project into a view
  where it is actually occluded; consumers that care (e.g. photometric checks)
  must filter separately.
- **GPU / batching.** Building one small `R×R` map per (patch, view) is cheap, but
  a whole `PatchCloud × views` sweep could be batched; deferred until a caller
  needs it.
