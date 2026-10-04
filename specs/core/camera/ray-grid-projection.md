# Affine Ray-Grid Projection for Patch Warps

## Purpose

Painting a small square of a 3D surface into a photograph means working out,
for every sample of that square, where it lands in the image. The square is
planar, so the map from its own coordinates to rays in the camera's frame is
affine: the whole grid of rays follows from three vectors, and the camera model
only ever has to turn rays into pixels. This spec describes that split — a
model-free geometry stage owned by the patch warp, and a camera-owned stage that
projects a whole ray grid at once — and the bound on the coarse-grid shortcut
the expensive camera models take.

## The seam

`WarpMap::from_patch` builds, for one (patch, view), the `r×r` grid of
source-image coordinates where the patch's `(s, t) ∈ [-1, 1]²` samples project.
Running the whole chain per sample — build the homogeneous corner, apply the
pose, then `ray_to_pixel` — costs one pose multiply per output pixel, and
`RigidTransform::transform_point_homogeneous` also rebuilds the rotation matrix
from the quaternion on every call. The split
below removes that: the pose is applied to three vectors once per patch, and
only the camera projection runs per pixel.

The patch → camera-frame map is **affine in `(s, t)`**. Write the patch plane as
`P(s,t) = C + s·U + t·V`, where `U = h_u·u_axis` and `V = −h_v·v_axis` are the
scaled in-plane axes (`h_u`, `h_v` the half-extents; `V` points along `−v_axis`
because the raster row counts downward, which renders the front face
un-mirrored). With the pose `x ↦ R·x + T` and the patch weight `w`,

```
Q(s,t) = q0 + s·qu + t·qv,   q0 = R·C + w·T,  qu = R·U,  qv = R·V
```

so the entire pose multiply collapses to **three precomputed vectors**. Re-expressed
on the integer grid (`s = (col+0.5)·step − 1`, `step = 2/r`), the camera-frame ray at
node `(col, row)` is `origin + col·col_step + row·row_step`. Points at infinity
(`w = 0`) need no special case: the weight only enters `q0`, dropping the translation.

This is the boundary the design keeps:

- **Geometry (caller, `from_patch`)** — builds `(origin, col_step, row_step)` from
  plane + pose. No camera-model branching; works for every model and for infinity.
- **Projection (camera, `ray_to_pixel_grid`)** — turns the affine ray grid into
  source pixels, owning all model-specific math (divide, distortion, intrinsics,
  validity domain). The homography that some texts expose is **not** surfaced: it
  depends on the patch + pose (not just the camera), so it stays an internal
  detail of the perspective branch rather than camera state.

Invalid nodes (behind the camera, outside the distortion model's invertible
domain, or outside the image rectangle) are written `(NaN, NaN)`, the same result
as `ray_to_pixel` followed by an in-frame test.

## `CameraIntrinsics::ray_to_pixel_grid`

The projection stage lives in
[ray_grid.rs](../../../crates/sfmtool-core/src/camera/distortion/ray_grid.rs);
the geometry stage that feeds it is `WarpMap::from_patch` in
[warp_map.rs](../../../crates/sfmtool-core/src/camera/warp_map.rs).

```rust
pub fn ray_to_pixel_grid(
    &self,
    origin: [f64; 3], col_step: [f64; 3], row_step: [f64; 3],
    cols: u32, rows: u32, out: &mut [f32],   // interleaved (sx, sy), len 2·cols·rows
)
```

Two paths, chosen by model and grid size:

- **Perspective** (`!needs_ray_path`) — **exact**. Every node is projected
  (`ray_to_pixel_grid_exact`); the win is purely that the affine basis removed the
  per-pixel pose multiply, and the divide + distortion are cheap. Bit-for-bit equal
  to scalar `ray_to_pixel` per node (test `ray_to_pixel_grid_perspective_matches_scalar`).
- **Fisheye / equirectangular** (`needs_ray_path`) — **bounded coarse-grid**. The
  per-node projection (`atan2`/`asin`) is expensive but spatially smooth, so the
  exact projection is evaluated on a sub-grid (stride `COARSE_GRID_STRIDE = 8`) and
  the interior is bilinearly interpolated. A grid with fewer than
  `2·COARSE_GRID_STRIDE` columns or rows is too small to repay the sub-grid
  setup and takes the exact path instead.

### Implementation notes

- **Intrinsics are read once per grid.** The scalar `ray_to_pixel` matches on the
  camera-model enum for its intrinsics on every call; both grid paths read them
  once and project each node with an inlined copy of `ray_to_pixel`.
- **The caller owns parallelism.** A grid is one patch-sized tile and
  `ray_to_pixel_grid` is called inside a per-patch `par_iter`, so both paths run
  sequentially rather than nesting rayon over a few dozen rows.
- **The coarse path writes every pixel exactly once.** It walks sub-grid cells,
  and each cell fills a half-open pixel block; the last cell on each axis also
  owns the final node row and column. Each sub-grid node is projected once and
  shared by the cells around it.
- **The probe predicts the fill exactly.** The acceptance probe and the fill use
  the same bilinear helper, so a value the probe accepts is the value written.

## Bounding the coarse-grid approximation

Accuracy does **not** depend on the stride. Each sub-grid cell is accepted for
interpolation only after a per-cell **probe**: its center and four edge-midpoints —
the points where bilinear interpolation of a separable-quadratic warp is least
accurate — are projected exactly and compared to the interpolant. If any probe
deviates by more than `COARSE_GRID_TOL_PX = 0.02` source pixels (or any of the four
cell corners is invalid), the cell is **demoted to exact** per-pixel projection.
The stride therefore trades only speed (how many cells qualify), never correctness;
a higher-curvature or peripheral tile simply falls back to exact where needed.

The tolerance is set an order of magnitude below the sub-pixel accuracy the
keypoint localizer needs, so a coarse-grid warp is indistinguishable from the
exact one at that scale.

The test `coarse_grid_error_within_bound` checks the bound over a sweep of the
`needs_ray_path` models (SimpleRadialFisheye, RadialFisheye, OpenCVFisheye,
Equirectangular) that mixes realistic small
tiles with wide-angle, depth-tilted ones. It requires that some cells take the
interpolated path, that the worst error against the exact map stays below
`2·COARSE_GRID_TOL_PX` (pixels inside an accepted cell but away from its probe
points can land slightly above the tolerance), and that fewer than 1% of pixels
disagree on validity.

The probe bounds the warp *position*, not its derivative, so a second test
(`coarse_grid_jacobian_degradation`) guards the central-difference Jacobian that
`compute_svd`/`compute_jacobians` feed to the anisotropic sampler and the GN
gradient. Over a sweep of the three fisheye models it requires a worst-case relative Jacobian
error below 1%, a relative RMS error below 0.5%, a major-axis direction error
below 1°, and no pixel crossing the `MAX_ANISOTROPY` clamp. The piecewise-bilinear
seams at stride boundaries do not degrade the Jacobian: central differencing
averages across the slope change, and the test requires the seam pixels' RMS error
to stay within twice that of cell interiors.
