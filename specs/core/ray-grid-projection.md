# Affine Ray-Grid Projection for Patch Warps

_Status: **landed** (branch `profile-embed-patches`). Splits
[`WarpMap::from_patch`](image-warping.md) into a model-free geometry stage and a
camera-owned projection stage, [`CameraIntrinsics::ray_to_pixel_grid`], so the
per-pixel cost of rendering a patch into a view drops sharply. Motivated by
profiling `sfm embed-patches`, where `from_patch` (via `render_context` in
[patch-keypoint-localization.md](patch-keypoint-localization.md) and
`cache_prerender` in [patch-normal-refinement.md](patch-normal-refinement.md))
was the single largest cost. Code in
`sfmtool-core/src/camera/distortion.rs` and `src/camera/warp_map.rs`._

## Problem

`WarpMap::from_patch` builds, for one (patch, view), the `r×r` grid of source-image
coordinates where the patch's `(s, t) ∈ [-1, 1]²` samples project. The original
implementation ran the **whole** projection chain per output pixel:

```
(s,t) → corner_homogeneous → R·x + t  (pose) → ray_to_pixel (divide + distort + K)
```

This is `r²` projections per render and, on `dino_dog_toy` (perspective,
SIMPLE_RADIAL, 48×48 tiles, ~1.04M renders in localization alone), it dominated
the phase — `render_project` was **52% of `localize_total`**. A hidden multiplier
made it worse: `RigidTransform::transform_point_homogeneous` rebuilds the rotation
**matrix from the quaternion on every call**, so the pose conversion ran once per
pixel (≈2,300× per render).

## The seam

The patch → camera-frame map is **affine in `(s, t)`**. With `P(s,t) = C + s·U + t·V`
(scaled in-plane axes) and `Q = R·P + t·w`,

```
Q(s,t) = q0 + s·qu + t·qv,   q0 = R·C + t·w,  qu = R·U,  qv = R·V
```

so the entire pose multiply collapses to **three precomputed vectors**. Re-expressed
on the integer grid (`s = (col+0.5)·step − 1`, `step = 2/r`), the camera-frame ray at
node `(col, row)` is `origin + col·col_step + row·row_step`. Points at infinity
(`w = 0`) are uniform: the weight only enters `q0`, dropping the translation.

This is the boundary the design keeps:

- **Geometry (caller, `from_patch`)** — builds `(origin, col_step, row_step)` from
  plane + pose. No camera-model branching; works for every model and for infinity.
- **Projection (camera, `ray_to_pixel_grid`)** — turns the affine ray grid into
  source pixels, owning all model-specific math (divide, distortion, intrinsics,
  validity domain). The homography that some texts expose is **not** surfaced: it
  depends on the patch + pose (not just the camera), so it stays an internal
  detail of the perspective branch rather than camera state.

Invalid nodes (behind the camera, outside the distortion model's invertible
domain, or outside the image rectangle) are written `(NaN, NaN)`, identical to
`ray_to_pixel` + the in-frame test the warp callers applied before.

## `CameraIntrinsics::ray_to_pixel_grid`

```rust
pub fn ray_to_pixel_grid(
    &self,
    origin: [f64; 3], col_step: [f64; 3], row_step: [f64; 3],
    cols: u32, rows: u32, out: &mut [f32],   // interleaved (sx, sy), len 2·cols·rows
)
```

Two paths, chosen by model:

- **Perspective** (`!needs_ray_path`) — **exact**. Every node is projected
  (`ray_to_pixel_grid_exact`); the win is purely that the affine basis removed the
  per-pixel pose multiply, and the divide + distortion are cheap. Bit-for-bit equal
  to scalar `ray_to_pixel` per node (test `ray_to_pixel_grid_perspective_matches_scalar`).
- **Fisheye / equirectangular** (`needs_ray_path`) — **bounded coarse-grid**. The
  per-node projection (`atan2`/`asin`) is expensive but spatially smooth, so the
  exact projection is evaluated on a sub-grid (stride `COARSE_GRID_STRIDE = 8`) and
  the interior is bilinearly interpolated.

### Implementation notes

- **Hoisted intrinsics.** Both paths build a `GridProj` (`fx, fy, cx, cy, w, h`)
  once per grid via `grid_proj()` and pass it to `project_ray_node`, which inlines
  `ray_to_pixel` (`model.distort_ray` + the affine + the in-frame test). The scalar
  `ray_to_pixel` re-fetches the (enum-match) intrinsics per call; the grid paths
  fetch them once, not once per node.
- **Sequential exact path.** `ray_to_pixel_grid_exact` renders one patch-sized tile
  and is always called inside a per-patch `par_iter`, so it is sequential — the
  caller owns parallelism (avoids nesting rayon over ~48 rows).
- **Cell-blocked coarse path.** `ray_to_pixel_grid_coarse` walks cells, not pixels:
  each cell fetches its four corners and makes the interpolate-or-exact decision
  once, then fills its half-open pixel block (`[r0,r1) × [c0,c1)`; the last cell on
  each axis also owns the final node row/column, so every pixel is written once).
  Sub-grid node positions are computed arithmetically (`min(i·stride, n−1)`, with
  `ceil((n−1)/stride)` cells) rather than materialized, so the only heap allocation
  is the `node_px` cache (each node projected once, shared by adjacent cells). The
  same `bilerp` helper serves the probe (acceptance) and the fill, so the probe
  predicts the emitted value exactly.

## Bounding the coarse-grid approximation

Accuracy does **not** depend on the stride. Each sub-grid cell is accepted for
interpolation only after a per-cell **probe**: its center and four edge-midpoints —
the points where bilinear interpolation of a separable-quadratic warp is least
accurate — are projected exactly and compared to the interpolant. If any probe
deviates by more than `COARSE_GRID_TOL_PX = 0.02` source pixels (or any of the four
cell corners is invalid), the cell is **demoted to exact** per-pixel projection.
The stride therefore trades only speed (how many cells qualify), never correctness;
a higher-curvature or peripheral tile simply falls back to exact where needed.

Measured over a fisheye sweep mixing realistic small tiles with aggressive
wide-angle, depth-tilted ones (`coarse_grid_error_within_bound`,
SimpleRadialFisheye / RadialFisheye / OpenCVFisheye):

| metric | value |
|---|---|
| max source-pixel error vs exact | **0.0197 px** (≈ tol) |
| RMS error | 0.0043 px |
| validity disagreement | 0 / 48,384 px |
| cells interpolated | 29.6% (≈all on gentle tiles, ~none on wide-angle) |

The worst-case error tracks the tolerance because the probe is the bound's
enforcement, not a heuristic.

The probe bounds the warp *position*, not its derivative, so a second test
(`coarse_grid_jacobian_degradation`) guards the central-difference Jacobian that
`compute_svd`/`compute_jacobians` feed to the anisotropic sampler and the GN
gradient. Measured over the same fisheye sweep (50,784 px): Jacobian error
≤ 0.29% worst / 0.04% RMS, σ-major ≤ 0.22%, **0** crossings of the `MAX_ANISOTROPY`
clamp (fisheye warps are near-conformal, ≤ 1.9× anisotropy observed). The
piecewise-bilinear seams at stride boundaries are harmless: central differencing
averages across the slope discontinuity, so seam pixels are actually marginally
*more* accurate than cell interiors (0.77× the error).

## Impact (measured, `sfm embed-patches`)

Both `from_patch` consumers benefit (shared primitive: localization's
`render_context`, refinement's `cache_prerender`, and the `warp_from_patch` leaf).

**dino_dog_toy** (perspective, 85 imgs, 55,701 pts):

| | before | after |
|---|--:|--:|
| `render_project` (localize, CPU) | 921.9 s | **52.2 s** (~18×) |
| localize wall | 55.4 s | **29.4 s** |
| `cache_prerender` (refine, CPU) | 287.2 s | **75.5 s** (3.8×) |
| refine wall | 14.8 s | **8.0 s** |
| **embed-patches total wall** | ~99.6 s | **52.1 s** (1.9×) |

**kerry_park** (fisheye rig, coarse-grid path):

| | before | after |
|---|--:|--:|
| `render_context` (localize, CPU) | 21.5 s | **6.3 s** |
| localize wall | 0.95 s | **0.33 s** (2.8×) |
| `cache_prerender` (refine, CPU) | 5.5 s | **1.6 s** |

After this change, localization is no longer projection-bound: `render_remap`,
`loo_template`, and `search_shift` become the co-dominant terms (~25–29% each on
dino). A `pixel_to_ray_grid` sibling — the undistort direction — would accelerate
`render_remap` and the rectification/undistort consumers next; it can reuse the
same coarse-grid-with-probe helper.
