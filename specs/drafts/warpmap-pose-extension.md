# WarpMap extension: pose-aware construction

**Status:** Implemented.

## Motivation

`WarpMap::from_cameras(src, dst)` builds a dense pixel-to-pixel map between two
cameras under the assumption that both observe the same ray through the world.
That's correct when both cameras share a world-space pose (the canonical
undistortion / re-projection use case) — but it's wrong whenever the two
cameras have different poses and we want the map to reflect the scene that
lives between them.

Two new construction paths are provided:

1. **Rotation-aware (at infinity).** "For every destination pixel, take its ray
   in dst-space, rotate it into src-space, call `src.ray_to_pixel(ray)`."
   Models the limit where the scene is infinitely far and only the rotation
   between the two cameras matters.

2. **Pose-and-depth-aware.** "For every destination pixel, take its ray in
   dst-space, trace it to a point at radial distance `r` from the dst camera
   center (expressed in world coordinates), transform that point into
   src-camera coordinates, project." Models a sphere of radius `r` around the
   dst camera — pixels on that sphere land exactly where the pose-aware map
   says they do.

Both paths reuse the existing `ray_to_pixel` machinery and the existing
`RigidTransform` type; neither requires new camera-model arithmetic. They
generalize `from_cameras`.

## API

The new methods live on `crates/sfmtool-core/src/warp_map.rs`. Rotations and
poses use the same types the rest of the codebase already uses: `RotQuaternion`
for rotations and `RigidTransform` for world-to-camera poses, matching the
convention in `SfmrImage::{quaternion_wxyz, translation_xyz}`.

```rust
impl WarpMap {
    /// Rotation-only construction. For each dst pixel center,
    ///   d_dst = dst.pixel_to_ray(u, v)
    ///   d_src = rot_src_from_dst * d_dst
    ///   (sx, sy) = src.ray_to_pixel(d_src)
    ///
    /// Equivalent to assuming the scene is infinitely far: only the relative
    /// rotation between the two cameras affects the projection.
    pub fn from_cameras_with_rotation(
        src: &CameraIntrinsics,
        dst: &CameraIntrinsics,
        rot_src_from_dst: &RotQuaternion,
    ) -> Self;

    /// Full pose + depth construction.
    ///
    /// Implemented as a single 3x3 matrix multiply and vector add per dst
    /// pixel:
    ///   d_dst = dst.pixel_to_ray(u, v)                    // unit, dst-cam frame
    ///   p_src = R_sd * (depth * d_dst) + T_sd             // src-cam frame
    ///   (sx, sy) = src.ray_to_pixel(p_src)
    /// where `R_sd = R_sw * R_dw^T` and `T_sd = t_sw - R_sd * t_dw`. This is
    /// the exact formulation, not a small-angle approximation.
    ///
    /// `src_from_world` and `dst_from_world` are world-to-camera extrinsics,
    /// matching `SfmrImage::{quaternion_wxyz, translation_xyz}`.
    ///
    /// `depth` is the radial distance from the dst camera center along the
    /// dst ray. Passing `f64::INFINITY` short-circuits to the
    /// `from_cameras_with_rotation` path (the only pose component that still
    /// matters is the relative rotation).
    pub fn from_cameras_with_pose(
        src: &CameraIntrinsics,
        dst: &CameraIntrinsics,
        src_from_world: &RigidTransform,
        dst_from_world: &RigidTransform,
        depth: f64,
    ) -> Self;
}
```

Both constructors share a single implementation helper
(`build_with_pose_impl`) that iterates dst rows in parallel via rayon,
matching `from_cameras`. The impl uses the collapsed
`p_src = R_sd * p_dst + T_sd` form — no quaternion multiplication, no
`inverse()` call, no small-angle approximation — so it's numerically exact
at all baselines.

### Why depth is radial, not Z

A "Z-depth" plane only makes sense for perspective cameras with a well-defined
optical axis. This method must also work when `dst` is equirectangular or
fisheye, where no single Z direction applies — so depth is expressed as radial
distance from the dst camera center along each dst ray (a sphere, not a plane).
For perspective dst the two agree up to a per-pixel `cos(theta)` factor, and
callers who want a fronto-parallel plane can convert.

## Python bindings

Exposed via `crates/sfmtool-py/src/py_warp_map.rs`. Rotations accept a
`RotQuaternion`; poses accept a `RigidTransform` that can be built from the
same `(quaternion_wxyz, translation_xyz)` tuple already stored on
reconstruction images:

```python
from sfmtool._sfmtool import WarpMap, RigidTransform

src_from_world = RigidTransform.from_wxyz_translation(
    recon.quaternions_wxyz[src_idx].tolist(),
    recon.translations[src_idx].tolist(),
)
dst_from_world = RigidTransform.from_wxyz_translation(
    recon.quaternions_wxyz[dst_idx].tolist(),
    recon.translations[dst_idx].tolist(),
)
warp = WarpMap.from_cameras_with_pose(
    src=src_camera, dst=dst_camera,
    src_from_world=src_from_world,
    dst_from_world=dst_from_world,
    depth=scene_radius,
)
```

`compute_svd()`, `remap_bilinear()`, `remap_aniso()`, and `to_numpy()` work
unchanged on maps built via either new constructor — the Jacobian is estimated
from the map via central differences and is agnostic to how the map was built.

## Testing

Rust unit tests (`crates/sfmtool-core/src/warp_map.rs` tests module):

- `from_cameras_with_rotation_identity_matches_from_cameras` — identity
  rotation recovers `from_cameras`, pinhole case.
- `from_cameras_with_rotation_identity_matches_from_cameras_equirect` —
  same invariance on an equirect→fisheye pair (ray path).
- `from_cameras_with_pose_infinity_matches_rotation_only` — `depth=INF`
  short-circuits to the rotation-only path built from `R_sd`.
- `from_cameras_with_pose_coincident_pose_matches_from_cameras` — when src
  and dst share the same world pose, the pose-aware map equals
  `from_cameras`.
- `from_cameras_with_pose_known_depth_synthetic_sphere` — for a
  spot-checked sample of dst pixels, the warp is compared against the
  hand-computed reprojection of `depth * d_dst` through the full pose
  chain. Max error <1e-2 px.
- `from_cameras_with_pose_baseline_comparable_to_depth` — at
  baseline/depth = 0.3, the exact formulation and the rotation-only
  approximation disagree by tens of pixels; the exact map also matches a
  hand-computed expected value at the centre pixel to within 1 px.
- `from_cameras_with_pose_svd_still_works` — `compute_svd()` succeeds on a
  pose-built map.
- `from_cameras_with_pose_equirect_dst` — pose-aware path handles
  equirectangular destinations.

Python tests (`tests/test_warp_map_pose.py`):

- Mirror the Rust identity / INF / coincident / known-depth / equirect
  tests, through the PyO3 surface.
- **`TestRealReconstruction::test_per_point_reprojection`** — uses the
  session-scoped `sfmrfile_reconstruction_with_17_images` fixture to load
  a real seoul_bull .sfmr. For ~10 reconstructed 3D points co-visible in
  image 0 and image 1, computes `(u_A, v_A)` (projection in src) and
  `(u_B, v_B, r)` (projection + radial distance in dst), builds a
  per-point warp at `depth=r`, and verifies the warp evaluated at
  `(u_B, v_B)` bilinearly samples to `(u_A, v_A)` within <1 px. This is
  the strongest test: it validates the full world-coordinate chain
  end-to-end against real reconstructed geometry.
- **`TestRealReconstruction::test_rotation_only_vs_pose_diverge_for_nearby_scene`**
  — on the same real reconstruction, at `depth = median(r_i)`, the
  rotation-only approximation disagrees with the exact formulation by
  >10 px maximum across the frame (in practice ~44 px). Confirms the
  exact formulation is not a silent identity under real-world poses.
- **`TestRealReconstruction::test_remap_real_image`** — builds a
  pose-aware warp between two real images using scene-median radial depth
  and runs `remap_bilinear` on the source JPEG. Asserts more than 50% of
  the warped frame falls within the source bounds.
- **`TestRealReconstruction::test_equirect_destination_from_pinhole`** —
  uses a real SIMPLE_RADIAL pinhole source and a 720x360 equirectangular
  destination co-located with the source pose, verifying that the forward
  patch of the equirect frame samples validly from the source image. This
  exercises the `fisheye/equirect needs_ray_path` branch on real data.

Run with:

```
pixi run cargo test -p sfmtool-core --lib warp_map   # Rust
pixi run test -- tests/test_warp_map_pose.py         # Python
```

## Implementation notes and adjustments from the draft

- The spec initially described the pose-aware path as three discrete
  operations (`dst_from_world.inverse().transform()` then
  `src_from_world.transform()`). The implementation collapses this into
  a single `R_sd * p_dst + T_sd` with precomputed `R_sd` and `T_sd`,
  which is faster (one mat-vec, one vec-add per pixel, no per-pixel
  inverse) and numerically identical.
- For `depth = INFINITY` the implementation detects the non-finite depth
  and dispatches to the rotation-only path with the precomputed `R_sd`,
  instead of relying on IEEE arithmetic to cancel the translation cleanly
  (which wouldn't happen — `inf * 0 = NaN`).
- The dst camera bounds are always validated via `ray_to_pixel` plus a
  subsequent `[0, src_w) × [0, src_h)` check, matching the `from_cameras`
  semantics. Rays projecting behind a perspective src camera return `None`
  from `ray_to_pixel` and therefore become `NaN` in the warp map.
- `pixel_to_ray` (not `unproject`) is used on the dst side so
  equirectangular and fisheye destinations work correctly. The existing
  `from_cameras` selects between the image-plane path and the ray path
  based on `needs_ray_path`; the new constructors always use the ray
  path (which is a superset).
- The PyO3 method signatures use keyword-only style
  (`from_cameras_with_pose(src=…, dst=…, src_from_world=…, …)`) to match
  the existing `WarpMap.from_cameras(src=…, dst=…)` convention.

## Non-goals

- No automatic derivation of the pose from reconstruction format metadata.
  That stays in the Python wrapper (e.g. the future `sfmtool.equirect`
  module).
- No per-dst-pixel depth map. For varying-per-pixel depths the caller
  rebuilds the map or uses a depth-aware resampler (future work).

## Status / dependencies

Implemented against `WarpMap`, `RigidTransform`, and `ray_to_pixel`, all of
which ship per `specs/core/image-warping.md`. No new camera-model arithmetic.
Unblocks the main `equirect-rendering.md` algorithm.
