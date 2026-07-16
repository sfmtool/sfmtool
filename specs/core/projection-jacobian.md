# Projection Jacobian (ray-to-pixel derivatives)

**Status:** Implemented (perspective family) —
`crates/sfmtool-core/src/camera/distortion.rs`
(`CameraIntrinsics::ray_to_pixel_with_jacobian`, `CameraModel::distort_jacobian`)
and `crates/sfmtool-core/src/camera/intrinsics.rs`
(`CameraModel::supports_pixel_jacobian`); tests in
`camera/distortion/tests.rs`. Core Rust only — no Python binding yet, as the
current consumer is the native pose refinement (see
[absolute-pose.md](absolute-pose.md)).

## Purpose

The analytic derivative of the forward projection, so gradient-based
optimizers over pose or structure (pose-only resection refinement, bundle
adjustment) stop finite-differencing through the camera model. A finite
difference costs one projection per parameter per step and carries a
step-size error that is worst exactly where distortion curves hardest; the
analytic Jacobian is exact, one pass, and cannot drift from the projection at
run time.

## Definitions

- Camera-frame ray / point in the canonical convention (the camera looks along
  `−Z`; a point in front has `z < 0`). The projection is scale-invariant in the
  ray, so the derivative with respect to the supplied ray components **is** the
  derivative with respect to a camera-frame point when one is passed directly.
- Pixel `(u, v)` from `ray_to_pixel`.
- The **projection Jacobian** `∂(u, v)/∂ray`, a 2×3 returned row-major
  `[[∂u/∂x, ∂u/∂y, ∂u/∂z], [∂v/∂x, ∂v/∂y, ∂v/∂z]]`.

## Scope

Perspective models only — pinhole, `SimpleRadial`, `Radial`, `OpenCV`,
`FullOpenCV` — reported by `CameraModel::supports_pixel_jacobian`. Fisheye and
equirectangular models take the ray path in `distort_ray` and have no analytic
Jacobian here yet; a caller that needs one for those falls back to a finite
difference. A caller checks `supports_pixel_jacobian` once per camera to choose
the analytic or fallback path.

## API

```rust
/// Pixel (u, v) plus the 2×3 ∂(u, v)/∂ray, row-major.
pub type PixelJacobian = ((f64, f64), [[f64; 3]; 2]);

impl CameraIntrinsics {
    pub fn ray_to_pixel_with_jacobian(&self, ray: [f64; 3]) -> Option<PixelJacobian>;
}

impl CameraModel {
    pub fn supports_pixel_jacobian(&self) -> bool;   // perspective family
}
```

`ray_to_pixel_with_jacobian` returns `None` on exactly the domain where
`ray_to_pixel` does — the ray behind the camera or outside the distortion
polynomial's invertible branch — and also for an unsupported model. The pixel
it returns equals `ray_to_pixel`'s.

## Mechanism

The forward map is `pixel = K ∘ distort ∘ divide ∘ S`, so the Jacobian is the
product of those stages' derivatives by the chain rule:

```
∂(u, v)/∂ray  =  diag(fx, fy) · D · (P · S)
     (2×3)          (2×2)      (2×2)  (2×3)
```

- **Frame flip** `S = diag(1, −1, −1)` maps the canonical ray to the optical
  frame the distortion kernels use.
- **Perspective divide** `P = ∂(x, y)/∂(rx, ry, rz)` for `(x, y) = (rx/rz,
  ry/rz)`; combined with `S` it is
  `[[1/rz, 0, rx/rz²], [0, −1/rz, ry/rz²]]`.
- **Distortion** `D = ∂(x_d, y_d)/∂(x, y)` (`CameraModel::distort_jacobian`).
  Every perspective model is `x_d = x·g(r²) + T_x`, `y_d = y·g(r²) + T_y` with
  radial factor `g`, `r² = x² + y²`, and tangential
  `T_x = 2 p1 x y + p2 (r² + 2x²)`, `T_y = p1 (r² + 2y²) + 2 p2 x y`. With
  `g' = dg/d(r²)` the 2×2 is
  `[[g + 2x²g' + 2p1 y + 6p2 x, c], [c, g + 2y²g' + 6p1 y + 2p2 x]]`, shared
  off-diagonal `c = 2xy g' + 2p1 x + 2p2 y`. `g` is `1` (pinhole), a radial
  polynomial (`SimpleRadial`/`Radial`/`OpenCV`), or a rational
  `(1 + k1 r² + k2 r⁴ + k3 r⁶)/(1 + k4 r² + k5 r⁴ + k6 r⁶)` (`FullOpenCV`).
- **Intrinsics** `K` scales the rows by `(fx, fy)`.

The derivative is with respect to the **ray** only; poses and 3D points
differentiate on top of this 2×3 in the caller (e.g. resection composes it with
`∂(R·X + t)/∂pose`).

## Testing requirements

- **Central-difference agreement**: across every perspective model and a wide
  sweep of ray directions (in-image pixels back-projected to rays) and depths,
  the analytic 2×3 matches a central difference of `ray_to_pixel` within
  tolerance. This is the primary correctness pin and the regression guard for
  both the projection math and the Jacobian.
- **Domain**: a ray behind the camera returns `None`, matching `ray_to_pixel`.
- **Scope**: fisheye and equirectangular models report
  `supports_pixel_jacobian() == false` and return `None`.

## Non-goals

- Fisheye / equirectangular analytic Jacobians — deferred; callers finite-
  difference for those.
- Derivatives with respect to intrinsics or distortion coefficients — this is
  the derivative with respect to the ray only.
- Any optimizer or normal-equation assembly; this is the measurement
  derivative a solver consumes.
