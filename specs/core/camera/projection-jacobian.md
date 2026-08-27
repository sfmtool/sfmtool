# Projection Jacobian (ray-to-pixel derivatives)

**Status:** Implemented (perspective family — `SfmtoolPinhole` included —
plus `EquidistantFisheye`, `SimpleRadialFisheye` and `SfmtoolFisheye`) —
`crates/sfmtool-core/src/camera/distortion.rs`
(`CameraIntrinsics::ray_to_pixel_with_jacobian`, `CameraModel::distort_jacobian`),
`crates/sfmtool-core/src/camera/distortion/kernels.rs`
(`radial_fisheye_ray_jacobian`, `sfmtool_fisheye_ray_jacobian`,
`sfmtool_pinhole_radial_factor`) and
`crates/sfmtool-core/src/camera/intrinsics.rs`
(`CameraModel::supports_pixel_jacobian`); tests in
`camera/distortion/tests.rs`. Core Rust only — no Python binding yet, as the
current consumer is the native pose refinement (see
[absolute-pose.md](../geometry/absolute-pose.md)).

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

Two families, both reported by `CameraModel::supports_pixel_jacobian`:

- the perspective models — pinhole, `SimpleRadial`, `Radial`, `OpenCV`,
  `FullOpenCV`, `SfmtoolPinhole` — which project through the image plane;
- the θ-map fisheyes `EquidistantFisheye`, `SimpleRadialFisheye` and
  `SfmtoolFisheye`, whose ray-path map is a closed form in `θ` alone —
  `θ_d = θ·(1 + k1·θ²)` for the first two (`k1 = 0` for the first),
  `θ_d = θ + δ(θ)` with `δ` a spline for the third
  ([../../formats/sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md)) — and so
  differentiates in closed form at every `θ`, `θ ≥ 90°` included.

The multi-coefficient fisheye models and equirectangular take the ray path
with no analytic derivative here; a caller that needs one for those falls back
to a finite difference. A caller checks `supports_pixel_jacobian` once per camera to
choose the analytic or fallback path.

## API

```rust
/// Pixel (u, v) plus the 2×3 ∂(u, v)/∂ray, row-major.
pub type PixelJacobian = ((f64, f64), [[f64; 3]; 2]);

impl CameraIntrinsics {
    pub fn ray_to_pixel_with_jacobian(&self, ray: [f64; 3]) -> Option<PixelJacobian>;
}

impl CameraModel {
    pub fn supports_pixel_jacobian(&self) -> bool;
}
```

`ray_to_pixel_with_jacobian` returns `None` on exactly the domain where
`ray_to_pixel` does — the ray behind the camera or outside the distortion
polynomial's invertible branch — and also for an unsupported model. The pixel
it returns equals `ray_to_pixel`'s. One documented exception: `θ = π` exactly
(`r_xy = 0` behind the camera) under the equidistant pair, where the
projection is defined but its derivative is unbounded — see "Equidistant map"
below.

## Mechanism — perspective family

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
  polynomial (`SimpleRadial`/`Radial`/`OpenCV`), a rational
  `(1 + k1 r² + k2 r⁴ + k3 r⁶)/(1 + k4 r² + k5 r⁴ + k6 r⁶)` (`FullOpenCV`), or
  the radial spline `1 + δ(ρ)/ρ` at `ρ = √(r²)` with
  `g' = (ρ·δ'(ρ) − δ(ρ))/(2ρ³)` (`SfmtoolPinhole` —
  [sfmtool-pinhole-kernels.md](sfmtool-pinhole-kernels.md)).
- **Intrinsics** `K` scales the rows by `(fx, fy)`.

The derivative is with respect to the **ray** only; poses and 3D points
differentiate on top of this 2×3 in the caller (e.g. resection composes it with
`∂(R·X + t)/∂pose`).

## Mechanism — equidistant map

Neither `EquidistantFisheye` nor `SimpleRadialFisheye` ever forms
`(rx/rz, ry/rz)`, so the perspective divide — and with it the `rz > 0` domain
restriction — does not appear. In the optical frame, with `ρ = √(rx² + ry²)`,
`n² = ρ² + rz²`, unit 2D direction `(ux, uy) = (rx, ry)/ρ` and
`θ = atan2(ρ, rz)`, the map is `(x_d, y_d) = θ_d·(ux, uy)` with
`θ_d = θ·(1 + k1·θ²)` (`k1 = 0` is the distortion-free model, and the shared
kernel reproduces it bit for bit). `SfmtoolFisheye` substitutes
`θ_d = θ + δ(θ)` and `θ_d' = 1 + δ'(θ)` into the same template, both read from
one spline evaluation; everything below holds unchanged for it. Writing
`θ_d' = dθ_d/dθ = 1 + 3·k1·θ²`, its derivative follows from

```
∂θ/∂rx = ux·rz/n²    ∂θ/∂ry = uy·rz/n²    ∂θ/∂rz = −ρ/n²
∂ux/∂rx = uy²/ρ      ∂ux/∂ry = −ux·uy/ρ   (mirrored for uy)
```

giving, with the shared off-diagonal factor `c = θ_d'·rz/n² − θ_d/ρ`,

```
∂(x_d, y_d)/∂(rx, ry, rz) =
    [[θ_d·uy²/ρ + θ_d'·ux²·rz/n²,  ux·uy·c,                     −θ_d'·rx/n²],
     [ux·uy·c,                     θ_d·ux²/ρ + θ_d'·uy²·rz/n²,  −θ_d'·ry/n²]]
```

`K` and the frame flip `S` compose on top exactly as above. Every expression is
finite for any `rz`, which is the point: the periphery past 90° is where a
fisheye carries its model information, so no in-front guard applies.

Two limits on the optical axis, where `(ux, uy)` is undefined:

- **In front** (`ρ → 0`, `rz > 0`): `θ_d/ρ → 1/rz`, `θ_d' → 1` and `c → 0`,
  so the matrix
  tends to `diag(1/rz, 1/rz)` with a zero third column — the pinhole
  small-angle Jacobian, independent of the direction the limit is taken from.
  This is the value returned on the axis.
- **At the antipode** (`ρ → 0`, `rz < 0`): `θ → π` while `ρ → 0`, so `θ_d/ρ`
  diverges and no finite Jacobian exists — `None`. The projection itself is
  still defined there (it aliases the principal point), so this is the single
  measure-zero direction where the Jacobian's domain is narrower than
  `ray_to_pixel`'s.

## Testing requirements

- **Central-difference agreement**: across every model with an analytic
  Jacobian and a wide sweep of ray directions (in-image pixels back-projected
  to rays) and depths, the analytic 2×3 matches a central difference of
  `ray_to_pixel` within tolerance. This is the primary correctness pin and the
  regression guard for both the projection math and the Jacobian. For the
  equidistant map the sweep covers a whole synthetic sensor plus explicit `θ`
  bands straddling 90°, at both signs of `k1` as well as `k1 = 0`.
- **`k1 = 0` degeneracy**: `SimpleRadialFisheye { k1 = 0 }` returns the
  `EquidistantFisheye` pixel and Jacobian **bit for bit** over a θ sweep —
  the promotion between the two representations moves nothing.
- **Domain**: a ray behind the camera returns `None` for the perspective
  family, matching `ray_to_pixel`; under the equidistant map the same ray
  returns a Jacobian, and only the exact antipode returns `None`.
- **Axis limit**: the equidistant Jacobian on the optical axis equals the
  pinhole small-angle form and is approached continuously from every azimuth.
- **Scale invariance**: the map is degree-0 homogeneous in the ray, so
  `J·r = 0` and `J(s·r) = J(r)/s`.
- **Scope**: the multi-coefficient fisheye models and equirectangular report
  `supports_pixel_jacobian() == false` and return `None`.
- **Shared domain**: a `k1` strong enough to fold `θ_d` non-positive inside
  the sampled field takes projection and derivative out of domain together.

## Non-goals

- Analytic Jacobians for the multi-coefficient fisheye models and
  equirectangular — deferred; callers finite-difference for those.
- Derivatives with respect to intrinsics or distortion coefficients — this is
  the derivative with respect to the ray only. (The bundle adjustment's
  `∂(u, v)/∂f`, `∂(u, v)/∂k1` and `∂(u, v)/∂cᵢ` columns are its own; see
  [bundle-adjustment.md](../geometry/bundle-adjustment.md).)
- Any optimizer or normal-equation assembly; this is the measurement
  derivative a solver consumes.
