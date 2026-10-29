# The `SFMTOOL_PINHOLE` kernels

## Summary

A perspective camera whose radial distortion is a monotone cubic B-spline
correction rather than a polynomial, so a wide lens can be calibrated without a
fitted curve that folds back on itself — these are the kernels that project,
unproject and differentiate it. The model itself — its parameter list and its
serialization — is defined in
[sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md); this spec is
the computation behind it, and covers forward projection, the inverse recovery of the image-plane
radius, the analytic ray Jacobian, enforcement of the monotonicity invariant,
the model's classification flags, and the two corrections one image of the
camera may carry, a radial correction added to the spline and a rolling shutter
applied after it ([Per-image corrections](#per-image-corrections)). Basis evaluation is the shared machinery
described in
[sfmtool-fisheye-kernels.md](sfmtool-fisheye-kernels.md) § "Basis evaluation",
read at the radial coordinate `ρ` on the domain `[0, ρ_max]` instead of `θ` on
`[0, θ_max]`. The bundle adjustment's `opt_bspline` release, which produces the
coefficients, is specified in [bundle-adjustment.md](../geometry/bundle-adjustment.md).

## Forward projection

The kernels live in
[kernels/sfmtool_pinhole.rs](../../../crates/sfmtool-core/src/camera/distortion/kernels/sfmtool_pinhole.rs)
(`distort_sfmtool_pinhole`, `undistort_sfmtool_pinhole`,
`sfmtool_pinhole_radial_factor`, `sfmtool_pinhole_unfolded`), sharing both
`recover_radial_bspline` — which lives with the fisheye sibling in
[kernels/sfmtool_fisheye.rs](../../../crates/sfmtool-core/src/camera/distortion/kernels/sfmtool_fisheye.rs)
— and the B-spline basis in
[bspline.rs](../../../crates/sfmtool-core/src/camera/distortion/bspline.rs);
dispatch is in
[distortion.rs](../../../crates/sfmtool-core/src/camera/distortion.rs) and the
classification arms in
[intrinsics.rs](../../../crates/sfmtool-core/src/camera/intrinsics.rs).

The map is the perspective family's: the optical-frame ray divides by `rz`,
and the quotient's radius **is** the model's radial coordinate,

```
ρ = √(rx² + ry²)/rz        (x, y) = (rx, ry)/rz
```

so the model needs no ray-space entry point of its own.
`distort_sfmtool_pinhole` takes the image-plane point to
`(x_d, y_d) = (ρ + δ(ρ))·(x, y)/ρ`, and the dispatch reaches it through the
same `rz > 0` guard, divide and `distort` call the pinhole and polynomial
models use. A point on the optical axis (`ρ < 1e-15`) passes through
unchanged: `δ(ρ)/ρ → 0` by the gauge, so the correction vanishes there.

The domain restriction is the pinhole's, `rz > 0`, plus the **fold gate**:
`sfmtool_pinhole_unfolded` reports whether `ρ_d = ρ + δ(ρ)` is positive at a
positive `ρ`. A spline that drives it non-positive has left the branch
connected to the origin, and the projection is `None` there, the analog of
[sfmtool-fisheye-kernels.md](sfmtool-fisheye-kernels.md)'s `θ_d ≤ 0` gate. It
is answered inside `forward_projection_invertible`, the perspective family's
branch predicate, which both the projection and the analytic Jacobian consult,
so the two leave the domain together by construction.

The slope half of that predicate for the polynomial models
(`g + 2r²g' > 0`, here `1 + δ'(ρ) > 0`) is this model's monotonicity
**construction** invariant, enforced where splines are produced rather than
probed per ray, so it is not repeated in the gate.

## Inverse

`undistort_sfmtool_pinhole` recovers `(x, y)` from the distorted image-plane
point by inverting `r_d = ρ + δ(ρ)` in `recover_radial_bspline`, then scaling
the distorted direction by `ρ/r_d`. The perspective dispatch turns that into a
ray by normalizing `(x, y, 1)`, so the model shares the family's
`undistort_to_ray` path.

`recover_radial_bspline` is the equation-level solver both spline models call:
its `r_d = d + δ(d)` is coordinate-agnostic, so it inverts `θ` for the fisheye
and `ρ` here, unchanged.

The solver itself — the closed-form inverse on the linear tail past
`r_end = ρ_max + δ(ρ_max)`, the bracket-safeguarded Newton inside
`[0, ρ_max]`, and the cases it reports as non-convergence — is specified in
[sfmtool-fisheye-kernels.md](sfmtool-fisheye-kernels.md) § "Inverse", read
with `ρ` and `ρ_max` in place of `θ` and `θ_max`. What this kernel adds around
it:

- A distorted point within `1e-15` of the axis (`PINHOLE_AXIS_EPS`) is
  returned unchanged before the solver runs, the same guard the forward map
  applies at `ρ < 1e-15`. `recover_radial_bspline`'s own `r_d ≤ 0 → ρ = 0`
  case is therefore never reached from this kernel.
- Where `recover_radial_bspline` returns `converged = false` (a spline folded
  so far that `ρ_max + δ(ρ_max) ≤ 0`, or a radius past `r_end` under a
  non-positive end slope), the kernel answers with the **identity** `ρ = r_d`,
  the base pinhole's inverse, so the distorted point passes through
  unchanged. Taking the reported radius instead would scale every pixel in the
  image onto the optical axis. `sfmtool_fisheye_to_ray` answers the same
  report with its own base model's inverse, `equidistant_to_ray` (`θ = r_d`).
  Both cases need a spline that violates the monotonicity invariant, so they
  are unreachable through a solve, which cannot persist a spline that folded
  ([Monotonicity enforcement](#monotonicity-enforcement)).
- The linear tail carries the periphery of every real image: `ρ` grows without
  bound toward `θ = 90°`, so a calibrated `ρ_max` is crossed well inside the
  frame. That is why the tail follows the fitted end slope rather than
  returning to the base pinhole's slope.

`CameraModel::undistort` carries an **explicit arm** for this model rather than
falling through to the generic fixed-point iteration. That iteration
(`x ← x + (x_d − distort(x))`) contracts only for weak distortion, while the
monotonicity invariant hands this solve a guaranteed bracket at any coefficient
magnitude. The projection pipeline reads this inverse for retriangulation and
direction re-estimation, where a silently mis-converged radius is a wrong ray.

## Ray Jacobian

`CameraModel::supports_pixel_jacobian` is **true**, so pose refinement, patch
sizing and the bundle adjustment take the analytic path.

The derivative is the **perspective family's** composition from
[projection-jacobian.md](projection-jacobian.md),
`diag(fx, fy) · D · (P · S)`, with `D` the 2×2 of a radially symmetric
image-plane map `x_d = x·g(r²)`. This model's forward map is exactly that form,
so the radial factor `g` is the only stage that differs from a pinhole's and
the frame flip, the divide, the domain gate and the intrinsics scaling are the
same code the rest of the family runs. Its fisheye sibling carries a dedicated
kernel instead because that map never forms `(rx/rz, ry/rz)` at all, so none of
the composition applies to it.

`sfmtool_pinhole_radial_factor` supplies the pair the family is parameterized
by. The family reads `g` as a function of `r²`, so the chain rule through
`ρ = √(r²)` (`dρ/d(r²) = 1/(2ρ)`) gives

```
g(ρ) = 1 + δ(ρ)/ρ        dg/d(r²) = (ρ·δ'(ρ) − δ(ρ))/(2·ρ³)
```

with `δ` and `δ'` from one basis evaluation. The tangential coefficients are
zero, so the 2×2 reduces to the radial form
`[[g + 2x²g', 2xy g'], [2xy g', g + 2y²g']]`.

The limits:

- **On axis** (`ρ` below `1e-15`): `(g, dg/d(r²)) = (1, 0)` outright. Only the
  first of those is a limit of its expression. The gauge pins `δ(0) = 0` and
  `δ'(0) = 0`, so `δ(ρ)/ρ → 0` and `g → 1`. It does **not** pin `δ''(0)`, so
  `δ = a·ρ² + O(ρ³)` and `dg/d(r²) = (ρ·δ' − δ)/(2ρ³)` diverges like `a/(2ρ)`.
  What is finite is the composition: `dg/d(r²)` enters the 2×2
  `[[g + 2x²g', 2xy g'], [2xy g', g + 2y²g']]` only against a factor of order
  `ρ²`, so each `g'` term is `O(ρ)` and the 2×2 approaches the identity
  linearly in `ρ`. At the `1e-15` guard the term the short-circuit drops is
  sub-ulp against `g = 1`, so `(1, 0)` is continuous with the general
  expression rather than an approximation of it. The second return is
  therefore not a usable radial derivative apart from those companion factors.
  `D` is the identity there and the Jacobian is the pinhole
  `diag(fx/rz, −fy/rz)` with the `rx/rz²` column, whatever the coefficients.
- **Behind the camera** (`rz ≤ 0`): `None`, the pinhole domain, shared with the
  rest of the perspective family.
- **Past the fold**: `None`, from the same `forward_projection_invertible` call
  the projection makes.

## Per-image corrections

An image of an `SFMTOOL_PINHOLE` camera may carry a radial correction and
rolling-shutter rates of its own, beside its focal scale and principal-point
offset: the four parts of its
[`IntrinsicAdjustment`](../../../crates/sfmtool-core/src/camera/intrinsics.rs)
(`focal`, `principal_point`, `radial_bspline: [f64; 2]`,
`rolling_shutter: [f64; 3]`, all zero by default; `is_zero` tests all four).
`CameraIntrinsics::with_adjustment(adjustment, scanlines)` builds the image's
intrinsics from its camera and the camera's scanlines setting, and
`ImageSet::image_intrinsics(i)` and `ImageIntrinsicsList::new` call it for every
image, so every routine that reads an image through its intrinsics gets both
corrections with no change of its own. `check_adjustment` says why a camera
cannot take an adjustment (a correction on another model, a degenerate spline
domain, or rates without a scanlines setting), and `with_adjustment` panics on
those; `SfmrReconstruction::from_sfmr_data` refuses them with an error first.
`check_rolling_shutter` says the same of a rolling shutter set on the field
directly: non-zero rates on another model, on which the projection methods
panic, or a rate that is not finite. `ImageSet::check_image_intrinsics` runs
both over every camera and image, for an edit that pairs images with other
cameras or changes a camera, its flags or its scanlines setting; the binding's
`clone_with_changes` calls it before anything is read, and the binding's
`CameraIntrinsics.from_dict` and unpickling check a rolling shutter they set,
each raising `ValueError`. The bundle-adjustment kernel checks its images'
adjustments up front and panics with the reason, with its other caller
errors.

```rust
let image = camera.with_adjustment(
    &IntrinsicAdjustment {
        radial_bspline: [0.004, -0.006],
        rolling_shutter: [2e-3, -1e-3, 5e-4],
        ..Default::default()
    },
    Some(RollingShutterScanlines::Rows),
);
let ray = image.pixel_to_ray(u, v); // inverts the radial map and the readout
```

Each correction is shaped to leave the existing kernels as they are.

**The radial correction** replaces the camera's coefficients `c` with `c + K·e`
(`K` from `radial_bspline_insertion` in
[bspline.rs](../../../crates/sfmtool-core/src/camera/distortion/bspline.rs); the
closed form, the matrix and the empty-spline case are in
[sfmtool-camera-models.md § Per-image radial
correction](../../formats/sfmtool-camera-models.md#per-image-radial-correction)).
The image's intrinsics stay an ordinary `SFMTOOL_PINHOLE` camera, so the forward
map, the inverse, the Jacobian and `bspline_is_monotone` read it unchanged.

**The rolling shutter** is not a radial map, so no `CameraModel` variant carries
it. It is `CameraIntrinsics::rolling_shutter: Option<RollingShutter>`, with
`RollingShutter { scanlines: RollingShutterScanlines, rates: [f64; 3] }`
(`RollingShutterScanlines` is the format crate's enum, re-exported by
`sfmtool_core::camera`). Only an image's intrinsics carry one, never a camera
entry, and `SfmrCamera::from` leaves it out. A field on the intrinsics reaches
every projection method at once; the cost is that every comparison of
`CameraIntrinsics` and every place that builds one names it, and that
`has_distortion` reports `true` for a camera with non-zero rates.

### The rolling-shutter step

The step is applied by `project`, `ray_to_pixel`, `ray_to_pixel_with_jacobian`,
`pixel_to_ray`, `unproject`, their batch forms and the ray grid's per-node
projection, all in
[projection.rs](../../../crates/sfmtool-core/src/camera/distortion/projection.rs)
and [rolling_shutter.rs](../../../crates/sfmtool-core/src/camera/distortion/rolling_shutter.rs).
Each method tests `rolling_shutter_step()` first: with no rolling shutter, or
rates of exactly zero, it runs its own arithmetic, so such a camera projects,
unprojects and differentiates bit for bit as one without the field. The step
is defined on this model's image plane only; a camera of another model with
non-zero rates panics in the projection methods.

In the `.sfmr` camera frame (+X right, +Y up, looking down −Z), the undistorted
point is `(x, y) = (Px, Py)/(−Pz)` and the radial map gives `(x_d, y_d)`. The
readout position is the pixel distance of the point's scanline from the
principal point's, in the short-side unit `m = min(width, height)`, read at the
pixel the point reaches before the step, `u = cx + f·x_d`, `v = cy − f·y_d`:

```
t = (v − cy)/m = −f·y_d/m        Rows
t = (u − cx)/m =  f·x_d/m        Columns
```

The step adds the image-plane velocity of a fixed point under the camera
rotation `ω = (ωx, ωy, ωz)`, evaluated at the undistorted point, times `t`:

```
x_r = x_d + t·ẋ,        ẋ =  ωy·(1 + x²) − ωx·x·y + ωz·y
y_r = y_d + t·ẏ,        ẏ = −ωx·(1 + y²) + ωy·x·y − ωz·x
u = cx + f·x_r,         v = cy − f·y_r
```

`(ẋ, ẏ)` is the derivative of `(Px, Py)/(−Pz)` under `Ṗ = −ω × P`. The rates
are in radians per short side of readout, whose meaning
[sfmr-file-format.md § Per-image intrinsic
adjustments](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13)
gives.

The code runs the step in the kernels' optical frame, where the radial
kernels return `(x_o, y_o) = (x_d, −y_d)` and the pixel is
`(f·x_o + cx, f·y_o + cy)`: there `t` is `f·y_o/m` for rows and `f·x_o/m` for
columns, and the velocity is `(ẋ, −ẏ)` at `(x_o, −y_o)`. The two `−y` flips are
the existing pixel mapping's; the step adds none.

### Unprojection with a rolling shutter

`pixel_to_ray` and `unproject` take the pixel to `(x_r, y_r)` by the affine
pixel step, then remove the displacement by a fixed-point iteration around the
exact radial inverse (`undistort_sfmtool_pinhole`, run on the image's spline):

```
(x_d)₀ = (x_r)
repeat:
    (x)ₖ   = radial_inverse((x_d)ₖ)
    (x_d)ₖ₊₁ = (x_r) − t((x_d)ₖ) · ṗ((x)ₖ)
until a pass moves (x_d) by less than 1e-12, or after 50 passes
```

and finish with the radial inverse of the last `(x_d)`. The iteration contracts
by about `|ω|·(f/m)·(1 + ρ²)`, about `10⁻³` on a measured phone capture, so it
reaches rounding in three or four passes. A pass that does not converge returns
its last iterate, as the radial inverse returns its base model's inverse rather
than failing, and a non-finite iterate stops the loop and is returned as it is.

### The ray Jacobian with a rolling shutter

`ray_to_pixel_with_jacobian` returns the derivative of the whole map. With `D`
the radial 2×2 at `(x, y)` (the same expression in either frame),
`∇t = ∂t/∂(x_d, y_d)`, which is `(0, −f/m)` for rows and `(f/m, 0)` for columns
in the `.sfmr` frame, and `ṗ = (ẋ, ẏ)`:

```
∂(x_r, y_r)/∂(x, y) = (I + ṗ·∇tᵀ)·D + t·∂ṗ/∂(x, y)
∂ṗ/∂(x, y)          = [[ 2ωy·x − ωx·y,   −ωx·x + ωz     ],
                       [ ωy·y − ωz,      −2ωx·y + ωy·x  ]]
```

followed by the pixel step's `f·diag(1, −1)` and the divide's `∂(x, y)/∂ray`.
It is the derivative of the map as stated, with the velocity read at the
undistorted point and `t` at the distorted one, not of a simplified map,
because the bundle adjustment's damping ladder compares costs of the map
itself. The domain is the model's: `None` behind the camera and past the fold,
from the same `forward_projection_invertible` call.

The bundle adjustment holds an image's corrections and differentiates its own
columns through them
([bundle-adjustment.md § Held image
adjustments](../geometry/bundle-adjustment.md#the-per-image-release)).

## Monotonicity enforcement

`bspline_is_monotone` decides the model's invariant, `1 + δ'(ρ) > 0` over
`[0, ρ_max]` and over the linear tail past it, by the procedure specified in
[sfmtool-fisheye-kernels.md](sfmtool-fisheye-kernels.md) § "Monotonicity
enforcement": the tail's single `1 + δ'(ρ_max) > 0` inequality, then the
derivative spline's control points, then a dense sampling when the conservative
test fails. The procedure is arithmetic on the coefficients and the domain end,
so it is the same code for both models.

Enforcement sites:

- **At solve time**: the bundle adjustment's `bspline_step_admissible` rejects,
  inside the damping ladder, any step whose candidate spline is non-finite or
  fails the check over `[0, ρ_max]`, and any step after which an image's
  spline `c + K·e` under its held radial correction would fail it. A released
  solve therefore cannot persist a folded spline for the camera or for any of
  its images.
- **At read time**: the Newton inverse's bracket `[0, ρ_max]` contains exactly
  one root because of it. The iteration is bisection-safeguarded anyway, so a
  spline that violates the invariant still converges to *a* root of the folded
  map rather than diverging, and the forward map's fold gate keeps projection
  and derivative out of domain together.

## The zero-spline short-circuit

`bspline_is_inactive` (fewer than `MIN_BSPLINE_COEFFS = 2` coefficients, every
coefficient exactly `0.0`, or a `ρ_max` that is not positive and finite)
short-circuits every map to the `SIMPLE_PINHOLE` arithmetic:

| entry point | short-circuits to |
|-------------|-------------------|
| `distort_sfmtool_pinhole` | the identity `(x, y)` |
| `undistort_sfmtool_pinhole` | the identity `(x_d, y_d)` |
| `sfmtool_pinhole_radial_factor` | `(g, dg/d(r²)) = (1, 0)` |
| `sfmtool_pinhole_unfolded` | `true` |
| `ray_to_pixel_with_jacobian` | the undistorted-pinhole fast path |

The last row is what makes the contract bitwise rather than merely equal to a
tolerance: an undistorted pinhole skips the 2×2 composition and writes
`J = diag(fx, fy)·(P·S)` directly, and the general composition rounds
`fx·rx/rz²` in a different association. The fast-path predicate therefore
admits a `SFMTOOL_PINHOLE` whose spline is inactive, so the promotion contract
of [../../formats/sfmtool-camera-models.md](../../formats/sfmtool-camera-models.md)
holds on the derivative as well as the projection.

## Classification

- `is_fisheye()` is **false** and so is `needs_ray_path()`: the map has a
  perspective divide and an `rz > 0` domain, so everything keyed off them
  (frustum placement, warp maps, ray-grid projection, the pinhole fits, the
  GUI) takes the image-plane path, which is the correct one for this model.
- `supports_pixel_jacobian()` is **true**.
- `has_distortion()` is true when the spline is active (`bspline_is_inactive`
  is false) and any `|cᵢ| > 1e-12` (`DISTORTION_EPS`), the same threshold the
  polynomial models' coefficients use. The magnitude test is looser than the
  exact-zero identity check, which governs bit-identity rather than
  classification; the activity test keeps a camera that projects as the base
  model from reporting distortion.
- `focal_lengths()` reports `(f, f)`; `principal_point()` the pair.
- `best_fit_inside_pinhole` / `best_fit_outside_pinhole` are defined for it
  (they reject `needs_ray_path` models), and search over the spline's forward
  and inverse maps.

## Testing requirements

- **Zero-spline bit-identity**: over a field sweep out past the frame corner,
  an empty spline and an all-zero spline reproduce the `SIMPLE_PINHOLE`
  camera's projection, unprojection and Jacobian bit for bit, including at the
  image boundary; so do live coefficients on a `ρ_max` that is not positive and
  finite (zero, negative, infinite, NaN), with no NaN ever produced.
- **Round trip with a live spline**: project → unproject returns the ray across
  the `ρ_max` seam, exercising both the Newton branch and the closed-form tail.
- **No invertible radius**: a spline with `ρ_max + δ(ρ_max) ≤ 0` unprojects to
  the identity across the field, leaving each distorted point where it is
  rather than collapsing the image onto the optical axis.
- **Jacobian against a central difference** of the projection over a field
  crossing `ρ_max`, at several ray scales, and the on-axis Jacobian equal to
  the pinhole limit with no NaN as `ρ → 0`.
- **Shared domain**: a spline folded inside the sampled field takes projection
  and Jacobian out of domain together (`None` from both); so does a ray at or
  behind the plane through the camera centre.
- **Classification**: the model reports perspective, image-plane and
  analytic-Jacobian support, reports no distortion for an empty or all-zero
  spline, and admits the pinhole fits.
- **Python surface** (`tests/rust_bindings/`): batch projection round trip
  across the seam, the exact pinhole identity, copy/dict round trip, and
  `best_fit_inside_pinhole` behavior.
- **Per-image corrections**
  ([rolling_shutter/tests.rs](../../../crates/sfmtool-core/src/camera/distortion/rolling_shutter/tests.rs),
  [intrinsics/tests.rs](../../../crates/sfmtool-core/src/camera/intrinsics/tests.rs)):
  zero rates project, unproject and differentiate bit for bit as no rolling
  shutter; `c + K·e` evaluates to `δ + Δδ` to rounding on and past `[0, ρ_max]`
  for several `N` including 2; the forward map equals the `.sfmr`-frame formulas
  above written out independently; the velocity equals the image motion of a
  small camera rotation; `pixel_to_ray` inverts `ray_to_pixel` to `1e-9` px over
  the image with a radial correction and rates ten times the measured size, on
  rows and on columns; the analytic Jacobian agrees with central differences;
  the batch forms and the ray grid agree with the scalar forms; a point at
  infinity projects as its direction; and rates on another model panic.
