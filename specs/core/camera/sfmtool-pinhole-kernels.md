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
and the model's classification flags. Basis evaluation is the shared machinery
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

- `r_d ≤ 0` is `ρ = 0`.
- A spline folded so far that `ρ_max + δ(ρ_max) ≤ 0` has no invertible radius
  at all: no positive `r_d` is reachable, and `recover_radial_bspline` reports
  it by returning `converged = false`. The kernel answers that report with the
  **identity** `ρ = r_d`, the base pinhole's inverse, so the distorted point
  passes through unchanged. Taking the reported radius instead would scale
  every pixel in the image onto the optical axis. It is the policy
  `sfmtool_fisheye_to_ray` applies to the same report, and it is unreachable
  through a solve, which cannot persist a spline that folded
  ([Monotonicity enforcement](#monotonicity-enforcement)).
- On the **linear tail** (`r_d ≥ ρ_max + δ(ρ_max)`) the inverse is closed form:
  `ρ = r_d − δ(ρ_max)`, no iteration. The tail carries the periphery of every
  real image: `ρ` grows without bound toward `θ = 90°`, so a calibrated
  `ρ_max` is crossed well inside the frame.
- Inside the spline's domain, a **bracket-safeguarded Newton** solves
  `g(ρ) = ρ + δ(ρ) − r_d` over `[0, ρ_max]`, where `g(0) = −r_d < 0` and
  `g(ρ_max) > 0`. The start is the identity guess `min(r_d, ρ_max)`; each
  iterate updates the bracket by the sign of `g`, and a Newton step landing
  outside the bracket (or non-finite, or on a non-positive `g'`) is replaced by
  a bisection. The bounds are inclusive: an underflowed step that reproduces
  `ρ` is convergence, not a reason to bisect away from the root.

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

## Monotonicity enforcement

`bspline_is_monotone` decides the model's invariant, `1 + δ'(ρ) > 0` over
`[0, ρ_max]`, by the two-stage procedure specified in
[sfmtool-fisheye-kernels.md](sfmtool-fisheye-kernels.md) § "Monotonicity
enforcement": the derivative spline's control points, then a dense sampling
when the conservative test fails. The procedure is arithmetic on the
coefficients and the domain end, so it is the same code for both models.

Enforcement sites:

- **At solve time**: the bundle adjustment's `bspline_step_admissible` rejects,
  inside the damping ladder, any step whose candidate spline is non-finite or
  fails the check over `[0, ρ_max]`. A released solve therefore cannot persist
  a folded spline.
- **At read time**: the Newton inverse's bracket `[0, ρ_max]` contains exactly
  one root because of it. The iteration is bisection-safeguarded anyway, so a
  spline that violates the invariant still converges to *a* root of the folded
  map rather than diverging, and the forward map's fold gate keeps projection
  and derivative out of domain together.

## The zero-spline short-circuit

`bspline_is_inactive` (identity coefficients, or a `ρ_max` that is not positive
and finite) short-circuits every map to the `SIMPLE_PINHOLE` arithmetic:

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
