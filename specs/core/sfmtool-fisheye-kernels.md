# The `SFMTOOL_FISHEYE` kernels

**Status:** Implemented.
`crates/sfmtool-core/src/camera/distortion/bspline.rs` (basis evaluation and
the monotonicity check), `crates/sfmtool-core/src/camera/distortion/kernels.rs`
(`distort_ray_sfmtool_fisheye`, `sfmtool_fisheye_to_ray`,
`recover_theta_bspline`, `sfmtool_fisheye_ray_jacobian`), the dispatch in
`crates/sfmtool-core/src/camera/distortion.rs` and the classification arms in
`crates/sfmtool-core/src/camera/intrinsics.rs`; tests in
`camera/distortion/tests.rs`, `camera/intrinsics/tests.rs` and
`tests/rust_bindings/test_sfmtool_fisheye_rust_bindings.py`.

## Summary

The computation behind the `SFMTOOL_FISHEYE` camera model defined in
[../formats/sfmtool-camera-models.md](../formats/sfmtool-camera-models.md).
This spec covers basis evaluation, forward projection, the inverse recovery of
the incidence angle, the analytic ray Jacobian, enforcement of the
monotonicity invariant, and the model's classification flags. The bundle
adjustment's `opt_bspline` release, which produces the coefficients, is
specified in [bundle-adjustment.md](bundle-adjustment.md).

## Basis evaluation

`basis_at(n_coeffs, theta_max, theta)` returns the full-basis index of the
first active function together with the four values and four derivatives at
`θ`. At most `BSPLINE_SUPPORT = 4` functions are active at any `θ`; entries
whose full index is below 2 are the anchored pair and carry no coefficient
(coefficient `i` is full-basis function `i + 2`). A spline shorter than
`MIN_BSPLINE_COEFFS = 2` evaluates as the identity `δ ≡ 0`, as does a `θ_max`
that is not positive and finite.

`basis_at` clamps `θ` into `[0, θ_max]`, so a coefficient derivative on the
held-constant tail is exactly `Bᵢ(θ_max)`, the derivative of the held
constant. `δ` and `δ'` at a point come from one `basis_at` evaluation
(`delta_and_deriv`).

## Forward projection

`distort_ray_sfmtool_fisheye` takes an optical-frame ray to `(x_d, y_d) =
θ_d·û`. A ray on the optical axis (`ρ < 1e-15`) maps to `(0, 0)`, the principal
point, the antipode included. The one domain restriction is the same **fold
gate** the polynomial fisheye family applies: a spline that drives `θ_d ≤ 0`
at a positive `θ` has left its principal monotonic branch, and the projection
is `None` there.

`distort_sfmtool_fisheye` / `undistort_sfmtool_fisheye` are the tangent-plane
forms (`r = tan θ` in, `r_d = θ_d` out and back), meaningful only for `θ < 90°`
where the tangent plane exists; the ray-space entry points are what the
projection pipeline calls.

## Inverse

`sfmtool_fisheye_to_ray` recovers the unit ray from the distorted coordinate by
inverting `r_d = θ + δ(θ)` in `recover_theta_bspline`, then returning
`(x_d·sin θ/r_d, y_d·sin θ/r_d, cos θ)`.

- `r_d ≤ 0` is `θ = 0`.
- On the **linear tail** (`r_d ≥ θ_max + δ(θ_max)`) the inverse is closed form:
  `θ = r_d − δ(θ_max)`, no iteration.
- Inside the spline's domain, a **bracket-safeguarded Newton** solves
  `g(θ) = θ + δ(θ) − r_d` over `[0, θ_max]`, where `g(0) = −r_d < 0` and
  `g(θ_max) > 0`. The start is the identity guess `min(r_d, θ_max)`; each
  iterate updates the bracket by the sign of `g`, and a Newton step landing
  outside the bracket (or non-finite, or on a non-positive `g'`) is replaced by
  a bisection. The bounds are inclusive: an underflowed step that reproduces
  `θ` is convergence, not a reason to bisect away from the root.
- Non-convergence is reported only when `θ_d(θ_max) ≤ 0` (a spline folded so
  far that no radius is representable) and is answered with the identity
  equidistant ray, since there is no inverse to return.

There is **no wide-angle blend**: the map is monotone by construction, so the
recovery is the exact inverse at every angle. A blend would drop the spline
exactly at the periphery, which is where the spline is largest. This is the
inverse the bundle adjustment's retriangulation and direction re-estimation
read.

## Ray Jacobian

`CameraModel::supports_pixel_jacobian` is **true**, so pose refinement, patch
sizing and the bundle adjustment take the analytic path.
`sfmtool_fisheye_ray_jacobian` is the equidistant-family template of
[projection-jacobian.md](projection-jacobian.md) (same optical frame, same
`ρ`, `n² = ρ² + rz²`, `û` and shared off-diagonal factor
`c = θ_d'·rz/n² − θ_d/ρ`) with

```
θ_d = θ + δ(θ)        θ_d' = 1 + δ'(θ)
```

substituted for the polynomial pair (`θ·(1 + k1·θ²)`, `1 + 3·k1·θ²`). `δ` and
`δ'` come from one `basis_at` evaluation, so the derivative is closed form at
every `θ`.

The two limits are the family's:

- **On axis, in front**: `diag(1/rz, 1/rz)`, the pinhole small-angle Jacobian.
  The gauge is what makes this exact: `δ(0) = 0` and `δ'(0) = 0` give
  `θ_d/ρ → 1/rz` and `θ_d' → 1` just as `k1` does.
- **At the antipode** (`ρ → 0`, `rz < 0`): `None`, the one measure-zero
  direction where the derivative's domain is narrower than the projection's.

`None` also past the fold gate, so projection and derivative leave the domain
together.

## Monotonicity enforcement

`bspline_is_monotone` decides the model's invariant, `1 + δ'(θ) > 0` over
`[0, θ_max]`, in two stages:

1. The **sufficient** condition on the derivative spline's control points. `δ'`
   is a quadratic B-spline whose control points are
   `3·(a_{i+1} − a_i)/(t_{i+4} − t_{i+1})` over the padded coefficients `a`
   (the anchored pair reading as zero), and a B-spline is a convex combination
   of its control points, so `1 + dᵢ > 0` for every `i` proves monotonicity
   outright over the whole domain (zero-width knot spans skipped).
2. When that conservative test fails, a dense sampling of `δ'` (64 samples per
   knot span, against a per-span quadratic) over the requested span decides.

Enforcement sites:

- **At solve time**: the bundle adjustment's `bspline_step_admissible`
  rejects, inside the damping ladder, any step whose candidate spline is
  non-finite or fails the check over `[0, θ_max]`. A released solve therefore
  cannot persist a folded spline.
- **At read time**: the Newton inverse's bracket `[0, θ_max]` contains exactly
  one root because of it. The iteration is bisection-safeguarded anyway, so a
  spline that violates the invariant still converges to *a* root of the folded
  map rather than diverging, and the forward map's fold gate keeps projection
  and derivative out of domain together.

## The zero-spline short-circuit

`bspline_is_identity` decides the coefficient half of the format spec's
zero-spline identity: the spline is shorter than `MIN_BSPLINE_COEFFS` (the
empty one included) or every coefficient is **exactly** `0.0`.
`bspline_is_inactive` widens it with the degenerate domain ends: identity
coefficients, or a `θ_max` that is not positive and finite. Every map
short-circuits on `bspline_is_inactive` to the corresponding
`EQUIDISTANT_FISHEYE` kernel:

| entry point | short-circuits to |
|-------------|-------------------|
| `distort_sfmtool_fisheye` | `distort_equidistant` |
| `undistort_sfmtool_fisheye` | `undistort_equidistant` |
| `distort_ray_sfmtool_fisheye` | `distort_ray_equidistant_exact` |
| `sfmtool_fisheye_to_ray` | `equidistant_to_ray` |
| `sfmtool_fisheye_ray_jacobian` | `radial_fisheye_ray_jacobian(…, k1 = 0)` |

The result is **bit-identical arithmetic**, not agreement to a tolerance,
which is what delivers the format spec's promotion contract.

## Classification

- `is_fisheye()` and therefore `needs_ray_path()` are **true**: everything
  keyed off them (frustum placement, warp maps, ray-grid projection, the GUI)
  takes the ray path.
- `supports_pixel_jacobian()` is **true**.
- `has_distortion()` is true when the spline is active (`bspline_is_inactive`
  is false) and any `|cᵢ| > 1e-12` (`DISTORTION_EPS`), the same threshold the
  polynomial models' coefficients use. The magnitude test is looser than the
  exact-zero identity check, which governs bit-identity rather than
  classification; the activity test keeps a camera that projects as the base
  model from reporting distortion.
- `focal_lengths()` reports `(f, f)`; `principal_point()` the pair.

## Testing requirements

- **Zero-spline bit-identity**: over a `θ` sweep past 90°, an empty spline and
  an all-zero spline reproduce the `EQUIDISTANT_FISHEYE` camera's projection,
  unprojection and Jacobian bit for bit; so do live coefficients on a `θ_max`
  that is not positive and finite (zero, negative, infinite, NaN), with no NaN
  ever produced.
- **Round trip with a live spline**: project → unproject returns the ray past
  90° and across the `θ_max` seam, exercising both the Newton branch and the
  closed-form tail.
- **Jacobian against a central difference** of the projection over a field
  sampled past 90°, and the on-axis Jacobian equal to the pinhole
  `diag(1/rz, 1/rz)` limit.
- **Shared domain**: a spline folded inside the sampled field takes projection
  and Jacobian out of domain together (`None` from both).
- **Basis unit tests**: `δ(0) = 0` and `δ'(0) = 0` (the gauge); the basis is a
  partition of unity; equal coefficients plateau once the anchored pair dies
  out; `δ'` matches a central difference of `δ`; `δ` is held constant beyond
  `θ_max`; a spline below the minimum length is the identity.
- **Classification**: the model reports fisheye, ray-path and analytic-Jacobian
  support, and reports no distortion for an empty or all-zero spline.
- **Python surface** (`tests/rust_bindings/`): batch projection round trip
  past 90°, the exact equidistant identity, copy/dict round trip, and
  `best_fit_inside_pinhole` behavior.
