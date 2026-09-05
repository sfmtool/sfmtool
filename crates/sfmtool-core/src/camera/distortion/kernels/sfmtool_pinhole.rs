// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `SFMTOOL_PINHOLE`: pinhole base + monotone radial spline.

use super::recover_radial_bspline;
use crate::camera::distortion::bspline;

/// Below this normalized image-plane radius the correction is unresolvable
/// against the gauge: `δ(0) = δ'(0) = 0` forces `δ(ρ)/ρ → 0` as `ρ → 0`, so
/// the radial factor `g = 1 + δ(ρ)/ρ` tends to `1` and the map is the base
/// pinhole's on the axis. The same `1e-15` the rest of the kernels use for an
/// on-axis point.
const PINHOLE_AXIS_EPS: f64 = 1e-15;

/// Forward `SFMTOOL_PINHOLE` map in normalized image-plane coordinates:
/// `(x, y)` with `ρ = √(x² + y²) = tan θ` in, `((ρ + δ(ρ))·x/ρ,
/// (ρ + δ(ρ))·y/ρ)` out, `δ` the radial spline ([`bspline::delta`]).
///
/// Unlike the fisheye's tangent-plane pair, this **is** the model's forward
/// map: the pinhole base's radial coordinate is exactly the image-plane radius
/// the caller passes, so no reparameterization happens here and the ray entry
/// point is the ordinary perspective divide followed by this.
///
/// An inactive spline ([`bspline::bspline_is_inactive`] — identity
/// coefficients, or a degenerate domain end) short-circuits to the identity,
/// keeping the zero-spline model bit-identical to `SIMPLE_PINHOLE`.
pub(in crate::camera::distortion) fn distort_sfmtool_pinhole(
    x: f64,
    y: f64,
    coeffs: &[f64],
    rho_max: f64,
) -> (f64, f64) {
    if bspline::bspline_is_inactive(coeffs, rho_max) {
        return (x, y);
    }
    let rho = (x * x + y * y).sqrt();
    if rho < PINHOLE_AXIS_EPS {
        return (x, y);
    }
    let rho_d = rho + bspline::delta(coeffs, rho_max, rho);
    let scale = rho_d / rho;
    (x * scale, y * scale)
}

/// Inverse of [`distort_sfmtool_pinhole`]: `(x_d, y_d)` with `r_d = ρ_d` in,
/// image-plane `(x, y)` with `ρ = tan θ` out.
///
/// The exact bracketed Newton of [`recover_radial_bspline`], not the generic
/// fixed-point iteration [`CameraModel::undistort`](crate::camera::CameraModel::undistort) falls back on: that
/// iteration contracts only for weak distortion, while the spline's
/// monotonicity invariant hands this solve a guaranteed bracket at any
/// coefficient magnitude.
///
/// Inactive splines short-circuit to the identity (bit-identity with
/// `SIMPLE_PINHOLE`). A spline folded so far that `ρ_max + δ(ρ_max) ≤ 0` has
/// no invertible radius at all, and [`recover_radial_bspline`] reports that
/// as `converged = false`; this returns the identity `ρ = r_d` there, the
/// base pinhole's inverse, rather than scaling every distorted point onto the
/// optical axis. That is the policy [`sfmtool_fisheye_to_ray`](super::sfmtool_fisheye_to_ray) applies to the
/// same report.
pub(in crate::camera::distortion) fn undistort_sfmtool_pinhole(
    x_d: f64,
    y_d: f64,
    coeffs: &[f64],
    rho_max: f64,
) -> (f64, f64) {
    if bspline::bspline_is_inactive(coeffs, rho_max) {
        return (x_d, y_d);
    }
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < PINHOLE_AXIS_EPS {
        return (x_d, y_d);
    }
    let (rho, converged) = recover_radial_bspline(r_d, coeffs, rho_max);
    if !converged {
        return (x_d, y_d);
    }
    let scale = rho / r_d;
    (x_d * scale, y_d * scale)
}

/// The radial factor `g` of the `SFMTOOL_PINHOLE` map and its derivative
/// `dg/d(r²)`, at squared image-plane radius `r2` — the pair the perspective
/// family's `distort_jacobian` is parameterized by.
///
/// The map is radially symmetric in the image plane, `x_d = x·g(ρ)` with
///
/// ```text
/// g(ρ) = 1 + δ(ρ)/ρ        dg/d(r²) = (ρ·δ'(ρ) − δ(ρ))/(2·ρ³)
/// ```
///
/// (chain rule through `ρ = √(r²)`, `dρ/d(r²) = 1/(2ρ)`). An inactive spline
/// or an on-axis point is `(1, 0)`. The first half of that is a limit: the
/// gauge pins `δ(0) = 0` and `δ'(0) = 0`, so `δ(ρ)/ρ → 0` and `g → 1`,
/// leaving the exact `SIMPLE_PINHOLE` factor at the axis.
///
/// The second half is **not**. The gauge says nothing about `δ''(0)`, so
/// `δ = aρ² + O(ρ³)` and `dg/d(r²) = (ρ·δ' − δ)/(2ρ³)` diverges like
/// `a/(2ρ)`. It stays bounded only in company: the caller reaches it through
/// `[[g + 2x²g', 2xy g'], [2xy g', g + 2y²g']]`, where every appearance
/// carries a `2x²`, `2y²` or `2xy` factor of order `ρ²`, so each entry's
/// `g'` term is `O(ρ)` and the composed 2×2 tends to the identity. At
/// `PINHOLE_AXIS_EPS = 1e-15` the term this short-circuit discards is
/// therefore sub-ulp against `g = 1`, which is what makes returning `(1, 0)`
/// continuous rather than merely close.
///
/// So the second return is not a bounded radial derivative on its own. Do not
/// reuse it apart from the `O(ρ²)` companion factors that make the product
/// finite.
pub(in crate::camera::distortion) fn sfmtool_pinhole_radial_factor(
    r2: f64,
    coeffs: &[f64],
    rho_max: f64,
) -> (f64, f64) {
    if bspline::bspline_is_inactive(coeffs, rho_max) {
        return (1.0, 0.0);
    }
    let rho = r2.sqrt();
    if rho < PINHOLE_AXIS_EPS {
        return (1.0, 0.0);
    }
    let (delta, deriv) = bspline::delta_and_deriv(coeffs, rho_max, rho);
    (1.0 + delta / rho, (rho * deriv - delta) / (2.0 * rho * r2))
}

/// Whether the `SFMTOOL_PINHOLE` map is on its principal monotonic branch at
/// squared image-plane radius `r2` — the fold gate, the same one
/// [`distort_ray_sfmtool_fisheye`](super::distort_ray_sfmtool_fisheye) applies in `θ`: a spline that drives
/// `ρ_d = ρ + δ(ρ)` non-positive at a positive `ρ` has left the branch
/// connected to the origin, and the projection is out of domain there.
///
/// The slope half of the polynomial family's branch test (`1 + δ'(ρ) > 0`) is
/// this model's monotonicity **construction** invariant, enforced where
/// splines are produced rather than probed per ray, so it is not repeated
/// here. An inactive spline is `SIMPLE_PINHOLE`, whose domain is every ray in
/// front of the camera.
pub(in crate::camera::distortion) fn sfmtool_pinhole_unfolded(
    r2: f64,
    coeffs: &[f64],
    rho_max: f64,
) -> bool {
    if bspline::bspline_is_inactive(coeffs, rho_max) {
        return true;
    }
    let rho = r2.sqrt();
    if rho < PINHOLE_AXIS_EPS {
        return true;
    }
    rho + bspline::delta(coeffs, rho_max, rho) > 0.0
}
