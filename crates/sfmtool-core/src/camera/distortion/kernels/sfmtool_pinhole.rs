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
/// optical axis. [`sfmtool_fisheye_to_ray`](super::sfmtool_fisheye_to_ray) answers the same report
/// with its own base model's inverse, `equidistant_to_ray` (`θ = r_d`).
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
/// family's `distort_jacobian` is parameterized by:
/// `g(ρ) = 1 + δ(ρ)/ρ`, `dg/d(r²) = (ρ·δ'(ρ) − δ(ρ))/(2·ρ³)`.
///
/// An inactive spline or an on-axis point (`ρ < PINHOLE_AXIS_EPS`) returns
/// `(1, 0)`. The second return can diverge like `1/ρ` toward the axis and is
/// finite only multiplied by the `O(ρ²)` factors of the radial 2×2, so do not
/// use it on its own as a radial derivative. The derivation and the on-axis
/// limit are in `specs/core/camera/sfmtool-pinhole-kernels.md` § "Ray Jacobian".
///
/// This applies no fold gate. The projection's Jacobian reads the pair from
/// [`checked_distort_sfmtool_pinhole_with_radial_factor`] instead, which gates
/// it and returns the distorted point from the same spline evaluation.
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
    radial_factor(rho, r2, delta, deriv)
}

/// `(g, dg/d(r²))` from one evaluation `(δ, δ')` at `ρ = √(r²)`, off the
/// axis: the expressions of [`sfmtool_pinhole_radial_factor`], written once
/// for it and for [`checked_distort_sfmtool_pinhole_with_radial_factor`].
fn radial_factor(rho: f64, r2: f64, delta: f64, deriv: f64) -> (f64, f64) {
    (1.0 + delta / rho, (rho * deriv - delta) / (2.0 * rho * r2))
}

/// [`distort_sfmtool_pinhole`] behind the fold gate: `None` where the
/// image-plane point `(x, y)` is off the map's principal monotonic branch, and
/// otherwise the distorted point, with one evaluation of `δ` serving both.
///
/// The gate is the one [`distort_ray_sfmtool_fisheye`](super::distort_ray_sfmtool_fisheye)
/// applies in `θ`: a spline that drives `ρ_d = ρ + δ(ρ)` non-positive at a
/// positive `ρ` has left the branch connected to the origin, and the
/// projection is out of domain there (so is a `NaN` `ρ_d`). The
/// slope half of the polynomial family's branch test (`1 + δ'(ρ) > 0`) is this
/// model's monotonicity **construction** invariant, enforced where splines are
/// produced rather than probed per ray, so it is not repeated here.
///
/// An inactive spline is `SIMPLE_PINHOLE`, whose domain is every ray in front
/// of the camera, and an on-axis point (`ρ < PINHOLE_AXIS_EPS`) is in domain;
/// both return `(x, y)`, as [`distort_sfmtool_pinhole`] does. Every point this
/// returns is [`distort_sfmtool_pinhole`]'s, bit for bit.
pub(in crate::camera::distortion) fn checked_distort_sfmtool_pinhole(
    x: f64,
    y: f64,
    coeffs: &[f64],
    rho_max: f64,
) -> Option<(f64, f64)> {
    if bspline::bspline_is_inactive(coeffs, rho_max) {
        return Some((x, y));
    }
    let rho = (x * x + y * y).sqrt();
    if rho < PINHOLE_AXIS_EPS {
        return Some((x, y));
    }
    let rho_d = rho + bspline::delta(coeffs, rho_max, rho);
    // Written so that a `NaN` `ρ_d` is out of domain too.
    if rho_d > 0.0 {
        let scale = rho_d / rho;
        Some((x * scale, y * scale))
    } else {
        None
    }
}

/// [`checked_distort_sfmtool_pinhole`] together with the pair
/// `(g, dg/d(r²))` of [`sfmtool_pinhole_radial_factor`], all from one
/// evaluation of `(δ, δ')`: the fold gate, the distorted point and the pair
/// the projection's Jacobian composes.
///
/// `None` on exactly the points where [`checked_distort_sfmtool_pinhole`]
/// returns `None`. Elsewhere the point is that function's and the pair is
/// [`sfmtool_pinhole_radial_factor`]'s, bit for bit: `δ` is
/// [`bspline::delta`]'s value whichever of the two evaluates it. An inactive
/// spline or an on-axis point returns `((x, y), (1, 0))`.
pub(in crate::camera::distortion) fn checked_distort_sfmtool_pinhole_with_radial_factor(
    x: f64,
    y: f64,
    coeffs: &[f64],
    rho_max: f64,
) -> Option<((f64, f64), (f64, f64))> {
    if bspline::bspline_is_inactive(coeffs, rho_max) {
        return Some(((x, y), (1.0, 0.0)));
    }
    let r2 = x * x + y * y;
    let rho = r2.sqrt();
    if rho < PINHOLE_AXIS_EPS {
        return Some(((x, y), (1.0, 0.0)));
    }
    let (delta, deriv) = bspline::delta_and_deriv(coeffs, rho_max, rho);
    let rho_d = rho + delta;
    // The same test as `checked_distort_sfmtool_pinhole`'s.
    if rho_d > 0.0 {
        let scale = rho_d / rho;
        Some(((x * scale, y * scale), radial_factor(rho, r2, delta, deriv)))
    } else {
        None
    }
}
