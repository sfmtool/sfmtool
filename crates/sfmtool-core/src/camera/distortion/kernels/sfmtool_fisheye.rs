// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `SFMTOOL_FISHEYE`: equidistant base + monotone radial spline.

use super::{
    distort_equidistant, distort_ray_equidistant_exact, equidistant_to_ray,
    radial_fisheye_ray_jacobian, undistort_equidistant, NormalizedRayJacobian,
    EQUIDISTANT_AXIS_EPS,
};
use crate::camera::distortion::{bspline, UNDISTORT_EPS, UNDISTORT_MAX_ITER};

/// Forward `SFMTOOL_FISHEYE` map in tangent-plane coordinates: `(x, y)` with
/// `r = tan θ` in, `((θ + δ(θ))·x/r, (θ + δ(θ))·y/r)` out, `δ` the radial
/// spline ([`bspline::delta`]).
///
/// An inactive spline ([`bspline::bspline_is_inactive`] — identity
/// coefficients, or a degenerate domain end) short-circuits to
/// [`distort_equidistant`], keeping the zero-spline model bit-identical to
/// `EQUIDISTANT_FISHEYE`. Like [`distort_equidistant`], only meaningful for
/// `θ < 90°`; the ray-space entry point is [`distort_ray_sfmtool_fisheye`].
pub(in crate::camera::distortion) fn distort_sfmtool_fisheye(
    x: f64,
    y: f64,
    coeffs: &[f64],
    theta_max: f64,
) -> (f64, f64) {
    if bspline::bspline_is_inactive(coeffs, theta_max) {
        return distort_equidistant(x, y);
    }
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }
    let theta = r.atan();
    let theta_d = theta + bspline::delta(coeffs, theta_max, theta);
    let scale = theta_d / r;
    (x * scale, y * scale)
}

/// Inverse of [`distort_sfmtool_fisheye`]: `(x_d, y_d)` with `r_d = θ_d` in,
/// tangent-plane `(x, y)` with `r = tan θ` out.
///
/// Inactive splines short-circuit to [`undistort_equidistant`] (bit-identity
/// with `EQUIDISTANT_FISHEYE`).
pub(in crate::camera::distortion) fn undistort_sfmtool_fisheye(
    x_d: f64,
    y_d: f64,
    coeffs: &[f64],
    theta_max: f64,
) -> (f64, f64) {
    if bspline::bspline_is_inactive(coeffs, theta_max) {
        return undistort_equidistant(x_d, y_d);
    }
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (theta, _) = recover_radial_bspline(r_d, coeffs, theta_max);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

/// Recover the undistorted radial coordinate `d` from the distorted radial
/// distance `r_d = d + δ(d)` for the spline map. Returns `(d, converged)`.
///
/// The equation is coordinate-agnostic, so both spline models invert through
/// it: `d` is the incidence angle `θ` for `SFMTOOL_FISHEYE` and the
/// normalized image-plane radius `ρ` for `SFMTOOL_PINHOLE`, with `d_max` the
/// matching domain end.
///
/// Beyond the spline's domain the map is exactly linear
/// (`r_d = d + δ(d_max)`), so `r_d ≥ d_max + δ(d_max)` inverts in closed
/// form. Inside `[0, d_max]` a safeguarded Newton iteration solves the
/// monotone `d_d(d)` — monotonicity is the model's construction invariant, so
/// the bracket `[0, d_max]` always contains exactly one root; the bisection
/// safeguard keeps the iteration well-defined even for a spline that
/// violates the invariant (it converges to *a* root of the folded map).
/// `converged` is `false` only when `d_d(d_max) ≤ 0` — a spline folded so
/// far that no radius is representable — mirroring
/// [`recover_theta_equidistant`](super::recover_theta_equidistant)'s unreachable-`r_d` report.
pub(in crate::camera::distortion) fn recover_radial_bspline(
    r_d: f64,
    coeffs: &[f64],
    d_max: f64,
) -> (f64, bool) {
    if r_d <= 0.0 {
        return (0.0, true);
    }
    let delta_end = bspline::delta(coeffs, d_max, d_max);
    let rd_end = d_max + delta_end;
    if rd_end <= 0.0 {
        return (0.0, false);
    }
    if r_d >= rd_end {
        // Linear region: exact inverse, no iteration.
        return (r_d - delta_end, true);
    }
    // Bracketed Newton on g(d) = d + δ(d) − r_d over [0, d_max]:
    // g(0) = −r_d < 0 and g(d_max) = rd_end − r_d > 0.
    let (mut lo, mut hi) = (0.0f64, d_max);
    let mut d = r_d.min(d_max); // identity start, like the k-family
    for _ in 0..UNDISTORT_MAX_ITER {
        let (delta, deriv) = bspline::delta_and_deriv(coeffs, d_max, d);
        let g = d + delta - r_d;
        if g == 0.0 {
            return (d, true); // Newton landed on the exact root
        }
        if g > 0.0 {
            hi = d;
        } else {
            lo = d;
        }
        let gp = 1.0 + deriv;
        let mut next = if gp > 0.0 { d - g / gp } else { f64::NAN };
        // Bracket safeguard (also catches NaN). Inclusive bounds: an
        // underflowed Newton step may reproduce `d` — that is convergence,
        // not a reason to bisect away from the root.
        if !(next >= lo && next <= hi) {
            next = 0.5 * (lo + hi);
        }
        let step = next - d;
        d = next;
        if step.abs() < UNDISTORT_EPS {
            break;
        }
    }
    (d, true)
}

/// Project an optical-frame ray through the spline map:
/// `θ = atan2(r_xy, rz)`, `θ_d = θ + δ(θ)`, distorted coordinate `θ_d` times
/// the unit 2D direction.
///
/// Inactive splines short-circuit to [`distort_ray_equidistant_exact`]
/// (bit-identity with `EQUIDISTANT_FISHEYE`, domain the whole sphere).
/// Otherwise the same fold gate as [`distort_ray_equidistant`](super::distort_ray_equidistant): a spline
/// that drives `θ_d` non-positive at a positive `θ` has left its principal
/// monotonic branch, and the projection is `None` there.
pub(in crate::camera::distortion) fn distort_ray_sfmtool_fisheye(
    rx: f64,
    ry: f64,
    rz: f64,
    coeffs: &[f64],
    theta_max: f64,
) -> Option<(f64, f64)> {
    if bspline::bspline_is_inactive(coeffs, theta_max) {
        return Some(distort_ray_equidistant_exact(rx, ry, rz));
    }
    let r_xy = (rx * rx + ry * ry).sqrt();
    if r_xy < 1e-15 {
        return Some((0.0, 0.0));
    }
    let theta = r_xy.atan2(rz);
    let theta_d = theta + bspline::delta(coeffs, theta_max, theta);
    if theta > 0.0 && theta_d <= 0.0 {
        return None;
    }
    let (dx, dy) = (rx / r_xy, ry / r_xy);
    Some((theta_d * dx, theta_d * dy))
}

/// Convert `SFMTOOL_FISHEYE` distorted coordinates to a unit ray direction —
/// Newton recovery of the monotone `θ_d(θ)` with **no wide-angle blend**,
/// exactly the [`simple_radial_fisheye_to_ray`](super::simple_radial_fisheye_to_ray) policy: the spline is
/// largest at the periphery, which is where a blend would drop it.
///
/// Inactive splines short-circuit to [`equidistant_to_ray`] (bit-identity
/// with `EQUIDISTANT_FISHEYE`). Past a fold (a spline violating the
/// monotonicity invariant so badly that no radius is representable) the
/// identity extrapolation is kept: there is no inverse to return there.
pub(in crate::camera::distortion) fn sfmtool_fisheye_to_ray(
    x_d: f64,
    y_d: f64,
    coeffs: &[f64],
    theta_max: f64,
) -> [f64; 3] {
    if bspline::bspline_is_inactive(coeffs, theta_max) {
        return equidistant_to_ray(x_d, y_d);
    }
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    let (theta, converged) = recover_radial_bspline(r_d, coeffs, theta_max);
    if !converged {
        return equidistant_to_ray(x_d, y_d);
    }
    let s = theta.sin() / r_d;
    [x_d * s, y_d * s, theta.cos()]
}

/// Distorted coordinate and analytic `∂(x_d, y_d)/∂(rx, ry, rz)` of the
/// spline map — [`radial_fisheye_ray_jacobian`]'s template with
/// `θ_d = θ + δ(θ)` and `θ_d' = 1 + δ'(θ)` substituted for the polynomial
/// pair. Same conventions throughout: optical frame, axis handling via
/// [`EQUIDISTANT_AXIS_EPS`], `None` at the antipode and past the fold.
///
/// The on-axis forward limit is the same pinhole `diag(1/rz, 1/rz)` as the
/// k-family's: the gauge pins `δ(0) = 0` and `δ'(0) = 0`, so `θ_d/ρ → 1/rz`
/// and `θ_d' → 1` exactly as with `k1`.
///
/// Inactive splines short-circuit to
/// `radial_fisheye_ray_jacobian(…, k1 = 0)`, the `EQUIDISTANT_FISHEYE`
/// arithmetic, bit for bit.
pub(in crate::camera::distortion) fn sfmtool_fisheye_ray_jacobian(
    rx: f64,
    ry: f64,
    rz: f64,
    coeffs: &[f64],
    theta_max: f64,
) -> Option<NormalizedRayJacobian> {
    if bspline::bspline_is_inactive(coeffs, theta_max) {
        return radial_fisheye_ray_jacobian(rx, ry, rz, 0.0);
    }
    let rho2 = rx * rx + ry * ry;
    let rho = rho2.sqrt();
    let n2 = rho2 + rz * rz;
    if n2 == 0.0 {
        return None;
    }
    if rho <= EQUIDISTANT_AXIS_EPS * n2.sqrt() {
        // On the optical axis: only the forward limit is finite.
        if rz <= 0.0 {
            return None;
        }
        let inv = 1.0 / rz;
        return Some(((0.0, 0.0), [[inv, 0.0, 0.0], [0.0, inv, 0.0]]));
    }
    let theta = rho.atan2(rz);
    let (d, dp) = bspline::delta_and_deriv(coeffs, theta_max, theta);
    let theta_d = theta + d;
    let dtheta_d = 1.0 + dp;
    // The forward map's fold gate (`distort_ray_sfmtool_fisheye`): past it
    // there is no projection to differentiate.
    if theta > 0.0 && theta_d <= 0.0 {
        return None;
    }
    let (ux, uy) = (rx / rho, ry / rho);
    let rz_n2 = rz / n2;
    let theta_rho = theta_d / rho;
    let cross = ux * uy * (dtheta_d * rz_n2 - theta_rho);
    Some((
        (theta_d * ux, theta_d * uy),
        [
            [
                theta_rho * uy * uy + dtheta_d * (ux * ux * rz_n2),
                cross,
                -dtheta_d * rx / n2,
            ],
            [
                cross,
                theta_rho * ux * ux + dtheta_d * (uy * uy * rz_n2),
                -dtheta_d * ry / n2,
            ],
        ],
    ))
}
