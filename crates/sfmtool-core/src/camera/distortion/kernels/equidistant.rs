// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The equidistant fisheye family and the distortion-free equidistant map.
//!
//! Three COLMAP models whose forward map is a polynomial in the incidence
//! angle `θ = atan r` — `OPENCV_FISHEYE` (k1–k4), `SIMPLE_RADIAL_FISHEYE`
//! (k) and `RADIAL_FISHEYE` (k1, k2) — share one Newton inverse,
//! [`recover_theta_equidistant`]. Below them sit the ray-direction helpers
//! those models project through, and the distortion-free `θ = r/f` map with
//! its exact ray-space entry point and 2×3 Jacobian, which the
//! `SFMTOOL_FISHEYE` spline models use as their base.

use super::{blend_fisheye_ray, equidistant_to_ray};
use crate::camera::distortion::{UNDISTORT_EPS, UNDISTORT_MAX_ITER};

/// OpenCV fisheye (equidistant) distortion.
///
/// Maps a 3D ray direction `(x, y, 1)` through the equidistant fisheye model:
/// `theta_d = theta * (1 + k1*theta² + k2*theta⁴ + k3*theta⁶ + k4*theta⁸)`
/// where `theta = atan(r)` and `r = sqrt(x² + y²)`.
pub(in crate::camera::distortion) fn distort_fisheye(
    x: f64,
    y: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }
    let theta = r.atan();
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;
    let theta6 = theta4 * theta2;
    let theta8 = theta4 * theta4;
    let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
    let scale = theta_d / r;
    (x * scale, y * scale)
}

/// Recover the incidence angle theta from the distorted radial distance
/// for the equidistant fisheye projection model.
///
/// Solves `r_d = theta * (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)` for theta
/// using Newton's method.
///
/// Returns `(theta, converged)`. When `converged` is false, `r_d` exceeds
/// the maximum value of the distortion function (its peak), and `theta` is
/// clamped to the peak angle — the largest angle the model can represent.
/// Callers that need smooth extrapolation beyond the model's valid range
/// (e.g. [`equidistant_fisheye_to_ray`]) can use this flag to fall back to
/// the identity equidistant model.
pub(in crate::camera::distortion) fn recover_theta_equidistant(
    r_d: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
) -> (f64, bool) {
    /// Evaluate f'(θ) = d/dθ [θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)].
    #[inline]
    fn f_prime(theta: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> f64 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
    }

    /// Bisect to find theta where f'(theta) = 0 (the peak of f).
    #[inline]
    fn find_peak(hi_start: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> f64 {
        let mut lo = 0.0_f64;
        let mut hi = hi_start;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if f_prime(mid, k1, k2, k3, k4) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    // Clamp the initial guess to π.
    let mut theta = r_d.min(std::f64::consts::PI);

    // The distortion polynomial can be non-monotonic for high-order
    // coefficients: f(θ) rises to a peak then falls. If the starting guess
    // is past the peak (f'(θ) ≤ 0), bisect to find the peak and use it as
    // both the starting point and a hard upper bound for Newton. For
    // out-of-range r_d (above the peak f-value), Newton converges to the
    // peak — the maximum angle the model can represent.
    let mut theta_max = std::f64::consts::PI;
    let mut hit_peak = false;
    if f_prime(theta, k1, k2, k3, k4) <= 0.0 {
        theta_max = find_peak(theta, k1, k2, k3, k4);
        theta = theta_max;
        hit_peak = true;
    }

    for _ in 0..UNDISTORT_MAX_ITER {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;
        let f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - r_d;
        let fp =
            1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8;
        if fp <= 0.0 {
            // Newton overshot past the peak. Bisect to find the true peak.
            theta = find_peak(theta, k1, k2, k3, k4);
            hit_peak = true;
            break;
        }
        let delta = f / fp;
        theta -= delta;
        theta = theta.clamp(0.0, theta_max);
        if delta.abs() < UNDISTORT_EPS {
            break;
        }
    }

    // If we ended up at the peak, check whether r_d is actually reachable.
    // If the peak f-value is less than r_d, the model can't represent this
    // r_d, so report non-convergence.
    if hit_peak {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        let f_at_peak = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
        if f_at_peak < r_d - UNDISTORT_EPS {
            return (theta, false);
        }
    }

    (theta, true)
}

/// Inverse of OpenCV fisheye distortion.
///
/// Given distorted coordinates `(x_d, y_d)`, recovers the undistorted `(x, y)`.
/// Uses Newton's method on the scalar `theta_d → theta` mapping, then recovers
/// the 2D direction.
pub(in crate::camera::distortion) fn undistort_fisheye(
    x_d: f64,
    y_d: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (theta, _) = recover_theta_equidistant(r_d, k1, k2, k3, k4);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

/// Simple radial fisheye distortion: equidistant + single radial k in theta space.
///
/// `theta_d = theta * (1 + k * theta²)` where `theta = atan(r)`.
pub(in crate::camera::distortion) fn distort_simple_radial_fisheye(
    x: f64,
    y: f64,
    k: f64,
) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }
    let theta = r.atan();
    let theta2 = theta * theta;
    let theta_d = theta * (1.0 + k * theta2);
    let scale = theta_d / r;
    (x * scale, y * scale)
}

/// Inverse of simple radial fisheye distortion.
pub(in crate::camera::distortion) fn undistort_simple_radial_fisheye(
    x_d: f64,
    y_d: f64,
    k: f64,
) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (theta, _) = recover_theta_equidistant(r_d, k, 0.0, 0.0, 0.0);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

/// Radial fisheye distortion: equidistant + two radial k1, k2 in theta space.
///
/// `theta_d = theta * (1 + k1 * theta² + k2 * theta⁴)` where `theta = atan(r)`.
pub(in crate::camera::distortion) fn distort_radial_fisheye(
    x: f64,
    y: f64,
    k1: f64,
    k2: f64,
) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }
    let theta = r.atan();
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;
    let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4);
    let scale = theta_d / r;
    (x * scale, y * scale)
}

/// Inverse of radial fisheye distortion.
pub(in crate::camera::distortion) fn undistort_radial_fisheye(
    x_d: f64,
    y_d: f64,
    k1: f64,
    k2: f64,
) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (theta, _) = recover_theta_equidistant(r_d, k1, k2, 0.0, 0.0);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

// ---------------------------------------------------------------------------
// Ray direction helpers
// ---------------------------------------------------------------------------

/// Convert distorted equidistant fisheye coordinates to a unit ray direction.
///
/// Recovers theta via Newton's method, then builds the ray as
/// `[sin(theta) * x_d/r_d, sin(theta) * y_d/r_d, cos(theta)]`.
/// Works correctly for any field of view, including beyond 180°.
pub(in crate::camera::distortion) fn equidistant_fisheye_to_ray(
    x_d: f64,
    y_d: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
) -> [f64; 3] {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    let (theta, converged) = recover_theta_equidistant(r_d, k1, k2, k3, k4);
    if !converged {
        // r_d exceeds the valid range of the distortion model (past the
        // peak of the distortion polynomial). Fall back to the identity
        // equidistant model which treats r_d directly as the incidence
        // angle. This avoids the broken peak-clamped theta and produces
        // a smooth extrapolation beyond the model's valid domain.
        return equidistant_to_ray(x_d, y_d);
    }
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let s = sin_theta / r_d;
    let recovered = [x_d * s, y_d * s, cos_theta];
    let undistorted = equidistant_to_ray(x_d, y_d);
    blend_fisheye_ray(r_d, recovered, undistorted)
}

/// Convert `SIMPLE_RADIAL_FISHEYE` distorted coordinates to a unit ray
/// direction — the same Newton recovery as the rest of the equidistant
/// family, WITHOUT the wide-angle blend.
///
/// [`blend_fisheye_ray`] exists because a high-order distortion polynomial
/// stops being trustworthy as it approaches its peak; from `r_d = 90°` it
/// blends toward the identity (`θ = r_d`) ray, and past `r_d = 100°` it hands
/// that ray back outright. With a single
/// coefficient there is nothing to distrust — `θ_d = θ·(1 + k1·θ²)` is
/// monotone over any field this model is used on, and the recovery is the
/// exact inverse at every angle — while the blend would DROP the `k1` term
/// exactly where it is largest: a 105° rim at `k1 = 0.02` comes back 6° off,
/// which is a ray, not a rounding error. This is the inverse the bundle
/// adjustment's retriangulation and direction re-estimation read.
///
/// `k1 = 0` short-circuits to [`equidistant_to_ray`], keeping the
/// `SimpleRadialFisheye { k1 = 0 }` convention bit-identical to the native
/// `EQUIDISTANT_FISHEYE` model at every radius. Past the fold (`r_d` above
/// the map's peak, where Newton reports non-convergence) the identity
/// extrapolation is kept: there is no inverse to return there.
pub(in crate::camera::distortion) fn simple_radial_fisheye_to_ray(
    x_d: f64,
    y_d: f64,
    k1: f64,
) -> [f64; 3] {
    if k1 == 0.0 {
        return equidistant_to_ray(x_d, y_d);
    }
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    let (theta, converged) = recover_theta_equidistant(r_d, k1, 0.0, 0.0, 0.0);
    if !converged {
        return equidistant_to_ray(x_d, y_d);
    }
    let s = theta.sin() / r_d;
    [x_d * s, y_d * s, theta.cos()]
}

/// Project a unit ray through the equidistant fisheye model, working in theta-space.
///
/// Computes `theta = atan2(sqrt(rx² + ry²), rz)`, applies the distortion
/// polynomial `theta_d = theta * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)`,
/// and returns `(theta_d * dx, theta_d * dy)` where `(dx, dy)` is the unit
/// direction in the image plane. Returns `None` only when the polynomial is
/// non-monotonic and `theta` exceeds the peak.
pub(in crate::camera::distortion) fn distort_ray_equidistant(
    rx: f64,
    ry: f64,
    rz: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
) -> Option<(f64, f64)> {
    let r_xy = (rx * rx + ry * ry).sqrt();
    let theta = r_xy.atan2(rz);
    if r_xy < 1e-15 {
        return Some((0.0, 0.0));
    }
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;
    let theta6 = theta4 * theta2;
    let theta8 = theta4 * theta4;
    let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

    // Check monotonicity: if theta_d is negative for positive theta, we've
    // exceeded the model's valid range.
    if theta > 0.0 && theta_d <= 0.0 {
        return None;
    }

    let (dx, dy) = (rx / r_xy, ry / r_xy);
    Some((theta_d * dx, theta_d * dy))
}

// ---------------------------------------------------------------------------
// Distortion-free equidistant fisheye (`θ = r/f`)
// ---------------------------------------------------------------------------

/// Angular width, relative to the ray norm, of the on-axis band where the
/// 2D direction `(rx, ry)/r_xy` is numerically meaningless and the Jacobian
/// is evaluated from its axis limit instead.
pub(in crate::camera::distortion) const EQUIDISTANT_AXIS_EPS: f64 = 1e-12;

/// Distorted normalized coordinate `(x_d, y_d)` paired with the 2×3
/// `∂(x_d, y_d)/∂(rx, ry, rz)`, row-major — the pre-intrinsics half of a
/// [`PixelJacobian`](crate::camera::distortion::PixelJacobian), in the optical frame.
pub(in crate::camera::distortion) type NormalizedRayJacobian = ((f64, f64), [[f64; 3]; 2]);

/// Forward equidistant map in tangent-plane coordinates: `(x, y)` with
/// `r = tan θ` in, `(θ·x/r, θ·y/r)` out.
///
/// Only meaningful for `θ < 90°`, where the tangent plane exists; the
/// ray-space entry points ([`distort_ray_equidistant_exact`] and
/// [`equidistant_to_ray`]) carry the model past that and are what the
/// projection pipeline actually calls.
pub(in crate::camera::distortion) fn distort_equidistant(x: f64, y: f64) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }
    let scale = r.atan() / r;
    (x * scale, y * scale)
}

/// Inverse of [`distort_equidistant`]: `(x_d, y_d)` with `r_d = θ` in,
/// tangent-plane `(x, y)` with `r = tan θ` out.
pub(in crate::camera::distortion) fn undistort_equidistant(x_d: f64, y_d: f64) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let scale = r_d.tan() / r_d;
    (x_d * scale, y_d * scale)
}

/// Project an optical-frame ray through the exact `θ = r/f` map.
///
/// `θ = atan2(r_xy, rz) ∈ [0, π]` and the distorted coordinate is `θ` times
/// the unit 2D direction — no polynomial, so the domain is the whole sphere
/// and the result is always `Some`. A ray on the optical axis maps to the
/// principal point; that includes the **antipode** (`θ = π`, `r_xy = 0`),
/// where the map is not injective — every antipodal direction aliases onto
/// `θ = 0`. See [`radial_fisheye_ray_jacobian`], whose derivative is unbounded
/// exactly there.
pub(in crate::camera::distortion) fn distort_ray_equidistant_exact(
    rx: f64,
    ry: f64,
    rz: f64,
) -> (f64, f64) {
    let r_xy = (rx * rx + ry * ry).sqrt();
    if r_xy < 1e-15 {
        return (0.0, 0.0);
    }
    let theta = r_xy.atan2(rz);
    (theta * rx / r_xy, theta * ry / r_xy)
}

/// Distorted coordinate and the analytic `∂(x_d, y_d)/∂(rx, ry, rz)` of the
/// one-coefficient equidistant map `θ_d = θ·(1 + k1·θ²)`, row-major, all in
/// the optical frame.
///
/// `k1 = 0` is [`distort_ray_equidistant_exact`] (the `EQUIDISTANT_FISHEYE`
/// model) and reproduces it bit for bit; a non-zero `k1` is
/// `SIMPLE_RADIAL_FISHEYE`, whose forward map is
/// [`distort_ray_equidistant`] with `k2 = k3 = k4 = 0`. Both are single-focal
/// models whose distorted coordinate is `θ_d` times the unit 2D direction, so
/// one derivative covers both.
///
/// With `ρ = r_xy`, `n² = ρ² + rz²`, unit direction `(ux, uy) = (rx, ry)/ρ`,
/// `θ = atan2(ρ, rz)` and `θ_d' = dθ_d/dθ = 1 + 3·k1·θ²`:
///
/// ```text
/// ∂θ/∂rx = ux·rz/n²   ∂θ/∂ry = uy·rz/n²   ∂θ/∂rz = −ρ/n²
/// ∂ux/∂rx = uy²/ρ     ∂ux/∂ry = −ux·uy/ρ  (and the mirror for uy)
/// ```
///
/// so, chaining `x_d = θ_d(θ)·ux` and writing `c = θ_d'·rz/n² − θ_d/ρ` for
/// the shared off-diagonal factor,
///
/// ```text
/// ∂x_d/∂rx = θ_d·uy²/ρ + θ_d'·ux²·rz/n²   ∂x_d/∂ry = ux·uy·c
///                                         ∂x_d/∂rz = −θ_d'·rx/n²
/// ∂y_d/∂rx = ux·uy·c                      ∂y_d/∂ry = θ_d·ux²/ρ + θ_d'·uy²·rz/n²
///                                         ∂y_d/∂rz = −θ_d'·ry/n²
/// ```
///
/// Nothing here is guarded on `rz`: the expressions are finite and correct
/// past 90°, which is the whole point of a fisheye-native derivative.
///
/// Two limits:
///
/// - **On axis, in front** (`ρ → 0`, `rz > 0`): `θ_d/ρ → 1/rz` and
///   `θ_d' → 1`, so the off-diagonal factor `c → 0` and the third column
///   vanishes, leaving `diag(1/rz, 1/rz)` — the pinhole small-angle
///   Jacobian, independent of both `k1` and the direction `(ux, uy)` that is
///   undefined there.
/// - **At the antipode** (`ρ → 0`, `rz < 0`): `θ → π` while `ρ → 0`, so
///   `θ_d/ρ` diverges and no finite Jacobian exists. Returns `None`; this is
///   the one measure-zero direction where the derivative is narrower than
///   [`distort_ray_equidistant_exact`]'s domain.
///
/// `None` also where the forward map itself is out of domain — a `k1` strong
/// enough to fold `θ_d` non-positive at this `θ`, the same gate
/// [`distort_ray_equidistant`] applies.
pub(in crate::camera::distortion) fn radial_fisheye_ray_jacobian(
    rx: f64,
    ry: f64,
    rz: f64,
    k1: f64,
) -> Option<NormalizedRayJacobian> {
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
    let theta2 = theta * theta;
    let theta_d = theta * (1.0 + k1 * theta2);
    let dtheta_d = 1.0 + 3.0 * k1 * theta2;
    // The forward map's domain gate (`distort_ray_equidistant`): past the
    // fold there is no projection to differentiate.
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
