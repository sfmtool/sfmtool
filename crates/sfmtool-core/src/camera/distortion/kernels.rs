// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-model distortion math kernels for [`super::CameraModel`].
//!
//! These are the private, model-specific forward/inverse distortion
//! implementations and ray-direction helpers used by the public
//! `distort` / `undistort` / `project` / `unproject` API in the parent
//! module. Kept separate so `distortion.rs` holds only the two public
//! `impl` blocks and the model dispatch.

use super::{FISHEYE_BLEND_END_RAD, FISHEYE_BLEND_START_RAD, UNDISTORT_EPS, UNDISTORT_MAX_ITER};

/// OpenCV distortion: radial (k1, k2) + tangential (p1, p2).
pub(super) fn distort_opencv(x: f64, y: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> (f64, f64) {
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    let x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (x_d, y_d)
}

/// Full OpenCV distortion: rational radial (k1..k6) + tangential (p1, p2).
#[allow(clippy::too_many_arguments)]
pub(super) fn distort_full_opencv(
    x: f64,
    y: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    k6: f64,
) -> (f64, f64) {
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
    let x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (x_d, y_d)
}

/// OpenCV fisheye (equidistant) distortion.
///
/// Maps a 3D ray direction `(x, y, 1)` through the equidistant fisheye model:
/// `theta_d = theta * (1 + k1*theta² + k2*theta⁴ + k3*theta⁶ + k4*theta⁸)`
/// where `theta = atan(r)` and `r = sqrt(x² + y²)`.
pub(super) fn distort_fisheye(x: f64, y: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> (f64, f64) {
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
pub(super) fn recover_theta_equidistant(
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
pub(super) fn undistort_fisheye(
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
pub(super) fn distort_simple_radial_fisheye(x: f64, y: f64, k: f64) -> (f64, f64) {
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
pub(super) fn undistort_simple_radial_fisheye(x_d: f64, y_d: f64, k: f64) -> (f64, f64) {
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
pub(super) fn distort_radial_fisheye(x: f64, y: f64, k1: f64, k2: f64) -> (f64, f64) {
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
pub(super) fn undistort_radial_fisheye(x_d: f64, y_d: f64, k1: f64, k2: f64) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (theta, _) = recover_theta_equidistant(r_d, k1, k2, 0.0, 0.0);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

/// Thin prism fisheye distortion.
///
/// Applies the equidistant base projection, then additive distortion in
/// equidistant (theta) space: radial + tangential + thin prism.
///
/// The distortion in theta-space (where `(uu, vv)` are equidistant coords):
/// ```text
/// θ² = uu² + vv²
/// radial = k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸
/// duu = uu·radial + 2·p1·uu·vv + p2·(θ² + 2·uu²) + sx1·θ²
/// dvv = vv·radial + 2·p2·uu·vv + p1·(θ² + 2·vv²) + sy1·θ²
/// ```
#[allow(clippy::too_many_arguments)]
pub(super) fn distort_thin_prism_fisheye(
    x: f64,
    y: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    sx1: f64,
    sy1: f64,
) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }

    // Convert perspective (x, y) to equidistant (uu, vv)
    let theta = r.atan();
    let scale_eq = theta / r;
    let uu = x * scale_eq;
    let vv = y * scale_eq;

    // Additive distortion in equidistant space
    let theta2 = uu * uu + vv * vv;
    let theta4 = theta2 * theta2;
    let theta6 = theta4 * theta2;
    let theta8 = theta4 * theta4;

    let radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
    let duu = uu * radial + 2.0 * p1 * uu * vv + p2 * (theta2 + 2.0 * uu * uu) + sx1 * theta2;
    let dvv = vv * radial + 2.0 * p2 * uu * vv + p1 * (theta2 + 2.0 * vv * vv) + sy1 * theta2;

    (uu + duu, vv + dvv)
}

/// Recover undistorted equidistant coordinates from distorted equidistant
/// coordinates for the thin prism fisheye model.
///
/// Uses 2D Newton's method with the analytical Jacobian of the forward
/// distortion function. The forward function is:
///   F(uu, vv) = (uu + duu(uu, vv), vv + dvv(uu, vv))
/// and we solve F(uu, vv) = (x_d, y_d).
///
/// When the radial distortion polynomial is non-monotonic (has a peak),
/// two distinct (uu, vv) can map to the same (x_d, y_d). In that case
/// we prefer the solution with larger theta (the descending side of the
/// peak), which is the physically correct branch for wide-angle fisheye.
#[allow(clippy::too_many_arguments)]
pub(super) fn recover_equidistant_thin_prism(
    x_d: f64,
    y_d: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    sx1: f64,
    sy1: f64,
) -> (f64, f64) {
    newton_thin_prism(x_d, y_d, x_d, y_d, k1, k2, p1, p2, k3, k4, sx1, sy1)
}

/// Run 2D Newton's method for thin prism fisheye undistortion.
#[allow(clippy::too_many_arguments)]
pub(super) fn newton_thin_prism(
    x_d: f64,
    y_d: f64,
    uu_init: f64,
    vv_init: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    sx1: f64,
    sy1: f64,
) -> (f64, f64) {
    let mut uu = uu_init;
    let mut vv = vv_init;
    for _ in 0..UNDISTORT_MAX_ITER {
        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let theta2 = uu2 + vv2;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;

        let radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
        let duu = uu * radial + 2.0 * p1 * uu * vv + p2 * (theta2 + 2.0 * uu2) + sx1 * theta2;
        let dvv = vv * radial + 2.0 * p2 * uu * vv + p1 * (theta2 + 2.0 * vv2) + sy1 * theta2;

        // Residual: F(uu, vv) - (x_d, y_d)
        let res_u = uu + duu - x_d;
        let res_v = vv + dvv - y_d;

        let res_norm = res_u * res_u + res_v * res_v;
        if res_u.abs() + res_v.abs() < UNDISTORT_EPS {
            break;
        }

        // Jacobian of F(uu, vv) = (uu + duu, vv + dvv)
        let d_radial = k1 + 2.0 * k2 * theta2 + 3.0 * k3 * theta4 + 4.0 * k4 * theta6;

        let j00 =
            1.0 + radial + 2.0 * uu2 * d_radial + 2.0 * p1 * vv + 6.0 * p2 * uu + 2.0 * sx1 * uu;
        let j01 = 2.0 * uu * vv * d_radial + 2.0 * p1 * uu + 2.0 * p2 * vv + 2.0 * sx1 * vv;
        let j10 = 2.0 * uu * vv * d_radial + 2.0 * p2 * vv + 2.0 * p1 * uu + 2.0 * sy1 * uu;
        let j11 =
            1.0 + radial + 2.0 * vv2 * d_radial + 2.0 * p2 * uu + 6.0 * p1 * vv + 2.0 * sy1 * vv;

        // Solve J * delta = residual via 2x2 inverse
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-30 {
            break;
        }
        let inv_det = 1.0 / det;
        let delta_uu = (j11 * res_u - j01 * res_v) * inv_det;
        let delta_vv = (-j10 * res_u + j00 * res_v) * inv_det;

        // Backtracking line search: halve the step until the residual decreases.
        let mut alpha = 1.0;
        for _ in 0..10 {
            let uu_t = uu - alpha * delta_uu;
            let vv_t = vv - alpha * delta_vv;
            let t2 = uu_t * uu_t + vv_t * vv_t;
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let t8 = t4 * t4;
            let rad_t = k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
            let du_t =
                uu_t * rad_t + 2.0 * p1 * uu_t * vv_t + p2 * (t2 + 2.0 * uu_t * uu_t) + sx1 * t2;
            let dv_t =
                vv_t * rad_t + 2.0 * p2 * uu_t * vv_t + p1 * (t2 + 2.0 * vv_t * vv_t) + sy1 * t2;
            let ru = uu_t + du_t - x_d;
            let rv = vv_t + dv_t - y_d;
            if ru * ru + rv * rv < res_norm {
                break;
            }
            alpha *= 0.5;
        }

        uu -= alpha * delta_uu;
        vv -= alpha * delta_vv;
    }
    (uu, vv)
}

/// Inverse of thin prism fisheye distortion.
///
/// Uses 2D Newton's method in equidistant space, then converts
/// back to perspective coordinates.
#[allow(clippy::too_many_arguments)]
pub(super) fn undistort_thin_prism_fisheye(
    x_d: f64,
    y_d: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    sx1: f64,
    sy1: f64,
) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (uu, vv) = recover_equidistant_thin_prism(x_d, y_d, k1, k2, p1, p2, k3, k4, sx1, sy1);
    let theta = (uu * uu + vv * vv).sqrt();
    if theta < 1e-15 {
        return (uu, vv);
    }
    let r = theta.tan();
    let scale = r / theta;
    (uu * scale, vv * scale)
}

/// Rad-tan thin prism fisheye distortion (Meta/Aria model).
///
/// Applies the equidistant base projection, then in equidistant (theta) space:
/// 1. Radial scaling: `th_radial = 1 + k0·θ² + k1·θ⁴ + k2·θ⁶ + k3·θ⁸ + k4·θ¹⁰ + k5·θ¹²`
/// 2. Tangential + thin prism on the radially-scaled coordinates
#[allow(clippy::too_many_arguments)]
pub(super) fn distort_rad_tan_thin_prism_fisheye(
    x: f64,
    y: f64,
    k0: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    p0: f64,
    p1: f64,
    s0: f64,
    s1: f64,
    s2: f64,
    s3: f64,
) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    if r < 1e-15 {
        return (x, y);
    }

    // Convert perspective (x, y) to equidistant (uu, vv)
    let theta = r.atan();
    let scale_eq = theta / r;
    let uu = x * scale_eq;
    let vv = y * scale_eq;

    // Radial scaling in equidistant space
    let th2 = uu * uu + vv * vv;
    let th4 = th2 * th2;
    let th6 = th4 * th2;
    let th8 = th4 * th4;
    let th10 = th8 * th2;
    let th12 = th8 * th4;
    let th_radial = 1.0 + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8 + k4 * th10 + k5 * th12;
    let uu_r = uu * th_radial;
    let vv_r = vv * th_radial;

    // Tangential + thin prism on radially-scaled coordinates
    let uu_r2 = uu_r * uu_r;
    let vv_r2 = vv_r * vv_r;
    let r2 = uu_r2 + vv_r2;
    let r4 = r2 * r2;
    let duu = 2.0 * p1 * uu_r * vv_r + p0 * (r2 + 2.0 * uu_r2) + s0 * r2 + s1 * r4;
    let dvv = p1 * (r2 + 2.0 * vv_r2) + 2.0 * p0 * uu_r * vv_r + s2 * r2 + s3 * r4;

    (uu_r + duu, vv_r + dvv)
}

/// Recover undistorted equidistant coordinates from distorted equidistant
/// coordinates for the rad-tan thin prism fisheye model.
///
/// Uses 2D Newton's method with the analytical Jacobian. When the radial
/// distortion is non-monotonic, prefers the larger-theta (descending side)
/// solution for wide-angle fisheye correctness.
#[allow(clippy::too_many_arguments)]
pub(super) fn recover_equidistant_rad_tan_thin_prism(
    x_d: f64,
    y_d: f64,
    k0: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    p0: f64,
    p1: f64,
    s0: f64,
    s1: f64,
    s2: f64,
    s3: f64,
) -> (f64, f64) {
    newton_rad_tan_thin_prism(
        x_d, y_d, x_d, y_d, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3,
    )
}

/// Run 2D Newton's method for rad-tan thin prism fisheye undistortion.
#[allow(clippy::too_many_arguments)]
pub(super) fn newton_rad_tan_thin_prism(
    x_d: f64,
    y_d: f64,
    uu_init: f64,
    vv_init: f64,
    k0: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    p0: f64,
    p1: f64,
    s0: f64,
    s1: f64,
    s2: f64,
    s3: f64,
) -> (f64, f64) {
    let mut uu = uu_init;
    let mut vv = vv_init;
    for _ in 0..UNDISTORT_MAX_ITER {
        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let th2 = uu2 + vv2;
        let th4 = th2 * th2;
        let th6 = th4 * th2;
        let th8 = th4 * th4;
        let th10 = th8 * th2;
        let th12 = th8 * th4;
        let th_radial = 1.0 + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8 + k4 * th10 + k5 * th12;
        let uu_r = uu * th_radial;
        let vv_r = vv * th_radial;

        let uu_r2 = uu_r * uu_r;
        let vv_r2 = vv_r * vv_r;
        let r2 = uu_r2 + vv_r2;
        let r4 = r2 * r2;
        let duu = 2.0 * p1 * uu_r * vv_r + p0 * (r2 + 2.0 * uu_r2) + s0 * r2 + s1 * r4;
        let dvv = p1 * (r2 + 2.0 * vv_r2) + 2.0 * p0 * uu_r * vv_r + s2 * r2 + s3 * r4;

        let res_u = uu_r + duu - x_d;
        let res_v = vv_r + dvv - y_d;

        let res_norm = res_u * res_u + res_v * res_v;
        if res_u.abs() + res_v.abs() < UNDISTORT_EPS {
            break;
        }

        // Jacobian via chain rule: J = J_tp * J_eq
        let d_th_radial = k0
            + 2.0 * k1 * th2
            + 3.0 * k2 * th4
            + 4.0 * k3 * th6
            + 5.0 * k4 * th8
            + 6.0 * k5 * th10;

        let eq00 = th_radial + 2.0 * uu2 * d_th_radial;
        let eq01 = 2.0 * uu * vv * d_th_radial;
        let eq11 = th_radial + 2.0 * vv2 * d_th_radial;

        let dduu_duur = 2.0 * p1 * vv_r + 6.0 * p0 * uu_r + (2.0 * s0 + 4.0 * s1 * r2) * uu_r;
        let dduu_dvvr = 2.0 * p1 * uu_r + 2.0 * p0 * vv_r + (2.0 * s0 + 4.0 * s1 * r2) * vv_r;
        let ddvv_duur = 2.0 * p1 * uu_r + 2.0 * p0 * vv_r + (2.0 * s2 + 4.0 * s3 * r2) * uu_r;
        let ddvv_dvvr = 6.0 * p1 * vv_r + 2.0 * p0 * uu_r + (2.0 * s2 + 4.0 * s3 * r2) * vv_r;

        let tp00 = 1.0 + dduu_duur;
        let tp01 = dduu_dvvr;
        let tp10 = ddvv_duur;
        let tp11 = 1.0 + ddvv_dvvr;

        let j00 = tp00 * eq00 + tp01 * eq01;
        let j01 = tp00 * eq01 + tp01 * eq11;
        let j10 = tp10 * eq00 + tp11 * eq01;
        let j11 = tp10 * eq01 + tp11 * eq11;

        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-30 {
            break;
        }
        let inv_det = 1.0 / det;
        let delta_uu = (j11 * res_u - j01 * res_v) * inv_det;
        let delta_vv = (-j10 * res_u + j00 * res_v) * inv_det;

        // Backtracking line search
        let mut alpha = 1.0;
        for _ in 0..10 {
            let uu_t = uu - alpha * delta_uu;
            let vv_t = vv - alpha * delta_vv;
            let t2 = uu_t * uu_t + vv_t * vv_t;
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let t8 = t4 * t4;
            let t10 = t8 * t2;
            let t12 = t8 * t4;
            let thr = 1.0 + k0 * t2 + k1 * t4 + k2 * t6 + k3 * t8 + k4 * t10 + k5 * t12;
            let ur = uu_t * thr;
            let vr = vv_t * thr;
            let ur2 = ur * ur;
            let vr2 = vr * vr;
            let rr2 = ur2 + vr2;
            let rr4 = rr2 * rr2;
            let du = 2.0 * p1 * ur * vr + p0 * (rr2 + 2.0 * ur2) + s0 * rr2 + s1 * rr4;
            let dv = p1 * (rr2 + 2.0 * vr2) + 2.0 * p0 * ur * vr + s2 * rr2 + s3 * rr4;
            let ru = ur + du - x_d;
            let rv = vr + dv - y_d;
            if ru * ru + rv * rv < res_norm {
                break;
            }
            alpha *= 0.5;
        }

        uu -= alpha * delta_uu;
        vv -= alpha * delta_vv;
    }
    (uu, vv)
}

/// Inverse of rad-tan thin prism fisheye distortion.
///
/// Uses 2D Newton's method in equidistant space, then converts
/// back to perspective coordinates.
#[allow(clippy::too_many_arguments)]
pub(super) fn undistort_rad_tan_thin_prism_fisheye(
    x_d: f64,
    y_d: f64,
    k0: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    p0: f64,
    p1: f64,
    s0: f64,
    s1: f64,
    s2: f64,
    s3: f64,
) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let (uu, vv) = recover_equidistant_rad_tan_thin_prism(
        x_d, y_d, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3,
    );
    let theta = (uu * uu + vv * vv).sqrt();
    if theta < 1e-15 {
        return (uu, vv);
    }
    let r = theta.tan();
    let scale = r / theta;
    (uu * scale, vv * scale)
}

// ---------------------------------------------------------------------------
// Ray direction helpers
// ---------------------------------------------------------------------------

/// Convert distorted equidistant fisheye coordinates to a unit ray direction.
///
/// Recovers theta via Newton's method, then builds the ray as
/// `[sin(theta) * x_d/r_d, sin(theta) * y_d/r_d, cos(theta)]`.
/// Works correctly for any field of view, including beyond 180°.
pub(super) fn equidistant_fisheye_to_ray(
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

/// Project a unit ray through the equidistant fisheye model, working in theta-space.
///
/// Computes `theta = atan2(sqrt(rx² + ry²), rz)`, applies the distortion
/// polynomial `theta_d = theta * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)`,
/// and returns `(theta_d * dx, theta_d * dy)` where `(dx, dy)` is the unit
/// direction in the image plane. Returns `None` only when the polynomial is
/// non-monotonic and `theta` exceeds the peak.
pub(super) fn distort_ray_equidistant(
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

/// Convert undistorted equidistant coordinates `(uu, vv)` to a unit ray direction.
///
/// `theta = sqrt(uu² + vv²)` is the incidence angle.
/// Returns `[sin(theta) * uu/theta, sin(theta) * vv/theta, cos(theta)]`.
pub(super) fn equidistant_to_ray(uu: f64, vv: f64) -> [f64; 3] {
    let theta = (uu * uu + vv * vv).sqrt();
    if theta < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let s = sin_theta / theta;
    [uu * s, vv * s, cos_theta]
}

/// Blend a recovered fisheye ray toward the undistorted (identity) ray for
/// wide angles, returning a unit ray.
///
/// `r_d` is the distorted radial distance (= undistorted angle when k=0).
/// `recovered` is the ray from the model-specific Newton solver, and
/// `undistorted` is the identity-model ray (`equidistant_to_ray(x_d, y_d)`).
///
/// High-order distortion polynomials become unreliable approaching their
/// peak. This blends from `recovered` to the identity ray over 80°–90° of
/// `r_d` using a smoothstep curve. Since `r_d` is monotonic across the
/// sensor, this produces a smooth spatial transition for all fisheye models.
pub(super) fn blend_fisheye_ray(r_d: f64, recovered: [f64; 3], undistorted: [f64; 3]) -> [f64; 3] {
    if r_d <= FISHEYE_BLEND_START_RAD {
        return recovered;
    }
    if r_d >= FISHEYE_BLEND_END_RAD {
        return undistorted;
    }
    let t = (r_d - FISHEYE_BLEND_START_RAD) / (FISHEYE_BLEND_END_RAD - FISHEYE_BLEND_START_RAD);
    let s = t * t * (3.0 - 2.0 * t); // smoothstep
    let rx = recovered[0] * (1.0 - s) + undistorted[0] * s;
    let ry = recovered[1] * (1.0 - s) + undistorted[1] * s;
    let rz = recovered[2] * (1.0 - s) + undistorted[2] * s;
    let len = (rx * rx + ry * ry + rz * rz).sqrt();
    [rx / len, ry / len, rz / len]
}
