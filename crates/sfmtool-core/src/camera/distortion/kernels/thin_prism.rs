// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `THIN_PRISM_FISHEYE`: an equidistant fisheye base with radial (k1–k4),
//! tangential (p1, p2) and thin-prism (s1–s4) terms in the distorted
//! tangent plane, inverted by a 2D Newton solve.

use crate::camera::distortion::{UNDISTORT_EPS, UNDISTORT_MAX_ITER};

/// The thin prism fisheye model's additive distortion, in the equidistant
/// (theta) space the model is actually defined in.
///
/// `(uu, vv)` is already `θ·(dx, dy)` for the incidence angle `θ` and the unit
/// 2D direction `(dx, dy)`. Every entry point below reaches the model through
/// here: the perspective wrapper [`distort_thin_prism_fisheye`] converts
/// `(x, y)` into this space first, and the ray entry point
/// [`distort_ray_thin_prism_fisheye`] forms `(uu, vv)` straight from a ray
/// direction, never passing through perspective coordinates at all.
///
/// ```text
/// θ² = uu² + vv²
/// radial = k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸
/// duu = uu·radial + 2·p1·uu·vv + p2·(θ² + 2·uu²) + sx1·θ²
/// dvv = vv·radial + 2·p2·uu·vv + p1·(θ² + 2·vv²) + sy1·θ²
/// ```
///
/// Regular at the origin — every term above carries a factor of `θ²` or of
/// `(uu, vv)` — so there is no on-axis guard here. The guards live in the
/// callers, which are the ones that divide by a radius.
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_thin_prism_equidistant(
    uu: f64,
    vv: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    sx1: f64,
    sy1: f64,
) -> (f64, f64) {
    let theta2 = uu * uu + vv * vv;
    let theta4 = theta2 * theta2;
    let theta6 = theta4 * theta2;
    let theta8 = theta4 * theta4;

    let radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
    let duu = uu * radial + 2.0 * p1 * uu * vv + p2 * (theta2 + 2.0 * uu * uu) + sx1 * theta2;
    let dvv = vv * radial + 2.0 * p2 * uu * vv + p1 * (theta2 + 2.0 * vv * vv) + sy1 * theta2;

    (uu + duu, vv + dvv)
}

/// Thin prism fisheye distortion of **perspective** normalized coordinates.
///
/// `(x, y)` is the quotient `(rx/rz, ry/rz)`; this converts it to the
/// equidistant `(uu, vv)` that [`distort_thin_prism_equidistant`] is defined on
/// (`θ = atan r`) and applies the model there. A caller holding a ray rather
/// than that quotient wants [`distort_ray_thin_prism_fisheye`], which skips the
/// conversion rather than undoing it.
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_thin_prism_fisheye(
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
    let scale_eq = r.atan() / r;
    distort_thin_prism_equidistant(x * scale_eq, y * scale_eq, k1, k2, p1, p2, k3, k4, sx1, sy1)
}

/// Thin prism fisheye distortion of an optical-frame ray (+Z forward, y down).
///
/// `θ = atan2(√(rx² + ry²), rz)` is the incidence angle off the optical axis,
/// which gives the equidistant `(uu, vv) = θ·(dx, dy)` the model is defined on
/// directly: no `tan θ`, so no singularity at 90°, and no detour through
/// perspective coordinates. The exact forward partner of the theta-space
/// Newton recovery in [`recover_equidistant_thin_prism`].
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_ray_thin_prism_fisheye(
    rx: f64,
    ry: f64,
    rz: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    k3: f64,
    k4: f64,
    sx1: f64,
    sy1: f64,
) -> (f64, f64) {
    let r_xy = (rx * rx + ry * ry).sqrt();
    if r_xy < 1e-15 {
        return (0.0, 0.0);
    }
    let scale = r_xy.atan2(rz) / r_xy;
    distort_thin_prism_equidistant(rx * scale, ry * scale, k1, k2, p1, p2, k3, k4, sx1, sy1)
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
pub(in crate::camera::distortion) fn recover_equidistant_thin_prism(
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
pub(in crate::camera::distortion) fn newton_thin_prism(
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
pub(in crate::camera::distortion) fn undistort_thin_prism_fisheye(
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
