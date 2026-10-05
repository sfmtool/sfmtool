// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `RAD_TAN_THIN_PRISM_FISHEYE`: the [`thin_prism`](super::thin_prism) model
//! with the radial and tangential terms evaluated in the *undistorted*
//! tangent plane rather than in `θ`, giving a second 2D Newton solve of the
//! same shape.

use super::newton::{newton_2d, Jacobian2};

/// The rad-tan thin prism fisheye model (Meta/Aria), in the equidistant
/// (theta) space it is actually defined in.
///
/// `(uu, vv)` is already `θ·(dx, dy)`. In that space:
/// 1. Radial scaling: `th_radial = 1 + k0·θ² + k1·θ⁴ + k2·θ⁶ + k3·θ⁸ + k4·θ¹⁰ + k5·θ¹²`
/// 2. Tangential + thin prism on the radially-scaled coordinates
///
/// The rad-tan counterpart of [`distort_thin_prism_equidistant`](super::distort_thin_prism_equidistant), and reached
/// the same two ways: [`distort_rad_tan_thin_prism_fisheye`] for a perspective
/// input, [`distort_ray_rad_tan_thin_prism_fisheye`] for a ray. Regular at the
/// origin, so the on-axis guards belong to those callers.
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_rad_tan_thin_prism_equidistant(
    uu: f64,
    vv: f64,
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

/// Rad-tan thin prism fisheye distortion of **perspective** normalized
/// coordinates — [`distort_rad_tan_thin_prism_equidistant`] behind the
/// `θ = atan r` conversion. See [`distort_thin_prism_fisheye`](super::distort_thin_prism_fisheye) for the same
/// division of labour.
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_rad_tan_thin_prism_fisheye(
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
    let scale_eq = r.atan() / r;
    distort_rad_tan_thin_prism_equidistant(
        x * scale_eq,
        y * scale_eq,
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        p0,
        p1,
        s0,
        s1,
        s2,
        s3,
    )
}

/// Rad-tan thin prism fisheye distortion of an optical-frame ray (+Z forward,
/// y down) — the incidence angle straight into
/// [`distort_rad_tan_thin_prism_equidistant`], with no `tan θ` in the way. The
/// exact forward partner of [`recover_equidistant_rad_tan_thin_prism`].
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn distort_ray_rad_tan_thin_prism_fisheye(
    rx: f64,
    ry: f64,
    rz: f64,
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
    let r_xy = (rx * rx + ry * ry).sqrt();
    if r_xy < 1e-15 {
        return (0.0, 0.0);
    }
    let scale = r_xy.atan2(rz) / r_xy;
    distort_rad_tan_thin_prism_equidistant(
        rx * scale,
        ry * scale,
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        p0,
        p1,
        s0,
        s1,
        s2,
        s3,
    )
}

/// Recover undistorted equidistant coordinates from distorted equidistant
/// coordinates for the rad-tan thin prism fisheye model.
///
/// Uses 2D Newton's method with the analytical Jacobian. When the radial
/// distortion is non-monotonic, prefers the larger-theta (descending side)
/// solution for wide-angle fisheye correctness.
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn recover_equidistant_rad_tan_thin_prism(
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
pub(in crate::camera::distortion) fn newton_rad_tan_thin_prism(
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
    newton_2d(
        x_d,
        y_d,
        uu_init,
        vv_init,
        |uu, vv| {
            distort_rad_tan_thin_prism_equidistant(
                uu, vv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3,
            )
        },
        |uu, vv| {
            rad_tan_thin_prism_with_jacobian(uu, vv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3)
        },
    )
}

/// [`distort_rad_tan_thin_prism_equidistant`] together with its analytical
/// Jacobian, for the Newton step in [`newton_rad_tan_thin_prism`]. The
/// Jacobian is the chain rule through the radial scaling: `J = J_tp · J_eq`.
#[allow(clippy::too_many_arguments)]
fn rad_tan_thin_prism_with_jacobian(
    uu: f64,
    vv: f64,
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
) -> ((f64, f64), Jacobian2) {
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

    let d_th_radial =
        k0 + 2.0 * k1 * th2 + 3.0 * k2 * th4 + 4.0 * k3 * th6 + 5.0 * k4 * th8 + 6.0 * k5 * th10;

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

    ((uu_r + duu, vv_r + dvv), [[j00, j01], [j10, j11]])
}

/// Inverse of rad-tan thin prism fisheye distortion.
///
/// Uses 2D Newton's method in equidistant space, then converts
/// back to perspective coordinates.
#[allow(clippy::too_many_arguments)]
pub(in crate::camera::distortion) fn undistort_rad_tan_thin_prism_fisheye(
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
