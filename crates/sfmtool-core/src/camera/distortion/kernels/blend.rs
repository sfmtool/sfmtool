// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The shared tail every fisheye inverse ends in: the undistorted-equidistant
//! ray, and the blend that hands the identity ray back past the radius where
//! the model's polynomial stops describing the lens.

use crate::camera::distortion::{FISHEYE_BLEND_END_RAD, FISHEYE_BLEND_START_RAD};

/// Convert undistorted equidistant coordinates `(uu, vv)` to a unit ray direction.
///
/// `theta = sqrt(uu² + vv²)` is the incidence angle.
/// Returns `[sin(theta) * uu/theta, sin(theta) * vv/theta, cos(theta)]`.
pub(in crate::camera::distortion) fn equidistant_to_ray(uu: f64, vv: f64) -> [f64; 3] {
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
/// peak. This blends from `recovered` to the identity ray with a smoothstep
/// curve over [`FISHEYE_BLEND_START_RAD`] to [`FISHEYE_BLEND_END_RAD`] of
/// `r_d` — 90° to 100°. Since `r_d` is monotonic across the sensor, this
/// produces a smooth spatial transition for all fisheye models.
///
/// The threshold is on the **distorted** radius, so where it lands in
/// incidence angle depends on the lens: `crate::camera::report`'s
/// `trustworthy_max_theta_deg` is what reports that angle for one camera.
pub(in crate::camera::distortion) fn blend_fisheye_ray(
    r_d: f64,
    recovered: [f64; 3],
    undistorted: [f64; 3],
) -> [f64; 3] {
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
