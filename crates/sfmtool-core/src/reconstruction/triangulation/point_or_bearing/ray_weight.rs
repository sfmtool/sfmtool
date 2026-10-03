// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-ray noise weights the point-or-bearing test reads, and the one
//! construction of a ray and its weight from an observation.
//!
//! A weight `W` is a 2×3 world-frame matrix that maps a small change of a
//! ray's direction to the change it makes in the observation's pixel, over the
//! pixel noise. The residual `W (I − d dᵀ) m` is then, to first order, the
//! pixel residual over σ_px, with the camera model's own stretch in each
//! direction: a fisheye's pixels per radian differ radially and tangentially,
//! and a scalar angular noise per ray cannot carry that.

use nalgebra::{Matrix2x3, Matrix3x2, UnitQuaternion, Vector3};

use super::tangent_basis;
use crate::camera::CameraIntrinsics;

/// How far, in pixels, the projection of a pixel's un-projected ray may land
/// from the pixel before [`observed_ray`] calls the pixel outside the camera
/// model's domain. Un-projection converges to far below this inside the
/// domain. Past it the un-projected ray does not project back to the pixel:
/// beyond the start of a fisheye's wide-angle blend (90° of distorted radius,
/// where un-projection deliberately moves toward the equidistant ray and
/// projection does not follow), and beyond a fold of the distortion, where
/// projection is not invertible.
const ROUND_TRIP_TOLERANCE_PX: f64 = 1e-3;

/// One observation as a world-frame ray and its noise weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservedRay {
    /// The unit world-frame ray through the observed pixel.
    pub dir: Vector3<f64>,
    /// `(1/σ_px) · J · R`: `J` the 2×3 derivative of the camera's projection
    /// with respect to the camera-frame ray at this ray, and `R` the
    /// world-to-camera rotation.
    pub weight: Matrix2x3<f64>,
}

/// The ray through `pixel` and its noise weight, for a camera with intrinsics
/// `camera` and world-to-camera rotation `cam_from_world`, at a per-axis pixel
/// noise `sigma_px`. The camera centre is the caller's: it enters the test
/// separately. The projection derivative is the camera's own pixel Jacobian,
/// analytic where the model has one (`CameraModel::supports_pixel_jacobian`)
/// and a central difference otherwise.
///
/// `None` when `sigma_px` is not finite and positive; when the pixel does not
/// un-project to a ray the model projects back to within
/// 1e-3 px of the pixel (it lies outside the model's domain: past a fisheye's
/// wide-angle blend, or past a fold of the distortion); or when a difference
/// probe around the ray leaves the domain.
pub fn observed_ray(
    camera: &CameraIntrinsics,
    cam_from_world: &UnitQuaternion<f64>,
    pixel: [f64; 2],
    sigma_px: f64,
) -> Option<ObservedRay> {
    if !sigma_px.is_finite() || sigma_px <= 0.0 {
        return None;
    }
    let ray = camera.pixel_to_ray(pixel[0], pixel[1]);
    let ray_cam = Vector3::new(ray[0], ray[1], ray[2]);
    let n = ray_cam.norm();
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    let ray_cam = ray_cam / n;
    let (u, v) = camera.ray_to_pixel([ray_cam.x, ray_cam.y, ray_cam.z])?;
    if (u - pixel[0]).hypot(v - pixel[1]) > ROUND_TRIP_TOLERANCE_PX {
        return None;
    }
    let j = camera.pixel_jacobian([ray_cam.x, ray_cam.y, ray_cam.z])?;
    let j = Matrix2x3::new(
        j[0][0], j[0][1], j[0][2], //
        j[1][0], j[1][1], j[1][2],
    );
    let r = cam_from_world.to_rotation_matrix().into_inner();
    let weight = j * r / sigma_px;
    let dir = cam_from_world.inverse() * ray_cam;
    (weight.iter().all(|x| x.is_finite()) && dir.iter().all(|x| x.is_finite()))
        .then_some(ObservedRay { dir, weight })
}

/// The weight of a ray with isotropic angular noise `sigma_rad` (radians):
/// `(1/σ) Bᵀ` with `B` an orthonormal basis perpendicular to `dir`, so that
/// `‖W (I − d dᵀ) m‖² = ‖(I − d dᵀ) m‖² / σ²`, the squared sine over the
/// variance. Which basis is used does not matter: any other differs by a 2×2
/// rotation, which leaves every cost unchanged.
pub fn isotropic_ray_weight(dir: &Vector3<f64>, sigma_rad: f64) -> Matrix2x3<f64> {
    let (b1, b2) = tangent_basis(&dir.normalize());
    Matrix3x2::from_columns(&[b1, b2]).transpose() / sigma_rad
}

/// [`isotropic_ray_weight`] for each ray, `sigma_rad` indexed as `dirs`.
pub fn isotropic_ray_weights(dirs: &[Vector3<f64>], sigma_rad: &[f64]) -> Vec<Matrix2x3<f64>> {
    assert_eq!(
        dirs.len(),
        sigma_rad.len(),
        "sigma_rad must have one entry per ray"
    );
    dirs.iter()
        .zip(sigma_rad)
        .map(|(d, &s)| isotropic_ray_weight(d, s))
        .collect()
}
