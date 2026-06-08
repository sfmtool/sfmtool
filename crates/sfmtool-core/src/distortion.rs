// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Lens distortion and undistortion for COLMAP camera models.
//!
//! Provides forward distortion (undistorted → distorted normalized coordinates)
//! and iterative undistortion (distorted → undistorted) for all supported camera
//! models. Convenience wrappers on [`CameraIntrinsics`] handle the full
//! pixel ↔ normalized coordinate conversion.
//!
//! # Coordinate systems
//!
//! This module uses two coordinate systems:
//!
//! **Image-plane coordinates** `(x, y)` are obtained by projecting a
//! camera-space 3D point onto the image plane: `(x, y) = (X/Z, Y/Z)`. The
//! origin `(0, 0)` is the optical axis (principal ray). Values are unbounded
//! and represent the tangent of the angle from the optical axis — a point at
//! 45° off-axis has `|x|` or `|y|` of 1.0. These are **not** the same as
//! normalized device coordinates (NDC), which typically span `[-1, 1]`.
//!
//! **Pixel coordinates** `(u, v)` have the origin at the top-left of the image,
//! with `u` increasing rightward and `v` increasing downward. The principal
//! point `(cx, cy)` maps to image-plane `(0, 0)`.
//!
//! ## COLMAP projection pipeline
//!
//! ```text
//! 3D point → image-plane (x = X/Z, y = Y/Z) → distort → pixel (u = fx*x_d + cx)
//! pixel → distorted image-plane (x_d = (u-cx)/fx) → undistort → ray direction (x, y, 1)
//! ```
//!
//! The `distort` and `undistort` methods on [`CameraModel`] operate in
//! image-plane coordinates. The `project` and `unproject` methods on
//! [`CameraIntrinsics`] handle the full pixel ↔ image-plane conversion.

use rayon::prelude::*;

use crate::camera_intrinsics::{CameraIntrinsics, CameraIntrinsicsError, CameraModel};

/// Maximum iterations for iterative undistortion.
const UNDISTORT_MAX_ITER: usize = 100;

/// Convergence threshold for iterative undistortion.
const UNDISTORT_EPS: f64 = 1e-10;

/// Fisheye distortion models are not coherent past ~90° from the optical axis,
/// so we blend from the distorted ray to the undistorted (identity) ray over
/// this angular range (in radians of the undistorted angle).
const FISHEYE_BLEND_START_RAD: f64 = 90.0 * (std::f64::consts::PI / 180.0); // 90°
const FISHEYE_BLEND_END_RAD: f64 = 100.0 * (std::f64::consts::PI / 180.0); // 100°

mod kernels;
use kernels::*;

// ---------------------------------------------------------------------------
// CameraModel: normalized-space distortion
// ---------------------------------------------------------------------------

impl CameraModel {
    /// Apply forward distortion: undistorted image-plane → distorted image-plane.
    ///
    /// For pinhole models (no distortion), returns `(x, y)` unchanged.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::Equirectangular { .. } => (x, y),

            CameraModel::SimpleRadial {
                radial_distortion_k1: k1,
                ..
            } => {
                let r2 = x * x + y * y;
                let radial = 1.0 + k1 * r2;
                (x * radial, y * radial)
            }

            CameraModel::Radial {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => {
                let r2 = x * x + y * y;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                (x * radial, y * radial)
            }

            CameraModel::OpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                ..
            } => distort_opencv(x, y, *k1, *k2, *p1, *p2),

            CameraModel::OpenCVFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                ..
            } => distort_fisheye(x, y, *k1, *k2, *k3, *k4),

            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k,
                ..
            } => distort_simple_radial_fisheye(x, y, *k),

            CameraModel::RadialFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => distort_radial_fisheye(x, y, *k1, *k2),

            CameraModel::ThinPrismFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                thin_prism_sx1: sx1,
                thin_prism_sy1: sy1,
                ..
            } => distort_thin_prism_fisheye(x, y, *k1, *k2, *p1, *p2, *k3, *k4, *sx1, *sy1),

            CameraModel::RadTanThinPrismFisheye {
                radial_distortion_k0: k0,
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                tangential_distortion_p0: p0,
                tangential_distortion_p1: p1,
                thin_prism_s0: s0,
                thin_prism_s1: s1,
                thin_prism_s2: s2,
                thin_prism_s3: s3,
                ..
            } => distort_rad_tan_thin_prism_fisheye(
                x, y, *k0, *k1, *k2, *k3, *k4, *k5, *p0, *p1, *s0, *s1, *s2, *s3,
            ),

            CameraModel::FullOpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                radial_distortion_k6: k6,
                ..
            } => distort_full_opencv(x, y, *k1, *k2, *p1, *p2, *k3, *k4, *k5, *k6),
        }
    }

    /// Whether the normalized image-plane point `(x, y)` lies in the
    /// distortion polynomial's principal monotonic branch — the branch
    /// connected to the origin via positive radial growth.
    ///
    /// Beyond the first inflection of the polynomial, the forward map
    /// stops being injective: the same distorted pixel can be reached from
    /// multiple ray directions, producing ghost / mirror projections
    /// outside the camera's true FOV. [`distort_ray`] uses this to gate
    /// rays before calling [`distort`].
    ///
    /// For radially-symmetric distortion (`xd = x · g(r²)`,
    /// `yd = y · g(r²)`) the principal branch is the region where the
    /// radial scalar `g > 0` and the radial Jacobian factor
    /// `g + 2r² g' > 0` are both positive. Either crossing zero means we
    /// have either folded sign or passed an inflection.
    ///
    /// For models with tangential terms (OpenCV / FullOpenCV) we apply
    /// the radial branch test to the radial part and additionally require
    /// the full Jacobian (computed via central differences) to be positive
    /// at `(x, y)`.
    ///
    /// Only meaningful for the perspective-model family that goes through
    /// [`distort`]; for fisheye and equirectangular models — which take
    /// different code paths in [`distort_ray`] — this returns `true`.
    fn forward_projection_invertible(&self, x: f64, y: f64) -> bool {
        match self {
            CameraModel::Pinhole { .. } | CameraModel::SimplePinhole { .. } => true,
            CameraModel::SimpleRadial {
                radial_distortion_k1: k1,
                ..
            } => {
                // Principal branch: 1 + k1 r² > 0 and 1 + 3 k1 r² > 0.
                let r2 = x * x + y * y;
                (1.0 + k1 * r2) > 0.0 && (1.0 + 3.0 * k1 * r2) > 0.0
            }
            CameraModel::Radial {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => {
                // Principal branch: g and (g + 2r² g') both positive.
                let r2 = x * x + y * y;
                let g = 1.0 + k1 * r2 + k2 * r2 * r2;
                let g_jac = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2;
                g > 0.0 && g_jac > 0.0
            }
            CameraModel::OpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            }
            | CameraModel::FullOpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => {
                // Radial sign check (rough proxy — picks up the dominant
                // fold even with k3..k6 / rational denominator at higher
                // orders) plus a numerical det(J) > 0 at (x, y) to catch
                // local non-invertibility from the tangential terms.
                let r2 = x * x + y * y;
                if (1.0 + k1 * r2 + k2 * r2 * r2) <= 0.0 {
                    return false;
                }
                let h = 1e-5;
                let (xpx, ypx) = self.distort(x + h, y);
                let (xmx, ymx) = self.distort(x - h, y);
                let (xpy, ypy) = self.distort(x, y + h);
                let (xmy, ymy) = self.distort(x, y - h);
                let dxd_dx = (xpx - xmx) / (2.0 * h);
                let dyd_dx = (ypx - ymx) / (2.0 * h);
                let dxd_dy = (xpy - xmy) / (2.0 * h);
                let dyd_dy = (ypy - ymy) / (2.0 * h);
                (dxd_dx * dyd_dy - dxd_dy * dyd_dx) > 0.0
            }
            // Non-perspective models reach this only via accidental call.
            _ => true,
        }
    }

    /// Remove distortion: distorted image-plane → undistorted image-plane.
    ///
    /// Uses iterative fixed-point solving. For pinhole models, returns the
    /// input unchanged. For fisheye, uses Newton's method on the scalar
    /// theta mapping.
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        match self {
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::Equirectangular { .. } => (x_d, y_d),

            CameraModel::OpenCVFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                ..
            } => undistort_fisheye(x_d, y_d, *k1, *k2, *k3, *k4),

            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k,
                ..
            } => undistort_simple_radial_fisheye(x_d, y_d, *k),

            CameraModel::RadialFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => undistort_radial_fisheye(x_d, y_d, *k1, *k2),

            CameraModel::ThinPrismFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                thin_prism_sx1: sx1,
                thin_prism_sy1: sy1,
                ..
            } => undistort_thin_prism_fisheye(x_d, y_d, *k1, *k2, *p1, *p2, *k3, *k4, *sx1, *sy1),

            CameraModel::RadTanThinPrismFisheye {
                radial_distortion_k0: k0,
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                tangential_distortion_p0: p0,
                tangential_distortion_p1: p1,
                thin_prism_s0: s0,
                thin_prism_s1: s1,
                thin_prism_s2: s2,
                thin_prism_s3: s3,
                ..
            } => undistort_rad_tan_thin_prism_fisheye(
                x_d, y_d, *k0, *k1, *k2, *k3, *k4, *k5, *p0, *p1, *s0, *s1, *s2, *s3,
            ),

            _ => {
                // Generic iterative fixed-point undistortion.
                // Initialize with the distorted point as the first guess.
                let mut x = x_d;
                let mut y = y_d;
                for _ in 0..UNDISTORT_MAX_ITER {
                    let (x_d_est, y_d_est) = self.distort(x, y);
                    let dx = x_d - x_d_est;
                    let dy = y_d - y_d_est;
                    x += dx;
                    y += dy;
                    if dx.abs() + dy.abs() < UNDISTORT_EPS {
                        break;
                    }
                }
                (x, y)
            }
        }
    }

    /// Apply forward distortion to a batch of points.
    ///
    /// Parallelized with rayon — negligible overhead for small inputs,
    /// scales to millions of points.
    pub fn distort_batch(&self, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        points
            .par_iter()
            .map(|&[x, y]| {
                let (xd, yd) = self.distort(x, y);
                [xd, yd]
            })
            .collect()
    }

    /// Remove distortion from a batch of points.
    ///
    /// Parallelized with rayon — negligible overhead for small inputs,
    /// scales to millions of points.
    pub fn undistort_batch(&self, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        points
            .par_iter()
            .map(|&[x_d, y_d]| {
                let (x, y) = self.undistort(x_d, y_d);
                [x, y]
            })
            .collect()
    }

    /// Project a unit ray direction in camera space to distorted normalized
    /// coordinates.
    ///
    /// For perspective models, computes `(rx/rz, ry/rz)` then applies
    /// distortion. For fisheye models, computes the distorted coordinates
    /// directly from the incidence angle `theta = atan2(sqrt(rx² + ry²), rz)`,
    /// avoiding the `tan(theta)` singularity. For equirectangular, maps via
    /// longitude/latitude. This is the true inverse of [`undistort_to_ray`].
    ///
    /// Returns `None` if the ray falls outside the model's valid domain:
    /// for perspective models, `theta >= pi/2`; for fisheye, only when the
    /// distortion polynomial's representable range is exceeded.
    pub fn distort_ray(&self, ray: [f64; 3]) -> Option<(f64, f64)> {
        let [rx, ry, rz] = ray;
        match self {
            // Equirectangular: longitude/latitude mapping
            CameraModel::Equirectangular { .. } => {
                let longitude = rx.atan2(rz);
                let r_len = (rx * rx + ry * ry + rz * rz).sqrt();
                let latitude = -(ry / r_len).clamp(-1.0, 1.0).asin();
                Some((longitude, latitude))
            }

            // Perspective models: divide by rz, then distort
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::SimpleRadial { .. }
            | CameraModel::Radial { .. }
            | CameraModel::OpenCV { .. }
            | CameraModel::FullOpenCV { .. } => {
                if rz <= 0.0 {
                    return None;
                }
                let x = rx / rz;
                let y = ry / rz;
                // Reject rays that fall outside the distortion polynomial's
                // principal monotonic branch. Beyond the first inflection
                // the forward map stops being injective and produces ghost
                // projections at spurious pixels inside the image rectangle.
                if !self.forward_projection_invertible(x, y) {
                    return None;
                }
                let (x_d, y_d) = self.distort(x, y);
                Some((x_d, y_d))
            }

            // Fisheye models: work in theta-space
            CameraModel::OpenCVFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                ..
            } => distort_ray_equidistant(rx, ry, rz, *k1, *k2, *k3, *k4),

            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k,
                ..
            } => distort_ray_equidistant(rx, ry, rz, *k, 0.0, 0.0, 0.0),

            CameraModel::RadialFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => distort_ray_equidistant(rx, ry, rz, *k1, *k2, 0.0, 0.0),

            CameraModel::ThinPrismFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                thin_prism_sx1: sx1,
                thin_prism_sy1: sy1,
                ..
            } => {
                let r_xy = (rx * rx + ry * ry).sqrt();
                let theta = r_xy.atan2(rz);
                if r_xy < 1e-15 {
                    return Some((0.0, 0.0));
                }
                let (dx, dy) = (rx / r_xy, ry / r_xy);
                let uu = theta * dx;
                let vv = theta * dy;
                let (x_d, y_d) =
                    distort_thin_prism_fisheye(uu, vv, *k1, *k2, *p1, *p2, *k3, *k4, *sx1, *sy1);
                Some((x_d, y_d))
            }

            CameraModel::RadTanThinPrismFisheye {
                radial_distortion_k0: k0,
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                tangential_distortion_p0: p0,
                tangential_distortion_p1: p1,
                thin_prism_s0: s0,
                thin_prism_s1: s1,
                thin_prism_s2: s2,
                thin_prism_s3: s3,
                ..
            } => {
                let r_xy = (rx * rx + ry * ry).sqrt();
                let theta = r_xy.atan2(rz);
                if r_xy < 1e-15 {
                    return Some((0.0, 0.0));
                }
                let (dx, dy) = (rx / r_xy, ry / r_xy);
                let uu = theta * dx;
                let vv = theta * dy;
                let (x_d, y_d) = distort_rad_tan_thin_prism_fisheye(
                    uu, vv, *k0, *k1, *k2, *k3, *k4, *k5, *p0, *p1, *s0, *s1, *s2, *s3,
                );
                Some((x_d, y_d))
            }
        }
    }

    /// Convert distorted normalized coordinates to a unit ray direction.
    ///
    /// For perspective models, equivalent to normalizing `(undistort(x_d, y_d), 1)`.
    /// For fisheye models, computes the ray directly from the incidence angle
    /// theta, avoiding the `tan(theta)` singularity that causes [`undistort`]
    /// to break down at and beyond 90° from the optical axis.
    ///
    /// The returned vector is unit-length and points in the direction the
    /// camera pixel is looking in camera space (z-forward).
    pub fn undistort_to_ray(&self, x_d: f64, y_d: f64) -> [f64; 3] {
        match self {
            // Equirectangular: x_d is longitude, y_d is latitude (negated v)
            CameraModel::Equirectangular { .. } => {
                let longitude = x_d;
                let latitude = -y_d;
                let cos_lat = latitude.cos();
                [
                    longitude.sin() * cos_lat,
                    latitude.sin(),
                    longitude.cos() * cos_lat,
                ]
            }

            // Perspective models: undistort then normalize (x, y, 1)
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::SimpleRadial { .. }
            | CameraModel::Radial { .. }
            | CameraModel::OpenCV { .. }
            | CameraModel::FullOpenCV { .. } => {
                let (x, y) = self.undistort(x_d, y_d);
                let len = (x * x + y * y + 1.0).sqrt();
                [x / len, y / len, 1.0 / len]
            }

            // Equidistant fisheye family: recover theta, build ray directly
            CameraModel::OpenCVFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                ..
            } => equidistant_fisheye_to_ray(x_d, y_d, *k1, *k2, *k3, *k4),

            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k,
                ..
            } => equidistant_fisheye_to_ray(x_d, y_d, *k, 0.0, 0.0, 0.0),

            CameraModel::RadialFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => equidistant_fisheye_to_ray(x_d, y_d, *k1, *k2, 0.0, 0.0),

            // Thin prism fisheye: recover equidistant coords, then build ray
            CameraModel::ThinPrismFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                thin_prism_sx1: sx1,
                thin_prism_sy1: sy1,
                ..
            } => {
                let r_d = (x_d * x_d + y_d * y_d).sqrt();
                if r_d < 1e-15 {
                    return [0.0, 0.0, 1.0];
                }
                let (uu, vv) = recover_equidistant_thin_prism(
                    x_d, y_d, *k1, *k2, *p1, *p2, *k3, *k4, *sx1, *sy1,
                );
                let recovered = equidistant_to_ray(uu, vv);
                let undistorted = equidistant_to_ray(x_d, y_d);
                blend_fisheye_ray(r_d, recovered, undistorted)
            }

            CameraModel::RadTanThinPrismFisheye {
                radial_distortion_k0: k0,
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                radial_distortion_k5: k5,
                tangential_distortion_p0: p0,
                tangential_distortion_p1: p1,
                thin_prism_s0: s0,
                thin_prism_s1: s1,
                thin_prism_s2: s2,
                thin_prism_s3: s3,
                ..
            } => {
                let r_d = (x_d * x_d + y_d * y_d).sqrt();
                if r_d < 1e-15 {
                    return [0.0, 0.0, 1.0];
                }
                let (uu, vv) = recover_equidistant_rad_tan_thin_prism(
                    x_d, y_d, *k0, *k1, *k2, *k3, *k4, *k5, *p0, *p1, *s0, *s1, *s2, *s3,
                );
                let recovered = equidistant_to_ray(uu, vv);
                let undistorted = equidistant_to_ray(x_d, y_d);
                blend_fisheye_ray(r_d, recovered, undistorted)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CameraIntrinsics: pixel-space convenience
// ---------------------------------------------------------------------------

impl CameraIntrinsics {
    /// Project undistorted image-plane point to pixel coordinates.
    ///
    /// Applies distortion then converts to pixel coordinates:
    /// `(x, y)` → distort → `(u, v)` where `u = fx * x_d + cx`.
    pub fn project(&self, x: f64, y: f64) -> (f64, f64) {
        let (x_d, y_d) = self.model.distort(x, y);
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        (fx * x_d + cx, fy * y_d + cy)
    }

    /// Unproject pixel coordinates to undistorted image-plane coordinates.
    ///
    /// Converts pixel to distorted image-plane, then removes distortion:
    /// `(u, v)` → `(x_d, y_d)` → undistort → `(x, y)`.
    ///
    /// The returned `(x, y)` can be used as a ray direction `(x, y, 1)`.
    pub fn unproject(&self, u: f64, v: f64) -> (f64, f64) {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        let x_d = (u - cx) / fx;
        let y_d = (v - cy) / fy;
        self.model.undistort(x_d, y_d)
    }

    /// Project a batch of undistorted image-plane points to pixel coordinates.
    pub fn project_batch(&self, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        points
            .par_iter()
            .map(|&[x, y]| {
                let (x_d, y_d) = self.model.distort(x, y);
                [fx * x_d + cx, fy * y_d + cy]
            })
            .collect()
    }

    /// Unproject a batch of pixel coordinates to undistorted image-plane coordinates.
    pub fn unproject_batch(&self, pixels: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        pixels
            .par_iter()
            .map(|&[u, v]| {
                let x_d = (u - cx) / fx;
                let y_d = (v - cy) / fy;
                let (x, y) = self.model.undistort(x_d, y_d);
                [x, y]
            })
            .collect()
    }

    /// Convert pixel coordinates to a unit ray direction in camera space.
    ///
    /// For perspective models, equivalent to normalizing `(unproject(u, v), 1)`.
    /// For fisheye models, computes the ray directly from the incidence angle,
    /// avoiding the `tan(theta)` singularity that causes [`unproject`] to break
    /// down at and beyond 90° from the optical axis. This makes it suitable for
    /// wide-angle fisheye lenses with field of view approaching or exceeding 180°.
    pub fn pixel_to_ray(&self, u: f64, v: f64) -> [f64; 3] {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        let x_d = (u - cx) / fx;
        let y_d = (v - cy) / fy;
        self.model.undistort_to_ray(x_d, y_d)
    }

    /// Project a unit ray direction in camera space to pixel coordinates.
    ///
    /// For perspective models, equivalent to `project(rx/rz, ry/rz)`, but
    /// for fisheye models computes the distorted coordinates directly from
    /// the incidence angle, avoiding the `tan(theta)` singularity. For
    /// equirectangular, maps via longitude/latitude. This is the true inverse
    /// of [`pixel_to_ray`].
    ///
    /// Returns `None` if the ray falls outside the model's valid domain.
    pub fn ray_to_pixel(&self, ray: [f64; 3]) -> Option<(f64, f64)> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        let (x_d, y_d) = self.model.distort_ray(ray)?;
        Some((fx * x_d + cx, fy * y_d + cy))
    }

    /// Batch version of [`ray_to_pixel`].
    pub fn ray_to_pixel_batch(&self, rays: &[[f64; 3]]) -> Vec<Option<[f64; 2]>> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        rays.par_iter()
            .map(|&ray| {
                let (x_d, y_d) = self.model.distort_ray(ray)?;
                Some([fx * x_d + cx, fy * y_d + cy])
            })
            .collect()
    }

    /// Convert a batch of pixel coordinates to unit ray directions.
    pub fn pixel_to_ray_batch(&self, pixels: &[[f64; 2]]) -> Vec<[f64; 3]> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        pixels
            .par_iter()
            .map(|&[u, v]| {
                let x_d = (u - cx) / fx;
                let y_d = (v - cy) / fy;
                self.model.undistort_to_ray(x_d, y_d)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Best-fit pinhole estimation
    // -----------------------------------------------------------------------

    /// Build a pinhole camera at the given resolution whose field of view is
    /// the largest that still maps every destination pixel to a valid location
    /// in this (source) camera.
    ///
    /// The resulting undistorted image will have no black borders — every pixel
    /// is backed by source data — but some peripheral source pixels may be
    /// cropped.
    ///
    /// The pinhole is centred at `(width/2, height/2)` with equal focal lengths
    /// `fx = fy`. The focal length is found via binary search.
    ///
    /// Returns [`CameraIntrinsicsError::UnsupportedModel`] if `self` is a
    /// fisheye or equirectangular model.
    pub fn best_fit_inside_pinhole(
        &self,
        width: u32,
        height: u32,
    ) -> Result<CameraIntrinsics, CameraIntrinsicsError> {
        if self.model.needs_ray_path() {
            return Err(CameraIntrinsicsError::UnsupportedModel(
                self.model.model_name().to_string(),
            ));
        }

        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let src_w = self.width as f64;
        let src_h = self.height as f64;

        let boundary = Self::boundary_samples(width, height);

        // Predicate: at this focal length, do ALL boundary points in the
        // pinhole frame map to valid source pixels?
        let all_inside = |focal: f64| -> bool {
            for &(u, v) in &boundary {
                let x = (u - cx) / focal;
                let y = (v - cy) / focal;
                let (sx, sy) = self.project(x, y);
                if sx < 0.0 || sy < 0.0 || sx >= src_w || sy >= src_h {
                    return false;
                }
            }
            true
        };

        // Search range: a very small focal length sees a wide FoV (likely
        // out of bounds), a very large focal length sees a narrow FoV
        // (likely all in bounds). We want the smallest focal length where
        // all_inside is true.
        let (fx, fy) = self.focal_lengths();
        let mut lo = 1.0_f64;
        let mut hi = fx.max(fy) * 4.0;

        // Ensure hi is actually valid (it should be for any reasonable camera).
        if !all_inside(hi) {
            hi *= 4.0;
        }

        for _ in 0..64 {
            let mid = (lo + hi) / 2.0;
            if all_inside(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        Ok(CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: hi,
                focal_length_y: hi,
                principal_point_x: cx,
                principal_point_y: cy,
            },
            width,
            height,
        })
    }

    /// Build a pinhole camera at the given resolution whose field of view is
    /// the smallest that still covers every pixel in this (source) camera.
    ///
    /// The resulting undistorted image will contain all source content — nothing
    /// is cropped — but may have black borders where no source data exists.
    ///
    /// The pinhole is centred at `(width/2, height/2)` with equal focal lengths
    /// `fx = fy`. The focal length is found via binary search.
    ///
    /// Returns [`CameraIntrinsicsError::UnsupportedModel`] if `self` is a
    /// fisheye or equirectangular model.
    pub fn best_fit_outside_pinhole(
        &self,
        width: u32,
        height: u32,
    ) -> Result<CameraIntrinsics, CameraIntrinsicsError> {
        if self.model.needs_ray_path() {
            return Err(CameraIntrinsicsError::UnsupportedModel(
                self.model.model_name().to_string(),
            ));
        }

        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let dst_w = width as f64;
        let dst_h = height as f64;

        let boundary = Self::boundary_samples(self.width, self.height);

        // Predicate: at this focal length, do ALL source boundary points
        // map to valid locations in the destination pinhole frame?
        let all_covered = |focal: f64| -> bool {
            for &(u, v) in &boundary {
                let (x, y) = self.unproject(u, v);
                let px = focal * x + cx;
                let py = focal * y + cy;
                if px < 0.0 || py < 0.0 || px >= dst_w || py >= dst_h {
                    return false;
                }
            }
            true
        };

        // Search range: a very large focal length maps source boundary
        // points outside the dst frame; a very small focal length pulls
        // them all in. We want the largest focal length where all_covered
        // is true.
        let (fx, fy) = self.focal_lengths();
        let mut lo = 1.0_f64;
        let mut hi = fx.max(fy) * 4.0;

        // Ensure lo is actually valid.
        if !all_covered(lo) {
            lo = 0.1;
        }

        for _ in 0..64 {
            let mid = (lo + hi) / 2.0;
            if all_covered(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Ok(CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: lo,
                focal_length_y: lo,
                principal_point_x: cx,
                principal_point_y: cy,
            },
            width,
            height,
        })
    }

    /// Sample 8 boundary points of an image: 4 corners + 4 edge midpoints.
    fn boundary_samples(width: u32, height: u32) -> Vec<(f64, f64)> {
        let w = width as f64;
        let h = height as f64;
        vec![
            (0.5, 0.5),         // top-left
            (w - 0.5, 0.5),     // top-right
            (0.5, h - 0.5),     // bottom-left
            (w - 0.5, h - 0.5), // bottom-right
            (w / 2.0, 0.5),     // top-center
            (w / 2.0, h - 0.5), // bottom-center
            (0.5, h / 2.0),     // left-center
            (w - 0.5, h / 2.0), // right-center
        ]
    }
}

#[cfg(test)]
mod tests;
