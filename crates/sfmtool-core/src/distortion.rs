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

use crate::camera_intrinsics::{CameraIntrinsics, CameraModel};

/// Maximum iterations for iterative undistortion.
const UNDISTORT_MAX_ITER: usize = 100;

/// Convergence threshold for iterative undistortion.
const UNDISTORT_EPS: f64 = 1e-10;

/// Fisheye distortion models are not coherent past ~90° from the optical axis,
/// so we blend from the distorted ray to the undistorted (identity) ray over
/// this angular range (in radians of the undistorted angle).
const FISHEYE_BLEND_START_RAD: f64 = 90.0 * (std::f64::consts::PI / 180.0); // 90°
const FISHEYE_BLEND_END_RAD: f64 = 100.0 * (std::f64::consts::PI / 180.0); // 100°

// ---------------------------------------------------------------------------
// CameraModel: normalized-space distortion
// ---------------------------------------------------------------------------

impl CameraModel {
    /// Apply forward distortion: undistorted image-plane → distorted image-plane.
    ///
    /// For pinhole models (no distortion), returns `(x, y)` unchanged.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            CameraModel::Pinhole { .. } | CameraModel::SimplePinhole { .. } => (x, y),

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

    /// Remove distortion: distorted image-plane → undistorted image-plane.
    ///
    /// Uses iterative fixed-point solving. For pinhole models, returns the
    /// input unchanged. For fisheye, uses Newton's method on the scalar
    /// theta mapping.
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        match self {
            CameraModel::Pinhole { .. } | CameraModel::SimplePinhole { .. } => (x_d, y_d),

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
}

// ---------------------------------------------------------------------------
// Distortion model implementations
// ---------------------------------------------------------------------------

/// OpenCV distortion: radial (k1, k2) + tangential (p1, p2).
fn distort_opencv(x: f64, y: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> (f64, f64) {
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    let x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (x_d, y_d)
}

/// Full OpenCV distortion: rational radial (k1..k6) + tangential (p1, p2).
#[allow(clippy::too_many_arguments)]
fn distort_full_opencv(
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
fn distort_fisheye(x: f64, y: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> (f64, f64) {
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
fn recover_theta_equidistant(r_d: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> f64 {
    /// Evaluate f'(θ) = d/dθ [θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)].
    #[inline]
    fn f_prime(theta: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> f64 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
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
    if f_prime(theta, k1, k2, k3, k4) <= 0.0 {
        let mut lo = 0.0_f64;
        let mut hi = theta;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if f_prime(mid, k1, k2, k3, k4) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        theta_max = lo;
        theta = lo;
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
            // Reached the peak — can't go further.
            break;
        }
        let delta = f / fp;
        theta -= delta;
        theta = theta.clamp(0.0, theta_max);
        if delta.abs() < UNDISTORT_EPS {
            break;
        }
    }
    theta
}

/// Inverse of OpenCV fisheye distortion.
///
/// Given distorted coordinates `(x_d, y_d)`, recovers the undistorted `(x, y)`.
/// Uses Newton's method on the scalar `theta_d → theta` mapping, then recovers
/// the 2D direction.
fn undistort_fisheye(x_d: f64, y_d: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let theta = recover_theta_equidistant(r_d, k1, k2, k3, k4);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

/// Simple radial fisheye distortion: equidistant + single radial k in theta space.
///
/// `theta_d = theta * (1 + k * theta²)` where `theta = atan(r)`.
fn distort_simple_radial_fisheye(x: f64, y: f64, k: f64) -> (f64, f64) {
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
fn undistort_simple_radial_fisheye(x_d: f64, y_d: f64, k: f64) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let theta = recover_theta_equidistant(r_d, k, 0.0, 0.0, 0.0);
    let r = theta.tan();
    let scale = r / r_d;
    (x_d * scale, y_d * scale)
}

/// Radial fisheye distortion: equidistant + two radial k1, k2 in theta space.
///
/// `theta_d = theta * (1 + k1 * theta² + k2 * theta⁴)` where `theta = atan(r)`.
fn distort_radial_fisheye(x: f64, y: f64, k1: f64, k2: f64) -> (f64, f64) {
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
fn undistort_radial_fisheye(x_d: f64, y_d: f64, k1: f64, k2: f64) -> (f64, f64) {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return (x_d, y_d);
    }
    let theta = recover_theta_equidistant(r_d, k1, k2, 0.0, 0.0);
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
fn distort_thin_prism_fisheye(
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
fn recover_equidistant_thin_prism(
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
fn newton_thin_prism(
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
fn undistort_thin_prism_fisheye(
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
fn distort_rad_tan_thin_prism_fisheye(
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
fn recover_equidistant_rad_tan_thin_prism(
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
fn newton_rad_tan_thin_prism(
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
fn undistort_rad_tan_thin_prism_fisheye(
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
fn equidistant_fisheye_to_ray(x_d: f64, y_d: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> [f64; 3] {
    let r_d = (x_d * x_d + y_d * y_d).sqrt();
    if r_d < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    let theta = recover_theta_equidistant(r_d, k1, k2, k3, k4);
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let s = sin_theta / r_d;
    let recovered = [x_d * s, y_d * s, cos_theta];
    // For the simple equidistant model the XY direction is the same for
    // recovered and undistorted, so we can pass the recovered ray as both.
    blend_fisheye_ray(r_d, recovered, recovered)
}

/// Convert undistorted equidistant coordinates `(uu, vv)` to a unit ray direction.
///
/// `theta = sqrt(uu² + vv²)` is the incidence angle.
/// Returns `[sin(theta) * uu/theta, sin(theta) * vv/theta, cos(theta)]`.
fn equidistant_to_ray(uu: f64, vv: f64) -> [f64; 3] {
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
fn blend_fisheye_ray(r_d: f64, recovered: [f64; 3], undistorted: [f64; 3]) -> [f64; 3] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // Test camera constructors (reused from camera_intrinsics tests)
    // -----------------------------------------------------------------------

    fn pinhole() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn simple_pinhole() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimplePinhole {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn simple_radial() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimpleRadial {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
            },
            width: 640,
            height: 480,
        }
    }

    fn radial() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::Radial {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
            },
            width: 640,
            height: 480,
        }
    }

    fn opencv() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::OpenCV {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
                tangential_distortion_p1: 0.001,
                tangential_distortion_p2: -0.002,
            },
            width: 640,
            height: 480,
        }
    }

    fn opencv_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
                radial_distortion_k3: 0.01,
                radial_distortion_k4: -0.005,
            },
            width: 640,
            height: 480,
        }
    }

    fn full_opencv() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::FullOpenCV {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.1,
                radial_distortion_k2: -0.05,
                tangential_distortion_p1: 0.001,
                tangential_distortion_p2: -0.002,
                radial_distortion_k3: 0.01,
                radial_distortion_k4: -0.005,
                radial_distortion_k5: 0.002,
                radial_distortion_k6: -0.001,
            },
            width: 640,
            height: 480,
        }
    }

    fn simple_radial_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimpleRadialFisheye {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
            },
            width: 640,
            height: 480,
        }
    }

    fn radial_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::RadialFisheye {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
                radial_distortion_k2: -0.02,
            },
            width: 640,
            height: 480,
        }
    }

    fn thin_prism_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::ThinPrismFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.05,
                radial_distortion_k2: -0.01,
                tangential_distortion_p1: 0.001,
                tangential_distortion_p2: -0.001,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
                thin_prism_sx1: 0.002,
                thin_prism_sy1: -0.001,
            },
            width: 640,
            height: 480,
        }
    }

    fn rad_tan_thin_prism_fisheye() -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::RadTanThinPrismFisheye {
                focal_length_x: 500.0,
                focal_length_y: 502.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k0: 0.03,
                radial_distortion_k1: -0.01,
                radial_distortion_k2: 0.005,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
                radial_distortion_k5: 0.0,
                tangential_distortion_p0: 0.001,
                tangential_distortion_p1: -0.001,
                thin_prism_s0: 0.001,
                thin_prism_s1: 0.0,
                thin_prism_s2: -0.001,
                thin_prism_s3: 0.0,
            },
            width: 640,
            height: 480,
        }
    }

    fn all_cameras() -> Vec<CameraIntrinsics> {
        vec![
            pinhole(),
            simple_pinhole(),
            simple_radial(),
            radial(),
            opencv(),
            opencv_fisheye(),
            simple_radial_fisheye(),
            radial_fisheye(),
            thin_prism_fisheye(),
            rad_tan_thin_prism_fisheye(),
            full_opencv(),
        ]
    }

    // -----------------------------------------------------------------------
    // Pinhole: distort/undistort are identity
    // -----------------------------------------------------------------------

    #[test]
    fn pinhole_distort_is_identity() {
        for cam in [pinhole(), simple_pinhole()] {
            let (xd, yd) = cam.model.distort(0.3, -0.4);
            assert_relative_eq!(xd, 0.3, epsilon = 1e-15);
            assert_relative_eq!(yd, -0.4, epsilon = 1e-15);
        }
    }

    #[test]
    fn pinhole_undistort_is_identity() {
        for cam in [pinhole(), simple_pinhole()] {
            let (x, y) = cam.model.undistort(0.3, -0.4);
            assert_relative_eq!(x, 0.3, epsilon = 1e-15);
            assert_relative_eq!(y, -0.4, epsilon = 1e-15);
        }
    }

    // -----------------------------------------------------------------------
    // Origin: all models should be identity at (0, 0)
    // -----------------------------------------------------------------------

    #[test]
    fn distort_at_origin_is_identity() {
        for cam in all_cameras() {
            let (xd, yd) = cam.model.distort(0.0, 0.0);
            assert_relative_eq!(xd, 0.0, epsilon = 1e-15);
            assert_relative_eq!(yd, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn undistort_at_origin_is_identity() {
        for cam in all_cameras() {
            let (x, y) = cam.model.undistort(0.0, 0.0);
            assert_relative_eq!(x, 0.0, epsilon = 1e-15);
            assert_relative_eq!(y, 0.0, epsilon = 1e-15);
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip: undistort(distort(x, y)) ≈ (x, y) for all models
    // -----------------------------------------------------------------------

    /// Test points spanning a range of distances from the optical axis.
    fn test_points() -> Vec<[f64; 2]> {
        vec![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [-0.2, 0.15],
            [0.3, -0.2],
            [-0.1, -0.3],
            [0.5, 0.5],
            [-0.4, 0.3],
            [0.05, -0.05],
        ]
    }

    #[test]
    fn round_trip_distort_then_undistort() {
        for cam in all_cameras() {
            for &[x, y] in &test_points() {
                let (xd, yd) = cam.model.distort(x, y);
                let (x_rt, y_rt) = cam.model.undistort(xd, yd);
                assert_relative_eq!(x_rt, x, epsilon = 1e-8,);
                assert_relative_eq!(y_rt, y, epsilon = 1e-8,);
            }
        }
    }

    #[test]
    fn round_trip_undistort_then_distort() {
        for cam in all_cameras() {
            for &[xd, yd] in &test_points() {
                let (x, y) = cam.model.undistort(xd, yd);
                let (xd_rt, yd_rt) = cam.model.distort(x, y);
                assert_relative_eq!(xd_rt, xd, epsilon = 1e-8);
                assert_relative_eq!(yd_rt, yd, epsilon = 1e-8);
            }
        }
    }

    // -----------------------------------------------------------------------
    // SimpleRadial: verify distort formula directly
    // -----------------------------------------------------------------------

    #[test]
    fn simple_radial_distort_formula() {
        let cam = simple_radial();
        let (x, y) = (0.3, 0.4);
        let r2 = x * x + y * y; // 0.25
        let k1 = 0.1;
        let expected_scale = 1.0 + k1 * r2; // 1.025
        let (xd, yd) = cam.model.distort(x, y);
        assert_relative_eq!(xd, x * expected_scale, epsilon = 1e-15);
        assert_relative_eq!(yd, y * expected_scale, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // Radial: verify distort formula directly
    // -----------------------------------------------------------------------

    #[test]
    fn radial_distort_formula() {
        let cam = radial();
        let (x, y) = (0.3, 0.4);
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let (k1, k2) = (0.1, -0.05);
        let expected_scale = 1.0 + k1 * r2 + k2 * r4;
        let (xd, yd) = cam.model.distort(x, y);
        assert_relative_eq!(xd, x * expected_scale, epsilon = 1e-15);
        assert_relative_eq!(yd, y * expected_scale, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // Distortion changes coordinates (non-zero distortion should differ)
    // -----------------------------------------------------------------------

    #[test]
    fn distortion_is_not_identity_for_distorted_models() {
        let point = (0.3, 0.4);
        for cam in [
            simple_radial(),
            radial(),
            opencv(),
            opencv_fisheye(),
            simple_radial_fisheye(),
            radial_fisheye(),
            thin_prism_fisheye(),
            rad_tan_thin_prism_fisheye(),
            full_opencv(),
        ] {
            let (xd, yd) = cam.model.distort(point.0, point.1);
            let differs = (xd - point.0).abs() > 1e-10 || (yd - point.1).abs() > 1e-10;
            assert!(
                differs,
                "{} distort should modify off-center points",
                cam.model_name()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Pixel-space project/unproject round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn project_unproject_round_trip() {
        for cam in all_cameras() {
            for &[x, y] in &test_points() {
                let (u, v) = cam.project(x, y);
                let (x_rt, y_rt) = cam.unproject(u, v);
                assert_relative_eq!(x_rt, x, epsilon = 1e-8);
                assert_relative_eq!(y_rt, y, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn project_pinhole_matches_intrinsic_matrix() {
        let cam = pinhole();
        let (x, y) = (0.3, -0.2);
        let (u, v) = cam.project(x, y);
        // For pinhole: u = fx * x + cx, v = fy * y + cy
        assert_relative_eq!(u, 500.0 * 0.3 + 320.0, epsilon = 1e-12);
        assert_relative_eq!(v, 502.0 * -0.2 + 240.0, epsilon = 1e-12);
    }

    #[test]
    fn unproject_pinhole_at_principal_point() {
        let cam = pinhole();
        let (x, y) = cam.unproject(320.0, 240.0);
        assert_relative_eq!(x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(y, 0.0, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // Batch variants
    // -----------------------------------------------------------------------

    #[test]
    fn distort_batch_matches_single() {
        for cam in all_cameras() {
            let pts = test_points();
            let batch_result = cam.model.distort_batch(&pts);
            for (i, &[x, y]) in pts.iter().enumerate() {
                let (xd, yd) = cam.model.distort(x, y);
                assert_relative_eq!(batch_result[i][0], xd, epsilon = 1e-15);
                assert_relative_eq!(batch_result[i][1], yd, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn undistort_batch_matches_single() {
        for cam in all_cameras() {
            let pts = test_points();
            let batch_result = cam.model.undistort_batch(&pts);
            for (i, &[x_d, y_d]) in pts.iter().enumerate() {
                let (x, y) = cam.model.undistort(x_d, y_d);
                assert_relative_eq!(batch_result[i][0], x, epsilon = 1e-15);
                assert_relative_eq!(batch_result[i][1], y, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn project_batch_matches_single() {
        for cam in all_cameras() {
            let pts = test_points();
            let batch_result = cam.project_batch(&pts);
            for (i, &[x, y]) in pts.iter().enumerate() {
                let (u, v) = cam.project(x, y);
                assert_relative_eq!(batch_result[i][0], u, epsilon = 1e-15);
                assert_relative_eq!(batch_result[i][1], v, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn unproject_batch_matches_single() {
        for cam in all_cameras() {
            let pixels: Vec<[f64; 2]> = test_points()
                .iter()
                .map(|&[x, y]| {
                    let (u, v) = cam.project(x, y);
                    [u, v]
                })
                .collect();
            let batch_result = cam.unproject_batch(&pixels);
            for (i, &[u, v]) in pixels.iter().enumerate() {
                let (x, y) = cam.unproject(u, v);
                assert_relative_eq!(batch_result[i][0], x, epsilon = 1e-15);
                assert_relative_eq!(batch_result[i][1], y, epsilon = 1e-15);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fisheye: specific behavior tests
    // -----------------------------------------------------------------------

    #[test]
    fn fisheye_distort_at_origin() {
        let cam = opencv_fisheye();
        let (xd, yd) = cam.model.distort(0.0, 0.0);
        assert_relative_eq!(xd, 0.0, epsilon = 1e-15);
        assert_relative_eq!(yd, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn fisheye_round_trip_wide_angle() {
        // Test at wider angles where fisheye diverges most from pinhole
        let cam = opencv_fisheye();
        for &[x, y] in &[[0.8, 0.0], [0.0, 0.8], [0.6, 0.6], [-0.7, 0.5]] {
            let (xd, yd) = cam.model.distort(x, y);
            let (x_rt, y_rt) = cam.model.undistort(xd, yd);
            assert_relative_eq!(x_rt, x, epsilon = 1e-8);
            assert_relative_eq!(y_rt, y, epsilon = 1e-8);
        }
    }

    // -----------------------------------------------------------------------
    // undistort_to_ray tests
    // -----------------------------------------------------------------------

    #[test]
    fn undistort_to_ray_at_origin_is_optical_axis() {
        for cam in all_cameras() {
            let ray = cam.model.undistort_to_ray(0.0, 0.0);
            assert_relative_eq!(ray[0], 0.0, epsilon = 1e-15);
            assert_relative_eq!(ray[1], 0.0, epsilon = 1e-15);
            assert_relative_eq!(ray[2], 1.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn undistort_to_ray_produces_unit_vectors() {
        for cam in all_cameras() {
            for &[x_d, y_d] in &test_points() {
                let ray = cam.model.undistort_to_ray(x_d, y_d);
                let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
                assert_relative_eq!(len, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn undistort_to_ray_agrees_with_undistort_for_perspective() {
        // For perspective models, undistort_to_ray should give the same
        // direction as normalize(undistort(x_d, y_d), 1)
        for cam in [
            pinhole(),
            simple_pinhole(),
            simple_radial(),
            radial(),
            opencv(),
            full_opencv(),
        ] {
            for &[x_d, y_d] in &test_points() {
                let ray = cam.model.undistort_to_ray(x_d, y_d);
                let (x, y) = cam.model.undistort(x_d, y_d);
                let len = (x * x + y * y + 1.0).sqrt();
                assert_relative_eq!(ray[0], x / len, epsilon = 1e-10);
                assert_relative_eq!(ray[1], y / len, epsilon = 1e-10);
                assert_relative_eq!(ray[2], 1.0 / len, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn undistort_to_ray_agrees_with_undistort_for_small_angles() {
        // For fisheye models at small angles, undistort_to_ray should agree
        // with normalize(undistort(x_d, y_d), 1) since tan(theta) ≈ theta
        let small_points = [[0.01, 0.0], [0.0, 0.01], [0.01, 0.01], [-0.02, 0.015]];
        let fisheye_cameras = vec![
            opencv_fisheye(),
            simple_radial_fisheye(),
            radial_fisheye(),
            thin_prism_fisheye(),
            rad_tan_thin_prism_fisheye(),
        ];
        for cam in fisheye_cameras {
            for &[x_d, y_d] in &small_points {
                let ray = cam.model.undistort_to_ray(x_d, y_d);
                let (x, y) = cam.model.undistort(x_d, y_d);
                let len = (x * x + y * y + 1.0).sqrt();
                assert_relative_eq!(ray[0], x / len, epsilon = 1e-6);
                assert_relative_eq!(ray[1], y / len, epsilon = 1e-6);
                assert_relative_eq!(ray[2], 1.0 / len, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn undistort_to_ray_fisheye_beyond_90_degrees() {
        // For a pure equidistant fisheye (no distortion coefficients),
        // a distorted radius of π/2 corresponds to theta = 90°,
        // and beyond that the ray should point backward (z < 0).
        let cam = CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 500.0,
                focal_length_y: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.0,
                radial_distortion_k2: 0.0,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
            },
            width: 640,
            height: 480,
        };

        // At exactly 90°: theta = π/2, r_d = π/2 in normalized coords
        let r_d_90 = std::f64::consts::FRAC_PI_2;
        let ray = cam.model.undistort_to_ray(r_d_90, 0.0);
        assert_relative_eq!(ray[2], 0.0, epsilon = 1e-10); // z ≈ 0 at 90°
        assert!(ray[0] > 0.0); // pointing rightward

        // Beyond 90°: theta > π/2, z should be negative
        let r_d_120 = std::f64::consts::FRAC_PI_3 * 2.0; // 120° = 2π/3
        let ray = cam.model.undistort_to_ray(r_d_120, 0.0);
        assert!(
            ray[2] < 0.0,
            "Ray beyond 90° should have negative z, got {}",
            ray[2]
        );
        assert!(ray[0] > 0.0, "Ray should still point rightward");
        let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
        assert_relative_eq!(len, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn thin_prism_fisheye_undistort_to_ray_wide_angle() {
        // Verify undistort_to_ray for thin prism fisheye with nonzero distortion.
        // Exact round-trip is tested below 80° (before the blend to undistorted
        // kicks in). Above 80°, we just verify unit-length and no NaNs.
        let (k1, k2, p1, p2, k3, k4, sx1, sy1) =
            (0.01, -0.0001, 0.001, -0.001, 0.0, 0.0, 0.002, -0.001);
        let cam = CameraModel::ThinPrismFisheye {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 0.0,
            principal_point_y: 0.0,
            radial_distortion_k1: k1,
            radial_distortion_k2: k2,
            tangential_distortion_p1: p1,
            tangential_distortion_p2: p2,
            radial_distortion_k3: k3,
            radial_distortion_k4: k4,
            thin_prism_sx1: sx1,
            thin_prism_sy1: sy1,
        };

        for deg in (0..=360).step_by(5) {
            let theta = (deg as f64).to_radians();
            let uu = theta * 0.8_f64.cos();
            let vv = theta * 0.8_f64.sin();

            // Forward distort in equidistant space
            let theta2 = uu * uu + vv * vv;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;
            let radial_val = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
            let duu =
                uu * radial_val + 2.0 * p1 * uu * vv + p2 * (theta2 + 2.0 * uu * uu) + sx1 * theta2;
            let dvv =
                vv * radial_val + 2.0 * p2 * uu * vv + p1 * (theta2 + 2.0 * vv * vv) + sy1 * theta2;
            let x_d = uu + duu;
            let y_d = vv + dvv;

            let ray = cam.undistort_to_ray(x_d, y_d);
            let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();

            assert!(
                !ray[0].is_nan() && !ray[1].is_nan() && !ray[2].is_nan(),
                "ThinPrism: NaN at {deg}°"
            );
            assert_relative_eq!(len, 1.0, epsilon = 1e-6);

            // Exact round-trip only below the blend range (80°)
            if deg < 90 {
                let expected = equidistant_to_ray(uu, vv);
                let err = ((ray[0] - expected[0]).powi(2)
                    + (ray[1] - expected[1]).powi(2)
                    + (ray[2] - expected[2]).powi(2))
                .sqrt();
                assert!(
                    err < 1e-6,
                    "ThinPrism: ray error {err:.2e} at {deg}° (ray={ray:?}, expected={expected:?})"
                );
            }
        }
    }

    #[test]
    fn rad_tan_thin_prism_fisheye_undistort_to_ray_wide_angle() {
        // Same test for RadTanThinPrismFisheye with small distortion.
        // Exact round-trip below 80°; unit-length and no NaNs everywhere.
        let (k0, k1, k2, k3, k4, k5) = (0.01, -0.0001, 0.0, 0.0, 0.0, 0.0);
        let (p0, p1) = (0.001, -0.001);
        let (s0, s1, s2, s3) = (0.001, 0.0, -0.001, 0.0);
        let cam = CameraModel::RadTanThinPrismFisheye {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 0.0,
            principal_point_y: 0.0,
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
        };

        for deg in (0..=360).step_by(5) {
            let theta = (deg as f64).to_radians();
            let uu = theta * 0.8_f64.cos();
            let vv = theta * 0.8_f64.sin();

            // Forward distort: radial scaling then tangential+thin prism
            let th2 = uu * uu + vv * vv;
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
            let x_d = uu_r + duu;
            let y_d = vv_r + dvv;

            let ray = cam.undistort_to_ray(x_d, y_d);
            let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();

            assert!(
                !ray[0].is_nan() && !ray[1].is_nan() && !ray[2].is_nan(),
                "RadTanThinPrism: NaN at {deg}°"
            );
            assert_relative_eq!(len, 1.0, epsilon = 1e-6);

            if deg < 90 {
                let expected = equidistant_to_ray(uu, vv);
                let err = ((ray[0] - expected[0]).powi(2)
                    + (ray[1] - expected[1]).powi(2)
                    + (ray[2] - expected[2]).powi(2))
                .sqrt();
                assert!(
                    err < 1e-6,
                    "RadTanThinPrism: ray error {err:.2e} at {deg}° (ray={ray:?}, expected={expected:?})"
                );
            }
        }
    }

    #[test]
    fn recover_theta_equidistant_out_of_range() {
        // Fisheye camera with distortion coefficients from a real 360 camera.
        // The distortion function f(theta) peaks at ~106° and then decreases,
        // so r_d values beyond ~1.878 have no valid inverse. Previously Newton's
        // method would diverge, producing garbage theta values (e.g. 2800°).
        let k1 = 0.04338287031606894;
        let k2 = -0.010311408690860134;
        let k3 = 0.00890875030327529;
        let k4 = -0.0026965936602161068;

        // In-range: should converge to a valid theta
        let theta = recover_theta_equidistant(1.5, k1, k2, k3, k4);
        assert!(theta > 0.0 && theta < std::f64::consts::PI, "theta={theta}");

        // Out-of-range (corner pixel): should NOT produce garbage
        let theta = recover_theta_equidistant(2.636, k1, k2, k3, k4);
        assert!(
            theta > 0.0 && theta <= std::f64::consts::PI,
            "Out-of-range r_d should produce bounded theta, got {theta} ({} degrees)",
            theta.to_degrees()
        );

        // The ray from an out-of-range theta must still be a valid unit vector
        let cam = CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 1033.0,
                focal_length_y: 1027.0,
                principal_point_x: 1920.0,
                principal_point_y: 1920.0,
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
            },
            width: 3840,
            height: 3840,
        };
        // Corner pixel — beyond valid distortion range
        let ray = cam.pixel_to_ray(3840.0, 3840.0);
        let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Ray should be approximately unit length, got {len}"
        );
        assert!(
            ray[2] > -1.1,
            "Ray z component should be reasonable, got {}",
            ray[2]
        );
    }

    // -----------------------------------------------------------------------
    // pixel_to_ray tests
    // -----------------------------------------------------------------------

    #[test]
    fn pixel_to_ray_at_principal_point() {
        for cam in all_cameras() {
            let (cx, cy) = cam.principal_point();
            let ray = cam.pixel_to_ray(cx, cy);
            assert_relative_eq!(ray[0], 0.0, epsilon = 1e-15);
            assert_relative_eq!(ray[1], 0.0, epsilon = 1e-15);
            assert_relative_eq!(ray[2], 1.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn pixel_to_ray_produces_unit_vectors() {
        for cam in all_cameras() {
            let pixels = [[0.0, 0.0], [320.0, 240.0], [639.0, 479.0], [100.0, 200.0]];
            for &[u, v] in &pixels {
                let ray = cam.pixel_to_ray(u, v);
                let len = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
                assert_relative_eq!(len, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn pixel_to_ray_batch_matches_single() {
        for cam in all_cameras() {
            let pixels = [[0.0, 0.0], [320.0, 240.0], [639.0, 479.0], [100.0, 200.0]];
            let batch = cam.pixel_to_ray_batch(&pixels);
            for (i, &[u, v]) in pixels.iter().enumerate() {
                let ray = cam.pixel_to_ray(u, v);
                assert_relative_eq!(batch[i][0], ray[0], epsilon = 1e-15);
                assert_relative_eq!(batch[i][1], ray[1], epsilon = 1e-15);
                assert_relative_eq!(batch[i][2], ray[2], epsilon = 1e-15);
            }
        }
    }
}
