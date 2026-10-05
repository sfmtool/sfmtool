// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Lens distortion and undistortion for COLMAP camera models.
//!
//! Provides forward distortion (undistorted → distorted normalized coordinates)
//! and iterative undistortion (distorted → undistorted) for all supported camera
//! models. Convenience wrappers on [`CameraIntrinsics`], in the private
//! `projection` submodule, handle the full pixel ↔ normalized coordinate
//! conversion.
//!
//! [`CameraIntrinsics`]: crate::camera::CameraIntrinsics
//!
//! # Coordinate systems
//!
//! **Camera space** follows the canonical `.sfmr` convention (see
//! `specs/formats/sfmr-file-format.md` § "Coordinate System Conventions"):
//! the camera looks down **−Z**, with **+X right** and **+Y up** in the image
//! plane (OpenGL-style). A point is in front of the camera iff its
//! camera-space `z < 0`, and its depth is `−z`.
//!
//! **Image-plane coordinates** `(x, y)` are obtained by projecting a
//! camera-space 3D point onto the image plane: `(x, y) = (X/(−Z), Y/(−Z))`,
//! so `+y` points up. The origin `(0, 0)` is the optical axis (principal
//! ray). Values are unbounded and represent the tangent of the angle from
//! the optical axis — a point at 45° off-axis has `|x|` or `|y|` of 1.0.
//! These are **not** normalized device coordinates (NDC).
//!
//! **Pixel coordinates** `(u, v)` have the origin at the top-left of the image,
//! with `u` increasing rightward and `v` increasing **downward**. The principal
//! point `(cx, cy)` maps to image-plane `(0, 0)`.
//!
//! ## Projection pipeline and the optical-frame boundary
//!
//! The distortion kernels are unchanged COLMAP/OpenCV math and operate in the
//! legacy **optical frame** (+Z forward, y down). Rather than rewriting them,
//! the flip `S = diag(1, −1, −1)` is applied exactly once at the camera-model
//! boundary (see `specs/formats/sfmr-file-format.md` § "Coordinate System
//! Conventions" → "Pixel space"):
//!
//! ```text
//! camera-space point p (z < 0 in front)
//!   → image-plane (x = p.x/(−p.z), y = p.y/(−p.z))     # y up
//!   → distort(x, −y) = (x_d, y_d)                       # kernels are y-down
//!   → pixel (u = fx·x_d + cx, v = fy·y_d + cy)
//!
//! pixel → distorted image-plane (x_d = (u−cx)/fx, y_d = (v−cy)/fy)
//!       → undistort → y-down (x, y_k) → y-up (x, −y_k)
//!       → ray direction (x, −y_k, −1)                   # canonical, −Z forward
//! ```
//!
//! The `distort` and `undistort` methods on [`CameraModel`] are the kernel
//! level: they operate in **y-down** (optical-frame) image-plane coordinates,
//! matching pixel rows. The `project` / `unproject` / `pixel_to_ray` /
//! `ray_to_pixel` methods on [`CameraIntrinsics`] (and `distort_ray` /
//! `undistort_to_ray` on [`CameraModel`]) speak the canonical y-up /
//! −Z-forward convention and perform the `S` flip internally.

use rayon::prelude::*;

use crate::camera::CameraModel;

/// A projected pixel `(u, v)` paired with the 2×3 Jacobian `∂(u, v)/∂ray`
/// (row-major `[[∂u/∂x, ∂u/∂y, ∂u/∂z], [∂v/∂x, ∂v/∂y, ∂v/∂z]]`), returned by
/// [`CameraIntrinsics::ray_to_pixel_with_jacobian`](crate::camera::CameraIntrinsics::ray_to_pixel_with_jacobian).
pub type PixelJacobian = ((f64, f64), [[f64; 3]; 2]);

/// Maximum iterations for iterative undistortion.
const UNDISTORT_MAX_ITER: usize = 100;

/// Convergence threshold for iterative undistortion.
const UNDISTORT_EPS: f64 = 1e-10;

/// Where the multi-coefficient fisheye polynomials stop being inverted, as a
/// **distorted** radius `r_d` in normalized image-plane units.
///
/// A high-order distortion polynomial becomes unreliable as it approaches its
/// peak, so past this radius `blend_fisheye_ray` blends the Newton-recovered
/// ray toward the identity (`θ = r_d`) ray, and past
/// [`FISHEYE_BLEND_END_RAD`] it hands back the identity ray outright. Above
/// this radius the model's forward and inverse maps are therefore no longer
/// each other's inverse, and neither describes the lens.
///
/// The quantity is `r_d`, **not** the incidence angle `θ`. For the equidistant
/// family the two carry the same units — radians — and coincide only for a
/// zero-coefficient model; a lens whose polynomial magnifies its rim crosses
/// this radius some way inside 90° off-axis, and
/// [`crate::camera::report::trustworthy_max_theta_deg`] is what converts one
/// to the other for a given camera.
pub const FISHEYE_BLEND_START_RAD: f64 = 90.0 * (std::f64::consts::PI / 180.0); // 90°
/// Where the fisheye blend that starts at [`FISHEYE_BLEND_START_RAD`] finishes,
/// in the same units: past this distorted radius the recovered ray is dropped
/// entirely in favour of the identity ray.
pub const FISHEYE_BLEND_END_RAD: f64 = 100.0 * (std::f64::consts::PI / 180.0); // 100°

mod kernels;
mod pinhole_fit;
mod projection;
// The SFMTOOL_FISHEYE radial spline: crate-visible because the bundle
// adjustment linearizes through the same basis evaluation the kernels use.
pub(crate) mod bspline;
mod ray_grid;
use kernels::*;

// Named directly by the sibling test module through its `use super::*`;
// production reads them inside `ray_grid`, so this is test-gated to stay
// warning-clean in release (mirrors `keypoint_subpixel`).
#[cfg(test)]
use ray_grid::{COARSE_GRID_STRIDE, COARSE_GRID_TOL_PX};

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

            CameraModel::EquidistantFisheye { .. } => distort_equidistant(x, y),

            CameraModel::SfmtoolFisheye {
                bspline,
                bspline_theta_max,
                ..
            } => distort_sfmtool_fisheye(x, y, bspline, *bspline_theta_max),

            CameraModel::SfmtoolPinhole {
                bspline,
                bspline_rho_max,
                ..
            } => distort_sfmtool_pinhole(x, y, bspline, *bspline_rho_max),

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

    /// Analytic Jacobian `∂(x_d, y_d)/∂(x, y)` of [`Self::distort`] at the normalized
    /// image-plane point `(x, y)`, row-major `[[∂x_d/∂x, ∂x_d/∂y], [∂y_d/∂x,
    /// ∂y_d/∂y]]`.
    ///
    /// Perspective-model family only; returns `None` for fisheye and
    /// equirectangular models, whose forward map does not go through
    /// [`Self::distort`] (see [`Self::distort_ray`]) and has no analytic pixel Jacobian yet.
    ///
    /// Every perspective model is `x_d = x·g(r²) + T_x`, `y_d = y·g(r²) + T_y`
    /// with radial factor `g`, `r² = x² + y²`, and tangential
    /// `T_x = 2 p1 x y + p2 (r² + 2x²)`, `T_y = p1 (r² + 2y²) + 2 p2 x y`. The
    /// 2×2 follows from `g`, `g' = dg/d(r²)`, and `(p1, p2)`.
    ///
    /// [`CameraModel::SfmtoolPinhole`] joins the family through that same
    /// form: its radial spline is `g(ρ) = 1 + δ(ρ)/ρ` with
    /// `dg/d(r²) = (ρ·δ'(ρ) − δ(ρ))/(2ρ³)` (`ρ = √(r²)`), computed in
    /// `sfmtool_pinhole_radial_factor`, so the composition, the tangential
    /// slots and the on-axis limit are all shared rather than re-derived.
    pub(crate) fn distort_jacobian(&self, x: f64, y: f64) -> Option<[[f64; 2]; 2]> {
        let s = x * x + y * y;
        let (g, gp, p1, p2) = match self {
            CameraModel::Pinhole { .. } | CameraModel::SimplePinhole { .. } => (1.0, 0.0, 0.0, 0.0),
            CameraModel::SimpleRadial {
                radial_distortion_k1: k1,
                ..
            } => (1.0 + k1 * s, *k1, 0.0, 0.0),
            CameraModel::Radial {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                ..
            } => (1.0 + k1 * s + k2 * s * s, k1 + 2.0 * k2 * s, 0.0, 0.0),
            CameraModel::OpenCV {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                tangential_distortion_p1: p1,
                tangential_distortion_p2: p2,
                ..
            } => (1.0 + k1 * s + k2 * s * s, k1 + 2.0 * k2 * s, *p1, *p2),
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
            } => {
                // Rational radial g = N/D, so g' = (N'·D − N·D')/D².
                let num = 1.0 + k1 * s + k2 * s * s + k3 * s * s * s;
                let den = 1.0 + k4 * s + k5 * s * s + k6 * s * s * s;
                let nump = k1 + 2.0 * k2 * s + 3.0 * k3 * s * s;
                let denp = k4 + 2.0 * k5 * s + 3.0 * k6 * s * s;
                let g = num / den;
                let gp = (nump * den - num * denp) / (den * den);
                (g, gp, *p1, *p2)
            }
            CameraModel::SfmtoolPinhole {
                bspline,
                bspline_rho_max,
                ..
            } => {
                let (g, gp) = sfmtool_pinhole_radial_factor(s, bspline, *bspline_rho_max);
                (g, gp, 0.0, 0.0)
            }
            // Fisheye / equirectangular: no analytic pixel Jacobian yet.
            _ => return None,
        };
        // Cross term is shared by both off-diagonals (radial `2xy g'` plus the
        // tangential `2 p1 x + 2 p2 y`).
        let cross = 2.0 * x * y * gp + 2.0 * p1 * x + 2.0 * p2 * y;
        let dxdx = g + 2.0 * x * x * gp + 2.0 * p1 * y + 6.0 * p2 * x;
        let dydy = g + 2.0 * y * y * gp + 6.0 * p1 * y + 2.0 * p2 * x;
        Some([[dxdx, cross], [cross, dydy]])
    }

    /// Whether the normalized image-plane point `(x, y)` lies in the
    /// distortion polynomial's principal monotonic branch — the branch
    /// connected to the origin via positive radial growth.
    ///
    /// Beyond the first inflection of the polynomial, the forward map
    /// stops being injective: the same distorted pixel can be reached from
    /// multiple ray directions, producing ghost / mirror projections
    /// outside the camera's true FOV. [`Self::distort_ray`] uses this to gate
    /// rays before calling [`Self::distort`].
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
    /// [`Self::distort`]; for fisheye and equirectangular models — which take
    /// different code paths in [`Self::distort_ray`] — this returns `true`.
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
            // Radial spline: the fold gate of `sfmtool_pinhole_unfolded`,
            // `ρ + δ(ρ) > 0`. Both the projection and the analytic Jacobian
            // reach the model through this predicate, so they leave the domain
            // together by construction.
            CameraModel::SfmtoolPinhole {
                bspline,
                bspline_rho_max,
                ..
            } => sfmtool_pinhole_unfolded(x * x + y * y, bspline, *bspline_rho_max),
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

            CameraModel::EquidistantFisheye { .. } => undistort_equidistant(x_d, y_d),

            // Explicit arm: the spline's θ-space Newton inverse. The generic
            // fixed-point fallback below is a perspective-model iteration and
            // would silently mishandle a θ-map model.
            CameraModel::SfmtoolFisheye {
                bspline,
                bspline_theta_max,
                ..
            } => undistort_sfmtool_fisheye(x_d, y_d, bspline, *bspline_theta_max),

            // Explicit arm for the same reason: the generic fixed-point
            // fallback below contracts only for weak distortion, while the
            // spline's monotonicity invariant gives this model an exact
            // bracketed Newton inverse at any coefficient magnitude.
            CameraModel::SfmtoolPinhole {
                bspline,
                bspline_rho_max,
                ..
            } => undistort_sfmtool_pinhole(x_d, y_d, bspline, *bspline_rho_max),

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

    /// Project a ray direction in **canonical camera space** (−Z forward,
    /// +Y up) to distorted normalized coordinates.
    ///
    /// The input is mapped through `S = diag(1, −1, −1)` into the optical
    /// frame the kernels expect (see the module docs). For perspective
    /// models this computes `(rx/(−rz), ry/(−rz))` y-flipped, then applies
    /// distortion. For fisheye models, the distorted coordinates come
    /// directly from the incidence angle off the −Z optical axis, avoiding
    /// the `tan(theta)` singularity. For equirectangular, maps via
    /// longitude/latitude. This is the true inverse of [`Self::undistort_to_ray`].
    ///
    /// Returns `None` if the ray falls outside the model's valid domain:
    /// for perspective models, when the ray is not in front of the camera
    /// (`rz >= 0`); for the polynomial fisheye family, only when the
    /// distortion polynomial's representable range is exceeded.
    /// [`CameraModel::EquidistantFisheye`] and
    /// [`CameraModel::Equirectangular`] have no invalid domain and always
    /// return `Some`.
    pub fn distort_ray(&self, ray: [f64; 3]) -> Option<(f64, f64)> {
        // Canonical → optical frame: (rx, ry, rz) ← S · ray. Every branch
        // below operates in the legacy optical frame (+Z forward, y down).
        let [rx, ry, rz] = [ray[0], -ray[1], -ray[2]];
        match self {
            // Equirectangular: longitude/latitude mapping. Pano-up is camera
            // +Y (optical −y): a ray above the horizon must land above the
            // image centre (y_d < 0), hence `asin(ry_optical)` here.
            CameraModel::Equirectangular { .. } => {
                let longitude = rx.atan2(rz);
                let r_len = (rx * rx + ry * ry + rz * rz).sqrt();
                let latitude = (ry / r_len).clamp(-1.0, 1.0).asin();
                Some((longitude, latitude))
            }

            // Perspective models: divide by the optical-frame rz, then
            // distort. `rz <= 0` here is a canonical-space z >= 0 — the ray
            // is not in front of the camera. `SFMTOOL_PINHOLE` belongs here
            // outright: its radial coordinate `ρ = √(rx² + ry²)/rz` IS the
            // quotient this arm forms, so the spline needs no ray-space entry
            // point of its own and an inactive spline reproduces the
            // `SIMPLE_PINHOLE` arithmetic bit for bit.
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::SimpleRadial { .. }
            | CameraModel::Radial { .. }
            | CameraModel::OpenCV { .. }
            | CameraModel::FullOpenCV { .. }
            | CameraModel::SfmtoolPinhole { .. } => {
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

            // Distortion-free equidistant: exact closed form at every θ, so
            // there is no polynomial range to fall out of — always `Some`.
            CameraModel::EquidistantFisheye { .. } => {
                Some(distort_ray_equidistant_exact(rx, ry, rz))
            }

            // Spline equidistant: `θ_d = θ + δ(θ)`, with the same
            // fold gate as the polynomial family (`None` where a
            // non-monotone spline drives `θ_d` non-positive).
            CameraModel::SfmtoolFisheye {
                bspline,
                bspline_theta_max,
                ..
            } => distort_ray_sfmtool_fisheye(rx, ry, rz, bspline, *bspline_theta_max),

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

            // Thin prism family: the incidence angle straight into the
            // theta-space kernel, which is where these two models are
            // defined. The `distort_*_fisheye` kernels the `distort` arms
            // call are the *perspective* front door to that same core, so
            // calling them from here would apply `atan` to an angle.
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
            } => Some(distort_ray_thin_prism_fisheye(
                rx, ry, rz, *k1, *k2, *p1, *p2, *k3, *k4, *sx1, *sy1,
            )),

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
            } => Some(distort_ray_rad_tan_thin_prism_fisheye(
                rx, ry, rz, *k0, *k1, *k2, *k3, *k4, *k5, *p0, *p1, *s0, *s1, *s2, *s3,
            )),
        }
    }

    /// Convert distorted normalized coordinates to a unit ray direction in
    /// **canonical camera space** (−Z forward, +Y up).
    ///
    /// For perspective models, equivalent to normalizing
    /// `(undistort(x_d, y_d), 1)` mapped through `S` — i.e.
    /// `(x, −y, −1)`-style rays. For fisheye models, computes the ray
    /// directly from the incidence angle theta, avoiding the `tan(theta)`
    /// singularity that causes [`Self::undistort`] to break down at and beyond 90°
    /// from the optical axis.
    ///
    /// The returned vector is unit-length and points in the direction the
    /// camera pixel is looking (a pixel at the principal point maps to
    /// `(0, 0, −1)`).
    pub fn undistort_to_ray(&self, x_d: f64, y_d: f64) -> [f64; 3] {
        // Equirectangular is derived directly in the canonical frame; every
        // other model runs the legacy optical-frame kernels and maps the
        // result back through S = diag(1, −1, −1) (module docs, D7).
        if let CameraModel::Equirectangular { .. } = self {
            // x_d is longitude (0 at −Z, +π/2 at +X); y_d is negated
            // latitude (pixel v grows down, latitude grows up).
            let longitude = x_d;
            let latitude = -y_d;
            let cos_lat = latitude.cos();
            return [
                longitude.sin() * cos_lat,
                latitude.sin(),
                -(longitude.cos() * cos_lat),
            ];
        }
        let [x, y, z] = self.undistort_to_ray_optical(x_d, y_d);
        [x, -y, -z]
    }

    /// Optical-frame (+Z forward, y down) body of [`Self::undistort_to_ray`]: the
    /// unchanged COLMAP/OpenCV kernel math. Callers outside the D7 boundary
    /// must use [`Self::undistort_to_ray`].
    fn undistort_to_ray_optical(&self, x_d: f64, y_d: f64) -> [f64; 3] {
        match self {
            CameraModel::Equirectangular { .. } => {
                unreachable!("equirectangular is handled canonically in undistort_to_ray")
            }

            // Perspective models: undistort then normalize (x, y, 1). The
            // spline pinhole's `undistort` arm is its exact Newton inverse, so
            // this is the exact inverse of `distort_ray` for it too.
            CameraModel::Pinhole { .. }
            | CameraModel::SimplePinhole { .. }
            | CameraModel::SimpleRadial { .. }
            | CameraModel::Radial { .. }
            | CameraModel::OpenCV { .. }
            | CameraModel::FullOpenCV { .. }
            | CameraModel::SfmtoolPinhole { .. } => {
                let (x, y) = self.undistort(x_d, y_d);
                let len = (x * x + y * y + 1.0).sqrt();
                [x / len, y / len, 1.0 / len]
            }

            // Distortion-free equidistant: `θ = r_d` outright — no Newton
            // recovery and no wide-angle blend, both of which exist only to
            // cope with the distortion polynomial.
            CameraModel::EquidistantFisheye { .. } => equidistant_to_ray(x_d, y_d),

            // Radial spline: the exact Newton inverse of the monotone
            // `θ_d(θ)`, no wide-angle blend (same policy as
            // SIMPLE_RADIAL_FISHEYE below — the spline is largest at the
            // periphery, exactly where a blend would drop it).
            CameraModel::SfmtoolFisheye {
                bspline,
                bspline_theta_max,
                ..
            } => sfmtool_fisheye_to_ray(x_d, y_d, bspline, *bspline_theta_max),

            // Equidistant fisheye family: recover theta, build ray directly
            CameraModel::OpenCVFisheye {
                radial_distortion_k1: k1,
                radial_distortion_k2: k2,
                radial_distortion_k3: k3,
                radial_distortion_k4: k4,
                ..
            } => equidistant_fisheye_to_ray(x_d, y_d, *k1, *k2, *k3, *k4),

            // One coefficient: the exact Newton inverse, no wide-angle blend
            // (which would drop `k1` past 90°, where it is largest).
            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k,
                ..
            } => simple_radial_fisheye_to_ray(x_d, y_d, *k),

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

#[cfg(test)]
mod tests;
