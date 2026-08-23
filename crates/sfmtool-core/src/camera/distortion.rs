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

use crate::camera::{CameraIntrinsics, CameraModel};

use self::bspline::bspline_is_inactive;

/// A projected pixel `(u, v)` paired with the 2×3 Jacobian `∂(u, v)/∂ray`
/// (row-major `[[∂u/∂x, ∂u/∂y, ∂u/∂z], [∂v/∂x, ∂v/∂y, ∂v/∂z]]`), returned by
/// [`CameraIntrinsics::ray_to_pixel_with_jacobian`].
pub type PixelJacobian = ((f64, f64), [[f64; 3]; 2]);

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
mod pinhole_fit;
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

// ---------------------------------------------------------------------------
// CameraIntrinsics: pixel-space convenience
// ---------------------------------------------------------------------------

impl CameraIntrinsics {
    /// Project an undistorted **canonical** (y-up) image-plane point to pixel
    /// coordinates.
    ///
    /// `(x, y)` is `(p.x/(−p.z), p.y/(−p.z))` of a canonical camera-space
    /// point in front of the camera. The y axis is flipped into the y-down
    /// kernel frame, distortion is applied, and the result is converted to
    /// pixels: `(x, y)` → distort(x, −y) → `(u, v)` where `u = fx * x_d + cx`.
    pub fn project(&self, x: f64, y: f64) -> (f64, f64) {
        let (x_d, y_d) = self.model.distort(x, -y);
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        (fx * x_d + cx, fy * y_d + cy)
    }

    /// Unproject pixel coordinates to undistorted **canonical** (y-up)
    /// image-plane coordinates.
    ///
    /// Converts pixel to distorted image-plane, removes distortion, then
    /// flips y back up: `(u, v)` → `(x_d, y_d)` → undistort → `(x, −y)`.
    ///
    /// The returned `(x, y)` can be used as a ray direction `(x, y, −1)`.
    pub fn unproject(&self, u: f64, v: f64) -> (f64, f64) {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        let x_d = (u - cx) / fx;
        let y_d = (v - cy) / fy;
        let (x, y) = self.model.undistort(x_d, y_d);
        (x, -y)
    }

    /// Project a batch of undistorted canonical image-plane points to pixel
    /// coordinates. See [`project`](Self::project).
    pub fn project_batch(&self, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        points
            .par_iter()
            .map(|&[x, y]| {
                let (x_d, y_d) = self.model.distort(x, -y);
                [fx * x_d + cx, fy * y_d + cy]
            })
            .collect()
    }

    /// Unproject a batch of pixel coordinates to undistorted canonical
    /// image-plane coordinates. See [`Self::unproject`].
    pub fn unproject_batch(&self, pixels: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        pixels
            .par_iter()
            .map(|&[u, v]| {
                let x_d = (u - cx) / fx;
                let y_d = (v - cy) / fy;
                let (x, y) = self.model.undistort(x_d, y_d);
                [x, -y]
            })
            .collect()
    }

    /// Convert pixel coordinates to a unit ray direction in canonical camera
    /// space (−Z forward, +Y up).
    ///
    /// For perspective models, equivalent to normalizing `(unproject(u, v), −1)`.
    /// For fisheye models, computes the ray directly from the incidence angle,
    /// avoiding the `tan(theta)` singularity that causes [`Self::unproject`] to break
    /// down at and beyond 90° from the optical axis. This makes it suitable for
    /// wide-angle fisheye lenses with field of view approaching or exceeding 180°.
    pub fn pixel_to_ray(&self, u: f64, v: f64) -> [f64; 3] {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        let x_d = (u - cx) / fx;
        let y_d = (v - cy) / fy;
        self.model.undistort_to_ray(x_d, y_d)
    }

    /// Project a ray direction in canonical camera space (−Z forward, +Y up)
    /// to pixel coordinates.
    ///
    /// For perspective models, equivalent to `project(rx/(−rz), ry/(−rz))`,
    /// but for fisheye models computes the distorted coordinates directly from
    /// the incidence angle, avoiding the `tan(theta)` singularity. For
    /// equirectangular, maps via longitude/latitude. This is the true inverse
    /// of [`Self::pixel_to_ray`].
    ///
    /// Returns `None` if the ray falls outside the model's valid domain.
    pub fn ray_to_pixel(&self, ray: [f64; 3]) -> Option<(f64, f64)> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        let (x_d, y_d) = self.model.distort_ray(ray)?;
        Some((fx * x_d + cx, fy * y_d + cy))
    }

    /// [`Self::ray_to_pixel`] plus the analytic Jacobian `∂(u, v)/∂ray` of the pixel
    /// with respect to the camera-frame ray direction, row-major
    /// `[[∂u/∂x, ∂u/∂y, ∂u/∂z], [∂v/∂x, ∂v/∂y, ∂v/∂z]]`.
    ///
    /// The perspective family — [`CameraModel::SfmtoolPinhole`] included, its
    /// radial spline entering as the family's `g(ρ) = 1 + δ(ρ)/ρ` — plus the
    /// θ-map fisheye trio [`CameraModel::EquidistantFisheye`],
    /// [`CameraModel::SimpleRadialFisheye`] and
    /// [`CameraModel::SfmtoolFisheye`] (`supports_pixel_jacobian`) — the
    /// first two share the closed-form `θ_d = θ·(1 + k1·θ²)` derivative (with
    /// `k1 = 0` for the distortion-free map), and the third substitutes the
    /// spline pair `θ_d = θ + δ(θ)`, `θ_d' = 1 + δ'(θ)` into the same radial
    /// template. Returns `None` when the ray is
    /// outside the model's valid domain — exactly where
    /// [`Self::ray_to_pixel`] returns `None`, with one documented exception
    /// below — or when the model has no analytic Jacobian (multi-coefficient
    /// fisheye / equirectangular), so a caller can fall back to a finite
    /// difference for those.
    ///
    /// The exception is the equidistant family at the **antipode**
    /// (`θ = π`, `r_xy = 0`): [`Self::ray_to_pixel`] maps it to the principal
    /// point, but the derivative there is unbounded, so this returns `None`.
    ///
    /// The projection is scale-invariant in the ray, so this is the derivative
    /// with respect to the supplied (possibly non-unit) ray components — i.e.
    /// with respect to a camera-frame point when one is passed directly.
    pub fn ray_to_pixel_with_jacobian(&self, ray: [f64; 3]) -> Option<PixelJacobian> {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        // Canonical → optical frame: (rx, ry, rz) = S·ray, S = diag(1, −1, −1).
        let [rx, ry, rz] = [ray[0], -ray[1], -ray[2]];

        // Equidistant fisheye family with a closed-form `θ_d(θ)` and
        // `θ_d'(θ)`: the distortion-free `θ = r/f` map, the
        // single-coefficient `θ_d = θ·(1 + k1·θ²)`, and the spline
        // `θ_d = θ + δ(θ)` — each arm hands its own `(θ_d, θ_d')` pair to the
        // shared radial Jacobian template. Dispatched BEFORE the perspective
        // in-front guard — rays past 90° (optical `rz ≤ 0`) are the periphery
        // these models exist to carry, not a domain error.
        let theta_map_jac = match &self.model {
            CameraModel::EquidistantFisheye { .. } => {
                Some(radial_fisheye_ray_jacobian(rx, ry, rz, 0.0))
            }
            CameraModel::SimpleRadialFisheye {
                radial_distortion_k1: k1,
                ..
            } => Some(radial_fisheye_ray_jacobian(rx, ry, rz, *k1)),
            CameraModel::SfmtoolFisheye {
                bspline,
                bspline_theta_max,
                ..
            } => Some(sfmtool_fisheye_ray_jacobian(
                rx,
                ry,
                rz,
                bspline,
                *bspline_theta_max,
            )),
            _ => None,
        };
        if let Some(jac) = theta_map_jac {
            let ((x_d, y_d), jd) = jac?;
            // J = diag(fx, fy) · Jd · S, and S negates the ry, rz columns.
            return Some((
                (fx * x_d + cx, fy * y_d + cy),
                [
                    [fx * jd[0][0], -fx * jd[0][1], -fx * jd[0][2]],
                    [fy * jd[1][0], -fy * jd[1][1], -fy * jd[1][2]],
                ],
            ));
        }

        // Perspective family: the ray must be in front of the camera.
        if rz <= 0.0 {
            return None;
        }
        let x = rx / rz;
        let y = ry / rz;

        // Pinhole fast path: no distortion, so the domain is unconditionally
        // valid and D is the identity. Skip the distortion Jacobian and the
        // 2×2 composition and write J = diag(fx, fy)·(P·S) directly.
        //
        // A `SFMTOOL_PINHOLE` whose spline is inactive projects as
        // `SIMPLE_PINHOLE`, and takes this path so it does so with the SAME
        // arithmetic: the general composition below rounds `fx·(rx/rz²)` in a
        // different association, which would cost the zero-spline promotion
        // its bit-identity.
        let undistorted_pinhole = match &self.model {
            CameraModel::Pinhole { .. } | CameraModel::SimplePinhole { .. } => true,
            CameraModel::SfmtoolPinhole {
                bspline,
                bspline_rho_max,
                ..
            } => bspline_is_inactive(bspline, *bspline_rho_max),
            _ => false,
        };
        if undistorted_pinhole {
            let inv = 1.0 / rz;
            return Some((
                (fx * x + cx, fy * y + cy),
                [
                    [fx * inv, 0.0, fx * rx * inv * inv],
                    [0.0, -fy * inv, fy * ry * inv * inv],
                ],
            ));
        }

        if !self.model.forward_projection_invertible(x, y) {
            return None;
        }
        // 2×2 distortion Jacobian ∂(x_d, y_d)/∂(x, y); None for unsupported models.
        let d = self.model.distort_jacobian(x, y)?;
        let (x_d, y_d) = self.model.distort(x, y);

        // ∂(x, y)/∂ray = P·S, where P = ∂(x, y)/∂(rx, ry, rz) and S flips the
        // sign of the ry, rz columns: [[1/rz, 0, rx/rz²], [0, −1/rz, ry/rz²]].
        let inv = 1.0 / rz;
        let ps = [[inv, 0.0, rx * inv * inv], [0.0, -inv, ry * inv * inv]];
        // J = diag(fx, fy) · D · (P·S).
        let mut jac = [[0.0f64; 3]; 2];
        for c in 0..3 {
            let m0 = d[0][0] * ps[0][c] + d[0][1] * ps[1][c];
            let m1 = d[1][0] * ps[0][c] + d[1][1] * ps[1][c];
            jac[0][c] = fx * m0;
            jac[1][c] = fy * m1;
        }
        Some(((fx * x_d + cx, fy * y_d + cy), jac))
    }

    /// The 2×3 pixel Jacobian `∂(u, v)/∂p_cam` at the camera-frame point
    /// `p_cam`, analytic where the model has one and a central difference of
    /// [`Self::ray_to_pixel`] otherwise.
    ///
    /// `None` when `p_cam` (or a difference probe around it) falls outside the
    /// model's domain, or when `p_cam` is the origin.
    fn pixel_jacobian(&self, p_cam: [f64; 3]) -> Option<[[f64; 3]; 2]> {
        if self.model.supports_pixel_jacobian() {
            return self.ray_to_pixel_with_jacobian(p_cam).map(|(_, j)| j);
        }
        // Polynomial fisheye / equirectangular: no analytic derivative, so
        // difference the projection itself. The step is relative to ‖p‖ because
        // the projection is scale-invariant in the ray — `1e-6·‖p‖` sits near
        // the central-difference optimum for f64 (truncation ~1e-12 relative,
        // round-off ~1e-10 relative).
        let n = (p_cam[0] * p_cam[0] + p_cam[1] * p_cam[1] + p_cam[2] * p_cam[2]).sqrt();
        if n <= 0.0 || n.is_nan() {
            return None;
        }
        let h = 1e-6 * n;
        let mut jac = [[0.0f64; 3]; 2];
        for col in 0..3 {
            let mut plus = p_cam;
            let mut minus = p_cam;
            plus[col] += h;
            minus[col] -= h;
            let (up, vp) = self.ray_to_pixel(plus)?;
            let (um, vm) = self.ray_to_pixel(minus)?;
            jac[0][col] = (up - um) / (2.0 * h);
            jac[1][col] = (vp - vm) / (2.0 * h);
        }
        Some(jac)
    }

    /// The local **pixel scale** at the camera-frame point `p_cam`: the smaller
    /// singular value `σ_min` of the pixel Jacobian `J = ∂(u, v)/∂p_cam`, in
    /// pixels per world unit.
    ///
    /// Every model here projects by *direction* only, so `J·p_cam = 0` — `J`'s
    /// null space is the viewing ray itself, and its two singular values are the
    /// pixels-per-world-unit along the two tangent directions at `p_cam` (both
    /// `∝ 1/‖p_cam‖`). `σ_min` is the conservative one: a tangent-plane offset
    /// `δ` moves the projection by **at least** `σ_min·‖δ‖` pixels, in every
    /// direction, so `pixels / σ_min` is the world size that fits a pixel
    /// budget however the surface is oriented. See
    /// [`Self::pixel_radius_to_world`], the sizing rule built on it.
    ///
    /// `None` when the Jacobian is undefined at `p_cam` — the ray is outside the
    /// model's domain (behind a perspective camera, past a distortion
    /// polynomial's principal branch, at the equidistant antipode) or `p_cam` is
    /// the camera centre.
    pub fn min_pixel_scale(&self, p_cam: [f64; 3]) -> Option<f64> {
        let j = self.pixel_jacobian(p_cam)?;
        // Singular values of the 2×3 `J` are the square roots of the eigenvalues
        // of the symmetric 2×2 `J·Jᵀ` — a closed form, no SVD needed.
        let a = j[0][0] * j[0][0] + j[0][1] * j[0][1] + j[0][2] * j[0][2];
        let b = j[0][0] * j[1][0] + j[0][1] * j[1][1] + j[0][2] * j[1][2];
        let c = j[1][0] * j[1][0] + j[1][1] * j[1][1] + j[1][2] * j[1][2];
        let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
        let lambda_max = 0.5 * (a + c + disc);
        if lambda_max <= 0.0 || lambda_max.is_nan() {
            return None;
        }
        // `det / λ_max` rather than `(tr − disc)/2`: the difference form
        // cancels catastrophically once the two singular values separate (they
        // differ by `sec θ` off axis).
        let lambda_min = ((a * c - b * b) / lambda_max).max(0.0);
        Some(lambda_min.sqrt())
    }

    /// The world-space radius at the camera-frame point `p_cam` that projects to
    /// `radius_px` pixels: `radius_px / σ_min(J)`, with `σ_min` the local pixel
    /// scale ([`Self::min_pixel_scale`]).
    ///
    /// One rule for every camera model — the pixel Jacobian already knows how
    /// each one magnifies, so nothing here branches on projection family. Two
    /// models have an exact closed form for `σ_min`, used directly (both are
    /// algebraic identities of the general rule, not approximations of it):
    ///
    /// - **Pinhole** (`fx == fy == f`), at `θ` off axis and range `R`: the
    ///   tangent scales are `f·sec²θ/R` (radial) and `f·secθ/R` (azimuthal), so
    ///   `σ_min = f·secθ/R = f/|z|` and the radius is `radius_px·|z|/f`.
    /// - **[`CameraModel::EquidistantFisheye`]**: the tangent scales are
    ///   `f·(θ/sin θ)/R` (azimuthal) and `f/R` (radial), so `σ_min = f/R` and the
    ///   radius is `radius_px·‖p_cam‖/f` — finite past 90°, where the pinhole
    ///   `|z|` collapses to zero and inverts beyond.
    ///
    /// Every other model goes through `σ_min` itself, which is strictly more
    /// correct than either closed form: a distorted perspective model picks up
    /// the local distortion magnification that `|z|/f` ignores, and the
    /// polynomial fisheye family picks up `dr_d/dθ ≠ f`, which `‖p‖/f` assumes.
    ///
    /// When `σ_min` is undefined (the ray is outside the model's domain, so no
    /// view of that point exists there) this falls back to the angular reading
    /// `radius_px·‖p_cam‖/f`, which stays finite at every angle. The distance is
    /// floored at `1e-6` so a point sitting on the camera centre still gets a
    /// size rather than zero.
    pub fn pixel_radius_to_world(&self, p_cam: [f64; 3], radius_px: f64) -> f64 {
        match &self.model {
            CameraModel::SimplePinhole { focal_length, .. } => {
                radius_px * p_cam[2].abs().max(1e-6) / focal_length
            }
            CameraModel::Pinhole {
                focal_length_x,
                focal_length_y,
                ..
            } if focal_length_x == focal_length_y => {
                radius_px * p_cam[2].abs().max(1e-6) / focal_length_x
            }
            CameraModel::EquidistantFisheye { focal_length, .. } => {
                radius_px * ray_range(p_cam).max(1e-6) / focal_length
            }
            _ => match self.min_pixel_scale(p_cam) {
                Some(scale) if scale > 0.0 => radius_px / scale,
                _ => radius_px * ray_range(p_cam).max(1e-6) / self.focal_lengths().0,
            },
        }
    }

    /// The **angular** radius (radians) around the direction `ray` that projects
    /// to `radius_px` pixels: `radius_px / (‖ray‖·σ_min)`, the angular sibling of
    /// [`Self::pixel_radius_to_world`].
    ///
    /// `σ_min` goes as `1/‖p_cam‖`, so `‖ray‖·σ_min` is range-free — it is the
    /// local **pixels per radian** in the least-magnified tangent direction, i.e.
    /// `σ_min` of the projection restricted to the unit sphere's tangent plane at
    /// `ray`. That makes this the right sizing rule for a patch anchored to a
    /// direction rather than a position (a point at infinity), whose extent is an
    /// angle. Only the direction of `ray` matters.
    ///
    /// The same two models have exact closed forms, used directly:
    ///
    /// - **Pinhole** (`fx == fy == f`): `‖ray‖·σ_min = f·secθ`, so the angle is
    ///   `radius_px·cosθ/f`. A pixel budget buys **less angle off axis**, because
    ///   the image plane magnifies there — `radius_px/f` is only the on-axis
    ///   value.
    /// - **[`CameraModel::EquidistantFisheye`]**: `‖ray‖·σ_min = f` at every `θ`,
    ///   so the angle is `radius_px/f` outright — the one model for which the
    ///   naive reading is exact, since it is angle-linear by construction.
    ///
    /// Every other model evaluates `σ_min`. When it is undefined (the ray is
    /// outside the model's domain) this falls back to `radius_px/f`.
    pub fn pixel_radius_to_angle(&self, ray: [f64; 3], radius_px: f64) -> f64 {
        // `cos θ = |z|/‖ray‖`; the closed forms below need nothing else.
        let range = ray_range(ray);
        match &self.model {
            CameraModel::SimplePinhole { focal_length, .. } if range > 0.0 => {
                radius_px * (ray[2].abs() / range) / focal_length
            }
            CameraModel::Pinhole {
                focal_length_x,
                focal_length_y,
                ..
            } if focal_length_x == focal_length_y && range > 0.0 => {
                radius_px * (ray[2].abs() / range) / focal_length_x
            }
            CameraModel::EquidistantFisheye { focal_length, .. } => radius_px / focal_length,
            _ => match self.min_pixel_scale(ray) {
                Some(scale) if scale > 0.0 && range > 0.0 => radius_px / (range * scale),
                _ => radius_px / self.focal_lengths().0,
            },
        }
    }

    /// Batch version of [`Self::ray_to_pixel`].
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
}

/// Euclidean length of a camera-frame point, associated left-to-right so it
/// agrees bit-for-bit with `nalgebra`'s `Vector3::norm` on the same components.
fn ray_range(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

#[cfg(test)]
mod tests;
