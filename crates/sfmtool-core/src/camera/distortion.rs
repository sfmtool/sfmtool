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

use crate::camera::{CameraIntrinsics, CameraIntrinsicsError, CameraModel};

/// Maximum iterations for iterative undistortion.
const UNDISTORT_MAX_ITER: usize = 100;

/// Convergence threshold for iterative undistortion.
const UNDISTORT_EPS: f64 = 1e-10;

/// Fisheye distortion models are not coherent past ~90° from the optical axis,
/// so we blend from the distorted ray to the undistorted (identity) ray over
/// this angular range (in radians of the undistorted angle).
const FISHEYE_BLEND_START_RAD: f64 = 90.0 * (std::f64::consts::PI / 180.0); // 90°
const FISHEYE_BLEND_END_RAD: f64 = 100.0 * (std::f64::consts::PI / 180.0); // 100°

// --- Coarse ray-grid projection (non-perspective path of `ray_to_pixel_grid`) ---

/// Sub-grid spacing (in destination grid pixels) for the non-perspective path of
/// [`CameraIntrinsics::ray_to_pixel_grid`]: the exact projection is evaluated
/// every `COARSE_GRID_STRIDE` pixels and the interior is bilinearly interpolated.
/// A larger stride speeds smooth (low-curvature) tiles but is bounded for free:
/// every cell is probe-checked against the exact projection and demoted to exact
/// when it would exceed [`COARSE_GRID_TOL_PX`], so accuracy never depends on this
/// value — only the speedup does. See `specs/core/ray-grid-projection.md` and the
/// `coarse_grid_error_*` tests.
const COARSE_GRID_STRIDE: u32 = 8;

/// Per-cell source-pixel error tolerance for the coarse-grid path of
/// [`CameraIntrinsics::ray_to_pixel_grid`]. A cell is interpolated only if its
/// center and edge-midpoints match the exact projection to within this many
/// source pixels; otherwise it is projected exactly. Set an order of magnitude
/// below the localizer's sub-pixel needs.
const COARSE_GRID_TOL_PX: f32 = 0.02;

/// Linear interpolation `a + (b − a)·f`.
#[inline]
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

/// Per-grid projection constants for [`CameraIntrinsics::project_ray_node`],
/// hoisted once per grid so the per-node projection touches no enum-match for
/// the (loop-invariant) intrinsics or image bounds.
#[derive(Clone, Copy)]
struct GridProj {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
}

/// Bilinear interpolation of the four cell corners (`p00` at `(0,0)`, `p10` at
/// `(1,0)`, `p01` at `(0,1)`, `p11` at `(1,1)`) at fractional `(sf, tf)`. Shared
/// by the coarse-grid probe (acceptance) and fill so the probe predicts the fill
/// exactly.
#[inline]
fn bilerp(
    p00: [f32; 2],
    p10: [f32; 2],
    p01: [f32; 2],
    p11: [f32; 2],
    sf: f32,
    tf: f32,
) -> [f32; 2] {
    [
        lerp(lerp(p00[0], p10[0], sf), lerp(p01[0], p11[0], sf), tf),
        lerp(lerp(p00[1], p10[1], sf), lerp(p01[1], p11[1], sf), tf),
    ]
}

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

    /// Project a ray direction in **canonical camera space** (−Z forward,
    /// +Y up) to distorted normalized coordinates.
    ///
    /// The input is mapped through `S = diag(1, −1, −1)` into the optical
    /// frame the kernels expect (see the module docs). For perspective
    /// models this computes `(rx/(−rz), ry/(−rz))` y-flipped, then applies
    /// distortion. For fisheye models, the distorted coordinates come
    /// directly from the incidence angle off the −Z optical axis, avoiding
    /// the `tan(theta)` singularity. For equirectangular, maps via
    /// longitude/latitude. This is the true inverse of [`undistort_to_ray`].
    ///
    /// Returns `None` if the ray falls outside the model's valid domain:
    /// for perspective models, when the ray is not in front of the camera
    /// (`rz >= 0`); for fisheye, only when the distortion polynomial's
    /// representable range is exceeded.
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
            // is not in front of the camera.
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

    /// Convert distorted normalized coordinates to a unit ray direction in
    /// **canonical camera space** (−Z forward, +Y up).
    ///
    /// For perspective models, equivalent to normalizing
    /// `(undistort(x_d, y_d), 1)` mapped through `S` — i.e.
    /// `(x, −y, −1)`-style rays. For fisheye models, computes the ray
    /// directly from the incidence angle theta, avoiding the `tan(theta)`
    /// singularity that causes [`undistort`] to break down at and beyond 90°
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

    /// Optical-frame (+Z forward, y down) body of [`undistort_to_ray`]: the
    /// unchanged COLMAP/OpenCV kernel math. Callers outside the D7 boundary
    /// must use [`undistort_to_ray`].
    fn undistort_to_ray_optical(&self, x_d: f64, y_d: f64) -> [f64; 3] {
        match self {
            CameraModel::Equirectangular { .. } => {
                unreachable!("equirectangular is handled canonically in undistort_to_ray")
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
    /// image-plane coordinates. See [`unproject`](Self::unproject).
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

    /// Project a ray direction in canonical camera space (−Z forward, +Y up)
    /// to pixel coordinates.
    ///
    /// For perspective models, equivalent to `project(rx/(−rz), ry/(−rz))`,
    /// but for fisheye models computes the distorted coordinates directly from
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

    /// Hoist the loop-invariant projection constants (intrinsics + image bounds)
    /// for a grid projection — see [`GridProj`] and [`project_ray_node`](Self::project_ray_node).
    #[inline]
    fn grid_proj(&self) -> GridProj {
        let (fx, fy) = self.focal_lengths();
        let (cx, cy) = self.principal_point();
        GridProj {
            fx,
            fy,
            cx,
            cy,
            w: self.width as f64,
            h: self.height as f64,
        }
    }

    /// Project a single camera-frame ray to a source pixel using pre-hoisted
    /// [`GridProj`] constants, returning `(NaN, NaN)` when the model rejects the
    /// ray (behind camera / outside the invertible domain) or the result falls
    /// outside the image rectangle `[0, w) × [0, h)`. This is
    /// [`ray_to_pixel`](Self::ray_to_pixel) inlined so the grid paths fetch the
    /// (loop-invariant) intrinsics once per grid instead of once per node.
    #[inline]
    fn project_ray_node(&self, ray: [f64; 3], p: GridProj) -> [f32; 2] {
        match self.model.distort_ray(ray) {
            Some((x_d, y_d)) => {
                let px = p.fx * x_d + p.cx;
                let py = p.fy * y_d + p.cy;
                if px >= 0.0 && py >= 0.0 && px < p.w && py < p.h {
                    [px as f32, py as f32]
                } else {
                    [f32::NAN, f32::NAN]
                }
            }
            None => [f32::NAN, f32::NAN],
        }
    }

    /// Exact, per-node version of [`ray_to_pixel_grid`](Self::ray_to_pixel_grid):
    /// projects every grid node through the full camera model. Used directly for
    /// perspective models (where it is the fast path — the affine ray basis
    /// removes the per-node pose multiply, and the divide + distortion are
    /// cheap) and as the coarse-grid fallback and test reference for the
    /// non-perspective path.
    ///
    /// Sequential: this renders one patch-sized tile and every caller runs it
    /// inside a per-patch/`par_iter` loop, so the caller owns parallelism (an
    /// inner `par_chunks` would just nest rayon over ~`rows` rows). Full-image
    /// warps that want row parallelism use [`WarpMap::from_cameras`] instead.
    pub(crate) fn ray_to_pixel_grid_exact(
        &self,
        origin: [f64; 3],
        col_step: [f64; 3],
        row_step: [f64; 3],
        cols: u32,
        rows: u32,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), 2 * cols as usize * rows as usize);
        let gp = self.grid_proj();
        let cols_u = cols as usize;
        for (row, dst) in out.chunks_exact_mut(2 * cols_u).enumerate() {
            let r = row as f64;
            let ox = origin[0] + r * row_step[0];
            let oy = origin[1] + r * row_step[1];
            let oz = origin[2] + r * row_step[2];
            for col in 0..cols_u {
                let c = col as f64;
                let ray = [
                    ox + c * col_step[0],
                    oy + c * col_step[1],
                    oz + c * col_step[2],
                ];
                let p = self.project_ray_node(ray, gp);
                dst[2 * col] = p[0];
                dst[2 * col + 1] = p[1];
            }
        }
    }

    /// Project an **affine grid of camera-frame rays** to source pixel
    /// coordinates — the grid sibling of [`ray_to_pixel`](Self::ray_to_pixel).
    ///
    /// The ray at integer grid node `(col, row)` (with `col ∈ 0..cols`,
    /// `row ∈ 0..rows`) is `origin + col·col_step + row·row_step`. Results are
    /// written as interleaved `(sx, sy)` f32 pairs, row-major, into `out` (which
    /// must have length `2·cols·rows`); a node that is behind the camera, outside
    /// the distortion model's invertible domain, or outside the image rectangle
    /// is written as `(NaN, NaN)` — identical to [`ray_to_pixel`] followed by the
    /// in-frame test.
    ///
    /// Affineness of the input grid is the contract that licenses the
    /// model-specific fast paths:
    /// * **Perspective** models project every node exactly (the divide +
    ///   distortion are cheap; the win over the scalar caller is that the pose
    ///   multiply has already been folded into the affine basis).
    /// * **Fisheye / equirectangular** models, whose per-node projection is
    ///   expensive (`atan2`/`asin`) but spatially smooth, evaluate the exact
    ///   projection only on a coarse sub-grid (stride [`COARSE_GRID_STRIDE`])
    ///   and bilinearly interpolate the interior, falling back to exact
    ///   projection wherever a bracketing sub-grid node is invalid. The
    ///   interpolation error is bounded; see
    ///   `specs/core/ray-grid-projection.md`.
    pub fn ray_to_pixel_grid(
        &self,
        origin: [f64; 3],
        col_step: [f64; 3],
        row_step: [f64; 3],
        cols: u32,
        rows: u32,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), 2 * cols as usize * rows as usize);
        // Perspective models, or grids too small to amortize the sub-grid setup,
        // go straight through the exact per-node path.
        if !self.model.needs_ray_path()
            || cols < 2 * COARSE_GRID_STRIDE
            || rows < 2 * COARSE_GRID_STRIDE
        {
            self.ray_to_pixel_grid_exact(origin, col_step, row_step, cols, rows, out);
            return;
        }
        let _ = self.ray_to_pixel_grid_coarse(origin, col_step, row_step, cols, rows, out);
    }

    /// Coarse-grid interpolation path for [`ray_to_pixel_grid`] (non-perspective
    /// models). See that method's docs and `specs/core/ray-grid-projection.md`.
    /// Returns `(interpolated_cells, total_cells)` for diagnostics/tests — the
    /// hit rate of the fast (interpolated) path on this tile.
    ///
    /// The error is bounded **by construction**: each sub-grid cell is accepted
    /// for bilinear interpolation only after its center and edge-midpoints — the
    /// points where bilinear is least accurate — are projected exactly and agree
    /// with the interpolant to within [`COARSE_GRID_TOL_PX`]. Cells that fail the
    /// probe (high curvature, or an invalid corner) are projected exactly per
    /// pixel, so the worst-case deviation from the exact map stays at the
    /// tolerance regardless of geometry.
    fn ray_to_pixel_grid_coarse(
        &self,
        origin: [f64; 3],
        col_step: [f64; 3],
        row_step: [f64; 3],
        cols: u32,
        rows: u32,
        out: &mut [f32],
    ) -> (usize, usize) {
        let gp = self.grid_proj();
        let proj = |c: f64, r: f64| {
            let ray = [
                origin[0] + c * col_step[0] + r * row_step[0],
                origin[1] + c * col_step[1] + r * row_step[1],
                origin[2] + c * col_step[2] + r * row_step[2],
            ];
            self.project_ray_node(ray, gp)
        };

        // Sub-grid nodes at 0, stride, 2·stride, …, with the final node forced to
        // (axis − 1) so the endpoints are always nodes. Positions are computed
        // arithmetically (no node-index Vec): node `i` sits at `min(i·stride, n−1)`,
        // and there are `ceil((n−1)/stride)` cells.
        let stride = COARSE_GRID_STRIDE;
        let n_cells = |n: u32| (n - 1).div_ceil(stride) as usize;
        let node_pos = |n: u32, i: usize| (i as u32 * stride).min(n - 1);
        let (seg_c, seg_r) = (n_cells(cols), n_cells(rows));
        let nc = seg_c + 1;

        // Project each sub-grid node exactly, once (shared by adjacent cells).
        let mut node_px = vec![[f32::NAN; 2]; nc * (seg_r + 1)];
        for j in 0..=seg_r {
            let r = node_pos(rows, j) as f64;
            for i in 0..=seg_c {
                node_px[j * nc + i] = proj(node_pos(cols, i) as f64, r);
            }
        }

        // `[(0.5,0.5),(0.5,0),(0.5,1),(0,0.5),(1,0.5)]` are the bilinear-error
        // extrema for a separable-quadratic warp.
        const PROBES: [(f32, f32); 5] =
            [(0.5, 0.5), (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)];
        let cols_u = cols as usize;
        let mut interp_count = 0usize;

        // Walk cells. Each cell owns the half-open pixel block `[r0,r1) × [c0,c1)`;
        // the last cell on each axis additionally owns the final node row/column,
        // so every pixel is written exactly once.
        for j in 0..seg_r {
            let (r0, r1) = (node_pos(rows, j), node_pos(rows, j + 1));
            let r_end = if j == seg_r - 1 { r1 + 1 } else { r1 };
            let inv_ch = 1.0 / (r1 - r0) as f32;
            for i in 0..seg_c {
                let (c0, c1) = (node_pos(cols, i), node_pos(cols, i + 1));
                let c_end = if i == seg_c - 1 { c1 + 1 } else { c1 };
                let inv_cw = 1.0 / (c1 - c0) as f32;
                let p00 = node_px[j * nc + i];
                let p10 = node_px[j * nc + i + 1];
                let p01 = node_px[(j + 1) * nc + i];
                let p11 = node_px[(j + 1) * nc + i + 1];

                // Interpolate the cell only if every corner is valid and the probe
                // points agree with the interpolant to within the tolerance.
                let interp = p00[0].is_finite()
                    && p10[0].is_finite()
                    && p01[0].is_finite()
                    && p11[0].is_finite()
                    && PROBES.iter().all(|&(sf, tf)| {
                        let est = bilerp(p00, p10, p01, p11, sf, tf);
                        let c = c0 as f64 + sf as f64 * (c1 - c0) as f64;
                        let r = r0 as f64 + tf as f64 * (r1 - r0) as f64;
                        let ex = proj(c, r);
                        ex[0].is_finite()
                            && (est[0] - ex[0]).hypot(est[1] - ex[1]) <= COARSE_GRID_TOL_PX
                    });
                if interp {
                    interp_count += 1;
                }

                for row in r0..r_end {
                    let tf = (row - r0) as f32 * inv_ch;
                    let base = row as usize * 2 * cols_u;
                    for col in c0..c_end {
                        let o = base + 2 * col as usize;
                        let p = if interp {
                            bilerp(p00, p10, p01, p11, (col - c0) as f32 * inv_cw, tf)
                        } else {
                            proj(col as f64, row as f64)
                        };
                        out[o] = p[0];
                        out[o + 1] = p[1];
                    }
                }
            }
        }
        (interp_count, seg_c * seg_r)
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
