// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pixel-space projection and unprojection on [`CameraIntrinsics`].
//!
//! These methods wrap the normalized-space `distort` / `undistort` /
//! `distort_ray` / `undistort_to_ray` of [`CameraModel`] with the focal
//! lengths and principal point, and speak the canonical y-up, −Z-forward
//! camera frame described in the parent module. They also hold the pixel
//! Jacobian of `ray_to_pixel` and the pixel-scale helpers built on it.

use rayon::prelude::*;

use crate::camera::{CameraIntrinsics, CameraModel};

use super::bspline::bspline_is_inactive;
use super::kernels::*;
use super::PixelJacobian;

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

    /// Project the homogeneous world point `(xyz, w)` into this camera at
    /// `cam_from_world`, as a pixel `[x, y]`, with no test against the image's
    /// bounds.
    ///
    /// `w = 1` is a finite point. `w = 0` is a direction, a point at infinity:
    /// it is rotated into the camera frame without the translation and
    /// projected as a ray.
    ///
    /// The camera looks along `−Z`, so a point in front of a perspective camera
    /// has `z < 0` in the camera frame, and one at `z >= 0` returns `None`. A
    /// ray-path model ([`CameraModel::needs_ray_path`]: fisheye and
    /// equirectangular) images past 90° off the axis, where a real sighting has
    /// `z >= 0`, so for those the model's own domain, whatever
    /// [`Self::ray_to_pixel`] accepts, is the only test.
    ///
    /// A pixel outside the frame is returned as it is: a residual a pixel past
    /// the border is a small error, not a missing measurement, and a tile partly
    /// off the photograph still has a geometry.
    pub fn project_homogeneous(
        &self,
        cam_from_world: &crate::geometry::RigidTransform,
        xyz: nalgebra::Vector3<f64>,
        w: f64,
    ) -> Option<[f64; 2]> {
        let pc = cam_from_world.transform_point_homogeneous(xyz, w);
        if !self.model.needs_ray_path() && pc.z >= 0.0 {
            return None;
        }
        self.ray_to_pixel([pc.x, pc.y, pc.z]).map(|(x, y)| [x, y])
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
    pub(crate) fn pixel_jacobian(&self, p_cam: [f64; 3]) -> Option<[[f64; 3]; 2]> {
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
