// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Warp map for image distortion and undistortion.
//!
//! A [`WarpMap`] stores a per-pixel mapping from destination image coordinates
//! to source image coordinates, enabling efficient image resampling for
//! distortion, undistortion, or re-projection between camera models.
//!
//! [`WarpMapSvd`] stores the singular value decomposition of the local Jacobian
//! at each pixel, useful for adaptive filtering during resampling.

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::camera_intrinsics::CameraIntrinsics;
use crate::rigid_transform::RigidTransform;
use crate::rot_quaternion::RotQuaternion;

/// A dense pixel-to-pixel warp map from a destination image to a source image.
///
/// Coordinates are stored as interleaved `(x, y)` f32 pairs in row-major order.
/// The total length of the internal buffer is `2 * width * height`.
/// Out-of-bounds or invalid pixels are stored as `(NaN, NaN)`.
pub struct WarpMap {
    width: u32,
    height: u32,
    /// Interleaved (x, y) pairs, row-major. Length = 2 * width * height.
    data: Vec<f32>,
    /// Optional SVD of the local Jacobian at each pixel.
    svd: Option<WarpMapSvd>,
}

/// Singular value decomposition of the 2x2 Jacobian at each warp map pixel.
///
/// For each pixel the Jacobian of the warp is decomposed as `J = U * S * V^T`
/// where `S = diag(sigma_major, sigma_minor)` with `sigma_major >= sigma_minor`.
/// `major_dir` stores the column of `V` corresponding to `sigma_major` (the
/// direction in destination space that maps to the largest stretch in source space).
pub struct WarpMapSvd {
    /// Major singular value per pixel. Length = width * height.
    pub sigma_major: Vec<f32>,
    /// Minor singular value per pixel. Length = width * height.
    pub sigma_minor: Vec<f32>,
    /// Interleaved (dx, dy) direction of the major singular vector in
    /// destination space. Length = 2 * width * height.
    pub major_dir: Vec<f32>,
}

impl WarpMap {
    /// Create a warp map from raw data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != 2 * width * height`.
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            2 * width as usize * height as usize,
            "WarpMap data length must be 2 * width * height"
        );
        Self {
            width,
            height,
            data,
            svd: None,
        }
    }

    /// Build a warp map that maps every pixel in `dst_camera` to its
    /// corresponding location in `src_camera`.
    ///
    /// For each destination pixel center `(col + 0.5, row + 0.5)`:
    /// - If either camera [`needs_ray_path`](crate::camera_intrinsics::CameraModel::needs_ray_path)
    ///   (fisheye or equirectangular), the ray-based path is used:
    ///   `dst_camera.pixel_to_ray()` then `src_camera.ray_to_pixel()`.
    /// - Otherwise both cameras are perspective and the image-plane path is
    ///   used: `dst_camera.unproject()` then `src_camera.project()`.
    /// - Source coordinates outside `[0, src_width) x [0, src_height)` are
    ///   stored as `(NaN, NaN)`.
    ///
    /// Rows are computed in parallel via rayon.
    pub fn from_cameras(src_camera: &CameraIntrinsics, dst_camera: &CameraIntrinsics) -> Self {
        let dst_w = dst_camera.width;
        let dst_h = dst_camera.height;
        let src_w = src_camera.width as f64;
        let src_h = src_camera.height as f64;
        let use_ray_path = src_camera.model.needs_ray_path() || dst_camera.model.needs_ray_path();

        // Each row produces 2 * dst_w f32 values.
        let row_len = 2 * dst_w as usize;
        let data: Vec<f32> = (0..dst_h)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![0.0f32; row_len];
                let v = row as f64 + 0.5;
                for col in 0..dst_w {
                    let u = col as f64 + 0.5;
                    let (sx, sy) = if use_ray_path {
                        let ray = dst_camera.pixel_to_ray(u, v);
                        match src_camera.ray_to_pixel(ray) {
                            Some((px, py)) => (px, py),
                            None => (f64::NAN, f64::NAN),
                        }
                    } else {
                        let (x, y) = dst_camera.unproject(u, v);
                        src_camera.project(x, y)
                    };

                    // Bounds check against source image.
                    let (sx, sy) = if sx >= 0.0 && sy >= 0.0 && sx < src_w && sy < src_h {
                        (sx, sy)
                    } else {
                        (f64::NAN, f64::NAN)
                    };

                    let idx = 2 * col as usize;
                    row_data[idx] = sx as f32;
                    row_data[idx + 1] = sy as f32;
                }
                row_data
            })
            .collect();

        WarpMap {
            width: dst_w,
            height: dst_h,
            data,
            svd: None,
        }
    }

    /// Build a warp map that assumes the scene is infinitely far away — only
    /// the relative rotation between the two cameras affects the projection.
    ///
    /// For each destination pixel center `(col + 0.5, row + 0.5)`:
    /// - `d_dst = dst_camera.pixel_to_ray(u, v)` (unit ray in dst-camera frame)
    /// - `d_src = rot_src_from_dst * d_dst` (rotate into src-camera frame)
    /// - `(sx, sy) = src_camera.ray_to_pixel(d_src)`
    /// - Source coordinates outside `[0, src_w) x [0, src_h)` are stored as `(NaN, NaN)`.
    ///
    /// Passing the identity rotation recovers [`from_cameras`] (both code
    /// paths project the same ray through the src camera).
    ///
    /// Rows are computed in parallel via rayon.
    pub fn from_cameras_with_rotation(
        src_camera: &CameraIntrinsics,
        dst_camera: &CameraIntrinsics,
        rot_src_from_dst: &RotQuaternion,
    ) -> Self {
        let r = rot_src_from_dst.to_rotation_matrix();
        Self::build_with_pose_impl(src_camera, dst_camera, &r, None, f64::INFINITY)
    }

    /// Build a warp map under the assumption that every dst ray hits a point
    /// at radial distance `depth` from the dst camera center.
    ///
    /// For each destination pixel center `(col + 0.5, row + 0.5)`:
    /// - `d_dst = dst_camera.pixel_to_ray(u, v)` (unit ray in dst-camera frame)
    /// - `p_dst = depth * d_dst` (point in dst-camera frame)
    /// - `p_world = dst_from_world.inverse().transform(p_dst)`
    /// - `p_src = src_from_world.transform(p_world)`
    /// - `(sx, sy) = src_camera.ray_to_pixel(p_src)`
    /// - Source coordinates outside `[0, src_w) x [0, src_h)`, or behind a
    ///   perspective src camera, are stored as `(NaN, NaN)`.
    ///
    /// Passing `depth = f64::INFINITY` short-circuits to the
    /// [`from_cameras_with_rotation`] path using only the relative rotation
    /// `R_src * R_dst^T`.
    ///
    /// The formulation is collapsed to
    /// `p_src = R_sd * p_dst + T_sd` where
    /// `R_sd = R_sw * R_dw^T` and `T_sd = t_sw - R_sd * t_dw`,
    /// so the math involves exactly one 3x3 matrix multiply and one vector
    /// add per dst pixel — no inverse, no small-angle approximation.
    ///
    /// Rows are computed in parallel via rayon.
    pub fn from_cameras_with_pose(
        src_camera: &CameraIntrinsics,
        dst_camera: &CameraIntrinsics,
        src_from_world: &RigidTransform,
        dst_from_world: &RigidTransform,
        depth: f64,
    ) -> Self {
        let r_sw = src_from_world.to_rotation_matrix();
        let r_dw = dst_from_world.to_rotation_matrix();
        let r_sd = r_sw * r_dw.transpose();

        if !depth.is_finite() {
            return Self::build_with_pose_impl(src_camera, dst_camera, &r_sd, None, depth);
        }

        let t_sd = src_from_world.translation - r_sd * dst_from_world.translation;
        Self::build_with_pose_impl(src_camera, dst_camera, &r_sd, Some(t_sd), depth)
    }

    /// Shared implementation for the pose-aware constructors.
    ///
    /// When `t_sd` is `None` the map is built from rays only (the depth is
    /// either infinite or irrelevant). When `t_sd` is `Some`, each dst ray is
    /// traced to `depth * d_dst` in dst-camera frame, then transformed to
    /// src-camera frame as `R_sd * p_dst + t_sd`.
    fn build_with_pose_impl(
        src_camera: &CameraIntrinsics,
        dst_camera: &CameraIntrinsics,
        r_sd: &Matrix3<f64>,
        t_sd: Option<Vector3<f64>>,
        depth: f64,
    ) -> Self {
        let dst_w = dst_camera.width;
        let dst_h = dst_camera.height;
        let src_w = src_camera.width as f64;
        let src_h = src_camera.height as f64;
        let row_len = 2 * dst_w as usize;

        let data: Vec<f32> = (0..dst_h)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![0.0f32; row_len];
                let v = row as f64 + 0.5;
                for col in 0..dst_w {
                    let u = col as f64 + 0.5;
                    let d_dst = dst_camera.pixel_to_ray(u, v);
                    let d_dst_vec = Vector3::new(d_dst[0], d_dst[1], d_dst[2]);

                    let p_src_vec = match t_sd {
                        Some(t) => r_sd * (depth * d_dst_vec) + t,
                        None => r_sd * d_dst_vec,
                    };
                    let ray_src = [p_src_vec.x, p_src_vec.y, p_src_vec.z];

                    let (sx, sy) = match src_camera.ray_to_pixel(ray_src) {
                        Some((px, py)) => (px, py),
                        None => (f64::NAN, f64::NAN),
                    };

                    let (sx, sy) = if sx >= 0.0 && sy >= 0.0 && sx < src_w && sy < src_h {
                        (sx, sy)
                    } else {
                        (f64::NAN, f64::NAN)
                    };

                    let idx = 2 * col as usize;
                    row_data[idx] = sx as f32;
                    row_data[idx + 1] = sy as f32;
                }
                row_data
            })
            .collect();

        WarpMap {
            width: dst_w,
            height: dst_h,
            data,
            svd: None,
        }
    }

    /// Returns the width of the destination image.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of the destination image.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns `true` if the pixel at `(col, row)` has a valid (non-NaN) mapping.
    #[inline]
    pub fn is_valid(&self, col: u32, row: u32) -> bool {
        let idx = 2 * (row as usize * self.width as usize + col as usize);
        !self.data[idx].is_nan()
    }

    /// Returns the source `(x, y)` coordinates for the destination pixel at
    /// `(col, row)`. May be `(NaN, NaN)` if the pixel is invalid.
    #[inline]
    pub fn get(&self, col: u32, row: u32) -> (f32, f32) {
        let idx = 2 * (row as usize * self.width as usize + col as usize);
        (self.data[idx], self.data[idx + 1])
    }

    /// Returns `true` if [`compute_svd`](Self::compute_svd) has been called.
    pub fn has_svd(&self) -> bool {
        self.svd.is_some()
    }

    /// Returns a reference to the SVD data, if computed.
    pub fn svd(&self) -> Option<&WarpMapSvd> {
        self.svd.as_ref()
    }

    /// Look up the precomputed SVD at a single pixel.
    ///
    /// Returns `(sigma_major, sigma_minor, major_dx, major_dy)`.
    ///
    /// # Panics
    ///
    /// Panics if [`compute_svd`](Self::compute_svd) has not been called.
    pub fn get_svd(&self, col: u32, row: u32) -> (f32, f32, f32, f32) {
        let svd = self
            .svd
            .as_ref()
            .expect("SVD not computed; call compute_svd() first");
        let idx = row as usize * self.width as usize + col as usize;
        let dir_idx = 2 * idx;
        (
            svd.sigma_major[idx],
            svd.sigma_minor[idx],
            svd.major_dir[dir_idx],
            svd.major_dir[dir_idx + 1],
        )
    }

    /// Compute the SVD of the local 2x2 Jacobian at each pixel using central
    /// differences, and store the result internally.
    ///
    /// The Jacobian `J` maps a unit displacement in destination pixel space to
    /// the corresponding displacement in source pixel space. It is estimated
    /// via central differences:
    ///
    /// ```text
    /// J = [[dx/dcol, dx/drow],
    ///      [dy/dcol, dy/drow]]
    /// ```
    ///
    /// At image boundaries or where any neighbour is NaN, the identity Jacobian
    /// is used, yielding `sigma_major = sigma_minor = 1` and `major_dir = (1, 0)`.
    pub fn compute_svd(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;

        let (sigma_major, sigma_minor, major_dir) = (0..h)
            .into_par_iter()
            .map(|row| {
                let mut row_major = Vec::with_capacity(w);
                let mut row_minor = Vec::with_capacity(w);
                let mut row_dir = Vec::with_capacity(2 * w);

                for col in 0..w {
                    let (s_maj, s_min, dx, dy) = self.jacobian_svd_at(col, row, w, h);
                    row_major.push(s_maj);
                    row_minor.push(s_min);
                    row_dir.push(dx);
                    row_dir.push(dy);
                }
                (row_major, row_minor, row_dir)
            })
            .reduce(
                || {
                    (
                        Vec::with_capacity(n),
                        Vec::with_capacity(n),
                        Vec::with_capacity(2 * n),
                    )
                },
                |mut acc, row| {
                    acc.0.extend(row.0);
                    acc.1.extend(row.1);
                    acc.2.extend(row.2);
                    acc
                },
            );

        self.svd = Some(WarpMapSvd {
            sigma_major,
            sigma_minor,
            major_dir,
        });
    }

    /// Compute the closed-form SVD of the 2x2 Jacobian at a single pixel.
    ///
    /// Returns `(sigma_major, sigma_minor, dir_x, dir_y)`.
    fn jacobian_svd_at(&self, col: usize, row: usize, w: usize, h: usize) -> (f32, f32, f32, f32) {
        // Identity fallback.
        let identity = (1.0f32, 1.0f32, 1.0f32, 0.0f32);

        // Need valid neighbours on both sides for central differences.
        if col == 0 || col >= w - 1 || row == 0 || row >= h - 1 {
            return identity;
        }

        let idx = |c: usize, r: usize| -> usize { 2 * (r * w + c) };

        let xl = self.data[idx(col - 1, row)];
        let xr = self.data[idx(col + 1, row)];
        let yl = self.data[idx(col - 1, row) + 1];
        let yr = self.data[idx(col + 1, row) + 1];

        let xt = self.data[idx(col, row - 1)];
        let xb = self.data[idx(col, row + 1)];
        let yt = self.data[idx(col, row - 1) + 1];
        let yb = self.data[idx(col, row + 1) + 1];

        // If any neighbour is NaN, fall back to identity.
        if xl.is_nan()
            || xr.is_nan()
            || yl.is_nan()
            || yr.is_nan()
            || xt.is_nan()
            || xb.is_nan()
            || yt.is_nan()
            || yb.is_nan()
        {
            return identity;
        }

        // Central differences (denominator is 2 pixels).
        let a = (xr - xl) * 0.5; // dx/dcol
        let b = (xb - xt) * 0.5; // dx/drow
        let c = (yr - yl) * 0.5; // dy/dcol
        let d = (yb - yt) * 0.5; // dy/drow

        // Closed-form SVD of 2x2 matrix [[a, b], [c, d]].
        svd_2x2(a, b, c, d)
    }
}

/// Closed-form SVD of a 2x2 matrix `[[a, b], [c, d]]`.
///
/// Returns `(sigma_major, sigma_minor, v_major_x, v_major_y)` where
/// `sigma_major >= sigma_minor >= 0` and `(v_major_x, v_major_y)` is the
/// right singular vector corresponding to `sigma_major`.
fn svd_2x2(a: f32, b: f32, c: f32, d: f32) -> (f32, f32, f32, f32) {
    // Using the standard closed-form for 2x2 SVD via the quantities:
    //   s1 = a^2 + b^2 + c^2 + d^2  (= ||M||_F^2)
    //   s2 = sqrt((a^2 + b^2 - c^2 - d^2)^2 + 4*(a*c + b*d)^2)
    // Then sigma_major = sqrt((s1 + s2) / 2), sigma_minor = sqrt((s1 - s2) / 2).

    let s1 = (a * a + b * b + c * c + d * d) as f64;
    let diff = (a * a + b * b - c * c - d * d) as f64;
    let cross = (a * c + b * d) as f64;
    let s2 = (diff * diff + 4.0 * cross * cross).sqrt();

    let sigma_major = (((s1 + s2) * 0.5).max(0.0)).sqrt() as f32;
    let sigma_minor = (((s1 - s2) * 0.5).max(0.0)).sqrt() as f32;

    // Right singular vector for sigma_major from M^T M.
    // M^T M = [[a^2+c^2, a*b+c*d], [a*b+c*d, b^2+d^2]]
    // The eigenvector for the larger eigenvalue (sigma_major^2) satisfies:
    //   (M^T M - sigma_major^2 I) v = 0
    let mtm00 = (a * a + c * c) as f64;
    let mtm01 = (a * b + c * d) as f64;
    let mtm11 = (b * b + d * d) as f64;
    let lambda = (sigma_major as f64) * (sigma_major as f64);

    // Use the row that has larger off-diagonal contribution to avoid
    // numerical issues when one row is near-zero.
    let (vx, vy) = if (mtm00 - lambda).abs() <= (mtm11 - lambda).abs() {
        // From first row: (mtm00 - lambda) * vx + mtm01 * vy = 0
        // => v proportional to (mtm01, lambda - mtm00)
        if mtm01.abs() < 1e-15 && (mtm00 - lambda).abs() < 1e-15 {
            (1.0_f64, 0.0_f64)
        } else {
            (mtm01, lambda - mtm00)
        }
    } else {
        // From second row: mtm01 * vx + (mtm11 - lambda) * vy = 0
        // => v proportional to (lambda - mtm11, mtm01)
        if mtm01.abs() < 1e-15 && (mtm11 - lambda).abs() < 1e-15 {
            (1.0_f64, 0.0_f64)
        } else {
            (lambda - mtm11, mtm01)
        }
    };

    let len = (vx * vx + vy * vy).sqrt();
    if len < 1e-30 {
        (sigma_major, sigma_minor, 1.0, 0.0)
    } else {
        (
            sigma_major,
            sigma_minor,
            (vx / len) as f32,
            (vy / len) as f32,
        )
    }
}

#[cfg(test)]
mod tests;
