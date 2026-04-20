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
mod tests {
    use super::*;
    use crate::camera_intrinsics::CameraModel;

    /// Helper: build a simple pinhole camera.
    fn pinhole(width: u32, height: u32, focal: f64) -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: focal,
                focal_length_y: focal,
                principal_point_x: width as f64 / 2.0,
                principal_point_y: height as f64 / 2.0,
            },
            width,
            height,
        }
    }

    /// Helper: build a simple radial camera with distortion.
    fn simple_radial(width: u32, height: u32, focal: f64, k1: f64) -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimpleRadial {
                focal_length: focal,
                principal_point_x: width as f64 / 2.0,
                principal_point_y: height as f64 / 2.0,
                radial_distortion_k1: k1,
            },
            width,
            height,
        }
    }

    /// Helper: build a simple radial fisheye camera.
    fn simple_radial_fisheye(width: u32, height: u32, focal: f64, k1: f64) -> CameraIntrinsics {
        CameraIntrinsics {
            model: CameraModel::SimpleRadialFisheye {
                focal_length: focal,
                principal_point_x: width as f64 / 2.0,
                principal_point_y: height as f64 / 2.0,
                radial_distortion_k1: k1,
            },
            width,
            height,
        }
    }

    /// Helper: build an equirectangular camera (full sphere).
    fn equirectangular(width: u32, height: u32) -> CameraIntrinsics {
        let fx = width as f64 / (2.0 * std::f64::consts::PI);
        let fy = height as f64 / std::f64::consts::PI;
        CameraIntrinsics {
            model: CameraModel::Equirectangular {
                focal_length_x: fx,
                focal_length_y: fy,
                principal_point_x: width as f64 / 2.0,
                principal_point_y: height as f64 / 2.0,
            },
            width,
            height,
        }
    }

    // -----------------------------------------------------------------------
    // Identity map
    // -----------------------------------------------------------------------

    #[test]
    fn identity_map_produces_pixel_centers() {
        let cam = pinhole(64, 48, 100.0);
        let warp = WarpMap::from_cameras(&cam, &cam);

        assert_eq!(warp.width(), 64);
        assert_eq!(warp.height(), 48);

        for row in 0..warp.height() {
            for col in 0..warp.width() {
                assert!(
                    warp.is_valid(col, row),
                    "pixel ({col}, {row}) should be valid"
                );
                let (x, y) = warp.get(col, row);
                let expected_x = col as f32 + 0.5;
                let expected_y = row as f32 + 0.5;
                assert!(
                    (x - expected_x).abs() < 1e-3,
                    "col={col} row={row}: x={x} expected {expected_x}"
                );
                assert!(
                    (y - expected_y).abs() < 1e-3,
                    "col={col} row={row}: y={y} expected {expected_y}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip: undistort then redistort recovers original
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_undistort_redistort() {
        let distorted = simple_radial(64, 48, 100.0, 0.1);
        let undistorted = pinhole(64, 48, 100.0);

        // Forward: distorted → undistorted (undistort map)
        let undistort_map = WarpMap::from_cameras(&distorted, &undistorted);
        // Reverse: undistorted → distorted (redistort map)
        let redistort_map = WarpMap::from_cameras(&undistorted, &distorted);

        // For pixels near the center (avoid boundary issues), composing the
        // two maps should recover roughly the pixel center.
        let margin = 8;
        for row in margin..(undistorted.height - margin) {
            for col in margin..(undistorted.width - margin) {
                if !undistort_map.is_valid(col, row) {
                    continue;
                }
                let (sx, sy) = undistort_map.get(col, row);
                // Look up the redistort map at the (fractional) source pixel.
                // Use nearest-neighbour for simplicity.
                let sc = (sx - 0.5).round() as u32;
                let sr = (sy - 0.5).round() as u32;
                if sc >= redistort_map.width() || sr >= redistort_map.height() {
                    continue;
                }
                if !redistort_map.is_valid(sc, sr) {
                    continue;
                }
                let (rx, ry) = redistort_map.get(sc, sr);
                let expected_x = col as f32 + 0.5;
                let expected_y = row as f32 + 0.5;
                // Tolerance is generous because of nearest-neighbour sampling.
                assert!(
                    (rx - expected_x).abs() < 2.0,
                    "col={col} row={row}: rx={rx} expected ~{expected_x}"
                );
                assert!(
                    (ry - expected_y).abs() < 2.0,
                    "col={col} row={row}: ry={ry} expected ~{expected_y}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Equirectangular target — centre should have valid pixels
    // -----------------------------------------------------------------------

    #[test]
    fn equirectangular_target_center_valid() {
        let fisheye = simple_radial_fisheye(200, 200, 100.0, 0.0);
        let equirect = equirectangular(400, 200);

        let warp = WarpMap::from_cameras(&fisheye, &equirect);

        // The centre of the equirectangular image corresponds to the forward
        // direction, which should definitely map into the fisheye image.
        let cx = warp.width() / 2;
        let cy = warp.height() / 2;
        assert!(
            warp.is_valid(cx, cy),
            "centre of equirectangular target should be valid"
        );

        // Check a small region around centre.
        let margin = 10;
        let mut valid_count = 0u32;
        let total = (2 * margin + 1) * (2 * margin + 1);
        for dr in 0..=(2 * margin) {
            for dc in 0..=(2 * margin) {
                let c = cx - margin + dc;
                let r = cy - margin + dr;
                if warp.is_valid(c, r) {
                    valid_count += 1;
                }
            }
        }
        assert!(
            valid_count == total,
            "expected all {total} centre pixels valid, got {valid_count}"
        );
    }

    // -----------------------------------------------------------------------
    // SVD computation
    // -----------------------------------------------------------------------

    #[test]
    fn compute_svd_runs_without_panic() {
        let cam = pinhole(32, 24, 50.0);
        let mut warp = WarpMap::from_cameras(&cam, &cam);
        assert!(!warp.has_svd());

        warp.compute_svd();
        assert!(warp.has_svd());

        let svd = warp.svd().unwrap();
        let n = 32 * 24;
        assert_eq!(svd.sigma_major.len(), n);
        assert_eq!(svd.sigma_minor.len(), n);
        assert_eq!(svd.major_dir.len(), 2 * n);
    }

    #[test]
    fn identity_svd_values_near_one() {
        let cam = pinhole(32, 24, 50.0);
        let mut warp = WarpMap::from_cameras(&cam, &cam);
        warp.compute_svd();
        let svd = warp.svd().unwrap();

        // Interior pixels (away from boundary) should have singular values
        // very close to 1.0 for an identity map.
        for row in 2..22 {
            for col in 2..30 {
                let idx = row * 32 + col;
                assert!(
                    (svd.sigma_major[idx] - 1.0).abs() < 0.01,
                    "sigma_major at ({col},{row}) = {} expected ~1",
                    svd.sigma_major[idx]
                );
                assert!(
                    (svd.sigma_minor[idx] - 1.0).abs() < 0.01,
                    "sigma_minor at ({col},{row}) = {} expected ~1",
                    svd.sigma_minor[idx]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // is_valid
    // -----------------------------------------------------------------------

    #[test]
    fn is_valid_returns_correct_values() {
        // Build a map where some pixels are intentionally out of bounds.
        // Use a small source image and a large destination with the same
        // intrinsics so that edge pixels in dst land outside src.
        let src = pinhole(32, 24, 50.0);
        let dst = pinhole(64, 48, 50.0);

        let warp = WarpMap::from_cameras(&src, &dst);

        // Centre of dst should map to centre of src and be valid.
        assert!(warp.is_valid(32, 24));

        // Far corners of dst should map outside the small src and be invalid.
        // The dst is 64x48 with cx=32, cy=24 and src is 32x24 with cx=16, cy=12.
        // dst pixel (0, 0) → unproject → project → will be at src pixel
        // (0 + 0.5 - 32) * 50/50 + 16 = -15.5 which is < 0 → invalid.
        assert!(
            !warp.is_valid(0, 0),
            "corner pixel (0,0) should be out of source bounds"
        );
        assert!(
            !warp.is_valid(63, 47),
            "corner pixel (63,47) should be out of source bounds"
        );
    }

    // -----------------------------------------------------------------------
    // SVD of scaled map
    // -----------------------------------------------------------------------

    #[test]
    fn svd_of_scaled_map() {
        // A 2x zoom should produce singular values near 2.0.
        // Use a large source so that dst pixels don't go out of bounds.
        let src = pinhole(128, 96, 100.0);
        let dst = pinhole(64, 48, 50.0); // half the focal length → 2x zoom out

        let mut warp = WarpMap::from_cameras(&src, &dst);
        warp.compute_svd();
        let svd = warp.svd().unwrap();

        // Check interior (skip boundary pixels where central differences
        // fall back to identity).
        for row in 2..46 {
            for col in 2..62 {
                let idx = row * 64 + col;
                assert!(
                    (svd.sigma_major[idx] - 2.0).abs() < 0.1,
                    "sigma_major at ({col},{row}) = {} expected ~2",
                    svd.sigma_major[idx]
                );
                assert!(
                    (svd.sigma_minor[idx] - 2.0).abs() < 0.1,
                    "sigma_minor at ({col},{row}) = {} expected ~2",
                    svd.sigma_minor[idx]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2x2 SVD helper
    // -----------------------------------------------------------------------

    #[test]
    fn svd_2x2_identity() {
        let (s1, s2, dx, dy) = svd_2x2(1.0, 0.0, 0.0, 1.0);
        assert!((s1 - 1.0).abs() < 1e-6);
        assert!((s2 - 1.0).abs() < 1e-6);
        // Direction can be either (1,0) or (0,1) since both singular values
        // are equal — just check it's unit length.
        assert!((dx * dx + dy * dy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn svd_2x2_diagonal() {
        let (s1, s2, _dx, _dy) = svd_2x2(3.0, 0.0, 0.0, 2.0);
        assert!((s1 - 3.0).abs() < 1e-5, "s1 = {s1}");
        assert!((s2 - 2.0).abs() < 1e-5, "s2 = {s2}");
    }

    #[test]
    fn svd_2x2_rotation() {
        // Pure rotation should have both singular values = 1.
        let angle = 0.7_f32;
        let (s1, s2, _, _) = svd_2x2(angle.cos(), -angle.sin(), angle.sin(), angle.cos());
        assert!((s1 - 1.0).abs() < 1e-5, "s1 = {s1}");
        assert!((s2 - 1.0).abs() < 1e-5, "s2 = {s2}");
    }

    // -----------------------------------------------------------------------
    // Pose-aware constructors
    // -----------------------------------------------------------------------

    /// Compare two warp maps pixel-by-pixel with a per-pixel tolerance.
    ///
    /// Both maps are required to agree on which pixels are valid (NaN vs not).
    fn assert_maps_equal(a: &WarpMap, b: &WarpMap, tol: f32) {
        assert_eq!(a.width(), b.width());
        assert_eq!(a.height(), b.height());
        for row in 0..a.height() {
            for col in 0..a.width() {
                let va = a.is_valid(col, row);
                let vb = b.is_valid(col, row);
                assert_eq!(va, vb, "validity mismatch at ({col}, {row})");
                if !va {
                    continue;
                }
                let (ax, ay) = a.get(col, row);
                let (bx, by) = b.get(col, row);
                let err = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                assert!(
                    err < tol,
                    "mismatch at ({col},{row}): ({ax},{ay}) vs ({bx},{by}), err={err}"
                );
            }
        }
    }

    #[test]
    fn from_cameras_with_rotation_identity_matches_from_cameras() {
        // rotation-aware with identity rotation must equal the baseline.
        let src = pinhole(64, 48, 100.0);
        let dst = pinhole(64, 48, 100.0);

        let baseline = WarpMap::from_cameras(&src, &dst);
        let rotated = WarpMap::from_cameras_with_rotation(&src, &dst, &RotQuaternion::identity());

        assert_maps_equal(&baseline, &rotated, 1e-4);
    }

    #[test]
    fn from_cameras_with_rotation_identity_matches_from_cameras_equirect() {
        // Same invariance must hold for equirectangular / fisheye, which go
        // through the ray-based path.
        let src = simple_radial_fisheye(200, 200, 100.0, 0.0);
        let dst = equirectangular(400, 200);

        let baseline = WarpMap::from_cameras(&src, &dst);
        let rotated = WarpMap::from_cameras_with_rotation(&src, &dst, &RotQuaternion::identity());

        assert_maps_equal(&baseline, &rotated, 1e-3);
    }

    #[test]
    fn from_cameras_with_pose_infinity_matches_rotation_only() {
        // With depth = +INF the translation drops out of the formulation.
        let src = pinhole(64, 48, 100.0);
        let dst = pinhole(64, 48, 100.0);

        let rot = RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.2).unwrap();
        let src_from_world = RigidTransform::new(rot.clone(), Vector3::new(0.5, -0.2, 3.0));
        let dst_from_world =
            RigidTransform::new(RotQuaternion::identity(), Vector3::new(-0.1, 0.3, 2.5));
        // Relative rotation from dst-frame to src-frame: R_sw * R_dw^T.
        let r_sd_mat =
            src_from_world.to_rotation_matrix() * dst_from_world.to_rotation_matrix().transpose();
        let r_sd = RotQuaternion::from_rotation_matrix(r_sd_mat);

        let pose_inf = WarpMap::from_cameras_with_pose(
            &src,
            &dst,
            &src_from_world,
            &dst_from_world,
            f64::INFINITY,
        );
        let rot_only = WarpMap::from_cameras_with_rotation(&src, &dst, &r_sd);
        assert_maps_equal(&pose_inf, &rot_only, 1e-3);
    }

    #[test]
    fn from_cameras_with_pose_coincident_pose_matches_from_cameras() {
        // If src and dst cameras sit at the same world pose, the pose
        // construction must agree with from_cameras (scene points on the
        // radius-r sphere around dst land in the same place in src).
        let src = pinhole(64, 48, 100.0);
        let dst = pinhole(64, 48, 100.0);

        let pose = RigidTransform::new(
            RotQuaternion::from_axis_angle(Vector3::new(1.0, 0.5, 0.2), 0.4).unwrap(),
            Vector3::new(2.0, -1.0, 0.5),
        );

        let baseline = WarpMap::from_cameras(&src, &dst);
        let posed = WarpMap::from_cameras_with_pose(&src, &dst, &pose, &pose, 5.0);
        assert_maps_equal(&baseline, &posed, 1e-2);
    }

    #[test]
    fn from_cameras_with_pose_known_depth_synthetic_sphere() {
        // Synthetic "scene" = a sphere of radius R centered at dst camera.
        // Every dst pixel center traces a ray hitting a point on that sphere.
        // When we transform that point into src coordinates and project, we
        // should get a pixel which is in-bounds and matches the warp exactly.
        let src = pinhole(160, 120, 200.0);
        let dst = pinhole(160, 120, 200.0);

        let src_from_world = RigidTransform::new(
            RotQuaternion::from_axis_angle(Vector3::new(0.1, 1.0, 0.05), 0.15).unwrap(),
            Vector3::new(0.5, 0.0, 0.0),
        );
        let dst_from_world = RigidTransform::new(
            RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.1), -0.05).unwrap(),
            Vector3::new(0.0, 0.0, 0.0),
        );
        let radius = 10.0_f64;

        let warp =
            WarpMap::from_cameras_with_pose(&src, &dst, &src_from_world, &dst_from_world, radius);

        // Precompute R_sd and T_sd just like the implementation does.
        let r_sw = src_from_world.to_rotation_matrix();
        let r_dw = dst_from_world.to_rotation_matrix();
        let r_sd = r_sw * r_dw.transpose();
        let t_sd = src_from_world.translation - r_sd * dst_from_world.translation;

        // Spot-check the middle 80% of pixels — edges may miss src bounds.
        let w = warp.width();
        let h = warp.height();
        let mut checked = 0usize;
        for row in (h / 10)..(9 * h / 10) {
            for col in (w / 10)..(9 * w / 10) {
                if !warp.is_valid(col, row) {
                    continue;
                }
                let u = col as f64 + 0.5;
                let v = row as f64 + 0.5;
                let d_dst = dst.pixel_to_ray(u, v);
                let d_dst_vec = Vector3::new(d_dst[0], d_dst[1], d_dst[2]);
                let p_dst = radius * d_dst_vec;
                let p_src = r_sd * p_dst + t_sd;
                let (exp_x, exp_y) = src.ray_to_pixel([p_src.x, p_src.y, p_src.z]).unwrap();

                let (gx, gy) = warp.get(col, row);
                assert!(
                    (gx as f64 - exp_x).abs() < 1e-3,
                    "x mismatch at ({col},{row}): got {gx}, expected {exp_x}"
                );
                assert!(
                    (gy as f64 - exp_y).abs() < 1e-3,
                    "y mismatch at ({col},{row}): got {gy}, expected {exp_y}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "too few pixels in-bounds: {checked}");
    }

    #[test]
    fn from_cameras_with_pose_baseline_comparable_to_depth() {
        // Spec: at r ≈ B_max, a small-angle approximation (rotation-only,
        // ignoring translation) errs by noticeable amounts. The exact ray
        // formula must still be pixel-accurate. We construct a synthetic
        // scene on a sphere and verify the two formulations disagree.

        let cam = pinhole(320, 240, 300.0);
        // Baseline = 0.3, depth = 1.0: comparable scale, but mild enough that
        // most pixels still project inside the image. The small-angle approx
        // (rotation-only) disagrees with the exact formulation by many
        // pixels under these conditions.
        let src_from_world =
            RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.3, 0.0, 0.0));
        let dst_from_world =
            RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let radius = 1.0_f64;

        let exact =
            WarpMap::from_cameras_with_pose(&cam, &cam, &src_from_world, &dst_from_world, radius);
        // The rotation-only (infinite-depth) approximation ignores the
        // translation entirely — it should be obviously wrong here.
        let approx = WarpMap::from_cameras_with_rotation(&cam, &cam, &RotQuaternion::identity());

        // Find the maximum error between the two maps across valid pixels.
        let mut max_err_sq = 0.0_f32;
        let mut counted = 0usize;
        for row in (cam.height / 4)..(3 * cam.height / 4) {
            for col in (cam.width / 4)..(3 * cam.width / 4) {
                if !exact.is_valid(col, row) || !approx.is_valid(col, row) {
                    continue;
                }
                let (ax, ay) = exact.get(col, row);
                let (bx, by) = approx.get(col, row);
                let err_sq = (ax - bx).powi(2) + (ay - by).powi(2);
                if err_sq > max_err_sq {
                    max_err_sq = err_sq;
                }
                counted += 1;
            }
        }
        assert!(counted > 100);
        let max_err = max_err_sq.sqrt();
        // Sanity: the approximation should be off by many pixels. If the two
        // agreed to <0.1 px we'd know one of the two was silently ignoring
        // its inputs.
        assert!(
            max_err > 5.0,
            "rotation-only approximation should disagree with exact; got max_err={max_err} px",
        );

        // Cross-check the exact map against a hand-computed reprojection at
        // the centre pixel. src_from_world.translation = (0.3, 0, 0) means
        // src camera sits at world (-0.3, 0, 0). Ray from dst centre is
        // (0, 0, 1); at depth 1 it lands at world point (0, 0, 1). In src
        // frame: p_src = R_sw * p_w + t_sw = (0, 0, 1) + (0.3, 0, 0)
        // = (0.3, 0, 1), which projects to pixel (cx + 0.3*f, cy) =
        // (160 + 90, 120) = (250, 120).
        let cx = cam.width / 2;
        let cy = cam.height / 2;
        assert!(exact.is_valid(cx, cy));
        let (gx, gy) = exact.get(cx, cy);
        // Allow ±1 px since (cx, cy) is the integer pixel index and the
        // ray traces from the pixel centre at (cx + 0.5, cy + 0.5).
        assert!(
            (gx - 250.5).abs() < 1.0,
            "centre sx = {gx}, expected ~250.5"
        );
        assert!(
            (gy - 120.5).abs() < 1.0,
            "centre sy = {gy}, expected ~120.5"
        );

        // The rotation-only approximation at the same pixel is the identity
        // map: pixel centre (160.5, 120.5).
        let (ax, ay) = approx.get(cx, cy);
        assert!((ax - 160.5).abs() < 0.5);
        assert!((ay - 120.5).abs() < 0.5);
        let center_err = ((gx - ax).powi(2) + (gy - ay).powi(2)).sqrt();
        assert!(
            center_err > 50.0,
            "rotation-only vs exact must disagree by many pixels; got {center_err}"
        );
    }

    #[test]
    fn from_cameras_with_pose_svd_still_works() {
        // compute_svd must work identically on a pose-built map.
        let src = pinhole(64, 48, 100.0);
        let dst = pinhole(64, 48, 100.0);
        let src_from_world = RigidTransform::new(
            RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.05).unwrap(),
            Vector3::new(0.02, 0.0, 0.0),
        );
        let dst_from_world = RigidTransform::identity();
        let mut warp =
            WarpMap::from_cameras_with_pose(&src, &dst, &src_from_world, &dst_from_world, 100.0);
        assert!(!warp.has_svd());
        warp.compute_svd();
        assert!(warp.has_svd());

        // Singular values must be finite and positive somewhere away from
        // the boundary.
        let svd = warp.svd().unwrap();
        let idx = (24 * 64 + 32) as usize;
        assert!(svd.sigma_major[idx].is_finite() && svd.sigma_major[idx] > 0.0);
        assert!(svd.sigma_minor[idx].is_finite() && svd.sigma_minor[idx] > 0.0);
    }

    #[test]
    fn from_cameras_with_pose_equirect_dst() {
        // Equirectangular dst: every ray is valid and all pixels with a
        // proper src projection should be in-bounds.
        let src = pinhole(200, 200, 100.0);
        let dst = equirectangular(400, 200);
        let pose = RigidTransform::identity();

        let warp = WarpMap::from_cameras_with_pose(&src, &dst, &pose, &pose, 1e6);

        // Centre of equirect (forward) must be valid in src.
        let cx = warp.width() / 2;
        let cy = warp.height() / 2;
        assert!(warp.is_valid(cx, cy));
    }
}
