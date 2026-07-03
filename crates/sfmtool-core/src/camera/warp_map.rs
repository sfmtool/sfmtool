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

use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;
use crate::geometry::RotQuaternion;
use crate::patch::cloud::OrientedPatch;

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
    /// Optional raw per-pixel 2×2 Jacobian `J = [[dx/dcol, dx/drow], [dy/dcol,
    /// dy/drow]]`. Populated by [`WarpMap::compute_svd`] (as a free
    /// by-product of the SVD's central differences) and also by
    /// [`WarpMap::compute_jacobians`] (when the caller wants `J` without
    /// paying for the SVD, e.g. the bilinear photometric refiner path).
    /// Length = 4 * width * height, interleaved
    /// `[a, b, c, d]` per pixel (`a = dx/dcol`, `b = dx/drow`,
    /// `c = dy/dcol`, `d = dy/drow`).
    jacobians: Option<Vec<f32>>,
}

/// Destination-pixel count at or below which the warp / remap row loops run
/// sequentially. For small destinations (e.g. the `R×R` patch grids of
/// patch-normal refinement, typically nested inside an already-parallel
/// per-patch loop) the rayon scaffolding costs an order of magnitude more
/// than the row work itself; full-image warps stay parallel.
pub(crate) const PAR_MIN_PIXELS: usize = 2048;

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
            jacobians: None,
        }
    }

    /// Build a warp map that maps every pixel in `dst_camera` to its
    /// corresponding location in `src_camera`.
    ///
    /// For each destination pixel center `(col + 0.5, row + 0.5)`:
    /// - If either camera [`needs_ray_path`](crate::camera::CameraModel::needs_ray_path)
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
            jacobians: None,
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
            jacobians: None,
        }
    }

    /// Build a `resolution × resolution` warp map sampling `camera`'s image over
    /// an oriented 3D `patch`.
    ///
    /// The destination is the patch's canonical `(s, t) ∈ [-1, 1]²` grid (pixel
    /// centers at `(col + 0.5, row + 0.5)`), with `col` stepping along `+u_axis`
    /// and `row` along `−v_axis` — the row index counts downward, so the frame's
    /// `v_axis` ("up") is reversed to render the front face un-mirrored (see
    /// [`OrientedPatch`]). Each entry is the source-image `(x, y)` where that
    /// patch pixel projects: the homogeneous corner is mapped to the camera frame
    /// via `cam_from_world` and projected with `ray_to_pixel`, so all camera models
    /// (including distortion / fisheye) are handled. This works for a finite
    /// patch (`w = 1`, a planar surfel) and a point at infinity (`w = 0`, a
    /// region of the sphere of directions — the corner is a direction, rotated
    /// without translation, then projected as a ray). Pixels are `(NaN, NaN)`
    /// when behind the camera, outside the model domain, or outside the image
    /// bounds — matching the other constructors.
    ///
    /// Generalizes [`Self::from_cameras_with_pose`] from a fronto-parallel depth
    /// plane to an arbitrary oriented plane. See `specs/core/patch-cloud.md`.
    pub fn from_patch(
        patch: &OrientedPatch,
        camera: &CameraIntrinsics,
        cam_from_world: &RigidTransform,
        resolution: u32,
    ) -> Self {
        let r = resolution.max(1);

        // Stage 1 — geometry (no camera model). The patch → camera-frame map is
        // affine in `(s, t)`: `Q(s, t) = q0 + s·qu + t·qv`. A finite patch
        // (`w = 1`) gets `R·x + t`; a point at infinity (`w = 0`) is rotated only
        // — the weight is folded into `q0` exactly as
        // [`RigidTransform::transform_point_homogeneous`] would. Building the
        // affine basis once replaces the per-pixel corner build + pose multiply.
        let rot = cam_from_world.rotation.to_rotation_matrix();
        let q0 = rot * patch.center.coords + cam_from_world.translation * patch.w;
        let qu = (rot * patch.u_axis) * patch.half_extent[0];
        // The row index increases *downward*, so we step along `−v_axis`: the
        // patch frame is right-handed (outward normal `u × v`), and walking
        // `−v` for the raster row is what renders the front face un-mirrored —
        // walking `+v` would bake a mirror image. See `OrientedPatch`.
        let qv = (rot * patch.v_axis) * (-patch.half_extent[1]);

        // Re-express on the integer `(col, row)` grid: pixel centers sit at
        // `s = (col + 0.5)·step − 1`, `step = 2/r`, so the ray at `(col, row)` is
        // `origin + col·col_step + row·row_step`.
        let step = 2.0 / r as f64;
        let s0 = 0.5 * step - 1.0;
        let origin = q0 + (qu + qv) * s0;
        let col_step = qu * step;
        let row_step = qv * step;

        // Stage 2 — projection (camera-owned; exact for perspective, bounded
        // coarse-grid interpolation for fisheye/equirectangular). Invalid pixels
        // come back as `(NaN, NaN)`, matching the other constructors.
        let mut data = vec![0.0f32; 2 * r as usize * r as usize];
        camera.ray_to_pixel_grid(
            [origin.x, origin.y, origin.z],
            [col_step.x, col_step.y, col_step.z],
            [row_step.x, row_step.y, row_step.z],
            r,
            r,
            &mut data,
        );

        WarpMap {
            width: r,
            height: r,
            data,
            svd: None,
            jacobians: None,
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
    /// differences, and store the result internally. Also stores the raw
    /// per-pixel Jacobian (as a by-product of the same central differences) so
    /// [`get_jacobian`](Self::get_jacobian) is callable afterwards without an
    /// extra pass.
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
    /// At image boundaries the gradient falls back to a one-sided central
    /// difference (so a non-identity warp still gets its actual scale at the
    /// frame edge, not a wrong identity); where any required neighbour is NaN,
    /// the identity Jacobian is used (`sigma_major = sigma_minor = 1`,
    /// `major_dir = (1, 0)`) — see [`jacobian_at`](Self::jacobian_at). Idempotent:
    /// a second call is a no-op (mirrors
    /// [`compute_jacobians`](Self::compute_jacobians)). The guard checks `svd`
    /// only because `compute_svd` always populates both `svd` and `jacobians`
    /// in one pass — the two stay in lockstep on this path.
    pub fn compute_svd(&mut self) {
        if self.svd.is_some() {
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;

        let per_pixel = |col: usize, row: usize| {
            let j = self.jacobian_at(col, row, w, h);
            let (s_maj, s_min, dx, dy) = svd_2x2(j[0], j[1], j[2], j[3]);
            (j, s_maj, s_min, dx, dy)
        };

        let (sigma_major, sigma_minor, major_dir, jacobians) = if n <= PAR_MIN_PIXELS {
            let mut sigma_major = Vec::with_capacity(n);
            let mut sigma_minor = Vec::with_capacity(n);
            let mut major_dir = Vec::with_capacity(2 * n);
            let mut jacobians = Vec::with_capacity(4 * n);
            for row in 0..h {
                for col in 0..w {
                    let (j, s_maj, s_min, dx, dy) = per_pixel(col, row);
                    sigma_major.push(s_maj);
                    sigma_minor.push(s_min);
                    major_dir.push(dx);
                    major_dir.push(dy);
                    jacobians.extend_from_slice(&j);
                }
            }
            (sigma_major, sigma_minor, major_dir, jacobians)
        } else {
            (0..h)
                .into_par_iter()
                .map(|row| {
                    let mut row_major = Vec::with_capacity(w);
                    let mut row_minor = Vec::with_capacity(w);
                    let mut row_dir = Vec::with_capacity(2 * w);
                    let mut row_jac = Vec::with_capacity(4 * w);

                    for col in 0..w {
                        let (j, s_maj, s_min, dx, dy) = per_pixel(col, row);
                        row_major.push(s_maj);
                        row_minor.push(s_min);
                        row_dir.push(dx);
                        row_dir.push(dy);
                        row_jac.extend_from_slice(&j);
                    }
                    (row_major, row_minor, row_dir, row_jac)
                })
                .reduce(
                    || {
                        (
                            Vec::with_capacity(n),
                            Vec::with_capacity(n),
                            Vec::with_capacity(2 * n),
                            Vec::with_capacity(4 * n),
                        )
                    },
                    |mut acc, row| {
                        acc.0.extend(row.0);
                        acc.1.extend(row.1);
                        acc.2.extend(row.2);
                        acc.3.extend(row.3);
                        acc
                    },
                )
        };

        self.svd = Some(WarpMapSvd {
            sigma_major,
            sigma_minor,
            major_dir,
        });
        self.jacobians = Some(jacobians);
    }

    /// Compute and store the raw per-pixel 2×2 Jacobian without the SVD.
    /// Cheaper than [`compute_svd`] when the caller only needs `J`
    /// (e.g. the bilinear photometric subpixel refiner path).
    ///
    /// Same central-difference scheme as `compute_svd`: at image boundaries or
    /// where any neighbour is NaN, the identity Jacobian is stored.
    pub fn compute_jacobians(&mut self) {
        if self.jacobians.is_some() {
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;

        let jacobians: Vec<f32> = if n <= PAR_MIN_PIXELS {
            let mut j = Vec::with_capacity(4 * n);
            for row in 0..h {
                for col in 0..w {
                    j.extend_from_slice(&self.jacobian_at(col, row, w, h));
                }
            }
            j
        } else {
            (0..h)
                .into_par_iter()
                .flat_map(|row| {
                    let mut row_j = Vec::with_capacity(4 * w);
                    for col in 0..w {
                        row_j.extend_from_slice(&self.jacobian_at(col, row, w, h));
                    }
                    row_j
                })
                .collect()
        };

        self.jacobians = Some(jacobians);
    }

    /// Returns `true` if [`compute_jacobians`](Self::compute_jacobians) (or
    /// [`compute_svd`](Self::compute_svd), which populates it as a by-product)
    /// has been called.
    pub fn has_jacobians(&self) -> bool {
        self.jacobians.is_some()
    }

    /// Look up the raw per-pixel 2×2 Jacobian at a single pixel:
    /// `[[dx/dcol, dx/drow], [dy/dcol, dy/drow]]`.
    ///
    /// Composing `∇_src I · J` gives `∂I/∂δ`, the image Jacobian against
    /// patch-grid displacements `δ = (δ_col, δ_row)` — the photometric subpixel
    /// refiner's GN inner step.
    ///
    /// # Panics
    ///
    /// Panics if neither [`compute_jacobians`](Self::compute_jacobians) nor
    /// [`compute_svd`](Self::compute_svd) has been called.
    pub fn get_jacobian(&self, col: u32, row: u32) -> [[f32; 2]; 2] {
        let jac = self
            .jacobians
            .as_ref()
            .expect("Jacobians not computed; call compute_jacobians() or compute_svd() first");
        let idx = 4 * (row as usize * self.width as usize + col as usize);
        [[jac[idx], jac[idx + 1]], [jac[idx + 2], jac[idx + 3]]]
    }

    /// Estimate the 2×2 Jacobian at a single pixel from the warp coords.
    /// Returns `[a, b, c, d]` with `a = dx/dcol`, `b = dx/drow`, `c = dy/dcol`,
    /// `d = dy/drow`.
    ///
    /// Uses central differences in the interior; **one-sided** (forward at col/row 0,
    /// backward at col/row w-1/h-1) at the image boundary so a non-identity warp
    /// still gets its actual scale at the frame edge — the previous identity-Jacobian
    /// fallback there silently biased downstream uses (e.g. the photometric
    /// subpixel refiner's GN normal equations) on the boundary ring of support
    /// pixels with non-zero window weight. Only when an axis collapses
    /// (`w < 2` or `h < 2`, no neighbour at all on that axis) or any required
    /// neighbour is NaN does it fall back to the identity Jacobian.
    fn jacobian_at(&self, col: usize, row: usize, w: usize, h: usize) -> [f32; 4] {
        let identity = [1.0f32, 0.0, 0.0, 1.0];

        // An axis with width/height < 2 has no neighbour to difference against —
        // fall back to identity wholesale for that case.
        if w < 2 || h < 2 {
            return identity;
        }

        let idx = |c: usize, r: usize| -> usize { 2 * (r * w + c) };

        // Per-axis difference: central in the interior, one-sided at the boundary.
        // (denominator: 2 for the central case, 1 for the one-sided case.)
        let (xl_col, xr_col, col_denom) = if col == 0 {
            (col, col + 1, 1.0)
        } else if col == w - 1 {
            (col - 1, col, 1.0)
        } else {
            (col - 1, col + 1, 2.0)
        };
        let (xt_row, xb_row, row_denom) = if row == 0 {
            (row, row + 1, 1.0)
        } else if row == h - 1 {
            (row - 1, row, 1.0)
        } else {
            (row - 1, row + 1, 2.0)
        };

        let xl = self.data[idx(xl_col, row)];
        let xr = self.data[idx(xr_col, row)];
        let yl = self.data[idx(xl_col, row) + 1];
        let yr = self.data[idx(xr_col, row) + 1];

        let xt = self.data[idx(col, xt_row)];
        let xb = self.data[idx(col, xb_row)];
        let yt = self.data[idx(col, xt_row) + 1];
        let yb = self.data[idx(col, xb_row) + 1];

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

        let a = (xr - xl) / col_denom; // dx/dcol
        let b = (xb - xt) / row_denom; // dx/drow
        let c = (yr - yl) / col_denom; // dy/dcol
        let d = (yb - yt) / row_denom; // dy/drow
        [a, b, c, d]
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
