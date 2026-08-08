// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Undistorted-pinhole approximation of a distorted camera.
//!
//! Derives the pinhole intrinsics whose field of view is fully backed by
//! source pixels ([`CameraIntrinsics::best_fit_inside_pinhole`]) or fully
//! contains them ([`CameraIntrinsics::best_fit_outside_pinhole`]), by
//! binary search over the focal length against the distorted image
//! boundary. Not a distortion model itself — it is what callers undistort
//! *into*.

use crate::camera::{CameraIntrinsics, CameraIntrinsicsError, CameraModel};

impl CameraIntrinsics {
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
    pub(super) fn boundary_samples(width: u32, height: u32) -> Vec<(f64, f64)> {
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
