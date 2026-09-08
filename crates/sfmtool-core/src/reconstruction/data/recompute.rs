// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Derived quantities recomputed from geometry for [`SfmrReconstruction`].
//!
//! Everything here reads observations back through the camera model and writes
//! the result onto the reconstruction: per-image observation errors, the
//! per-point error refresh for finite and infinity points, and the depth
//! statistics/histograms (which also fill in missing point normals). Split out
//! of [`super`] so the type definitions are not interleaved with this algebra.
//!
//! Named `recompute` rather than `errors` on purpose: the error *type*,
//! [`ReconstructionError`], is part of the data model and stays in [`super`].
//! `rebuild_derived_fields` is the one recompute that stays there too — it is
//! pure bookkeeping over the track arrays with no camera model involved.

use nalgebra::{Point3, UnitQuaternion, Vector3};

use sfmr_format::SfmrError;

use crate::camera::CameraIntrinsics;

use super::{ReconstructionError, SfmrReconstruction};

impl SfmrReconstruction {
    /// Compute per-observation reprojection errors for a single image.
    ///
    /// Projects each observed 3D point through the camera and measures pixel
    /// distance to where that observation says it sits. Points at infinity
    /// (`w = 0`) project their stored bearing direction (translation-free), so
    /// their error is well-defined like any finite point's.
    ///
    /// The observed pixel comes from the reconstruction's inline `keypoints_xy`
    /// when it carries one, and otherwise from the image's `.sift` file -- so a
    /// `sift_files` reconstruction with the optional inline copy needs no
    /// `.sift` companion on disk.
    ///
    /// Returns a vector of `(feature_index, reprojection_error_px)` pairs,
    /// one per track observation for this image. Points behind the camera
    /// produce `f32::NAN`.
    pub fn compute_observation_reprojection_errors(
        &self,
        image_index: usize,
    ) -> Result<Vec<(u32, f32)>, ReconstructionError> {
        let image = &self.image_table.images[image_index];
        let camera = &self.image_table.cameras[image.camera_index as usize];

        let inline = self.keypoints_xy();
        // Feature positions from the image's `.sift` file, read only when there
        // is no inline column to answer from.
        let positions = match inline {
            Some(_) => Vec::new(),
            None => {
                let read_count = self.point_set.max_track_feature_index[image_index] as usize + 1;
                let sift_path = self.sift_path_for_image(image_index);
                sift_format::read_sift_positions(&sift_path, read_count).map_err(|e| {
                    ReconstructionError::SiftRead {
                        path: sift_path,
                        source: e.to_string(),
                    }
                })?
            }
        };

        // World-to-camera rotation (the stored unit quaternion rotates the point
        // into the camera frame directly, no matrix needed).
        let r = &image.quaternion_wxyz;
        let t = &image.translation_xyz;

        // Iterate all track observations for this image
        let feat_to_point = &self.point_set.image_feature_to_point[image_index];
        let mut results = Vec::with_capacity(feat_to_point.len());

        for (&feat_idx, &point_idx) in feat_to_point {
            let observed = match inline {
                Some(keypoints_xy) => {
                    match self.observation_row(image_index, point_idx, feat_idx) {
                        Some(row) => [keypoints_xy[[row, 0]] as f64, keypoints_xy[[row, 1]] as f64],
                        None => {
                            results.push((feat_idx, f32::NAN));
                            continue;
                        }
                    }
                }
                None => match positions.get(feat_idx as usize) {
                    Some(&xy) => [xy[0] as f64, xy[1] as f64],
                    None => {
                        results.push((feat_idx, f32::NAN));
                        continue;
                    }
                },
            };

            let point = &self.point_set.points[point_idx as usize];
            let error = match observation_reprojection_error(
                r,
                t,
                camera,
                &point.position,
                point.is_at_infinity(),
                observed,
            ) {
                Some(e) => e as f32,
                // Point behind the camera: no defined reprojection.
                None => f32::NAN,
            };

            results.push((feat_idx, error));
        }

        Ok(results)
    }

    /// Mean reprojection error (px) per point from the reconstruction's inline
    /// `keypoints_xy` -- one entry per point, parallel to `points`. Each
    /// observation reprojects its point through the observing camera and
    /// measures pixel distance to the inline keypoint; points at infinity
    /// project their bearing (translation-free). A point with no valid
    /// (in-front) observation gets `0.0`.
    ///
    /// Returns `None` when the reconstruction carries no inline column -- a
    /// `sift_files` one without the optional copy, whose 2D observations live in
    /// external `.sift` files and must be read via
    /// [`Self::compute_observation_reprojection_errors`] instead.
    fn inline_point_reprojection_errors(&self) -> Option<Vec<f32>> {
        let keypoints_xy = self.keypoints_xy()?;
        let num_points = self.point_set.points.len();
        let mut out = vec![0.0f32; num_points];
        for (point_idx, slot) in out.iter_mut().enumerate() {
            let point = &self.point_set.points[point_idx];
            let at_infinity = point.is_at_infinity();
            let start = self.point_set.observation_offsets[point_idx];
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for (k, obs) in self.observations_for_point(point_idx).iter().enumerate() {
                let image = &self.image_table.images[obs.image_index as usize];
                let camera = &self.image_table.cameras[image.camera_index as usize];
                let obs_global = start + k;
                let observed = [
                    keypoints_xy[[obs_global, 0]] as f64,
                    keypoints_xy[[obs_global, 1]] as f64,
                ];
                if let Some(e) = observation_reprojection_error(
                    &image.quaternion_wxyz,
                    &image.translation_xyz,
                    camera,
                    &point.position,
                    at_infinity,
                    observed,
                ) {
                    sum += e;
                    count += 1;
                }
            }
            if count > 0 {
                *slot = (sum / count as f64) as f32;
            }
        }
        Some(out)
    }

    /// Recompute per-point mean reprojection errors from scratch.
    ///
    /// For each image, loads feature positions from the `.sift` file and
    /// reprojects all observed 3D points through the camera model. Each
    /// point's `error` is set to the mean pixel-space reprojection error
    /// across all its observations. Points with no valid observations
    /// (e.g., all behind camera) get `error = 0.0`.
    ///
    /// This replaces any errors read from COLMAP/GLOMAP binary files,
    /// which may use different coordinate conventions (GLOMAP stores
    /// errors in normalized image coordinates, not pixels).
    ///
    /// A reconstruction carrying inline `keypoints_xy` -- every
    /// `embedded_patches` one, and a `sift_files` one with the optional copy --
    /// uses those directly and reads no `.sift` file.
    pub fn recompute_point_errors(&mut self) -> Result<(), ReconstructionError> {
        if let Some(errors) = self.inline_point_reprojection_errors() {
            for (pt, error) in self.point_set.points.iter_mut().zip(errors) {
                pt.error = error;
            }
            return Ok(());
        }

        let num_points = self.point_set.points.len();
        let mut error_sums = vec![0.0f64; num_points];
        let mut error_counts = vec![0u32; num_points];

        for img_idx in 0..self.image_table.images.len() {
            let results = self.compute_observation_reprojection_errors(img_idx)?;
            let feat_to_point = &self.point_set.image_feature_to_point[img_idx];
            for (feat_idx, error) in results {
                if error.is_nan() {
                    continue;
                }
                if let Some(&point_idx) = feat_to_point.get(&feat_idx) {
                    error_sums[point_idx as usize] += error as f64;
                    error_counts[point_idx as usize] += 1;
                }
            }
        }

        for i in 0..num_points {
            self.point_set.points[i].error = if error_counts[i] > 0 {
                (error_sums[i] / error_counts[i] as f64) as f32
            } else {
                0.0
            };
        }

        Ok(())
    }

    /// Recompute mean reprojection errors for points at infinity only, leaving
    /// finite points' errors untouched.
    ///
    /// Used after bundle adjustment: a point that was materialised to a finite
    /// landmark, refined, then reclassified back to `w = 0` carries an error
    /// describing the landmark, not its bearing. Only those points need fixing,
    /// so finite points keep the errors the solve produced and `.sift` files are
    /// read only for images that observe a point at infinity.
    pub fn recompute_infinity_point_errors(&mut self) -> Result<(), ReconstructionError> {
        let num_points = self.point_set.points.len();
        let is_infinity: Vec<bool> = self
            .point_set
            .points
            .iter()
            .map(|p| p.is_at_infinity())
            .collect();
        if !is_infinity.iter().any(|&b| b) {
            return Ok(());
        }

        // With an inline keypoint column there is nothing to read from disk;
        // recompute the infinity points' errors from it, leaving finite points
        // untouched.
        if let Some(errors) = self.inline_point_reprojection_errors() {
            for (i, &inf) in is_infinity.iter().enumerate() {
                if inf {
                    self.point_set.points[i].error = errors[i];
                }
            }
            return Ok(());
        }

        let mut error_sums = vec![0.0f64; num_points];
        let mut error_counts = vec![0u32; num_points];

        for img_idx in 0..self.image_table.images.len() {
            let feat_to_point = &self.point_set.image_feature_to_point[img_idx];
            // Skip images observing no point at infinity — avoids reading their
            // `.sift` file just to discard every observation.
            if !feat_to_point.values().any(|&p| is_infinity[p as usize]) {
                continue;
            }
            let results = self.compute_observation_reprojection_errors(img_idx)?;
            let feat_to_point = &self.point_set.image_feature_to_point[img_idx];
            for (feat_idx, error) in results {
                if error.is_nan() {
                    continue;
                }
                if let Some(&point_idx) = feat_to_point.get(&feat_idx) {
                    if is_infinity[point_idx as usize] {
                        error_sums[point_idx as usize] += error as f64;
                        error_counts[point_idx as usize] += 1;
                    }
                }
            }
        }

        for i in 0..num_points {
            if is_infinity[i] {
                self.point_set.points[i].error = if error_counts[i] > 0 {
                    (error_sums[i] / error_counts[i] as f64) as f32
                } else {
                    0.0
                };
            }
        }

        Ok(())
    }

    /// Recompute depth statistics, histograms, and estimated normals from the
    /// current poses, points, and tracks. Uses the same
    /// [`sfmr_format::compute_depth_statistics`] function that `.sfmr` file
    /// writing uses.
    pub fn recompute_depth_statistics(&mut self) -> Result<(), SfmrError> {
        use ndarray::{Array1, Array2};

        let image_count = self.image_table.images.len();
        let points3d_count = self.point_set.points.len();
        let observation_count = self.point_set.tracks.len();

        // Build columnar arrays from the reconstruction data
        let mut quaternions_wxyz = Array2::<f64>::zeros((image_count, 4));
        let mut translations_xyz = Array2::<f64>::zeros((image_count, 3));
        for (i, im) in self.image_table.images.iter().enumerate() {
            let q = im.quaternion_wxyz.quaternion();
            quaternions_wxyz[[i, 0]] = q.w;
            quaternions_wxyz[[i, 1]] = q.i;
            quaternions_wxyz[[i, 2]] = q.j;
            quaternions_wxyz[[i, 3]] = q.k;
            translations_xyz[[i, 0]] = im.translation_xyz.x;
            translations_xyz[[i, 1]] = im.translation_xyz.y;
            translations_xyz[[i, 2]] = im.translation_xyz.z;
        }

        let mut positions_xyzw = Array2::<f64>::zeros((points3d_count, 4));
        for (i, pt) in self.point_set.points.iter().enumerate() {
            positions_xyzw[[i, 0]] = pt.position.x;
            positions_xyzw[[i, 1]] = pt.position.y;
            positions_xyzw[[i, 2]] = pt.position.z;
            positions_xyzw[[i, 3]] = pt.w;
        }

        let mut image_indexes = Array1::<u32>::zeros(observation_count);
        let mut point_indexes = Array1::<u32>::zeros(observation_count);
        for (i, obs) in self.point_set.tracks.iter().enumerate() {
            image_indexes[i] = obs.image_index;
            point_indexes[i] = obs.point_index;
        }

        let result = sfmr_format::compute_depth_statistics(
            &quaternions_wxyz,
            &translations_xyz,
            &positions_xyzw,
            &image_indexes,
            &point_indexes,
        )?;

        // Store results back
        self.image_table.depth_statistics = result.depth_statistics;
        let num_buckets = result.observed_depth_histogram_counts.ncols();
        self.image_table.depth_histogram_counts = (0..image_count)
            .map(|i| {
                (0..num_buckets)
                    .map(|j| result.observed_depth_histogram_counts[[i, j]])
                    .collect()
            })
            .collect();
        // Only materialize normals when this reconstruction carries them.
        if self.point_set.has_normals {
            for (i, pt) in self.point_set.points.iter_mut().enumerate() {
                pt.normal = Vector3::new(
                    result.mean_viewing_normals_xyz[[i, 0]],
                    result.mean_viewing_normals_xyz[[i, 1]],
                    result.mean_viewing_normals_xyz[[i, 2]],
                );
            }
        }

        Ok(())
    }
}

/// Reprojection error (pixels) of one observation, or `None` when the camera
/// cannot image the point at all.
///
/// `rotation`/`translation` are the world→camera pose; `point` is the 3D point's
/// stored coordinate. A point at infinity (`at_infinity`) projects its bearing
/// direction through rotation + intrinsics only — the camera translation is
/// negligible at infinity — while a finite point projects `R·p + t`. Shared by
/// per-image error computation and points-at-infinity discovery so the two stay
/// in sync.
///
/// **"Cannot image" is the camera model's decision, not a hard-coded
/// half-space.** The camera-frame point goes through
/// [`CameraIntrinsics::ray_to_pixel`] and that function's `None` is the
/// domain: the cheirality half-space plus the distortion polynomial's
/// invertible branch for the perspective family, the imaged sphere for an
/// equidistant map. This function used to re-implement the perspective
/// decision as a hard-coded `z ≥ 0` rejection and gnomonic projection for
/// every model, so every observation past 90° off axis returned `None` and
/// dropped out of the point's mean — silently biasing the stored error that
/// the points-at-infinity classifier calibrates its angular noise from — and
/// a distorted perspective ray beyond the polynomial's principal branch
/// scored a residual against its ghost projection instead of being rejected.
pub(crate) fn observation_reprojection_error(
    rotation: &UnitQuaternion<f64>,
    translation: &Vector3<f64>,
    camera: &CameraIntrinsics,
    point: &Point3<f64>,
    at_infinity: bool,
    observed_xy: [f64; 2],
) -> Option<f64> {
    let p_cam = if at_infinity {
        rotation * point.coords
    } else {
        rotation * point.coords + translation
    };
    let (u_proj, v_proj) = camera.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])?;
    let du = u_proj - observed_xy[0];
    let dv = v_proj - observed_xy[1];
    Some((du * du + dv * dv).sqrt())
}
