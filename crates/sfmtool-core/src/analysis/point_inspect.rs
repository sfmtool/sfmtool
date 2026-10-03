// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-point triangulation inspection.
//!
//! Re-derives a 3D point's observation rays by un-projecting its observed
//! pixels (the inline keypoints, or the workspace `.sift` files), runs the
//! triangulation and its observability diagnostics (condition number,
//! inverse-depth z, resolvable distance) over them, and reports them beside the
//! verdict of the point-or-bearing test that reclassification, discovery and the
//! bench decide with. Backs `sfm inspect pt3d_*`.

use nalgebra::{Point3, Vector3};

use crate::analysis::infinity::camera_extents;
use crate::reconstruction::triangulation::{
    depth_uncertainty_batch, is_finite, triangulate_batch, BearingScore, PointBearingFit,
    PointBearingFitOptions, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
};
use crate::reconstruction::{ReconstructionError, SfmrReconstruction};

/// One observation in a point's track, with where it lands in its image.
pub struct ObservationInspection {
    pub image_index: usize,
    pub image_name: String,
    /// The `.sift` feature the observation is, for a `sift_files`
    /// reconstruction; `None` for an `embedded_patches` one, which keeps the
    /// keypoint itself.
    pub feature_index: Option<u32>,
    /// Angle of the un-projected ray off the camera optical axis (degrees).
    /// Large values (→ 90°) sit near the fisheye edge, where the un-projection
    /// is least reliable.
    pub incidence_deg: f64,
}

/// Full triangulation analysis of one 3D point.
pub struct PointInspection {
    /// Stored homogeneous kind: `1.0` finite, `0.0` at infinity.
    pub w: f64,
    /// Stored Euclidean position (finite) or unit direction (at infinity).
    pub position: Point3<f64>,
    pub error: f32,
    pub color: [u8; 3],
    pub observations: Vec<ObservationInspection>,
    /// The point-or-bearing test on the point, at the reconstruction's
    /// measured noise level; `None` when no level can be measured or fewer
    /// than two of its observations give a usable ray.
    pub point_or_bearing: Option<PointOrBearingVerdict>,
    /// Least-squares point from the re-derived rays.
    pub triangulated_point: Point3<f64>,
    pub eigenvalues: [f64; 3],
    pub condition_number: f64,
    pub in_front: bool,
    pub depth: f64,
    pub sigma: f64,
    pub inverse_depth_z: f64,
    pub resolvable_distance: f64,
    /// Camera extents — the `finite_horizon` the resolvable distance is read
    /// against.
    pub finite_horizon: f64,
    /// Bounding-box diagonal of the *observing* camera centers.
    pub baseline_span: f64,
    /// Largest angle (degrees) of any observation ray to the mean direction.
    pub max_ray_angle_deg: f64,
}

/// The point-or-bearing test's reading of one point.
pub struct PointOrBearingVerdict {
    /// The per-axis pixel noise the rays were weighted by: the
    /// reconstruction's measured reprojection noise.
    pub sigma_px: f64,
    /// The bearing, its cost, the depth score and the midpoint bound.
    pub score: BearingScore,
    /// The plain least-squares point fit, with `Λ`, warm-started from the
    /// stored position of a finite point.
    pub fit: Option<PointBearingFit>,
    /// The verdict at `DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`.
    pub finite: bool,
}

impl SfmrReconstruction {
    /// Inspect a single 3D point's triangulation.
    ///
    /// Observation coordinates come from the reconstruction's inline
    /// `keypoints_xy` when it carries one (every `embedded_patches`
    /// reconstruction, and a `sift_files` one with the optional copy);
    /// otherwise they are read from the observing images' `.sift` files, which
    /// must then be present in the workspace. An observation whose pixel cannot
    /// be read (a feature index past the end of its file) is listed with a
    /// `NaN` incidence and casts no ray.
    ///
    /// `point_idx` must be in range; callers should validate it first.
    pub fn inspect_point(
        &self,
        point_idx: usize,
        noise_floor_px: f64,
    ) -> Result<PointInspection, ReconstructionError> {
        let pt = &self.point_set.points[point_idx];
        let start = self.point_set.observation_offsets[point_idx];
        let observations = self.observations_for_point(point_idx);
        let noise = (pt.error as f64).max(noise_floor_px);
        let feature_indexes = self.feature_indexes();
        let mut wanted = vec![false; self.image_table.images.len()];
        for obs in observations {
            wanted[obs.image_index as usize] = true;
        }
        let pixels = self.observation_pixels(&wanted)?;

        let mut dirs: Vec<Vector3<f64>> = Vec::with_capacity(observations.len());
        let mut centers: Vec<Point3<f64>> = Vec::with_capacity(observations.len());
        let mut sigma_rad: Vec<f64> = Vec::with_capacity(observations.len());
        let mut obs_out: Vec<ObservationInspection> = Vec::with_capacity(observations.len());

        for (k, obs) in observations.iter().enumerate() {
            let feature_index = feature_indexes.map(|f| f[start + k]);
            let img_idx = obs.image_index as usize;
            let image = &self.image_table.images[img_idx];
            let camera = &self.image_table.cameras[image.camera_index as usize];
            let (fx, fy) = camera.focal_lengths();
            let (u, v) = (pixels[[start + k, 0]] as f64, pixels[[start + k, 1]] as f64);
            let mut inspection = ObservationInspection {
                image_index: img_idx,
                image_name: image.name.clone(),
                feature_index,
                incidence_deg: f64::NAN,
            };
            if !(u.is_finite() && v.is_finite()) {
                obs_out.push(inspection);
                continue;
            }

            let ray_cam = camera.pixel_to_ray(u, v);
            let ray_cam = Vector3::new(ray_cam[0], ray_cam[1], ray_cam[2]);
            let rc_norm = ray_cam.norm();
            // Angle off the optical axis, which is −Z in the canonical frame.
            inspection.incidence_deg = if rc_norm > 0.0 {
                (-ray_cam.z / rc_norm).clamp(-1.0, 1.0).acos().to_degrees()
            } else {
                0.0
            };

            let world = image.quaternion_wxyz.inverse() * ray_cam;
            let wn = world.norm();
            dirs.push(if wn > 0.0 { world / wn } else { world });
            centers.push(image.camera_center());
            sigma_rad.push(noise / fx.max(fy));
            obs_out.push(inspection);
        }

        let centers_all: Vec<Point3<f64>> = self
            .image_table
            .images
            .iter()
            .map(|im| im.camera_center())
            .collect();
        let finite_horizon = camera_extents(&centers_all);

        let offsets = [0usize, dirs.len()];
        let tri = triangulate_batch(&dirs, &centers, &offsets)[0];
        let du = depth_uncertainty_batch(&[tri], &dirs, &centers, &offsets, &sigma_rad)[0];
        // The test reads the observations' own pixels and the measured noise,
        // as reclassification does; a point without either is reported without
        // a verdict rather than refused, since the diagnostics above stand.
        let plain = PointBearingFitOptions {
            soft_l1_scale: None,
            ..PointBearingFitOptions::default()
        };
        let point_or_bearing = self
            .point_or_bearing_scores(Some(&[point_idx]), None, Some(&plain))
            .ok()
            .and_then(|scored| {
                let score = scored.scores[0]?;
                Some(PointOrBearingVerdict {
                    sigma_px: scored.sigma_px,
                    score,
                    fit: scored.fits.and_then(|fits| fits[0]),
                    finite: is_finite(&score, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD),
                })
            });

        // Largest angle of any ray to the mean viewing direction.
        let mut mean = Vector3::zeros();
        for d in &dirs {
            mean += d;
        }
        let max_ray_angle_deg = if mean.norm() > 0.0 {
            let m = mean.normalize();
            dirs.iter()
                .map(|d| d.dot(&m).clamp(-1.0, 1.0).acos().to_degrees())
                .fold(0.0_f64, f64::max)
        } else {
            0.0
        };

        Ok(PointInspection {
            w: pt.w,
            position: pt.position,
            error: pt.error,
            color: pt.color,
            observations: obs_out,
            point_or_bearing,
            triangulated_point: tri.point,
            eigenvalues: tri.eigenvalues,
            condition_number: tri.condition_number,
            in_front: tri.in_front_of_all_cameras,
            depth: du.depth,
            sigma: du.sigma,
            inverse_depth_z: du.inverse_depth_z,
            resolvable_distance: du.resolvable_distance,
            finite_horizon,
            baseline_span: camera_extents(&centers),
            max_ray_angle_deg,
        })
    }
}
